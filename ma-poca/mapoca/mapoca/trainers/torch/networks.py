import abc
import gc

from collections import defaultdict
from collections.abc import Callable
from typing import Optional, Union

import torchtune
import torchtune.data

from mlagents_envs.base_env import ActionSpec, ObservationSpec, ObservationType

from mapoca.torch_utils import nn, torch
from mapoca.trainers.buffer import AgentBuffer
from mapoca.trainers.exception import UnityTrainerException
from mapoca.trainers.settings import ConditioningType, EncoderType, NetworkSettings
from mapoca.trainers.torch.action_log_probs import ActionLogProbs
from mapoca.trainers.torch.action_model import ActionModel
from mapoca.trainers.torch.agent_action import AgentAction
from mapoca.trainers.torch.attention import EntityEmbedding, PositionalSelfAttention, ResidualSelfAttention, get_zero_entities_mask
from mapoca.trainers.torch.conditioning import ConditionalEncoder
from mapoca.trainers.torch.decoders import ValueHeads
from mapoca.trainers.torch.encoders import VectorInput
from mapoca.trainers.torch.layers import LSTM, Initialization, LinearEncoder, linear_layer
from mapoca.trainers.torch.utils import ModelUtils
from mapoca.trainers.trajectory import ObsUtil

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
EncoderFunction = Callable[
    [torch.Tensor, int, ActivationFunction, int, str, bool],
    torch.Tensor,
]

EPSILON = 1e-7


class ObservationEncoder(nn.Module):
    ATTENTION_EMBEDDING_SIZE = 128  # The embedding size of attention is fixed

    def __init__(
        self,
        observation_specs: list[ObservationSpec],
        h_size: int,
        vis_encode_type: EncoderType,
        normalize: bool = False,
    ):
        """
        Returns an ObservationEncoder that can process and encode a set of observations.
        Will use an RSA if needed for variable length observations.
        """
        super().__init__()
        self.processors, self.embedding_sizes = ModelUtils.create_input_processors(
            observation_specs,
            h_size,
            vis_encode_type,
            self.ATTENTION_EMBEDDING_SIZE,
            normalize=normalize,
        )
        self.rsa, self.x_self_encoder = ModelUtils.create_residual_self_attention(
            self.processors,
            self.embedding_sizes,
            self.ATTENTION_EMBEDDING_SIZE,
        )
        total_enc_size = sum(self.embedding_sizes) + self.ATTENTION_EMBEDDING_SIZE if self.rsa is not None else sum(self.embedding_sizes)
        self.normalize = normalize
        self._total_enc_size = total_enc_size

        self._total_goal_enc_size = 0
        self._goal_processor_indices: list[int] = []
        for i in range(len(observation_specs)):
            if observation_specs[i].observation_type == ObservationType.GOAL_SIGNAL:
                self._total_goal_enc_size += self.embedding_sizes[i]
                self._goal_processor_indices.append(i)

    @property
    def total_enc_size(self) -> int:
        """Returns the total encoding size for this ObservationEncoder."""
        return self._total_enc_size

    @property
    def total_goal_enc_size(self) -> int:
        """Returns the total goal encoding size for this ObservationEncoder."""
        return self._total_goal_enc_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        obs = ObsUtil.from_buffer(buffer, len(self.processors))
        for vec_input, enc in zip(obs, self.processors, strict=False):
            if isinstance(enc, VectorInput):
                enc.update_normalization(torch.as_tensor(vec_input))

    def copy_normalization(self, other_encoder: "ObservationEncoder") -> None:
        if self.normalize:
            for n1, n2 in zip(self.processors, other_encoder.processors, strict=False):
                if isinstance(n1, VectorInput) and isinstance(n2, VectorInput):
                    n1.copy_normalization(n2)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Encode observations using a list of processors and an RSA.
        :param inputs: List of Tensors corresponding to a set of obs.
        """
        encodes = []
        var_len_processor_inputs: list[tuple[nn.Module, torch.Tensor]] = []

        for idx, processor in enumerate(self.processors):
            if not isinstance(processor, EntityEmbedding):
                # The input can be encoded without having to process other inputs
                obs_input = inputs[idx]
                processed_obs = processor(obs_input)
                encodes.append(processed_obs)
            else:
                var_len_processor_inputs.append((processor, inputs[idx]))
        if len(encodes) != 0:
            encoded_self = torch.cat(encodes, dim=1)
            input_exist = True
        else:
            input_exist = False
        if len(var_len_processor_inputs) > 0 and self.rsa is not None:
            # Some inputs need to be processed with a variable length encoder
            masks = get_zero_entities_mask([p_i[1] for p_i in var_len_processor_inputs])
            embeddings: list[torch.Tensor] = []
            processed_self = self.x_self_encoder(encoded_self) if input_exist and self.x_self_encoder is not None else None
            for processor, var_len_input in var_len_processor_inputs:
                embeddings.append(processor(processed_self, var_len_input))
            qkv = torch.cat(embeddings, dim=1)
            attention_embedding = self.rsa(qkv, masks)
            if not input_exist:
                encoded_self = torch.cat([attention_embedding], dim=1)
                input_exist = True
            else:
                encoded_self = torch.cat([encoded_self, attention_embedding], dim=1)

        if not input_exist:
            raise UnityTrainerException(
                "The trainer was unable to process any of the provided inputs. "
                "Make sure the trained agents has at least one sensor attached to them.",
            )

        return encoded_self

    def get_goal_encoding(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Encode observations corresponding to goals using a list of processors.
        :param inputs: List of Tensors corresponding to a set of obs.
        """
        encodes = []
        for idx in self._goal_processor_indices:
            processor = self.processors[idx]
            if not isinstance(processor, EntityEmbedding):
                # The input can be encoded without having to process other inputs
                obs_input = inputs[idx]
                processed_obs = processor(obs_input)
                encodes.append(processed_obs)
            else:
                raise UnityTrainerException(
                    "The one of the goals uses variable length observations. This use case is not supported.",
                )
        if len(encodes) != 0:
            encoded = torch.cat(encodes, dim=1)
        else:
            raise UnityTrainerException(
                "Trainer was unable to process any of the goals provided as input.",
            )
        return encoded


class NetworkBody(nn.Module):
    def __init__(
        self,
        observation_specs: list[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = network_settings.memory.memory_size if network_settings.memory is not None else 0
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,
            self.normalize,
        )
        self.processors = self.observation_encoder.processors
        total_enc_size = self.observation_encoder.total_enc_size
        total_enc_size += encoded_act_size

        if self.observation_encoder.total_goal_enc_size > 0 and network_settings.goal_conditioning_type == ConditioningType.HYPER:
            self._body_encoder = ConditionalEncoder(
                total_enc_size,
                self.observation_encoder.total_goal_enc_size,
                self.h_size,
                network_settings.num_layers,
                1,
            )
        else:
            self._body_encoder = LinearEncoder(
                total_enc_size,
                network_settings.num_layers,
                self.h_size,
            )

        if self.use_lstm:
            self.lstm: LSTM | None = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def forward(
        self,
        inputs: list[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_self = self.observation_encoder(inputs)
        if actions is not None:
            encoded_self = torch.cat([encoded_self, actions], dim=1)
        if isinstance(self._body_encoder, ConditionalEncoder):
            goal = self.observation_encoder.get_goal_encoding(inputs)
            encoding = self._body_encoder(encoded_self, goal)
        else:
            encoding = self._body_encoder(encoded_self)

        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        return encoding, memories


class MultiAgentNetworkBody(torch.nn.Module):
    """
    A network body that uses a self attention layer to handle state
    and action input from a potentially variable number of agents that
    share the same observation and action space.
    """

    def __init__(
        self,
        observation_specs: list[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = network_settings.memory.memory_size if network_settings.memory is not None else 0
        self.action_spec = action_spec
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,
            self.normalize,
        )
        self.processors = self.observation_encoder.processors

        # Modules for multi-agent self-attention
        obs_only_ent_size = self.observation_encoder.total_enc_size
        q_ent_size = obs_only_ent_size + sum(self.action_spec.discrete_branches) + self.action_spec.continuous_size

        attention_embedding_size = self.h_size
        self.obs_encoder = EntityEmbedding(
            obs_only_ent_size,
            None,
            attention_embedding_size,
        )
        self.obs_action_encoder = EntityEmbedding(
            q_ent_size,
            None,
            attention_embedding_size,
        )

        self.self_attn = ResidualSelfAttention(attention_embedding_size)

        self.linear_encoder = LinearEncoder(
            attention_embedding_size,
            network_settings.num_layers,
            self.h_size,
            kernel_init=Initialization.XavierGlorotNormal,
            kernel_gain=(0.125 / self.h_size) ** 0.5,
        )

        if self.use_lstm:
            self.lstm: LSTM | None = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None
        self._current_max_agents = torch.nn.Parameter(
            torch.as_tensor(1),
            requires_grad=False,
        )

    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "MultiAgentNetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    def _get_masks_from_nans(self, obs_tensors: list[torch.Tensor]) -> torch.Tensor:  # noqa: PLR6301
        """
        Get attention masks by grabbing an arbitrary obs across all the agents
        Since these are raw obs, the padded values are still NaN.
        """
        only_first_obs = [_all_obs[0] for _all_obs in obs_tensors]
        # Just get the first element in each obs regardless of its dimension. This will speed up
        # searching for NaNs.
        only_first_obs_flat = torch.stack(
            [_obs.flatten(start_dim=1)[:, 0] for _obs in only_first_obs],
            dim=1,
        )
        # Get the mask from NaNs
        return only_first_obs_flat.isnan().float()

    def _copy_and_remove_nans_from_obs(  # noqa: PLR6301
        self,
        all_obs: list[list[torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """Helper function to remove NaNs from observations using an attention mask."""
        obs_with_no_nans = []
        for i_agent, single_agent_obs in enumerate(all_obs):
            no_nan_obs = []
            for obs in single_agent_obs:
                new_obs = obs.clone()
                new_obs[attention_mask.bool()[:, i_agent], ::] = 0.0  # Remove NaNs fast
                no_nan_obs.append(new_obs)
            obs_with_no_nans.append(no_nan_obs)
        return obs_with_no_nans

    def forward(
        self,
        obs_only: list[list[torch.Tensor]],
        obs: list[list[torch.Tensor]],
        actions: list[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param obs_only: Observations to be processed that do not have corresponding actions.
            These are encoded with the obs_encoder.
        :param obs: Observations to be processed that do have corresponding actions.
            After concatenation with actions, these are processed with obs_action_encoder.
        :param actions: After concatenation with obs, these are processed with obs_action_encoder.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        """
        self_attn_masks = []
        self_attn_inputs = []
        concat_f_inp = []
        if obs:
            obs_attn_mask = self._get_masks_from_nans(obs)
            obs = self._copy_and_remove_nans_from_obs(obs, obs_attn_mask)
            for inputs, action in zip(obs, actions, strict=False):
                encoded = self.observation_encoder(inputs)
                cat_encodes = [
                    encoded,
                    action.to_flat(self.action_spec.discrete_branches),
                ]
                concat_f_inp.append(torch.cat(cat_encodes, dim=1))
            f_inp = torch.stack(concat_f_inp, dim=1)
            self_attn_masks.append(obs_attn_mask)
            self_attn_inputs.append(self.obs_action_encoder(None, f_inp))

        concat_encoded_obs = []
        if obs_only:
            obs_only_attn_mask = self._get_masks_from_nans(obs_only)
            obs_only = self._copy_and_remove_nans_from_obs(obs_only, obs_only_attn_mask)
            for inputs in obs_only:
                encoded = self.observation_encoder(inputs)
                concat_encoded_obs.append(encoded)
            g_inp = torch.stack(concat_encoded_obs, dim=1)
            self_attn_masks.append(obs_only_attn_mask)
            self_attn_inputs.append(self.obs_encoder(None, g_inp))

        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        encoded_state = self.self_attn(encoded_entity, self_attn_masks)

        flipped_masks = 1 - torch.cat(self_attn_masks, dim=1)
        num_agents = torch.sum(flipped_masks, dim=1, keepdim=True)
        if torch.max(num_agents).item() > self._current_max_agents:
            self._current_max_agents = torch.nn.Parameter(
                torch.as_tensor(torch.max(num_agents).item()),
                requires_grad=False,
            )

        # num_agents will be -1 for a single agent and +1 when the current maximum is reached
        num_agents = num_agents * 2.0 / self._current_max_agents - 1

        encoding = self.linear_encoder(encoded_state)
        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        encoding = torch.cat([encoding, num_agents], dim=1)
        return encoding, memories


class Critic(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """

    def critic_pass(  # noqa: B027
        self,
        inputs: list[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Get value outputs for the given obs.
        :param inputs: List of inputs as tensors.
        :param memories: Tensor of memories, if using memory. Otherwise, None.
        :returns: Dict of reward stream to output tensor for values.
        """


class ValueNetwork(nn.Module, Critic):
    def __init__(
        self,
        stream_names: list[str],
        observation_specs: list[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ):
        # This is not a typo, we want to call __init__ of nn.Module
        nn.Module.__init__(self)
        self.network_body = NetworkBody(
            observation_specs,
            network_settings,
            encoded_act_size=encoded_act_size,
        )
        encoding_size = network_settings.memory.memory_size // 2 if network_settings.memory is not None else network_settings.hidden_units
        self.value_heads = ValueHeads(stream_names, encoding_size, outputs_per_stream)

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def critic_pass(
        self,
        inputs: list[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        value_outputs, critic_mem_out = self.forward(
            inputs,
            memories=memories,
            sequence_length=sequence_length,
        )
        return value_outputs, critic_mem_out

    def forward(
        self,
        inputs: list[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs,
            actions,
            memories,
            sequence_length,
        )
        output = self.value_heads(encoding)
        return output, memories


class Actor(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """

    def get_action_and_stats(  # noqa: B027
        self,
        inputs: list[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """

    def get_stats(  # noqa: B027
        self,
        inputs: list[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[ActionLogProbs, torch.Tensor]:
        """
        Returns log_probs for actions and entropies.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param actions: AgentAction of actions.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """

    @abc.abstractmethod
    def forward(
        self,
        inputs: list[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> tuple[Union[int, torch.Tensor], ...]:
        """
        Forward pass of the Actor for inference. This is required for export to ONNX, and
        the inputs and outputs of this method should not be changed without a respective change
        in the ONNX export code.
        """


class SimpleActor(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        observation_specs: list[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]),
            requires_grad=False,
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]),
            requires_grad=False,
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]),
            requires_grad=False,
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]),
            requires_grad=False,
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [
                    self.action_spec.continuous_size + sum(self.action_spec.discrete_branches),
                ],
            ),
            requires_grad=False,
        )
        self.network_body = NetworkBody(observation_specs, network_settings)
        if network_settings.memory is not None:
            self.encoding_size = network_settings.memory.memory_size // 2
        else:
            self.encoding_size = network_settings.hidden_units
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]),
            requires_grad=False,
        )

        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
        )

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def get_action_and_stats(
        self,
        inputs: list[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:
        encoding, memories = self.network_body(
            inputs,
            memories=memories,
            sequence_length=sequence_length,
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        return action, log_probs, entropies, memories

    def get_stats(
        self,
        inputs: list[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[ActionLogProbs, torch.Tensor]:
        encoding, _actor_mem_outs = self.network_body(
            inputs,
            memories=memories,
            sequence_length=sequence_length,
        )
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)

        return log_probs, entropies

    def forward(
        self,
        inputs: list[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> tuple[Union[int, torch.Tensor], ...]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.

        At this moment, torch.onnx.export() doesn't accept None as tensor to be exported,
        so the size of return tuple varies with action spec.
        """
        encoding, memories_out = self.network_body(
            inputs,
            memories=memories,
            sequence_length=1,
        )

        (
            cont_action_out,
            disc_action_out,
            _action_out_deprecated,
        ) = self.action_model.get_action_out(encoding, masks)
        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [cont_action_out, self.continuous_act_size_vector]
        if self.action_spec.discrete_size > 0:
            export_out += [disc_action_out, self.discrete_act_size_vector]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]
        return tuple(export_out)


class SharedActorCritic(SimpleActor, Critic):
    def __init__(
        self,
        observation_specs: list[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: list[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        self.use_lstm = network_settings.memory is not None
        super().__init__(
            observation_specs,
            network_settings,
            action_spec,
            conditional_sigma,
            tanh_squash,
        )
        self.stream_names = stream_names
        self.value_heads = ValueHeads(stream_names, self.encoding_size)

    def critic_pass(
        self,
        inputs: list[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories_out = self.network_body(
            inputs,
            memories=memories,
            sequence_length=sequence_length,
        )
        return self.value_heads(encoding), memories_out


class GlobalSteps(nn.Module):
    def __init__(self):
        super().__init__()
        self.__global_step = nn.Parameter(
            torch.Tensor([0]).to(torch.int64),
            requires_grad=False,
        )

    @property
    def current_step(self):
        return int(self.__global_step.item())

    @current_step.setter
    def current_step(self, value):
        self.__global_step[:] = value

    def increment(self, value):
        self.__global_step += value


class LearningRate(nn.Module):
    def __init__(self, lr):
        # TODO: add learning rate decay
        super().__init__()
        self.learning_rate = torch.Tensor([lr])


class CLUBCategorical(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super().__init__()
        self.varnet = nn.Sequential(
            linear_layer(x_dim, hidden_size),
            nn.ReLU(),
            linear_layer(hidden_size, y_dim),
        )

    def forward(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
        # x_samples [sample_size, x_dim]  # noqa: ERA001
        # y_samples [sample_size, y_dim]  # noqa: ERA001

        logits = self.varnet(x_samples)

        # log of conditional probability of positive sample pairs
        sample_size, y_dim = logits.shape

        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)
        y_samples_extend = y_samples.unsqueeze(0).repeat(sample_size, 1, 1)

        gc.collect()
        torch.cuda.empty_cache()
        # log of conditional probability of negative sample pairs
        log_mat = -nn.functional.cross_entropy(
            logits_extend.reshape(-1, y_dim),
            y_samples_extend.reshape(-1, y_dim).argmax(dim=-1),
            reduction="none",
        )

        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat)
        negative = log_mat.mean(1)
        return (positive - negative).mean(dim=-1, keepdim=True)

    def loglikeli(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
        logits = self.varnet(x_samples)
        return nn.functional.cross_entropy(logits, y_samples.argmax(dim=-1), reduction="none").neg()

    def learning_loss(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
        return self.loglikeli(x_samples, y_samples).neg()


class CLUBContinuous(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_size: int):
        super().__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(linear_layer(x_dim, hidden_size // 2), nn.ReLU(), linear_layer(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(linear_layer(x_dim, hidden_size // 2), nn.ReLU(), linear_layer(hidden_size // 2, y_dim), nn.Tanh())

    def get_mu_logvar(self, x_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = -((mu - y_samples) ** 2) / 2.0 / logvar.exp()

        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)

        # log of conditional probability of negative sample pairs
        negative = -((y_samples_1 - prediction_1) ** 2).mean(1) / 2.0 / logvar.exp()

        return (positive.sum(1) - negative.sum(1)).mean(dim=-1, keepdim=True)

    def loglikeli(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (((mu - y_samples) ** 2).neg() / logvar.exp() - logvar).sum(1)

    def learning_loss(self, x_samples: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
        return self.loglikeli(x_samples, y_samples).neg()


class MIR3Body(nn.Module):
    def __init__(self, observation_specs: list[ObservationSpec], network_settings: NetworkSettings, action_spec: ActionSpec):
        super().__init__()
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]),
            requires_grad=False,
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]),
            requires_grad=False,
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]),
            requires_grad=False,
        )
        self.network_body = NetworkBody(observation_specs, network_settings)
        if network_settings.memory is not None:
            self.encoding_size = network_settings.memory.memory_size // 2
        else:
            self.encoding_size = network_settings.hidden_units
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]),
            requires_grad=False,
        )
        self.continuous_club_head = None
        self.discrete_club_head = None
        if self.action_spec.continuous_size > 0:
            self.continuous_club_head = CLUBContinuous(self.encoding_size, int(self.action_spec.continuous_size))
        if self.action_spec.discrete_size > 0:
            self.discrete_club_head = CLUBCategorical(self.encoding_size, sum(self.action_spec.discrete_branches))

    def forward(
        self,
        inputs: list[torch.Tensor],
        actions: list[AgentAction],
        memories: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoding, memories_out = self.network_body(
            inputs,
            memories=memories,
            sequence_length=1,
        )
        continuous_club = None
        discrete_club = None
        if self.continuous_club_head is not None:
            continuous_act = torch.stack([act.continuous_tensor for act in actions], dim=1)
            continuous_club = self.continuous_club_head(encoding, continuous_act)
        if self.discrete_club_head is not None:
            discrete_act = torch.stack(
                [torch.cat(ModelUtils.actions_to_onehot(act.discrete_list, self.action_spec.discrete_branches), dim=1) for act in actions],
                dim=1,
            )
            discrete_club = self.discrete_club_head(encoding, discrete_act)

        return continuous_club, discrete_club, memories_out


class MultiAgentMIR3Body(nn.Module):
    def __init__(
        self,
        observation_specs: list[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
    ):
        super().__init__()
        self.network_body = MultiAgentNetworkBody(
            observation_specs,
            network_settings,
            action_spec,
        )
        encoding_size = network_settings.memory.memory_size // 2 if network_settings.memory is not None else network_settings.hidden_units

        self.continuous_club_head = None
        self.discrete_club_head = None
        self.obs_act_encoder = LinearEncoder(
            2 * (encoding_size + 1)
            + int(self.network_body.action_spec.continuous_size)
            + sum(self.network_body.action_spec.discrete_branches),
            network_settings.num_layers,
            encoding_size,
        )
        if self.network_body.action_spec.continuous_size > 0:
            self.continuous_club_head = CLUBContinuous(encoding_size, int(self.network_body.action_spec.continuous_size), encoding_size)
        if self.network_body.action_spec.discrete_size > 0:
            self.discrete_club_head = CLUBCategorical(
                encoding_size,
                sum(self.network_body.action_spec.discrete_branches),
                encoding_size,
            )
        self.sequence_encoding_head = PositionalSelfAttention(encoding_size)

    def _get_encoding(
        self,
        obs_only: list[list[torch.Tensor]],
        actions: AgentAction,
        first_steps: torch.Tensor,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoding, memories = self.network_body(
            obs_only=obs_only,
            obs=[],
            actions=[],
            memories=memories,
            sequence_length=sequence_length,
        )
        return self._get_encoding_and_actions(
            encoding=encoding,
            actions=actions,
            first_steps=first_steps,
        )

    def _get_baseline_encoding(
        self,
        obs_without_actions: list[torch.Tensor],
        obs_with_actions: tuple[list[list[torch.Tensor]], list[AgentAction]],
        actions: AgentAction,
        first_steps: torch.Tensor,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (obs, act) = obs_with_actions
        encoding, memories = self.network_body(
            obs_only=[obs_without_actions],
            obs=obs,
            actions=act,
            memories=memories,
            sequence_length=sequence_length,
        )
        return self._get_encoding_and_actions(
            encoding=encoding,
            actions=actions,
            first_steps=first_steps,
        )

    def _get_encoding_and_actions(
        self,
        encoding: torch.Tensor,
        actions: AgentAction,
        first_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        discrete_act = torch.cat(
            ModelUtils.actions_to_onehot(actions.discrete_tensor, self.network_body.action_spec.discrete_branches),
            dim=1,
        )

        encoding = self.obs_act_encoder(
            torch.cat([encoding[:-1], actions.to_flat(self.network_body.action_spec.discrete_branches)[:-1], encoding[1:]], dim=-1),
        )
        episodes = torch.argwhere(torch.cat([first_steps, torch.ones(1, dtype=first_steps.dtype, device=first_steps.device)]))
        episode_lengths = episodes[1:] - episodes[:-1]
        encoding_slices = []
        encoding_actions = defaultdict(list)

        # construct a sequence of histories for each step in the episode
        for episode, episode_length in zip(episodes[:-1], episode_lengths, strict=True):
            if episode_length:
                encoding_slices.extend([encoding[episode : episode + length] for length in range(1, episode_length + 1)])
                encoding_actions["continuous"].extend([
                    actions.continuous_tensor[episode + length - 1] for length in range(1, episode_length + 1)
                ])
                encoding_actions["discrete"].extend([discrete_act[episode + length - 1] for length in range(1, episode_length + 1)])
        encoding_actions = {
            "continuous": torch.stack(encoding_actions["continuous"]),
            "discrete": torch.stack(encoding_actions["discrete"]).squeeze(),
        }
        encoding = self.sequence_encoding_head(torchtune.data.left_pad_sequence(encoding_slices, batch_first=True)).mean(dim=1)
        return encoding, encoding_actions

    def _get_training_slices_and_actions(
        self,
        encoding: torch.Tensor,
        actions: AgentAction,
        first_steps: torch.Tensor,
        loss_masks: torch.Tensor | None = None,
        samples_per_episode: int = 10,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        discrete_act = torch.cat(
            ModelUtils.actions_to_onehot(actions.discrete_tensor, self.network_body.action_spec.discrete_branches),
            dim=1,
        )
        encoding = self.obs_act_encoder(
            torch.cat([encoding[:-1], actions.to_flat(self.network_body.action_spec.discrete_branches)[:-1], encoding[1:]], dim=-1),
        )
        # append a True to capture the final step of the episode
        episodes = torch.argwhere(torch.cat([first_steps, torch.ones(1, dtype=first_steps.dtype, device=first_steps.device)]))
        episode_lengths = episodes[1:] - episodes[:-1]
        training_slices = []
        training_actions = defaultdict(list)
        training_masks = []

        # construct a sequence of random sub-histories for each step in the episode
        for episode, episode_length in zip(episodes[:-1], episode_lengths, strict=True):
            if episode_length:
                slice_lengths = torch.randint(0, episode_length, (samples_per_episode,))
                slice_starts = torch.randint(0, episode_length, (samples_per_episode,))
                slice_ends = (slice_starts + slice_lengths).clamp(max=episode_length)
                slice_lengths = slice_ends - slice_starts
                slice_starts += episode
                slice_ends += episode
                slices = [
                    (slice_start, slice_end)
                    for slice_start, slice_end in zip(slice_starts, slice_ends, strict=True)
                    if slice_end > slice_start
                ]
                training_slices.extend([encoding[slice_start:slice_end] for slice_start, slice_end in slices])
                training_actions["continuous"].extend([actions.continuous_tensor[slice_end - 1] for _, slice_end in slices])
                training_actions["discrete"].extend([discrete_act[slice_end - 1] for _, slice_end in slices])
                training_masks.extend([loss_masks[slice_end - 1] for _, slice_end in slices])
        encoding = self.sequence_encoding_head(torchtune.data.left_pad_sequence(training_slices, batch_first=True)).mean(dim=1)
        training_actions = {
            "continuous": torch.stack(training_actions["continuous"]),
            "discrete": torch.stack(training_actions["discrete"]),
        }
        return encoding, training_actions, torch.stack(training_masks).view(-1)

    def learning_loss(
        self,
        obs_only: list[list[torch.Tensor]],
        actions: AgentAction,
        loss_masks: torch.Tensor,
        first_steps: torch.Tensor,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> torch.Tensor:
        encoding, memories = self.network_body(
            obs_only=obs_only,
            obs=[],
            actions=[],
            memories=memories,
            sequence_length=sequence_length,
        )
        loss = torch.zeros((1,))
        encoding, training_actions, loss_masks = self._get_training_slices_and_actions(
            encoding,
            actions,
            first_steps=first_steps,
            loss_masks=loss_masks,
        )

        if self.continuous_club_head is not None:
            loss = ModelUtils.masked_mean(self.continuous_club_head.learning_loss(encoding, training_actions["continuous"]), loss_masks)
        if self.discrete_club_head is not None:
            loss = ModelUtils.masked_mean(self.discrete_club_head.learning_loss(encoding, training_actions["discrete"]), loss_masks) + (
                loss or 0
            )
        return loss

    def baseline_loss(
        self,
        obs_without_actions: list[torch.Tensor],
        obs_with_actions: tuple[list[list[torch.Tensor]], list[AgentAction]],
        actions: AgentAction,
        loss_masks: torch.Tensor,
        first_steps: torch.Tensor,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> torch.Tensor:
        (obs, act) = obs_with_actions
        encoding, memories = self.network_body(
            obs_only=[obs_without_actions],
            obs=obs,
            actions=act,
            memories=memories,
            sequence_length=sequence_length,
        )
        loss = torch.zeros((1,))
        encoding, training_actions, loss_masks = self._get_training_slices_and_actions(
            encoding,
            actions,
            first_steps=first_steps,
            loss_masks=loss_masks,
        )
        if self.continuous_club_head is not None:
            loss = ModelUtils.masked_mean(self.continuous_club_head.learning_loss(encoding, training_actions["continuous"]), loss_masks)
        if self.discrete_club_head is not None:
            loss = ModelUtils.masked_mean(self.discrete_club_head.learning_loss(encoding, training_actions["discrete"]), loss_masks) + (
                loss or 0
            )
        return loss

    def forward(
        self,
        encoding: torch.Tensor,
        actions: dict[str, torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # begin MIR3 predictions
        club = torch.zeros(encoding.shape[0])
        if self.continuous_club_head is not None:
            club += self.continuous_club_head(encoding, actions["continuous"])
        if self.discrete_club_head is not None:
            club += self.discrete_club_head(encoding, actions["discrete"])

        return club, memories

    def club(
        self,
        obs: list[list[torch.Tensor]],
        actions: AgentAction,
        first_steps: torch.Tensor,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The POCA baseline marginalizes the action of the agent associated with self_obs.
        It calls the forward pass of the MultiAgentNetworkBody with the state action
        pairs of groupmates but just the state of the agent in question.
        :param obs_without_actions: The obs of the agent for which to compute the baseline.
        :param obs_with_actions: Tuple of observations and actions for all groupmates.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.

        :return: A Tuple of Dict of reward stream to tensor and critic memories.
        """
        encoding, sampled_actions = self._get_encoding(
            obs_only=obs,
            actions=actions,
            first_steps=first_steps,
            memories=memories,
            sequence_length=sequence_length,
        )
        return self.forward(encoding, sampled_actions, memories=memories, sequence_length=sequence_length)

    def club_baseline(
        self,
        obs_without_actions: list[torch.Tensor],
        obs_with_actions: tuple[list[list[torch.Tensor]], list[AgentAction]],
        actions: AgentAction,
        first_steps: torch.Tensor,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The POCA baseline marginalizes the action of the agent associated with self_obs.
        It calls the forward pass of the MultiAgentNetworkBody with the state action
        pairs of groupmates but just the state of the agent in question.
        :param obs_without_actions: The obs of the agent for which to compute the baseline.
        :param obs_with_actions: Tuple of observations and actions for all groupmates.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.

        :return: A Tuple of Dict of reward stream to tensor and critic memories.
        """
        encoding, sampled_actions = self._get_baseline_encoding(
            obs_without_actions=obs_without_actions,
            obs_with_actions=obs_with_actions,
            actions=actions,
            first_steps=first_steps,
            memories=memories,
            sequence_length=sequence_length,
        )

        return self.forward(encoding, sampled_actions, memories=memories, sequence_length=sequence_length)
