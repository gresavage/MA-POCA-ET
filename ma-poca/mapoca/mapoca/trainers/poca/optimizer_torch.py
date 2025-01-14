from collections import defaultdict
from typing import Optional, cast

import numpy as np

from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import timed

from mapoca.torch_utils import default_device, torch
from mapoca.trainers.buffer import AgentBuffer, AgentBufferField, BufferKey, RewardSignalUtil
from mapoca.trainers.optimizer.torch_optimizer import TorchOptimizer
from mapoca.trainers.policy.torch_policy import TorchPolicy
from mapoca.trainers.settings import NetworkSettings, POCASettings, RewardSignalSettings, RewardSignalType, TrainerSettings
from mapoca.trainers.torch.action_log_probs import ActionLogProbs
from mapoca.trainers.torch.agent_action import AgentAction
from mapoca.trainers.torch.components.reward_providers.extrinsic_reward_provider import ExtrinsicRewardProvider
from mapoca.trainers.torch.decoders import ValueHeads
from mapoca.trainers.torch.networks import Critic, MultiAgentNetworkBody
from mapoca.trainers.torch.utils import ModelUtils
from mapoca.trainers.trajectory import GroupObsUtil, ObsUtil

logger = get_logger(__name__)


class TorchPOCAOptimizer(TorchOptimizer):
    class POCAValueNetwork(torch.nn.Module, Critic):
        """
        The POCAValueNetwork uses the MultiAgentNetworkBody to compute the value
        and POCA baseline for a variable number of agents in a group that all
        share the same observation and action space.
        """

        def __init__(
            self,
            stream_names: list[str],
            observation_specs: list[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            torch.nn.Module.__init__(self)
            self.network_body = MultiAgentNetworkBody(
                observation_specs,
                network_settings,
                action_spec,
            )
            if network_settings.memory is not None:
                encoding_size = network_settings.memory.memory_size // 2
            else:
                encoding_size = network_settings.hidden_units

            self.value_heads = ValueHeads(stream_names, encoding_size + 1, 1)
            # The + 1 is for the normalized number of agents

        @property
        def memory_size(self) -> int:
            return self.network_body.memory_size

        def update_normalization(self, buffer: AgentBuffer) -> None:
            self.network_body.update_normalization(buffer)

        def baseline(
            self,
            obs_without_actions: list[torch.Tensor],
            obs_with_actions: tuple[list[list[torch.Tensor]], list[AgentAction]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
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
            (obs, actions) = obs_with_actions
            encoding, memories = self.network_body(
                obs_only=[obs_without_actions],
                obs=obs,
                actions=actions,
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding,
                memories,
                sequence_length,
            )
            return value_outputs, critic_mem_out

        def critic_pass(
            self,
            obs: list[list[torch.Tensor]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            """
            A centralized value function. It calls the forward pass of MultiAgentNetworkBody
            with just the states of all agents.
            :param obs: List of observations for all agents in group
            :param memories: If using memory, a Tensor of initial memories.
            :param sequence_length: If using memory, the sequence length.
            :return: A Tuple of Dict of reward stream to tensor and critic memories.
            """
            encoding, memories = self.network_body(
                obs_only=obs,
                obs=[],
                actions=[],
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding,
                memories,
                sequence_length,
            )
            return value_outputs, critic_mem_out

        def forward(
            self,
            encoding: torch.Tensor,
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = self.value_heads(encoding)
            return output, memories

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        :param policy: A TorchPolicy object that will be updated by this POCA Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self._critic = TorchPOCAOptimizer.POCAValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            network_settings=trainer_settings.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        # Move to GPU if needed
        self._critic.to(default_device())

        params = list(self.policy.actor.parameters()) + list(self.critic.parameters())
        self.hyperparameters: POCASettings = cast(
            "POCASettings",
            trainer_settings.hyperparameters,
        )
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params,
            lr=self.trainer_settings.hyperparameters.learning_rate,
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())
        self.value_memory_dict: dict[str, torch.Tensor] = {}
        self.baseline_memory_dict: dict[str, torch.Tensor] = {}

    def create_reward_signals(
        self,
        reward_signal_configs: dict[RewardSignalType, RewardSignalSettings],
    ) -> None:
        """
        Create reward signals. Override default to provide warnings for Curiosity and
        GAIL, and make sure Extrinsic adds team rewards.
        :param reward_signal_configs: Reward signal config.
        """
        for reward_signal in reward_signal_configs:
            if reward_signal != RewardSignalType.EXTRINSIC:
                logger.warning(
                    f"Reward signal {reward_signal.value.capitalize()} is not supported with the POCA trainer; results may be unexpected.",
                )
        super().create_reward_signals(reward_signal_configs)
        # Make sure we add the groupmate rewards in POCA, so agents learn how to help each
        # other achieve individual rewards as well
        for reward_provider in self.reward_signals.values():
            if isinstance(reward_provider, ExtrinsicRewardProvider):
                reward_provider.add_groupmate_rewards = True

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> dict[str, float]:  # noqa: PLR0914
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        old_baseline_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)],
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)],
            )
            old_baseline_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.baseline_estimates_key(name)],
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = [[ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs] for _groupmate_obs in groupmate_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)
        groupmate_actions = AgentAction.group_from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0,
                len(batch[BufferKey.CRITIC_MEMORY]),
                self.policy.sequence_length,
            )
        ]

        baseline_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.BASELINE_MEMORY][i])
            for i in range(
                0,
                len(batch[BufferKey.BASELINE_MEMORY]),
                self.policy.sequence_length,
            )
        ]

        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)
            baseline_memories = torch.stack(baseline_memories).unsqueeze(0)

        log_probs, entropy = self.policy.evaluate_actions(
            current_obs,
            masks=act_masks,
            actions=actions,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        all_obs = [current_obs, *groupmate_obs]
        values, _ = self.critic.critic_pass(
            all_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
        baselines, _ = self.critic.baseline(
            current_obs,
            groupmate_obs_and_actions,
            memories=baseline_memories,
            sequence_length=self.policy.sequence_length,
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        baseline_loss = ModelUtils.trust_region_value_loss(
            baselines,
            old_baseline_values,
            returns,
            decay_eps,
            loss_masks,
        )
        value_loss = ModelUtils.trust_region_value_loss(
            values,
            old_values,
            returns,
            decay_eps,
            loss_masks,
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )

        loss = policy_loss + 0.5 * (value_loss + 0.5 * baseline_loss) - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Baseline Loss": baseline_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))

        return update_stats

    def get_modules(self):
        modules = {"Optimizer:adam": self.optimizer, "Optimizer:critic": self._critic}
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

    def _evaluate_by_sequence_team(  # noqa: PLR0914, PLR0915
        self,
        self_obs: list[torch.Tensor],
        obs: list[list[torch.Tensor]],
        actions: list[AgentAction],
        init_value_mem: torch.Tensor,
        init_baseline_mem: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        AgentBufferField,
        AgentBufferField,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Evaluate a trajectory sequence-by-sequence, assembling the result. This enables us to get the
        intermediate memories for the critic.
        :param tensor_obs: A List of tensors of shape (trajectory_len, <obs_dim>) that are the agent's
            observations for this trajectory.
        :param initial_memory: The memory that preceeds this trajectory. Of shape (1,1,<mem_size>), i.e.
            what is returned as the output of a MemoryModules.
        :return: A Tuple of the value estimates as a Dict of [name, tensor], an AgentBufferField of the initial
            memories to be used during value function update, and the final memory at the end of the trajectory.
        """
        num_experiences = self_obs[0].shape[0]
        all_next_value_mem = AgentBufferField()
        all_next_baseline_mem = AgentBufferField()

        # When using LSTM, we need to divide the trajectory into sequences of equal length. Sometimes,
        # that division isn't even, and we must pad the leftover sequence.
        # In the buffer, the last sequence are the ones that are padded. So if seq_len = 3 and
        # trajectory is of length 10, the last sequence is [obs,pad,pad].
        # Compute the number of elements in this padded seq.
        leftover_seq_len = num_experiences % self.policy.sequence_length

        all_values: dict[str, list[np.ndarray]] = defaultdict(list)
        all_baseline: dict[str, list[np.ndarray]] = defaultdict(list)
        baseline_mem = init_baseline_mem
        value_mem = init_value_mem

        # Evaluate other trajectories, carrying over _mem after each
        # trajectory
        for seq_num in range(num_experiences // self.policy.sequence_length):
            for _ in range(self.policy.sequence_length):
                all_next_value_mem.append(ModelUtils.to_numpy(value_mem.squeeze()))
                all_next_baseline_mem.append(
                    ModelUtils.to_numpy(baseline_mem.squeeze()),
                )

            start = seq_num * self.policy.sequence_length
            end = (seq_num + 1) * self.policy.sequence_length

            self_seq_obs = []
            groupmate_seq_obs = []
            groupmate_seq_act = []
            seq_obs = [_self_obs[start:end] for _self_obs in self_obs]
            self_seq_obs.append(seq_obs)

            for groupmate_obs, groupmate_action in zip(obs, actions, strict=False):
                seq_obs = []
                for _obs in groupmate_obs:
                    sliced_seq_obs = _obs[start:end]
                    seq_obs.append(sliced_seq_obs)
                groupmate_seq_obs.append(seq_obs)
                act = groupmate_action.slice(start, end)
                groupmate_seq_act.append(act)

            all_seq_obs = self_seq_obs + groupmate_seq_obs
            values, value_mem = self.critic.critic_pass(
                all_seq_obs,
                value_mem,
                sequence_length=self.policy.sequence_length,
            )
            for signal_name, _val in values.items():
                all_values[signal_name].append(_val)

            groupmate_obs_and_actions = (groupmate_seq_obs, groupmate_seq_act)
            baselines, baseline_mem = self.critic.baseline(
                self_seq_obs[0],
                groupmate_obs_and_actions,
                baseline_mem,
                sequence_length=self.policy.sequence_length,
            )
            for signal_name, _val in baselines.items():
                all_baseline[signal_name].append(_val)

        # Compute values for the potentially truncated initial sequence
        if leftover_seq_len > 0:
            self_seq_obs = []
            groupmate_seq_obs = []
            groupmate_seq_act = []
            seq_obs = []
            for _self_obs in self_obs:
                last_seq_obs = _self_obs[-leftover_seq_len:]
                seq_obs.append(last_seq_obs)
            self_seq_obs.append(seq_obs)

            for groupmate_obs, groupmate_action in zip(obs, actions, strict=False):
                seq_obs = []
                for _obs in groupmate_obs:
                    last_seq_obs = _obs[-leftover_seq_len:]
                    seq_obs.append(last_seq_obs)
                groupmate_seq_obs.append(seq_obs)
                act = groupmate_action.slice(len(_obs) - leftover_seq_len, len(_obs))
                groupmate_seq_act.append(act)

            # For the last sequence, the initial memory should be the one at the
            # beginning of this trajectory.
            seq_obs = []
            for _ in range(leftover_seq_len):
                all_next_value_mem.append(ModelUtils.to_numpy(value_mem.squeeze()))
                all_next_baseline_mem.append(
                    ModelUtils.to_numpy(baseline_mem.squeeze()),
                )

            all_seq_obs = self_seq_obs + groupmate_seq_obs
            last_values, value_mem = self.critic.critic_pass(
                all_seq_obs,
                value_mem,
                sequence_length=leftover_seq_len,
            )
            for signal_name, _val in last_values.items():
                all_values[signal_name].append(_val)
            groupmate_obs_and_actions = (groupmate_seq_obs, groupmate_seq_act)
            last_baseline, baseline_mem = self.critic.baseline(
                self_seq_obs[0],
                groupmate_obs_and_actions,
                baseline_mem,
                sequence_length=leftover_seq_len,
            )
            for signal_name, _val in last_baseline.items():
                all_baseline[signal_name].append(_val)
        # Create one tensor per reward signal
        all_value_tensors = {signal_name: torch.cat(value_list, dim=0) for signal_name, value_list in all_values.items()}
        all_baseline_tensors = {signal_name: torch.cat(baseline_list, dim=0) for signal_name, baseline_list in all_baseline.items()}
        next_value_mem = value_mem
        next_baseline_mem = baseline_mem
        return (
            all_value_tensors,
            all_baseline_tensors,
            all_next_value_mem,
            all_next_baseline_mem,
            next_value_mem,
            next_baseline_mem,
        )

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: list[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> tuple[dict[str, np.ndarray], dict[str, float], Optional[AgentBufferField]]:
        """
        Override base class method. Unused in the trainer, but needed to make sure class heirarchy is maintained.
        Assume that there are no group obs.
        """
        (
            value_estimates,
            _,
            next_value_estimates,
            all_next_value_mem,
            _,
        ) = self.get_trajectory_and_baseline_value_estimates(
            batch,
            next_obs,
            [],
            done,
            agent_id,
        )

        return value_estimates, next_value_estimates, all_next_value_mem

    def get_trajectory_and_baseline_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: list[np.ndarray],
        next_groupmate_obs: list[list[np.ndarray]],
        done: bool,
        agent_id: str = "",
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, float],
        Optional[AgentBufferField],
        Optional[AgentBufferField],
    ]:
        """
        Get value estimates, baseline estimates, and memories for a trajectory, in batch form.
        :param batch: An AgentBuffer that consists of a trajectory.
        :param next_obs: the next observation (after the trajectory). Used for boostrapping
            if this is not a termiinal trajectory.
        :param next_groupmate_obs: the next observations from other members of the group.
        :param done: Set true if this is a terminal trajectory.
        :param agent_id: Agent ID of the agent that this trajectory belongs to.
        :returns: A Tuple of the Value Estimates as a Dict of [name, np.ndarray(trajectory_len)],
            the baseline estimates as a Dict, the final value estimate as a Dict of [name, float], and
            optionally (if using memories) an AgentBufferField of initial critic and baseline memories to be used
            during update.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)

        current_obs = ObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)

        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = [[ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs] for _groupmate_obs in groupmate_obs]

        groupmate_actions = AgentAction.group_from_buffer(batch)

        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        next_groupmate_obs = [ModelUtils.list_to_tensor_list(_list_obs) for _list_obs in next_groupmate_obs]
        # Expand dimensions of next critic obs
        next_groupmate_obs = [[_obs.unsqueeze(0) for _obs in _list_obs] for _list_obs in next_groupmate_obs]

        if agent_id in self.value_memory_dict:
            # The agent_id should always be in both since they are added together
            init_value_mem = self.value_memory_dict[agent_id]
            init_baseline_mem = self.baseline_memory_dict[agent_id]
        else:
            init_value_mem = torch.zeros((1, 1, self.critic.memory_size)) if self.policy.use_recurrent else None
            init_baseline_mem = torch.zeros((1, 1, self.critic.memory_size)) if self.policy.use_recurrent else None

        all_obs = [current_obs, *groupmate_obs] if groupmate_obs is not None else [current_obs]
        all_next_value_mem: Optional[AgentBufferField] = None
        all_next_baseline_mem: Optional[AgentBufferField] = None
        with torch.no_grad():
            if self.policy.use_recurrent:
                (
                    value_estimates,
                    baseline_estimates,
                    all_next_value_mem,
                    all_next_baseline_mem,
                    next_value_mem,
                    next_baseline_mem,
                ) = self._evaluate_by_sequence_team(
                    current_obs,
                    groupmate_obs,
                    groupmate_actions,
                    init_value_mem,
                    init_baseline_mem,
                )
            else:
                value_estimates, next_value_mem = self.critic.critic_pass(
                    all_obs,
                    init_value_mem,
                    sequence_length=batch.num_experiences,
                )
                groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
                baseline_estimates, next_baseline_mem = self.critic.baseline(
                    current_obs,
                    groupmate_obs_and_actions,
                    init_baseline_mem,
                    sequence_length=batch.num_experiences,
                )
        # Store the memory for the next trajectory
        self.value_memory_dict[agent_id] = next_value_mem
        self.baseline_memory_dict[agent_id] = next_baseline_mem

        all_next_obs = [next_obs, *next_groupmate_obs] if next_groupmate_obs is not None else [next_obs]

        next_value_estimates, _ = self.critic.critic_pass(
            all_next_obs,
            next_value_mem,
            sequence_length=1,
        )

        for name, estimate in baseline_estimates.items():
            baseline_estimates[name] = ModelUtils.to_numpy(estimate)

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)

        # the base line and V shpuld  not be on the same done flag
        for name, estimate in next_value_estimates.items():
            next_value_estimates[name] = ModelUtils.to_numpy(estimate)

        if done:
            for k in next_value_estimates:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimates[k][-1] = 0.0

        return (
            value_estimates,
            baseline_estimates,
            next_value_estimates,
            all_next_value_mem,
            all_next_baseline_mem,
        )


class TorchETPOCAOptimizer(TorchPOCAOptimizer):
    class POCAValueNetwork(TorchPOCAOptimizer.POCAValueNetwork):
        """
        The POCAValueNetwork uses the MultiAgentNetworkBody to compute the value
        and POCA baseline for a variable number of agents in a group that all
        share the same observation and action space.
        """

        def __init__(
            self,
            stream_names: list[str],
            observation_specs: list[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            torch.nn.Module.__init__(self)
            self.network_body = MultiAgentNetworkBody(
                observation_specs,
                network_settings,
                action_spec,
            )
            if network_settings.memory is not None:
                encoding_size = network_settings.memory.memory_size // 2
            else:
                encoding_size = network_settings.hidden_units

            self.value_heads = ValueHeads(stream_names, encoding_size + 1, 1)

            trace_flat_output_size = 0
            trace_param_shapes = {}
            for name, param in self.value_heads.named_parameters():
                trace_param_shapes[name] = param.shape
                trace_flat_output_size += param.numel()
            self.trace_param_shapes = trace_param_shapes
            self.trace_heads = ValueHeads(stream_names, encoding_size + 1, trace_flat_output_size)
            # The + 1 is for the normalized number of agents

        def trace_pass(
            self,
            obs: list[list[torch.Tensor]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            """
            A centralized value function. It calls the forward pass of MultiAgentNetworkBody
            with just the states of all agents.
            :param obs: List of observations for all agents in group
            :param memories: If using memory, a Tensor of initial memories.
            :param sequence_length: If using memory, the sequence length.
            :return: A Tuple of Dict of reward stream to tensor and critic memories.
            """
            encoding, memories = self.network_body(
                obs_only=obs,
                obs=[],
                actions=[],
                memories=memories,
                sequence_length=sequence_length,
            )

            trace_outputs = self.trace_heads(encoding)
            return trace_outputs, memories

        def baseline_trace_pass(
            self,
            obs_without_actions: list[torch.Tensor],
            obs_with_actions: tuple[list[list[torch.Tensor]], list[AgentAction]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
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
            (obs, actions) = obs_with_actions
            encoding, memories = self.network_body(
                obs_only=[obs_without_actions],
                obs=obs,
                actions=actions,
                memories=memories,
                sequence_length=sequence_length,
            )

            trace_outputs = self.trace_heads(encoding)
            return trace_outputs, memories

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        :param policy: A TorchPolicy object that will be updated by this POCA Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self._critic = TorchETPOCAOptimizer.POCAValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            network_settings=trainer_settings.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        # Move to GPU if needed
        self._critic.to(default_device())

        params = list(self.policy.actor.parameters()) + list(self.critic.parameters())
        self.hyperparameters: POCASettings = cast(
            "POCASettings",
            trainer_settings.hyperparameters,
        )
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params,
            lr=self.trainer_settings.hyperparameters.learning_rate,
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())
        self.value_memory_dict: dict[str, torch.Tensor] = {}
        self.baseline_memory_dict: dict[str, torch.Tensor] = {}

    def init_buffer_traces(self, batch: AgentBuffer, num_sequences: int, lambd: float) -> dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        old_baseline_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)],
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)],
            )
            old_baseline_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.baseline_estimates_key(name)],
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = [[ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs] for _groupmate_obs in groupmate_obs]

        groupmate_actions = AgentAction.group_from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0,
                len(batch[BufferKey.CRITIC_MEMORY]),
                self.policy.sequence_length,
            )
        ]

        baseline_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.BASELINE_MEMORY][i])
            for i in range(
                0,
                len(batch[BufferKey.BASELINE_MEMORY]),
                self.policy.sequence_length,
            )
        ]

        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)
            baseline_memories = torch.stack(baseline_memories).unsqueeze(0)

        all_obs = [current_obs, *groupmate_obs]
        values, _ = self.critic.critic_pass(
            all_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
        baselines, _ = self.critic.baseline(
            current_obs,
            groupmate_obs_and_actions,
            memories=baseline_memories,
            sequence_length=self.policy.sequence_length,
        )
        traces, _ = self.critic.trace_pass(
            all_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        baseline_traces, _ = self.critic.baseline_trace_pass(
            current_obs,
            groupmate_obs_and_actions,
            memories=baseline_memories,
            sequence_length=self.policy.sequence_length,
        )
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        baseline_loss = ModelUtils.trust_region_value_loss(
            baselines,
            old_baseline_values,
            returns,
            decay_eps,
            loss_masks,
            reduce=False,
        )
        value_loss = ModelUtils.trust_region_value_loss(
            values,
            old_values,
            returns,
            decay_eps,
            loss_masks,
            reduce=False,
        )
        # flatten into [R * B]
        value_loss = 0.5 * (value_loss + 0.5 * baseline_loss)

        critic_params = {
            name: param for name, param in self.critic.named_parameters() if not name.startswith("trace_head") and param.requires_grad
        }
        for rew_name in self.reward_signals:
            rew_traces = list(
                ModelUtils.to_numpy(traces[rew_name]) + 0.5 * ModelUtils.to_numpy(baseline_traces[rew_name]),
            )
            batch["trace_preds", rew_name] = [np.zeros_like(rew_traces[0]), *rew_traces[:-1]]
            for param_name in critic_params:
                batch["instantaneous_trace", rew_name, param_name] = []

        for rew_name, rew_loss in zip(self.reward_signals, value_loss, strict=True):
            chunk_start = 0
            while chunk_start < batch.num_experiences:
                chunk_size = min(chunk_start + 512, batch.num_experiences)
                chunk_end = chunk_start + min(chunk_start + 512, batch.num_experiences)
                grads = torch.autograd.grad(
                    rew_loss[chunk_start:chunk_end],
                    list(critic_params.values()),
                    grad_outputs=torch.eye(chunk_size),
                    is_grads_batched=True,
                    retain_graph=True,
                    materialize_grads=True,
                )
                grads = [ModelUtils.to_numpy(grad.view(chunk_size, -1)) for grad in grads]

                for idx, step in enumerate(batch[BufferKey.STEP][chunk_start:chunk_end]):
                    if step == 0:
                        # first step trace should always be 0
                        for p_idx, name in enumerate(critic_params):
                            batch["instantaneous_trace", rew_name, name].append(grads[p_idx][idx])
                    else:
                        for p_idx, name in enumerate(critic_params):
                            batch["instantaneous_trace", rew_name, name].append(
                                self.reward_signals[rew_name].gamma * lambd * batch["instantaneous_trace", rew_name, name][-1]
                                + grads[p_idx][idx],
                            )
                chunk_start = chunk_end

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int, original_batch: AgentBuffer, lambd: float) -> dict[str, float]:
        # def update(self, batch: AgentBuffer, num_sequences: int) -> dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        old_baseline_values = {}
        prev_traces = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)],
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)],
            )
            old_baseline_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.baseline_estimates_key(name)],
            )
            prev_traces[name] = 0.5 * ModelUtils.list_to_tensor(batch["trace_preds", name])

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = [[ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs] for _groupmate_obs in groupmate_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)
        groupmate_actions = AgentAction.group_from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0,
                len(batch[BufferKey.CRITIC_MEMORY]),
                self.policy.sequence_length,
            )
        ]

        baseline_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.BASELINE_MEMORY][i])
            for i in range(
                0,
                len(batch[BufferKey.BASELINE_MEMORY]),
                self.policy.sequence_length,
            )
        ]

        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)
            baseline_memories = torch.stack(baseline_memories).unsqueeze(0)

        log_probs, entropy = self.policy.evaluate_actions(
            current_obs,
            masks=act_masks,
            actions=actions,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        all_obs = [current_obs, *groupmate_obs]
        values, _ = self.critic.critic_pass(
            all_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
        baselines, _ = self.critic.baseline(
            current_obs,
            groupmate_obs_and_actions,
            memories=baseline_memories,
            sequence_length=self.policy.sequence_length,
        )
        traces, _ = self.critic.trace_pass(
            all_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        baseline_traces, _ = self.critic.baseline_trace_pass(
            current_obs,
            groupmate_obs_and_actions,
            memories=baseline_memories,
            sequence_length=self.policy.sequence_length,
        )
        trace_tensor = (torch.stack(list(traces.values()))).sum(dim=0) + (torch.stack(list(baseline_traces.values()))).sum(dim=0)

        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        baseline_loss = ModelUtils.trust_region_value_loss(
            baselines,
            old_baseline_values,
            returns,
            decay_eps,
            loss_masks,
            reduce=False,
        )
        value_loss = ModelUtils.trust_region_value_loss(
            values,
            old_values,
            returns,
            decay_eps,
            loss_masks,
            reduce=False,
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )

        loss = policy_loss - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)

        # leave the reward signal and batch dimension intact
        while value_loss.ndim > 2:
            value_loss = value_loss.sum(dim=-1)
        while baseline_loss.ndim > 2:
            baseline_loss = baseline_loss.sum(dim=-1)

        trace_value_loss = (0.5 * (value_loss + 0.5 * baseline_loss)).view(-1)

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        critic_params = {
            name: param for name, param in self.critic.named_parameters() if not name.startswith("trace_head") and param.requires_grad
        }
        value_head_params = {
            name: param for name, param in self.critic.named_parameters() if name.startswith("value_head") and param.requires_grad
        }
        trace_head_params = {
            name: param for name, param in self.critic.named_parameters() if name.startswith("trace_head") and param.requires_grad
        }
        grads = torch.autograd.grad(
            trace_value_loss,
            inputs=list(critic_params.values()),
            grad_outputs=torch.eye(trace_value_loss.shape[0]),
            retain_graph=True,
            is_grads_batched=True,
            allow_unused=True,
            materialize_grads=True,
        )
        grads = [grad.view(len(self.reward_signals), batch.num_experiences, grad.shape[1:]) for grad in grads]
        critic_traces = {}

        episodes = list(batch[BufferKey.EPISODE_ID])
        episode_steps = list(batch[BufferKey.STEP])

        for name, grad in zip(critic_params, grads, strict=True):
            for rew_idx, rew_name in enumerate(self.reward_signals):
                critic_traces[name, rew_name] = {(episode_steps[step], episodes[step]): g for step, g in enumerate(grad[rew_idx])}

        updated_trace_steps = []
        batch_indices_map = {}
        for idx, (step, episode) in enumerate(zip(original_batch[BufferKey.STEP], original_batch[BufferKey.EPISODE_ID], strict=True)):
            for b_idx, (b_step, b_episode) in enumerate(zip(episode_steps, episodes, strict=True)):
                if b_episode == episode and b_step == step:
                    batch_indices_map[b_step, b_episode] = (idx, b_idx)
                    updated_trace_steps.append(idx)

                    episodes.pop(b_idx)
                    episode_steps.pop(b_idx)
                    break

            if not episode_steps or not episodes:
                break

        for (step, episode), (idx, b_idx) in batch_indices_map.items():
            if (step + 1, episode) in batch_indices_map:
                next_idx, _ = batch_indices_map[step + 1, episode]
                for _, rew_name in critic_traces:
                    # update the trace target for the next timestep using the prediction from this timestep
                    original_batch["trace_preds", rew_name][next_idx] = ModelUtils.to_numpy(traces[rew_name][b_idx])

            if step == 0:
                loss_masks[b_idx] = torch.zeros_like(loss_masks[b_idx])
                continue

            for (name, rew_name), grad_dict in critic_traces.items():
                # moving mean
                original_batch["instantaneous_trace", rew_name, name][idx] = (
                    self.reward_signals[rew_name].gamma * lambd * original_batch["instantaneous_trace", rew_name, name][idx - 1]
                    + grad_dict[step, episode]
                )

        # reshape from [P, R, B] to [B, -1] (batch dimension first & flatten)
        grads = torch.stack(grads, dim=2).sum(dim=2).transpose(0, 1).view(batch.num_experiences, -1)
        trace_loss = (
            (
                grads + lambd * torch.stack([rew_signal.gamma * prev_traces[name] for name, rew_signal in self.reward_signals]).sum(dim=0)
            ).detach()
            - trace_tensor
        ) ** 2.0
        trace_loss = ModelUtils.masked_mean(trace_loss, loss_masks)
        trace_head_grads = torch.autograd.grad(trace_loss, list(trace_head_params.values()))

        self.optimizer.zero_grad()
        loss.backward()

        # combine the policy loss gradients with the updated trace gradients
        for name, param in critic_params.items():
            for rew_name in self.reward_signals:
                if name in value_head_params or name in trace_head_params:
                    # skip the value head since we are using the mixed trace instead
                    continue
                new_grad = torch.stack([original_batch["instantaneous_trace", rew_name, name][idx] for idx in updated_trace_steps]).mean(
                    dim=0,
                )
                if param.grad is None:
                    param.grad = new_grad
                else:
                    param.grad += new_grad

        trace_ptr = 0
        for name, param in value_head_params.items():
            shape = self.critic.trace_param_shapes[name]
            ptr_end = np.prod(shape)
            new_grads = trace_tensor[trace_ptr : trace_ptr + ptr_end].view(shape)
            if param.grad is None:
                param.grad = new_grads
            else:
                param.grad += new_grads
            trace_ptr += ptr_end
        if trace_ptr != len(trace_tensor):
            raise RuntimeError("Did not consume all trace predictions")

        for param, grad in zip(trace_head_params.values(), trace_head_grads, strict=True):
            param.grad = grad

        self.optimizer.step()
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Baseline Loss": baseline_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))

        return update_stats
