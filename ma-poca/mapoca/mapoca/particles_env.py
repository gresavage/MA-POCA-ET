from collections.abc import Mapping as MappingType
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from mlagents_envs.base_env import (
    ActionSpec,
    ActionTuple,
    AgentId,
    BaseEnv,
    BehaviorName,
    BehaviorSpec,
    DecisionSteps,
    DimensionProperty,
    ObservationSpec,
    ObservationType,
    TerminalSteps,
)
from mlagents_envs.communicator_objects.capabilities_pb2 import UnityRLCapabilitiesProto

if TYPE_CHECKING:
    from pettingzoo import ParallelEnv
    from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv


def _make_env(scenario_name: str, benchmark: bool = False) -> "ParallelEnv":  # noqa: ARG001
    """
    Create a MultiAgentEnv object as env.

    This can be used similar to a gym environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents.

    Parameters
    ----------
    scenario_name   :   str
        name of the scenario from ./scenarios/ to be Returns (without the .py extension)
    benchmark       :   bool
        whether you want to produce benchmarking data (usually only done during evaluation)

    Returns
    -------
    ParallelEnv

    Raises
    ------
    ValueError
        If an invalid scenario_name is given
    """
    from pettingzoo import mpe  # noqa: PLC0415

    for file in Path(mpe.__file__).parent.iterdir():
        if file.name.startswith(scenario_name):
            return getattr(mpe, file.stem).parallel_env()
    raise ValueError(f"{scenario_name} is not a valid MPE scenario.")


class ParticlesEnvironment(BaseEnv):
    def __init__(self, name: str = "simple_spread", worker_id: int = 0):
        self._obs: Optional[dict[str, np.ndarray]] = None
        self._rew: Optional[dict[str, int]] = None
        self._done: Optional[dict[str, bool]] = None
        self._actions: Optional[dict[str, int | np.ndarray]] = None
        self._name = name
        self._env = _make_env(name)
        self._env.discrete_action_input = not getattr(getattr(self._env, "unwrapped", self._env), "continuous_actions", False)
        self._id_to_agent = dict(enumerate(self._env.possible_agents))
        self._worker_id = worker_id

        # :(
        self.academy_capabilities = UnityRLCapabilitiesProto()
        self.academy_capabilities.baseRLCapabilities = True
        self.academy_capabilities.concatenatedPngObservations = True
        self.academy_capabilities.compressedChannelMapping = True
        self.academy_capabilities.hybridActions = True
        self.academy_capabilities.trainingAnalytics = True
        self.academy_capabilities.variableLengthObservation = True
        self.academy_capabilities.multiAgentGroups = True
        self.count = 0
        self.episode_count = 0

    def step(self) -> None:
        if self._actions is None:
            base_env = getattr(self._env, "unwrapped", self._env)
            if self._env.discrete_action_input:
                actions = dict.fromkeys(base_env.action_spaces.keys(), 0)
            else:
                actions = {agent: np.zeros(space.shape, dtype=space.dtype) for agent, space in base_env.action_spaces.items()}
        else:
            actions = self._actions

        self._obs, self._rew, self._done, _ = self._env.step(actions)

        if self.count >= 25:
            self._done = dict.fromkeys(self._id_to_agent.values(), True)
        self.count += 1

    def reset(self) -> None:
        base_env: SimpleEnv = getattr(self._env, "unwrapped", self._env)
        self._obs = self._env.reset()
        self._rew = base_env.rewards
        self._done = base_env.dones
        self._reset_actions()
        self.episode_count += 1
        self.count = 0

    def close(self) -> None:
        self._env.close()

    @property
    def behavior_specs(self) -> MappingType[str, BehaviorSpec]:
        if self._env.discrete_action_input:
            act_spec = ActionSpec(0, (list(self._env.action_spaces.values())[0].n,))  # noqa: RUF015
        else:
            act_spec = ActionSpec(np.prod(list(self._env.action_spaces.values())[0].shape).item(), (0,))  # noqa: RUF015
        return {
            self._name: BehaviorSpec(
                [
                    ObservationSpec(
                        list(self._env.observation_spaces.values())[0].shape,  # noqa: RUF015
                        (DimensionProperty.NONE,),
                        ObservationType.DEFAULT,
                        "obs_0",
                    ),
                ],
                act_spec,
            ),
        }

    def set_actions(self, behavior_name: BehaviorName, action: ActionTuple) -> None:
        assert behavior_name == self._name
        actions = [x.item() for x in action.discrete] if self._env.discrete_action_input else action.continuous
        self._actions = {agent: actions[i] for i, agent in self._id_to_agent.items()}

    def set_action_for_agent(
        self,
        behavior_name: BehaviorName,
        agent_id: AgentId,
        action: ActionTuple,
    ) -> None:
        assert behavior_name == self._name
        if self._actions is None:
            self._reset_actions()
        agent = self._id_to_agent[agent_id]
        agent_action = action.discrete[0].item() if self._env.discrete_action_input else action.continuous[0]
        self._actions[agent] = agent_action

    def _reset_actions(self) -> None:
        if self._env.discrete_action_input:
            self._actions = dict.fromkeys(self._env.action_spaces.keys(), 0)
        else:
            self._actions = {agent: np.zeros(space.shape, dtype=space.dtype) for agent, space in self._env.action_spaces.items()}

    def get_steps(
        self,
        behavior_name: BehaviorName,
    ) -> tuple[DecisionSteps, TerminalSteps]:
        terminal_steps = self.get_terminal_steps()
        if any(self._done.values()):
            # if any is done, reset the environment and
            # get the next steps
            self.reset()
        decision_steps = self.get_decision_steps()
        return decision_steps, terminal_steps

    def get_decision_steps(self) -> DecisionSteps:
        reward_scale = 1
        decision_obs = np.array(
            [self._obs[agent] for agent, done in self._done.items() if not done],
            dtype=np.float32,
        )
        decision_reward = np.array(
            [0 for done in self._done.values() if not done],
            dtype=np.float32,
        )
        decision_id = np.array([i for i, agent in self._id_to_agent.items() if not self._done[agent]])
        decision_group_reward = np.array(
            [self._rew[agent] * reward_scale for agent, done in self._done.items() if not done],
            dtype=np.float32,
        )
        decision_group_id = np.array(
            [1 for done in self._done.values() if not done],
        )
        return DecisionSteps(
            [decision_obs],
            decision_reward,
            decision_id,
            None,
            decision_group_id,
            decision_group_reward,
        )

    def get_terminal_steps(self) -> TerminalSteps:
        reward_scale = 1.0
        terminal_obs = np.array(
            [self._obs[agent] for agent, done in self._done.items() if done],
            dtype=np.float32,
        )
        terminal_reward = np.array(
            [0.0 for done in self._done.values() if done],
            dtype=np.float32,
        )
        terminal_id = np.array([i for i, agent in self._id_to_agent.items() if self._done[agent]])
        terminal_group_reward = np.array(
            [self._rew[agent] * reward_scale for agent, done in self._done.items() if done],
            dtype=np.float32,
        )
        terminal_group_id = np.array([1 for done in self._done.values() if done])
        # TODO : Figureout the type of interruption
        terminal_interruption = np.array(
            [False for done in self._done.values() if done],
        )
        return TerminalSteps(
            [terminal_obs],
            terminal_reward,
            terminal_interruption,
            terminal_id,
            terminal_group_id,
            terminal_group_reward,
        )
