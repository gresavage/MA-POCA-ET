from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import NamedTuple

from mlagents_envs.base_env import BehaviorName, BehaviorSpec, DecisionSteps, TerminalSteps
from mlagents_envs.logging_util import get_logger
from mlagents_envs.side_channel.stats_side_channel import EnvironmentStats

from mapoca.trainers.action_info import ActionInfo
from mapoca.trainers.agent_processor import AgentManager, AgentManagerQueue
from mapoca.trainers.policy import Policy
from mapoca.trainers.settings import TrainerSettings

AllStepResult = dict[BehaviorName, tuple[DecisionSteps, TerminalSteps]]
AllGroupSpec = dict[BehaviorName, BehaviorSpec]

logger = get_logger(__name__)


class EnvironmentStep(NamedTuple):
    current_all_step_result: AllStepResult
    worker_id: int
    brain_name_to_action_info: dict[BehaviorName, ActionInfo]
    environment_stats: EnvironmentStats

    @property
    def name_behavior_ids(self) -> Iterable[BehaviorName]:
        return self.current_all_step_result.keys()

    @staticmethod
    def empty(worker_id: int) -> "EnvironmentStep":
        return EnvironmentStep({}, worker_id, {}, {})


class EnvManager(ABC):
    def __init__(self):
        self.policies: dict[BehaviorName, Policy] = {}
        self.agent_managers: dict[BehaviorName, AgentManager] = {}
        self.first_step_infos: list[EnvironmentStep] = []

    def set_policy(self, brain_name: BehaviorName, policy: Policy) -> None:
        self.policies[brain_name] = policy
        if brain_name in self.agent_managers:
            self.agent_managers[brain_name].policy = policy

    def set_agent_manager(
        self,
        brain_name: BehaviorName,
        manager: AgentManager,
    ) -> None:
        self.agent_managers[brain_name] = manager

    @abstractmethod
    def _step(self) -> list[EnvironmentStep]:
        pass

    @abstractmethod
    def _reset_env(self, config: dict | None = None) -> list[EnvironmentStep]:
        pass

    def reset(self, config: dict | None = None) -> int:
        for manager in self.agent_managers.values():
            manager.end_episode()
        # Save the first step infos, after the reset.
        # They will be processed on the first advance().
        self.first_step_infos = self._reset_env(config)
        return len(self.first_step_infos)

    @abstractmethod
    def set_env_parameters(self, config: dict | None = None) -> None:
        """
        Sends environment parameter settings to C# via the
        EnvironmentParametersSideChannel.
        :param config: Dict of environment parameter keys and values.
        """

    def on_training_started(  # noqa: B027
        self,
        behavior_name: str,
        trainer_settings: TrainerSettings,
    ) -> None:
        """
        Handle traing starting for a new behavior type. Generally nothing is necessary here.
        :param behavior_name:
        :param trainer_settings:
        :return:
        """

    @property
    @abstractmethod
    def training_behaviors(self) -> dict[BehaviorName, BehaviorSpec]:
        pass

    @abstractmethod
    def close(self):
        pass

    def get_steps(self) -> list[EnvironmentStep]:
        """
        Updates the policies, steps the environments, and returns the step information from the environments.
        Calling code should pass the returned EnvironmentSteps to process_steps() after calling this.
        :return: The list of EnvironmentSteps.
        """
        # If we had just reset, process the first EnvironmentSteps.
        # Note that we do it here instead of in reset() so that on the very first reset(),
        # we can create the needed AgentManagers before calling advance() and processing the EnvironmentSteps.
        if self.first_step_infos:
            self._process_step_infos(self.first_step_infos)
            self.first_step_infos = []
        # Get new policies if found. Always get the latest policy.
        for brain_name in self.agent_managers:
            policy = None
            try:
                # We make sure to empty the policy queue before continuing to produce steps.
                # This halts the trainers until the policy queue is empty.
                while True:
                    policy = self.agent_managers[brain_name].policy_queue.get_nowait()
            except AgentManagerQueue.Empty:
                if policy is not None:
                    self.set_policy(brain_name, policy)
        # Step the environments
        return self._step()

    def process_steps(self, new_step_infos: list[EnvironmentStep]) -> int:
        # Add to AgentProcessor
        return self._process_step_infos(new_step_infos)

    def _process_step_infos(self, step_infos: list[EnvironmentStep]) -> int:
        for step_info in step_infos:
            for name_behavior_id in step_info.name_behavior_ids:
                if name_behavior_id not in self.agent_managers:
                    logger.warning(
                        f"Agent manager was not created for behavior id {name_behavior_id}.",
                    )
                    continue
                decision_steps, terminal_steps = step_info.current_all_step_result[name_behavior_id]
                self.agent_managers[name_behavior_id].add_experiences(
                    decision_steps,
                    terminal_steps,
                    step_info.worker_id,
                    step_info.brain_name_to_action_info.get(
                        name_behavior_id,
                        ActionInfo.empty(),
                    ),
                )

                self.agent_managers[name_behavior_id].record_environment_stats(
                    step_info.environment_stats,
                    step_info.worker_id,
                )
        return len(step_infos)
