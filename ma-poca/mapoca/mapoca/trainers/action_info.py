from typing import Any, NamedTuple

import numpy as np

from mlagents_envs.base_env import AgentId

ActionInfoOutputs = dict[str, np.ndarray]


class ActionInfo(NamedTuple):
    """
    A NamedTuple containing actions and related quantities to the policy forward
    pass. Additionally contains the agent ids in the corresponding DecisionStep
    :param action: The action output of the policy
    :param env_action: The possibly clipped action to be executed in the environment
    :param outputs: Dict of all quantities associated with the policy forward pass
    :param agent_ids: List of int agent ids in DecisionStep.
    """

    action: Any
    env_action: Any
    outputs: ActionInfoOutputs
    agent_ids: list[AgentId]

    @staticmethod
    def empty() -> "ActionInfo":
        return ActionInfo([], [], {}, [])
