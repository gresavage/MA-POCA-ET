
from mlagents_envs.base_env import BehaviorSpec

from mapoca.trainers.exception import UnityTrainerException
from mapoca.trainers.settings import RewardSignalSettings, RewardSignalType
from mapoca.trainers.torch.components.reward_providers.base_reward_provider import BaseRewardProvider
from mapoca.trainers.torch.components.reward_providers.curiosity_reward_provider import CuriosityRewardProvider
from mapoca.trainers.torch.components.reward_providers.extrinsic_reward_provider import ExtrinsicRewardProvider
from mapoca.trainers.torch.components.reward_providers.gail_reward_provider import GAILRewardProvider
from mapoca.trainers.torch.components.reward_providers.rnd_reward_provider import RNDRewardProvider

NAME_TO_CLASS: dict[RewardSignalType, type[BaseRewardProvider]] = {
    RewardSignalType.EXTRINSIC: ExtrinsicRewardProvider,
    RewardSignalType.CURIOSITY: CuriosityRewardProvider,
    RewardSignalType.GAIL: GAILRewardProvider,
    RewardSignalType.RND: RNDRewardProvider,
}


def create_reward_provider(
    name: RewardSignalType, specs: BehaviorSpec, settings: RewardSignalSettings,
) -> BaseRewardProvider:
    """
    Creates a reward provider class based on the name and config entry provided as a dict.
    :param name: The name of the reward signal
    :param specs: The BehaviorSpecs of the policy
    :param settings: The RewardSignalSettings for that reward signal
    :return: The reward signal class instantiated.
    """
    rcls = NAME_TO_CLASS.get(name)
    if not rcls:
        raise UnityTrainerException(f"Unknown reward signal type {name}")

    return rcls(specs, settings)
