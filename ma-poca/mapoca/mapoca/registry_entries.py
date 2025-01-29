from typing import Any, Optional

from mlagents_envs.base_env import BaseEnv
from mlagents_envs.registry import UnityEnvRegistry
from mlagents_envs.registry.base_registry_entry import BaseRegistryEntry

from mapoca.particles_env import ParticlesEnvironment

mapoca_registry = UnityEnvRegistry()
mapoca_registry.register_from_yaml("https://storage.googleapis.com/mlagents-test-environments/1.1.0/manifest.yaml")


class ParticleEnvEntry(BaseRegistryEntry):
    def __init__(
        self,
        identifier: str,
        expected_reward: Optional[float],
        description: Optional[str],
    ):
        super().__init__(identifier, expected_reward, description)

    def make(self, **kwargs: Any) -> BaseEnv:  # noqa: PLR6301
        return ParticlesEnvironment(worker_id=kwargs["worker_id"])


mapoca_registry.register(
    ParticleEnvEntry(
        "ParticlesEnv",
        -160,
        "The particles environment from https://github.com/openai/multiagent-particle-envs",
    ),
)
