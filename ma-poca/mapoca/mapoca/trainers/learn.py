# # Unity ML-Agents Toolkit
import json

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np

import mlagents_envs
import yaml

from mlagents_envs import logging_util
from mlagents_envs.base_env import BaseEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.timers import (
    add_metadata as add_timer_metadata,
    get_timer_tree,
    hierarchical_timer,
)

import mapoca.trainers

from mapoca import torch_utils
from mapoca.plugins.stats_writer import register_stats_writer_plugins
from mapoca.registry_entries import mapoca_registry
from mapoca.trainers.cli_utils import parser
from mapoca.trainers.directory_utils import validate_existing_directories
from mapoca.trainers.environment_parameter_manager import EnvironmentParameterManager
from mapoca.trainers.settings import RunOptions
from mapoca.trainers.stats import StatsReporter
from mapoca.trainers.subprocess_env_manager import SubprocessEnvManager
from mapoca.trainers.trainer import TrainerFactory
from mapoca.trainers.trainer_controller import TrainerController
from mapoca.trainers.training_status import GlobalTrainingStatus

logger = logging_util.get_logger(__name__)

TRAINING_STATUS_FILE_NAME = "training_status.json"


def get_version_string() -> str:
    return f""" Version information:
  ml-agents: {mapoca.trainers.__version__},
  ml-agents-envs: {mlagents_envs.__version__},
  Communicator API: {UnityEnvironment.API_VERSION},
  PyTorch: {torch_utils.torch.__version__}"""


def parse_command_line(argv: Optional[list[str]] = None) -> RunOptions:
    args = parser.parse_args(argv)
    return RunOptions.from_argparse(args)


def run_training(run_seed: int, options: RunOptions) -> None:
    """
    Launches training session.
    :param run_seed: Random seed used for training.
    :param options: parsed command line arguments.
    """
    with hierarchical_timer("run_training.setup"):
        torch_utils.set_torch_config(options.torch_settings)
        checkpoint_settings = options.checkpoint_settings
        env_settings = options.env_settings
        engine_settings = options.engine_settings

        run_logs_dir = checkpoint_settings.run_logs_dir
        port: Optional[int] = env_settings.base_port
        # Check if directory exists
        validate_existing_directories(
            checkpoint_settings.write_path,
            checkpoint_settings.resume,
            checkpoint_settings.force,
            checkpoint_settings.maybe_init_path,
        )
        # Make run logs directory
        Path(run_logs_dir).mkdir(parents=True, exist_ok=True)
        # Load any needed states
        if checkpoint_settings.resume:
            GlobalTrainingStatus.load_state(
                Path(run_logs_dir) / "training_status.json",
            )

        # Configure Tensorboard Writers and StatsReporter
        stats_writers = register_stats_writer_plugins(options)
        for sw in stats_writers:
            StatsReporter.add_writer(sw)

        env_factory = create_environment_factory(
            env_settings.env_name,
            engine_settings.no_graphics,
            run_seed,
            port,
            env_settings.env_args,
            Path(run_logs_dir).absolute(),  # Unity environment requires absolute path
        )

        env_manager = SubprocessEnvManager(env_factory, options, env_settings.num_envs)
        env_parameter_manager = EnvironmentParameterManager(
            options.environment_parameters,
            run_seed,
            restore=checkpoint_settings.resume,
        )

        trainer_factory = TrainerFactory(
            trainer_config=options.behaviors,
            output_path=checkpoint_settings.write_path,
            train_model=not checkpoint_settings.inference,
            load_model=checkpoint_settings.resume,
            seed=run_seed,
            param_manager=env_parameter_manager,
            init_path=checkpoint_settings.maybe_init_path,
            multi_gpu=False,
        )
        # Create controller and begin training.
        tc = TrainerController(
            trainer_factory,
            checkpoint_settings.write_path,
            checkpoint_settings.run_id,
            env_parameter_manager,
            not checkpoint_settings.inference,
            run_seed,
        )

    # Begin training
    try:
        tc.start_learning(env_manager)
    finally:
        env_manager.close()
        write_run_options(checkpoint_settings.write_path, options)
        write_timing_tree(run_logs_dir)
        write_training_status(run_logs_dir)


def write_run_options(output_dir: str, run_options: RunOptions) -> None:
    run_options_path = Path(output_dir) / "configuration.yaml"
    try:
        with run_options_path.open("w") as f:
            try:
                yaml.dump(run_options.as_dict(), f, sort_keys=False)
            except TypeError:  # Older versions of pyyaml don't support sort_keys
                yaml.dump(run_options.as_dict(), f)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save configuration to {run_options_path}. Make sure the directory exists",
        )


def write_training_status(output_dir: str) -> None:
    GlobalTrainingStatus.save_state(Path(output_dir) / TRAINING_STATUS_FILE_NAME)


def write_timing_tree(output_dir: str) -> None:
    timing_path = Path(output_dir) / "timers.json"
    try:
        with Path(timing_path).open("w", encoding="utf-8") as f:
            json.dump(get_timer_tree(), f, indent=4)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save to {timing_path}. Make sure the directory exists",
        )


def create_environment_factory(
    env_name: Optional[str],
    no_graphics: bool,
    seed: int,
    start_port: Optional[int],
    env_args: Optional[list[str]],
    log_folder: str,
) -> Callable[[int, list[SideChannel]], BaseEnv]:
    def create_unity_environment(
        worker_id: int,
        side_channels: list[SideChannel],
    ) -> UnityEnvironment:
        # Make sure that each environment gets a different seed
        env_seed = seed + worker_id
        return mapoca_registry[env_name].make(
            worker_id=worker_id,
            seed=env_seed,
            no_graphics=no_graphics,
            base_port=start_port,
            additional_args=env_args,
            side_channels=side_channels,
            log_folder=log_folder,
        )

    return create_unity_environment


def run_cli(options: RunOptions) -> None:
    try:
        print(
            """

                        ▄▄▄▓▓▓▓
                   ╓▓▓▓▓▓▓█▓▓▓▓▓
              ,▄▄▄m▀▀▀'  ,▓▓▓▀▓▓▄                           ▓▓▓  ▓▓▌
            ▄▓▓▓▀'      ▄▓▓▀  ▓▓▓      ▄▄     ▄▄ ,▄▄ ▄▄▄▄   ,▄▄ ▄▓▓▌▄ ▄▄▄    ,▄▄
          ▄▓▓▓▀        ▄▓▓▀   ▐▓▓▌     ▓▓▌   ▐▓▓ ▐▓▓▓▀▀▀▓▓▌ ▓▓▓ ▀▓▓▌▀ ^▓▓▌  ╒▓▓▌
        ▄▓▓▓▓▓▄▄▄▄▄▄▄▄▓▓▓      ▓▀      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌   ▐▓▓▄ ▓▓▌
        ▀▓▓▓▓▀▀▀▀▀▀▀▀▀▀▓▓▄     ▓▓      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌    ▐▓▓▐▓▓
          ^█▓▓▓        ▀▓▓▄   ▐▓▓▌     ▓▓▓▓▄▓▓▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▓▄    ▓▓▓▓`
            '▀▓▓▓▄      ^▓▓▓  ▓▓▓       └▀▀▀▀ ▀▀ ^▀▀    `▀▀ `▀▀   '▀▀    ▐▓▓▌
               ▀▀▀▀▓▄▄▄   ▓▓▓▓▓▓,                                      ▓▓▓▓▀
                   `▀█▓▓▓▓▓▓▓▓▓▌
                        ¬`▀▀▀█▓

        """,
        )
    except Exception:  # noqa: BLE001
        print("\n\n\tUnity Technologies\n")
    print(get_version_string())

    log_level = logging_util.DEBUG if options.debug else logging_util.INFO

    logging_util.set_log_level(log_level)

    logger.debug("Configuration for this run:")
    logger.debug(json.dumps(options.as_dict(), indent=4))

    # Options deprecation warnings
    if options.checkpoint_settings.load_model:
        logger.warning(
            "The --load option has been deprecated. Please use the --resume option instead.",
        )
    if options.checkpoint_settings.train_model:
        logger.warning(
            "The --train option has been deprecated. Train mode is now the default. Use --inference to run in inference mode.",
        )

    run_seed = options.env_settings.seed

    # Add some timer metadata
    add_timer_metadata("mlagents_version", mapoca.trainers.__version__)
    add_timer_metadata("mlagents_envs_version", mlagents_envs.__version__)
    add_timer_metadata("communication_protocol_version", UnityEnvironment.API_VERSION)
    add_timer_metadata("pytorch_version", torch_utils.torch.__version__)
    add_timer_metadata("numpy_version", np.__version__)

    if options.env_settings.seed == -1:
        run_seed = np.random.randint(0, 10000)
        logger.debug(f"run_seed set to {run_seed}")
    run_training(run_seed, options)


def main():
    run_cli(parse_command_line())


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
