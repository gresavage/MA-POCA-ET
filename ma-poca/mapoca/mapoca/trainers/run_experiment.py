import argparse

from typing import Optional

from mapoca.trainers.cli_utils import load_config
from mapoca.trainers.learn import run_cli
from mapoca.trainers.settings import RunOptions


def parse_command_line(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("experiment_config_path")
    return parser.parse_args(argv)


def main():
    """
    Provides an alternative CLI interface to mlagents-learn, 'mlagents-run-experiment'.
    Accepts a JSON/YAML formatted mapoca.trainers.learn.RunOptions object, and executes
    the run loop as defined in mapoca.trainers.learn.run_cli.
    """
    args = parse_command_line()
    expt_config = load_config(args.experiment_config_path)
    run_cli(RunOptions.from_dict(expt_config))


if __name__ == "__main__":
    main()
