from pathlib import Path

from mapoca.trainers.exception import UnityTrainerException


def validate_existing_directories(
    output_path: str,
    resume: bool,
    force: bool,
    init_path: str | None = None,
) -> None:
    """
    Validates that if the run_id model exists, we do not overwrite it unless --force is specified.
    Throws an exception if resume isn't specified and run_id exists. Throws an exception
    if --resume is specified and run-id was not found.
    :param model_path: The model path specified.
    :param summary_path: The summary path to be used.
    :param resume: Whether or not the --resume flag was passed.
    :param force: Whether or not the --force flag was passed.
    """
    output_path_exists = Path(output_path).is_dir()

    if output_path_exists:
        if not resume and not force:
            raise UnityTrainerException(
                "Previous data from this run ID was found. "
                "Either specify a new run ID, use --resume to resume this run, "
                "or use the --force parameter to overwrite existing data.",
            )
    elif resume:
        raise UnityTrainerException(
            "Previous data from this run ID was not found. Train a new run by removing the --resume flag.",
        )

    # Verify init path if specified.
    if init_path is not None and not Path(init_path).is_dir():
        raise UnityTrainerException(
            f"Could not initialize from {init_path}. Make sure models have already been saved with that run ID.",
        )
