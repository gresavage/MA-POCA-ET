"""Contains exceptions for the trainers package."""


class TrainerError(Exception):
    """Any error related to the trainers in the ML-Agents Toolkit."""


class TrainerConfigError(Exception):
    """Any error related to the configuration of trainers in the ML-Agents Toolkit."""


class TrainerConfigWarning(Warning):
    """Any warning related to the configuration of trainers in the ML-Agents Toolkit."""


class CurriculumError(TrainerError):
    """Any error related to training with a curriculum."""


class CurriculumLoadingError(CurriculumError):
    """Any error related to loading the Curriculum config file."""


class CurriculumConfigError(CurriculumError):
    """Any error related to processing the Curriculum config file."""


class MetaCurriculumError(TrainerError):
    """Any error related to the configuration of a metacurriculum."""


class SamplerException(TrainerError):  # noqa: N818
    """Related to errors with the sampler actions."""


class UnityTrainerException(TrainerError):  # noqa: N818
    """Related to errors with the Trainer."""
