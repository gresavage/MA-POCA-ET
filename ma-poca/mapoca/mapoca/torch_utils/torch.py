import contextlib
import os

from distutils.version import LooseVersion

import pkg_resources

from mlagents_envs.logging_util import get_logger

from mapoca.torch_utils import cpu_utils
from mapoca.trainers.settings import TorchSettings

logger = get_logger(__name__)


def assert_torch_installed():
    # Check that torch version 1.6.0 or later has been installed. If not, refer
    # user to the PyTorch webpage for install instructions.
    torch_pkg = None
    with contextlib.suppress(pkg_resources.DistributionNotFound):
        torch_pkg = pkg_resources.get_distribution("torch")
    assert torch_pkg is not None and LooseVersion(torch_pkg.version) >= LooseVersion("1.6.0"), (  # noqa: PT018
        "A compatible version of PyTorch was not installed. Please visit the PyTorch homepage "
        "(https://pytorch.org/get-started/locally/) and follow the instructions to install. "
        "Version 1.6.0 and later are supported."
    )


assert_torch_installed()

# This should be the only place that we import torch directly.
# Everywhere else is caught by the banned-modules setting for flake8
import torch  # noqa: E402

torch.set_num_threads(cpu_utils.get_num_threads_to_use())
os.environ["KMP_BLOCKTIME"] = "0"


_device = torch.device("cpu")


def set_torch_config(torch_settings: TorchSettings) -> None:
    global _device  # noqa: PLW0603

    device_str = ("cuda" if torch.cuda.is_available() else "cpu") if torch_settings.device is None else torch_settings.device

    _device = torch.device(device_str)

    torch.set_default_device(_device)
    torch.set_default_dtype(torch.float32)
    logger.debug(f"default Torch device: {_device}")


# Initialize to default settings
set_torch_config(TorchSettings(device=None))

nn = torch.nn

import torchtune as torchtune  # noqa: E402, PLC0414


def default_device():
    return _device
