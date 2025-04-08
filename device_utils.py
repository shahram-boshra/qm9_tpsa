# --- device_utils.py ---
import torch
import logging
import os
from exceptions import DeviceInitializationError 

logger = logging.getLogger(__name__)


def get_device(use_gpu: bool = True, use_tpu: bool = False) -> torch.device:
    """
    Determines and returns the appropriate device (CPU, GPU, or TPU).

    Args:
        use_gpu (bool): Whether to use GPU if available.
        use_tpu (bool): Whether to use TPU if available.

    Returns:
        torch.device: The determined device.

    Raises:
        DeviceInitializationError: If the device cannot be initialized.
    """
    try:
        if use_tpu and 'COLAB_TPU_ADDR' in os.environ:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            logger.info("Using TPU.")
        elif use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using GPU.")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU.")
        return device

    except Exception as e:
        logger.error(f"Error initializing device: {e}")
        raise DeviceInitializationError(f"Failed to initialize device: {e}")
