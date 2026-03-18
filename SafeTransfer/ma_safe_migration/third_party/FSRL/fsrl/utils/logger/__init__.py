"""Logger package."""

from fsrl.utils.logger.base_logger import BaseLogger, DummyLogger
from fsrl.utils.logger.tb_logger import TensorboardLogger

try:
    from fsrl.utils.logger.wandb_logger import WandbLogger
except ImportError:
    class WandbLogger(TensorboardLogger):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "wandb is not installed. Please install wandb or use TensorboardLogger."
            )

__all__ = [
    "BaseLogger",
    "DummyLogger",
    "TensorboardLogger",
    "WandbLogger",
]
