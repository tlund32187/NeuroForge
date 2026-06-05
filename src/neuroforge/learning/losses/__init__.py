"""Supervised loss functions used by learning workflows."""

from neuroforge.learning.losses.bce import BceLogitsLoss
from neuroforge.learning.losses.mse import MseCountLoss
from neuroforge.learning.losses.registry import build_loss_registry

__all__ = [
    "BceLogitsLoss",
    "MseCountLoss",
    "build_loss_registry",
]
