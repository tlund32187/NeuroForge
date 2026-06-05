"""Factory helpers for supervised losses."""

from __future__ import annotations

from neuroforge.construction.registry import Registry
from neuroforge.learning.losses.bce import BceLogitsLoss
from neuroforge.learning.losses.mse import MseCountLoss

__all__ = ["build_loss_registry"]


def build_loss_registry() -> Registry:
    """Build a registry populated with built-in supervised losses."""
    registry = Registry("losses")
    registry.register("mse_count", MseCountLoss)
    registry.register("bce_logits", BceLogitsLoss)
    return registry
