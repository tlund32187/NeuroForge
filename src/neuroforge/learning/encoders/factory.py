"""Factory helpers for learning encoders."""

from __future__ import annotations

from neuroforge.construction.registry import Registry
from neuroforge.learning.encoders.rate import RateEncoder

__all__ = ["build_encoder_registry"]


def build_encoder_registry() -> Registry:
    """Build a registry populated with built-in encoders."""
    registry = Registry("encoders")
    registry.register("rate", RateEncoder)
    return registry
