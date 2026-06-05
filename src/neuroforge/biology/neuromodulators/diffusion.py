"""Neuromodulator diffusion helpers."""

from __future__ import annotations

__all__ = ["diffuse_scalar"]


def diffuse_scalar(value: float, *, decay: float) -> float:
    """Apply one scalar diffusion/decay step."""
    return value * decay
