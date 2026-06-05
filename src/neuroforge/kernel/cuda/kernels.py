"""CUDA kernel extension points."""

from __future__ import annotations

__all__ = ["custom_kernels_enabled"]


def custom_kernels_enabled() -> bool:
    """Return whether custom CUDA kernels are enabled."""
    return False
