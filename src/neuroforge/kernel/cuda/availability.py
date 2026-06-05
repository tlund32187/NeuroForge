"""CUDA availability helpers."""

from __future__ import annotations

from typing import Any

from neuroforge.kernel.torch_utils import require_torch

__all__ = ["cuda_available", "current_cuda_device"]


def cuda_available() -> bool:
    """Return whether torch CUDA is available."""
    torch = require_torch()
    return bool(torch.cuda.is_available())


def current_cuda_device() -> Any | None:
    """Return the current CUDA device, or ``None`` when CUDA is unavailable."""
    torch = require_torch()
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda", torch.cuda.current_device())
