"""CUDA stream helpers."""

from __future__ import annotations

from typing import Any

from neuroforge.kernel.torch_utils import require_torch

__all__ = ["current_stream"]


def current_stream(device: Any = None) -> Any | None:
    """Return the current CUDA stream, or ``None`` when CUDA is unavailable."""
    torch = require_torch()
    if not torch.cuda.is_available():
        return None
    return torch.cuda.current_stream(device=device)
