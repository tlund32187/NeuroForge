"""CUDA memory helpers."""

from __future__ import annotations

from typing import Any

from neuroforge.kernel.torch_utils import require_torch

__all__ = ["cuda_memory_stats"]


def cuda_memory_stats(device: Any = None) -> dict[str, int]:
    """Return CUDA memory stats, or zeros when CUDA is unavailable."""
    torch = require_torch()
    if not torch.cuda.is_available():
        return {
            "allocated": 0,
            "reserved": 0,
            "max_allocated": 0,
            "max_reserved": 0,
        }
    return {
        "allocated": int(torch.cuda.memory_allocated(device=device)),
        "reserved": int(torch.cuda.memory_reserved(device=device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device=device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device=device)),
    }
