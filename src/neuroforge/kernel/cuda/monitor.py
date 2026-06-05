"""CUDA synchronization helpers for timing and monitoring."""

from __future__ import annotations

from typing import Any

from neuroforge.kernel.torch_utils import require_torch

__all__ = ["cuda_maybe_sync"]


def cuda_maybe_sync(device: Any = None) -> None:
    """Synchronize pending CUDA work when CUDA applies."""
    torch = require_torch()
    if not torch.cuda.is_available():
        return
    if device is not None:
        device_type = str(getattr(device, "type", device))
        if not device_type.startswith("cuda"):
            return
        torch.cuda.synchronize(device=device)
        return
    torch.cuda.synchronize()
