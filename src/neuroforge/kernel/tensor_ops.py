"""Common tensor operation helpers."""

from __future__ import annotations

from typing import Any

from neuroforge.kernel.torch_utils import require_torch

__all__ = ["as_tensor", "zeros_like_shape"]


def as_tensor(value: Any, *, device: str = "", dtype: str = "float32") -> Any:
    """Convert a value to a torch tensor using kernel device/dtype resolution."""
    from neuroforge.kernel.torch_utils import resolve_device_dtype

    torch = require_torch()
    torch_device, torch_dtype = resolve_device_dtype(device, dtype)
    return torch.as_tensor(value, device=torch_device, dtype=torch_dtype)


def zeros_like_shape(shape: tuple[int, ...], *, device: str = "", dtype: str = "float32") -> Any:
    """Create a zero tensor for a shape using kernel device/dtype resolution."""
    from neuroforge.kernel.torch_utils import resolve_device_dtype

    torch = require_torch()
    torch_device, torch_dtype = resolve_device_dtype(device, dtype)
    return torch.zeros(shape, device=torch_device, dtype=torch_dtype)
