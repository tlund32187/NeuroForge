"""Dtype resolution helpers."""

from __future__ import annotations

from typing import Any

from neuroforge.kernel.torch_utils import require_torch

__all__ = ["resolve_dtype"]


def resolve_dtype(dtype: str = "float32") -> Any:
    """Resolve a dtype string to a torch dtype object."""
    torch = require_torch()
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        msg = f"Unsupported dtype: {dtype!r}. Use one of {list(dtype_map)}"
        raise ValueError(msg)
    return torch_dtype
