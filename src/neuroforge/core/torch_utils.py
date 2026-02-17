"""Torch utilities — lazy import, device/dtype resolution, helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["require_torch", "resolve_device_dtype"]

_torch: Any = None


def require_torch() -> Any:
    """Lazily import and return the ``torch`` module.

    Raises
    ------
    ImportError:
        If PyTorch is not installed.
    """
    global _torch  # noqa: PLW0603
    if _torch is None:
        try:
            import torch as _t

            _torch = _t
        except ImportError as exc:
            msg = (
                "PyTorch is required but not installed. "
                "Install it with: pip install neuroforge[torch]"
            )
            raise ImportError(msg) from exc
    return _torch


def resolve_device_dtype(
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[Any, Any]:
    """Convert string device/dtype to torch objects.

    Parameters
    ----------
    device:
        Device string (``"cpu"`` or ``"cuda"``).
    dtype:
        Dtype string (``"float32"``, ``"float64"``, ``"float16"``).

    Returns
    -------
    tuple:
        Resolved (torch.device, torch.dtype).

    Raises
    ------
    ValueError:
        If dtype string is not recognised.
    """
    torch = require_torch()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    torch_device = torch.device(device)
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        msg = f"Unsupported dtype: {dtype!r}. Use one of {list(dtype_map.keys())}"
        raise ValueError(msg)

    return torch_device, torch_dtype
