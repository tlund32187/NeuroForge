"""Torch utilities — lazy import, device/dtype resolution, helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "default_device",
    "require_torch",
    "resolve_device_dtype",
    "smart_device",
]

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


# Minimum total neuron count for CUDA to outperform CPU.  Below this
# threshold the kernel-launch overhead dominates and CPU is faster.
CUDA_NEURON_THRESHOLD: int = 500


def default_device() -> str:
    """Return ``"cuda"`` when CUDA is available, otherwise ``"cpu"``.

    This is the recommended way to pick a device string at config
    construction time so that GPU acceleration is used automatically.
    """
    torch = require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def smart_device(n_neurons: int) -> str:
    """Pick the fastest device for a network of *n_neurons* total neurons.

    Small networks (< :data:`CUDA_NEURON_THRESHOLD` neurons) run faster
    on CPU because CUDA kernel-launch and sync overhead dominate.  For
    larger networks CUDA is used when available.

    Parameters
    ----------
    n_neurons:
        Total number of neurons in the network (input + hidden + output).

    Returns
    -------
    str:
        ``"cuda"`` or ``"cpu"``.
    """
    torch = require_torch()
    if n_neurons >= CUDA_NEURON_THRESHOLD and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device_dtype(
    device: str = "",
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

    torch_device = torch.device(device if device else default_device())
    torch_dtype = dtype_map.get(dtype)
    if torch_dtype is None:
        msg = f"Unsupported dtype: {dtype!r}. Use one of {list(dtype_map.keys())}"
        raise ValueError(msg)

    return torch_device, torch_dtype
