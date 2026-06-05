"""Dense static projection operator."""

from __future__ import annotations

from typing import Any

__all__ = ["dense_static_current"]


def dense_static_current(pre_spikes: Any, weight_matrix: Any) -> Any:
    """Return dense post current using ``pre @ W``."""
    pre = pre_spikes.to(dtype=weight_matrix.dtype)
    if int(weight_matrix.ndim) == 1:
        return (pre * weight_matrix).sum().reshape(1)
    return pre @ weight_matrix

