"""Dense Dale's-law static projection operator."""

from __future__ import annotations

from typing import Any

from neuroforge.biology.synapses.operators.dense_static import dense_static_current

__all__ = ["dales_dense_current"]


def dales_dense_current(pre_spikes: Any, weight_matrix: Any, sign_pre: Any | None) -> Any:
    """Return dense post current after applying Dale's-law signs."""
    if sign_pre is None:
        return dense_static_current(pre_spikes, weight_matrix)
    sign = sign_pre.to(device=weight_matrix.device, dtype=weight_matrix.dtype)
    if int(weight_matrix.ndim) == 1:
        effective = weight_matrix.abs() * sign
    else:
        effective = weight_matrix.abs() * sign.reshape(-1, 1)
    return dense_static_current(pre_spikes, effective)

