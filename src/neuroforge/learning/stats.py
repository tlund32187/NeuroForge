"""Small tensor/gradient stats helpers for lightweight scalar emission."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["grad_stats", "tensor_stats"]


def _norm_maxabs(t: Tensor | None) -> tuple[float, float]:
    if t is None:
        return 0.0, 0.0
    td = t.detach()
    if td.numel() == 0:
        return 0.0, 0.0
    td_any = cast("Any", td)
    norm = float(td_any.norm().item())
    maxabs = float(td_any.abs().max().item())
    return norm, maxabs


def tensor_stats(t: Tensor | None, prefix: str) -> dict[str, float]:
    """Return L2 norm and max-abs stats for a tensor."""
    norm, maxabs = _norm_maxabs(t)
    return {
        f"{prefix}_norm": norm,
        f"{prefix}_maxabs": maxabs,
    }


def grad_stats(t: Tensor | None, prefix: str) -> dict[str, float]:
    """Return grad norm/max-abs stats using ``t.grad`` when available."""
    grad = None if t is None else t.grad
    norm, maxabs = _norm_maxabs(grad)
    return {
        f"g_norm_{prefix}": norm,
        f"g_maxabs_{prefix}": maxabs,
    }
