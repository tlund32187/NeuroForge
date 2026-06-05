"""Mean-squared-error losses for spike-count learning."""

from __future__ import annotations

from typing import Any

__all__ = ["MseCountLoss"]


class MseCountLoss:
    """MSE between spike count and a target count."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, count: Any, target_count: Any) -> Any:
        """Compute MSE loss."""
        diff = count - target_count
        sq = diff * diff
        if self.reduction == "sum":
            return sq.sum()
        return sq.mean()
