"""Learning objective helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ObjectiveResult", "supervised_loss_objective"]


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Result from evaluating a learning objective."""

    loss: Any
    metrics: dict[str, float]


def supervised_loss_objective(
    prediction: Any,
    target: Any,
    loss_fn: Any,
) -> ObjectiveResult:
    """Evaluate a supervised loss and return objective metrics."""
    loss = loss_fn(prediction, target)
    value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
    return ObjectiveResult(loss=loss, metrics={"loss": value})
