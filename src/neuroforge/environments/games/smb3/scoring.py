"""SMB3 score/progress delta helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import GameObservation

__all__ = ["SMB3ScoreDelta", "score_delta"]


@dataclass(frozen=True, slots=True)
class SMB3ScoreDelta:
    """Frame-to-frame SMB3 scoring deltas."""

    score: float = 0.0
    x_progress: float = 0.0
    time_left: float = 0.0


def score_delta(previous: GameObservation, current: GameObservation) -> SMB3ScoreDelta:
    """Return score, progress, and timer deltas for two observations."""
    return SMB3ScoreDelta(
        score=_delta(previous.metrics.score, current.metrics.score),
        x_progress=_delta(previous.metrics.x_progress, current.metrics.x_progress),
        time_left=_delta(previous.metrics.time_left, current.metrics.time_left),
    )


def _delta(previous: int | float | None, current: int | float | None) -> float:
    if previous is None or current is None:
        return 0.0
    return float(current) - float(previous)
