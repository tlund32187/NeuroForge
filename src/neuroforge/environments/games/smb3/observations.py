"""SMB3 observation summaries derived from vision metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import GameObservation

__all__ = ["SMB3ObservationSummary"]


@dataclass(frozen=True, slots=True)
class SMB3ObservationSummary:
    """Compact SMB3 observation fields useful for logging and decisions."""

    step: int
    score: int | None = None
    lives: int | None = None
    time_left: int | None = None
    x_progress: float | None = None

    @classmethod
    def from_observation(cls, observation: GameObservation) -> SMB3ObservationSummary:
        """Build a summary from a frame-derived game observation."""
        metrics = observation.metrics
        return cls(
            step=int(observation.step),
            score=metrics.score,
            lives=metrics.lives,
            time_left=metrics.time_left,
            x_progress=metrics.x_progress,
        )
