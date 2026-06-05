"""SMB3 termination helpers derived from vision metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import GameObservation

__all__ = ["SMB3TerminationDecision", "level_clear_decision"]


@dataclass(frozen=True, slots=True)
class SMB3TerminationDecision:
    """SMB3-specific termination decision."""

    terminated: bool
    reason: str = ""


def level_clear_decision(
    observation: GameObservation,
    *,
    clear_progress_threshold: float = 0.985,
) -> SMB3TerminationDecision:
    """Return a level-clear decision from vision-derived progress."""
    x_progress = observation.metrics.x_progress
    cleared = x_progress is not None and float(x_progress) >= clear_progress_threshold
    return SMB3TerminationDecision(
        terminated=cleared,
        reason="level_clear" if cleared else "",
    )
