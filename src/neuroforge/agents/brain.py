"""Brain protocols and decision DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation

__all__ = ["BrainDecision", "IBrain"]


def _empty_decision_signals() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, slots=True)
class BrainDecision:
    """Decision signals produced by a brain before action decoding."""

    signals: dict[str, Any] = field(default_factory=_empty_decision_signals)


@runtime_checkable
class IBrain(Protocol):
    """Decision-making model inside an agent."""

    def decide(self, observation: AgentObservation) -> BrainDecision:
        """Return decision signals for one observation."""
        ...
