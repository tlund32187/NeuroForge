"""State sensor protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation

__all__ = ["IStateSensor"]


@runtime_checkable
class IStateSensor(Protocol):
    """Converts structured environment state into observations."""

    def observe_state(self, state: object) -> AgentObservation:
        """Build an observation from structured state."""
        ...
