"""Visual sensor protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation

__all__ = ["IVisualSensor"]


@runtime_checkable
class IVisualSensor(Protocol):
    """Converts visual environment output into observations."""

    def observe(self, frame: object) -> AgentObservation:
        """Build an observation from a frame-like object."""
        ...
