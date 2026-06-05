"""RAM sensor protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation

__all__ = ["IRAMSensor"]


@runtime_checkable
class IRAMSensor(Protocol):
    """Optional sensor for environments that expose RAM-like state."""

    def observe_ram(self, state: object) -> AgentObservation:
        """Build an observation from RAM-like state."""
        ...
