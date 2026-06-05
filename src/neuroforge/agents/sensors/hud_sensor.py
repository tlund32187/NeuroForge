"""HUD sensor protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.agents.observation import AgentObservation

__all__ = ["IHUDSensor"]


@runtime_checkable
class IHUDSensor(Protocol):
    """Converts HUD metrics into observations."""

    def observe_hud(self, hud: object) -> AgentObservation:
        """Build an observation from HUD data."""
        ...
