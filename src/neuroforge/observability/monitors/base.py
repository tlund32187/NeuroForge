"""Base monitor implementation for observability subscribers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuroforge.contracts.messaging import MonitorEvent

__all__ = ["MonitorBase"]


class MonitorBase(ABC):
    """Small base class for event subscriber monitors."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    @abstractmethod
    def on_event(self, event: MonitorEvent) -> None:
        """Handle an incoming monitor event."""

    def reset(self) -> None:
        """Reset monitor state."""
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable monitor snapshot."""
        return {"enabled": self.enabled}
