"""Monitoring contracts for observability subscribers.

Monitor implementations are event subscribers with lifecycle and snapshot
semantics. The event transport itself lives in :mod:
euroforge.messaging`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from neuroforge.contracts.messaging import IEventSubscriber, MonitorEvent

__all__ = ["IMonitor"]


@runtime_checkable
class IMonitor(IEventSubscriber, Protocol):
    """Protocol for a monitor that subscribes to events."""

    @property
    def enabled(self) -> bool:
        """Whether this monitor is currently recording."""
        ...

    @enabled.setter
    def enabled(self, value: bool) -> None: ...

    def on_event(self, event: MonitorEvent) -> None:
        """Handle an incoming event."""
        ...

    def reset(self) -> None:
        """Clear all recorded data."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of recorded data."""
        ...
