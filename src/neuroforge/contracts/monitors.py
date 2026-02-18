"""Monitor contracts — interfaces for the observer / pub-sub system.

Monitors follow SOLID:
- **S** — each monitor has a single responsibility (record one signal).
- **O** — the system is open for extension (new monitors) without
  modifying existing code.
- **L** — any ``IMonitor`` implementation can be used interchangeably.
- **I** — ``IMonitor`` is slim; optional granularity via ``IEventBus``.
- **D** — production code depends on abstractions, not concrete monitors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "EventTopic",
    "MonitorEvent",
    "IEventBus",
    "IMonitor",
]


# ── Event topics ────────────────────────────────────────────────────


class EventTopic(StrEnum):
    """Well-known event topics for the monitor system."""

    SPIKE = auto()
    VOLTAGE = auto()
    WEIGHT = auto()
    TRAINING_TRIAL = auto()
    TRAINING_START = auto()
    TRAINING_END = auto()
    TOPOLOGY = auto()


# ── Event DTO ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MonitorEvent:
    """Immutable event emitted by the simulation / training loop.

    Attributes
    ----------
    topic:
        The event topic.
    step:
        Simulation step index (or trial index for training events).
    t:
        Simulation time in seconds (or 0.0 for training events).
    source:
        Name of the population / projection / task that emitted the event.
    data:
        Topic-specific payload (dict of tensors, scalars, etc.).
    """

    topic: EventTopic
    step: int
    t: float
    source: str
    data: dict[str, Any]


# ── Protocols ───────────────────────────────────────────────────────


@runtime_checkable
class IMonitor(Protocol):
    """Protocol for a monitor that subscribes to events.

    Monitors can be enabled/disabled at runtime.
    """

    @property
    def enabled(self) -> bool:
        """Whether this monitor is currently recording."""
        ...

    @enabled.setter
    def enabled(self, value: bool) -> None: ...

    def on_event(self, event: MonitorEvent) -> None:
        """Handle an incoming event.

        Parameters
        ----------
        event:
            The event to process. Monitors should early-return if
            ``self.enabled`` is False or the topic is irrelevant.
        """
        ...

    def reset(self) -> None:
        """Clear all recorded data."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of recorded data.

        This is used by dashboards and other consumers.
        """
        ...


@runtime_checkable
class IEventBus(Protocol):
    """Protocol for the publish/subscribe event bus.

    Decouples event producers (engine, task) from consumers (monitors).
    """

    def subscribe(
        self,
        topic: EventTopic,
        monitor: IMonitor,
    ) -> None:
        """Subscribe *monitor* to events of the given *topic*.

        Parameters
        ----------
        topic:
            The topic to listen to.
        monitor:
            The monitor instance.
        """
        ...

    def unsubscribe(
        self,
        topic: EventTopic,
        monitor: IMonitor,
    ) -> None:
        """Remove *monitor*'s subscription to *topic*.

        Parameters
        ----------
        topic:
            The topic to unsubscribe from.
        monitor:
            The monitor instance.
        """
        ...

    def publish(self, event: MonitorEvent) -> None:
        """Dispatch *event* to all subscribers of ``event.topic``.

        Parameters
        ----------
        event:
            The event to publish. Delivery order is not guaranteed.
        """
        ...

    def clear(self) -> None:
        """Remove all subscriptions."""
        ...
