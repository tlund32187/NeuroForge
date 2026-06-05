"""Messaging contracts for NeuroForge's publish/subscribe boundary.

This module contains event topics, event payload DTOs, and lightweight
protocols for event publishers/subscribers. It intentionally has no dependency
on concrete monitor implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Event",
    "EventTopic",
    "IEventBus",
    "IEventPublisher",
    "IEventSubscriber",
    "MonitorEvent",
]


class EventTopic(StrEnum):
    """Well-known event topics for simulation, training, and applications."""

    SPIKE = auto()
    VOLTAGE = auto()
    WEIGHT = auto()
    TRAINING_TRIAL = auto()
    TRAINING_START = auto()
    TRAINING_END = auto()
    TOPOLOGY = auto()
    TOPOLOGY_STATS = auto()
    TOPOLOGY_TRACE = auto()
    RUN_START = auto()
    RUN_END = auto()
    SCALAR = auto()
    EVALUATION_PROGRESS = auto()


@dataclass(frozen=True, slots=True)
class MonitorEvent:
    """Immutable event emitted by simulation, training, or application code."""

    topic: EventTopic
    step: int
    t: float
    source: str
    data: dict[str, Any]


Event = MonitorEvent


@runtime_checkable
class IEventSubscriber(Protocol):
    """Protocol for any object that can receive published events."""

    def on_event(self, event: MonitorEvent) -> None:
        """Handle an incoming event."""
        ...


@runtime_checkable
class IEventPublisher(Protocol):
    """Protocol for objects that publish events."""

    def publish(self, event: MonitorEvent) -> None:
        """Publish an event to interested subscribers."""
        ...


@runtime_checkable
class IEventBus(IEventPublisher, Protocol):
    """Protocol for a topic-based publish/subscribe bus."""

    def subscribe(
        self,
        topic: EventTopic,
        subscriber: IEventSubscriber,
    ) -> None:
        """Subscribe *subscriber* to events of the given *topic*."""
        ...

    def unsubscribe(
        self,
        topic: EventTopic,
        subscriber: IEventSubscriber,
    ) -> None:
        """Remove *subscriber*'s subscription to *topic*."""
        ...

    def clear(self) -> None:
        """Remove all subscriptions."""
        ...
