"""Unit tests for the messaging boundary."""

from __future__ import annotations

import pytest

from neuroforge.contracts.messaging import EventTopic, IEventSubscriber, MonitorEvent
from neuroforge.messaging.bus import EventBus


class _Subscriber:
    def __init__(self) -> None:
        self.events: list[MonitorEvent] = []

    def on_event(self, event: MonitorEvent) -> None:
        self.events.append(event)


@pytest.mark.unit
def test_event_bus_uses_messaging_contracts() -> None:
    bus = EventBus()
    subscriber = _Subscriber()
    event = MonitorEvent(
        topic=EventTopic.SCALAR,
        step=1,
        t=0.0,
        source="test",
        data={"loss": 0.25},
    )

    assert isinstance(subscriber, IEventSubscriber)

    bus.subscribe(EventTopic.SCALAR, subscriber)
    bus.publish(event)

    assert subscriber.events == [event]

