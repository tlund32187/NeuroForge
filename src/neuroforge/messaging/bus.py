"""Thread-safe topic-based publish/subscribe event bus."""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict

from neuroforge.contracts.messaging import EventTopic, IEventSubscriber, MonitorEvent

__all__ = ["EventBus"]


class EventBus:
    """Concrete event bus with topic-based pub/sub.

    Subscribers are any objects that implement ``on_event``. Monitor lifecycle
    semantics stay outside the bus so messaging does not depend on monitoring.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[EventTopic, list[IEventSubscriber]] = defaultdict(list)

    def subscribe(self, topic: EventTopic, subscriber: IEventSubscriber) -> None:
        """Subscribe *subscriber* to *topic*."""
        with self._lock:
            subs = self._subs[topic]
            if subscriber not in subs:
                subs.append(subscriber)

    def unsubscribe(self, topic: EventTopic, subscriber: IEventSubscriber) -> None:
        """Unsubscribe *subscriber* from *topic*."""
        with self._lock:
            subs = self._subs[topic]
            with contextlib.suppress(ValueError):
                subs.remove(subscriber)

    def publish(self, event: MonitorEvent) -> None:
        """Dispatch *event* to all subscribers of ``event.topic``."""
        with self._lock:
            subscribers = list(self._subs.get(event.topic, []))
        for subscriber in subscribers:
            subscriber.on_event(event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subs.clear()

    def subscribe_all(self, subscriber: IEventSubscriber) -> None:
        """Subscribe *subscriber* to every known topic."""
        for topic in EventTopic:
            self.subscribe(topic, subscriber)

    @property
    def subscriber_count(self) -> int:
        """Total number of (topic, subscriber) pairs."""
        with self._lock:
            return sum(len(subscribers) for subscribers in self._subs.values())
