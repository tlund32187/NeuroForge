"""EventBus — concrete implementation of IEventBus.

Thread-safe publish/subscribe hub. Monitors subscribe to topics; the
engine/task publishes events. CUDA-compatible (events carry device-
agnostic payloads — monitors decide whether to copy to CPU).
"""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict
from typing import TYPE_CHECKING

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

if TYPE_CHECKING:
    from neuroforge.contracts.monitors import IMonitor

__all__ = ["EventBus"]


class EventBus:
    """Concrete event bus with topic-based pub/sub.

    Thread-safe via a reentrant lock so monitors can be added/removed
    from any thread (useful when a dashboard runs in a background thread).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[EventTopic, list[IMonitor]] = defaultdict(list)

    # ── IEventBus implementation ────────────────────────────────────

    def subscribe(self, topic: EventTopic, monitor: IMonitor) -> None:
        """Subscribe *monitor* to *topic*."""
        with self._lock:
            subs = self._subs[topic]
            if monitor not in subs:
                subs.append(monitor)

    def unsubscribe(self, topic: EventTopic, monitor: IMonitor) -> None:
        """Unsubscribe *monitor* from *topic*."""
        with self._lock:
            subs = self._subs[topic]
            with contextlib.suppress(ValueError):
                subs.remove(monitor)

    def publish(self, event: MonitorEvent) -> None:
        """Dispatch *event* to all subscribers of ``event.topic``."""
        with self._lock:
            # Snapshot subscriber list to avoid mutation during dispatch.
            subs = list(self._subs.get(event.topic, []))
        for mon in subs:
            mon.on_event(event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subs.clear()

    # ── Convenience ─────────────────────────────────────────────────

    def subscribe_all(self, monitor: IMonitor) -> None:
        """Subscribe *monitor* to every known topic."""
        for topic in EventTopic:
            self.subscribe(topic, monitor)

    @property
    def subscriber_count(self) -> int:
        """Total number of (topic, monitor) pairs."""
        with self._lock:
            return sum(len(v) for v in self._subs.values())
