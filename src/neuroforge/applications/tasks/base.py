"""Shared scaffolding for training tasks.

Every task (logic gate, multi-gate, vision classification) publishes
:class:`MonitorEvent` objects to an optional event bus and consults an
optional stop-check callback. That wiring used to be copy-pasted into each
task; :class:`BaseTask` houses it once so the publish/subscribe glue lives in
a single place (DRY).

Concrete tasks own their own (differently typed) ``config`` and construction
logic. They only need to call ``super().__init__(event_bus, stop_check)`` -
or set ``self._bus`` / ``self._stop_check`` directly - and then use the
``_emit`` and ``_should_stop`` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.contracts.messaging import EventTopic, MonitorEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.messaging import IEventBus

__all__ = ["BaseTask"]


class BaseTask:
    """Base class supplying the shared event-emission + stop-check helpers."""

    _bus: IEventBus | None
    _stop_check: Callable[[], bool] | None

    def __init__(
        self,
        event_bus: IEventBus | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> None:
        self._bus = event_bus
        self._stop_check = stop_check

    #

    def _emit(
        self,
        topic: str,
        step: int,
        source: str,
        data: dict[str, Any],
        *,
        t: float = 0.0,
    ) -> None:
        """Publish a :class:`MonitorEvent` if an event bus is connected."""
        if self._bus is None:
            return
        self._bus.publish(
            MonitorEvent(
                topic=EventTopic(topic),
                step=step,
                t=t,
                source=source,
                data=data,
            )
        )

    #

    def _should_stop(self) -> bool:
        """Return ``True`` when an external stop has been requested."""
        return self._stop_check is not None and self._stop_check()
