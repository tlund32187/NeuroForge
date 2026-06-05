"""Unit tests for TopologyActivityMonitor."""

from __future__ import annotations

from typing import Any

import pytest

from neuroforge.contracts.messaging import EventTopic, MonitorEvent
from neuroforge.messaging.bus import EventBus
from neuroforge.observability.monitors.topology_activity_monitor import TopologyActivityMonitor


class _CaptureMonitor:
    def __init__(self) -> None:
        self.enabled = True
        self.events: list[MonitorEvent] = []

    def on_event(self, event: MonitorEvent) -> None:
        self.events.append(event)

    def reset(self) -> None:
        self.events.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"count": len(self.events)}


def _event(
    topic: EventTopic,
    *,
    step: int = 0,
    source: str = "test",
    data: dict[str, Any] | None = None,
) -> MonitorEvent:
    return MonitorEvent(topic=topic, step=step, t=0.0, source=source, data=data or {})


def _topology_payload() -> dict[str, Any]:
    return {
        "layers": ["input(4)", "hidden(3)", "output(1)"],
        "projection_meta": [
            {
                "name": "input_hidden",
                "src": "input",
                "dst": "hidden",
                "n_pre": 4,
                "n_post": 3,
                "n_edges": 12,
                "dense": True,
                "dtype": "float32",
                "topology_type": "dense",
            },
            {
                "name": "hidden_output",
                "src": "hidden",
                "dst": "output",
                "n_pre": 3,
                "n_post": 1,
                "n_edges": 3,
                "dense": True,
                "dtype": "float32",
                "topology_type": "dense",
            },
        ],
    }


def _trace_events(events: list[MonitorEvent]) -> list[MonitorEvent]:
    return [event for event in events if event.topic == EventTopic.TOPOLOGY_TRACE]


def test_derives_layer_activity_from_training_trial() -> None:
    bus = EventBus()
    cap = _CaptureMonitor()
    mon = TopologyActivityMonitor(event_bus=bus, trace_every_n_steps=1)
    bus.subscribe_all(mon)
    bus.subscribe_all(cap)

    bus.publish(_event(EventTopic.TOPOLOGY, data=_topology_payload()))
    bus.publish(
        _event(
            EventTopic.TRAINING_TRIAL,
            step=1,
            data={
                "input_spikes": [1, 0, 2, 0],
                "hidden_spikes": [0, 3, 1],
                "output_spikes": [1],
            },
        )
    )

    traces = _trace_events(cap.events)
    assert len(traces) == 1
    payload = traces[0].data
    by_layer = {row["name"]: row for row in payload["layers"]}
    assert by_layer["input"]["latest_spikes"] == 3.0
    assert by_layer["input"]["active_count"] == 2
    assert by_layer["hidden"]["active_count"] == 2
    assert by_layer["output"]["activity"] == 1.0


def test_emits_no_trace_before_topology_is_known() -> None:
    bus = EventBus()
    cap = _CaptureMonitor()
    mon = TopologyActivityMonitor(event_bus=bus, trace_every_n_steps=1)
    bus.subscribe_all(mon)
    bus.subscribe_all(cap)

    bus.publish(
        _event(
            EventTopic.TRAINING_TRIAL,
            step=0,
            data={"input_spikes": [1, 0], "hidden_spikes": [1]},
        )
    )

    assert _trace_events(cap.events) == []


def test_respects_cadence_and_caps_edge_samples() -> None:
    bus = EventBus()
    cap = _CaptureMonitor()
    mon = TopologyActivityMonitor(
        event_bus=bus,
        trace_every_n_steps=2,
        max_edge_samples=3,
        max_weight_samples=5,
    )
    bus.subscribe_all(mon)
    bus.subscribe_all(cap)

    bus.publish(_event(EventTopic.TOPOLOGY, data=_topology_payload()))
    for step in range(1, 5):
        bus.publish(
            _event(
                EventTopic.TRAINING_TRIAL,
                step=step,
                data={
                    "input_spikes": [1, 0, 1, 1],
                    "hidden_spikes": [1, 1, 0],
                    "output_spikes": [1],
                },
            )
        )

    traces = _trace_events(cap.events)
    assert [event.step for event in traces] == [2, 4]

    bus.publish(
        _event(
            EventTopic.WEIGHT,
            step=4,
            source="XOR/input-hidden",
            data={"weights": list(range(12))},
        )
    )
    weight_trace = _trace_events(cap.events)[-1].data
    projections = {row["name"]: row for row in weight_trace["projections"]}
    assert len(projections["input_hidden"]["sample_edges"]) <= 3


def test_handles_tensor_weights_without_full_trace_growth() -> None:
    torch = pytest.importorskip("torch")
    bus = EventBus()
    cap = _CaptureMonitor()
    mon = TopologyActivityMonitor(
        event_bus=bus,
        trace_every_n_steps=1,
        max_edge_samples=4,
        max_weight_samples=7,
    )
    bus.subscribe_all(mon)
    bus.subscribe_all(cap)

    bus.publish(_event(EventTopic.TOPOLOGY, data=_topology_payload()))
    bus.publish(
        _event(
            EventTopic.TRAINING_TRIAL,
            step=0,
            data={
                "input_spikes": [1, 0, 1, 0],
                "hidden_spikes": [1, 1, 0],
                "output_spikes": [1],
            },
        )
    )
    bus.publish(
        _event(
            EventTopic.WEIGHT,
            step=0,
            source="XOR/input-hidden",
            data={"weights": torch.arange(128, dtype=torch.float32)},
        )
    )

    payload = _trace_events(cap.events)[-1].data
    projection = {row["name"]: row for row in payload["projections"]}["input_hidden"]
    assert projection["weight_abs_mean"] is not None
    assert len(projection["sample_edges"]) <= 4
    assert mon.snapshot()["latest"]["step"] == 0
