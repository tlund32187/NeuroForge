"""Unit tests for TopologyStatsMonitor."""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.topology_stats_monitor import TopologyStatsMonitor


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


def _topic_events(events: list[MonitorEvent], topic: EventTopic) -> list[MonitorEvent]:
    return [e for e in events if e.topic == topic]


def test_topology_stats_monitor_computes_totals_and_emits_events() -> None:
    bus = EventBus()
    cap = _CaptureMonitor()
    mon = TopologyStatsMonitor(event_bus=bus, enabled=True)

    bus.subscribe_all(mon)
    bus.subscribe_all(cap)

    bus.publish(
        MonitorEvent(
            topic=EventTopic.TOPOLOGY,
            step=0,
            t=0.0,
            source="test",
            data={
                "layers": ["input(64)", "hidden(24)", "output(1)"],
                "projection_meta": [
                    {
                        "name": "input_hidden",
                        "src": "input",
                        "dst": "hidden",
                        "n_pre": 64,
                        "n_post": 24,
                        "n_edges": 1536,
                        "dense": True,
                        "dtype": "float64",
                        "topology_type": "dense",
                    },
                    {
                        "name": "hidden_output",
                        "src": "hidden",
                        "dst": "output",
                        "n_pre": 24,
                        "n_post": 1,
                        "n_edges": 60,
                        "dense": False,
                        "dtype": "float32",
                        "topology_type": "sparse_random",
                    },
                ],
            },
        )
    )

    top_stats_events = _topic_events(cap.events, EventTopic.TOPOLOGY_STATS)
    scalar_events = _topic_events(cap.events, EventTopic.SCALAR)
    assert len(top_stats_events) == 1
    assert len(scalar_events) >= 1

    totals = top_stats_events[0].data["totals"]
    assert totals["projection_count"] == 2
    assert totals["edges_total"] == 1596
    assert totals["bytes_idx_total_est"] == 25536
    assert totals["bytes_delays_total_est"] == 12768
    assert totals["bytes_weights_total_est"] == 12528
    assert totals["bytes_dense_total_est"] == 12288
    assert totals["bytes_total_est"] == 63120

    scalar = scalar_events[-1].data
    assert scalar["topology.edges_total"] == 1596
    assert scalar["topology.bytes_total_est"] == 63120


def test_topology_stats_monitor_is_one_shot() -> None:
    bus = EventBus()
    cap = _CaptureMonitor()
    mon = TopologyStatsMonitor(event_bus=bus, enabled=True)
    bus.subscribe_all(mon)
    bus.subscribe_all(cap)

    payload = {
        "projection_meta": [
            {
                "name": "p",
                "src": "a",
                "dst": "b",
                "n_pre": 4,
                "n_post": 4,
                "n_edges": 16,
                "dense": True,
                "dtype": "float32",
                "topology_type": "dense",
            }
        ]
    }
    bus.publish(
        MonitorEvent(
            topic=EventTopic.TOPOLOGY,
            step=0,
            t=0.0,
            source="test",
            data=dict(payload),
        )
    )
    bus.publish(
        MonitorEvent(
            topic=EventTopic.TOPOLOGY,
            step=1,
            t=0.0,
            source="test",
            data=dict(payload),
        )
    )

    assert len(_topic_events(cap.events, EventTopic.TOPOLOGY_STATS)) == 1
