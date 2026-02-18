"""Unit tests for the monitor infrastructure.

Tests cover: EventBus, SpikeMonitor, VoltageMonitor, WeightMonitor,
TrainingMonitor — all using math-predictive assertions where applicable.
"""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.spike_monitor import SpikeMonitor
from neuroforge.monitors.training_monitor import TrainingMonitor
from neuroforge.monitors.voltage_monitor import VoltageMonitor
from neuroforge.monitors.weight_monitor import WeightMonitor

# ── Helpers ─────────────────────────────────────────────────────────


def _make_event(
    topic: EventTopic,
    step: int = 0,
    source: str = "test",
    data: dict[str, Any] | None = None,
    t: float = 0.0,
) -> MonitorEvent:
    return MonitorEvent(
        topic=topic, step=step, t=t, source=source, data=data or {}
    )


# ── EventBus tests ──────────────────────────────────────────────────


class TestEventBus:
    """Tests for the EventBus publish/subscribe hub."""

    def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        mon = SpikeMonitor()
        bus.subscribe(EventTopic.SPIKE, mon)

        event = _make_event(EventTopic.SPIKE, data={"spikes": [True, False, True]})
        bus.publish(event)

        snap = mon.snapshot()
        assert snap["populations"]["test"]["counts"] == [2]

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus()
        mon = SpikeMonitor()
        bus.subscribe(EventTopic.SPIKE, mon)
        bus.unsubscribe(EventTopic.SPIKE, mon)

        bus.publish(_make_event(EventTopic.SPIKE, data={"spikes": [True]}))
        assert mon.snapshot()["populations"] == {}

    def test_unsubscribe_nonexistent_is_noop(self) -> None:
        bus = EventBus()
        mon = SpikeMonitor()
        bus.unsubscribe(EventTopic.SPIKE, mon)  # should not raise

    def test_clear_removes_all(self) -> None:
        bus = EventBus()
        mon = SpikeMonitor()
        bus.subscribe(EventTopic.SPIKE, mon)
        assert bus.subscriber_count == 1
        bus.clear()
        assert bus.subscriber_count == 0

    def test_subscribe_all_topics(self) -> None:
        bus = EventBus()
        mon = TrainingMonitor()
        bus.subscribe_all(mon)
        assert bus.subscriber_count == len(EventTopic)

    def test_duplicate_subscribe_prevented(self) -> None:
        bus = EventBus()
        mon = SpikeMonitor()
        bus.subscribe(EventTopic.SPIKE, mon)
        bus.subscribe(EventTopic.SPIKE, mon)  # duplicate
        assert bus.subscriber_count == 1

    def test_multiple_monitors_receive_same_event(self) -> None:
        bus = EventBus()
        m1 = SpikeMonitor()
        m2 = SpikeMonitor()
        bus.subscribe(EventTopic.SPIKE, m1)
        bus.subscribe(EventTopic.SPIKE, m2)

        bus.publish(_make_event(EventTopic.SPIKE, data={"spikes": [True]}))

        assert m1.snapshot()["populations"]["test"]["counts"] == [1]
        assert m2.snapshot()["populations"]["test"]["counts"] == [1]

    def test_topic_isolation(self) -> None:
        """Events on one topic don't reach monitors on another."""
        bus = EventBus()
        spike_mon = SpikeMonitor()
        bus.subscribe(EventTopic.SPIKE, spike_mon)

        bus.publish(_make_event(EventTopic.VOLTAGE, data={"voltage": [0.5]}))
        assert spike_mon.snapshot()["populations"] == {}


# ── SpikeMonitor tests ──────────────────────────────────────────────


class TestSpikeMonitor:
    """Tests for the SpikeMonitor."""

    def test_counts_spikes_correctly(self) -> None:
        mon = SpikeMonitor()
        mon.on_event(_make_event(
            EventTopic.SPIKE, step=0, source="pop_a",
            data={"spikes": [True, False, True, True]},
        ))
        snap = mon.snapshot()
        assert snap["populations"]["pop_a"]["counts"] == [3]

    def test_raster_records_firing_indices(self) -> None:
        mon = SpikeMonitor()
        mon.on_event(_make_event(
            EventTopic.SPIKE, step=5, source="pop_a",
            data={"spikes": [False, True, False, True]},
        ))
        raster = mon.snapshot()["populations"]["pop_a"]["raster"]
        assert [1, 5] in raster
        assert [3, 5] in raster
        assert len(raster) == 2

    def test_disabled_monitor_ignores_events(self) -> None:
        mon = SpikeMonitor(enabled=False)
        mon.on_event(_make_event(EventTopic.SPIKE, data={"spikes": [True]}))
        assert mon.snapshot()["populations"] == {}

    def test_toggle_enabled_at_runtime(self) -> None:
        mon = SpikeMonitor(enabled=True)
        mon.on_event(_make_event(
            EventTopic.SPIKE, step=0, source="p", data={"spikes": [True]},
        ))
        mon.enabled = False
        mon.on_event(_make_event(
            EventTopic.SPIKE, step=1, source="p", data={"spikes": [True]},
        ))
        # Only the first event recorded.
        assert mon.snapshot()["populations"]["p"]["counts"] == [1]

    def test_wrong_topic_ignored(self) -> None:
        mon = SpikeMonitor()
        mon.on_event(_make_event(EventTopic.VOLTAGE, data={"voltage": [0.5]}))
        assert mon.snapshot()["populations"] == {}

    def test_reset_clears_data(self) -> None:
        mon = SpikeMonitor()
        mon.on_event(_make_event(EventTopic.SPIKE, data={"spikes": [True]}))
        mon.reset()
        assert mon.snapshot()["populations"] == {}

    def test_multiple_populations(self) -> None:
        mon = SpikeMonitor()
        mon.on_event(_make_event(EventTopic.SPIKE, source="a", data={"spikes": [True]}))
        mon.on_event(_make_event(EventTopic.SPIKE, source="b", data={"spikes": [False, True]}))
        snap = mon.snapshot()
        assert snap["populations"]["a"]["counts"] == [1]
        assert snap["populations"]["b"]["counts"] == [1]


# ── VoltageMonitor tests ────────────────────────────────────────────


class TestVoltageMonitor:
    """Tests for the VoltageMonitor."""

    def test_records_voltage_snapshot(self) -> None:
        mon = VoltageMonitor()
        mon.on_event(_make_event(
            EventTopic.VOLTAGE, step=0, source="pop",
            data={"voltage": [0.5, 0.3, 0.8]},
        ))
        snap = mon.snapshot()
        assert snap["populations"]["pop"]["voltages"] == [[0.5, 0.3, 0.8]]
        assert snap["populations"]["pop"]["steps"] == [0]

    def test_bounded_window(self) -> None:
        mon = VoltageMonitor(max_window=3)
        for i in range(5):
            mon.on_event(_make_event(
                EventTopic.VOLTAGE, step=i, source="p",
                data={"voltage": [float(i)]},
            ))
        snap = mon.snapshot()
        assert snap["populations"]["p"]["steps"] == [2, 3, 4]
        assert snap["populations"]["p"]["voltages"] == [[2.0], [3.0], [4.0]]

    def test_disabled_ignores(self) -> None:
        mon = VoltageMonitor(enabled=False)
        mon.on_event(_make_event(EventTopic.VOLTAGE, data={"voltage": [0.5]}))
        assert mon.snapshot()["populations"] == {}

    def test_reset_clears_data(self) -> None:
        mon = VoltageMonitor()
        mon.on_event(_make_event(EventTopic.VOLTAGE, data={"voltage": [0.5]}))
        mon.reset()
        assert mon.snapshot()["populations"] == {}


# ── WeightMonitor tests ─────────────────────────────────────────────


class TestWeightMonitor:
    """Tests for the WeightMonitor."""

    def test_records_weight_snapshot(self) -> None:
        mon = WeightMonitor()
        mon.on_event(_make_event(
            EventTopic.WEIGHT, step=0, source="proj",
            data={"weights": [0.1, -0.2, 0.3]},
        ))
        snap = mon.snapshot()
        assert snap["projections"]["proj"]["weights"] == [[0.1, -0.2, 0.3]]

    def test_bounded_window(self) -> None:
        mon = WeightMonitor(max_window=2)
        for i in range(4):
            mon.on_event(_make_event(
                EventTopic.WEIGHT, step=i, source="p",
                data={"weights": [float(i)]},
            ))
        snap = mon.snapshot()
        assert snap["projections"]["p"]["steps"] == [2, 3]

    def test_disabled_ignores(self) -> None:
        mon = WeightMonitor(enabled=False)
        mon.on_event(_make_event(EventTopic.WEIGHT, data={"weights": [0.5]}))
        assert mon.snapshot()["projections"] == {}

    def test_reset_clears_data(self) -> None:
        mon = WeightMonitor()
        mon.on_event(_make_event(EventTopic.WEIGHT, data={"weights": [0.5]}))
        mon.reset()
        assert mon.snapshot()["projections"] == {}


# ── TrainingMonitor tests ───────────────────────────────────────────


class TestTrainingMonitor:
    """Tests for the TrainingMonitor."""

    def test_start_event_sets_gate(self) -> None:
        mon = TrainingMonitor()
        mon.on_event(_make_event(
            EventTopic.TRAINING_START, data={"gate": "XOR"},
        ))
        assert mon.snapshot()["gate"] == "XOR"

    def test_trial_events_accumulate(self) -> None:
        mon = TrainingMonitor()
        mon.on_event(_make_event(
            EventTopic.TRAINING_START, data={"gate": "OR"},
        ))
        mon.on_event(_make_event(
            EventTopic.TRAINING_TRIAL, step=0,
            data={
                "input": (0, 1), "expected": 1, "predicted": 1,
                "correct": True, "error": 0.0, "accuracy": 1.0,
                "out_spike_count": 5,
            },
        ))
        snap = mon.snapshot()
        assert snap["total_trials"] == 1
        assert snap["accuracy_history"] == [1.0]

    def test_confidence_calculation(self) -> None:
        """Confidence = windowed correct / window_size per pattern."""
        mon = TrainingMonitor()
        mon.on_event(_make_event(EventTopic.TRAINING_START, data={"gate": "AND"}))

        # 3 trials for pattern (1, 1): 2 correct, 1 wrong → 66.67%
        for i, correct in enumerate([True, True, False]):
            mon.on_event(_make_event(
                EventTopic.TRAINING_TRIAL, step=i,
                data={
                    "input": (1, 1), "expected": 1,
                    "predicted": 1 if correct else 0,
                    "correct": correct, "error": 0.0, "accuracy": 0.0,
                },
            ))

        snap = mon.snapshot()
        entry = snap["truth_table"]["(1, 1)"]
        assert entry["total_count"] == 3
        # 2/3 ≈ 0.6667
        assert abs(entry["confidence"] - 2 / 3) < 1e-4

    def test_end_event_sets_converged(self) -> None:
        mon = TrainingMonitor()
        mon.on_event(_make_event(
            EventTopic.TRAINING_END, data={"converged": True},
        ))
        assert mon.snapshot()["converged"] is True

    def test_start_resets_state(self) -> None:
        mon = TrainingMonitor()
        mon.on_event(_make_event(
            EventTopic.TRAINING_TRIAL, step=0,
            data={"input": (0, 0), "expected": 0, "predicted": 0,
                  "correct": True, "error": 0.0, "accuracy": 1.0},
        ))
        # Starting a new run should clear.
        mon.on_event(_make_event(
            EventTopic.TRAINING_START, data={"gate": "NOR"},
        ))
        snap = mon.snapshot()
        assert snap["total_trials"] == 0
        assert snap["gate"] == "NOR"

    def test_disabled_ignores(self) -> None:
        mon = TrainingMonitor(enabled=False)
        mon.on_event(_make_event(
            EventTopic.TRAINING_TRIAL, step=0,
            data={"input": (0, 0), "expected": 0, "predicted": 0,
                  "correct": True, "error": 0.0, "accuracy": 1.0},
        ))
        assert mon.snapshot()["total_trials"] == 0

    def test_reset_clears_everything(self) -> None:
        mon = TrainingMonitor()
        mon.on_event(_make_event(EventTopic.TRAINING_START, data={"gate": "OR"}))
        mon.on_event(_make_event(
            EventTopic.TRAINING_TRIAL, step=0,
            data={"input": (1, 0), "expected": 1, "predicted": 1,
                  "correct": True, "error": 0.0, "accuracy": 1.0},
        ))
        mon.reset()
        snap = mon.snapshot()
        assert snap["gate"] == ""
        assert snap["total_trials"] == 0

    def test_topology_event_recorded(self) -> None:
        mon = TrainingMonitor()
        topo = {"layers": ["input(2)", "output(1)"], "edges": []}
        mon.on_event(_make_event(EventTopic.TOPOLOGY, data=topo))
        snap = mon.snapshot()
        assert snap["topology"]["layers"] == ["input(2)", "output(1)"]


# ── Integration: EventBus + LogicGateTask ───────────────────────────


class TestLogicGateTaskWithMonitors:
    """Verify that LogicGateTask emits events through the bus."""

    def test_training_emits_events(self) -> None:
        """Run a quick training and verify monitors received data."""
        bus = EventBus()
        train_mon = TrainingMonitor()
        weight_mon = WeightMonitor()

        bus.subscribe_all(train_mon)
        bus.subscribe(EventTopic.WEIGHT, weight_mon)

        from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask

        cfg = LogicGateConfig(gate="OR", max_trials=100, seed=42)
        task = LogicGateTask(cfg, event_bus=bus)
        result = task.run()

        # Training monitor should have received trial events.
        snap = train_mon.snapshot()
        assert snap["gate"] == "OR"
        assert snap["total_trials"] > 0

        # Weight monitor should have weight snapshots.
        wsnap = weight_mon.snapshot()
        assert len(wsnap["projections"]) > 0

        # Converged status should match result.
        assert snap["converged"] == result.converged

    def test_no_bus_still_works(self) -> None:
        """Without a bus, the task should run identically."""
        from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask

        cfg = LogicGateConfig(gate="AND", max_trials=100, seed=42)
        task = LogicGateTask(cfg)  # no event_bus
        result = task.run()
        assert isinstance(result.trials, int)
