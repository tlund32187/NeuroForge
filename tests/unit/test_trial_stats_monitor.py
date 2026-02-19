# pyright: basic
"""Unit tests for TrialStatsMonitor."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

import pytest

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.artifact_writer import ArtifactWriter
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.trial_stats_monitor import TrialStatsMonitor

if TYPE_CHECKING:
    from pathlib import Path


def _event(
    topic: EventTopic,
    *,
    step: int = 0,
    data: dict[str, Any] | None = None,
) -> MonitorEvent:
    return MonitorEvent(
        topic=topic,
        step=step,
        t=0.0,
        source="test",
        data=data or {},
    )


def test_trial_stats_monitor_injects_rate_and_sparsity_fields() -> None:
    mon = TrialStatsMonitor(enabled=True)
    mon.on_event(
        _event(
            EventTopic.TRAINING_START,
            data={
                "dt": 0.002,
                "window_steps": 20,
            },
        )
    )

    trial = _event(
        EventTopic.TRAINING_TRIAL,
        step=1,
        data={
            "gate": "XOR",
            "input": (0, 1),
            "correct": True,
            "out_spike_count": 5,
            "input_spike_counts": [0, 5, 10],
            "hidden_spike_counts": [2, 0],
        },
    )
    mon.on_event(trial)

    assert trial.data["out_spike_count"] == 5
    assert trial.data["rate_out_hz"] == pytest.approx(125.0)
    assert trial.data["rate_in_mean_hz"] == pytest.approx(125.0)
    assert trial.data["rate_in_max_hz"] == pytest.approx(250.0)
    assert trial.data["sparsity_in"] == pytest.approx(1.0 / 3.0)
    assert trial.data["rate_hid_mean_hz"] == pytest.approx(25.0)
    assert trial.data["rate_hid_max_hz"] == pytest.approx(50.0)
    assert trial.data["sparsity_hid"] == pytest.approx(0.5)
    assert trial.data["conv_streak"] == 1
    assert trial.data["conv_per_pattern_min"] == pytest.approx(1.0)
    assert trial.data["conv_per_pattern_mean"] == pytest.approx(1.0)


def test_trial_stats_monitor_supports_spike_alias_fields() -> None:
    mon = TrialStatsMonitor(enabled=True, dt_default=1e-3, window_steps_default=50)
    mon.on_event(_event(EventTopic.TRAINING_START, data={}))

    trial = _event(
        EventTopic.TRAINING_TRIAL,
        step=2,
        data={
            "gate": "AND",
            "input_pattern": "01",
            "correct": 1,
            "output_spikes": [1, 0, 1],
            "input_spikes": [0, 2],
            "hidden_spikes": [1, 1],
        },
    )
    mon.on_event(trial)

    assert trial.data["out_spike_count"] == 2
    assert trial.data["rate_out_hz"] == pytest.approx(40.0)
    assert trial.data["rate_in_mean_hz"] == pytest.approx(20.0)
    assert trial.data["rate_in_max_hz"] == pytest.approx(40.0)
    assert trial.data["sparsity_in"] == pytest.approx(0.5)
    assert trial.data["rate_hid_mean_hz"] == pytest.approx(20.0)
    assert trial.data["rate_hid_max_hz"] == pytest.approx(20.0)
    assert trial.data["sparsity_hid"] == pytest.approx(0.0)
    assert trial.data["conv_streak"] == 1


def test_trial_stats_monitor_tracks_per_pattern_streaks() -> None:
    mon = TrialStatsMonitor(enabled=True)
    mon.on_event(_event(EventTopic.TRAINING_START, data={}))

    ev1 = _event(
        EventTopic.TRAINING_TRIAL,
        step=0,
        data={"gate": "XOR", "input": [0, 0], "correct": True, "out_spike_count": 1},
    )
    ev2 = _event(
        EventTopic.TRAINING_TRIAL,
        step=1,
        data={"gate": "XOR", "input": [0, 0], "correct": True, "out_spike_count": 1},
    )
    ev3 = _event(
        EventTopic.TRAINING_TRIAL,
        step=2,
        data={"gate": "XOR", "input": [0, 1], "correct": True, "out_spike_count": 1},
    )
    ev4 = _event(
        EventTopic.TRAINING_TRIAL,
        step=3,
        data={"gate": "XOR", "input": [0, 0], "correct": False, "out_spike_count": 0},
    )

    mon.on_event(ev1)
    mon.on_event(ev2)
    mon.on_event(ev3)
    mon.on_event(ev4)

    assert ev1.data["conv_streak"] == 1
    assert ev2.data["conv_streak"] == 2
    assert ev3.data["conv_streak"] == 3
    assert ev4.data["conv_streak"] == 0
    assert ev3.data["conv_per_pattern_min"] == pytest.approx(1.0)
    assert ev3.data["conv_per_pattern_mean"] == pytest.approx(1.5)
    assert ev4.data["conv_per_pattern_min"] == pytest.approx(0.0)
    assert ev4.data["conv_per_pattern_mean"] == pytest.approx(0.5)


def test_trial_stats_monitor_values_land_in_scalars_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_trial_stats"
    run_dir.mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "logs").mkdir()

    bus = EventBus()
    mon = TrialStatsMonitor(enabled=True)
    writer = ArtifactWriter(run_dir, flush_every_n=1)

    bus.subscribe_all(mon)
    bus.subscribe_all(writer)

    bus.publish(
        _event(
            EventTopic.TRAINING_START,
            data={"dt": 0.001, "window_steps": 10},
        )
    )
    bus.publish(
        _event(
            EventTopic.TRAINING_TRIAL,
            step=5,
            data={
                "gate": "XOR",
                "input": [1, 0],
                "correct": True,
                "out_spike_count": 3,
                "input_spike_counts": [0, 1],
                "hidden_spike_counts": [2, 0],
            },
        )
    )
    writer.flush()

    with (run_dir / "metrics" / "scalars.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["rate_out_hz"] == "300.0"
    assert row["rate_in_mean_hz"] == "50.0"
    assert row["sparsity_in"] == "0.5"
    assert row["conv_streak"] == "1"
