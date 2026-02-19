"""Unit tests for StabilityMonitor."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

import pytest

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.artifact_writer import ArtifactWriter
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.stability_monitor import StabilityConfig, StabilityMonitor

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


def test_stability_monitor_flags_nan_inf_from_scalar() -> None:
    monitor = StabilityMonitor(
        StabilityConfig(enabled=True, check_every_n_trials=1),
    )
    monitor.on_event(_event(EventTopic.TRAINING_START, data={}))
    scalar_ev = _event(
        EventTopic.SCALAR,
        step=1,
        data={"trial": 1, "loss": float("nan")},
    )

    monitor.on_event(scalar_ev)

    assert scalar_ev.data["stab_nan_inf"] == 1
    assert scalar_ev.data["stab_weight_explode"] == 0


def test_stability_monitor_flags_weight_explode() -> None:
    monitor = StabilityMonitor(
        StabilityConfig(
            enabled=True,
            check_every_n_trials=1,
            weight_maxabs_threshold=10.0,
        ),
    )
    monitor.on_event(_event(EventTopic.TRAINING_START, data={}))
    scalar_ev = _event(
        EventTopic.SCALAR,
        step=2,
        data={"trial": 2, "w_maxabs_ih": 12.5},
    )

    monitor.on_event(scalar_ev)

    assert scalar_ev.data["stab_weight_explode"] == 1


def test_stability_monitor_flags_oscillation_pattern() -> None:
    monitor = StabilityMonitor(
        StabilityConfig(
            enabled=True,
            check_every_n_trials=1,
            oscillation_window=12,
        ),
    )
    monitor.on_event(_event(EventTopic.TRAINING_START, data={}))

    last_ev: MonitorEvent | None = None
    pattern = [0, 2, 0, 2, 0, 2, 0, 2]
    for idx, count in enumerate(pattern, start=1):
        last_ev = _event(
            EventTopic.TRAINING_TRIAL,
            step=idx,
            data={"trial": idx, "out_spike_count": count, "accuracy": 0.5},
        )
        monitor.on_event(last_ev)

    assert last_ev is not None
    assert last_ev.data["stab_oscillation"] == 1


def test_stability_monitor_fail_fast_raises_on_critical_flag() -> None:
    monitor = StabilityMonitor(
        StabilityConfig(
            enabled=True,
            check_every_n_trials=1,
            fail_fast=True,
            weight_maxabs_threshold=1.0,
        ),
    )
    monitor.on_event(_event(EventTopic.TRAINING_START, data={}))
    scalar_ev = _event(
        EventTopic.SCALAR,
        step=1,
        data={"trial": 1, "w_maxabs_ho": 2.0},
    )

    with pytest.raises(RuntimeError, match="weight maxabs threshold exceeded"):
        monitor.on_event(scalar_ev)


def test_stability_flags_written_to_scalars_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_stability"
    run_dir.mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "logs").mkdir()

    bus = EventBus()
    monitor = StabilityMonitor(
        StabilityConfig(enabled=True, check_every_n_trials=1),
    )
    writer = ArtifactWriter(run_dir, flush_every_n=1)

    bus.subscribe_all(monitor)
    bus.subscribe_all(writer)

    bus.publish(_event(EventTopic.TRAINING_START, data={}))
    bus.publish(
        _event(
            EventTopic.SCALAR,
            step=1,
            data={"trial": 1, "loss": float("nan")},
        )
    )
    writer.flush()

    with (run_dir / "metrics" / "scalars.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    assert rows[0]["stab_nan_inf"] == "1"
