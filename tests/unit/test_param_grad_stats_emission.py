# pyright: basic, reportMissingImports=false
"""Unit tests for task-emitted parameter/gradient scalar stats."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.artifact_writer import ArtifactWriter
from neuroforge.monitors.bus import EventBus
from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask
from neuroforge.tasks.multi_gate import MultiGateConfig, MultiGateTask

if TYPE_CHECKING:
    from pathlib import Path


class _ScalarCaptureMonitor:
    def __init__(self) -> None:
        self.enabled = True
        self.rows: list[dict[str, Any]] = []

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        self.rows.append(dict(event.data))

    def reset(self) -> None:
        self.rows.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"rows": len(self.rows)}


def _has_param_grad_stats(row: dict[str, Any]) -> bool:
    keys = (
        "w_norm_ih",
        "w_maxabs_ih",
        "w_norm_ho",
        "w_maxabs_ho",
        "g_norm_ih",
        "g_maxabs_ih",
        "g_norm_ho",
        "g_maxabs_ho",
    )
    for key in keys:
        if key not in row:
            return False
        if row[key] == "":
            return False
    return True


def test_logic_gate_emits_param_grad_stats_in_scalar_events() -> None:
    bus = EventBus()
    capture = _ScalarCaptureMonitor()
    bus.subscribe(EventTopic.SCALAR, capture)

    cfg = LogicGateConfig(
        gate="XOR",
        max_trials=8,
        seed=7,
        device="cpu",
        emit_param_grad_stats=True,
        stats_every_n_trials=1,
    )
    task = LogicGateTask(cfg, event_bus=bus)
    _ = task.run()

    assert capture.rows
    assert any(_has_param_grad_stats(row) for row in capture.rows)


def test_multi_gate_emits_param_grad_stats_in_scalar_events() -> None:
    bus = EventBus()
    capture = _ScalarCaptureMonitor()
    bus.subscribe(EventTopic.SCALAR, capture)

    cfg = MultiGateConfig(
        max_epochs=1,
        seed=7,
        device="cpu",
        emit_param_grad_stats=True,
        stats_every_n_trials=1,
    )
    task = MultiGateTask(cfg, event_bus=bus)
    _ = task.run()

    assert capture.rows
    assert any(_has_param_grad_stats(row) for row in capture.rows)


def test_logic_gate_param_grad_stats_written_to_scalars_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_stats_csv"
    run_dir.mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "logs").mkdir()

    bus = EventBus()
    writer = ArtifactWriter(run_dir, flush_every_n=1)
    bus.subscribe_all(writer)

    cfg = LogicGateConfig(
        gate="XOR",
        max_trials=8,
        seed=13,
        device="cpu",
        emit_param_grad_stats=True,
        stats_every_n_trials=1,
    )
    task = LogicGateTask(cfg, event_bus=bus)
    _ = task.run()
    writer.flush()

    with (run_dir / "metrics" / "scalars.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert rows
    assert any(row.get("w_norm_ih", "") != "" for row in rows)
    assert any(row.get("g_norm_ih", "") != "" for row in rows)
