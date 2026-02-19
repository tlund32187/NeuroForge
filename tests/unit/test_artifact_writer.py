"""Unit tests for ArtifactWriter monitor."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.artifact_writer import ArtifactWriter
from neuroforge.monitors.bus import EventBus

if TYPE_CHECKING:
    from pathlib import Path


def _make_run_dir(tmp_path: Path) -> Path:
    """Create a run directory with metrics/ and logs/ subdirs."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "logs").mkdir()
    return run_dir


def _event(topic: str, data: dict[str, object], step: int = 0) -> MonitorEvent:
    return MonitorEvent(
        topic=EventTopic(topic),
        step=step,
        t=0.0,
        source="TEST",
        data=data,
    )


# ── RUN_START writes run_meta + config ──────────────────────────────


class TestRunStart:
    def test_writes_run_meta_and_config(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        writer.on_event(_event("run_start", {
            "run_meta": {"run_id": "run_test", "seed": 42},
            "config": {"gate": "XOR", "lr": 0.005},
        }))

        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["run_id"] == "run_test"
        assert meta["seed"] == 42

        config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
        assert config["gate"] == "XOR"
        assert config["lr"] == 0.005

    def test_skips_when_disabled(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir, enabled=False)

        writer.on_event(_event("run_start", {
            "run_meta": {"run_id": "x"},
            "config": {"gate": "OR"},
        }))

        assert not (run_dir / "run_meta.json").exists()


# ── TOPOLOGY writes topology.json (once) ────────────────────────────


class TestTopology:
    def test_writes_topology(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        writer.on_event(_event("topology", {
            "layers": ["input(2)", "hidden(6)", "output(1)"],
            "edges": [],
        }))

        topo = json.loads((run_dir / "topology.json").read_text(encoding="utf-8"))
        assert topo["layers"] == ["input(2)", "hidden(6)", "output(1)"]

    def test_writes_once(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        writer.on_event(_event("topology", {"layers": ["A"], "edges": []}))
        writer.on_event(_event("topology", {"layers": ["B"], "edges": []}))

        topo = json.loads((run_dir / "topology.json").read_text(encoding="utf-8"))
        assert topo["layers"] == ["A"]

    def test_handles_torch_tensors(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        writer.on_event(_event("topology", {
            "layers": ["input(3)"],
            "edges": [{"src": "a", "dst": "b", "weights": t}],
        }))

        topo = json.loads((run_dir / "topology.json").read_text(encoding="utf-8"))
        assert topo["edges"][0]["weights"] == [1.0, 2.0, 3.0]


# ── SCALAR writes CSV rows ──────────────────────────────────────────


class TestScalar:
    def test_appends_csv_rows(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir, flush_every_n=2)

        for i in range(4):
            writer.on_event(_event("scalar", {
                "trial": i,
                "epoch": 0,
                "gate": "XOR",
                "accuracy": 0.5,
                "error": 0.1,
                "correct": True,
            }, step=i))

        writer.flush()

        csv_path = run_dir / "metrics" / "scalars.csv"
        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as fh:
            reader = list(csv.DictReader(fh))
        assert len(reader) == 4
        assert reader[0]["gate"] == "XOR"

    def test_training_trial_also_writes_csv(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir, flush_every_n=1)

        writer.on_event(_event("training_trial", {
            "gate": "AND",
            "accuracy": 1.0,
            "error": 0.0,
            "correct": True,
        }, step=5))

        csv_path = run_dir / "metrics" / "scalars.csv"
        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as fh:
            reader = list(csv.DictReader(fh))
        assert len(reader) == 1
        assert reader[0]["gate"] == "AND"

    def test_dynamic_resource_columns_are_added(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir, flush_every_n=1)

        writer.on_event(_event("scalar", {
            "trial": 0,
            "gate": "XOR",
            "accuracy": 0.5,
            "error": 0.5,
            "resource.cpu.system_percent": 12.5,
            "resource.ram.process_rss_mb": 256.0,
            "torch_cuda_allocated_mb": 0.0,
        }, step=0))
        writer.on_event(_event("scalar", {
            "trial": 1,
            "gate": "XOR",
            "accuracy": 0.75,
            "error": 0.25,
            "resource.gpu.util_percent": 88.0,
        }, step=1))
        writer.flush()

        csv_path = run_dir / "metrics" / "scalars.csv"
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            assert reader.fieldnames is not None
            assert "resource.cpu.system_percent" in reader.fieldnames
            assert "resource.ram.process_rss_mb" in reader.fieldnames
            assert "resource.gpu.util_percent" in reader.fieldnames
            assert "torch_cuda_allocated_mb" in reader.fieldnames

        assert len(rows) == 2
        assert rows[0]["resource.cpu.system_percent"] == "12.5"
        assert rows[0]["resource.gpu.util_percent"] == ""
        assert rows[1]["resource.gpu.util_percent"] == "88.0"


# ── RUN_END flushes and writes summary ──────────────────────────────


class TestRunEnd:
    def test_writes_training_end_and_log(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        writer.add_log_line("line1")
        writer.add_log_line("line2")

        writer.on_event(_event("run_end", {
            "converged": True,
            "trials": 100,
            "wall_ms": 42.0,
        }))

        end = json.loads((run_dir / "training_end.json").read_text(encoding="utf-8"))
        assert end["converged"] is True
        assert end["trials"] == 100

        log = (run_dir / "logs" / "run.log").read_text(encoding="utf-8")
        assert "line1" in log
        assert "line2" in log

    def test_training_end_also_triggers(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        writer.on_event(_event("training_end", {"converged": False, "trials": 50}))

        assert (run_dir / "training_end.json").exists()


# ── Integration with EventBus ───────────────────────────────────────


class TestBusIntegration:
    def test_full_lifecycle(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        bus = EventBus()
        writer = ArtifactWriter(run_dir, flush_every_n=1)
        bus.subscribe_all(writer)

        bus.publish(_event("run_start", {
            "run_meta": {"run_id": "integration"},
            "config": {"seed": 1},
        }))
        bus.publish(_event("topology", {
            "layers": ["input(2)", "output(1)"],
            "edges": [],
        }))
        bus.publish(_event("scalar", {
            "trial": 0, "epoch": 0, "gate": "OR",
            "accuracy": 0.75, "error": 0.25, "correct": False,
        }))
        writer.add_log_line("done")
        bus.publish(_event("run_end", {
            "converged": True, "trials": 1, "wall_ms": 10.0,
        }))

        assert (run_dir / "run_meta.json").exists()
        assert (run_dir / "config_resolved.json").exists()
        assert (run_dir / "topology.json").exists()
        assert (run_dir / "metrics" / "scalars.csv").exists()
        assert (run_dir / "training_end.json").exists()
        assert (run_dir / "logs" / "run.log").exists()


# ── reset / snapshot ────────────────────────────────────────────────


class TestResetSnapshot:
    def test_reset_clears_state(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        writer.on_event(_event("topology", {"layers": [], "edges": []}))
        assert writer.snapshot()["topology_written"] is True

        writer.reset()
        assert writer.snapshot()["topology_written"] is False
        assert writer.snapshot()["csv_rows"] == 0

    def test_snapshot_reports_csv_rows(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        writer = ArtifactWriter(run_dir)

        for i in range(3):
            writer.on_event(_event("scalar", {
                "trial": i, "error": 0,
            }, step=i))

        assert writer.snapshot()["csv_rows"] == 3
