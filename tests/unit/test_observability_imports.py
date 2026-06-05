"""Observability package shape and canonical imports."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_observability_layout_matches_target() -> None:
    root = _repo_root() / "src" / "neuroforge" / "observability"

    expected_files = [
        "__init__.py",
        "events/__init__.py",
        "events/recorder.py",
        "events/replay.py",
        "monitors/__init__.py",
        "monitors/base.py",
        "monitors/artifact_writer.py",
        "monitors/resource_monitor.py",
        "monitors/spike_monitor.py",
        "monitors/voltage_monitor.py",
        "monitors/weight_monitor.py",
        "monitors/stability_monitor.py",
        "monitors/topology_activity_monitor.py",
        "monitors/topology_stats_monitor.py",
        "monitors/training_monitor.py",
        "monitors/trial_stats_monitor.py",
        "monitors/vision_monitors.py",
        "artifacts/__init__.py",
        "artifacts/writer.py",
        "artifacts/schemas.py",
        "artifacts/run_layout.py",
        "metrics/__init__.py",
        "metrics/scalar_schema.py",
        "metrics/aggregation.py",
    ]
    for rel in expected_files:
        assert (root / rel).is_file(), rel

    removed_files = [
        "monitors/event_recorder.py",
        "monitors/scalars_schema.py",
        "monitors/cuda_monitor.py",
    ]
    for rel in removed_files:
        assert not (root / rel).exists(), rel


def test_observability_canonical_imports() -> None:
    from neuroforge.observability.artifacts import JsonArtifactWriter, RunLayout
    from neuroforge.observability.events import EventRecorderMonitor, iter_event_records
    from neuroforge.observability.metrics import build_scalar_fields
    from neuroforge.observability.monitors.artifact_writer import ArtifactWriter
    from neuroforge.observability.monitors.base import MonitorBase
    from neuroforge.observability.monitors.resource_monitor import (
        CudaMetricsMonitor,
        ResourceMonitor,
    )

    assert ArtifactWriter is not None
    assert CudaMetricsMonitor is not None
    assert EventRecorderMonitor is not None
    assert JsonArtifactWriter is not None
    assert MonitorBase is not None
    assert ResourceMonitor is not None
    assert RunLayout is not None
    assert callable(build_scalar_fields)
    assert callable(iter_event_records)


def test_removed_observability_modules_are_not_importable() -> None:
    removed_modules = [
        "neuroforge.observability.monitors.event_recorder",
        "neuroforge.observability.monitors.scalars_schema",
        "neuroforge.observability.monitors.cuda_monitor",
    ]
    for module in removed_modules:
        assert importlib.util.find_spec(module) is None, module


def test_no_code_imports_removed_observability_modules() -> None:
    old_paths = [
        "neuroforge.observability.monitors.event_recorder",
        "neuroforge.observability.monitors.scalars_schema",
        "neuroforge.observability.monitors.cuda_monitor",
    ]
    roots = [
        _repo_root() / "src",
        _repo_root() / "scripts",
        _repo_root() / "tests" / "unit",
    ]

    offenders: list[tuple[str, str]] = []
    this_file = Path(__file__).resolve()
    for root in roots:
        for path in root.rglob("*.py"):
            if path.resolve() == this_file:
                continue
            text = path.read_text(encoding="utf-8")
            for old_path in old_paths:
                if old_path in text:
                    offenders.append((str(path.relative_to(_repo_root())), old_path))

    assert offenders == []
