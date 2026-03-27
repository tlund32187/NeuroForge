"""Unit tests for vision monitors and exporters."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.core.torch_utils import require_torch
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.vision_monitors import (
    ConfusionMatrixExporter,
    ConfusionMatrixMonitor,
    VisionLayerStatsExporter,
    VisionLayerStatsMonitor,
    VisionSampleGridExporter,
    VisionSampleGridMonitor,
)

if TYPE_CHECKING:
    from pathlib import Path

torch = require_torch()


def _make_run_dir(tmp_path: Path) -> Path:
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
        source="VISION",
        data=data,
    )


@pytest.mark.unit
def test_vision_monitors_export_expected_artifacts(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    bus = EventBus()

    layer_monitor = VisionLayerStatsMonitor(interval_steps=1, enabled=True)
    confusion_monitor = ConfusionMatrixMonitor(enabled=True)
    sample_monitor = VisionSampleGridMonitor(max_samples=4, enabled=True)

    bus.subscribe_all(layer_monitor)
    bus.subscribe_all(confusion_monitor)
    bus.subscribe_all(sample_monitor)
    bus.subscribe_all(VisionLayerStatsExporter(run_dir, layer_monitor, enabled=True))
    bus.subscribe_all(ConfusionMatrixExporter(run_dir, confusion_monitor, enabled=True))
    bus.subscribe_all(VisionSampleGridExporter(run_dir, sample_monitor, enabled=True))

    bus.publish(_event("training_start", {"n_classes": 3}, step=0))
    bus.publish(
        _event(
            "scalar",
            {
                "trial": 1,
                "gate": "VISION",
                "vision.layer.conv_0.spike_rate": 0.12,
                "vision.layer.conv_0.mean_activation": 0.34,
                "vision.layer.conv_0.max_activation": 0.56,
            },
            step=1,
        )
    )
    bus.publish(
        _event(
            "training_trial",
            {
                "predicted": [0, 1, 2, 1],
                "expected": [0, 2, 2, 1],
                "images": torch.rand(4, 1, 8, 8, generator=torch.Generator().manual_seed(7)),
            },
            step=1,
        )
    )
    bus.publish(_event("training_end", {"converged": False}, step=1))

    layer_stats_path = run_dir / "vision" / "metrics" / "vision_layer_stats.csv"
    confusion_path = run_dir / "vision" / "metrics" / "confusion_matrix.npy"
    per_class_path = run_dir / "vision" / "metrics" / "per_class_accuracy.json"
    grid_path = run_dir / "vision" / "samples" / "input_grid.png"

    assert layer_stats_path.exists()
    assert confusion_path.exists()
    assert per_class_path.exists()
    assert grid_path.exists()

    with layer_stats_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert rows[0]["layer"] == "conv_0"

    np = pytest.importorskip("numpy")
    matrix = np.load(confusion_path)
    assert matrix.shape == (3, 3)
    assert int(matrix[2, 2]) == 1

    per_class = json.loads(per_class_path.read_text(encoding="utf-8"))
    assert per_class["n_classes"] == 3
    assert "0" in per_class["per_class_accuracy"]
