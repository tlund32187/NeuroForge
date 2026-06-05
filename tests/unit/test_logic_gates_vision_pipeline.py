"""Smoke test: logic gates through the vision pipeline produce expected artifacts.

Runs 3 train steps on CPU with tiny batch to verify the full pipeline
(dataset -> backbone -> readout -> monitors -> artifact export) works end-to-end.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from neuroforge.contracts.messaging import EventTopic, MonitorEvent
from neuroforge.interfaces.cli.commands.vision import VisionRunnerConfig, run_vision_classification
from neuroforge.messaging.bus import EventBus
from neuroforge.observability.monitors.vision_monitors import (
    ConfusionMatrixExporter,
    ConfusionMatrixMonitor,
    VisionLayerStatsExporter,
    VisionLayerStatsMonitor,
    VisionSampleGridExporter,
    VisionSampleGridMonitor,
)


def _make_config(*, backbone_type: str = "lif_convnet_v1") -> VisionRunnerConfig:
    """Build a minimal CPU-friendly vision config for logic gates."""
    blocks: tuple[dict[str, object], ...] | None = None
    output_dim = 32
    if backbone_type == "lif_convnet_v1":
        blocks = (
            {"type": "conv", "params": {"out_channels": 8, "kernel_size": 3}},
            {"type": "pool", "params": {"kernel_size": 2, "mode": "avg"}},
            {"type": "conv", "params": {"out_channels": 12, "kernel_size": 3}},
        )
    elif backbone_type in {"none", "null", "identity"}:
        output_dim = 64  # NoBackbone: C*H*W = 1*8*8 = 64
    return VisionRunnerConfig(
        seed=0,
        device="cpu",
        dtype="float32",
        deterministic=True,
        steps=3,
        batch_size=8,
        n_classes=4,
        image_channels=1,
        image_h=8,
        image_w=8,
        dataset="logic_gates_pixels",
        dataset_root=".cache/logic_gates_pixels",
        dataset_download=False,
        dataset_logic_image_size=8,
        dataset_logic_gates=("AND", "OR", "NAND", "NOR"),
        dataset_logic_mode="multiclass",
        dataset_logic_samples_per_gate=64,
        lr=1e-3,
        loss_fn="bce_logits",
        readout="spike_count",
        backbone_type=backbone_type,
        backbone_time_steps=4,
        backbone_encoding_mode="rate",
        backbone_output_dim=output_dim,
        backbone_blocks=blocks,
    )


def _run_pipeline(
    tmp_path: Path,
    *,
    backbone_type: str = "lif_convnet_v1",
) -> Path:
    """Run the vision pipeline and return the run_dir surrogate (tmp_path)."""
    cfg = _make_config(backbone_type=backbone_type)
    bus = EventBus()

    layer_stats = VisionLayerStatsMonitor(interval_steps=1, enabled=True)
    confusion = ConfusionMatrixMonitor(enabled=True)
    samples = VisionSampleGridMonitor(max_samples=16, enabled=True)

    bus.subscribe_all(layer_stats)
    bus.subscribe_all(confusion)
    bus.subscribe_all(samples)
    bus.subscribe_all(VisionLayerStatsExporter(tmp_path, layer_stats, enabled=True))
    bus.subscribe_all(ConfusionMatrixExporter(tmp_path, confusion, enabled=True))
    bus.subscribe_all(VisionSampleGridExporter(tmp_path, samples, enabled=True))

    run_vision_classification(cfg, event_bus=bus)

    # Manually trigger end-of-run so exporters flush
    bus.publish(
        MonitorEvent(
            topic=EventTopic.RUN_END,
            step=cfg.steps,
            t=0.0,
            source="test",
            data={"converged": True},
        )
    )
    return tmp_path


class TestPipelineWithBackbone:
    """Logic gates through the vision pipeline with a backbone."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return _run_pipeline(tmp_path, backbone_type="lif_convnet_v1")

    def test_confusion_matrix_exists(self, run_dir: Path) -> None:
        metrics = run_dir / "vision" / "metrics"
        assert (metrics / "confusion_matrix.npy").exists()

    def test_per_class_accuracy_json_exists(self, run_dir: Path) -> None:
        metrics = run_dir / "vision" / "metrics"
        assert (metrics / "per_class_accuracy.json").exists()

    def test_sample_grid_exists(self, run_dir: Path) -> None:
        samples = run_dir / "vision" / "samples"
        assert (samples / "input_grid.png").exists()

    def test_layer_stats_exists(self, run_dir: Path) -> None:
        metrics = run_dir / "vision" / "metrics"
        assert (metrics / "vision_layer_stats.csv").exists()

    def test_per_class_accuracy_has_4_classes(self, run_dir: Path) -> None:
        path = run_dir / "vision" / "metrics" / "per_class_accuracy.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["n_classes"] == 4


class TestPipelineNoBackbone:
    """Logic gates through the vision pipeline without a backbone."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return _run_pipeline(tmp_path, backbone_type="none")

    def test_confusion_matrix_exists(self, run_dir: Path) -> None:
        metrics = run_dir / "vision" / "metrics"
        assert (metrics / "confusion_matrix.npy").exists()

    def test_sample_grid_exists(self, run_dir: Path) -> None:
        samples = run_dir / "vision" / "samples"
        assert (samples / "input_grid.png").exists()

    def test_layer_stats_csv_exists(self, run_dir: Path) -> None:
        metrics = run_dir / "vision" / "metrics"
        assert (metrics / "vision_layer_stats.csv").exists()
