"""Unit tests for synthetic vision classification task wiring."""

from __future__ import annotations

from typing import Any

import pytest

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.core.torch_utils import require_torch
from neuroforge.data.datasets import DatasetLoaders, DatasetMeta
from neuroforge.monitors.bus import EventBus
from neuroforge.tasks.vision_classification import (
    VisionClassificationConfig,
    VisionClassificationTask,
)

torch = require_torch()


class _CaptureMonitor:
    """Collects all monitor events for assertions."""

    def __init__(self) -> None:
        self._enabled = True
        self.events: list[MonitorEvent] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def on_event(self, event: MonitorEvent) -> None:
        if not self._enabled:
            return
        self.events.append(event)

    def reset(self) -> None:
        self.events.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"count": len(self.events)}


@pytest.mark.unit
def test_vision_classification_task_one_step_emits_scalars() -> None:
    bus = EventBus()
    capture = _CaptureMonitor()
    bus.subscribe_all(capture)

    cfg = VisionClassificationConfig(
        seed=11,
        device="cpu",
        dtype="float32",
        train_steps=1,
        batch_size=4,
        n_classes=3,
        image_channels=1,
        image_h=8,
        image_w=8,
        backbone_time_steps=4,
        backbone_output_dim=16,
    )
    task = VisionClassificationTask(cfg, event_bus=bus)
    result = task.run()

    assert result.steps == 1
    assert len(result.loss_history) == 1
    assert len(result.accuracy_history) == 1

    scalar_events = [evt for evt in capture.events if evt.topic == EventTopic.SCALAR]
    assert scalar_events, "Expected at least one scalar event"
    assert "loss" in scalar_events[-1].data
    assert "accuracy" in scalar_events[-1].data

    topics = {evt.topic for evt in capture.events}
    assert EventTopic.TRAINING_START in topics
    assert EventTopic.TRAINING_END in topics
    assert EventTopic.TRAINING_TRIAL in topics


@pytest.mark.unit
def test_vision_classification_task_fixed_seed_reproducible_cpu() -> None:
    cfg = VisionClassificationConfig(
        seed=99,
        device="cpu",
        dtype="float32",
        train_steps=1,
        batch_size=3,
        n_classes=3,
        image_channels=1,
        image_h=8,
        image_w=8,
        backbone_time_steps=3,
        backbone_output_dim=12,
    )

    result_a = VisionClassificationTask(cfg).run()
    result_b = VisionClassificationTask(cfg).run()
    assert result_a.steps == result_b.steps == 1
    assert result_a.final_loss == pytest.approx(result_b.final_loss, rel=1e-8, abs=1e-8)
    assert result_a.final_accuracy == pytest.approx(
        result_b.final_accuracy, rel=1e-8, abs=1e-8,
    )


@pytest.mark.unit
def test_vision_classification_task_mnist_mode_runs_multiple_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_build_loaders(_self: Any, cfg: Any) -> DatasetLoaders:
        images = torch.linspace(
            0.0,
            1.0,
            steps=32 * 1 * 8 * 8,
            dtype=torch.float32,
        ).reshape(32, 1, 8, 8)
        labels = torch.arange(32, dtype=torch.long) % 10
        ds = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False)
        meta = DatasetMeta(
            name="mnist",
            root="fake-cache",
            split_sizes={"train": 32, "val": 0, "test": 32},
            transforms_summary={"train": ["ToTensor()"], "eval": ["ToTensor()"]},
            n_classes=10,
            channels=1,
            height=8,
            width=8,
        )
        return DatasetLoaders(train=loader, val=loader, test=loader, meta=meta)

    monkeypatch.setattr(
        "neuroforge.data.datasets.DatasetFactory.build_loaders",
        _fake_build_loaders,
    )

    cfg = VisionClassificationConfig(
        seed=21,
        device="cpu",
        dtype="float32",
        train_steps=3,
        batch_size=4,
        n_classes=10,
        dataset_name="mnist",
        dataset_download=False,
        image_channels=1,
        image_h=8,
        image_w=8,
        backbone_time_steps=3,
        backbone_output_dim=16,
    )
    result = VisionClassificationTask(cfg).run()

    assert result.steps == 3
    assert len(result.loss_history) == 3
    assert len(result.accuracy_history) == 3
    assert result.dataset_meta is not None
    assert result.dataset_meta["name"] == "mnist"


@pytest.mark.unit
def test_vision_classification_task_nmnist_event_frames_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_build_loaders(_self: Any, cfg: Any) -> DatasetLoaders:
        frames = torch.linspace(
            0.0,
            1.0,
            steps=32 * 4 * 2 * 8 * 8,
            dtype=torch.float32,
        ).reshape(32, 4, 2, 8, 8)
        labels = torch.arange(32, dtype=torch.long) % 10
        ds = torch.utils.data.TensorDataset(frames, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False)
        meta = DatasetMeta(
            name="nmnist",
            root="fake-cache",
            split_sizes={"train": 32, "val": 0, "test": 32},
            transforms_summary={
                "train": ["EventSlice(mode=count,event_count=1000)", "EventTensor(mode=frames)"],
                "eval": ["EventSlice(mode=count,event_count=1000)", "EventTensor(mode=frames)"],
            },
            n_classes=10,
            channels=2,
            height=8,
            width=8,
            sensor_size=(8, 8, 2),
        )
        return DatasetLoaders(train=loader, val=loader, test=loader, meta=meta)

    monkeypatch.setattr(
        "neuroforge.data.datasets.DatasetFactory.build_loaders",
        _fake_build_loaders,
    )

    cfg = VisionClassificationConfig(
        seed=31,
        device="cpu",
        dtype="float32",
        train_steps=2,
        batch_size=4,
        n_classes=10,
        dataset_name="nmnist",
        dataset_download=False,
        image_channels=2,
        image_h=8,
        image_w=8,
        backbone_time_steps=4,
        backbone_output_dim=16,
    )
    result = VisionClassificationTask(cfg).run()

    assert result.steps == 2
    assert len(result.loss_history) == 2
    assert len(result.accuracy_history) == 2
    assert result.dataset_meta is not None
    assert result.dataset_meta["name"] == "nmnist"
    assert result.dataset_meta["sensor_size"] == [8, 8, 2]


@pytest.mark.unit
def test_vision_classification_task_logic_dataset_without_backbone_emits_layer_stats() -> None:
    bus = EventBus()
    capture = _CaptureMonitor()
    bus.subscribe_all(capture)

    cfg = VisionClassificationConfig(
        seed=41,
        device="cpu",
        dtype="float32",
        train_steps=2,
        batch_size=8,
        n_classes=4,
        dataset_name="logic_gates_pixels",
        dataset_logic_mode="multiclass",
        dataset_logic_gates=("AND", "OR", "NAND", "NOR"),
        dataset_logic_samples_per_gate=40,
        dataset_logic_image_size=8,
        image_channels=1,
        image_h=8,
        image_w=8,
        backbone_type="none",
    )
    result = VisionClassificationTask(cfg, event_bus=bus).run()

    assert result.steps == 2
    assert result.dataset_meta is not None
    assert result.dataset_meta["name"] == "logic_gates_pixels"

    scalar_events = [evt for evt in capture.events if evt.topic == EventTopic.SCALAR]
    assert scalar_events
    layer_keys = {
        str(key)
        for evt in scalar_events
        for key in evt.data.keys()
        if str(key).startswith("vision.layer.")
    }
    assert "vision.layer.head.spike_rate" in layer_keys
    assert "vision.layer.features.spike_rate" in layer_keys
    assert any(key.startswith("vision.layer.input.") for key in layer_keys)


@pytest.mark.unit
def test_vision_classification_task_logic_dataset_backbone_handles_short_last_batch() -> None:
    cfg = VisionClassificationConfig(
        seed=43,
        device="cpu",
        dtype="float32",
        train_steps=4,
        batch_size=16,
        n_classes=4,
        dataset_name="logic_gates_pixels",
        dataset_logic_mode="multiclass",
        dataset_logic_gates=("AND", "OR", "NAND", "NOR"),
        dataset_logic_samples_per_gate=19,
        dataset_logic_image_size=8,
        image_channels=1,
        image_h=8,
        image_w=8,
        backbone_type="lif_convnet_v1",
        backbone_time_steps=4,
        backbone_output_dim=16,
    )

    result = VisionClassificationTask(cfg).run()
    assert result.steps == 4
