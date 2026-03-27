"""Unit tests for vision runner dataset integration."""

from __future__ import annotations

from typing import Any

import pytest

from neuroforge.core.torch_utils import require_torch
from neuroforge.data.datasets import DatasetLoaders, DatasetMeta
from neuroforge.runners.vision import VisionRunnerConfig, run_vision_classification

torch = require_torch()


@pytest.mark.unit
def test_run_vision_classification_with_dataset_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_loaders(_self: Any, cfg: Any) -> DatasetLoaders:
        images = torch.rand(24, 1, 8, 8, generator=torch.Generator().manual_seed(3))
        labels = torch.arange(24, dtype=torch.long) % 10
        ds = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False)
        meta = DatasetMeta(
            name="mnist",
            root="fake-cache",
            split_sizes={"train": 24, "val": 0, "test": 24},
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

    cfg = VisionRunnerConfig(
        seed=5,
        device="cpu",
        dtype="float32",
        steps=3,
        batch_size=4,
        dataset="mnist",
        dataset_download=False,
        n_classes=10,
        image_channels=1,
        image_h=8,
        image_w=8,
        backbone_time_steps=2,
    )

    summary = run_vision_classification(cfg)
    result = summary["result"]
    assert int(result["steps"]) == 3
    assert len(result["loss_history"]) == 3
    assert len(result["accuracy_history"]) == 3
    train_perf = result["performance"]["train"]
    eval_perf = result["performance"]["eval"]
    assert int(train_perf["steps"]) == 3
    assert isinstance(train_perf["samples_per_sec"], float)
    assert isinstance(eval_perf["steps_per_sec"], float)
    assert summary["dataset_meta"] is not None


@pytest.mark.unit
def test_run_vision_classification_with_nmnist_event_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_build_loaders(_self: Any, cfg: Any) -> DatasetLoaders:
        frames = torch.rand(24, 4, 2, 8, 8, generator=torch.Generator().manual_seed(7))
        labels = torch.arange(24, dtype=torch.long) % 10
        ds = torch.utils.data.TensorDataset(frames, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False)
        meta = DatasetMeta(
            name="nmnist",
            root="fake-cache",
            split_sizes={"train": 24, "val": 0, "test": 24},
            transforms_summary={
                "train": ["EventSlice(mode=time,window_us=100000)", "EventTensor(mode=frames)"],
                "eval": ["EventSlice(mode=time,window_us=100000)", "EventTensor(mode=frames)"],
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

    cfg = VisionRunnerConfig(
        seed=8,
        device="cpu",
        dtype="float32",
        steps=2,
        batch_size=4,
        dataset="nmnist",
        dataset_download=False,
        n_classes=10,
        image_channels=2,
        image_h=8,
        image_w=8,
        backbone_time_steps=4,
    )

    summary = run_vision_classification(cfg)
    result = summary["result"]
    assert int(result["steps"]) == 2
    assert len(result["loss_history"]) == 2
    assert summary["dataset_meta"] is not None
    assert summary["dataset_meta"]["name"] == "nmnist"
    assert summary["dataset_meta"]["sensor_size"] == [8, 8, 2]
