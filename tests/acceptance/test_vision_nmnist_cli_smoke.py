"""Acceptance smoke test for NMNIST vision CLI path."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import neuroforge.runners.cli as cli
from neuroforge.core.torch_utils import require_torch
from neuroforge.data.datasets import DatasetLoaders, DatasetMeta

if TYPE_CHECKING:
    from pathlib import Path

torch = require_torch()


@pytest.mark.acceptance
def test_vision_cli_nmnist_short_train_eval_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_build_loaders(_self: Any, cfg: Any) -> DatasetLoaders:
        frames = torch.rand(24, 4, 2, 8, 8, generator=torch.Generator().manual_seed(77))
        labels = torch.arange(24, dtype=torch.long) % 10
        ds = torch.utils.data.TensorDataset(frames, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False)
        meta = DatasetMeta(
            name="nmnist",
            root="fake-tonic-cache",
            split_sizes={"train": 24, "val": 0, "test": 24},
            transforms_summary={
                "train": [
                    "EventSlice(mode=time,window_us=100000,event_count=20000)",
                    "EventTensor(mode=frames,time_steps=4,polarity_channels=2)",
                ],
                "eval": [
                    "EventSlice(mode=time,window_us=100000,event_count=20000)",
                    "EventTensor(mode=frames,time_steps=4,polarity_channels=2)",
                ],
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

    artifacts_dir = tmp_path / "artifacts"
    parser = cli._build_parser()
    args = parser.parse_args([
        "vision",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--dataset",
        "nmnist",
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--n-classes",
        "10",
        "--image-channels",
        "2",
        "--image-h",
        "8",
        "--image-w",
        "8",
        "--backbone-time-steps",
        "4",
        "--dataset-event-tensor-mode",
        "frames",
        "--dataset-event-slice-mode",
        "time",
        "--dataset-event-window-us",
        "100000",
        "--artifacts",
        str(artifacts_dir),
    ])
    rc = cli._cmd_vision(args)
    assert rc == 0

    run_dirs = sorted(p for p in artifacts_dir.glob("run_*") if p.is_dir())
    assert run_dirs
    run_dir = run_dirs[-1]
    summary = json.loads((run_dir / "vision" / "vision_summary.json").read_text(encoding="utf-8"))
    assert int(summary["result"]["steps"]) == 2
    assert int(summary["result"]["eval_steps"]) >= 1
    assert summary["dataset_meta"]["name"] == "nmnist"
    assert summary["dataset_meta"]["sensor_size"] == [8, 8, 2]
