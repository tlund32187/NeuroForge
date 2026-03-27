"""Unit tests for event dataset adaptation to [T, C, H, W] tensors."""

from __future__ import annotations

from typing import Any

import pytest

from neuroforge.core.torch_utils import require_torch
from neuroforge.data.event_dataset_adapter import (
    EventDatasetAdapter,
    EventSliceConfig,
    EventTensorConfig,
    TonicDatasetConfig,
    build_tonic_dataset,
)

torch = require_torch()


class _FakeEventDataset:
    def __init__(self) -> None:
        self._events = torch.tensor(
            [
                [10, 1, 1, 0],
                [20, 2, 1, 1],
                [35, 2, 2, 0],
                [50, 3, 2, 1],
                [90, 4, 3, 0],
                [120, 4, 4, 1],
            ],
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[Any, int]:
        del index
        return self._events.clone(), 3


class _FakeMappingEventDataset:
    def __init__(self) -> None:
        self._events = {
            "t": torch.tensor([10, 30, 45, 70, 120], dtype=torch.int64),
            "x": torch.tensor([1, 1, 2, 2, 3], dtype=torch.int64),
            "y": torch.tensor([1, 2, 2, 3, 3], dtype=torch.int64),
            "polarity": torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[Any, int]:
        del index
        return dict(self._events), 7


class _FakeNMNIST:
    def __init__(self, *, save_to: str, train: bool, download: bool) -> None:
        self.save_to = save_to
        self.train = bool(train)
        self.download = bool(download)


class _FakePOKERDVS:
    def __init__(self, *, save_to: str, train: bool, download: bool) -> None:
        self.save_to = save_to
        self.train = bool(train)
        self.download = bool(download)


class _FakeTonicDatasets:
    NMNIST = _FakeNMNIST
    POKERDVS = _FakePOKERDVS


class _FakeTonic:
    datasets = _FakeTonicDatasets


@pytest.mark.unit
def test_event_adapter_frames_mode_count_window_is_stable() -> None:
    dataset = _FakeEventDataset()
    tensor_cfg = EventTensorConfig(
        time_steps=5,
        height=8,
        width=8,
        polarity_channels=2,
        mode="frames",
    )
    slice_cfg = EventSliceConfig(mode="count", event_count=4)
    adapter = EventDatasetAdapter(
        dataset,
        tensor_config=tensor_cfg,
        slice_config=slice_cfg,
    )

    sample_a, label_a = adapter[0]
    sample_b, label_b = adapter[0]

    assert tuple(sample_a.shape) == (5, 2, 8, 8)
    assert label_a == 3
    assert label_b == 3
    assert torch.equal(sample_a, sample_b)
    assert float(sample_a.sum().item()) == pytest.approx(4.0, abs=1e-8)


@pytest.mark.unit
def test_event_adapter_frames_mode_time_window_uses_fixed_horizon() -> None:
    dataset = _FakeMappingEventDataset()
    tensor_cfg = EventTensorConfig(
        time_steps=6,
        height=6,
        width=6,
        polarity_channels=2,
        mode="frames",
    )
    slice_cfg = EventSliceConfig(mode="time", window_us=40)
    adapter = EventDatasetAdapter(
        dataset,
        tensor_config=tensor_cfg,
        slice_config=slice_cfg,
    )

    sample, label = adapter[0]
    assert tuple(sample.shape) == (6, 2, 6, 6)
    assert label == 7
    # t starts at 10, window is [10, 50) -> first three events.
    assert float(sample.sum().item()) == pytest.approx(3.0, abs=1e-8)


@pytest.mark.unit
def test_event_adapter_voxel_grid_mode_is_deterministic() -> None:
    dataset = _FakeEventDataset()
    tensor_cfg = EventTensorConfig(
        time_steps=4,
        height=8,
        width=8,
        polarity_channels=2,
        mode="voxel_grid",
    )
    slice_cfg = EventSliceConfig(mode="count", event_count=5)
    adapter_1 = EventDatasetAdapter(
        dataset,
        tensor_config=tensor_cfg,
        slice_config=slice_cfg,
    )
    adapter_2 = EventDatasetAdapter(
        dataset,
        tensor_config=tensor_cfg,
        slice_config=slice_cfg,
    )

    sample_a, _ = adapter_1[0]
    sample_b, _ = adapter_2[0]

    assert tuple(sample_a.shape) == (4, 2, 8, 8)
    assert torch.equal(sample_a, sample_b)
    assert float(sample_a.sum().item()) == pytest.approx(5.0, abs=1e-6)


@pytest.mark.unit
def test_build_tonic_dataset_uses_signature_aware_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "neuroforge.data.event_dataset_adapter._require_tonic",
        lambda: _FakeTonic,
    )
    cfg = TonicDatasetConfig(
        name="nmnist",
        root="cache-tonic",
        train=False,
        download=False,
    )
    ds = build_tonic_dataset(cfg)
    assert isinstance(ds, _FakeNMNIST)
    assert ds.save_to == "cache-tonic"
    assert ds.train is False
    assert ds.download is False


@pytest.mark.unit
def test_build_tonic_dataset_resolves_pokerdvs_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "neuroforge.data.event_dataset_adapter._require_tonic",
        lambda: _FakeTonic,
    )
    cfg = TonicDatasetConfig(
        name="pokerdvs",
        root="cache-tonic",
        train=True,
        download=False,
    )
    ds = build_tonic_dataset(cfg)
    assert isinstance(ds, _FakePOKERDVS)
    assert ds.save_to == "cache-tonic"


@pytest.mark.unit
def test_build_tonic_dataset_rejects_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "neuroforge.data.event_dataset_adapter._require_tonic",
        lambda: _FakeTonic,
    )
    with pytest.raises(ValueError, match="Unsupported tonic dataset"):
        build_tonic_dataset(TonicDatasetConfig(name="unknown_events"))
