"""Unit tests for LogicGatesPixelsDataset: shapes, labels, determinism."""

from __future__ import annotations

import pytest

from neuroforge.core.torch_utils import require_torch
from neuroforge.data.logic_gates_pixels import (
    LogicGatesPixelsConfig,
    LogicGatesPixelsSplits,
    build_logic_gates_pixels_splits,
)

torch = require_torch()


def _build_splits(
    *,
    mode: str = "multiclass",
    gates: tuple[str, ...] = ("AND", "OR", "NAND", "NOR"),
    seed: int = 42,
    samples_per_gate: int = 64,
    image_size: int = 8,
) -> LogicGatesPixelsSplits:
    cfg = LogicGatesPixelsConfig(
        image_size=image_size,
        gates=gates,
        mode=mode,
        samples_per_gate=samples_per_gate,
        seed=seed,
    )
    return build_logic_gates_pixels_splits(cfg)


class TestMulticlassShapesAndLabels:
    """Multiclass mode: gate classification across AND/OR/NAND/NOR."""

    def test_train_image_shape(self) -> None:
        splits = _build_splits()
        img, _label = splits.train[0]
        assert tuple(img.shape) == (1, 8, 8)

    def test_train_image_dtype(self) -> None:
        splits = _build_splits()
        img, _label = splits.train[0]
        assert img.dtype == torch.float32

    def test_label_dtype(self) -> None:
        splits = _build_splits()
        _img, label = splits.train[0]
        assert label.dtype == torch.long

    def test_label_range(self) -> None:
        splits = _build_splits()
        for idx in range(len(splits.train)):
            _img, label = splits.train[idx]
            assert 0 <= int(label.item()) < splits.n_classes

    def test_n_classes_matches_gates(self) -> None:
        splits = _build_splits(gates=("AND", "OR", "NAND", "NOR"))
        assert splits.n_classes == 4

    def test_class_names_match_gates(self) -> None:
        splits = _build_splits(gates=("AND", "OR"))
        assert splits.class_names == ("AND", "OR")

    def test_splits_non_empty(self) -> None:
        splits = _build_splits()
        assert len(splits.train) > 0
        assert len(splits.val) > 0
        assert len(splits.test) > 0

    def test_total_sample_count(self) -> None:
        splits = _build_splits(gates=("AND", "OR"), samples_per_gate=64)
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert total == 128  # 2 gates * 64


class TestSingleGateMode:
    """Single-gate mode: binary classification for one gate."""

    def test_n_classes_is_two(self) -> None:
        splits = _build_splits(mode="single_gate", gates=("XOR",))
        assert splits.n_classes == 2

    def test_labels_are_binary(self) -> None:
        cfg = LogicGatesPixelsConfig(
            mode="single_gate",
            single_gate="AND",
            samples_per_gate=32,
            seed=42,
        )
        splits = build_logic_gates_pixels_splits(cfg)
        for idx in range(len(splits.train)):
            _img, label = splits.train[idx]
            assert int(label.item()) in {0, 1}


class TestDeterminism:
    """Same seed must produce identical datasets."""

    def test_same_seed_same_data(self) -> None:
        a = _build_splits(seed=99)
        b = _build_splits(seed=99)
        assert len(a.train) == len(b.train)
        for idx in range(len(a.train)):
            img_a, lbl_a = a.train[idx]
            img_b, lbl_b = b.train[idx]
            assert torch.equal(img_a, img_b)
            assert int(lbl_a.item()) == int(lbl_b.item())

    def test_different_seed_different_data(self) -> None:
        a = _build_splits(seed=1)
        b = _build_splits(seed=2)
        # With different seeds, at least one sample should differ
        any_differ = False
        n = min(len(a.train), len(b.train))
        for idx in range(n):
            img_a, _la = a.train[idx]
            img_b, _lb = b.train[idx]
            if not torch.equal(img_a, img_b):
                any_differ = True
                break
        assert any_differ


class TestImageSizeVariation:
    """Verify non-default image sizes work."""

    @pytest.mark.parametrize("size", [6, 8, 12, 16])
    def test_image_size(self, size: int) -> None:
        splits = _build_splits(image_size=size, samples_per_gate=8)
        img, _label = splits.train[0]
        assert tuple(img.shape) == (1, size, size)


class TestConfigValidation:
    """Invalid configs must raise ValueError."""

    def test_image_size_too_small(self) -> None:
        with pytest.raises(ValueError, match="image_size"):
            LogicGatesPixelsConfig(image_size=4)

    def test_unsupported_gate(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            LogicGatesPixelsConfig(gates=("INVALID",))

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            LogicGatesPixelsConfig(mode="bogus")
