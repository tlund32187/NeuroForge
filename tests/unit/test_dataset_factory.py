"""Unit tests for torchvision dataset factory wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from neuroforge.core.torch_utils import require_torch
from neuroforge.data.datasets import DatasetFactory, DatasetFactoryConfig, DatasetTransformConfig

if TYPE_CHECKING:
    from pathlib import Path

torch = require_torch()


class _Compose:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = [*transforms]

    def __call__(self, value: Any) -> Any:
        out = value
        for transform in self.transforms:
            out = transform(out)
        return out

    def __repr__(self) -> str:
        names = ", ".join(t.__class__.__name__ for t in self.transforms)
        return f"Compose([{names}])"


class _ToTensor:
    def __call__(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32)

    def __repr__(self) -> str:
        return "ToTensor()"


class _Normalize:
    def __init__(self, *, mean: tuple[float], std: tuple[float]) -> None:
        self._mean = float(mean[0])
        self._std = float(std[0])

    def __call__(self, value: Any) -> Any:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return (tensor - self._mean) / self._std

    def __repr__(self) -> str:
        return f"Normalize(mean={[self._mean]}, std={[self._std]})"


class _RandomCrop:
    def __init__(self, size: int, *, padding: int = 0) -> None:
        self._size = int(size)
        self._padding = int(padding)

    def __call__(self, value: Any) -> Any:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if self._padding > 0:
            tensor = torch.nn.functional.pad(
                tensor,
                (self._padding, self._padding, self._padding, self._padding),
            )
        height = int(tensor.shape[-2])
        width = int(tensor.shape[-1])
        top = max((height - self._size) // 2, 0)
        left = max((width - self._size) // 2, 0)
        return tensor[..., top:top + self._size, left:left + self._size]

    def __repr__(self) -> str:
        return f"RandomCrop(size={self._size}, padding={self._padding})"


class _RandomHorizontalFlip:
    def __call__(self, value: Any) -> Any:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return torch.flip(tensor, dims=(-1,))

    def __repr__(self) -> str:
        return "RandomHorizontalFlip()"


class _FakeTransforms:
    Compose = _Compose
    ToTensor = _ToTensor
    Normalize = _Normalize
    RandomCrop = _RandomCrop
    RandomHorizontalFlip = _RandomHorizontalFlip


class _FakeMNIST(torch.utils.data.Dataset[Any]):
    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Any | None = None,
    ) -> None:
        del root, download
        self._transform = transform
        self._length = 64 if train else 20

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[Any, int]:
        base = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 28, 28)
        image = torch.clamp(base + float(index % 7) * 0.01, 0.0, 1.0)
        if self._transform is not None:
            image = self._transform(image)
        label = int(index % 10)
        return image, label


class _FakeDatasets:
    MNIST = _FakeMNIST
    FashionMNIST = _FakeMNIST
    Subset = torch.utils.data.Subset


def _patch_fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "neuroforge.data.datasets._require_torchvision",
        lambda: (_FakeDatasets, _FakeTransforms),
    )


class _FakeEventDataset:
    def __init__(self, *, dataset_name: str, train: bool) -> None:
        self._dataset_name = dataset_name
        self._train = bool(train)
        if dataset_name == "nmnist":
            self.sensor_size = (34, 34, 2)
            self.classes = list(range(10))
        elif dataset_name == "pokerdvs":
            self.sensor_size = (35, 35, 2)
            self.classes = list(range(4))
        else:
            self.sensor_size = (32, 32, 2)
            self.classes = list(range(2))
        self._length = 40 if self._train else 16

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[Any, int]:
        n_events = 24
        base = torch.arange(n_events, dtype=torch.int64)
        t = base * 1000
        width = int(self.sensor_size[0])
        height = int(self.sensor_size[1])
        x = (base + int(index)) % width
        y = ((base * 2) + int(index)) % height
        p = (base + int(index)) % 2
        events = torch.stack([t, x, y, p], dim=1)
        label = int(index % len(self.classes))
        return events, label


def _patch_fake_tonic(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_tonic_dataset(cfg: Any) -> Any:
        return _FakeEventDataset(
            dataset_name=str(cfg.name).strip().lower(),
            train=bool(cfg.train),
        )

    monkeypatch.setattr(
        "neuroforge.data.datasets.build_tonic_dataset",
        _fake_build_tonic_dataset,
    )


@pytest.mark.unit
def test_dataset_factory_mnist_smoke_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_torchvision(monkeypatch)
    cfg = DatasetFactoryConfig(
        name="mnist",
        root=str(tmp_path / "torchvision"),
        batch_size=8,
        val_fraction=0.2,
        download=False,
        seed=7,
        transforms=DatasetTransformConfig(
            normalize=True,
            random_crop=False,
            random_horizontal_flip=False,
        ),
    )

    loaders = DatasetFactory().build_loaders(cfg)
    images, labels = next(iter(loaders.train))

    assert tuple(images.shape) == (8, 1, 28, 28)
    assert tuple(labels.shape) == (8,)
    assert labels.dtype == torch.long
    assert loaders.meta.name == "mnist"
    assert loaders.meta.split_sizes["train"] == 51
    assert loaders.meta.split_sizes["val"] == 13
    assert loaders.meta.split_sizes["test"] == 20
    assert "ToTensor()" in loaders.meta.transforms_summary["train"][0]


@pytest.mark.unit
def test_dataset_factory_requires_fashion_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow_fashion_mnist=False"):
        DatasetFactoryConfig(
            name="fashion_mnist",
            root=str(tmp_path),
        )


@pytest.mark.unit
def test_dataset_factory_nmnist_smoke_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_tonic(monkeypatch)
    cfg = DatasetFactoryConfig(
        name="nmnist",
        root=str(tmp_path / "tonic"),
        batch_size=4,
        val_fraction=0.25,
        download=False,
        seed=9,
        event_time_steps=5,
        event_tensor_mode="frames",
        event_slice_mode="count",
        event_count=12,
        event_polarity_channels=2,
    )

    loaders = DatasetFactory().build_loaders(cfg)
    images, labels = next(iter(loaders.train))

    assert tuple(images.shape) == (4, 5, 2, 34, 34)
    assert tuple(labels.shape) == (4,)
    assert labels.dtype == torch.long
    assert loaders.meta.name == "nmnist"
    assert loaders.meta.split_sizes["train"] == 30
    assert loaders.meta.split_sizes["val"] == 10
    assert loaders.meta.split_sizes["test"] == 16
    assert loaders.meta.sensor_size == (34, 34, 2)
    assert any(
        "EventTensor(" in row
        for row in loaders.meta.transforms_summary["train"]
    )


@pytest.mark.unit
def test_dataset_factory_pokerdvs_loads_and_yields_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_tonic(monkeypatch)
    cfg = DatasetFactoryConfig(
        name="pokerdvs",
        root=str(tmp_path / "tonic"),
        batch_size=3,
        val_fraction=0.2,
        download=False,
        seed=5,
        event_time_steps=6,
        event_tensor_mode="voxel_grid",
        event_slice_mode="time",
        event_window_us=50_000,
        event_polarity_channels=2,
    )

    loaders = DatasetFactory().build_loaders(cfg)
    train_images, train_labels = next(iter(loaders.train))
    val_images, val_labels = next(iter(loaders.val))

    assert tuple(train_images.shape) == (3, 6, 2, 35, 35)
    assert tuple(train_labels.shape) == (3,)
    assert tuple(val_images.shape[1:]) == (6, 2, 35, 35)
    assert tuple(val_labels.shape) == (3,)
    assert loaders.meta.name == "pokerdvs"
    assert loaders.meta.n_classes == 4
    assert loaders.meta.sensor_size == (35, 35, 2)


@pytest.mark.unit
def test_dataset_factory_logic_gates_pixels_is_deterministic(tmp_path: Path) -> None:
    cfg = DatasetFactoryConfig(
        name="logic_gates_pixels",
        root=str(tmp_path / "logic"),
        batch_size=8,
        seed=123,
        logic_image_size=8,
        logic_gates=("AND", "OR", "NAND", "NOR"),
        logic_mode="multiclass",
        logic_samples_per_gate=40,
        logic_train_ratio=0.7,
        logic_val_ratio=0.15,
        logic_test_ratio=0.15,
    )

    loaders_a = DatasetFactory().build_loaders(cfg)
    loaders_b = DatasetFactory().build_loaders(cfg)
    images_a, labels_a = next(iter(loaders_a.train))
    images_b, labels_b = next(iter(loaders_b.train))

    assert tuple(images_a.shape) == (8, 1, 8, 8)
    assert tuple(labels_a.shape) == (8,)
    assert torch.equal(images_a, images_b)
    assert torch.equal(labels_a, labels_b)
    assert int(labels_a.min().item()) >= 0
    assert int(labels_a.max().item()) < int(loaders_a.meta.n_classes)
    assert loaders_a.meta.name == "logic_gates_pixels"
    assert loaders_a.meta.class_names is not None
    assert len(loaders_a.meta.class_names) == 4
