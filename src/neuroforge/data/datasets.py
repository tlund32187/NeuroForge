"""Dataset factory for vision task datasets (torchvision + Tonic)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuroforge.core.torch_utils import require_torch
from neuroforge.data.event_dataset_adapter import (
    EventDatasetAdapter,
    EventSliceConfig,
    EventTensorConfig,
    TonicDatasetConfig,
    build_tonic_dataset,
)
from neuroforge.data.logic_gates_pixels import (
    LogicGatesPixelsConfig,
    build_logic_gates_pixels_splits,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

torch = require_torch()

__all__ = [
    "DatasetFactory",
    "DatasetFactoryConfig",
    "DatasetLoaders",
    "DatasetMeta",
    "DatasetTransformConfig",
]

_IMAGE_DATASETS: frozenset[str] = frozenset({"mnist", "fashion_mnist"})
_EVENT_DATASETS: frozenset[str] = frozenset({"nmnist", "pokerdvs"})
_LOGIC_DATASETS: frozenset[str] = frozenset({"logic_gates_pixels"})
_SUPPORTED_DATASETS: frozenset[str] = _IMAGE_DATASETS | _EVENT_DATASETS | _LOGIC_DATASETS


@dataclass(frozen=True, slots=True)
class DatasetTransformConfig:
    """Transform options for torchvision dataset pipelines."""

    normalize: bool = True
    random_crop: bool = False
    crop_padding: int = 2
    random_horizontal_flip: bool = False

    def __post_init__(self) -> None:
        if self.crop_padding < 0:
            msg = "crop_padding must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DatasetFactoryConfig:
    """Factory config for train/val/test dataset and dataloader creation."""

    name: str = "mnist"
    root: str = ".cache/torchvision"
    batch_size: int = 64
    val_fraction: float = 0.1
    download: bool = True
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    allow_fashion_mnist: bool = False
    event_time_steps: int = 6
    event_tensor_mode: str = "frames"
    event_slice_mode: str = "time"
    event_window_us: int = 100_000
    event_count: int = 20_000
    event_polarity_channels: int = 2
    logic_image_size: int = 8
    logic_gates: tuple[str, ...] = ("AND", "OR", "NAND", "NOR")
    logic_mode: str = "multiclass"
    logic_single_gate: str = "AND"
    logic_samples_per_gate: int = 128
    logic_total_samples: int = 0
    logic_train_ratio: float = 0.7
    logic_val_ratio: float = 0.15
    logic_test_ratio: float = 0.15
    transforms: DatasetTransformConfig = field(default_factory=DatasetTransformConfig)

    def __post_init__(self) -> None:
        dataset_name = self.name.strip().lower()
        if dataset_name not in _SUPPORTED_DATASETS:
            msg = (
                "DatasetFactoryConfig.name must be one of "
                f"{sorted(_SUPPORTED_DATASETS)}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "name", dataset_name)
        if self.batch_size <= 0:
            msg = "batch_size must be > 0"
            raise ValueError(msg)
        if not (0.0 <= self.val_fraction < 1.0):
            msg = "val_fraction must be in [0.0, 1.0)"
            raise ValueError(msg)
        if self.num_workers < 0:
            msg = "num_workers must be >= 0"
            raise ValueError(msg)
        if dataset_name == "fashion_mnist" and not self.allow_fashion_mnist:
            msg = "fashion_mnist requested but allow_fashion_mnist=False"
            raise ValueError(msg)
        event_tensor_mode = self.event_tensor_mode.strip().lower()
        if event_tensor_mode not in {"frames", "voxel_grid"}:
            msg = "event_tensor_mode must be one of ['frames', 'voxel_grid']"
            raise ValueError(msg)
        object.__setattr__(self, "event_tensor_mode", event_tensor_mode)
        event_slice_mode = self.event_slice_mode.strip().lower()
        if event_slice_mode not in {"time", "count"}:
            msg = "event_slice_mode must be one of ['time', 'count']"
            raise ValueError(msg)
        object.__setattr__(self, "event_slice_mode", event_slice_mode)
        if self.event_time_steps <= 0:
            msg = "event_time_steps must be > 0"
            raise ValueError(msg)
        if self.event_window_us <= 0:
            msg = "event_window_us must be > 0"
            raise ValueError(msg)
        if self.event_count <= 0:
            msg = "event_count must be > 0"
            raise ValueError(msg)
        if self.event_polarity_channels not in {1, 2}:
            msg = "event_polarity_channels must be 1 or 2"
            raise ValueError(msg)
        logic_mode = str(self.logic_mode).strip().lower()
        if logic_mode not in {"multiclass", "single_gate"}:
            msg = "logic_mode must be one of ['multiclass', 'single_gate']"
            raise ValueError(msg)
        object.__setattr__(self, "logic_mode", logic_mode)
        if self.logic_image_size < 6:
            msg = "logic_image_size must be >= 6"
            raise ValueError(msg)
        if self.logic_samples_per_gate <= 0:
            msg = "logic_samples_per_gate must be > 0"
            raise ValueError(msg)
        if self.logic_total_samples < 0:
            msg = "logic_total_samples must be >= 0"
            raise ValueError(msg)
        if any(
            value < 0.0
            for value in (
                self.logic_train_ratio,
                self.logic_val_ratio,
                self.logic_test_ratio,
            )
        ):
            msg = "logic_train_ratio/logic_val_ratio/logic_test_ratio must be >= 0"
            raise ValueError(msg)
        if self.logic_train_ratio <= 0.0:
            msg = "logic_train_ratio must be > 0"
            raise ValueError(msg)
        if (self.logic_train_ratio + self.logic_val_ratio + self.logic_test_ratio) <= 0.0:
            msg = "logic split ratios must sum to > 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DatasetMeta:
    """Resolved dataset metadata for artifacts and reproducibility."""

    name: str
    root: str
    split_sizes: dict[str, int]
    transforms_summary: dict[str, list[str]]
    n_classes: int
    channels: int
    height: int
    width: int
    sensor_size: tuple[int, ...] | None = None
    class_names: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": self.root,
            "split_sizes": dict(self.split_sizes),
            "transforms_summary": {
                key: [*value]
                for key, value in self.transforms_summary.items()
            },
            "n_classes": self.n_classes,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "sensor_size": (
                [int(v) for v in self.sensor_size]
                if self.sensor_size is not None
                else None
            ),
            "class_names": (
                [str(name) for name in self.class_names]
                if self.class_names is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class DatasetLoaders:
    """Bundle of train/val/test dataloaders and resolved metadata."""

    train: Any
    val: Any
    test: Any
    meta: DatasetMeta


def _require_torchvision() -> tuple[Any, Any]:
    try:
        torchvision = importlib.import_module("torchvision")
    except ImportError as exc:
        msg = (
            "torchvision is required for MNIST datasets. "
            "Install with: pip install neuroforge[torch,vision]"
        )
        raise ImportError(msg) from exc
    return torchvision.datasets, torchvision.transforms


def _normalization_stats(name: str) -> tuple[tuple[float], tuple[float]]:
    if name == "fashion_mnist":
        return (0.2860,), (0.3530,)
    return (0.1307,), (0.3081,)


def _transform_summary(parts: Sequence[Any]) -> list[str]:
    summary: list[str] = []
    for part in parts:
        text = str(part)
        if " object at 0x" in text:
            text = part.__class__.__name__
        summary.append(text)
    return summary


def _build_transform_pair(
    *,
    dataset_name: str,
    cfg: DatasetTransformConfig,
) -> tuple[Any, Any, dict[str, list[str]]]:
    _, transforms = _require_torchvision()

    train_parts: list[Any] = []
    eval_parts: list[Any] = []

    if cfg.random_crop:
        train_parts.append(transforms.RandomCrop(28, padding=int(cfg.crop_padding)))
    if cfg.random_horizontal_flip:
        train_parts.append(transforms.RandomHorizontalFlip())

    train_parts.append(transforms.ToTensor())
    eval_parts.append(transforms.ToTensor())

    if cfg.normalize:
        mean, std = _normalization_stats(dataset_name)
        norm = transforms.Normalize(mean=mean, std=std)
        train_parts.append(norm)
        eval_parts.append(norm)

    train_tf = transforms.Compose(train_parts)
    eval_tf = transforms.Compose(eval_parts)
    summary = {
        "train": _transform_summary(train_parts),
        "eval": _transform_summary(eval_parts),
    }
    return train_tf, eval_tf, summary


def _dataset_class(name: str, *, allow_fashion_mnist: bool) -> Any:
    datasets, _ = _require_torchvision()
    if name == "mnist":
        return datasets.MNIST
    if name == "fashion_mnist":
        if not allow_fashion_mnist:
            msg = "fashion_mnist requested but allow_fashion_mnist=False"
            raise ValueError(msg)
        return datasets.FashionMNIST
    msg = f"Unsupported dataset: {name!r}"
    raise ValueError(msg)


def _event_defaults(name: str) -> tuple[int, tuple[int, int, int]]:
    if name == "nmnist":
        return 10, (34, 34, 2)
    if name == "pokerdvs":
        return 4, (35, 35, 2)
    msg = f"Unsupported event dataset: {name!r}"
    raise ValueError(msg)


def _infer_event_n_classes(dataset: Any, *, dataset_name: str) -> int:
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, (tuple, list, set)):
        count = int(len(classes))
        if count > 1:
            return count

    targets = getattr(dataset, "targets", None)
    if isinstance(targets, torch.Tensor) and targets.numel() > 0:
        return int(targets.max().item()) + 1
    if isinstance(targets, tuple | list) and len(targets) > 0:
        try:
            return int(max(int(v) for v in targets)) + 1
        except (TypeError, ValueError):
            pass

    fallback, _ = _event_defaults(dataset_name)
    return fallback


def _resolve_sensor_size(dataset: Any, *, dataset_name: str) -> tuple[int, int, int]:
    _default_classes, default_sensor = _event_defaults(dataset_name)
    raw = getattr(dataset, "sensor_size", None)
    if not isinstance(raw, tuple | list):
        return default_sensor
    values: list[int] = []
    for value in raw:
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            return default_sensor
    if len(values) >= 3:
        return (values[0], values[1], max(1, values[2]))
    if len(values) == 2:
        return (values[0], values[1], default_sensor[2])
    return default_sensor


def _event_transforms_summary(cfg: DatasetFactoryConfig) -> dict[str, list[str]]:
    summary = [
        (
            "EventSlice("
            f"mode={cfg.event_slice_mode},"
            f"window_us={cfg.event_window_us},"
            f"event_count={cfg.event_count})"
        ),
        (
            "EventTensor("
            f"mode={cfg.event_tensor_mode},"
            f"time_steps={cfg.event_time_steps},"
            f"polarity_channels={cfg.event_polarity_channels})"
        ),
    ]
    return {"train": [*summary], "eval": [*summary]}


def _split_train_val_indices(
    *,
    n_train_total: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    n_val = int(round(n_train_total * val_fraction))
    n_val = max(0, min(n_train_total, n_val))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(n_train_total, generator=gen).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _build_loader_bundle(
    *,
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    cfg: DatasetFactoryConfig,
) -> tuple[Any, Any, Any]:
    loader_gen = torch.Generator(device="cpu")
    loader_gen.manual_seed(int(cfg.seed))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        generator=loader_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    return train_loader, val_loader, test_loader


class DatasetFactory:
    """Build train/val/test datasets and dataloaders for Phase 8 tasks."""

    def build_loaders(self, cfg: DatasetFactoryConfig) -> DatasetLoaders:
        if cfg.name in _IMAGE_DATASETS:
            return self._build_torchvision_loaders(cfg)
        if cfg.name in _LOGIC_DATASETS:
            return self._build_logic_gate_loaders(cfg)
        return self._build_tonic_loaders(cfg)

    def _build_torchvision_loaders(self, cfg: DatasetFactoryConfig) -> DatasetLoaders:
        datasets, _ = _require_torchvision()
        dataset_cls = _dataset_class(cfg.name, allow_fashion_mnist=cfg.allow_fashion_mnist)

        root = Path(cfg.root)
        root.mkdir(parents=True, exist_ok=True)

        train_tf, eval_tf, tf_summary = _build_transform_pair(
            dataset_name=cfg.name,
            cfg=cfg.transforms,
        )

        train_aug = dataset_cls(
            root=str(root),
            train=True,
            download=bool(cfg.download),
            transform=train_tf,
        )
        train_eval = dataset_cls(
            root=str(root),
            train=True,
            download=bool(cfg.download),
            transform=eval_tf,
        )
        test_ds = dataset_cls(
            root=str(root),
            train=False,
            download=bool(cfg.download),
            transform=eval_tf,
        )

        train_idx, val_idx = _split_train_val_indices(
            n_train_total=len(train_aug),
            val_fraction=float(cfg.val_fraction),
            seed=int(cfg.seed),
        )

        train_ds = datasets.Subset(train_aug, train_idx)
        val_ds = datasets.Subset(train_eval, val_idx)
        train_loader, val_loader, test_loader = _build_loader_bundle(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            cfg=cfg,
        )

        meta = DatasetMeta(
            name=cfg.name,
            root=str(root.resolve()),
            split_sizes={
                "train": int(len(train_ds)),
                "val": int(len(val_ds)),
                "test": int(len(test_ds)),
            },
            transforms_summary=tf_summary,
            n_classes=10,
            channels=1,
            height=28,
            width=28,
            sensor_size=(28, 28, 1),
            class_names=tuple(str(idx) for idx in range(10)),
        )
        return DatasetLoaders(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            meta=meta,
        )

    def _build_logic_gate_loaders(self, cfg: DatasetFactoryConfig) -> DatasetLoaders:
        root = Path(cfg.root)
        root.mkdir(parents=True, exist_ok=True)

        logic_cfg = LogicGatesPixelsConfig(
            image_size=int(cfg.logic_image_size),
            gates=tuple(str(gate) for gate in cfg.logic_gates),
            mode=str(cfg.logic_mode),
            single_gate=str(cfg.logic_single_gate),
            samples_per_gate=int(cfg.logic_samples_per_gate),
            total_samples=int(cfg.logic_total_samples),
            train_ratio=float(cfg.logic_train_ratio),
            val_ratio=float(cfg.logic_val_ratio),
            test_ratio=float(cfg.logic_test_ratio),
            seed=int(cfg.seed),
        )
        splits = build_logic_gates_pixels_splits(logic_cfg)
        train_loader, val_loader, test_loader = _build_loader_bundle(
            train_ds=splits.train,
            val_ds=splits.val,
            test_ds=splits.test,
            cfg=cfg,
        )

        tf_summary = {
            "train": [
                (
                    "LogicGatesPixels("
                    f"mode={logic_cfg.mode},image_size={logic_cfg.image_size},"
                    f"gates={list(logic_cfg.gates)},single_gate={logic_cfg.single_gate},"
                    f"samples_per_gate={logic_cfg.samples_per_gate},"
                    f"total_samples={logic_cfg.total_samples})"
                ),
            ],
            "eval": [
                (
                    "LogicGatesPixels("
                    f"mode={logic_cfg.mode},image_size={logic_cfg.image_size},"
                    f"gates={list(logic_cfg.gates)},single_gate={logic_cfg.single_gate},"
                    f"samples_per_gate={logic_cfg.samples_per_gate},"
                    f"total_samples={logic_cfg.total_samples})"
                ),
            ],
        }
        meta = DatasetMeta(
            name=cfg.name,
            root=str(root.resolve()),
            split_sizes={
                "train": int(len(splits.train)),
                "val": int(len(splits.val)),
                "test": int(len(splits.test)),
            },
            transforms_summary=tf_summary,
            n_classes=int(splits.n_classes),
            channels=1,
            height=int(logic_cfg.image_size),
            width=int(logic_cfg.image_size),
            sensor_size=(int(logic_cfg.image_size), int(logic_cfg.image_size), 1),
            class_names=tuple(str(name) for name in splits.class_names),
        )
        return DatasetLoaders(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            meta=meta,
        )

    def _build_tonic_loaders(self, cfg: DatasetFactoryConfig) -> DatasetLoaders:
        root = Path(cfg.root)
        root.mkdir(parents=True, exist_ok=True)

        train_raw = build_tonic_dataset(
            TonicDatasetConfig(
                name=cfg.name,
                root=str(root),
                train=True,
                download=bool(cfg.download),
            )
        )
        test_raw = build_tonic_dataset(
            TonicDatasetConfig(
                name=cfg.name,
                root=str(root),
                train=False,
                download=bool(cfg.download),
            )
        )

        sensor_size = _resolve_sensor_size(train_raw, dataset_name=cfg.name)
        sensor_w, sensor_h, sensor_c = sensor_size
        polarity_channels = max(1, int(cfg.event_polarity_channels))
        tensor_cfg = EventTensorConfig(
            time_steps=int(cfg.event_time_steps),
            height=int(sensor_h),
            width=int(sensor_w),
            polarity_channels=polarity_channels if polarity_channels > 0 else int(sensor_c),
            mode=cfg.event_tensor_mode,
        )
        slice_cfg = EventSliceConfig(
            mode=cfg.event_slice_mode,
            window_us=int(cfg.event_window_us),
            event_count=int(cfg.event_count),
        )

        train_aug = EventDatasetAdapter(
            train_raw,
            tensor_config=tensor_cfg,
            slice_config=slice_cfg,
        )
        train_eval = EventDatasetAdapter(
            train_raw,
            tensor_config=tensor_cfg,
            slice_config=slice_cfg,
        )
        test_ds = EventDatasetAdapter(
            test_raw,
            tensor_config=tensor_cfg,
            slice_config=slice_cfg,
        )

        train_idx, val_idx = _split_train_val_indices(
            n_train_total=len(train_aug),
            val_fraction=float(cfg.val_fraction),
            seed=int(cfg.seed),
        )
        train_ds = torch.utils.data.Subset(train_aug, train_idx)
        val_ds = torch.utils.data.Subset(train_eval, val_idx)
        train_loader, val_loader, test_loader = _build_loader_bundle(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            cfg=cfg,
        )

        meta = DatasetMeta(
            name=cfg.name,
            root=str(root.resolve()),
            split_sizes={
                "train": int(len(train_ds)),
                "val": int(len(val_ds)),
                "test": int(len(test_ds)),
            },
            transforms_summary=_event_transforms_summary(cfg),
            n_classes=_infer_event_n_classes(train_raw, dataset_name=cfg.name),
            channels=int(tensor_cfg.polarity_channels),
            height=int(tensor_cfg.height),
            width=int(tensor_cfg.width),
            sensor_size=(int(sensor_w), int(sensor_h), int(sensor_c)),
            class_names=tuple(str(idx) for idx in range(_infer_event_n_classes(train_raw, dataset_name=cfg.name))),
        )
        return DatasetLoaders(
            train=train_loader,
            val=val_loader,
            test=test_loader,
            meta=meta,
        )
