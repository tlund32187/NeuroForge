"""Event dataset adapter for Tonic-style event streams.

This module is data/transform only:
- No UI concerns.
- No artifact export concerns.

It adapts event samples into tensors shaped ``[T, C, H, W]`` so they can
flow into existing vision backbones.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

from neuroforge.core.torch_utils import require_torch

torch = require_torch()

__all__ = [
    "EventDatasetAdapter",
    "EventSliceConfig",
    "EventTensorConfig",
    "TonicDatasetConfig",
    "build_tonic_dataset",
]


@dataclass(frozen=True, slots=True)
class EventSliceConfig:
    """Deterministic event slicing settings.

    ``mode="time"`` keeps events in ``[t0, t0 + window_us)``.
    ``mode="count"`` keeps the first ``event_count`` events.
    """

    mode: str = "time"
    window_us: int = 100_000
    event_count: int = 20_000

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        if mode not in {"time", "count"}:
            msg = "EventSliceConfig.mode must be one of ['time', 'count']"
            raise ValueError(msg)
        object.__setattr__(self, "mode", mode)
        if mode == "time" and self.window_us <= 0:
            msg = "EventSliceConfig.window_us must be > 0 when mode='time'"
            raise ValueError(msg)
        if mode == "count" and self.event_count <= 0:
            msg = "EventSliceConfig.event_count must be > 0 when mode='count'"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EventTensorConfig:
    """Output tensor settings for event conversion."""

    time_steps: int
    height: int
    width: int
    polarity_channels: int = 2
    mode: str = "frames"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.time_steps <= 0:
            msg = "EventTensorConfig.time_steps must be > 0"
            raise ValueError(msg)
        if self.height <= 0 or self.width <= 0:
            msg = "EventTensorConfig.height/width must be > 0"
            raise ValueError(msg)
        if self.polarity_channels not in {1, 2}:
            msg = "EventTensorConfig.polarity_channels must be 1 or 2"
            raise ValueError(msg)
        mode = self.mode.strip().lower()
        if mode not in {"frames", "voxel_grid"}:
            msg = "EventTensorConfig.mode must be one of ['frames', 'voxel_grid']"
            raise ValueError(msg)
        object.__setattr__(self, "mode", mode)
        if self.dtype not in {"float16", "float32", "float64"}:
            msg = "EventTensorConfig.dtype must be one of ['float16', 'float32', 'float64']"
            raise ValueError(msg)

    def torch_dtype(self) -> Any:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        return dtype_map[self.dtype]


@dataclass(frozen=True, slots=True)
class TonicDatasetConfig:
    """Config for creating a Tonic dataset instance."""

    name: str = "nmnist"
    root: str = ".cache/tonic"
    train: bool = True
    download: bool = True

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            msg = "TonicDatasetConfig.name must be a non-empty string"
            raise ValueError(msg)
        object.__setattr__(self, "name", name)
        root = self.root.strip()
        if not root:
            msg = "TonicDatasetConfig.root must be a non-empty string"
            raise ValueError(msg)
        object.__setattr__(self, "root", root)


def _require_tonic() -> Any:
    try:
        return importlib.import_module("tonic")
    except ImportError as exc:
        msg = "Tonic is required for event datasets. Install with: pip install neuroforge[events]"
        raise ImportError(msg) from exc


def _tonic_dataset_class(tonic_module: Any, name: str) -> Any:
    datasets = tonic_module.datasets
    aliases = {
        "nmnist": "NMNIST",
        "pokerdvs": "POKERDVS",
        "poker_dvs": "POKERDVS",
        "poker-dvs": "POKERDVS",
        "cifar10dvs": "CIFAR10DVS",
        "cifar10_dvs": "CIFAR10DVS",
        "dvsgesture": "DVSGesture",
        "dvs_gesture": "DVSGesture",
    }
    key = name.strip().lower()
    cls_name = aliases.get(key, name)
    dataset_cls = getattr(datasets, cls_name, None)
    if dataset_cls is None and key in {"pokerdvs", "poker_dvs", "poker-dvs"}:
        dataset_cls = getattr(datasets, "PokerDVS", None)
    if dataset_cls is not None:
        return dataset_cls
    msg = f"Unsupported tonic dataset: {name!r}"
    raise ValueError(msg)


def build_tonic_dataset(cfg: TonicDatasetConfig) -> Any:
    """Build a Tonic dataset with signature-aware kwargs.

    Supports dataset classes that accept either ``save_to=...`` or ``root=...``.
    """
    tonic = _require_tonic()
    dataset_cls = _tonic_dataset_class(tonic, cfg.name)
    param_names: set[str]
    try:
        param_names = set(inspect.signature(dataset_cls).parameters)
    except (TypeError, ValueError):
        param_names = set()

    kwargs: dict[str, Any] = {}
    if "save_to" in param_names:
        kwargs["save_to"] = cfg.root
    elif "root" in param_names:
        kwargs["root"] = cfg.root
    if "train" in param_names:
        kwargs["train"] = bool(cfg.train)
    if "download" in param_names:
        kwargs["download"] = bool(cfg.download)
    return dataset_cls(**kwargs)


def _as_1d_tensor(values: Any, *, dtype: Any) -> Any:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)
    return tensor.to(dtype=dtype)


def _extract_event_fields(events: Any) -> tuple[Any, Any, Any, Any]:
    # Mapping case: {"t": ..., "x": ..., "y": ..., "polarity"/"p": ...}
    if isinstance(events, dict):
        polarity_key = "polarity" if "polarity" in events else "p"
        required = ("t", "x", "y", polarity_key)
        if any(key not in events for key in required):
            msg = "Event mapping must contain keys 't', 'x', 'y', and 'polarity' (or 'p')"
            raise ValueError(msg)
        return (
            _as_1d_tensor(events["t"], dtype=torch.int64),
            _as_1d_tensor(events["x"], dtype=torch.int64),
            _as_1d_tensor(events["y"], dtype=torch.int64),
            _as_1d_tensor(events[polarity_key], dtype=torch.int64),
        )

    # Tensor case: [N, 4] ordered as [t, x, y, polarity].
    if isinstance(events, torch.Tensor):
        if events.ndim != 2 or int(events.shape[1]) != 4:
            msg = "Event tensor must have shape [N, 4] with columns [t, x, y, polarity]"
            raise ValueError(msg)
        return (
            events[:, 0].to(dtype=torch.int64),
            events[:, 1].to(dtype=torch.int64),
            events[:, 2].to(dtype=torch.int64),
            events[:, 3].to(dtype=torch.int64),
        )

    # Numpy structured arrays are common in Tonic datasets.
    dtype_obj = getattr(events, "dtype", None)
    names = getattr(dtype_obj, "names", None)
    if isinstance(names, tuple):
        polarity_key = "polarity" if "polarity" in names else "p"
        required = ("t", "x", "y", polarity_key)
        if any(key not in names for key in required):
            msg = "Structured events must have fields 't', 'x', 'y', and 'polarity' (or 'p')"
            raise ValueError(msg)
        return (
            _as_1d_tensor(events["t"], dtype=torch.int64),
            _as_1d_tensor(events["x"], dtype=torch.int64),
            _as_1d_tensor(events["y"], dtype=torch.int64),
            _as_1d_tensor(events[polarity_key], dtype=torch.int64),
        )

    # Generic sequence case: (t, x, y, polarity).
    if isinstance(events, tuple | list) and len(events) >= 4:
        return (
            _as_1d_tensor(events[0], dtype=torch.int64),
            _as_1d_tensor(events[1], dtype=torch.int64),
            _as_1d_tensor(events[2], dtype=torch.int64),
            _as_1d_tensor(events[3], dtype=torch.int64),
        )

    msg = (
        "Unsupported event payload type. "
        "Expected mapping, [N,4] tensor, structured array, or (t,x,y,p) tuple."
    )
    raise TypeError(msg)


def _coerce_label(label: Any) -> Any:
    if isinstance(label, torch.Tensor) and label.ndim == 0:
        return int(label.item())
    if isinstance(label, bool | int):
        return int(label)
    return label


class EventDatasetAdapter:
    """Wrap an event dataset and convert samples to ``[T, C, H, W]`` tensors."""

    def __init__(
        self,
        dataset: Any,
        *,
        tensor_config: EventTensorConfig,
        slice_config: EventSliceConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self._tensor_config = tensor_config
        self._slice_config = slice_config or EventSliceConfig()

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (
            int(self._tensor_config.time_steps),
            int(self._tensor_config.polarity_channels),
            int(self._tensor_config.height),
            int(self._tensor_config.width),
        )

    def __len__(self) -> int:
        return int(len(self._dataset))

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample = self._dataset[index]
        if not isinstance(sample, tuple | list) or len(sample) < 2:
            msg = "Dataset samples must be (events, label)"
            raise ValueError(msg)
        events_raw = sample[0]
        label = _coerce_label(sample[1])

        t, x, y, p = _extract_event_fields(events_raw)
        t, x, y, p = self._deterministic_slice(t=t, x=x, y=y, p=p)
        out = self._to_tensor(t=t, x=x, y=y, p=p)
        return out, label

    def _deterministic_slice(self, *, t: Any, x: Any, y: Any, p: Any) -> tuple[Any, Any, Any, Any]:
        if t.numel() <= 1:
            return t, x, y, p

        order = torch.argsort(t)
        t = t[order]
        x = x[order]
        y = y[order]
        p = p[order]

        if self._slice_config.mode == "count":
            keep = min(int(self._slice_config.event_count), int(t.numel()))
            return t[:keep], x[:keep], y[:keep], p[:keep]

        # mode == "time"
        t0 = int(t[0].item())
        horizon = t0 + int(self._slice_config.window_us)
        mask = t < horizon
        if not bool(mask.any()):
            return t[:1], x[:1], y[:1], p[:1]
        return t[mask], x[mask], y[mask], p[mask]

    def _to_tensor(self, *, t: Any, x: Any, y: Any, p: Any) -> Any:
        out = torch.zeros(
            self.output_shape,
            dtype=self._tensor_config.torch_dtype(),
        )
        if t.numel() == 0:
            return out

        width = int(self._tensor_config.width)
        height = int(self._tensor_config.height)
        c = int(self._tensor_config.polarity_channels)
        time_steps = int(self._tensor_config.time_steps)

        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not bool(in_bounds.any()):
            return out
        t = t[in_bounds]
        x = x[in_bounds]
        y = y[in_bounds]
        p = p[in_bounds]

        pol = (
            (p > 0).to(dtype=torch.int64)
            if c == 2
            else torch.zeros_like(p, dtype=torch.int64)
        )

        t_rel = (t - t.min()).to(dtype=torch.float32)
        duration_us = self._duration_us(t=t)

        flat = out.view(-1)
        if self._tensor_config.mode == "frames":
            frame_idx = torch.floor(t_rel * float(time_steps) / float(duration_us)).to(torch.int64)
            frame_idx = frame_idx.clamp(min=0, max=time_steps - 1)
            lin = (((frame_idx * c + pol) * height + y) * width + x).to(torch.int64)
            ones = torch.ones(int(lin.numel()), dtype=out.dtype)
            flat.scatter_add_(0, lin, ones)
            return out

        # mode == "voxel_grid"
        if time_steps == 1 or duration_us <= 1:
            left = torch.zeros_like(t, dtype=torch.int64)
            right = left
            w_left = torch.ones(int(t.numel()), dtype=out.dtype)
            w_right = torch.zeros(int(t.numel()), dtype=out.dtype)
        else:
            t_norm = t_rel * float(time_steps - 1) / float(duration_us - 1)
            left = torch.floor(t_norm).to(torch.int64).clamp(min=0, max=time_steps - 1)
            right = (left + 1).clamp(max=time_steps - 1)
            w_right = (t_norm - left.to(torch.float32)).to(dtype=out.dtype)
            w_left = (1.0 - w_right).to(dtype=out.dtype)

        lin_left = (((left * c + pol) * height + y) * width + x).to(torch.int64)
        lin_right = (((right * c + pol) * height + y) * width + x).to(torch.int64)
        flat.scatter_add_(0, lin_left, w_left)
        flat.scatter_add_(0, lin_right, w_right)
        return out

    def _duration_us(self, *, t: Any) -> int:
        if self._slice_config.mode == "time":
            return max(1, int(self._slice_config.window_us))
        span = int((t.max() - t.min()).item()) + 1
        return max(1, span)
