"""Vision monitors and exporters for Phase 8 artifact outputs."""

from __future__ import annotations

import csv
import json
import math
import re
from typing import TYPE_CHECKING, Any, cast

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.core.torch_utils import require_torch

torch = require_torch()

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ConfusionMatrixExporter",
    "ConfusionMatrixMonitor",
    "VisionLayerStatsExporter",
    "VisionLayerStatsMonitor",
    "VisionSampleGridExporter",
    "VisionSampleGridMonitor",
]

_LAYER_PREFIX = "vision.layer."
_FIELD_SPIKE_RATE = "spike_rate"
_FIELD_MEAN_ACT = "mean_activation"
_FIELD_MAX_ACT = "max_activation"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        msg = "numpy is required for confusion_matrix.npy export"
        raise ImportError(msg) from exc
    return np


def _require_pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        msg = "Pillow is required for vision sample grid export"
        raise ImportError(msg) from exc
    return Image, ImageDraw


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        _val = cast("Any", value)
        out: list[int] = []
        for item in _val:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    if hasattr(value, "detach"):
        tensor = value.detach().reshape(-1).cpu()
        out = []
        for item in tensor.tolist():
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


class VisionLayerStatsMonitor:
    """Extract per-layer scalar stats at fixed intervals from SCALAR events."""

    def __init__(self, *, interval_steps: int = 1, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.interval_steps = max(1, int(interval_steps))
        self._rows: list[dict[str, Any]] = []

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        if event.topic == EventTopic.TRAINING_START:
            self.reset()
            return
        if event.topic != EventTopic.SCALAR:
            return
        if int(event.step) % self.interval_steps != 0:
            return
        if str(event.data.get("gate", "")).upper() != "VISION":
            return

        layer_rows: dict[str, dict[str, Any]] = {}
        for key, raw_value in event.data.items():
            if not isinstance(key, str) or not key.startswith(_LAYER_PREFIX):  # pyright: ignore[reportUnnecessaryIsInstance]
                continue
            tail = key[len(_LAYER_PREFIX):]
            if "." not in tail:
                continue
            layer, _, metric = tail.rpartition(".")
            if metric not in {_FIELD_SPIKE_RATE, _FIELD_MEAN_ACT, _FIELD_MAX_ACT}:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            row = layer_rows.setdefault(
                layer,
                {
                    "trial": int(event.data.get("trial", event.step)),
                    "layer": layer,
                    _FIELD_SPIKE_RATE: 0.0,
                    _FIELD_MEAN_ACT: 0.0,
                    _FIELD_MAX_ACT: 0.0,
                },
            )
            row[metric] = value

        self._rows.extend(layer_rows.values())

    def reset(self) -> None:
        self._rows.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_steps": self.interval_steps,
            "rows": [dict(row) for row in self._rows],
            "row_count": len(self._rows),
        }

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


class VisionLayerStatsExporter:
    """Write `vision_layer_stats.csv` from VisionLayerStatsMonitor snapshot."""

    def __init__(
        self,
        run_dir: Path,
        monitor: VisionLayerStatsMonitor,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._run_dir = run_dir
        self._monitor = monitor
        self._written = False

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled or self._written:
            return
        if event.topic not in (EventTopic.TRAINING_END, EventTopic.RUN_END):
            return
        rows = self._monitor.rows()
        metrics_dir = self._run_dir / "vision" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_path = metrics_dir / "vision_layer_stats.csv"
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "trial",
                    "layer",
                    _FIELD_SPIKE_RATE,
                    _FIELD_MEAN_ACT,
                    _FIELD_MAX_ACT,
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        self._written = True

    def reset(self) -> None:
        self._written = False

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "written": self._written}


class ConfusionMatrixMonitor:
    """Accumulate confusion matrix statistics from TRAINING_TRIAL events."""

    def __init__(self, *, n_classes: int | None = None, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._n_classes = int(n_classes) if n_classes is not None else 0
        self._matrix: list[list[int]] = []
        self._total_samples = 0
        self._class_names: list[str] = []

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        if event.topic == EventTopic.TRAINING_START:
            self.reset()
            n_classes = event.data.get("n_classes")
            if n_classes is not None:
                try:
                    self._n_classes = max(0, int(n_classes))
                except (TypeError, ValueError):
                    self._n_classes = 0
                self._ensure_size(max(1, self._n_classes))
            dataset_meta = event.data.get("dataset_meta")
            class_names: list[str] = []
            if isinstance(dataset_meta, dict):
                _meta = cast("Any", dataset_meta)
                raw_class_names = _meta.get("class_names")
                if isinstance(raw_class_names, list):
                    _names = cast("Any", raw_class_names)
                    class_names = [str(name) for name in _names if str(name).strip()]
            if not class_names:
                raw_class_names = event.data.get("class_names")
                if isinstance(raw_class_names, list):
                    _names = cast("Any", raw_class_names)
                    class_names = [str(name) for name in _names if str(name).strip()]
            self._class_names = class_names
            return
        if event.topic != EventTopic.TRAINING_TRIAL:
            return
        predicted = _as_int_list(event.data.get("predicted"))
        expected = _as_int_list(event.data.get("expected"))
        n = min(len(predicted), len(expected))
        for idx in range(n):
            true_label = expected[idx]
            pred_label = predicted[idx]
            if true_label < 0 or pred_label < 0:
                continue
            self._ensure_size(max(true_label, pred_label) + 1)
            self._matrix[true_label][pred_label] += 1
            self._total_samples += 1

    def reset(self) -> None:
        self._matrix = []
        self._total_samples = 0
        self._class_names = []
        if self._n_classes > 0:
            self._ensure_size(self._n_classes)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_classes": len(self._matrix),
            "total_samples": self._total_samples,
            "matrix": [list(row) for row in self._matrix],
            "class_names": [*self._class_names],
            "per_class_accuracy": self.per_class_accuracy(),
        }

    def matrix(self) -> list[list[int]]:
        return [list(row) for row in self._matrix]

    def per_class_accuracy(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for class_idx, row in enumerate(self._matrix):
            row_sum = int(sum(row))
            correct = int(row[class_idx]) if class_idx < len(row) else 0
            out[str(class_idx)] = 0.0 if row_sum <= 0 else float(correct / row_sum)
        return out

    def class_names(self) -> list[str]:
        return [*self._class_names]

    def _ensure_size(self, size: int) -> None:
        target = max(0, int(size))
        current = len(self._matrix)
        if target <= current:
            return
        for row in self._matrix:
            row.extend([0] * (target - len(row)))
        for _ in range(target - current):
            self._matrix.append([0] * target)


class ConfusionMatrixExporter:
    """Write confusion matrix `.npy` and per-class accuracy `.json` artifacts."""

    def __init__(
        self,
        run_dir: Path,
        monitor: ConfusionMatrixMonitor,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._run_dir = run_dir
        self._monitor = monitor
        self._written = False

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled or self._written:
            return
        if event.topic not in (EventTopic.TRAINING_END, EventTopic.RUN_END):
            return
        np = _require_numpy()
        metrics_dir = self._run_dir / "vision" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        matrix = self._monitor.matrix()
        matrix_np = np.asarray(matrix, dtype=np.int64)
        np.save(metrics_dir / "confusion_matrix.npy", matrix_np)

        payload = {
            "n_classes": int(len(matrix)),
            "total_samples": int(sum(sum(row) for row in matrix)),
            "class_names": self._monitor.class_names(),
            "per_class_accuracy": self._monitor.per_class_accuracy(),
        }
        (metrics_dir / "per_class_accuracy.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        self._written = True

    def reset(self) -> None:
        self._written = False

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "written": self._written}


class VisionSampleGridMonitor:
    """Capture a small number of labeled samples for artifact grid export."""

    def __init__(self, *, max_samples: int = 16, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.max_samples = max(1, int(max_samples))
        self._samples: list[tuple[Any, int, int]] = []
        self._event_sample: tuple[Any, int, int] | None = None
        self._dataset_meta: dict[str, Any] | None = None

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        if event.topic == EventTopic.TRAINING_START:
            self.reset()
            dataset_meta = event.data.get("dataset_meta")
            if isinstance(dataset_meta, dict):
                self._dataset_meta = dataset_meta
            return
        if event.topic != EventTopic.TRAINING_TRIAL:
            return
        if len(self._samples) >= self.max_samples and self._event_sample is not None:
            return

        images = event.data.get("images")
        if images is None or not hasattr(images, "detach"):
            return
        image_tensor = images.detach().cpu()
        if image_tensor.ndim not in {4, 5}:
            return

        predicted = _as_int_list(event.data.get("predicted"))
        expected = _as_int_list(event.data.get("expected"))
        if image_tensor.ndim == 5:
            # Event batches are [B, T, C, H, W]. Keep first sample for event artifacts.
            if self._event_sample is None and image_tensor.shape[0] > 0:
                gt0 = int(expected[0]) if expected else -1
                pred0 = int(predicted[0]) if predicted else -1
                self._event_sample = (image_tensor[0].clone(), gt0, pred0)
            # Collapse over time for the generic sample grid preview.
            image_tensor = image_tensor.sum(dim=1)

        batch_size = int(image_tensor.shape[0])
        n = min(batch_size, len(predicted), len(expected))
        remaining = self.max_samples - len(self._samples)
        for idx in range(min(n, remaining)):
            self._samples.append(
                (
                    image_tensor[idx].clone(),
                    int(expected[idx]),
                    int(predicted[idx]),
                )
            )

    def reset(self) -> None:
        self._samples.clear()
        self._event_sample = None
        self._dataset_meta = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_samples": self.max_samples,
            "sample_count": len(self._samples),
            "has_event_sample": self._event_sample is not None,
        }

    def samples(self) -> list[tuple[Any, int, int]]:
        return [(img.clone(), gt, pred) for img, gt, pred in self._samples]

    def event_sample(self) -> tuple[Any, int, int] | None:
        if self._event_sample is None:
            return None
        frames, gt, pred = self._event_sample
        return frames.clone(), gt, pred

    def dataset_meta(self) -> dict[str, Any] | None:
        if not isinstance(self._dataset_meta, dict):
            return None
        return dict(self._dataset_meta)


class VisionSampleGridExporter:
    """Write `input_grid.png` for a small set of labeled input samples."""

    def __init__(
        self,
        run_dir: Path,
        monitor: VisionSampleGridMonitor,
        *,
        enabled: bool = True,
        ncols: int = 4,
    ) -> None:
        self.enabled = bool(enabled)
        self._run_dir = run_dir
        self._monitor = monitor
        self._ncols = max(1, int(ncols))
        self._written = False

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled or self._written:
            return
        if event.topic not in (EventTopic.TRAINING_END, EventTopic.RUN_END):
            return
        samples = self._monitor.samples()
        event_sample = self._monitor.event_sample()
        dataset_meta = self._monitor.dataset_meta()
        samples_dir = self._run_dir / "vision" / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = self._run_dir / "vision" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_path = samples_dir / "input_grid.png"
        self._write_grid(samples, out_path)
        if event_sample is not None:
            frames, gt, pred = event_sample
            self._write_event_artifacts(
                frames=frames,
                expected=gt,
                predicted=pred,
                dataset_meta=dataset_meta,
                samples_dir=samples_dir,
                metrics_dir=metrics_dir,
            )
        self._written = True

    def reset(self) -> None:
        self._written = False

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "written": self._written}

    def _write_grid(self, samples: list[tuple[Any, int, int]], out_path: Path) -> None:
        Image, ImageDraw = _require_pillow()

        if not samples:
            empty = Image.new("RGB", (128, 48), color=(255, 255, 255))
            draw = ImageDraw.Draw(empty)
            draw.text((8, 16), "No samples", fill=(0, 0, 0))
            empty.save(out_path)
            return

        tile_images: list[Any] = []
        labels: list[str] = []
        for image, gt, pred in samples:
            tile_images.append(self._to_image(image, Image))
            labels.append(f"gt:{gt} pred:{pred}")

        width, height = tile_images[0].size
        ncols = min(self._ncols, len(tile_images))
        nrows = int(math.ceil(len(tile_images) / ncols))
        label_h = 14

        canvas = Image.new(
            "RGB",
            (ncols * width, nrows * (height + label_h)),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(canvas)
        for idx, tile in enumerate(tile_images):
            row = idx // ncols
            col = idx % ncols
            x = col * width
            y = row * (height + label_h)
            draw.text((x + 2, y + 1), labels[idx], fill=(0, 0, 0))
            canvas.paste(tile, (x, y + label_h))
        canvas.save(out_path)

    def _write_event_artifacts(
        self,
        *,
        frames: Any,
        expected: int,
        predicted: int,
        dataset_meta: dict[str, Any] | None,
        samples_dir: Path,
        metrics_dir: Path,
    ) -> None:
        Image, ImageDraw = _require_pillow()

        if frames.ndim != 4:
            return
        t_steps = int(frames.shape[0])
        if t_steps <= 0:
            return

        bins_grid_path = samples_dir / "event_bins_grid.png"
        sum_image_path = samples_dir / "event_sum.png"
        stats_path = metrics_dir / "event_sample_stats.json"

        bin_tiles: list[Any] = []
        bin_labels: list[str] = []
        for t_idx in range(t_steps):
            bin_tiles.append(self._to_image(frames[t_idx], Image))
            bin_labels.append(f"t:{t_idx}")

        self._write_labeled_tiles(
            image_cls=Image,
            draw_cls=ImageDraw,
            tiles=bin_tiles,
            labels=bin_labels,
            out_path=bins_grid_path,
            ncols=min(self._ncols, max(1, t_steps)),
        )
        summed = frames.sum(dim=0)
        self._to_image(summed, Image).save(sum_image_path)

        total_events = float(frames.sum().item())
        if int(frames.shape[1]) >= 2:
            negative_events = float(frames[:, 0].sum().item())
            positive_events = float(frames[:, 1].sum().item())
        else:
            positive_events = float(total_events)
            negative_events = 0.0
        polarity_total = positive_events + negative_events
        polarity_balance = (
            float((positive_events - negative_events) / polarity_total)
            if polarity_total > 1e-12
            else 0.0
        )
        positive_fraction = (
            float(positive_events / polarity_total)
            if polarity_total > 1e-12
            else 0.0
        )
        duration_us, slice_mode = self._extract_event_duration(dataset_meta)

        payload: dict[str, Any] = {
            "total_events": total_events,
            "positive_events": positive_events,
            "negative_events": negative_events,
            "positive_fraction": positive_fraction,
            "polarity_balance": polarity_balance,
            "duration_bins": t_steps,
            "duration_us": duration_us,
            "slice_mode": slice_mode,
            "expected_label": int(expected),
            "predicted_label": int(predicted),
            "image_files": {
                "bins_grid": "event_bins_grid.png",
                "sum": "event_sum.png",
            },
        }
        stats_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _extract_event_duration(
        dataset_meta: dict[str, Any] | None,
    ) -> tuple[int | None, str | None]:
        if not isinstance(dataset_meta, dict):
            return None, None
        transforms_summary = dataset_meta.get("transforms_summary")
        if not isinstance(transforms_summary, dict):
            return None, None
        _tsdict = cast("Any", transforms_summary)
        train_rows = _tsdict.get("train")
        if not isinstance(train_rows, list):
            return None, None
        _rows = cast("Any", train_rows)
        duration_us: int | None = None
        slice_mode: str | None = None
        for row in _rows:
            text = str(row)
            if "EventSlice(" not in text:
                continue
            mode_match = re.search(r"mode=([a-zA-Z_]+)", text)
            if mode_match is not None:
                slice_mode = str(mode_match.group(1))
            window_match = re.search(r"window_us=(\d+)", text)
            if window_match is not None:
                try:
                    duration_us = int(window_match.group(1))
                except ValueError:
                    duration_us = None
            break
        return duration_us, slice_mode

    @staticmethod
    def _write_labeled_tiles(
        *,
        image_cls: Any,
        draw_cls: Any,
        tiles: list[Any],
        labels: list[str],
        out_path: Path,
        ncols: int,
    ) -> None:
        if not tiles:
            empty = image_cls.new("RGB", (128, 48), color=(255, 255, 255))
            draw = draw_cls.Draw(empty)
            draw.text((8, 16), "No event bins", fill=(0, 0, 0))
            empty.save(out_path)
            return

        width, height = tiles[0].size
        ncols_safe = max(1, int(ncols))
        nrows = int(math.ceil(len(tiles) / ncols_safe))
        label_h = 14
        canvas = image_cls.new(
            "RGB",
            (ncols_safe * width, nrows * (height + label_h)),
            color=(255, 255, 255),
        )
        draw = draw_cls.Draw(canvas)
        for idx, tile in enumerate(tiles):
            row = idx // ncols_safe
            col = idx % ncols_safe
            x = col * width
            y = row * (height + label_h)
            text = labels[idx] if idx < len(labels) else f"idx:{idx}"
            draw.text((x + 2, y + 1), text, fill=(0, 0, 0))
            canvas.paste(tile, (x, y + label_h))
        canvas.save(out_path)

    @staticmethod
    def _to_image(image: Any, image_cls: Any) -> Any:
        tensor = image.detach().cpu().to(dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            msg = f"Expected image tensor [C,H,W], got shape {tuple(tensor.shape)}"
            raise ValueError(msg)
        channels = int(tensor.shape[0])
        if channels == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif channels == 2:
            tensor = torch.cat([tensor, tensor[:1]], dim=0)
        elif channels >= 3:
            tensor = tensor[:3]

        lo = float(tensor.min().item())
        hi = float(tensor.max().item())
        tensor = torch.zeros_like(tensor) if hi - lo <= 1e-8 else (tensor - lo) / (hi - lo)
        tensor_u8 = (tensor * 255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8)
        array = tensor_u8.permute(1, 2, 0).contiguous().numpy()
        pil_image = image_cls.fromarray(array, mode="RGB")
        width, height = pil_image.size
        scale = 1
        if max(width, height) <= 8:
            scale = 6
        elif max(width, height) <= 16:
            scale = 3
        if scale > 1:
            resampling = (
                image_cls.Resampling.NEAREST
                if hasattr(image_cls, "Resampling")
                else image_cls.NEAREST
            )
            pil_image = pil_image.resize(
                (int(width * scale), int(height * scale)),
                resample=resampling,
            )
        return pil_image
