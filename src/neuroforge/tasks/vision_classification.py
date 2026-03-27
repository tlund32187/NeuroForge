"""Vision classification task using FactoryHub backbone/readout/loss wiring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
from neuroforge.core.torch_utils import require_torch, resolve_device_dtype, smart_device
from neuroforge.data.datasets import (
    DatasetFactory,
    DatasetFactoryConfig,
    DatasetLoaders,
    DatasetTransformConfig,
)
from neuroforge.factories.hub import DEFAULT_HUB
from neuroforge.network.specs import VisionBackboneSpec, VisionBlockSpec, VisionInputSpec
from neuroforge.vision.backbones import NoBackbone
from neuroforge.vision.classification_engine import VisionClassificationEngine
from neuroforge.vision.factory import VisionBackboneFactory

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neuroforge.contracts.encoding import ILoss, IReadout
    from neuroforge.contracts.monitors import IEventBus
    from neuroforge.vision.backbones import VisionState

torch = require_torch()

__all__ = [
    "SyntheticVisionBatch",
    "SyntheticVisionDataLoader",
    "VisionClassificationConfig",
    "VisionClassificationResult",
    "VisionClassificationTask",
]

SUPPORTED_VISION_DATASETS: tuple[str, ...] = (
    "synthetic",
    "mnist",
    "fashion_mnist",
    "nmnist",
    "pokerdvs",
    "logic_gates_pixels",
)


@dataclass(frozen=True, slots=True)
class SyntheticVisionBatch:
    """One image classification mini-batch."""

    images: Any
    labels: Any


@dataclass(frozen=True, slots=True)
class VisionClassificationConfig:
    """Configuration for vision classification with synthetic or torchvision datasets."""

    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    deterministic: bool = True
    benchmark: bool = False
    warn_only: bool = True
    train_steps: int = 1
    batch_size: int = 8
    n_classes: int = 4
    lr: float = 1e-3
    loss_fn: str = "bce_logits"
    readout: str = "spike_count"
    readout_threshold: float = 0.0
    target_count: float = 1.0
    image_channels: int = 1
    image_h: int = 16
    image_w: int = 16
    dataset_name: str = "synthetic"
    dataset_root: str = ".cache/torchvision"
    dataset_download: bool = True
    dataset_val_fraction: float = 0.1
    dataset_num_workers: int = 0
    dataset_pin_memory: bool = False
    dataset_allow_fashion_mnist: bool = False
    dataset_normalize: bool = True
    dataset_random_crop: bool = False
    dataset_crop_padding: int = 2
    dataset_random_horizontal_flip: bool = False
    dataset_event_tensor_mode: str = "frames"
    dataset_event_slice_mode: str = "time"
    dataset_event_window_us: int = 100_000
    dataset_event_count: int = 20_000
    dataset_event_polarity_channels: int = 2
    dataset_logic_image_size: int = 8
    dataset_logic_gates: tuple[str, ...] = ("AND", "OR", "NAND", "NOR")
    dataset_logic_mode: str = "multiclass"
    dataset_logic_single_gate: str = "AND"
    dataset_logic_samples_per_gate: int = 128
    dataset_logic_total_samples: int = 0
    dataset_logic_train_ratio: float = 0.7
    dataset_logic_val_ratio: float = 0.15
    dataset_logic_test_ratio: float = 0.15
    backbone_type: str = "lif_convnet_v1"
    backbone_time_steps: int = 6
    backbone_encoding_mode: str = "rate"
    backbone_output_dim: int = 64
    backbone_blocks: tuple[VisionBlockSpec, ...] = field(
        default_factory=lambda: (
            VisionBlockSpec(type="conv", params={"out_channels": 16, "kernel_size": 3}),
            VisionBlockSpec(type="pool", params={"kernel_size": 2, "mode": "avg"}),
            VisionBlockSpec(type="res", params={"channels": 16, "depth": 1}),
        )
    )

    def __post_init__(self) -> None:
        if self.train_steps <= 0:
            msg = "train_steps must be > 0"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be > 0"
            raise ValueError(msg)
        if self.n_classes <= 1:
            msg = "n_classes must be > 1"
            raise ValueError(msg)
        if self.lr <= 0.0:
            msg = "lr must be > 0"
            raise ValueError(msg)
        if self.target_count <= 0.0:
            msg = "target_count must be > 0"
            raise ValueError(msg)
        if self.readout != "spike_count":
            msg = "VisionClassificationTask currently supports readout='spike_count'"
            raise ValueError(msg)
        if self.loss_fn not in {"bce_logits", "mse_count"}:
            msg = "loss_fn must be one of ['bce_logits', 'mse_count']"
            raise ValueError(msg)
        if self.dataset_num_workers < 0:
            msg = "dataset_num_workers must be >= 0"
            raise ValueError(msg)
        if not (0.0 <= self.dataset_val_fraction < 1.0):
            msg = "dataset_val_fraction must be in [0.0, 1.0)"
            raise ValueError(msg)
        dataset_name = self.dataset_name.strip().lower()
        if dataset_name not in SUPPORTED_VISION_DATASETS:
            msg = (
                "dataset_name must be one of "
                f"{list(SUPPORTED_VISION_DATASETS)}"
            )
            raise ValueError(msg)
        if dataset_name == "fashion_mnist" and not self.dataset_allow_fashion_mnist:
            msg = "fashion_mnist requested but dataset_allow_fashion_mnist=False"
            raise ValueError(msg)
        event_tensor_mode = self.dataset_event_tensor_mode.strip().lower()
        if event_tensor_mode not in {"frames", "voxel_grid"}:
            msg = "dataset_event_tensor_mode must be one of ['frames', 'voxel_grid']"
            raise ValueError(msg)
        object.__setattr__(self, "dataset_event_tensor_mode", event_tensor_mode)
        event_slice_mode = self.dataset_event_slice_mode.strip().lower()
        if event_slice_mode not in {"time", "count"}:
            msg = "dataset_event_slice_mode must be one of ['time', 'count']"
            raise ValueError(msg)
        object.__setattr__(self, "dataset_event_slice_mode", event_slice_mode)
        if self.dataset_event_window_us <= 0:
            msg = "dataset_event_window_us must be > 0"
            raise ValueError(msg)
        if self.dataset_event_count <= 0:
            msg = "dataset_event_count must be > 0"
            raise ValueError(msg)
        if self.dataset_event_polarity_channels not in {1, 2}:
            msg = "dataset_event_polarity_channels must be 1 or 2"
            raise ValueError(msg)
        logic_mode = self.dataset_logic_mode.strip().lower()
        if logic_mode not in {"multiclass", "single_gate"}:
            msg = "dataset_logic_mode must be one of ['multiclass', 'single_gate']"
            raise ValueError(msg)
        object.__setattr__(self, "dataset_logic_mode", logic_mode)
        if self.dataset_logic_image_size < 6:
            msg = "dataset_logic_image_size must be >= 6"
            raise ValueError(msg)
        if self.dataset_logic_samples_per_gate <= 0:
            msg = "dataset_logic_samples_per_gate must be > 0"
            raise ValueError(msg)
        if self.dataset_logic_total_samples < 0:
            msg = "dataset_logic_total_samples must be >= 0"
            raise ValueError(msg)
        if any(
            value < 0.0
            for value in (
                self.dataset_logic_train_ratio,
                self.dataset_logic_val_ratio,
                self.dataset_logic_test_ratio,
            )
        ):
            msg = "dataset logic split ratios must be >= 0"
            raise ValueError(msg)
        if self.dataset_logic_train_ratio <= 0.0:
            msg = "dataset_logic_train_ratio must be > 0"
            raise ValueError(msg)
        if (
            self.dataset_logic_train_ratio
            + self.dataset_logic_val_ratio
            + self.dataset_logic_test_ratio
            <= 0.0
        ):
            msg = "dataset logic split ratios must sum to > 0"
            raise ValueError(msg)
        backbone_type = self.backbone_type.strip()
        if not backbone_type:
            msg = "backbone_type must be a non-empty string"
            raise ValueError(msg)
        object.__setattr__(self, "backbone_type", backbone_type)
        if self.backbone_output_dim <= 0:
            msg = "backbone_output_dim must be > 0"
            raise ValueError(msg)
        object.__setattr__(self, "dataset_name", dataset_name)
        if self.device == "auto":
            image_neurons = self.image_channels * self.image_h * self.image_w
            if dataset_name in {"mnist", "fashion_mnist"}:
                image_neurons = 1 * 28 * 28
            if dataset_name == "nmnist":
                image_neurons = 2 * 34 * 34
            if dataset_name == "pokerdvs":
                image_neurons = 2 * 35 * 35
            if dataset_name == "logic_gates_pixels":
                image_neurons = 1 * self.dataset_logic_image_size * self.dataset_logic_image_size
            n_total = image_neurons + self.backbone_output_dim + self.n_classes
            object.__setattr__(self, "device", smart_device(n_total))

    def use_backbone(self) -> bool:
        """Return ``True`` when a named vision backbone should be resolved from hub."""
        return self.backbone_type.strip().lower() not in {"none", "null", "identity"}

    def build_backbone_spec(
        self,
        *,
        channels: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> VisionBackboneSpec:
        """Build a validated backbone spec from this config."""
        if not self.use_backbone():
            msg = "backbone_type='none' does not produce a VisionBackboneSpec"
            raise ValueError(msg)
        return VisionBackboneSpec(
            type=self.backbone_type,
            input=VisionInputSpec(
                channels=self.image_channels if channels is None else int(channels),
                height=self.image_h if height is None else int(height),
                width=self.image_w if width is None else int(width),
            ),
            time_steps=self.backbone_time_steps,
            encoding_mode=self.backbone_encoding_mode,
            blocks=[*self.backbone_blocks],
            output_dim=self.backbone_output_dim,
        )

    def build_dataset_factory_config(self) -> DatasetFactoryConfig:
        """Build dataset factory config for torchvision/Tonic dataset modes."""
        if self.dataset_name == "synthetic":
            msg = "Synthetic mode does not use DatasetFactoryConfig"
            raise ValueError(msg)
        return DatasetFactoryConfig(
            name=self.dataset_name,
            root=self.dataset_root,
            batch_size=self.batch_size,
            val_fraction=self.dataset_val_fraction,
            download=self.dataset_download,
            seed=self.seed,
            num_workers=self.dataset_num_workers,
            pin_memory=self.dataset_pin_memory,
            allow_fashion_mnist=self.dataset_allow_fashion_mnist,
            event_time_steps=self.backbone_time_steps,
            event_tensor_mode=self.dataset_event_tensor_mode,
            event_slice_mode=self.dataset_event_slice_mode,
            event_window_us=self.dataset_event_window_us,
            event_count=self.dataset_event_count,
            event_polarity_channels=self.dataset_event_polarity_channels,
            logic_image_size=self.dataset_logic_image_size,
            logic_gates=tuple(self.dataset_logic_gates),
            logic_mode=self.dataset_logic_mode,
            logic_single_gate=self.dataset_logic_single_gate,
            logic_samples_per_gate=self.dataset_logic_samples_per_gate,
            logic_total_samples=self.dataset_logic_total_samples,
            logic_train_ratio=self.dataset_logic_train_ratio,
            logic_val_ratio=self.dataset_logic_val_ratio,
            logic_test_ratio=self.dataset_logic_test_ratio,
            transforms=DatasetTransformConfig(
                normalize=self.dataset_normalize,
                random_crop=self.dataset_random_crop,
                crop_padding=self.dataset_crop_padding,
                random_horizontal_flip=self.dataset_random_horizontal_flip,
            ),
        )


@dataclass(slots=True)
class VisionClassificationResult:
    """Training result summary for the vision classification task."""

    steps: int
    final_loss: float
    final_accuracy: float
    loss_history: list[float] = field(default_factory=lambda: [])
    accuracy_history: list[float] = field(default_factory=lambda: [])
    dataset_meta: dict[str, Any] | None = None
    eval_loss: float = 0.0
    eval_accuracy: float = 0.0
    eval_steps: int = 0
    eval_samples: int = 0
    train_samples: int = 0
    train_steps_per_sec: float = 0.0
    train_samples_per_sec: float = 0.0
    train_ms_per_step: float = 0.0
    train_wall_ms: float = 0.0
    eval_steps_per_sec: float = 0.0
    eval_samples_per_sec: float = 0.0
    eval_ms_per_step: float = 0.0
    eval_wall_ms: float = 0.0
    runtime_device: str = ""
    runtime_dtype: str = ""


class SyntheticVisionDataLoader:
    """Deterministic synthetic batch generator (dataset stub)."""

    def __init__(
        self,
        *,
        seed: int,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        n_classes: int,
        device: Any,
        dtype: Any,
    ) -> None:
        self._batch_size = int(batch_size)
        self._channels = int(channels)
        self._height = int(height)
        self._width = int(width)
        self._n_classes = int(n_classes)
        self._device = device
        self._dtype = dtype
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(int(seed))

    def iter_batches(self, steps: int) -> Iterator[SyntheticVisionBatch]:
        """Yield ``steps`` synthetic mini-batches."""
        for _ in range(int(steps)):
            images_cpu = torch.rand(
                self._batch_size,
                self._channels,
                self._height,
                self._width,
                generator=self._gen,
                dtype=torch.float32,
            )
            labels_cpu = torch.randint(
                low=0,
                high=self._n_classes,
                size=(self._batch_size,),
                generator=self._gen,
                dtype=torch.long,
            )
            yield SyntheticVisionBatch(
                images=images_cpu.to(device=self._device, dtype=self._dtype),
                labels=labels_cpu.to(device=self._device),
            )


class VisionClassificationTask:
    """Supervised vision classification using vision backbone + head."""

    def __init__(
        self,
        config: VisionClassificationConfig | None = None,
        event_bus: IEventBus | None = None,
    ) -> None:
        self.config = config or VisionClassificationConfig()
        self._bus = event_bus

    def _emit(
        self,
        topic: str,
        step: int,
        source: str,
        data: dict[str, Any],
        *,
        t: float = 0.0,
    ) -> None:
        if self._bus is None:
            return
        from neuroforge.contracts.monitors import EventTopic, MonitorEvent

        self._bus.publish(
            MonitorEvent(
                topic=EventTopic(topic),
                step=step,
                t=t,
                source=source,
                data=data,
            )
        )

    @staticmethod
    def _build_target(labels: Any, n_classes: int, *, dtype: Any) -> Any:
        return torch.nn.functional.one_hot(labels, num_classes=n_classes).to(dtype=dtype)

    @staticmethod
    def _coerce_dataset_batch(
        batch: Any,
        *,
        device: Any,
        dtype: Any,
    ) -> SyntheticVisionBatch:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:  # pyright: ignore[reportUnknownArgumentType]
            msg = "Dataset loader must yield (images, labels) pairs"
            raise ValueError(msg)
        images = cast("Any", batch[0])
        labels = cast("Any", batch[1])
        if not hasattr(images, "to") or not hasattr(labels, "to"):
            msg = "Dataset batch tensors must support .to(device=..., dtype=...)"
            raise TypeError(msg)
        return SyntheticVisionBatch(
            images=images.to(device=device, dtype=dtype),
            labels=labels.to(device=device, dtype=torch.long),
        )

    @classmethod
    def _iter_dataset_batches(
        cls,
        loader: Any,
        *,
        steps: int,
        device: Any,
        dtype: Any,
    ) -> Iterator[SyntheticVisionBatch]:
        iterator = iter(loader)
        for _ in range(int(steps)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    msg = "Dataset loader is empty; cannot run training"
                    raise ValueError(msg) from exc
            yield cls._coerce_dataset_batch(
                batch,
                device=device,
                dtype=dtype,
            )

    @staticmethod
    def _sync_cuda_if_needed(device: Any) -> None:
        """Synchronise CUDA work before/after timing windows."""
        if not torch.cuda.is_available():
            return
        device_type = str(getattr(device, "type", device))
        if not device_type.startswith("cuda"):
            return
        torch.cuda.synchronize(device=device)

    def _resolve_batch_source(
        self,
        cfg: VisionClassificationConfig,
        *,
        device: Any,
        dtype: Any,
    ) -> tuple[
        Iterator[SyntheticVisionBatch],
        Iterator[SyntheticVisionBatch],
        dict[str, Any] | None,
        int,
        int,
        int,
        int,
    ]:
        def _loader_len(loader: Any) -> int:
            try:
                return max(0, int(len(loader)))
            except (TypeError, ValueError):
                return 0

        if cfg.dataset_name == "synthetic":
            loader = SyntheticVisionDataLoader(
                seed=cfg.seed,
                batch_size=cfg.batch_size,
                channels=cfg.image_channels,
                height=cfg.image_h,
                width=cfg.image_w,
                n_classes=cfg.n_classes,
                device=device,
                dtype=dtype,
            )
            eval_loader = SyntheticVisionDataLoader(
                seed=cfg.seed + 1,
                batch_size=cfg.batch_size,
                channels=cfg.image_channels,
                height=cfg.image_h,
                width=cfg.image_w,
                n_classes=cfg.n_classes,
                device=device,
                dtype=dtype,
            )
            eval_steps = max(1, min(int(cfg.train_steps), 8))
            return (
                loader.iter_batches(cfg.train_steps),
                eval_loader.iter_batches(eval_steps),
                None,
                cfg.image_channels,
                cfg.image_h,
                cfg.image_w,
                cfg.n_classes,
            )

        ds_cfg = cfg.build_dataset_factory_config()
        ds_bundle: DatasetLoaders = DatasetFactory().build_loaders(ds_cfg)
        dataset_meta = ds_bundle.meta.to_dict()
        eval_loader = ds_bundle.val
        eval_steps = _loader_len(eval_loader)
        if eval_steps <= 0:
            eval_loader = ds_bundle.test
            eval_steps = _loader_len(eval_loader)
        eval_steps = max(1, eval_steps)
        return (
            self._iter_dataset_batches(
                ds_bundle.train,
                steps=cfg.train_steps,
                device=device,
                dtype=dtype,
            ),
            self._iter_dataset_batches(
                eval_loader,
                steps=eval_steps,
                device=device,
                dtype=dtype,
            ),
            dataset_meta,
            int(ds_bundle.meta.channels),
            int(ds_bundle.meta.height),
            int(ds_bundle.meta.width),
            int(ds_bundle.meta.n_classes),
        )

    def run(self) -> VisionClassificationResult:
        """Run training for ``config.train_steps`` mini-batches."""
        cfg = self.config
        apply_determinism(
            DeterminismConfig(
                seed=cfg.seed,
                deterministic=cfg.deterministic,
                benchmark=cfg.benchmark,
                warn_only=cfg.warn_only,
            )
        )

        dev, tdt = resolve_device_dtype(cfg.device, cfg.dtype)
        (
            batch_iter,
            eval_iter,
            dataset_meta,
            input_channels,
            input_h,
            input_w,
            effective_n_classes,
        ) = self._resolve_batch_source(
            cfg,
            device=dev,
            dtype=tdt,
        )
        if cfg.use_backbone():
            backbone_spec = cfg.build_backbone_spec(
                channels=input_channels,
                height=input_h,
                width=input_w,
            )
            factory_obj = DEFAULT_HUB.vision_backbones.create(
                backbone_spec.type, spec=backbone_spec,
            )
            if not isinstance(factory_obj, VisionBackboneFactory):
                msg = (
                    "DEFAULT_HUB.vision_backbones must resolve VisionBackboneFactory; "
                    f"got {type(factory_obj).__name__}"
                )
                raise TypeError(msg)
            backbone = cast("Any", factory_obj.build())
            if not isinstance(backbone, torch.nn.Module):
                msg = "Resolved vision backbone must be torch.nn.Module"
                raise TypeError(msg)
            backbone = backbone.to(device=dev, dtype=tdt)
            backbone_name = backbone_spec.type
            backbone_encoding_mode = backbone_spec.encoding_mode
            backbone_time_steps = int(backbone_spec.time_steps)
            backbone_output_dim = int(backbone_spec.output_dim)
            topology_blocks = [name for name in getattr(backbone, "block_names", ())]
        else:
            backbone = NoBackbone(
                channels=int(input_channels),
                height=int(input_h),
                width=int(input_w),
            ).to(device=dev, dtype=tdt)
            backbone_name = "none"
            backbone_encoding_mode = "none"
            backbone_time_steps = 1
            backbone_output_dim = int(backbone.output_dim)
            topology_blocks = ["input"]

        readout = cast("IReadout", DEFAULT_HUB.readouts.create(
            cfg.readout,
            threshold=float(cfg.readout_threshold),
        ))
        loss_fn = cast("ILoss", DEFAULT_HUB.losses.create(cfg.loss_fn))
        engine = VisionClassificationEngine(
            backbone=backbone,
            n_classes=effective_n_classes,
            readout=readout,
        ).to(device=dev, dtype=tdt)

        optimizer = torch.optim.Adam(engine.parameters(), lr=cfg.lr)

        self._emit(
            "run_start",
            0,
            "VISION",
            {
                "task": "vision_classification",
                "seed": cfg.seed,
                "device": str(dev),
                "dtype": cfg.dtype,
                "dataset": cfg.dataset_name,
            },
        )
        self._emit(
            "training_start",
            0,
            "VISION",
            {
                "task": "vision_classification",
                "steps": cfg.train_steps,
                "batch_size": cfg.batch_size,
                "n_classes": effective_n_classes,
                "backbone": backbone_name,
                "encoding_mode": backbone_encoding_mode,
                "time_steps": backbone_time_steps,
                "dataset": cfg.dataset_name,
                "dataset_meta": dataset_meta,
            },
        )
        self._emit(
            "topology",
            0,
            "VISION",
            {
                "layers": [
                    f"image({input_channels}x{input_h}x{input_w})",
                    f"backbone({backbone_name})",
                    f"features({backbone_output_dim})",
                    f"head({effective_n_classes})",
                ],
                "edges": [
                    {"name": "image_backbone", "src": "image", "dst": "backbone"},
                    {"name": "backbone_head", "src": "backbone", "dst": "head"},
                ],
                "vision_blocks": topology_blocks,
                "dataset": cfg.dataset_name,
            },
        )

        loss_history: list[float] = []
        accuracy_history: list[float] = []
        train_samples = 0
        state: VisionState | None = None
        self._sync_cuda_if_needed(dev)
        train_t0 = time.perf_counter()
        for step_idx, batch in enumerate(batch_iter, start=1):
            self._sync_cuda_if_needed(dev)
            t0 = time.perf_counter()
            optimizer.zero_grad()
            step_out = engine.forward_step(batch.images, state=state)
            state = step_out.state

            target = self._build_target(
                batch.labels,
                effective_n_classes,
                dtype=step_out.logits.dtype,
            )
            if cfg.loss_fn == "bce_logits":
                loss = loss_fn(step_out.readout.logits, target)
            else:
                loss = loss_fn(step_out.readout.count, target * float(cfg.target_count))

            loss.backward()
            optimizer.step()
            self._sync_cuda_if_needed(dev)

            pred = step_out.readout.logits.argmax(dim=-1)
            accuracy = float((pred == batch.labels).float().mean().item())
            loss_value = float(loss.detach().item())
            wall_ms = (time.perf_counter() - t0) * 1000.0
            batch_size = int(batch.labels.shape[0])
            train_samples += batch_size
            loss_history.append(loss_value)
            accuracy_history.append(accuracy)

            sample_cap = min(int(batch.images.shape[0]), 16)
            trial_data: dict[str, Any] = {
                "batch_size": batch_size,
                "loss": loss_value,
                "accuracy": accuracy,
                "predicted": pred.detach().cpu().tolist(),
                "expected": batch.labels.detach().cpu().tolist(),
                "images": batch.images[:sample_cap].detach().cpu(),
            }
            self._emit("training_trial", step_idx, "VISION", trial_data)

            scalar_data: dict[str, Any] = {
                "trial": step_idx,
                "epoch": 0,
                "gate": "VISION",
                "loss": loss_value,
                "accuracy": accuracy,
                "wall_ms": wall_ms,
                "vision.layer.head.spike_rate": float(step_out.logits.abs().mean().item()),
                "vision.layer.head.mean_activation": float(step_out.logits.mean().item()),
                "vision.layer.head.max_activation": float(step_out.logits.abs().max().item()),
                "vision.layer.features.spike_rate": float(step_out.features.abs().mean().item()),
                "vision.layer.features.mean_activation": float(step_out.features.mean().item()),
                "vision.layer.features.max_activation": float(
                    step_out.features.abs().max().item()
                ),
            }
            if state is not None:
                for name, counts in state.per_layer_spike_counts.items():
                    scalar_data[f"spikes.{name}.mean"] = float(counts.mean().item())
                    neurons = int(state.per_layer_neuron_count.get(name, 0))
                    denom = float(max(1, neurons * max(1, state.time_steps)))
                    scalar_data[f"vision.layer.{name}.spike_rate"] = float(
                        counts.mean().item() / denom
                    )
                    scalar_data[f"vision.layer.{name}.mean_activation"] = float(
                        counts.mean().item() / float(max(1, state.time_steps))
                    )
                    scalar_data[f"vision.layer.{name}.max_activation"] = float(
                        counts.max().item() / float(max(1, state.time_steps))
                    )
            self._emit("scalar", step_idx, "VISION", scalar_data)

        self._sync_cuda_if_needed(dev)
        train_elapsed_s = max(1e-9, time.perf_counter() - train_t0)
        train_steps = len(loss_history)
        train_steps_per_sec = float(train_steps / train_elapsed_s) if train_steps > 0 else 0.0
        train_samples_per_sec = (
            float(train_samples / train_elapsed_s) if train_samples > 0 else 0.0
        )
        train_ms_per_step = (
            float((train_elapsed_s * 1000.0) / train_steps) if train_steps > 0 else 0.0
        )

        eval_steps = 0
        eval_samples = 0
        eval_loss_sum = 0.0
        eval_correct = 0
        engine.eval()
        self._sync_cuda_if_needed(dev)
        eval_t0 = time.perf_counter()
        with torch.no_grad():
            for eval_batch in eval_iter:
                self._sync_cuda_if_needed(dev)
                eval_out = engine.forward_step(eval_batch.images, state=None)
                eval_target = self._build_target(
                    eval_batch.labels,
                    effective_n_classes,
                    dtype=eval_out.logits.dtype,
                )
                if cfg.loss_fn == "bce_logits":
                    eval_loss_tensor = loss_fn(eval_out.readout.logits, eval_target)
                else:
                    eval_loss_tensor = loss_fn(
                        eval_out.readout.count,
                        eval_target * float(cfg.target_count),
                    )
                eval_loss_sum += float(eval_loss_tensor.detach().item())
                eval_pred = eval_out.readout.logits.argmax(dim=-1)
                eval_correct += int((eval_pred == eval_batch.labels).sum().item())
                eval_samples += int(eval_batch.labels.shape[0])
                eval_steps += 1
        self._sync_cuda_if_needed(dev)
        eval_elapsed_s = max(1e-9, time.perf_counter() - eval_t0)
        engine.train()

        eval_loss = float(eval_loss_sum / max(1, eval_steps))
        eval_acc = float(eval_correct / max(1, eval_samples))
        eval_steps_per_sec = float(eval_steps / eval_elapsed_s) if eval_steps > 0 else 0.0
        eval_samples_per_sec = float(eval_samples / eval_elapsed_s) if eval_samples > 0 else 0.0
        eval_ms_per_step = float((eval_elapsed_s * 1000.0) / max(1, eval_steps))

        final_loss = loss_history[-1]
        final_acc = accuracy_history[-1]
        self._emit(
            "scalar",
            cfg.train_steps,
            "VISION",
            {
                "trial": cfg.train_steps,
                "epoch": 0,
                "gate": "VISION",
                "loss": final_loss,
                "accuracy": final_acc,
                "perf.steps_per_sec": train_steps_per_sec,
                "perf.ms_per_step": train_ms_per_step,
                "perf.samples_per_sec": train_samples_per_sec,
                "perf.train.steps_per_sec": train_steps_per_sec,
                "perf.train.samples_per_sec": train_samples_per_sec,
                "perf.eval.steps_per_sec": eval_steps_per_sec,
                "perf.eval.samples_per_sec": eval_samples_per_sec,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_acc,
            },
        )
        self._emit(
            "training_end",
            cfg.train_steps,
            "VISION",
            {
                # Vision tasks are budget-driven (fixed steps), so a completed run
                # should be treated as successful by dashboard status logic.
                "converged": True,
                "steps": cfg.train_steps,
                "final_loss": final_loss,
                "final_accuracy": final_acc,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_acc,
                "train_steps_per_sec": train_steps_per_sec,
                "train_samples_per_sec": train_samples_per_sec,
                "eval_steps_per_sec": eval_steps_per_sec,
                "eval_samples_per_sec": eval_samples_per_sec,
            },
        )
        return VisionClassificationResult(
            steps=cfg.train_steps,
            final_loss=final_loss,
            final_accuracy=final_acc,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
            dataset_meta=dataset_meta,
            eval_loss=eval_loss,
            eval_accuracy=eval_acc,
            eval_steps=eval_steps,
            eval_samples=eval_samples,
            train_samples=train_samples,
            train_steps_per_sec=train_steps_per_sec,
            train_samples_per_sec=train_samples_per_sec,
            train_ms_per_step=train_ms_per_step,
            train_wall_ms=float(train_elapsed_s * 1000.0),
            eval_steps_per_sec=eval_steps_per_sec,
            eval_samples_per_sec=eval_samples_per_sec,
            eval_ms_per_step=eval_ms_per_step,
            eval_wall_ms=float(eval_elapsed_s * 1000.0),
            runtime_device=str(dev),
            runtime_dtype=str(tdt),
        )
