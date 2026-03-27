"""Runner helper for vision classification task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.tasks.vision_classification import (
    VisionClassificationConfig,
    VisionClassificationTask,
)

if TYPE_CHECKING:
    from neuroforge.contracts.monitors import IEventBus

__all__ = ["VisionRunnerConfig", "run_vision_classification"]


@dataclass(frozen=True, slots=True)
class VisionRunnerConfig:
    """CLI-friendly runner config for vision classification."""

    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    deterministic: bool = True
    benchmark: bool = False
    warn_only: bool = True
    steps: int = 1
    batch_size: int = 8
    n_classes: int = 4
    image_channels: int = 1
    image_h: int = 16
    image_w: int = 16
    dataset: str = "synthetic"
    dataset_root: str = ".cache/torchvision"
    dataset_download: bool = True
    dataset_val_fraction: float = 0.1
    dataset_num_workers: int = 0
    dataset_pin_memory: bool = False
    allow_fashion_mnist: bool = False
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
    lr: float = 1e-3
    loss_fn: str = "bce_logits"
    readout: str = "spike_count"
    readout_threshold: float = 0.0
    backbone_type: str = "lif_convnet_v1"
    backbone_time_steps: int = 6
    backbone_encoding_mode: str = "rate"
    backbone_output_dim: int = 64
    backbone_blocks: tuple[dict[str, Any], ...] | None = None


def run_vision_classification(
    cfg: VisionRunnerConfig,
    *,
    event_bus: IEventBus | None = None,
) -> dict[str, Any]:
    """Run a vision classification task and return a summary dict."""
    task_kwargs: dict[str, Any] = {
        "seed": cfg.seed,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "deterministic": cfg.deterministic,
        "benchmark": cfg.benchmark,
        "warn_only": cfg.warn_only,
        "train_steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "n_classes": cfg.n_classes,
        "lr": cfg.lr,
        "loss_fn": cfg.loss_fn,
        "readout": cfg.readout,
        "readout_threshold": cfg.readout_threshold,
        "image_channels": cfg.image_channels,
        "image_h": cfg.image_h,
        "image_w": cfg.image_w,
        "dataset_name": cfg.dataset,
        "dataset_root": cfg.dataset_root,
        "dataset_download": cfg.dataset_download,
        "dataset_val_fraction": cfg.dataset_val_fraction,
        "dataset_num_workers": cfg.dataset_num_workers,
        "dataset_pin_memory": cfg.dataset_pin_memory,
        "dataset_allow_fashion_mnist": cfg.allow_fashion_mnist,
        "dataset_normalize": cfg.dataset_normalize,
        "dataset_random_crop": cfg.dataset_random_crop,
        "dataset_crop_padding": cfg.dataset_crop_padding,
        "dataset_random_horizontal_flip": cfg.dataset_random_horizontal_flip,
        "dataset_event_tensor_mode": cfg.dataset_event_tensor_mode,
        "dataset_event_slice_mode": cfg.dataset_event_slice_mode,
        "dataset_event_window_us": cfg.dataset_event_window_us,
        "dataset_event_count": cfg.dataset_event_count,
        "dataset_event_polarity_channels": cfg.dataset_event_polarity_channels,
        "dataset_logic_image_size": cfg.dataset_logic_image_size,
        "dataset_logic_gates": tuple(cfg.dataset_logic_gates),
        "dataset_logic_mode": cfg.dataset_logic_mode,
        "dataset_logic_single_gate": cfg.dataset_logic_single_gate,
        "dataset_logic_samples_per_gate": cfg.dataset_logic_samples_per_gate,
        "dataset_logic_total_samples": cfg.dataset_logic_total_samples,
        "dataset_logic_train_ratio": cfg.dataset_logic_train_ratio,
        "dataset_logic_val_ratio": cfg.dataset_logic_val_ratio,
        "dataset_logic_test_ratio": cfg.dataset_logic_test_ratio,
        "backbone_time_steps": cfg.backbone_time_steps,
        "backbone_type": cfg.backbone_type,
        "backbone_encoding_mode": cfg.backbone_encoding_mode,
        "backbone_output_dim": cfg.backbone_output_dim,
    }
    if cfg.backbone_blocks is not None:
        task_kwargs["backbone_blocks"] = tuple(cfg.backbone_blocks)
    task_cfg = VisionClassificationConfig(**task_kwargs)
    task = VisionClassificationTask(task_cfg, event_bus=event_bus)
    result = task.run()
    return {
        "task": "vision_classification",
        "config": asdict(task_cfg),
        "result": {
            "steps": result.steps,
            "final_loss": result.final_loss,
            "final_accuracy": result.final_accuracy,
            "eval_loss": result.eval_loss,
            "eval_accuracy": result.eval_accuracy,
            "eval_steps": result.eval_steps,
            "eval_samples": result.eval_samples,
            "runtime_device": result.runtime_device,
            "runtime_dtype": result.runtime_dtype,
            "performance": {
                "train": {
                    "steps": result.steps,
                    "samples": result.train_samples,
                    "steps_per_sec": result.train_steps_per_sec,
                    "samples_per_sec": result.train_samples_per_sec,
                    "ms_per_step": result.train_ms_per_step,
                    "wall_ms": result.train_wall_ms,
                },
                "eval": {
                    "steps": result.eval_steps,
                    "samples": result.eval_samples,
                    "steps_per_sec": result.eval_steps_per_sec,
                    "samples_per_sec": result.eval_samples_per_sec,
                    "ms_per_step": result.eval_ms_per_step,
                    "wall_ms": result.eval_wall_ms,
                    "loss": result.eval_loss,
                    "accuracy": result.eval_accuracy,
                },
            },
            "loss_history": [*result.loss_history],
            "accuracy_history": [*result.accuracy_history],
        },
        "dataset_meta": result.dataset_meta,
    }
