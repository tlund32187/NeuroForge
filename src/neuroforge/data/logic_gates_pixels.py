"""Deterministic synthetic logic-gate pixel dataset for vision pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuroforge.core.torch_utils import require_torch

torch = require_torch()

__all__ = [
    "LOGIC_GATE_TABLES",
    "LogicGatesPixelsConfig",
    "LogicGatesPixelsDataset",
    "LogicGatesPixelsSplits",
    "build_logic_gates_pixels_splits",
]

LOGIC_GATE_TABLES: dict[str, dict[tuple[int, int], int]] = {
    "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "NAND": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "NOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
    "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
}
_DEFAULT_GATES: tuple[str, ...] = ("AND", "OR", "NAND", "NOR")
_VALID_MODES: frozenset[str] = frozenset({"multiclass", "single_gate"})
_PATTERNS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True, slots=True)
class LogicGatesPixelsConfig:
    """Config for deterministic logic-gates pixel dataset generation."""

    image_size: int = 8
    gates: tuple[str, ...] = _DEFAULT_GATES
    mode: str = "multiclass"
    single_gate: str = "AND"
    samples_per_gate: int = 128
    total_samples: int = 0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if self.image_size < 6:
            msg = "LogicGatesPixelsConfig.image_size must be >= 6"
            raise ValueError(msg)

        gate_names: list[str] = []
        seen: set[str] = set()
        for gate in self.gates:
            norm = str(gate).strip().upper()
            if norm not in LOGIC_GATE_TABLES:
                msg = f"Unsupported logic gate: {gate!r}"
                raise ValueError(msg)
            if norm in seen:
                continue
            seen.add(norm)
            gate_names.append(norm)
        if not gate_names:
            msg = "LogicGatesPixelsConfig.gates must contain at least one gate"
            raise ValueError(msg)
        object.__setattr__(self, "gates", tuple(gate_names))

        mode = str(self.mode).strip().lower()
        if mode not in _VALID_MODES:
            msg = f"LogicGatesPixelsConfig.mode must be one of {sorted(_VALID_MODES)}"
            raise ValueError(msg)
        object.__setattr__(self, "mode", mode)

        single_gate = str(self.single_gate).strip().upper()
        if single_gate not in LOGIC_GATE_TABLES:
            msg = f"Unsupported single_gate value: {self.single_gate!r}"
            raise ValueError(msg)
        object.__setattr__(self, "single_gate", single_gate)

        if self.samples_per_gate <= 0:
            msg = "LogicGatesPixelsConfig.samples_per_gate must be > 0"
            raise ValueError(msg)
        if self.total_samples < 0:
            msg = "LogicGatesPixelsConfig.total_samples must be >= 0"
            raise ValueError(msg)

        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(float(r) < 0.0 for r in ratios):
            msg = "LogicGatesPixelsConfig train/val/test ratios must be >= 0"
            raise ValueError(msg)
        if float(self.train_ratio) <= 0.0:
            msg = "LogicGatesPixelsConfig.train_ratio must be > 0"
            raise ValueError(msg)
        ratio_sum = float(self.train_ratio + self.val_ratio + self.test_ratio)
        if ratio_sum <= 0.0:
            msg = "LogicGatesPixelsConfig train/val/test ratios must sum to > 0"
            raise ValueError(msg)


class LogicGatesPixelsDataset(torch.utils.data.Dataset[Any]):
    """In-memory tensor dataset yielding ``([1,H,W], label)`` pairs."""

    def __init__(
        self,
        *,
        images: Any,
        labels: Any,
        class_names: tuple[str, ...],
    ) -> None:
        image_tensor = images.to(dtype=torch.float32).contiguous()
        label_tensor = labels.to(dtype=torch.long).contiguous()
        if image_tensor.ndim != 4:
            msg = f"images must have shape [N,1,H,W], got {tuple(image_tensor.shape)}"
            raise ValueError(msg)
        if image_tensor.shape[1] != 1:
            msg = f"images must have channel dimension C=1, got {int(image_tensor.shape[1])}"
            raise ValueError(msg)
        if label_tensor.ndim != 1:
            msg = f"labels must have shape [N], got {tuple(label_tensor.shape)}"
            raise ValueError(msg)
        if int(image_tensor.shape[0]) != int(label_tensor.shape[0]):
            msg = "images and labels length mismatch"
            raise ValueError(msg)
        self._images = image_tensor
        self._labels = label_tensor
        self.class_names = tuple(str(name) for name in class_names)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self._images[index], self._labels[index]


@dataclass(frozen=True, slots=True)
class LogicGatesPixelsSplits:
    """Bundle of generated train/val/test logic-gate pixel datasets."""

    train: LogicGatesPixelsDataset
    val: LogicGatesPixelsDataset
    test: LogicGatesPixelsDataset
    n_classes: int
    class_names: tuple[str, ...]


def _split_counts(*, total: int, cfg: LogicGatesPixelsConfig) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    ratio_sum = float(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio)
    train_f = float(cfg.train_ratio) / ratio_sum
    val_f = float(cfg.val_ratio) / ratio_sum
    train_n = int(round(float(total) * train_f))
    val_n = int(round(float(total) * val_f))
    train_n = max(0, min(total, train_n))
    val_n = max(0, min(total - train_n, val_n))
    test_n = int(total - train_n - val_n)
    return train_n, val_n, test_n


def _sample_count_per_gate(cfg: LogicGatesPixelsConfig) -> list[int]:
    n_gates = len(cfg.gates)
    if cfg.mode == "single_gate":
        total = int(cfg.total_samples) if cfg.total_samples > 0 else int(cfg.samples_per_gate)
        return [max(1, total)]

    if cfg.total_samples > 0:
        total = int(cfg.total_samples)
        base = total // n_gates
        rem = total % n_gates
        out = [base for _ in range(n_gates)]
        for idx in range(rem):
            out[idx] += 1
        return [max(1, value) for value in out]

    return [int(cfg.samples_per_gate) for _ in range(n_gates)]


def _render_logic_image(
    *,
    bits: tuple[int, int],
    gate_idx: int,
    n_gates: int,
    image_size: int,
    include_gate_context: bool,
) -> Any:
    image = torch.zeros((1, int(image_size), int(image_size)), dtype=torch.float32)
    row = min(2, image_size - 1)
    col_a = min(2, image_size - 1)
    col_b = max(0, image_size - 3)
    if bits[0] == 1:
        image[0, row, col_a] = 1.0
    if bits[1] == 1:
        image[0, row, col_b] = 1.0
    if include_gate_context and 0 <= gate_idx < n_gates and gate_idx < image_size:
        image[0, 0, gate_idx] = 1.0
    return image


def build_logic_gates_pixels_splits(cfg: LogicGatesPixelsConfig) -> LogicGatesPixelsSplits:
    """Generate deterministic train/val/test logic-gate pixel datasets."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.seed))

    images: list[Any] = []
    labels: list[int] = []

    gate_names: tuple[str, ...]
    class_names: tuple[str, ...]
    if cfg.mode == "single_gate":
        gate_names = (cfg.single_gate,)
        class_names = (f"{cfg.single_gate}:0", f"{cfg.single_gate}:1")
    else:
        gate_names = cfg.gates
        class_names = tuple(gate_names)

    per_gate_counts = _sample_count_per_gate(cfg)
    include_gate_context = cfg.mode == "multiclass"
    for gate_idx, gate_name in enumerate(gate_names):
        table = LOGIC_GATE_TABLES[gate_name]
        n_samples = per_gate_counts[gate_idx]
        for _ in range(int(n_samples)):
            pattern_idx = int(torch.randint(0, len(_PATTERNS), (1,), generator=gen).item())
            bits = _PATTERNS[pattern_idx]
            image = _render_logic_image(
                bits=bits,
                gate_idx=gate_idx,
                n_gates=len(gate_names),
                image_size=int(cfg.image_size),
                include_gate_context=include_gate_context,
            )
            if cfg.mode == "single_gate":
                label = int(table[bits])
            else:
                label = int(gate_idx)
            images.append(image)
            labels.append(label)

    total = len(images)
    if total <= 0:
        msg = "LogicGatesPixelsConfig produced zero samples"
        raise ValueError(msg)

    perm = torch.randperm(total, generator=gen).tolist()
    image_tensor = torch.stack([images[idx] for idx in perm], dim=0)
    label_tensor = torch.tensor([labels[idx] for idx in perm], dtype=torch.long)

    train_n, val_n, test_n = _split_counts(total=total, cfg=cfg)
    train_end = train_n
    val_end = train_n + val_n
    test_end = train_n + val_n + test_n

    train_ds = LogicGatesPixelsDataset(
        images=image_tensor[:train_end],
        labels=label_tensor[:train_end],
        class_names=class_names,
    )
    val_ds = LogicGatesPixelsDataset(
        images=image_tensor[train_end:val_end],
        labels=label_tensor[train_end:val_end],
        class_names=class_names,
    )
    test_ds = LogicGatesPixelsDataset(
        images=image_tensor[val_end:test_end],
        labels=label_tensor[val_end:test_end],
        class_names=class_names,
    )

    return LogicGatesPixelsSplits(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        n_classes=len(class_names),
        class_names=class_names,
    )
