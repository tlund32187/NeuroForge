"""Determinism controls for reproducible runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from neuroforge.core.torch_utils import require_torch

__all__ = ["DeterminismConfig", "apply_determinism"]


@dataclass(frozen=True, slots=True)
class DeterminismConfig:
    """Configuration for global determinism behaviour."""

    seed: int
    deterministic: bool = True
    benchmark: bool = False
    warn_only: bool = True
    cublas_workspace: str = ":4096:8"


def apply_determinism(cfg: DeterminismConfig) -> None:
    """Apply reproducibility settings for Python, NumPy, and Torch."""
    random.seed(cfg.seed)

    try:
        import numpy as np

        np.random.seed(cfg.seed)
    except ImportError:
        pass

    torch = require_torch()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg.cublas_workspace
        torch.use_deterministic_algorithms(True, warn_only=cfg.warn_only)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = bool(cfg.benchmark)
        return

    torch.use_deterministic_algorithms(False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(cfg.benchmark)
