"""Legacy seed helper for backward compatibility."""

from __future__ import annotations

__all__ = ["set_global_seed"]


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds while preserving current determinism mode."""
    from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
    from neuroforge.core.torch_utils import require_torch

    torch = require_torch()
    deterministic_enabled = bool(torch.are_deterministic_algorithms_enabled())
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    benchmark_enabled = bool(getattr(cudnn_backend, "benchmark", False))

    apply_determinism(
        DeterminismConfig(
            seed=seed,
            deterministic=deterministic_enabled,
            benchmark=benchmark_enabled,
            warn_only=True,
        )
    )
