"""Micro-benchmark helpers for engine step throughput."""

from __future__ import annotations

from time import perf_counter
from typing import Any

__all__ = ["measure_steps_per_sec"]


def measure_steps_per_sec(
    engine: Any,
    steps: int,
    external_drive_fn: Any | None = None,
    *,
    sync_cuda: bool = False,
) -> dict[str, float]:
    """Measure simulation throughput in steps/sec and ms/step.

    Parameters
    ----------
    engine:
        Engine instance with ``run_steps`` or ``run``.
    steps:
        Number of steps to execute.
    external_drive_fn:
        Optional callback used per step.
    sync_cuda:
        If ``True`` and CUDA is available, synchronize before/after timing.
    """
    from neuroforge.core.torch_utils import require_torch

    torch = require_torch()
    should_sync = bool(sync_cuda and torch.cuda.is_available())

    if should_sync:
        torch.cuda.synchronize()
    t0 = perf_counter()

    if hasattr(engine, "run_steps"):
        engine.run_steps(steps, external_drive_fn, collect=False)
    else:
        engine.run(steps, external_drive_fn)

    if should_sync:
        torch.cuda.synchronize()
    elapsed_s = perf_counter() - t0

    if steps <= 0:
        steps_per_sec = 0.0
        ms_per_step = 0.0
    elif elapsed_s <= 0.0:
        steps_per_sec = float("inf")
        ms_per_step = 0.0
    else:
        steps_per_sec = float(steps / elapsed_s)
        ms_per_step = float((elapsed_s * 1000.0) / steps)

    return {
        "steps": float(steps),
        "elapsed_s": float(elapsed_s),
        "steps_per_sec": steps_per_sec,
        "ms_per_step": ms_per_step,
    }
