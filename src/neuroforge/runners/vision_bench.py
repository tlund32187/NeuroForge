"""Phase 8 benchmark harness for vision training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
from neuroforge.core.torch_utils import require_torch
from neuroforge.monitors.artifact_writer import ArtifactWriter
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.cuda_monitor import CudaMetricsMonitor
from neuroforge.monitors.topology_stats_monitor import TopologyStatsMonitor
from neuroforge.monitors.vision_monitors import (
    ConfusionMatrixExporter,
    ConfusionMatrixMonitor,
    VisionLayerStatsExporter,
    VisionLayerStatsMonitor,
    VisionSampleGridExporter,
    VisionSampleGridMonitor,
)
from neuroforge.runners.run_context import RunContext, create_run_dir
from neuroforge.runners.vision import VisionRunnerConfig, run_vision_classification

__all__ = [
    "VisionBenchPlan",
    "build_plan_from_config",
    "run_vision_benchmark",
]

_ROOT_RESERVED_KEYS: frozenset[str] = frozenset({"artifacts", "seeds", "runner", "vision"})


@dataclass(frozen=True, slots=True)
class VisionBenchPlan:
    """Resolved plan for running one or multiple vision benchmark seeds."""

    artifacts: str
    seeds: tuple[int, ...]
    runner: VisionRunnerConfig
    config_path: str


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        msg = "At least one seed is required"
        raise ValueError(msg)
    out: list[int] = []
    for part in parts:
        try:
            out.append(int(part))
        except ValueError as exc:
            msg = f"Invalid seed value: {part!r}"
            raise ValueError(msg) from exc
    return tuple(out)


def _coerce_seed_values(raw: object) -> tuple[int, ...]:
    if isinstance(raw, int):
        return (int(raw),)
    if isinstance(raw, str):
        return _parse_seed_list(raw)
    if isinstance(raw, list):
        out: list[int] = []
        for item in raw:
            if isinstance(item, bool):
                msg = "Boolean values are not valid seeds"
                raise ValueError(msg)
            try:
                out.append(int(item))
            except (TypeError, ValueError) as exc:
                msg = f"Invalid seed value in list: {item!r}"
                raise ValueError(msg) from exc
        if not out:
            msg = "At least one seed is required"
            raise ValueError(msg)
        return tuple(out)
    msg = f"Unsupported seeds value: {raw!r}"
    raise ValueError(msg)


def _runner_from_raw(raw: dict[str, Any]) -> VisionRunnerConfig:
    runner_fields = {f.name for f in fields(VisionRunnerConfig)}
    unknown = [key for key in raw if key not in runner_fields]
    if unknown:
        msg = f"Unknown vision runner config keys: {unknown}"
        raise ValueError(msg)
    return VisionRunnerConfig(**raw)


def _extract_runner_config(raw: dict[str, Any]) -> dict[str, Any]:
    if "runner" in raw:
        runner_obj = raw["runner"]
        if not isinstance(runner_obj, dict):
            msg = "config['runner'] must be an object"
            raise ValueError(msg)
        unknown_runner_root = [key for key in raw if key not in _ROOT_RESERVED_KEYS]
        if unknown_runner_root:
            msg = (
                "Top-level keys are not allowed when 'runner' is present; "
                f"unknown keys: {unknown_runner_root}"
            )
            raise ValueError(msg)
        return {str(k): v for k, v in runner_obj.items()}

    if "vision" in raw:
        vision_obj = raw["vision"]
        if not isinstance(vision_obj, dict):
            msg = "config['vision'] must be an object"
            raise ValueError(msg)
        unknown_vision_root = [key for key in raw if key not in _ROOT_RESERVED_KEYS]
        if unknown_vision_root:
            msg = (
                "Top-level keys are not allowed when 'vision' is present; "
                f"unknown keys: {unknown_vision_root}"
            )
            raise ValueError(msg)
        return {str(k): v for k, v in vision_obj.items()}

    runner_fields = {f.name for f in fields(VisionRunnerConfig)}
    out: dict[str, Any] = {}
    unknown_root: list[str] = []
    for key, value in raw.items():
        key_s = str(key)
        if key_s in _ROOT_RESERVED_KEYS:
            continue
        if key_s in runner_fields:
            out[key_s] = value
            continue
        unknown_root.append(key_s)
    if unknown_root:
        msg = f"Unknown top-level config keys: {unknown_root}"
        raise ValueError(msg)
    return out


def build_plan_from_config(
    config_path: Path,
    *,
    seeds_override: tuple[int, ...] | None = None,
    artifacts_override: str | None = None,
) -> VisionBenchPlan:
    """Load and validate a benchmark plan from JSON config."""
    try:
        raw_obj = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Config is not valid JSON: {config_path}"
        raise ValueError(msg) from exc
    if not isinstance(raw_obj, dict):
        msg = "Config root must be a JSON object"
        raise ValueError(msg)
    raw = cast("dict[str, Any]", raw_obj)

    runner_cfg = _runner_from_raw(_extract_runner_config(raw))
    artifacts = artifacts_override or str(raw.get("artifacts", "artifacts"))

    if seeds_override is not None:
        seeds = seeds_override
    elif "seeds" in raw:
        seeds = _coerce_seed_values(raw["seeds"])
    else:
        seeds = (int(runner_cfg.seed),)

    return VisionBenchPlan(
        artifacts=artifacts,
        seeds=tuple(int(seed) for seed in seeds),
        runner=runner_cfg,
        config_path=str(config_path),
    )


def _config_hash(cfg: VisionRunnerConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def _resolve_cuda_device(device: str) -> Any | None:
    torch = require_torch()
    if not torch.cuda.is_available():
        return None

    raw = str(device).strip().lower()
    if raw == "cpu":
        return None
    if raw in {"", "auto"}:
        return torch.device("cuda")
    if not raw.startswith("cuda"):
        return None
    try:
        return torch.device(str(device))
    except (RuntimeError, ValueError):
        return torch.device("cuda")


def _reset_cuda_peak_memory(device: str) -> None:
    cuda_device = _resolve_cuda_device(device)
    if cuda_device is None:
        return
    torch = require_torch()
    torch.cuda.synchronize(device=cuda_device)
    torch.cuda.reset_peak_memory_stats(device=cuda_device)


def _capture_cuda_peak_memory(device: str) -> dict[str, Any] | None:
    cuda_device = _resolve_cuda_device(device)
    if cuda_device is None:
        return None
    torch = require_torch()
    torch.cuda.synchronize(device=cuda_device)
    max_allocated = int(torch.cuda.max_memory_allocated(device=cuda_device))
    max_reserved = int(torch.cuda.max_memory_reserved(device=cuda_device))
    return {
        "device": str(cuda_device),
        "max_allocated_bytes": max_allocated,
        "max_reserved_bytes": max_reserved,
        "max_allocated_mb": float(max_allocated / (1024.0 * 1024.0)),
        "max_reserved_mb": float(max_reserved / (1024.0 * 1024.0)),
    }


def _publish(
    bus: EventBus,
    *,
    topic: str,
    step: int,
    source: str,
    data: dict[str, Any],
) -> None:
    bus.publish(
        MonitorEvent(
            topic=EventTopic(topic),
            step=int(step),
            t=0.0,
            source=source,
            data=data,
        )
    )


def _build_run_summary(
    *,
    cfg: VisionRunnerConfig,
    ctx: RunContext,
    config_hash: str,
    wall_ms: float,
    task_summary: dict[str, Any] | None,
    cuda_memory: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dataset_meta = None
    if task_summary is not None:
        result = _as_mapping(task_summary.get("result"))
        dataset_meta = task_summary.get("dataset_meta")

    perf = _as_mapping(result.get("performance"))
    train_perf = _as_mapping(perf.get("train"))
    eval_perf = _as_mapping(perf.get("eval"))

    steps = _to_int(result.get("steps"))
    if steps is None:
        steps = _to_int(train_perf.get("steps"))
    if steps is None:
        steps = int(cfg.steps)

    train_steps = _to_int(train_perf.get("steps"))
    if train_steps is None:
        train_steps = steps

    train_samples = _to_int(train_perf.get("samples"))
    if train_samples is None:
        train_samples = int(train_steps * int(cfg.batch_size))

    eval_steps = _to_int(result.get("eval_steps"))
    if eval_steps is None:
        eval_steps = _to_int(eval_perf.get("steps"))
    if eval_steps is None:
        eval_steps = 0

    eval_samples = _to_int(result.get("eval_samples"))
    if eval_samples is None:
        eval_samples = _to_int(eval_perf.get("samples"))
    if eval_samples is None:
        eval_samples = 0

    loss_history = [float(v) for v in cast("list[Any]", result.get("loss_history", []))]
    acc_history = [float(v) for v in cast("list[Any]", result.get("accuracy_history", []))]
    final_loss = float(result.get("final_loss", loss_history[-1] if loss_history else 0.0))
    final_acc = float(result.get("final_accuracy", acc_history[-1] if acc_history else 0.0))
    eval_loss = _to_float(result.get("eval_loss"))
    if eval_loss is None:
        eval_loss = _to_float(eval_perf.get("loss"))
    if eval_loss is None:
        eval_loss = 0.0
    eval_acc = _to_float(result.get("eval_accuracy"))
    if eval_acc is None:
        eval_acc = _to_float(eval_perf.get("accuracy"))
    if eval_acc is None:
        eval_acc = 0.0

    best_loss = float(min(loss_history) if loss_history else final_loss)
    best_acc = float(max(acc_history) if acc_history else final_acc)
    best_val_acc = float(eval_acc if eval_steps > 0 else best_acc)

    elapsed_s = max(1e-9, wall_ms / 1000.0)
    fallback_steps_per_sec = 0.0 if train_steps <= 0 else float(train_steps / elapsed_s)
    fallback_samples_per_sec = 0.0 if train_samples <= 0 else float(train_samples / elapsed_s)
    fallback_ms_per_step = 0.0 if train_steps <= 0 else float(wall_ms / float(train_steps))

    train_steps_per_sec = _to_float(train_perf.get("steps_per_sec"))
    if train_steps_per_sec is None:
        train_steps_per_sec = fallback_steps_per_sec
    train_samples_per_sec = _to_float(train_perf.get("samples_per_sec"))
    if train_samples_per_sec is None:
        train_samples_per_sec = fallback_samples_per_sec
    train_ms_per_step = _to_float(train_perf.get("ms_per_step"))
    if train_ms_per_step is None:
        train_ms_per_step = fallback_ms_per_step
    train_wall_ms = _to_float(train_perf.get("wall_ms"))
    if train_wall_ms is None:
        train_wall_ms = float(wall_ms)

    eval_steps_per_sec = _to_float(eval_perf.get("steps_per_sec"))
    eval_samples_per_sec = _to_float(eval_perf.get("samples_per_sec"))
    eval_ms_per_step = _to_float(eval_perf.get("ms_per_step"))
    eval_wall_ms = _to_float(eval_perf.get("wall_ms"))
    if eval_wall_ms is None and eval_steps > 0 and eval_ms_per_step is not None:
        eval_wall_ms = float(eval_steps * eval_ms_per_step)
    if eval_wall_ms is None:
        eval_wall_ms = 0.0

    runtime_device = str(result.get("runtime_device", cfg.device))
    runtime_dtype = str(result.get("runtime_dtype", cfg.dtype))

    summary: dict[str, Any] = {
        "task": "vision_bench",
        "seed": int(cfg.seed),
        "device": runtime_device,
        "dtype": runtime_dtype,
        "dataset": str(cfg.dataset),
        "determinism": {
            "seed": int(cfg.seed),
            "deterministic": bool(cfg.deterministic),
            "benchmark": bool(cfg.benchmark),
            "warn_only": bool(cfg.warn_only),
        },
        "steps": steps,
        "config_hash": config_hash,
        "run_id": ctx.run_id,
        "run_dir": str(ctx.run_dir),
        "failed": error is not None,
        "error": error,
        "final_metrics": {
            "loss": final_loss,
            "accuracy": final_acc,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_acc,
        },
        "best_metrics": {
            "loss": best_loss,
            "accuracy": best_acc,
            "val_accuracy": best_val_acc,
            "test_accuracy": float(eval_acc if eval_steps > 0 else best_acc),
        },
        "throughput": {
            "steps_per_sec": train_steps_per_sec,
            "samples_per_sec": train_samples_per_sec,
            "ms_per_step": train_ms_per_step,
            "wall_ms": float(wall_ms),
            "train": {
                "steps": int(train_steps),
                "samples": int(train_samples),
                "steps_per_sec": train_steps_per_sec,
                "samples_per_sec": train_samples_per_sec,
                "ms_per_step": train_ms_per_step,
                "wall_ms": train_wall_ms,
            },
            "eval": {
                "steps": int(eval_steps),
                "samples": int(eval_samples),
                "steps_per_sec": float(eval_steps_per_sec or 0.0),
                "samples_per_sec": float(eval_samples_per_sec or 0.0),
                "ms_per_step": float(eval_ms_per_step or 0.0),
                "wall_ms": float(eval_wall_ms),
                "loss": eval_loss,
                "accuracy": eval_acc,
            },
        },
        "dataset_meta": dataset_meta,
        "performance": {
            "train": {
                "steps": int(train_steps),
                "samples": int(train_samples),
                "steps_per_sec": train_steps_per_sec,
                "samples_per_sec": train_samples_per_sec,
                "ms_per_step": train_ms_per_step,
                "wall_ms": train_wall_ms,
            },
            "eval": {
                "steps": int(eval_steps),
                "samples": int(eval_samples),
                "steps_per_sec": float(eval_steps_per_sec or 0.0),
                "samples_per_sec": float(eval_samples_per_sec or 0.0),
                "ms_per_step": float(eval_ms_per_step or 0.0),
                "wall_ms": float(eval_wall_ms),
                "loss": eval_loss,
                "accuracy": eval_acc,
            },
        },
    }
    if cuda_memory is not None:
        summary["gpu_memory"] = cuda_memory
    return summary


def _build_perf_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task": str(summary.get("task", "vision_bench")),
        "run_id": str(summary.get("run_id", "")),
        "seed": int(summary.get("seed", 0)),
        "device": str(summary.get("device", "")),
        "dtype": str(summary.get("dtype", "")),
        "steps": int(summary.get("steps", 0)),
        "throughput": _as_mapping(summary.get("throughput")),
    }
    gpu_memory = summary.get("gpu_memory")
    if isinstance(gpu_memory, dict):
        out["gpu_memory"] = gpu_memory
    return out


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _run_single_seed(cfg: VisionRunnerConfig, *, artifacts: Path) -> dict[str, Any]:
    task_name = "vision_classification"
    apply_determinism(
        DeterminismConfig(
            seed=int(cfg.seed),
            deterministic=bool(cfg.deterministic),
            benchmark=bool(cfg.benchmark),
            warn_only=bool(cfg.warn_only),
        )
    )
    ctx = create_run_dir(base_dir=artifacts, seed=cfg.seed, device=cfg.device)

    bus = EventBus()
    writer = ArtifactWriter(ctx.run_dir)
    layer_stats = VisionLayerStatsMonitor(interval_steps=1, enabled=True)
    confusion = ConfusionMatrixMonitor(enabled=True)
    samples = VisionSampleGridMonitor(max_samples=16, enabled=True)

    bus.subscribe_all(TopologyStatsMonitor(event_bus=bus, enabled=True))
    bus.subscribe_all(layer_stats)
    bus.subscribe_all(confusion)
    bus.subscribe_all(samples)
    if str(cfg.device).startswith("cuda"):
        bus.subscribe_all(CudaMetricsMonitor(enabled=True))
    bus.subscribe_all(VisionLayerStatsExporter(ctx.run_dir, layer_stats, enabled=True))
    bus.subscribe_all(ConfusionMatrixExporter(ctx.run_dir, confusion, enabled=True))
    bus.subscribe_all(VisionSampleGridExporter(ctx.run_dir, samples, enabled=True))
    bus.subscribe_all(writer)

    _publish(
        bus,
        topic="run_start",
        step=0,
        source=task_name,
        data={
            "run_meta": ctx.to_dict(),
            "config": asdict(cfg),
        },
    )

    _reset_cuda_peak_memory(str(cfg.device))
    t0 = time.perf_counter()
    run_error: str | None = None
    task_summary: dict[str, Any] | None = None
    runtime_device = str(cfg.device)
    try:
        task_summary = run_vision_classification(cfg, event_bus=bus)
        result_payload = _as_mapping(task_summary.get("result"))
        runtime_device = str(result_payload.get("runtime_device", runtime_device))
    except Exception as exc:
        run_error = str(exc)

    cuda_memory = _capture_cuda_peak_memory(runtime_device)
    wall_ms = (time.perf_counter() - t0) * 1_000.0
    if task_summary is not None:
        result = cast("dict[str, Any]", task_summary.get("result", {}))
        step = int(result.get("steps", cfg.steps))
        _publish(
            bus,
            topic="run_end",
            step=step,
            source=task_name,
            data={
                "converged": False,
                "failed": False,
                "steps": step,
                "final_loss": float(result.get("final_loss", 0.0)),
                "final_accuracy": float(result.get("final_accuracy", 0.0)),
                "wall_ms": round(wall_ms, 1),
            },
        )
    else:
        _publish(
            bus,
            topic="run_end",
            step=0,
            source=task_name,
            data={
                "converged": False,
                "failed": True,
                "error": run_error,
                "wall_ms": round(wall_ms, 1),
            },
        )
    writer.flush()

    summary = _build_run_summary(
        cfg=cfg,
        ctx=ctx,
        config_hash=_config_hash(cfg),
        wall_ms=wall_ms,
        task_summary=task_summary,
        cuda_memory=cuda_memory,
        error=run_error,
    )
    perf_summary = _build_perf_summary(summary)
    reports_dir = ctx.run_dir / "reports"
    _write_json(reports_dir / "benchmark_summary.json", summary)
    _write_json(reports_dir / "perf_summary.json", perf_summary)
    return summary


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    mean = float(sum(values) / len(values))
    variance = float(sum((value - mean) ** 2 for value in values) / len(values))
    return {
        "mean": mean,
        "std": float(math.sqrt(variance)),
    }


def _build_aggregate(
    *,
    seeds: tuple[int, ...],
    per_run: list[dict[str, Any]],
) -> dict[str, Any]:
    acc_values: list[float] = []
    throughput_values: list[float] = []
    throughput_sample_values: list[float] = []
    for run in per_run:
        if bool(run.get("failed", False)):
            continue
        best = _as_mapping(run.get("best_metrics"))
        throughput = _as_mapping(run.get("throughput"))
        acc_values.append(float(best.get("val_accuracy", 0.0)))
        throughput_values.append(float(throughput.get("steps_per_sec", 0.0)))
        throughput_sample_values.append(float(throughput.get("samples_per_sec", 0.0)))

    runs_out: list[dict[str, Any]] = []
    for run in per_run:
        best = _as_mapping(run.get("best_metrics"))
        throughput = _as_mapping(run.get("throughput"))
        runs_out.append(
            {
                "seed": int(run.get("seed", 0)),
                "run_id": str(run.get("run_id", "")),
                "run_dir": str(run.get("run_dir", "")),
                "failed": bool(run.get("failed", False)),
                "error": run.get("error"),
                "best_val_accuracy": float(best.get("val_accuracy", 0.0)),
                "steps_per_sec": float(throughput.get("steps_per_sec", 0.0)),
                "samples_per_sec": float(throughput.get("samples_per_sec", 0.0)),
            }
        )

    return {
        "task": "vision_bench",
        "mode": "multi_seed",
        "seeds": [int(seed) for seed in seeds],
        "determinism": {
            "deterministic": bool(per_run[0].get("determinism", {}).get("deterministic", True))
            if per_run else None,
            "benchmark": bool(per_run[0].get("determinism", {}).get("benchmark", False))
            if per_run else None,
            "warn_only": bool(per_run[0].get("determinism", {}).get("warn_only", True))
            if per_run else None,
            "seeds": [int(run.get("seed", 0)) for run in per_run],
        },
        "n_runs": len(per_run),
        "n_succeeded": int(sum(0 if bool(run.get("failed", False)) else 1 for run in per_run)),
        "n_failed": int(sum(1 if bool(run.get("failed", False)) else 0 for run in per_run)),
        "metrics": {
            "best_val_accuracy": _mean_std(acc_values),
            "throughput_steps_per_sec": _mean_std(throughput_values),
            "throughput_samples_per_sec": _mean_std(throughput_sample_values),
        },
        "runs": runs_out,
    }


def run_vision_benchmark(plan: VisionBenchPlan) -> dict[str, Any]:
    """Run one or many vision benchmark seeds and write report artifacts."""
    artifacts = Path(plan.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    per_run: list[dict[str, Any]] = []
    for seed in plan.seeds:
        cfg = replace(plan.runner, seed=int(seed))
        per_run.append(_run_single_seed(cfg, artifacts=artifacts))

    aggregate_path: Path | None = None
    aggregate: dict[str, Any] | None = None
    if len(plan.seeds) > 1 and per_run:
        aggregate = _build_aggregate(seeds=plan.seeds, per_run=per_run)
        anchor_run = Path(str(per_run[0]["run_dir"]))
        aggregate_path = anchor_run / "reports" / "multi_seed_aggregate.json"
        _write_json(aggregate_path, aggregate)

    return {
        "config_path": plan.config_path,
        "artifacts": str(artifacts.resolve()),
        "seeds": [int(seed) for seed in plan.seeds],
        "runs": per_run,
        "aggregate_path": str(aggregate_path) if aggregate_path is not None else None,
        "aggregate": aggregate,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neuroforge.runners.vision_bench",
        description="Phase 8 vision benchmark harness",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark JSON config",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional seed override (comma-separated list, e.g. 0,1,2)",
    )
    parser.add_argument(
        "--artifacts",
        default=None,
        help="Optional override for artifacts base directory",
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(str(args.config))
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)  # noqa: T201
        return 1

    try:
        seeds_override = (
            _parse_seed_list(str(args.seeds))
            if args.seeds is not None
            else None
        )
        plan = build_plan_from_config(
            config_path,
            seeds_override=seeds_override,
            artifacts_override=str(args.artifacts) if args.artifacts is not None else None,
        )
        result = run_vision_benchmark(plan)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201
        return 1

    print(json.dumps(result, indent=2), flush=True)  # noqa: T201
    any_failed = any(bool(run.get("failed", False)) for run in result["runs"])
    return 2 if any_failed else 0


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``python -m neuroforge.runners.vision_bench``."""
    sys.exit(_main(argv))


if __name__ == "__main__":
    main()
