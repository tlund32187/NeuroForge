"""Phase 8 Gate B validator for MNIST/NMNIST + Poker-DVS benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from neuroforge.runners.vision_bench import (
    VisionBenchPlan,
    build_plan_from_config,
    run_vision_benchmark,
)

__all__ = [
    "evaluate_gate_b",
    "run_gate_b_validation",
]

_DATASET_ALIAS: dict[str, str] = {
    "mnist": "mnist",
    "fashion_mnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
    "nmnist": "nmnist",
    "n_mnist": "nmnist",
    "n-mnist": "nmnist",
    "pokerdvs": "pokerdvs",
    "poker_dvs": "pokerdvs",
    "poker-dvs": "pokerdvs",
}


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


def _derive_seed_sweep(*, base_seed: int, n_seeds: int) -> tuple[int, ...]:
    if n_seeds <= 0:
        msg = "n_seeds must be > 0"
        raise ValueError(msg)
    return tuple(int(base_seed + idx) for idx in range(int(n_seeds)))


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


def _accuracy_fraction(value: Any) -> float | None:
    num = _to_float(value)
    if num is None:
        return None
    if num < 0.0:
        return None
    if num > 1.0:
        if num <= 100.0:
            return num / 100.0
        return None
    return num


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def _normalize_dataset_name(value: Any) -> str:
    raw = str(value).strip().lower()
    if not raw:
        return ""
    return _DATASET_ALIAS.get(raw, raw.replace("-", "_"))


def _run_dataset_name(run: dict[str, Any], *, fallback: str) -> str:
    run_dataset = _normalize_dataset_name(run.get("dataset"))
    if run_dataset:
        return run_dataset
    return _normalize_dataset_name(fallback)


def _has_confusion_artifacts(run: dict[str, Any]) -> bool:
    run_dir = str(run.get("run_dir", "")).strip()
    if not run_dir:
        return False
    metrics_dir = Path(run_dir) / "vision" / "metrics"
    return any(
        (metrics_dir / name).is_file()
        for name in (
            "confusion_matrix.npy",
            "confusion_matrix.json",
            "per_class_accuracy.json",
        )
    )


def _extract_accuracy_fraction(run: dict[str, Any], *, split: str) -> float | None:
    best = _as_mapping(run.get("best_metrics"))
    final = _as_mapping(run.get("final_metrics"))

    if split == "val":
        keys = ("val_accuracy", "accuracy")
    elif split == "test":
        keys = ("test_accuracy", "val_accuracy", "accuracy")
    else:
        msg = f"Unsupported split: {split!r}"
        raise ValueError(msg)

    for key in keys:
        maybe = _accuracy_fraction(best.get(key))
        if maybe is not None:
            return maybe
    for key in keys:
        maybe = _accuracy_fraction(final.get(key))
        if maybe is not None:
            return maybe
    return None


def _extract_steps(run: dict[str, Any]) -> int | None:
    steps = _to_int(run.get("steps"))
    if steps is not None and steps >= 0:
        return steps
    final = _as_mapping(run.get("final_metrics"))
    final_steps = _to_int(final.get("steps"))
    if final_steps is not None and final_steps >= 0:
        return final_steps
    return None


def _determinism_requested(plan: VisionBenchPlan) -> dict[str, Any]:
    return {
        "deterministic": bool(plan.runner.deterministic),
        "benchmark": bool(plan.runner.benchmark),
        "warn_only": bool(plan.runner.warn_only),
    }


def _evaluate_gating_dataset(
    *,
    result: dict[str, Any],
    plan: VisionBenchPlan,
    expected_dataset: str,
    min_accuracy_pct: float,
    max_budget_steps: int,
    gate_label: str,
) -> dict[str, Any]:
    threshold_fraction = float(min_accuracy_pct) / 100.0
    runs_raw = result.get("runs", [])
    if not isinstance(runs_raw, list):
        runs_raw = []
    runs = cast("list[dict[str, Any]]", runs_raw)

    expected_norm = _normalize_dataset_name(expected_dataset)
    plan_dataset = _normalize_dataset_name(plan.runner.dataset)
    dataset_ok = plan_dataset == expected_norm

    evaluated_runs: list[dict[str, Any]] = []
    passed_runs = 0
    for run in runs:
        seed = _to_int(run.get("seed"))
        run_id = str(run.get("run_id", ""))
        run_dir = str(run.get("run_dir", ""))
        run_failed = bool(run.get("failed", False))
        steps = _extract_steps(run)
        within_budget = steps is not None and int(steps) <= int(max_budget_steps)
        val_acc = _extract_accuracy_fraction(run, split="val")
        test_acc = _extract_accuracy_fraction(run, split="test")
        run_dataset = _run_dataset_name(run, fallback=plan.runner.dataset)
        dataset_match = run_dataset == expected_norm

        reasons: list[str] = []
        if run_failed:
            reasons.append("run_failed")
        if not dataset_match:
            reasons.append("dataset_mismatch")
        if not within_budget:
            reasons.append("budget_exceeded_or_missing_steps")
        if val_acc is None:
            reasons.append("missing_val_accuracy")
        elif val_acc < threshold_fraction:
            reasons.append("val_accuracy_below_threshold")
        if test_acc is None:
            reasons.append("missing_test_accuracy")
        elif test_acc < threshold_fraction:
            reasons.append("test_accuracy_below_threshold")

        run_pass = not reasons
        if run_pass:
            passed_runs += 1

        evaluated_runs.append({
            "seed": seed,
            "run_id": run_id,
            "run_dir": run_dir,
            "dataset": run_dataset,
            "dataset_match": dataset_match,
            "failed": run_failed,
            "steps": steps,
            "within_budget": within_budget,
            "best_val_accuracy_pct": None if val_acc is None else float(val_acc * 100.0),
            "best_test_accuracy_pct": None if test_acc is None else float(test_acc * 100.0),
            "pass": run_pass,
            "reasons": reasons,
        })

    criteria = {
        "dataset": expected_norm,
        "min_best_val_accuracy_pct": float(min_accuracy_pct),
        "min_best_test_accuracy_pct": float(min_accuracy_pct),
        "max_budget_steps": int(max_budget_steps),
    }
    gate_pass = (
        dataset_ok
        and len(evaluated_runs) > 0
        and passed_runs == len(evaluated_runs)
    )
    aggregate = _as_mapping(result.get("aggregate"))
    aggregate_metrics = _as_mapping(aggregate.get("metrics"))
    summary: dict[str, Any] = {
        "total_runs": len(evaluated_runs),
        "passed_runs": int(passed_runs),
        "failed_runs": int(len(evaluated_runs) - passed_runs),
        "gate_pass": gate_pass,
    }
    if not dataset_ok:
        summary["dataset_reason"] = (
            f"{gate_label} requires runner.dataset={expected_norm!r}"
        )

    return {
        "gate": gate_label,
        "status": "PASS" if gate_pass else "FAIL",
        "dataset": expected_norm,
        "dataset_ok": dataset_ok,
        "criteria": criteria,
        "runs": evaluated_runs,
        "aggregate_metrics": aggregate_metrics,
        "summary": summary,
    }


def _evaluate_non_gating_poker(
    *,
    result: dict[str, Any],
    plan: VisionBenchPlan,
) -> dict[str, Any]:
    runs_raw = result.get("runs", [])
    if not isinstance(runs_raw, list):
        runs_raw = []
    runs = cast("list[dict[str, Any]]", runs_raw)

    evaluated_runs: list[dict[str, Any]] = []
    succeeded_runs = 0
    failed_runs = 0
    runs_with_confusion = 0
    for run in runs:
        seed = _to_int(run.get("seed"))
        run_id = str(run.get("run_id", ""))
        run_dir = str(run.get("run_dir", ""))
        run_failed = bool(run.get("failed", False))
        steps = _extract_steps(run)
        val_acc = _extract_accuracy_fraction(run, split="val")
        test_acc = _extract_accuracy_fraction(run, split="test")
        run_dataset = _run_dataset_name(run, fallback=plan.runner.dataset)
        confusion_present = _has_confusion_artifacts(run)

        if run_failed:
            failed_runs += 1
        else:
            succeeded_runs += 1
        if confusion_present:
            runs_with_confusion += 1

        evaluated_runs.append({
            "seed": seed,
            "run_id": run_id,
            "run_dir": run_dir,
            "dataset": run_dataset,
            "failed": run_failed,
            "steps": steps,
            "best_val_accuracy_pct": None if val_acc is None else float(val_acc * 100.0),
            "best_test_accuracy_pct": None if test_acc is None else float(test_acc * 100.0),
            "confusion_matrix_present": confusion_present,
        })

    aggregate = _as_mapping(result.get("aggregate"))
    aggregate_metrics = _as_mapping(aggregate.get("metrics"))
    status = "OK"
    if failed_runs > 0 or runs_with_confusion < len(evaluated_runs):
        status = "WARN"

    return {
        "gate": "POKER_DVS",
        "status": status,
        "non_gating": True,
        "dataset": "pokerdvs",
        "criteria": {
            "non_gating": True,
            "notes": (
                "Poker-DVS is a benchmark-only section. "
                "It never changes overall PASS/FAIL."
            ),
        },
        "runs": evaluated_runs,
        "aggregate_metrics": aggregate_metrics,
        "summary": {
            "total_runs": len(evaluated_runs),
            "succeeded_runs": int(succeeded_runs),
            "failed_runs": int(failed_runs),
            "runs_with_confusion": int(runs_with_confusion),
            "runs_missing_confusion": int(len(evaluated_runs) - runs_with_confusion),
            "gate_impact": "none",
        },
    }


def evaluate_gate_b(
    *,
    result: dict[str, Any],
    plan: VisionBenchPlan,
    min_accuracy_pct: float,
    max_budget_steps: int,
) -> dict[str, Any]:
    """Evaluate Gate B1 (MNIST) criteria against benchmark run summaries."""
    section = _evaluate_gating_dataset(
        result=result,
        plan=plan,
        expected_dataset="mnist",
        min_accuracy_pct=float(min_accuracy_pct),
        max_budget_steps=int(max_budget_steps),
        gate_label="B1_MNIST",
    )
    gate_pass = bool(_as_mapping(section.get("summary")).get("gate_pass", False))
    return {
        "gate": "phase8_gate_b",
        "status": "PASS" if gate_pass else "FAIL",
        "criteria": _as_mapping(section.get("criteria")),
        "config_path": plan.config_path,
        "artifacts_dir": plan.artifacts,
        "dataset": _normalize_dataset_name(plan.runner.dataset),
        "dataset_ok": bool(section.get("dataset_ok", False)),
        "seeds": [int(seed) for seed in plan.seeds],
        "determinism": _determinism_requested(plan),
        "runs": section.get("runs", []),
        "aggregate_metrics": section.get("aggregate_metrics", {}),
        "summary": _as_mapping(section.get("summary")),
    }


def _resolve_report_dir(*, result: dict[str, Any], plan: VisionBenchPlan) -> Path:
    aggregate_path = result.get("aggregate_path")
    if aggregate_path:
        return Path(str(aggregate_path)).resolve().parent

    runs_raw = result.get("runs", [])
    if isinstance(runs_raw, list) and runs_raw:
        first = _as_mapping(runs_raw[0])
        run_dir = first.get("run_dir")
        if run_dir:
            return Path(str(run_dir)).resolve() / "reports"
    return Path(plan.artifacts).resolve() / "gate_reports"


def run_gate_b_validation(
    *,
    config_path: Path,
    seeds_override: tuple[int, ...] | None,
    n_seeds: int,
    min_accuracy_pct: float,
    max_budget_steps: int | None,
    artifacts_override: str | None,
    min_accuracy_pct_b2: float | None = None,
) -> tuple[bool, Path, dict[str, Any]]:
    """Run B1/B2 benchmark sweeps + Poker benchmark, then persist gate_report.json."""
    base_plan = build_plan_from_config(
        config_path,
        seeds_override=None,
        artifacts_override=artifacts_override,
    )
    seeds = (
        seeds_override
        if seeds_override is not None
        else _derive_seed_sweep(base_seed=int(base_plan.runner.seed), n_seeds=int(n_seeds))
    )
    plan = replace(base_plan, seeds=seeds)
    budget_steps = (
        int(max_budget_steps)
        if max_budget_steps is not None
        else int(plan.runner.steps)
    )
    b2_threshold = (
        float(min_accuracy_pct_b2)
        if min_accuracy_pct_b2 is not None
        else float(min_accuracy_pct)
    )

    plan_b1 = replace(plan, runner=replace(plan.runner, dataset="mnist"))
    plan_b2 = replace(plan, runner=replace(plan.runner, dataset="nmnist"))
    plan_poker = replace(plan, runner=replace(plan.runner, dataset="pokerdvs"))

    result_b1 = run_vision_benchmark(plan_b1)
    result_b2 = run_vision_benchmark(plan_b2)
    result_poker = run_vision_benchmark(plan_poker)

    section_b1 = _evaluate_gating_dataset(
        result=result_b1,
        plan=plan_b1,
        expected_dataset="mnist",
        min_accuracy_pct=float(min_accuracy_pct),
        max_budget_steps=budget_steps,
        gate_label="B1_MNIST",
    )
    section_b2 = _evaluate_gating_dataset(
        result=result_b2,
        plan=plan_b2,
        expected_dataset="nmnist",
        min_accuracy_pct=b2_threshold,
        max_budget_steps=budget_steps,
        gate_label="B2_NMNIST",
    )
    section_poker = _evaluate_non_gating_poker(
        result=result_poker,
        plan=plan_poker,
    )

    b1_pass = bool(_as_mapping(section_b1.get("summary")).get("gate_pass", False))
    b2_pass = bool(_as_mapping(section_b2.get("summary")).get("gate_pass", False))
    gate_pass = bool(b1_pass and b2_pass)

    report: dict[str, Any] = {
        "gate": "phase8_gate_b",
        "status": "PASS" if gate_pass else "FAIL",
        "config_path": plan.config_path,
        "artifacts_dir": plan.artifacts,
        "seeds": [int(seed) for seed in plan.seeds],
        "determinism": _determinism_requested(plan),
        "criteria": {
            "b1": _as_mapping(section_b1.get("criteria")),
            "b2": _as_mapping(section_b2.get("criteria")),
            "poker_dvs": _as_mapping(section_poker.get("criteria")),
        },
        "b1": section_b1,
        "b2": section_b2,
        "poker_dvs": section_poker,
        "summary": {
            "b1_pass": b1_pass,
            "b2_pass": b2_pass,
            "gate_pass": gate_pass,
            "poker_runs": _as_mapping(section_poker.get("summary")).get("total_runs", 0),
            "poker_failed_runs": _as_mapping(section_poker.get("summary")).get("failed_runs", 0),
        },
        # Backward-compatible mirrors of old single-section fields (B1/MNIST).
        "dataset": "mnist",
        "dataset_ok": bool(section_b1.get("dataset_ok", False)),
        "runs": section_b1.get("runs", []),
        "aggregate_metrics": section_b1.get("aggregate_metrics", {}),
    }

    report_dir = _resolve_report_dir(result=result_b1, plan=plan_b1)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "gate_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return gate_pass, report_path, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neuroforge.runners.vision_gate_b",
        description="Phase 8 Gate B validator (B1=MNIST, B2=NMNIST, Poker-DVS benchmark)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to vision benchmark JSON config",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional explicit seeds override (comma-separated, e.g. 0,1,2)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Seed count when --seeds is not provided (default: 3)",
    )
    parser.add_argument(
        "--min-accuracy-pct",
        type=float,
        default=99.0,
        help="Gate B1 threshold for MNIST best val/test accuracy percentage (default: 99.0)",
    )
    parser.add_argument(
        "--min-accuracy-pct-b2",
        type=float,
        default=None,
        help=(
            "Optional Gate B2 threshold for NMNIST best val/test accuracy percentage. "
            "Defaults to --min-accuracy-pct."
        ),
    )
    parser.add_argument(
        "--max-budget-steps",
        type=int,
        default=None,
        help="Maximum training step budget per run (defaults to runner.steps in config)",
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
        gate_pass, report_path, report = run_gate_b_validation(
            config_path=config_path,
            seeds_override=seeds_override,
            n_seeds=int(args.n_seeds),
            min_accuracy_pct=float(args.min_accuracy_pct),
            max_budget_steps=args.max_budget_steps,
            artifacts_override=str(args.artifacts) if args.artifacts is not None else None,
            min_accuracy_pct_b2=args.min_accuracy_pct_b2,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201
        return 1

    summary = _as_mapping(report.get("summary"))
    b1 = _as_mapping(report.get("b1"))
    b2 = _as_mapping(report.get("b2"))
    poker = _as_mapping(report.get("poker_dvs"))
    b1_summary = _as_mapping(b1.get("summary"))
    b2_summary = _as_mapping(b2.get("summary"))
    poker_summary = _as_mapping(poker.get("summary"))
    b1_criteria = _as_mapping(b1.get("criteria"))
    b2_criteria = _as_mapping(b2.get("criteria"))
    print(f"Gate B: {'PASS' if gate_pass else 'FAIL'}")  # noqa: T201
    print(  # noqa: T201
        "B1 (MNIST): "
        f"{b1.get('status', 'FAIL')} "
        f"val/test>={b1_criteria.get('min_best_val_accuracy_pct')}% "
        f"budget<={b1_criteria.get('max_budget_steps')} steps "
        f"runs={b1_summary.get('passed_runs', 0)}/{b1_summary.get('total_runs', 0)}"
    )
    print(  # noqa: T201
        "B2 (NMNIST): "
        f"{b2.get('status', 'FAIL')} "
        f"val/test>={b2_criteria.get('min_best_val_accuracy_pct')}% "
        f"budget<={b2_criteria.get('max_budget_steps')} steps "
        f"runs={b2_summary.get('passed_runs', 0)}/{b2_summary.get('total_runs', 0)}"
    )
    print(  # noqa: T201
        "Poker-DVS (non-gating): "
        f"status={poker.get('status', 'WARN')} "
        f"succeeded={poker_summary.get('succeeded_runs', 0)}"
        f"/{poker_summary.get('total_runs', 0)} "
        f"confusion={poker_summary.get('runs_with_confusion', 0)}"
        f"/{poker_summary.get('total_runs', 0)}"
    )
    print(f"Gate summary: B1={summary.get('b1_pass')} B2={summary.get('b2_pass')}")  # noqa: T201
    print(f"Report: {report_path}")  # noqa: T201
    return 0 if gate_pass else 2


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``python -m neuroforge.runners.vision_gate_b``."""
    sys.exit(_main(argv))


if __name__ == "__main__":
    main()
