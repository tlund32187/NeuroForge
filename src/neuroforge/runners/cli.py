"""NeuroForge CLI — ``neuroforge run | ui | list-runs``.

Entrypoint registered in ``pyproject.toml``::

    [project.scripts]
    neuroforge = "neuroforge.runners.cli:main"

Also works as ``python -m neuroforge.runners.cli``.
"""
# pyright: basic

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neuroforge.api.version import __version__

# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a training task and write M0 artifacts."""
    from neuroforge.contracts.monitors import EventTopic, MonitorEvent
    from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
    from neuroforge.monitors.artifact_writer import ArtifactWriter
    from neuroforge.monitors.bus import EventBus
    from neuroforge.monitors.cuda_monitor import CudaMetricsMonitor
    from neuroforge.monitors.stability_monitor import StabilityConfig, StabilityMonitor
    from neuroforge.monitors.trial_stats_monitor import TrialStatsMonitor
    from neuroforge.runners.run_context import create_run_dir

    task_name: str = args.task
    seed: int = args.seed
    device: str = args.device

    apply_determinism(
        DeterminismConfig(
            seed=seed,
            deterministic=bool(args.deterministic),
            benchmark=bool(args.benchmark),
            warn_only=bool(args.warn_only),
        )
    )

    # ── Create run directory and set up event bus ───────────────────
    ctx = create_run_dir(base_dir=args.artifacts, seed=seed, device=device)
    bus = EventBus()
    writer = ArtifactWriter(ctx.run_dir)
    pre_write_monitors: list[Any] = []

    # Trial-level enrichment (rates/sparsity/convergence) is opt-in.
    if bool(args.trial_stats):
        pre_write_monitors.append(
            TrialStatsMonitor(enabled=True),
        )

    if bool(args.stability):
        pre_write_monitors.append(
            StabilityMonitor(
                StabilityConfig(
                    enabled=True,
                    check_every_n_trials=max(1, int(args.stability_every)),
                    fail_fast=bool(args.fail_fast),
                )
            ),
        )

    # CUDA metrics should be injected before ArtifactWriter serialises.
    if device.startswith("cuda"):
        pre_write_monitors.append(CudaMetricsMonitor(enabled=True))

    for monitor in pre_write_monitors:
        bus.subscribe_all(monitor)
    bus.subscribe_all(writer)

    def _log(msg: str) -> None:
        ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
        line = f"[{ts}] {msg}"
        writer.add_log_line(line)
        print(line, flush=True)  # noqa: T201

    def _emit(topic: str, data: dict[str, Any]) -> None:
        bus.publish(MonitorEvent(
            topic=EventTopic(topic),
            step=0,
            t=0.0,
            source=task_name,
            data=data,
        ))

    def _config_dict(config: Any) -> dict[str, Any]:
        if hasattr(config, "__dataclass_fields__"):
            return asdict(config)
        return dict(vars(config))

    _log(f"run_id   : {ctx.run_id}")
    _log(f"task     : {task_name}")
    _log(f"seed     : {seed}")
    _log(f"device   : {device}")
    _log(f"deterministic: {bool(args.deterministic)}")
    _log(f"benchmark    : {bool(args.benchmark)}")
    _log(f"warn_only    : {bool(args.warn_only)}")
    _log(f"trial_stats  : {bool(args.trial_stats)}")
    _log(f"stability    : {bool(args.stability)}")
    _log(f"stability_N  : {max(1, int(args.stability_every))}")
    _log(f"fail_fast    : {bool(args.fail_fast)}")

    if task_name == "multi_gate":
        from neuroforge.tasks.multi_gate import MultiGateConfig, MultiGateTask

        cfg_multi = MultiGateConfig(seed=seed, max_epochs=args.max_epochs, device=device)
        _log(f"epochs   : {cfg_multi.max_epochs}")

        _emit("run_start", {
            "run_meta": ctx.to_dict(),
            "config": _config_dict(cfg_multi),
        })

        bus_any: Any = bus
        task_multi = MultiGateTask(cfg_multi, event_bus=bus_any)

        _log("Training started\u2026")
        t0 = time.perf_counter()
        try:
            result_multi = task_multi.run()
        except RuntimeError as exc:
            wall_ms = (time.perf_counter() - t0) * 1_000
            _log(f"Run aborted: {exc}")
            _log(f"Wall time: {wall_ms:,.0f} ms")
            _log(f"Artifacts: {ctx.run_dir}")
            _emit("run_end", {
                "converged": False,
                "failed": True,
                "error": str(exc),
                "wall_ms": round(wall_ms, 1),
            })
            writer.flush()
            return 2
        wall_ms = (time.perf_counter() - t0) * 1_000

        _log(f"Converged: {result_multi.converged}")
        _log(f"Trials   : {result_multi.trials}")
        _log(f"Epochs   : {result_multi.epochs}")
        for g, ok in result_multi.per_gate_converged.items():
            _log(f"  {g:5s}  : {'OK' if ok else 'FAIL'}")
        _log(f"Wall time: {wall_ms:,.0f} ms")
        _log(f"Artifacts: {ctx.run_dir}")

        _emit("run_end", {
            "converged": result_multi.converged,
            "trials": result_multi.trials,
            "epochs": result_multi.epochs,
            "wall_ms": round(wall_ms, 1),
        })
        return 0

    if task_name == "logic_gate":
        from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask

        gate: str = args.gate.upper()
        cfg_single = LogicGateConfig(
            gate=gate, seed=seed, max_trials=args.max_trials, device=device,
        )
        _log(f"gate     : {gate}")
        _log(f"trials   : {cfg_single.max_trials}")

        _emit("run_start", {
            "run_meta": ctx.to_dict(),
            "config": _config_dict(cfg_single),
        })

        bus_any2: Any = bus
        task_single = LogicGateTask(cfg_single, event_bus=bus_any2)

        _log("Training started\u2026")
        t0 = time.perf_counter()
        try:
            result_single = task_single.run()
        except RuntimeError as exc:
            wall_ms = (time.perf_counter() - t0) * 1_000
            _log(f"Run aborted: {exc}")
            _log(f"Wall time: {wall_ms:,.0f} ms")
            _log(f"Artifacts: {ctx.run_dir}")
            _emit("run_end", {
                "converged": False,
                "failed": True,
                "error": str(exc),
                "wall_ms": round(wall_ms, 1),
            })
            writer.flush()
            return 2
        wall_ms = (time.perf_counter() - t0) * 1_000

        _log(f"Converged: {result_single.converged}")
        _log(f"Trials   : {result_single.trials}")
        _log(f"Wall time: {wall_ms:,.0f} ms")
        _log(f"Artifacts: {ctx.run_dir}")

        _emit("run_end", {
            "converged": result_single.converged,
            "trials": result_single.trials,
            "wall_ms": round(wall_ms, 1),
        })
        return 0

    print(f"Unknown task: {task_name!r}", file=sys.stderr)  # noqa: T201
    return 1


# ---------------------------------------------------------------------------
# Subcommand: stability
# ---------------------------------------------------------------------------


def _parse_seeds(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
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
    return out


def _cmd_stability(args: argparse.Namespace) -> int:
    """Execute a multi-seed stability harness and print JSON summary."""
    from neuroforge.runners.stability_harness import run_multi_seed

    try:
        seeds = _parse_seeds(args.seeds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201
        return 1

    hidden = int(args.hidden) if args.hidden is not None else None
    base_config: dict[str, Any] = {
        "device": args.device,
        "trial_stats": True,
        "stability_every": max(1, int(args.stability_every)),
        "fail_fast": bool(args.fail_fast),
    }
    if args.max_epochs is not None:
        if args.task == "multi_gate":
            base_config["max_epochs"] = int(args.max_epochs)
        else:
            base_config["max_trials"] = int(args.max_epochs) * 24
    if hidden is not None:
        base_config["n_hidden"] = hidden
        base_config["n_inhibitory"] = min(max(0, hidden // 4), hidden)
    if args.task == "logic_gate":
        base_config["gate"] = str(args.gate).upper()

    summary = run_multi_seed(
        task_name=str(args.task),
        seeds=seeds,
        base_config=base_config,
        deterministic=bool(args.deterministic),
    )
    print(json.dumps(summary, indent=2), flush=True)  # noqa: T201
    return 0


# ---------------------------------------------------------------------------
# Subcommand: ui
# ---------------------------------------------------------------------------


def _cmd_ui(args: argparse.Namespace) -> int:
    """Launch the dashboard UI."""
    from neuroforge.dashboard import run_dashboard

    mode = "live" if args.live else "default"
    print(  # noqa: T201
        f"Starting dashboard ({mode}) on http://{args.host}:{args.port}",
        flush=True,
    )
    run_dashboard(host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: list-runs
# ---------------------------------------------------------------------------


def _cmd_list_runs(args: argparse.Namespace) -> int:
    """List all run directories under the artifacts folder."""
    base = Path(args.artifacts)
    if not base.exists():
        print(f"No artifacts directory: {base}", file=sys.stderr)  # noqa: T201
        return 1

    runs = sorted(
        (d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")),
        key=lambda p: p.name,
    )
    if not runs:
        print("No runs found.", file=sys.stderr)  # noqa: T201
        return 0

    for run_dir in runs:
        meta_path = run_dir / "run_meta.json"
        suffix = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                suffix = f"  seed={meta.get('seed', '?')}  device={meta.get('device', '?')}"
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"  {run_dir.name}{suffix}")  # noqa: T201

    print(f"\n{len(runs)} run(s) in {base.resolve()}")  # noqa: T201
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroforge",
        description="NeuroForge — spiking neural network toolkit",
    )
    parser.add_argument(
        "--version", action="version", version=f"neuroforge {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    # ── neuroforge run ──────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run a training task and write artifacts")
    p_run.add_argument(
        "--task",
        choices=["logic_gate", "multi_gate"],
        default="multi_gate",
        help="Task to run (default: multi_gate)",
    )
    p_run.add_argument("--gate", default="XOR", help="Gate for logic_gate task (default: XOR)")
    p_run.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    det_group = p_run.add_mutually_exclusive_group()
    det_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Enable deterministic torch algorithms (default: on)",
    )
    det_group.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic torch algorithms",
    )
    p_run.set_defaults(deterministic=True)
    p_run.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable backend benchmark mode (default: off)",
    )
    warn_group = p_run.add_mutually_exclusive_group()
    warn_group.add_argument(
        "--warn-only",
        dest="warn_only",
        action="store_true",
        help="Warn on non-deterministic ops (default)",
    )
    warn_group.add_argument(
        "--strict-determinism",
        dest="warn_only",
        action="store_false",
        help="Raise on non-deterministic ops when deterministic mode is enabled",
    )
    p_run.set_defaults(warn_only=True)
    trial_stats_group = p_run.add_mutually_exclusive_group()
    trial_stats_group.add_argument(
        "--trial-stats",
        dest="trial_stats",
        action="store_true",
        help="Enable TrialStatsMonitor enrichment (default: on)",
    )
    trial_stats_group.add_argument(
        "--no-trial-stats",
        dest="trial_stats",
        action="store_false",
        help="Disable TrialStatsMonitor enrichment",
    )
    p_run.set_defaults(trial_stats=True)
    stability_group = p_run.add_mutually_exclusive_group()
    stability_group.add_argument(
        "--stability",
        dest="stability",
        action="store_true",
        help="Enable StabilityMonitor flags (default: on)",
    )
    stability_group.add_argument(
        "--no-stability",
        dest="stability",
        action="store_false",
        help="Disable StabilityMonitor flags",
    )
    p_run.set_defaults(stability=True)
    p_run.add_argument(
        "--stability-every",
        type=int,
        default=5,
        metavar="N",
        help="Evaluate stability flags every N trials (default: 5)",
    )
    p_run.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Abort the run on critical stability flags (default: off)",
    )
    p_run.add_argument(
        "--max-epochs", type=int, default=1500,
        help="Max epochs for multi_gate (default: 1500)",
    )
    p_run.add_argument(
        "--max-trials", type=int, default=5000,
        help="Max trials for logic_gate (default: 5000)",
    )
    p_run.add_argument(
        "--artifacts", default="artifacts",
        help="Base dir for run artifacts (default: artifacts)",
    )
    p_run.add_argument(
        "--device", default="auto",
        help="Torch device: auto, cpu, or cuda (default: auto)",
    )

    # ── neuroforge ui ───────────────────────────────────────────────
    p_stability = sub.add_parser("stability", help="Run multi-seed stability harness")
    p_stability.add_argument(
        "--task",
        choices=["logic_gate", "multi_gate"],
        default="multi_gate",
        help="Task to run across seeds (default: multi_gate)",
    )
    p_stability.add_argument(
        "--gate",
        default="XOR",
        help="Gate for logic_gate harness runs (default: XOR)",
    )
    p_stability.add_argument(
        "--seeds",
        default="1,2,3,4,5",
        help="Comma-separated seed list (default: 1,2,3,4,5)",
    )
    p_stability.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs for multi_gate (or scaled max trials for logic_gate)",
    )
    p_stability.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Override hidden neuron count (default: task config)",
    )
    det_stab_group = p_stability.add_mutually_exclusive_group()
    det_stab_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Enable deterministic mode (default: on)",
    )
    det_stab_group.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic mode",
    )
    p_stability.set_defaults(deterministic=True)
    p_stability.add_argument(
        "--stability-every",
        type=int,
        default=5,
        metavar="N",
        help="Evaluate stability flags every N trials (default: 5)",
    )
    p_stability.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Abort each seed run on critical stability flags",
    )
    p_stability.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, or cuda (default: auto)",
    )

    p_ui = sub.add_parser("ui", help="Launch the dashboard web server")
    p_ui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_ui.add_argument("--port", type=int, default=8050, help="Bind port (default: 8050)")
    p_ui.add_argument("--live", action="store_true", help="Enable live training mode")

    # ── neuroforge list-runs ────────────────────────────────────────
    p_list = sub.add_parser("list-runs", help="List artifacts runs")
    p_list.add_argument(
        "--artifacts", default="artifacts",
        help="Base dir for run artifacts (default: artifacts)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "run": _cmd_run,
        "stability": _cmd_stability,
        "ui": _cmd_ui,
        "list-runs": _cmd_list_runs,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
