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
from typing import Any, cast

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
    from neuroforge.monitors.topology_stats_monitor import TopologyStatsMonitor
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
    pre_write_monitors: list[Any] = [
        TopologyStatsMonitor(event_bus=bus, enabled=True),
    ]

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
# Subcommand: bench
# ---------------------------------------------------------------------------


def _cmd_bench(args: argparse.Namespace) -> int:
    """Run a synthetic benchmark and optionally write artifacts."""
    from neuroforge.contracts.monitors import EventTopic, MonitorEvent
    from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
    from neuroforge.monitors.artifact_writer import ArtifactWriter
    from neuroforge.monitors.bus import EventBus
    from neuroforge.monitors.cuda_monitor import CudaMetricsMonitor
    from neuroforge.monitors.resource_monitor import ResourceMonitor
    from neuroforge.monitors.topology_stats_monitor import TopologyStatsMonitor
    from neuroforge.runners.bench import BenchRunConfig, run_bench
    from neuroforge.runners.run_context import create_run_dir

    task_name = "bench"
    apply_determinism(
        DeterminismConfig(
            seed=int(args.seed),
            deterministic=False,
            benchmark=True,
            warn_only=True,
        )
    )

    sync_cuda_timing = (
        bool(args.sync_cuda_timing)
        if args.sync_cuda_timing is not None
        else str(args.device).startswith("cuda")
    )
    cfg = BenchRunConfig(
        device=str(args.device),
        dtype=str(args.dtype),
        seed=int(args.seed),
        n_input=int(args.n_input),
        n_hidden=int(args.n_hidden),
        n_output=int(args.n_output),
        topology=str(args.topology),
        synapse_model=str(args.synapse_model),
        p_connect=float(args.p_connect),
        fanout=int(args.fanout),
        fanin=int(args.fanin),
        block_pre=int(args.block_pre),
        block_post=int(args.block_post),
        p_block=float(args.p_block),
        max_delay=int(args.max_delay),
        steps=int(args.steps),
        warmup=int(args.warmup),
        amplitude=float(args.amplitude),
        sync_cuda_timing=bool(sync_cuda_timing),
    )

    bus = EventBus()
    ctx = None
    writer: ArtifactWriter | None = None

    pre_write_monitors: list[Any] = [
        TopologyStatsMonitor(event_bus=bus, enabled=True),
    ]
    if bool(args.resources):
        pre_write_monitors.append(
            ResourceMonitor(
                enabled=True,
                every_n_steps=1,
                include_gpu=str(cfg.device).startswith("cuda"),
            ),
        )
    if str(cfg.device).startswith("cuda"):
        pre_write_monitors.append(CudaMetricsMonitor(enabled=True))

    for monitor in pre_write_monitors:
        bus.subscribe_all(monitor)

    if bool(args.write_artifacts):
        ctx = create_run_dir(
            base_dir=args.artifacts,
            seed=cfg.seed,
            device=cfg.device,
        )
        writer = ArtifactWriter(
            ctx.run_dir,
            include_resource_fields=bool(args.resources),
        )
        bus.subscribe_all(writer)

    def _log(msg: str) -> None:
        ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
        line = f"[{ts}] {msg}"
        if writer is not None:
            writer.add_log_line(line)
        print(line, flush=True)  # noqa: T201

    def _publish(
        topic: str,
        *,
        step: int,
        data: dict[str, Any],
        source: str = "BENCH",
    ) -> MonitorEvent:
        event = MonitorEvent(
            topic=EventTopic(topic),
            step=step,
            t=0.0,
            source=source,
            data=data,
        )
        bus.publish(event)
        return event

    if ctx is not None:
        _publish(
            "run_start",
            step=0,
            source=task_name,
            data={
                "run_meta": ctx.to_dict(),
                "config": asdict(cfg),
            },
        )

    _publish(
        "training_start",
        step=0,
        source=task_name,
        data={
            "task": "bench",
            "device": cfg.device,
            "dtype": cfg.dtype,
            "seed": cfg.seed,
            "steps": cfg.steps,
            "warmup": cfg.warmup,
            "topology": cfg.topology,
            "synapse_model": cfg.synapse_model,
        },
    )

    _publish(
        "scalar",
        step=0,
        source=task_name,
        data={
            "trial": 0,
            "gate": "BENCH",
            "perf.phase": "start",
            "perf.warmup_steps": cfg.warmup,
            "perf.steps_target": cfg.steps,
        },
    )

    _log(f"device   : {cfg.device}")
    _log(f"dtype    : {cfg.dtype}")
    _log(f"topology : {cfg.topology}")
    _log(f"synapse  : {cfg.synapse_model}")
    _log(f"shape    : in={cfg.n_input} hidden={cfg.n_hidden} out={cfg.n_output}")
    _log(f"steps    : warmup={cfg.warmup} timed={cfg.steps}")

    bench_t0 = time.perf_counter()
    try:
        summary = run_bench(cfg)
    except Exception as exc:
        wall_ms = (time.perf_counter() - bench_t0) * 1_000.0
        _publish(
            "run_end",
            step=0,
            source=task_name,
            data={
                "converged": False,
                "failed": True,
                "error": str(exc),
                "wall_ms": round(wall_ms, 1),
            },
        )
        if writer is not None:
            writer.flush()
        print(f"Benchmark failed: {exc}", file=sys.stderr)  # noqa: T201
        return 2

    wall_ms = (time.perf_counter() - bench_t0) * 1_000.0
    topology_json = summary.pop("topology_json", None)
    if topology_json is not None:
        _publish(
            "topology",
            step=0,
            source=task_name,
            data=topology_json,
        )

    final_scalar = _publish(
        "scalar",
        step=cfg.steps,
        source=task_name,
        data={
            "trial": cfg.steps,
            "epoch": 0,
            "gate": "BENCH",
            "wall_ms": round(float(summary["wall_elapsed_s"]) * 1000.0, 3),
            "perf.steps_per_sec": float(summary["steps_per_sec"]),
            "perf.ms_per_step": float(summary["ms_per_step"]),
            "perf.edge_count_total": int(summary["edge_count_total"]),
            "perf.topology": cfg.topology,
            "perf.synapse_model": cfg.synapse_model,
            "perf.n_input": cfg.n_input,
            "perf.n_hidden": cfg.n_hidden,
            "perf.n_output": cfg.n_output,
        },
    )
    for key, value in final_scalar.data.items():
        if (
            key.startswith("resource.")
            or key.startswith("torch_cuda_")
            or key.startswith("cuda_mem_")
        ):
            summary[key] = value

    summary["wall_ms_cli"] = round(wall_ms, 1)

    if ctx is not None:
        summary["run_id"] = ctx.run_id
        summary["run_dir"] = str(ctx.run_dir)
        bench_dir = ctx.run_dir / "bench"
        bench_dir.mkdir(parents=True, exist_ok=True)
        (bench_dir / "bench_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

    _publish(
        "training_end",
        step=cfg.steps,
        source=task_name,
        data={
            "converged": True,
            "bench": True,
            "steps": cfg.steps,
            "wall_ms": round(wall_ms, 1),
            "steps_per_sec": float(summary["steps_per_sec"]),
            "ms_per_step": float(summary["ms_per_step"]),
        },
    )
    _publish(
        "run_end",
        step=cfg.steps,
        source=task_name,
        data={
            "converged": True,
            "bench": True,
            "steps": cfg.steps,
            "wall_ms": round(wall_ms, 1),
        },
    )

    if writer is not None:
        writer.flush()

    _log(f"steps/s  : {summary['steps_per_sec']:.2f}")
    _log(f"ms/step  : {summary['ms_per_step']:.4f}")
    if "run_dir" in summary:
        _log(f"artifacts: {summary['run_dir']}")
    print(json.dumps(summary, indent=2), flush=True)  # noqa: T201
    return 0


# ---------------------------------------------------------------------------
# Subcommand: vision
# ---------------------------------------------------------------------------


def _cmd_vision(args: argparse.Namespace) -> int:
    """Run one vision classification training session."""
    from neuroforge.contracts.monitors import EventTopic, MonitorEvent
    from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
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
    from neuroforge.runners.run_context import create_run_dir
    from neuroforge.runners.vision import VisionRunnerConfig, run_vision_classification

    task_name = "vision_classification"
    apply_determinism(
        DeterminismConfig(
            seed=int(args.seed),
            deterministic=bool(args.deterministic),
            benchmark=bool(args.benchmark),
            warn_only=bool(args.warn_only),
        )
    )
    cfg = VisionRunnerConfig(
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        deterministic=bool(args.deterministic),
        benchmark=bool(args.benchmark),
        warn_only=bool(args.warn_only),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        n_classes=int(args.n_classes),
        image_channels=int(args.image_channels),
        image_h=int(args.image_h),
        image_w=int(args.image_w),
        dataset=str(args.dataset),
        dataset_root=str(args.dataset_root),
        dataset_download=bool(args.dataset_download),
        dataset_val_fraction=float(args.dataset_val_fraction),
        dataset_num_workers=int(args.dataset_num_workers),
        dataset_pin_memory=bool(args.dataset_pin_memory),
        allow_fashion_mnist=bool(args.allow_fashion_mnist),
        dataset_normalize=bool(args.dataset_normalize),
        dataset_random_crop=bool(args.dataset_random_crop),
        dataset_crop_padding=int(args.dataset_crop_padding),
        dataset_random_horizontal_flip=bool(args.dataset_random_horizontal_flip),
        dataset_event_tensor_mode=str(args.dataset_event_tensor_mode),
        dataset_event_slice_mode=str(args.dataset_event_slice_mode),
        dataset_event_window_us=int(args.dataset_event_window_us),
        dataset_event_count=int(args.dataset_event_count),
        dataset_event_polarity_channels=int(args.dataset_event_polarity_channels),
        dataset_logic_image_size=int(args.dataset_logic_image_size),
        dataset_logic_gates=tuple(
            str(v) for v in str(args.dataset_logic_gates).split(",") if str(v).strip()
        ),
        dataset_logic_mode=str(args.dataset_logic_mode),
        dataset_logic_single_gate=str(args.dataset_logic_single_gate),
        dataset_logic_samples_per_gate=int(args.dataset_logic_samples_per_gate),
        dataset_logic_total_samples=int(args.dataset_logic_total_samples),
        dataset_logic_train_ratio=float(args.dataset_logic_train_ratio),
        dataset_logic_val_ratio=float(args.dataset_logic_val_ratio),
        dataset_logic_test_ratio=float(args.dataset_logic_test_ratio),
        lr=float(args.lr),
        loss_fn=str(args.loss_fn),
        readout=str(args.readout),
        readout_threshold=float(args.readout_threshold),
        backbone_type=str(args.backbone_type),
        backbone_time_steps=int(args.backbone_time_steps),
        backbone_encoding_mode=str(args.backbone_encoding_mode),
        backbone_output_dim=int(args.backbone_output_dim),
    )

    ctx = create_run_dir(
        base_dir=args.artifacts,
        seed=cfg.seed,
        device=cfg.device,
    )
    bus = EventBus()
    writer = ArtifactWriter(ctx.run_dir)

    layer_stats = VisionLayerStatsMonitor(interval_steps=1, enabled=True)
    confusion = ConfusionMatrixMonitor(enabled=True)
    samples = VisionSampleGridMonitor(max_samples=16, enabled=True)

    pre_write_monitors: list[Any] = [
        TopologyStatsMonitor(event_bus=bus, enabled=True),
        layer_stats,
        confusion,
        samples,
    ]
    if str(cfg.device).startswith("cuda"):
        pre_write_monitors.append(CudaMetricsMonitor(enabled=True))
    for monitor in pre_write_monitors:
        bus.subscribe_all(monitor)
    bus.subscribe_all(VisionLayerStatsExporter(ctx.run_dir, layer_stats, enabled=True))
    bus.subscribe_all(ConfusionMatrixExporter(ctx.run_dir, confusion, enabled=True))
    bus.subscribe_all(VisionSampleGridExporter(ctx.run_dir, samples, enabled=True))
    bus.subscribe_all(writer)

    def _log(msg: str) -> None:
        ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
        line = f"[{ts}] {msg}"
        writer.add_log_line(line)
        print(line, flush=True)  # noqa: T201

    def _publish(topic: str, *, step: int, data: dict[str, Any]) -> None:
        bus.publish(
            MonitorEvent(
                topic=EventTopic(topic),
                step=step,
                t=0.0,
                source=task_name,
                data=data,
            )
        )

    _publish(
        "run_start",
        step=0,
        data={
            "run_meta": ctx.to_dict(),
            "config": asdict(cfg),
        },
    )
    _log(f"run_id   : {ctx.run_id}")
    _log(f"task     : {task_name}")
    _log(f"seed     : {cfg.seed}")
    _log(f"device   : {cfg.device}")
    _log(f"dtype    : {cfg.dtype}")
    _log(f"steps    : {cfg.steps}")
    _log(f"batch    : {cfg.batch_size}")
    _log(f"dataset  : {cfg.dataset}")

    t0 = time.perf_counter()
    try:
        summary = run_vision_classification(cfg, event_bus=bus)
    except Exception as exc:
        wall_ms = (time.perf_counter() - t0) * 1_000.0
        _publish(
            "run_end",
            step=0,
            data={
                "converged": False,
                "failed": True,
                "error": str(exc),
                "wall_ms": round(wall_ms, 1),
            },
        )
        writer.flush()
        print(f"Vision run failed: {exc}", file=sys.stderr)  # noqa: T201
        return 2

    wall_ms = (time.perf_counter() - t0) * 1_000.0
    result_data = cast("dict[str, Any]", summary["result"])
    _publish(
        "run_end",
        step=int(result_data["steps"]),
        data={
            "converged": False,
            "steps": int(result_data["steps"]),
            "final_loss": float(result_data["final_loss"]),
            "final_accuracy": float(result_data["final_accuracy"]),
            "wall_ms": round(wall_ms, 1),
        },
    )

    vision_dir = ctx.run_dir / "vision"
    vision_dir.mkdir(parents=True, exist_ok=True)
    summary_out = dict(summary)
    summary_out["run_id"] = ctx.run_id
    summary_out["run_dir"] = str(ctx.run_dir)
    summary_out["wall_ms_cli"] = round(wall_ms, 1)
    (vision_dir / "vision_summary.json").write_text(
        json.dumps(summary_out, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset_meta = summary_out.get("dataset_meta")
    if isinstance(dataset_meta, dict):
        dataset_dir = ctx.run_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset_meta.json").write_text(
            json.dumps(dataset_meta, indent=2) + "\n",
            encoding="utf-8",
        )
    writer.flush()

    _log(f"final_loss : {result_data['final_loss']:.6f}")
    _log(f"final_acc  : {result_data['final_accuracy']:.4f}")
    _log(f"wall_ms    : {wall_ms:,.1f}")
    _log(f"artifacts  : {ctx.run_dir}")
    print(json.dumps(summary_out, indent=2), flush=True)  # noqa: T201
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
    from neuroforge.runners.bench import (
        SUPPORTED_BENCH_SYNAPSES,
        SUPPORTED_BENCH_TOPOLOGIES,
    )

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

    p_bench = sub.add_parser("bench", help="Run synthetic performance benchmark")
    p_bench.add_argument(
        "--device",
        default="cpu",
        help="Torch device (e.g. cpu, cuda, cuda:0). Default: cpu",
    )
    p_bench.add_argument(
        "--dtype",
        default="float32",
        help="Torch dtype string (default: float32)",
    )
    p_bench.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    p_bench.add_argument("--n-input", type=int, default=64, help="Input population size")
    p_bench.add_argument("--n-hidden", type=int, default=256, help="Hidden population size")
    p_bench.add_argument("--n-output", type=int, default=1, help="Output population size")
    p_bench.add_argument(
        "--topology",
        choices=list(SUPPORTED_BENCH_TOPOLOGIES),
        default="sparse_random",
        help="Projection topology type",
    )
    p_bench.add_argument(
        "--synapse-model",
        choices=list(SUPPORTED_BENCH_SYNAPSES),
        default="static",
        help="Synapse model key",
    )
    p_bench.add_argument(
        "--p-connect",
        type=float,
        default=0.1,
        help="Connection probability for sparse_random",
    )
    p_bench.add_argument(
        "--fanout",
        type=int,
        default=16,
        help="Fanout per pre-neuron for sparse_fanout",
    )
    p_bench.add_argument(
        "--fanin",
        type=int,
        default=16,
        help="Fanin per post-neuron for sparse_fanin",
    )
    p_bench.add_argument(
        "--block-pre",
        type=int,
        default=16,
        help="Block pre size for block_sparse",
    )
    p_bench.add_argument(
        "--block-post",
        type=int,
        default=16,
        help="Block post size for block_sparse",
    )
    p_bench.add_argument(
        "--p-block",
        type=float,
        default=0.25,
        help="Block keep probability for block_sparse",
    )
    p_bench.add_argument(
        "--max-delay",
        type=int,
        default=3,
        help="Max random edge delay for static_delayed",
    )
    p_bench.add_argument("--steps", type=int, default=2000, help="Timed steps")
    p_bench.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    p_bench.add_argument(
        "--amplitude",
        type=float,
        default=20.0,
        help="Input drive amplitude",
    )
    sync_group = p_bench.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--sync-cuda-timing",
        dest="sync_cuda_timing",
        action="store_true",
        help="Use CUDA event timing (default on for CUDA devices)",
    )
    sync_group.add_argument(
        "--no-sync-cuda-timing",
        dest="sync_cuda_timing",
        action="store_false",
        help="Disable CUDA event timing",
    )
    p_bench.set_defaults(sync_cuda_timing=None)
    write_group = p_bench.add_mutually_exclusive_group()
    write_group.add_argument(
        "--write-artifacts",
        dest="write_artifacts",
        action="store_true",
        help="Write bench artifacts (default: on)",
    )
    write_group.add_argument(
        "--no-write-artifacts",
        dest="write_artifacts",
        action="store_false",
        help="Do not write run artifacts",
    )
    p_bench.set_defaults(write_artifacts=True)
    resources_group = p_bench.add_mutually_exclusive_group()
    resources_group.add_argument(
        "--resources",
        dest="resources",
        action="store_true",
        help="Enable resource monitor during bench (default: on)",
    )
    resources_group.add_argument(
        "--no-resources",
        dest="resources",
        action="store_false",
        help="Disable resource monitor during bench",
    )
    p_bench.set_defaults(resources=True)
    p_bench.add_argument(
        "--artifacts",
        default="artifacts",
        help="Base dir for run artifacts (default: artifacts)",
    )

    p_vision = sub.add_parser(
        "vision",
        help="Run vision classification task",
    )
    p_vision.add_argument("--seed", type=int, default=42, help="Random seed")
    p_vision.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu, cuda, cuda:0, or auto (default: cpu)",
    )
    p_vision.add_argument(
        "--dtype",
        default="float32",
        help="Torch dtype string (default: float32)",
    )
    det_v_group = p_vision.add_mutually_exclusive_group()
    det_v_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Enable deterministic torch algorithms (default: on)",
    )
    det_v_group.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic torch algorithms",
    )
    p_vision.set_defaults(deterministic=True)
    p_vision.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable backend benchmark mode (default: off)",
    )
    warn_v_group = p_vision.add_mutually_exclusive_group()
    warn_v_group.add_argument(
        "--warn-only",
        dest="warn_only",
        action="store_true",
        help="Warn on non-deterministic ops (default)",
    )
    warn_v_group.add_argument(
        "--strict-determinism",
        dest="warn_only",
        action="store_false",
        help="Raise on non-deterministic ops",
    )
    p_vision.set_defaults(warn_only=True)
    p_vision.add_argument("--steps", type=int, default=1, help="Training steps")
    p_vision.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p_vision.add_argument("--n-classes", type=int, default=4, help="Class count")
    p_vision.add_argument(
        "--dataset",
        choices=[
            "synthetic", "mnist", "fashion_mnist",
            "nmnist", "pokerdvs", "logic_gates_pixels",
        ],
        default="synthetic",
        help="Dataset source (default: synthetic)",
    )
    p_vision.add_argument(
        "--dataset-root",
        default=".cache/torchvision",
        help="Dataset cache root (torchvision/Tonic)",
    )
    dataset_dl_group = p_vision.add_mutually_exclusive_group()
    dataset_dl_group.add_argument(
        "--dataset-download",
        dest="dataset_download",
        action="store_true",
        help="Allow torchvision dataset download (default: on)",
    )
    dataset_dl_group.add_argument(
        "--no-dataset-download",
        dest="dataset_download",
        action="store_false",
        help="Require dataset files to already exist in --dataset-root",
    )
    p_vision.set_defaults(dataset_download=True)
    p_vision.add_argument(
        "--dataset-val-fraction",
        type=float,
        default=0.1,
        help="Validation split fraction from train set for torchvision datasets",
    )
    p_vision.add_argument(
        "--dataset-num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for torchvision datasets",
    )
    dataset_pin_group = p_vision.add_mutually_exclusive_group()
    dataset_pin_group.add_argument(
        "--dataset-pin-memory",
        dest="dataset_pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory for torchvision datasets",
    )
    dataset_pin_group.add_argument(
        "--no-dataset-pin-memory",
        dest="dataset_pin_memory",
        action="store_false",
        help="Disable DataLoader pin_memory (default)",
    )
    p_vision.set_defaults(dataset_pin_memory=False)
    p_vision.add_argument(
        "--allow-fashion-mnist",
        action="store_true",
        help="Allow dataset=fashion_mnist",
    )
    p_vision.add_argument(
        "--dataset-event-tensor-mode",
        choices=["frames", "voxel_grid"],
        default="frames",
        help="Event tensorization mode for NMNIST/POKERDVS datasets",
    )
    p_vision.add_argument(
        "--dataset-event-slice-mode",
        choices=["time", "count"],
        default="time",
        help="Deterministic event slicing mode for NMNIST/POKERDVS datasets",
    )
    p_vision.add_argument(
        "--dataset-event-window-us",
        type=int,
        default=100_000,
        help="Fixed time window in microseconds for event slice mode=time",
    )
    p_vision.add_argument(
        "--dataset-event-count",
        type=int,
        default=20_000,
        help="Fixed event count for event slice mode=count",
    )
    p_vision.add_argument(
        "--dataset-event-polarity-channels",
        type=int,
        default=2,
        help="Output event polarity channels (1 or 2)",
    )
    p_vision.add_argument(
        "--dataset-logic-image-size",
        type=int,
        default=8,
        help="Image size for logic_gates_pixels dataset (default: 8)",
    )
    p_vision.add_argument(
        "--dataset-logic-gates",
        default="AND,OR,NAND,NOR",
        help="Comma-separated gate set for logic_gates_pixels (default: AND,OR,NAND,NOR)",
    )
    p_vision.add_argument(
        "--dataset-logic-mode",
        choices=["multiclass", "single_gate"],
        default="multiclass",
        help="Label mode for logic_gates_pixels dataset",
    )
    p_vision.add_argument(
        "--dataset-logic-single-gate",
        default="AND",
        help="Single gate name when --dataset-logic-mode=single_gate",
    )
    p_vision.add_argument(
        "--dataset-logic-samples-per-gate",
        type=int,
        default=128,
        help="Samples per gate for logic_gates_pixels when total-samples is 0",
    )
    p_vision.add_argument(
        "--dataset-logic-total-samples",
        type=int,
        default=0,
        help="Total sample count override for logic_gates_pixels (0 disables)",
    )
    p_vision.add_argument(
        "--dataset-logic-train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio for logic_gates_pixels",
    )
    p_vision.add_argument(
        "--dataset-logic-val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio for logic_gates_pixels",
    )
    p_vision.add_argument(
        "--dataset-logic-test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio for logic_gates_pixels",
    )
    norm_group = p_vision.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--dataset-normalize",
        dest="dataset_normalize",
        action="store_true",
        help="Enable dataset normalization (default: on)",
    )
    norm_group.add_argument(
        "--no-dataset-normalize",
        dest="dataset_normalize",
        action="store_false",
        help="Disable dataset normalization",
    )
    p_vision.set_defaults(dataset_normalize=True)
    crop_group = p_vision.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--dataset-random-crop",
        dest="dataset_random_crop",
        action="store_true",
        help="Enable random crop augmentation (default: off)",
    )
    crop_group.add_argument(
        "--no-dataset-random-crop",
        dest="dataset_random_crop",
        action="store_false",
        help="Disable random crop augmentation",
    )
    p_vision.set_defaults(dataset_random_crop=False)
    p_vision.add_argument(
        "--dataset-crop-padding",
        type=int,
        default=2,
        help="Padding used by random crop augmentation",
    )
    flip_group = p_vision.add_mutually_exclusive_group()
    flip_group.add_argument(
        "--dataset-random-horizontal-flip",
        dest="dataset_random_horizontal_flip",
        action="store_true",
        help="Enable random horizontal flip augmentation (default: off)",
    )
    flip_group.add_argument(
        "--no-dataset-random-horizontal-flip",
        dest="dataset_random_horizontal_flip",
        action="store_false",
        help="Disable random horizontal flip augmentation",
    )
    p_vision.set_defaults(dataset_random_horizontal_flip=False)
    p_vision.add_argument(
        "--image-channels", type=int, default=1, help="Image channel count",
    )
    p_vision.add_argument("--image-h", type=int, default=16, help="Image height")
    p_vision.add_argument("--image-w", type=int, default=16, help="Image width")
    p_vision.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_vision.add_argument(
        "--loss-fn",
        choices=["bce_logits", "mse_count"],
        default="bce_logits",
        help="Loss factory key",
    )
    p_vision.add_argument(
        "--readout",
        choices=["spike_count"],
        default="spike_count",
        help="Readout factory key",
    )
    p_vision.add_argument(
        "--readout-threshold",
        type=float,
        default=0.0,
        help="Threshold for spike_count readout logits",
    )
    p_vision.add_argument(
        "--backbone-type",
        choices=["lif_convnet_v1", "none"],
        default="lif_convnet_v1",
        help="Backbone type (default: lif_convnet_v1)",
    )
    p_vision.add_argument(
        "--backbone-time-steps",
        type=int,
        default=6,
        help="Temporal steps in vision backbone encoding",
    )
    p_vision.add_argument(
        "--backbone-encoding-mode",
        choices=["rate", "poisson", "constant"],
        default="rate",
        help="Static image encoding mode",
    )
    p_vision.add_argument(
        "--backbone-output-dim",
        type=int,
        default=64,
        help="Backbone output feature dimension (ignored when backbone-type=none)",
    )
    p_vision.add_argument(
        "--artifacts",
        default="artifacts",
        help="Base dir for run artifacts (default: artifacts)",
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
        "bench": _cmd_bench,
        "vision": _cmd_vision,
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
