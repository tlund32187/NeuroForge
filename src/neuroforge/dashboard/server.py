# pyright: basic
"""Dashboard server — aiohttp + WebSocket for real-time SNN visualisation.

Serves static files (HTML/CSS/JS) and provides:

**Live mode** (real-time training):
- ``POST /api/train``  — start a training run (returns immediately)
- ``GET  /api/status``  — poll current training state
- ``GET  /ws``          — WebSocket for live event streaming

**Replay mode** (artifact-driven, read-only):
- ``GET /api/runs``              — list completed runs
- ``GET /api/run/<run_id>/meta``    — run_meta.json
- ``GET /api/run/<run_id>/config``  — config_resolved.json
- ``GET /api/run/<run_id>/topology``— topology.json
- ``GET /api/run/<run_id>/topology-stats`` — topology/topology_stats.json
- ``GET /api/run/<run_id>/scalars`` — metrics/scalars.csv as JSON
- ``GET /api/run/<run_id>/vision/sample-grid`` — vision/samples/input_grid.png
- ``GET /api/run/<run_id>/vision/event-sample`` — event sample stats summary
- ``GET /api/run/<run_id>/vision/event-image?mode=bins|sum`` — event sample images
- ``GET /api/run/<run_id>/vision/confusion`` — confusion matrix + per-class accuracy
- ``GET /api/run/<run_id>/vision/layer-stats`` — vision layer stats CSV as JSON
- ``GET /api/run/<run_id>/vision/summary`` — throughput summary from reports/scalars

The training task runs in a background thread so the event loop stays
responsive.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

from aiohttp import web  # pyright: ignore[reportMissingImports]

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.event_recorder import EventRecorderMonitor
from neuroforge.monitors.resource_monitor import ResourceMonitor
from neuroforge.monitors.spike_monitor import SpikeMonitor
from neuroforge.monitors.stability_monitor import StabilityConfig, StabilityMonitor
from neuroforge.monitors.topology_stats_monitor import TopologyStatsMonitor
from neuroforge.monitors.training_monitor import TrainingMonitor
from neuroforge.monitors.trial_stats_monitor import TrialStatsMonitor
from neuroforge.monitors.weight_monitor import WeightMonitor

__all__ = ["start_server"]

STATIC_DIR = Path(__file__).parent / "static"


def _compute_asset_hash() -> str:
    """Hash all static files so the version changes on any edit."""
    h = hashlib.md5()  # noqa: S324
    for f in sorted(STATIC_DIR.glob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


_ASSET_HASH = _compute_asset_hash()

# ── Shared state ────────────────────────────────────────────────────

_bus = EventBus()
_training_monitor = TrainingMonitor()
_spike_monitor = SpikeMonitor()
_weight_monitor = WeightMonitor()

_ws_clients: list[web.WebSocketResponse] = []
_training_thread: threading.Thread | None = None
_training_lock = threading.Lock()
_stop_event = threading.Event()
_current_config: dict[str, Any] | None = None


def _setup_monitors() -> None:
    """Wire monitors to the event bus."""
    _bus.clear()
    _training_monitor.reset()
    _spike_monitor.reset()
    _weight_monitor.reset()

    for topic in EventTopic:
        _bus.subscribe(topic, _training_monitor)
    _bus.subscribe(EventTopic.SPIKE, _spike_monitor)
    _bus.subscribe(EventTopic.WEIGHT, _weight_monitor)


def _parse_resource_monitor_cfg(body: dict[str, Any]) -> dict[str, Any]:
    """Parse opt-in resource monitor config from request JSON."""
    default: dict[str, Any] = {
        "enabled": False,
        "every_n_steps": 10,
        "include_process": True,
        "include_system": True,
        "include_gpu": True,
        "gpu_index": 0,
    }

    raw_monitoring = body.get("monitoring")
    if not isinstance(raw_monitoring, dict):
        return default
    raw_resource = raw_monitoring.get("resource")
    if not isinstance(raw_resource, dict):
        return default

    def _as_bool(value: Any, fallback: bool) -> bool:
        if isinstance(value, bool):
            return value
        return fallback

    def _as_int(value: Any, fallback: int, *, minimum: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return fallback
        return max(minimum, parsed)

    return {
        "enabled": _as_bool(raw_resource.get("enabled"), default["enabled"]),
        "every_n_steps": _as_int(
            raw_resource.get("every_n_steps"),
            default["every_n_steps"],
            minimum=1,
        ),
        "include_process": _as_bool(
            raw_resource.get("include_process"),
            default["include_process"],
        ),
        "include_system": _as_bool(
            raw_resource.get("include_system"),
            default["include_system"],
        ),
        "include_gpu": _as_bool(raw_resource.get("include_gpu"), default["include_gpu"]),
        "gpu_index": _as_int(raw_resource.get("gpu_index"), default["gpu_index"], minimum=0),
    }


def _attach_monitoring_cfg(
    config_dict: dict[str, Any],
    resource_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(config_dict)
    out["monitoring"] = {"resource": dict(resource_cfg)}
    return out


# ── WebSocket broadcaster ──────────────────────────────────────────


class _WsBroadcastMonitor:
    """Monitor that forwards events to all connected WebSocket clients."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.enabled = True
        self._loop = loop
        self._step_interval = 1  # send every Nth trial to control bandwidth

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        # Throttle TRAINING_TRIAL events.
        if (
            event.topic == EventTopic.TRAINING_TRIAL
            and event.step % self._step_interval != 0
        ):
            return
        # Skip per-step WEIGHT except every 10 trials.
        if event.topic == EventTopic.WEIGHT and event.step % 10 != 0:
            return

        payload = _event_to_json(event)
        asyncio.run_coroutine_threadsafe(
            _broadcast(payload), self._loop
        )

    def reset(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {}


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert tensors to Python lists for JSON serialisation."""
    # Tensor-like?
    try:
        return obj.detach().cpu().tolist()
    except (AttributeError, TypeError):
        pass
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    return obj


def _event_to_json(event: MonitorEvent) -> str:
    """Serialise a MonitorEvent to JSON."""
    safe_data = _make_json_safe(event.data)
    return json.dumps({
        "topic": event.topic.value,
        "step": event.step,
        "t": event.t,
        "source": event.source,
        "data": safe_data,
    })


async def _broadcast(payload: str) -> None:
    """Send *payload* to all connected WebSocket clients."""
    closed: list[web.WebSocketResponse] = []
    for ws in _ws_clients:
        if ws.closed:
            closed.append(ws)
            continue
        try:
            await ws.send_str(payload)
        except ConnectionResetError:
            closed.append(ws)
    for ws in closed:
        _ws_clients.remove(ws)


# ── Route handlers ──────────────────────────────────────────────────


async def _handle_index(_request: web.Request) -> web.Response:
    raw = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = raw.replace("__CACHE_BUST__", _compute_asset_hash())
    return web.Response(
        text=html,
        content_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


async def _handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.append(ws)

    # Send current snapshot immediately.
    snap_data: dict[str, Any] = {
        "topic": "snapshot",
        "training": _training_monitor.snapshot(),
        "weights": _weight_monitor.snapshot(),
        "spikes": _spike_monitor.snapshot(),
    }
    if _current_config is not None:
        snap_data["config"] = _current_config
    await ws.send_str(json.dumps(snap_data))

    async for _msg in ws:
        pass  # keep alive; client doesn't send data
    _ws_clients.remove(ws)
    return ws


async def _handle_train(request: web.Request) -> web.Response:
    """Start a training run in a background thread."""
    global _training_thread

    with _training_lock:
        if _training_thread is not None and _training_thread.is_alive():
            return web.json_response(
                {"error": "Training already in progress"}, status=409
            )

    body = await request.json()
    gate = body.get("gate", "OR")
    max_trials = int(body.get("max_trials", 5000))
    device: str = str(body.get("device", ""))
    resource_cfg = _parse_resource_monitor_cfg(body)

    # Resolve device: explicit "cpu"/"cuda" is honoured.  Anything else
    # (including "auto" and empty) becomes "auto" so the config's
    # __post_init__ picks the fastest option for the topology.
    if device not in ("cpu", "cuda"):
        device = "auto"

    # Weights are always initialised on CPU so the same seed gives
    # identical starting values regardless of the target device.
    seed = 42

    _setup_monitors()
    _stop_event.clear()

    # Phase 6 enrichment monitors for live dashboard + replay artifacts.
    trial_stats_monitor = TrialStatsMonitor(enabled=True)
    stability_monitor = StabilityMonitor(
        StabilityConfig(enabled=True, check_every_n_trials=5, fail_fast=False),
    )
    topology_stats_monitor = TopologyStatsMonitor(event_bus=_bus, enabled=True)
    _bus.subscribe_all(trial_stats_monitor)
    _bus.subscribe_all(stability_monitor)
    _bus.subscribe_all(topology_stats_monitor)

    # Optional resource monitor: enriches SCALAR payloads.
    resource_monitor: ResourceMonitor | None = None
    if resource_cfg["enabled"]:
        resource_monitor = ResourceMonitor(
            enabled=True,
            every_n_steps=resource_cfg["every_n_steps"],
            include_process=resource_cfg["include_process"],
            include_system=resource_cfg["include_system"],
            include_gpu=resource_cfg["include_gpu"],
            gpu_index=resource_cfg["gpu_index"],
        )
        _bus.subscribe_all(resource_monitor)

    # Subscribe the WS broadcaster.
    loop = asyncio.get_running_loop()
    ws_mon = _WsBroadcastMonitor(loop)
    for topic in EventTopic:
        _bus.subscribe(topic, ws_mon)

    # Also record artifacts so the run can be replayed.
    from dataclasses import asdict

    from neuroforge.monitors.artifact_writer import ArtifactWriter
    from neuroforge.runners.run_context import create_run_dir

    ctx = create_run_dir(base_dir="artifacts", seed=seed)
    artifact_writer = ArtifactWriter(ctx.run_dir)
    event_recorder = EventRecorderMonitor(ctx.run_dir)
    _bus.subscribe_all(artifact_writer)
    _bus.subscribe_all(event_recorder)

    # Emit RUN_START so ArtifactWriter writes run_meta + config.
    def _make_config_dict(config: Any) -> dict[str, Any]:
        if hasattr(config, "__dataclass_fields__"):
            return asdict(config)
        return dict(vars(config))

    def _run() -> None:
        global _current_config

        try:
            if gate == "LOGIC_BACKBONE_TINY":
                from neuroforge.monitors.vision_monitors import (
                    ConfusionMatrixExporter,
                    ConfusionMatrixMonitor,
                    VisionLayerStatsExporter,
                    VisionLayerStatsMonitor,
                    VisionSampleGridExporter,
                    VisionSampleGridMonitor,
                )
                from neuroforge.runners.vision import (
                    VisionRunnerConfig,
                    run_vision_classification,
                )

                layer_stats = VisionLayerStatsMonitor(interval_steps=1, enabled=True)
                confusion = ConfusionMatrixMonitor(enabled=True)
                samples = VisionSampleGridMonitor(max_samples=16, enabled=True)
                _bus.subscribe_all(layer_stats)
                _bus.subscribe_all(confusion)
                _bus.subscribe_all(samples)
                _bus.subscribe_all(
                    VisionLayerStatsExporter(ctx.run_dir, layer_stats, enabled=True)
                )
                _bus.subscribe_all(
                    ConfusionMatrixExporter(ctx.run_dir, confusion, enabled=True)
                )
                _bus.subscribe_all(VisionSampleGridExporter(ctx.run_dir, samples, enabled=True))

                vision_cfg = VisionRunnerConfig(
                    seed=seed,
                    device=device,
                    dtype="float32",
                    deterministic=True,
                    benchmark=False,
                    warn_only=True,
                    steps=max(1, max_trials),
                    batch_size=16,
                    n_classes=4,
                    image_channels=1,
                    image_h=8,
                    image_w=8,
                    dataset="logic_gates_pixels",
                    dataset_root=".cache/logic_gates_pixels",
                    dataset_download=False,
                    dataset_num_workers=0,
                    dataset_pin_memory=False,
                    dataset_logic_image_size=8,
                    dataset_logic_gates=("AND", "OR", "NAND", "NOR"),
                    dataset_logic_mode="multiclass",
                    dataset_logic_samples_per_gate=256,
                    dataset_logic_train_ratio=0.7,
                    dataset_logic_val_ratio=0.15,
                    dataset_logic_test_ratio=0.15,
                    lr=1e-3,
                    loss_fn="bce_logits",
                    readout="spike_count",
                    readout_threshold=0.0,
                    backbone_type="lif_convnet_v1",
                    backbone_time_steps=8,
                    backbone_encoding_mode="rate",
                    backbone_output_dim=32,
                    backbone_blocks=(
                        {"type": "conv", "params": {"out_channels": 8, "kernel_size": 3}},
                        {"type": "pool", "params": {"kernel_size": 2, "mode": "avg"}},
                        {"type": "conv", "params": {"out_channels": 12, "kernel_size": 3}},
                    ),
                )
                _current_config = _attach_monitoring_cfg(
                    _make_config_dict(vision_cfg),
                    resource_cfg,
                )
                _bus.publish(MonitorEvent(
                    topic=EventTopic.RUN_START, step=0, t=0.0,
                    source="dashboard",
                    data={
                        "run_meta": ctx.to_dict(),
                        "config": _current_config,
                    },
                ))
                summary = run_vision_classification(vision_cfg, event_bus=_bus)
                result = summary.get("result", {})
                steps = int(result.get("steps", max_trials))
                _bus.publish(MonitorEvent(
                    topic=EventTopic.RUN_END, step=steps, t=0.0,
                    source="dashboard",
                    data={
                        "converged": True,
                        "trials": steps,
                        "steps": steps,
                        "final_loss": float(result.get("final_loss", 0.0)),
                        "final_accuracy": float(result.get("final_accuracy", 0.0)),
                    },
                ))
            elif gate == "MULTI":
                from neuroforge.tasks.multi_gate import MultiGateConfig, MultiGateTask

                multi_cfg = MultiGateConfig(
                    max_epochs=max_trials,  # UI field doubles as max_epochs
                    device=device,
                    seed=seed,
                )
                _current_config = _attach_monitoring_cfg(
                    _make_config_dict(multi_cfg),
                    resource_cfg,
                )
                _bus.publish(MonitorEvent(
                    topic=EventTopic.RUN_START, step=0, t=0.0,
                    source="dashboard",
                    data={
                        "run_meta": ctx.to_dict(),
                        "config": _current_config,
                    },
                ))
                multi_result = MultiGateTask(
                    multi_cfg, event_bus=_bus, stop_check=_stop_event.is_set,
                ).run()
                _bus.publish(MonitorEvent(
                    topic=EventTopic.RUN_END, step=0, t=0.0,
                    source="dashboard",
                    data={
                        "converged": multi_result.converged,
                        "trials": multi_result.trials,
                        "epochs": multi_result.epochs,
                    },
                ))
            else:
                from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask

                single_cfg = LogicGateConfig(
                    gate=gate, max_trials=max_trials,
                    device=device,
                    seed=seed,
                )
                _current_config = _attach_monitoring_cfg(
                    _make_config_dict(single_cfg),
                    resource_cfg,
                )
                _bus.publish(MonitorEvent(
                    topic=EventTopic.RUN_START, step=0, t=0.0,
                    source="dashboard",
                    data={
                        "run_meta": ctx.to_dict(),
                        "config": _current_config,
                    },
                ))
                single_result = LogicGateTask(
                    single_cfg, event_bus=_bus, stop_check=_stop_event.is_set,
                ).run()
                _bus.publish(MonitorEvent(
                    topic=EventTopic.RUN_END, step=0, t=0.0,
                    source="dashboard",
                    data={
                        "converged": single_result.converged,
                        "trials": single_result.trials,
                    },
                ))
        finally:
            event_recorder.close()
            if resource_monitor is not None:
                resource_monitor.reset()

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()

    return web.json_response({
        "status": "started", "gate": gate, "run_id": ctx.run_id,
    })


async def _handle_stop(_request: web.Request) -> web.Response:
    """Signal the training thread to stop."""
    _stop_event.set()
    return web.json_response({"status": "stopping"})


async def _handle_status(_request: web.Request) -> web.Response:
    """Return current training snapshot."""
    snap: dict[str, Any] = {
        "training": _training_monitor.snapshot(),
        "weights": _weight_monitor.snapshot(),
        "spikes": _spike_monitor.snapshot(),
    }
    if _current_config is not None:
        snap["config"] = _current_config
    return web.json_response(snap)


# ── Replay API (artifact-driven, read-only) ────────────────────────

_ARTIFACTS_DIR = Path("artifacts")
_RUN_ID_RE = re.compile(r"^run_\d{8}_\d{6}_[0-9a-f]{8}$")


def _safe_run_dir(run_id: str) -> Path | None:
    """Return the run directory if *run_id* is valid and exists."""
    if not _RUN_ID_RE.match(run_id):
        return None
    run_dir = _ARTIFACTS_DIR / run_id
    # Resolve to prevent traversal and ensure it's under artifacts.
    try:
        resolved = run_dir.resolve(strict=False)
        artifacts_resolved = _ARTIFACTS_DIR.resolve(strict=False)
        if not str(resolved).startswith(str(artifacts_resolved)):
            return None
    except (OSError, ValueError):
        return None
    if run_dir.is_dir():
        return run_dir
    return None


def _parse_csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            if value == "True":
                return True
            if value == "False":
                return False
            return value


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                parsed[str(key)] = _parse_csv_value(value)
            rows.append(parsed)
    return rows


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


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _is_vision_config(cfg: dict[str, Any]) -> bool:
    task_name = str(cfg.get("task", "")).strip().lower()
    if "vision" in task_name:
        return True
    if "backbone_time_steps" in cfg or "backbone_encoding_mode" in cfg:
        return True
    if "dataset_name" in cfg:
        return True
    return str(cfg.get("dataset", "")).strip().lower() in {
        "synthetic",
        "mnist",
        "fashion_mnist",
        "nmnist",
        "pokerdvs",
        "logic_gates_pixels",
    }


def _vision_sample_grid_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "samples" / "input_grid.png"


def _vision_event_bins_grid_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "samples" / "event_bins_grid.png"


def _vision_event_sum_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "samples" / "event_sum.png"


def _vision_event_stats_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "metrics" / "event_sample_stats.json"


def _vision_layer_stats_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "metrics" / "vision_layer_stats.csv"


def _vision_confusion_npy_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "metrics" / "confusion_matrix.npy"


def _vision_confusion_json_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "metrics" / "confusion_matrix.json"


def _vision_per_class_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "metrics" / "per_class_accuracy.json"


def _vision_benchmark_summary_path(run_dir: Path) -> Path:
    return run_dir / "reports" / "benchmark_summary.json"


def _vision_summary_path(run_dir: Path) -> Path:
    return run_dir / "vision" / "vision_summary.json"


def _load_confusion_matrix_from_npy(path: Path) -> list[list[int]] | None:
    if not path.is_file():
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        arr = np.load(path)
        rows_raw = arr.tolist()
    except Exception:
        return None
    if not isinstance(rows_raw, list):
        return None
    out: list[list[int]] = []
    for row in rows_raw:
        if not isinstance(row, list):
            return None
        out.append([int(v) for v in row])
    return out


def _load_confusion_matrix_from_json(path: Path) -> list[list[int]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    matrix_raw: Any = payload
    if isinstance(payload, dict):
        if "matrix" in payload:
            matrix_raw = payload.get("matrix")
        elif "confusion_matrix" in payload:
            matrix_raw = payload.get("confusion_matrix")
    if not isinstance(matrix_raw, list):
        return None
    out: list[list[int]] = []
    for row in matrix_raw:
        if not isinstance(row, list):
            return None
        out.append([int(v) for v in row])
    return out


def _throughput_from_scalars(rows: list[dict[str, Any]]) -> dict[str, Any]:
    last_steps_per_sec: float | None = None
    last_ms_per_step: float | None = None
    last_wall_ms: float | None = None
    last_trial: int | None = None
    for row in rows:
        sps = _to_float(row.get("perf.steps_per_sec"))
        if sps is None:
            sps = _to_float(row.get("steps_per_sec"))
        if sps is not None:
            last_steps_per_sec = sps

        ms = _to_float(row.get("perf.ms_per_step"))
        if ms is None:
            ms = _to_float(row.get("ms_per_step"))
        wall_ms = _to_float(row.get("wall_ms"))
        if ms is None and wall_ms is not None:
            ms = wall_ms
        if ms is not None:
            last_ms_per_step = ms
            if ms > 0:
                last_steps_per_sec = 1000.0 / ms if sps is None else last_steps_per_sec
        if wall_ms is not None:
            last_wall_ms = wall_ms

        trial = _to_int(row.get("trial"))
        if trial is not None:
            last_trial = trial

    return {
        "steps_per_sec": last_steps_per_sec,
        "ms_per_step": last_ms_per_step,
        "wall_ms": last_wall_ms,
        "steps": int(last_trial) if last_trial is not None else len(rows),
    }


async def _handle_runs(_request: web.Request) -> web.Response:
    """List completed runs under artifacts/, newest first."""
    if not _ARTIFACTS_DIR.is_dir():
        return web.json_response([])

    runs: list[dict[str, Any]] = []
    for d in sorted(_ARTIFACTS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not _RUN_ID_RE.match(d.name):
            continue
        entry: dict[str, Any] = {"run_id": d.name}
        meta_path = d / "run_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                entry["created_at"] = meta.get("started_at", "")
                entry["device"] = meta.get("device", "")
                entry["seed"] = meta.get("seed", "")
            except (json.JSONDecodeError, OSError):
                pass
        # Try to figure out the task name from config.
        cfg_path = d / "config_resolved.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                if not isinstance(cfg, dict):
                    cfg = {}
                if _is_vision_config(cfg):
                    entry["task_name"] = "vision"
                elif "max_epochs" in cfg:
                    entry["task_name"] = "multi_gate"
                elif "gate" in cfg:
                    entry["task_name"] = f"logic_gate ({cfg['gate']})"
            except (json.JSONDecodeError, OSError):
                pass
        runs.append(entry)

    return web.json_response(runs)


async def _handle_run_meta(request: web.Request) -> web.Response:
    """Serve run_meta.json for a given run."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)
    path = run_dir / "run_meta.json"
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    return web.Response(
        text=path.read_text(encoding="utf-8"),
        content_type="application/json",
    )


async def _handle_run_config(request: web.Request) -> web.Response:
    """Serve config_resolved.json for a given run."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)
    path = run_dir / "config_resolved.json"
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    return web.Response(
        text=path.read_text(encoding="utf-8"),
        content_type="application/json",
    )


async def _handle_run_topology(request: web.Request) -> web.Response:
    """Serve topology.json for a given run."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)
    path = run_dir / "topology.json"
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    return web.Response(
        text=path.read_text(encoding="utf-8"),
        content_type="application/json",
    )


async def _handle_run_topology_stats(request: web.Request) -> web.Response:
    """Serve topology/topology_stats.json for a given run."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)
    path = run_dir / "topology" / "topology_stats.json"
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    return web.Response(
        text=path.read_text(encoding="utf-8"),
        content_type="application/json",
    )


async def _handle_run_scalars(request: web.Request) -> web.Response:
    """Serve metrics/scalars.csv as a JSON array of objects."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)
    path = run_dir / "metrics" / "scalars.csv"
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    rows = _read_csv_rows(path)
    return web.json_response(rows)


async def _handle_run_vision_sample_grid(request: web.Request) -> web.StreamResponse:
    """Serve vision sample grid image for a given run."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)
    path = _vision_sample_grid_path(run_dir)
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    return web.FileResponse(path)


async def _handle_run_vision_event_image(request: web.Request) -> web.StreamResponse:
    """Serve event-sample artifact image for a given run and display mode."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)

    mode_raw = str(request.rel_url.query.get("mode", "bins")).strip().lower()
    mode = "sum" if mode_raw in {"sum", "summed", "aggregate"} else "bins"
    path = (
        _vision_event_sum_path(run_dir)
        if mode == "sum"
        else _vision_event_bins_grid_path(run_dir)
    )
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    return web.FileResponse(path)


async def _handle_run_vision_event_sample(request: web.Request) -> web.Response:
    """Serve event-sample stats + artifact availability for a given run."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)

    stats = _safe_json_load(_vision_event_stats_path(run_dir))
    has_bins_grid = _vision_event_bins_grid_path(run_dir).is_file()
    has_sum = _vision_event_sum_path(run_dir).is_file()
    if stats is None and not has_bins_grid and not has_sum:
        return web.json_response({"error": "not found"}, status=404)

    payload = dict(stats or {})
    image_files_raw = payload.get("image_files")
    image_files = dict(image_files_raw) if isinstance(image_files_raw, dict) else {}
    image_files.setdefault("bins_grid", "event_bins_grid.png")
    image_files.setdefault("sum", "event_sum.png")

    payload["image_files"] = image_files
    payload["has_bins_grid"] = has_bins_grid
    payload["has_sum"] = has_sum
    return web.json_response(payload)


async def _handle_run_vision_layer_stats(request: web.Request) -> web.Response:
    """Serve vision_layer_stats.csv as JSON rows."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)
    path = _vision_layer_stats_path(run_dir)
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)
    rows = _read_csv_rows(path)
    layers = sorted({
        str(row.get("layer"))
        for row in rows
        if row.get("layer") not in (None, "")
    })
    return web.json_response({
        "rows": rows,
        "layers": layers,
        "metrics": ["spike_rate", "mean_activation", "max_activation"],
    })


async def _handle_run_vision_confusion(request: web.Request) -> web.Response:
    """Serve confusion matrix summary from vision artifacts."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)

    npy_path = _vision_confusion_npy_path(run_dir)
    matrix_json_path = _vision_confusion_json_path(run_dir)
    per_class_path = _vision_per_class_path(run_dir)

    per_class_payload = _safe_json_load(per_class_path) or {}
    if (
        not npy_path.is_file()
        and not matrix_json_path.is_file()
        and not per_class_payload
    ):
        return web.json_response({"error": "not found"}, status=404)

    matrix: list[list[int]] = []
    matrix_source = "none"

    matrix_npy = _load_confusion_matrix_from_npy(npy_path)
    if matrix_npy is not None:
        matrix = matrix_npy
        matrix_source = "npy"
    else:
        matrix_json = _load_confusion_matrix_from_json(matrix_json_path)
        if matrix_json is not None:
            matrix = matrix_json
            matrix_source = "json"

    total_samples = int(sum(sum(row) for row in matrix))
    payload_total = _to_int(per_class_payload.get("total_samples"))
    if payload_total is not None and payload_total > total_samples:
        total_samples = payload_total

    n_classes = len(matrix)
    payload_classes = _to_int(per_class_payload.get("n_classes"))
    if payload_classes is not None and payload_classes > n_classes:
        n_classes = payload_classes

    per_class_accuracy = per_class_payload.get("per_class_accuracy", {})
    if not isinstance(per_class_accuracy, dict):
        per_class_accuracy = {}

    return web.json_response({
        "matrix": matrix,
        "matrix_source": matrix_source,
        "n_classes": n_classes,
        "total_samples": total_samples,
        "per_class_accuracy": per_class_accuracy,
    })


async def _handle_run_vision_summary(request: web.Request) -> web.Response:
    """Serve throughput/metrics summary for vision runs."""
    run_dir = _safe_run_dir(request.match_info["run_id"])
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)

    benchmark_summary = _safe_json_load(_vision_benchmark_summary_path(run_dir))
    vision_summary = _safe_json_load(_vision_summary_path(run_dir))

    summary_source = "none"
    throughput: dict[str, Any] = {
        "steps_per_sec": None,
        "ms_per_step": None,
        "wall_ms": None,
        "steps": None,
    }
    final_metrics: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    summary_payload: dict[str, Any] = {}

    if benchmark_summary is not None:
        summary_source = "benchmark_summary"
        summary_payload = benchmark_summary
        raw_throughput = benchmark_summary.get("throughput", {})
        if isinstance(raw_throughput, dict):
            throughput["steps_per_sec"] = _to_float(raw_throughput.get("steps_per_sec"))
            throughput["ms_per_step"] = _to_float(raw_throughput.get("ms_per_step"))
            throughput["wall_ms"] = _to_float(raw_throughput.get("wall_ms"))
        throughput["steps"] = _to_int(benchmark_summary.get("steps"))
        raw_final = benchmark_summary.get("final_metrics", {})
        raw_best = benchmark_summary.get("best_metrics", {})
        if isinstance(raw_final, dict):
            final_metrics = raw_final
        if isinstance(raw_best, dict):
            best_metrics = raw_best
    elif vision_summary is not None:
        summary_source = "vision_summary"
        summary_payload = vision_summary
        result = vision_summary.get("result", {})
        if isinstance(result, dict):
            throughput["steps"] = _to_int(result.get("steps"))
            final_metrics = {
                "loss": _to_float(result.get("final_loss")),
                "accuracy": _to_float(result.get("final_accuracy")),
            }
            best_metrics = {}

    scalars_path = run_dir / "metrics" / "scalars.csv"
    scalars_rows = _read_csv_rows(scalars_path) if scalars_path.is_file() else []
    scalars_throughput = _throughput_from_scalars(scalars_rows)
    for key in ("steps_per_sec", "ms_per_step", "wall_ms", "steps"):
        if throughput.get(key) is None:
            throughput[key] = scalars_throughput.get(key)

    if summary_source == "none" and any(
        throughput.get(key) is not None for key in ("steps_per_sec", "ms_per_step", "steps")
    ):
        summary_source = "scalars"

    if summary_source == "none":
        return web.json_response({"error": "not found"}, status=404)

    return web.json_response({
        "summary_source": summary_source,
        "throughput": throughput,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "summary": summary_payload,
    })


def _events_path(run_dir: Path) -> Path:
    return run_dir / "events" / "events.ndjson"


def _iter_ndjson(path: Path) -> list[tuple[int, int, dict[str, Any]]]:
    """Return ``(index, byte_offset, event)`` tuples from an NDJSON file."""
    rows: list[tuple[int, int, dict[str, Any]]] = []
    offset = 0
    idx = 0
    with path.open("rb") as fh:
        for raw in fh:
            line_offset = offset
            offset += len(raw)
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            rows.append((idx, line_offset, event))
            idx += 1
    return rows


async def _handle_run_events(request: web.Request) -> web.Response:
    """Serve paginated recorded monitor events from events.ndjson."""
    run_id = request.match_info["run_id"]
    run_dir = _safe_run_dir(run_id)
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)

    path = _events_path(run_dir)
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)

    q = request.rel_url.query
    try:
        start = max(0, int(q.get("start", "0")))
        limit = max(1, min(5000, int(q.get("limit", "500"))))
    except ValueError:
        return web.json_response({"error": "invalid pagination"}, status=400)

    rows = _iter_ndjson(path)
    total = len(rows)
    sliced = rows[start:start + limit]
    events = [event for _idx, _offset, event in sliced]
    return web.json_response({
        "run_id": run_id,
        "start": start,
        "limit": limit,
        "total": total,
        "events": events,
    })


async def _handle_run_events_index(request: web.Request) -> web.Response:
    """Serve offsets + timestamps for replay scrubber timelines."""
    run_id = request.match_info["run_id"]
    run_dir = _safe_run_dir(run_id)
    if run_dir is None:
        return web.json_response({"error": "not found"}, status=404)

    path = _events_path(run_dir)
    if not path.is_file():
        return web.json_response({"error": "not found"}, status=404)

    rows = _iter_ndjson(path)
    items = []
    for idx, offset, event in rows:
        items.append({
            "idx": idx,
            "offset": offset,
            "ts_wall": event.get("ts_wall"),
            "t_step": event.get("t_step", event.get("step")),
            "topic": event.get("topic"),
        })
    return web.json_response({
        "run_id": run_id,
        "count": len(items),
        "items": items,
    })


# ── Server bootstrap ───────────────────────────────────────────────


def start_server(*, host: str = "127.0.0.1", port: int = 8050) -> None:
    """Create and run the aiohttp application."""

    @web.middleware
    async def _no_cache(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        resp = await handler(request)
        if isinstance(resp, web.StreamResponse):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
        return resp  # type: ignore[no-any-return]

    app = web.Application(middlewares=[_no_cache])
    app.router.add_get("/", _handle_index)
    app.router.add_get("/ws", _handle_ws)
    app.router.add_post("/api/train", _handle_train)
    app.router.add_post("/api/stop", _handle_stop)
    app.router.add_get("/api/status", _handle_status)

    # Replay API (artifact-driven, read-only).
    app.router.add_get("/api/runs", _handle_runs)
    app.router.add_get("/api/run/{run_id}/meta", _handle_run_meta)
    app.router.add_get("/api/run/{run_id}/config", _handle_run_config)
    app.router.add_get("/api/run/{run_id}/topology", _handle_run_topology)
    app.router.add_get("/api/run/{run_id}/topology-stats", _handle_run_topology_stats)
    app.router.add_get("/api/run/{run_id}/scalars", _handle_run_scalars)
    app.router.add_get("/api/run/{run_id}/vision/sample-grid", _handle_run_vision_sample_grid)
    app.router.add_get("/api/run/{run_id}/vision/event-sample", _handle_run_vision_event_sample)
    app.router.add_get("/api/run/{run_id}/vision/event-image", _handle_run_vision_event_image)
    app.router.add_get("/api/run/{run_id}/vision/confusion", _handle_run_vision_confusion)
    app.router.add_get("/api/run/{run_id}/vision/layer-stats", _handle_run_vision_layer_stats)
    app.router.add_get("/api/run/{run_id}/vision/summary", _handle_run_vision_summary)
    app.router.add_get("/api/run/{run_id}/events", _handle_run_events)
    app.router.add_get("/api/run/{run_id}/events/index", _handle_run_events_index)

    app.router.add_static("/static", STATIC_DIR, show_index=False)

    print(f"\n  NeuroForge Dashboard -> http://{host}:{port}\n")
    web.run_app(app, host=host, port=port, print=None)
