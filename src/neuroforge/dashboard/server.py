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

from aiohttp import web

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
            if gate == "MULTI":
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
                if "max_epochs" in cfg:
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
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Convert numeric fields.
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = int(v)
                    except (ValueError, TypeError):
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            if v == "True":
                                parsed[k] = True
                            elif v == "False":
                                parsed[k] = False
                            else:
                                parsed[k] = v
            rows.append(parsed)
    return web.json_response(rows)


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
    app.router.add_get("/api/run/{run_id}/events", _handle_run_events)
    app.router.add_get("/api/run/{run_id}/events/index", _handle_run_events_index)

    app.router.add_static("/static", STATIC_DIR, show_index=False)

    print(f"\n  NeuroForge Dashboard -> http://{host}:{port}\n")
    web.run_app(app, host=host, port=port, print=None)
