# pyright: basic
"""Dashboard server — aiohttp + WebSocket for real-time SNN visualisation.

Serves static files (HTML/CSS/JS) and provides:
- ``POST /api/train``  — start a training run (returns immediately)
- ``GET  /api/status``  — poll current training state
- ``GET  /ws``          — WebSocket for live event streaming

The training task runs in a background thread so the event loop stays
responsive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from aiohttp import web

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.spike_monitor import SpikeMonitor
from neuroforge.monitors.training_monitor import TrainingMonitor
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


def _event_to_json(event: MonitorEvent) -> str:
    """Serialise a MonitorEvent to JSON."""
    safe_data: dict[str, Any] = {}
    for k, v in event.data.items():
        try:
            # Try tensor → list conversion.
            safe_data[k] = v.detach().cpu().tolist()
        except (AttributeError, TypeError):
            safe_data[k] = v
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
    html = raw.replace("__CACHE_BUST__", _ASSET_HASH)
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
    snap = json.dumps({
        "topic": "snapshot",
        "training": _training_monitor.snapshot(),
        "weights": _weight_monitor.snapshot(),
        "spikes": _spike_monitor.snapshot(),
    })
    await ws.send_str(snap)

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

    _setup_monitors()
    _stop_event.clear()

    # Subscribe the WS broadcaster.
    loop = asyncio.get_running_loop()
    ws_mon = _WsBroadcastMonitor(loop)
    for topic in EventTopic:
        _bus.subscribe(topic, ws_mon)

    def _run() -> None:
        if gate == "MULTI":
            from neuroforge.tasks.multi_gate import MultiGateConfig, MultiGateTask

            multi_cfg = MultiGateConfig(
                max_epochs=max_trials,  # UI field doubles as max_epochs
            )
            MultiGateTask(
                multi_cfg, event_bus=_bus, stop_check=_stop_event.is_set,
            ).run()
        else:
            from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask

            single_cfg = LogicGateConfig(gate=gate, max_trials=max_trials)
            LogicGateTask(
                single_cfg, event_bus=_bus, stop_check=_stop_event.is_set,
            ).run()

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()

    return web.json_response({"status": "started", "gate": gate})


async def _handle_stop(_request: web.Request) -> web.Response:
    """Signal the training thread to stop."""
    _stop_event.set()
    return web.json_response({"status": "stopping"})


async def _handle_status(_request: web.Request) -> web.Response:
    """Return current training snapshot."""
    snap = {
        "training": _training_monitor.snapshot(),
        "weights": _weight_monitor.snapshot(),
        "spikes": _spike_monitor.snapshot(),
    }
    return web.json_response(snap)


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
    app.router.add_static("/static", STATIC_DIR, show_index=False)

    print(f"\n  NeuroForge Dashboard -> http://{host}:{port}\n")
    web.run_app(app, host=host, port=port, print=None)
