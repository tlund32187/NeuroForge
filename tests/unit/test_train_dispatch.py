# pyright: basic, reportMissingImports=false
"""Dashboard ``/api/train`` dispatch tests.

These lock in the gate->handler routing that replaced the former if/elif chain
in :func:
euroforge.interfaces.dashboard.server._handle_train`. Real training is faked so
the test is fast and deterministic - we only assert that the correct branch
ran and emitted a sensible ``run_end`` event.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _make_app() -> Any:
    from aiohttp import web

    import neuroforge.interfaces.dashboard.server as srv

    app = web.Application()
    app.router.add_post("/api/train", srv._handle_train)
    app.router.add_post("/api/stop", srv._handle_stop)
    return app


async def _train_and_collect(body: dict[str, Any], artifacts_root: Path) -> dict[str, Any]:
    """POST a train request, wait for the background run, return parsed events."""
    from aiohttp.test_utils import TestClient, TestServer

    import neuroforge.interfaces.dashboard.server as srv

    app = _make_app()
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.post("/api/train", json=body)
        started = await resp.json()
        assert resp.status == 200, (resp.status, started)
        run_id = started["run_id"]

        for _ in range(300):
            with srv._training_lock:
                thread = srv._training_thread
            if thread is None or not thread.is_alive():
                break
            await asyncio.sleep(0.05)

    events_path = artifacts_root / run_id / "events" / "events.ndjson"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"run_id": run_id, "events": events}


@pytest.fixture()
def _fake_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect run dirs to tmp and replace the real tasks with fast fakes."""
    import neuroforge.applications.tasks.logic_gates as logic_gates
    import neuroforge.applications.tasks.multi_gate as multi_gate
    import neuroforge.simulation.runtime.run_context as run_context

    artifacts_root = tmp_path / "artifacts"
    real_create = run_context.create_run_dir

    def _fake_create(*args: Any, **kwargs: Any) -> Any:
        kwargs["base_dir"] = str(artifacts_root)
        return real_create(*args, **kwargs)

    monkeypatch.setattr(run_context, "create_run_dir", _fake_create)

    class _FakeSingle:
        def __init__(self, _cfg: Any, *, event_bus: Any = None, stop_check: Any = None) -> None:
            pass

        def run(self) -> Any:
            return SimpleNamespace(converged=True, trials=7)

    class _FakeMulti:
        def __init__(self, _cfg: Any, *, event_bus: Any = None, stop_check: Any = None) -> None:
            pass

        def run(self) -> Any:
            return SimpleNamespace(converged=True, trials=24, epochs=3)

    monkeypatch.setattr(logic_gates, "LogicGateTask", _FakeSingle)
    monkeypatch.setattr(multi_gate, "MultiGateTask", _FakeMulti)
    return artifacts_root


def _find(events: list[dict[str, Any]], topic: str) -> dict[str, Any] | None:
    return next((e for e in events if e.get("topic") == topic), None)


def test_single_gate_dispatch(_fake_training: Path) -> None:
    out = asyncio.run(
        _train_and_collect(
            {"gate": "AND", "max_trials": 5, "device": "cpu", "params": {"seed": 1}},
            _fake_training,
        )
    )
    run_end = _find(out["events"], "run_end")
    assert run_end is not None
    assert run_end["data"]["trials"] == 7
    assert "epochs" not in run_end["data"]  # single-gate branch


def test_multi_gate_dispatch(_fake_training: Path) -> None:
    out = asyncio.run(
        _train_and_collect(
            {"gate": "MULTI", "max_trials": 1, "device": "cpu", "params": {"seed": 1}},
            _fake_training,
        )
    )
    run_end = _find(out["events"], "run_end")
    assert run_end is not None
    assert run_end["data"]["epochs"] == 3  # multi-gate branch
    assert run_end["data"]["trials"] == 24


#


class _FakeTensor:
    """Minimal tensor-like object matching what ``_make_json_safe`` expects."""

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[float]:
        return [0.1, 0.2]


class _SnapMonitor:
    def __init__(self, snap: dict[str, Any]) -> None:
        self._snap = snap

    def snapshot(self) -> dict[str, Any]:
        return self._snap


async def _get_status() -> tuple[int, Any]:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    import neuroforge.interfaces.dashboard.server as srv

    app = web.Application()
    app.router.add_get("/api/status", srv._handle_status)
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.get("/api/status")
        body = await resp.json()
        return resp.status, body


def test_status_with_tensor_weights_is_json_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live snapshot embedding raw tensors must not 500 the status endpoint."""
    import neuroforge.interfaces.dashboard.server as srv

    training = _SnapMonitor({"topology": {"edges": [{"weights": _FakeTensor()}]}})
    monkeypatch.setattr(srv, "_training_monitor", training)
    monkeypatch.setattr(srv, "_weight_monitor", _SnapMonitor({}))
    monkeypatch.setattr(srv, "_spike_monitor", _SnapMonitor({}))

    status, data = asyncio.run(_get_status())
    assert status == 200
    assert data["training"]["topology"]["edges"][0]["weights"] == [0.1, 0.2]
