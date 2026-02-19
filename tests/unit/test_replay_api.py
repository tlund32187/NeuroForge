# pyright: basic
"""Unit tests for the dashboard replay API endpoints."""

from __future__ import annotations

import asyncio
import csv
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_RUN_ID = "run_20260101_120000_aabbccdd"


@pytest.fixture()
def _artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a fake artifacts directory and patch the server constant."""
    arts = tmp_path / "artifacts"
    arts.mkdir()

    run_dir = arts / _RUN_ID
    run_dir.mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "events").mkdir()

    (run_dir / "run_meta.json").write_text(
        json.dumps({
            "run_id": _RUN_ID,
            "started_at": "2026-01-01T12:00:00+00:00",
            "device": "cpu",
            "seed": 42,
        }),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.json").write_text(
        json.dumps({"gate": "XOR", "max_trials": 5000, "lr": 0.005}),
        encoding="utf-8",
    )
    (run_dir / "topology.json").write_text(
        json.dumps({
            "layers": ["input(2)", "hidden(6)", "output(1)"],
            "edges": [
                {"src": "input", "dst": "hidden", "weights": [[0.1, 0.2]]},
            ],
        }),
        encoding="utf-8",
    )

    csv_path = run_dir / "metrics" / "scalars.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "trial", "epoch", "gate", "accuracy", "loss", "correct", "wall_ms",
                "resource.cpu.system_percent",
                "resource.gpu.util_percent",
                "torch_cuda_allocated_mb",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "trial": 0, "epoch": 0, "gate": "XOR",
            "accuracy": 0.5, "loss": 1.0, "correct": "False", "wall_ms": "",
            "resource.cpu.system_percent": "21.5",
            "resource.gpu.util_percent": "",
            "torch_cuda_allocated_mb": "0.0",
        })
        writer.writerow({
            "trial": 1, "epoch": 0, "gate": "XOR",
            "accuracy": 0.75, "loss": 0.25, "correct": "True", "wall_ms": "",
            "resource.cpu.system_percent": "22.0",
            "resource.gpu.util_percent": "55.0",
            "torch_cuda_allocated_mb": "",
        })

    events_path = run_dir / "events" / "events.ndjson"
    events_path.write_text(
        "\n".join([
            json.dumps({
                "ts_wall": "2026-01-01T12:00:00+00:00",
                "run_id": _RUN_ID,
                "topic": "training_start",
                "source": "XOR",
                "step": 0,
                "t_step": 0,
                "t": 0.0,
                "data": {"gate": "XOR"},
            }),
            json.dumps({
                "ts_wall": "2026-01-01T12:00:01+00:00",
                "run_id": _RUN_ID,
                "topic": "training_trial",
                "source": "XOR",
                "step": 1,
                "t_step": 1,
                "t": 0.0,
                "data": {"accuracy": 0.5, "correct": False},
            }),
            json.dumps({
                "ts_wall": "2026-01-01T12:00:02+00:00",
                "run_id": _RUN_ID,
                "topic": "training_end",
                "source": "XOR",
                "step": 2,
                "t_step": 2,
                "t": 0.0,
                "data": {"converged": False},
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    # Patch the module-level _ARTIFACTS_DIR constant in the server.
    import neuroforge.dashboard.server as srv

    monkeypatch.setattr(srv, "_ARTIFACTS_DIR", arts)
    return arts


# ── Helpers ────────────────────────────────────────────────────────────


def _make_app() -> Any:
    """Build a minimal aiohttp app with replay routes."""
    from aiohttp import web

    import neuroforge.dashboard.server as srv

    app = web.Application()
    app.router.add_get("/api/runs", srv._handle_runs)
    app.router.add_get("/api/run/{run_id}/meta", srv._handle_run_meta)
    app.router.add_get("/api/run/{run_id}/config", srv._handle_run_config)
    app.router.add_get("/api/run/{run_id}/topology", srv._handle_run_topology)
    app.router.add_get("/api/run/{run_id}/scalars", srv._handle_run_scalars)
    app.router.add_get("/api/run/{run_id}/events", srv._handle_run_events)
    app.router.add_get("/api/run/{run_id}/events/index", srv._handle_run_events_index)
    return app


async def _get(path: str) -> tuple[int, Any]:
    """Issue a GET request via TestClient and return (status, json)."""
    from aiohttp.test_utils import TestClient, TestServer

    app = _make_app()
    async with TestClient(TestServer(app)) as tc:
        resp = await tc.get(path)
        body = await resp.json()
        return resp.status, body


# ── Synchronous _safe_run_dir tests ─────────────────────────────────


class TestRunIdValidation:
    """Test the _safe_run_dir helper for directory traversal prevention."""

    def test_valid_run_id(self, _artifacts: Path) -> None:
        from neuroforge.dashboard.server import _safe_run_dir

        result = _safe_run_dir(_RUN_ID)
        assert result is not None
        assert result.name == _RUN_ID

    def test_invalid_run_id_traversal(self, _artifacts: Path) -> None:
        from neuroforge.dashboard.server import _safe_run_dir

        assert _safe_run_dir("../etc/passwd") is None
        assert _safe_run_dir("run_../../etc") is None

    def test_invalid_run_id_format(self, _artifacts: Path) -> None:
        from neuroforge.dashboard.server import _safe_run_dir

        assert _safe_run_dir("not_a_run") is None
        assert _safe_run_dir("") is None
        assert _safe_run_dir("run_abc") is None

    def test_nonexistent_run_id(self, _artifacts: Path) -> None:
        from neuroforge.dashboard.server import _safe_run_dir

        assert _safe_run_dir("run_99991231_235959_deadbeef") is None


# ── Async endpoint tests (no pytest-asyncio needed) ────────────────


class TestReplayEndpoints:
    """Test replay API endpoints via aiohttp TestClient."""

    def test_runs_list(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get("/api/runs"))
        assert status == 200
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["run_id"] == _RUN_ID
        assert data[0]["device"] == "cpu"
        assert data[0]["seed"] == 42

    def test_run_meta(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get(f"/api/run/{_RUN_ID}/meta"))
        assert status == 200
        assert data["run_id"] == _RUN_ID
        assert data["seed"] == 42

    def test_run_config(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get(f"/api/run/{_RUN_ID}/config"))
        assert status == 200
        assert data["gate"] == "XOR"
        assert data["lr"] == 0.005

    def test_run_topology(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get(f"/api/run/{_RUN_ID}/topology"))
        assert status == 200
        assert "layers" in data
        assert data["layers"] == ["input(2)", "hidden(6)", "output(1)"]

    def test_run_scalars(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get(f"/api/run/{_RUN_ID}/scalars"))
        assert status == 200
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["gate"] == "XOR"
        assert data[0]["accuracy"] == 0.5
        assert data[1]["correct"] is True

    def test_run_scalars_with_optional_resource_columns(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get(f"/api/run/{_RUN_ID}/scalars"))
        assert status == 200
        assert data[0]["resource.cpu.system_percent"] == 21.5
        assert data[0]["resource.gpu.util_percent"] is None
        assert data[1]["resource.gpu.util_percent"] == 55.0
        assert data[1]["torch_cuda_allocated_mb"] is None

    def test_run_events(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get(f"/api/run/{_RUN_ID}/events?start=1&limit=1"))
        assert status == 200
        assert data["run_id"] == _RUN_ID
        assert data["total"] == 3
        assert len(data["events"]) == 1
        assert data["events"][0]["topic"] == "training_trial"

    def test_run_events_index(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, data = asyncio.run(_get(f"/api/run/{_RUN_ID}/events/index"))
        assert status == 200
        assert data["run_id"] == _RUN_ID
        assert data["count"] == 3
        assert data["items"][0]["idx"] == 0
        assert data["items"][0]["topic"] == "training_start"

    def test_invalid_run_returns_404(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, _data = asyncio.run(_get("/api/run/not_a_valid_run/meta"))
        assert status == 404

    def test_traversal_returns_404(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, _data = asyncio.run(_get("/api/run/..%2F..%2Fetc/meta"))
        assert status == 404

    def test_nonexistent_run_returns_404(self, _artifacts: Path) -> None:  # noqa: ARG002
        status, _data = asyncio.run(_get("/api/run/run_99991231_235959_deadbeef/meta"))
        assert status == 404
