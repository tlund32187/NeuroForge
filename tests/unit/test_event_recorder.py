# pyright: basic
"""Unit tests for the EventRecorderMonitor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.event_recorder import EventRecorderMonitor

if TYPE_CHECKING:
    from pathlib import Path


def _event(
    topic: EventTopic,
    *,
    step: int = 0,
    source: str = "test",
    data: dict[str, Any] | None = None,
) -> MonitorEvent:
    return MonitorEvent(topic=topic, step=step, t=0.0, source=source, data=data or {})


def test_event_recorder_writes_ndjson(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260101_120000_aabbccdd"
    run_dir.mkdir(parents=True)

    recorder = EventRecorderMonitor(
        run_dir,
        flush_every=1,
        sample_rules={
            "weight_every_k": 2,
            "weight_max_values": 3,
        },
    )

    recorder.on_event(_event(EventTopic.RUN_START, data={"ok": True}))
    recorder.on_event(_event(EventTopic.TOPOLOGY, data={"layers": ["input(2)", "output(1)"]}))
    recorder.on_event(_event(EventTopic.TOPOLOGY, data={"layers": ["ignored"]}))
    recorder.on_event(_event(EventTopic.SCALAR, step=1, data={"accuracy": 0.75}))
    recorder.on_event(_event(EventTopic.WEIGHT, step=1, data={"weights": [1, 2, 3, 4]}))
    recorder.on_event(_event(EventTopic.WEIGHT, step=2, data={"weights": [1, 2, 3, 4]}))
    recorder.on_event(_event(EventTopic.RUN_END, step=3, data={"done": True}))
    recorder.close()

    out_path = run_dir / "events" / "events.ndjson"
    assert out_path.is_file()

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    topics = [row["topic"] for row in rows]
    assert topics.count("topology") == 1
    assert "scalar" in topics
    assert "weight" in topics

    weight_rows = [r for r in rows if r["topic"] == "weight"]
    assert len(weight_rows) == 1
    assert weight_rows[0]["step"] == 2
    assert weight_rows[0]["data"]["weights"] == [1, 2, 3]
    assert weight_rows[0]["data"]["weights_numel"] == 4
    assert weight_rows[0]["data"]["weights_sampled"] is True

    for row in rows:
        assert row["run_id"] == "run_20260101_120000_aabbccdd"
        assert "ts_wall" in row
        assert "t_step" in row
