"""Scalar-header regression tests for ArtifactWriter."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.artifact_writer import ArtifactWriter

if TYPE_CHECKING:
    from pathlib import Path


def _event(topic: str, data: dict[str, object], *, step: int = 0) -> MonitorEvent:
    return MonitorEvent(
        topic=EventTopic(topic),
        step=step,
        t=0.0,
        source="TEST",
        data=data,
    )


def test_phase6_scalar_header_is_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_header"
    run_dir.mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "logs").mkdir()

    writer = ArtifactWriter(run_dir, flush_every_n=1)
    writer.on_event(
        _event(
            "scalar",
            {
                "trial": 0,
                "epoch": 0,
                "gate": "MULTI",
                "accuracy": 0.25,
                "loss": 0.75,
                "error": 1.0,
                "correct": False,
            },
            step=0,
        )
    )
    writer.flush()

    csv_path = run_dir / "metrics" / "scalars.csv"
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        header = set(reader.fieldnames)

    assert "trial" in header
    assert "accuracy" in header
    assert "loss" in header
    assert "rate_out_hz" in header
    assert "w_norm_ih" in header
    assert "stab_nan_inf" in header
