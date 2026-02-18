"""ArtifactWriter — monitor that writes M0 run artifacts to disk.

Subscribes to events published by tasks and writes:

- ``run_meta.json``          — on ``RUN_START``
- ``config_resolved.json``   — on ``RUN_START``
- ``topology.json``          — on ``TOPOLOGY`` (once)
- ``metrics/scalars.csv``    — rows appended on ``SCALAR`` / ``TRAINING_TRIAL``
- ``training_end.json``      — on ``RUN_END`` / ``TRAINING_END``
- ``logs/run.log``           — on ``RUN_END``

**SOLID**: tasks only *publish* lightweight events (plain-python
scalars).  This monitor decides what to copy to CPU and how to
serialise.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["ArtifactWriter"]

_SCALAR_FIELDS = (
    "trial", "epoch", "gate", "accuracy", "loss", "correct", "wall_ms",
    "cuda_mem_allocated", "cuda_mem_reserved", "cuda_mem_peak",
)


def _json_default(obj: object) -> Any:
    """``json.dumps`` fallback for tensors / Paths."""
    if hasattr(obj, "detach"):
        tensor: Any = obj
        return tensor.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ArtifactWriter:
    """Monitor that streams M0 artifacts to a run directory.

    Parameters
    ----------
    run_dir:
        Absolute path to the run directory (must already exist
        with ``metrics/`` and ``logs/`` subdirectories).
    enabled:
        Whether the monitor is active.
    flush_every_n:
        Flush the CSV buffer to disk every *N* scalar rows.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        enabled: bool = True,
        flush_every_n: int = 50,
    ) -> None:
        self.enabled = enabled
        self._run_dir = run_dir
        self._flush_every_n = flush_every_n

        # CSV state — we buffer in a StringIO then flush periodically.
        self._csv_buf = StringIO()
        self._csv_writer = csv.DictWriter(
            self._csv_buf,
            fieldnames=list(_SCALAR_FIELDS),
        )
        self._csv_rows = 0
        self._csv_header_written = False

        # Topology written flag (write exactly once).
        self._topology_written = False

        # Log lines collected during the run.
        self._log_lines: list[str] = []

    # ── IMonitor interface ──────────────────────────────────────────

    def on_event(self, event: MonitorEvent) -> None:  # noqa: C901
        """Route an incoming event to the appropriate writer."""
        if not self.enabled:
            return

        topic = event.topic

        if topic == EventTopic.RUN_START:
            self._on_run_start(event)
        elif topic == EventTopic.TOPOLOGY:
            self._on_topology(event)
        elif topic in (EventTopic.SCALAR, EventTopic.TRAINING_TRIAL):
            self._on_scalar(event)
        elif topic in (EventTopic.RUN_END, EventTopic.TRAINING_END):
            self._on_run_end(event)

    def reset(self) -> None:
        """Clear internal buffers."""
        self._csv_buf = StringIO()
        self._csv_writer = csv.DictWriter(
            self._csv_buf, fieldnames=list(_SCALAR_FIELDS),
        )
        self._csv_rows = 0
        self._csv_header_written = False
        self._topology_written = False
        self._log_lines.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a summary of what has been written so far."""
        return {
            "run_dir": str(self._run_dir),
            "csv_rows": self._csv_rows,
            "topology_written": self._topology_written,
        }

    # ── Event handlers ──────────────────────────────────────────────

    def _on_run_start(self, event: MonitorEvent) -> None:
        """Write ``run_meta.json`` and ``config_resolved.json``."""
        data = event.data

        run_meta = data.get("run_meta")
        if run_meta is not None:
            (self._run_dir / "run_meta.json").write_text(
                json.dumps(run_meta, indent=2, default=_json_default) + "\n",
                encoding="utf-8",
            )

        config = data.get("config")
        if config is not None:
            (self._run_dir / "config_resolved.json").write_text(
                json.dumps(config, indent=2, default=_json_default) + "\n",
                encoding="utf-8",
            )

    def _on_topology(self, event: MonitorEvent) -> None:
        """Write ``topology.json`` (once)."""
        if self._topology_written:
            return
        (self._run_dir / "topology.json").write_text(
            json.dumps(event.data, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        self._topology_written = True

    def _on_scalar(self, event: MonitorEvent) -> None:
        """Append a row to ``metrics/scalars.csv``."""
        if not self._csv_header_written:
            self._csv_writer.writeheader()
            self._csv_header_written = True

        d = event.data
        error = d.get("error", 0)
        loss_val = d.get("loss")
        if loss_val is None and error != "":
            loss_val = round(float(error) ** 2, 6)

        self._csv_writer.writerow({
            "trial": d.get("trial", event.step),
            "epoch": d.get("epoch", ""),
            "gate": d.get("gate", ""),
            "accuracy": d.get("accuracy", ""),
            "loss": loss_val if loss_val is not None else "",
            "correct": d.get("correct", ""),
            "wall_ms": d.get("wall_ms", ""),
            "cuda_mem_allocated": d.get("cuda_mem_allocated", ""),
            "cuda_mem_reserved": d.get("cuda_mem_reserved", ""),
            "cuda_mem_peak": d.get("cuda_mem_peak", ""),
        })
        self._csv_rows += 1

        if self._csv_rows % self._flush_every_n == 0:
            self._flush_csv()

    def _on_run_end(self, event: MonitorEvent) -> None:
        """Flush CSV, write ``training_end.json`` and ``logs/run.log``."""
        self._flush_csv()

        # Write small summary.
        (self._run_dir / "training_end.json").write_text(
            json.dumps(event.data, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

        # Write log if any lines were collected.
        if self._log_lines:
            (self._run_dir / "logs" / "run.log").write_text(
                "\n".join(self._log_lines) + "\n",
                encoding="utf-8",
            )

    # ── Utilities ───────────────────────────────────────────────────

    def add_log_line(self, line: str) -> None:
        """Add a line to the in-memory log (written on RUN_END)."""
        self._log_lines.append(line)

    def flush(self) -> None:
        """Force-flush the CSV buffer to disk."""
        self._flush_csv()

    def _flush_csv(self) -> None:
        """Write buffered CSV text to ``metrics/scalars.csv``."""
        text = self._csv_buf.getvalue()
        if not text:
            return
        csv_path = self._run_dir / "metrics" / "scalars.csv"
        # Append if file already has content, else write fresh.
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            fh.write(text)
        # Reset the buffer (keep the writer pointing to a fresh StringIO).
        self._csv_buf = StringIO()
        self._csv_writer = csv.DictWriter(
            self._csv_buf, fieldnames=list(_SCALAR_FIELDS),
        )
