"""ArtifactWriter - monitor that writes M0 run artifacts to disk.

Subscribes to monitor events and writes:
- run_meta.json (RUN_START)
- config_resolved.json (RUN_START)
- topology.json (TOPOLOGY, once)
- metrics/scalars.csv (SCALAR / TRAINING_TRIAL)
- training_end.json (RUN_END / TRAINING_END)
- logs/run.log (RUN_END)

SOLID boundary: tasks publish lightweight events; this monitor owns serialization.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["ArtifactWriter"]

_BASE_SCALAR_FIELDS = (
    "trial", "epoch", "gate", "accuracy", "loss", "correct", "wall_ms",
    "cuda_mem_allocated", "cuda_mem_reserved", "cuda_mem_peak",
)


def _json_default(obj: object) -> Any:
    """json.dumps fallback for tensors / Paths."""
    if hasattr(obj, "detach"):
        tensor: Any = obj
        return tensor.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _is_scalar_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if hasattr(value, "detach"):
        try:
            t = value.detach()
            return bool(getattr(t, "numel", lambda: 2)() == 1)
        except (TypeError, ValueError):
            return False
    return False


def _to_csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "detach"):
        try:
            return value.detach().item()
        except (TypeError, ValueError):
            return ""
    return ""


class ArtifactWriter:
    """Monitor that streams run artifacts to a run directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        enabled: bool = True,
        flush_every_n: int = 50,
    ) -> None:
        self.enabled = enabled
        self._run_dir = run_dir
        self._flush_every_n = max(1, int(flush_every_n))

        # CSV state.
        self._csv_fields: list[str] = list(_BASE_SCALAR_FIELDS)
        self._pending_rows: list[dict[str, Any]] = []
        self._csv_header_dirty = False
        self._csv_rows = 0

        self._topology_written = False
        self._log_lines: list[str] = []

    # IMonitor interface

    def on_event(self, event: MonitorEvent) -> None:
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
        self._csv_fields = list(_BASE_SCALAR_FIELDS)
        self._pending_rows.clear()
        self._csv_header_dirty = False
        self._csv_rows = 0
        self._topology_written = False
        self._log_lines.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a summary of what has been written so far."""
        return {
            "run_dir": str(self._run_dir),
            "csv_rows": self._csv_rows,
            "csv_fields": list(self._csv_fields),
            "topology_written": self._topology_written,
        }

    # Event handlers

    def _on_run_start(self, event: MonitorEvent) -> None:
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
        if self._topology_written:
            return
        (self._run_dir / "topology.json").write_text(
            json.dumps(event.data, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        self._topology_written = True

    def _on_scalar(self, event: MonitorEvent) -> None:
        row = self._build_scalar_row(event)

        new_fields = [k for k in row if k not in self._csv_fields]
        if new_fields:
            self._csv_fields.extend(new_fields)
            self._csv_header_dirty = True

        self._pending_rows.append(row)
        self._csv_rows += 1

        if self._csv_rows % self._flush_every_n == 0:
            self._flush_csv()

    def _on_run_end(self, event: MonitorEvent) -> None:
        self._flush_csv()

        (self._run_dir / "training_end.json").write_text(
            json.dumps(event.data, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

        if self._log_lines:
            (self._run_dir / "logs" / "run.log").write_text(
                "\n".join(self._log_lines) + "\n",
                encoding="utf-8",
            )

    # Utilities

    def _build_scalar_row(self, event: MonitorEvent) -> dict[str, Any]:
        d = event.data
        error = d.get("error", 0)
        loss_val = d.get("loss")
        if loss_val is None and error not in ("", None):
            try:
                loss_val = round(float(error) ** 2, 6)
            except (TypeError, ValueError):
                loss_val = ""

        row: dict[str, Any] = {
            "trial": _to_csv_scalar(d.get("trial", event.step)),
            "epoch": _to_csv_scalar(d.get("epoch", "")),
            "gate": _to_csv_scalar(d.get("gate", "")),
            "accuracy": _to_csv_scalar(d.get("accuracy", "")),
            "loss": _to_csv_scalar(loss_val),
            "correct": _to_csv_scalar(d.get("correct", "")),
            "wall_ms": _to_csv_scalar(d.get("wall_ms", "")),
            "cuda_mem_allocated": _to_csv_scalar(d.get("cuda_mem_allocated", "")),
            "cuda_mem_reserved": _to_csv_scalar(d.get("cuda_mem_reserved", "")),
            "cuda_mem_peak": _to_csv_scalar(d.get("cuda_mem_peak", "")),
        }

        for key, value in d.items():
            if key in row:
                continue
            if not _is_scalar_value(value):
                continue
            row[key] = _to_csv_scalar(value)

        return row

    def add_log_line(self, line: str) -> None:
        """Add a line to the in-memory log (written on RUN_END)."""
        self._log_lines.append(line)

    def flush(self) -> None:
        """Force-flush buffered scalar rows to disk."""
        self._flush_csv()

    def _flush_csv(self) -> None:
        """Write buffered scalar rows to metrics/scalars.csv."""
        if not self._pending_rows:
            return

        csv_path = self._run_dir / "metrics" / "scalars.csv"

        if not csv_path.exists() or self._csv_header_dirty:
            existing_rows: list[dict[str, Any]] = []
            if csv_path.exists():
                with csv_path.open(encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    if reader.fieldnames:
                        for field in reader.fieldnames:
                            if field not in self._csv_fields:
                                self._csv_fields.append(field)
                    for old_row in reader:
                        existing_rows.append(old_row)

            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._csv_fields)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow(self._normalise_row(old_row))
                for row in self._pending_rows:
                    writer.writerow(self._normalise_row(row))

            self._csv_header_dirty = False
            self._pending_rows.clear()
            return

        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._csv_fields)
            for row in self._pending_rows:
                writer.writerow(self._normalise_row(row))

        self._pending_rows.clear()

    def _normalise_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {field: _to_csv_scalar(row.get(field, "")) for field in self._csv_fields}
