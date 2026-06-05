"""ArtifactWriter - monitor that writes M0 run artifacts to disk.

Subscribes to monitor events and writes:
- run_meta.json (RUN_START)
- config_resolved.json (RUN_START)
- topology.json (TOPOLOGY, once)
- topology/topology_stats.json (TOPOLOGY_STATS)
- metrics/scalars.csv (SCALAR / TRAINING_TRIAL)
- training_end.json (RUN_END / TRAINING_END)
- logs/run.log (RUN_END)

SOLID boundary: tasks publish lightweight events; this monitor owns serialization.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.messaging import EventTopic, MonitorEvent
from neuroforge.observability.artifacts.run_layout import RunLayout
from neuroforge.observability.artifacts.schemas import is_scalar_value, to_csv_scalar
from neuroforge.observability.artifacts.writer import JsonArtifactWriter
from neuroforge.observability.metrics.aggregation import bytes_to_mb
from neuroforge.observability.metrics.scalar_schema import build_scalar_fields

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ArtifactWriter"]


class ArtifactWriter:
    """Monitor that streams run artifacts to a run directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        enabled: bool = True,
        flush_every_n: int = 50,
        include_resource_fields: bool = False,
    ) -> None:
        self.enabled = enabled
        self._run_dir = run_dir
        self._layout = RunLayout(run_dir).ensure()
        self._json_writer = JsonArtifactWriter()
        self._flush_every_n = max(1, int(flush_every_n))
        self._include_resource_fields = include_resource_fields

        # CSV state.
        self._csv_fields: list[str] = build_scalar_fields(
            include_resource=self._include_resource_fields,
        )
        self._pending_rows: list[dict[str, Any]] = []
        self._csv_header_dirty = False
        self._csv_rows = 0

        self._topology_written = False
        self._topology_stats_written = False
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
        elif topic == EventTopic.TOPOLOGY_STATS:
            self._on_topology_stats(event)
        elif topic in (EventTopic.SCALAR, EventTopic.TRAINING_TRIAL):
            self._on_scalar(event)
        elif topic in (EventTopic.RUN_END, EventTopic.TRAINING_END):
            self._on_run_end(event)

    def reset(self) -> None:
        """Clear internal buffers."""
        self._csv_fields = build_scalar_fields(
            include_resource=self._include_resource_fields,
        )
        self._pending_rows.clear()
        self._csv_header_dirty = False
        self._csv_rows = 0
        self._topology_written = False
        self._topology_stats_written = False
        self._log_lines.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a summary of what has been written so far."""
        return {
            "run_dir": str(self._run_dir),
            "csv_rows": self._csv_rows,
            "csv_fields": list(self._csv_fields),
            "topology_written": self._topology_written,
            "topology_stats_written": self._topology_stats_written,
        }

    # Event handlers

    def _on_run_start(self, event: MonitorEvent) -> None:
        data = event.data

        run_meta = data.get("run_meta")
        if run_meta is not None:
            self._json_writer.write_json(self._layout.run_meta, run_meta)

        config = data.get("config")
        if config is not None:
            self._json_writer.write_json(self._layout.config_resolved, config)

    def _on_topology(self, event: MonitorEvent) -> None:
        if self._topology_written:
            return
        self._json_writer.write_json(self._layout.topology, event.data)
        self._topology_written = True

    def _on_topology_stats(self, event: MonitorEvent) -> None:
        self._json_writer.write_json(self._layout.topology_stats, event.data)
        self._topology_stats_written = True

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

        self._json_writer.write_json(self._layout.training_end, event.data)

        if self._log_lines:
            self._layout.run_log.write_text(
                "\n".join(self._log_lines) + "\n",
                encoding="utf-8",
            )

    # Utilities

    @staticmethod
    def _resolve_cuda_mb(
        data: dict[str, Any],
        *,
        mb_key: str,
        legacy_bytes_key: str,
    ) -> Any:
        if mb_key in data:
            return data.get(mb_key)
        raw = data.get(legacy_bytes_key)
        if raw is None:
            return ""
        if raw == "":
            return ""
        converted = bytes_to_mb(raw)
        return "" if converted is None else converted

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
            "trial": to_csv_scalar(d.get("trial", event.step)),
            "epoch": to_csv_scalar(d.get("epoch", "")),
            "gate": to_csv_scalar(d.get("gate", "")),
            "accuracy": to_csv_scalar(d.get("accuracy", "")),
            "loss": to_csv_scalar(loss_val),
            "error": to_csv_scalar(d.get("error", "")),
            "correct": to_csv_scalar(d.get("correct", "")),
            "wall_ms": to_csv_scalar(d.get("wall_ms", "")),
            "cuda_mem_allocated_mb": to_csv_scalar(
                self._resolve_cuda_mb(
                    d,
                    mb_key="cuda_mem_allocated_mb",
                    legacy_bytes_key="cuda_mem_allocated",
                )
            ),
            "cuda_mem_reserved_mb": to_csv_scalar(
                self._resolve_cuda_mb(
                    d,
                    mb_key="cuda_mem_reserved_mb",
                    legacy_bytes_key="cuda_mem_reserved",
                )
            ),
            "cuda_mem_peak_mb": to_csv_scalar(
                self._resolve_cuda_mb(
                    d,
                    mb_key="cuda_mem_peak_mb",
                    legacy_bytes_key="cuda_mem_peak",
                )
            ),
        }

        for field in self._csv_fields:
            if field in row:
                continue
            row[field] = to_csv_scalar(d.get(field, ""))

        for key, value in d.items():
            if key in row:
                continue
            if not is_scalar_value(value):
                continue
            row[key] = to_csv_scalar(value)

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

        csv_path = self._layout.scalars

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
        return {field: to_csv_scalar(row.get(field, "")) for field in self._csv_fields}
