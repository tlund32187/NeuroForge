"""EventRecorderMonitor - asynchronously persist monitor events as NDJSON."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import queue
import threading
from pathlib import Path
from typing import Any, cast

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["EventRecorderMonitor"]

_HEAVY_TOPICS = {EventTopic.WEIGHT, EventTopic.SPIKE, EventTopic.VOLTAGE}
_DEFAULT_SAMPLE_RULES: dict[str, int] = {
    "weight_every_k": 10,
    "spike_every_k": 10,
    "voltage_every_k": 10,
    "weight_max_values": 256,
    "spike_max_values": 256,
    "voltage_max_values": 256,
    "default_max_values": 512,
}


def _utc_iso_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat()


class EventRecorderMonitor:
    """Monitor that records events to ``run_dir/events/events.ndjson``."""

    def __init__(
        self,
        run_dir: Path,
        enabled: bool = True,
        flush_every: int = 50,
        max_queue: int = 10_000,
        sample_rules: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self._run_dir = run_dir
        self._run_id = run_dir.name
        self._events_dir = self._run_dir / "events"
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._events_dir / "events.ndjson"

        self._flush_every = max(1, int(flush_every))
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._sample_rules = dict(_DEFAULT_SAMPLE_RULES)
        if sample_rules:
            for key, value in sample_rules.items():
                with contextlib.suppress(TypeError, ValueError):
                    self._sample_rules[str(key)] = int(value)

        self._topology_written = False
        self._closed = False
        self._lock = threading.Lock()
        self._enqueued = 0
        self._written = 0
        self._dropped = 0

        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"event-recorder-{self._run_id}",
            daemon=True,
        )
        self._thread.start()

    def on_event(self, event: MonitorEvent) -> None:
        """Record an event if it passes sampling rules."""
        if not self.enabled or self._closed:
            return

        record = self._build_record(event)
        if record is None:
            return

        line = json.dumps(record, separators=(",", ":"), sort_keys=False)
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return

        with self._lock:
            self._enqueued += 1

        if event.topic in (EventTopic.TRAINING_END, EventTopic.RUN_END):
            self.flush()

    def reset(self) -> None:
        """Reset recorder state for topology-once tracking."""
        self._topology_written = False
        with self._lock:
            self._enqueued = 0
            self._written = 0
            self._dropped = 0

    def snapshot(self) -> dict[str, Any]:
        """Return write counters and output path."""
        with self._lock:
            enqueued = self._enqueued
            written = self._written
            dropped = self._dropped
        return {
            "run_id": self._run_id,
            "path": str(self._events_path),
            "enqueued": enqueued,
            "written": written,
            "dropped": dropped,
            "topology_written": self._topology_written,
        }

    def flush(self) -> None:
        """Block until all currently enqueued events are persisted."""
        self._queue.join()

    def close(self) -> None:
        """Flush and stop the background writer thread."""
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _writer_loop(self) -> None:
        pending = 0
        with self._events_path.open("a", encoding="utf-8") as fh:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        break
                    fh.write(item)
                    fh.write("\n")
                    pending += 1
                    with self._lock:
                        self._written += 1
                    if pending >= self._flush_every:
                        fh.flush()
                        pending = 0
                finally:
                    self._queue.task_done()

            if pending > 0:
                fh.flush()

    def _rule(self, key: str, default: int) -> int:
        raw = self._sample_rules.get(key, default)
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = default
        return max(1, val)

    def _should_record(self, topic: EventTopic, step: int) -> bool:
        if topic == EventTopic.SCALAR:
            return True
        if topic == EventTopic.TOPOLOGY:
            if self._topology_written:
                return False
            self._topology_written = True
            return True
        if topic in _HEAVY_TOPICS:
            key = f"{topic.value}_every_k"
            every_k = self._rule(key, 10)
            return (step % every_k) == 0
        return True

    def _build_record(self, event: MonitorEvent) -> dict[str, Any] | None:
        if not self._should_record(event.topic, event.step):
            return None

        return {
            "ts_wall": _utc_iso_now(),
            "run_id": self._run_id,
            "topic": event.topic.value,
            "source": event.source,
            "step": event.step,
            "t_step": event.step,
            "t": event.t,
            "data": self._serialise_payload(event.topic, event.data),
        }

    def _serialise_payload(
        self,
        topic: EventTopic,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if topic == EventTopic.WEIGHT:
            max_values = self._rule("weight_max_values", 256)
            return self._serialise_sampled_key(payload, "weights", max_values)
        if topic == EventTopic.SPIKE:
            max_values = self._rule("spike_max_values", 256)
            return self._serialise_sampled_key(payload, "spikes", max_values)
        if topic == EventTopic.VOLTAGE:
            max_values = self._rule("voltage_max_values", 256)
            return self._serialise_sampled_key(payload, "voltage", max_values)

        default_max = self._rule("default_max_values", 512)
        out: dict[str, Any] = {}
        for key, value in payload.items():
            out[str(key)] = self._small_json(value, default_max)
        return out

    def _serialise_sampled_key(
        self,
        payload: dict[str, Any],
        key_name: str,
        max_values: int,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if key != key_name:
                out[str(key)] = self._small_json(value, max_values)
                continue
            sampled, total, sampled_flag = self._sample_flat_values(value, max_values)
            out[key_name] = sampled
            out[f"{key_name}_numel"] = total
            out[f"{key_name}_sampled"] = sampled_flag
        if key_name not in out:
            out[key_name] = []
            out[f"{key_name}_numel"] = 0
            out[f"{key_name}_sampled"] = False
        return out

    def _sample_flat_values(
        self,
        value: Any,
        max_values: int,
    ) -> tuple[list[Any], int, bool]:
        max_values = max(1, int(max_values))
        flat = self._flatten_value(value)
        total = len(flat)
        if total <= max_values:
            return flat, total, False
        stride = max(1, total // max_values)
        sampled = flat[::stride][:max_values]
        return sampled, total, True

    def _flatten_value(self, value: Any) -> list[Any]:
        try:
            tensor = value.detach()
            tensor = tensor.reshape(-1).cpu()
            flat_tensor = tensor.tolist()
            if isinstance(flat_tensor, list):
                flat_list = cast("list[Any]", flat_tensor)  # type: ignore[redundant-cast]
                return [self._coerce_scalar(v) for v in flat_list]
            return [self._coerce_scalar(flat_tensor)]
        except AttributeError:
            pass

        if isinstance(value, dict):
            return [self._small_json(value, self._rule("default_max_values", 512))]

        if isinstance(value, (list, tuple)):
            out: list[Any] = []
            seq = cast("list[Any] | tuple[Any, ...]", value)  # type: ignore[redundant-cast]
            for item in seq:
                out.extend(self._flatten_value(item))
            return out

        return [self._coerce_scalar(value)]

    def _small_json(self, value: Any, max_values: int) -> Any:
        try:
            tensor = value.detach()
            sampled, total, sampled_flag = self._sample_flat_values(tensor, max_values)
            if sampled_flag:
                return {
                    "sample": sampled,
                    "numel": total,
                    "sampled": True,
                }
            return sampled
        except AttributeError:
            pass

        if isinstance(value, dict):
            mapping = cast("dict[Any, Any]", value)  # type: ignore[redundant-cast]
            return {str(k): self._small_json(v, max_values) for k, v in mapping.items()}
        if isinstance(value, (list, tuple)):
            seq = cast("list[Any] | tuple[Any, ...]", value)  # type: ignore[redundant-cast]
            if len(seq) <= max_values:
                return [self._small_json(v, max_values) for v in seq]
            sampled, total, _ = self._sample_flat_values(seq, max_values)
            return {
                "sample": sampled,
                "numel": total,
                "sampled": True,
            }
        return self._coerce_scalar(value)

    @staticmethod
    def _coerce_scalar(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        with contextlib.suppress(TypeError, ValueError):
            return int(value)
        with contextlib.suppress(TypeError, ValueError):
            return float(value)
        return str(value)
