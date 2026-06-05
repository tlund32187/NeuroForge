"""TopologyActivityMonitor - compact runtime traces for topology views.

The monitor listens to existing pub/sub events and derives a bounded,
JSON-friendly view of what is currently active in the network. It deliberately
publishes projection/layer summaries instead of full spike paths so live
dashboards and replay artifacts stay small.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from neuroforge.contracts.messaging import EventTopic, MonitorEvent

if TYPE_CHECKING:
    from neuroforge.contracts.messaging import IEventBus

__all__ = ["TopologyActivityMonitor"]


def _empty_float_list() -> list[float]:
    """Typed default factory for a list of sampled weights."""
    return []


def _empty_edge_list() -> list[dict[str, Any]]:
    """Typed default factory for a list of sampled edges."""
    return []


@dataclass(slots=True)
class _LayerSummary:
    name: str
    size: int = 0
    latest_spikes: float = 0.0
    active_count: int = 0
    activity: float = 0.0
    mean_activation: float | None = None
    max_activation: float | None = None


@dataclass(slots=True)
class _ProjectionSummary:
    name: str
    src: str = ""
    dst: str = ""
    n_pre: int = 0
    n_post: int = 0
    n_edges: int = 0
    dense: bool = False
    topology_type: str = "unknown"
    dtype: str = "float32"
    active_edge_count: int = 0
    weight_abs_mean: float | None = None
    weight_delta_mean: float | None = None
    weight_sample: list[float] = field(default_factory=_empty_float_list)
    weight_numel: int = 0
    sample_edges: list[dict[str, Any]] = field(default_factory=_empty_edge_list)


class TopologyActivityMonitor:
    """Derive bounded topology trace events from existing monitor traffic."""

    def __init__(
        self,
        *,
        event_bus: IEventBus | None = None,
        enabled: bool = True,
        trace_every_n_steps: int = 5,
        max_edge_samples: int = 64,
        max_projection_entries: int = 32,
        max_weight_samples: int = 512,
        max_trace_frames: int = 200,
    ) -> None:
        self.enabled = enabled
        self._bus = event_bus
        self._trace_every = max(1, int(trace_every_n_steps))
        self._max_edge_samples = max(1, int(max_edge_samples))
        self._max_projection_entries = max(1, int(max_projection_entries))
        self._max_weight_samples = max(1, int(max_weight_samples))
        self._traces: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_trace_frames)))
        self._layers: dict[str, _LayerSummary] = {}
        self._layer_order: list[str] = []
        self._projections: dict[str, _ProjectionSummary] = {}
        self._projection_order: list[str] = []
        self._active_indices: dict[str, list[int]] = {}
        self._topology_ready = False
        self._last_emit_step: dict[str, int] = {}

    def on_event(self, event: MonitorEvent) -> None:
        """Ingest topology-relevant events and publish bounded trace frames."""
        if not self.enabled or event.topic == EventTopic.TOPOLOGY_TRACE:
            return

        channel: str | None = None
        if event.topic == EventTopic.TOPOLOGY:
            self._ingest_topology(event.data)
            return
        if event.topic == EventTopic.TOPOLOGY_STATS:
            self._ingest_topology_stats(event.data)
            return
        if event.topic == EventTopic.TRAINING_TRIAL:
            self._ingest_training_trial(event.data)
            channel = "activity"
        elif event.topic == EventTopic.SPIKE:
            self._ingest_spike(event.source, event.data)
            channel = "activity"
        elif event.topic == EventTopic.SCALAR:
            self._ingest_scalar(event.data)
            channel = "activity"
        elif event.topic == EventTopic.WEIGHT:
            self._ingest_weight(event.source, event.data)
            channel = "weight"
        else:
            return

        # Every non-returning branch above set ``channel``.
        self._maybe_emit(event, channel)

    def reset(self) -> None:
        """Clear all derived topology activity state."""
        self._traces.clear()
        self._layers.clear()
        self._layer_order.clear()
        self._projections.clear()
        self._projection_order.clear()
        self._active_indices.clear()
        self._topology_ready = False
        self._last_emit_step.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return the latest compact topology traces."""
        return {
            "enabled": self.enabled,
            "topology_ready": self._topology_ready,
            "latest": self._traces[-1] if self._traces else None,
            "traces": list(self._traces),
        }

    def _ingest_topology(self, data: dict[str, Any]) -> None:
        self._layers.clear()
        self._layer_order.clear()
        self._projections.clear()
        self._projection_order.clear()
        self._active_indices.clear()

        details = self._as_list(data.get("layer_details"))
        if details:
            for raw in details:
                item = self._as_dict(raw)
                if item is None:
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                size = self._safe_int(item.get("neurons"), 0)
                self._ensure_layer(name, size=size)
        else:
            for raw in self._as_list(data.get("layers")):
                name, size = self._parse_layer_label(raw)
                if name:
                    self._ensure_layer(name, size=size)

        projection_meta = self._as_list(data.get("projection_meta"))
        if projection_meta:
            for raw in projection_meta:
                item = self._as_dict(raw)
                if item is not None:
                    self._upsert_projection(item)
        else:
            for raw in self._as_list(data.get("edges")):
                item = self._as_dict(raw)
                if item is not None:
                    self._upsert_projection(item)

        self._topology_ready = bool(self._layers)

    def _ingest_topology_stats(self, data: dict[str, Any]) -> None:
        for raw in self._as_list(data.get("projections")):
            item = self._as_dict(raw)
            if item is not None:
                self._upsert_projection(item)

    def _ingest_training_trial(self, data: dict[str, Any]) -> None:
        layer_keys = {
            "input": data.get("input_spikes"),
            "hidden": data.get("hidden_spikes"),
            "output": data.get("output_spikes"),
        }
        for layer, values in layer_keys.items():
            if values is not None:
                self._update_layer_values(layer, values)

    def _ingest_spike(self, source: str, data: dict[str, Any]) -> None:
        layer = self._layer_from_source(source)
        if not layer:
            return
        values = data.get("spikes")
        if values is not None:
            self._update_layer_values(layer, values)

    def _ingest_scalar(self, data: dict[str, Any]) -> None:
        prefix = "vision.layer."
        for key, raw in data.items():
            key_s = str(key)
            if not key_s.startswith(prefix):
                continue
            tail = key_s[len(prefix):]
            idx = tail.rfind(".")
            if idx <= 0:
                continue
            layer = tail[:idx]
            metric = tail[idx + 1:]
            value = self._safe_float(raw)
            if value is None:
                continue
            summary = self._ensure_layer(layer)
            if metric == "spike_rate":
                summary.activity = max(0.0, min(1.0, float(value)))
                summary.active_count = int(round(summary.activity * max(1, summary.size)))
                summary.latest_spikes = float(summary.active_count)
            elif metric == "mean_activation":
                summary.mean_activation = float(value)
            elif metric == "max_activation":
                summary.max_activation = float(value)

    def _ingest_weight(self, source: str, data: dict[str, Any]) -> None:
        weights = data.get("weights")
        projection = self._projection_from_source(source)
        if projection is None or weights is None:
            return

        sample, total = self._sample_numeric_values(weights, self._max_weight_samples)
        if not sample:
            return

        prev = projection.weight_abs_mean
        mean_abs = sum(abs(v) for v in sample) / len(sample)
        projection.weight_abs_mean = float(mean_abs)
        projection.weight_delta_mean = (
            None if prev is None else float(mean_abs - prev)
        )
        projection.weight_sample = sample
        projection.weight_numel = int(total)
        if projection.n_edges <= 0:
            projection.n_edges = int(total)
        projection.sample_edges = self._build_projection_edges(projection)

    def _maybe_emit(self, event: MonitorEvent, channel: str) -> None:
        if not self._topology_ready or self._bus is None:
            return
        if int(event.step) % self._trace_every != 0:
            return
        if self._last_emit_step.get(channel) == int(event.step):
            return

        payload = self._build_trace_payload(event, channel)
        self._traces.append(payload)
        self._last_emit_step[channel] = int(event.step)
        self._bus.publish(
            MonitorEvent(
                topic=EventTopic.TOPOLOGY_TRACE,
                step=int(event.step),
                t=float(event.t),
                source="topology_activity_monitor",
                data=payload,
            )
        )

    def _build_trace_payload(self, event: MonitorEvent, mode: str) -> dict[str, Any]:
        projections = [self._projections[name] for name in self._projection_order]
        projections.sort(
            key=lambda p: (
                int(p.active_edge_count),
                float(abs(p.weight_delta_mean or 0.0)),
                float(p.weight_abs_mean or 0.0),
            ),
            reverse=True,
        )
        projections = projections[: self._max_projection_entries]

        return {
            "step": int(event.step),
            "t": float(event.t),
            "mode": mode,
            "layers": [
                {
                    "name": layer.name,
                    "latest_spikes": float(layer.latest_spikes),
                    "active_count": int(layer.active_count),
                    "activity": float(layer.activity),
                    "mean_activation": layer.mean_activation,
                    "max_activation": layer.max_activation,
                }
                for layer in (self._layers[name] for name in self._layer_order)
            ],
            "projections": [
                {
                    "name": proj.name,
                    "src": proj.src,
                    "dst": proj.dst,
                    "active_edge_count": int(proj.active_edge_count),
                    "weight_abs_mean": proj.weight_abs_mean,
                    "weight_delta_mean": proj.weight_delta_mean,
                    "sample_edges": list(proj.sample_edges[: self._max_edge_samples]),
                }
                for proj in projections
            ],
        }

    def _ensure_layer(self, name: str, *, size: int | None = None) -> _LayerSummary:
        if name not in self._layers:
            self._layers[name] = _LayerSummary(name=name, size=max(0, int(size or 0)))
            self._layer_order.append(name)
        elif size is not None and int(size) > 0:
            self._layers[name].size = int(size)
        return self._layers[name]

    def _upsert_projection(self, raw: dict[str, Any]) -> _ProjectionSummary:
        src = str(raw.get("src", raw.get("source", "")))
        dst = str(raw.get("dst", raw.get("target", "")))
        name = str(raw.get("name") or raw.get("projection") or f"{src}->{dst}")
        name = self._normalise_projection_name(name)

        if name not in self._projections:
            self._projections[name] = _ProjectionSummary(name=name)
            self._projection_order.append(name)

        proj = self._projections[name]
        proj.src = src or proj.src
        proj.dst = dst or proj.dst
        proj.n_pre = self._safe_int(raw.get("n_pre"), proj.n_pre)
        proj.n_post = self._safe_int(raw.get("n_post"), proj.n_post)
        proj.n_edges = self._safe_int(raw.get("n_edges"), proj.n_edges)
        proj.dense = bool(raw.get("dense", proj.dense))
        proj.dtype = str(raw.get("dtype", proj.dtype))
        proj.topology_type = str(raw.get("topology_type", proj.topology_type))
        return proj

    def _projection_from_source(self, source: str) -> _ProjectionSummary | None:
        candidates = [
            str(source),
            str(source).split("/")[-1],
            str(source).split("/")[-1].replace("-", "_"),
        ]
        for candidate in candidates:
            norm = self._normalise_projection_name(candidate)
            if norm in self._projections:
                return self._projections[norm]

        tail = self._normalise_projection_name(str(source).split("/")[-1])
        parts = [p for p in tail.replace("->", "_").split("_") if p]
        src = parts[0] if len(parts) >= 2 else ""
        dst = parts[-1] if len(parts) >= 2 else ""
        if not tail:
            return None
        return self._upsert_projection({"name": tail, "src": src, "dst": dst})

    def _build_projection_edges(self, proj: _ProjectionSummary) -> list[dict[str, Any]]:
        pre_active = self._active_indices.get(proj.src, [])
        post_active = self._active_indices.get(proj.dst, [])
        edges: list[dict[str, Any]] = []

        if pre_active and post_active:
            for pre in pre_active:
                for post in post_active:
                    if len(edges) >= self._max_edge_samples:
                        break
                    weight = self._sample_weight_for_dense_pair(proj, pre, post)
                    edges.append(
                        {
                            "pre": int(pre),
                            "post": int(post),
                            "weight": weight,
                            "signal": self._edge_signal(weight, 1.0),
                        }
                    )
                if len(edges) >= self._max_edge_samples:
                    break
            proj.active_edge_count = int(len(pre_active) * len(post_active))
            return edges

        proj.active_edge_count = 0
        if not proj.weight_sample:
            return edges

        n_post = max(1, proj.n_post)
        for idx, weight in enumerate(proj.weight_sample[: self._max_edge_samples]):
            edges.append(
                {
                    "pre": int(idx // n_post),
                    "post": int(idx % n_post),
                    "weight": float(weight),
                    "signal": self._edge_signal(float(weight), 0.35),
                }
            )
        return edges

    def _sample_weight_for_dense_pair(
        self,
        proj: _ProjectionSummary,
        pre: int,
        post: int,
    ) -> float | None:
        if not proj.weight_sample:
            return None
        if proj.dense and proj.n_post > 0:
            idx = int(pre) * int(proj.n_post) + int(post)
            if 0 <= idx < len(proj.weight_sample):
                return float(proj.weight_sample[idx])
        idx = (int(pre) + int(post)) % len(proj.weight_sample)
        return float(proj.weight_sample[idx])

    @staticmethod
    def _edge_signal(weight: float | None, activity: float) -> float:
        if weight is None:
            return float(max(0.0, min(1.0, activity)))
        return float(max(0.0, min(1.0, abs(weight) * max(0.1, activity))))

    def _update_layer_values(self, layer_name: str, values: Any) -> None:
        vals = self._sample_numeric_values(values, self._max_weight_samples)[0]
        if not vals:
            return
        summary = self._ensure_layer(layer_name, size=len(vals))
        active = [idx for idx, value in enumerate(vals) if float(value) > 0.0]
        active = active[: self._max_edge_samples]
        total_spikes = sum(float(v) for v in vals)
        summary.latest_spikes = float(total_spikes)
        summary.active_count = len(active)
        summary.activity = float(len(active) / max(1, len(vals)))
        summary.mean_activation = float(total_spikes / max(1, len(vals)))
        summary.max_activation = float(max(vals))
        self._active_indices[layer_name] = active

        for proj in self._projections.values():
            if proj.src == layer_name or proj.dst == layer_name:
                proj.sample_edges = self._build_projection_edges(proj)

    @staticmethod
    def _layer_from_source(source: str) -> str:
        return str(source).split("/")[-1].strip()

    @staticmethod
    def _normalise_projection_name(name: str) -> str:
        return str(name).strip().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _parse_layer_label(value: Any) -> tuple[str, int]:
        label = str(value)
        if "(" not in label or not label.endswith(")"):
            return label, 0
        name, _, tail = label.partition("(")
        count_s = tail[:-1]
        try:
            return name, int(count_s)
        except ValueError:
            parts = [p for p in count_s.replace("X", "x").split("x") if p]
            product = 1
            for part in parts:
                try:
                    product *= int(part)
                except ValueError:
                    return name, 0
            return name, product

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            # pyright narrows Any-of-list to Unknown and needs the cast; mypy
            # considers it redundant — suppress only mypy's redundant-cast.
            return cast("list[Any]", value)  # type: ignore[redundant-cast]
        return []

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        source = cast("dict[Any, Any]", value)  # type: ignore[redundant-cast]
        return {str(k): v for k, v in source.items()}

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out

    def _sample_numeric_values(self, value: Any, max_values: int) -> tuple[list[float], int]:
        max_values = max(1, int(max_values))
        if value is None:
            return [], 0

        try:
            tensor = value.detach().reshape(-1)
            total = int(tensor.numel())
            if total == 0:
                return [], 0
            if total <= max_values:
                raw = tensor.cpu().tolist()
            else:
                import torch

                idx = torch.linspace(
                    0,
                    total - 1,
                    steps=max_values,
                    device=tensor.device,
                ).to(dtype=torch.long)
                raw = tensor.index_select(0, idx).cpu().tolist()
            return [float(v) for v in raw], total
        except AttributeError:
            pass

        out: list[float] = []
        total = self._flatten_numeric_sequence(value, out, max_values)
        return out, total

    def _flatten_numeric_sequence(
        self,
        value: Any,
        out: list[float],
        max_values: int,
    ) -> int:
        if isinstance(value, dict):
            return 0
        if isinstance(value, (list, tuple)):
            total = 0
            for item in cast("list[Any]", value):
                total += self._flatten_numeric_sequence(item, out, max_values)
            return total
        parsed = self._safe_float(value)
        if parsed is None:
            return 0
        if len(out) < max_values:
            out.append(float(parsed))
        return 1
