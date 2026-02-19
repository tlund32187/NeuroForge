"""TopologyStatsMonitor - one-shot edge/memory estimates from topology metadata.

Consumes lightweight topology metadata (preferably ``projection_meta``)
and emits:
- a ``SCALAR`` event with aggregate ``topology.*`` keys
- a ``TOPOLOGY_STATS`` event with per-projection + total estimates

No tensor/file I/O is performed here; this monitor is computation-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

if TYPE_CHECKING:
    from neuroforge.contracts.monitors import IEventBus

__all__ = ["TopologyStatsMonitor"]


class TopologyStatsMonitor:
    """Compute one-shot topology stats from lightweight metadata."""

    def __init__(
        self,
        *,
        event_bus: IEventBus | None = None,
        enabled: bool = True,
        default_float_bytes: int = 4,
    ) -> None:
        self.enabled = enabled
        self._bus = event_bus
        self._default_float_bytes = max(1, int(default_float_bytes))
        self._emitted = False
        self._last_stats: dict[str, Any] = {}

    def on_event(self, event: MonitorEvent) -> None:
        """Handle topology-bearing events and emit one-shot derived stats."""
        if not self.enabled or self._emitted:
            return
        if event.topic not in (EventTopic.TOPOLOGY, EventTopic.TRAINING_START):
            return

        projection_meta = self._extract_projection_meta(event.data)
        if not projection_meta:
            return

        stats = self._compute_stats(projection_meta)
        self._last_stats = stats
        self._emitted = True

        # Make totals available on the topology payload itself for UI consumers.
        if event.topic == EventTopic.TOPOLOGY:
            event.data["projection_meta"] = projection_meta
            event.data["topology_stats"] = dict(stats["totals"])

        if self._bus is None:
            return

        totals = stats["totals"]
        self._bus.publish(
            MonitorEvent(
                topic=EventTopic.SCALAR,
                step=int(event.step),
                t=float(event.t),
                source="topology_stats_monitor",
                data={
                    "trial": int(event.step),
                    "topology.edges_total": int(totals["edges_total"]),
                    "topology.bytes_total_est": int(totals["bytes_total_est"]),
                    "topology.bytes_dense_total_est": int(totals["bytes_dense_total_est"]),
                    "topology.projections": int(totals["projection_count"]),
                },
            )
        )
        self._bus.publish(
            MonitorEvent(
                topic=EventTopic.TOPOLOGY_STATS,
                step=int(event.step),
                t=float(event.t),
                source="topology_stats_monitor",
                data=stats,
            )
        )

    def reset(self) -> None:
        """Reset one-shot state."""
        self._emitted = False
        self._last_stats = {}

    def snapshot(self) -> dict[str, Any]:
        """Return monitor status and the last computed stats."""
        return {
            "enabled": self.enabled,
            "emitted": self._emitted,
            "last_stats": self._last_stats,
        }

    def _extract_projection_meta(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._as_object_list(data.get("projection_meta")):
            item_dict = self._as_str_key_dict(item)
            if item_dict is None:
                continue
            out.append(self._normalise_projection_meta(item_dict))
        if out:
            return out

        # Fallback for older payloads: derive from topology edges + layer labels.
        layer_counts = self._layer_count_map(data.get("layers"))
        edges = self._as_object_list(data.get("edges"))
        if not edges:
            return []

        derived: list[dict[str, Any]] = []
        for idx, edge_raw in enumerate(edges):
            edge = self._as_str_key_dict(edge_raw)
            if edge is None:
                continue
            src = str(edge.get("src", ""))
            dst = str(edge.get("dst", ""))
            n_pre = self._safe_int(edge.get("n_pre"), layer_counts.get(src, 0))
            n_post = self._safe_int(edge.get("n_post"), layer_counts.get(dst, 0))
            dense = bool(
                edge.get("dense", False)
                or str(edge.get("topology_type", "")).lower() == "dense"
            )
            n_edges = self._safe_int(
                edge.get("n_edges"),
                n_pre * n_post if dense else self._edge_count_from_weights(edge.get("weights")),
            )
            derived.append(
                {
                    "name": str(
                        edge.get("name") or edge.get("projection") or f"{src}->{dst}#{idx}"
                    ),
                    "src": src,
                    "dst": dst,
                    "n_pre": int(n_pre),
                    "n_post": int(n_post),
                    "n_edges": int(n_edges),
                    "dense": dense,
                    "dtype": str(edge.get("dtype") or "float32"),
                    "topology_type": str(
                        edge.get("topology_type") or ("dense" if dense else "sparse")
                    ),
                }
            )
        return derived

    @staticmethod
    def _layer_count_map(layers_raw: Any) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in TopologyStatsMonitor._as_object_list(layers_raw):
            s = str(entry)
            if "(" not in s or not s.endswith(")"):
                continue
            name, _, tail = s.partition("(")
            count_s = tail[:-1]
            try:
                out[name] = int(count_s)
            except ValueError:
                continue
        return out

    @staticmethod
    def _edge_count_from_weights(weights: Any) -> int:
        if weights is None:
            return 0
        if hasattr(weights, "numel"):
            try:
                return int(weights.numel())
            except (TypeError, ValueError):
                return 0
        if isinstance(weights, list):
            rows = TopologyStatsMonitor._as_object_list(weights)
            if not rows:
                return 0
            if isinstance(rows[0], list):
                total = 0
                for row in rows:
                    total += len(TopologyStatsMonitor._as_object_list(row))
                return total
            return len(rows)
        return 0

    @staticmethod
    def _as_object_list(value: Any) -> list[object]:
        if not isinstance(value, list):
            return []
        return list(cast("list[object]", value))

    @staticmethod
    def _as_str_key_dict(value: object) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        value_dict = cast("dict[object, object]", value)
        out: dict[str, Any] = {}
        for key, item in value_dict.items():
            out[str(key)] = item
        return out

    def _normalise_projection_meta(self, raw: dict[str, Any]) -> dict[str, Any]:
        dense_val = raw.get("dense", False)
        dense = bool(dense_val)
        return {
            "name": str(raw.get("name", "")),
            "src": str(raw.get("src", "")),
            "dst": str(raw.get("dst", "")),
            "n_pre": self._safe_int(raw.get("n_pre"), 0),
            "n_post": self._safe_int(raw.get("n_post"), 0),
            "n_edges": self._safe_int(raw.get("n_edges"), 0),
            "dense": dense,
            "dtype": str(raw.get("dtype", "float32")),
            "topology_type": str(raw.get("topology_type", "dense" if dense else "sparse")),
        }

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _dtype_nbytes(self, dtype_name: str) -> int:
        name = str(dtype_name).replace("torch.", "").lower()
        mapping = {
            "float16": 2,
            "half": 2,
            "bfloat16": 2,
            "float32": 4,
            "float": 4,
            "float64": 8,
            "double": 8,
        }
        return int(mapping.get(name, self._default_float_bytes))

    def _compute_stats(self, projection_meta: list[dict[str, Any]]) -> dict[str, Any]:
        projections_out: list[dict[str, Any]] = []

        edges_total = 0
        bytes_idx_total = 0
        bytes_delays_total = 0
        bytes_weights_total = 0
        bytes_dense_total = 0
        bytes_total = 0

        for meta in projection_meta:
            n_pre = self._safe_int(meta.get("n_pre"), 0)
            n_post = self._safe_int(meta.get("n_post"), 0)
            n_edges = self._safe_int(meta.get("n_edges"), 0)
            dense = bool(meta.get("dense", False))
            dtype = str(meta.get("dtype", "float32"))
            float_bytes = self._dtype_nbytes(dtype)

            bytes_idx = n_edges * 8 * 2
            bytes_delays = n_edges * 8
            bytes_weights = n_edges * float_bytes
            bytes_dense = (n_pre * n_post * float_bytes) if dense else 0
            bytes_proj_total = bytes_idx + bytes_delays + bytes_weights + bytes_dense

            edges_total += n_edges
            bytes_idx_total += bytes_idx
            bytes_delays_total += bytes_delays
            bytes_weights_total += bytes_weights
            bytes_dense_total += bytes_dense
            bytes_total += bytes_proj_total

            projections_out.append(
                {
                    "name": str(meta.get("name", "")),
                    "src": str(meta.get("src", "")),
                    "dst": str(meta.get("dst", "")),
                    "topology_type": str(
                        meta.get("topology_type", "dense" if dense else "sparse")
                    ),
                    "dtype": dtype,
                    "dense": dense,
                    "n_pre": int(n_pre),
                    "n_post": int(n_post),
                    "n_edges": int(n_edges),
                    "bytes_idx_est": int(bytes_idx),
                    "bytes_delays_est": int(bytes_delays),
                    "bytes_weights_est": int(bytes_weights),
                    "bytes_dense_matrix_est": int(bytes_dense),
                    "bytes_total_est": int(bytes_proj_total),
                }
            )

        return {
            "projections": projections_out,
            "totals": {
                "projection_count": int(len(projections_out)),
                "edges_total": int(edges_total),
                "bytes_idx_total_est": int(bytes_idx_total),
                "bytes_delays_total_est": int(bytes_delays_total),
                "bytes_weights_total_est": int(bytes_weights_total),
                "bytes_dense_total_est": int(bytes_dense_total),
                "bytes_total_est": int(bytes_total),
            },
        }
