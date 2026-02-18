"""WeightMonitor — records synaptic weight snapshots.

Captures weight tensors on ``EventTopic.WEIGHT`` events. Stores a
bounded history of weight snapshots per projection.

CUDA-compatible: tensors are moved to CPU before storage.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["WeightMonitor"]


class WeightMonitor:
    """Records weight snapshots emitted on ``EventTopic.WEIGHT``.

    Parameters
    ----------
    max_window:
        Maximum number of snapshots to keep per projection.
    enabled:
        Whether the monitor is active.
    """

    def __init__(
        self, *, enabled: bool = True, max_window: int = 200
    ) -> None:
        self.enabled = enabled
        self._max_window = max_window
        self._data: dict[str, dict[str, deque[Any]]] = defaultdict(
            lambda: {"steps": deque(maxlen=self._max_window),
                     "weights": deque(maxlen=self._max_window)}
        )

    # ── IMonitor implementation ─────────────────────────────────────

    def on_event(self, event: MonitorEvent) -> None:
        """Record a weight snapshot from a WEIGHT event."""
        if not self.enabled or event.topic != EventTopic.WEIGHT:
            return

        weights_raw = event.data.get("weights")
        if weights_raw is None:
            return

        try:
            weights_list: list[Any] = weights_raw.detach().cpu().tolist()
        except AttributeError:
            weights_list = list(weights_raw)

        rec = self._data[event.source]
        rec["steps"].append(event.step)
        rec["weights"].append(weights_list)

    def reset(self) -> None:
        """Clear all recorded weight data."""
        self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serialisable snapshot."""
        return {
            "projections": {
                name: {
                    "steps": list(rec["steps"]),
                    "weights": list(rec["weights"]),
                }
                for name, rec in self._data.items()
            }
        }
