"""VoltageMonitor — records per-population membrane voltages.

Samples voltage at each step for each population.  For dashboards,
only the last *window* steps are kept (ring-buffer style) to bound
memory.

CUDA-compatible: tensors are moved to CPU before storage.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["VoltageMonitor"]


class VoltageMonitor:
    """Records voltage snapshots emitted on ``EventTopic.VOLTAGE``.

    Parameters
    ----------
    max_window:
        Maximum number of steps to keep in memory (oldest discarded).
    enabled:
        Whether the monitor is active.
    """

    def __init__(
        self, *, enabled: bool = True, max_window: int = 500
    ) -> None:
        self.enabled = enabled
        self._max_window = max_window
        self._data: dict[str, dict[str, deque[Any]]] = defaultdict(
            lambda: {"steps": deque(maxlen=self._max_window),
                     "voltages": deque(maxlen=self._max_window)}
        )

    # ── IMonitor implementation ─────────────────────────────────────

    def on_event(self, event: MonitorEvent) -> None:
        """Record voltage snapshot from a VOLTAGE event."""
        if not self.enabled or event.topic != EventTopic.VOLTAGE:
            return

        voltage_raw = event.data.get("voltage")
        if voltage_raw is None:
            return

        try:
            voltage_list: list[float] = voltage_raw.detach().cpu().tolist()
        except AttributeError:
            voltage_list = list(voltage_raw)

        rec = self._data[event.source]
        rec["steps"].append(event.step)
        rec["voltages"].append(voltage_list)

    def reset(self) -> None:
        """Clear all recorded voltage data."""
        self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serialisable snapshot."""
        return {
            "populations": {
                name: {
                    "steps": list(rec["steps"]),
                    "voltages": list(rec["voltages"]),
                }
                for name, rec in self._data.items()
            }
        }
