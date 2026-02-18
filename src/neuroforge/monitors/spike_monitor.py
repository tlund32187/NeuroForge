"""SpikeMonitor — records per-population spike events.

Stores spike counts per population per step and can produce a raster
(step × neuron boolean) for small populations.

CUDA-compatible: if spike tensors live on GPU, the monitor copies them
to CPU before storing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["SpikeMonitor"]


class SpikeMonitor:
    """Records spike events emitted on ``EventTopic.SPIKE``.

    Attributes
    ----------
    enabled:
        Toggle recording on/off at runtime.

    Recorded data layout (``snapshot()``)::

        {
            "populations": {
                "<name>": {
                    "steps": [step_idx, ...],
                    "counts": [spike_count_per_step, ...],
                    "raster": [[neuron_i, step_j], ...],
                }
            }
        }
    """

    def __init__(self, *, enabled: bool = True, max_raster_neurons: int = 200) -> None:
        self.enabled = enabled
        self._max_raster = max_raster_neurons
        self._data: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: {"steps": [], "counts": [], "raster": []}
        )

    # ── IMonitor implementation ─────────────────────────────────────

    def on_event(self, event: MonitorEvent) -> None:
        """Record spike data from a SPIKE event."""
        if not self.enabled or event.topic != EventTopic.SPIKE:
            return

        spikes_raw = event.data.get("spikes")
        if spikes_raw is None:
            return

        # Convert to CPU list for storage (CUDA-safe).
        try:
            spikes_list: list[bool] = spikes_raw.detach().cpu().tolist()
        except AttributeError:
            spikes_list = list(spikes_raw)

        count = sum(1 for s in spikes_list if s)
        rec = self._data[event.source]
        rec["steps"].append(event.step)
        rec["counts"].append(count)

        # Raster: store indices of firing neurons (bounded).
        if len(spikes_list) <= self._max_raster:
            for i, fired in enumerate(spikes_list):
                if fired:
                    rec["raster"].append([i, event.step])

    def reset(self) -> None:
        """Clear all recorded spike data."""
        self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serialisable snapshot."""
        return {"populations": dict(self._data)}
