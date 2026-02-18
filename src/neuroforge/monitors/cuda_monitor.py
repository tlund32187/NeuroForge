"""CudaMetricsMonitor — records GPU memory usage into SCALAR events.

When the training device is CUDA, this monitor intercepts every
``SCALAR`` event and enriches its ``data`` dict with:

- ``cuda_mem_allocated``  — bytes currently allocated on the default
  CUDA device (``torch.cuda.memory_allocated()``).
- ``cuda_mem_reserved``   — bytes reserved by the caching allocator
  (``torch.cuda.memory_reserved()``).
- ``cuda_mem_peak``       — high-water mark of allocated bytes since
  the last ``torch.cuda.reset_peak_memory_stats()`` call
  (``torch.cuda.max_memory_allocated()``).

Because the monitor mutates ``event.data`` *in-place* before
``ArtifactWriter`` sees the event, those columns show up automatically
in ``metrics/scalars.csv``.

On CPU-only systems the monitor is a no-op.
"""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["CudaMetricsMonitor"]


class CudaMetricsMonitor:
    """Lightweight monitor that stamps CUDA memory stats onto SCALAR events.

    Parameters
    ----------
    enabled:
        Whether the monitor is active.  Set to ``False`` on CPU-only
        systems so the monitor is a cheap no-op.
    reset_peak_on_start:
        If ``True``, call ``torch.cuda.reset_peak_memory_stats()`` when
        the first ``SCALAR`` event arrives.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        reset_peak_on_start: bool = True,
    ) -> None:
        self.enabled = enabled
        self._reset_peak_on_start = reset_peak_on_start
        self._first_scalar = True

    # ── IMonitor interface ──────────────────────────────────────────

    def on_event(self, event: MonitorEvent) -> None:
        """Enrich SCALAR events with CUDA memory statistics."""
        if not self.enabled:
            return
        if event.topic not in (EventTopic.SCALAR, EventTopic.TRAINING_TRIAL):
            return

        try:
            from neuroforge.core.torch_utils import require_torch

            torch = require_torch()
            if not torch.cuda.is_available():
                return

            if self._first_scalar and self._reset_peak_on_start:
                torch.cuda.reset_peak_memory_stats()
                self._first_scalar = False

            # Inject memory stats into the event's data dict (in-place).
            event.data["cuda_mem_allocated"] = torch.cuda.memory_allocated()
            event.data["cuda_mem_reserved"] = torch.cuda.memory_reserved()
            event.data["cuda_mem_peak"] = torch.cuda.max_memory_allocated()
        except ImportError:
            pass

    def reset(self) -> None:
        """Reset internal state."""
        self._first_scalar = True

    def snapshot(self) -> dict[str, Any]:
        """Return monitor status summary."""
        return {"enabled": self.enabled, "first_scalar": self._first_scalar}
