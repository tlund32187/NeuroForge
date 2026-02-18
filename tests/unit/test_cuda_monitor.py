# pyright: basic
"""Unit tests for CudaMetricsMonitor — works on CPU-only systems."""

from __future__ import annotations

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.cuda_monitor import CudaMetricsMonitor


class TestCudaMetricsMonitorCpu:
    """Tests that work regardless of CUDA availability."""

    def test_disabled_monitor_is_noop(self) -> None:
        """A disabled monitor should not touch event data."""
        mon = CudaMetricsMonitor(enabled=False)
        event = MonitorEvent(
            topic=EventTopic.SCALAR,
            step=0,
            t=0.0,
            source="test",
            data={"trial": 0},
        )
        mon.on_event(event)
        assert "cuda_mem_allocated" not in event.data

    def test_non_scalar_events_ignored(self) -> None:
        """Monitor should only act on SCALAR and TRAINING_TRIAL."""
        mon = CudaMetricsMonitor(enabled=True)
        event = MonitorEvent(
            topic=EventTopic.WEIGHT,
            step=0,
            t=0.0,
            source="test",
            data={"weights": []},
        )
        mon.on_event(event)
        assert "cuda_mem_allocated" not in event.data

    def test_reset(self) -> None:
        """Reset should re-arm the first-scalar flag."""
        mon = CudaMetricsMonitor(enabled=True)
        mon._first_scalar = False
        mon.reset()
        assert mon._first_scalar is True

    def test_snapshot(self) -> None:
        """Snapshot should return monitor status."""
        mon = CudaMetricsMonitor(enabled=True)
        snap = mon.snapshot()
        assert snap["enabled"] is True
        assert snap["first_scalar"] is True
