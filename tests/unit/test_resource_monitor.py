# pyright: basic, reportMissingImports=false
"""Unit tests for ResourceMonitor."""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.monitors.resource_monitor import ResourceMonitor


def _event(
    step: int,
    data: dict[str, Any] | None = None,
    *,
    topic: EventTopic = EventTopic.SCALAR,
) -> MonitorEvent:
    return MonitorEvent(
        topic=topic,
        step=step,
        t=0.0,
        source="test",
        data=data or {},
    )


def test_resource_monitor_enriches_scalar_payload_on_cadence() -> None:
    mon = ResourceMonitor(
        enabled=True,
        every_n_steps=2,
        include_system=False,
        include_process=False,
        include_gpu=False,
    )

    mon._collect_payload = lambda: {  # type: ignore[method-assign]
        "resource.cpu.system_percent": 42.0,
        "resource.ram.process_rss_mb": 128.5,
    }

    e0 = _event(0, {"trial": 0})
    e1 = _event(1, {"trial": 1})
    e2 = _event(2, {"trial": 2})

    mon.on_event(e0)
    mon.on_event(e1)
    mon.on_event(e2)

    assert isinstance(e0.data["resource.cpu.system_percent"], float)
    assert isinstance(e0.data["resource.ram.process_rss_mb"], float)
    assert "resource.cpu.system_percent" not in e1.data
    assert isinstance(e2.data["resource.cpu.system_percent"], float)


def test_resource_monitor_disabled_is_noop() -> None:
    mon = ResourceMonitor(
        enabled=False,
        include_system=False,
        include_process=False,
        include_gpu=False,
    )
    mon._collect_payload = lambda: {  # type: ignore[method-assign]
        "resource.cpu.system_percent": 5.0,
    }
    ev = _event(0, {"trial": 0})
    mon.on_event(ev)
    assert "resource.cpu.system_percent" not in ev.data


def test_resource_monitor_enriches_training_trial_payload() -> None:
    mon = ResourceMonitor(
        enabled=True,
        every_n_steps=1,
        include_system=False,
        include_process=False,
        include_gpu=False,
    )
    mon._collect_payload = lambda: {  # type: ignore[method-assign]
        "resource.cpu.system_percent": 11.0,
    }
    ev = _event(4, {"trial": 4}, topic=EventTopic.TRAINING_TRIAL)
    mon.on_event(ev)
    assert ev.data["resource.cpu.system_percent"] == 11.0
