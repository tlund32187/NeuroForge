"""Register built-in application-facing construction components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.construction.hub import FactoryHub

__all__ = ["register"]


def register(hub: FactoryHub) -> None:
    """Register built-in monitor constructors used by applications."""
    from neuroforge.observability.monitors.resource_monitor import (
        CudaMetricsMonitor,
        ResourceMonitor,
    )
    from neuroforge.observability.monitors.spike_monitor import SpikeMonitor
    from neuroforge.observability.monitors.stability_monitor import StabilityMonitor
    from neuroforge.observability.monitors.topology_stats_monitor import TopologyStatsMonitor
    from neuroforge.observability.monitors.training_monitor import TrainingMonitor
    from neuroforge.observability.monitors.trial_stats_monitor import TrialStatsMonitor
    from neuroforge.observability.monitors.voltage_monitor import VoltageMonitor
    from neuroforge.observability.monitors.weight_monitor import WeightMonitor

    hub.monitors.register("cuda", CudaMetricsMonitor)
    hub.monitors.register("resource", ResourceMonitor)
    hub.monitors.register("spikes", SpikeMonitor)
    hub.monitors.register("stability", StabilityMonitor)
    hub.monitors.register("topology_stats", TopologyStatsMonitor)
    hub.monitors.register("training", TrainingMonitor)
    hub.monitors.register("trial_stats", TrialStatsMonitor)
    hub.monitors.register("voltage", VoltageMonitor)
    hub.monitors.register("weights", WeightMonitor)
