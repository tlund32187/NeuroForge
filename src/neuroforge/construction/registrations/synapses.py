"""Register built-in synapse models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.construction.hub import FactoryHub

__all__ = ["register"]


def register(hub: FactoryHub) -> None:
    """Register built-in synapse models."""
    from neuroforge.biology.synapses.models.dales_static import DalesStaticSynapseModel
    from neuroforge.biology.synapses.models.delayed_static import DelayedStaticSynapseModel
    from neuroforge.biology.synapses.models.static import StaticSynapseModel

    hub.synapses.register("static", StaticSynapseModel)
    hub.synapses.register("static_dales", DalesStaticSynapseModel)
    hub.synapses.register("static_delayed", DelayedStaticSynapseModel)
