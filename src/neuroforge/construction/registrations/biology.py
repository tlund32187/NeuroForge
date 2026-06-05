"""Register built-in biological neuron models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.construction.hub import FactoryHub

__all__ = ["register"]


def register(hub: FactoryHub) -> None:
    """Register built-in neuron models."""
    from neuroforge.biology.neurons.models.lif.model import LIFModel
    from neuroforge.biology.neurons.models.lif.surrogate import SurrogateLIFModel

    hub.neurons.register("lif", LIFModel)
    hub.neurons.register("lif_surr", SurrogateLIFModel)
    hub.neurons.register("lif_surrogate", SurrogateLIFModel)
