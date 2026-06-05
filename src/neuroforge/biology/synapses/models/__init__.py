"""Concrete synapse models."""

from __future__ import annotations

from neuroforge.biology.synapses.models.chemical import ChemicalSynapseModel
from neuroforge.biology.synapses.models.dales_static import DalesStaticSynapseModel
from neuroforge.biology.synapses.models.delayed_static import DelayedStaticSynapseModel
from neuroforge.biology.synapses.models.electrical_gap_junction import (
    ElectricalGapJunctionModel,
)
from neuroforge.biology.synapses.models.static import StaticSynapseModel

__all__ = [
    "ChemicalSynapseModel",
    "DalesStaticSynapseModel",
    "DelayedStaticSynapseModel",
    "ElectricalGapJunctionModel",
    "StaticSynapseModel",
]
