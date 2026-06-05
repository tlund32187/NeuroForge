"""Chemical synapse model extension point."""

from __future__ import annotations

from neuroforge.biology.synapses.models.static import StaticSynapseModel

__all__ = ["ChemicalSynapseModel"]


class ChemicalSynapseModel(StaticSynapseModel):
    """Current chemical synapse implementation backed by static propagation."""
