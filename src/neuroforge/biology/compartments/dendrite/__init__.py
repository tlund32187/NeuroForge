"""Dendrite compartment model."""

from __future__ import annotations

from neuroforge.biology.compartments.dendrite.factory import DendriteFactory
from neuroforge.biology.compartments.dendrite.model import DendriteCompartmentModel
from neuroforge.biology.compartments.dendrite.params import DendriteParams

__all__ = ["DendriteCompartmentModel", "DendriteFactory", "DendriteParams"]
