"""Soma compartment model."""

from __future__ import annotations

from neuroforge.biology.compartments.soma.factory import SomaFactory
from neuroforge.biology.compartments.soma.model import SomaCompartmentModel
from neuroforge.biology.compartments.soma.params import SomaParams

__all__ = ["SomaCompartmentModel", "SomaFactory", "SomaParams"]
