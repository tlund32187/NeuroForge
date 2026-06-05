"""Factory for soma compartment models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.biology.compartments.soma.model import SomaCompartmentModel

if TYPE_CHECKING:
    from neuroforge.biology.compartments.soma.params import SomaParams

__all__ = ["SomaFactory"]


class SomaFactory:
    """Create soma compartment models."""

    def create(self, params: SomaParams | None = None) -> SomaCompartmentModel:
        """Create a soma compartment model."""
        return SomaCompartmentModel(params)
