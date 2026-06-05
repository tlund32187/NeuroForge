"""Factory for axon compartment models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.biology.compartments.axon.model import AxonCompartmentModel

if TYPE_CHECKING:
    from neuroforge.biology.compartments.axon.params import AxonParams

__all__ = ["AxonFactory"]


class AxonFactory:
    """Create axon compartment models."""

    def create(self, params: AxonParams | None = None) -> AxonCompartmentModel:
        """Create an axon compartment model."""
        return AxonCompartmentModel(params)
