"""Factory for dendrite compartment models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.biology.compartments.dendrite.model import DendriteCompartmentModel

if TYPE_CHECKING:
    from neuroforge.biology.compartments.dendrite.params import DendriteParams

__all__ = ["DendriteFactory"]


class DendriteFactory:
    """Create dendrite compartment models."""

    def create(
        self, params: DendriteParams | None = None
    ) -> DendriteCompartmentModel:
        """Create a dendrite compartment model."""
        return DendriteCompartmentModel(params)
