"""Multi-compartment neuron builder specs."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.biology.compartments.types import Compartment

__all__ = ["MultiCompartmentSpec"]


@dataclass(frozen=True, slots=True)
class MultiCompartmentSpec:
    """Description of a neuron with an arbitrary compartment sequence."""

    compartments: tuple[Compartment, ...] = (
        Compartment.SOMA,
        Compartment.DENDRITE,
        Compartment.AXON,
    )
