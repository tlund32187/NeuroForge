"""Two-compartment neuron builder specs."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.biology.compartments.types import Compartment

__all__ = ["TwoCompartmentSpec"]


@dataclass(frozen=True, slots=True)
class TwoCompartmentSpec:
    """Description of a neuron with soma and dendrite compartments."""

    soma: Compartment = Compartment.SOMA
    dendrite: Compartment = Compartment.DENDRITE
