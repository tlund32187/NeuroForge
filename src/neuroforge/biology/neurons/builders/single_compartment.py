"""Single-compartment neuron builder specs."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.biology.compartments.types import Compartment

__all__ = ["SingleCompartmentSpec"]


@dataclass(frozen=True, slots=True)
class SingleCompartmentSpec:
    """Description of a neuron with one compartment."""

    compartment: Compartment = Compartment.SOMA
