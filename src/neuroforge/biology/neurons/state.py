"""Neuron state and step DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.biology.compartments.types import Compartment
    from neuroforge.contracts.tensors import Tensor

__all__ = ["NeuronInputs", "NeuronStepResult", "StepContext"]


@dataclass(frozen=True, slots=True)
class StepContext:
    """Immutable context passed to every neuron or synapse step."""

    step: int
    dt: float
    t: float


@dataclass(frozen=True, slots=True)
class NeuronInputs:
    """Input drives delivered to a neuron population for one step."""

    drive: dict[Compartment, Tensor]


@dataclass(frozen=True, slots=True)
class NeuronStepResult:
    """Outputs produced by a neuron population after one step."""

    spikes: Tensor
    voltage: dict[Compartment, Tensor]
