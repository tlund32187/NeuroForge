"""Synapse input and step-result DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.biology.compartments.types import Compartment
    from neuroforge.contracts.tensors import Tensor

__all__ = ["SynapseInputs", "SynapseStepResult"]


@dataclass(frozen=True, slots=True)
class SynapseInputs:
    """Inputs delivered to a synapse model each step."""

    pre_spikes: Tensor
    post_spikes: Tensor


@dataclass(frozen=True, slots=True)
class SynapseStepResult:
    """Output of a synapse model step."""

    post_current: dict[Compartment, Tensor]
