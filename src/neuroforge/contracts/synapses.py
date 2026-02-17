"""Synapse model contract.

Defines the protocol that every synapse model must implement, plus the
frozen DTOs for topology, inputs, and step results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.contracts.tensor import Tensor
    from neuroforge.contracts.types import Compartment

__all__ = [
    "ISynapseModel",
    "SynapseTopology",
    "SynapseInputs",
    "SynapseStepResult",
]


@dataclass(frozen=True, slots=True)
class SynapseTopology:
    """Immutable description of a projection's wiring.

    Attributes
    ----------
    pre_idx:
        Source neuron indices, int tensor ``[E]``.
    post_idx:
        Target neuron indices, int tensor ``[E]``.
    weights:
        Synaptic weights, float tensor ``[E]``.
    delays:
        Per-edge delay in simulation steps, int tensor ``[E]``.
    n_pre:
        Number of pre-synaptic neurons.
    n_post:
        Number of post-synaptic neurons.
    """

    pre_idx: Tensor
    post_idx: Tensor
    weights: Tensor
    delays: Tensor
    n_pre: int
    n_post: int


@dataclass(frozen=True, slots=True)
class SynapseInputs:
    """Inputs delivered to a synapse model each step.

    Attributes
    ----------
    pre_spikes:
        Boolean tensor ``[N_pre]`` — which pre-synaptic neurons fired.
    post_spikes:
        Boolean tensor ``[N_post]`` — which post-synaptic neurons fired
        (needed for some learning rules, not for propagation).
    """

    pre_spikes: Tensor
    post_spikes: Tensor


@dataclass(frozen=True, slots=True)
class SynapseStepResult:
    """Output of a synapse model step.

    Attributes
    ----------
    post_current:
        Mapping from compartment to post-synaptic current tensor ``[N_post]``.
    """

    post_current: dict[Compartment, Tensor]


@runtime_checkable
class ISynapseModel(Protocol):
    """Protocol for edge-population synapse models.

    A synapse model computes post-synaptic currents from pre-synaptic
    spike events, given a fixed topology (weights, indices, delays).

    Lifecycle: ``init_state`` → (``step`` …) → (state is mutable dict).
    """

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Tensor]:
        """Allocate synapse-specific state (e.g. delay ring buffers).

        Parameters
        ----------
        topology:
            The wiring spec for this projection.
        device:
            Target device.
        dtype:
            Tensor dtype string.

        Returns
        -------
        dict[str, Tensor]:
            Named state tensors.
        """
        ...

    def step(
        self,
        state: dict[str, Tensor],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Propagate spikes through the projection for one step.

        Parameters
        ----------
        state:
            Mutable state dict.
        topology:
            Projection wiring.
        inputs:
            Pre/post spike tensors.
        ctx:
            Step context.

        Returns
        -------
        SynapseStepResult:
            Post-synaptic currents by compartment.
        """
        ...

    def state_tensors(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return state tensors for checkpointing."""
        ...
