"""Learning rule contract.

Defines the protocol for synaptic learning rules and the frozen DTOs
for learning batches and step results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.contracts.tensor import Tensor

__all__ = [
    "ILearningRule",
    "LearningBatch",
    "LearningStepResult",
]


@dataclass(frozen=True, slots=True)
class LearningBatch:
    """Data passed to a learning rule each update step.

    Attributes
    ----------
    pre_spikes:
        Boolean ``[E]`` — which pre-synaptic neurons spiked (edge-indexed).
    post_spikes:
        Boolean ``[E]`` — which post-synaptic neurons spiked (edge-indexed).
    weights:
        Current weight tensor ``[E]``.
    eligibility:
        Per-edge eligibility trace ``[E]`` or None if not yet initialised.
    reward:
        Scalar or ``[1]`` reward signal for this update.
    """

    pre_spikes: Tensor
    post_spikes: Tensor
    weights: Tensor
    eligibility: Tensor | None
    reward: Tensor | None


@dataclass(frozen=True, slots=True)
class LearningStepResult:
    """Output of a learning rule update.

    Attributes
    ----------
    dw:
        Weight delta tensor ``[E]`` to be added to weights.
    new_eligibility:
        Updated eligibility trace ``[E]`` or None.
    """

    dw: Tensor
    new_eligibility: Tensor | None


@runtime_checkable
class ILearningRule(Protocol):
    """Protocol for synaptic learning rules.

    A learning rule computes weight updates from spike activity and
    (optionally) a reward/modulator signal.

    Lifecycle: ``init_state`` → (``step`` …).
    """

    def init_state(
        self,
        n_edges: int,
        device: str,
        dtype: str,
    ) -> dict[str, Tensor]:
        """Allocate learning-rule state (e.g. eligibility traces).

        Parameters
        ----------
        n_edges:
            Number of edges in the projection.
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
        batch: LearningBatch,
        ctx: object,
    ) -> LearningStepResult:
        """Compute one learning update.

        Parameters
        ----------
        state:
            Mutable state dict (updated in-place).
        batch:
            Current spike / weight / reward data.
        ctx:
            Step context.

        Returns
        -------
        LearningStepResult:
            Weight deltas and updated eligibility.
        """
        ...

    def state_tensors(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return state tensors for checkpointing."""
        ...
