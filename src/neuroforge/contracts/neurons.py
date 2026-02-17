"""Neuron model contract.

Defines the protocol that every neuron model must implement, plus the
frozen DTOs for inputs, outputs, and step context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.contracts.tensor import Tensor
    from neuroforge.contracts.types import Compartment

__all__ = [
    "INeuronModel",
    "NeuronInputs",
    "NeuronStepResult",
    "StepContext",
]


@dataclass(frozen=True, slots=True)
class StepContext:
    """Immutable context passed to every ``step()`` call.

    Attributes
    ----------
    step:
        Current simulation step index (0-based).
    dt:
        Simulation time-step in seconds.
    t:
        Current simulation time in seconds (``step * dt``).
    """

    step: int
    dt: float
    t: float


@dataclass(frozen=True, slots=True)
class NeuronInputs:
    """Input currents/drives delivered to a neuron population this step.

    Attributes
    ----------
    drive:
        Mapping from compartment to drive current tensor ``[N]``.
        For single-compartment models, only ``Compartment.SOMA`` is present.
    """

    drive: dict[Compartment, Tensor]


@dataclass(frozen=True, slots=True)
class NeuronStepResult:
    """Output produced by a neuron model after one simulation step.

    Attributes
    ----------
    spikes:
        Boolean tensor ``[N]`` — True where the neuron fired.
    voltage:
        Mapping from compartment to membrane voltage tensor ``[N]``.
    """

    spikes: Tensor
    voltage: dict[Compartment, Tensor]


@runtime_checkable
class INeuronModel(Protocol):
    """Protocol for population-level neuron models.

    A neuron model manages the state of *N* neurons in a single
    population.  All operations are vectorised over *N*.

    Lifecycle: ``init_state`` → (``step`` …) → ``reset_state``.
    """

    def init_state(
        self,
        n: int,
        device: str,
        dtype: str,
    ) -> dict[str, Tensor]:
        """Allocate and return initial state tensors for *n* neurons.

        Parameters
        ----------
        n:
            Number of neurons in the population.
        device:
            Target device (``"cpu"`` or ``"cuda"``).
        dtype:
            Tensor dtype string (``"float32"``).

        Returns
        -------
        dict[str, Tensor]:
            Named state tensors (e.g. ``{"v": Tensor[N]}``).
        """
        ...

    def step(
        self,
        state: dict[str, Tensor],
        inputs: NeuronInputs,
        ctx: StepContext,
    ) -> NeuronStepResult:
        """Advance the population by one time-step.

        Parameters
        ----------
        state:
            Mutable state dict returned by ``init_state`` (updated in-place).
        inputs:
            Drive currents for this step.
        ctx:
            Simulation context (step index, dt, t).

        Returns
        -------
        NeuronStepResult:
            Spikes and voltages for this step.
        """
        ...

    def reset_state(self, state: dict[str, Tensor]) -> None:
        """Reset state tensors to their initial values (in-place)."""
        ...

    def state_tensors(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return the subset of *state* that should be checkpointed."""
        ...
