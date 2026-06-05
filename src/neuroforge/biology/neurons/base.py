"""Abstract base class for neuron models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuroforge.biology.compartments.types import Compartment
    from neuroforge.biology.neurons.state import (
        NeuronInputs,
        NeuronStepResult,
        StepContext,
    )

__all__ = ["NeuronModelBase"]


class NeuronModelBase(ABC):
    """Template-method base for population-level neuron models."""

    @abstractmethod
    def _init_state_tensors(
        self,
        n: int,
        device: Any,
        dtype: Any,
    ) -> dict[str, Any]:
        """Allocate model-specific state tensors."""
        ...

    @abstractmethod
    def _step(
        self,
        state: dict[str, Any],
        inputs: NeuronInputs,
        ctx: StepContext,
    ) -> NeuronStepResult:
        """Compute one step of the neuron dynamics."""
        ...

    @abstractmethod
    def _reset_state_tensors(self, state: dict[str, Any]) -> None:
        """Reset state tensors to their initial values in-place."""
        ...

    def init_state(
        self,
        n: int,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Allocate and return initial state tensors for `
`` neurons."""
        from neuroforge.kernel.torch_utils import resolve_device_dtype

        torch_device, torch_dtype = resolve_device_dtype(device, dtype)
        return self._init_state_tensors(n, torch_device, torch_dtype)

    def step(
        self,
        state: dict[str, Any],
        inputs: NeuronInputs,
        ctx: StepContext,
    ) -> NeuronStepResult:
        """Advance the population by one time-step."""
        return self._step(state, inputs, ctx)

    def reset_state(self, state: dict[str, Any]) -> None:
        """Reset state tensors to initial values."""
        self._reset_state_tensors(state)

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return all persistable state tensors."""
        return state

    def compartments(self) -> tuple[Compartment, ...]:
        """Return compartment(s) this model supports."""
        from neuroforge.biology.compartments.types import Compartment

        return (Compartment.SOMA,)
