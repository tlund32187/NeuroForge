"""Abstract base class for neuron models (template method pattern).

Concrete neuron models inherit from ``NeuronModelBase`` and implement
the hook methods.  The base class provides validation and the
``INeuronModel`` lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuroforge.contracts.neurons import (
        NeuronInputs,
        NeuronStepResult,
        StepContext,
    )
    from neuroforge.contracts.types import Compartment

__all__ = ["NeuronModelBase", "NeuronParams"]


@dataclass(frozen=True, slots=True)
class NeuronParams:
    """Base parameter dataclass — subclasses extend with model-specific fields."""


class NeuronModelBase(ABC):
    """Template-method base for neuron models.

    Subclasses implement:
    - ``_init_state_tensors`` — allocate model-specific state
    - ``_step`` — one time-step of the dynamics
    - ``_reset_state_tensors`` — zero/reinit state
    """

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
        """Reset state tensors to initial values in-place."""
        ...

    # ── INeuronModel implementation ─────────────────────────────────

    def init_state(
        self,
        n: int,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Allocate and return initial state tensors for *n* neurons."""
        from neuroforge.core.torch_utils import resolve_device_dtype

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
        """Return all state tensors (default: return state as-is)."""
        return state

    def compartments(self) -> tuple[Compartment, ...]:
        """Return compartment(s) this model supports."""
        from neuroforge.contracts.types import Compartment as _Compartment

        return (_Compartment.SOMA,)
