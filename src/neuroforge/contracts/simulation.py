"""Simulation engine contract.

Defines the protocol for the top-level simulation engine and the
configuration DTO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuroforge.contracts.tensor import Tensor

__all__ = [
    "ISimulationEngine",
    "SimulationConfig",
]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable configuration for a simulation run.

    Attributes
    ----------
    dt:
        Simulation time-step in seconds.
    seed:
        Random seed for reproducibility.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    dtype:
        Torch dtype string (``"float32"``).
    """

    dt: float = 1e-3
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class StepResult:
    """Output of a single simulation step.

    Attributes
    ----------
    step:
        Step index.
    t:
        Simulation time.
    spikes:
        Mapping from population name to spike tensor ``[N]``.
    extra:
        Any additional data (voltages, metrics, etc.).
    """

    step: int
    t: float
    spikes: dict[str, Tensor]
    extra: dict[str, Any] = field(default_factory=lambda: {})


@runtime_checkable
class ISimulationEngine(Protocol):
    """Protocol for the top-level simulation engine.

    The engine orchestrates populations (neuron models), projections
    (synapse models), and optional learning rules.

    Lifecycle: ``build`` → ``reset`` → (``step`` …).
    """

    def build(self) -> None:
        """Initialise all state tensors and prepare for stepping."""
        ...

    def reset(self) -> None:
        """Reset all state to initial values."""
        ...

    def step(
        self,
        external_input: dict[str, dict[str, Tensor]] | None = None,
    ) -> StepResult:
        """Advance the simulation by one time-step.

        Parameters
        ----------
        external_input:
            Optional mapping ``{pop_name: {compartment: drive_tensor}}``.

        Returns
        -------
        StepResult:
            Spikes and metadata for this step.
        """
        ...

    def run(self, steps: int) -> list[StepResult]:
        """Run multiple steps and return all results.

        Parameters
        ----------
        steps:
            Number of steps to run.

        Returns
        -------
        list[StepResult]:
            One StepResult per step.
        """
        ...
