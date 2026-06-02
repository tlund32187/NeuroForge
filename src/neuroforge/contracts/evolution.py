# pyright: basic
"""Contracts for the neuroevolution track.

These protocols keep population search decoupled from the concrete SMB3 fitness
backend. Early tests can use a toy evaluator, while live runs can plug in a
BizHawk/GameTrainingTask evaluator later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "FitnessResult",
    "IGenome",
    "IFitnessEvaluator",
    "IReproduction",
    "ISpeciation",
]


@dataclass(frozen=True, slots=True)
class FitnessResult:
    """Fitness and telemetry produced by evaluating one genome."""

    fitness: float
    metrics: dict[str, float] = field(default_factory=dict)
    episodes: int = 0
    frames: int = 0


@runtime_checkable
class IGenome(Protocol):
    """Minimal genome interface needed by the evolution loop."""

    @property
    def id(self) -> str:
        """Stable genome identifier."""
        ...

    @property
    def generation(self) -> int:
        """Generation in which this genome was created."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        ...


@runtime_checkable
class IFitnessEvaluator(Protocol):
    """Score a genome."""

    def evaluate(self, genome: IGenome) -> FitnessResult:
        """Return the genome's fitness result."""
        ...


@runtime_checkable
class IReproduction(Protocol):
    """Create a next generation from evaluated genomes."""

    def next_generation(self, evaluated: list[Any]) -> list[IGenome]:
        """Return the child population."""
        ...


@runtime_checkable
class ISpeciation(Protocol):
    """Assign genomes to species."""

    def assign(self, genomes: list[IGenome]) -> dict[str, int]:
        """Return ``genome_id -> species_id``."""
        ...
