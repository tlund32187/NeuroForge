"""Task, readout, and loss contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ILoss",
    "IReadout",
    "ReadoutResult",
]


@dataclass(frozen=True, slots=True)
class ReadoutResult:
    """Result of a readout operation."""

    count: Any
    logits: Any


@runtime_checkable
class IReadout(Protocol):
    """Protocol for spike readout decoders."""

    def __call__(self, spikes: Any) -> ReadoutResult:
        """Decode spikes into a readout result."""
        ...


@runtime_checkable
class ILoss(Protocol):
    """Protocol for supervised loss functions."""

    def __call__(self, prediction: Any, target: Any) -> Any:
        """Compute scalar loss."""
        ...
