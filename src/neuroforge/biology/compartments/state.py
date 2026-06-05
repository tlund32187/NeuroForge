"""Compartment state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuroforge.biology.compartments.types import Compartment

__all__ = ["CompartmentState"]


def _empty_tensors() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class CompartmentState:
    """Named state tensors for one biological compartment."""

    compartment: Compartment
    tensors: dict[str, Any] = field(default_factory=_empty_tensors)
