"""Dendrite compartment parameters."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DendriteParams"]


@dataclass(frozen=True, slots=True)
class DendriteParams:
    """Parameters for dendritic integration."""

    attenuation: float = 1.0
