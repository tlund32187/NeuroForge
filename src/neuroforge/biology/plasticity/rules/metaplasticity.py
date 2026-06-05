"""Metaplasticity rule extension point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MetaplasticityParams", "MetaplasticityRule"]


@dataclass(frozen=True, slots=True)
class MetaplasticityParams:
    """Parameters for a metaplasticity rule."""

    lr: float = 1e-3


class MetaplasticityRule:
    """Minimal metaplasticity rule interface."""

    def __init__(self, params: MetaplasticityParams | None = None) -> None:
        self.params = params or MetaplasticityParams()

    def init_state(self, n_edges: int, device: str, dtype: str) -> dict[str, Any]:
        """Allocate rule state."""
        _ = (n_edges, device, dtype)
        return {}

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return persistable state tensors."""
        return state
