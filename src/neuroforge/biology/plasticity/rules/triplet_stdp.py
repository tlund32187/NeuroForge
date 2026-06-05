"""Triplet STDP rule extension point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["TripletSTDPParams", "TripletSTDPRule"]


@dataclass(frozen=True, slots=True)
class TripletSTDPParams:
    """Parameters for a triplet STDP rule."""

    lr: float = 1e-3


class TripletSTDPRule:
    """Minimal triplet STDP rule interface."""

    def __init__(self, params: TripletSTDPParams | None = None) -> None:
        self.params = params or TripletSTDPParams()

    def init_state(self, n_edges: int, device: str, dtype: str) -> dict[str, Any]:
        """Allocate rule state."""
        _ = (n_edges, device, dtype)
        return {}

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return persistable state tensors."""
        return state
