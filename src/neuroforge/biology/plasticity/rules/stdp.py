"""Spike-Timing-Dependent Plasticity rule extension point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["STDPParams", "STDPRule"]


@dataclass(frozen=True, slots=True)
class STDPParams:
    """Parameters for a pair-based STDP rule."""

    lr: float = 1e-3


class STDPRule:
    """Minimal STDP rule interface for future biological implementations."""

    def __init__(self, params: STDPParams | None = None) -> None:
        self.params = params or STDPParams()

    def init_state(self, n_edges: int, device: str, dtype: str) -> dict[str, Any]:
        """Allocate rule state."""
        _ = (n_edges, device, dtype)
        return {}

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return persistable state tensors."""
        return state
