"""Soma compartment model."""

from __future__ import annotations

from typing import Any

from neuroforge.biology.compartments.base import CompartmentModelBase
from neuroforge.biology.compartments.soma.params import SomaParams
from neuroforge.biology.compartments.state import CompartmentState
from neuroforge.biology.compartments.types import Compartment

__all__ = ["SomaCompartmentModel"]


class SomaCompartmentModel(CompartmentModelBase):
    """Minimal soma compartment model with voltage state."""

    def __init__(self, params: SomaParams | None = None) -> None:
        self.params = params or SomaParams()

    def init_state(self, n: int, device: str, dtype: str) -> CompartmentState:
        """Create initial soma voltage state."""
        from neuroforge.kernel.torch_utils import require_torch, resolve_device_dtype

        torch = require_torch()
        torch_device, torch_dtype = resolve_device_dtype(device, dtype)
        return CompartmentState(
            compartment=Compartment.SOMA,
            tensors={"v": torch.zeros(n, device=torch_device, dtype=torch_dtype)},
        )

    def step(self, state: CompartmentState, drive: Any, dt: float) -> CompartmentState:
        """Advance soma state; detailed dynamics live in neuron models."""
        _ = (drive, dt)
        return state
