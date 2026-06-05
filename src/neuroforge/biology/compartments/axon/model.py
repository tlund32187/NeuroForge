"""Axon compartment model."""

from __future__ import annotations

from typing import Any

from neuroforge.biology.compartments.axon.params import AxonParams
from neuroforge.biology.compartments.base import CompartmentModelBase
from neuroforge.biology.compartments.state import CompartmentState
from neuroforge.biology.compartments.types import Compartment

__all__ = ["AxonCompartmentModel"]


class AxonCompartmentModel(CompartmentModelBase):
    """Minimal axon compartment model."""

    def __init__(self, params: AxonParams | None = None) -> None:
        self.params = params or AxonParams()

    def init_state(self, n: int, device: str, dtype: str) -> CompartmentState:
        """Create initial axon state."""
        from neuroforge.kernel.torch_utils import require_torch, resolve_device_dtype

        torch = require_torch()
        torch_device, torch_dtype = resolve_device_dtype(device, dtype)
        return CompartmentState(
            compartment=Compartment.AXON,
            tensors={"spikes": torch.zeros(n, device=torch_device, dtype=torch_dtype)},
        )

    def step(self, state: CompartmentState, drive: Any, dt: float) -> CompartmentState:
        """Advance axon state; detailed propagation can extend this model."""
        _ = (drive, dt)
        return state
