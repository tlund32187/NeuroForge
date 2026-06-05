"""Leaky Integrate-and-Fire neuron model."""

from __future__ import annotations

import math
from typing import Any

from neuroforge.biology.compartments.types import Compartment
from neuroforge.biology.neurons.base import NeuronModelBase
from neuroforge.biology.neurons.models.lif.params import LIFParams as _LIFParams
from neuroforge.biology.neurons.state import NeuronInputs, NeuronStepResult, StepContext

__all__ = ["LIFModel"]


class LIFModel(NeuronModelBase):
    """Leaky Integrate-and-Fire neuron model for one soma compartment."""

    def __init__(self, params: _LIFParams | None = None) -> None:
        self.params = params or _LIFParams()

    def decay_factor(self, dt: float) -> float:
        """Compute the voltage decay factor."""
        return math.exp(-dt / self.params.tau_mem)

    def drive_factor(self, dt: float) -> float:
        """Compute the drive scaling factor."""
        return dt / self.params.tau_mem

    def steady_state_voltage(self, drive: float, dt: float) -> float:
        """Predict the steady-state voltage for a constant drive."""
        alpha = self.decay_factor(dt)
        r = self.drive_factor(dt)
        return self.params.v_rest + drive * r / (1.0 - alpha)

    def voltage_after_k_steps(
        self,
        v0: float,
        drive: float,
        dt: float,
        k: int,
    ) -> float:
        """Predict voltage after ``k`` steps with constant drive and no spiking."""
        alpha = self.decay_factor(dt)
        r = self.drive_factor(dt)
        v_rest = self.params.v_rest
        alpha_k = alpha**k
        return (
            v_rest
            + (v0 - v_rest) * alpha_k
            + drive * r * (1.0 - alpha_k) / (1.0 - alpha)
        )

    def steps_to_fire(
        self, drive: float, dt: float, v0: float | None = None
    ) -> int | None:
        """Predict the first step on which the neuron fires."""
        if v0 is None:
            v0 = self.params.v_rest

        v_ss = self.steady_state_voltage(drive, dt)
        if v_ss < self.params.v_thresh:
            return None

        alpha = self.decay_factor(dt)
        r = self.drive_factor(dt)
        v_rest = self.params.v_rest
        v = v0
        for k in range(1, 100_000):
            v = v_rest + (v - v_rest) * alpha + drive * r
            if v >= self.params.v_thresh:
                return k
        return None

    def impulses_to_fire(self, impulse_dv: float, dt: float) -> int | None:
        """Predict how many discrete impulses are needed to reach threshold."""
        if impulse_dv <= 0:
            return None

        alpha = self.decay_factor(dt)
        v = self.params.v_rest
        for n in range(1, 100_000):
            v = self.params.v_rest + (v - self.params.v_rest) * alpha + impulse_dv
            if v >= self.params.v_thresh:
                return n
        return None

    def _init_state_tensors(
        self,
        n: int,
        device: Any,
        dtype: Any,
    ) -> dict[str, Any]:
        from neuroforge.kernel.torch_utils import require_torch

        torch = require_torch()
        v = torch.full((n,), self.params.v_rest, device=device, dtype=dtype)
        return {"v": v}

    def _step(
        self,
        state: dict[str, Any],
        inputs: NeuronInputs,
        ctx: StepContext,
    ) -> NeuronStepResult:
        from neuroforge.kernel.torch_utils import require_torch

        torch = require_torch()

        v = state["v"]
        dt = ctx.dt
        p = self.params

        drive = inputs.drive.get(Compartment.SOMA)
        if drive is None:
            drive = torch.zeros_like(v)

        alpha = math.exp(-dt / p.tau_mem)
        r = dt / p.tau_mem
        v_new = p.v_rest + (v - p.v_rest) * alpha + drive * r
        spikes = v_new >= p.v_thresh
        v_new = torch.where(
            spikes, torch.tensor(p.v_reset, device=v.device, dtype=v.dtype), v_new
        )

        state["v"] = v_new

        return NeuronStepResult(
            spikes=spikes,
            voltage={Compartment.SOMA: v_new},
        )

    def _reset_state_tensors(self, state: dict[str, Any]) -> None:
        state["v"].fill_(self.params.v_rest)
