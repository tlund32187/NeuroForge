"""Leaky Integrate-and-Fire (LIF) neuron model.

Single-compartment (soma only).  Dynamics:

    v(t+dt) = v_rest + (v(t) - v_rest) * exp(-dt / tau_mem) + I(t) * R_factor

where R_factor = dt / tau_mem  (normalised so that constant I = v_thresh - v_rest
drives the neuron just to threshold at steady state).

When v >= v_thresh the neuron spikes and voltage resets to v_reset.

All operations are vectorised over N neurons — no Python loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from neuroforge.contracts.neurons import NeuronInputs, NeuronStepResult, StepContext
from neuroforge.contracts.types import Compartment
from neuroforge.neurons.base import NeuronModelBase

__all__ = ["LIFModel", "LIFParams"]


@dataclass(frozen=True, slots=True)
class LIFParams:
    """Parameters for the LIF neuron model.

    Attributes
    ----------
    tau_mem:
        Membrane time constant in seconds (default 20 ms).
    v_thresh:
        Spike threshold voltage (default 1.0).
    v_reset:
        Reset voltage after spike (default 0.0).
    v_rest:
        Resting membrane potential (default 0.0).
    """

    tau_mem: float = 20e-3
    v_thresh: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0


class LIFModel(NeuronModelBase):
    """Leaky Integrate-and-Fire neuron model (single compartment).

    Parameters
    ----------
    params:
        LIF model parameters.  Uses defaults if not provided.
    """

    def __init__(self, params: LIFParams | None = None) -> None:
        self.params = params or LIFParams()

    # ── math helpers (public for testing) ───────────────────────────

    def decay_factor(self, dt: float) -> float:
        """Compute the voltage decay factor: exp(-dt / tau_mem)."""
        return math.exp(-dt / self.params.tau_mem)

    def drive_factor(self, dt: float) -> float:
        """Compute the drive scaling factor: dt / tau_mem."""
        return dt / self.params.tau_mem

    def steady_state_voltage(self, drive: float, dt: float) -> float:
        """Predict the steady-state voltage for a constant drive.

        v_ss = v_rest + drive * (dt / tau_mem) / (1 - exp(-dt / tau_mem))
        """
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
        """Predict voltage after k steps with constant drive (no spiking).

        Uses the closed-form recurrence:
            v_k = v_rest + (v0 - v_rest) * alpha^k + drive * R * (1 - alpha^k) / (1 - alpha)
        where alpha = exp(-dt/tau_mem), R = dt/tau_mem.
        """
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
        """Predict the step number on which the neuron first fires.

        Returns None if drive is too weak (steady state < threshold).
        """
        if v0 is None:
            v0 = self.params.v_rest

        v_ss = self.steady_state_voltage(drive, dt)
        if v_ss < self.params.v_thresh:
            return None  # will never fire

        # Binary/linear search: find smallest k where v_k >= v_thresh
        alpha = self.decay_factor(dt)
        r = self.drive_factor(dt)
        v_rest = self.params.v_rest
        v = v0
        for k in range(1, 100_000):
            v = v_rest + (v - v_rest) * alpha + drive * r
            if v >= self.params.v_thresh:
                return k
        return None  # shouldn't reach here for reasonable params

    def impulses_to_fire(self, impulse_dv: float, dt: float) -> int | None:
        """Predict how many discrete impulses are needed to reach threshold.

        Each impulse adds impulse_dv to voltage, with decay between impulses.
        Returns None if impulse_dv <= 0.
        """
        if impulse_dv <= 0:
            return None

        alpha = self.decay_factor(dt)
        v = self.params.v_rest
        for n in range(1, 100_000):
            # Decay from previous step, then add impulse
            v = self.params.v_rest + (v - self.params.v_rest) * alpha + impulse_dv
            if v >= self.params.v_thresh:
                return n
        return None

    # ── NeuronModelBase hooks ───────────────────────────────────────

    def _init_state_tensors(
        self,
        n: int,
        device: Any,
        dtype: Any,
    ) -> dict[str, Any]:
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()
        v = torch.full((n,), self.params.v_rest, device=device, dtype=dtype)
        return {"v": v}

    def _step(
        self,
        state: dict[str, Any],
        inputs: NeuronInputs,
        ctx: StepContext,
    ) -> NeuronStepResult:
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        v = state["v"]
        dt = ctx.dt
        p = self.params

        # Get soma drive (default to zero if not provided)
        drive = inputs.drive.get(Compartment.SOMA)
        if drive is None:
            drive = torch.zeros_like(v)

        # LIF dynamics: exponential decay + drive
        alpha = math.exp(-dt / p.tau_mem)
        r = dt / p.tau_mem
        v_new = p.v_rest + (v - p.v_rest) * alpha + drive * r

        # Spike detection
        spikes = v_new >= p.v_thresh

        # Reset spiked neurons
        v_new = torch.where(
            spikes, torch.tensor(p.v_reset, device=v.device, dtype=v.dtype), v_new
        )

        # Update state in-place
        state["v"] = v_new

        return NeuronStepResult(
            spikes=spikes,
            voltage={Compartment.SOMA: v_new},
        )

    def _reset_state_tensors(self, state: dict[str, Any]) -> None:
        state["v"].fill_(self.params.v_rest)
