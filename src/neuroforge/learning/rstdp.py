"""Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP).

Each edge maintains an eligibility trace *e* that accumulates STDP-like
correlations.  The actual weight change is gated by a reward signal:

    dw = learning_rate * reward * e

Eligibility trace update (simplified one-factor trace):

    e(t+1) = e(t) * decay + (pre_spike * A_plus - post_spike * A_minus)

Where:
    decay = exp(-dt / tau_e)    — trace decay per time step
    A_plus                      — potentiation amplitude (pre→post correlation)
    A_minus                     — depression amplitude (post→pre anti-correlation)

Weight clamping ensures weights stay in [w_min, w_max].

This simplified R-STDP is sufficient for the logic gate task and can
be extended with more sophisticated timing windows later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from neuroforge.contracts.learning import LearningBatch, LearningStepResult

__all__ = ["RSTDPRule", "RSTDPParams"]


@dataclass(frozen=True, slots=True)
class RSTDPParams:
    """Parameters for the R-STDP learning rule.

    Attributes
    ----------
    lr:
        Learning rate.
    tau_e:
        Eligibility trace time constant (seconds).
    a_plus:
        Potentiation amplitude (pre before post).
    a_minus:
        Depression amplitude (post before pre).
    w_min:
        Minimum weight.
    w_max:
        Maximum weight.
    """

    lr: float = 1e-3
    tau_e: float = 20e-3
    a_plus: float = 1.0
    a_minus: float = 1.0
    w_min: float = -1.0
    w_max: float = 1.0


class RSTDPRule:
    """Reward-modulated STDP learning rule.

    Implements ``ILearningRule`` protocol.
    """

    def __init__(self, params: RSTDPParams | None = None) -> None:
        self.params = params or RSTDPParams()

    # ── math helpers (public for testing) ───────────────────────────

    def eligibility_decay(self, dt: float) -> float:
        """Compute eligibility trace decay factor: exp(-dt / tau_e)."""
        return math.exp(-dt / self.params.tau_e)

    def predict_eligibility_update(
        self,
        e_prev: float,
        pre_spike: bool,
        post_spike: bool,
        dt: float,
    ) -> float:
        """Predict eligibility trace after one step.

        e_new = e_prev * decay + pre_spike * A_plus - post_spike * A_minus
        """
        decay = self.eligibility_decay(dt)
        delta = 0.0
        if pre_spike:
            delta += self.params.a_plus
        if post_spike:
            delta -= self.params.a_minus
        return e_prev * decay + delta

    def predict_dw(self, reward: float, eligibility: float) -> float:
        """Predict weight delta: lr * reward * eligibility."""
        return self.params.lr * reward * eligibility

    def predict_clamped_weight(self, w: float, dw: float) -> float:
        """Predict weight after clamping to [w_min, w_max]."""
        return max(self.params.w_min, min(self.params.w_max, w + dw))

    # ── ILearningRule implementation ────────────────────────────────

    def init_state(
        self,
        n_edges: int,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Allocate eligibility trace (zeros)."""
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()
        from neuroforge.core.torch_utils import resolve_device_dtype

        dev, dt_type = resolve_device_dtype(device, dtype)
        e = torch.zeros(n_edges, device=dev, dtype=dt_type)
        return {"eligibility": e}

    def step(
        self,
        state: dict[str, Any],
        batch: LearningBatch,
        dt: float,
    ) -> LearningStepResult:
        """Compute one R-STDP update.

        Parameters
        ----------
        state:
            Must contain ``"eligibility"`` tensor.
        batch:
            Current spike / weight / reward data.
        dt:
            Simulation time step.

        Returns
        -------
        LearningStepResult:
            Weight deltas and updated eligibility.
        """
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        p = self.params
        e = state["eligibility"]

        # Decay eligibility
        decay = math.exp(-dt / p.tau_e)
        e_new = e * decay

        # Add STDP contribution
        pre = batch.pre_spikes.float()
        post = batch.post_spikes.float()
        e_new = e_new + pre * p.a_plus - post * p.a_minus

        # Compute weight delta: modulated by reward
        reward = batch.reward
        if reward is None:
            reward = torch.tensor(0.0, device=e.device, dtype=e.dtype)

        dw = p.lr * reward * e_new

        # Update state in-place
        state["eligibility"] = e_new

        return LearningStepResult(
            dw=dw,
            new_eligibility=e_new,
        )

    def apply_dw(
        self,
        weights: Any,
        dw: Any,
    ) -> Any:
        """Apply weight delta with clamping.

        Parameters
        ----------
        weights:
            Current weight tensor.
        dw:
            Weight delta from ``step()``.

        Returns
        -------
        Tensor:
            Updated weights clamped to [w_min, w_max].
        """
        return (weights + dw).clamp(self.params.w_min, self.params.w_max)
