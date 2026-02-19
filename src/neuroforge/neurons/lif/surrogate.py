"""Surrogate-gradient LIF neuron model.

Identical LIF dynamics to :class:`~neuroforge.neurons.lif.model.LIFModel`
but produces **float** spikes (0.0 / 1.0) via a surrogate spike function
so that ``torch.autograd`` can back-propagate through the spike decision.

Voltage reset uses ``spikes.detach()`` to avoid gradient flow through
the reset path (consistent with the existing training task style).
"""

from __future__ import annotations

import math
from typing import Any

from neuroforge.contracts.neurons import NeuronInputs, NeuronStepResult, StepContext
from neuroforge.contracts.types import Compartment
from neuroforge.core.surrogate import surrogate_spike
from neuroforge.neurons.base import NeuronModelBase
from neuroforge.neurons.lif.model import LIFParams

__all__ = ["SurrogateLIFModel"]


class SurrogateLIFModel(NeuronModelBase):
    """Leaky Integrate-and-Fire with surrogate-gradient spiking.

    Parameters
    ----------
    params:
        LIF parameters (shared with the standard :class:`LIFModel`).
    beta:
        Steepness of the fast-sigmoid surrogate gradient (default 5.0).
    """

    def __init__(
        self,
        params: LIFParams | None = None,
        *,
        beta: float = 5.0,
    ) -> None:
        self.params = params or LIFParams()
        self.beta = beta

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

        # Soma drive (default to zero if absent).
        drive = inputs.drive.get(Compartment.SOMA)
        if drive is None:
            drive = torch.zeros_like(v)

        # LIF dynamics — identical to LIFModel.
        alpha = math.exp(-dt / p.tau_mem)
        r = dt / p.tau_mem
        v_new = p.v_rest + (v - p.v_rest) * alpha + drive * r

        # Surrogate spike: float 0/1 with smooth backward gradient.
        spikes = surrogate_spike(v_new - p.v_thresh, beta=self.beta)

        # Reset using *detached* spikes so the reset doesn't carry
        # gradient (same convention as the existing training tasks).
        v_reset = torch.tensor(p.v_reset, device=v.device, dtype=v.dtype)
        v_new = v_new * (1.0 - spikes.detach()) + v_reset * spikes.detach()

        # Update state in-place.
        state["v"] = v_new

        return NeuronStepResult(
            spikes=spikes,
            voltage={Compartment.SOMA: v_new},
        )

    def _reset_state_tensors(self, state: dict[str, Any]) -> None:
        state["v"].fill_(self.params.v_rest)
