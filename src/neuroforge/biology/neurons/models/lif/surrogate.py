"""Surrogate-gradient Leaky Integrate-and-Fire neuron model."""

from __future__ import annotations

import math
from typing import Any

from neuroforge.biology.compartments.types import Compartment
from neuroforge.biology.neurons.base import NeuronModelBase
from neuroforge.biology.neurons.models.lif.params import LIFParams as _LIFParams
from neuroforge.biology.neurons.state import NeuronInputs, NeuronStepResult, StepContext
from neuroforge.kernel.surrogate import surrogate_spike

__all__ = ["SurrogateLIFModel"]


class SurrogateLIFModel(NeuronModelBase):
    """LIF dynamics with surrogate-gradient spiking."""

    def __init__(
        self,
        params: _LIFParams | None = None,
        *,
        beta: float = 5.0,
    ) -> None:
        self.params = params or _LIFParams()
        self.beta = beta

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
        spikes = surrogate_spike(v_new - p.v_thresh, beta=self.beta)

        v_reset = torch.tensor(p.v_reset, device=v.device, dtype=v.dtype)
        v_new = v_new * (1.0 - spikes.detach()) + v_reset * spikes.detach()
        state["v"] = v_new

        return NeuronStepResult(
            spikes=spikes,
            voltage={Compartment.SOMA: v_new},
        )

    def _reset_state_tensors(self, state: dict[str, Any]) -> None:
        state["v"].fill_(self.params.v_rest)
