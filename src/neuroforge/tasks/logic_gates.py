"""Logic gate task — train an SNN to solve all 6 logic gates.

Architecture (simple gates: AND, OR, NAND, NOR):
    2 input neurons → 1 output neuron  (single layer)

Architecture (XOR, XNOR):
    2 input neurons → N hidden neurons → 1 output neuron  (two layers)

Training loop (spike-count error learning):
1. Pick a random input pattern (00, 01, 10, 11)
2. Encode inputs as constant drive currents (rate encoding)
3. Run simulation for ``window`` steps
4. Count input, (hidden,) and output spikes
5. Decode output (binary decode based on spike count)
6. Compute error signal: expected - predicted  (+1 / 0 / -1)
7. Update weights: dw = lr * error * input_rate  (perceptron-like rule
   operating on spike counts — biologically plausible rate-coded update)
8. Repeat until convergence or max trials

All 6 gates must pass for the milestone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

__all__ = ["LogicGateTask", "LogicGateConfig", "LogicGateResult", "GATE_TABLES"]


# ── Truth tables ────────────────────────────────────────────────────

GATE_TABLES: dict[str, dict[tuple[int, int], int]] = {
    "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
    "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    "NAND": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "NOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
    "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
    "XNOR": {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
}


@dataclass(frozen=True, slots=True)
class LogicGateConfig:
    """Configuration for the logic gate task.

    Attributes
    ----------
    gate:
        Gate name (AND, OR, NAND, NOR, XOR, XNOR).
    max_trials:
        Maximum training trials before declaring failure.
    window_steps:
        Simulation steps per trial (presentation window).
    dt:
        Simulation time step.
    convergence_streak:
        Number of consecutive correct trials to declare convergence.
    n_hidden:
        Number of hidden neurons (used for XOR/XNOR only).
    amplitude:
        Input encoding amplitude.
    spike_threshold:
        Spike count threshold for binary output decode.
    seed:
        Random seed.
    lr:
        Learning rate.
    w_min:
        Minimum weight.
    w_max:
        Maximum weight.
    """

    gate: str = "OR"
    max_trials: int = 5000
    window_steps: int = 50
    dt: float = 1e-3
    convergence_streak: int = 20
    n_hidden: int = 6
    amplitude: float = 50.0
    spike_threshold: int = 3
    seed: int = 42
    lr: float = 0.02
    w_min: float = -2.0
    w_max: float = 2.0


@dataclass
class LogicGateResult:
    """Result of a logic gate task run.

    Attributes
    ----------
    gate:
        Gate name.
    converged:
        Whether the task converged.
    trials:
        Number of trials run.
    final_weights:
        Final weight tensors (by projection name).
    accuracy_history:
        Sliding window accuracy at each trial.
    """

    gate: str
    converged: bool
    trials: int
    final_weights: dict[str, Any] = field(default_factory=lambda: {})
    accuracy_history: list[float] = field(default_factory=lambda: [])


class LogicGateTask:
    """Trains an SNN on a single logic gate via spike-count error learning.

    Usage::

        task = LogicGateTask(LogicGateConfig(gate="XOR"))
        result = task.run()
        assert result.converged
    """

    def __init__(self, config: LogicGateConfig | None = None) -> None:
        self.config = config or LogicGateConfig()
        self._truth_table = GATE_TABLES[self.config.gate]

    @property
    def needs_hidden(self) -> bool:
        """Whether this gate requires a hidden layer."""
        return self.config.gate in ("XOR", "XNOR")

    def run(self) -> LogicGateResult:
        """Execute the training loop. Returns result with convergence status."""
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        cfg = self.config

        # ── Import components ───────────────────────────────────────
        from neuroforge.encoding.rate import RateEncoder, RateEncoderParams
        from neuroforge.neurons.lif.model import LIFModel, LIFParams

        encoder = RateEncoder(RateEncoderParams(amplitude=cfg.amplitude))
        lif_params = LIFParams(tau_mem=20e-3, v_thresh=1.0, v_reset=0.0, v_rest=0.0)
        lif = LIFModel(lif_params)

        # ── Initialise weights + biases ─────────────────────────────
        # Pre-bind all variables so pyright flow analysis is satisfied.
        w_io = torch.empty(0, dtype=torch.float64)
        w_ih = torch.empty(0, 0, dtype=torch.float64)
        b_h = torch.empty(0, dtype=torch.float64)
        w_ho = torch.empty(0, dtype=torch.float64)
        b_o = torch.zeros(1, dtype=torch.float64)

        if self.needs_hidden:
            n_h = cfg.n_hidden
            w_ih = torch.empty(2, n_h, dtype=torch.float64).uniform_(-0.3, 0.3)
            b_h = torch.zeros(n_h, dtype=torch.float64)
            w_ho = torch.empty(n_h, dtype=torch.float64).uniform_(-0.3, 0.3)
            b_o = torch.zeros(1, dtype=torch.float64)
        else:
            w_io = torch.empty(2, dtype=torch.float64).uniform_(-0.3, 0.3)
            b_o = torch.zeros(1, dtype=torch.float64)

        # ── Training loop ───────────────────────────────────────────
        patterns = [(0, 0), (0, 1), (1, 0), (1, 1)]
        streak = 0
        accuracy_history: list[float] = []
        recent_correct: list[bool] = []

        for trial in range(cfg.max_trials):
            inp = random.choice(patterns)
            expected = self._truth_table[inp]

            # Encode inputs
            drives = encoder.encode(
                torch.tensor([float(inp[0]), float(inp[1])], dtype=torch.float64)
            )

            hid_counts = torch.zeros(0, dtype=torch.float64)
            if self.needs_hidden:
                out_count, in_counts, hid_counts = self._forward_hidden(
                    lif,
                    drives,
                    w_ih,
                    b_h,
                    w_ho,
                    b_o,
                    cfg.dt,
                    cfg.window_steps,
                    torch,
                )
            else:
                out_count, in_counts = self._forward_simple(
                    lif, drives, w_io, b_o, cfg.dt, cfg.window_steps, torch
                )

            # Decode
            predicted = 1 if out_count >= cfg.spike_threshold else 0
            correct = predicted == expected

            # Error signal: direction of needed weight change
            # +1 when should fire more, -1 when should fire less, 0 when correct
            error = float(expected - predicted)

            # Update weights + biases using spike-count error rule
            if self.needs_hidden:
                self._update_hidden(
                    w_ih,
                    b_h,
                    w_ho,
                    b_o,
                    in_counts,
                    hid_counts,
                    error,
                    cfg,
                    torch,
                )
            else:
                self._update_simple(w_io, b_o, in_counts, error, cfg, torch)

            # Track convergence
            recent_correct.append(correct)
            if len(recent_correct) > cfg.convergence_streak:
                recent_correct.pop(0)
            accuracy = sum(recent_correct) / len(recent_correct)
            accuracy_history.append(accuracy)

            if correct:
                streak += 1
            else:
                streak = 0

            if streak >= cfg.convergence_streak:
                weights = self._collect_weights(
                    w_ih if self.needs_hidden else w_io,
                    w_ho if self.needs_hidden else None,
                    b_o,
                    b_h if self.needs_hidden else None,
                )
                return LogicGateResult(
                    gate=cfg.gate,
                    converged=True,
                    trials=trial + 1,
                    final_weights=weights,
                    accuracy_history=accuracy_history,
                )

        weights = self._collect_weights(
            w_ih if self.needs_hidden else w_io,
            w_ho if self.needs_hidden else None,
            b_o,
            b_h if self.needs_hidden else None,
        )
        return LogicGateResult(
            gate=cfg.gate,
            converged=False,
            trials=cfg.max_trials,
            final_weights=weights,
            accuracy_history=accuracy_history,
        )

    # ── Forward pass helpers ────────────────────────────────────────

    def _forward_simple(
        self,
        lif: Any,
        drives: Any,
        weights: Any,
        bias: Any,
        dt: float,
        window: int,
        torch: Any,
    ) -> tuple[int, Any]:
        """2→1 network. Returns (output_spike_count, input_spike_counts)."""
        from neuroforge.contracts.neurons import NeuronInputs, StepContext
        from neuroforge.contracts.types import Compartment

        input_state = lif.init_state(2, "cpu", "float64")
        output_state = lif.init_state(1, "cpu", "float64")

        out_count = 0
        in_counts = torch.zeros(2, dtype=torch.float64)

        for s in range(window):
            ctx = StepContext(dt=dt, step=s, t=s * dt)

            inp_result = lif.step(
                input_state,
                NeuronInputs(drive={Compartment.SOMA: drives}),
                ctx,
            )
            in_counts += inp_result.spikes.float()

            syn_current = (
                (inp_result.spikes.float() * weights).sum() + bias[0]
            ).unsqueeze(0)

            out_result = lif.step(
                output_state,
                NeuronInputs(drive={Compartment.SOMA: syn_current}),
                ctx,
            )
            if out_result.spikes.any().item():
                out_count += 1

        return out_count, in_counts

    def _forward_hidden(
        self,
        lif: Any,
        drives: Any,
        w_ih: Any,
        b_h: Any,
        w_ho: Any,
        b_o: Any,
        dt: float,
        window: int,
        torch: Any,
    ) -> tuple[int, Any, Any]:
        """2→N→1 network. Returns (out_count, input_counts, hidden_counts)."""
        from neuroforge.contracts.neurons import NeuronInputs, StepContext
        from neuroforge.contracts.types import Compartment

        n_h = w_ho.shape[0]
        input_state = lif.init_state(2, "cpu", "float64")
        hidden_state = lif.init_state(n_h, "cpu", "float64")
        output_state = lif.init_state(1, "cpu", "float64")

        out_count = 0
        in_counts = torch.zeros(2, dtype=torch.float64)
        hid_counts = torch.zeros(n_h, dtype=torch.float64)

        for s in range(window):
            ctx = StepContext(dt=dt, step=s, t=s * dt)

            inp_result = lif.step(
                input_state,
                NeuronInputs(drive={Compartment.SOMA: drives}),
                ctx,
            )
            in_counts += inp_result.spikes.float()

            ih_current = (inp_result.spikes.float().unsqueeze(1) * w_ih).sum(0) + b_h

            hid_result = lif.step(
                hidden_state,
                NeuronInputs(drive={Compartment.SOMA: ih_current}),
                ctx,
            )
            hid_counts += hid_result.spikes.float()

            ho_current = ((hid_result.spikes.float() * w_ho).sum() + b_o[0]).unsqueeze(
                0
            )

            out_result = lif.step(
                output_state,
                NeuronInputs(drive={Compartment.SOMA: ho_current}),
                ctx,
            )
            if out_result.spikes.any().item():
                out_count += 1

        return out_count, in_counts, hid_counts

    # ── Weight update helpers ───────────────────────────────────────

    def _update_simple(
        self,
        weights: Any,
        bias: Any,
        in_counts: Any,
        error: float,
        cfg: LogicGateConfig,
        torch: Any,
    ) -> None:
        """Spike-count perceptron update for 2→1 network.

        dw_i = lr * error * input_rate_i
        d_bias = lr * error
        """
        if error == 0.0:
            return
        in_rates = in_counts / cfg.window_steps
        dw = cfg.lr * error * in_rates
        weights.add_(dw)
        weights.clamp_(cfg.w_min, cfg.w_max)
        bias.add_(cfg.lr * error)
        bias.clamp_(cfg.w_min, cfg.w_max)

    def _update_hidden(
        self,
        w_ih: Any,
        b_h: Any,
        w_ho: Any,
        b_o: Any,
        in_counts: Any,
        hid_counts: Any,
        error: float,
        cfg: LogicGateConfig,
        torch: Any,
    ) -> None:
        """Spike-count error-backprop update for 2→N→1 network.

        Output layer: dw_ho_j = lr * error * hidden_rate_j
        Hidden layer: dw_ih_ij = lr * (error * sign(w_ho_j)) * input_rate_i
        Biases update with activity = 1 (always active).
        """
        if error == 0.0:
            return

        in_rates = in_counts / cfg.window_steps  # [2]
        hid_rates = hid_counts / cfg.window_steps  # [n_h]

        # Output layer
        dw_ho = cfg.lr * error * hid_rates
        w_ho.add_(dw_ho)
        w_ho.clamp_(cfg.w_min, cfg.w_max)
        b_o.add_(cfg.lr * error)
        b_o.clamp_(cfg.w_min, cfg.w_max)

        # Hidden layer — propagate error through output weights
        hidden_error = error * w_ho.sign()  # [n_h]
        dw_ih = cfg.lr * in_rates.unsqueeze(1) * hidden_error.unsqueeze(0)  # [2, n_h]
        w_ih.add_(dw_ih)
        w_ih.clamp_(cfg.w_min, cfg.w_max)
        b_h.add_(cfg.lr * hidden_error)
        b_h.clamp_(cfg.w_min, cfg.w_max)

    def _collect_weights(
        self,
        w1: Any,
        w2: Any | None,
        b_o: Any,
        b_h: Any | None = None,
    ) -> dict[str, Any]:
        """Collect weights and biases for the result."""
        weights: dict[str, Any] = {
            "layer_1": w1.clone().detach(),
            "bias_o": b_o.clone().detach(),
        }
        if w2 is not None:
            weights["layer_2"] = w2.clone().detach()
        if b_h is not None:
            weights["bias_h"] = b_h.clone().detach()
        return weights
