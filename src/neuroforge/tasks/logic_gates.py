"""Logic gate task — train an SNN to solve all 6 logic gates.

Architecture (simple gates: AND, OR, NAND, NOR):
    2 input neurons → 1 output neuron  (single layer)

Architecture (XOR, XNOR):
    2 input neurons → N hidden neurons → 1 output neuron  (two layers)

Training — simple gates (spike-count error rule):
1. Pick a random input pattern (00, 01, 10, 11)
2. Encode inputs as constant drive currents (rate encoding)
3. Run simulation for ``window`` steps
4. Decode output (binary decode based on spike count)
5. Update weights: dw = lr × error × input_rate  (perceptron-like)
6. Repeat until convergence or max trials

Training — XOR/XNOR (surrogate-gradient descent + Dale's Law):
1-4 same as above, but the forward pass builds a differentiable
    computation graph via a **surrogate spike function** (fast-sigmoid
    backward through the Heaviside spike non-linearity).
5. **Dale's Law** — effective weights are ``|w_raw| × sign_mask``:
    input neurons are excitatory, hidden neurons are split E/I.
6. MSE loss on differentiable spike count → ``loss.backward()`` →
    gradient-clipped weight update.

All 6 gates must pass for the milestone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.monitors import IEventBus

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
    per_pattern_streak:
        Each pattern must be individually correct for this many
        consecutive appearances before declaring convergence.
    n_hidden:
        Number of hidden neurons (used for XOR/XNOR only).
    n_inhibitory:
        Number of inhibitory hidden neurons (Dale's Law).
        The remaining ``n_hidden - n_inhibitory`` are excitatory.
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
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    dtype:
        Torch dtype string (``"float32"`` or ``"float64"``).
    """

    gate: str = "OR"
    max_trials: int = 5000
    window_steps: int = 50
    dt: float = 1e-3
    convergence_streak: int = 20
    per_pattern_streak: int = 5
    n_hidden: int = 6
    n_inhibitory: int = 2
    amplitude: float = 50.0
    spike_threshold: int = 3
    seed: int = 42
    lr: float = 5e-3
    w_min: float = -2.0
    w_max: float = 2.0
    device: str = "cpu"
    dtype: str = "float64"


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

    def __init__(
        self,
        config: LogicGateConfig | None = None,
        event_bus: IEventBus | None = None,
        *,
        stop_check: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config or LogicGateConfig()
        self._truth_table = GATE_TABLES[self.config.gate]
        self._bus = event_bus
        self._stop_check = stop_check

    # ── Event helpers ───────────────────────────────────────────────

    def _emit(
        self,
        topic: str,
        step: int,
        source: str,
        data: dict[str, Any],
        *,
        t: float = 0.0,
    ) -> None:
        """Publish an event if the bus is connected."""
        if self._bus is None:
            return
        from neuroforge.contracts.monitors import EventTopic, MonitorEvent

        self._bus.publish(
            MonitorEvent(
                topic=EventTopic(topic),
                step=step,
                t=t,
                source=source,
                data=data,
            )
        )

    @property
    def needs_hidden(self) -> bool:
        """Whether this gate requires a hidden layer."""
        return self.config.gate in ("XOR", "XNOR")

    def run(self) -> LogicGateResult:
        """Execute the training loop. Returns result with convergence status."""
        from neuroforge.core.torch_utils import require_torch, resolve_device_dtype

        torch = require_torch()
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        cfg = self.config
        dev, dt = resolve_device_dtype(cfg.device, cfg.dtype)

        # ── Import components ───────────────────────────────────────
        from neuroforge.encoding.rate import RateEncoder, RateEncoderParams
        from neuroforge.neurons.lif.model import LIFModel, LIFParams

        encoder = RateEncoder(RateEncoderParams(amplitude=cfg.amplitude))
        lif_params = LIFParams(tau_mem=20e-3, v_thresh=1.0, v_reset=0.0, v_rest=0.0)
        lif = LIFModel(lif_params)

        # ── Initialise weights + biases ─────────────────────────────
        # Pre-bind all variables so pyright flow analysis is satisfied.
        w_io = torch.empty(0, dtype=dt, device=dev)
        w_ih = torch.empty(0, 0, dtype=dt, device=dev)
        b_h = torch.empty(0, dtype=dt, device=dev)
        w_ho = torch.empty(0, dtype=dt, device=dev)
        b_o = torch.zeros(1, dtype=dt, device=dev)
        spike_fn: Any = None
        dale_in: Any = None
        dale_h: Any = None

        if self.needs_hidden:
            n_h = cfg.n_hidden
            w_ih = torch.empty(2, n_h, dtype=dt, device=dev).uniform_(-0.3, 0.3)
            b_h = torch.zeros(n_h, dtype=dt, device=dev)
            w_ho = torch.empty(n_h, dtype=dt, device=dev).uniform_(-0.3, 0.3)
            b_o = torch.zeros(1, dtype=dt, device=dev)
            # Enable gradient tracking for surrogate-gradient training.
            for _p in (w_ih, b_h, w_ho, b_o):
                _p.requires_grad_(True)
            spike_fn = self._make_surrogate_spike(torch)

            # Dale's Law: build per-population sign masks.
            from neuroforge.core.dales_law import make_dale_mask

            n_exc = n_h - cfg.n_inhibitory
            # Input neurons → excitatory (all outgoing weights ≥ 0).
            dale_in = make_dale_mask(2, 0, device=cfg.device, dtype=cfg.dtype)
            # Hidden neurons → first n_exc excitatory, rest inhibitory.
            dale_h = make_dale_mask(n_exc, cfg.n_inhibitory, device=cfg.device, dtype=cfg.dtype)
        else:
            w_io = torch.empty(2, dtype=dt, device=dev).uniform_(-0.3, 0.3)
            b_o = torch.zeros(1, dtype=dt, device=dev)

        # ── Emit run_start + topology ────────────────────────────
        self._emit(
            "run_start", 0, cfg.gate,
            {
                "task": "logic_gate",
                "device": cfg.device,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
            },
        )
        self._emit(
            "training_start", 0, cfg.gate,
            {"gate": cfg.gate, "max_trials": cfg.max_trials,
             "window_steps": cfg.window_steps,
             "per_pattern_streak": cfg.per_pattern_streak},
        )
        if self.needs_hidden:
            topo_data: dict[str, Any] = {
                "layers": ["input(2)", f"hidden({cfg.n_hidden})", "output(1)"],
                "edges": [
                    {"src": "input", "dst": "hidden",
                     "weights": w_ih.detach()},
                    {"src": "hidden", "dst": "output",
                     "weights": w_ho.detach()},
                ],
            }
        else:
            topo_data = {
                "layers": ["input(2)", "output(1)"],
                "edges": [
                    {"src": "input", "dst": "output",
                     "weights": w_io.detach()},
                ],
            }
        self._emit("topology", 0, cfg.gate, topo_data)

        # ── Training loop ───────────────────────────────────────────
        patterns = [(0, 0), (0, 1), (1, 0), (1, 1)]
        streak = 0
        pattern_streaks = {p: 0 for p in patterns}
        accuracy_history: list[float] = []
        recent_correct: list[bool] = []

        for trial in range(cfg.max_trials):
            # Check for external cancellation.
            if self._stop_check is not None and self._stop_check():
                weights = self._collect_weights(
                    w_ih if self.needs_hidden else w_io,
                    w_ho if self.needs_hidden else None,
                    b_o,
                    b_h if self.needs_hidden else None,
                )
                self._emit(
                    "training_end", trial, cfg.gate,
                    {"converged": False, "trials": trial, "stopped": True},
                )
                return LogicGateResult(
                    gate=cfg.gate,
                    converged=False,
                    trials=trial,
                    final_weights=weights,
                    accuracy_history=accuracy_history,
                )

            inp = random.choice(patterns)
            expected = self._truth_table[inp]

            # Encode inputs
            drives = encoder.encode(
                torch.tensor([float(inp[0]), float(inp[1])], dtype=dt, device=dev)
            )

            hid_counts = torch.zeros(0, dtype=dt, device=dev)
            out_diff: Any = None
            if self.needs_hidden:
                # Zero gradients before the differentiable forward pass.
                for _p in (w_ih, b_h, w_ho, b_o):
                    if _p.grad is not None:
                        _p.grad.zero_()
                out_diff, out_count, in_counts, hid_counts = (
                    self._forward_hidden(
                        drives, w_ih, b_h, w_ho, b_o, cfg, torch, spike_fn,
                        dale_in, dale_h,
                    )
                )
            else:
                out_count, in_counts = self._forward_simple(
                    lif, drives, w_io, b_o, cfg.dt, cfg.window_steps, torch,
                    device=cfg.device, dtype=cfg.dtype,
                )

            # Decode
            predicted = 1 if out_count >= cfg.spike_threshold else 0
            correct = predicted == expected

            # Error signal
            error = float(expected - predicted)

            # Weight update
            if self.needs_hidden:
                if error != 0.0:
                    # Surrogate-gradient MSE loss on spike count.
                    target = float(expected) * cfg.spike_threshold
                    loss = (out_diff - target) ** 2
                    loss.backward()
                    with torch.no_grad():
                        for _p in (w_ih, b_h, w_ho, b_o):
                            if _p.grad is not None:
                                _p.grad.clamp_(-1.0, 1.0)
                                _p.sub_(cfg.lr * _p.grad)
                                _p.clamp_(cfg.w_min, cfg.w_max)
            else:
                self._update_simple(w_io, b_o, in_counts, error, cfg, torch)

            # Track convergence
            recent_correct.append(correct)
            if len(recent_correct) > cfg.convergence_streak:
                recent_correct.pop(0)
            accuracy = sum(recent_correct) / len(recent_correct)
            accuracy_history.append(accuracy)

            # ── Emit per-trial events ───────────────────────────────
            # Build per-neuron spike data for the dashboard.
            spike_data: dict[str, Any] = {
                "input": list(inp),
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "error": error,
                "accuracy": accuracy,
                "out_spike_count": out_count,
                "input_spikes": in_counts.detach().cpu().tolist(),
            }
            if self.needs_hidden:
                spike_data["hidden_spikes"] = hid_counts.detach().cpu().tolist()
            spike_data["output_spikes"] = [out_count]
            self._emit("training_trial", trial, cfg.gate, spike_data)
            # Lightweight scalar for ArtifactWriter.
            self._emit(
                "scalar", trial, cfg.gate,
                {
                    "trial": trial,
                    "epoch": 0,
                    "gate": cfg.gate,
                    "accuracy": accuracy,
                    "error": error,
                    "correct": correct,
                },
            )
            # Emit weight snapshot (every trial — monitors bound memory).
            if self.needs_hidden:
                self._emit(
                    "weight", trial, f"{cfg.gate}/input-hidden",
                    {"weights": w_ih.detach()},
                )
                self._emit(
                    "weight", trial, f"{cfg.gate}/hidden-output",
                    {"weights": w_ho.detach()},
                )
            else:
                self._emit(
                    "weight", trial, f"{cfg.gate}/input-output",
                    {"weights": w_io},
                )

            if correct:
                streak += 1
                pattern_streaks[inp] += 1
            else:
                streak = 0
                pattern_streaks[inp] = 0

            all_patterns_reliable = all(
                s >= cfg.per_pattern_streak
                for s in pattern_streaks.values()
            )
            if streak >= cfg.convergence_streak and all_patterns_reliable:
                weights = self._collect_weights(
                    w_ih if self.needs_hidden else w_io,
                    w_ho if self.needs_hidden else None,
                    b_o,
                    b_h if self.needs_hidden else None,
                )
                self._emit(
                    "training_end", trial + 1, cfg.gate,
                    {"converged": True, "trials": trial + 1},
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
        self._emit(
            "training_end", cfg.max_trials, cfg.gate,
            {"converged": False, "trials": cfg.max_trials},
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
        *,
        device: str = "cpu",
        dtype: str = "float64",
    ) -> tuple[int, Any]:
        """2→1 network. Returns (output_spike_count, input_spike_counts)."""
        from neuroforge.contracts.neurons import NeuronInputs, StepContext
        from neuroforge.contracts.types import Compartment

        input_state = lif.init_state(2, device, dtype)
        output_state = lif.init_state(1, device, dtype)

        out_count = 0
        in_counts = torch.zeros(2, dtype=weights.dtype, device=weights.device)

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
        drives: Any,
        w_ih: Any,
        b_h: Any,
        w_ho: Any,
        b_o: Any,
        cfg: LogicGateConfig,
        torch: Any,
        spike_fn: Any,
        dale_in: Any,
        dale_h: Any,
    ) -> tuple[Any, int, Any, Any]:
        """2→N→1 forward pass with surrogate gradients and Dale's Law.

        Uses inline LIF dynamics (tau_mem=20 ms, v_thresh=1.0, hard reset)
        so the computation graph is differentiable through the surrogate
        spike function.  Input neurons use hard spikes (non-trainable).

        Dale's Law is enforced via ``|w| × sign`` reparameterization:
        - Input neurons are excitatory → ``w_ih_eff = |w_ih|``
        - Hidden neurons obey *dale_h* signs → ``w_ho_eff = |w_ho| × sign``

        Returns
        -------
        out_diff : Tensor
            Differentiable output spike count (for ``loss.backward()``).
        out_count : int
            Integer output spike count (for convergence checking).
        in_counts : Tensor
            Input spike counts (detached).
        hid_counts : Tensor
            Hidden spike counts (detached).
        """
        import math as _math

        from neuroforge.core.dales_law import apply_dales_constraint

        n_h = w_ho.shape[0]
        alpha = _math.exp(-cfg.dt / 20e-3)
        r_factor = cfg.dt / 20e-3
        v_thresh = torch.tensor(1.0, dtype=w_ih.dtype, device=w_ih.device)

        # Dale's Law reparameterization.
        w_ih_eff = apply_dales_constraint(
            w_ih, dale_in.signs.unsqueeze(1),
        )  # [2, n_h] — input neurons excitatory
        w_ho_eff = apply_dales_constraint(
            w_ho, dale_h.signs,
        )  # [n_h] — hidden neurons E/I

        _dev = w_ih.device
        _dt = w_ih.dtype
        v_in = torch.zeros(2, dtype=_dt, device=_dev)
        v_hid = torch.zeros(n_h, dtype=_dt, device=_dev)
        v_out = torch.zeros(1, dtype=_dt, device=_dev)

        in_counts = torch.zeros(2, dtype=_dt, device=_dev)
        hid_counts = torch.zeros(n_h, dtype=_dt, device=_dev)
        out_diff = torch.zeros(1, dtype=_dt, device=_dev)

        for _s in range(cfg.window_steps):
            # Input neurons — hard spikes, not trainable.
            v_in = v_in * alpha + drives * r_factor
            in_spikes = (v_in >= 1.0).to(_dt)
            v_in = (v_in * (1.0 - in_spikes)).detach()
            in_counts = in_counts + in_spikes

            # Hidden neurons — surrogate spike for gradient flow.
            ih_current = (in_spikes @ w_ih_eff) + b_h
            v_hid = v_hid * alpha + ih_current * r_factor
            hid_spikes = spike_fn(v_hid, v_thresh)
            v_hid = v_hid * (1.0 - hid_spikes.detach())
            hid_counts = hid_counts + hid_spikes.detach()

            # Output neuron — surrogate spike.
            ho_current = (hid_spikes * w_ho_eff).sum() + b_o[0]
            v_out = v_out * alpha + ho_current.unsqueeze(0) * r_factor
            out_spikes = spike_fn(v_out, v_thresh)
            v_out = v_out * (1.0 - out_spikes.detach())
            out_diff = out_diff + out_spikes

        out_count = int(out_diff.detach().sum().item())
        return out_diff.sum(), out_count, in_counts.detach(), hid_counts.detach()

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

    @staticmethod
    def _make_surrogate_spike(torch: Any) -> Any:
        """Create a spike function with surrogate gradient for backprop.

        The forward pass produces hard Heaviside spikes (0 or 1).
        The backward pass uses a fast-sigmoid surrogate derivative
        so that gradients flow through the non-differentiable spike.
        """
        beta = 5.0

        class _Fn(torch.autograd.Function):  # type: ignore[misc]
            @staticmethod
            def forward(ctx: Any, v: Any, threshold: Any) -> Any:  # noqa: N805
                spikes = (v >= threshold).to(v.dtype)
                ctx.save_for_backward(v, threshold)
                return spikes

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:  # noqa: N805
                v, threshold = ctx.saved_tensors
                x = v - threshold
                sg = beta / (1.0 + beta * torch.abs(x)) ** 2
                return grad_output * sg, None

        return _Fn.apply

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
