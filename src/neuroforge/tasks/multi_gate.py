"""Multi-gate task — train ONE brain on ALL 6 logic gates simultaneously.

Architecture
------------
(2 data + 6 gate-context) input neurons → N hidden → 1 output

The 6 gate-context neurons form a one-hot encoding so the brain knows
which gate it is currently solving.  **All** weights are shared —
input→hidden (``w_ih``), hidden→output (``w_ho``), and biases
(``b_h``, ``b_o``).  The network must learn to route signals
differently based on the gate-context input alone.

Training uses surrogate-gradient descent (fast-sigmoid β=5) with
Dale's Law and the Adam optimiser.  Each *epoch* presents all 24
gate × pattern combos in shuffled order for balanced coverage.
"""

from __future__ import annotations

import math as _math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec
from neuroforge.tasks.logic_gates import GATE_TABLES

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.monitors import IEventBus

__all__ = [
    "MultiGateTask",
    "MultiGateConfig",
    "MultiGateResult",
    "ALL_GATES",
    "GATE_INDEX",
]

ALL_GATES: tuple[str, ...] = ("AND", "OR", "NAND", "NOR", "XOR", "XNOR")
GATE_INDEX: dict[str, int] = {g: i for i, g in enumerate(ALL_GATES)}


@dataclass(frozen=True, slots=True)
class MultiGateConfig:
    """Configuration for the multi-gate task.

    Attributes
    ----------
    gates:
        Which gates to train (default: all six).
    max_epochs:
        Maximum training epochs (each epoch = one pass through every
        gate × pattern combo, i.e. 24 trials for 6 gates).
    window_steps:
        LIF simulation steps per trial.
    dt:
        Simulation time-step (seconds).
    convergence_streak:
        Overall consecutive-correct streak required.
    per_pattern_streak:
        Per (gate, pattern) combo streak required.
    n_hidden:
        Hidden neuron count.
    n_inhibitory:
        Inhibitory hidden neurons (Dale's Law).
    amplitude:
        Rate-encoding amplitude for active inputs.
    spike_threshold:
        Output spike count for binary decode (≥ threshold ⇒ 1).
    seed:
        Random seed.
    lr:
        Adam learning rate.
    w_min:
        Weight clamp minimum.
    w_max:
        Weight clamp maximum.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    dtype:
        Torch dtype string (``"float32"`` or ``"float64"``).
    """

    gates: tuple[str, ...] = ALL_GATES
    max_epochs: int = 1_500
    window_steps: int = 25
    dt: float = 1e-3
    convergence_streak: int = 50
    per_pattern_streak: int = 5
    n_hidden: int = 24
    n_inhibitory: int = 6
    amplitude: float = 50.0
    spike_threshold: int = 3
    seed: int = 42
    lr: float = 5e-3
    w_min: float = -2.0
    w_max: float = 2.0
    device: str = "cpu"
    dtype: str = "float64"


@dataclass
class MultiGateResult:
    """Result of a multi-gate training run.

    Attributes
    ----------
    gates:
        Gates that were trained.
    converged:
        Whether *all* gates converged.
    trials:
        Total trials run.
    epochs:
        Epochs completed.
    per_gate_converged:
        Convergence status per gate.
    final_weights:
        Shared weight tensors (``w_ih``, ``w_ho``, ``b_h``, ``b_o``).
    accuracy_history:
        Rolling accuracy per trial.
    """

    gates: tuple[str, ...]
    converged: bool
    trials: int
    epochs: int = 0
    per_gate_converged: dict[str, bool] = field(default_factory=lambda: {})
    final_weights: dict[str, Any] = field(default_factory=lambda: {})
    accuracy_history: list[float] = field(default_factory=lambda: [])


class MultiGateTask:
    """Train one brain on all logic gates with gate-context inputs.

    Each training *epoch* presents every (gate, pattern) combo once in
    shuffled order.  All weights are fully shared — the network must
    learn to route differently based on the one-hot gate-context input.

    Usage::

        task = MultiGateTask(MultiGateConfig())
        result = task.run()
        assert result.converged
    """

    def __init__(
        self,
        config: MultiGateConfig | None = None,
        event_bus: IEventBus | None = None,
        *,
        stop_check: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config or MultiGateConfig()
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

    # ── Network spec builder ────────────────────────────────────────

    def _build_network_spec(self) -> NetworkSpec:
        """Return a :class:`NetworkSpec` for the multi-gate architecture."""
        cfg = self.config
        gates = list(cfg.gates)
        n_gates = len(gates)
        n_input = 2 + n_gates

        return NetworkSpec(
            populations=[
                PopulationSpec("input", n_input, "lif"),
                PopulationSpec("hidden", cfg.n_hidden, "lif"),
                PopulationSpec("output", 1, "lif"),
            ],
            projections=[
                ProjectionSpec(
                    "input_hidden", "input", "hidden", "static",
                    topology={
                        "type": "dense", "init": "uniform",
                        "low": -0.3, "high": 0.3, "bias": True,
                    },
                ),
                ProjectionSpec(
                    "hidden_output", "hidden", "output", "static",
                    topology={
                        "type": "dense", "init": "uniform",
                        "low": -0.3, "high": 0.3, "bias": True,
                    },
                ),
            ],
            metadata={"task": "multi_gate", "gates": gates},
        )

    # ── Main training loop ──────────────────────────────────────────

    def run(self) -> MultiGateResult:
        """Train all gates with shared weights. Returns result."""
        from neuroforge.core.dales_law import apply_dales_constraint, make_dale_mask
        from neuroforge.core.torch_utils import require_torch, resolve_device_dtype
        from neuroforge.encoding.rate import RateEncoder, RateEncoderParams

        torch = require_torch()
        cfg = self.config
        dev, tdt = resolve_device_dtype(cfg.device, cfg.dtype)

        gates = list(cfg.gates)
        n_gates = len(gates)
        n_input = 2 + n_gates  # 2 data + one-hot gate context
        n_h = cfg.n_hidden

        # ── Build network via NetworkFactory ────────────────────────
        from neuroforge.network.factory import NetworkFactory, to_topology_json
        from neuroforge.neurons.registry import NEURON_MODELS
        from neuroforge.synapses.registry import SYNAPSE_MODELS

        spec = self._build_network_spec()
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(spec, device=cfg.device, dtype=cfg.dtype, seed=cfg.seed)

        # Extract trainable weight tensors owned by the projections.
        proj_ih = engine.projections["input_hidden"]
        proj_ho = engine.projections["hidden_output"]
        w_ih = proj_ih.state["weight_matrix"]   # [n_input, n_h]
        b_h = proj_ih.state["bias"]             # [n_h]
        w_ho = proj_ho.state["weight_matrix"]   # [n_h]
        b_o = proj_ho.state["bias"]             # [1]

        params = [w_ih, b_h, w_ho, b_o]
        for _p in params:
            _p.requires_grad_(True)

        optimizer = torch.optim.Adam(params, lr=cfg.lr)

        spike_fn = self._make_surrogate_spike(torch)

        # Dale's Law masks.
        dale_in = make_dale_mask(n_input, 0, device=cfg.device, dtype=cfg.dtype)
        n_exc = n_h - cfg.n_inhibitory
        dale_h = make_dale_mask(n_exc, cfg.n_inhibitory, device=cfg.device, dtype=cfg.dtype)

        encoder = RateEncoder(RateEncoderParams(amplitude=cfg.amplitude))

        # ── Emit run_start + topology ─────────────────────────────
        self._emit(
            "run_start", 0, "MULTI",
            {
                "task": "multi_gate",
                "device": cfg.device,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
            },
        )
        self._emit(
            "training_start", 0, "MULTI",
            {
                "gate": "MULTI",
                "gates": list(gates),
                "max_epochs": cfg.max_epochs,
                "window_steps": cfg.window_steps,
                "per_pattern_streak": cfg.per_pattern_streak,
                "n_input": n_input,
                "n_hidden": n_h,
            },
        )
        self._emit("topology", 0, "MULTI", to_topology_json(engine))

        # ── Convergence tracking ────────────────────────────────────
        patterns: list[tuple[int, int]] = [
            (0, 0), (0, 1), (1, 0), (1, 1),
        ]
        all_combos: list[tuple[str, tuple[int, int]]] = [
            (g, p) for g in gates for p in patterns
        ]
        combo_streaks: dict[tuple[str, tuple[int, int]], int] = {
            c: 0 for c in all_combos
        }
        streak = 0
        recent_correct: list[bool] = []
        accuracy_history: list[float] = []

        # ── LIF constants ───────────────────────────────────────────
        alpha = _math.exp(-cfg.dt / 20e-3)
        r_factor = cfg.dt / 20e-3
        v_thresh = torch.tensor(1.0, dtype=tdt, device=dev)

        trial = 0

        for epoch in range(cfg.max_epochs):
            # Each epoch: present every combo in shuffled order.
            order = list(all_combos)
            random.shuffle(order)

            for gate, inp in order:
                if self._stop_check is not None and self._stop_check():
                    return self._make_result(
                        cfg, gates, combo_streaks, False, trial,
                        epoch, w_ih, w_ho, b_h, b_o, accuracy_history,
                        stopped=True,
                    )

                gate_idx = GATE_INDEX[gate]
                expected = GATE_TABLES[gate][inp]

                # Build drive vector: [data0, data1, ctx0..ctx5].
                data_drives = encoder.encode(
                    torch.tensor(
                        [float(inp[0]), float(inp[1])],
                        dtype=tdt, device=dev,
                    ),
                )
                ctx_drives = torch.zeros(n_gates, dtype=tdt, device=dev)
                ctx_drives[gate_idx] = cfg.amplitude
                drives = torch.cat([data_drives, ctx_drives])

                optimizer.zero_grad()

                # Dale's Law reparameterization.
                w_ih_eff = apply_dales_constraint(
                    w_ih, dale_in.signs.unsqueeze(1),
                )
                w_ho_eff = apply_dales_constraint(
                    w_ho, dale_h.signs,
                )

                # ── Forward pass (inline LIF) ───────────────────────
                v_in = torch.zeros(n_input, dtype=tdt, device=dev)
                v_hid = torch.zeros(n_h, dtype=tdt, device=dev)
                v_out = torch.zeros(1, dtype=tdt, device=dev)
                hid_counts = torch.zeros(n_h, dtype=tdt, device=dev)
                out_diff = torch.zeros(1, dtype=tdt, device=dev)

                for _s in range(cfg.window_steps):
                    # Input neurons — hard threshold spikes.
                    v_in = v_in * alpha + drives * r_factor
                    in_spikes = (v_in >= 1.0).to(tdt)
                    v_in = (v_in * (1.0 - in_spikes)).detach()

                    # Hidden neurons — surrogate spike.
                    ih_current = (in_spikes @ w_ih_eff) + b_h
                    v_hid = v_hid * alpha + ih_current * r_factor
                    hid_spikes = spike_fn(v_hid, v_thresh)
                    v_hid = v_hid * (1.0 - hid_spikes.detach())
                    hid_counts = hid_counts + hid_spikes.detach()

                    # Output neuron — surrogate spike.
                    ho_current = (
                        (hid_spikes * w_ho_eff).sum() + b_o[0]
                    )
                    v_out = (
                        v_out * alpha
                        + ho_current.unsqueeze(0) * r_factor
                    )
                    out_spikes = spike_fn(v_out, v_thresh)
                    v_out = v_out * (1.0 - out_spikes.detach())
                    out_diff = out_diff + out_spikes

                out_count = int(out_diff.detach().sum().item())
                predicted = 1 if out_count >= cfg.spike_threshold else 0
                correct = predicted == expected

                # ── Adam weight update on error ─────────────────────
                if not correct:
                    target = float(expected) * cfg.spike_threshold
                    loss = (out_diff.sum() - target) ** 2
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for _p in params:
                            _p.clamp_(cfg.w_min, cfg.w_max)

                # ── Convergence tracking ────────────────────────────
                recent_correct.append(correct)
                if len(recent_correct) > cfg.convergence_streak:
                    recent_correct.pop(0)
                accuracy = sum(recent_correct) / len(recent_correct)
                accuracy_history.append(accuracy)

                combo_key = (gate, inp)
                if correct:
                    streak += 1
                    combo_streaks[combo_key] += 1
                else:
                    streak = 0
                    combo_streaks[combo_key] = 0

                # ── Emit events ─────────────────────────────────────
                error_val = float(expected - predicted)
                spike_data: dict[str, Any] = {
                    "gate": gate,
                    "input": list(inp),
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                    "error": error_val,
                    "accuracy": accuracy,
                    "out_spike_count": out_count,
                    "hidden_spikes": hid_counts.detach().cpu().tolist(),
                    "output_spikes": [out_count],
                }
                self._emit(
                    "training_trial", trial, f"MULTI/{gate}", spike_data,
                )
                # Lightweight scalar for ArtifactWriter.
                self._emit(
                    "scalar", trial, "MULTI",
                    {
                        "trial": trial,
                        "epoch": epoch,
                        "gate": gate,
                        "accuracy": accuracy,
                        "error": error_val,
                        "correct": correct,
                    },
                )

                if trial % 10 == 0:
                    self._emit(
                        "weight", trial, "MULTI/input-hidden",
                        {"weights": w_ih.detach()},
                    )
                    self._emit(
                        "weight", trial, "MULTI/hidden-output",
                        {"weights": w_ho.detach()},
                    )

                trial += 1

            # ── End-of-epoch convergence check ──────────────────────
            all_combos_reliable = all(
                s >= cfg.per_pattern_streak
                for s in combo_streaks.values()
            )
            if streak >= cfg.convergence_streak and all_combos_reliable:
                return self._make_result(
                    cfg, gates, combo_streaks, True, trial,
                    epoch + 1, w_ih, w_ho, b_h, b_o, accuracy_history,
                )

        return self._make_result(
            cfg, gates, combo_streaks, False, trial,
            cfg.max_epochs, w_ih, w_ho, b_h, b_o, accuracy_history,
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _make_result(
        self,
        cfg: MultiGateConfig,
        gates: list[str],
        combo_streaks: dict[tuple[str, tuple[int, int]], int],
        converged: bool,
        trials: int,
        epochs: int,
        w_ih: Any,
        w_ho: Any,
        b_h: Any,
        b_o: Any,
        accuracy_history: list[float],
        *,
        stopped: bool = False,
    ) -> MultiGateResult:
        """Build a ``MultiGateResult`` and emit the training_end event."""
        patterns: list[tuple[int, int]] = [
            (0, 0), (0, 1), (1, 0), (1, 1),
        ]
        per_gate: dict[str, bool] = {}
        for g in gates:
            per_gate[g] = all(
                combo_streaks[(g, p)] >= cfg.per_pattern_streak
                for p in patterns
            )

        end_data: dict[str, Any] = {
            "converged": converged,
            "trials": trials,
            "epochs": epochs,
            "per_gate_converged": per_gate,
        }
        if stopped:
            end_data["stopped"] = True
        self._emit("training_end", trials, "MULTI", end_data)

        return MultiGateResult(
            gates=tuple(gates),
            converged=converged,
            trials=trials,
            epochs=epochs,
            per_gate_converged=per_gate,
            final_weights={
                "w_ih": w_ih.clone().detach(),
                "w_ho": w_ho.clone().detach(),
                "b_h": b_h.clone().detach(),
                "b_o": b_o.clone().detach(),
            },
            accuracy_history=accuracy_history,
        )

    @staticmethod
    def _make_surrogate_spike(torch: Any) -> Any:
        """Surrogate spike function (fast-sigmoid backward)."""
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
