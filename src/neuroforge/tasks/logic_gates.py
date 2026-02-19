"""Logic gate task — train an SNN to solve all 6 logic gates.

Architecture (all gates — one brain, one topology):
    H*W input neurons → N hidden neurons → 1 output neuron

Every gate uses the **same** architecture with **Dale's Law** enforced
on every projection: effective weights are ``|w_raw| × sign_mask``.
Input neurons are excitatory; hidden neurons are split E/I.

Inputs are **pixel-encoded**: each 2-bit pattern is rendered as a tiny
``(H, W)`` image with two bright squares at fixed coordinates.  The
flattened image is rate-encoded into current drives.

Training — all gates use surrogate-gradient descent + Adam:
1. Pick a random input pattern (00, 01, 10, 11)
2. Render the pattern as a pixel image, flatten, rate-encode.
3. Run simulation for ``window_steps`` using population neuron models
   with inline synaptic-current computation so gradients flow through
   trainable weights.
4. Readout via :class:`SpikeCountReadout`; decode binary output.
5. On error: MSE-count (or BCE-logits) loss → ``loss.backward()`` →
   gradient-clipped Adam step → weight clamp.
6. Repeat until convergence or max trials.

All 6 gates must pass for the milestone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from neuroforge.core.torch_utils import smart_device
from neuroforge.learning.stats import grad_stats, tensor_stats

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.encoding import ILoss, IReadout
    from neuroforge.contracts.monitors import IEventBus
    from neuroforge.encoding.rate import RateEncoder

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
        Number of hidden neurons.
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
        Torch device string (``"cpu"``, ``"cuda"``, or ``"auto"``).
        ``"auto"`` (the default) picks the fastest device for the
        network size — CPU for small topologies, CUDA for large ones.
    dtype:
        Torch dtype string (``"float32"`` or ``"float64"``).
    image_h:
        Height of the pixel-encoded input image.
    image_w:
        Width of the pixel-encoded input image.
    loss_fn:
        Loss function factory key (``"mse_count"`` or ``"bce_logits"``).
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
    device: str = "auto"
    dtype: str = "float64"
    image_h: int = 8
    image_w: int = 8
    loss_fn: str = "mse_count"
    emit_param_grad_stats: bool = True
    stats_every_n_trials: int = 10

    def __post_init__(self) -> None:
        if self.stats_every_n_trials <= 0:
            msg = "stats_every_n_trials must be > 0"
            raise ValueError(msg)
        if self.device == "auto":
            n_total = self.image_h * self.image_w + self.n_hidden + 1
            object.__setattr__(
                self, "device", smart_device(n_total),
            )


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


# ── Pixel encoding helper ──────────────────────────────────────────


def _pattern_to_image(
    bits: tuple[int, int],
    h: int,
    w: int,
    torch_mod: Any,
) -> Any:
    """Convert a 2-bit input pattern to an ``[H, W]`` pixel image.

    Bit 0 controls a bright pixel at ``(2, 2)``.
    Bit 1 controls a bright pixel at ``(2, 5)``.

    Parameters
    ----------
    bits:
        ``(b0, b1)`` where each value is 0 or 1.
    h, w:
        Image height and width (default 8×8).
    torch_mod:
        The ``torch`` module.

    Returns
    -------
    Tensor:
        Float image ``[H, W]`` with values in ``{0.0, 1.0}``.
    """
    img = torch_mod.zeros(h, w)
    if bits[0]:
        img[2, 2] = 1.0
    if bits[1]:
        img[2, 5] = 1.0
    return img


def _param_grad_stats(
    w_ih: Any,
    w_ho: Any,
) -> dict[str, float]:
    w_ih_stats = tensor_stats(w_ih, "w_ih")
    w_ho_stats = tensor_stats(w_ho, "w_ho")
    g_ih_stats = grad_stats(w_ih, "ih")
    g_ho_stats = grad_stats(w_ho, "ho")
    return {
        "w_norm_ih": w_ih_stats["w_ih_norm"],
        "w_maxabs_ih": w_ih_stats["w_ih_maxabs"],
        "w_norm_ho": w_ho_stats["w_ho_norm"],
        "w_maxabs_ho": w_ho_stats["w_ho_maxabs"],
        "g_norm_ih": g_ih_stats["g_norm_ih"],
        "g_maxabs_ih": g_ih_stats["g_maxabs_ih"],
        "g_norm_ho": g_ho_stats["g_norm_ho"],
        "g_maxabs_ho": g_ho_stats["g_maxabs_ho"],
    }


class LogicGateTask:
    """Trains an SNN on a single logic gate.

    Uses :func:`build_gate_network` for network construction,
    pixel-encoded inputs, and factory-created readout / loss functions.
    All gates use surrogate gradients + Adam optimizer.

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

    # ── Forward pass ────────────────────────────────────────────────

    def _forward(
        self,
        drives: Any,
        gn: Any,
        cfg: LogicGateConfig,
        torch_mod: Any,
    ) -> tuple[Any, Any, Any]:
        """Run a full presentation window and accumulate output spikes.

        Uses population neuron models for dynamics and computes synaptic
        currents inline so that gradients flow through trainable weights.

        Parameters
        ----------
        drives:
            Encoded input drive tensor ``[N_in]``.
        gn:
            :class:`GateNetwork` (from ``build_gate_network``).
        cfg:
            Task configuration.
        torch_mod:
            The ``torch`` module.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]:
            ``(spike_tensor, in_counts, hid_counts)`` where
            *spike_tensor* is ``[T, 1]`` with differentiable output
            spikes, *in_counts* is detached ``[N_in]``, and
            *hid_counts* is detached ``[N_hid]`` (empty if no hidden).
        """
        from neuroforge.contracts.neurons import NeuronInputs, StepContext
        from neuroforge.contracts.types import Compartment

        pops = gn.engine._populations  # pyright: ignore[reportPrivateUsage]

        # Reset all neuron states and detach from prior graph.
        gn.engine.reset()
        for _pop in pops.values():
            for _k in list(_pop.state):
                _pop.state[_k] = _pop.state[_k].detach()

        inp_pop = pops["input"]
        hid_pop = pops["hidden"]
        out_pop = pops["output"]

        dev = drives.device
        dt_t = drives.dtype

        # Resolve effective weights (Dale's Law: |w| × sign).
        w_ih = gn.trainables["raw_w_in_to_hidden"]
        b_ih = gn.trainables["bias_in_to_hidden"]
        w_ho = gn.trainables["raw_w_hidden_to_out"]
        b_ho = gn.trainables["bias_hidden_to_out"]
        sign_in = gn.meta["sign_input"]
        sign_h = gn.meta["sign_hidden"]
        w_ih_eff = torch_mod.abs(w_ih) * sign_in.unsqueeze(1)
        w_ho_eff = torch_mod.abs(w_ho) * sign_h

        in_counts = torch_mod.zeros(inp_pop.n, dtype=dt_t, device=dev)
        hid_counts = torch_mod.zeros(hid_pop.n, dtype=dt_t, device=dev)

        out_spikes_list: list[Any] = []

        for s in range(cfg.window_steps):
            ctx = StepContext(dt=cfg.dt, step=s, t=s * cfg.dt)

            # ── Input population ────────────────────────────────
            inp_result = inp_pop.model.step(
                inp_pop.state,
                NeuronInputs(drive={Compartment.SOMA: drives}),
                ctx,
            )
            in_spikes = inp_result.spikes
            in_counts = in_counts + in_spikes.detach()

            # ── Input → Hidden (Dale's Law) ─────────────────────
            h_current = (in_spikes @ w_ih_eff) + b_ih
            hid_result = hid_pop.model.step(
                hid_pop.state,
                NeuronInputs(drive={Compartment.SOMA: h_current}),
                ctx,
            )
            hid_spikes = hid_result.spikes
            hid_counts = hid_counts + hid_spikes.detach()

            # ── Hidden → Output (Dale's Law) ────────────────────
            o_current: Any = (
                (hid_spikes * w_ho_eff).sum() + b_ho[0]
            ).unsqueeze(0)

            # ── Output population ───────────────────────────────
            out_result = out_pop.model.step(
                out_pop.state,
                NeuronInputs(drive={Compartment.SOMA: o_current}),
                ctx,
            )
            out_spikes_list.append(out_result.spikes)

        spike_tensor = torch_mod.stack(out_spikes_list, dim=0)  # [T, 1]
        return spike_tensor, in_counts, hid_counts

    # ── Weight collection ───────────────────────────────────────────

    def _collect_weights(self, gn: Any) -> dict[str, Any]:
        """Collect final weights in backward-compatible format."""
        return {
            "layer_1": gn.trainables["raw_w_in_to_hidden"].clone().detach(),
            "layer_2": gn.trainables["raw_w_hidden_to_out"].clone().detach(),
            "bias_o": gn.trainables["bias_hidden_to_out"].clone().detach(),
            "bias_h": gn.trainables["bias_in_to_hidden"].clone().detach(),
        }

    # ── Main training loop ──────────────────────────────────────────

    def run(self) -> LogicGateResult:  # noqa: C901, PLR0912, PLR0915
        """Execute the training loop. Returns result with convergence status."""
        from neuroforge.core.torch_utils import require_torch, resolve_device_dtype
        from neuroforge.encoding.rate import RateEncoderParams
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.network.gate_builder import build_gate_network
        from neuroforge.network.specs import GateNetworkSpec

        torch = require_torch()
        cfg = self.config
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # ── Build network via factory ───────────────────────────────
        spec = GateNetworkSpec(
            input_size=cfg.image_h * cfg.image_w,
            hidden_size=cfg.n_hidden,
            output_size=1,
            n_inhibitory_hidden=cfg.n_inhibitory,
            seed=cfg.seed,
            device=cfg.device,
            dtype=cfg.dtype,
            init_scale=0.3,
            neuron_model="lif_surr",
            synapse_model="static_dales",
        )
        gn = build_gate_network(spec)

        # ── Encoder / readout / loss from factories ─────────────────
        encoder = cast("RateEncoder", DEFAULT_HUB.encoders.create(
            "rate", params=RateEncoderParams(amplitude=cfg.amplitude),
        ))
        readout = cast("IReadout", DEFAULT_HUB.readouts.create(
            "spike_count", threshold=float(cfg.spike_threshold),
        ))
        loss_fn = cast("ILoss", DEFAULT_HUB.losses.create(cfg.loss_fn))

        # ── Optimizer ───────────────────────────────────────────────
        params_list = list(gn.trainables.values())
        optimizer = torch.optim.Adam(params_list, lr=cfg.lr)

        dev, dt = resolve_device_dtype(cfg.device, cfg.dtype)

        # ── Emit run_start + topology ───────────────────────────────
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
             "dt": cfg.dt,
             "window_steps": cfg.window_steps,
             "per_pattern_streak": cfg.per_pattern_streak},
        )

        n_in = cfg.image_h * cfg.image_w
        dtype_name = str(gn.trainables["raw_w_in_to_hidden"].dtype).replace("torch.", "")
        topo_data: dict[str, Any] = {
            "layers": [f"input({n_in})", f"hidden({cfg.n_hidden})",
                       "output(1)"],
            "edges": [
                {"name": "input_hidden", "src": "input", "dst": "hidden",
                 "n_pre": int(n_in), "n_post": int(cfg.n_hidden),
                 "n_edges": int(n_in * cfg.n_hidden),
                 "dense": True, "dtype": dtype_name, "topology_type": "dense",
                 "weights": gn.trainables["raw_w_in_to_hidden"].detach()},
                {"name": "hidden_output", "src": "hidden", "dst": "output",
                 "n_pre": int(cfg.n_hidden), "n_post": 1,
                 "n_edges": int(cfg.n_hidden),
                 "dense": True, "dtype": dtype_name, "topology_type": "dense",
                 "weights": gn.trainables["raw_w_hidden_to_out"].detach()},
            ],
            "projection_meta": [
                {
                    "name": "input_hidden",
                    "src": "input",
                    "dst": "hidden",
                    "n_pre": int(n_in),
                    "n_post": int(cfg.n_hidden),
                    "n_edges": int(n_in * cfg.n_hidden),
                    "dense": True,
                    "dtype": dtype_name,
                    "topology_type": "dense",
                },
                {
                    "name": "hidden_output",
                    "src": "hidden",
                    "dst": "output",
                    "n_pre": int(cfg.n_hidden),
                    "n_post": 1,
                    "n_edges": int(cfg.n_hidden),
                    "dense": True,
                    "dtype": dtype_name,
                    "topology_type": "dense",
                },
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
            trial_t0 = perf_counter()
            # Check for external cancellation.
            if self._stop_check is not None and self._stop_check():
                self._emit(
                    "training_end", trial, cfg.gate,
                    {"converged": False, "trials": trial, "stopped": True},
                )
                return LogicGateResult(
                    gate=cfg.gate,
                    converged=False,
                    trials=trial,
                    final_weights=self._collect_weights(gn),
                    accuracy_history=accuracy_history,
                )

            inp = random.choice(patterns)
            expected = self._truth_table[inp]

            # ── Pixel encode ────────────────────────────────────────
            img = _pattern_to_image(inp, cfg.image_h, cfg.image_w, torch)
            flat = img.reshape(-1).to(device=dev, dtype=dt)
            drives = encoder.encode(flat)

            # ── Zero gradients ──────────────────────────────────────
            optimizer.zero_grad()

            # ── Forward pass ────────────────────────────────────────
            spike_tensor, in_counts, hid_counts = self._forward(
                drives, gn, cfg, torch,
            )

            # ── Readout + decode ────────────────────────────────────
            ro_result = readout(spike_tensor)
            out_count = int(ro_result.count.item())
            predicted = 1 if out_count >= cfg.spike_threshold else 0
            correct = predicted == expected
            error = float(expected - predicted)

            # ── Weight update (only on error) ───────────────────────
            if cfg.loss_fn == "bce_logits":
                target_t = torch.tensor(
                    [float(expected)], dtype=dt, device=dev,
                )
                loss = loss_fn(ro_result.logits, target_t)
            else:
                target_count = float(expected) * cfg.spike_threshold
                target_t = torch.tensor(
                    [target_count], dtype=dt, device=dev,
                )
                loss = loss_fn(ro_result.count, target_t)

            emit_stats = (
                cfg.emit_param_grad_stats
                and (trial % cfg.stats_every_n_trials == 0)
            )
            param_grad_payload: dict[str, float] = {}

            if error != 0.0:
                loss.backward()
                if emit_stats:
                    param_grad_payload = _param_grad_stats(
                        gn.trainables["raw_w_in_to_hidden"],
                        gn.trainables["raw_w_hidden_to_out"],
                    )
                # Gradient clipping.
                for _p in params_list:
                    if _p.grad is not None:
                        _p.grad.clamp_(-1.0, 1.0)
                optimizer.step()
                # Weight clamping.
                with torch.no_grad():
                    for _p in params_list:
                        _p.clamp_(cfg.w_min, cfg.w_max)
            elif emit_stats:
                param_grad_payload = _param_grad_stats(
                    gn.trainables["raw_w_in_to_hidden"],
                    gn.trainables["raw_w_hidden_to_out"],
                )

            # ── Convergence tracking ────────────────────────────────
            recent_correct.append(correct)
            if len(recent_correct) > cfg.convergence_streak:
                recent_correct.pop(0)
            accuracy = sum(recent_correct) / len(recent_correct)
            accuracy_history.append(accuracy)

            # ── Emit per-trial events ───────────────────────────────
            spike_data: dict[str, Any] = {
                "input": list(inp),
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "error": error,
                "accuracy": accuracy,
                "epoch": 0,
                "out_spike_count": out_count,
                "input_spikes": in_counts.detach().cpu().tolist(),
                "hidden_spikes": hid_counts.detach().cpu().tolist(),
                "output_spikes": [out_count],
            }
            self._emit("training_trial", trial, cfg.gate, spike_data)

            self._emit(
                "scalar", trial, cfg.gate,
                {
                    "trial": trial,
                    "epoch": 0,
                    "gate": cfg.gate,
                    "accuracy": accuracy,
                    "error": error,
                    "loss": float(loss.detach().item()),
                    "wall_ms": (perf_counter() - trial_t0) * 1_000.0,
                    "correct": correct,
                    **param_grad_payload,
                },
            )

            self._emit(
                "weight", trial, f"{cfg.gate}/input-hidden",
                {"weights": gn.trainables[
                    "raw_w_in_to_hidden"
                ].detach()},
            )
            self._emit(
                "weight", trial, f"{cfg.gate}/hidden-output",
                {"weights": gn.trainables[
                    "raw_w_hidden_to_out"
                ].detach()},
            )

            # ── Streak / convergence check ──────────────────────────
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
                self._emit(
                    "training_end", trial + 1, cfg.gate,
                    {"converged": True, "trials": trial + 1},
                )
                return LogicGateResult(
                    gate=cfg.gate,
                    converged=True,
                    trials=trial + 1,
                    final_weights=self._collect_weights(gn),
                    accuracy_history=accuracy_history,
                )

        self._emit(
            "training_end", cfg.max_trials, cfg.gate,
            {"converged": False, "trials": cfg.max_trials},
        )
        return LogicGateResult(
            gate=cfg.gate,
            converged=False,
            trials=cfg.max_trials,
            final_weights=self._collect_weights(gn),
            accuracy_history=accuracy_history,
        )
