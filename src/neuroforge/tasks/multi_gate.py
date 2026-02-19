"""Multi-gate task: train one shared brain on all six logic gates.

This task uses pixel-only inputs with gate-context pixels:
- 8x8 image by default
- data bit pixels at (2, 2) and (2, 5)
- gate one-hot pixels at top row columns 0..5

Network and training are factory-driven:
- CoreEngine built through NetworkFactory
- input population: lif
- hidden/output populations: lif_surr
- synapses: static_dales
- encoder/readout/loss from DEFAULT_HUB factories
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.contracts.neurons import NeuronInputs, StepContext
from neuroforge.contracts.synapses import SynapseInputs, SynapseTopology
from neuroforge.contracts.types import Compartment
from neuroforge.core.torch_utils import require_torch, resolve_device_dtype, smart_device
from neuroforge.encoding.rate import RateEncoderParams
from neuroforge.factories.hub import DEFAULT_HUB
from neuroforge.learning.stats import grad_stats, tensor_stats
from neuroforge.network.factory import NetworkFactory, to_topology_json
from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec
from neuroforge.tasks.logic_gates import GATE_TABLES

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import Generator, Tensor
    from torch.optim import Optimizer

    from neuroforge.contracts.encoding import ILoss, IReadout
    from neuroforge.contracts.monitors import IEventBus
    from neuroforge.encoding.rate import RateEncoder
    from neuroforge.engine.core_engine import CoreEngine, Projection

__all__ = [
    "ALL_GATES",
    "GATE_INDEX",
    "MultiGateConfig",
    "MultiGateResult",
    "MultiGateTask",
    "build_gate_network",
]

Pattern = tuple[int, int]
Combo = tuple[str, Pattern]
RunStatus = Literal["converged", "stopped", "max_epochs"]

ALL_GATES: tuple[str, ...] = ("AND", "OR", "NAND", "NOR", "XOR", "XNOR")
GATE_INDEX: dict[str, int] = {gate: idx for idx, gate in enumerate(ALL_GATES)}
PATTERNS: tuple[Pattern, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
MIN_IMAGE_H = 3
MIN_IMAGE_W = 6


@dataclass(slots=True)
class _Trainables:
    w_ih: Tensor
    b_h: Tensor
    w_ho: Tensor
    b_o: Tensor


@dataclass(slots=True)
class _ForwardSignals:
    spikes: Tensor
    input_counts: Tensor
    hidden_counts: Tensor


@dataclass(slots=True)
class _Runtime:
    cfg: MultiGateConfig
    dev: torch.device
    tdt: torch.dtype
    engine: CoreEngine
    trainables: _Trainables
    params: tuple[Tensor, Tensor, Tensor, Tensor]
    optimizer: Optimizer
    encoder: RateEncoder
    readout: IReadout
    loss_fn: ILoss
    gates: list[str]
    all_combos: list[Combo]
    shuffle_gen: Generator


@dataclass(slots=True)
class _LoopState:
    trial: int
    epoch: int
    streak: int
    recent_correct: list[bool]
    accuracy_history: list[float]
    combo_streaks: dict[Combo, int]


@dataclass(slots=True)
class _TrialOutcome:
    gate: str
    pattern: Pattern
    expected: int
    predicted: int
    correct: bool
    error: float
    accuracy: float
    loss: float
    out_count: int
    input_counts: Tensor
    hidden_counts: Tensor
    wall_ms: float
    param_grad_stats: dict[str, float]


def _new_per_gate_converged() -> dict[str, bool]:
    return {}


def _new_final_weights() -> dict[str, object]:
    return {}


def _new_accuracy_history() -> list[float]:
    return []


def _tensor_to_float_list(values: Tensor) -> list[float]:
    """Convert a tensor to a flat Python float list."""
    flat = values.detach().reshape(-1)
    return [float(flat[idx].item()) for idx in range(flat.numel())]


@dataclass(frozen=True, slots=True)
class MultiGateConfig:
    """Configuration for the multi-gate task."""

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
    lr: float = 1e-2
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
        """Validate configuration constraints and resolve auto device."""
        unknown = [gate for gate in self.gates if gate not in GATE_INDEX]
        if unknown:
            msg = f"Unsupported gates: {unknown}"
            raise ValueError(msg)
        if self.n_inhibitory < 0 or self.n_inhibitory > self.n_hidden:
            msg = "n_inhibitory must be in [0, n_hidden]"
            raise ValueError(msg)
        if self.image_h < MIN_IMAGE_H:
            msg = f"image_h must be >= {MIN_IMAGE_H}"
            raise ValueError(msg)
        if self.image_w < MIN_IMAGE_W:
            msg = f"image_w must be >= {MIN_IMAGE_W}"
            raise ValueError(msg)
        if self.image_w < len(ALL_GATES):
            msg = f"image_w must be >= {len(ALL_GATES)} for gate-context pixels"
            raise ValueError(msg)
        if self.stats_every_n_trials <= 0:
            msg = "stats_every_n_trials must be > 0"
            raise ValueError(msg)
        if self.device == "auto":
            n_total = self.image_h * self.image_w + self.n_hidden + 1
            object.__setattr__(self, "device", smart_device(n_total))


@dataclass
class MultiGateResult:
    """Result of a multi-gate training run."""

    gates: tuple[str, ...]
    converged: bool
    trials: int
    epochs: int = 0
    per_gate_converged: dict[str, bool] = field(default_factory=_new_per_gate_converged)
    final_weights: dict[str, object] = field(default_factory=_new_final_weights)
    accuracy_history: list[float] = field(default_factory=_new_accuracy_history)


def _build_hidden_signs(
    n_hidden: int,
    n_inhibitory: int,
    *,
    dev: torch.device,
    tdt: torch.dtype,
) -> Tensor:
    """Build hidden pre-synaptic sign mask (+1 excit, -1 inhib)."""
    torch = require_torch()
    sign_hidden = torch.ones(n_hidden, device=dev, dtype=tdt)
    n_exc = max(0, n_hidden - n_inhibitory)
    sign_hidden[n_exc:] = -1.0
    return cast("Tensor", sign_hidden)


def _pattern_gate_to_image(
    bits: Pattern,
    gate_idx: int,
    n_gates: int,
    *,
    h: int,
    w: int,
) -> Tensor:
    """Convert (pattern, gate) to an [H, W] pixel image."""
    torch = require_torch()
    img = torch.zeros(h, w)
    if bits[0]:
        img[2, 2] = 1.0
    if bits[1]:
        img[2, 5] = 1.0
    if 0 <= gate_idx < n_gates:
        img[0, gate_idx] = 1.0
    return cast("Tensor", img)


def build_gate_network(
    cfg: MultiGateConfig,
    *,
    dev: torch.device,
    tdt: torch.dtype,
) -> tuple[CoreEngine, _Trainables]:
    """Build the shared multi-gate CoreEngine via NetworkFactory."""
    torch = require_torch()

    input_size = cfg.image_h * cfg.image_w
    hidden_size = cfg.n_hidden

    sign_input = torch.ones(input_size, device=dev, dtype=tdt)
    sign_hidden = _build_hidden_signs(
        hidden_size,
        cfg.n_inhibitory,
        dev=dev,
        tdt=tdt,
    )

    spec = NetworkSpec(
        populations=[
            PopulationSpec("input", input_size, "lif"),
            PopulationSpec("hidden", hidden_size, "lif_surr"),
            PopulationSpec("output", 1, "lif_surr"),
        ],
        projections=[
            ProjectionSpec(
                "input_hidden",
                "input",
                "hidden",
                "static_dales",
                synapse_params={"sign_pre": sign_input},
                topology={
                    "type": "dense",
                    "init": "uniform",
                    "low": -0.3,
                    "high": 0.3,
                    "bias": True,
                },
            ),
            ProjectionSpec(
                "hidden_output",
                "hidden",
                "output",
                "static_dales",
                synapse_params={"sign_pre": sign_hidden},
                topology={
                    "type": "dense",
                    "init": "uniform",
                    "low": -0.3,
                    "high": 0.3,
                    "bias": True,
                },
            ),
        ],
        metadata={"task": "multi_gate", "gates": list(cfg.gates)},
    )

    factory = NetworkFactory(DEFAULT_HUB.neurons, DEFAULT_HUB.synapses)
    engine = factory.build(spec, device=cfg.device, dtype=cfg.dtype, seed=cfg.seed)

    proj_ih = engine.projections["input_hidden"]
    proj_ho = engine.projections["hidden_output"]
    trainables = _Trainables(
        w_ih=proj_ih.state["weight_matrix"],
        b_h=proj_ih.state["bias"],
        w_ho=proj_ho.state["weight_matrix"],
        b_o=proj_ho.state["bias"],
    )
    return engine, trainables


def _rebind_projection_weights(proj: Projection, weight_matrix: Tensor) -> None:
    """Rebind projection topology weights to a live view of raw weights."""
    topo = proj.topology
    proj.topology = SynapseTopology(
        pre_idx=topo.pre_idx,
        post_idx=topo.post_idx,
        weights=weight_matrix.reshape(-1),
        delays=topo.delays,
        n_pre=topo.n_pre,
        n_post=topo.n_post,
    )


class MultiGateTask:
    """Train one shared network on all configured logic gates."""

    def __init__(
        self,
        config: MultiGateConfig | None = None,
        event_bus: IEventBus | None = None,
        *,
        stop_check: Callable[[], bool] | None = None,
    ) -> None:
        """Create a multi-gate trainer with optional event bus/stop hook."""
        self.config = config or MultiGateConfig()
        self._bus = event_bus
        self._stop_check = stop_check

    @staticmethod
    def _pattern_gate_to_image(
        bits: Pattern,
        gate_idx: int,
        n_gates: int,
        *,
        h: int = 8,
        w: int = 8,
    ) -> Tensor:
        """Backward-compatible wrapper for image encoding helper."""
        return _pattern_gate_to_image(bits, gate_idx, n_gates, h=h, w=w)

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
        self._bus.publish(
            MonitorEvent(
                topic=EventTopic(topic),
                step=step,
                t=t,
                source=source,
                data=data,
            ),
        )

    def _prepare_runtime(self) -> _Runtime:
        cfg = self.config
        torch = require_torch()

        torch.manual_seed(cfg.seed)
        dev, tdt = resolve_device_dtype(cfg.device, cfg.dtype)

        engine, trainables = build_gate_network(cfg, dev=dev, tdt=tdt)
        params = (trainables.w_ih, trainables.b_h, trainables.w_ho, trainables.b_o)
        for param in params:
            param.requires_grad = True

        _rebind_projection_weights(engine.projections["input_hidden"], trainables.w_ih)
        _rebind_projection_weights(engine.projections["hidden_output"], trainables.w_ho)

        optimizer = torch.optim.Adam([*params], lr=cfg.lr)
        encoder = cast(
            "RateEncoder",
            DEFAULT_HUB.encoders.create(
                "rate",
                params=RateEncoderParams(amplitude=cfg.amplitude),
            ),
        )
        readout = cast(
            "IReadout",
            DEFAULT_HUB.readouts.create(
                "spike_count",
                threshold=float(cfg.spike_threshold),
            ),
        )
        loss_fn = cast("ILoss", DEFAULT_HUB.losses.create(cfg.loss_fn))

        gates = list(cfg.gates)
        all_combos = [(gate, pattern) for gate in gates for pattern in PATTERNS]
        shuffle_gen = torch.Generator(device="cpu")
        shuffle_gen.manual_seed(cfg.seed)

        return _Runtime(
            cfg=cfg,
            dev=dev,
            tdt=tdt,
            engine=engine,
            trainables=trainables,
            params=params,
            optimizer=optimizer,
            encoder=encoder,
            readout=readout,
            loss_fn=loss_fn,
            gates=gates,
            all_combos=all_combos,
            shuffle_gen=shuffle_gen,
        )

    def _emit_training_start(self, runtime: _Runtime) -> None:
        cfg = runtime.cfg
        self._emit(
            "run_start",
            0,
            "MULTI",
            {
                "task": "multi_gate",
                "device": cfg.device,
                "seed": cfg.seed,
                "dtype": cfg.dtype,
            },
        )
        self._emit(
            "training_start",
            0,
            "MULTI",
            {
                "gate": "MULTI",
                "gates": list(runtime.gates),
                "max_epochs": cfg.max_epochs,
                "dt": cfg.dt,
                "window_steps": cfg.window_steps,
                "per_pattern_streak": cfg.per_pattern_streak,
                "n_input": cfg.image_h * cfg.image_w,
                "n_hidden": cfg.n_hidden,
                "image_h": cfg.image_h,
                "image_w": cfg.image_w,
            },
        )
        self._emit("topology", 0, "MULTI", to_topology_json(runtime.engine))

    @staticmethod
    def _param_grad_stats(runtime: _Runtime) -> dict[str, float]:
        w_ih_stats = tensor_stats(runtime.trainables.w_ih, "w_ih")
        w_ho_stats = tensor_stats(runtime.trainables.w_ho, "w_ho")
        g_ih_stats = grad_stats(runtime.trainables.w_ih, "ih")
        g_ho_stats = grad_stats(runtime.trainables.w_ho, "ho")
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

    def _shuffled_combos(self, runtime: _Runtime) -> list[Combo]:
        torch = require_torch()
        order = torch.randperm(
            len(runtime.all_combos),
            generator=runtime.shuffle_gen,
        ).tolist()
        return [runtime.all_combos[idx] for idx in order]

    def _forward(self, drives: Tensor, runtime: _Runtime) -> _ForwardSignals:
        """Run one presentation window and return output spikes + counts."""
        cfg = runtime.cfg
        torch = require_torch()

        pops = runtime.engine.populations
        projs = runtime.engine.projections

        runtime.engine.reset()
        for pop in pops.values():
            for key in list(pop.state):
                pop.state[key] = pop.state[key].detach()

        inp_pop = pops["input"]
        hid_pop = pops["hidden"]
        out_pop = pops["output"]
        proj_ih = projs["input_hidden"]
        proj_ho = projs["hidden_output"]

        in_counts = torch.zeros(inp_pop.n, dtype=drives.dtype, device=drives.device)
        hid_counts = torch.zeros(hid_pop.n, dtype=drives.dtype, device=drives.device)
        out_spikes_list: list[Tensor] = []

        zero_h_post = torch.zeros(hid_pop.n, dtype=drives.dtype, device=drives.device)
        zero_o_post = torch.zeros(out_pop.n, dtype=drives.dtype, device=drives.device)

        for step in range(cfg.window_steps):
            ctx = StepContext(dt=cfg.dt, step=step, t=step * cfg.dt)

            inp_result = inp_pop.model.step(
                inp_pop.state,
                NeuronInputs(drive={Compartment.SOMA: drives}),
                ctx,
            )
            in_spikes = inp_result.spikes
            in_counts = in_counts + in_spikes.detach().to(drives.dtype)

            ih_syn = proj_ih.model.step(
                proj_ih.state,
                proj_ih.topology,
                SynapseInputs(pre_spikes=in_spikes, post_spikes=zero_h_post),
                ctx,
            )
            h_current = ih_syn.post_current[Compartment.SOMA] + runtime.trainables.b_h

            hid_result = hid_pop.model.step(
                hid_pop.state,
                NeuronInputs(drive={Compartment.SOMA: h_current}),
                ctx,
            )
            hid_spikes = hid_result.spikes
            hid_counts = hid_counts + hid_spikes.detach().to(drives.dtype)

            ho_syn = proj_ho.model.step(
                proj_ho.state,
                proj_ho.topology,
                SynapseInputs(pre_spikes=hid_spikes, post_spikes=zero_o_post),
                ctx,
            )
            o_current = ho_syn.post_current[Compartment.SOMA] + runtime.trainables.b_o

            out_result = out_pop.model.step(
                out_pop.state,
                NeuronInputs(drive={Compartment.SOMA: o_current}),
                ctx,
            )
            out_spikes_list.append(out_result.spikes)

        spike_tensor = torch.stack(out_spikes_list, dim=0)
        return _ForwardSignals(
            spikes=cast("Tensor", spike_tensor),
            input_counts=cast("Tensor", in_counts),
            hidden_counts=cast("Tensor", hid_counts),
        )

    def _run_trial(self, runtime: _Runtime, combo: Combo, state: _LoopState) -> _TrialOutcome:
        cfg = runtime.cfg
        torch = require_torch()
        trial_t0 = perf_counter()

        gate, inp = combo
        gate_idx = GATE_INDEX[gate]
        expected = GATE_TABLES[gate][inp]

        img = _pattern_gate_to_image(
            inp,
            gate_idx,
            len(ALL_GATES),
            h=cfg.image_h,
            w=cfg.image_w,
        )
        flat = img.reshape(-1).to(device=runtime.dev, dtype=runtime.tdt)
        drives = runtime.encoder.encode(flat)

        runtime.optimizer.zero_grad()
        signals = self._forward(drives, runtime)

        ro_result = runtime.readout(signals.spikes)
        out_count = int(ro_result.count.detach().item())
        predicted = 1 if out_count >= cfg.spike_threshold else 0
        correct = predicted == expected

        target_count = float(expected) * cfg.spike_threshold
        target_t = torch.tensor([target_count], dtype=runtime.tdt, device=runtime.dev)
        loss = runtime.loss_fn(ro_result.count, target_t)
        emit_stats = (
            cfg.emit_param_grad_stats
            and (state.trial % cfg.stats_every_n_trials == 0)
        )
        param_grad = {}
        if not correct:
            loss.backward()
            if emit_stats:
                param_grad = self._param_grad_stats(runtime)
            runtime.optimizer.step()

            with torch.no_grad():
                for param in runtime.params:
                    param.clamp_(cfg.w_min, cfg.w_max)
        elif emit_stats:
            param_grad = self._param_grad_stats(runtime)

        state.recent_correct.append(correct)
        if len(state.recent_correct) > cfg.convergence_streak:
            state.recent_correct.pop(0)
        accuracy = sum(state.recent_correct) / len(state.recent_correct)
        state.accuracy_history.append(accuracy)

        if correct:
            state.streak += 1
            state.combo_streaks[combo] += 1
        else:
            state.streak = 0
            state.combo_streaks[combo] = 0

        return _TrialOutcome(
            gate=gate,
            pattern=inp,
            expected=expected,
            predicted=predicted,
            correct=correct,
            error=float(expected - predicted),
            accuracy=accuracy,
            loss=float(loss.detach().item()),
            out_count=out_count,
            input_counts=signals.input_counts.detach(),
            hidden_counts=signals.hidden_counts.detach(),
            wall_ms=(perf_counter() - trial_t0) * 1_000.0,
            param_grad_stats=param_grad,
        )

    def _emit_trial_events(
        self,
        runtime: _Runtime,
        state: _LoopState,
        outcome: _TrialOutcome,
    ) -> None:
        input_spikes = _tensor_to_float_list(outcome.input_counts)
        hidden_spikes = _tensor_to_float_list(outcome.hidden_counts)
        self._emit(
            "training_trial",
            state.trial,
            f"MULTI/{outcome.gate}",
            {
                "gate": outcome.gate,
                "input": list(outcome.pattern),
                "expected": outcome.expected,
                "predicted": outcome.predicted,
                "correct": outcome.correct,
                "error": outcome.error,
                "accuracy": outcome.accuracy,
                "epoch": state.epoch,
                "out_spike_count": outcome.out_count,
                "input_spikes": input_spikes,
                "hidden_spikes": hidden_spikes,
                "output_spikes": [outcome.out_count],
            },
        )

        scalar_payload: dict[str, Any] = {
            "trial": state.trial,
            "epoch": state.epoch,
            "gate": outcome.gate,
            "accuracy": outcome.accuracy,
            "error": outcome.error,
            "loss": outcome.loss,
            "wall_ms": outcome.wall_ms,
            "correct": outcome.correct,
        }
        scalar_payload.update(outcome.param_grad_stats)
        self._emit("scalar", state.trial, "MULTI", scalar_payload)

        if state.trial % 10 != 0:
            return

        torch = require_torch()
        self._emit(
            "weight",
            state.trial,
            "MULTI/input-hidden",
            {"weights": runtime.trainables.w_ih.detach()},
        )
        self._emit(
            "weight",
            state.trial,
            "MULTI/hidden-output",
            {"weights": runtime.trainables.w_ho.detach()},
        )
        self._emit(
            "spike",
            state.trial,
            "MULTI/input",
            {"spikes": (outcome.input_counts > 0)},
        )
        self._emit(
            "spike",
            state.trial,
            "MULTI/hidden",
            {"spikes": (outcome.hidden_counts > 0)},
        )
        self._emit(
            "spike",
            state.trial,
            "MULTI/output",
            {"spikes": torch.tensor([outcome.out_count > 0], device=runtime.dev)},
        )

    def _all_combos_reliable(self, runtime: _Runtime, state: _LoopState) -> bool:
        return all(
            streak >= runtime.cfg.per_pattern_streak for streak in state.combo_streaks.values()
        )

    def _build_result(
        self,
        runtime: _Runtime,
        state: _LoopState,
        *,
        status: RunStatus,
        epochs: int,
    ) -> MultiGateResult:
        converged = status == "converged"
        stopped = status == "stopped"

        per_gate: dict[str, bool] = {}
        for gate in runtime.gates:
            per_gate[gate] = all(
                state.combo_streaks[(gate, pattern)] >= runtime.cfg.per_pattern_streak
                for pattern in PATTERNS
            )

        end_data: dict[str, Any] = {
            "converged": converged,
            "trials": state.trial,
            "epochs": epochs,
            "per_gate_converged": per_gate,
        }
        if stopped:
            end_data["stopped"] = True
        self._emit("training_end", state.trial, "MULTI", end_data)

        return MultiGateResult(
            gates=tuple(runtime.gates),
            converged=converged,
            trials=state.trial,
            epochs=epochs,
            per_gate_converged=per_gate,
            final_weights={
                "w_ih": runtime.trainables.w_ih.clone().detach(),
                "w_ho": runtime.trainables.w_ho.clone().detach(),
                "b_h": runtime.trainables.b_h.clone().detach(),
                "b_o": runtime.trainables.b_o.clone().detach(),
            },
            accuracy_history=[*state.accuracy_history],
        )

    def run(self) -> MultiGateResult:
        """Train all gates with shared weights and return result."""
        runtime = self._prepare_runtime()
        self._emit_training_start(runtime)

        state = _LoopState(
            trial=0,
            epoch=0,
            streak=0,
            recent_correct=[],
            accuracy_history=[],
            combo_streaks=dict.fromkeys(runtime.all_combos, 0),
        )

        for epoch in range(runtime.cfg.max_epochs):
            state.epoch = epoch
            for combo in self._shuffled_combos(runtime):
                if self._stop_check is not None and self._stop_check():
                    return self._build_result(
                        runtime,
                        state,
                        status="stopped",
                        epochs=epoch,
                    )

                outcome = self._run_trial(runtime, combo, state)
                self._emit_trial_events(runtime, state, outcome)
                state.trial += 1

            if (
                state.streak >= runtime.cfg.convergence_streak
                and self._all_combos_reliable(runtime, state)
            ):
                return self._build_result(
                    runtime,
                    state,
                    status="converged",
                    epochs=epoch + 1,
                )

        return self._build_result(
            runtime,
            state,
            status="max_epochs",
            epochs=runtime.cfg.max_epochs,
        )
