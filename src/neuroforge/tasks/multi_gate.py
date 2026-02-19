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

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from neuroforge.core.torch_utils import smart_device
from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec
from neuroforge.tasks.logic_gates import GATE_TABLES

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.encoding import ILoss, IReadout
    from neuroforge.contracts.monitors import IEventBus
    from neuroforge.encoding.rate import RateEncoder
    from neuroforge.engine.core_engine import CoreEngine

__all__ = [
    "MultiGateTask",
    "MultiGateConfig",
    "MultiGateResult",
    "ALL_GATES",
    "GATE_INDEX",
    "build_gate_network",
]

ALL_GATES: tuple[str, ...] = ("AND", "OR", "NAND", "NOR", "XOR", "XNOR")
GATE_INDEX: dict[str, int] = {g: i for i, g in enumerate(ALL_GATES)}


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

    def __post_init__(self) -> None:
        unknown = [g for g in self.gates if g not in GATE_INDEX]
        if unknown:
            msg = f"Unsupported gates: {unknown}"
            raise ValueError(msg)
        if self.n_inhibitory < 0 or self.n_inhibitory > self.n_hidden:
            msg = "n_inhibitory must be in [0, n_hidden]"
            raise ValueError(msg)
        if self.image_h < 3:
            msg = "image_h must be >= 3"
            raise ValueError(msg)
        if self.image_w < 6:
            msg = "image_w must be >= 6"
            raise ValueError(msg)
        if self.image_w < len(ALL_GATES):
            msg = f"image_w must be >= {len(ALL_GATES)} for gate-context pixels"
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
    per_gate_converged: dict[str, bool] = field(default_factory=lambda: {})
    final_weights: dict[str, Any] = field(default_factory=lambda: {})
    accuracy_history: list[float] = field(default_factory=lambda: [])


def _build_hidden_signs(n_hidden: int, n_inhibitory: int, torch_mod: Any, dev: Any, tdt: Any) -> Any:
    """Build hidden pre-synaptic sign mask (+1 excit, -1 inhib)."""
    sign_hidden = torch_mod.ones(n_hidden, device=dev, dtype=tdt)
    n_exc = max(0, n_hidden - n_inhibitory)
    sign_hidden[n_exc:] = -1.0
    return sign_hidden


def _pattern_gate_to_image(
    bits: tuple[int, int],
    gate_idx: int,
    n_gates: int,
    h: int,
    w: int,
    torch_mod: Any,
) -> Any:
    """Convert (pattern, gate) to an [H, W] pixel image."""
    img = torch_mod.zeros(h, w)
    if bits[0]:
        img[2, 2] = 1.0
    if bits[1]:
        img[2, 5] = 1.0
    if 0 <= gate_idx < n_gates:
        img[0, gate_idx] = 1.0
    return img


def build_gate_network(
    cfg: MultiGateConfig,
    *,
    torch_mod: Any,
    dev: Any,
    tdt: Any,
) -> tuple[CoreEngine, dict[str, Any]]:
    """Build the shared multi-gate CoreEngine via NetworkFactory."""
    from neuroforge.factories.hub import DEFAULT_HUB
    from neuroforge.network.factory import NetworkFactory

    input_size = cfg.image_h * cfg.image_w
    hidden_size = cfg.n_hidden

    sign_input = torch_mod.ones(input_size, device=dev, dtype=tdt)
    sign_hidden = _build_hidden_signs(
        hidden_size,
        cfg.n_inhibitory,
        torch_mod,
        dev,
        tdt,
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
    trainables = {
        "w_ih": proj_ih.state["weight_matrix"],
        "b_h": proj_ih.state["bias"],
        "w_ho": proj_ho.state["weight_matrix"],
        "b_o": proj_ho.state["bias"],
    }
    return engine, trainables


def _rebind_projection_weights(proj: Any, weight_matrix: Any) -> None:
    """Rebind projection topology weights to a live view of raw weights."""
    from neuroforge.contracts.synapses import SynapseTopology

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
        self.config = config or MultiGateConfig()
        self._bus = event_bus
        self._stop_check = stop_check

    @staticmethod
    def _pattern_gate_to_image(
        bits: tuple[int, int],
        gate_idx: int,
        n_gates: int,
        h: int = 8,
        w: int = 8,
        torch_mod: Any = None,
    ) -> Any:
        """Backward-compatible wrapper for image encoding helper."""
        if torch_mod is None:
            import torch as torch_mod

        return _pattern_gate_to_image(bits, gate_idx, n_gates, h, w, torch_mod)

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

    def _forward(
        self,
        drives: Any,
        engine: CoreEngine,
        cfg: MultiGateConfig,
        b_h: Any,
        b_o: Any,
        torch_mod: Any,
    ) -> tuple[Any, Any, Any]:
        """Run one presentation window and return output spikes + counts."""
        from neuroforge.contracts.neurons import NeuronInputs, StepContext
        from neuroforge.contracts.synapses import SynapseInputs
        from neuroforge.contracts.types import Compartment

        pops = engine._populations  # pyright: ignore[reportPrivateUsage]
        projs = engine._projections  # pyright: ignore[reportPrivateUsage]

        engine.reset()
        for pop in pops.values():
            for k in list(pop.state):
                pop.state[k] = pop.state[k].detach()

        inp_pop = pops["input"]
        hid_pop = pops["hidden"]
        out_pop = pops["output"]
        proj_ih = projs["input_hidden"]
        proj_ho = projs["hidden_output"]

        in_counts = torch_mod.zeros(inp_pop.n, dtype=drives.dtype, device=drives.device)
        hid_counts = torch_mod.zeros(hid_pop.n, dtype=drives.dtype, device=drives.device)
        out_spikes_list: list[Any] = []
        zero_h_post = torch_mod.zeros(hid_pop.n, dtype=drives.dtype, device=drives.device)
        zero_o_post = torch_mod.zeros(out_pop.n, dtype=drives.dtype, device=drives.device)

        for _s in range(cfg.window_steps):
            ctx = StepContext(dt=cfg.dt, step=_s, t=_s * cfg.dt)

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
                SynapseInputs(
                    pre_spikes=in_spikes,
                    post_spikes=zero_h_post,
                ),
                ctx,
            )
            h_current = ih_syn.post_current[Compartment.SOMA] + b_h

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
                SynapseInputs(
                    pre_spikes=hid_spikes,
                    post_spikes=zero_o_post,
                ),
                ctx,
            )
            o_current = ho_syn.post_current[Compartment.SOMA] + b_o

            out_result = out_pop.model.step(
                out_pop.state,
                NeuronInputs(drive={Compartment.SOMA: o_current}),
                ctx,
            )
            out_spikes_list.append(out_result.spikes)

        spike_tensor = torch_mod.stack(out_spikes_list, dim=0)
        return spike_tensor, in_counts, hid_counts

    def run(self) -> MultiGateResult:  # noqa: C901, PLR0912, PLR0915
        """Train all gates with shared weights and return result."""
        from neuroforge.core.torch_utils import require_torch, resolve_device_dtype
        from neuroforge.encoding.rate import RateEncoderParams
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.network.factory import to_topology_json

        torch = require_torch()
        cfg = self.config

        rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)

        dev, tdt = resolve_device_dtype(cfg.device, cfg.dtype)

        gates = list(cfg.gates)
        n_input = cfg.image_h * cfg.image_w
        n_h = cfg.n_hidden

        engine, trainables = build_gate_network(cfg, torch_mod=torch, dev=dev, tdt=tdt)
        proj_ih = engine.projections["input_hidden"]
        proj_ho = engine.projections["hidden_output"]

        w_ih = trainables["w_ih"]
        b_h = trainables["b_h"]
        w_ho = trainables["w_ho"]
        b_o = trainables["b_o"]

        params = [w_ih, b_h, w_ho, b_o]
        for p in params:
            p.requires_grad_(True)
        _rebind_projection_weights(proj_ih, w_ih)
        _rebind_projection_weights(proj_ho, w_ho)

        optimizer = torch.optim.Adam(params, lr=cfg.lr)

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
                "gates": list(gates),
                "max_epochs": cfg.max_epochs,
                "window_steps": cfg.window_steps,
                "per_pattern_streak": cfg.per_pattern_streak,
                "n_input": n_input,
                "n_hidden": n_h,
                "image_h": cfg.image_h,
                "image_w": cfg.image_w,
            },
        )
        self._emit("topology", 0, "MULTI", to_topology_json(engine))

        patterns: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        all_combos: list[tuple[str, tuple[int, int]]] = [(g, p) for g in gates for p in patterns]
        combo_streaks: dict[tuple[str, tuple[int, int]], int] = {c: 0 for c in all_combos}

        streak = 0
        recent_correct: list[bool] = []
        accuracy_history: list[float] = []
        trial = 0

        for epoch in range(cfg.max_epochs):
            order = list(all_combos)
            rng.shuffle(order)

            for gate, inp in order:
                if self._stop_check is not None and self._stop_check():
                    return self._make_result(
                        cfg,
                        gates,
                        combo_streaks,
                        False,
                        trial,
                        epoch,
                        w_ih,
                        w_ho,
                        b_h,
                        b_o,
                        accuracy_history,
                        stopped=True,
                    )

                gate_idx = GATE_INDEX[gate]
                expected = GATE_TABLES[gate][inp]

                img = _pattern_gate_to_image(
                    inp,
                    gate_idx,
                    len(ALL_GATES),
                    cfg.image_h,
                    cfg.image_w,
                    torch,
                )
                flat = img.reshape(-1).to(device=dev, dtype=tdt)
                drives = encoder.encode(flat)

                optimizer.zero_grad()

                spike_tensor, in_counts, hid_counts = self._forward(
                    drives,
                    engine,
                    cfg,
                    b_h,
                    b_o,
                    torch,
                )

                ro_result = readout(spike_tensor)
                out_count = int(ro_result.count.detach().item())
                predicted = 1 if out_count >= cfg.spike_threshold else 0
                correct = predicted == expected

                target_count = float(expected) * cfg.spike_threshold
                target_t = torch.tensor([target_count], dtype=tdt, device=dev)
                loss = loss_fn(ro_result.count, target_t)
                if not correct:
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        for p in params:
                            p.clamp_(cfg.w_min, cfg.w_max)

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

                error_val = float(expected - predicted)
                spike_data: dict[str, Any] = {
                    "gate": gate,
                    "input": list(inp),
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                    "error": error_val,
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "out_spike_count": out_count,
                    "input_spikes": in_counts.detach().cpu().tolist(),
                    "hidden_spikes": hid_counts.detach().cpu().tolist(),
                    "output_spikes": [out_count],
                }
                self._emit("training_trial", trial, f"MULTI/{gate}", spike_data)

                self._emit(
                    "scalar",
                    trial,
                    "MULTI",
                    {
                        "trial": trial,
                        "epoch": epoch,
                        "gate": gate,
                        "accuracy": accuracy,
                        "error": error_val,
                        "loss": float(loss.detach().item()),
                        "correct": correct,
                    },
                )

                if trial % 10 == 0:
                    self._emit(
                        "weight",
                        trial,
                        "MULTI/input-hidden",
                        {"weights": w_ih.detach()},
                    )
                    self._emit(
                        "weight",
                        trial,
                        "MULTI/hidden-output",
                        {"weights": w_ho.detach()},
                    )

                    self._emit(
                        "spike",
                        trial,
                        "MULTI/input",
                        {"spikes": (in_counts > 0)},
                    )
                    self._emit(
                        "spike",
                        trial,
                        "MULTI/hidden",
                        {"spikes": (hid_counts > 0)},
                    )
                    self._emit(
                        "spike",
                        trial,
                        "MULTI/output",
                        {"spikes": torch.tensor([out_count > 0], device=dev)},
                    )

                trial += 1

            all_combos_reliable = all(
                s >= cfg.per_pattern_streak for s in combo_streaks.values()
            )
            if streak >= cfg.convergence_streak and all_combos_reliable:
                return self._make_result(
                    cfg,
                    gates,
                    combo_streaks,
                    True,
                    trial,
                    epoch + 1,
                    w_ih,
                    w_ho,
                    b_h,
                    b_o,
                    accuracy_history,
                )

        return self._make_result(
            cfg,
            gates,
            combo_streaks,
            False,
            trial,
            cfg.max_epochs,
            w_ih,
            w_ho,
            b_h,
            b_o,
            accuracy_history,
        )

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
        """Build a MultiGateResult and emit training_end."""
        patterns: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]

        per_gate: dict[str, bool] = {}
        for g in gates:
            per_gate[g] = all(
                combo_streaks[(g, p)] >= cfg.per_pattern_streak for p in patterns
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
