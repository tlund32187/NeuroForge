# pyright: basic
"""Build the spiking policy network: input -> hidden (E/I) -> motor.

A small, stateful :class:`CoreEngine` the live policy drives directly (unlike
the stateless batch vision backbone): pixel drive enters ``input`` LIF neurons,
flows through a Dale's-law-split ``hidden`` layer, and excites a ``motor`` layer
of 8 button assemblies. Projections use the Dale static synapse (``|w|*sign``),
so weights stay non-negative magnitudes — the form the Phase-3 R-STDP loop
wants. Weights are random here; learning shapes them later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.game import NINTENDO_BUTTONS
from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.engine.core_engine import CoreEngine, Population, Projection
from neuroforge.network.gate_builder import build_dale_signs
from neuroforge.network.topology_builders import (
    build_dense_topology,
    build_sparse_fanin_topology,
)

if TYPE_CHECKING:
    from neuroforge.factories.hub import FactoryHub

__all__ = ["N_BUTTONS", "PolicyNetwork", "PolicyNetworkConfig", "build_policy_network"]

N_BUTTONS: int = len(NINTENDO_BUTTONS)  # 8


@dataclass(frozen=True, slots=True)
class PolicyNetworkConfig:
    """Shape/initialisation of the spiking policy network."""

    n_input: int
    n_hidden: int = 128
    n_inhibitory_hidden: int = -1       # -1 => auto (~20% of hidden are inhibitory, Dale)
    motor_per_button: int = 4           # motor neurons per button (assembly size)
    input_fanin: int = 64               # sparse fan-in for input->hidden (0 = dense)
    recurrent_hidden: bool = False
    init_scale: float = 0.5
    tau_mem: float = 5e-3       # fast membrane so the cascade fires within the tick window
    v_thresh: float = 1.0
    dt: float = 1e-3
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.n_input <= 0 or self.n_hidden <= 0:
            msg = "PolicyNetworkConfig n_input/n_hidden must be > 0"
            raise ValueError(msg)
        if self.motor_per_button < 1:
            msg = "PolicyNetworkConfig.motor_per_button must be >= 1"
            raise ValueError(msg)
        if self.n_inhibitory_hidden < 0:  # auto: ~20% inhibitory
            object.__setattr__(self, "n_inhibitory_hidden", round(0.2 * self.n_hidden))
        if not 0 <= self.n_inhibitory_hidden <= self.n_hidden:
            msg = "PolicyNetworkConfig.n_inhibitory_hidden must be in [0, n_hidden]"
            raise ValueError(msg)


@dataclass
class PolicyNetwork:
    """A built spiking policy network and the metadata the engine/decoder need."""

    engine: CoreEngine
    n_input: int
    n_hidden: int
    n_motor: int
    input_pop: str
    motor_pop: str
    motor_per_button: int
    n_buttons: int
    plastic_projections: tuple[str, ...]


def build_policy_network(
    config: PolicyNetworkConfig, *, hub: FactoryHub | None = None,
) -> PolicyNetwork:
    """Construct and build a :class:`CoreEngine` policy network."""
    from neuroforge.core.torch_utils import require_torch, resolve_device_dtype
    from neuroforge.factories.hub import DEFAULT_HUB
    from neuroforge.neurons.lif.model import LIFParams

    cfg = config
    factory_hub = hub or DEFAULT_HUB
    torch = require_torch()
    dev, tdt = resolve_device_dtype(cfg.device, cfg.dtype)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(cfg.seed)

    n_motor = N_BUTTONS * cfg.motor_per_button
    engine = CoreEngine(
        SimulationConfig(dt=cfg.dt, seed=cfg.seed, device=cfg.device, dtype=cfg.dtype),
    )

    def _lif() -> Any:
        return factory_hub.neurons.create(
            "lif", params=LIFParams(tau_mem=cfg.tau_mem, v_thresh=cfg.v_thresh),
        )

    engine.add_population(Population("input", _lif(), cfg.n_input))
    engine.add_population(Population("hidden", _lif(), cfg.n_hidden))
    engine.add_population(Population("motor", _lif(), n_motor))

    signs = build_dale_signs(
        cfg.n_input, cfg.n_hidden, cfg.n_inhibitory_hidden, dev, tdt, torch,
    )
    sign_input = signs["sign_input"]
    sign_hidden = signs["sign_hidden"]

    def _add_proj(
        name: str, src: str, tgt: str, n_pre: int, n_post: int, sign_pre: Any, fanin: int,
    ) -> None:
        syn = factory_hub.synapses.create("static_dales", sign_pre=sign_pre)
        # Non-negative magnitudes: the Dale synapse supplies the sign (|w|*sign),
        # so stored weights stay >= 0 and the R-STDP [0, w_max] clamp is correct.
        if 0 < fanin < n_pre:
            topo, _weights = build_sparse_fanin_topology(
                n_pre, n_post, fanin,
                init_cfg={"init": "uniform", "low": 0.0, "high": cfg.init_scale},
                delay_cfg={}, dev=dev, tdt=tdt, torch=torch, rng=rng, sort=True,
            )
        else:
            matrix = torch.empty(n_pre, n_post, dtype=tdt).uniform_(
                0.0, cfg.init_scale,
            ).to(dev)
            topo = build_dense_topology(matrix, n_pre, n_post, dev, torch)
        # R-STDP is gradient-free and mutates weights in place; don't track grads.
        topo.weights.requires_grad_(False)
        engine.add_projection(
            Projection(name=name, model=syn, source=src, target=tgt, topology=topo),
        )

    _add_proj(
        "in_to_hidden", "input", "hidden",
        cfg.n_input, cfg.n_hidden, sign_input, cfg.input_fanin,
    )
    _add_proj("hidden_to_motor", "hidden", "motor", cfg.n_hidden, n_motor, sign_hidden, 0)
    plastic = ["in_to_hidden", "hidden_to_motor"]
    if cfg.recurrent_hidden:
        _add_proj(
            "hidden_to_hidden", "hidden", "hidden",
            cfg.n_hidden, cfg.n_hidden, sign_hidden, 0,
        )
        plastic.append("hidden_to_hidden")

    engine.build()
    return PolicyNetwork(
        engine=engine,
        n_input=cfg.n_input,
        n_hidden=cfg.n_hidden,
        n_motor=n_motor,
        input_pop="input",
        motor_pop="motor",
        motor_per_button=cfg.motor_per_button,
        n_buttons=N_BUTTONS,
        plastic_projections=tuple(plastic),
    )
