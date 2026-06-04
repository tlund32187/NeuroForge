# pyright: basic
"""Compile a structural ``GraphGenome`` into a spiking policy network.

Generalises :func:`~neuroforge.game.policies.network.build_policy_network` from
the fixed ``input -> hidden -> motor`` chain to the arbitrary node/connection
graph an evolved genome describes — multiple hidden populations, skip edges,
parallel pathways, recurrence — while keeping the same Dale's-law non-negative
synapses the R-STDP loop needs. It returns the same :class:`PolicyNetwork` bundle
the stateful engine, decoder, and trainer already consume, so an *invented*
topology drops straight into the live game-training stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.engine.core_engine import CoreEngine, Population, Projection
from neuroforge.game.policies.network import N_BUTTONS, PolicyNetwork
from neuroforge.network.topology_builders import build_dense_topology

if TYPE_CHECKING:
    from neuroforge.evolution.graph_genome import GraphGenome
    from neuroforge.factories.hub import FactoryHub

__all__ = ["build_graph_policy_network"]


def build_graph_policy_network(
    genome: GraphGenome,
    *,
    n_input: int,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float32",
    v_thresh: float = 1.0,
    dt: float = 1e-3,
    hub: FactoryHub | None = None,
) -> PolicyNetwork:
    """Build a :class:`CoreEngine` policy from an evolved network graph."""
    from neuroforge.core.torch_utils import require_torch, resolve_device_dtype
    from neuroforge.factories.hub import DEFAULT_HUB
    from neuroforge.neurons.lif.model import LIFParams

    torch = require_torch()
    factory_hub = hub or DEFAULT_HUB
    dev, tdt = resolve_device_dtype(device, dtype)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    motor_per_button = int(genome.value("motor_per_button"))
    tau_mem = float(genome.value("tau_mem"))
    n_motor = N_BUTTONS * motor_per_button

    # Resolve population sizes (I/O sizes are fixed by the environment, not evolved).
    sizes: dict[str, int] = {}
    for node in genome.nodes:
        if node.role == "input":
            sizes[node.id] = n_input
        elif node.role == "motor":
            sizes[node.id] = n_motor
        else:
            sizes[node.id] = max(1, int(node.size))

    engine = CoreEngine(
        SimulationConfig(dt=dt, seed=seed, device=device, dtype=dtype),
    )

    def _lif() -> Any:
        return factory_hub.neurons.create(
            "lif", params=LIFParams(tau_mem=tau_mem, v_thresh=v_thresh),
        )

    for node in genome.nodes:
        engine.add_population(Population(node.id, _lif(), sizes[node.id]))

    # Dale sign per source population (input/motor excitatory; hidden split E/I).
    signs: dict[str, Any] = {
        node.id: _sign_vector(
            sizes[node.id],
            node.inhibitory_frac if node.role == "hidden" else 0.0,
            dev, tdt, torch,
        )
        for node in genome.nodes
    }

    plastic: list[str] = []
    for conn in genome.enabled_connections():
        if conn.src not in sizes or conn.dst not in sizes:
            continue
        n_pre, n_post = sizes[conn.src], sizes[conn.dst]
        syn = factory_hub.synapses.create("static_dales", sign_pre=signs[conn.src])
        # Non-negative magnitudes (Dale supplies the sign); per-connection init scale.
        matrix = (
            torch.empty(n_pre, n_post, dtype=tdt)
            .uniform_(0.0, max(1e-3, float(conn.weight_scale)))
            .to(dev)
        )
        topo = build_dense_topology(matrix, n_pre, n_post, dev, torch)
        topo.weights.requires_grad_(False)  # R-STDP mutates in place, gradient-free
        name = _projection_name(conn.src, conn.dst)
        engine.add_projection(
            Projection(name=name, model=syn, source=conn.src, target=conn.dst, topology=topo),
        )
        if conn.plastic:
            plastic.append(name)

    engine.build()
    n_hidden = sum(sizes[node.id] for node in genome.nodes if node.role == "hidden")
    return PolicyNetwork(
        engine=engine,
        n_input=n_input,
        n_hidden=n_hidden,
        n_motor=n_motor,
        input_pop="input",
        motor_pop="motor",
        motor_per_button=motor_per_button,
        n_buttons=N_BUTTONS,
        plastic_projections=tuple(plastic),
    )


def _projection_name(src: str, dst: str) -> str:
    return f"{src}__to__{dst}"


def _sign_vector(size: int, inhibitory_frac: float, dev: Any, tdt: Any, torch: Any) -> Any:
    signs = torch.ones(size, dtype=tdt)
    n_inhibitory = int(round(inhibitory_frac * size))
    if n_inhibitory > 0:
        signs[:n_inhibitory] = -1.0
    return signs.to(dev)
