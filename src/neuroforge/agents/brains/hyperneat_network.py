"""Compile a HyperNEAT genome into a spiking policy network.

This is the HyperNEAT analog of the graph policy builder. Instead of reading an
explicit graph, it lays the input/hidden/motor populations on a fixed substrate
and queries the genome CPPN over every potential connection to paint each
projection weight matrix.

It produces the same Dale-law, non-negative-magnitude PolicyNetwork consumed by
the stateful engine, decoder, and R-STDP trainer, so evolved HyperNEAT topology
drops into the live game-training stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.agents.brains.policy_network import N_BUTTONS, PolicyNetwork
from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.neuroevolution.genomes.substrate import Substrate
from neuroforge.simulation.engine.core import CoreEngine, Population, Projection
from neuroforge.simulation.topology.builders import build_dense_topology

if TYPE_CHECKING:
    from neuroforge.construction.hub import FactoryHub
    from neuroforge.neuroevolution.genomes.hyperneat import HyperNEATGenome

__all__ = ["build_hyperneat_policy_network"]

_DEFAULT_INHIBITORY_FRAC = 0.2  # Dale E/I split for the hidden sheet


def build_hyperneat_policy_network(
    genome: HyperNEATGenome,
    *,
    n_input: int,
    seed: int = 42,
    device: str = "cpu",
    dtype: str = "float32",
    v_thresh: float = 1.0,
    dt: float = 1e-3,
    hub: FactoryHub | None = None,
) -> PolicyNetwork:
    """Decode *genome*'s CPPN over its substrate into a :class:`CoreEngine` policy."""
    from neuroforge.biology.neurons.models.lif.params import LIFParams
    from neuroforge.construction.composition_root import DEFAULT_HUB
    from neuroforge.construction.gate_builder import build_dale_signs
    from neuroforge.kernel.torch_utils import require_torch, resolve_device_dtype

    torch = require_torch()
    factory_hub = hub or DEFAULT_HUB
    dev, tdt = resolve_device_dtype(device, dtype)

    motor_per_button = int(genome.value("motor_per_button"))
    tau_mem = float(genome.value("tau_mem"))
    init_scale = float(genome.value("init_scale"))
    n_motor = N_BUTTONS * motor_per_button

    substrate = Substrate(
        genome.substrate,
        n_input=n_input,
        n_buttons=N_BUTTONS,
        motor_per_button=motor_per_button,
        torch=torch,
        dev=dev,
        dtype=tdt,
    )
    n_hidden = substrate.count("hidden")

    engine = CoreEngine(SimulationConfig(dt=dt, seed=seed, device=device, dtype=dtype))

    def _lif() -> Any:
        return factory_hub.neurons.create(
            "lif", params=LIFParams(tau_mem=tau_mem, v_thresh=v_thresh),
        )

    sizes = {"input": n_input, "hidden": n_hidden, "motor": n_motor}
    for layer, size in sizes.items():
        engine.add_population(Population(layer, _lif(), size))

    # Dale signs: input excitatory, hidden split E/I â€” reuse the shared builder so the
    # sign convention matches the hand-built policy network exactly (DRY).
    n_inhibitory = round(_DEFAULT_INHIBITORY_FRAC * n_hidden)
    dale = build_dale_signs(n_input, n_hidden, n_inhibitory, dev, tdt, torch)
    signs: dict[str, Any] = {"input": dale["sign_input"], "hidden": dale["sign_hidden"]}

    weight_idx = genome.cppn.outputs.index("weight")
    has_leo = "expression" in genome.cppn.outputs
    expression_idx = genome.cppn.outputs.index("expression") if has_leo else -1
    threshold = float(genome.substrate.weight_threshold)

    plastic: list[str] = []
    for src, dst in substrate.connection_pairs():
        n_pre, n_post = sizes[src], sizes[dst]
        features = substrate.query_features(src, dst)            # [P, query_dim]
        out = genome.cppn.query(features, torch=torch)           # [P, n_out], tanh-bounded
        raw = out[:, weight_idx]                                 # [-1, 1] connection weight
        # Locality: prune the sub-threshold band, and (if LEO is on) any link the
        # expression output switches off â€” yielding sparse, localized receptive fields.
        keep = raw.abs() >= threshold
        if has_leo:
            keep = keep & (out[:, expression_idx] > 0.0)
        magnitude = raw.abs() * init_scale * keep.to(tdt)        # non-negative (Dale adds sign)
        matrix = magnitude.reshape(n_pre, n_post)

        syn = factory_hub.synapses.create("static_dales", sign_pre=signs[src])
        topo = build_dense_topology(matrix, n_pre, n_post, dev, torch)
        topo.weights.requires_grad_(False)  # R-STDP mutates in place, gradient-free
        name = f"{src}__to__{dst}"
        engine.add_projection(
            Projection(name=name, model=syn, source=src, target=dst, topology=topo),
        )
        plastic.append(name)

    engine.build()
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
