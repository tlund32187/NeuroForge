"""Structural genome for neuroevolution (NEAT-style topology invention).

Unlike :class:`~neuroforge.neuroevolution.genomes.policy.PolicyGenome` (a fixed vector of
hyperparameters), this genome encodes the network *graph explicitly* — a set of
**node genes** (populations) and **connection genes** (projections) — so
evolution can *invent structure*: grow hidden nodes, add skip/parallel/recurrent
pathways, and disable connections. Connections carry NEAT innovation numbers
(from :class:`~neuroforge.neuroevolution.genomes.innovations.InnovationRegistry`) so crossover
aligns shared genes and speciation measures real structural distance.

Fixed I/O nodes ``input`` and ``motor`` anchor the graph (their sizes are set at
build time from the encoder/buttons); hidden nodes and all connections evolve.
A handful of global hyperparameter genes (learning + neuron + decode knobs) ride
alongside the graph. The phenotype is compiled to a spiking policy by
:func:`neuroforge.agents.brains.graph_network.build_graph_policy_network`.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from neuroforge.neuroevolution.genomes.policy import Gene, GeneDef

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuroforge.neuroevolution.genomes.innovations import InnovationRegistry

__all__ = [
    "ConnGene",
    "GraphGenome",
    "GraphReproduction",
    "NodeGene",
    "graph_hyperparam_defs",
    "make_graph_seed_population",
]

INPUT_NODE = "input"
MOTOR_NODE = "motor"


@dataclass(frozen=True, slots=True)
class NodeGene:
    """One population in the network graph."""

    id: str
    role: str               # "input" | "hidden" | "motor"
    size: int = 32
    inhibitory_frac: float = 0.2   # Dale E/I split for this population's outputs


@dataclass(frozen=True, slots=True)
class ConnGene:
    """One projection (edge) in the network graph, with a NEAT innovation number."""

    innovation: int
    src: str
    dst: str
    weight_scale: float = 0.5
    plastic: bool = True
    enabled: bool = True


def graph_hyperparam_defs() -> tuple[GeneDef, ...]:
    """Global (non-structural) hyperparameter genes carried by the graph genome."""
    return (
        GeneDef(0, "motor_per_button", "int", 4, 1, 8, 1),
        GeneDef(0, "init_scale", "float", 0.5, 0.05, 1.5, 0.12),
        GeneDef(0, "tau_mem", "float", 5e-3, 2e-3, 20e-3, 1.5e-3),
        GeneDef(0, "decide_ticks", "int", 12, 4, 28, 3),
        GeneDef(0, "noise_amp", "float", 0.25, 0.0, 0.6, 0.06),
        GeneDef(0, "commit_frames", "int", 4, 0, 20, 2),
        GeneDef(0, "lr", "float", 3e-4, 1e-5, 3e-3, 2e-4),
        GeneDef(0, "tau_e", "float", 80e-3, 20e-3, 250e-3, 18e-3),
        GeneDef(0, "reward_scale", "float", 0.05, 0.005, 0.2, 0.02),
    )


_HYPER_DEFS: tuple[GeneDef, ...] = graph_hyperparam_defs()
_HYPER_BY_KEY: dict[str, GeneDef] = {d.key: d for d in _HYPER_DEFS}

_HIDDEN_SIZE_MIN, _HIDDEN_SIZE_MAX = 8, 256


@dataclass(frozen=True, slots=True)
class GraphGenome:
    """A network graph (nodes + connections) plus global hyperparameter genes."""

    id: str
    generation: int
    nodes: tuple[NodeGene, ...]
    connections: tuple[ConnGene, ...]
    hyperparams: tuple[Gene, ...]
    parent_ids: tuple[str, ...] = ()

    # -- construction --------------------------------------------------

    @classmethod
    def seed(
        cls,
        genome_id: str,
        *,
        innovations: InnovationRegistry,
        generation: int = 0,
        rng: random.Random | None = None,
        randomise: bool = True,
    ) -> GraphGenome:
        """Minimal graph: input -> one hidden node -> motor, with hyperparams."""
        local_rng = rng or random.Random()
        hidden_size = (
            local_rng.randint(_HIDDEN_SIZE_MIN, _HIDDEN_SIZE_MAX) if randomise else 64
        )
        nodes = (
            NodeGene(INPUT_NODE, "input", size=0),
            NodeGene("h0", "hidden", size=hidden_size),
            NodeGene(MOTOR_NODE, "motor", size=0),
        )
        connections = (
            ConnGene(innovations.connection(INPUT_NODE, "h0"), INPUT_NODE, "h0"),
            ConnGene(innovations.connection("h0", MOTOR_NODE), "h0", MOTOR_NODE),
        )
        hyperparams = tuple(
            Gene(0, d.key, d.random_value(local_rng) if randomise else d.default)
            for d in _HYPER_DEFS
        )
        return cls(
            id=genome_id,
            generation=generation,
            nodes=nodes,
            connections=connections,
            hyperparams=hyperparams,
        )

    # -- gene access ---------------------------------------------------

    def value(self, key: str) -> int | float | bool:
        """Return one hyperparameter gene value by key."""
        for gene in self.hyperparams:
            if gene.key == key:
                return gene.value
        msg = f"hyperparameter {key!r} not found"
        raise KeyError(msg)

    def hidden_nodes(self) -> tuple[NodeGene, ...]:
        """Hidden (evolvable) populations only."""
        return tuple(node for node in self.nodes if node.role == "hidden")

    def enabled_connections(self) -> tuple[ConnGene, ...]:
        """Connections currently expressed in the phenotype."""
        return tuple(conn for conn in self.connections if conn.enabled)

    def content_key(self) -> str:
        """Stable key over structure + hyperparameters (ignores id/lineage)."""
        node_part = ";".join(
            f"{n.id}:{n.role}:{n.size}:{n.inhibitory_frac:.4f}"
            for n in sorted(self.nodes, key=lambda node: node.id)
        )
        conn_part = ";".join(
            f"{c.innovation}:{c.src}->{c.dst}:{c.weight_scale:.4f}:{int(c.plastic)}:{int(c.enabled)}"
            for c in sorted(self.connections, key=lambda conn: conn.innovation)
        )
        hyper_part = ";".join(
            f"{g.key}={g.value!r}" for g in sorted(self.hyperparams, key=lambda g: g.key)
        )
        return f"N[{node_part}]|C[{conn_part}]|H[{hyper_part}]"

    # -- lineage -------------------------------------------------------

    def as_offspring(
        self, *, child_id: str, generation: int, parent_ids: tuple[str, ...] | None = None,
    ) -> GraphGenome:
        """Copy with new lineage but identical structure (elite / asexual clone)."""
        return dataclasses.replace(
            self,
            id=child_id,
            generation=generation,
            parent_ids=parent_ids if parent_ids is not None else (self.id,),
        )

    # -- variation -----------------------------------------------------

    def mutate(
        self,
        *,
        child_id: str,
        generation: int,
        rng: random.Random,
        rate: float,
        power: float = 1.0,
        innovations: InnovationRegistry,
    ) -> GraphGenome:
        """Return a mutated child: structural growth + attribute perturbation."""
        nodes = list(self.nodes)
        connections = list(self.connections)

        # Structural mutations (rarer than attribute tweaks, gated by `rate`).
        if rng.random() < rate * 0.5:
            self._mutate_add_connection(nodes, connections, rng, innovations)
        if rng.random() < rate * 0.3:
            self._mutate_add_node(nodes, connections, rng, innovations)
        if connections and rng.random() < rate * 0.2:
            idx = rng.randrange(len(connections))
            conn = connections[idx]
            connections[idx] = dataclasses.replace(conn, enabled=not conn.enabled)

        # Attribute perturbations (hidden sizes, connection scales).
        nodes = [self._maybe_perturb_node(node, rng, rate, power) for node in nodes]
        connections = [
            self._maybe_perturb_connection(conn, rng, rate, power) for conn in connections
        ]
        hyperparams = tuple(
            Gene(0, gene.key, _HYPER_BY_KEY[gene.key].mutate(gene.value, rng, power=power))
            if gene.key in _HYPER_BY_KEY and rng.random() < rate
            else gene
            for gene in self.hyperparams
        )
        return GraphGenome(
            id=child_id,
            generation=generation,
            nodes=tuple(nodes),
            connections=tuple(connections),
            hyperparams=hyperparams,
            parent_ids=(self.id,),
        )

    def crossover(
        self,
        other: GraphGenome,
        *,
        child_id: str,
        generation: int,
        rng: random.Random,
    ) -> GraphGenome:
        """Innovation-aligned crossover: shared genes random, disjoint kept from self."""
        other_by_innovation = {conn.innovation: conn for conn in other.connections}
        child_conns: list[ConnGene] = []
        for conn in self.connections:
            mate = other_by_innovation.get(conn.innovation)
            chosen = mate if mate is not None and rng.random() < 0.5 else conn
            child_conns.append(chosen)
        needed = {INPUT_NODE, MOTOR_NODE}
        for conn in child_conns:
            needed.add(conn.src)
            needed.add(conn.dst)
        nodes_by_id = {node.id: node for node in self.nodes}
        for node in other.nodes:
            nodes_by_id.setdefault(node.id, node)
        child_nodes = tuple(nodes_by_id[node_id] for node_id in nodes_by_id if node_id in needed)
        hyperparams = tuple(
            other_gene if rng.random() < 0.5 else self_gene
            for self_gene, other_gene in zip(self.hyperparams, other.hyperparams, strict=False)
        )
        return GraphGenome(
            id=child_id,
            generation=generation,
            nodes=child_nodes,
            connections=tuple(child_conns),
            hyperparams=hyperparams or self.hyperparams,
            parent_ids=(self.id, other.id),
        )

    def distance(self, other: GraphGenome) -> float:
        """NEAT-style compatibility: disjoint connections + matched weight diffs."""
        mine = {conn.innovation: conn for conn in self.connections}
        theirs = {conn.innovation: conn for conn in other.connections}
        all_innovations = set(mine) | set(theirs)
        if not all_innovations:
            return 0.0
        disjoint = 0
        weight_diff = 0.0
        matched = 0
        for innovation in all_innovations:
            left = mine.get(innovation)
            right = theirs.get(innovation)
            if left is None or right is None:
                disjoint += 1
            else:
                weight_diff += abs(left.weight_scale - right.weight_scale)
                matched += 1
        normaliser = max(len(mine), len(theirs), 1)
        avg_weight = (weight_diff / matched) if matched else 0.0
        node_diff = abs(len(self.hidden_nodes()) - len(other.hidden_nodes()))
        return disjoint / normaliser + 0.4 * avg_weight + 0.1 * node_diff

    # -- phenotype wiring ----------------------------------------------

    def to_game_training_config(self, *, base: Any | None = None, seed: int) -> Any:
        """Apply this genome's hyperparameters onto a GameTrainingConfig.

        The *structure* is supplied separately by :meth:`make_network_builder`;
        this only sets the learning/decode/neuron knobs the task still reads.
        """
        from neuroforge.applications.tasks.game_training import GameTrainingConfig
        from neuroforge.learning.training_loop import OnlineRSTDPConfig

        cfg = base or GameTrainingConfig()
        rstdp = dataclasses.replace(
            cfg.rstdp if hasattr(cfg, "rstdp") else OnlineRSTDPConfig(),
            lr=float(self.value("lr")),
            tau_e=float(self.value("tau_e")),
            reward_scale=float(self.value("reward_scale")),
        )
        return dataclasses.replace(
            cfg,
            decide_ticks=int(self.value("decide_ticks")),
            noise_amp=float(self.value("noise_amp")),
            commit_frames=int(self.value("commit_frames")),
            motor_per_button=int(self.value("motor_per_button")),
            tau_mem=float(self.value("tau_mem")),
            init_scale=float(self.value("init_scale")),
            rstdp=rstdp,
            seed=seed,
        )

    def make_network_builder(
        self, *, seed: int, device: str = "cpu", dtype: str = "float32",
    ) -> Any:
        """Return ``n_input -> PolicyNetwork`` that compiles this genome's graph."""
        from neuroforge.agents.brains.graph_network import build_graph_policy_network

        def builder(n_input: int) -> Any:
            return build_graph_policy_network(
                self, n_input=n_input, seed=seed, device=device, dtype=dtype,
            )

        return builder

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "type": "graph",
            "id": self.id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "nodes": [dataclasses.asdict(node) for node in self.nodes],
            "connections": [dataclasses.asdict(conn) for conn in self.connections],
            "hyperparams": [dataclasses.asdict(gene) for gene in self.hyperparams],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphGenome:
        """Decode a genome from :meth:`to_dict` output."""
        nodes = tuple(
            NodeGene(
                id=str(item["id"]),
                role=str(item["role"]),
                size=int(item["size"]),
                inhibitory_frac=float(item.get("inhibitory_frac", 0.2)),
            )
            for item in payload.get("nodes", [])
        )
        connections = tuple(
            ConnGene(
                innovation=int(item["innovation"]),
                src=str(item["src"]),
                dst=str(item["dst"]),
                weight_scale=float(item.get("weight_scale", 0.5)),
                plastic=bool(item.get("plastic", True)),
                enabled=bool(item.get("enabled", True)),
            )
            for item in payload.get("connections", [])
        )
        hyper_by_key = {
            str(item["key"]): item["value"] for item in payload.get("hyperparams", [])
        }
        hyperparams = tuple(
            Gene(0, d.key, hyper_by_key.get(d.key, d.default)) for d in _HYPER_DEFS
        )
        parents = payload.get("parent_ids", [])
        return cls(
            id=str(payload["id"]),
            generation=int(payload["generation"]),
            nodes=nodes,
            connections=connections,
            hyperparams=hyperparams,
            parent_ids=(
                tuple(str(pid) for pid in cast("Sequence[Any]", parents))
                if isinstance(parents, list)
                else ()
            ),
        )

    # -- internals -----------------------------------------------------

    def _mutate_add_connection(
        self,
        nodes: list[NodeGene],
        connections: list[ConnGene],
        rng: random.Random,
        innovations: InnovationRegistry,
    ) -> None:
        existing = {(conn.src, conn.dst) for conn in connections}
        sources = [n.id for n in nodes if n.role != "motor"]
        targets = [n.id for n in nodes if n.role != "input"]
        rng.shuffle(sources)
        for src in sources:
            for dst in targets:
                if src == INPUT_NODE and dst == MOTOR_NODE:
                    continue  # the trivial reflex edge is already discoverable via h-nodes
                if (src, dst) not in existing:
                    connections.append(
                        ConnGene(innovations.connection(src, dst), src, dst,
                                 weight_scale=rng.uniform(0.1, 0.8)),
                    )
                    return

    def _mutate_add_node(
        self,
        nodes: list[NodeGene],
        connections: list[ConnGene],
        rng: random.Random,
        innovations: InnovationRegistry,
    ) -> None:
        enabled = [idx for idx, conn in enumerate(connections) if conn.enabled]
        if not enabled:
            return
        idx = rng.choice(enabled)
        conn = connections[idx]
        node_id_num, in_innovation, out_innovation = innovations.node_split(conn.innovation)
        new_id = f"h{node_id_num}"
        connections[idx] = dataclasses.replace(conn, enabled=False)
        nodes.append(
            NodeGene(new_id, "hidden", size=rng.randint(_HIDDEN_SIZE_MIN, _HIDDEN_SIZE_MAX)),
        )
        connections.append(ConnGene(in_innovation, conn.src, new_id, weight_scale=1.0))
        connections.append(
            ConnGene(out_innovation, new_id, conn.dst, weight_scale=conn.weight_scale),
        )

    def _maybe_perturb_node(
        self, node: NodeGene, rng: random.Random, rate: float, power: float,
    ) -> NodeGene:
        if node.role != "hidden" or rng.random() >= rate:
            return node
        delta = int(round(rng.gauss(0.0, 16.0 * power)))
        size = max(_HIDDEN_SIZE_MIN, min(_HIDDEN_SIZE_MAX, node.size + delta))
        frac = min(0.5, max(0.0, node.inhibitory_frac + rng.gauss(0.0, 0.05 * power)))
        return dataclasses.replace(node, size=size, inhibitory_frac=frac)

    def _maybe_perturb_connection(
        self, conn: ConnGene, rng: random.Random, rate: float, power: float,
    ) -> ConnGene:
        if rng.random() >= rate:
            return conn
        scale = max(0.02, min(1.5, conn.weight_scale + rng.gauss(0.0, 0.12 * power)))
        return dataclasses.replace(conn, weight_scale=scale)


class GraphReproduction:
    """Reproduction for :class:`GraphGenome` that threads the innovation registry.

    Mirrors the default :class:`~neuroforge.neuroevolution.search.engine.SimpleReproduction`
    (elitism + fitness-proportionate selection + crossover) but passes the shared
    :class:`InnovationRegistry` into structural mutation so add-node/add-connection
    get consistent innovation numbers across the population. Inject it into
    :class:`~neuroforge.neuroevolution.search.engine.EvolutionEngine`.
    """

    def __init__(self, config: Any, innovations: InnovationRegistry) -> None:
        self._cfg = config
        self._innovations = innovations

    def next_generation(
        self, evaluated: list[Any], *, generation: int, rng: random.Random,
    ) -> list[Any]:
        """Return the next graph-genome population."""
        cfg = self._cfg
        children: list[Any] = [
            item.genome.as_offspring(child_id=f"g{generation}_{idx}", generation=generation)
            for idx, item in enumerate(evaluated[: cfg.elite_count])
        ]
        while len(children) < cfg.population_size:
            child_index = len(children)
            parent_a = _select_parent(evaluated, rng)
            if rng.random() < cfg.crossover_rate:
                parent_b = _select_parent(evaluated, rng)
                child = parent_a.genome.crossover(
                    parent_b.genome,
                    child_id=f"g{generation}_{child_index}",
                    generation=generation,
                    rng=rng,
                )
            else:
                child = parent_a.genome.as_offspring(
                    child_id=f"g{generation}_{child_index}", generation=generation,
                )
            children.append(
                child.mutate(
                    child_id=child.id,
                    generation=generation,
                    rng=rng,
                    rate=cfg.mutation_rate,
                    power=cfg.mutation_power,
                    innovations=self._innovations,
                )
            )
        return children


def make_graph_seed_population(
    innovations: InnovationRegistry,
) -> Any:
    """Return a ``(size, rng) -> list[GraphGenome]`` seeder sharing *innovations*."""
    def seed(size: int, rng: random.Random) -> list[GraphGenome]:
        return [
            GraphGenome.seed(
                f"g0_{idx}", innovations=innovations, rng=rng, randomise=idx != 0,
            )
            for idx in range(size)
        ]

    return seed


def _select_parent(evaluated: list[Any], rng: random.Random) -> Any:
    min_fit = min(item.adjusted_fitness for item in evaluated)
    weights = [item.adjusted_fitness - min_fit + 1e-9 for item in evaluated]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(evaluated)
    pick = rng.random() * total
    acc = 0.0
    for item, weight in zip(evaluated, weights, strict=True):
        acc += weight
        if acc >= pick:
            return item
    return evaluated[-1]
