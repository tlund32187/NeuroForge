"""HyperNEAT genome built around a CPPN and substrate geometry."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from neuroforge.neuroevolution.genomes.cppn import (
    CPPN,
    CPPNConn,
    CPPNNode,
    activation_for_index,
)
from neuroforge.neuroevolution.genomes.graph import graph_hyperparam_defs
from neuroforge.neuroevolution.genomes.policy import Gene
from neuroforge.neuroevolution.genomes.substrate import (
    DEFAULT_SUBSTRATE,
    InputChannelLayout,
    SubstrateConfig,
)

if TYPE_CHECKING:
    from neuroforge.neuroevolution.genomes.innovations import InnovationRegistry

__all__ = [
    "HyperNEATGenome",
    "HyperNEATReproduction",
    "make_hyperneat_seed_population",
]

_HYPER_DEFS = graph_hyperparam_defs()
_HYPER_BY_KEY = {gene.key: gene for gene in _HYPER_DEFS}


def _as_int(value: object) -> int | None:
    try:
        return int(value) if isinstance(value, bool | int | float | str) else None
    except ValueError:
        return None


def _as_float(value: object) -> float | None:
    try:
        return float(value) if isinstance(value, bool | int | float | str) else None
    except ValueError:
        return None


def _int_tuple2(value: object, default: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return default
    items = cast("Sequence[object]", value)
    if len(items) != 2:
        return default
    first = _as_int(items[0])
    second = _as_int(items[1])
    if first is None or second is None:
        return default
    return (first, second)


def _int_tuple3(value: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return default
    items = cast("Sequence[object]", value)
    if len(items) != 3:
        return default
    first = _as_int(items[0])
    second = _as_int(items[1])
    third = _as_int(items[2])
    if first is None or second is None or third is None:
        return default
    return (first, second, third)


def _input_layout_from_payload(value: object) -> tuple[InputChannelLayout, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    layout: list[InputChannelLayout] = []
    for idx, item in enumerate(cast("Sequence[object]", value)):
        if not isinstance(item, Mapping):
            continue
        payload = cast("Mapping[str, object]", item)
        height = _as_int(payload.get("height"))
        width = _as_int(payload.get("width"))
        channel = _as_float(payload.get("channel"))
        kind = _as_float(payload.get("kind"))
        if height is None or width is None or channel is None or kind is None:
            continue
        layout.append(
            InputChannelLayout(
                name=str(payload.get("name", f"input_{idx}")),
                height=height,
                width=width,
                channel=channel,
                kind=kind,
            ),
        )
    return tuple(layout)


def _substrate_from_payload(value: object) -> SubstrateConfig:
    if not isinstance(value, Mapping):
        return DEFAULT_SUBSTRATE
    payload = cast("Mapping[str, object]", value)
    leo_value = payload.get("leo", DEFAULT_SUBSTRATE.leo)
    threshold_value = payload.get("weight_threshold", DEFAULT_SUBSTRATE.weight_threshold)
    return SubstrateConfig(
        input_shape=_int_tuple3(payload.get("input_shape"), DEFAULT_SUBSTRATE.input_shape),
        input_layout=_input_layout_from_payload(payload.get("input_layout", ())),
        hidden_shape=_int_tuple2(payload.get("hidden_shape"), DEFAULT_SUBSTRATE.hidden_shape),
        leo=leo_value if isinstance(leo_value, bool) else DEFAULT_SUBSTRATE.leo,
        weight_threshold=(
            float(threshold_value)
            if isinstance(threshold_value, int | float)
            else DEFAULT_SUBSTRATE.weight_threshold
        ),
    )


@dataclass(frozen=True, slots=True)
class HyperNEATGenome:
    """CPPN genotype for HyperNEAT policy-network generation."""

    id: str
    generation: int
    cppn: CPPN
    hyperparams: tuple[Gene, ...]
    substrate: SubstrateConfig = DEFAULT_SUBSTRATE
    parent_ids: tuple[str, ...] = ()
    learned_checkpoint_path: str = ""

    @classmethod
    def seed(
        cls,
        genome_id: str,
        *,
        innovations: InnovationRegistry,
        generation: int = 0,
        rng: random.Random | None = None,
        randomise: bool = True,
        substrate: SubstrateConfig | None = None,
    ) -> HyperNEATGenome:
        """Return a minimal fully-connected perceptron CPPN."""
        local_rng = rng or random.Random()
        sub = substrate or DEFAULT_SUBSTRATE
        inputs = tuple(f"in{i}" for i in range(sub.query_dim()))
        outputs = ("weight", "expression") if sub.leo else ("weight",)
        nodes = (
            *(CPPNNode(name, "input", "identity") for name in inputs),
            *(CPPNNode(name, "output", "tanh") for name in outputs),
        )
        connections = tuple(
            CPPNConn(
                innovations.connection(src, dst),
                src,
                dst,
                weight=local_rng.uniform(-1.0, 1.0) if randomise else 1.0,
            )
            for src in inputs
            for dst in outputs
        )
        hyperparams = tuple(
            Gene(0, d.key, d.random_value(local_rng) if randomise else d.default)
            for d in _HYPER_DEFS
        )
        return cls(
            id=genome_id,
            generation=generation,
            cppn=CPPN(inputs=inputs, outputs=outputs, nodes=nodes, connections=connections),
            hyperparams=hyperparams,
            substrate=sub,
        )

    def value(self, key: str) -> int | float | bool:
        """Return one hyperparameter by key."""
        for gene in self.hyperparams:
            if gene.key == key:
                return gene.value
        msg = f"hyperparameter {key!r} not found"
        raise KeyError(msg)

    def content_key(self) -> str:
        """Stable key over CPPN structure and hyperparameters."""
        hyper_part = ";".join(
            f"{g.key}={g.value!r}" for g in sorted(self.hyperparams, key=lambda g: g.key)
        )
        return f"S[{self.substrate!r}]|{self.cppn.content_key()}|H[{hyper_part}]"

    def as_offspring(
        self,
        *,
        child_id: str,
        generation: int,
        parent_ids: tuple[str, ...] | None = None,
    ) -> HyperNEATGenome:
        """Copy with new lineage but identical structure."""
        return dataclasses.replace(
            self,
            id=child_id,
            generation=generation,
            parent_ids=parent_ids if parent_ids is not None else (self.id,),
        )

    def with_learned_checkpoint(self, path: str) -> HyperNEATGenome:
        """Return this genome with an inherited learned-weight checkpoint path."""
        return dataclasses.replace(self, learned_checkpoint_path=str(path))

    def with_substrate(
        self,
        substrate: SubstrateConfig,
        *,
        innovations: InnovationRegistry,
    ) -> HyperNEATGenome:
        """Return this genome on a new substrate, expanding CPPN inputs if needed."""
        required = substrate.query_dim()
        current = len(self.cppn.inputs)
        if required < current:
            msg = (
                f"cannot migrate HyperNEAT CPPN from {current} inputs to "
                f"{required} query features"
            )
            raise ValueError(msg)
        if required == current:
            return dataclasses.replace(self, substrate=substrate)

        inputs = list(self.cppn.inputs)
        nodes = list(self.cppn.nodes)
        connections = list(self.cppn.connections)
        insert_at = sum(1 for node in nodes if node.kind == "input")
        existing = {node.id for node in nodes}
        for idx in range(current, required):
            name = f"in{idx}"
            if name in existing:
                msg = f"cannot add CPPN input {name!r}; node id already exists"
                raise ValueError(msg)
            inputs.append(name)
            existing.add(name)
            nodes.insert(insert_at, CPPNNode(name, "input", "identity"))
            insert_at += 1
            for output in self.cppn.outputs:
                connections.append(
                    CPPNConn(
                        innovations.connection(name, output),
                        name,
                        output,
                        weight=0.0,
                    ),
                )

        cppn = dataclasses.replace(
            self.cppn,
            inputs=tuple(inputs),
            nodes=tuple(nodes),
            connections=tuple(connections),
        )
        return dataclasses.replace(self, cppn=cppn, substrate=substrate)

    def mutate(
        self,
        *,
        child_id: str,
        generation: int,
        rng: random.Random,
        rate: float,
        power: float = 1.0,
        innovations: InnovationRegistry,
    ) -> HyperNEATGenome:
        """Return a mutated HyperNEAT child."""
        nodes = list(self.cppn.nodes)
        connections = list(self.cppn.connections)
        if rng.random() < rate * 0.5:
            _add_connection(nodes, connections, rng, innovations)
        if rng.random() < rate * 0.35:
            _add_node(nodes, connections, rng, innovations)
        connections = [
            _perturb_connection(conn, rng, rate=rate, power=power) for conn in connections
        ]
        hyperparams = tuple(
            Gene(0, gene.key, _HYPER_BY_KEY[gene.key].mutate(gene.value, rng, power=power))
            if gene.key in _HYPER_BY_KEY and rng.random() < rate
            else gene
            for gene in self.hyperparams
        )
        cppn = dataclasses.replace(self.cppn, nodes=tuple(nodes), connections=tuple(connections))
        return HyperNEATGenome(
            id=child_id,
            generation=generation,
            cppn=cppn,
            hyperparams=hyperparams,
            substrate=self.substrate,
            parent_ids=self.parent_ids if self.parent_ids else (self.id,),
            learned_checkpoint_path=self.learned_checkpoint_path,
        )

    def crossover(
        self,
        other: HyperNEATGenome,
        *,
        child_id: str,
        generation: int,
        rng: random.Random,
    ) -> HyperNEATGenome:
        """NEAT-style aligned crossover over CPPN connection innovations."""
        mine = {conn.innovation: conn for conn in self.cppn.connections}
        theirs = {conn.innovation: conn for conn in other.cppn.connections}
        child_connections = tuple(
            _inherit_connection(i, mine, theirs, rng)
            for i in sorted(set(mine) | set(theirs))
        )
        node_by_id = {node.id: node for node in (*self.cppn.nodes, *other.cppn.nodes)}
        child_nodes = tuple(node_by_id[key] for key in sorted(node_by_id))
        other_hyper = {gene.key: gene for gene in other.hyperparams}
        child_hyper = tuple(
            rng.choice([gene, other_hyper[gene.key]]) if gene.key in other_hyper else gene
            for gene in self.hyperparams
        )
        return HyperNEATGenome(
            id=child_id,
            generation=generation,
            cppn=dataclasses.replace(
                self.cppn,
                nodes=child_nodes,
                connections=child_connections,
            ),
            hyperparams=child_hyper,
            substrate=self.substrate,
            parent_ids=(self.id, other.id),
            learned_checkpoint_path=_inherit_learned_checkpoint(self, other, rng),
        )

    def distance(self, other: HyperNEATGenome) -> float:
        """Compatibility distance over CPPN structure."""
        mine = {conn.innovation: conn for conn in self.cppn.connections}
        theirs = {conn.innovation: conn for conn in other.cppn.connections}
        all_ids = set(mine) | set(theirs)
        if not all_ids:
            return 0.0
        disjoint = 0
        weight = 0.0
        matched = 0
        for innovation in all_ids:
            left = mine.get(innovation)
            right = theirs.get(innovation)
            if left is None or right is None:
                disjoint += 1
            else:
                weight += abs(left.weight - right.weight)
                matched += 1
        hidden_diff = abs(len(self.cppn.hidden_nodes()) - len(other.cppn.hidden_nodes()))
        return disjoint / max(len(mine), len(theirs), 1) + 0.4 * (
            weight / matched if matched else 0.0
        ) + 0.1 * hidden_diff

    def to_game_training_config(self, *, base: Any | None = None, seed: int) -> Any:
        """Apply HyperNEAT hyperparameters onto a game-training config."""
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
        self,
        *,
        seed: int,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> Any:
        """Return ``n_input -> PolicyNetwork`` that compiles this HyperNEAT genome."""
        from neuroforge.agents.brains.hyperneat_network import build_hyperneat_policy_network

        def builder(n_input: int) -> Any:
            return build_hyperneat_policy_network(
                self,
                n_input=n_input,
                seed=seed,
                device=device,
                dtype=dtype,
            )

        return builder

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload: dict[str, Any] = {
            "type": "hyperneat",
            "id": self.id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "cppn": self.cppn.to_dict(),
            "hyperparams": [dataclasses.asdict(gene) for gene in self.hyperparams],
            "substrate": dataclasses.asdict(self.substrate),
        }
        if self.learned_checkpoint_path:
            payload["learned_checkpoint_path"] = self.learned_checkpoint_path
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HyperNEATGenome:
        """Decode a genome from :meth:`to_dict` output."""
        hyper_by_key = {
            str(item["key"]): item["value"] for item in payload.get("hyperparams", [])
        }
        substrate = _substrate_from_payload(payload.get("substrate"))
        parents_payload: object = payload.get("parent_ids", [])
        parents = (
            cast("Sequence[object]", parents_payload)
            if isinstance(parents_payload, list)
            else ()
        )
        return cls(
            id=str(payload["id"]),
            generation=int(payload["generation"]),
            parent_ids=tuple(str(pid) for pid in parents),
            cppn=CPPN.from_dict(payload["cppn"]),
            hyperparams=tuple(
                Gene(0, d.key, hyper_by_key.get(d.key, d.default)) for d in _HYPER_DEFS
            ),
            substrate=substrate,
            learned_checkpoint_path=_learned_checkpoint_path_from_payload(payload),
        )


class HyperNEATReproduction:
    """Reproduction operator for :class:`HyperNEATGenome`."""

    def __init__(self, config: Any, innovations: InnovationRegistry) -> None:
        self._cfg = config
        self._innovations = innovations

    def next_generation(
        self,
        evaluated: list[Any],
        *,
        generation: int,
        rng: random.Random,
    ) -> list[Any]:
        """Create the next HyperNEAT population."""
        ranked = sorted(evaluated, key=lambda item: item.adjusted_fitness, reverse=True)
        next_population: list[HyperNEATGenome] = [
            item.genome.as_offspring(child_id=f"g{generation}_elite{i}", generation=generation)
            for i, item in enumerate(ranked[: self._cfg.elite_count])
        ]
        while len(next_population) < self._cfg.population_size:
            parent_a = _select(ranked, rng, self._cfg).genome
            child = parent_a
            parent_ids: tuple[str, ...] = (parent_a.id,)
            if rng.random() < self._cfg.crossover_rate and len(ranked) > 1:
                parent_b = _select_distinct(
                    ranked,
                    rng,
                    exclude_genome_id=str(parent_a.id),
                    config=self._cfg,
                ).genome
                child = parent_a.crossover(
                    parent_b,
                    child_id=f"g{generation}_{len(next_population)}",
                    generation=generation,
                    rng=rng,
                )
                parent_ids = (parent_a.id, parent_b.id)
            child = child.mutate(
                child_id=f"g{generation}_{len(next_population)}",
                generation=generation,
                rng=rng,
                rate=self._cfg.mutation_rate,
                power=self._cfg.mutation_power,
                innovations=self._innovations,
            )
            child = dataclasses.replace(child, parent_ids=parent_ids)
            next_population.append(child)
        return next_population

    def reproduce(
        self,
        evaluated: list[Any],
        *,
        generation: int,
        rng: random.Random,
    ) -> list[Any]:
        """Compatibility alias for callers that use a reproduction verb."""
        return self.next_generation(evaluated, generation=generation, rng=rng)


def make_hyperneat_seed_population(
    innovations: InnovationRegistry,
    *,
    substrate: SubstrateConfig | None = None,
) -> Any:
    """Return a seed-population factory for :class:`EvolutionEngine`."""

    def seed(size: int, rng: random.Random) -> list[HyperNEATGenome]:
        return [
            HyperNEATGenome.seed(
                f"g0_{idx}",
                innovations=innovations,
                rng=rng,
                randomise=idx != 0,
                substrate=substrate,
            )
            for idx in range(size)
        ]

    return seed


def _add_connection(
    nodes: list[CPPNNode],
    connections: list[CPPNConn],
    rng: random.Random,
    innovations: InnovationRegistry,
) -> None:
    existing = {(conn.src, conn.dst) for conn in connections}
    ordered = {node.id: idx for idx, node in enumerate(nodes)}
    sources = [node for node in nodes if node.kind != "output"]
    targets = [node for node in nodes if node.kind != "input"]
    rng.shuffle(sources)
    rng.shuffle(targets)
    for src in sources:
        for dst in targets:
            if ordered[src.id] >= ordered[dst.id] or (src.id, dst.id) in existing:
                continue
            connections.append(
                CPPNConn(
                    innovations.connection(src.id, dst.id),
                    src.id,
                    dst.id,
                    weight=rng.uniform(-1.0, 1.0),
                )
            )
            return


def _add_node(
    nodes: list[CPPNNode],
    connections: list[CPPNConn],
    rng: random.Random,
    innovations: InnovationRegistry,
) -> None:
    enabled = [idx for idx, conn in enumerate(connections) if conn.enabled]
    if not enabled:
        return
    idx = rng.choice(enabled)
    conn = connections[idx]
    node_id, in_innovation, out_innovation = innovations.node_split(conn.innovation)
    new_id = f"h{node_id}"
    connections[idx] = dataclasses.replace(conn, enabled=False)
    dst_index = next((i for i, node in enumerate(nodes) if node.id == conn.dst), len(nodes))
    nodes.insert(dst_index, CPPNNode(new_id, "hidden", activation_for_index(node_id)))
    connections.append(CPPNConn(in_innovation, conn.src, new_id, weight=1.0))
    connections.append(CPPNConn(out_innovation, new_id, conn.dst, weight=conn.weight))


def _perturb_connection(
    conn: CPPNConn,
    rng: random.Random,
    *,
    rate: float,
    power: float,
) -> CPPNConn:
    if rng.random() >= rate:
        return conn
    enabled = (not conn.enabled) if rng.random() < rate * 0.1 else conn.enabled
    return dataclasses.replace(
        conn,
        weight=max(-5.0, min(5.0, conn.weight + rng.gauss(0.0, power))),
        enabled=enabled,
    )


def _select(ranked: list[Any], rng: random.Random, config: Any | None = None) -> Any:
    if config is not None and hasattr(config, "selection_mode"):
        from neuroforge.neuroevolution.search.engine import select_parent_by_mode

        return select_parent_by_mode(
            ranked,
            rng,
            mode=str(config.selection_mode),
            tournament_size=int(config.tournament_size),
            rank_pressure=float(config.rank_selection_pressure),
        )
    min_fit = min(item.adjusted_fitness for item in ranked)
    weights = [item.adjusted_fitness - min_fit + 1e-9 for item in ranked]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(ranked)
    pick = rng.random() * total
    accum = 0.0
    for item, weight in zip(ranked, weights, strict=True):
        accum += weight
        if accum >= pick:
            return item
    return ranked[-1]


def _select_distinct(
    ranked: list[Any],
    rng: random.Random,
    *,
    exclude_genome_id: str,
    config: Any | None = None,
) -> Any:
    for _ in range(8):
        candidate = _select(ranked, rng, config)
        if str(candidate.genome.id) != exclude_genome_id:
            return candidate
    for candidate in ranked:
        if str(candidate.genome.id) != exclude_genome_id:
            return candidate
    msg = "distinct crossover parent not found"
    raise RuntimeError(msg)


def _inherit_connection(
    innovation: int,
    mine: dict[int, CPPNConn],
    theirs: dict[int, CPPNConn],
    rng: random.Random,
) -> CPPNConn:
    left = mine.get(innovation)
    right = theirs.get(innovation)
    if left is not None and right is not None:
        return rng.choice([left, right])
    if left is not None:
        return left
    if right is None:
        msg = f"missing connection innovation {innovation}"
        raise KeyError(msg)
    return right


def _learned_checkpoint_path_from_payload(payload: dict[str, Any]) -> str:
    """Return the optional learned-policy checkpoint path from a genome payload."""
    raw = payload.get("learned_checkpoint_path", "")
    if isinstance(raw, str):
        return raw
    state = payload.get("learned_state")
    if isinstance(state, Mapping):
        path = cast("Mapping[str, object]", state).get("policy_checkpoint", "")
        return path if isinstance(path, str) else ""
    return ""


def _inherit_learned_checkpoint(
    left: HyperNEATGenome,
    right: HyperNEATGenome,
    rng: random.Random,
) -> str:
    """Choose one available learned checkpoint during crossover inheritance."""
    paths = tuple(
        path
        for path in (left.learned_checkpoint_path, right.learned_checkpoint_path)
        if path
    )
    if not paths:
        return ""
    return rng.choice(paths)
