"""Tests for structural neuroevolution (Track E #5): the GraphGenome.

Covers innovation stability, structure-inventing mutations (add-node /
add-connection), NEAT distance + crossover, serialization, that an evolved graph
*compiles to a working spiking policy*, and that the genome-agnostic engine
actually selects for structure.
"""

from __future__ import annotations

import random

import pytest

from neuroforge.contracts.evolution import FitnessResult
from neuroforge.evolution import (
    CallableFitnessEvaluator,
    EvolutionConfig,
    GraphGenome,
    GraphReproduction,
    InnovationRegistry,
    make_graph_seed_population,
)
from neuroforge.tasks.evolution import EvolutionTask


def _seed(reg: InnovationRegistry, *, randomise: bool = False) -> GraphGenome:
    return GraphGenome.seed("g0", innovations=reg, generation=0, randomise=randomise)


@pytest.mark.unit
def test_innovation_registry_is_stable() -> None:
    reg = InnovationRegistry()
    a = reg.connection("input", "h0")
    b = reg.connection("input", "h0")
    c = reg.connection("h0", "motor")
    assert a == b           # same edge -> same innovation
    assert c != a
    split_one = reg.node_split(a)
    split_two = reg.node_split(a)
    assert split_one == split_two   # splitting the same connection is stable


@pytest.mark.unit
def test_seed_genome_has_io_and_one_hidden() -> None:
    genome = _seed(InnovationRegistry())
    roles = sorted(node.role for node in genome.nodes)
    assert roles == ["hidden", "input", "motor"]
    assert len(genome.enabled_connections()) == 2


@pytest.mark.unit
def test_mutation_invents_structure() -> None:
    reg = InnovationRegistry()
    genome = _seed(reg)
    rng = random.Random(0)
    max_nodes = len(genome.nodes)
    max_conns = len(genome.connections)
    current = genome
    for i in range(60):
        current = current.mutate(
            child_id=f"c{i}", generation=1, rng=rng, rate=1.0, power=1.0, innovations=reg,
        )
        max_nodes = max(max_nodes, len(current.nodes))
        max_conns = max(max_conns, len(current.connections))
    assert max_nodes > len(genome.nodes)        # add-node grew the graph
    assert max_conns > len(genome.connections)  # add-connection grew the graph


@pytest.mark.unit
def test_distance_zero_for_twins_and_positive_after_growth() -> None:
    reg = InnovationRegistry()
    genome = _seed(reg)
    twin = genome.as_offspring(child_id="twin", generation=0)
    assert genome.distance(twin) == pytest.approx(0.0)

    rng = random.Random(1)
    grown = genome
    for i in range(40):
        grown = grown.mutate(
            child_id=f"m{i}", generation=1, rng=rng, rate=1.0, innovations=reg,
        )
    assert genome.distance(grown) > 0.0


@pytest.mark.unit
def test_crossover_and_serialization_round_trip() -> None:
    reg = InnovationRegistry()
    parent = _seed(reg)
    rng = random.Random(2)
    other = parent.mutate(child_id="b", generation=1, rng=rng, rate=1.0, innovations=reg)
    child = parent.crossover(other, child_id="c", generation=2, rng=random.Random(3))
    assert child.parent_ids == ("g0", "b")

    restored = GraphGenome.from_dict(child.to_dict())
    assert restored.content_key() == child.content_key()
    assert restored.id == child.id


@pytest.mark.unit
def test_content_key_ignores_id_but_tracks_structure() -> None:
    reg = InnovationRegistry()
    genome = _seed(reg)
    assert genome.content_key() == genome.as_offspring(
        child_id="x", generation=9,
    ).content_key()
    grown = genome.mutate(
        child_id="y", generation=1, rng=random.Random(4), rate=1.0, innovations=reg,
    )
    assert grown.content_key() != genome.content_key()


@pytest.mark.unit
def test_graph_compiles_to_a_working_spiking_policy() -> None:
    torch = pytest.importorskip("torch")
    from neuroforge.game.policies.graph_network import build_graph_policy_network
    from neuroforge.game.policies.stateful_engine import CoreEnginePolicyEngine

    reg = InnovationRegistry()
    genome = _seed(reg)
    # Grow the structure, then confirm the bigger graph still compiles + runs.
    rng = random.Random(5)
    for i in range(20):
        genome = genome.mutate(
            child_id=f"g{i}", generation=1, rng=rng, rate=1.0, innovations=reg,
        )

    net = build_graph_policy_network(genome, n_input=16, seed=1)
    engine = CoreEnginePolicyEngine(
        net.engine,
        motor_pop=net.motor_pop,
        motor_per_button=net.motor_per_button,
        n_buttons=net.n_buttons,
        input_pop=net.input_pop,
    )
    decision = engine.decide(torch.ones(16) * 10.0, ticks=4)
    assert tuple(decision.motor_rates.shape) == (8,)


@pytest.mark.unit
def test_engine_selects_for_structure() -> None:
    reg = InnovationRegistry()

    def objective(genome: GraphGenome) -> FitnessResult:
        # Reward inventing hidden nodes (a stand-in for "more capable structure").
        return FitnessResult(fitness=float(len(genome.hidden_nodes())))

    cfg = EvolutionConfig(
        population_size=10, generations=12, elite_count=2, mutation_rate=0.9, seed=7,
    )
    result = EvolutionTask(
        cfg,
        evaluator=CallableFitnessEvaluator(objective),
        reproduction=GraphReproduction(cfg, reg),
        seed_population=make_graph_seed_population(reg),
    ).run()

    assert result.best_genome is not None
    # Started from a single hidden node; structure search must grow beyond it.
    assert result.best_fitness >= 2.0


@pytest.mark.unit
def test_graph_genome_runs_through_the_game_training_backend() -> None:
    pytest.importorskip("torch")
    from neuroforge.evolution.evaluators import GameTrainingFitnessEvaluator
    from neuroforge.game.clients.scripted import ScriptedGameClient
    from neuroforge.game.policies.preprocess import FramePreprocessConfig
    from neuroforge.tasks.game_training import GameTrainingConfig

    evaluator = GameTrainingFitnessEvaluator(
        client_factory=lambda: ScriptedGameClient(
            width=12, height=10, channels=1, max_steps=4,
        ),
        base_config=GameTrainingConfig(
            preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
            max_episodes=1, frames_per_episode=4, telemetry_every=0,
        ),
    )
    genome = _seed(InnovationRegistry())

    # The evaluator must accept the graph genome, compile its invented topology
    # via make_network_builder, and run the policy through the training loop.
    result = evaluator.evaluate(genome)
    assert result.frames == 4


@pytest.mark.unit
def test_decode_genome_dispatches_by_type() -> None:
    from neuroforge.evolution import decode_genome, max_connection_innovation
    from neuroforge.evolution.genome import PolicyGenome

    graph = _seed(InnovationRegistry())
    assert isinstance(decode_genome(graph.to_dict()), GraphGenome)

    policy = PolicyGenome.seed("p", generation=0, randomise=False)
    assert type(decode_genome(policy.to_dict())).__name__ == "PolicyGenome"

    legacy = policy.to_dict()
    legacy.pop("type", None)  # pre-tag checkpoints decode as PolicyGenome
    assert type(decode_genome(legacy)).__name__ == "PolicyGenome"

    assert max_connection_innovation([graph.to_dict()]) >= 1


@pytest.mark.unit
def test_graph_evolution_resumes_from_checkpoint(tmp_path: object) -> None:
    import json

    from neuroforge.evolution import max_connection_innovation

    def objective(genome: GraphGenome) -> float:
        return float(len(genome.hidden_nodes()))

    checkpoint = tmp_path / "evo.json"  # type: ignore[operator]

    reg1 = InnovationRegistry()
    cfg1 = EvolutionConfig(
        population_size=8, generations=2, elite_count=1, mutation_rate=0.9, seed=3,
        checkpoint_path=str(checkpoint),
    )
    EvolutionTask(
        cfg1,
        evaluator=CallableFitnessEvaluator(objective),
        reproduction=GraphReproduction(cfg1, reg1),
        seed_population=make_graph_seed_population(reg1),
    ).run()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    assert payload["population"][0]["type"] == "graph"  # graph genomes were checkpointed

    # Resume with a registry continued past the checkpoint's innovations.
    resume_payloads = [*payload["population"], payload["best"]["genome"]]
    reg2 = InnovationRegistry(start=max_connection_innovation(resume_payloads) + 1)
    cfg2 = EvolutionConfig(
        population_size=8, generations=4, elite_count=1, mutation_rate=0.9, seed=3,
        checkpoint_path=str(checkpoint), resume=True,
    )
    result = EvolutionTask(
        cfg2,
        evaluator=CallableFitnessEvaluator(objective),
        reproduction=GraphReproduction(cfg2, reg2),
        seed_population=make_graph_seed_population(reg2),
    ).run()

    assert result.generations == 2  # gens 2 and 3 of a 4-generation target
    assert isinstance(result.best_genome, GraphGenome)  # decoded back as a graph
