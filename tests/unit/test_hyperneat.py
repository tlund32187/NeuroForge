"""Tests for HyperNEAT-CPPN neuroevolution (Track E): spatial receptive fields.

Covers the CPPN evaluator (deterministic + vectorised), structure-inventing
mutation, NEAT distance/crossover/serialization, the headline property that
connectivity is a *function of substrate geometry* (localized receptive fields),
the Dale's-law non-negative-magnitude invariant the R-STDP loop needs, that an
evolved CPPN *compiles to a working spiking policy*, that the genome-agnostic
engine selects over CPPN structure, end-to-end scoring through the game-training
backend, codec dispatch, and checkpoint resume.
"""

from __future__ import annotations

import random

import pytest

from neuroforge.applications.tasks.evolution import EvolutionTask
from neuroforge.contracts.applications.evolution import FitnessResult
from neuroforge.neuroevolution import (
    CallableFitnessEvaluator,
    EvolutionConfig,
    HyperNEATGenome,
    HyperNEATReproduction,
    InnovationRegistry,
    SubstrateConfig,
    make_hyperneat_seed_population,
)
from neuroforge.neuroevolution.genomes.cppn import CPPN, CPPNConn, CPPNNode


def _seed(
    reg: InnovationRegistry,
    *,
    randomise: bool = False,
    substrate: SubstrateConfig | None = None,
) -> HyperNEATGenome:
    return HyperNEATGenome.seed(
        "g0",
        innovations=reg,
        generation=0,
        randomise=randomise,
        substrate=substrate or SubstrateConfig(),
    )


@pytest.mark.unit
def test_seed_genome_is_a_fully_connected_perceptron_cppn() -> None:
    sub = SubstrateConfig()
    genome = _seed(InnovationRegistry(), substrate=sub)
    assert len(genome.cppn.inputs) == sub.query_dim()
    assert genome.cppn.outputs == ("weight", "expression")  # LEO on by default
    # minimal CPPN: every input wired to every output, no hidden nodes yet.
    assert len(genome.cppn.hidden_nodes()) == 0
    assert len(genome.cppn.connections) == sub.query_dim() * len(genome.cppn.outputs)


@pytest.mark.unit
def test_cppn_query_is_deterministic_and_vectorised() -> None:
    torch = pytest.importorskip("torch")
    genome = _seed(InnovationRegistry(), randomise=True)
    coords = torch.randn(7, genome.substrate.query_dim())

    first = genome.cppn.query(coords, torch=torch)
    second = genome.cppn.query(coords, torch=torch)
    assert torch.allclose(first, second)  # deterministic

    # Batched query equals row-by-row query (vectorisation is correct).
    rows = torch.cat(
        [genome.cppn.query(coords[i : i + 1], torch=torch) for i in range(coords.shape[0])],
        dim=0,
    )
    assert torch.allclose(first, rows, atol=1e-6)


@pytest.mark.unit
def test_mutation_invents_cppn_structure() -> None:
    reg = InnovationRegistry()
    genome = _seed(reg)
    rng = random.Random(0)
    max_nodes = len(genome.cppn.nodes)
    max_conns = len(genome.cppn.connections)
    current = genome
    for i in range(60):
        current = current.mutate(
            child_id=f"c{i}", generation=1, rng=rng, rate=1.0, power=1.0, innovations=reg,
        )
        max_nodes = max(max_nodes, len(current.cppn.nodes))
        max_conns = max(max_conns, len(current.cppn.connections))
    assert max_nodes > len(genome.cppn.nodes)        # add-node grew the CPPN
    assert max_conns > len(genome.cppn.connections)  # add-connection grew the CPPN


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

    payload = child.to_dict()
    assert payload["type"] == "hyperneat"
    restored = HyperNEATGenome.from_dict(payload)
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
def test_decode_genome_dispatches_hyperneat_and_finds_innovations() -> None:
    from neuroforge.neuroevolution import decode_genome, max_connection_innovation

    genome = _seed(InnovationRegistry())
    assert isinstance(decode_genome(genome.to_dict()), HyperNEATGenome)
    # CPPN edges live under cppn.connections; the resume helper must still see them.
    assert max_connection_innovation([genome.to_dict()]) >= 1


@pytest.mark.unit
def test_spatial_receptive_field_is_localized() -> None:
    """A CPPN that is a gaussian of pre↔post distance must paint *local* weights.

    This is the defining HyperNEAT property: connectivity is a function of geometry.
    We hand-build ``weight = gauss(k · distance)`` and confirm the compiled
    input→hidden magnitudes are far larger for nearby neuron pairs than distant ones.
    """
    torch = pytest.importorskip("torch")
    from neuroforge.agents.brains.hyperneat_network import (
        build_hyperneat_policy_network,
    )
    from neuroforge.neuroevolution.genomes.substrate import Substrate

    sub = SubstrateConfig(
        input_shape=(1, 6, 6), hidden_shape=(4, 4), leo=False, weight_threshold=0.2,
    )
    dist_channel = sub.query_dim() - 1  # distance is the last query feature
    inputs = tuple(f"in{i}" for i in range(sub.query_dim()))
    cppn = CPPN(
        inputs=inputs,
        outputs=("weight",),
        nodes=(
            *(CPPNNode(name, "input", "identity") for name in inputs),
            CPPNNode("weight", "output", "gauss"),
        ),
        connections=(CPPNConn(0, inputs[dist_channel], "weight", weight=4.0),),
    )
    base = _seed(InnovationRegistry(), substrate=sub)
    import dataclasses

    genome = dataclasses.replace(base, cppn=cppn)

    net = build_hyperneat_policy_network(genome, n_input=sub.input_count(), seed=1)
    weights = net.engine.projections["input__to__hidden"].topology.weights.detach()

    # Recover the per-edge pre↔post distances in the same (pre-major) edge order.
    substrate = Substrate(
        sub, n_input=sub.input_count(), n_buttons=8, motor_per_button=4,
        torch=torch, dev=torch.device("cpu"), dtype=torch.float32,
    )
    distances = substrate.query_features("input", "hidden")[:, dist_channel]
    order = torch.argsort(distances)
    cut = max(1, distances.numel() // 5)            # nearest vs farthest quintile
    near_mean = weights[order[:cut]].mean().item()
    far_mean = weights[order[-cut:]].mean().item()
    assert near_mean > far_mean                      # weights concentrate locally
    assert far_mean == pytest.approx(0.0, abs=1e-6)  # distant links pruned away


@pytest.mark.unit
def test_compiled_weights_obey_dale_non_negative_invariant() -> None:
    pytest.importorskip("torch")
    from neuroforge.agents.brains.hyperneat_network import (
        build_hyperneat_policy_network,
    )

    sub = SubstrateConfig(input_shape=(1, 5, 5), hidden_shape=(3, 3))
    genome = _seed(InnovationRegistry(), randomise=True, substrate=sub)
    net = build_hyperneat_policy_network(genome, n_input=sub.input_count(), seed=7)
    for projection in net.engine.projections.values():
        assert float(projection.topology.weights.min()) >= 0.0  # |w| magnitudes only


@pytest.mark.unit
def test_cppn_compiles_to_a_working_spiking_policy() -> None:
    torch = pytest.importorskip("torch")
    from neuroforge.agents.brains.hyperneat_network import (
        build_hyperneat_policy_network,
    )
    from neuroforge.agents.brains.stateful_engine import CoreEnginePolicyEngine

    sub = SubstrateConfig(input_shape=(1, 4, 4), hidden_shape=(3, 3))
    reg = InnovationRegistry()
    genome = _seed(reg, randomise=True, substrate=sub)
    rng = random.Random(5)
    for i in range(15):  # grow the CPPN, then confirm it still compiles + runs
        genome = genome.mutate(
            child_id=f"g{i}", generation=1, rng=rng, rate=1.0, innovations=reg,
        )

    net = build_hyperneat_policy_network(genome, n_input=16, seed=1)
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
def test_encoder_agnostic_fallback_grid() -> None:
    # A non-matching n_input (default substrate expects 1792) falls back to a
    # near-square single-channel grid, so HyperNEAT compiles for any encoder.
    sub = SubstrateConfig()
    assert sub.resolve_input_grid(100) == (1, 10, 10)
    assert sub.resolve_input_grid(sub.input_count()) == sub.input_shape


@pytest.mark.unit
def test_engine_selects_for_cppn_structure() -> None:
    reg = InnovationRegistry()

    def objective(genome: HyperNEATGenome) -> FitnessResult:
        return FitnessResult(fitness=float(len(genome.cppn.hidden_nodes())))

    cfg = EvolutionConfig(
        population_size=10, generations=15, elite_count=2, mutation_rate=0.9, seed=7,
    )
    result = EvolutionTask(
        cfg,
        evaluator=CallableFitnessEvaluator(objective),
        reproduction=HyperNEATReproduction(cfg, reg),
        seed_population=make_hyperneat_seed_population(reg),
    ).run()

    assert result.best_genome is not None
    assert result.best_fitness >= 2.0  # structure search grew hidden CPPN nodes


@pytest.mark.unit
def test_hyperneat_genome_runs_through_the_game_training_backend() -> None:
    pytest.importorskip("torch")
    from neuroforge.applications.tasks.game_training import GameTrainingConfig
    from neuroforge.environments.games.clients.scripted import ScriptedGameClient
    from neuroforge.neuroevolution.fitness.evaluators import GameTrainingFitnessEvaluator
    from neuroforge.perception.vision.encoding.frame_preprocess import FramePreprocessConfig

    evaluator = GameTrainingFitnessEvaluator(
        client_factory=lambda: ScriptedGameClient(
            width=12, height=10, channels=1, max_steps=4,
        ),
        base_config=GameTrainingConfig(
            preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
            max_episodes=1, frames_per_episode=4, telemetry_every=0,
        ),
    )
    genome = _seed(InnovationRegistry())  # default substrate; encoder-agnostic fallback

    result = evaluator.evaluate(genome)
    assert result.frames == 4


@pytest.mark.unit
def test_hyperneat_evolution_resumes_from_checkpoint(tmp_path: object) -> None:
    import json

    from neuroforge.neuroevolution import max_connection_innovation

    def objective(genome: HyperNEATGenome) -> float:
        return float(len(genome.cppn.hidden_nodes()))

    checkpoint = tmp_path / "evo.json"  # type: ignore[operator]

    reg1 = InnovationRegistry()
    cfg1 = EvolutionConfig(
        population_size=8, generations=2, elite_count=1, mutation_rate=0.9, seed=3,
        checkpoint_path=str(checkpoint),
    )
    EvolutionTask(
        cfg1,
        evaluator=CallableFitnessEvaluator(objective),
        reproduction=HyperNEATReproduction(cfg1, reg1),
        seed_population=make_hyperneat_seed_population(reg1),
    ).run()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    assert payload["population"][0]["type"] == "hyperneat"

    resume_payloads = [*payload["population"], payload["best"]["genome"]]
    reg2 = InnovationRegistry(start=max_connection_innovation(resume_payloads) + 1)
    cfg2 = EvolutionConfig(
        population_size=8, generations=4, elite_count=1, mutation_rate=0.9, seed=3,
        checkpoint_path=str(checkpoint), resume=True,
    )
    result = EvolutionTask(
        cfg2,
        evaluator=CallableFitnessEvaluator(objective),
        reproduction=HyperNEATReproduction(cfg2, reg2),
        seed_population=make_hyperneat_seed_population(reg2),
    ).run()

    assert result.generations == 2  # gens 2 and 3 of a 4-generation target
    assert isinstance(result.best_genome, HyperNEATGenome)  # decoded back as HyperNEAT
