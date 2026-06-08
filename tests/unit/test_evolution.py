"""Unit tests for the first neuroevolution slice."""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from typing import TYPE_CHECKING, Any

import pytest

import neuroforge.interfaces.cli.main as cli
from neuroforge.applications.smb3.fitness import (
    ScriptedGameFitnessConfig,
    build_scripted_progress_fitness_evaluator,
)
from neuroforge.applications.tasks.evolution import EvolutionTask
from neuroforge.applications.tasks.game_training import GameTrainingConfig
from neuroforge.contracts.applications.evolution import FitnessResult
from neuroforge.contracts.messaging import EventTopic, MonitorEvent
from neuroforge.messaging.bus import EventBus
from neuroforge.neuroevolution import (
    AdaptiveSpeciation,
    BestGenomeCheckpoint,
    CallableFitnessEvaluator,
    EvaluatedGenome,
    EvolutionConfig,
    Gene,
    PolicyGenome,
    SpeciesAwareReproduction,
    ThreadLocalFitnessEvaluatorPool,
    attach_learned_checkpoint_to_evolution_checkpoint,
    decode_genome,
    find_latest_evolution_checkpoint,
    get_policy_objective,
    load_best_genome_checkpoint,
    policy_objective_names,
    select_parent_by_mode,
)

if TYPE_CHECKING:
    from pathlib import Path


class _Collector:
    enabled = True

    def __init__(self) -> None:
        self.events: list[MonitorEvent] = []

    def on_event(self, event: MonitorEvent) -> None:
        self.events.append(event)

    def reset(self) -> None:
        self.events.clear()

    def snapshot(self) -> dict[str, object]:
        return {"count": len(self.events)}


@pytest.mark.unit
def test_policy_genome_round_trips_and_builds_configs() -> None:
    genome = PolicyGenome.seed("g0", generation=0, randomise=False)
    decoded = PolicyGenome.from_dict(genome.to_dict())

    assert decoded == genome
    cfg = genome.to_policy_network_config(n_input=32, seed=7)
    assert cfg.n_hidden == 128
    assert cfg.n_input == 32
    assert cfg.n_hidden_layers == 1

    spec = genome.to_network_spec(n_input=32)
    assert spec.metadata["genome_id"] == "g0"
    assert spec.metadata["n_hidden_layers"] == 1
    assert [pop.name for pop in spec.populations] == ["input", "hidden", "motor"]
    assert {proj.name for proj in spec.projections} >= {"in_to_hidden", "hidden_to_motor"}


@pytest.mark.unit
def test_policy_genome_learned_checkpoint_round_trip_does_not_change_content_key() -> None:
    genome = PolicyGenome.seed("g0", generation=0, randomise=False)
    learned = genome.with_learned_checkpoint("artifacts/smb3_policy.pt")
    decoded = PolicyGenome.from_dict(learned.to_dict())

    assert decoded.learned_checkpoint_path == "artifacts/smb3_policy.pt"
    assert decoded.content_key() == genome.content_key()
    assert decoded.as_offspring(child_id="child", generation=1).learned_checkpoint_path == (
        "artifacts/smb3_policy.pt"
    )


@pytest.mark.unit
def test_policy_genome_loads_old_checkpoints_with_new_gene_defaults() -> None:
    payload = PolicyGenome.seed("legacy", generation=0, randomise=False).to_dict()
    payload["genes"] = [
        gene
        for gene in payload["genes"]
        if gene["key"] not in {"n_hidden_layers", "hidden_fanin", "input_to_motor_skip"}
    ]

    decoded = PolicyGenome.from_dict(payload)

    assert decoded.value("n_hidden_layers") == 1
    assert decoded.value("hidden_fanin") == 0
    assert decoded.value("input_to_motor_skip") is False


@pytest.mark.unit
def test_policy_genome_can_apply_checkpoint_compatible_training_genes() -> None:
    genome = PolicyGenome.seed("g0", generation=0, randomise=False).mutate(
        child_id="g1",
        generation=1,
        rng=random.Random(7),
        rate=1.0,
    )
    base = GameTrainingConfig(
        n_hidden=64,
        n_hidden_layers=2,
        motor_per_button=2,
        input_fanin=32,
        hidden_fanin=16,
        input_to_motor_skip=True,
        seed=99,
    )

    cfg = genome.to_game_training_config(
        base=base,
        seed=123,
        include_network_shape=False,
    )

    assert cfg.n_hidden == 64
    assert cfg.n_hidden_layers == 2
    assert cfg.motor_per_button == 2
    assert cfg.input_fanin == 32
    assert cfg.hidden_fanin == 16
    assert cfg.input_to_motor_skip is True
    assert cfg.seed == 123
    assert cfg.decide_ticks == int(genome.value("decide_ticks"))
    assert cfg.rstdp.reward_scale == pytest.approx(float(genome.value("reward_scale")))


@pytest.mark.unit
def test_policy_genome_full_phenotype_applies_architecture_genes() -> None:
    replacements: dict[str, int | bool] = {
        "n_hidden_layers": 3,
        "hidden_fanin": 24,
        "input_to_motor_skip": True,
    }
    genome = PolicyGenome(
        id="deep",
        generation=1,
        genes=tuple(
            Gene(gene.innovation, gene.key, replacements[gene.key])
            if gene.key in replacements
            else gene
            for gene in PolicyGenome.seed("base", randomise=False).genes
        ),
    )
    base = GameTrainingConfig(
        n_hidden_layers=1,
        hidden_fanin=0,
        input_to_motor_skip=False,
    )

    cfg = genome.to_game_training_config(base=base, seed=456, include_network_shape=True)
    spec = genome.to_network_spec(n_input=32)

    assert cfg.n_hidden_layers == 3
    assert cfg.hidden_fanin == 24
    assert cfg.input_to_motor_skip is True
    assert [pop.name for pop in spec.populations] == [
        "input",
        "hidden_0",
        "hidden_1",
        "hidden_2",
        "motor",
    ]
    assert {proj.name for proj in spec.projections} >= {
        "hidden_0_to_hidden_1",
        "hidden_1_to_hidden_2",
        "input_to_motor",
    }


@pytest.mark.unit
def test_load_and_find_best_evolution_checkpoint(tmp_path: Path) -> None:
    genome = PolicyGenome.seed("best", generation=2, randomise=False)
    checkpoint = tmp_path / "run_1" / "evolution" / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "generation": 3,
                "evaluations": 12,
                "population": [genome.to_dict()],
                "best": {
                    "genome": genome.to_dict(),
                    "fitness": 42.5,
                    "metrics": {"max_x_progress": 0.12},
                    "episodes": 1,
                    "frames": 600,
                    "species_id": 0,
                    "adjusted_fitness": 42.5,
                },
            }
        ),
        encoding="utf-8",
    )

    found = find_latest_evolution_checkpoint(tmp_path)
    loaded = load_best_genome_checkpoint(checkpoint)

    assert found == checkpoint
    assert isinstance(loaded, BestGenomeCheckpoint)
    assert loaded.genome == genome
    assert loaded.fitness == pytest.approx(42.5)
    assert loaded.metrics["max_x_progress"] == pytest.approx(0.12)
    assert loaded.generation == 3
    assert loaded.evaluations == 12
    assert loaded.schema_version == 1
    assert loaded.config == {}


@pytest.mark.unit
def test_attach_learned_checkpoint_to_evolution_checkpoint(tmp_path: Path) -> None:
    genome = PolicyGenome.seed("best", generation=2, randomise=False)
    twin = genome.as_offspring(child_id="twin", generation=3)
    other = genome.mutate(child_id="other", generation=3, rng=random.Random(5), rate=1.0)
    checkpoint = tmp_path / "run" / "evolution" / "checkpoint.json"
    learned = tmp_path / "policy.pt"
    learned.write_text("placeholder", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps(
            {
                "generation": 3,
                "evaluations": 12,
                "population": [twin.to_dict(), other.to_dict()],
                "best": {
                    "genome": genome.to_dict(),
                    "fitness": 42.5,
                    "metrics": {},
                    "episodes": 1,
                    "frames": 600,
                    "species_id": 0,
                    "adjusted_fitness": 42.5,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = attach_learned_checkpoint_to_evolution_checkpoint(
        checkpoint,
        learned,
        genome_id="best",
        source="unit",
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    loaded = load_best_genome_checkpoint(checkpoint)

    assert summary["population_updates"] == 1
    assert payload["best"]["lamarckian"]["source"] == "unit"
    assert payload["best"]["genome"]["learned_checkpoint_path"] == str(learned)
    assert payload["population"][0]["learned_checkpoint_path"] == str(learned)
    assert "learned_checkpoint_path" not in payload["population"][1]
    assert loaded.learned_checkpoint_path == str(learned)
    assert loaded.genome.learned_checkpoint_path == str(learned)


@pytest.mark.unit
def test_policy_genome_mutation_keeps_bounds_and_changes_value() -> None:
    genome = PolicyGenome.seed("parent", randomise=False)
    import random

    child = genome.mutate(
        child_id="child",
        generation=1,
        rng=random.Random(1),
        rate=1.0,
        power=1.0,
    )

    assert child.id == "child"
    assert child.parent_ids == ("parent",)
    assert child != genome
    assert 32 <= int(child.value("n_hidden")) <= 256
    assert 1e-5 <= float(child.value("lr")) <= 3e-3


@pytest.mark.unit
def test_policy_genome_crossover_uses_parent_gene_values() -> None:
    import random

    a = PolicyGenome.seed("a", randomise=False)
    b = a.mutate(child_id="b", generation=0, rng=random.Random(2), rate=1.0)
    child = a.crossover(b, child_id="c", generation=1, rng=random.Random(3))
    parent_values = {
        key: {a.value(key), b.value(key)}
        for key in ("n_hidden", "motor_per_button", "tau_mem", "noise_amp")
    }

    assert child.parent_ids == ("a", "b")
    for key, values in parent_values.items():
        assert child.value(key) in values


@pytest.mark.unit
def test_genome_content_key_ignores_id_and_generation() -> None:
    base = PolicyGenome.seed("a", generation=0, randomise=False)
    twin = PolicyGenome(id="b", generation=5, genes=base.genes, parent_ids=("x",))
    assert base.content_key() == twin.content_key()  # same genes -> same key
    mutated = base.mutate(child_id="c", generation=1, rng=random.Random(1), rate=1.0)
    assert mutated.content_key() != base.content_key()  # different genes -> different key


@pytest.mark.unit
def test_adaptive_speciation_adjusts_threshold_toward_target_species_count() -> None:
    base = PolicyGenome.seed("base", generation=0, randomise=False)
    twins = [
        base.as_offspring(child_id=f"twin_{idx}", generation=0)
        for idx in range(4)
    ]
    too_few = AdaptiveSpeciation(
        threshold=0.5,
        target_min=2,
        target_max=4,
        adjustment=0.1,
    )

    too_few.assign(twins)

    assert too_few.last_species_count == 1
    assert too_few.threshold == pytest.approx(0.45)

    rng = random.Random(2)
    diverse = [base]
    current = base
    for idx in range(4):
        current = current.mutate(
            child_id=f"diverse_{idx}",
            generation=0,
            rng=rng,
            rate=1.0,
        )
        diverse.append(current)
    too_many = AdaptiveSpeciation(
        threshold=1e-6,
        target_min=1,
        target_max=2,
        adjustment=0.1,
    )

    too_many.assign(diverse)

    assert too_many.last_species_count > 2
    assert too_many.threshold == pytest.approx(1.1e-6)


@pytest.mark.unit
def test_species_aware_reproduction_preserves_weaker_species() -> None:
    strong = PolicyGenome.seed("strong", generation=0, randomise=False)
    weak = strong.mutate(child_id="weak", generation=0, rng=random.Random(3), rate=1.0)
    cfg = EvolutionConfig(
        population_size=6,
        generations=2,
        elite_count=1,
        mutation_rate=0.8,
        seed=9,
    )
    evaluated = [
        EvaluatedGenome(
            strong,
            FitnessResult(100.0),
            species_id=0,
            adjusted_fitness=50.0,
            species_uid=0,
        ),
        EvaluatedGenome(
            strong.mutate(child_id="strong_2", generation=0, rng=random.Random(4), rate=1.0),
            FitnessResult(90.0),
            species_id=0,
            adjusted_fitness=45.0,
            species_uid=0,
        ),
        EvaluatedGenome(
            weak,
            FitnessResult(1.0),
            species_id=1,
            adjusted_fitness=1.0,
            species_uid=1,
        ),
    ]

    children = SpeciesAwareReproduction(cfg).next_generation(
        evaluated,
        generation=1,
        rng=random.Random(5),
    )

    assert len(children) == cfg.population_size
    assert any(child.content_key() == strong.content_key() for child in children)
    assert any(child.content_key() == weak.content_key() for child in children)


@pytest.mark.unit
def test_tournament_selection_prefers_order_over_score_magnitude() -> None:
    base = PolicyGenome.seed("base", generation=0, randomise=False)
    evaluated = [
        EvaluatedGenome(
            base.as_offspring(child_id="low", generation=0),
            FitnessResult(-10_000.0),
            species_id=0,
            adjusted_fitness=-10_000.0,
            species_uid=0,
        ),
        EvaluatedGenome(
            base.as_offspring(child_id="mid", generation=0),
            FitnessResult(0.5),
            species_id=0,
            adjusted_fitness=0.5,
            species_uid=0,
        ),
        EvaluatedGenome(
            base.as_offspring(child_id="high", generation=0),
            FitnessResult(0.6),
            species_id=0,
            adjusted_fitness=0.6,
            species_uid=0,
        ),
    ]

    selected = select_parent_by_mode(
        evaluated,
        random.Random(3),
        mode="tournament",
        tournament_size=len(evaluated),
        rank_pressure=1.7,
    )

    assert selected.genome.id == "high"


@pytest.mark.unit
def test_fitness_cache_avoids_re_evaluating_identical_genomes() -> None:
    evaluated: list[str] = []

    def objective(genome: PolicyGenome) -> float:
        evaluated.append(genome.content_key())
        return float(genome.value("n_hidden"))

    EvolutionTask(
        EvolutionConfig(
            population_size=6, generations=4, elite_count=2, mutation_rate=0.8, seed=11,
        ),
        evaluator=CallableFitnessEvaluator(objective),
    ).run()

    # The cache means each distinct genome is scored once; carried elites are not
    # re-rolled, so total evaluate() calls fall below the naive pop*generations.
    assert len(evaluated) == len(set(evaluated))
    assert len(evaluated) < 6 * 4


@pytest.mark.unit
def test_evolution_task_adds_novelty_selection_metrics() -> None:
    def objective(genome: PolicyGenome) -> FitnessResult:
        return FitnessResult(
            fitness=1.0,
            metrics={"behavior": float(genome.value("n_hidden")) / 256.0},
        )

    bus = EventBus()
    collector = _Collector()
    bus.subscribe(EventTopic.TRAINING_TRIAL, collector)
    bus.subscribe(EventTopic.SCALAR, collector)

    EvolutionTask(
        EvolutionConfig(
            population_size=5,
            generations=1,
            elite_count=1,
            seed=31,
            novelty_weight=10.0,
            novelty_metric_keys=("behavior",),
        ),
        event_bus=bus,
        evaluator=CallableFitnessEvaluator(objective),
    ).run()

    trials = [
        event.data
        for event in collector.events
        if event.topic == EventTopic.TRAINING_TRIAL
    ]
    scalars = [
        event.data
        for event in collector.events
        if event.topic == EventTopic.SCALAR
    ]

    assert trials
    assert any(trial["novelty_score"] > 0.0 for trial in trials)
    assert any(trial["fitness"] > trial["fitness_objective"] for trial in trials)
    assert "best_novelty_score" in scalars[0]


@pytest.mark.unit
def test_evolution_task_improves_toy_objective_and_emits_events(tmp_path: Path) -> None:
    def objective(genome: PolicyGenome) -> FitnessResult:
        hidden = float(genome.value("n_hidden"))
        # Peak near 192 hidden neurons; enough structure for selection to prefer.
        fitness = 100.0 - abs(hidden - 192.0)
        return FitnessResult(fitness=fitness, metrics={"hidden": hidden})

    bus = EventBus()
    collector = _Collector()
    for topic in EventTopic:
        bus.subscribe(topic, collector)

    checkpoint = tmp_path / "evolution.json"
    task = EvolutionTask(
        EvolutionConfig(
            population_size=8,
            generations=4,
            elite_count=2,
            mutation_rate=0.8,
            seed=11,
            checkpoint_path=str(checkpoint),
        ),
        event_bus=bus,
        evaluator=CallableFitnessEvaluator(objective),
    )
    result = task.run()

    assert result.generations == 4
    assert result.evaluations == 32
    assert result.best_genome is not None
    assert result.best_fitness > 0.0
    assert checkpoint.exists()
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["schema_version"] == 2
    assert isinstance(checkpoint_payload["rng_state"], list)
    assert checkpoint_payload["config"] == {
        "population_size": 8,
        "generations": 4,
        "elite_count": 2,
        "mutation_rate": 0.8,
        "mutation_power": 1.0,
        "crossover_rate": 0.7,
        "species_threshold": 0.18,
        "seed": 11,
        "checkpoint_path": str(checkpoint),
        "resume": False,
        "max_workers": 1,
        "preserve_global_best": True,
        "selection_mode": "tournament",
        "tournament_size": 3,
        "rank_selection_pressure": 1.7,
        "novelty_weight": 0.0,
        "novelty_k": 5,
        "novelty_archive_size": 256,
        "novelty_metric_keys": [],
    }
    loaded_checkpoint = load_best_genome_checkpoint(checkpoint)
    assert loaded_checkpoint.schema_version == 2
    assert loaded_checkpoint.config["population_size"] == 8
    assert loaded_checkpoint.config["seed"] == 11

    topics = [event.topic for event in collector.events]
    assert EventTopic.RUN_START in topics
    assert EventTopic.RUN_END in topics
    assert topics.count(EventTopic.SCALAR) == 4
    assert topics.count(EventTopic.TRAINING_TRIAL) == 32
    assert topics.count(EventTopic.EVALUATION_PROGRESS) == 64

    progress = [
        event for event in collector.events
        if event.topic == EventTopic.EVALUATION_PROGRESS
    ]
    first_start = progress[0].data
    first_complete = progress[1].data
    final_complete = progress[-1].data
    assert first_start["phase"] == "start"
    assert first_start["generation"] == 0
    assert first_start["generations"] == 4
    assert first_start["individual"] == 0
    assert first_start["population_size"] == 8
    assert first_start["run_evaluations"] == 0
    assert "fitness" not in first_start
    assert first_complete["phase"] == "complete"
    assert first_complete["fitness"] > 0.0
    assert first_complete["evaluation_cache_hit"] == pytest.approx(0.0)
    assert first_complete["evaluation_wall_seconds"] >= 0.0
    assert first_complete["run_evaluations"] == 1
    assert final_complete["phase"] == "complete"
    assert final_complete["run_evaluations"] == 32
    assert final_complete["remaining_evaluations"] == 0


@pytest.mark.unit
def test_policy_objective_lookup_scores_proxy_genome() -> None:
    genome = PolicyGenome.seed("g0", generation=0, randomise=False)
    objective = get_policy_objective("proxy")
    result = objective(genome)

    assert "proxy_policy_gene_target" in policy_objective_names()
    assert result.fitness > 0.0
    assert result.metrics["proxy.target_cost"] >= 0.0


@pytest.mark.unit
def test_evolution_task_supports_parallel_evaluation_workers() -> None:
    def objective(genome: PolicyGenome) -> float:
        return float(genome.value("n_hidden"))

    task = EvolutionTask(
        EvolutionConfig(
            population_size=6,
            generations=2,
            elite_count=1,
            max_workers=2,
            seed=13,
        ),
        evaluator=CallableFitnessEvaluator(objective),
    )
    result = task.run()

    assert result.evaluations == 12
    assert result.best_genome is not None


@pytest.mark.unit
def test_thread_local_fitness_evaluator_pool_reuses_worker_evaluators() -> None:
    created: list[int] = []
    closed: list[int] = []

    class _WorkerEvaluator:
        def __init__(self, index: int) -> None:
            self.index = index
            created.append(index)

        def evaluate(self, genome: Any) -> FitnessResult:
            time.sleep(0.01)
            return FitnessResult(
                fitness=float(genome.value("n_hidden")) + self.index,
                metrics={"worker_index": float(self.index)},
            )

        def close(self) -> None:
            closed.append(self.index)

    pool = ThreadLocalFitnessEvaluatorPool(
        lambda worker_index: _WorkerEvaluator(worker_index),
        max_workers=2,
    )
    task = EvolutionTask(
        EvolutionConfig(
            population_size=6,
            generations=2,
            elite_count=1,
            max_workers=2,
            seed=13,
        ),
        evaluator=pool,
    )
    result = task.run()
    pool.close()

    assert result.evaluations == 12
    assert created == [0, 1]
    assert sorted(closed) == [0, 1]


@pytest.mark.unit
def test_scripted_progress_evaluator_uses_game_training_backend() -> None:
    evaluator = build_scripted_progress_fitness_evaluator(
        ScriptedGameFitnessConfig(
            max_episodes=1,
            frames_per_episode=4,
            telemetry_every=0,
            width=12,
            height=10,
            channels=1,
            device="cpu",
        )
    )
    genome = PolicyGenome.seed("g0", generation=0, randomise=False)

    result = evaluator.evaluate(genome)

    assert result.episodes == 1
    assert result.frames == 4
    assert "reward_mean" in result.metrics
    assert "max_x_progress" in result.metrics
    assert "score_gain" in result.metrics
    assert "action_button_mean" in result.metrics
    assert "action_up_frac" in result.metrics


@pytest.mark.unit
def test_cli_run_evolution_uses_shared_runner_and_writes_artifacts(tmp_path: Path) -> None:
    parser = cli._build_parser()
    args = parser.parse_args([
        "run",
        "--task",
        "evolution",
        "--seed",
        "23",
        "--device",
        "cpu",
        "--population-size",
        "4",
        "--generations",
        "2",
        "--elite-count",
        "1",
        "--evolution-workers",
        "2",
        "--no-stability",
        "--no-trial-stats",
        "--artifacts",
        str(tmp_path),
    ])

    rc = cli._cmd_run(args)

    assert rc == 0
    run_dirs = sorted(path for path in tmp_path.glob("run_*") if path.is_dir())
    assert run_dirs
    run_dir = run_dirs[-1]
    config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    training_end = json.loads((run_dir / "training_end.json").read_text(encoding="utf-8"))

    assert config["population_size"] == 4
    assert config["max_workers"] == 2
    assert training_end["evaluations"] == 8
    assert math.isfinite(float(training_end["best_fitness"]))
    assert (run_dir / "evolution" / "checkpoint.json").exists()
    assert (run_dir / "metrics" / "scalars.csv").exists()


@pytest.mark.unit
def test_cli_run_evolution_supports_scripted_game_backend(tmp_path: Path) -> None:
    parser = cli._build_parser()
    args = parser.parse_args([
        "run",
        "--task",
        "evolution",
        "--seed",
        "29",
        "--device",
        "cpu",
        "--population-size",
        "3",
        "--generations",
        "1",
        "--elite-count",
        "1",
        "--evolution-backend",
        "scripted-game",
        "--evolution-eval-frames",
        "3",
        "--no-stability",
        "--no-trial-stats",
        "--artifacts",
        str(tmp_path),
    ])

    rc = cli._cmd_run(args)

    assert rc == 0
    run_dirs = sorted(path for path in tmp_path.glob("run_*") if path.is_dir())
    training_end = json.loads(
        (run_dirs[-1] / "training_end.json").read_text(encoding="utf-8"),
    )
    assert training_end["evaluations"] == 3
    assert math.isfinite(float(training_end["best_fitness"]))


@pytest.mark.unit
def test_evaluator_reuses_one_client_across_genomes_when_requested() -> None:
    from neuroforge.environments.games.clients.scripted import ScriptedGameClient
    from neuroforge.neuroevolution.fitness.evaluators import GameTrainingFitnessEvaluator
    from neuroforge.perception.vision.encoding.frame_preprocess import FramePreprocessConfig

    created = {"count": 0}

    def factory() -> ScriptedGameClient:
        created["count"] += 1
        return ScriptedGameClient(width=12, height=10, channels=1, max_steps=4)

    evaluator = GameTrainingFitnessEvaluator(
        client_factory=factory,
        base_config=GameTrainingConfig(
            preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
            n_hidden=16, motor_per_button=1, input_fanin=8,
            decide_ticks=3, max_episodes=1, frames_per_episode=4, telemetry_every=0,
        ),
        reuse_client=True,
        eval_repeats=2,
    )
    genome = PolicyGenome.seed("g0", generation=0, randomise=False)
    result = evaluator.evaluate(genome)
    evaluator.evaluate(genome)
    evaluator.close()

    assert created["count"] == 1  # one client served all genomes/rollouts
    assert result.metrics["rollout_count"] == pytest.approx(2.0)
    assert result.metrics["rollout_frames_total"] == pytest.approx(8.0)
    assert result.frames == 8
    assert result.episodes == 2
    assert "fitness_std" in result.metrics
    assert "max_x_progress_std" in result.metrics
    assert "survival_frac" in result.metrics


@pytest.mark.unit
def test_game_training_evaluator_applies_terminal_fitness_penalties() -> None:
    from neuroforge.contracts.applications.games import EpisodeDecision, GameObservation
    from neuroforge.environments.games.clients.scripted import ActionProgressGameClient
    from neuroforge.neuroevolution.fitness.evaluators import GameTrainingFitnessEvaluator
    from neuroforge.perception.vision.encoding.frame_preprocess import FramePreprocessConfig

    class DeathAfterOneStep:
        def begin_episode(self) -> None:
            return

        def should_end(
            self,
            before: GameObservation,
            after: GameObservation,
        ) -> EpisodeDecision:
            del before, after
            return EpisodeDecision(terminated=True, reason="death")

    class ZeroReward:
        def reward(self, previous: GameObservation, current: GameObservation) -> float:
            del previous, current
            return 0.0

    def build_evaluator(*, death_penalty: float) -> GameTrainingFitnessEvaluator:
        return GameTrainingFitnessEvaluator(
            client_factory=lambda: ActionProgressGameClient(
                width=12, height=10, channels=1, max_steps=3,
            ),
            base_config=GameTrainingConfig(
                preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
                n_hidden=16,
                motor_per_button=1,
                input_fanin=8,
                decide_ticks=3,
                max_episodes=1,
                frames_per_episode=3,
                telemetry_every=0,
            ),
            reward_model_factory=ZeroReward,
            episode_manager_factory=DeathAfterOneStep,
            death_penalty=death_penalty,
        )

    genome = PolicyGenome.seed("g0", generation=0, randomise=False)
    base = build_evaluator(death_penalty=0.0).evaluate(genome)
    penalized = build_evaluator(death_penalty=7.5).evaluate(genome)

    assert base.metrics["termination.death"] == pytest.approx(1.0)
    assert penalized.metrics["fitness_terminal_score"] == pytest.approx(-7.5)
    assert base.fitness - penalized.fitness == pytest.approx(7.5)


@pytest.mark.unit
def test_game_training_evaluator_rewards_score_gain() -> None:
    from neuroforge.contracts.applications.games import (
        ControllerAction,
        GameClientStep,
        GameObservation,
        ScreenFrame,
        VisionGameMetrics,
    )
    from neuroforge.environments.games.smb3.actions import ActionEnergyConfig
    from neuroforge.neuroevolution.fitness.evaluators import GameTrainingFitnessEvaluator
    from neuroforge.perception.vision.encoding.frame_preprocess import FramePreprocessConfig

    class ScoreClient:
        def __init__(self) -> None:
            self._step = 0
            self._score = 0

        def reset(self) -> GameObservation:
            self._step = 0
            self._score = 0
            return self._observation()

        def step(self, action: ControllerAction) -> GameClientStep:
            del action
            self._step += 1
            self._score += 100
            return GameClientStep(self._observation(), truncated=self._step >= 3)

        def close(self) -> None:
            return

        def _observation(self) -> GameObservation:
            frame = ScreenFrame(width=12, height=10, channels=1, data=bytes(120))
            return GameObservation(
                step=self._step,
                t=float(self._step),
                frame=frame,
                metrics=VisionGameMetrics(score=self._score, x_progress=0.0),
            )

    class ZeroReward:
        def reward(self, previous: GameObservation, current: GameObservation) -> float:
            del previous, current
            return 0.0

    evaluator = GameTrainingFitnessEvaluator(
        client_factory=ScoreClient,
        base_config=GameTrainingConfig(
            preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
            n_hidden=16,
            motor_per_button=1,
            input_fanin=8,
            decide_ticks=3,
            max_episodes=1,
            frames_per_episode=3,
            telemetry_every=0,
            action_energy=ActionEnergyConfig(enabled=False),
        ),
        reward_model_factory=ZeroReward,
        progress_scale=0.0,
        score_gain_scale=0.01,
    )
    result = evaluator.evaluate(PolicyGenome.seed("g0", generation=0, randomise=False))

    assert result.metrics["score_gain_total"] == pytest.approx(300.0)
    assert result.metrics["fitness_score_gain_score"] == pytest.approx(3.0)
    assert result.fitness == pytest.approx(3.0)


@pytest.mark.unit
def test_trial_metric_collisions_do_not_replace_primary_rollout_metrics() -> None:
    from neuroforge.neuroevolution.fitness.evaluators import _non_colliding_metrics

    merged = _non_colliding_metrics(
        {"max_x_progress": 0.1, "termination.death": 1.0},
        existing={"max_x_progress": 0.3},
    )

    assert "max_x_progress" not in merged
    assert merged["max_x_progress_mean"] == pytest.approx(0.1)
    assert merged["termination.death"] == pytest.approx(1.0)


@pytest.mark.unit
def test_game_training_evaluator_config_transform_clamps_decide_ticks() -> None:
    from neuroforge.environments.games.clients.scripted import ScriptedGameClient
    from neuroforge.neuroevolution.fitness.evaluators import GameTrainingFitnessEvaluator
    from neuroforge.perception.vision.encoding.frame_preprocess import FramePreprocessConfig

    genome = PolicyGenome.seed("g0", generation=0, randomise=False)

    def clamp(config: GameTrainingConfig, _genome: object) -> GameTrainingConfig:
        return dataclasses.replace(config, decide_ticks=min(config.decide_ticks, 5))

    evaluator = GameTrainingFitnessEvaluator(
        client_factory=lambda: ScriptedGameClient(width=12, height=10, channels=1, max_steps=3),
        base_config=GameTrainingConfig(
            preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
            n_hidden=16,
            motor_per_button=1,
            input_fanin=8,
            decide_ticks=3,
            max_episodes=1,
            frames_per_episode=3,
            telemetry_every=0,
        ),
        config_transform=clamp,
    )

    result = evaluator.evaluate(genome)

    assert int(genome.value("decide_ticks")) == 12
    assert result.metrics["genome_decide_ticks"] == pytest.approx(12.0)
    assert result.metrics["effective_decide_ticks"] == pytest.approx(5.0)


@pytest.mark.unit
def test_evolution_preserves_global_raw_best_in_next_population(tmp_path: Path) -> None:
    champion = PolicyGenome.seed("champion", generation=0, randomise=False)
    challenger = champion.mutate(
        child_id="challenger",
        generation=0,
        rng=random.Random(3),
        rate=1.0,
    )
    checkpoint = tmp_path / "evolution.json"

    def objective(genome: PolicyGenome) -> float:
        return 10.0 if genome.content_key() == champion.content_key() else 0.0

    class _DropsBestReproduction:
        def next_generation(
            self,
            evaluated: list[Any],
            *,
            generation: int,
            rng: Any,
        ) -> list[Any]:
            del evaluated
            return [
                challenger.mutate(
                    child_id=f"g{generation}_{idx}",
                    generation=generation,
                    rng=rng,
                    rate=1.0,
                )
                for idx in range(4)
            ]

    EvolutionTask(
        EvolutionConfig(
            population_size=4,
            generations=2,
            elite_count=1,
            checkpoint_path=str(checkpoint),
            seed=19,
        ),
        evaluator=CallableFitnessEvaluator(objective),
        reproduction=_DropsBestReproduction(),
        seed_population=lambda _size, _rng: [champion, challenger, challenger, challenger],
    ).run()

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    population_keys = [
        decode_genome(item).content_key()
        for item in payload["population"]
    ]

    assert champion.content_key() in population_keys


@pytest.mark.unit
def test_evolution_task_requires_evaluator() -> None:
    with pytest.raises(ValueError, match="fitness evaluator"):
        EvolutionTask(EvolutionConfig(population_size=4)).run()


@pytest.mark.unit
def test_evolution_task_resumes_checkpoint(tmp_path: Path) -> None:
    def objective(genome: PolicyGenome) -> float:
        return float(genome.value("n_hidden"))

    checkpoint = tmp_path / "evolution.json"
    first = EvolutionTask(
        EvolutionConfig(
            population_size=6,
            generations=2,
            elite_count=1,
            checkpoint_path=str(checkpoint),
            seed=5,
        ),
        evaluator=CallableFitnessEvaluator(objective),
    )
    first.run()

    second = EvolutionTask(
        EvolutionConfig(
            population_size=6,
            generations=3,
            elite_count=1,
            checkpoint_path=str(checkpoint),
            resume=True,
            seed=5,
        ),
        evaluator=CallableFitnessEvaluator(objective),
    )
    result = second.run()

    assert result.generations == 1
    assert result.evaluations == 18


@pytest.mark.unit
def test_resume_progress_uses_checkpoint_population_before_reproducing(
    tmp_path: Path,
) -> None:
    def objective(genome: PolicyGenome) -> float:
        return float(genome.value("n_hidden"))

    checkpoint = tmp_path / "evolution.json"
    EvolutionTask(
        EvolutionConfig(
            population_size=6,
            generations=2,
            elite_count=1,
            checkpoint_path=str(checkpoint),
            seed=5,
        ),
        evaluator=CallableFitnessEvaluator(objective),
    ).run()

    bus = EventBus()
    collector = _Collector()
    bus.subscribe(EventTopic.RUN_START, collector)
    bus.subscribe(EventTopic.EVALUATION_PROGRESS, collector)
    result = EvolutionTask(
        EvolutionConfig(
            population_size=4,
            generations=4,
            elite_count=1,
            checkpoint_path=str(checkpoint),
            resume=True,
            seed=5,
        ),
        event_bus=bus,
        evaluator=CallableFitnessEvaluator(objective),
    ).run()

    progress = [
        event.data
        for event in collector.events
        if event.topic == EventTopic.EVALUATION_PROGRESS
    ]
    run_start = [
        event.data
        for event in collector.events
        if event.topic == EventTopic.RUN_START
    ][0]
    completes = [event for event in progress if event["phase"] == "complete"]

    assert result.generations == 2
    assert result.evaluations == 22
    assert run_start["resumed"] is True
    assert run_start["population_size"] == 4
    assert run_start["start_population_size"] == 6
    assert run_start["resume"]["loaded"] is True
    assert run_start["resume"]["schema_version"] == 2
    assert run_start["resume"]["rng_state_restored"] is True
    assert run_start["resume"]["generation"] == 2
    assert run_start["resume"]["evaluations"] == 12
    assert run_start["resume"]["population_size"] == 6
    assert run_start["resume"]["checkpoint_config"]["population_size"] == 6
    assert run_start["resume"]["config_differences"]["population_size"] == {
        "checkpoint": 6,
        "active": 4,
    }
    assert run_start["resume"]["config_differences"]["generations"] == {
        "checkpoint": 2,
        "active": 4,
    }
    assert completes[0]["generation"] == 2
    assert completes[0]["population_size"] == 6
    assert completes[0]["run_total_evaluations"] == 10
    assert completes[0]["total_evaluations"] == 22
    assert completes[5]["run_evaluations"] == 6
    assert completes[6]["generation"] == 3
    assert completes[6]["population_size"] == 4
    assert completes[-1]["run_evaluations"] == 10
    assert completes[-1]["remaining_evaluations"] == 0


@pytest.mark.unit
def test_resume_restores_rng_stream_for_deterministic_continuation(
    tmp_path: Path,
) -> None:
    def objective(genome: PolicyGenome) -> float:
        return float(genome.value("n_hidden")) + float(genome.value("decide_ticks"))

    full_checkpoint = tmp_path / "full.json"
    split_checkpoint = tmp_path / "split.json"

    def config(
        generations: int,
        checkpoint_path: Path,
        *,
        resume: bool = False,
    ) -> EvolutionConfig:
        return EvolutionConfig(
            generations=generations,
            checkpoint_path=str(checkpoint_path),
            population_size=6,
            elite_count=1,
            mutation_rate=0.8,
            mutation_power=0.9,
            crossover_rate=0.75,
            seed=17,
            resume=resume,
        )

    EvolutionTask(
        config(4, full_checkpoint),
        evaluator=CallableFitnessEvaluator(objective),
    ).run()
    EvolutionTask(
        config(2, split_checkpoint),
        evaluator=CallableFitnessEvaluator(objective),
    ).run()
    EvolutionTask(
        config(4, split_checkpoint, resume=True),
        evaluator=CallableFitnessEvaluator(objective),
    ).run()

    full_payload = json.loads(full_checkpoint.read_text(encoding="utf-8"))
    split_payload = json.loads(split_checkpoint.read_text(encoding="utf-8"))

    assert split_payload["generation"] == full_payload["generation"] == 4
    assert split_payload["evaluations"] == full_payload["evaluations"] == 24
    assert split_payload["population"] == full_payload["population"]
    assert split_payload["best"] == full_payload["best"]
    assert split_payload["rng_state"] == full_payload["rng_state"]
