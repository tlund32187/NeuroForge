"""Unit tests for the first neuroevolution slice."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import neuroforge.runners.cli as cli
from neuroforge.contracts.evolution import FitnessResult
from neuroforge.contracts.monitors import EventTopic
from neuroforge.evolution import (
    CallableFitnessEvaluator,
    EvolutionConfig,
    PolicyGenome,
    ScriptedGameFitnessConfig,
    build_scripted_progress_fitness_evaluator,
    get_policy_objective,
    policy_objective_names,
)
from neuroforge.monitors.bus import EventBus
from neuroforge.tasks.evolution import EvolutionTask

if TYPE_CHECKING:
    from pathlib import Path


class _Collector:
    enabled = True

    def __init__(self) -> None:
        self.events: list[object] = []

    def on_event(self, event: object) -> None:
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

    spec = genome.to_network_spec(n_input=32)
    assert spec.metadata["genome_id"] == "g0"
    assert [pop.name for pop in spec.populations] == ["input", "hidden", "motor"]
    assert {proj.name for proj in spec.projections} >= {"in_to_hidden", "hidden_to_motor"}


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

    topics = [event.topic for event in collector.events]  # type: ignore[attr-defined]
    assert EventTopic.RUN_START in topics
    assert EventTopic.RUN_END in topics
    assert topics.count(EventTopic.SCALAR) == 4
    assert topics.count(EventTopic.TRAINING_TRIAL) == 32


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
    assert training_end["best_fitness"] >= 0.0
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
    assert training_end["best_fitness"] >= 0.0


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
