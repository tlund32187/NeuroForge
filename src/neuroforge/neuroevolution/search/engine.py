"""Small, deterministic neuroevolution engine.

This is the first working Track E engine: it evaluates a population of
``PolicyGenome`` instances, groups them into compatibility species, applies
fitness sharing, and reproduces the next generation through elitism,
crossover, and mutation.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from neuroforge.contracts.applications.evolution import (
    FitnessResult,
    IFitnessEvaluator,
    IReproduction,
    ISpeciation,
)
from neuroforge.neuroevolution.genomes.policy import PolicyGenome
from neuroforge.neuroevolution.io.genome_codec import decode_genome
from neuroforge.neuroevolution.io.serde import (
    cast_json_object,
    rng_state_from_json,
    rng_state_to_json,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "EvaluatedGenome",
    "EvaluationProgress",
    "EvolutionConfig",
    "EvolutionEngine",
    "EvolutionState",
    "GenerationSummary",
    "ProgressCallback",
    "SimpleReproduction",
    "SimpleSpeciation",
    "default_seed_population",
    "evolution_config_to_dict",
]


_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Configuration for population evolution."""

    population_size: int = 16
    generations: int = 10
    elite_count: int = 2
    mutation_rate: float = 0.25
    mutation_power: float = 1.0
    crossover_rate: float = 0.7
    species_threshold: float = 0.18
    seed: int = 42
    checkpoint_path: str | None = None
    resume: bool = False
    max_workers: int = 1
    preserve_global_best: bool = True

    def __post_init__(self) -> None:
        if self.population_size < 2:
            msg = "EvolutionConfig.population_size must be >= 2"
            raise ValueError(msg)
        if self.generations < 1:
            msg = "EvolutionConfig.generations must be >= 1"
            raise ValueError(msg)
        if self.elite_count < 1 or self.elite_count >= self.population_size:
            msg = "EvolutionConfig.elite_count must be in [1, population_size)"
            raise ValueError(msg)
        for name in ("mutation_rate", "crossover_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                msg = f"EvolutionConfig.{name} must be in [0, 1]"
                raise ValueError(msg)
        if self.mutation_power <= 0:
            msg = "EvolutionConfig.mutation_power must be > 0"
            raise ValueError(msg)
        if self.species_threshold <= 0:
            msg = "EvolutionConfig.species_threshold must be > 0"
            raise ValueError(msg)
        if self.max_workers < 1:
            msg = "EvolutionConfig.max_workers must be >= 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EvaluatedGenome:
    """A genome plus fitness and species assignment.

    ``genome`` is typed loosely so any genome implementing the evolvable surface
    (``distance``/``crossover``/``mutate``/``as_offspring``/``content_key``) works
    — that is what lets a structural graph genome share this engine.
    """

    genome: Any
    result: FitnessResult
    species_id: int
    adjusted_fitness: float


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    """In-flight progress for one genome fitness evaluation."""

    phase: str
    generation: int
    individual: int
    population_size: int
    genome: PolicyGenome
    completed: int
    result: FitnessResult | None = None
    error: str | None = None


ProgressCallback = Callable[[EvaluationProgress], None]


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Aggregate metrics for one generation."""

    generation: int
    best: EvaluatedGenome
    mean_fitness: float
    species_count: int
    evaluations: int


@dataclass(slots=True)
class EvolutionState:
    """Mutable engine state that can be checkpointed."""

    generation: int
    population: list[Any]
    best: EvaluatedGenome | None = None
    evaluations: int = 0
    checkpoint_path: str = ""
    checkpoint_schema_version: int = 0
    checkpoint_config: dict[str, Any] = field(default_factory=dict[str, Any])
    rng_state_restored: bool = False


@dataclass(slots=True)
class _Species:
    id: int
    representative: PolicyGenome
    members: list[PolicyGenome] = field(default_factory=list[PolicyGenome])


class SimpleSpeciation:
    """Compatibility-distance species assignment."""

    def __init__(self, *, threshold: float) -> None:
        self._threshold = float(threshold)

    def assign(self, genomes: list[PolicyGenome]) -> dict[str, int]:
        """Return ``genome_id -> species_id``."""
        species: list[_Species] = []
        assignments: dict[str, int] = {}
        for genome in genomes:
            assigned = False
            for group in species:
                if genome.distance(group.representative) <= self._threshold:
                    group.members.append(genome)
                    assignments[genome.id] = group.id
                    assigned = True
                    break
            if not assigned:
                species_id = len(species)
                species.append(_Species(species_id, genome, [genome]))
                assignments[genome.id] = species_id
        return assignments


class SimpleReproduction:
    """Default reproduction: elitism + fitness-proportionate selection + variation.

    Genome-agnostic — it only calls the genome's own ``as_offspring`` / ``crossover``
    / ``mutate``, so the same strategy reproduces hyperparameter and structural
    genomes alike. Injectable via :class:`EvolutionEngine` for custom schemes.
    """

    def __init__(self, config: EvolutionConfig) -> None:
        self._cfg = config

    def next_generation(
        self, evaluated: list[EvaluatedGenome], *, generation: int, rng: random.Random,
    ) -> list[Any]:
        """Return the child population for *generation*."""
        cfg = self._cfg
        children: list[Any] = [
            item.genome.as_offspring(child_id=f"g{generation}_{idx}", generation=generation)
            for idx, item in enumerate(evaluated[: cfg.elite_count])
        ]
        while len(children) < cfg.population_size:
            child_index = len(children)
            parent_a = self._select_parent(evaluated, rng)
            if rng.random() < cfg.crossover_rate:
                parent_b = self._select_parent(evaluated, rng)
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
                )
            )
        return children

    @staticmethod
    def _select_parent(
        evaluated: list[EvaluatedGenome], rng: random.Random,
    ) -> EvaluatedGenome:
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


def default_seed_population(size: int, rng: random.Random) -> list[Any]:
    """Seed an initial :class:`PolicyGenome` population (genome 0 = defaults)."""
    return [
        PolicyGenome.seed(f"g0_{idx}", generation=0, rng=rng, randomise=idx != 0)
        for idx in range(size)
    ]


class EvolutionEngine:
    """Evaluate and reproduce a population of genomes (genome-agnostic)."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        *,
        evaluator: IFitnessEvaluator,
        speciation: ISpeciation | None = None,
        reproduction: IReproduction | None = None,
        seed_population: Callable[[int, random.Random], list[Any]] | None = None,
    ) -> None:
        self._cfg = config or EvolutionConfig()
        self._evaluator = evaluator
        self._rng = random.Random(self._cfg.seed)
        self._speciation = speciation or SimpleSpeciation(threshold=self._cfg.species_threshold)
        self._reproduction = reproduction or SimpleReproduction(self._cfg)
        self._seed_population = seed_population or default_seed_population
        # Cache fitness by gene content so an unchanged elite keeps its measured
        # score instead of being re-rolled with a new seed each generation (the
        # bug that stopped best-fitness from ratcheting). Identical genomes within
        # a run are deduped too.
        self._fitness_cache: dict[str, FitnessResult] = {}

    def initial_state(self) -> EvolutionState:
        """Create a fresh population or load one from checkpoint."""
        checkpoint = self._checkpoint_path()
        if self._cfg.resume and checkpoint is not None and checkpoint.exists():
            return self.load_state(checkpoint)
        population = self._seed_population(self._cfg.population_size, self._rng)
        return EvolutionState(generation=0, population=population)

    def evaluate_generation(
        self, state: EvolutionState,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[EvaluatedGenome], GenerationSummary]:
        """Evaluate the current population and return sorted results."""
        assignments = self._speciation.assign(state.population)
        species_sizes: dict[int, int] = {}
        for species_id in assignments.values():
            species_sizes[species_id] = species_sizes.get(species_id, 0) + 1

        results = self._evaluate_population(
            state.population,
            generation=state.generation,
            progress_callback=progress_callback,
        )
        evaluated: list[EvaluatedGenome] = []
        for genome, result in zip(state.population, results, strict=True):
            species_id = assignments[genome.id]
            adjusted = result.fitness / max(1, species_sizes[species_id])
            evaluated.append(
                EvaluatedGenome(
                    genome=genome,
                    result=result,
                    species_id=species_id,
                    adjusted_fitness=adjusted,
                )
            )
        evaluated.sort(key=lambda item: item.result.fitness, reverse=True)
        best = evaluated[0]
        if state.best is None or best.result.fitness > state.best.result.fitness:
            state.best = best
        state.evaluations += len(evaluated)
        mean = sum(item.result.fitness for item in evaluated) / len(evaluated)
        summary = GenerationSummary(
            generation=state.generation,
            best=best,
            mean_fitness=mean,
            species_count=len(set(assignments.values())),
            evaluations=state.evaluations,
        )
        return evaluated, summary

    def _evaluate_population(
        self,
        population: list[PolicyGenome],
        *,
        generation: int,
        progress_callback: ProgressCallback | None,
    ) -> list[FitnessResult]:
        """Evaluate genomes, optionally in parallel, preserving population order."""
        completed = 0
        lock = threading.Lock()
        population_size = len(population)
        results: list[FitnessResult | None] = [None] * population_size

        def _completed_count() -> int:
            with lock:
                return completed

        def _mark_completed() -> int:
            nonlocal completed
            with lock:
                completed += 1
                return completed

        def _publish_progress(
            *,
            phase: str,
            index: int,
            genome: PolicyGenome,
            completed_count: int,
            result: FitnessResult | None = None,
            error: str | None = None,
        ) -> None:
            if progress_callback is None:
                return
            progress_callback(
                EvaluationProgress(
                    phase=phase,
                    generation=generation,
                    individual=index,
                    population_size=population_size,
                    genome=genome,
                    completed=completed_count,
                    result=result,
                    error=error,
                )
            )

        def _evaluate(index: int, genome: PolicyGenome) -> tuple[int, FitnessResult]:
            _publish_progress(
                phase="start",
                index=index,
                genome=genome,
                completed_count=_completed_count(),
            )
            started = time.perf_counter()
            cached = self._fitness_cache.get(_content_key(genome))
            if cached is not None:
                progress_result = _with_evaluation_metrics(
                    cached,
                    wall_seconds=time.perf_counter() - started,
                    cache_hit=True,
                )
                _publish_progress(
                    phase="complete",
                    index=index,
                    genome=genome,
                    completed_count=_mark_completed(),
                    result=progress_result,
                )
                return index, cached
            try:
                result = self._evaluator.evaluate(genome)
            except Exception as exc:
                _publish_progress(
                    phase="error",
                    index=index,
                    genome=genome,
                    completed_count=_mark_completed(),
                    error=str(exc),
                )
                raise
            progress_result = _with_evaluation_metrics(
                result,
                wall_seconds=time.perf_counter() - started,
                cache_hit=False,
            )
            _publish_progress(
                phase="complete",
                index=index,
                genome=genome,
                completed_count=_mark_completed(),
                result=progress_result,
            )
            return index, result

        if self._cfg.max_workers <= 1:
            for index, genome in enumerate(population):
                result_index, result = _evaluate(index, genome)
                results[result_index] = result
        else:
            with ThreadPoolExecutor(max_workers=self._cfg.max_workers) as pool:
                futures = [
                    pool.submit(_evaluate, index, genome)
                    for index, genome in enumerate(population)
                ]
                for future in as_completed(futures):
                    result_index, result = future.result()
                    results[result_index] = result

        completed_results = _complete_results(results)
        for genome, result in zip(population, completed_results, strict=True):
            self._fitness_cache.setdefault(_content_key(genome), result)
        return completed_results

    def reproduce(
        self, evaluated: list[EvaluatedGenome], *, generation: int,
    ) -> list[Any]:
        """Create the next population via the injected reproduction strategy."""
        return self._reproduction.next_generation(
            evaluated, generation=generation, rng=self._rng,
        )

    def advance(self, state: EvolutionState, evaluated: list[EvaluatedGenome]) -> None:
        """Mutate *state* to the next generation."""
        next_generation = state.generation + 1
        population = self.reproduce(evaluated, generation=next_generation)
        state.population = self._with_global_best(
            population,
            state=state,
            generation=next_generation,
        )
        state.generation = next_generation

    def maybe_checkpoint(self, state: EvolutionState) -> None:
        """Persist state if checkpointing is enabled."""
        path = self._checkpoint_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_to_dict(state), indent=2), encoding="utf-8")

    def state_to_dict(self, state: EvolutionState) -> dict[str, Any]:
        """Encode state as JSON-safe data."""
        best_payload: dict[str, Any] | None = None
        if state.best is not None:
            best_payload = {
                "genome": state.best.genome.to_dict(),
                "fitness": state.best.result.fitness,
                "metrics": state.best.result.metrics,
                "episodes": state.best.result.episodes,
                "frames": state.best.result.frames,
                "species_id": state.best.species_id,
                "adjusted_fitness": state.best.adjusted_fitness,
            }
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "generation": state.generation,
            "evaluations": state.evaluations,
            "config": evolution_config_to_dict(self._cfg),
            "rng_state": rng_state_to_json(self._rng.getstate()),
            "population": [genome.to_dict() for genome in state.population],
            "best": best_payload,
        }

    def load_state(self, path: str | Path) -> EvolutionState:
        """Load state from a JSON checkpoint."""
        checkpoint_path = Path(path)
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        raw_population = payload.get("population", [])
        if not isinstance(raw_population, list):
            msg = "evolution checkpoint population must be a list"
            raise ValueError(msg)
        rng_state_restored = False
        raw_rng_state = payload.get("rng_state")
        if raw_rng_state is not None:
            self._rng.setstate(rng_state_from_json(raw_rng_state))
            rng_state_restored = True
        population = [
            decode_genome(cast_json_object(item))
            for item in cast("Sequence[Any]", raw_population)
        ]
        checkpoint_config = payload.get("config")
        state = EvolutionState(
            generation=int(payload.get("generation", 0)),
            population=population,
            evaluations=int(payload.get("evaluations", 0)),
            checkpoint_path=str(checkpoint_path),
            checkpoint_schema_version=int(payload.get("schema_version", 1)),
            checkpoint_config=(
                cast_json_object(checkpoint_config) if isinstance(checkpoint_config, dict) else {}
            ),
            rng_state_restored=rng_state_restored,
        )
        best_raw = payload.get("best")
        if isinstance(best_raw, dict):
            best_obj = cast("dict[str, Any]", best_raw)
            genome = decode_genome(cast_json_object(best_obj["genome"]))
            result = FitnessResult(
                fitness=float(best_obj["fitness"]),
                metrics={
                    str(key): float(value)
                    for key, value in cast_json_object(best_obj.get("metrics", {})).items()
                },
                episodes=int(best_obj.get("episodes", 0)),
                frames=int(best_obj.get("frames", 0)),
            )
            state.best = EvaluatedGenome(
                genome=genome,
                result=result,
                species_id=int(best_obj.get("species_id", 0)),
                adjusted_fitness=float(best_obj.get("adjusted_fitness", result.fitness)),
            )
        return state

    def _checkpoint_path(self) -> Path | None:
        if self._cfg.checkpoint_path is None:
            return None
        return Path(self._cfg.checkpoint_path)

    def _with_global_best(
        self,
        population: list[Any],
        *,
        state: EvolutionState,
        generation: int,
    ) -> list[Any]:
        """Reserve one future slot for the best raw-fitness genome seen so far."""
        if not self._cfg.preserve_global_best or state.best is None or not population:
            return population
        best_genome = state.best.genome
        best_key = _content_key(best_genome)
        if any(_content_key(genome) == best_key for genome in population):
            return population
        out = list(population)
        out[-1] = _as_global_best_offspring(best_genome, generation=generation)
        return out


def _content_key(genome: PolicyGenome) -> str:
    """Cache key over gene content; falls back to id for genomes without one."""
    content_key = getattr(genome, "content_key", None)
    return str(content_key()) if callable(content_key) else genome.id


def _as_global_best_offspring(genome: Any, *, generation: int) -> Any:
    """Clone *genome* into a reserved global-best slot for the next generation."""
    clone = getattr(genome, "as_offspring", None)
    child_id = f"g{generation}_global_best"
    if not callable(clone):
        return genome
    try:
        return clone(child_id=child_id, generation=generation, parent_ids=(str(genome.id),))
    except TypeError:
        return clone(child_id=child_id, generation=generation)


def evolution_config_to_dict(config: EvolutionConfig) -> dict[str, Any]:
    """Return the checkpoint-relevant evolution config as JSON-safe metadata."""
    return dict(asdict(config))


def _complete_results(results: list[FitnessResult | None]) -> list[FitnessResult]:
    """Return fully populated fitness results after worker collection."""
    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        msg = f"missing fitness results for population index(es): {missing}"
        raise RuntimeError(msg)
    return [result for result in results if result is not None]


def _with_evaluation_metrics(
    result: FitnessResult,
    *,
    wall_seconds: float,
    cache_hit: bool,
) -> FitnessResult:
    metrics = dict(result.metrics)
    metrics["evaluation_wall_seconds"] = float(max(0.0, wall_seconds))
    metrics["evaluation_cache_hit"] = 1.0 if cache_hit else 0.0
    frames = max(0, int(result.frames))
    if wall_seconds > 0.0 and frames > 0:
        metrics["evaluation_fps"] = frames / wall_seconds
    return FitnessResult(
        fitness=result.fitness,
        metrics=metrics,
        episodes=result.episodes,
        frames=result.frames,
    )
