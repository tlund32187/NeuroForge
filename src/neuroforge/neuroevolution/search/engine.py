"""Small, deterministic neuroevolution engine.

This is the first working Track E engine: it evaluates a population of
``PolicyGenome`` instances, groups them into compatibility species, applies
fitness sharing, and reproduces the next generation through elitism,
crossover, and mutation.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

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

__all__ = [
    "EvaluatedGenome",
    "EvaluationProgress",
    "AdaptiveSpeciation",
    "EvolutionConfig",
    "EvolutionEngine",
    "EvolutionState",
    "GenerationSummary",
    "ProgressCallback",
    "SimpleReproduction",
    "SimpleSpeciation",
    "SpeciesAwareReproduction",
    "default_seed_population",
    "evolution_config_to_dict",
    "select_parent_by_mode",
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
    selection_mode: str = "tournament"
    tournament_size: int = 3
    rank_selection_pressure: float = 1.7
    novelty_weight: float = 0.0
    novelty_k: int = 5
    novelty_archive_size: int = 256
    novelty_metric_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        selection_mode = self.selection_mode.strip().lower()
        if selection_mode not in {"roulette", "rank", "tournament"}:
            msg = "EvolutionConfig.selection_mode must be roulette, rank, or tournament"
            raise ValueError(msg)
        object.__setattr__(self, "selection_mode", selection_mode)
        object.__setattr__(
            self,
            "novelty_metric_keys",
            tuple(
                str(key).strip()
                for key in self.novelty_metric_keys
                if str(key).strip()
            ),
        )
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
        if self.tournament_size < 1:
            msg = "EvolutionConfig.tournament_size must be >= 1"
            raise ValueError(msg)
        if self.rank_selection_pressure < 1.0:
            msg = "EvolutionConfig.rank_selection_pressure must be >= 1"
            raise ValueError(msg)
        if self.novelty_weight < 0.0:
            msg = "EvolutionConfig.novelty_weight must be >= 0"
            raise ValueError(msg)
        if self.novelty_k < 1:
            msg = "EvolutionConfig.novelty_k must be >= 1"
            raise ValueError(msg)
        if self.novelty_archive_size < 0:
            msg = "EvolutionConfig.novelty_archive_size must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EvaluatedGenome:
    """A genome plus fitness and species assignment.

    ``genome`` is typed loosely so any genome implementing the evolvable surface
    (``distance``/``crossover``/``mutate``/``as_offspring``/``content_key``) works
    That is what lets a structural graph genome share this engine.
    """

    genome: Any
    result: FitnessResult
    species_id: int
    adjusted_fitness: float
    species_uid: int


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    """In-flight progress for one genome fitness evaluation."""

    phase: str
    generation: int
    individual: int
    population_size: int
    genome: PolicyGenome
    completed: int
    species_id: int
    species_uid: int
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
    representative: Any
    members: list[Any] = field(default_factory=list[Any])


class SimpleSpeciation:
    """Compatibility-distance species assignment."""

    def __init__(self, *, threshold: float) -> None:
        self._threshold = float(threshold)

    def assign(self, genomes: list[Any]) -> dict[str, int]:
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


class AdaptiveSpeciation:
    """Compatibility speciation with a threshold that tracks a target range.

    A fixed threshold is brittle for open-ended structure search: the same value
    can collapse a run into one species early and fragment it later. This assigner
    uses the current threshold for the generation, then nudges it for the next
    generation based on the observed species count.
    """

    def __init__(
        self,
        *,
        threshold: float,
        target_min: int = 6,
        target_max: int = 12,
        adjustment: float = 0.08,
        min_threshold: float = 1e-6,
        max_threshold: float = 100.0,
    ) -> None:
        if threshold <= 0.0:
            msg = "AdaptiveSpeciation.threshold must be > 0"
            raise ValueError(msg)
        if target_min < 1 or target_max < target_min:
            msg = "AdaptiveSpeciation target range must satisfy 1 <= min <= max"
            raise ValueError(msg)
        self._threshold = float(threshold)
        self._target_min = int(target_min)
        self._target_max = int(target_max)
        self._adjustment = max(0.0, float(adjustment))
        self._min_threshold = max(1e-12, float(min_threshold))
        self._max_threshold = max(self._min_threshold, float(max_threshold))
        self._last_species_count = 0

    @property
    def threshold(self) -> float:
        """Threshold that will be used for the next assignment."""
        return self._threshold

    @property
    def last_species_count(self) -> int:
        """Species count observed during the previous assignment."""
        return self._last_species_count

    def assign(self, genomes: list[Any]) -> dict[str, int]:
        """Return ``genome_id -> species_id`` and adapt for the next generation."""
        assignments = SimpleSpeciation(threshold=self._threshold).assign(genomes)
        species_count = len(set(assignments.values()))
        self._last_species_count = species_count
        self._adapt(species_count)
        return assignments

    def _adapt(self, species_count: int) -> None:
        if species_count < self._target_min:
            self._threshold *= max(0.0, 1.0 - self._adjustment)
        elif species_count > self._target_max:
            self._threshold *= 1.0 + self._adjustment
        self._threshold = min(
            self._max_threshold,
            max(self._min_threshold, self._threshold),
        )


class SimpleReproduction:
    """Default reproduction: elitism + configurable selection + variation.

    Genome-agnostic: it only calls the genome's own ``as_offspring`` / ``crossover``
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
                parent_b = self._select_distinct_parent(
                    evaluated,
                    rng,
                    exclude_genome_id=str(parent_a.genome.id),
                )
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

    def _select_parent(
        self, evaluated: list[EvaluatedGenome], rng: random.Random,
    ) -> EvaluatedGenome:
        return select_parent_by_mode(
            evaluated,
            rng,
            mode=self._cfg.selection_mode,
            tournament_size=self._cfg.tournament_size,
            rank_pressure=self._cfg.rank_selection_pressure,
        )

    def _select_distinct_parent(
        self,
        evaluated: list[EvaluatedGenome],
        rng: random.Random,
        *,
        exclude_genome_id: str,
    ) -> EvaluatedGenome:
        for _ in range(8):
            candidate = self._select_parent(evaluated, rng)
            if str(candidate.genome.id) != exclude_genome_id:
                return candidate
        for candidate in evaluated:
            if str(candidate.genome.id) != exclude_genome_id:
                return candidate
        msg = "distinct crossover parent not found"
        raise RuntimeError(msg)


MutationHook = Callable[
    [Any, str, int, random.Random, float, float],
    Any,
]


class SpeciesAwareReproduction:
    """Reproduce within species and allocate offspring per species quality.

    This is still genome-agnostic, but unlike :class:`SimpleReproduction` it
    protects each active species with at least one child and mostly mates parents
    inside their species. Structural genomes can inject a mutation hook so NEAT
    innovation registries stay consistent.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        *,
        mutate_child: MutationHook | None = None,
        species_elite_count: int = 1,
    ) -> None:
        if species_elite_count < 0:
            msg = "SpeciesAwareReproduction.species_elite_count must be >= 0"
            raise ValueError(msg)
        self._cfg = config
        self._mutate_child = mutate_child
        self._species_elite_count = int(species_elite_count)

    def next_generation(
        self, evaluated: list[EvaluatedGenome], *, generation: int, rng: random.Random,
    ) -> list[Any]:
        """Return a child population using species-aware allocation."""
        if not evaluated:
            return []
        groups = _species_groups(evaluated)
        allocations = _species_allocations(groups, self._cfg.population_size)
        children: list[Any] = []
        for species_id, members in groups:
            requested = allocations.get(species_id, 0)
            if requested <= 0:
                continue
            children.extend(
                self._species_children(
                    members,
                    count=requested,
                    generation=generation,
                    rng=rng,
                    child_offset=len(children),
                )
            )
        while len(children) < self._cfg.population_size:
            child = self._make_child(
                evaluated,
                generation=generation,
                rng=rng,
                child_index=len(children),
            )
            children.append(child)
        return children[: self._cfg.population_size]

    def _species_children(
        self,
        members: list[EvaluatedGenome],
        *,
        count: int,
        generation: int,
        rng: random.Random,
        child_offset: int,
    ) -> list[Any]:
        ranked = sorted(members, key=lambda item: item.result.fitness, reverse=True)
        out: list[Any] = []
        elites = min(self._species_elite_count, count, len(ranked))
        for item in ranked[:elites]:
            out.append(
                item.genome.as_offspring(
                    child_id=f"g{generation}_{child_offset + len(out)}",
                    generation=generation,
                )
            )
        while len(out) < count:
            child = self._make_child(
                ranked,
                generation=generation,
                rng=rng,
                child_index=child_offset + len(out),
            )
            out.append(child)
        return out

    def _make_child(
        self,
        pool: list[EvaluatedGenome],
        *,
        generation: int,
        rng: random.Random,
        child_index: int,
    ) -> Any:
        parent_a = select_parent_by_mode(
            pool,
            rng,
            mode=self._cfg.selection_mode,
            tournament_size=self._cfg.tournament_size,
            rank_pressure=self._cfg.rank_selection_pressure,
        )
        child_id = f"g{generation}_{child_index}"
        if rng.random() < self._cfg.crossover_rate and len(pool) > 1:
            parent_b = _select_distinct_parent_by_mode(
                pool,
                rng,
                exclude_genome_id=str(parent_a.genome.id),
                mode=self._cfg.selection_mode,
                tournament_size=self._cfg.tournament_size,
                rank_pressure=self._cfg.rank_selection_pressure,
            )
            child = parent_a.genome.crossover(
                parent_b.genome,
                child_id=child_id,
                generation=generation,
                rng=rng,
            )
        else:
            child = parent_a.genome.as_offspring(child_id=child_id, generation=generation)
        return self._mutate(
            child,
            child_id=child.id,
            generation=generation,
            rng=rng,
        )

    def _mutate(
        self,
        child: Any,
        *,
        child_id: str,
        generation: int,
        rng: random.Random,
    ) -> Any:
        if self._mutate_child is not None:
            return self._mutate_child(
                child,
                child_id,
                generation,
                rng,
                self._cfg.mutation_rate,
                self._cfg.mutation_power,
            )
        return child.mutate(
            child_id=child_id,
            generation=generation,
            rng=rng,
            rate=self._cfg.mutation_rate,
            power=self._cfg.mutation_power,
        )


def default_seed_population(size: int, rng: random.Random) -> list[Any]:
    """Seed an initial :class:`PolicyGenome` population (genome 0 = defaults)."""
    return [
        PolicyGenome.seed(f"g0_{idx}", generation=0, rng=rng, randomise=idx != 0)
        for idx in range(size)
    ]


class _NoveltyArchive:
    """Small behavior archive used to add novelty pressure to selection fitness."""

    def __init__(self, *, keys: tuple[str, ...], k: int, max_size: int) -> None:
        self._keys = keys
        self._k = max(1, int(k))
        self._max_size = max(0, int(max_size))
        self._archive: list[tuple[float, ...]] = []

    def apply(self, results: list[FitnessResult], *, weight: float) -> list[FitnessResult]:
        """Return results with novelty metrics and an optional selection bonus."""
        if not self._keys:
            return results
        descriptors = [self._descriptor(result) for result in results]
        scores = [
            self._score_descriptor(descriptor, descriptors, index=index)
            for index, descriptor in enumerate(descriptors)
        ]
        self._remember(descriptors)
        weighted = max(0.0, float(weight))
        return [
            _with_novelty_metrics(result, novelty_score=score, novelty_weight=weighted)
            for result, score in zip(results, scores, strict=True)
        ]

    def _descriptor(self, result: FitnessResult) -> tuple[float, ...]:
        return tuple(_safe_metric(result.metrics.get(key, 0.0)) for key in self._keys)

    def _score_descriptor(
        self,
        descriptor: tuple[float, ...],
        peers: list[tuple[float, ...]],
        *,
        index: int,
    ) -> float:
        neighbors = [
            candidate for peer_index, candidate in enumerate(peers) if peer_index != index
        ]
        neighbors.extend(self._archive)
        if not neighbors:
            return 0.0
        distances = sorted(_descriptor_distance(descriptor, candidate) for candidate in neighbors)
        nearest = distances[: min(self._k, len(distances))]
        return sum(nearest) / max(1, len(nearest))

    def _remember(self, descriptors: list[tuple[float, ...]]) -> None:
        if self._max_size <= 0:
            self._archive.clear()
            return
        self._archive.extend(descriptors)
        overflow = len(self._archive) - self._max_size
        if overflow > 0:
            del self._archive[:overflow]


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
        self._novelty_archive = (
            _NoveltyArchive(
                keys=self._cfg.novelty_metric_keys,
                k=self._cfg.novelty_k,
                max_size=self._cfg.novelty_archive_size,
            )
            if self._cfg.novelty_metric_keys
            else None
        )
        self._next_species_uid = 0
        self._species_representative_by_uid: dict[int, Any] = {}

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
        species_uid_by_local = self._assign_species_uids(state.population, assignments)
        species_uid_assignments = {
            genome.id: species_uid_by_local[assignments[genome.id]] for genome in state.population
        }
        species_sizes: dict[int, int] = {}
        for species_id in assignments.values():
            species_sizes[species_id] = species_sizes.get(species_id, 0) + 1

        results = self._evaluate_population(
            state.population,
            generation=state.generation,
            species_assignments=assignments,
            species_uid_assignments=species_uid_assignments,
            progress_callback=progress_callback,
        )
        results = self._with_novelty(results)
        evaluated: list[EvaluatedGenome] = []
        for genome, result in zip(state.population, results, strict=True):
            species_id = assignments[genome.id]
            adjusted = result.fitness / max(1, species_sizes[species_id])
            evaluated.append(
                EvaluatedGenome(
                    genome=genome,
                    result=result,
                    species_id=species_id,
                    species_uid=species_uid_assignments[genome.id],
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

    def _with_novelty(self, results: list[FitnessResult]) -> list[FitnessResult]:
        if self._novelty_archive is None:
            return results
        return self._novelty_archive.apply(results, weight=self._cfg.novelty_weight)

    def _evaluate_population(
        self,
        population: list[PolicyGenome],
        *,
        generation: int,
        species_assignments: dict[str, int] | None,
        species_uid_assignments: dict[str, int] | None,
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
            species_id = -1
            species_uid = -1
            if species_assignments is not None:
                species_id = int(species_assignments.get(genome.id, -1))
            if species_uid_assignments is not None:
                species_uid = int(species_uid_assignments.get(genome.id, -1))
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
                    species_id=species_id,
                    species_uid=species_uid,
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
                "species_uid": state.best.species_uid,
                "adjusted_fitness": state.best.adjusted_fitness,
            }
        species_tracking = {
            "next_uid": self._next_species_uid,
            "representatives": [
                {
                    "uid": uid,
                    "genome": representative.to_dict(),
                }
                for uid, representative in sorted(self._species_representative_by_uid.items())
            ],
        }
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "generation": state.generation,
            "evaluations": state.evaluations,
            "config": evolution_config_to_dict(self._cfg),
            "rng_state": rng_state_to_json(self._rng.getstate()),
            "population": [genome.to_dict() for genome in state.population],
            "best": best_payload,
            "species_tracking": species_tracking,
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
                species_uid=int(best_obj.get("species_uid", -1)),
                adjusted_fitness=float(best_obj.get("adjusted_fitness", result.fitness)),
            )
        tracking_raw = payload.get("species_tracking")
        if isinstance(tracking_raw, dict):
            tracking = cast("dict[str, Any]", tracking_raw)
            self._next_species_uid = int(tracking.get("next_uid", 0))
            restored: dict[int, Any] = {}
            reps_raw = tracking.get("representatives", [])
            if isinstance(reps_raw, Sequence):
                for item in cast("Sequence[object]", reps_raw):
                    if not isinstance(item, dict):
                        continue
                    entry = cast("dict[str, Any]", item)
                    genome_raw = entry.get("genome")
                    if not isinstance(genome_raw, dict):
                        continue
                    uid = int(entry.get("uid", -1))
                    if uid < 0:
                        continue
                    restored[uid] = decode_genome(cast_json_object(genome_raw))
            self._species_representative_by_uid = restored
            self._next_species_uid = max(
                self._next_species_uid,
                max(self._species_representative_by_uid.keys(), default=-1) + 1,
            )
        return state

    def _assign_species_uids(
        self,
        population: list[Any],
        assignments: dict[str, int],
    ) -> dict[int, int]:
        local_representatives: dict[int, Any] = {}
        for genome in population:
            local_id = int(assignments[genome.id])
            if local_id not in local_representatives:
                local_representatives[local_id] = genome
        available = set(self._species_representative_by_uid.keys())
        matched: dict[int, int] = {}
        threshold = self._species_match_threshold()
        for local_id in sorted(local_representatives):
            representative = local_representatives[local_id]
            best_uid = -1
            best_distance = float("inf")
            for uid in sorted(available):
                previous = self._species_representative_by_uid[uid]
                distance = float(representative.distance(previous))
                if distance < best_distance:
                    best_distance = distance
                    best_uid = uid
            if best_uid >= 0 and best_distance <= threshold:
                matched[local_id] = best_uid
                available.remove(best_uid)
            else:
                matched[local_id] = self._next_species_uid
                self._next_species_uid += 1
        self._species_representative_by_uid = {
            uid: local_representatives[local_id] for local_id, uid in matched.items()
        }
        return matched

    def _species_match_threshold(self) -> float:
        threshold = getattr(self._speciation, "threshold", self._cfg.species_threshold)
        if isinstance(threshold, int | float) and not isinstance(threshold, bool):
            return float(max(1e-12, threshold))
        return float(max(1e-12, self._cfg.species_threshold))

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
        """Reserve one future slot for the best reported-fitness genome seen so far."""
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
    """Cache key over gene content plus inherited learned-state artifact."""
    content_key = getattr(genome, "content_key", None)
    base = str(content_key()) if callable(content_key) else genome.id
    learned = getattr(genome, "learned_checkpoint_path", "")
    return f"{base}|learned={learned}" if isinstance(learned, str) and learned else base


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
    data = dict(asdict(config))
    data["novelty_metric_keys"] = list(config.novelty_metric_keys)
    return data


def _species_groups(evaluated: list[EvaluatedGenome]) -> list[tuple[int, list[EvaluatedGenome]]]:
    """Return evaluated genomes grouped by species, strongest species first."""
    grouped: dict[int, list[EvaluatedGenome]] = {}
    for item in evaluated:
        grouped.setdefault(item.species_id, []).append(item)
    return sorted(
        grouped.items(),
        key=lambda pair: max(item.result.fitness for item in pair[1]),
        reverse=True,
    )


def _species_allocations(
    groups: list[tuple[int, list[EvaluatedGenome]]],
    population_size: int,
) -> dict[int, int]:
    """Allocate offspring counts across species while preserving diversity."""
    if population_size <= 0 or not groups:
        return {}
    active = groups[:population_size]
    allocations = {species_id: 1 for species_id, _members in active}
    remaining = population_size - len(active)
    if remaining <= 0:
        return allocations

    scores = [sum(item.adjusted_fitness for item in members) for _sid, members in active]
    weights = _shifted_weights(scores)
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0 for _score in scores]
        total = float(len(weights))
    quotas = [remaining * weight / total for weight in weights]
    floors = [int(quota) for quota in quotas]
    for (species_id, _members), count in zip(active, floors, strict=True):
        allocations[species_id] += count
    leftover = remaining - sum(floors)
    order = sorted(
        range(len(active)),
        key=lambda idx: quotas[idx] - floors[idx],
        reverse=True,
    )
    for idx in order[:leftover]:
        species_id = active[idx][0]
        allocations[species_id] += 1
    return allocations


def _select_adjusted_parent(
    evaluated: list[EvaluatedGenome],
    rng: random.Random,
) -> EvaluatedGenome:
    """Fitness-proportionate parent selection over shifted adjusted fitness."""
    weights = _shifted_weights([item.adjusted_fitness for item in evaluated])
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


def select_parent_by_mode(
    evaluated: list[EvaluatedGenome],
    rng: random.Random,
    *,
    mode: str,
    tournament_size: int,
    rank_pressure: float,
) -> EvaluatedGenome:
    """Select one parent using the configured selection strategy."""
    if not evaluated:
        msg = "cannot select from an empty evaluated population"
        raise ValueError(msg)
    if mode == "tournament":
        return _select_tournament_parent(evaluated, rng, tournament_size=tournament_size)
    if mode == "rank":
        return _select_rank_parent(evaluated, rng, pressure=rank_pressure)
    return _select_adjusted_parent(evaluated, rng)


def _select_distinct_parent_by_mode(
    evaluated: list[EvaluatedGenome],
    rng: random.Random,
    *,
    exclude_genome_id: str,
    mode: str,
    tournament_size: int,
    rank_pressure: float,
) -> EvaluatedGenome:
    for _ in range(8):
        candidate = select_parent_by_mode(
            evaluated,
            rng,
            mode=mode,
            tournament_size=tournament_size,
            rank_pressure=rank_pressure,
        )
        if str(candidate.genome.id) != exclude_genome_id:
            return candidate
    for candidate in evaluated:
        if str(candidate.genome.id) != exclude_genome_id:
            return candidate
    msg = "distinct crossover parent not found"
    raise RuntimeError(msg)


def _select_tournament_parent(
    evaluated: list[EvaluatedGenome],
    rng: random.Random,
    *,
    tournament_size: int,
) -> EvaluatedGenome:
    """Select the strongest genome from a random tournament sample."""
    size = min(max(1, tournament_size), len(evaluated))
    contenders = (
        list(evaluated) if size >= len(evaluated) else rng.sample(evaluated, size)
    )
    return max(
        contenders,
        key=lambda item: (item.adjusted_fitness, item.result.fitness),
    )


def _select_rank_parent(
    evaluated: list[EvaluatedGenome],
    rng: random.Random,
    *,
    pressure: float,
) -> EvaluatedGenome:
    """Rank-based parent selection that ignores absolute score magnitude."""
    ranked = sorted(evaluated, key=lambda item: item.adjusted_fitness)
    if len(ranked) == 1:
        return ranked[0]
    span = len(ranked) - 1
    weights = [
        1.0 + (max(1.0, pressure) - 1.0) * rank / span
        for rank in range(len(ranked))
    ]
    selected = _weighted_pick(ranked, weights, rng)
    return selected if selected is not None else ranked[-1]


def _weighted_pick(
    items: list[EvaluatedGenome],
    weights: list[float],
    rng: random.Random,
) -> EvaluatedGenome | None:
    """Pick one item from non-negative weights."""
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0.0:
        return None
    pick = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights, strict=True):
        acc += max(0.0, weight)
        if acc >= pick:
            return item
    return items[-1] if items else None


def _shifted_weights(values: list[float]) -> list[float]:
    """Return non-negative weights that preserve ordering, even below zero."""
    if not values:
        return []
    minimum = min(values)
    return [value - minimum + 1e-9 for value in values]


def _complete_results(results: list[FitnessResult | None]) -> list[FitnessResult]:
    """Return fully populated fitness results after worker collection."""
    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        msg = f"missing fitness results for population index(es): {missing}"
        raise RuntimeError(msg)
    return [result for result in results if result is not None]


def _with_novelty_metrics(
    result: FitnessResult,
    *,
    novelty_score: float,
    novelty_weight: float,
) -> FitnessResult:
    """Return a result whose fitness includes a novelty bonus."""
    objective = float(result.fitness)
    novelty = max(0.0, float(novelty_score))
    bonus = novelty * max(0.0, float(novelty_weight))
    metrics = dict(result.metrics)
    metrics["fitness_objective"] = objective
    metrics["novelty_score"] = novelty
    metrics["novelty_weight"] = max(0.0, float(novelty_weight))
    metrics["fitness_novelty_bonus"] = bonus
    metrics["fitness_selection"] = objective + bonus
    return FitnessResult(
        fitness=objective + bonus,
        metrics=metrics,
        episodes=result.episodes,
        frames=result.frames,
    )


def _safe_metric(value: object) -> float:
    """Return a finite float for behavior descriptors."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        out = float(value)
        return out if math.isfinite(out) else 0.0
    return 0.0


def _descriptor_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Euclidean distance between same-length behavior descriptors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


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
