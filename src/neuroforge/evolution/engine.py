# pyright: basic
"""Small, deterministic neuroevolution engine.

This is the first working Track E engine: it evaluates a population of
``PolicyGenome`` instances, groups them into compatibility species, applies
fitness sharing, and reproduces the next generation through elitism,
crossover, and mutation.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuroforge.contracts.evolution import FitnessResult, IFitnessEvaluator
from neuroforge.evolution.genome import PolicyGenome

__all__ = [
    "EvaluatedGenome",
    "EvolutionConfig",
    "EvolutionEngine",
    "EvolutionState",
    "GenerationSummary",
    "SimpleSpeciation",
]


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
    """A genome plus fitness and species assignment."""

    genome: PolicyGenome
    result: FitnessResult
    species_id: int
    adjusted_fitness: float


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
    population: list[PolicyGenome]
    best: EvaluatedGenome | None = None
    evaluations: int = 0


@dataclass(slots=True)
class _Species:
    id: int
    representative: PolicyGenome
    members: list[PolicyGenome] = field(default_factory=list)


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


class EvolutionEngine:
    """Evaluate and reproduce a population of policy genomes."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        *,
        evaluator: IFitnessEvaluator,
    ) -> None:
        self._cfg = config or EvolutionConfig()
        self._evaluator = evaluator
        self._rng = random.Random(self._cfg.seed)
        self._speciation = SimpleSpeciation(threshold=self._cfg.species_threshold)

    def initial_state(self) -> EvolutionState:
        """Create a fresh population or load one from checkpoint."""
        checkpoint = self._checkpoint_path()
        if self._cfg.resume and checkpoint is not None and checkpoint.exists():
            return self.load_state(checkpoint)
        population = [
            PolicyGenome.seed(
                f"g0_{idx}",
                generation=0,
                rng=self._rng,
                randomise=idx != 0,
            )
            for idx in range(self._cfg.population_size)
        ]
        return EvolutionState(generation=0, population=population)

    def evaluate_generation(
        self, state: EvolutionState,
    ) -> tuple[list[EvaluatedGenome], GenerationSummary]:
        """Evaluate the current population and return sorted results."""
        assignments = self._speciation.assign(state.population)
        species_sizes: dict[int, int] = {}
        for species_id in assignments.values():
            species_sizes[species_id] = species_sizes.get(species_id, 0) + 1

        results = self._evaluate_population(state.population)
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

    def _evaluate_population(self, population: list[PolicyGenome]) -> list[FitnessResult]:
        """Evaluate genomes, optionally in parallel, preserving population order."""
        if self._cfg.max_workers <= 1:
            return [self._evaluator.evaluate(genome) for genome in population]

        def _evaluate(genome: PolicyGenome) -> FitnessResult:
            return self._evaluator.evaluate(genome)

        with ThreadPoolExecutor(max_workers=self._cfg.max_workers) as pool:
            return list(pool.map(_evaluate, population))

    def reproduce(
        self, evaluated: list[EvaluatedGenome], *, generation: int,
    ) -> list[PolicyGenome]:
        """Create the next population."""
        cfg = self._cfg
        elites = [
            PolicyGenome(
                id=f"g{generation}_{idx}",
                generation=generation,
                genes=item.genome.genes,
                parent_ids=(item.genome.id,),
            )
            for idx, item in enumerate(evaluated[: cfg.elite_count])
        ]
        children = list(elites)
        while len(children) < cfg.population_size:
            child_index = len(children)
            parent_a = self._select_parent(evaluated)
            if self._rng.random() < cfg.crossover_rate:
                parent_b = self._select_parent(evaluated)
                child = parent_a.genome.crossover(
                    parent_b.genome,
                    child_id=f"g{generation}_{child_index}",
                    generation=generation,
                    rng=self._rng,
                )
            else:
                child = PolicyGenome(
                    id=f"g{generation}_{child_index}",
                    generation=generation,
                    genes=parent_a.genome.genes,
                    parent_ids=(parent_a.genome.id,),
                )
            children.append(
                child.mutate(
                    child_id=child.id,
                    generation=generation,
                    rng=self._rng,
                    rate=cfg.mutation_rate,
                    power=cfg.mutation_power,
                )
            )
        return children

    def advance(self, state: EvolutionState, evaluated: list[EvaluatedGenome]) -> None:
        """Mutate *state* to the next generation."""
        next_generation = state.generation + 1
        state.population = self.reproduce(evaluated, generation=next_generation)
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
            "generation": state.generation,
            "evaluations": state.evaluations,
            "population": [genome.to_dict() for genome in state.population],
            "best": best_payload,
        }

    def load_state(self, path: str | Path) -> EvolutionState:
        """Load state from a JSON checkpoint."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_population = payload.get("population", [])
        if not isinstance(raw_population, list):
            msg = "evolution checkpoint population must be a list"
            raise ValueError(msg)
        population = [PolicyGenome.from_dict(cast_dict(item)) for item in raw_population]
        state = EvolutionState(
            generation=int(payload.get("generation", 0)),
            population=population,
            evaluations=int(payload.get("evaluations", 0)),
        )
        best_raw = payload.get("best")
        if isinstance(best_raw, dict):
            genome = PolicyGenome.from_dict(cast_dict(best_raw["genome"]))
            result = FitnessResult(
                fitness=float(best_raw["fitness"]),
                metrics={
                    str(key): float(value)
                    for key, value in cast_dict(best_raw.get("metrics", {})).items()
                },
                episodes=int(best_raw.get("episodes", 0)),
                frames=int(best_raw.get("frames", 0)),
            )
            state.best = EvaluatedGenome(
                genome=genome,
                result=result,
                species_id=int(best_raw.get("species_id", 0)),
                adjusted_fitness=float(best_raw.get("adjusted_fitness", result.fitness)),
            )
        return state

    def _select_parent(self, evaluated: list[EvaluatedGenome]) -> EvaluatedGenome:
        min_fit = min(item.adjusted_fitness for item in evaluated)
        weights = [item.adjusted_fitness - min_fit + 1e-9 for item in evaluated]
        total = sum(weights)
        if total <= 0.0:
            return self._rng.choice(evaluated)
        pick = self._rng.random() * total
        acc = 0.0
        for item, weight in zip(evaluated, weights, strict=True):
            acc += weight
            if acc >= pick:
                return item
        return evaluated[-1]

    def _checkpoint_path(self) -> Path | None:
        if self._cfg.checkpoint_path is None:
            return None
        return Path(self._cfg.checkpoint_path)


def cast_dict(value: Any) -> dict[str, Any]:
    """Validate a decoded JSON object."""
    if not isinstance(value, dict):
        msg = "expected JSON object"
        raise ValueError(msg)
    return {str(key): item for key, item in value.items()}
