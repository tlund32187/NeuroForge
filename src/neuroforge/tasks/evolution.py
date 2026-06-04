# pyright: basic
"""EvolutionTask for policy-network neuroevolution.

This task is intentionally evaluator-injected. Unit tests can evolve against a
toy objective; live SMB3 work can inject ``GameTrainingFitnessEvaluator`` once
the runtime budget is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.evolution.engine import (
    EvaluationProgress,
    EvolutionConfig,
    EvolutionEngine,
    EvolutionState,
    evolution_config_to_dict,
)
from neuroforge.tasks.base import BaseTask

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.evolution import (
        IFitnessEvaluator,
        IReproduction,
        ISpeciation,
    )
    from neuroforge.contracts.monitors import IEventBus
    from neuroforge.evolution.engine import EvaluatedGenome, GenerationSummary
    from neuroforge.evolution.genome import PolicyGenome

__all__ = ["EvolutionConfig", "EvolutionResult", "EvolutionTask"]


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """Summary returned by :meth:`EvolutionTask.run`."""

    generations: int
    evaluations: int
    best_fitness: float
    best_genome: PolicyGenome | None
    stopped: bool


class EvolutionTask(BaseTask):
    """Run population search over policy genomes."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        event_bus: IEventBus | None = None,
        *,
        stop_check: Callable[[], bool] | None = None,
        evaluator: IFitnessEvaluator | None = None,
        speciation: ISpeciation | None = None,
        reproduction: IReproduction | None = None,
        seed_population: Callable[[int, Any], list[Any]] | None = None,
    ) -> None:
        super().__init__(event_bus, stop_check)
        self._cfg = config or EvolutionConfig()
        self._evaluator = evaluator
        self._speciation = speciation
        self._reproduction = reproduction
        self._seed_population = seed_population

    def run(self) -> EvolutionResult:
        """Run evolution and return a summary."""
        if self._evaluator is None:
            msg = "EvolutionTask.run() requires a fitness evaluator"
            raise ValueError(msg)

        engine = EvolutionEngine(
            self._cfg,
            evaluator=self._evaluator,
            speciation=self._speciation,
            reproduction=self._reproduction,
            seed_population=self._seed_population,
        )
        state = engine.initial_state()
        resumed = bool(self._cfg.resume and self._cfg.checkpoint_path and state.generation > 0)
        active_config = evolution_config_to_dict(self._cfg)
        self._emit(
            "run_start",
            state.generation,
            "evolution",
            {
                "population_size": self._cfg.population_size,
                "start_population_size": len(state.population),
                "generations": self._cfg.generations,
                "resumed": resumed,
                "resume": _resume_info(
                    state,
                    requested=bool(self._cfg.resume),
                    loaded=resumed,
                    active_config=active_config,
                    requested_path=self._cfg.checkpoint_path,
                ),
            },
        )
        start_generation = state.generation
        start_population_size = len(state.population)

        summaries: list[GenerationSummary] = []
        stopped = False
        while state.generation < self._cfg.generations:
            if self._should_stop():
                stopped = True
                break
            evaluated, summary = engine.evaluate_generation(
                state,
                progress_callback=lambda progress: self._emit_evaluation_progress(
                    progress,
                    start_generation=start_generation,
                    start_population_size=start_population_size,
                    completed_before_run=state.evaluations,
                ),
            )
            summaries.append(summary)
            self._emit_individuals(evaluated, generation=summary.generation)
            self._emit_generation(summary)

            if state.generation + 1 >= self._cfg.generations:
                engine.advance(state, evaluated)
                engine.maybe_checkpoint(state)
                break
            engine.advance(state, evaluated)
            engine.maybe_checkpoint(state)

        best = state.best.genome if state.best is not None else None
        best_fitness = state.best.result.fitness if state.best is not None else float("-inf")
        data = {
            "generations": len(summaries),
            "evaluations": state.evaluations,
            "best_fitness": best_fitness,
            "best_genome_id": best.id if best is not None else "",
            "stopped": stopped,
        }
        self._emit("run_end", state.evaluations, "evolution", data)
        return EvolutionResult(
            generations=len(summaries),
            evaluations=state.evaluations,
            best_fitness=best_fitness,
            best_genome=best,
            stopped=stopped,
        )

    def _emit_evaluation_progress(
        self,
        progress: EvaluationProgress,
        *,
        start_generation: int,
        start_population_size: int,
        completed_before_run: int,
    ) -> None:
        run_completed = _run_completed_evaluations(
            generation=progress.generation,
            completed_in_generation=progress.completed,
            start_generation=start_generation,
            start_population_size=start_population_size,
            configured_population_size=self._cfg.population_size,
        )
        run_total = _run_total_evaluations(
            generations=self._cfg.generations,
            start_generation=start_generation,
            start_population_size=start_population_size,
            configured_population_size=self._cfg.population_size,
        )
        evaluations = completed_before_run + progress.completed
        data: dict[str, Any] = {
            "phase": progress.phase,
            "generation": progress.generation,
            "generations": self._cfg.generations,
            "individual": progress.individual,
            "population_size": progress.population_size,
            "genome_id": progress.genome.id,
            "completed": progress.completed,
            "evaluations": evaluations,
            "total_evaluations": completed_before_run + run_total,
            "run_evaluations": run_completed,
            "run_total_evaluations": run_total,
            "remaining_evaluations": max(0, run_total - run_completed),
        }
        if progress.result is not None:
            data.update(
                {
                    "fitness": progress.result.fitness,
                    "episodes": progress.result.episodes,
                    "frames": progress.result.frames,
                    **progress.result.metrics,
                }
            )
        if progress.error:
            data["error"] = progress.error
        self._emit("evaluation_progress", evaluations, "evolution", data)

    def _emit_individuals(
        self, evaluated: list[EvaluatedGenome], *, generation: int,
    ) -> None:
        for index, item in enumerate(evaluated):
            gene_data = _genome_gene_telemetry(item.genome)
            data = {
                "generation": generation,
                "individual": index,
                "genome_id": item.genome.id,
                "fitness": item.result.fitness,
                "adjusted_fitness": item.adjusted_fitness,
                "species_id": item.species_id,
                "parent_ids": ",".join(item.genome.parent_ids),
                "episodes": item.result.episodes,
                "frames": item.result.frames,
                **item.result.metrics,
                **gene_data,
            }
            self._emit("training_trial", generation, "evolution", data)

    def _emit_generation(self, summary: GenerationSummary) -> None:
        self._emit(
            "scalar",
            summary.generation,
            "evolution",
            {
                "generation": summary.generation,
                "best_fitness": summary.best.result.fitness,
                "mean_fitness": summary.mean_fitness,
                "species_count": summary.species_count,
                "evaluations": summary.evaluations,
                "best_genome_id": summary.best.genome.id,
            },
        )


def _genome_gene_telemetry(genome: Any) -> dict[str, Any]:
    """Per-individual gene telemetry, agnostic to genome type.

    Hyperparameter genomes report each gene; structural (graph) genomes report
    their hyperparameter genes plus the size of the invented structure.
    """
    genes = getattr(genome, "genes", None)
    if genes is not None:
        return {f"gene.{gene.key}": gene.value for gene in genes}
    out: dict[str, Any] = {}
    hyperparams = getattr(genome, "hyperparams", None)
    if hyperparams is not None:
        out.update({f"gene.{gene.key}": gene.value for gene in hyperparams})
    hidden = getattr(genome, "hidden_nodes", None)
    connections = getattr(genome, "enabled_connections", None)
    if callable(hidden):
        hidden_seq: Any = hidden()
        out["graph.hidden_nodes"] = len(hidden_seq)
    if callable(connections):
        conn_seq: Any = connections()
        out["graph.connections"] = len(conn_seq)
    return out


def _run_total_evaluations(
    *,
    generations: int,
    start_generation: int,
    start_population_size: int,
    configured_population_size: int,
) -> int:
    """Return expected evaluations remaining in this run.

    On checkpoint resume, the first generation uses the checkpoint's actual
    population. Later generations use the active config after reproduction.
    """
    remaining_generations = max(0, generations - start_generation)
    if remaining_generations == 0:
        return 0
    return max(0, start_population_size) + (
        (remaining_generations - 1) * max(0, configured_population_size)
    )


def _run_completed_evaluations(
    *,
    generation: int,
    completed_in_generation: int,
    start_generation: int,
    start_population_size: int,
    configured_population_size: int,
) -> int:
    """Return evaluations completed in this run through current progress."""
    if generation <= start_generation:
        return max(0, completed_in_generation)
    completed_after_start = max(0, generation - start_generation - 1)
    return (
        max(0, start_population_size)
        + completed_after_start * max(0, configured_population_size)
        + max(0, completed_in_generation)
    )


def _resume_info(
    state: EvolutionState,
    *,
    requested: bool,
    loaded: bool,
    active_config: dict[str, Any],
    requested_path: str | None,
) -> dict[str, Any]:
    """Return JSON-safe checkpoint resume context for run logs."""
    best_fitness = float(state.best.result.fitness) if loaded and state.best is not None else 0.0
    checkpoint_config = dict(state.checkpoint_config) if loaded else {}
    return {
        "requested": requested,
        "loaded": loaded,
        "path": state.checkpoint_path if loaded else (requested_path or ""),
        "schema_version": int(state.checkpoint_schema_version) if loaded else 0,
        "rng_state_restored": bool(state.rng_state_restored) if loaded else False,
        "generation": int(state.generation) if loaded else 0,
        "evaluations": int(state.evaluations) if loaded else 0,
        "population_size": len(state.population) if loaded else 0,
        "best_genome_id": state.best.genome.id if loaded and state.best is not None else "",
        "best_fitness": best_fitness,
        "checkpoint_config": checkpoint_config,
        "config_differences": _config_differences(checkpoint_config, active_config),
    }


def _config_differences(
    checkpoint_config: dict[str, Any],
    active_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return active config values that differ from the checkpoint config."""
    differences: dict[str, dict[str, Any]] = {}
    for key, checkpoint_value in checkpoint_config.items():
        if key not in active_config:
            continue
        active_value = active_config[key]
        if active_value != checkpoint_value:
            differences[key] = {
                "checkpoint": checkpoint_value,
                "active": active_value,
            }
    return differences
