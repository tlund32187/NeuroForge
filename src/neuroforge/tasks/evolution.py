# pyright: basic
"""EvolutionTask for policy-network neuroevolution.

This task is intentionally evaluator-injected. Unit tests can evolve against a
toy objective; live SMB3 work can inject ``GameTrainingFitnessEvaluator`` once
the runtime budget is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neuroforge.evolution.engine import EvolutionConfig, EvolutionEngine
from neuroforge.tasks.base import BaseTask

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.evolution import IFitnessEvaluator
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
    ) -> None:
        super().__init__(event_bus, stop_check)
        self._cfg = config or EvolutionConfig()
        self._evaluator = evaluator

    def run(self) -> EvolutionResult:
        """Run evolution and return a summary."""
        if self._evaluator is None:
            msg = "EvolutionTask.run() requires a fitness evaluator"
            raise ValueError(msg)

        engine = EvolutionEngine(self._cfg, evaluator=self._evaluator)
        state = engine.initial_state()
        resumed = bool(self._cfg.resume and self._cfg.checkpoint_path and state.generation > 0)
        self._emit(
            "run_start",
            state.generation,
            "evolution",
            {
                "population_size": self._cfg.population_size,
                "generations": self._cfg.generations,
                "resumed": resumed,
            },
        )

        summaries: list[GenerationSummary] = []
        stopped = False
        while state.generation < self._cfg.generations:
            if self._should_stop():
                stopped = True
                break
            evaluated, summary = engine.evaluate_generation(state)
            summaries.append(summary)
            self._emit_individuals(evaluated, generation=summary.generation)
            self._emit_generation(summary)

            if state.generation + 1 >= self._cfg.generations:
                state.generation += 1
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

    def _emit_individuals(
        self, evaluated: list[EvaluatedGenome], *, generation: int,
    ) -> None:
        for index, item in enumerate(evaluated):
            gene_data = {f"gene.{gene.key}": gene.value for gene in item.genome.genes}
            data = {
                "generation": generation,
                "individual": index,
                "genome_id": item.genome.id,
                "fitness": item.result.fitness,
                "adjusted_fitness": item.adjusted_fitness,
                "species_id": item.species_id,
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
