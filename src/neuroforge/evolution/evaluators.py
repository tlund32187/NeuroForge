# pyright: basic
"""Fitness evaluators for neuroevolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neuroforge.contracts.evolution import FitnessResult, IGenome
from neuroforge.evolution.genome import PolicyGenome
from neuroforge.monitors.bus import EventBus

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.game import (
        IEpisodeManager,
        IFrameMetricExtractor,
        IGameClient,
        IRewardModel,
    )
    from neuroforge.game.curriculum import ICurriculum
    from neuroforge.game.policies.encoder import IFrameEncoder
    from neuroforge.tasks.game_training import GameTrainingConfig

__all__ = ["CallableFitnessEvaluator", "GameTrainingFitnessEvaluator"]


@dataclass(slots=True)
class CallableFitnessEvaluator:
    """Wrap a simple callable as an ``IFitnessEvaluator``."""

    fn: Callable[[PolicyGenome], float | FitnessResult]

    def evaluate(self, genome: IGenome) -> FitnessResult:
        """Evaluate *genome* with the wrapped callable."""
        if not isinstance(genome, PolicyGenome):
            msg = "CallableFitnessEvaluator requires a PolicyGenome"
            raise TypeError(msg)
        result = self.fn(genome)
        if isinstance(result, FitnessResult):
            return result
        return FitnessResult(fitness=float(result))


class _TrialCollector:
    enabled = True

    def __init__(self) -> None:
        self.trials: list[dict[str, float]] = []

    def on_event(self, event: object) -> None:
        if getattr(getattr(event, "topic", None), "value", None) != "training_trial":
            return
        raw = getattr(event, "data", {})
        if isinstance(raw, dict):
            self.trials.append({str(k): float(v) for k, v in raw.items() if _is_number(v)})

    def reset(self) -> None:
        self.trials.clear()

    def snapshot(self) -> dict[str, object]:
        return {"trials": list(self.trials)}


class GameTrainingFitnessEvaluator:
    """Evaluate a policy genome by running ``GameTrainingTask`` episodes.

    The evaluator owns no emulator-specific state. Callers inject factories so
    live use can create a BizHawk client per genome, while tests can use scripted
    clients. Fitness defaults to ``mean(reward_mean) + max_x_progress * 100``.
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[], IGameClient],
        base_config: GameTrainingConfig | None = None,
        metric_extractor_factory: Callable[[], IFrameMetricExtractor] | None = None,
        reward_model_factory: Callable[[], IRewardModel] | None = None,
        episode_manager_factory: Callable[[], IEpisodeManager] | None = None,
        curriculum_factory: Callable[[], ICurriculum] | None = None,
        encoder_factory: Callable[[], IFrameEncoder] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._base_config = base_config
        self._metric_extractor_factory = metric_extractor_factory
        self._reward_model_factory = reward_model_factory
        self._episode_manager_factory = episode_manager_factory
        self._curriculum_factory = curriculum_factory
        self._encoder_factory = encoder_factory

    def evaluate(self, genome: IGenome) -> FitnessResult:
        """Run the genome through the game training loop and return fitness."""
        if not isinstance(genome, PolicyGenome):
            msg = "GameTrainingFitnessEvaluator requires a PolicyGenome"
            raise TypeError(msg)
        from neuroforge.contracts.monitors import EventTopic
        from neuroforge.tasks.game_training import GameTrainingTask

        collector = _TrialCollector()
        bus = EventBus()
        bus.subscribe(EventTopic.TRAINING_TRIAL, collector)

        config = genome.to_game_training_config(
            base=self._base_config,
            seed=_stable_genome_seed(genome),
        )
        task = GameTrainingTask(
            config,
            event_bus=bus,
            client=self._client_factory(),
            metric_extractor=(
                self._metric_extractor_factory() if self._metric_extractor_factory else None
            ),
            reward_model=self._reward_model_factory() if self._reward_model_factory else None,
            episode_manager=(
                self._episode_manager_factory() if self._episode_manager_factory else None
            ),
            curriculum=self._curriculum_factory() if self._curriculum_factory else None,
            encoder=self._encoder_factory() if self._encoder_factory else None,
        )
        result = task.run()
        reward_means = [trial.get("reward_mean", 0.0) for trial in collector.trials]
        max_x = max((trial.get("max_x_progress", 0.0) for trial in collector.trials), default=0.0)
        reward_mean = sum(reward_means) / max(1, len(reward_means))
        fitness = reward_mean + max_x * 100.0
        return FitnessResult(
            fitness=fitness,
            metrics={"reward_mean": reward_mean, "max_x_progress": max_x},
            episodes=result.episodes,
            frames=result.frames,
        )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _stable_genome_seed(genome: PolicyGenome) -> int:
    total = genome.generation * 100_003
    for index, char in enumerate(genome.id):
        total += (index + 1) * ord(char)
    return total % (2**31 - 1)
