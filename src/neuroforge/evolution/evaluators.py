# pyright: basic
"""Fitness evaluators for neuroevolution."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.contracts.evolution import FitnessResult, IGenome
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
    """Wrap a simple callable as an ``IFitnessEvaluator`` (any genome type)."""

    fn: Callable[[Any], float | FitnessResult]

    def evaluate(self, genome: IGenome) -> FitnessResult:
        """Evaluate *genome* with the wrapped callable."""
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
        eval_repeats: int = 1,
        reuse_client: bool = False,
    ) -> None:
        if eval_repeats < 1:
            msg = "GameTrainingFitnessEvaluator.eval_repeats must be >= 1"
            raise ValueError(msg)
        self._client_factory = client_factory
        self._base_config = base_config
        self._metric_extractor_factory = metric_extractor_factory
        self._reward_model_factory = reward_model_factory
        self._episode_manager_factory = episode_manager_factory
        self._curriculum_factory = curriculum_factory
        self._encoder_factory = encoder_factory
        self._eval_repeats = int(eval_repeats)
        # reuse_client keeps one client (e.g. a single launched emulator) alive
        # across every genome/rollout instead of cold-starting one per evaluation
        # — the fix for the per-genome emulator-launch tax. Call close() at the end.
        self._reuse_client = bool(reuse_client)
        self._shared_client: IGameClient | None = None

    def close(self) -> None:
        """Close the shared client, if one was created (no-op otherwise)."""
        if self._shared_client is not None:
            self._shared_client.close()
            self._shared_client = None

    def _acquire_client(self) -> IGameClient:
        if not self._reuse_client:
            return self._client_factory()
        if self._shared_client is None:
            self._shared_client = self._client_factory()
        return self._shared_client

    def evaluate(self, genome: IGenome) -> FitnessResult:
        """Average the genome's fitness over ``eval_repeats`` rollouts.

        Each rollout uses a content-derived seed offset by the repeat index, so a
        single noisy game episode no longer dominates selection (the #1 reason
        evolution wasn't climbing) while staying reproducible across runs.
        """
        if not (
            hasattr(genome, "to_game_training_config") and hasattr(genome, "content_key")
        ):
            msg = "GameTrainingFitnessEvaluator requires a policy/graph genome"
            raise TypeError(msg)

        base_seed = _stable_genome_seed(genome)
        rollouts = [
            self._run_once(genome, seed=base_seed + repeat)
            for repeat in range(self._eval_repeats)
        ]
        return _average_results(rollouts)

    def _run_once(self, genome: Any, *, seed: int) -> FitnessResult:
        """Run one game-training rollout and score it."""
        from neuroforge.contracts.monitors import EventTopic
        from neuroforge.tasks.game_training import GameTrainingTask

        collector = _TrialCollector()
        bus = EventBus()
        bus.subscribe(EventTopic.TRAINING_TRIAL, collector)

        config = genome.to_game_training_config(base=self._base_config, seed=seed)
        # A reused client is borrowed: the per-genome task must not close it.
        config = dataclasses.replace(config, close_client=not self._reuse_client)
        # A structural genome supplies its own compiled phenotype; a hyperparameter
        # genome leaves this None and the task builds from config as before.
        make_builder = getattr(genome, "make_network_builder", None)
        network_builder: Any = (
            make_builder(seed=seed, device=config.device, dtype=config.dtype)
            if callable(make_builder)
            else None
        )
        task = GameTrainingTask(
            config,
            event_bus=bus,
            network_builder=network_builder,
            client=self._acquire_client(),
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
        metrics = {
            "reward_mean": reward_mean,
            "max_x_progress": max_x,
            **_mean_trial_metrics(
                collector.trials,
                keys=(
                    "reward_action_energy_sum",
                    "action_button_mean",
                    "action_change_mean",
                    "action_energy_min",
                    "action_up_frac",
                    "action_down_frac",
                    "action_left_frac",
                    "action_right_frac",
                    "action_a_frac",
                    "action_b_frac",
                ),
            ),
        }
        return FitnessResult(
            fitness=fitness,
            metrics=metrics,
            episodes=result.episodes,
            frames=result.frames,
        )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _mean_trial_metrics(
    trials: list[dict[str, float]],
    *,
    keys: tuple[str, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = [trial[key] for trial in trials if key in trial]
        if values:
            out[key] = sum(values) / len(values)
    return out


def _stable_genome_seed(genome: Any) -> int:
    """Seed derived from gene *content* (not id), so identical genes evaluate alike."""
    digest = hashlib.sha256(genome.content_key().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _average_results(results: list[FitnessResult]) -> FitnessResult:
    """Mean fitness/metrics over rollouts; episodes/frames from the first rollout."""
    if not results:
        return FitnessResult(fitness=0.0)
    if len(results) == 1:
        return results[0]
    count = len(results)
    fitness = sum(item.fitness for item in results) / count
    metric_keys = {key for item in results for key in item.metrics}
    metrics = {
        key: sum(item.metrics.get(key, 0.0) for item in results) / count
        for key in metric_keys
    }
    return FitnessResult(
        fitness=fitness,
        metrics=metrics,
        episodes=results[0].episodes,
        frames=results[0].frames,
    )
