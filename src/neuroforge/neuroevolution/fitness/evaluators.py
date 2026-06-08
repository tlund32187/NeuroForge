"""Fitness evaluators for neuroevolution."""

from __future__ import annotations

import dataclasses
import hashlib
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from neuroforge.contracts.applications.evolution import (
    FitnessResult,
    IFitnessEvaluator,
    IGenome,
)
from neuroforge.messaging.bus import EventBus
from neuroforge.neuroevolution.io.learned_state import learned_checkpoint_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.applications.tasks.game_training import GameTrainingConfig
    from neuroforge.contracts.applications.games import (
        IEpisodeManager,
        IFrameMetricExtractor,
        IGameClient,
        IRewardModel,
    )
    from neuroforge.perception.vision.encoding.frame_encoder import IFrameEncoder

__all__ = [
    "CallableFitnessEvaluator",
    "GameTrainingFitnessEvaluator",
    "ThreadLocalFitnessEvaluatorPool",
]


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


class ThreadLocalFitnessEvaluatorPool:
    """Lease one evaluator per concurrent worker and keep them reusable.

    The wrapper is generic: callers provide a ``worker_index -> evaluator``
    factory, so reusable neuroevolution stays independent of SMB3/BizHawk while
    live SMB3 can assign one emulator port to each worker.
    """

    def __init__(
        self,
        factory: Callable[[int], IFitnessEvaluator],
        *,
        max_workers: int,
    ) -> None:
        if max_workers < 1:
            msg = "ThreadLocalFitnessEvaluatorPool.max_workers must be >= 1"
            raise ValueError(msg)
        self._factory = factory
        self._max_workers = int(max_workers)
        self._available: queue.LifoQueue[IFitnessEvaluator] = queue.LifoQueue()
        self._created: list[IFitnessEvaluator] = []
        self._lock = threading.Lock()

    @property
    def max_workers(self) -> int:
        """Maximum number of worker evaluators this pool will create."""
        return self._max_workers

    @property
    def created_count(self) -> int:
        """Number of evaluators created so far."""
        with self._lock:
            return len(self._created)

    def evaluate(self, genome: IGenome) -> FitnessResult:
        """Evaluate *genome* with an exclusive worker evaluator."""
        evaluator = self._acquire()
        try:
            return evaluator.evaluate(genome)
        finally:
            self._available.put(evaluator)

    def close(self) -> None:
        """Close every evaluator that exposes a close hook."""
        with self._lock:
            created = tuple(self._created)
            self._created.clear()
        while True:
            try:
                self._available.get_nowait()
            except queue.Empty:
                break
        for evaluator in created:
            close = getattr(evaluator, "close", None)
            if callable(close):
                close()

    def _acquire(self) -> IFitnessEvaluator:
        try:
            return self._available.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            worker_index = len(self._created)
            if worker_index < self._max_workers:
                evaluator = self._factory(worker_index)
                self._created.append(evaluator)
                return evaluator
        return self._available.get()


class _TrialCollector:
    enabled = True

    def __init__(self) -> None:
        self.trials: list[dict[str, float]] = []

    def on_event(self, event: object) -> None:
        if getattr(getattr(event, "topic", None), "value", None) != "training_trial":
            return
        raw = getattr(event, "data", {})
        if isinstance(raw, dict):
            data = cast("dict[str, Any]", raw)
            self.trials.append({str(k): float(v) for k, v in data.items() if _is_number(v)})

    def reset(self) -> None:
        self.trials.clear()

    def snapshot(self) -> dict[str, object]:
        return {"trials": list(self.trials)}


class GameTrainingFitnessEvaluator:
    """Evaluate a policy genome by running ``GameTrainingTask`` episodes.

    The evaluator owns no emulator-specific state. Callers inject factories so
    live use can create a concrete client per genome, while tests can use scripted
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
        curriculum_factory: Callable[[], Any] | None = None,
        encoder_factory: Callable[[], IFrameEncoder] | None = None,
        eval_repeats: int = 1,
        reuse_client: bool = False,
        config_transform: (
            Callable[[GameTrainingConfig, IGenome], GameTrainingConfig] | None
        ) = None,
        progress_scale: float = 100.0,
        score_gain_scale: float = 0.0,
        survival_scale: float = 0.0,
        durable_progress_weight: float = 0.0,
        death_penalty: float = 0.0,
        stall_penalty: float = 0.0,
        min_progress_penalty: float = 0.0,
        level_clear_bonus: float = 0.0,
        button_overuse_penalty: float = 0.0,
        button_overuse_threshold: float = 2.0,
        horizontal_conflict_penalty: float = 0.0,
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
        self._config_transform = config_transform
        # reuse_client keeps one client (e.g. a single launched emulator) alive
        # across every genome/rollout instead of cold-starting one per evaluation
        # - the fix for the per-genome emulator-launch tax. Call close() at the end.
        self._reuse_client = bool(reuse_client)
        self._shared_client: IGameClient | None = None
        self._progress_scale = float(progress_scale)
        self._score_gain_scale = float(score_gain_scale)
        self._survival_scale = float(survival_scale)
        self._durable_progress_weight = float(durable_progress_weight)
        self._death_penalty = float(death_penalty)
        self._stall_penalty = float(stall_penalty)
        self._min_progress_penalty = float(min_progress_penalty)
        self._level_clear_bonus = float(level_clear_bonus)
        self._button_overuse_penalty = float(button_overuse_penalty)
        self._button_overuse_threshold = max(0.0, float(button_overuse_threshold))
        self._horizontal_conflict_penalty = float(horizontal_conflict_penalty)

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

        started = time.perf_counter()
        base_seed = _stable_genome_seed(genome)
        rollouts = [
            self._run_once(genome, seed=base_seed + repeat)
            for repeat in range(self._eval_repeats)
        ]
        result = _average_results(rollouts)
        elapsed = time.perf_counter() - started
        return dataclasses.replace(
            result,
            metrics={
                **result.metrics,
                "evaluation_wall_seconds": elapsed,
                "evaluation_repeats": float(self._eval_repeats),
                "evaluation_cache_hit": 0.0,
            },
        )

    def _run_once(self, genome: Any, *, seed: int) -> FitnessResult:
        """Run one game-training rollout and score it."""
        from neuroforge.applications.tasks.game_training import GameTrainingTask
        from neuroforge.contracts.messaging import EventTopic

        collector = _TrialCollector()
        bus = EventBus()
        bus.subscribe(EventTopic.TRAINING_TRIAL, collector)

        config = genome.to_game_training_config(base=self._base_config, seed=seed)
        genome_decide_ticks = _genome_value(genome, "decide_ticks")
        if self._config_transform is not None:
            config = self._config_transform(config, genome)
        lamarckian_path = learned_checkpoint_path(genome)
        lamarckian_available = bool(lamarckian_path and Path(lamarckian_path).exists())
        if lamarckian_available:
            config = dataclasses.replace(
                config,
                resume=True,
                resume_checkpoint_path=lamarckian_path,
                resume_allow_partial=True,
            )
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
        started = time.perf_counter()
        result = task.run()
        elapsed = time.perf_counter() - started
        reward_means = [trial.get("reward_mean", 0.0) for trial in collector.trials]
        max_x = max((trial.get("max_x_progress", 0.0) for trial in collector.trials), default=0.0)
        reward_mean = sum(reward_means) / max(1, len(reward_means))
        trial_metrics = _mean_trial_metrics(
            collector.trials,
            keys=_numeric_trial_keys(collector.trials),
        )
        expected_frames = max(1, config.max_episodes * config.frames_per_episode)
        survival_frac = min(1.0, max(0.0, result.frames / expected_frames))
        progress_score = max_x * self._progress_scale
        score_gain_total = sum(
            max(0.0, trial.get("score_gain", 0.0)) for trial in collector.trials
        )
        score_gain_score = score_gain_total * self._score_gain_scale
        durable_progress_bonus = (
            progress_score * survival_frac * self._durable_progress_weight
        )
        survival_score = survival_frac * self._survival_scale
        terminal_score = (
            trial_metrics.get("termination.level_clear", 0.0) * self._level_clear_bonus
            - trial_metrics.get("termination.death", 0.0) * self._death_penalty
            - trial_metrics.get("termination.stall", 0.0) * self._stall_penalty
            - trial_metrics.get("termination.min_progress", 0.0)
            * self._min_progress_penalty
        )
        button_overuse = max(
            0.0,
            trial_metrics.get("action_button_mean", 0.0) - self._button_overuse_threshold,
        )
        button_overuse_score = -button_overuse * self._button_overuse_penalty
        horizontal_conflict = min(
            trial_metrics.get("action_left_frac", 0.0),
            trial_metrics.get("action_right_frac", 0.0),
        )
        horizontal_conflict_score = -horizontal_conflict * self._horizontal_conflict_penalty
        fitness = (
            reward_mean
            + progress_score
            + score_gain_score
            + durable_progress_bonus
            + survival_score
            + terminal_score
            + button_overuse_score
            + horizontal_conflict_score
        )
        metrics = {
            "reward_mean": reward_mean,
            "max_x_progress": max_x,
            "fitness_progress_score": progress_score,
            "fitness_score_gain_score": score_gain_score,
            "fitness_survival_score": survival_score,
            "fitness_durable_progress_bonus": durable_progress_bonus,
            "fitness_terminal_score": terminal_score,
            "fitness_button_overuse_score": button_overuse_score,
            "fitness_horizontal_conflict_score": horizontal_conflict_score,
            "action_button_overuse": button_overuse,
            "action_horizontal_conflict": horizontal_conflict,
            "score_gain_total": score_gain_total,
            "survival_frac": survival_frac,
            "rollout_wall_seconds": elapsed,
            "rollout_fps": result.frames / elapsed if elapsed > 0.0 else 0.0,
            "effective_decide_ticks": float(config.decide_ticks),
            "lamarckian_resume_requested": 1.0 if lamarckian_path else 0.0,
            "lamarckian_resume_available": 1.0 if lamarckian_available else 0.0,
        }
        metrics.update(_non_colliding_metrics(trial_metrics, existing=metrics))
        if genome_decide_ticks is not None:
            metrics["genome_decide_ticks"] = genome_decide_ticks
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


def _numeric_trial_keys(trials: list[dict[str, float]]) -> tuple[str, ...]:
    """Return stable numeric trial keys that should be averaged into fitness metrics."""
    return tuple(sorted({key for trial in trials for key in trial}))


def _non_colliding_metrics(
    metrics: dict[str, float],
    *,
    existing: dict[str, float],
) -> dict[str, float]:
    """Preserve primary rollout metrics and suffix averaged trial collisions."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        out[key if key not in existing else f"{key}_mean"] = value
    return out


def _stable_genome_seed(genome: Any) -> int:
    """Seed derived from gene *content* (not id), so identical genes evaluate alike."""
    digest = hashlib.sha256(genome.content_key().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _genome_value(genome: Any, key: str) -> float | None:
    value = getattr(genome, "value", None)
    if not callable(value):
        return None
    try:
        raw = value(key)
    except KeyError:
        return None
    if isinstance(raw, bool):
        return float(int(raw))
    if isinstance(raw, int | float):
        return float(raw)
    return None


def _average_results(results: list[FitnessResult]) -> FitnessResult:
    """Mean fitness/metrics over rollouts, including repeat variance telemetry."""
    if not results:
        return FitnessResult(fitness=0.0)
    count = len(results)
    fitness = sum(item.fitness for item in results) / count
    metric_keys = {key for item in results for key in item.metrics}
    metrics = {
        key: sum(item.metrics.get(key, 0.0) for item in results) / count
        for key in metric_keys
    }
    fitness_values = [item.fitness for item in results]
    frame_values = [float(item.frames) for item in results]
    episode_values = [float(item.episodes) for item in results]
    frames_total = sum(frame_values)
    episodes_total = sum(episode_values)
    metrics.update(
        {
            "rollout_count": float(count),
            "rollout_frames_total": frames_total,
            "rollout_episodes_total": episodes_total,
            "fitness_std": _std(fitness_values),
            "fitness_min": min(fitness_values),
            "fitness_max": max(fitness_values),
            "frames_mean": sum(frame_values) / count,
            "frames_std": _std(frame_values),
            "frames_min": min(frame_values),
            "frames_max": max(frame_values),
            "episodes_mean": sum(episode_values) / count,
            "episodes_std": _std(episode_values),
        }
    )
    for key in metric_keys:
        values = [item.metrics[key] for item in results if key in item.metrics]
        if values:
            metrics[f"{key}_std"] = _std(values)
            metrics[f"{key}_min"] = min(values)
            metrics[f"{key}_max"] = max(values)
    return FitnessResult(
        fitness=fitness,
        metrics=metrics,
        episodes=int(episodes_total),
        frames=int(frames_total),
    )


def _std(values: list[float]) -> float:
    """Return population standard deviation for small rollout samples."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)
