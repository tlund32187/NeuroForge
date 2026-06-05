"""Reward models built from vision-derived game metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import GameObservation

__all__ = [
    "SMB3RewardConfig",
    "SMB3RewardModel",
    "VisionMetricRewardConfig",
    "VisionMetricRewardModel",
]


@dataclass(frozen=True, slots=True)
class VisionMetricRewardConfig:
    """Weights for frame-derived game reward shaping."""

    score_scale: float = 0.01
    progress_scale: float = 1.0
    time_delta_scale: float = 0.0
    life_loss_penalty: float = -100.0
    life_gain_bonus: float = 25.0
    missing_metric_reward: float = 0.0


class VisionMetricRewardModel:
    """Reward model using only :class:`VisionGameMetrics`.

    It is intentionally tolerant of missing HUD fields: an unavailable metric
    contributes ``missing_metric_reward`` instead of failing the episode loop.
    """

    def __init__(self, cfg: VisionMetricRewardConfig | None = None) -> None:
        self._cfg = cfg or VisionMetricRewardConfig()

    def reward(self, previous: GameObservation, current: GameObservation) -> float:
        """Return reward from two frame-derived observations."""
        cfg = self._cfg
        reward = 0.0
        reward += self._delta_reward(
            previous.metrics.score,
            current.metrics.score,
            scale=cfg.score_scale,
        )
        reward += self._delta_reward(
            previous.metrics.x_progress,
            current.metrics.x_progress,
            scale=cfg.progress_scale,
        )
        reward += self._delta_reward(
            previous.metrics.time_left,
            current.metrics.time_left,
            scale=cfg.time_delta_scale,
        )

        prev_lives = previous.metrics.lives
        cur_lives = current.metrics.lives
        if prev_lives is None or cur_lives is None:
            reward += cfg.missing_metric_reward
        else:
            delta = int(cur_lives) - int(prev_lives)
            if delta < 0:
                reward += abs(delta) * cfg.life_loss_penalty
            elif delta > 0:
                reward += delta * cfg.life_gain_bonus

        return float(reward)

    def _delta_reward(
        self,
        previous: int | float | None,
        current: int | float | None,
        *,
        scale: float,
    ) -> float:
        if previous is None or current is None:
            return self._cfg.missing_metric_reward
        return (float(current) - float(previous)) * scale


def _default_base_reward() -> VisionMetricRewardConfig:
    """Base SMB3 delta-reward config tuned so forward progress dominates."""
    return VisionMetricRewardConfig(
        progress_scale=100.0,
        score_scale=0.05,
        time_delta_scale=0.0,
        life_loss_penalty=-50.0,
        life_gain_bonus=25.0,
        missing_metric_reward=0.0,
    )


@dataclass(frozen=True, slots=True)
class SMB3RewardConfig:
    """Configuration for :class:`SMB3RewardModel`."""

    base: VisionMetricRewardConfig = field(default_factory=_default_base_reward)
    idle_penalty: float = -0.5
    stall_patience: int = 240
    stall_penalty: float = -20.0
    level_clear_bonus: float = 200.0
    clear_progress_threshold: float = 0.985
    progress_epsilon: float = 1e-4


class SMB3RewardModel:
    """Dense, learnable SMB3 reward built on the vision-metric reward model."""

    def __init__(self, config: SMB3RewardConfig | None = None) -> None:
        self._cfg = config or SMB3RewardConfig()
        self._base = VisionMetricRewardModel(self._cfg.base)
        self._stall_frames = 0
        self._stall_fired = False
        self._cleared = False

    def begin_episode(self) -> None:
        """Clear per-episode stall/clear bookkeeping."""
        self._stall_frames = 0
        self._stall_fired = False
        self._cleared = False

    def reward(self, previous: GameObservation, current: GameObservation) -> float:
        """Return the shaped reward for one transition."""
        cfg = self._cfg
        value = self._base.reward(previous, current)

        delta = _progress_delta(previous, current)
        if delta is not None and delta > cfg.progress_epsilon:
            self._stall_frames = 0
            self._stall_fired = False
        else:
            self._stall_frames += 1
            value += cfg.idle_penalty
            if (
                cfg.stall_penalty != 0.0
                and not self._stall_fired
                and self._stall_frames >= cfg.stall_patience
            ):
                value += cfg.stall_penalty
                self._stall_fired = True

        if not self._cleared and self._is_clear(current):
            value += cfg.level_clear_bonus
            self._cleared = True

        return float(value)

    def _is_clear(self, current: GameObservation) -> bool:
        x = current.metrics.x_progress
        return x is not None and float(x) >= self._cfg.clear_progress_threshold


def _progress_delta(
    previous: GameObservation,
    current: GameObservation,
) -> float | None:
    """Forward ``x_progress`` change, or ``None`` if either value is missing."""
    prev_x = previous.metrics.x_progress
    cur_x = current.metrics.x_progress
    if prev_x is None or cur_x is None:
        return None
    return float(cur_x) - float(prev_x)
