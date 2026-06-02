"""Reward models built from vision-derived game metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.contracts.game import GameObservation

__all__ = ["VisionMetricRewardConfig", "VisionMetricRewardModel"]


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
