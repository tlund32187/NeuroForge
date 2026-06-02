"""SMB3-specific reward shaping for learnable in-level progress.

The base :class:`~neuroforge.game.rewards.VisionMetricRewardModel` scores the
stateless per-frame deltas it does well — forward ``x_progress``, score, and
life loss/gain. This model *composes* it (it does not fork or reimplement that
logic) and adds the **stateful** shaping SMB3 needs to actually learn:

* a per-frame **idle penalty** when no forward progress is made, so "stand
  still" is never a local optimum;
* a one-shot **anti-stall penalty** once a brain has been stuck for a while;
* a one-shot **level-clear bonus** when ``x_progress`` reaches the far end.

``x_progress`` is the *furthest* point reached (backtracking earns no credit),
so the progress term is already farming-proof. Everything is configurable; the
defaults are sensible starting points and expected to need tuning (Phase 4 is
the highest-learnability-risk phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from neuroforge.game.rewards import VisionMetricRewardConfig, VisionMetricRewardModel

if TYPE_CHECKING:
    from neuroforge.contracts.game import GameObservation

__all__ = ["SMB3RewardConfig", "SMB3RewardModel"]


def _default_base_reward() -> VisionMetricRewardConfig:
    """Base delta-reward config tuned so forward progress dominates."""
    return VisionMetricRewardConfig(
        progress_scale=100.0,      # reward per unit of furthest x_progress (0..1)
        score_scale=0.05,          # reward per game-score point gained
        time_delta_scale=0.0,      # time is informative but not directly rewarded
        life_loss_penalty=-50.0,   # death (per life lost; lives are confidence-gated)
        life_gain_bonus=25.0,      # a 1-up
        missing_metric_reward=0.0,
    )


@dataclass(frozen=True, slots=True)
class SMB3RewardConfig:
    """Configuration for :class:`SMB3RewardModel`."""

    base: VisionMetricRewardConfig = field(default_factory=_default_base_reward)
    idle_penalty: float = -0.5          # per frame with no forward progress
    stall_patience: int = 240           # frames of no progress before the extra hit
    stall_penalty: float = -20.0        # one-shot when stall_patience is reached
    level_clear_bonus: float = 200.0    # one-shot when the level end is reached
    clear_progress_threshold: float = 0.985
    progress_epsilon: float = 1e-4      # min x_progress delta counted as progress


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
    previous: GameObservation, current: GameObservation,
) -> float | None:
    """Forward ``x_progress`` change, or ``None`` if either value is missing."""
    prev_x = previous.metrics.x_progress
    cur_x = current.metrics.x_progress
    if prev_x is None or cur_x is None:
        return None
    return float(cur_x) - float(prev_x)
