"""Vision-derived episode termination for SMB3.

The emulator client cannot know when a *level attempt* is over — game-over,
level-clear, and "hopelessly stuck" are all judgements about the pixels. This
:class:`~neuroforge.contracts.applications.games.IEpisodeManager` makes that call from the
extracted metrics alone (lives, ``x_progress``), so :class:`VisionOnlyGameLoop`
can end the episode and the curriculum can reload the level-start savestate for
a fresh attempt — the loop that turns a single boot into many learning trials.

It stays strictly within the vision-only contract: every signal comes from
:class:`VisionGameMetrics`, never emulator RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neuroforge.contracts.applications.games import EpisodeDecision

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import GameObservation

__all__ = ["SMB3EpisodeConfig", "SMB3EpisodeManager"]


@dataclass(frozen=True, slots=True)
class SMB3EpisodeConfig:
    """Configuration for :class:`SMB3EpisodeManager`."""

    terminate_on_death: bool = True
    terminate_on_clear: bool = True
    terminate_on_stall: bool = True
    clear_progress_threshold: float = 0.985
    stall_patience: int = 600           # frames of no progress (~10s at 60 fps)
    progress_epsilon: float = 1e-4      # min x_progress delta counted as progress
    lives_confidence: float = 0.5       # only trust the lives metric above this
    min_progress_frames: int = 0        # 0 disables the early poor-progress gate
    min_progress: float = 0.0


class SMB3EpisodeManager:
    """Decide when an SMB3 level attempt ends, from pixels-derived metrics only."""

    def __init__(self, config: SMB3EpisodeConfig | None = None) -> None:
        self._cfg = config or SMB3EpisodeConfig()
        self._last_lives: int | None = None
        self._stall_frames = 0
        self._episode_frames = 0
        self._episode_max_x = 0.0

    def begin_episode(self) -> None:
        """Clear per-episode death/stall bookkeeping."""
        self._last_lives = None
        self._stall_frames = 0
        self._episode_frames = 0
        self._episode_max_x = 0.0

    def should_end(
        self, before: GameObservation, after: GameObservation,
    ) -> EpisodeDecision:
        """Return the termination verdict for one transition."""
        cfg = self._cfg
        self._episode_frames += 1

        if cfg.terminate_on_death and self._died(after):
            return EpisodeDecision(terminated=True, reason="death")

        x = after.metrics.x_progress
        if x is not None:
            self._episode_max_x = max(self._episode_max_x, float(x))
        if (
            cfg.terminate_on_clear
            and x is not None
            and float(x) >= cfg.clear_progress_threshold
        ):
            return EpisodeDecision(terminated=True, reason="level_clear")

        self._update_stall(before, after)
        if cfg.terminate_on_stall and self._stall_frames >= cfg.stall_patience:
            return EpisodeDecision(terminated=True, reason="stall")

        if (
            cfg.min_progress_frames > 0
            and self._episode_frames >= cfg.min_progress_frames
            and self._episode_max_x < cfg.min_progress
        ):
            return EpisodeDecision(terminated=True, reason="min_progress")

        return EpisodeDecision(terminated=False)

    def _died(self, after: GameObservation) -> bool:
        """True on a *confident* decrease in the lives count."""
        lives = after.metrics.lives
        confidence = after.metrics.confidence.get("lives", 0.0)
        if lives is None or confidence < self._cfg.lives_confidence:
            return False
        previous = self._last_lives
        self._last_lives = int(lives)
        return previous is not None and int(lives) < previous

    def _update_stall(
        self, before: GameObservation, after: GameObservation,
    ) -> None:
        prev_x = before.metrics.x_progress
        cur_x = after.metrics.x_progress
        if prev_x is None or cur_x is None:
            self._stall_frames += 1
            return
        if float(cur_x) - float(prev_x) > self._cfg.progress_epsilon:
            self._stall_frames = 0
        else:
            self._stall_frames += 1
