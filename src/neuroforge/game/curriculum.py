"""Savestate curriculum for SMB3 training.

A curriculum decides *where each episode begins*. Booting from the ROM drops
the brain at the title screen, where there is no reward to learn from; a
curriculum instead starts each episode from a savestate placed at the start of
a level, so learning happens where the vision reward is dense. As the brain
masters a stage it can be advanced to a harder savestate.

Loading a savestate is an *environment reset* (like resetting a Gym env to a
start state) — it is not memory reading, and the policy still only ever sees
pixels. The curriculum yields a savestate path; the BizHawk client loads it on
the next reset via ``queue_savestate``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ICurriculum", "SMB3Curriculum"]


@runtime_checkable
class ICurriculum(Protocol):
    """Choose the savestate each episode starts from and track mastery."""

    def savestate_for(self, episode: int) -> str | None:
        """Return the savestate path for *episode*, or ``None`` to boot the ROM."""
        ...

    def report_episode(self, x_progress: float) -> None:
        """Record how far the just-finished episode reached (furthest x_progress)."""
        ...


class SMB3Curriculum:
    """Advance through increasingly hard savestates as the brain masters each.

    Stays on a stage until the brain has had at least ``min_episodes_per_stage``
    attempts *and* its best episode reached ``advance_threshold`` of the level,
    then steps to the next savestate. With no stages it yields ``None`` (boot
    the ROM), preserving the pre-curriculum behaviour.
    """

    def __init__(
        self,
        stages: Sequence[str] = (),
        *,
        advance_threshold: float = 0.9,
        min_episodes_per_stage: int = 3,
    ) -> None:
        if not 0.0 <= advance_threshold <= 1.0:
            msg = "advance_threshold must be in [0, 1]"
            raise ValueError(msg)
        if min_episodes_per_stage < 1:
            msg = "min_episodes_per_stage must be >= 1"
            raise ValueError(msg)
        self._stages = tuple(stages)
        self._advance_threshold = float(advance_threshold)
        self._min_episodes = int(min_episodes_per_stage)
        self._stage = 0
        self._episodes_in_stage = 0
        self._best_in_stage = 0.0

    @property
    def stage(self) -> int:
        """Index of the current curriculum stage."""
        return self._stage

    @property
    def num_stages(self) -> int:
        """Number of configured savestate stages."""
        return len(self._stages)

    def savestate_for(self, episode: int) -> str | None:  # noqa: ARG002
        """Return the current stage's savestate, or ``None`` if there are none."""
        if not self._stages:
            return None
        return self._stages[min(self._stage, len(self._stages) - 1)]

    def report_episode(self, x_progress: float) -> None:
        """Update mastery for the current stage and advance when earned."""
        self._episodes_in_stage += 1
        self._best_in_stage = max(self._best_in_stage, float(x_progress))
        if (
            self._stage < len(self._stages) - 1
            and self._episodes_in_stage >= self._min_episodes
            and self._best_in_stage >= self._advance_threshold
        ):
            self._stage += 1
            self._episodes_in_stage = 0
            self._best_in_stage = 0.0
