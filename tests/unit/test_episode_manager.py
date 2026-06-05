"""Tests for vision-derived SMB3 episode termination (Phase 4)."""

from __future__ import annotations

import pytest

from neuroforge.contracts.applications.games import (
    GameObservation,
    IEpisodeManager,
    ScreenFrame,
    VisionGameMetrics,
)
from neuroforge.environments.games.smb3.episode import SMB3EpisodeConfig, SMB3EpisodeManager


def _obs(
    *,
    x: float | None = None,
    lives: int | None = None,
    lives_conf: float = 1.0,
) -> GameObservation:
    frame = ScreenFrame(width=1, height=1, channels=1, data=b"\x00")
    confidence = {"lives": lives_conf} if lives is not None else {}
    metrics = VisionGameMetrics(lives=lives, x_progress=x, confidence=confidence)
    return GameObservation(step=0, t=0.0, frame=frame, metrics=metrics)


@pytest.mark.unit
def test_is_episode_manager() -> None:
    assert isinstance(SMB3EpisodeManager(), IEpisodeManager)


@pytest.mark.unit
def test_no_termination_under_normal_progress() -> None:
    mgr = SMB3EpisodeManager()
    mgr.begin_episode()
    decision = mgr.should_end(_obs(x=0.1, lives=3), _obs(x=0.2, lives=3))
    assert decision.terminated is False


@pytest.mark.unit
def test_confident_life_loss_terminates_as_death() -> None:
    mgr = SMB3EpisodeManager()
    mgr.begin_episode()
    mgr.should_end(_obs(x=0.0, lives=3), _obs(x=0.1, lives=3))  # seeds last_lives=3
    decision = mgr.should_end(_obs(x=0.1, lives=3), _obs(x=0.1, lives=2))
    assert decision.terminated is True
    assert decision.reason == "death"


@pytest.mark.unit
def test_low_confidence_life_reading_is_ignored() -> None:
    mgr = SMB3EpisodeManager(SMB3EpisodeConfig(lives_confidence=0.5))
    mgr.begin_episode()
    mgr.should_end(_obs(x=0.0, lives=3), _obs(x=0.1, lives=3))
    # lives "drops" but the reading is not trusted → no death.
    decision = mgr.should_end(_obs(x=0.1, lives=3), _obs(x=0.2, lives=1, lives_conf=0.2))
    assert decision.terminated is False


@pytest.mark.unit
def test_reaching_the_end_terminates_as_level_clear() -> None:
    mgr = SMB3EpisodeManager(SMB3EpisodeConfig(clear_progress_threshold=0.985))
    mgr.begin_episode()
    decision = mgr.should_end(_obs(x=0.9), _obs(x=0.99))
    assert decision.terminated is True
    assert decision.reason == "level_clear"


@pytest.mark.unit
def test_prolonged_stall_terminates() -> None:
    mgr = SMB3EpisodeManager(SMB3EpisodeConfig(stall_patience=3))
    mgr.begin_episode()
    terminated = [mgr.should_end(_obs(x=0.5), _obs(x=0.5)).terminated for _ in range(3)]
    assert terminated == [False, False, True]


@pytest.mark.unit
def test_stall_can_be_disabled() -> None:
    mgr = SMB3EpisodeManager(SMB3EpisodeConfig(terminate_on_stall=False, stall_patience=2))
    mgr.begin_episode()
    terminated = [mgr.should_end(_obs(x=0.5), _obs(x=0.5)).terminated for _ in range(5)]
    assert not any(terminated)


@pytest.mark.unit
def test_min_progress_gate_terminates_poor_runs() -> None:
    mgr = SMB3EpisodeManager(
        SMB3EpisodeConfig(
            stall_patience=100,
            min_progress_frames=3,
            min_progress=0.05,
        )
    )
    mgr.begin_episode()

    decisions = [mgr.should_end(_obs(x=0.0), _obs(x=0.02)) for _ in range(3)]

    assert [decision.terminated for decision in decisions] == [False, False, True]
    assert decisions[-1].reason == "min_progress"
