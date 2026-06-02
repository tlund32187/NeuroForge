"""Tests for SMB3 reward shaping (Phase 4).

Exercises the stateful shaping the base reward model cannot express: a
progress-dominant signal, a per-frame idle penalty, a one-shot anti-stall
penalty, and a one-shot level-clear bonus.
"""

from __future__ import annotations

import pytest

from neuroforge.contracts.game import GameObservation, ScreenFrame, VisionGameMetrics
from neuroforge.game.rewards_smb3 import SMB3RewardConfig, SMB3RewardModel


def _obs(
    *,
    x: float | None = None,
    lives: int | None = None,
    score: int | None = None,
) -> GameObservation:
    frame = ScreenFrame(width=1, height=1, channels=1, data=b"\x00")
    metrics = VisionGameMetrics(score=score, lives=lives, x_progress=x)
    return GameObservation(step=0, t=0.0, frame=frame, metrics=metrics)


@pytest.mark.unit
def test_forward_progress_is_rewarded_and_dominant() -> None:
    model = SMB3RewardModel()
    model.begin_episode()
    reward = model.reward(_obs(x=0.0), _obs(x=0.1))
    # progress_scale defaults to 100 → 0.1 of the level == +10, no idle penalty.
    assert reward == pytest.approx(10.0)


@pytest.mark.unit
def test_no_progress_incurs_idle_penalty() -> None:
    model = SMB3RewardModel(SMB3RewardConfig(idle_penalty=-0.5, stall_penalty=0.0))
    model.begin_episode()
    assert model.reward(_obs(x=0.5), _obs(x=0.5)) == pytest.approx(-0.5)


@pytest.mark.unit
def test_death_applies_the_base_life_loss_penalty() -> None:
    model = SMB3RewardModel(SMB3RewardConfig(idle_penalty=0.0, stall_penalty=0.0))
    model.begin_episode()
    # base life_loss_penalty defaults to -50 per life lost.
    assert model.reward(_obs(x=0.5, lives=3), _obs(x=0.5, lives=2)) == pytest.approx(-50.0)


@pytest.mark.unit
def test_anti_stall_penalty_fires_once_at_patience() -> None:
    model = SMB3RewardModel(
        SMB3RewardConfig(idle_penalty=0.0, stall_patience=3, stall_penalty=-20.0)
    )
    model.begin_episode()
    rewards = [model.reward(_obs(x=0.2), _obs(x=0.2)) for _ in range(4)]
    assert rewards == pytest.approx([0.0, 0.0, -20.0, 0.0])  # one-shot at frame 3


@pytest.mark.unit
def test_stall_counter_resets_on_progress() -> None:
    model = SMB3RewardModel(
        SMB3RewardConfig(idle_penalty=0.0, stall_patience=2, stall_penalty=-20.0)
    )
    model.begin_episode()
    model.reward(_obs(x=0.2), _obs(x=0.2))          # stall=1
    model.reward(_obs(x=0.2), _obs(x=0.3))          # progress → stall resets
    assert model.reward(_obs(x=0.3), _obs(x=0.3)) == pytest.approx(0.0)  # stall=1, no fire


@pytest.mark.unit
def test_level_clear_bonus_is_one_shot() -> None:
    model = SMB3RewardModel(SMB3RewardConfig(clear_progress_threshold=0.985))
    model.begin_episode()
    first = model.reward(_obs(x=0.9), _obs(x=0.99))   # crosses threshold
    second = model.reward(_obs(x=0.99), _obs(x=0.995))
    assert first == pytest.approx((0.99 - 0.9) * 100.0 + 200.0)
    assert second == pytest.approx((0.995 - 0.99) * 100.0)  # no second bonus


@pytest.mark.unit
def test_begin_episode_rearms_clear_and_stall() -> None:
    model = SMB3RewardModel()
    model.begin_episode()
    model.reward(_obs(x=0.9), _obs(x=0.99))           # clear fires
    model.begin_episode()                             # new episode
    reward = model.reward(_obs(x=0.9), _obs(x=0.99))  # clear fires again
    assert reward == pytest.approx((0.99 - 0.9) * 100.0 + 200.0)
