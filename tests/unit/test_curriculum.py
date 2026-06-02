"""Tests for the SMB3 savestate curriculum (Phase 4)."""

from __future__ import annotations

import pytest

from neuroforge.game.curriculum import ICurriculum, SMB3Curriculum


@pytest.mark.unit
def test_is_curriculum() -> None:
    assert isinstance(SMB3Curriculum(), ICurriculum)


@pytest.mark.unit
def test_empty_curriculum_boots_the_rom() -> None:
    curriculum = SMB3Curriculum()
    assert curriculum.savestate_for(0) is None
    curriculum.report_episode(1.0)  # must not raise or advance
    assert curriculum.stage == 0


@pytest.mark.unit
def test_first_stage_savestate_is_used() -> None:
    curriculum = SMB3Curriculum(("a.State", "b.State"))
    assert curriculum.savestate_for(0) == "a.State"
    assert curriculum.num_stages == 2


@pytest.mark.unit
def test_advances_only_after_min_episodes_and_threshold() -> None:
    curriculum = SMB3Curriculum(
        ("a.State", "b.State"), advance_threshold=0.9, min_episodes_per_stage=2
    )
    curriculum.report_episode(0.95)          # mastered, but only 1 episode
    assert curriculum.stage == 0
    curriculum.report_episode(0.10)          # 2 episodes, best 0.95 ≥ 0.9 → advance
    assert curriculum.stage == 1
    assert curriculum.savestate_for(99) == "b.State"


@pytest.mark.unit
def test_does_not_advance_without_mastery() -> None:
    curriculum = SMB3Curriculum(("a.State", "b.State"), advance_threshold=0.9)
    for _ in range(10):
        curriculum.report_episode(0.4)       # never reaches threshold
    assert curriculum.stage == 0


@pytest.mark.unit
def test_does_not_advance_past_last_stage() -> None:
    curriculum = SMB3Curriculum(("only.State",), min_episodes_per_stage=1)
    curriculum.report_episode(1.0)
    curriculum.report_episode(1.0)
    assert curriculum.stage == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("threshold", "min_eps"),
    [(-0.1, 1), (1.5, 1), (0.9, 0)],
)
def test_invalid_config_rejected(threshold: float, min_eps: int) -> None:
    with pytest.raises(ValueError, match="threshold|min_episodes"):
        SMB3Curriculum(("a.State",), advance_threshold=threshold, min_episodes_per_stage=min_eps)
