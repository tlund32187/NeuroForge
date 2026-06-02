"""Tests for temporal action commitment (the anti-dither / run-momentum fix)."""

from __future__ import annotations

import pytest

from neuroforge.contracts.game import ControllerAction
from neuroforge.game.policies.commitment import TemporalCommitment, TemporalCommitmentConfig


@pytest.mark.unit
def test_commitment_holds_heading_through_dither() -> None:
    commit = TemporalCommitment(TemporalCommitmentConfig(commit_frames=4))
    # Dithering input: Right, then three Lefts, then Lefts again.
    inputs = [
        ControllerAction(right=True),
        ControllerAction(left=True),
        ControllerAction(left=True),
        ControllerAction(left=True),
        ControllerAction(left=True),
        ControllerAction(left=True),
        ControllerAction(left=True),
        ControllerAction(left=True),
    ]
    out = [commit.apply(a) for a in inputs]
    assert all(o.right and not o.left for o in out[:4])   # heading held Right for the window
    assert all(o.left and not o.right for o in out[4:])   # then re-commits to Left


@pytest.mark.unit
def test_commitment_passes_through_other_buttons() -> None:
    commit = TemporalCommitment(TemporalCommitmentConfig(commit_frames=4))
    commit.apply(ControllerAction(right=True))            # commit to Right
    out = commit.apply(ControllerAction(left=True, a=True, b=True))  # mid-commitment
    assert out.right and not out.left                     # heading still Right
    assert out.a and out.b                                # jump/run pass through every frame


@pytest.mark.unit
def test_commit_frames_one_is_passthrough_direction() -> None:
    commit = TemporalCommitment(TemporalCommitmentConfig(commit_frames=1))
    assert commit.apply(ControllerAction(right=True)).right
    assert commit.apply(ControllerAction(left=True)).left   # re-adopts every frame


@pytest.mark.unit
def test_reset_clears_commitment() -> None:
    commit = TemporalCommitment(TemporalCommitmentConfig(commit_frames=8))
    commit.apply(ControllerAction(right=True))            # commit Right
    commit.reset()
    out = commit.apply(ControllerAction(left=True))       # should adopt Left immediately
    assert out.left and not out.right
