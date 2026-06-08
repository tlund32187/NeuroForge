"""Tests for the A4 integration: PerceptionStack, the eval probe, and wiring.

Covers the IFrameEncoder seam (both the raw preprocessor and the bio stack
satisfy it), the composed stack's shape/lifecycle, the label-free-training eval
probe (invariant content separability across palettes), and that the stack drives
the spiking policy end-to-end through the training task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from neuroforge.contracts.applications.games import ScreenFrame

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytest.importorskip("torch")

from neuroforge.perception.vision.encoding import (
    PerceptionStack,
    PerceptionStackConfig,
    RetinaEncoder,
    RetinaEncoderConfig,
    representation_separability,
)
from neuroforge.perception.vision.encoding.frame_encoder import IFrameEncoder
from neuroforge.perception.vision.encoding.frame_preprocess import (
    FramePreprocessConfig,
    FramePreprocessor,
)

_H, _W = 56, 64


def _scene(hole_col: int, *, block: bool) -> NDArray[np.float32]:
    luma = np.full((_H, _W), 0.2, dtype=np.float32)
    luma[44:, :] = 0.8
    luma[44:, hole_col : hole_col + 10] = 0.2
    if block:
        luma[18:26, 28:36] = 0.6
    return luma


def _frame(luma: NDArray[np.float32], *, recolor: bool) -> ScreenFrame:
    if recolor:
        luma = np.clip(luma * 0.6 + 0.25, 0.0, 1.0)
        rgb = np.stack([luma, luma * 0.7 + 0.1, 0.3 + 0.5 * luma], axis=-1)
    else:
        rgb = np.stack([luma, luma, luma], axis=-1)
    data = (rgb * 255).astype(np.uint8).tobytes()
    return ScreenFrame(width=_W, height=_H, channels=3, data=data)


@pytest.mark.unit
def test_encoders_satisfy_the_protocol() -> None:
    assert isinstance(PerceptionStack(), IFrameEncoder)
    assert isinstance(RetinaEncoder(), IFrameEncoder)
    assert isinstance(FramePreprocessor(), IFrameEncoder)


@pytest.mark.unit
def test_stack_input_size_and_shape() -> None:
    cfg = PerceptionStackConfig(retina=RetinaEncoderConfig(out_h=28, out_w=32), motion=True)
    stack = PerceptionStack(cfg)
    assert stack.input_size == 28 * 32 * 3            # ON + OFF + motion channels
    drive = stack.to_drive(_frame(_scene(20, block=True), recolor=False))
    assert tuple(drive.shape) == (28 * 32 * 3,)


@pytest.mark.unit
def test_stack_without_motion_drops_the_channel() -> None:
    stack = PerceptionStack(PerceptionStackConfig(motion=False))
    assert stack.input_size == 28 * 32 * 2


@pytest.mark.unit
def test_probe_separates_content_across_palettes() -> None:
    # 3 content classes, each shown in two different palettes; reset-per-frame
    # means motion is zero, so this measures invariant *content* separability.
    luma = [_scene(8, block=False), _scene(46, block=False), _scene(20, block=True)]
    frames = [_frame(luma[c], recolor=r) for c in range(3) for r in (False, True)]
    labels = [0, 0, 1, 1, 2, 2]
    accuracy = representation_separability(RetinaEncoder(), frames, labels)
    assert accuracy >= 0.8                            # content survives the palette change


@pytest.mark.unit
def test_stack_drives_the_policy_through_the_task() -> None:
    from neuroforge.applications.tasks.game_training import GameTrainingConfig, GameTrainingTask
    from neuroforge.environments.games.clients.scripted import ScriptedGameClient

    stack = PerceptionStack(
        PerceptionStackConfig(retina=RetinaEncoderConfig(out_h=12, out_w=12), motion=True),
    )
    cfg = GameTrainingConfig(
        preprocess=FramePreprocessConfig(out_h=12, out_w=12),  # ignored when encoder is injected
        n_hidden=16, motor_per_button=1, input_fanin=8,
        decide_ticks=3, max_episodes=1, frames_per_episode=4, telemetry_every=0, seed=2,
    )
    client = ScriptedGameClient(width=64, height=56, channels=3, max_steps=4)
    task = GameTrainingTask(cfg, client=client, encoder=stack)
    result = task.run()
    assert result.frames == 4                          # the invariant drive ran the brain


#


def _small_stack(**kw: Any) -> PerceptionStack:
    return PerceptionStack(
        PerceptionStackConfig(retina=RetinaEncoderConfig(out_h=12, out_w=12), **kw),
    )


@pytest.mark.unit
def test_objects_requires_features() -> None:
    with pytest.raises(ValueError, match="features"):
        PerceptionStackConfig(features=False, objects=True)


@pytest.mark.unit
def test_full_stack_input_size_and_shape() -> None:
    stack = _small_stack(features=True, objects=True, motion=True)
    # fh=fw=(12-5)//4+1=2, F=16 -> 64; objects 16; motion 12*12=144.
    assert stack.input_size == 16 * 2 * 2 + 16 + 144
    drive = stack.to_drive(_frame(_scene(8, block=True), recolor=False))
    assert tuple(drive.shape) == (stack.input_size,)


@pytest.mark.unit
def test_features_learn_online_only_when_learning() -> None:
    import torch

    frame = _frame(_scene(8, block=True), recolor=False)

    learning = _small_stack(features=True, learn=True)
    assert learning._a1 is not None
    before = learning._a1.weights.clone()
    for _ in range(5):
        learning.to_drive(frame)
    assert not torch.allclose(before, learning._a1.weights)

    frozen = _small_stack(features=True, learn=False)
    assert frozen._a1 is not None
    fixed = frozen._a1.weights.clone()
    for _ in range(5):
        frozen.to_drive(frame)
    assert torch.allclose(fixed, frozen._a1.weights)


@pytest.mark.unit
def test_reset_persists_features_but_clears_trace() -> None:
    import torch

    stack = _small_stack(features=True, objects=True)
    assert stack._a1 is not None and stack._a2 is not None
    frame = _frame(_scene(8, block=True), recolor=False)
    stack.to_drive(frame)
    stack.to_drive(frame)
    weights = stack._a1.weights.clone()

    stack.reset()

    assert torch.allclose(weights, stack._a1.weights)
    assert float(stack._a2._trace.abs().sum()) == 0.0


@pytest.mark.unit
def test_state_dict_round_trip() -> None:
    import torch

    trained = _small_stack(features=True, objects=True)
    frame = _frame(_scene(8, block=True), recolor=False)
    for _ in range(5):
        trained.to_drive(frame)

    fresh = _small_stack(features=True, objects=True)
    fresh.load_state_dict(trained.state_dict())
    assert trained._a1 is not None and fresh._a1 is not None
    assert torch.allclose(trained._a1.weights, fresh._a1.weights)
