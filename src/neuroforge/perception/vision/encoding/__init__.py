"""Biologically-plausible visual encoding (object-centric perception, Layer A).

Stages that turn raw pixels into invariant, object-aware spiking representations
without labels or prepopulated sprite shapes:

* :mod:`~neuroforge.perception.vision.encoding.retina` (A0) — retinal center-surround ON/OFF
  contrast encoding with gain control, giving palette/brightness invariance.
* :mod:`~neuroforge.perception.vision.encoding.stdp_features` (A1) — competitive STDP feature
  maps that discover differentiated local features (edges/corners/parts) unsupervised.
* :mod:`~neuroforge.perception.vision.encoding.trace_invariance` (A2) — trace-rule object cells
  that bind temporally-adjacent appearances into invariant codes.
* :mod:`~neuroforge.perception.vision.encoding.motion_figure_ground` (A3) — scroll-compensated
  motion that separates moving sprites from the scrolling background.
* :mod:`~neuroforge.perception.vision.encoding.stack` (A4) — ``PerceptionStack``: composes the
  stages into a drop-in invariant front-end for the spiking policy.
* :mod:`~neuroforge.perception.vision.encoding.probe` (A4) — label-free-training eval probe.
"""

from neuroforge.perception.vision.encoding.frame_encoder import IFrameEncoder
from neuroforge.perception.vision.encoding.frame_preprocess import (
    FramePreprocessConfig,
    FramePreprocessor,
)
from neuroforge.perception.vision.encoding.motion_figure_ground import (
    MotionFigureGround,
    MotionFigureGroundConfig,
)
from neuroforge.perception.vision.encoding.probe import (
    nearest_centroid_accuracy,
    representation_separability,
)
from neuroforge.perception.vision.encoding.retina import RetinaEncoder, RetinaEncoderConfig
from neuroforge.perception.vision.encoding.stack import PerceptionStack, PerceptionStackConfig
from neuroforge.perception.vision.encoding.stdp_features import STDPFeatureConfig, STDPFeatureLayer
from neuroforge.perception.vision.encoding.trace_invariance import (
    TraceInvariantConfig,
    TraceInvariantLayer,
)
from neuroforge.perception.vision.encoding.visualize import render_feature_atlas

__all__ = [
    "MotionFigureGround",
    "MotionFigureGroundConfig",
    "PerceptionStack",
    "PerceptionStackConfig",
    "FramePreprocessConfig",
    "FramePreprocessor",
    "IFrameEncoder",
    "RetinaEncoder",
    "RetinaEncoderConfig",
    "STDPFeatureConfig",
    "STDPFeatureLayer",
    "TraceInvariantConfig",
    "TraceInvariantLayer",
    "nearest_centroid_accuracy",
    "render_feature_atlas",
    "representation_separability",
]
