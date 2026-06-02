"""Biologically-plausible visual encoding (object-centric perception, Layer A).

Stages that turn raw pixels into invariant, object-aware spiking representations
without labels or prepopulated sprite shapes:

* :mod:`~neuroforge.vision.encoding.retina` (A0) — retinal center-surround ON/OFF
  contrast encoding with gain control, giving palette/brightness invariance.
* :mod:`~neuroforge.vision.encoding.stdp_features` (A1) — competitive STDP feature
  maps that discover differentiated local features (edges/corners/parts) unsupervised.
* :mod:`~neuroforge.vision.encoding.trace_invariance` (A2) — trace-rule object cells
  that bind temporally-adjacent appearances into invariant codes.
* :mod:`~neuroforge.vision.encoding.motion_figure_ground` (A3) — scroll-compensated
  motion that separates moving sprites from the scrolling background.
* :mod:`~neuroforge.vision.encoding.stack` (A4) — ``PerceptionStack``: composes the
  stages into a drop-in invariant front-end for the spiking policy.
* :mod:`~neuroforge.vision.encoding.probe` (A4) — label-free-training eval probe.
"""

from neuroforge.vision.encoding.motion_figure_ground import (
    MotionFigureGround,
    MotionFigureGroundConfig,
)
from neuroforge.vision.encoding.probe import (
    nearest_centroid_accuracy,
    representation_separability,
)
from neuroforge.vision.encoding.retina import RetinaEncoder, RetinaEncoderConfig
from neuroforge.vision.encoding.stack import PerceptionStack, PerceptionStackConfig
from neuroforge.vision.encoding.stdp_features import STDPFeatureConfig, STDPFeatureLayer
from neuroforge.vision.encoding.trace_invariance import (
    TraceInvariantConfig,
    TraceInvariantLayer,
)
from neuroforge.vision.encoding.visualize import render_feature_atlas

__all__ = [
    "MotionFigureGround",
    "MotionFigureGroundConfig",
    "PerceptionStack",
    "PerceptionStackConfig",
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
