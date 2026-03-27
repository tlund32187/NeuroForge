"""Minimal spiking vision block building blocks."""

from neuroforge.vision.blocks.conv import SpikingConvBlock, build_2d_norm
from neuroforge.vision.blocks.pool import SpikingPool
from neuroforge.vision.blocks.res import SpikingResBlock
from neuroforge.vision.blocks.spike import SurrogateSpike

__all__ = [
    "SpikingConvBlock",
    "SpikingPool",
    "SpikingResBlock",
    "SurrogateSpike",
    "build_2d_norm",
]

