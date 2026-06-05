"""Biological plasticity rules and trace primitives."""

from __future__ import annotations

from neuroforge.biology.plasticity.base import PlasticityRuleBase
from neuroforge.biology.plasticity.eligibility import eligibility_decay
from neuroforge.biology.plasticity.traces import PlasticityTrace

__all__ = ["PlasticityRuleBase", "PlasticityTrace", "eligibility_decay"]
