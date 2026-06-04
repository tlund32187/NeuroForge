# pyright: basic
"""Decode a checkpointed genome back into the right concrete type.

Checkpoints store a ``"type"`` tag (``"policy"`` or ``"graph"``). This dispatcher
reads it so the engine and checkpoint reader reconstruct the correct genome —
without that, a resumed structural run would try to load graph genomes as
hyperparameter genomes and fail. Legacy payloads without a tag decode as
``PolicyGenome`` for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from neuroforge.evolution.genome import PolicyGenome
from neuroforge.evolution.graph_genome import GraphGenome

__all__ = ["decode_genome", "max_connection_innovation"]


def decode_genome(payload: dict[str, Any]) -> Any:
    """Reconstruct a genome from its :meth:`to_dict` payload by its ``type`` tag."""
    kind = str(payload.get("type", "policy"))
    if kind == "graph":
        return GraphGenome.from_dict(payload)
    return PolicyGenome.from_dict(payload)


def max_connection_innovation(payloads: list[dict[str, Any]]) -> int:
    """Highest connection innovation across graph-genome payloads (-1 if none).

    Used to resume an :class:`~neuroforge.evolution.innovations.InnovationRegistry`
    above the numbers already in a checkpoint, so new structural mutations never
    collide with existing genes.
    """
    highest = -1
    for payload in payloads:
        for conn in payload.get("connections", []) or []:
            if isinstance(conn, dict) and "innovation" in conn:
                highest = max(highest, int(conn["innovation"]))
    return highest
