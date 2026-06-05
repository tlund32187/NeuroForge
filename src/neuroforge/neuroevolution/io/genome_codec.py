"""Decode a checkpointed genome back into the right concrete type.

Checkpoints store a ``"type"`` tag. This dispatcher
reads it so the engine and checkpoint reader reconstruct the correct genome —
without that, a resumed structural run would try to load graph genomes as
hyperparameter genomes and fail. Legacy payloads without a tag decode as
``PolicyGenome`` for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from neuroforge.neuroevolution.genomes.graph import GraphGenome
from neuroforge.neuroevolution.genomes.hyperneat import HyperNEATGenome
from neuroforge.neuroevolution.genomes.policy import PolicyGenome

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["decode_genome", "max_connection_innovation"]


def decode_genome(payload: dict[str, Any]) -> Any:
    """Reconstruct a genome from its :meth:`to_dict` payload by its ``type`` tag."""
    kind = str(payload.get("type", "policy"))
    if kind == "graph":
        return GraphGenome.from_dict(payload)
    if kind == "hyperneat":
        return HyperNEATGenome.from_dict(payload)
    return PolicyGenome.from_dict(payload)


def max_connection_innovation(payloads: list[dict[str, Any]]) -> int:
    """Highest connection innovation across graph-genome payloads (-1 if none).

    Used to resume an :class:`~neuroforge.neuroevolution.genomes.innovations.InnovationRegistry`
    above the numbers already in a checkpoint, so new structural mutations never
    collide with existing genes.
    """
    innovations: list[int] = [-1]
    for payload in payloads:
        innovations.extend(_connection_innovations(payload.get("connections")))
        cppn = payload.get("cppn")
        if isinstance(cppn, dict):
            innovations.extend(
                _connection_innovations(cast("dict[str, Any]", cppn).get("connections")),
            )
    return max(innovations)


def _connection_innovations(raw: Any) -> list[int]:
    """Innovation numbers from a (possibly untyped) JSON ``connections`` list."""
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for conn in cast("Sequence[Any]", raw):
        if isinstance(conn, dict) and "innovation" in conn:
            out.append(int(cast("dict[str, Any]", conn)["innovation"]))
    return out
