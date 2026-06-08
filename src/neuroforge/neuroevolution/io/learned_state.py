"""Lamarckian learned-state metadata for evolution checkpoints.

The genome remains the genetic recipe. Learned weights are stored in the normal
policy checkpoint artifact, while the evolution checkpoint records which genome
should warm-start from that artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from neuroforge.neuroevolution.io.genome_codec import decode_genome
from neuroforge.neuroevolution.io.serde import cast_json_object

__all__ = [
    "attach_learned_checkpoint_to_evolution_checkpoint",
    "learned_checkpoint_path",
    "with_learned_checkpoint",
]


def learned_checkpoint_path(genome: object) -> str:
    """Return an optional learned policy checkpoint path carried by *genome*."""
    raw = getattr(genome, "learned_checkpoint_path", "")
    return raw if isinstance(raw, str) else ""


def with_learned_checkpoint(genome: object, path: str | Path) -> Any:
    """Return *genome* with a learned policy checkpoint path attached."""
    attach = getattr(genome, "with_learned_checkpoint", None)
    if callable(attach):
        return attach(str(path))
    return genome


def attach_learned_checkpoint_to_evolution_checkpoint(
    checkpoint_path: str | Path,
    learned_checkpoint: str | Path,
    *,
    genome_id: str = "",
    source: str = "training",
) -> dict[str, Any]:
    """Patch an evolution checkpoint so its matching genome inherits learned weights.

    The learned state itself stays in *learned_checkpoint* (usually a ``.pt`` file
    written by :class:`PolicyCheckpoint`). This function stores only that path on
    the best genome and any population genome with the same genetic content.
    """
    checkpoint = Path(checkpoint_path)
    learned = str(Path(learned_checkpoint))
    payload = cast_json_object(json.loads(checkpoint.read_text(encoding="utf-8")))
    best = payload.get("best")
    if not isinstance(best, dict):
        msg = f"evolution checkpoint has no best genome: {checkpoint}"
        raise ValueError(msg)
    best_obj = cast("dict[str, Any]", best)
    best_genome_payload = cast_json_object(best_obj.get("genome"))
    target_id = genome_id or str(best_genome_payload.get("id", ""))
    if genome_id and str(best_genome_payload.get("id", "")) != genome_id:
        msg = (
            f"checkpoint best genome id {best_genome_payload.get('id')!r} "
            f"does not match requested genome id {genome_id!r}"
        )
        raise ValueError(msg)

    target_key = decode_genome(best_genome_payload).content_key()
    _attach_path(best_genome_payload, learned)
    best_obj["genome"] = best_genome_payload
    best_obj["lamarckian"] = {
        "policy_checkpoint": learned,
        "genome_id": target_id,
        "source": source,
    }

    population_updates = 0
    raw_population = payload.get("population", [])
    if isinstance(raw_population, list):
        population = cast("list[object]", raw_population)
        for index, raw_item in enumerate(population):
            if not isinstance(raw_item, dict):
                continue
            genome_payload = cast_json_object(raw_item)
            if decode_genome(genome_payload).content_key() != target_key:
                continue
            _attach_path(genome_payload, learned)
            population[index] = genome_payload
            population_updates += 1
        payload["population"] = population

    checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint),
        "learned_checkpoint": learned,
        "genome_id": target_id,
        "population_updates": population_updates,
    }


def _attach_path(genome_payload: dict[str, Any], learned_checkpoint: str) -> None:
    genome_payload["learned_checkpoint_path"] = learned_checkpoint
