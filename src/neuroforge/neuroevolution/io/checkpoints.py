"""Helpers for reading evolution checkpoints.

Evolution checkpoints are JSON files written by :class:`EvolutionEngine`.
Training scripts only need a small, stable view of them: the best genome and
its score. Keeping that parsing here avoids each script learning the checkpoint
schema independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from neuroforge.neuroevolution.io.genome_codec import decode_genome
from neuroforge.neuroevolution.io.serde import cast_json_object

__all__ = [
    "BestGenomeCheckpoint",
    "find_latest_evolution_checkpoint",
    "load_best_genome_checkpoint",
]


@dataclass(frozen=True, slots=True)
class BestGenomeCheckpoint:
    """Best genome plus metadata loaded from one evolution checkpoint."""

    path: Path
    genome: Any            # PolicyGenome or GraphGenome, by checkpoint "type"
    fitness: float
    metrics: dict[str, float]
    generation: int
    evaluations: int
    schema_version: int = 1
    config: dict[str, Any] = field(default_factory=dict[str, Any])


def find_latest_evolution_checkpoint(runs_dir: str | Path) -> Path | None:
    """Return the newest evolution checkpoint under *runs_dir*, if one exists."""
    root = Path(runs_dir)
    candidates = [
        path
        for path in root.glob("**/evolution/checkpoint.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_best_genome_checkpoint(path: str | Path) -> BestGenomeCheckpoint:
    """Load the best genome from an evolution checkpoint JSON file."""
    checkpoint_path = Path(path)
    payload = cast_json_object(json.loads(checkpoint_path.read_text(encoding="utf-8")))
    best = payload.get("best")
    if not isinstance(best, dict):
        msg = f"evolution checkpoint has no best genome: {checkpoint_path}"
        raise ValueError(msg)
    best_obj = cast("dict[str, Any]", best)
    genome = decode_genome(cast_json_object(best_obj.get("genome")))
    metrics = {
        str(key): float(value)
        for key, value in cast_json_object(best_obj.get("metrics", {})).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    return BestGenomeCheckpoint(
        path=checkpoint_path,
        genome=genome,
        fitness=float(best_obj.get("fitness", 0.0)),
        metrics=metrics,
        generation=int(payload.get("generation", genome.generation)),
        evaluations=int(payload.get("evaluations", 0)),
        schema_version=int(payload.get("schema_version", 1)),
        config=cast_json_object(payload.get("config") or {}),
    )
