"""Helpers for applying evolved SMB3 genomes to live script configs."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuroforge.evolution import (
    find_latest_evolution_checkpoint,
    load_best_genome_checkpoint,
)


@dataclass(frozen=True, slots=True)
class EvolvedConfigSelection:
    """Result of applying an evolution checkpoint to a game-training config."""

    config: Any
    requested: bool
    applied: bool = False
    checkpoint_path: Path | None = None
    genome_id: str = ""
    fitness: float = 0.0
    include_network_shape: bool = False
    partial_resume_enabled: bool = False
    warnings: tuple[str, ...] = ()


def apply_evolved_genome_config(
    base_config: Any,
    *,
    use_evolved: bool,
    evolved_mode: str,
    evolution_checkpoint: str | None,
    runs_dir: Path,
) -> EvolvedConfigSelection:
    """Apply the selected evolved genome to *base_config*, if requested."""
    if not use_evolved:
        return EvolvedConfigSelection(config=base_config, requested=False)

    warnings: list[str] = []
    checkpoint_path = Path(evolution_checkpoint) if evolution_checkpoint else None
    if checkpoint_path is None:
        checkpoint_path = find_latest_evolution_checkpoint(runs_dir)
    if checkpoint_path is None or not checkpoint_path.exists():
        warnings.append("no evolution checkpoint found; using base SMB3 config.")
        return EvolvedConfigSelection(
            config=base_config,
            requested=True,
            checkpoint_path=checkpoint_path,
            warnings=tuple(warnings),
        )

    mode = evolved_mode.strip().lower()
    include_network_shape = mode == "full"
    if mode not in {"compatible", "full"}:
        warnings.append(f"unknown evolved mode {evolved_mode!r}; using compatible mode.")
        include_network_shape = False

    selected = load_best_genome_checkpoint(checkpoint_path)
    next_config = selected.genome.to_game_training_config(
        base=base_config,
        seed=base_config.seed,
        include_network_shape=include_network_shape,
    )
    partial_resume_enabled = bool(
        include_network_shape and getattr(next_config, "resume", False)
    )
    if partial_resume_enabled:
        next_config = dataclasses.replace(next_config, resume_allow_partial=True)

    return EvolvedConfigSelection(
        config=next_config,
        requested=True,
        applied=True,
        checkpoint_path=selected.path,
        genome_id=selected.genome.id,
        fitness=float(selected.fitness),
        include_network_shape=include_network_shape,
        partial_resume_enabled=partial_resume_enabled,
        warnings=tuple(warnings),
    )


def evolved_config_status_lines(
    selection: EvolvedConfigSelection,
    *,
    action: str,
) -> list[str]:
    """Return concise console lines for an evolved-config selection."""
    if not selection.requested:
        return []
    lines = [f"WARNING: {warning}" for warning in selection.warnings]
    if not selection.applied:
        return lines

    mode = (
        "full phenotype"
        if selection.include_network_shape
        else "checkpoint-compatible controls"
    )
    lines.append(
        f"Using evolved best genome {selection.genome_id} for {action} ({mode}) "
        f"fitness={selection.fitness:.3f} from {selection.checkpoint_path}"
    )
    if selection.partial_resume_enabled:
        lines.append(
            "Evolved genome mode=full changes network shape; "
            "partial checkpoint warm-start enabled."
        )
    return lines
