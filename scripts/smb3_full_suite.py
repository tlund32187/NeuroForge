"""Run the live SMB3 loop: evolve -> train from evolved controls -> evaluate.

This script is the safe one-click workflow. Evolution searches a genome, then
training applies the best genome's checkpoint-compatible control/learning genes
while preserving the existing policy weight shape. Evaluation then compares the
updated checkpoint against a random baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from smb3_suite_common import (
    base_env,
    evolution_checkpoint_path,
    new_evolution_run_dir,
    run_phase,
    with_evolution_run_dir,
)

_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = _REPO / "artifacts" / "runs"
EVOLVE_SCRIPT = _REPO / "scripts" / "evolve_smb3.py"
TRAIN_SCRIPT = _REPO / "scripts" / "train_smb3.py"
EVAL_SCRIPT = _REPO / "scripts" / "evaluate_smb3_checkpoint.py"


def main() -> int:
    env = base_env(_REPO)
    evolution_run_dir = new_evolution_run_dir(RUNS_DIR, label="evolve_suite")
    evolve_env = with_evolution_run_dir(env, evolution_run_dir)

    rc = run_phase(
        repo=_REPO,
        suite_name="NeuroForge SMB3 suite",
        label="1/3 evolve",
        script=EVOLVE_SCRIPT,
        env=evolve_env,
    )
    if rc != 0:
        return rc

    checkpoint = evolution_checkpoint_path(evolution_run_dir)
    if not checkpoint.exists():
        print(f"ERROR: evolution finished but no checkpoint was found at {checkpoint}.")
        return 1
    print(f"\n  Best evolved genome source: {checkpoint}\n")

    train_env = dict(env)
    train_env.update(
        {
            "NEUROFORGE_SMB3_USE_EVOLVED": "1",
            "NEUROFORGE_SMB3_EVOLVED_MODE": "compatible",
            "NEUROFORGE_SMB3_EVOLUTION_CHECKPOINT": str(checkpoint),
            "NEUROFORGE_SMB3_RESUME": "1",
        }
    )
    rc = run_phase(
        repo=_REPO,
        suite_name="NeuroForge SMB3 suite",
        label="2/3 train",
        script=TRAIN_SCRIPT,
        env=train_env,
    )
    if rc != 0:
        return rc

    eval_env = dict(env)
    eval_env.update(
        {
            "NEUROFORGE_EVAL_EPISODES": "3",
            "NEUROFORGE_EVAL_FRAMES": "600",
            "NEUROFORGE_EVAL_DETERMINISTIC": "0",
            "NEUROFORGE_EVAL_RANDOM_BASELINE": "1",
        }
    )
    return run_phase(
        repo=_REPO,
        suite_name="NeuroForge SMB3 suite",
        label="3/3 evaluate",
        script=EVAL_SCRIPT,
        env=eval_env,
    )


if __name__ == "__main__":
    sys.exit(main())
