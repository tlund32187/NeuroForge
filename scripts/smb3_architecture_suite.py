"""Run fresh full-architecture SMB3 evolution -> training -> evaluation.

This is the architecture-search lane. It lets evolution change the policy
network shape, then trains that evolved phenotype from a fresh checkpoint and
evaluates it against a random policy with the same evolved architecture.
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
ARCH_CHECKPOINT = _REPO / "artifacts" / "smb3_arch_policy.pt"
BASE_CHECKPOINT = _REPO / "artifacts" / "smb3_policy.pt"


def main() -> int:
    env = base_env(_REPO)
    evolution_run_dir = new_evolution_run_dir(RUNS_DIR, label="evolve_architecture")
    evolve_env = with_evolution_run_dir(env, evolution_run_dir)

    rc = run_phase(
        repo=_REPO,
        suite_name="NeuroForge SMB3 architecture suite",
        label="1/3 evolve full phenotype",
        script=EVOLVE_SCRIPT,
        env=evolve_env,
    )
    if rc != 0:
        return rc

    evolution_checkpoint = evolution_checkpoint_path(evolution_run_dir)
    if not evolution_checkpoint.exists():
        print(f"ERROR: evolution finished but no checkpoint was found at {evolution_checkpoint}.")
        return 1
    print(f"\n  Full phenotype source: {evolution_checkpoint}")
    print(f"  Fresh architecture policy checkpoint: {ARCH_CHECKPOINT}\n")

    train_env = dict(env)
    train_env.update(
        {
            "NEUROFORGE_SMB3_USE_EVOLVED": "1",
            "NEUROFORGE_SMB3_EVOLVED_MODE": "full",
            "NEUROFORGE_SMB3_EVOLUTION_CHECKPOINT": str(evolution_checkpoint),
            "NEUROFORGE_SMB3_CHECKPOINT": str(ARCH_CHECKPOINT),
            "NEUROFORGE_SMB3_RESUME_CHECKPOINT": str(BASE_CHECKPOINT),
            "NEUROFORGE_SMB3_RESUME": "1",
        }
    )
    rc = run_phase(
        repo=_REPO,
        suite_name="NeuroForge SMB3 architecture suite",
        label="2/3 train fresh architecture",
        script=TRAIN_SCRIPT,
        env=train_env,
    )
    if rc != 0:
        return rc

    eval_env = dict(env)
    eval_env.update(
        {
            "NEUROFORGE_SMB3_USE_EVOLVED": "1",
            "NEUROFORGE_SMB3_EVOLVED_MODE": "full",
            "NEUROFORGE_SMB3_EVOLUTION_CHECKPOINT": str(evolution_checkpoint),
            "NEUROFORGE_SMB3_CHECKPOINT": str(ARCH_CHECKPOINT),
            "NEUROFORGE_EVAL_EPISODES": "3",
            "NEUROFORGE_EVAL_FRAMES": "600",
            "NEUROFORGE_EVAL_DETERMINISTIC": "0",
            "NEUROFORGE_EVAL_RANDOM_BASELINE": "1",
        }
    )
    return run_phase(
        repo=_REPO,
        suite_name="NeuroForge SMB3 architecture suite",
        label="3/3 evaluate fresh architecture",
        script=EVAL_SCRIPT,
        env=eval_env,
    )


if __name__ == "__main__":
    sys.exit(main())
