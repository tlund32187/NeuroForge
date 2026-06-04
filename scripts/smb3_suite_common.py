"""Shared helpers for SMB3 subprocess suites."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

EVOLUTION_RUN_DIR_ENV = "NEUROFORGE_SMB3_EVOLUTION_RUN_DIR"


def base_env(repo: Path) -> dict[str, str]:
    """Return an environment that can import the local ``src`` tree."""
    env = dict(os.environ)
    src = str(repo / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def new_evolution_run_dir(runs_dir: Path, *, label: str) -> Path:
    """Build the deterministic evolution run directory for one suite phase."""
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return runs_dir / f"{label}_{stamp}"


def with_evolution_run_dir(env: dict[str, str], run_dir: Path) -> dict[str, str]:
    """Return a copy of *env* that forces ``evolve_smb3.py`` into *run_dir*."""
    out = dict(env)
    out[EVOLUTION_RUN_DIR_ENV] = str(run_dir)
    return out


def evolution_checkpoint_path(run_dir: Path) -> Path:
    """Return the evolution checkpoint path inside a suite-owned run directory."""
    return run_dir / "evolution" / "checkpoint.json"


def run_phase(
    *,
    repo: Path,
    suite_name: str,
    label: str,
    script: Path,
    env: dict[str, str],
) -> int:
    """Run one suite phase and return its process exit code."""
    print("\n" + "=" * 70)
    print(f"{suite_name} - {label}: {script.name}")
    print("=" * 70)
    completed = subprocess.run([sys.executable, str(script)], env=env, cwd=str(repo))
    if completed.returncode != 0:
        print(f"\n  Phase failed ({label}) with exit code {completed.returncode}.")
    return int(completed.returncode)
