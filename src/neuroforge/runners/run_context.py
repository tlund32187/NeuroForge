"""Run context — metadata and directory setup for M0 artifact runs."""

from __future__ import annotations

import datetime
import platform
import socket
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable metadata describing a single training run.

    Attributes
    ----------
    run_id:
        Unique identifier (``run_<YYYYmmdd_HHMMSS>_<shortid>``).
    run_dir:
        Absolute path to the run output directory.
    started_at:
        ISO-8601 timestamp of when the run was created.
    device:
        Compute device (e.g. ``"cpu"`` or ``"cuda:0"``).
    seed:
        Random seed used for reproducibility.
    git_commit:
        Short git SHA of the working tree, or ``"unknown"``.
    torch_version:
        PyTorch version string.
    cuda_available:
        Whether CUDA is available at run time.
    cuda_version:
        CUDA runtime version string (empty when unavailable).
    hostname:
        Machine hostname.
    platform:
        Platform identifier (e.g. ``"Windows-10-…"``).
    extra:
        Arbitrary additional metadata.
    """

    run_id: str
    run_dir: Path
    started_at: str
    device: str = "cpu"
    seed: int = 42
    git_commit: str = "unknown"
    torch_version: str = "unknown"
    cuda_available: bool = False
    cuda_version: str = ""
    hostname: str = ""
    platform: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (Path → str)."""
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        return d


def _git_short_sha() -> str:
    """Return the short git SHA of HEAD, or ``'unknown'``."""
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _torch_info() -> tuple[str, bool, str]:
    """Return ``(version, cuda_available, cuda_version)``."""
    try:
        import torch

        ver = torch.__version__
        cuda = torch.cuda.is_available()
        cuda_ver = torch.version.cuda or "" if cuda else ""
        return ver, cuda, cuda_ver
    except ImportError:
        return "unavailable", False, ""


def create_run_dir(
    *,
    base_dir: str | Path = "artifacts",
    seed: int = 42,
    device: str = "cpu",
) -> RunContext:
    """Create a timestamped run directory and return its context.

    Parameters
    ----------
    base_dir:
        Parent directory for all runs (default ``"artifacts"``).
    seed:
        Seed to record in the context.
    device:
        Device string to record.

    Returns
    -------
    RunContext
        Immutable metadata about the new run, with ``run_dir`` created
        on disk (including ``metrics/`` and ``logs/`` subdirectories).
    """
    now = datetime.datetime.now(tz=datetime.UTC)
    short_id = uuid.uuid4().hex[:8]
    run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{short_id}"

    base = Path(base_dir)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    torch_ver, cuda_avail, cuda_ver = _torch_info()

    return RunContext(
        run_id=run_id,
        run_dir=run_dir.resolve(),
        started_at=now.isoformat(),
        device=device,
        seed=seed,
        git_commit=_git_short_sha(),
        torch_version=torch_ver,
        cuda_available=cuda_avail,
        cuda_version=cuda_ver,
        hostname=socket.gethostname(),
        platform=platform.platform(),
    )
