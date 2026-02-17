"""Shared pytest fixtures for NeuroForge tests."""

from __future__ import annotations

import os
import pathlib

import pytest


@pytest.fixture()
def artifact_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a temporary directory for test artifacts.

    Uses PYTEST_BASEDIR env var if set (avoids writing into repo tree),
    otherwise falls back to pytest's tmp_path.
    """
    base = os.environ.get("PYTEST_BASEDIR")
    if base:
        d = pathlib.Path(base) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path


@pytest.fixture()
def device() -> str:
    """Return the best available torch device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@pytest.fixture()
def seed() -> int:
    """Fixed seed for reproducibility."""
    return 42
