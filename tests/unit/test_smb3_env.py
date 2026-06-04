"""Unit tests for SMB3 script environment parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_env_module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "smb3_env.py"
    spec = importlib.util.spec_from_file_location("smb3_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_env_bool_accepts_common_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _load_env_module()

    monkeypatch.setenv("NF_BOOL", "yes")
    assert env.env_bool("NF_BOOL", False) is True

    monkeypatch.setenv("NF_BOOL", "off")
    assert env.env_bool("NF_BOOL", True) is False

    monkeypatch.setenv("NF_BOOL", "maybe")
    assert env.env_bool("NF_BOOL", True) is True


@pytest.mark.unit
def test_env_number_helpers_fallback_on_invalid_or_too_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _load_env_module()

    monkeypatch.setenv("NF_INT", "12")
    monkeypatch.setenv("NF_FLOAT", "0.25")
    assert env.env_int("NF_INT", 3, min_value=1) == 12
    assert env.env_float("NF_FLOAT", 0.5, min_value=0.0) == pytest.approx(0.25)

    monkeypatch.setenv("NF_INT", "0")
    monkeypatch.setenv("NF_FLOAT", "-1")
    assert env.env_int("NF_INT", 3, min_value=1) == 3
    assert env.env_float("NF_FLOAT", 0.5, min_value=0.0) == pytest.approx(0.5)

    monkeypatch.setenv("NF_INT", "nope")
    monkeypatch.setenv("NF_FLOAT", "nope")
    assert env.env_int("NF_INT", 3) == 3
    assert env.env_float("NF_FLOAT", 0.5) == pytest.approx(0.5)
