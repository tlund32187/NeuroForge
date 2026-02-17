"""Smoke test: verify the package is importable and has a version."""

import pytest


@pytest.mark.unit
def test_import_neuroforge() -> None:
    import neuroforge

    assert hasattr(neuroforge, "__version__")


@pytest.mark.unit
def test_version_is_string() -> None:
    from neuroforge import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


@pytest.mark.unit
def test_version_format() -> None:
    from neuroforge import __version__

    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected semver format x.y.z, got {__version__}"
    for part in parts:
        assert part.isdigit(), f"Non-numeric version part: {part}"
