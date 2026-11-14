from __future__ import annotations

import os

import pytest

from blueprint_pipeline import mujoco_gl_environment as gl_env


def test_mujoco_import_default_is_scoped_and_operator_value_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str | None] = []

    def importer(_name: str) -> object:
        observed.append(os.environ.get("MUJOCO_GL"))
        return object()

    monkeypatch.setattr(gl_env.importlib, "import_module", importer)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    module, selected = gl_env.import_mujoco_with_scoped_gl_default(
        default="egl", platform_name="Linux"
    )
    assert module is not None
    assert selected == "egl"
    assert observed == ["egl"]
    assert "MUJOCO_GL" not in os.environ

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    _, selected = gl_env.import_mujoco_with_scoped_gl_default(
        default="egl", platform_name="Linux"
    )
    assert selected == "osmesa"
    assert observed[-1] == "osmesa"
    assert os.environ["MUJOCO_GL"] == "osmesa"


def test_mujoco_import_failure_restores_absent_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_importer(_name: str) -> object:
        assert os.environ["MUJOCO_GL"] == "disable"
        raise ImportError("mujoco unavailable")

    monkeypatch.setattr(gl_env.importlib, "import_module", failing_importer)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    with pytest.raises(ImportError, match="mujoco unavailable"):
        gl_env.import_mujoco_with_scoped_gl_default(
            default="disable", platform_name="Linux"
        )
    assert "MUJOCO_GL" not in os.environ


