from __future__ import annotations

import sys
from pathlib import Path

import pytest

from blueprint_pipeline.native_task_arena_import_scope import (
    install_scoped_arena_embodiment,
)


def _write(path: Path, value: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


@pytest.mark.parametrize("robot_id,selected_module", [
    ("franka_panda", "isaaclab_arena.embodiments.droid.droid"),
    # Public PiPER/YAM/TRLC and scripted named-arm bindings use this same
    # generic Arena base, without importing G1's unrelated Pink dependency.
    ("team_embodiment", "isaaclab_arena.embodiments.embodiment_base"),
])
def test_scoped_arena_import_loads_selected_robot_without_unrelated_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, robot_id: str, selected_module: str
) -> None:
    package = tmp_path / "isaaclab_arena"
    _write(package / "__init__.py")
    _write(
        package / "embodiments/__init__.py",
        "raise AssertionError('eager embodiment package executed')\n",
    )
    _write(package / "embodiments/droid/__init__.py")
    _write(package / "embodiments/droid/droid.py", "SELECTED = True\n")
    _write(package / "embodiments/embodiment_base.py", "SELECTED = True\n")
    _write(
        package / "embodiments/g1/__init__.py",
        "raise AssertionError('unrelated G1 imported')\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in tuple(sys.modules):
        if name == "isaaclab_arena" or name.startswith("isaaclab_arena."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    receipt = install_scoped_arena_embodiment(robot_id)

    assert receipt["selected_module"] == selected_module
    assert receipt["eager_all_embodiments_imported"] is False
    assert "isaaclab_arena.embodiments.g1" not in sys.modules


def test_scoped_arena_import_rejects_unadmitted_robot() -> None:
    with pytest.raises(
        RuntimeError,
        match="native_task_arena_robot_embodiment_unadmitted:unknown_robot",
    ):
        install_scoped_arena_embodiment("unknown_robot")
