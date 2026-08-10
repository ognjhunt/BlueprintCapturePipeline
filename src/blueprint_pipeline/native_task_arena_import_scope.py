"""Admit only the Arena embodiment required by one native task contract.

The released Arena package eagerly imports every robot implementation from
``isaaclab_arena.embodiments.__init__``.  That makes a Franka task depend on
unrelated G1/ONNX/Pinocchio packages before the environment builder can import.
This module preserves the exact released source files while replacing that
package-level side effect with an explicit, fail-closed embodiment admission.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any


ROBOT_EMBODIMENT_MODULES = {
    "franka_panda": "isaaclab_arena.embodiments.droid.droid",
}


def install_scoped_arena_embodiment(robot_id: str) -> dict[str, Any]:
    """Install and import the one Arena embodiment selected by ``robot_id``."""

    selected_module = ROBOT_EMBODIMENT_MODULES.get(str(robot_id))
    if selected_module is None:
        raise RuntimeError(f"native_task_arena_robot_embodiment_unadmitted:{robot_id}")

    arena = importlib.import_module("isaaclab_arena")
    package_name = "isaaclab_arena.embodiments"
    existing = sys.modules.get(package_name)
    if existing is not None and getattr(existing, "__blueprint_scoped__", False):
        imported = importlib.import_module(selected_module)
        return {
            "schema_version": "native_task_arena_embodiment_scope.v1",
            "robot_id": str(robot_id),
            "selected_module": selected_module,
            "selected_module_file": str(getattr(imported, "__file__", "")),
            "eager_all_embodiments_imported": False,
        }
    arena_file = getattr(arena, "__file__", None)
    if not arena_file:
        raise RuntimeError("native_task_arena_package_identity_unavailable")
    embodiments_path = Path(arena_file).resolve().parent / "embodiments"
    if not embodiments_path.is_dir():
        raise RuntimeError("native_task_arena_embodiments_package_missing")

    if existing is not None and not getattr(existing, "__blueprint_scoped__", False):
        raise RuntimeError("native_task_arena_embodiments_already_imported_unscoped")
    if existing is None:
        scoped = types.ModuleType(package_name)
        scoped.__package__ = package_name
        scoped.__path__ = [str(embodiments_path)]
        scoped.__blueprint_scoped__ = True
        spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(embodiments_path)]
        scoped.__spec__ = spec
        sys.modules[package_name] = scoped
        setattr(arena, "embodiments", scoped)

    imported = importlib.import_module(selected_module)
    return {
        "schema_version": "native_task_arena_embodiment_scope.v1",
        "robot_id": str(robot_id),
        "selected_module": selected_module,
        "selected_module_file": str(getattr(imported, "__file__", "")),
        "eager_all_embodiments_imported": False,
    }


__all__ = ["ROBOT_EMBODIMENT_MODULES", "install_scoped_arena_embodiment"]
