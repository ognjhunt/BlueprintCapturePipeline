"""Existing filesystem path checks used by scene readers and writers."""
from __future__ import annotations

from pathlib import Path


def require(condition, code):
    if not condition:
        raise ValueError("scene_progression_" + code)


def safe_path(path):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "path_unsafe")
    return path

