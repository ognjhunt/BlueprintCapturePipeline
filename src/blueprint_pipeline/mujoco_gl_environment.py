"""Process-scoped MuJoCo import environment."""

from __future__ import annotations

import importlib
import os
import platform
from typing import Any


def import_mujoco_with_scoped_gl_default(
    *, default: str, platform_name: str | None = None
) -> tuple[Any, str | None]:
    """Import MuJoCo with a Linux-only default without leaking process state."""

    key = "MUJOCO_GL"
    explicit = key in os.environ
    previous = os.environ.get(key)
    if not explicit and (platform_name or platform.system()).lower() == "linux":
        os.environ[key] = default
    selected = os.environ.get(key)
    try:
        module = importlib.import_module("mujoco")
    finally:
        if explicit and previous is not None:
            os.environ[key] = previous
        elif not explicit:
            os.environ.pop(key, None)
    return module, selected


__all__ = ["import_mujoco_with_scoped_gl_default"]
