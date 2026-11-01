"""What an OS sandbox needs to run this process's own Python interpreter.

The Isaac base image installs Kit's Python under ``/isaac-sim``, which it ships
as ``drwxr-x--- isaac-sim:isaac-sim``. Container root reads it only with
``CAP_DAC_OVERRIDE``, and the Landlock launcher clears every capability before
it opens its rules, so a sandbox that admits ``sys.base_prefix`` must first make
that runtime code readable (``asset_runtime_permissions``). Kit's interpreter
also finds ``libpython`` through a launcher-provided library path rather than
ELF RUNPATH, so the sandboxed interpreter needs that path back.

capture-447ae4f4, 2026-11-01: ARTISAN's sandbox preflight on a Vast Isaac host
failed with ``Permission denied: '/isaac-sim/kit/python'``; the Astra backend
already prepared both, ARTISAN did neither.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


def sandboxed_interpreter() -> dict[str, Any]:
    """Roots, loader environment and executable for running ``sys.executable`` sandboxed."""

    base = Path(sys.base_prefix).resolve()
    library_paths: list[Path] = []
    for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        path = Path(value) if value and Path(value).is_absolute() else None
        # Only the interpreter's own libraries; a launcher path can name much
        # larger trees that generated code has no reason to read.
        if path is not None and path.is_dir() and path.resolve().is_relative_to(base):
            library_paths.append(path.resolve())
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    for parent in (base / "lib", base.parent):
        if parent.is_dir() and any(parent.glob(f"libpython{version}.so*")):
            library_paths.append(parent.resolve())
    library_paths = list(dict.fromkeys(library_paths))
    roots = list(dict.fromkeys([base, *library_paths]))
    environment = {"LD_LIBRARY_PATH": os.pathsep.join(map(str, library_paths))} if library_paths else {}
    return {"roots": roots, "library_environment": environment,
            "executable": Path(sys.executable).resolve()}


__all__ = ["sandboxed_interpreter"]
