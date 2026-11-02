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
already prepared both, ARTISAN did neither. A later run then failed in the
sandbox with ``libpython3.12.so.1.0: cannot open shared object file``: this
module kept only launcher library entries under ``sys.base_prefix`` and looked
for libpython only beside the prefix, and on that image it lives elsewhere.
The launcher's whole library path is now passed back (as the Astra backend,
proven on the same image, does), and the directory of the libpython this
process actually loaded is read from ``/proc/self/maps``.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


def _loaded_libpython_directories(version: str, maps_path: Path = Path("/proc/self/maps")) -> list[Path]:
    """Directories of the libpython this process mapped, as the loader resolved it."""

    try:
        maps = maps_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(f"libpython{version}.so"):
            path = Path(fields[5])
            if path.is_absolute() and path.is_file():
                found.append(path.resolve().parent)
    return list(dict.fromkeys(found))


def sandboxed_interpreter() -> dict[str, Any]:
    """Roots, loader environment and executable for running ``sys.executable`` sandboxed."""

    base = Path(sys.base_prefix).resolve()
    library_paths: list[Path] = []
    # The launcher's own library path, as this interpreter was started with it.
    for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        path = Path(value) if value and Path(value).is_absolute() else None
        if path is not None and path.is_dir():
            library_paths.append(path.resolve())
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    library_paths.extend(_loaded_libpython_directories(version))
    for parent in (base / "lib", base.parent):
        if parent.is_dir() and any(parent.glob(f"libpython{version}.so*")):
            library_paths.append(parent.resolve())
    library_paths = list(dict.fromkeys(library_paths))
    roots = list(dict.fromkeys([base, *library_paths]))
    environment = {"LD_LIBRARY_PATH": os.pathsep.join(map(str, library_paths))} if library_paths else {}
    return {"roots": roots, "library_environment": environment,
            "executable": Path(sys.executable).resolve()}


__all__ = ["sandboxed_interpreter"]
