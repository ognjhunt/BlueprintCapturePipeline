"""Stage a module and everything it needs into a provider bundle's flat layout.

Provider bundles are flat: the modules that live in a package here are dropped
side by side there, which is why they all carry a ``try: from x import ...
except ModuleNotFoundError: from .x import ...`` pair. That fallback only works
if every module the entry point reaches is actually present, and the natural
mistake is to stage the obvious two or three and let the rest be discovered by
the runtime.

They are discovered on a GPU that is already billing, one missing module per
launch. So the closure is computed by following imports rather than listed by
hand, and - the part that matters - the staged directory is then imported for
real, flat, in a subprocess. A layout that cannot import is free to find here.

Verification runs in a subprocess deliberately. Importing the staged copies
into this interpreter would shadow the package versions for everything that
follows, and a check that corrupts the process it runs in is not a check worth
having.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


FLAT_MODULE_CLOSURE_SCHEMA_VERSION = "flat_module_closure.v1"


class FlatModuleClosureError(ValueError):
    """Stable, sorted flat-staging failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def resolve_flat_module_closure(
    *, package_root: str | Path, entry_modules: Sequence[str]
) -> list[str]:
    """Every module inside the package that the entry points can reach."""

    root = Path(package_root).expanduser().resolve()
    if not root.is_dir():
        raise FlatModuleClosureError(["flat_module_closure_package_root_missing"])
    entries = [str(name) for name in entry_modules if str(name)]
    if not entries:
        raise FlatModuleClosureError(["flat_module_closure_entry_modules_missing"])

    errors = [
        f"flat_module_closure_entry_module_missing:{name}"
        for name in entries
        if not (root / f"{name}.py").is_file()
    ]
    if errors:
        raise FlatModuleClosureError(errors)

    seen: set[str] = set()
    stack = list(entries)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        source = root / f"{name}.py"
        if not source.is_file():
            continue
        seen.add(name)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            # Relative imports name a sibling; absolute ones only count when a
            # sibling of that name exists, so third-party imports are ignored.
            if node.level == 1 or (root / f"{node.module}.py").is_file():
                stack.append(node.module)
    return sorted(seen)


def stage_flat_module_closure(
    *,
    package_root: str | Path,
    entry_modules: Sequence[str],
    destination: str | Path,
    verify_import: bool = True,
) -> dict[str, Any]:
    """Copy the closure flat, then prove it imports that way."""

    root = Path(package_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    closure = resolve_flat_module_closure(
        package_root=root, entry_modules=entry_modules
    )
    target.mkdir(parents=True, exist_ok=True)
    for name in closure:
        shutil.copy2(root / f"{name}.py", target / f"{name}.py")

    verified = None
    failures: list[str] = []
    if verify_import:
        for name in [str(value) for value in entry_modules]:
            # A subprocess, so a staged copy cannot shadow the package version
            # for whatever runs next in this interpreter.
            probe = subprocess.run(
                [sys.executable, "-c", f"import {name}"],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if probe.returncode != 0:
                tail = (probe.stderr or "").strip().splitlines()[-1:] or [""]
                failures.append(
                    f"flat_module_closure_flat_import_failed:{name}:{tail[0][:200]}"
                )
        verified = not failures
    if failures:
        raise FlatModuleClosureError(failures)

    return {
        "schema_version": FLAT_MODULE_CLOSURE_SCHEMA_VERSION,
        "package_root": str(root),
        "destination": str(target),
        "entry_modules": [str(value) for value in entry_modules],
        "staged_modules": closure,
        "staged_module_count": len(closure),
        "import_verified": verified,
        "claim_boundary": {
            "import_check_is_syntax_and_resolution_not_behaviour": True,
            "third_party_dependencies_are_not_staged": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--no-verify", action="store_true")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    receipt = stage_flat_module_closure(
        package_root=arguments.package_root,
        entry_modules=arguments.entry,
        destination=arguments.destination,
        verify_import=not arguments.no_verify,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FLAT_MODULE_CLOSURE_SCHEMA_VERSION",
    "FlatModuleClosureError",
    "resolve_flat_module_closure",
    "stage_flat_module_closure",
]


if __name__ == "__main__":
    raise SystemExit(main())
