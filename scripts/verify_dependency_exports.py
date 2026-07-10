#!/usr/bin/env python3
"""Verify pip compatibility requirements are frozen exports of uv.lock."""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = {
    "requirements.txt": (),
    "requirements-geometry.txt": ("--extra", "geometry"),
}
BASE_COMMAND = (
    "uv",
    "export",
    "--frozen",
    "--no-dev",
    "--format",
    "requirements-txt",
)


def verify_lock(*, root: Path = ROOT) -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("uv executable is required to verify dependency exports")
    completed = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "uv.lock is inconsistent with pyproject.toml")


def render(extra_args: tuple[str, ...], *, root: Path = ROOT) -> str:
    if shutil.which("uv") is None:
        raise RuntimeError("uv executable is required to verify dependency exports")
    completed = subprocess.run(
        [*BASE_COMMAND, *extra_args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "uv export failed")
    return completed.stdout


def verify(*, root: Path = ROOT, write: bool = False) -> list[str]:
    errors: list[str] = []
    verify_lock(root=root)
    for relative, extra_args in EXPORTS.items():
        path = root / relative
        expected = render(extra_args, root=root)
        if write:
            path.write_text(expected, encoding="utf-8")
            continue
        actual = path.read_text(encoding="utf-8") if path.is_file() else ""
        if actual == expected:
            continue
        diff = list(
            difflib.unified_diff(
                actual.splitlines(),
                expected.splitlines(),
                fromfile=relative,
                tofile=f"uv.lock export for {relative}",
                lineterm="",
            )
        )
        preview = "\n".join(diff[:80])
        errors.append(f"dependency_export_drift:{relative}\n{preview}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate compatibility exports.")
    args = parser.parse_args(argv)
    try:
        errors = verify(write=args.write)
    except (OSError, RuntimeError) as exc:
        print(f"[dependency-exports] ERROR {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"[dependency-exports] ERROR {error}", file=sys.stderr)
        return 1
    action = "wrote" if args.write else "ok"
    print(f"[dependency-exports] {action} ({len(EXPORTS)} exports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
