#!/usr/bin/env python3
"""Verify unchanged service units in an isolated CI filesystem layout.

Only known interpreter paths are staged from real executable files. This is
static unit/security validation, not evidence of an installed production runtime
or a running service. Docker smoke and deployment runtime checks remain separate.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PIPELINE_PYTHON = Path("opt/blueprint/BlueprintCapturePipeline/.venv/bin/python")


def stage_root(unit_dir: Path, root: Path, python_executable: Path) -> list[Path]:
    """Stage fixed executables, never invent an executable from an ExecStart."""
    units = sorted(unit_dir.glob("blueprint-*.service"))
    if not units:
        raise ValueError("no_blueprint_service_units")
    executables = {
        Path("bin/bash"): Path("/bin/bash"),
        Path("usr/bin/python3"): python_executable,
        PIPELINE_PYTHON: python_executable,
    }
    for source in executables.values():
        if not source.is_file() or not os.access(source, os.X_OK):
            raise ValueError(f"interpreter_missing_or_not_executable:{source}")
    root.mkdir()
    # Include ordinary target/dependency units so verification is not reduced
    # with --recursive-errors or disabled default dependencies.
    shutil.copytree("/usr/lib/systemd/system", root / "usr/lib/systemd/system", symlinks=True)
    destination = root / "etc/systemd/system"
    destination.mkdir(parents=True)
    staged = []
    for unit in units:
        target = destination / unit.name
        shutil.copy2(unit, target)
        staged.append(target)
    for target_path, source in executables.items():
        target = root / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return staged


def verify_root(root: Path, units: list[Path]) -> None:
    command = ["systemd-analyze", f"--root={root}"]
    subprocess.run([*command, "verify", *map(str, units)], check=True, timeout=60)
    for unit in units:
        subprocess.run(
            [*command, "security", "--offline=yes", "--no-pager", "--threshold=40", str(unit)],
            check=True,
            timeout=60,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir", type=Path, default=Path("deploy/systemd"))
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        parser.error("canonical CI unit validation requires Python 3.12")
    print("Static systemd unit/security validation; no installed-runtime or deployment proof.", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="blueprint-systemd-ci-") as temporary:
            root = Path(temporary) / "root"
            units = stage_root(args.unit_dir, root, Path(sys.executable))
            verify_root(root, units)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"systemd_ci_verification_failed:{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
