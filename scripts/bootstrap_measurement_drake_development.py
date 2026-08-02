#!/usr/bin/env python3
"""Create an isolated exact Drake runtime for the development adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DRAKE_VERSION = "1.55.0"
ROOT = Path(__file__).resolve().parents[1]


class DrakeBootstrapError(RuntimeError):
    pass


def _run(argv: Sequence[str]) -> None:
    completed = subprocess.run(  # nosec B603 - explicit argv, no shell
        list(argv),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "command_failed"
        raise DrakeBootstrapError(detail)


def _environment_python(environment: Path) -> Path:
    candidates = (environment / "bin/python", environment / "Scripts/python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    raise DrakeBootstrapError("drake_environment_python_missing")


def bootstrap(*, python: Path, environment: Path, uv: Path) -> dict[str, object]:
    python = python.expanduser().absolute()
    environment = environment.expanduser().absolute()
    uv = uv.expanduser().absolute()
    if not python.is_file():
        raise DrakeBootstrapError("drake_bootstrap_python_invalid")
    if not uv.is_file():
        raise DrakeBootstrapError("drake_bootstrap_uv_invalid")
    protected = {Path("/"), Path.home().resolve(), ROOT.resolve(), ROOT.parent.resolve()}
    if environment.resolve() in protected or environment.exists():
        raise DrakeBootstrapError("drake_bootstrap_environment_must_be_new_explicit_path")
    environment.parent.mkdir(parents=True, exist_ok=True)
    _run([str(uv), "venv", "--python", str(python), str(environment)])
    worker_python = _environment_python(environment)
    _run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(worker_python),
            f"drake=={DRAKE_VERSION}",
        ]
    )
    verification = subprocess.run(  # nosec B603 - exact interpreter and source
        [
            str(worker_python),
            "-c",
            (
                "import importlib.metadata,json,platform; "
                "import pydrake.all; "
                "print(json.dumps({'drake_version':importlib.metadata.version('drake'),"
                "'python_version':platform.python_version()}))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if verification.returncode != 0:
        raise DrakeBootstrapError(
            verification.stderr.strip() or "drake_bootstrap_import_verification_failed"
        )
    observed = json.loads(verification.stdout)
    if observed.get("drake_version") != DRAKE_VERSION:
        raise DrakeBootstrapError("drake_bootstrap_version_mismatch")
    result: dict[str, object] = {
        "schema_version": "measurement_drake_development_environment.v1",
        "drake_version": DRAKE_VERSION,
        "python_version": str(observed["python_version"]),
        "worker_python": str(worker_python),
        "environment": str(environment),
        "runtime_scope": "isolated_external_python",
        "development_only": True,
        "production_route_eligible": False,
        "r7_admission": False,
    }
    result["bootstrap_receipt_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--uv", type=Path, default=Path(shutil.which("uv") or ""))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = bootstrap(python=args.python, environment=args.environment, uv=args.uv)
    except (DrakeBootstrapError, OSError, json.JSONDecodeError) as exc:
        print(f"[measurement-drake-bootstrap] ERROR {exc}", file=sys.stderr)
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
