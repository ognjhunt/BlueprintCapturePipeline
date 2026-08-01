#!/usr/bin/env python3
"""Compile the reconstruction worker's Linux/amd64 hash lock."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "deploy/docker/reconstruction_worker/requirements.in"
OUTPUT = ROOT / "deploy/docker/reconstruction_worker/requirements.lock"
UV_VERSION = "0.10.7"
EXCLUDE_NEWER = "2026-08-01T00:00:00Z"
TARGET = "cpython-3.11 linux-x86_64-manylinux_2_28 torch-cu124"
INPUT_DIGEST_PREFIX = "# blueprint-input-sha256 requirements.in "


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_lock(text: str, *, input_digest: str) -> list[str]:
    errors: list[str] = []
    if f"{INPUT_DIGEST_PREFIX}{input_digest}" not in text.splitlines()[:8]:
        errors.append("requirements_input_digest_mismatch")
    if f"# blueprint-target {TARGET}" not in text.splitlines()[:8]:
        errors.append("requirements_target_missing")
    if f"# blueprint-exclude-newer {EXCLUDE_NEWER}" not in text.splitlines()[:8]:
        errors.append("requirements_cutoff_missing")

    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+==)", text)
    requirements = [block for block in blocks if re.match(r"^[A-Za-z0-9_.-]+==", block)]
    if not requirements:
        errors.append("requirements_missing")
    for block in requirements:
        first = block.splitlines()[0]
        if "==" not in first:
            errors.append("requirement_not_exact:" + first)
        if "--hash=sha256:" not in block:
            errors.append("requirement_hash_missing:" + first)
    return errors


def _uv_version() -> str:
    completed = subprocess.run(
        ["uv", "--version"], check=True, capture_output=True, text=True
    )
    match = re.match(r"^uv ([0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)", completed.stdout.strip())
    if match is None:
        raise SystemExit("uv_version_unparseable")
    return match.group(1)


def _compile() -> None:
    observed_uv = _uv_version()
    if observed_uv != UV_VERSION:
        raise SystemExit(f"uv_version_mismatch:{observed_uv}:expected:{UV_VERSION}")
    input_digest = _sha256(INPUT)
    with tempfile.TemporaryDirectory(prefix="blueprint-reconstruction-lock-") as temp_dir:
        generated = Path(temp_dir) / "requirements.lock"
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(INPUT),
                "--python-version",
                "3.11",
                "--python-platform",
                "x86_64-manylinux_2_28",
                "--torch-backend",
                "cu124",
                "--generate-hashes",
                "--only-binary=:all:",
                "--no-annotate",
                "--exclude-newer",
                EXCLUDE_NEWER,
                "--custom-compile-command",
                "python scripts/compile_reconstruction_worker_lock.py",
                "--output-file",
                str(generated),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        body = generated.read_text(encoding="utf-8")
        header = (
            f"{INPUT_DIGEST_PREFIX}{input_digest}\n"
            f"# blueprint-target {TARGET}\n"
            f"# blueprint-exclude-newer {EXCLUDE_NEWER}\n"
        )
        content = header + body
        errors = _validate_lock(content, input_digest=input_digest)
        if errors:
            raise SystemExit(";".join(errors))
        OUTPUT.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the checked-in lock without accessing package indexes",
    )
    args = parser.parse_args()
    if args.check:
        errors = _validate_lock(
            OUTPUT.read_text(encoding="utf-8"), input_digest=_sha256(INPUT)
        )
        if errors:
            raise SystemExit(";".join(errors))
    else:
        _compile()
    print(f"requirements_lock_sha256={_sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
