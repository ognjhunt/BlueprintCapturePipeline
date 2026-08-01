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
IMAGE_ROOT = ROOT / "deploy/docker/reconstruction_worker"
BOOTSTRAP_INPUT = IMAGE_ROOT / "build-requirements.in"
BOOTSTRAP_OUTPUT = IMAGE_ROOT / "build-requirements.lock"
RUNTIME_INPUT = IMAGE_ROOT / "requirements.in"
RUNTIME_OUTPUT = IMAGE_ROOT / "requirements.lock"
UV_VERSION = "0.10.7"
EXCLUDE_NEWER = "2026-08-01T00:00:00Z"
TARGET = "cpython-3.11 linux-x86_64-manylinux_2_28 torch-cu124"
SOURCE_EXCEPTIONS = ("antlr4-python3-runtime", "asciitree")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_lock(
    text: str,
    *,
    input_path: Path,
    input_digest: str,
    source_exceptions: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    digest_line = f"# blueprint-input-sha256 {input_path.name} {input_digest}"
    if digest_line not in text.splitlines()[:8]:
        errors.append("requirements_input_digest_mismatch")
    if f"# blueprint-target {TARGET}" not in text.splitlines()[:8]:
        errors.append("requirements_target_missing")
    if f"# blueprint-exclude-newer {EXCLUDE_NEWER}" not in text.splitlines()[:8]:
        errors.append("requirements_cutoff_missing")
    source_line = f"# blueprint-source-exceptions {','.join(source_exceptions) or 'none'}"
    if source_line not in text.splitlines()[:8]:
        errors.append("requirements_source_exceptions_mismatch")

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


def _compile_one(
    *, input_path: Path, output_path: Path, source_exceptions: tuple[str, ...]
) -> None:
    input_digest = _sha256(input_path)
    with tempfile.TemporaryDirectory(prefix="blueprint-reconstruction-lock-") as temp_dir:
        generated = Path(temp_dir) / "requirements.lock"
        binary_options = ["--only-binary=:all:"]
        for package in source_exceptions:
            binary_options.append(f"--no-binary={package}")
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(input_path),
                "--python-version",
                "3.11",
                "--python-platform",
                "x86_64-manylinux_2_28",
                "--torch-backend",
                "cu124",
                "--generate-hashes",
                *binary_options,
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
            f"# blueprint-input-sha256 {input_path.name} {input_digest}\n"
            f"# blueprint-target {TARGET}\n"
            f"# blueprint-exclude-newer {EXCLUDE_NEWER}\n"
            f"# blueprint-source-exceptions {','.join(source_exceptions) or 'none'}\n"
        )
        content = header + body
        errors = _validate_lock(
            content,
            input_path=input_path,
            input_digest=input_digest,
            source_exceptions=source_exceptions,
        )
        if errors:
            raise SystemExit(";".join(errors))
        output_path.write_text(content, encoding="utf-8")


def _compile() -> None:
    observed_uv = _uv_version()
    if observed_uv != UV_VERSION:
        raise SystemExit(f"uv_version_mismatch:{observed_uv}:expected:{UV_VERSION}")
    _compile_one(
        input_path=BOOTSTRAP_INPUT,
        output_path=BOOTSTRAP_OUTPUT,
        source_exceptions=(),
    )
    _compile_one(
        input_path=RUNTIME_INPUT,
        output_path=RUNTIME_OUTPUT,
        source_exceptions=SOURCE_EXCEPTIONS,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the checked-in lock without accessing package indexes",
    )
    args = parser.parse_args()
    if args.check:
        errors: list[str] = []
        for input_path, output_path, source_exceptions in (
            (BOOTSTRAP_INPUT, BOOTSTRAP_OUTPUT, ()),
            (RUNTIME_INPUT, RUNTIME_OUTPUT, SOURCE_EXCEPTIONS),
        ):
            errors.extend(
                _validate_lock(
                    output_path.read_text(encoding="utf-8"),
                    input_path=input_path,
                    input_digest=_sha256(input_path),
                    source_exceptions=source_exceptions,
                )
            )
        if errors:
            raise SystemExit(";".join(errors))
    else:
        _compile()
    print(f"build_requirements_lock_sha256={_sha256(BOOTSTRAP_OUTPUT)}")
    print(f"requirements_lock_sha256={_sha256(RUNTIME_OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
