#!/usr/bin/env python3
"""Normalize the official simready-validate CLI into Blueprint worker JSON."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-version", required=True)
    parser.add_argument("--expected-validator-version", required=True)
    args = parser.parse_args()
    specs = args.source_root.resolve() / "nv_core" / "sr_specs" / "docs"
    native_output = args.output.with_name(args.output.stem + "_native.json")
    command = [
        "simready-validate",
        "--rules-path",
        str(specs / "capabilities"),
        "--features-path",
        str(specs / "features"),
        "--profiles-path",
        str(specs / "profiles" / "profiles.toml"),
        "--profile",
        args.profile,
        "--version",
        args.profile_version,
        "--output",
        str(native_output),
        str(args.input),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    try:
        installed = importlib.metadata.version("simready-validate")
    except importlib.metadata.PackageNotFoundError:
        installed = "not-installed"
    native: object = {}
    if native_output.is_file():
        try:
            native = json.loads(native_output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            native = {"status": "invalid_native_json"}
    identity_ok = installed == args.expected_validator_version
    report = {
        "schema_version": "blueprint_simready_validator_worker.v1",
        "status": "passed" if completed.returncode == 0 and identity_ok else "failed",
        "validator_version": installed,
        "profile_name": args.profile,
        "profile_version": args.profile_version,
        "native_report": native,
        "native_report_path": str(native_output.resolve()),
        "subprocess_returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "findings": native.get("findings", []) if isinstance(native, dict) else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not identity_ok:
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
