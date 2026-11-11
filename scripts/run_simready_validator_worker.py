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
    content = parser.add_mutually_exclusive_group(required=True)
    content.add_argument("--source-root", type=Path)
    content.add_argument("--installed-tier-version")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-version", required=True)
    parser.add_argument("--expected-validator-version", required=True)
    args = parser.parse_args()
    native_output = args.output.with_name(args.output.stem + "_native.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native_output.unlink(missing_ok=True)
    content_args = []
    installed_tier = None
    if args.source_root:
        source = args.source_root.resolve()
        tier = source / "nv_core/tiers/simready_foundation_tier_core/simready/foundation/tier_core"
        specs = tier if tier.is_dir() else source / "nv_core/sr_specs/docs"
        profiles = specs / "profiles" if tier.is_dir() else specs / "profiles/profiles.toml"
        content_args = ["--rules-path", str(specs / "capabilities"),
                        "--features-path", str(specs / "features"), "--profiles-path", str(profiles)]
    else:
        installed_tier = importlib.metadata.version("simready-foundation-tier-core")
        if installed_tier != args.installed_tier_version:
            raise ValueError("simready_installed_tier_version_mismatch")
    # Use the CLI from this interpreter's environment, never a floating PATH entry.
    import sys
    executable = Path(sys.executable).parent / "simready-validate"
    command = [str(executable), *content_args,
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
    report_available = isinstance(native, dict) and bool(native) and native.get("status") != "invalid_native_json"
    report = {
        "schema_version": "blueprint_simready_validator_worker.v1",
        "status": "passed" if completed.returncode == 0 and identity_ok and report_available else "failed",
        "validator_version": installed,
        "foundation_tier_version": installed_tier,
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
    if not identity_ok or not report_available:
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
