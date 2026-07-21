#!/usr/bin/env python3
"""Machine-readable wrapper around the isolated usd-convert-gsplat CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    try:
        installed = importlib.metadata.version("usd-convert-gsplat")
    except importlib.metadata.PackageNotFoundError:
        installed = "not-installed"
    command = [
        sys.executable,
        "-m",
        "usd_convert_gsplat",
        "-i",
        str(args.input),
        "-o",
        str(args.oracle_output),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    identity_ok = installed == args.expected_version
    output_ok = args.oracle_output.is_file() and args.oracle_output.stat().st_size > 0
    status = 0 if completed.returncode == 0 and identity_ok and output_ok else 2
    report = {
        "schema_version": "usd_convert_gsplat_worker_result.v1",
        "status": "completed" if status == 0 else "blocked",
        "converter_version": installed,
        "source_revision": args.source_revision,
        "input_sha256": _sha(args.input),
        "output_path": str(args.oracle_output.resolve()),
        "output_sha256": _sha(args.oracle_output),
        "subprocess_returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "blockers": [
            blocker
            for blocker, triggered in (
                ("converter_version_mismatch", not identity_ok),
                ("converter_cli_failed", completed.returncode != 0),
                ("converter_output_missing_or_empty", not output_ok),
            )
            if triggered
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
