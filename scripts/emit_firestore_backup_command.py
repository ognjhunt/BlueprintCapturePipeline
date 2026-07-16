#!/usr/bin/env python3
"""Emit (and optionally run) the scheduled Firestore export command from the backup config.

Finding **R053 (P1)**: no scheduled backup for the authoritative Firestore control-plane.

This is the committed, honest half of the backup job: it reads the validated backup config
(``configs/firestore_backup_schedule.json``), renders the exact ``gcloud firestore export``
command for a timestamped destination path, and prints it. It NEVER touches infrastructure
by default -- ``--execute`` invokes ``gcloud`` (which must be installed and authenticated
as the backup service account), so a live export remains a deploy/ops step.

The config is validated first (via ``validate_firestore_backup_config``); a malformed config
fails closed and no command is emitted.

Usage:
    python3 scripts/emit_firestore_backup_command.py            # print the export command
    python3 scripts/emit_firestore_backup_command.py --execute  # run it (requires gcloud + auth)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "firestore_backup_schedule.json"
_VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_firestore_backup_config.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("_bp_backup_validator", _VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def render_command(config: dict, timestamp: str) -> str:
    export = config["firestore_export"]
    return export["gcloud_command_template"].format(
        destination_bucket=export["destination_bucket"],
        destination_prefix=export["destination_prefix"],
        project_id=export["project_id"],
        database=export["database"],
        timestamp=timestamp,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the export via gcloud (requires gcloud installed + authenticated).",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Override the export id timestamp (default: current UTC, e.g. 20260709T070000Z).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Fail closed on a bad config before emitting anything.
    validator = _load_validator()
    if not args.config.exists():
        print(f"Backup config missing: {args.config}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Keep stdout clean (command only) by routing the validator's success line to stderr.
    _saved_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        validator.validate_backup_config(config, args.config.name)
    finally:
        sys.stdout = _saved_stdout

    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    command = render_command(config, timestamp)

    if not args.execute:
        print(command)
        return

    command_argv = shlex.split(command)
    if command_argv[:3] != ["gcloud", "firestore", "export"]:
        print("Backup command did not resolve to gcloud firestore export", file=sys.stderr)
        sys.exit(1)
    print(f"# executing: {shlex.join(command_argv)}", file=sys.stderr)
    result = subprocess.run(command_argv, shell=False, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
