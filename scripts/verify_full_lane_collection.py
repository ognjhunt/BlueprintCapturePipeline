#!/usr/bin/env python3
"""Verify a full pytest run used the exact planned collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "blueprint_full_lane_collection.v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest_not_object:{path}")
    return value


def _digest(nodeids: list[str]) -> str:
    return hashlib.sha256("\n".join(nodeids).encode("utf-8")).hexdigest()


def _validate_manifest(payload: dict[str, Any], *, expected_phase: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{expected_phase}_schema_version_invalid")
    if payload.get("phase") != expected_phase:
        errors.append(f"{expected_phase}_phase_invalid")
    raw_nodeids = payload.get("nodeids")
    nodeids = [str(item) for item in raw_nodeids] if isinstance(raw_nodeids, list) else []
    if not nodeids:
        errors.append(f"{expected_phase}_nodeids_empty")
    if payload.get("test_count") != len(nodeids):
        errors.append(f"{expected_phase}_test_count_mismatch")
    if payload.get("nodeids_sha256") != _digest(nodeids):
        errors.append(f"{expected_phase}_nodeids_digest_mismatch")
    return errors


def verify(planned_path: Path, executed_path: Path) -> list[str]:
    planned = _read(planned_path)
    executed = _read(executed_path)
    errors = [
        *_validate_manifest(planned, expected_phase="planned"),
        *_validate_manifest(executed, expected_phase="executed"),
    ]
    if planned.get("test_count") != executed.get("test_count"):
        errors.append("planned_executed_test_count_mismatch")
    if planned.get("nodeids_sha256") != executed.get("nodeids_sha256"):
        errors.append("planned_executed_nodeids_digest_mismatch")
    if planned.get("nodeids") != executed.get("nodeids"):
        errors.append("planned_executed_nodeids_mismatch")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planned", type=Path, required=True)
    parser.add_argument("--executed", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        errors = verify(args.planned.resolve(), args.executed.resolve())
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"[full-lane-collection] ERROR {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"[full-lane-collection] ERROR {error}", file=sys.stderr)
        return 1
    planned = _read(args.planned.resolve())
    print(
        "[full-lane-collection] ok "
        f"(tests={planned['test_count']} sha256={planned['nodeids_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
