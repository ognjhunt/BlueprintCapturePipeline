#!/usr/bin/env python3
"""Verify that release-critical source fixtures exist in a clean checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE_FIXTURE_FILES = (
    "tests/fixtures/warehouse_task_min/README.md",
    "tests/fixtures/warehouse_task_min/assets/initial_policy_observation.jpg",
    "tests/fixtures/warehouse_task_min/assets/robot_state.json",
    "tests/fixtures/warehouse_task_min/assets/unitree_g1.xml",
    "tests/fixtures/warehouse_task_min/assets/warehouse_min.splat",
    "tests/fixtures/warehouse_task_min/raw/manifest.json",
    "tests/fixtures/warehouse_task_min/raw/object_index.json",
    "tests/fixtures/warehouse_task_min/pipeline/evaluation_prep/task_anchor_manifest.json",
    "tests/fixtures/warehouse_task_min/pipeline/geometry/camera/intrinsics.json",
)


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in WAREHOUSE_FIXTURE_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing_release_input:{relative}")

    json_expectations = {
        "tests/fixtures/warehouse_task_min/raw/object_index.json": "objects",
        "tests/fixtures/warehouse_task_min/pipeline/evaluation_prep/task_anchor_manifest.json": "tasks",
        "tests/fixtures/warehouse_task_min/pipeline/geometry/camera/intrinsics.json": "fx",
    }
    for relative, required_key in json_expectations.items():
        path = root / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_release_input_json:{relative}:{exc.__class__.__name__}")
            continue
        if not isinstance(payload, dict) or required_key not in payload:
            errors.append(f"release_input_key_missing:{relative}:{required_key}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    errors = validate(args.root.resolve())
    if errors:
        for error in errors:
            print(f"[clean-checkout-inputs] ERROR {error}", file=sys.stderr)
        return 1
    print("[clean-checkout-inputs] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
