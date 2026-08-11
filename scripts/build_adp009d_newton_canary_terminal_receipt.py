#!/usr/bin/env python3
"""Compile one observed Newton canary terminal receipt from retained evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blueprint_pipeline.adp009d_physics_backend_comparison import (
    build_newton_canary_terminal_receipt,
)
from blueprint_pipeline.common import write_json


def _read(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "admission",
        "bundle_receipt",
        "allocator_result",
        "artifact_manifest",
        "teardown_manifest",
        "provider_inventory",
        "vast_billing_response",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--native-result")
    args = parser.parse_args()
    teardown = _read(args.teardown_manifest)
    instance_ids = teardown.get("vast_instance_ids")
    if not isinstance(instance_ids, list) or len(instance_ids) != 1:
        raise ValueError("exactly_one_vast_instance_required")
    response = _read(args.vast_billing_response)
    matches = [
        row
        for row in response.get("results") or []
        if isinstance(row, dict)
        and row.get("source") == f"instance-{instance_ids[0]}"
    ]
    if len(matches) != 1:
        raise ValueError("exact_vast_instance_charge_not_found")
    receipt = build_newton_canary_terminal_receipt(
        admission=_read(args.admission),
        bundle_receipt=_read(args.bundle_receipt),
        allocator_result=_read(args.allocator_result),
        native_result=_read(args.native_result) if args.native_result else None,
        artifact_manifest=_read(args.artifact_manifest),
        teardown_manifest=teardown,
        provider_inventory=_read(args.provider_inventory),
        vast_charge=matches[0],
    )
    write_json(Path(args.output), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
