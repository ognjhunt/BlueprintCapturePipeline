"""Sign same-allocation startup gates into attempt-bound proof rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .g1_kitchen_leaf_evidence import load_attempt_identity, write_attested_leaf


SPECS = {
    "startup": ("supervised_startup_gates.json", "groot_oscar_same_allocation_startup_gates.v1"),
    "fast_canary": ("isaac_worker_runtime_preflight.json", "isaac_worker_runtime_preflight.v1"),
    "review_canary": (
        "review/isaac_review_renderer_canary.json",
        "isaac_review_renderer_canary.v1",
    ),
    "asset_gate": (
        "kitchen/kitchen_asset_startup_gate.json",
        "kitchen_asset_startup_gate.v1",
    ),
}


def sign_startup_proof_rows(
    *, startup_dir: str | Path, attempt_input_manifest: str | Path
) -> dict[str, Any]:
    root = Path(startup_dir)
    identity = load_attempt_identity(attempt_input_manifest)
    rows: dict[str, Any] = {}
    for row_id, (relative, schema) in SPECS.items():
        source = root / relative
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != schema:
            raise ValueError(f"startup_leaf_schema_mismatch:{row_id}")
        target_name = f"startup_{row_id}.json"
        ref = write_attested_leaf(
            payload=payload,
            path=root.parent / "proof_leaves" / target_name,
            reference_path=f"closed_loop_out/proof_leaves/{target_name}",
            identity=identity,
            role="startup",
        )
        rows[row_id] = {
            "status": "passed",
            "identity_binding": dict(identity),
            "leaf_artifacts": [ref],
            "blockers": [],
        }
    output = root / "startup_proof_rows.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-dir", required=True)
    parser.add_argument("--attempt-input-manifest", required=True)
    args = parser.parse_args()
    sign_startup_proof_rows(
        startup_dir=args.startup_dir,
        attempt_input_manifest=args.attempt_input_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
