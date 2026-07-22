"""Provider- and campaign-neutral projection of terminal attempt closures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def closure_sha256(closure: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(closure), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def project_attempt_closure(
    closure: Mapping[str, Any],
    *,
    expected_schema_version: str,
    incomplete_blocker: str,
) -> dict[str, Any]:
    """Expose only claims authorized by a completed aggregate closure."""

    value = _mapping(closure)
    valid = (
        value.get("schema_version") == expected_schema_version
        and value.get("status") == "completed"
    )
    rows = {
        str(row.get("row_id") or ""): str(row.get("status") or "")
        for row in value.get("proof_rows", [])
        if isinstance(row, Mapping)
    }
    verified_digests: dict[str, list[str]] = {}
    for row in value.get("proof_rows", []):
        if not isinstance(row, Mapping):
            continue
        leafs = _mapping(row.get("evidence")).get("verified_leaf_artifacts")
        digests = [
            str(item.get("sha256") or "")
            for item in (leafs if isinstance(leafs, Sequence) else [])
            if isinstance(item, Mapping) and item.get("sha256")
        ]
        if digests:
            verified_digests[str(row.get("row_id") or "")] = digests
    return {
        "identity": _mapping(value.get("identity")),
        "verified_leaf_artifact_sha256s": verified_digests,
        "source_schema_version": value.get("schema_version"),
        "source_closure_sha256": closure_sha256(value),
        "status": "ready" if valid else "blocked",
        "task_success_proven": bool(
            valid and rows.get("persistent_simulator_transition") == "passed"
        ),
        "semantic_review_passed": bool(
            valid and rows.get("semantic_review") == "passed"
        ),
        "forward_consistency_passed": bool(
            valid and rows.get("forward_consistency") == "passed"
        ),
        "inverse_consistency_passed": bool(
            valid and rows.get("inverse_consistency") == "passed"
        ),
        "teardown_and_zero_inventory_proven": bool(
            valid
            and rows.get("teardown") == "passed"
            and rows.get("final_inventory") == "passed"
        ),
        "blockers": [] if valid else [incomplete_blocker],
    }
