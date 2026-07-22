"""Fail-closed closure for one G1 kitchen simulator attempt.

Leaf artifacts in this lane use words such as ``completed`` for very different
things.  This module is the single authority that joins those artifacts without
promoting renderer, provider, or evaluator completion into task success.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .attempt_closure_projection import (
    project_attempt_closure,
)


SCHEMA_VERSION = "g1_kitchen_attempt_closure.v1"
IDENTITY_FIELDS = (
    "run_id",
    "attempt_id",
    "launch_nonce",
    "source_commit",
    "source_dirty_patch_sha256",
    "image_digest",
    "bundle_digest",
    "kitchen_asset_digest",
    "active_selection_sha256",
    "task_contract_sha256",
    "provider_allocation_id",
)
PROOF_ROW_IDS = (
    "allocation",
    "startup",
    "fast_canary",
    "review_canary",
    "asset_gate",
    "scene_load",
    "target",
    "stance",
    "collision",
    "robot_pov",
    "controller_fk",
    "persistent_simulator_transition",
    "semantic_review",
    "forward_consistency",
    "inverse_consistency",
    "teardown",
    "final_inventory",
)
ROW_STATUSES = frozenset({"passed", "blocked", "not_requested"})


def persistent_task_identity_rows(
    task_completion_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Project stage and target identity without claiming task transition success."""
    results = [dict(row) for row in task_completion_results]
    stage_ids = sorted(
        {str(row.get("stage_id") or "").strip() for row in results if row.get("stage_id")}
    )
    prim_paths = sorted(
        {
            str(row.get("articulation_prim_path") or "").strip()
            for row in results
            if row.get("articulation_prim_path")
        }
    )
    return {
        "scene_load": {
            "status": "passed" if results and len(stage_ids) == 1 else "blocked",
            "evidence": {
                "persistent_stage_ids": stage_ids,
                "task_completion_results": results,
            },
        },
        "target": {
            "status": "passed"
            if results
            and all(
                str(row.get("articulation_prim_path") or "").startswith("/")
                for row in results
            )
            else "blocked",
            "evidence": {"resolved_articulation_prim_paths": prim_paths},
        },
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return text


def _valid_sha256(value: Any) -> bool:
    text = _sha256_text(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _valid_source_commit(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return 7 <= len(text) <= 64 and all(char in "0123456789abcdef" for char in text)


def validate_identity(identity: Mapping[str, Any]) -> list[str]:
    """Return deterministic blockers for an immutable attempt identity."""
    detail = _mapping(identity)
    blockers: list[str] = []
    for field in IDENTITY_FIELDS:
        if not str(detail.get(field) or "").strip():
            blockers.append(f"identity_missing:{field}")
    if detail.get("source_commit") and not _valid_source_commit(detail["source_commit"]):
        blockers.append("identity_source_commit_invalid")
    for field in (
        "image_digest",
        "bundle_digest",
        "kitchen_asset_digest",
        "active_selection_sha256",
        "task_contract_sha256",
    ):
        if detail.get(field) and not _valid_sha256(detail[field]):
            blockers.append(f"identity_digest_invalid:{field}")
    if detail.get("run_id") == detail.get("attempt_id"):
        blockers.append("identity_run_and_attempt_ids_must_differ")
    return sorted(set(blockers))


def _row_binding_blockers(
    row_id: str, row: Mapping[str, Any], identity: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    binding = _mapping(row.get("identity_binding"))
    if row.get("status") == "passed":
        for field in IDENTITY_FIELDS:
            if field not in binding or not str(binding.get(field) or "").strip():
                blockers.append(f"{row_id}:passed_row_identity_binding_missing:{field}")
    for field, observed in binding.items():
        if field not in IDENTITY_FIELDS:
            blockers.append(f"{row_id}:unknown_identity_binding:{field}")
        elif str(observed or "") != str(identity.get(field) or ""):
            blockers.append(f"{row_id}:identity_binding_mismatch:{field}")
    return blockers


def _terminal_proof_blockers(row_id: str, row: Mapping[str, Any]) -> list[str]:
    evidence = _mapping(row.get("evidence"))
    if row_id == "teardown" and row.get("status") == "passed":
        if evidence.get("api_confirmed") is not True:
            return ["teardown:provider_api_confirmation_missing"]
        if str(evidence.get("terminal_state") or "").lower() not in {
            "not_found",
            "deleted",
            "terminated",
        }:
            return ["teardown:provider_terminal_state_not_proven"]
    if row_id == "final_inventory" and row.get("status") == "passed":
        blockers = []
        if evidence.get("api_confirmed") is not True:
            blockers.append("final_inventory:provider_api_confirmation_missing")
        try:
            count = int(evidence.get("live_resource_count"))
        except (TypeError, ValueError):
            count = -1
        if count != 0:
            blockers.append("final_inventory:live_resource_count_not_zero")
        return blockers
    return []


def build_attempt_closure(
    *,
    identity: Mapping[str, Any],
    proof_rows: Mapping[str, Mapping[str, Any]],
    required_rows: Sequence[str] = PROOF_ROW_IDS,
    terminal_reason: str | None = None,
) -> dict[str, Any]:
    """Normalize one attempt and close it only when every required row passes."""
    immutable_identity = {field: identity.get(field) for field in IDENTITY_FIELDS}
    blockers = validate_identity(immutable_identity)
    required = tuple(dict.fromkeys(str(item) for item in required_rows))
    unknown_required = sorted(set(required) - set(PROOF_ROW_IDS))
    blockers.extend(f"unknown_required_proof_row:{item}" for item in unknown_required)
    rows: list[dict[str, Any]] = []
    for row_id in PROOF_ROW_IDS:
        raw = _mapping(proof_rows.get(row_id))
        status = str(raw.get("status") or "blocked")
        row_blockers = [str(item) for item in raw.get("blockers", []) if str(item)]
        if status not in ROW_STATUSES:
            row_blockers.append(f"{row_id}:invalid_status:{status or 'missing'}")
            status = "blocked"
        if row_id in required and status == "not_requested":
            row_blockers.append(f"{row_id}:required_row_not_requested")
            status = "blocked"
        normalized = {
            "row_id": row_id,
            "required": row_id in required,
            "status": status,
            "blockers": sorted(set(row_blockers)),
            "identity_binding": _mapping(raw.get("identity_binding")),
            "evidence": _mapping(raw.get("evidence")),
            "artifact_refs": list(raw.get("artifact_refs") or []),
        }
        normalized["blockers"].extend(
            _row_binding_blockers(row_id, normalized, immutable_identity)
        )
        normalized["blockers"].extend(_terminal_proof_blockers(row_id, normalized))
        normalized["blockers"] = sorted(set(normalized["blockers"]))
        if normalized["blockers"] and normalized["status"] == "passed":
            normalized["status"] = "blocked"
        if normalized["required"] and normalized["status"] != "passed":
            blockers.extend(normalized["blockers"] or [f"proof_row_not_passed:{row_id}"])
        rows.append(normalized)
    status = "completed" if not blockers else "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": status,
        "terminal": True,
        "terminal_reason": terminal_reason or ("all_required_proof_passed" if status == "completed" else "proof_incomplete"),
        "identity": immutable_identity,
        "required_rows": list(required),
        "proof_rows": rows,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "renderer_completion_is_not_task_success": True,
            "structural_loop_completion_is_not_task_success": True,
            "marker_verification_is_not_worker_result": True,
            "sim_only": True,
            "physical_robot_readiness_proven": False,
        },
    }


def append_attempt_closure(path: str | Path, closure: Mapping[str, Any]) -> dict[str, Any]:
    """Append a closure as JSONL while atomically rejecting duplicate IDs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = _mapping(closure)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("terminal") is not True:
        raise ValueError("terminal g1 kitchen attempt closure required")
    identity = _mapping(value.get("identity"))
    key = (str(identity.get("run_id") or ""), str(identity.get("attempt_id") or ""))
    flags = "a+" if target.exists() else "x+"
    with target.open(flags, encoding="utf-8") as handle:
        handle.seek(0)
        for line in handle:
            if not line.strip():
                continue
            existing = _mapping(json.loads(line))
            prior = _mapping(existing.get("identity"))
            if (str(prior.get("run_id") or ""), str(prior.get("attempt_id") or "")) == key:
                raise ValueError("duplicate run/attempt closure identity")
        handle.seek(0, 2)
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    return {"status": "appended", "path": str(target), "run_id": key[0], "attempt_id": key[1]}


def buyer_readout_projection(closure: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only claims authorized by this closure, never by leaf statuses."""
    return project_attempt_closure(
        closure,
        expected_schema_version=SCHEMA_VERSION,
        incomplete_blocker="g1_kitchen_attempt_closure_not_completed",
    )
