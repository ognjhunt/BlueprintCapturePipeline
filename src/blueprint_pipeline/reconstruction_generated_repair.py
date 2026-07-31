"""Fail-closed registered gate for generated reconstruction repair candidates."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .reconstruction_enhancement_audit import enhancement_method_audit


GENERATED_REPAIR_REQUEST_SCHEMA_VERSION = "generated_repair_candidate_request.v1"
GENERATED_REPAIR_RESULT_SCHEMA_VERSION = "generated_repair_candidate_result.v1"


class GeneratedRepairContractError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise GeneratedRepairContractError(["generated_repair_artifact_not_json"]) from exc


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def build_generated_repair_candidate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(dict(value))
    errors: list[str] = []
    if request.get("schema_version") != GENERATED_REPAIR_REQUEST_SCHEMA_VERSION:
        errors.append("generated_repair_request_schema_invalid")
    method_id = str(request.get("method_id") or "")
    try:
        expected_audit = enhancement_method_audit(method_id)
    except ValueError as exc:
        raise GeneratedRepairContractError(["generated_repair_method_unknown"]) from exc
    if request.get("enhancement_method_audit") != expected_audit:
        errors.append("generated_repair_audit_binding_invalid")
    for key in (
        "source_capture_digest",
        "baseline_reconstruction_digest",
        "frozen_split_digest",
        "heldout_evaluation_contract_digest",
    ):
        if not _digest(request.get(key)):
            errors.append(f"generated_repair_request_{key}_invalid")
    if request.get("baseline_preserved") is not True:
        errors.append("generated_repair_request_baseline_preservation_required")
    if request.get("candidate_may_read_hidden_heldout") is not False:
        errors.append("generated_repair_request_hidden_access_forbidden")
    if request.get("generated_pixels_may_be_promoted_to_capture") is not False:
        errors.append("generated_repair_request_capture_promotion_forbidden")
    if not isinstance(request.get("authority_used"), Mapping):
        errors.append("generated_repair_request_authority_invalid")
    if not str(request.get("stable_run_identity") or "").strip() or not str(
        request.get("timestamp") or ""
    ).strip():
        errors.append("generated_repair_request_identity_or_timestamp_missing")
    supplied = request.pop("generated_repair_candidate_request_digest", None)
    request["generated_repair_candidate_request_digest"] = canonical_digest(
        request, digest_field="generated_repair_candidate_request_digest"
    )
    if supplied is not None and supplied != request[
        "generated_repair_candidate_request_digest"
    ]:
        errors.append("generated_repair_request_digest_mismatch")
    if errors:
        raise GeneratedRepairContractError(errors)
    return request


def run_generated_repair_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Record the current deterministic qualification rejection without execution."""

    request = build_generated_repair_candidate_request(value)
    audit = request["enhancement_method_audit"]
    if not str(audit["status"]).startswith("rejected_") or not audit["blockers"]:
        raise GeneratedRepairContractError(["generated_repair_qualified_runtime_not_registered"])
    result = {
        "schema_version": GENERATED_REPAIR_RESULT_SCHEMA_VERSION,
        "stable_run_identity": request["stable_run_identity"],
        "source_capture_digest": request["source_capture_digest"],
        "generated_repair_candidate_request_digest": request[
            "generated_repair_candidate_request_digest"
        ],
        "method_id": request["method_id"],
        "enhancement_method_audit_digest": audit["enhancement_method_audit_digest"],
        "baseline_reconstruction_digest": request["baseline_reconstruction_digest"],
        "frozen_split_digest": request["frozen_split_digest"],
        "status": "blocked_not_qualified",
        "execution_started": False,
        "generated_artifact_references": [],
        "blockers": list(audit["blockers"]),
        "legal_next_actions": list(audit["legal_next_actions"]),
        "baseline_preserved": True,
        "hidden_heldout_available_to_candidate": False,
        "independent_heldout_evaluation_executed": False,
        "generated_pixels_are_captured_evidence": False,
        "metric_or_collision_proof_effect": False,
        "physical_or_deployment_proof_effect": False,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": request["authority_used"],
        "proof_effect": "deterministic_enhancement_rejection_only",
        "claim_ceiling": "generated_visual_support",
        "parent_artifact_or_event": {
            "baseline_reconstruction_digest": request["baseline_reconstruction_digest"]
        },
        "timestamp": request["timestamp"],
    }
    result["generated_repair_candidate_result_digest"] = canonical_digest(
        result, digest_field="generated_repair_candidate_result_digest"
    )
    return result


__all__ = [
    "GENERATED_REPAIR_REQUEST_SCHEMA_VERSION",
    "GENERATED_REPAIR_RESULT_SCHEMA_VERSION",
    "GeneratedRepairContractError",
    "build_generated_repair_candidate_request",
    "run_generated_repair_candidate",
]
