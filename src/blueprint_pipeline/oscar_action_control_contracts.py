"""Fail-closed controls for OSCAR action-conditioning sensitivity evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "oscar_action_control_suite.v1"
EXECUTED_CONTROL_KINDS = {"zero", "sign_flipped", "scaled", "reordered"}
REJECTION_CONTROL_KINDS = {"stale", "repeated", "unrelated"}
REQUIRED_CONTROL_KINDS = EXECUTED_CONTROL_KINDS | REJECTION_CONTROL_KINDS
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _digest(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "").strip().lower()))


def validate_oscar_action_control_suite(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Validate counterfactual executions and fail-closed replay rejection controls."""

    blockers: list[str] = []
    if suite.get("schema_version") != SCHEMA_VERSION:
        blockers.append("oscar_action_control_suite_schema_missing_or_unsupported")
    for field in (
        "base_commanded_action_sha256",
        "base_skeleton_conditioning_sha256",
        "base_model_output_sha256",
        "oscar_checkpoint_sha256",
        "provider_execution_manifest_sha256",
    ):
        if not _digest(suite.get(field)):
            blockers.append(f"oscar_action_control_base_digest_missing_or_invalid:{field}")
    if suite.get("base_execution_fresh_official_oscar_model") is not True:
        blockers.append("oscar_action_control_base_execution_not_fresh_official_model")
    if suite.get("controls_are_excluded_from_decision_rows") is not True:
        blockers.append("oscar_action_controls_not_excluded_from_decision_rows")

    rows = _rows(suite.get("controls"))
    kinds = [str(row.get("control_kind") or "") for row in rows]
    if set(kinds) != REQUIRED_CONTROL_KINDS or len(kinds) != len(REQUIRED_CONTROL_KINDS):
        blockers.append("oscar_action_control_kind_coverage_invalid")
    executed_outputs: set[str] = set()
    base_action = str(suite.get("base_commanded_action_sha256") or "").lower()
    base_output = str(suite.get("base_model_output_sha256") or "").lower()
    for index, row in enumerate(rows):
        kind = str(row.get("control_kind") or "")
        if not _digest(row.get("control_action_sha256")):
            blockers.append(f"oscar_action_control_digest_invalid:{index}:control_action_sha256")
        elif str(row.get("control_action_sha256") or "").lower() == base_action:
            blockers.append(f"oscar_action_control_action_not_distinct_from_base:{index}")
        if row.get("transformation_or_replay_condition_verified") is not True:
            blockers.append(f"oscar_action_control_transformation_not_verified:{index}")
        if kind in EXECUTED_CONTROL_KINDS:
            for field in (
                "skeleton_conditioning_sha256",
                "model_output_sha256",
                "provider_execution_sha256",
                "next_policy_query_sha256",
            ):
                if not _digest(row.get(field)):
                    blockers.append(f"oscar_action_control_digest_invalid:{index}:{field}")
            output = str(row.get("model_output_sha256") or "").lower()
            if output == base_output or output in executed_outputs:
                blockers.append(f"oscar_action_control_model_output_reused:{index}")
            executed_outputs.add(output)
            if row.get("status") != "fresh_counterfactual_completed":
                blockers.append(f"oscar_action_control_fresh_execution_not_completed:{index}")
            if row.get("fresh_official_oscar_model_execution_proven") is not True:
                blockers.append(f"oscar_action_control_fresh_model_execution_not_proven:{index}")
            run_steps = row.get("fresh_oscar_provider_model_run_steps")
            if isinstance(run_steps, bool) or not isinstance(run_steps, int) or run_steps <= 0:
                blockers.append(f"oscar_action_control_provider_model_steps_invalid:{index}")
        elif kind in REJECTION_CONTROL_KINDS:
            if row.get("status") != "admission_rejected_before_ranking":
                blockers.append(f"oscar_action_replay_control_not_rejected:{index}")
            if not str(row.get("rejection_blocker") or "").strip():
                blockers.append(f"oscar_action_replay_control_rejection_reason_missing:{index}")
            if row.get("decision_grade_eligible") is not False:
                blockers.append(f"oscar_action_replay_control_not_decision_blocked:{index}")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "oscar_action_control_suite_validation.v1",
        "status": "passed" if not blockers else "blocked",
        "required_control_kinds": sorted(REQUIRED_CONTROL_KINDS),
        "executed_control_count": sum(kind in EXECUTED_CONTROL_KINDS for kind in kinds),
        "rejection_control_count": sum(kind in REJECTION_CONTROL_KINDS for kind in kinds),
        "decision_grade_eligible": not blockers,
        "blockers": blockers,
        "claim_boundary": {
            "controls_are_diagnostics_not_policy_ranking_rows": True,
            "control_pass_does_not_prove_real_world_task_success": True,
        },
    }
