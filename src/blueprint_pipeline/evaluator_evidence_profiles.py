"""Versioned evidence profiles for live evaluator-bounded policy rows.

These profiles describe evidence produced by Blueprint evaluation runs.  They
are deliberately separate from the frozen external-study profiles used for
correlation calibration.  Selecting an SC3- or OSCAR-inspired evaluator may
add protocol-specific requirements, but neither benchmark is the identity of
the compute provider or the product architecture.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


EVALUATOR_EVIDENCE_VALIDATION_SCHEMA_VERSION = "evaluator_evidence_validation.v1"
EVALUATOR_BACKEND_MANIFEST_SCHEMA_VERSION = "evaluator_backend_manifest.v1"

COMMON_DIGEST_FIELDS = (
    "site_task_condition_seed_manifest_sha256",
    "observation_sha256",
    "commanded_action_chunk_sha256",
    "policy_runtime_output_sha256",
    "initial_condition_sha256",
    "evaluator_profile_manifest_sha256",
    "evaluator_backend_manifest_sha256",
    "evaluator_request_sha256",
    "evaluator_checkpoint_sha256",
    "model_output_sha256",
    "provider_execution_sha256",
    "next_policy_query_sha256",
    "action_control_suite_sha256",
    "criterion_result_sha256",
    "authoritative_manifest_sha256",
)

EVALUATOR_EVIDENCE_PROFILES: dict[str, dict[str, Any]] = {
    "generic_evaluator_bounded_v1": {
        "family": "generic_evaluator_bounded",
        "required_digest_fields": (),
        "required_status_fields": {
            "generic_evaluator_contract_status": ("validated",),
        },
        "requires_sc3_multiview": False,
        "requires_oscar_action_skeleton_chain": False,
    },
    "oscar_roboarena_v2": {
        "family": "oscar_roboarena",
        "required_digest_fields": (
            "official_runtime_contract_sha256",
            "fk_result_sha256",
            "camera_projection_sha256",
            "skeleton_conditioning_sha256",
        ),
        "required_status_fields": {
            "official_runtime_contract_status": ("validated",),
            "fk_status": ("passed",),
            "camera_projection_status": ("passed",),
            "skeleton_validation_status": ("passed",),
        },
        "requires_sc3_multiview": False,
        "requires_oscar_action_skeleton_chain": True,
    },
    "sc3_eval_v3": {
        "family": "sc3_eval",
        "required_digest_fields": (
            "synchronized_multiview_manifest_sha256",
            "recovered_inverse_actions_sha256",
            "per_chunk_error_sha256",
            "inverse_calibration_set_sha256",
        ),
        "required_status_fields": {
            "strict_scorer_request_status": ("validated",),
            "multiview_consistency_status": ("passed",),
            "inverse_action_recovery_status": ("passed", "abstained"),
        },
        "requires_sc3_multiview": True,
        "requires_oscar_action_skeleton_chain": False,
    },
}

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _digest(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "").strip().lower()))


def _normalized_digest(value: Any) -> str:
    digest = str(value or "").strip().lower()
    return digest.removeprefix("sha256:") if _SHA256_RE.fullmatch(digest) else ""


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strict_rows(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    if any(not isinstance(item, Mapping) for item in value):
        return [], False
    return [dict(item) for item in value], True


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def required_evaluator_evidence_digest_fields(profile_id: str) -> tuple[str, ...]:
    """Return the common and profile-specific digest chain for ``profile_id``.

    Unknown profiles still return the common chain so downstream validators do
    not accidentally stop checking shared evidence while reporting the separate
    unsupported-profile blocker.
    """

    profile = EVALUATOR_EVIDENCE_PROFILES.get(str(profile_id or "").strip())
    if profile is None:
        return COMMON_DIGEST_FIELDS
    return (*COMMON_DIGEST_FIELDS, *profile["required_digest_fields"])


def validate_evaluator_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one row under its selected evaluator evidence profile."""

    blockers: list[str] = []
    profile_id = str(row.get("evaluator_profile_id") or "").strip()
    profile = EVALUATOR_EVIDENCE_PROFILES.get(profile_id)
    if profile is None:
        return {
            "schema_version": EVALUATOR_EVIDENCE_VALIDATION_SCHEMA_VERSION,
            "status": "blocked",
            "evaluator_profile_id": profile_id or None,
            "evaluator_family": None,
            "model_abstained": False,
            "blockers": ["evaluator_profile_missing_or_unsupported"],
        }

    backend = _mapping(row.get("evaluator_backend"))
    if backend.get("schema_version") != EVALUATOR_BACKEND_MANIFEST_SCHEMA_VERSION:
        blockers.append("evaluator_backend_schema_missing_or_unsupported")
    for field in ("backend_id", "model_family", "model_version", "adapter_version"):
        if not str(backend.get(field) or "").strip():
            blockers.append(f"evaluator_backend_identity_missing:{field}")
    if backend.get("backend_kind") not in {
        "world_model",
        "inverse_action_scorer",
        "generic_task_evaluator",
    }:
        blockers.append("evaluator_backend_kind_missing_or_invalid")
    if backend.get("execution_interface") not in {
        "local_container",
        "model_runtime_service",
        "provider_worker",
        "remote_api",
    }:
        blockers.append("evaluator_backend_execution_interface_missing_or_invalid")
    for field in (
        "model_artifact_sha256",
        "adapter_code_sha256",
        "runtime_manifest_sha256",
        "license_manifest_sha256",
    ):
        if not _digest(backend.get(field)):
            blockers.append(f"evaluator_backend_digest_missing_or_invalid:{field}")
    if _normalized_digest(backend.get("model_artifact_sha256")) != _normalized_digest(
        row.get("evaluator_checkpoint_sha256")
    ):
        blockers.append("evaluator_backend_model_digest_does_not_match_checkpoint")
    if backend.get("backend_is_compute_provider") is not False:
        blockers.append("evaluator_backend_must_not_be_compute_provider")

    for field in required_evaluator_evidence_digest_fields(profile_id):
        if not _digest(row.get(field)):
            blockers.append(f"evaluator_evidence_digest_missing_or_invalid:{field}")

    if row.get("fresh_evaluator_model_execution_proven") is not True:
        blockers.append("fresh_evaluator_model_execution_not_proven")
    run_steps = row.get("fresh_evaluator_model_run_steps")
    if isinstance(run_steps, bool) or not isinstance(run_steps, int) or run_steps <= 0:
        blockers.append("fresh_evaluator_model_run_steps_missing_or_invalid")
    if row.get("action_control_suite_status") != "passed":
        blockers.append("action_control_suite_not_passed")
    if row.get("authoritative_manifest_status") != "completed":
        blockers.append("authoritative_manifest_not_completed")
    if row.get("infrastructure_status") != "succeeded":
        blockers.append("evaluator_infrastructure_not_succeeded")
    if row.get("evaluator_identity_is_compute_provider") is not False:
        blockers.append("compute_provider_must_not_be_evaluator_identity")

    outcome_status = str(row.get("evaluator_outcome_status") or "")
    criterion_status = str(row.get("criterion_result_status") or "")
    if outcome_status not in {"valid", "abstained"}:
        blockers.append("evaluator_outcome_status_missing_or_invalid")
    if criterion_status not in {"valid", "abstained"}:
        blockers.append("criterion_result_status_missing_or_invalid")
    if outcome_status in {"valid", "abstained"} and criterion_status != outcome_status:
        blockers.append("criterion_result_status_does_not_match_evaluator_outcome")

    for field, allowed_values in profile["required_status_fields"].items():
        if row.get(field) not in allowed_values:
            blockers.append(f"evaluator_profile_status_invalid:{field}")

    if profile_id == "sc3_eval_v3":
        termination_chunk = row.get("termination_chunk_index")
        threshold = _finite(row.get("inverse_error_threshold"))
        recovered_dimensions, recovered_dimensions_payload_valid = _strict_rows(
            row.get("recovered_inverse_action_dimensions")
        )
        if not recovered_dimensions_payload_valid:
            blockers.append("sc3_recovered_inverse_action_dimensions_payload_invalid")
        if (
            isinstance(termination_chunk, bool)
            or not isinstance(termination_chunk, int)
            or termination_chunk < 0
        ):
            blockers.append("sc3_termination_chunk_missing_or_invalid")
        if threshold is None or threshold < 0:
            blockers.append("sc3_inverse_error_threshold_missing_or_invalid")
        if not recovered_dimensions:
            blockers.append("sc3_recovered_inverse_action_dimensions_missing")
        recovered_dimension_ids: set[int] = set()
        for dimension_index, dimension_row in enumerate(recovered_dimensions):
            dimension_id = dimension_row.get("dimension")
            maximum_error = _finite(dimension_row.get("maximum_error"))
            if (
                isinstance(dimension_id, bool)
                or not isinstance(dimension_id, int)
                or dimension_id < 0
                or dimension_id in recovered_dimension_ids
            ):
                blockers.append(f"sc3_recovered_inverse_action_dimension_invalid:{dimension_index}")
            else:
                recovered_dimension_ids.add(dimension_id)
            if not str(dimension_row.get("unit") or "").strip():
                blockers.append(f"sc3_recovered_inverse_action_unit_missing:{dimension_index}")
            if maximum_error is None or maximum_error < 0:
                blockers.append(f"sc3_recovered_inverse_action_error_invalid:{dimension_index}")
            elif threshold is not None and maximum_error > threshold:
                blockers.append(
                    f"sc3_recovered_inverse_action_error_exceeds_threshold:{dimension_index}"
                )
        inverse_status = row.get("inverse_action_recovery_status")
        if outcome_status == "valid" and inverse_status != "passed":
            blockers.append("sc3_valid_outcome_requires_passed_inverse_recovery")
        if outcome_status == "abstained" and inverse_status != "abstained":
            blockers.append("sc3_abstention_requires_inverse_recovery_abstention")

    blockers = sorted(set(blockers))
    return {
        "schema_version": EVALUATOR_EVIDENCE_VALIDATION_SCHEMA_VERSION,
        "status": "validated" if not blockers else "blocked",
        "evaluator_profile_id": profile_id,
        "evaluator_family": profile["family"],
        "evaluator_backend_id": str(backend.get("backend_id") or "").strip(),
        "evaluator_model_family": str(backend.get("model_family") or "").strip(),
        "evaluator_model_version": str(backend.get("model_version") or "").strip(),
        "model_abstained": outcome_status == "abstained",
        "blockers": blockers,
        "claim_boundary": {
            "compute_provider_is_not_evaluator_identity": True,
            "generated_episode_status_cannot_override_authoritative_manifest": True,
            "model_abstention_is_not_infrastructure_failure": True,
            "profile_validation_is_not_task_success_or_real_world_correlation": True,
        },
    }
