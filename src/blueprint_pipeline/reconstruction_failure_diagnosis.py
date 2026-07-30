"""Deterministic reconstruction failure diagnosis and bounded recovery policy."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_contracts import FAILURE_CODES, LEGAL_RECOVERY_ACTIONS


FAILURE_DIAGNOSIS_REQUEST_SCHEMA_VERSION = "reconstruction_failure_diagnosis_request.v1"
FAILURE_DIAGNOSIS_SCHEMA_VERSION = "reconstruction_failure_diagnosis.v1"


class ReconstructionFailureDiagnosisError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReconstructionFailureDiagnosisError(["failure_diagnosis_not_json"]) from exc


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def build_reconstruction_failure_diagnosis_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    request = _clone(dict(value))
    errors: list[str] = []
    if request.get("schema_version") != FAILURE_DIAGNOSIS_REQUEST_SCHEMA_VERSION:
        errors.append("failure_diagnosis_request_schema_invalid")
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "stage_id",
        "timestamp",
    ):
        if not str(request.get(key) or "").strip():
            errors.append(f"failure_diagnosis_request_{key}_missing")
    for key in ("source_capture_digest", "failed_event_digest"):
        if not _is_digest(request.get(key)):
            errors.append(f"failure_diagnosis_request_{key}_invalid")
    if request.get("failure_code") not in FAILURE_CODES:
        errors.append("failure_diagnosis_request_failure_code_invalid")
    authority_state = request.get("authority_state")
    if not isinstance(authority_state, Mapping) or any(
        not isinstance(authority_state.get(key), bool)
        for key in ("paid_execution_authorized", "provider_execution_authorized")
    ):
        errors.append("failure_diagnosis_request_authority_state_invalid")
    for key in ("execution_requires_paid_compute", "execution_requires_provider"):
        if not isinstance(request.get(key), bool):
            errors.append(f"failure_diagnosis_request_{key}_invalid")
    attempts = request.get("attempt_ledger")
    if not isinstance(attempts, list) or not attempts:
        errors.append("failure_diagnosis_request_attempt_ledger_missing")
    else:
        seen: set[str] = set()
        for index, raw_attempt in enumerate(attempts):
            if not isinstance(raw_attempt, Mapping):
                errors.append(f"failure_diagnosis_request_attempt_invalid:{index}")
                continue
            attempt = dict(raw_attempt)
            attempt_id = str(attempt.get("attempt_id") or "").strip()
            if not attempt_id or attempt_id in seen:
                errors.append(f"failure_diagnosis_request_attempt_id_invalid:{index}")
            seen.add(attempt_id)
            if attempt.get("failure_code") not in FAILURE_CODES:
                errors.append(f"failure_diagnosis_request_attempt_failure_invalid:{attempt_id}")
            for key in ("input_digest", "configuration_digest", "event_digest"):
                if not _is_digest(attempt.get(key)):
                    errors.append(
                        f"failure_diagnosis_request_attempt_digest_invalid:{attempt_id}:{key}"
                    )
            if attempt.get("failed_evidence_preserved") is not True:
                errors.append(f"failure_diagnosis_request_evidence_not_preserved:{attempt_id}")
        if attempts and isinstance(attempts[-1], Mapping) and attempts[-1].get(
            "failure_code"
        ) != request.get("failure_code"):
            errors.append("failure_diagnosis_request_latest_failure_mismatch")
    supplied_digest = request.pop("reconstruction_failure_diagnosis_request_digest", None)
    request["reconstruction_failure_diagnosis_request_digest"] = canonical_digest(
        request, digest_field="reconstruction_failure_diagnosis_request_digest"
    )
    if supplied_digest is not None and supplied_digest != request[
        "reconstruction_failure_diagnosis_request_digest"
    ]:
        errors.append("failure_diagnosis_request_digest_mismatch")
    if errors:
        raise ReconstructionFailureDiagnosisError(errors)
    return request


_RECOVERY_BY_FAILURE: dict[str, tuple[str, ...]] = {
    "invalid_capture_contract": ("request_targeted_recapture", "preserve_evidence_and_stop"),
    "missing_rights_or_consent": ("request_additional_authority", "preserve_evidence_and_stop"),
    "missing_retained_media": ("request_targeted_recapture", "preserve_evidence_and_stop"),
    "invalid_pts_mapping": ("request_targeted_recapture", "preserve_evidence_and_stop"),
    "insufficient_coverage": ("request_targeted_recapture", "abstain"),
    "excessive_blur": ("request_targeted_recapture", "abstain"),
    "unsupported_camera_mode": ("choose_prequalified_reconstruction_method", "abstain"),
    "corrupt_insv": ("request_targeted_recapture", "preserve_evidence_and_stop"),
    "unsynchronized_lens_streams": ("request_targeted_recapture", "abstain"),
    "missing_rig_calibration": ("request_targeted_recapture", "abstain"),
    "pose_estimation_failure": ("choose_prequalified_matching_method", "abstain"),
    "weak_registration": ("choose_prequalified_matching_method", "request_targeted_recapture"),
    "loop_closure_failure": ("request_targeted_recapture", "abstain"),
    "ambiguous_metric_scale": ("request_metric_anchor", "abstain"),
    "scale_anchor_rejection": ("request_metric_anchor", "abstain"),
    "invalid_depth_alignment": ("request_targeted_recapture", "abstain"),
    "training_divergence": ("choose_prequalified_reconstruction_method", "abstain"),
    "nan_output": ("retry_once_same_worker", "choose_prequalified_reconstruction_method"),
    "gpu_out_of_memory": ("resume_bound_checkpoint", "choose_prequalified_reconstruction_method"),
    "provider_capacity": ("retry_once_same_worker", "use_already_authorized_provider"),
    "provider_admission_failure": ("request_additional_authority", "preserve_evidence_and_stop"),
    "worker_startup_failure": ("retry_once_same_worker", "preserve_evidence_and_stop"),
    "checkpoint_acquisition_failure": ("resume_bound_checkpoint", "preserve_evidence_and_stop"),
    "malformed_output": ("retry_once_same_worker", "preserve_evidence_and_stop"),
    "invalid_artifact_digest": ("preserve_evidence_and_stop",),
    "heldout_evaluation_failure": ("preserve_evidence_and_stop", "abstain"),
    "collider_qualification_failure": ("request_targeted_recapture", "abstain"),
    "isaac_load_failure": ("choose_prequalified_reconstruction_method", "abstain"),
    "blank_render": ("retry_once_same_worker", "preserve_evidence_and_stop"),
    "missing_collision_properties": ("preserve_evidence_and_stop", "abstain"),
    "budget_exhaustion": ("request_additional_authority", "abstain"),
    "ttl_expiration": ("request_additional_authority", "abstain"),
    "repeated_identical_blocker": ("preserve_evidence_and_stop", "abstain"),
    "teardown_verification_failure": ("preserve_evidence_and_stop",),
    "provider_interruption": ("resume_bound_checkpoint", "retry_once_same_worker"),
    "permanent_incompatibility": ("preserve_evidence_and_stop", "abstain"),
}


def diagnose_reconstruction_failure(value: Mapping[str, Any]) -> dict[str, Any]:
    request = build_reconstruction_failure_diagnosis_request(value)
    attempts = request["attempt_ledger"]
    latest = attempts[-1]
    fingerprint = canonical_digest(
        {
            "failure_code": latest["failure_code"],
            "input_digest": latest["input_digest"],
            "configuration_digest": latest["configuration_digest"],
        }
    )
    identical_count = sum(
        1
        for attempt in attempts
        if canonical_digest(
            {
                "failure_code": attempt["failure_code"],
                "input_digest": attempt["input_digest"],
                "configuration_digest": attempt["configuration_digest"],
            }
        )
        == fingerprint
    )
    diagnosed_code = (
        "repeated_identical_blocker" if identical_count > 1 else request["failure_code"]
    )
    legal_actions = list(_RECOVERY_BY_FAILURE[diagnosed_code])
    unchanged_retry_allowed = identical_count == 1 and "retry_once_same_worker" in legal_actions
    if identical_count > 1:
        legal_actions = ["preserve_evidence_and_stop", "abstain"]
    authority_state = request["authority_state"]
    missing_execution_authority = bool(
        (
            request["execution_requires_paid_compute"]
            and authority_state["paid_execution_authorized"] is False
        )
        or (
            request["execution_requires_provider"]
            and authority_state["provider_execution_authorized"] is False
        )
    )
    if missing_execution_authority and not (identical_count > 1):
        execution_actions = {
            "retry_once_same_worker",
            "resume_bound_checkpoint",
            "use_already_authorized_provider",
        }
        legal_actions = [
            "request_additional_authority",
            *[action for action in legal_actions if action not in execution_actions],
        ]
        legal_actions = list(dict.fromkeys(legal_actions))
        unchanged_retry_allowed = False
    report = {
        "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        "stable_run_identity": request["stable_run_identity"],
        "source_capture_identity": request["source_capture_identity"],
        "source_capture_digest": request["source_capture_digest"],
        "failure_diagnosis_request_digest": request[
            "reconstruction_failure_diagnosis_request_digest"
        ],
        "failed_event_digest": request["failed_event_digest"],
        "stage_id": request["stage_id"],
        "reported_failure_code": request["failure_code"],
        "diagnosed_failure_code": diagnosed_code,
        "blocker_fingerprint": fingerprint,
        "identical_attempt_count": identical_count,
        "attempt_ids": [attempt["attempt_id"] for attempt in attempts],
        "failed_evidence_preserved": all(
            attempt["failed_evidence_preserved"] is True for attempt in attempts
        ),
        "unchanged_deterministic_retry_allowed": unchanged_retry_allowed,
        "terminal_for_current_configuration": identical_count > 1
        or diagnosed_code
        in {
            "invalid_artifact_digest",
            "teardown_verification_failure",
            "permanent_incompatibility",
        },
        "legal_next_actions": legal_actions,
        "authority_requested_not_granted": "request_additional_authority" in legal_actions,
        "recovery_executed": False,
        "agent_changed_failure_code": False,
        "agent_granted_authority": False,
        "proof_effect": "none",
        "claim_ceiling": "failure_diagnosis_only",
        "parent_artifact_or_event": {"failed_event_digest": request["failed_event_digest"]},
        "timestamp": request["timestamp"],
    }
    if any(action not in LEGAL_RECOVERY_ACTIONS for action in legal_actions):
        raise ReconstructionFailureDiagnosisError(["failure_diagnosis_action_invalid"])
    report["reconstruction_failure_diagnosis_digest"] = canonical_digest(
        report, digest_field="reconstruction_failure_diagnosis_digest"
    )
    return report


__all__ = [
    "FAILURE_DIAGNOSIS_REQUEST_SCHEMA_VERSION",
    "FAILURE_DIAGNOSIS_SCHEMA_VERSION",
    "ReconstructionFailureDiagnosisError",
    "build_reconstruction_failure_diagnosis_request",
    "diagnose_reconstruction_failure",
]
