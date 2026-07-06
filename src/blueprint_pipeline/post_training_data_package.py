"""Post-Training Data Package export, checksum, and archive builder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import tarfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .buyer_package_readout import (
    build_buyer_package_readout,
    render_buyer_package_readout_markdown,
)
from .lerobot_export_validation import (
    round_trip_validation_summary,
    validate_lerobot_export,
)
from .rl_post_training_handoff import build_rl_post_training_handoff_packet


POST_TRAINING_DATA_PACKAGE_EXPORT_SCHEMA_VERSION = "post_training_data_package_export.v1"
CUSTOMER_HANDOFF_REPORT_SCHEMA_VERSION = "post_training_customer_handoff_report.v1"
DELIVERY_MANIFEST_SCHEMA_VERSION = "post_training_delivery_manifest.v1"
SIGNED_ACCESS_MANIFEST_SCHEMA_VERSION = "post_training_signed_access_manifest.v1"
HANDOFF_SUMMARY_SCHEMA_VERSION = "post_training_data_package_handoff_summary.v1"
CURATION_REPORT_SCHEMA_VERSION = "post_training_curation_report.v1"
SEMANTIC_DEDUP_REPORT_SCHEMA_VERSION = "post_training_semantic_dedup_report.v1"
SC3_ACTION_REPORT_SCHEMA_VERSION = "post_training_sc3_action_normalization_report.v1"
REVOCATION_TAKEDOWN_MANIFEST_SCHEMA_VERSION = "post_training_revocation_takedown_manifest.v1"
DOWNSTREAM_TAKEDOWN_EXECUTION_LEDGER_SCHEMA_VERSION = (
    "post_training_downstream_takedown_execution_ledger.v1"
)
WEBAPP_RIGHTS_PRIVACY_TAKEDOWN_NOTICE_SCHEMA_VERSION = (
    "post_training_webapp_rights_privacy_takedown_notice.v1"
)
HOSTED_SESSION_TAKEDOWN_REQUEST_SCHEMA_VERSION = (
    "post_training_hosted_session_takedown_request.v1"
)
# Attempt-trace producers that never claimed to capture SC3 7D action vectors
# (e.g. isaac_lab_arena result ingestion) legitimately have no action data at
# all; that absence is surfaced in sc3_action_report / sc3_action_contract_status
# but must not hard-block curation or export. Malformed action data (present but
# invalid shape/values) still blocks.
SC3_NO_ACTION_DATA_BLOCKERS = frozenset(
    {"sc3_attempt_trace_missing", "sc3_action_trace_missing"}
)
OSCAR_MIN_FRAME_COUNT = 16
OSCAR_MAX_STATIC_CAMERA_MOTION_M = 0.05
OSCAR_MIN_ACTION_MOTION_SCORE = 1e-4
OSCAR_MIN_VISIBLE_SKELETON_FRACTION = 0.5
OSCAR_MIN_SHARPNESS_SCORE = 5.0
LEROBOT_V3_EXPORT_FPS = 5
# Honesty floor for training exports: below this fraction of measured (non
# zero-fill-synthesized) observation.state rows the lerobot-format exports are
# downgraded with insufficient_measured_state_fraction and the buyer readout's
# robot-POV-evidence section blocks. Zero-filled state is a format placeholder,
# never robot-state evidence; a buyer must see how much of the package is real.
MEASURED_STATE_FRACTION_FLOOR_DEFAULT = 0.5
MEASURED_STATE_FRACTION_FLOOR_ENV = "BLUEPRINT_PTDP_MEASURED_STATE_FRACTION_FLOOR"


def _measured_state_fraction_floor() -> float:
    raw = str(os.environ.get(MEASURED_STATE_FRACTION_FLOOR_ENV) or "").strip()
    if raw:
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            pass
    return MEASURED_STATE_FRACTION_FLOOR_DEFAULT

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "post_training_data_package_export",
    "export_manifest_only": False,
    "export_files_written": True,
    "review_acceptance_proven": False,
    "rights_privacy_scope_proven": False,
    "signed_delivery_access_proven": False,
    "customer_handoff_ready": False,
    "delivery_access_is_deployment_approval": False,
    "package_delivery_is_deployment_approval": False,
    "deployment_approval_proven": False,
    "physical_robot_readiness_proven": False,
    "safety_validation_proven": False,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "training_completed": False,
    "public_claim_upgrade_allowed": False,
}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Sequence[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = [value]
    out: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None:
        return None
    return int(number)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _explicit_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "revoked", "withdrawn", "rescinded"}:
            return True
        if normalized in {"0", "false", "no", "n", "active", "documented"}:
            return False
    return None


def _explicit_true(*values: Any) -> bool:
    return any(_explicit_bool(value) is True for value in values)


def _revocation_takedown_required(payload: Mapping[str, Any]) -> bool:
    revocation_takedown = _mapping(payload.get("revocation_takedown"))
    return bool(
        _explicit_true(payload.get("consent_revoked"))
        or payload.get("status") == "blocked_consent_revoked_takedown_required"
        or revocation_takedown.get("status") == "takedown_required"
    )


def _clip_rows(clips: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(clips.get("clips") or []):
        if isinstance(item, Mapping):
            row = dict(item)
        elif isinstance(item, str):
            row = {"clip_id": item, "clip_path": item}
        else:
            continue
        row.setdefault("clip_id", row.get("id") or row.get("path") or row.get("clip_path") or f"clip_{index}")
        rows.append(row)
    return rows


def _clip_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("clip_id") or row.get("id") or row.get("clip_path") or row.get("path") or f"clip_{index}").strip()


def _safe_path_component(value: Any, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text)
    safe = safe.strip("._")
    return safe[:96] or fallback


def _attempt_actions(attempt: Mapping[str, Any]) -> List[Any]:
    actions = attempt.get("actions")
    if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes, bytearray)):
        return list(actions)
    action_trace = attempt.get("action_trace")
    if isinstance(action_trace, Sequence) and not isinstance(action_trace, (str, bytes, bytearray)):
        return list(action_trace)
    return []


def _action_vector_from_mapping(action: Mapping[str, Any]) -> List[float] | None:
    candidates = [
        action.get("sc3_7d_delta_ee_pose"),
        action.get("sc3_action_vector"),
        action.get("action_vector_7d"),
        action.get("delta_end_effector_pose_7d"),
        action.get("delta_ee_pose_7d"),
    ]
    normalized = _mapping(action.get("normalized_action"))
    if normalized:
        candidates.extend(
            [
                normalized.get("sc3_7d_delta_ee_pose"),
                normalized.get("sc3_action_vector"),
                normalized.get("action_vector_7d"),
                normalized.get("delta_end_effector_pose_7d"),
                normalized.get("delta_ee_pose_7d"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)) and len(candidate) == 7:
            vector = [_finite_float(value) for value in candidate]
            if all(value is not None for value in vector):
                return [float(value) for value in vector if value is not None]

    delta_position = action.get("delta_position_m") or action.get("translation_delta_m")
    delta_rotation = action.get("delta_rotation_axis_angle") or action.get("rotation_delta_axis_angle")
    gripper = (
        action.get("gripper_delta")
        if action.get("gripper_delta") is not None
        else action.get("gripper")
    )
    if (
        isinstance(delta_position, Sequence)
        and not isinstance(delta_position, (str, bytes, bytearray))
        and len(delta_position) == 3
        and isinstance(delta_rotation, Sequence)
        and not isinstance(delta_rotation, (str, bytes, bytearray))
        and len(delta_rotation) == 3
    ):
        values = [
            *[_finite_float(value) for value in delta_position],
            *[_finite_float(value) for value in delta_rotation],
            _finite_float(gripper if gripper is not None else 0.0),
        ]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    return None


def _sc3_action_vector(action: Any) -> List[float] | None:
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes, bytearray)) and len(action) == 7:
        values = [_finite_float(value) for value in action]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    if isinstance(action, Mapping):
        return _action_vector_from_mapping(action)
    return None


def _sc3_action_vectors_for_attempt(attempt: Mapping[str, Any]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for action in _attempt_actions(attempt):
        vector = _sc3_action_vector(action)
        if vector is not None:
            vectors.append(vector)
    return vectors


def _build_sc3_action_report(
    *,
    trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    attempts = _rows(trace, "attempts")
    rows: List[Dict[str, Any]] = []
    blockers: List[str] = []
    action_count = 0
    valid_count = 0
    for attempt_index, attempt in enumerate(attempts):
        attempt_id = str(attempt.get("attempt_id") or f"attempt_{attempt_index}").strip()
        actions = _attempt_actions(attempt)
        vectors = _sc3_action_vectors_for_attempt(attempt)
        action_count += len(actions)
        valid_count += len(vectors)
        attempt_blockers: List[str] = []
        if not actions:
            attempt_blockers.append("sc3_action_trace_missing")
        if len(vectors) != len(actions):
            attempt_blockers.append("sc3_7d_delta_end_effector_pose_missing_or_invalid")
        if vectors and not any(any(abs(value) > OSCAR_MIN_ACTION_MOTION_SCORE for value in vector) for vector in vectors):
            attempt_blockers.append("sc3_action_vectors_all_zero")
        blockers.extend(f"{attempt_id}:{blocker}" for blocker in attempt_blockers)
        rows.append(
            {
                "attempt_id": attempt_id,
                "action_count": len(actions),
                "valid_sc3_7d_action_count": len(vectors),
                "vectors": vectors,
                "status": "passed" if not attempt_blockers else "blocked",
                "blockers": attempt_blockers,
            }
        )
    if not attempts:
        blockers.append("sc3_attempt_trace_missing")
    if action_count == 0:
        blockers.append("sc3_action_trace_missing")
    status = "passed" if not blockers else "blocked"
    return {
        "schema_version": SC3_ACTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "action_representation": "7d_delta_end_effector_pose",
        "attempt_count": len(attempts),
        "action_count": action_count,
        "valid_sc3_7d_action_count": valid_count,
        "rows": rows,
        "blockers": _string_list(blockers),
        "claim_boundary": {
            "action_normalization_validated": status == "passed",
            "missing_actions_exported_as_identity_pose": False,
            "sc3_action_contract_satisfied": status == "passed",
        },
    }


def _attempts_by_id(trace: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for index, attempt in enumerate(_rows(trace, "attempts")):
        attempt_id = str(attempt.get("attempt_id") or f"attempt_{index}").strip()
        if attempt_id:
            out[attempt_id] = attempt
        scenario_id = str(attempt.get("scenario_id") or "").strip()
        if scenario_id:
            out.setdefault(scenario_id, attempt)
    return out


def _evidence_gate(
    *,
    explicit: Any,
    measured: float | None,
    threshold: float,
    op: str,
    missing_blocker: str,
) -> tuple[bool, List[str], Dict[str, Any]]:
    evidence: Dict[str, Any] = {"threshold": threshold, "operator": op}
    if isinstance(explicit, bool):
        evidence["explicit"] = explicit
    if measured is None:
        evidence["value"] = None
        evidence["explicit_boolean_is_not_measured_evidence"] = isinstance(explicit, bool)
        blockers = [missing_blocker]
        if explicit is True:
            blockers.append(f"{missing_blocker}:explicit_true_without_measured_evidence")
        elif explicit is False:
            blockers.append(missing_blocker.replace("_missing", "_failed"))
        return False, blockers, evidence
    passed = measured >= threshold if op == ">=" else measured <= threshold
    blockers = [] if passed else [missing_blocker.replace("_missing", "_failed")]
    if explicit is False:
        passed = False
        blockers.append("explicit_source_filter_failed")
    evidence["value"] = measured
    return passed, blockers, evidence


def _first_string(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _success_claim_ledger_from_sources(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    for source in sources:
        if source.get("schema_version") == "success_claim_ledger.v1":
            return dict(source)
        direct = _mapping(source.get("success_claim_ledger"))
        if direct:
            return direct
        report = _mapping(source.get("task_eval_run_report"))
        nested = _mapping(report.get("success_claim_ledger"))
        if nested:
            return nested
    return {}


def _extract_product_handoff(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    for source in sources:
        handoff = _mapping(source.get("product_handoff"))
        if handoff:
            return handoff
    direct_keys = ("product_type", "product_sku", "entitlement_id", "buyer_review_url")
    for source in sources:
        handoff = {key: source.get(key) for key in direct_keys if source.get(key)}
        if handoff:
            return handoff
    return {}


def _consent_source_payload(capture_root: Path) -> Dict[str, Any]:
    for relative in (
        "raw/rights_consent.json",
        "rights_consent.json",
        "raw/manifest.json",
        "capture_descriptor.json",
    ):
        payload = _read_optional_mapping(capture_root / relative)
        if payload:
            nested = _mapping(
                payload.get("capture_rights")
                or payload.get("rights_consent")
                or payload.get("rights")
            )
            return nested or payload
    return {}


def _rights_commercialization_terms(*payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for payload in payloads:
        data = _mapping(
            payload.get("commercialization_terms")
            or payload.get("commercializationTerms")
            or payload.get("commercial_terms")
            or payload.get("commercialTerms")
        )
        if data:
            return data
    return {}


def _rights_operator_revenue_terms(
    *,
    commercialization_terms: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    for payload in payloads:
        data = _mapping(
            payload.get("operator_revenue_terms")
            or payload.get("operatorRevenueTerms")
            or payload.get("revenue_share_terms")
            or payload.get("revenueShareTerms")
        )
        if data:
            return data
    return _mapping(
        commercialization_terms.get("operator_revenue_terms")
        or commercialization_terms.get("operatorRevenueTerms")
        or commercialization_terms.get("revenue_share_terms")
        or commercialization_terms.get("revenueShareTerms")
        or commercialization_terms.get("revenue_share")
        or commercialization_terms.get("revenueShare")
    )


def _rights_exclusivity_terms(
    *,
    commercialization_terms: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    for payload in payloads:
        data = _mapping(
            payload.get("exclusivity_terms")
            or payload.get("exclusivityTerms")
            or payload.get("exclusivity")
        )
        if data:
            return data
    return _mapping(
        commercialization_terms.get("exclusivity_terms")
        or commercialization_terms.get("exclusivityTerms")
        or commercialization_terms.get("exclusivity")
    )


def _data_processing_terms_review(
    *,
    generated_at: str,
    rights_packet: Mapping[str, Any],
    consent_source: Mapping[str, Any],
) -> Dict[str, Any]:
    packet_terms = _mapping(
        rights_packet.get("data_processing_terms")
        or rights_packet.get("dataProcessingTerms")
        or rights_packet.get("dpa_terms")
        or rights_packet.get("dpaTerms")
    )
    source_terms = _mapping(
        consent_source.get("data_processing_terms")
        or consent_source.get("dataProcessingTerms")
        or consent_source.get("dpa_terms")
        or consent_source.get("dpaTerms")
    )
    terms = {**source_terms, **packet_terms}
    retention_policy = _mapping(
        terms.get("retention_policy")
        or terms.get("retentionPolicy")
        or terms.get("data_retention")
        or terms.get("dataRetention")
    )
    subprocessors = terms.get("subprocessors") or terms.get("subprocessor_list") or []
    if isinstance(subprocessors, Mapping):
        subprocessors = [dict(subprocessors)]
    elif isinstance(subprocessors, Sequence) and not isinstance(subprocessors, (str, bytes)):
        subprocessors = [
            dict(item) if isinstance(item, Mapping) else {"name": str(item)}
            for item in subprocessors
        ]
    else:
        subprocessors = []
    access_audit = _mapping(
        terms.get("access_audit")
        or terms.get("accessAudit")
        or terms.get("access_audit_terms")
        or terms.get("accessAuditTerms")
    )
    blockers: List[str] = []
    if not retention_policy:
        blockers.append("retention_policy_missing")
    if not subprocessors:
        blockers.append("subprocessor_list_missing")
    if not access_audit:
        blockers.append("access_audit_terms_missing")
    return {
        "schema_version": "post_training_data_processing_terms_review.v1",
        "generated_at": generated_at,
        "status": "recorded_review_required" if not blockers else "review_required",
        "required_before_external_delivery_or_paid_reuse": True,
        "retention_policy_present": bool(retention_policy),
        "subprocessor_list_present": bool(subprocessors),
        "access_audit_terms_present": bool(access_audit),
        "retention_policy": retention_policy,
        "subprocessors": subprocessors,
        "access_audit_terms": access_audit,
        "external_delivery_claim_allowed": False,
        "dpa_approval_claimed": False,
        "blockers": blockers,
        "source": "robot_eval_dataset_rights_packet_or_consent_source"
        if terms
        else "missing",
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "data_processing_terms_are_review_metadata_not_legal_approval": True,
            "external_delivery_requires_operator_dpa_or_equivalent_terms": True,
            "dpa_approval_proven": False,
        },
    }


def _revenue_review_from_rights(
    *,
    generated_at: str,
    rights_packet: Mapping[str, Any],
    consent_source: Mapping[str, Any],
) -> Dict[str, Any]:
    packet_review = _mapping(rights_packet.get("revenue_share_review"))
    records = [
        _mapping(record)
        for record in rights_packet.get("records") or []
        if isinstance(record, Mapping)
    ]
    record_by_scope = {
        str(record.get("rights_scope") or "").strip(): record for record in records
    }
    commercial_record = _mapping(record_by_scope.get("commercial_licensing"))
    revenue_record = _mapping(record_by_scope.get("revenue_share"))
    exclusivity_record = _mapping(record_by_scope.get("exclusivity_limits"))
    commercialization_terms = _rights_commercialization_terms(
        packet_review,
        rights_packet,
        commercial_record,
        consent_source,
    )
    operator_revenue_terms = _rights_operator_revenue_terms(
        commercialization_terms=commercialization_terms,
        payloads=[
            packet_review,
            rights_packet,
            revenue_record,
            consent_source,
        ],
    )
    exclusivity_terms = _rights_exclusivity_terms(
        commercialization_terms=commercialization_terms,
        payloads=[
            packet_review,
            rights_packet,
            exclusivity_record,
            consent_source,
        ],
    )
    owner_record_present = bool(
        packet_review.get("owner_revenue_share_record_present") is True
        or operator_revenue_terms
        or revenue_record.get("terms_record_present") is True
    )
    blockers = [] if owner_record_present else ["owner_revenue_share_record_missing"]
    return {
        "schema_version": "post_training_revenue_share_review.v1",
        "source_schema_version": packet_review.get("schema_version"),
        "generated_at": generated_at,
        "status": "recorded_review_required"
        if owner_record_present
        else "review_required",
        "upstream_status": packet_review.get("status"),
        "required_before_paid_reuse_or_resale": True,
        "owner_revenue_share_record_present": owner_record_present,
        "operator_revenue_terms": operator_revenue_terms,
        "commercialization_terms": commercialization_terms,
        "exclusivity_terms": exclusivity_terms,
        "commercial_use_claim_allowed": False,
        "external_licensing_claim_allowed": False,
        "revenue_share_commitment_made": False,
        "payout_commitment_allowed": False,
        "blockers": blockers,
        "source": "robot_eval_dataset_rights_packet"
        if packet_review or operator_revenue_terms or commercialization_terms
        else "missing",
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "package_delivery_is_not_revenue_share_commitment": True,
            "paid_reuse_requires_separate_owner_record": True,
            "operator_revenue_terms_are_review_metadata_not_payment_or_resale_clearance": True,
        },
    }


def _build_consent_evidence_record(
    *,
    capture_root: Path,
    output_dir: Path,
    rights_packet: Mapping[str, Any],
    scene_id: str,
    capture_id: str,
    generated_at: str,
) -> Dict[str, Any]:
    source = _consent_source_payload(capture_root)
    records = [
        dict(record)
        for record in rights_packet.get("records") or []
        if isinstance(record, Mapping)
    ]
    evidence_uris = _string_list(
        [
            *[
                record.get("evidence_uri")
                or record.get("permission_document_uri")
                or record.get("source_uri")
                for record in records
            ],
            source.get("permission_document_uri"),
            source.get("permissionDocumentUri"),
            source.get("evidence_uri"),
        ]
    )
    consent_status = _first_string(
        source.get("consent_status"),
        source.get("consentStatus"),
        rights_packet.get("consent_status"),
    )
    consent_status_normalized = consent_status.lower() if consent_status else ""
    consent_revoked = (
        consent_status_normalized in {"revoked", "withdrawn", "rescinded"}
        or _explicit_true(source.get("consent_revoked"))
        or _explicit_true(source.get("consentRevoked"))
        or bool(source.get("consent_revoked_at") or source.get("consentRevokedAt"))
        or _explicit_true(rights_packet.get("consent_revoked"))
    )
    consent_revoked_at = _first_string(
        source.get("consent_revoked_at"),
        source.get("consentRevokedAt"),
        rights_packet.get("consent_revoked_at"),
    )
    consent_scope = _string_list(
        source.get("consent_scope") or source.get("consentScope")
    )
    if not consent_scope:
        for record in records:
            consent_scope.extend(_string_list(record.get("rights_scope")))
    consent_scope = _string_list(consent_scope)

    blockers: List[str] = []
    if not consent_status:
        blockers.append("consent_status_missing")
    if consent_status_normalized == "documented" and not evidence_uris:
        blockers.append("permission_document_uri_missing_for_documented_consent")
    if not consent_scope:
        blockers.append("consent_scope_missing")
    if consent_revoked:
        blockers.append("consent_revoked_takedown_required")
    consent_evidence_present = bool(consent_status and consent_scope and not blockers)
    status = "consent_evidence_present" if consent_evidence_present else "blocked_missing_consent_evidence"
    if consent_revoked:
        status = "blocked_consent_revoked_takedown_required"
    revenue_share_review = _revenue_review_from_rights(
        generated_at=generated_at,
        rights_packet=rights_packet,
        consent_source=source,
    )
    data_processing_terms = _data_processing_terms_review(
        generated_at=generated_at,
        rights_packet=rights_packet,
        consent_source=source,
    )
    required_takedown_actions = [
        "block_new_package_exports",
        "disable_signed_delivery_access",
        "disable_signed_access_manifest",
        "mark_delivery_manifest_revoked",
        "remove_hosted_review_assets",
        "remove_or_expire_hosted_sessions",
        "mark_webapp_rights_privacy_blocking",
        "notify_webapp_rights_privacy_blocking",
        "stop_downstream_training_or_finetuning_use",
        "prevent_post_training_export_reuse",
        "notify_buyer_and_owner",
        "queue_customer_notice_if_required",
    ]
    downstream_takedown_artifacts = (
        {
            "webapp_rights_privacy_takedown_notice": (
                "webapp_rights_privacy_takedown_notice.json"
            ),
            "hosted_session_takedown_request": "hosted_session_takedown_request.json",
            "downstream_takedown_execution_ledger": (
                "downstream_takedown_execution_ledger.json"
            ),
        }
        if consent_revoked
        else {}
    )
    downstream_takedown_execution_ledger = {
        "schema_version": DOWNSTREAM_TAKEDOWN_EXECUTION_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "status": "queued_unexecuted_downstream_takedown"
        if consent_revoked
        else "not_required",
        "consent_revoked": consent_revoked,
        "consent_revoked_at": consent_revoked_at or None,
        "local_package_access_revoked": consent_revoked,
        "delivery_blocked_by_consent_revocation": consent_revoked,
        "signed_access_revoked_by_consent": consent_revoked,
        "webapp_takedown_executed": False,
        "hosted_session_takedown_executed": False,
        "webapp_or_hosted_takedown_execution_proven": False,
        "external_takedown_executor_present": False,
        "surfaces": [
            {
                "surface": "post_training_data_package",
                "status": "blocked_locally" if consent_revoked else "not_required",
                "execution_proven": consent_revoked,
                "artifact_path": "post_training_data_package_export_manifest.json",
                "required_action": "block_new_package_exports",
            },
            {
                "surface": "signed_delivery_access",
                "status": "revoked_locally" if consent_revoked else "not_required",
                "execution_proven": consent_revoked,
                "artifact_path": "signed_access_manifest.json",
                "required_action": "disable_signed_delivery_access",
            },
            {
                "surface": "webapp_projection",
                "status": "queued_unexecuted"
                if consent_revoked
                else "not_required",
                "execution_proven": False,
                "artifact_path": "webapp_rights_privacy_takedown_notice.json"
                if consent_revoked
                else None,
                "required_action": "notify_webapp_rights_privacy_blocking",
            },
            {
                "surface": "hosted_sessions",
                "status": "queued_unexecuted"
                if consent_revoked
                else "not_required",
                "execution_proven": False,
                "artifact_path": "hosted_session_takedown_request.json"
                if consent_revoked
                else None,
                "required_action": "remove_or_expire_hosted_sessions",
            },
        ],
        "blockers": [
            "webapp_takedown_execution_not_proven",
            "hosted_session_takedown_execution_not_proven",
            "external_takedown_executor_missing",
        ]
        if consent_revoked
        else [],
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "ledger_is_local_execution_state_not_downstream_execution": True,
            "local_package_delivery_blocked": consent_revoked,
            "webapp_or_hosted_takedown_execution_proven": False,
            "deployment_approval_proven": False,
        },
    }
    revocation_takedown = {
        "schema_version": REVOCATION_TAKEDOWN_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "status": "takedown_required" if consent_revoked else "not_required",
        "consent_revoked": consent_revoked,
        "consent_revoked_at": consent_revoked_at or None,
        "local_package_access_revoked": consent_revoked,
        "delivery_blocked": consent_revoked,
        "signed_access_revoked": consent_revoked,
        "downstream_takedown_required": consent_revoked,
        "webapp_takedown_executed": False,
        "hosted_session_takedown_executed": False,
        "affected_surfaces": [
            "post_training_data_package",
            "optional_training_exports",
            "hosted_review_assets",
            "signed_delivery_access",
            "webapp_projection",
            "buyer_package_readout",
        ],
        "affected_artifacts": [
            "post_training_data_package_export_manifest.json",
            "consent_evidence.json",
            "revocation_takedown_manifest.json",
            "customer_handoff_report.json",
            "delivery_manifest.json",
            "signed_access_manifest.json",
            "package_index.json",
            "archive_manifest.json",
            "license_manifest.json",
            "optional_export_manifest.json",
            *downstream_takedown_artifacts.values(),
        ],
        "downstream_takedown_artifacts": downstream_takedown_artifacts,
        "downstream_takedown_execution_ledger_path": (
            "downstream_takedown_execution_ledger.json" if consent_revoked else None
        ),
        "required_actions": required_takedown_actions if consent_revoked else [],
        "downstream_unexecuted_actions": [
            "remove_or_expire_hosted_sessions",
            "notify_webapp_rights_privacy_blocking",
            "queue_customer_notice_if_required",
        ]
        if consent_revoked
        else [],
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "revocation_blocks_delivery_and_training_use": consent_revoked,
            "local_package_takedown_instruction_written": True,
            "downstream_takedown_handoff_written": consent_revoked,
            "downstream_takedown_handoff_is_not_takedown_execution": True,
            "webapp_or_hosted_takedown_execution_proven": False,
            "takedown_manifest_is_not_legal_advice": True,
        },
    }
    record = {
        "schema_version": "post_training_consent_evidence.v1",
        "generated_at": generated_at,
        "status": status,
        "consent_evidence_present": consent_evidence_present,
        "consent_status": consent_status,
        "consent_revoked": consent_revoked,
        "consent_revoked_at": consent_revoked_at or None,
        "consent_scope": consent_scope,
        "evidence_uris": evidence_uris,
        "rights_packet_status": rights_packet.get("status"),
        "rights_packet_record_count": int(rights_packet.get("record_count") or len(records)),
        "blockers": blockers,
        "revenue_share_review": revenue_share_review,
        "data_processing_terms_review": data_processing_terms,
        "revocation_takedown_manifest_path": "revocation_takedown_manifest.json",
        "downstream_takedown_execution_ledger_path": (
            "downstream_takedown_execution_ledger.json" if consent_revoked else None
        ),
        "downstream_takedown_execution_ledger": (
            downstream_takedown_execution_ledger if consent_revoked else {}
        ),
        "downstream_takedown_artifacts": downstream_takedown_artifacts,
        "revocation_takedown": revocation_takedown,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "consent_record_documents_source_evidence_only": True,
            "consent_record_is_not_external_use_approval": True,
            "consent_revocation_blocks_downstream_use": consent_revoked,
        },
    }
    write_json(output_dir / "consent_evidence.json", record)
    write_json(output_dir / "revocation_takedown_manifest.json", revocation_takedown)
    if consent_revoked:
        write_json(
            output_dir / "downstream_takedown_execution_ledger.json",
            downstream_takedown_execution_ledger,
        )
        webapp_notice = {
            "schema_version": WEBAPP_RIGHTS_PRIVACY_TAKEDOWN_NOTICE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "queued_unexecuted_webapp_rights_privacy_blocking",
            "consent_revoked": True,
            "consent_revoked_at": consent_revoked_at or None,
            "required_webapp_state": "blocked_consent_revoked_takedown_required",
            "revocation_takedown_manifest_path": "revocation_takedown_manifest.json",
            "required_actions": [
                "mark_webapp_rights_privacy_blocking",
                "notify_webapp_rights_privacy_blocking",
                "hide_package_delivery_affordances",
                "hide_training_export_affordances",
            ],
            "webapp_takedown_executed": False,
            "claim_boundary": {
                **dict(CLAIM_BOUNDARY),
                "notice_is_downstream_handoff_only": True,
                "webapp_takedown_execution_proven": False,
                "hosted_session_takedown_execution_proven": False,
                "deployment_approval_proven": False,
            },
        }
        hosted_request = {
            "schema_version": HOSTED_SESSION_TAKEDOWN_REQUEST_SCHEMA_VERSION,
            "generated_at": generated_at,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "queued_unexecuted_hosted_session_takedown",
            "consent_revoked": True,
            "consent_revoked_at": consent_revoked_at or None,
            "revocation_takedown_manifest_path": "revocation_takedown_manifest.json",
            "required_actions": [
                "remove_hosted_review_assets",
                "remove_or_expire_hosted_sessions",
                "disable_hosted_session_replay_access",
            ],
            "hosted_review_assets_access_allowed": False,
            "hosted_session_takedown_executed": False,
            "claim_boundary": {
                **dict(CLAIM_BOUNDARY),
                "request_is_downstream_handoff_only": True,
                "hosted_session_takedown_execution_proven": False,
                "webapp_takedown_execution_proven": False,
                "deployment_approval_proven": False,
            },
        }
        write_json(
            output_dir / "webapp_rights_privacy_takedown_notice.json",
            webapp_notice,
        )
        write_json(output_dir / "hosted_session_takedown_request.json", hosted_request)
    return record


def _package_clip_rights_metadata(
    *,
    consent_evidence: Mapping[str, Any],
    revocation_takedown: Mapping[str, Any],
) -> Dict[str, Any]:
    consent_revoked = bool(
        _explicit_true(
            consent_evidence.get("consent_revoked"),
            revocation_takedown.get("consent_revoked"),
        )
        or _first_text(
            consent_evidence.get("consent_revoked_at"),
            revocation_takedown.get("consent_revoked_at"),
        )
        or revocation_takedown.get("status") == "takedown_required"
    )
    status = _first_text(
        consent_evidence.get("status"),
        "blocked_consent_revoked_takedown_required" if consent_revoked else None,
    )
    metadata = {
        "metadata_source": "package_consent_evidence",
        "license_status": status or "review_required",
        "consent_evidence_status": status or None,
        "consent_scope": _string_list(consent_evidence.get("consent_scope")),
        "consent_revoked": consent_revoked,
        "consent_revoked_at": consent_evidence.get("consent_revoked_at")
        or revocation_takedown.get("consent_revoked_at"),
        "delivery_blocked_by_consent_revocation": consent_revoked,
        "signed_access_revoked_by_consent": consent_revoked,
        "commercial_use_claim_allowed": False,
        "external_licensing_claim_allowed": False,
        "redaction_status": _first_text(
            consent_evidence.get("redaction_status"),
            consent_evidence.get("privacy_redaction_status"),
            "not_declared",
        ),
        "fallback_redaction_used": bool(consent_evidence.get("fallback_redaction_used")),
        "manual_rights_review_recommended": bool(
            consent_revoked or status != "consent_evidence_present"
        ),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [])}


def _build_curation_report(
    *,
    clips: Mapping[str, Any],
    trace: Mapping[str, Any],
    sc3_action_report: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    rows = _clip_rows(clips)
    attempts = _attempts_by_id(trace)
    accepted: List[str] = []
    rejected: List[Dict[str, Any]] = []
    blockers: List[str] = []
    clip_reports: List[Dict[str, Any]] = []
    for index, clip in enumerate(rows):
        clip_name = _clip_id(clip, index)
        curation = _mapping(clip.get("curation") or clip.get("quality") or clip.get("oscar_curation"))
        attempt_ref = str(clip.get("attempt_id") or clip.get("scenario_id") or "").strip()
        attempt = attempts.get(attempt_ref, {})
        vectors = _sc3_action_vectors_for_attempt(attempt) if attempt else []

        frame_count = _finite_int(
            _first_present(
                curation.get("frame_count"),
                clip.get("frame_count"),
                clip.get("num_frames"),
                clip.get("video_frame_count"),
            )
        )
        static_motion = _finite_float(
            _first_present(
                curation.get("camera_motion_m"),
                clip.get("camera_motion_m"),
                clip.get("max_camera_motion_m"),
            )
        )
        action_motion = _finite_float(
            _first_present(
                curation.get("action_motion_score"),
                clip.get("action_motion_score"),
                clip.get("manipulator_motion_score"),
            )
        )
        if action_motion is None and vectors:
            action_motion = max(sum(abs(value) for value in vector) for vector in vectors)
        visibility = _finite_float(
            _first_present(
                curation.get("visible_skeleton_fraction"),
                clip.get("visible_skeleton_fraction"),
                clip.get("target_visibility_fraction"),
                clip.get("visibility_fraction"),
            )
        )
        sharpness = _finite_float(
            _first_present(
                curation.get("sharpness_score"),
                clip.get("sharpness_score"),
                curation.get("mean_sharpness_score"),
                clip.get("mean_sharpness_score"),
            )
        )

        clip_blockers: List[str] = []
        frame_passed, frame_blockers, frame_evidence = _evidence_gate(
            explicit=curation.get("min_frame_filter_passed") if "min_frame_filter_passed" in curation else clip.get("min_frame_filter_passed"),
            measured=float(frame_count) if frame_count is not None else None,
            threshold=float(OSCAR_MIN_FRAME_COUNT),
            op=">=",
            missing_blocker="min_frame_count_missing",
        )
        static_passed, static_blockers, static_evidence = _evidence_gate(
            explicit=curation.get("static_camera_filter_passed") if "static_camera_filter_passed" in curation else clip.get("static_camera_filter_passed"),
            measured=static_motion,
            threshold=OSCAR_MAX_STATIC_CAMERA_MOTION_M,
            op="<=",
            missing_blocker="static_camera_evidence_missing",
        )
        action_passed, action_blockers, action_evidence = _evidence_gate(
            explicit=curation.get("meaningful_action_filter_passed") if "meaningful_action_filter_passed" in curation else clip.get("meaningful_action_filter_passed"),
            measured=action_motion,
            threshold=OSCAR_MIN_ACTION_MOTION_SCORE,
            op=">=",
            missing_blocker="meaningful_action_evidence_missing",
        )
        visibility_passed, visibility_blockers, visibility_evidence = _evidence_gate(
            explicit=curation.get("visible_skeleton_filter_passed") if "visible_skeleton_filter_passed" in curation else clip.get("visible_skeleton_filter_passed"),
            measured=visibility,
            threshold=OSCAR_MIN_VISIBLE_SKELETON_FRACTION,
            op=">=",
            missing_blocker="visible_skeleton_evidence_missing",
        )
        sharpness_passed, sharpness_blockers, sharpness_evidence = _evidence_gate(
            explicit=curation.get("blur_filter_passed") if "blur_filter_passed" in curation else clip.get("blur_filter_passed"),
            measured=sharpness,
            threshold=OSCAR_MIN_SHARPNESS_SCORE,
            op=">=",
            missing_blocker="blur_or_sharpness_evidence_missing",
        )
        for gate_blockers in (
            frame_blockers,
            static_blockers,
            action_blockers,
            visibility_blockers,
            sharpness_blockers,
        ):
            clip_blockers.extend(gate_blockers)
        passed = bool(
            frame_passed
            and static_passed
            and action_passed
            and visibility_passed
            and sharpness_passed
        )
        if passed:
            accepted.append(clip_name)
        else:
            rejected.append({"clip_id": clip_name, "blockers": clip_blockers})
            blockers.extend(f"{clip_name}:{blocker}" for blocker in clip_blockers)
        clip_reports.append(
            {
                "clip_id": clip_name,
                "status": "accepted" if passed else "rejected",
                "attempt_ref": attempt_ref or None,
                "gates": {
                    "min_frame": {"passed": frame_passed, **frame_evidence},
                    "static_camera": {"passed": static_passed, **static_evidence},
                    "meaningful_action": {"passed": action_passed, **action_evidence},
                    "visible_skeleton": {"passed": visibility_passed, **visibility_evidence},
                    "blur_or_sharpness": {"passed": sharpness_passed, **sharpness_evidence},
                },
                "blockers": clip_blockers,
            }
        )
    if not rows:
        blockers.append("clips_manifest_missing_or_empty")
    sc3_hard_blockers = [
        blocker
        for blocker in _string_list(sc3_action_report.get("blockers"))
        if blocker.rsplit(":", 1)[-1] not in SC3_NO_ACTION_DATA_BLOCKERS
    ]
    if sc3_action_report.get("status") != "passed" and sc3_hard_blockers:
        blockers.append("sc3_action_normalization_blocked")
    status = "passed" if rows and not blockers else "blocked"
    return {
        "schema_version": CURATION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "filter_family": "oscar_style_source_data_curation",
        "thresholds": {
            "min_frame_count": OSCAR_MIN_FRAME_COUNT,
            "max_static_camera_motion_m": OSCAR_MAX_STATIC_CAMERA_MOTION_M,
            "min_action_motion_score": OSCAR_MIN_ACTION_MOTION_SCORE,
            "min_visible_skeleton_fraction": OSCAR_MIN_VISIBLE_SKELETON_FRACTION,
            "min_sharpness_score": OSCAR_MIN_SHARPNESS_SCORE,
        },
        "source_clip_count": len(rows),
        "accepted_clip_count": len(accepted),
        "rejected_clip_count": len(rejected),
        "accepted_clip_ids": accepted,
        "rejected_clips": rejected,
        "clips": clip_reports,
        "blockers": _string_list(blockers),
    }


def _semantic_dedup_key(clip: Mapping[str, Any]) -> str:
    curation = _mapping(clip.get("curation") or clip.get("quality") or clip.get("oscar_curation"))
    direct = str(
        curation.get("semantic_dedup_key")
        or clip.get("semantic_dedup_key")
        or clip.get("dedup_key")
        or ""
    ).strip()
    if direct:
        return direct
    visual = str(
        curation.get("visual_embedding_hash")
        or clip.get("visual_embedding_hash")
        or clip.get("visual_hash")
        or ""
    ).strip()
    trajectory = str(
        curation.get("trajectory_hash")
        or clip.get("trajectory_hash")
        or clip.get("action_trajectory_hash")
        or ""
    ).strip()
    if visual and trajectory:
        return f"visual:{visual}|trajectory:{trajectory}"
    return ""


def _build_semantic_dedup_report(
    *,
    clips: Mapping[str, Any],
    curation_report: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    accepted_ids = set(_string_list(curation_report.get("accepted_clip_ids")))
    rows = [
        clip
        for index, clip in enumerate(_clip_rows(clips))
        if _clip_id(clip, index) in accepted_ids
    ]
    groups: Dict[str, List[str]] = {}
    missing: List[str] = []
    for index, clip in enumerate(rows):
        clip_name = _clip_id(clip, index)
        key = _semantic_dedup_key(clip)
        if not key:
            missing.append(clip_name)
            continue
        groups.setdefault(key, []).append(clip_name)
    duplicate_groups = [
        {"semantic_dedup_key": key, "clip_ids": ids}
        for key, ids in groups.items()
        if len(ids) > 1
    ]
    blockers: List[str] = []
    if missing and len(rows) > 1:
        blockers.extend(f"{clip_id}:semantic_dedup_evidence_missing" for clip_id in missing)
    if duplicate_groups:
        blockers.extend(
            f"semantic_duplicate_group:{group['semantic_dedup_key']}"
            for group in duplicate_groups
        )
    status = "passed" if not blockers else "blocked"
    return {
        "schema_version": SEMANTIC_DEDUP_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "source": "visual_semantic_key_plus_trajectory_key",
        "accepted_input_clip_count": len(rows),
        "deduped_clip_count": len(rows) - sum(max(0, len(group["clip_ids"]) - 1) for group in duplicate_groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "clips_missing_dedup_evidence": missing,
        "blockers": _string_list(blockers),
    }


def _rows(payload: Mapping[str, Any], key: str) -> List[Dict[str, Any]]:
    values = payload.get(key)
    if isinstance(values, list):
        return [dict(item) for item in values if isinstance(item, Mapping)]
    return []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _artifact(base_dir: Path, path: Path) -> Dict[str, Any]:
    return {
        "path": _relative_to(base_dir, path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha_file(path) if path.is_file() else None,
    }


def _job_artifact(job_dir: Path, name: str) -> str | None:
    path = job_dir / name
    return name.replace("\\", "/") if path.is_file() else None


def _pipeline_artifact(pipeline_dir: Path, relative_path: str) -> str | None:
    path = pipeline_dir / relative_path
    return relative_path if path.is_file() else None


def _optional_export_formats() -> Dict[str, Dict[str, Any]]:
    formats: Dict[str, Dict[str, Any]] = {}
    dependencies = {
        "rlds": ("rlds",),
        "lerobot": ("lerobot",),
        "lerobot_v3": ("pyarrow", "pandas"),
        "gr00t_lerobot": ("pyarrow", "pandas"),
        "hdf5": ("h5py",),
        "parquet": ("pyarrow", "pandas"),
    }
    for name, packages in dependencies.items():
        available = all(importlib.util.find_spec(package) is not None for package in packages)
        formats[name] = {
            "status": "available_not_written" if available else "blocked_optional_dependency_missing",
            "dependencies": list(packages),
            "format_written": False,
        }
    formats["video_bundle"] = {
        "status": "degraded_manifest_only",
        "dependencies": ["clips_manifest.json", "clip files when present"],
        "format_written": False,
    }
    return formats


def _live_closure_gate_reference(
    live_closure: Mapping[str, Any],
    gate_id: str,
) -> Dict[str, Any]:
    gate = _mapping(_mapping(live_closure.get("gates")).get(gate_id))
    evidence = _mapping(gate.get("evidence"))
    return {
        "gate_id": gate_id,
        "present": bool(gate),
        "passed": bool(gate.get("passed")),
        "blockers": _string_list(gate.get("blockers")),
        "evidence_keys": sorted(evidence),
    }


def _gate_blockers(
    gate_reference: Mapping[str, Any],
    gate_id: str,
    fallback_blocker: str,
) -> List[str]:
    if not gate_reference.get("present"):
        return [f"{gate_id}_gate_missing"]
    if gate_reference.get("passed"):
        return []
    blockers = _string_list(gate_reference.get("blockers"))
    if not blockers:
        blockers = [fallback_blocker]
    return [f"{gate_id}:{blocker}" for blocker in blockers]


def _handoff_claim_boundary(
    *,
    export_ready: bool,
    customer_handoff_ready: bool,
    review_acceptance_proven: bool,
    rights_privacy_scope_proven: bool,
    signed_delivery_access_proven: bool,
) -> Dict[str, Any]:
    return {
        **dict(CLAIM_BOUNDARY),
        "post_training_package_export_ready": bool(export_ready),
        "customer_handoff_ready": bool(customer_handoff_ready),
        "hosted_access_ready": bool(customer_handoff_ready),
        "review_acceptance_proven": bool(review_acceptance_proven),
        "rights_privacy_scope_proven": bool(rights_privacy_scope_proven),
        "signed_delivery_access_proven": bool(signed_delivery_access_proven),
        "delivery_approval_proven": False,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "field_readiness_proven": False,
        "safety_validation_proven": False,
        "simulator_execution_proven": False,
        "robot_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "training_completed": False,
        "public_claim_upgrade_allowed": False,
    }


def _merge_claim_boundary(
    existing: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(existing),
        **dict(boundary),
        "delivery_approval_proven": False,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "field_readiness_proven": False,
        "safety_validation_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def _handoff_status(*, export_ready: bool, customer_handoff_ready: bool) -> str:
    if customer_handoff_ready:
        return "customer_handoff_ready"
    if export_ready:
        return "export_ready_handoff_blocked"
    return "blocked_missing_package_export_inputs"


def _handoff_summary(
    *,
    generated_at: str,
    export_ready: bool,
    customer_handoff_ready: bool,
    handoff_blockers: Sequence[str],
    gate_blockers: Mapping[str, Sequence[str]],
    live_gate_references: Mapping[str, Mapping[str, Any]],
    webapp_ids: Mapping[str, Any],
    included_artifacts: Mapping[str, str],
    boundary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": HANDOFF_SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": _handoff_status(
            export_ready=export_ready,
            customer_handoff_ready=customer_handoff_ready,
        ),
        "post_training_package_export_ready": bool(export_ready),
        "customer_handoff_ready": bool(customer_handoff_ready),
        "handoff_ready": bool(customer_handoff_ready),
        "blockers": _string_list(handoff_blockers),
        "gate_blockers": {
            key: _string_list(value) for key, value in gate_blockers.items()
        },
        "webapp_ids": dict(webapp_ids),
        "artifact_paths": {
            "post_training_data_package_export_manifest": (
                "post_training_data_package_export_manifest.json"
            ),
            "customer_handoff_report": "customer_handoff_report.json",
            "delivery_manifest": "delivery_manifest.json",
            "signed_access_manifest": "signed_access_manifest.json",
            "proof_boundary": included_artifacts.get("proof_boundary"),
            "live_eval_closure_manifest": included_artifacts.get("live_eval_closure_manifest"),
            "rights_packet": included_artifacts.get("rights_packet"),
            "review_resolution_ledger": included_artifacts.get("review_resolution_ledger"),
            "accepted_failure_labels": included_artifacts.get("accepted_failure_labels"),
        },
        "live_closure_gate_references": {
            key: dict(value) for key, value in live_gate_references.items()
        },
        "claim_boundary": dict(boundary),
    }


def _read_existing_handoff_payload(
    *,
    output_dir: Path,
    job_dir: Path | None,
    name: str,
) -> Dict[str, Any]:
    output_payload = _read_optional_mapping(output_dir / name)
    if output_payload:
        return output_payload
    if job_dir and (job_dir / name).resolve() != (output_dir / name).resolve():
        return _read_optional_mapping(job_dir / name)
    return {}


def _write_customer_handoff_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = _mapping(report.get("post_training_data_package_handoff"))
    blockers = _string_list(summary.get("blockers") or report.get("blockers"))
    lines = [
        "# Post-Training Data Package Handoff",
        "",
        f"- Status: `{summary.get('status') or report.get('status')}`",
        f"- Export ready: `{bool(summary.get('post_training_package_export_ready'))}`",
        f"- Customer handoff ready: `{bool(summary.get('customer_handoff_ready'))}`",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Package export readiness does not prove hosted access, delivery approval, deployment approval, safety validation, or field readiness.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_handoff_manifests(
    *,
    output_dir: Path,
    job_dir: Path | None,
    generated_at: str,
    scene_id: str,
    capture_id: str,
    export_ready: bool,
    included_artifacts: Mapping[str, str],
    live_closure: Mapping[str, Any],
    live_gate_references: Mapping[str, Mapping[str, Any]],
    trace: Mapping[str, Any],
    labels: Mapping[str, Any],
    clips: Mapping[str, Any],
    consent_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    revocation_takedown = _mapping(consent_evidence.get("revocation_takedown"))
    consent_revoked = _revocation_takedown_required(consent_evidence)
    revocation_blockers = (
        [
            "consent:consent_revoked_takedown_required",
            "signed_access_revoked_by_consent",
            "delivery_revoked_by_consent",
        ]
        if consent_revoked
        else []
    )
    gate_blockers = {
        "webapp_upstream_truth": _gate_blockers(
            live_gate_references["webapp_upstream_truth"],
            "webapp_upstream_truth",
            "webapp_upstream_truth_not_proven",
        ),
        "rights_privacy_scope": _gate_blockers(
            live_gate_references["rights_privacy_scope"],
            "rights_privacy_scope",
            "rights_privacy_scope_not_proven",
        ),
        "review_acceptance": _gate_blockers(
            live_gate_references["review_acceptance"],
            "review_acceptance",
            "review_acceptance_not_proven",
        ),
        "signed_delivery_access": _gate_blockers(
            live_gate_references["signed_delivery_access"],
            "signed_delivery_access",
            "signed_delivery_access_not_proven",
        ),
    }
    handoff_blockers = _string_list(
        [
            *(["post_training_data_package_export_not_ready"] if not export_ready else []),
            *gate_blockers["webapp_upstream_truth"],
            *gate_blockers["rights_privacy_scope"],
            *gate_blockers["review_acceptance"],
            *gate_blockers["signed_delivery_access"],
            *revocation_blockers,
        ]
    )
    customer_handoff_ready = bool(export_ready and not handoff_blockers)
    boundary = _handoff_claim_boundary(
        export_ready=export_ready,
        customer_handoff_ready=customer_handoff_ready,
        review_acceptance_proven=bool(live_gate_references["review_acceptance"]["passed"]),
        rights_privacy_scope_proven=bool(
            live_gate_references["rights_privacy_scope"]["passed"]
        ),
        signed_delivery_access_proven=bool(
            live_gate_references["signed_delivery_access"]["passed"]
        ),
    )
    if consent_revoked:
        boundary.update(
            {
                "post_training_package_export_ready": False,
                "customer_handoff_ready": False,
                "hosted_access_ready": False,
                "rights_privacy_scope_proven": False,
                "signed_delivery_access_proven": False,
                "consent_revocation_blocks_downstream_use": True,
                "local_package_takedown_instruction_written": True,
                "webapp_or_hosted_takedown_execution_proven": False,
            }
        )
    webapp_evidence = _mapping(
        _mapping(_mapping(live_closure.get("gates")).get("webapp_upstream_truth")).get(
            "evidence"
        )
    )
    webapp_ids = _mapping(webapp_evidence.get("ids"))
    summary = _handoff_summary(
        generated_at=generated_at,
        export_ready=export_ready,
        customer_handoff_ready=customer_handoff_ready,
        handoff_blockers=handoff_blockers,
        gate_blockers=gate_blockers,
        live_gate_references=live_gate_references,
        webapp_ids=webapp_ids,
        included_artifacts=included_artifacts,
        boundary=boundary,
    )
    if consent_revoked:
        summary.update(
            {
                "status": "revoked_consent_takedown_required",
                "post_training_package_export_ready": False,
                "customer_handoff_ready": False,
                "handoff_ready": False,
                "local_package_access_revoked": True,
                "delivery_blocked_by_consent_revocation": True,
                "signed_access_revoked_by_consent": True,
                "revocation_takedown_manifest_path": "revocation_takedown_manifest.json",
                "revocation_takedown": dict(revocation_takedown),
            }
        )
        summary["artifact_paths"]["revocation_takedown_manifest"] = (
            "revocation_takedown_manifest.json"
        )

    existing_report = _read_existing_handoff_payload(
        output_dir=output_dir,
        job_dir=job_dir,
        name="customer_handoff_report.json",
    )
    report_status = (
        "revoked_consent_takedown_required"
        if consent_revoked
        else existing_report.get("status")
        if existing_report and existing_report.get("status")
        else summary["status"]
    )
    report_blockers = _string_list(
        [
            *_string_list(existing_report.get("blockers")),
            *handoff_blockers,
        ]
        if consent_revoked
        else existing_report.get("blockers") or handoff_blockers
    )
    report = {
        **existing_report,
        "schema_version": existing_report.get(
            "schema_version",
            CUSTOMER_HANDOFF_REPORT_SCHEMA_VERSION,
        ),
        "generated_at": generated_at,
        "status": report_status,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "blockers": report_blockers,
        "post_training_data_package_export_path": (
            "post_training_data_package_export_manifest.json"
        ),
        "delivery_manifest_path": "delivery_manifest.json",
        "signed_access_manifest_path": "signed_access_manifest.json",
        "revocation_takedown_manifest_path": "revocation_takedown_manifest.json",
        "local_package_access_revoked": bool(consent_revoked),
        "delivery_blocked_by_consent_revocation": bool(consent_revoked),
        "signed_access_revoked_by_consent": bool(consent_revoked),
        "buyer_summary": {
            **_mapping(existing_report.get("buyer_summary")),
            "post_training_package_export_ready": bool(export_ready),
            "customer_handoff_ready": bool(customer_handoff_ready),
            "consent_revoked": bool(consent_revoked),
            "local_package_access_revoked": bool(consent_revoked),
            "attempt_count": int(trace.get("attempt_count") or 0),
            "failure_label_count": int(labels.get("label_count") or 0),
            "clip_count": int(clips.get("clip_count") or 0),
        },
        "known_limits": _string_list(existing_report.get("known_limits"))
        or [
            "Post-Training Data Package export readiness is not hosted access.",
            "Signed delivery access is package access only, not deployment approval.",
            "This handoff does not prove safety validation or physical field readiness.",
        ],
        "post_training_data_package_handoff": summary,
        "revocation_takedown": dict(revocation_takedown),
        "claim_boundary": _merge_claim_boundary(
            _mapping(existing_report.get("claim_boundary")),
            {
                **boundary,
                "consent_revocation_blocks_downstream_use": bool(consent_revoked),
            },
        ),
    }
    write_json(output_dir / "customer_handoff_report.json", report)
    _write_customer_handoff_markdown(output_dir / "customer_handoff_report.md", report)

    existing_signed_access = _read_existing_handoff_payload(
        output_dir=output_dir,
        job_dir=job_dir,
        name="signed_access_manifest.json",
    )
    signed_access_ready = bool(
        live_gate_references["signed_delivery_access"]["passed"]
    ) and not consent_revoked
    signed_access_blockers = _string_list(
        [
            *gate_blockers["signed_delivery_access"],
            *(["signed_access_revoked_by_consent"] if consent_revoked else []),
            *(["consent:consent_revoked_takedown_required"] if consent_revoked else []),
        ]
    )
    signed_access_status = (
        "revoked_consent_takedown_required"
        if consent_revoked
        else existing_signed_access.get("status")
        if existing_signed_access and existing_signed_access.get("status")
        else ("signed_access_ready" if signed_access_ready else "blocked_signed_delivery_access")
    )
    signed_access_manifest_blockers = _string_list(
        [
            *_string_list(existing_signed_access.get("blockers")),
            *signed_access_blockers,
        ]
        if consent_revoked
        else existing_signed_access.get("blockers") or signed_access_blockers
    )
    signed_access = {
        **existing_signed_access,
        "schema_version": existing_signed_access.get(
            "schema_version",
            SIGNED_ACCESS_MANIFEST_SCHEMA_VERSION,
        ),
        "generated_at": generated_at,
        "status": signed_access_status,
        "blockers": signed_access_manifest_blockers,
        "signed_delivery_access_proven": bool(signed_access_ready),
        "signed_access_ready": bool(signed_access_ready),
        "signed_access_revoked_by_consent": bool(consent_revoked),
        "local_package_access_revoked": bool(consent_revoked),
        "revocation_takedown_manifest_path": "revocation_takedown_manifest.json",
        "revocation_takedown": dict(revocation_takedown),
        "customer_handoff_ready": bool(customer_handoff_ready),
        "handoff_blockers": handoff_blockers,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "claim_boundary": _merge_claim_boundary(
            _mapping(existing_signed_access.get("claim_boundary")),
            {
                **boundary,
                "consent_revocation_blocks_downstream_use": bool(consent_revoked),
            },
        ),
    }
    write_json(output_dir / "signed_access_manifest.json", signed_access)

    existing_delivery = _read_existing_handoff_payload(
        output_dir=output_dir,
        job_dir=job_dir,
        name="delivery_manifest.json",
    )
    delivery_status = (
        "revoked_consent_takedown_required"
        if consent_revoked
        else existing_delivery.get("status")
        if existing_delivery and existing_delivery.get("status")
        else summary["status"]
    )
    delivery_blockers = _string_list(
        [
            *_string_list(existing_delivery.get("blockers")),
            *handoff_blockers,
        ]
        if consent_revoked
        else existing_delivery.get("blockers") or handoff_blockers
    )
    delivery = {
        **existing_delivery,
        "schema_version": existing_delivery.get(
            "schema_version",
            DELIVERY_MANIFEST_SCHEMA_VERSION,
        ),
        "generated_at": generated_at,
        "status": delivery_status,
        "blockers": delivery_blockers,
        "post_training_data_package_export_path": (
            "post_training_data_package_export_manifest.json"
        ),
        "customer_handoff_report_path": "customer_handoff_report.json",
        "signed_access_manifest_path": "signed_access_manifest.json",
        "revocation_takedown_manifest_path": "revocation_takedown_manifest.json",
        "local_package_index_path": "package_index.json",
        "archive_manifest_path": "archive_manifest.json",
        "delivery_blocked_by_consent_revocation": bool(consent_revoked),
        "local_package_access_revoked": bool(consent_revoked),
        "signed_access_revoked_by_consent": bool(consent_revoked),
        "post_training_data_package_handoff": summary,
        "revocation_takedown": dict(revocation_takedown),
        "claim_boundary": _merge_claim_boundary(
            _mapping(existing_delivery.get("claim_boundary")),
            {
                **boundary,
                "consent_revocation_blocks_downstream_use": bool(consent_revoked),
            },
        ),
    }
    write_json(output_dir / "delivery_manifest.json", delivery)

    return {
        "customer_handoff_report": report,
        "delivery_manifest": delivery,
        "signed_access_manifest": signed_access,
        "summary": summary,
    }


def _annotate_live_closure_with_handoff(
    *,
    job_dir: Path | None,
    handoff_summary: Mapping[str, Any],
) -> None:
    if not job_dir:
        return
    path = job_dir / "live_eval_closure_manifest.json"
    live_closure = _read_optional_mapping(path)
    if not live_closure:
        return
    boundary = _mapping(handoff_summary.get("claim_boundary"))
    live_closure["post_training_data_package_handoff"] = dict(handoff_summary)
    proof_boundary = _mapping(live_closure.get("proof_boundary"))
    proof_boundary.update(
        {
            "post_training_package_export_ready": bool(
                handoff_summary.get("post_training_package_export_ready")
            ),
            "customer_handoff_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "hosted_access_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "delivery_approval_proven": False,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "field_readiness_proven": False,
            "safety_validation_proven": bool(boundary.get("safety_validation_proven")),
            "public_claim_upgrade_allowed": False,
        }
    )
    live_closure["proof_boundary"] = proof_boundary
    claim_boundary = _mapping(live_closure.get("claim_boundary"))
    claim_boundary.update(
        {
            "post_training_package_export_ready": bool(
                handoff_summary.get("post_training_package_export_ready")
            ),
            "customer_handoff_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "hosted_access_ready": bool(handoff_summary.get("customer_handoff_ready")),
            "delivery_approval_proven": False,
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "field_readiness_proven": False,
            "safety_validation_proven": False,
            "public_claim_upgrade_allowed": False,
        }
    )
    live_closure["claim_boundary"] = claim_boundary
    write_json(path, live_closure)


def _rows_for_optional_exports(
    *,
    attempts: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    scene_id: str,
    capture_id: str,
) -> List[Dict[str, Any]]:
    labels_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
    labels_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for label in label_rows:
        label_payload = dict(label)
        attempt_id = str(label.get("attempt_id") or "").strip()
        scenario_id = str(label.get("scenario_id") or "").strip()
        if attempt_id:
            labels_by_attempt.setdefault(attempt_id, []).append(label_payload)
        if scenario_id:
            labels_by_scenario.setdefault(scenario_id, []).append(label_payload)

    rows: List[Dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        attempt_id = str(attempt.get("attempt_id") or f"attempt_{index}").strip()
        scenario_id = str(attempt.get("scenario_id") or "").strip()
        labels = labels_by_attempt.get(attempt_id) or labels_by_scenario.get(scenario_id) or []
        sc3_vectors = _sc3_action_vectors_for_attempt(attempt)
        rows.append(
            {
                "episode_id": attempt_id,
                "episode_index": index - 1,
                "scene_id": scene_id,
                "capture_id": capture_id,
                "task_id": attempt.get("task_id"),
                "scenario_id": scenario_id or None,
                "policy_id": attempt.get("policy_id"),
                "success": bool(attempt.get("success")),
                "status": attempt.get("status") or "unknown",
                "metrics": dict(_mapping(attempt.get("metrics"))),
                "actions": attempt.get("actions") or attempt.get("action_trace") or [],
                "sc3_7d_delta_end_effector_actions": sc3_vectors,
                "sc3_action_contract_valid": bool(sc3_vectors),
                "observations": attempt.get("observations") or attempt.get("observation_refs") or [],
                "failure_labels": labels,
                "package_metrics": dict(metrics),
                "source_format": "blueprint_normalized_attempt_trace.v1",
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
        )
    if rows:
        return rows
    return [
        {
            "episode_id": "missing_attempts",
            "episode_index": 0,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "task_id": None,
            "scenario_id": None,
            "policy_id": None,
            "success": False,
            "status": "missing_source_attempts",
            "metrics": dict(metrics),
            "actions": [],
            "sc3_7d_delta_end_effector_actions": [],
            "sc3_action_contract_valid": False,
            "observations": [],
            "failure_labels": list(label_rows),
            "package_metrics": dict(metrics),
            "source_format": "blueprint_normalized_attempt_trace.v1",
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    ]


def _flat_export_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "episode_id": row.get("episode_id"),
                "episode_index": row.get("episode_index"),
                "scene_id": row.get("scene_id"),
                "capture_id": row.get("capture_id"),
                "task_id": row.get("task_id"),
                "scenario_id": row.get("scenario_id"),
                "policy_id": row.get("policy_id"),
                "success": bool(row.get("success")),
                "status": row.get("status"),
                "metrics_json": json.dumps(row.get("metrics") or {}, sort_keys=True),
                "actions_json": json.dumps(row.get("actions") or [], sort_keys=True),
                "sc3_7d_actions_json": json.dumps(
                    row.get("sc3_7d_delta_end_effector_actions") or [],
                    sort_keys=True,
                ),
                "observations_json": json.dumps(row.get("observations") or [], sort_keys=True),
                "failure_labels_json": json.dumps(
                    row.get("failure_labels") or [],
                    sort_keys=True,
                ),
            }
        )
    return out


def _write_native_hdf5(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return False
    ensure_dir(path.parent)
    payloads = [json.dumps(dict(row), sort_keys=True) for row in rows]
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "blueprint_post_training_hdf5.v1"
        handle.attrs["source_format"] = "blueprint_normalized_attempt_trace.v1"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("episodes_json", data=payloads, dtype=string_dtype)
    return True


def _write_native_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    if importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None:
        return False
    import pandas as pd  # type: ignore[import-not-found]

    ensure_dir(path.parent)
    pd.DataFrame(_flat_export_rows(rows)).to_parquet(path, index=False)
    return True


def _write_structured_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    if importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None:
        return False
    import pandas as pd  # type: ignore[import-not-found]

    ensure_dir(path.parent)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)
    return True


def _write_lerobot_tasks_parquet(path: Path, tasks: Sequence[Mapping[str, Any]]) -> bool:
    if importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None:
        return False
    import pandas as pd  # type: ignore[import-not-found]

    ensure_dir(path.parent)
    task_names = [str(task.get("task") or task.get("name") or "").strip() for task in tasks]
    task_indices = [int(task.get("task_index") or index) for index, task in enumerate(tasks)]
    frame = pd.DataFrame({"task_index": task_indices}, index=task_names)
    frame.to_parquet(path)
    return True


def _clip_reference_values(row: Mapping[str, Any]) -> List[str]:
    refs: List[str] = []
    for key in (
        "materialized_path",
        "local_path",
        "clip_path",
        "video_path",
        "source_video_path",
        "source_path",
        "path",
        "uri",
        "url",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in refs:
            refs.append(value.strip())
    return refs


def _resolve_clip_source_path(
    row: Mapping[str, Any],
    source_roots: Sequence[Path],
) -> tuple[Path | None, str | None]:
    refs = _clip_reference_values(row)
    for reference in refs:
        candidate = Path(reference)
        if candidate.is_absolute() and candidate.is_file():
            return candidate, reference
        if not candidate.is_absolute():
            for root in source_roots:
                rooted = root / candidate
                if rooted.is_file():
                    return rooted, reference
    return None, refs[0] if refs else None


def _materialize_video_bundle(
    *,
    output_dir: Path,
    clips: Mapping[str, Any],
    generated_at: str,
    source_roots: Sequence[Path],
    package_rights_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    bundle_dir = output_dir / "exports" / "video_bundle"
    videos_dir = bundle_dir / "clips"
    clip_rows = _clip_rows(clips)
    materialized_clips: List[Dict[str, Any]] = []
    missing_clip_ids: List[str] = []
    for index, clip in enumerate(clip_rows):
        clip_id = _clip_id(clip, index)
        safe_id = _safe_path_component(clip_id, f"clip_{index:06d}")
        source_path, source_reference = _resolve_clip_source_path(clip, source_roots)
        row = dict(clip)
        clip_rights = _mapping(
            row.get("rights_metadata")
            or row.get("rights")
            or row.get("license")
            or row.get("privacy")
        )
        top_level_rights = {
            key: row.get(key)
            for key in (
                "license_status",
                "consent_scope",
                "consent_revoked",
                "consent_revoked_at",
                "redaction_status",
                "fallback_redaction_used",
                "manual_rights_review_recommended",
                "commercial_use_claim_allowed",
                "external_licensing_claim_allowed",
            )
            if row.get(key) not in (None, "", [])
        }
        if top_level_rights:
            clip_rights = {
                **clip_rights,
                **top_level_rights,
                "clip_metadata_source": "clip_manifest_top_level_fields",
            }
        package_rights = dict(package_rights_metadata or {})
        merged_rights = {
            **package_rights,
            **clip_rights,
        }
        for key in (
            "consent_revoked",
            "delivery_blocked_by_consent_revocation",
            "signed_access_revoked_by_consent",
            "fallback_redaction_used",
            "manual_rights_review_recommended",
        ):
            if package_rights.get(key) is True:
                merged_rights[key] = True
        if (
            package_rights.get("consent_revoked_at")
            and merged_rights.get("consent_revoked") is True
        ):
            merged_rights["consent_revoked_at"] = package_rights.get(
                "consent_revoked_at"
            )
        for key in (
            "commercial_use_claim_allowed",
            "external_licensing_claim_allowed",
        ):
            if package_rights.get(key) is False:
                merged_rights[key] = False
        if merged_rights:
            row["rights_metadata"] = merged_rights
            row["consent_scope"] = _string_list(
                row.get("consent_scope") or merged_rights.get("consent_scope")
            )
            row["license_status"] = (
                row.get("license_status") or merged_rights.get("license_status")
            )
            row["redaction_status"] = (
                row.get("redaction_status") or merged_rights.get("redaction_status")
            )
            row["fallback_redaction_used"] = _explicit_true(
                row.get("fallback_redaction_used"),
                merged_rights.get("fallback_redaction_used"),
            )
            row["manual_rights_review_recommended"] = _explicit_true(
                row.get("manual_rights_review_recommended"),
                merged_rights.get("manual_rights_review_recommended"),
            )
        row["clip_id"] = clip_id
        row["source_reference"] = source_reference
        if source_path is None:
            row.update(
                {
                    "materialized": False,
                    "missing_reason": "clip_file_not_found",
                    "materialized_path": None,
                    "sha256": None,
                    "size_bytes": 0,
                }
            )
            missing_clip_ids.append(clip_id)
            materialized_clips.append(row)
            continue
        suffix = source_path.suffix.lower() if source_path.suffix else ".mp4"
        destination = videos_dir / f"{index:06d}_{safe_id}{suffix}"
        ensure_dir(destination.parent)
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        artifact = _artifact(output_dir, destination)
        row.update(
            {
                "materialized": True,
                "materialized_path": artifact["path"],
                "sha256": artifact["sha256"],
                "size_bytes": artifact["size_bytes"],
                "video_format": suffix.lstrip("."),
            }
        )
        sidecar_path = destination.with_name(f"{destination.name}.metadata.json")
        rights_metadata = _mapping(row.get("rights_metadata"))
        sidecar_payload = {
            "schema_version": "post_training_clip_metadata_sidecar.v1",
            "generated_at": generated_at,
            "clip_id": clip_id,
            "source_reference": source_reference,
            "materialized_path": artifact["path"],
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
            "video_format": suffix.lstrip("."),
            "observation_source": row.get("observation_source"),
            "observation_source_detail": row.get("observation_source_detail"),
            "observation_source_is_model_derived": _explicit_true(
                row.get("observation_source_is_model_derived"),
                row.get("model_derived"),
            ),
            "observation_source_is_raw_capture_evidence": _explicit_true(
                row.get("observation_source_is_raw_capture_evidence")
            ),
            "rights_metadata": rights_metadata,
            "license_status": row.get("license_status"),
            "consent_scope": _string_list(row.get("consent_scope")),
            "consent_revoked": _strict_true(rights_metadata.get("consent_revoked")),
            "redaction_status": row.get("redaction_status"),
            "fallback_redaction_used": _explicit_true(
                row.get("fallback_redaction_used"),
                rights_metadata.get("fallback_redaction_used"),
            ),
            "manual_rights_review_recommended": bool(
                _explicit_true(
                    row.get("manual_rights_review_recommended"),
                    rights_metadata.get("manual_rights_review_recommended"),
                )
            ),
            "commercial_use_claim_allowed": _strict_true(
                rights_metadata.get("commercial_use_claim_allowed")
            ),
            "external_licensing_claim_allowed": _strict_true(
                rights_metadata.get("external_licensing_claim_allowed")
            ),
            "claim_boundary": {
                **dict(CLAIM_BOUNDARY),
                "clip_sidecar_is_rights_metadata_not_clearance": True,
                "standalone_clip_requires_sidecar_review": True,
            },
        }
        write_json(sidecar_path, sidecar_payload)
        sidecar_artifact = _artifact(output_dir, sidecar_path)
        row["metadata_sidecar_path"] = sidecar_artifact["path"]
        row["metadata_sidecar_sha256"] = sidecar_artifact["sha256"]
        row["metadata_sidecar_schema_version"] = sidecar_payload["schema_version"]
        materialized_clips.append(row)
    materialized_count = sum(1 for row in materialized_clips if row.get("materialized"))
    if not clip_rows:
        status = "written_manifest_no_clips"
    elif missing_clip_ids:
        status = "written_manifest_missing_clip_files"
    else:
        status = "written_materialized"
    manifest = {
        "schema_version": "post_training_video_bundle_manifest.v2",
        "generated_at": generated_at,
        "status": status,
        "source_clips": dict(clips),
        "clips": materialized_clips,
        "clip_count": len(clip_rows),
        "materialized_clip_count": materialized_count,
        "missing_clip_file_count": len(missing_clip_ids),
        "missing_clip_ids": missing_clip_ids,
        "all_declared_clips_materialized": bool(clip_rows) and not missing_clip_ids,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "materialized_video_is_training_media": materialized_count > 0,
            "clip_manifest_reference_alone_is_not_delivery": True,
        },
    }
    manifest_path = bundle_dir / "clips_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "path": _relative_to(output_dir, manifest_path),
        "manifest": manifest,
        "materialized_clips": materialized_clips,
    }


def _lerobot_episode_materialized_video_map(
    *,
    clips: Mapping[str, Any],
    source_roots: Sequence[Path],
) -> Dict[str, Dict[str, Any]]:
    video_by_attempt: Dict[str, Dict[str, Any]] = {}
    for index, clip in enumerate(_clip_rows(clips)):
        attempt_id = str(clip.get("attempt_id") or "").strip()
        if not attempt_id or attempt_id in video_by_attempt:
            continue
        source_path, source_reference = _resolve_clip_source_path(clip, source_roots)
        if source_path is None:
            continue
        video_by_attempt[attempt_id] = {
            "path": str(source_path),
            "source_reference": source_reference,
            "clip_id": _clip_id(clip, index),
        }
    return video_by_attempt


def _numeric_vector(value: Any) -> List[float] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        vector = [_finite_float(item) for item in value]
        if all(item is not None for item in vector):
            return [float(item) for item in vector if item is not None]
    return None


def _action_vectors_for_attempt(row: Mapping[str, Any]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for candidate in row.get("sc3_7d_delta_end_effector_actions") or []:
        vector = _numeric_vector(candidate)
        if vector:
            vectors.append(vector)
    if vectors:
        return vectors
    for action in row.get("actions") or []:
        vector = _numeric_vector(action)
        if vector:
            vectors.append(vector)
            continue
        if isinstance(action, Mapping):
            mapped = _action_vector_from_mapping(action)
            if mapped:
                vectors.append(mapped)
    return vectors


def _state_vector_for_attempt(row: Mapping[str, Any], fallback_width: int) -> tuple[List[float], bool]:
    for key in ("observation.state", "observation_state", "state", "robot_state"):
        vector = _numeric_vector(row.get(key))
        if vector:
            return vector, False
    for observation in row.get("observations") or []:
        if isinstance(observation, Mapping):
            for key in ("observation.state", "state", "robot_state"):
                vector = _numeric_vector(observation.get(key))
                if vector:
                    return vector, False
        else:
            vector = _numeric_vector(observation)
            if vector:
                return vector, False
    width = max(1, fallback_width)
    return [0.0] * width, True


def _first_observation_mapping(row: Mapping[str, Any]) -> Dict[str, Any]:
    for observation in row.get("observations") or []:
        if isinstance(observation, Mapping):
            return dict(observation)
    return {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _truthy_source_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "model_derived"}


def _strict_optional_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _strict_true(*values: Any) -> bool:
    return any(value is True for value in values)


def _rights_claim_allowed_metadata_bool(*values: Any) -> bool | None:
    present = False
    for value in values:
        if value in (None, "", []):
            continue
        present = True
        if value is True:
            return True
    if present:
        return False
    return None


def _clip_rights_metadata(clip: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not clip:
        return {}
    rights = _mapping(
        clip.get("rights_metadata")
        or clip.get("rights")
        or clip.get("license")
        or clip.get("privacy")
    )
    consent_scope = _string_list(
        clip.get("consent_scope")
        or clip.get("rights_scope")
        or rights.get("consent_scope")
        or rights.get("rights_scope")
    )
    metadata = {
        "metadata_source": _first_text(
            clip.get("metadata_source"),
            rights.get("metadata_source"),
        )
        or None,
        "license_status": _first_text(
            clip.get("license_status"),
            rights.get("license_status"),
            rights.get("status"),
        )
        or None,
        "consent_scope": consent_scope,
        "redaction_status": _first_text(
            clip.get("redaction_status"),
            rights.get("redaction_status"),
            rights.get("privacy_redaction_status"),
        )
        or None,
        "consent_revoked": _strict_optional_bool(
            clip.get("consent_revoked"),
            rights.get("consent_revoked"),
        ),
        "consent_revoked_at": _first_text(
            clip.get("consent_revoked_at"),
            rights.get("consent_revoked_at"),
        )
        or None,
        "delivery_blocked_by_consent_revocation": _strict_optional_bool(
            clip.get("delivery_blocked_by_consent_revocation"),
            rights.get("delivery_blocked_by_consent_revocation"),
        ),
        "signed_access_revoked_by_consent": _strict_optional_bool(
            clip.get("signed_access_revoked_by_consent"),
            rights.get("signed_access_revoked_by_consent"),
        ),
        "commercial_use_claim_allowed": _rights_claim_allowed_metadata_bool(
            clip.get("commercial_use_claim_allowed"),
            rights.get("commercial_use_claim_allowed"),
        ),
        "external_licensing_claim_allowed": _rights_claim_allowed_metadata_bool(
            clip.get("external_licensing_claim_allowed"),
            rights.get("external_licensing_claim_allowed"),
        ),
        "fallback_redaction_used": _explicit_true(
            clip.get("fallback_redaction_used"),
            rights.get("fallback_redaction_used"),
        ),
        "manual_rights_review_recommended": bool(
            _explicit_true(
                clip.get("manual_rights_review_recommended"),
                rights.get("manual_rights_review_recommended"),
            )
            or not consent_scope
        ),
    }
    return {key: value for key, value in metadata.items() if value not in (None, [], "")}


def _observation_source_for_attempt(
    row: Mapping[str, Any],
    clip: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    observation = _first_observation_mapping(row)
    source_text = _first_text(
        row.get("observation_source"),
        row.get("source_observation_type"),
        row.get("source_type"),
        observation.get("observation_source"),
        observation.get("source_observation_type"),
        observation.get("source_type"),
        clip.get("observation_source") if clip else None,
        clip.get("source_type") if clip else None,
    )
    model_derived = bool(
        _truthy_source_flag(row.get("model_derived"))
        or _truthy_source_flag(observation.get("model_derived"))
        or (clip is not None and _truthy_source_flag(clip.get("model_derived")))
        or source_text.lower() in {"generated", "model_derived", "synthetic"}
    )
    if not source_text:
        if clip and clip.get("materialized"):
            source_text = "materialized_capture_clip"
        elif row.get("observations"):
            source_text = "source_capture_reference"
        else:
            source_text = "not_available"
    frame_reference = _first_text(
        observation.get("frame"),
        observation.get("frame_id"),
        observation.get("frame_path"),
        row.get("source_frame_reference"),
    )
    return {
        "observation_source": source_text,
        "observation_source_detail": _first_text(
            observation.get("source_detail"),
            observation.get("uri"),
            observation.get("path"),
            clip.get("source_reference") if clip else None,
        )
        or None,
        "source_frame_reference": frame_reference or None,
        "observation_source_is_model_derived": model_derived,
        "observation_source_is_raw_capture_evidence": bool(
            not model_derived
            and source_text
            in {
                "materialized_capture_clip",
                "source_capture_reference",
                "raw_capture",
                "capture",
            }
        ),
        "source_clip_id": clip.get("clip_id") if clip else None,
        "source_materialized_video_path": clip.get("materialized_path") if clip else None,
        "source_rights_metadata": _clip_rights_metadata(clip),
    }


def _training_export_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    materialized_clips: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    clips_by_attempt: Dict[str, Mapping[str, Any]] = {
        str(clip.get("attempt_id")): clip
        for clip in materialized_clips
        if str(clip.get("attempt_id") or "").strip()
    }
    task_indexes: Dict[str, int] = {}
    tasks: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    state_width = 0
    action_width = 0
    synthesized_state_rows = 0
    measured_state_rows = 0
    synthesized_action_rows = 0
    measured_action_rows = 0
    model_derived_frame_rows = 0
    raw_capture_frame_rows = 0
    rights_metadata_frame_rows = 0
    global_index = 0
    for episode_index, row in enumerate(rows):
        attempt_id = row.get("attempt_id") or row.get("episode_id")
        task_id = str(row.get("task_id") or "task").strip() or "task"
        if task_id not in task_indexes:
            task_indexes[task_id] = len(task_indexes)
            tasks.append({"task_index": task_indexes[task_id], "task": task_id})
        actions = _action_vectors_for_attempt(row)
        action_synthesized = not actions
        if not actions:
            actions = [[1.0 if row.get("success") else 0.0]]
        state, state_synthesized = _state_vector_for_attempt(row, len(actions[0]))
        if state_synthesized:
            synthesized_state_rows += len(actions)
        else:
            measured_state_rows += len(actions)
        if action_synthesized:
            synthesized_action_rows += len(actions)
        else:
            measured_action_rows += len(actions)
        state_width = max(state_width, len(state))
        action_width = max(action_width, len(actions[0]))
        clip = clips_by_attempt.get(str(attempt_id or ""))
        video_path = clip.get("materialized_path") if clip else None
        source = _observation_source_for_attempt(row, clip)
        rights_metadata = _mapping(source.get("source_rights_metadata"))
        episode_start = global_index
        for frame_index, action in enumerate(actions):
            if source["observation_source_is_model_derived"]:
                model_derived_frame_rows += 1
            if source["observation_source_is_raw_capture_evidence"]:
                raw_capture_frame_rows += 1
            if rights_metadata:
                rights_metadata_frame_rows += 1
            frame_rows.append(
                {
                    "observation.state": state,
                    "action": action,
                    "timestamp": float(frame_index) / float(LEROBOT_V3_EXPORT_FPS),
                    "annotation.human.action.task_description": task_indexes[task_id],
                    "task_index": task_indexes[task_id],
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "index": global_index,
                    "next.reward": 1.0 if row.get("success") and frame_index == len(actions) - 1 else 0.0,
                    "next.done": frame_index == len(actions) - 1,
                    "attempt_id": attempt_id,
                    "policy_id": row.get("policy_id"),
                    "scenario_id": row.get("scenario_id"),
                    "video_path": video_path,
                    "state_synthesized_zero_fill": state_synthesized,
                    "action_synthesized_fallback": action_synthesized,
                    "observation_source": source["observation_source"],
                    "observation_source_detail": source["observation_source_detail"],
                    "source_frame_reference": source["source_frame_reference"],
                    "observation_source_is_model_derived": source[
                        "observation_source_is_model_derived"
                    ],
                    "observation_source_is_raw_capture_evidence": source[
                        "observation_source_is_raw_capture_evidence"
                    ],
                    "source_clip_id": source["source_clip_id"],
                    "source_materialized_video_path": source[
                        "source_materialized_video_path"
                    ],
                    "source_rights_metadata_json": json.dumps(
                        rights_metadata,
                        sort_keys=True,
                    ),
                }
            )
            global_index += 1
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task_indexes[task_id]],
                "length": len(actions),
                "start_index": episode_start,
                "end_index": global_index,
                "dataset_from_index": episode_start,
                "dataset_to_index": global_index,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "attempt_id": attempt_id,
                "clip_id": clip.get("clip_id") if clip else None,
                "video_path": video_path,
                "observation_source": source["observation_source"],
                "observation_source_is_model_derived": source[
                    "observation_source_is_model_derived"
                ],
                "source_rights_metadata": rights_metadata or {"metadata_present": False},
                "state_action_provenance": {
                    "episode_rows": len(actions),
                    "measured_state_rows": 0 if state_synthesized else len(actions),
                    "synthesized_state_rows": len(actions) if state_synthesized else 0,
                    "measured_action_rows": 0 if action_synthesized else len(actions),
                    "synthesized_action_rows": len(actions) if action_synthesized else 0,
                },
                "videos/observation.images.ego_view/chunk_index": 0,
                "videos/observation.images.ego_view/file_index": episode_index,
                "videos/observation.images.ego_view/from_timestamp": 0.0,
                "videos/observation.images.ego_view/to_timestamp": float(len(actions))
                / float(LEROBOT_V3_EXPORT_FPS),
            }
        )
    for frame in frame_rows:
        state_values = list(frame.get("observation.state") or [])
        action_values = list(frame.get("action") or [])
        frame["observation.state"] = [
            float(value) for value in state_values[:state_width]
        ] + [0.0] * max(0, state_width - len(state_values))
        frame["action"] = [float(value) for value in action_values[:action_width]] + [
            0.0
        ] * max(0, action_width - len(action_values))
    provenance_frame_rows = measured_state_rows + synthesized_state_rows
    return frame_rows, episodes, tasks, {
        "state_width": state_width,
        "action_width": action_width,
        "synthesized_state_rows": synthesized_state_rows,
        "measured_state_rows": measured_state_rows,
        "synthesized_action_rows": synthesized_action_rows,
        "measured_action_rows": measured_action_rows,
        "real_state_fraction": (
            measured_state_rows / provenance_frame_rows if provenance_frame_rows else 0.0
        ),
        "real_action_fraction": (
            measured_action_rows / provenance_frame_rows if provenance_frame_rows else 0.0
        ),
        "model_derived_frame_rows": model_derived_frame_rows,
        "raw_capture_frame_rows": raw_capture_frame_rows,
        "rights_metadata_frame_rows": rights_metadata_frame_rows,
    }


def _state_action_provenance_gate(
    shape: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    floor = _measured_state_fraction_floor()
    measured_state = int(shape.get("measured_state_rows") or 0)
    synthesized_state = int(shape.get("synthesized_state_rows") or 0)
    frame_rows = measured_state + synthesized_state
    real_state_fraction = float(shape.get("real_state_fraction") or 0.0)
    real_action_fraction = float(shape.get("real_action_fraction") or 0.0)
    state_floor_passed = bool(frame_rows) and real_state_fraction >= floor
    action_floor_passed = bool(frame_rows) and real_action_fraction >= floor
    blockers: List[str] = []
    if not state_floor_passed:
        blockers.append("insufficient_measured_state_fraction")
    if not action_floor_passed:
        blockers.append("insufficient_measured_action_fraction")
    return {
        "schema_version": "ptdp_state_action_provenance.v1",
        "frame_rows": frame_rows,
        "measured_state_rows": measured_state,
        "synthesized_state_rows": synthesized_state,
        "measured_action_rows": int(shape.get("measured_action_rows") or 0),
        "synthesized_action_rows": int(shape.get("synthesized_action_rows") or 0),
        "real_state_fraction": real_state_fraction,
        "real_action_fraction": real_action_fraction,
        "measured_state_fraction_floor": floor,
        "measured_state_fraction_floor_passed": state_floor_passed,
        "measured_action_fraction_floor_passed": action_floor_passed,
        "blockers": blockers,
        "per_episode": [
            {
                "episode_index": episode.get("episode_index"),
                "attempt_id": episode.get("attempt_id"),
                **_mapping(episode.get("state_action_provenance")),
            }
            for episode in episodes
        ],
    }


def _write_lerobot_v3_export(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    materialized_clips: Sequence[Mapping[str, Any]],
    generated_at: str,
    scene_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    root = output_dir / "exports" / "lerobot_v3"
    frame_rows, episodes, tasks, shape = _training_export_rows(
        rows=rows,
        materialized_clips=materialized_clips,
    )
    data_parquet = root / "data" / "chunk-000" / "file-000.parquet"
    native_data = _write_structured_parquet(data_parquet, frame_rows)
    if native_data:
        data_path = "data/chunk-000/file-000.parquet"
    else:
        data_fallback = root / "data" / "chunk-000" / "file-000.parquet.jsonl"
        _write_jsonl(data_fallback, frame_rows)
        data_path = "data/chunk-000/file-000.parquet.jsonl"

    episodes_parquet = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    native_episodes = _write_structured_parquet(episodes_parquet, episodes)
    if native_episodes:
        episodes_path = "meta/episodes/chunk-000/file-000.parquet"
    else:
        episodes_fallback = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet.jsonl"
        _write_jsonl(episodes_fallback, episodes)
        episodes_path = "meta/episodes/chunk-000/file-000.parquet.jsonl"

    tasks_parquet = root / "meta" / "tasks.parquet"
    native_tasks = _write_lerobot_tasks_parquet(tasks_parquet, tasks)
    tasks_path = "meta/tasks.parquet"
    if not native_tasks:
        tasks_fallback = root / "meta" / "tasks.parquet.jsonl"
        _write_jsonl(tasks_fallback, tasks)
        tasks_path = "meta/tasks.parquet.jsonl"
    _write_jsonl(root / "meta" / "tasks.jsonl", tasks)

    video_files: List[Dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode.get("episode_index") or 0)
        source_rel = str(episode.get("video_path") or "")
        source_path = output_dir / source_rel
        if not source_rel or not source_path.is_file():
            continue
        dest = (
            root
            / "videos"
            / "observation.images.ego_view"
            / "chunk-000"
            / f"file-{episode_index:03d}.mp4"
        )
        ensure_dir(dest.parent)
        shutil.copy2(source_path, dest)
        video_files.append(
            {
                "clip_id": episode.get("clip_id"),
                "path": _relative_to(root, dest),
                "sha256": _sha_file(dest),
                "episode_index": episode_index,
            }
        )
    all_episode_videos_materialized = bool(episodes) and len(video_files) == len(episodes)

    state_action_provenance = _state_action_provenance_gate(shape, episodes)
    stats = {
        "schema_version": "lerobot_v3_stats.v1",
        "frame_count": len(frame_rows),
        "episode_count": len(episodes),
        "task_count": len(tasks),
        "state_width": shape["state_width"],
        "action_width": shape["action_width"],
        "synthesized_state_rows": shape["synthesized_state_rows"],
        "measured_state_rows": shape["measured_state_rows"],
        "synthesized_action_rows": shape["synthesized_action_rows"],
        "measured_action_rows": shape["measured_action_rows"],
        "real_state_fraction": shape["real_state_fraction"],
        "real_action_fraction": shape["real_action_fraction"],
        "model_derived_frame_rows": shape["model_derived_frame_rows"],
        "raw_capture_frame_rows": shape["raw_capture_frame_rows"],
        "rights_metadata_frame_rows": shape["rights_metadata_frame_rows"],
    }
    info = {
        "schema_version": "lerobot_v3_info.v1",
        "source": "blueprint_post_training_data_package",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": generated_at,
        "codebase_version": "v3.0",
        "robot_type": "blueprint_capture",
        "total_episodes": len(episodes),
        "total_frames": len(frame_rows),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": LEROBOT_V3_EXPORT_FPS,
        "splits": {"train": f"0:{len(episodes)}"} if episodes else {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "data_path_template": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path_template": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [shape["state_width"]]},
            "action": {"dtype": "float32", "shape": [shape["action_width"]]},
            # LeRobot's loader normalizes every feature shape with tuple(...);
            # video features still need an iterable shape even when dimensions
            # are unknown at export time.
            "observation.images.ego_view": {"dtype": "video", "shape": [0, 0, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1],
            },
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "attempt_id": {"dtype": "string", "shape": [1]},
            "policy_id": {"dtype": "string", "shape": [1]},
            "scenario_id": {"dtype": "string", "shape": [1]},
            "video_path": {"dtype": "string", "shape": [1]},
            "state_synthesized_zero_fill": {"dtype": "bool", "shape": [1]},
            "action_synthesized_fallback": {"dtype": "bool", "shape": [1]},
            "observation_source": {"dtype": "string", "shape": [1]},
            "observation_source_detail": {"dtype": "string", "shape": [1]},
            "source_frame_reference": {"dtype": "string", "shape": [1]},
            "observation_source_is_model_derived": {"dtype": "bool", "shape": [1]},
            "observation_source_is_raw_capture_evidence": {
                "dtype": "bool",
                "shape": [1],
            },
            "source_clip_id": {"dtype": "string", "shape": [1]},
            "source_materialized_video_path": {"dtype": "string", "shape": [1]},
            "source_rights_metadata_json": {"dtype": "string", "shape": [1]},
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(root / "meta" / "info.json", info)
    write_json(root / "meta" / "stats.json", stats)
    native_parquet = native_data and native_episodes and native_tasks
    complete = bool(frame_rows and all_episode_videos_materialized and native_parquet)
    provenance_passed = not state_action_provenance["blockers"]
    manifest = {
        "schema_version": "blueprint_lerobot_v3_export_manifest.v1",
        "generated_at": generated_at,
        "status": "written_native" if complete and provenance_passed else "written_degraded",
        "data_path": data_path,
        "episodes_path": episodes_path,
        "tasks_path": tasks_path,
        "info_path": "meta/info.json",
        "stats_path": "meta/stats.json",
        "video_files": video_files,
        "episode_count": len(episodes),
        "frame_count": len(frame_rows),
        "native_parquet_written": bool(native_parquet),
        "materialized_video_count": len(video_files),
        "episode_video_count": len(video_files),
        "missing_video_episode_count": max(0, len(episodes) - len(video_files)),
        "all_episode_videos_materialized": all_episode_videos_materialized,
        "consumer_layout_complete": complete,
        "state_action_provenance": state_action_provenance,
        "blockers": [
            *(["lerobot_v3_native_parquet_not_written"] if not native_parquet else []),
            *(
                ["lerobot_v3_video_files_missing"]
                if not all_episode_videos_materialized
                else []
            ),
            *(["lerobot_v3_no_frame_rows"] if not frame_rows else []),
            *state_action_provenance["blockers"],
        ],
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "lerobot_layout_is_training_format_adapter": True,
            "synthesized_zero_state_rows": shape["synthesized_state_rows"],
            "synthesized_zero_state_is_not_robot_state_evidence": shape["synthesized_state_rows"] > 0,
            "real_state_fraction": state_action_provenance["real_state_fraction"],
            "real_action_fraction": state_action_provenance["real_action_fraction"],
            "measured_state_fraction_floor": state_action_provenance[
                "measured_state_fraction_floor"
            ],
            "measured_state_fraction_floor_passed": state_action_provenance[
                "measured_state_fraction_floor_passed"
            ],
            "observation_source_columns_written": True,
            "source_rights_metadata_columns_written": True,
        },
    }
    write_json(root / "lerobot_v3_export_manifest.json", manifest)
    return manifest


def _write_gr00t_lerobot_export(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    materialized_clips: Sequence[Mapping[str, Any]],
    generated_at: str,
    scene_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    root = output_dir / "exports" / "gr00t_lerobot"
    frame_rows, episodes, tasks, shape = _training_export_rows(
        rows=rows,
        materialized_clips=materialized_clips,
    )
    episode_data_files: List[Dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode.get("episode_index") or 0)
        start = int(episode.get("start_index") or 0)
        end = int(episode.get("end_index") or start)
        rows_for_episode = frame_rows[start:end]
        dest = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        native = _write_structured_parquet(dest, rows_for_episode)
        if native:
            path = _relative_to(root, dest)
        else:
            fallback = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet.jsonl"
            _write_jsonl(fallback, rows_for_episode)
            path = _relative_to(root, fallback)
        episode_data_files.append(
            {
                "episode_index": episode_index,
                "path": path,
                "native_parquet_written": native,
            }
        )

    video_files: List[Dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode.get("episode_index") or 0)
        source_rel = str(episode.get("video_path") or "")
        source_path = output_dir / source_rel
        if not source_rel or not source_path.is_file():
            continue
        dest = (
            root
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / f"episode_{episode_index:06d}.mp4"
        )
        ensure_dir(dest.parent)
        shutil.copy2(source_path, dest)
        video_files.append(
            {
                "clip_id": episode.get("clip_id"),
                "path": _relative_to(root, dest),
                "sha256": _sha_file(dest),
                "episode_index": episode_index,
            }
        )
    all_episode_videos_materialized = bool(episodes) and len(video_files) == len(episodes)

    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "tasks.jsonl", tasks)
    modality = {
        "state": {
            "blueprint_observation_state": {
                "start": 0,
                "end": shape["state_width"],
            }
        },
        "action": {
            "sc3_7d_delta_end_effector_action": {
                "start": 0,
                "end": shape["action_width"],
            }
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view",
            }
        },
        "annotation": {
            "human.action.task_description": {},
        },
        "metadata": {
            "observation_source": {
                "original_key": "observation_source",
            },
            "source_rights_metadata_json": {
                "original_key": "source_rights_metadata_json",
            },
        },
    }
    info = {
        "schema_version": "gr00t_lerobot_info.v1",
        "source": "blueprint_post_training_data_package",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": generated_at,
        "episode_count": len(episodes),
        "frame_count": len(frame_rows),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [shape["state_width"]]},
            "action": {"dtype": "float32", "shape": [shape["action_width"]]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
            "observation_source": {"dtype": "string", "shape": [1]},
            "observation_source_is_model_derived": {"dtype": "bool", "shape": [1]},
            "source_rights_metadata_json": {"dtype": "string", "shape": [1]},
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    state_action_provenance = _state_action_provenance_gate(shape, episodes)
    stats = {
        "schema_version": "gr00t_lerobot_stats.v1",
        "frame_count": len(frame_rows),
        "episode_count": len(episodes),
        "task_count": len(tasks),
        "state_width": shape["state_width"],
        "action_width": shape["action_width"],
        "synthesized_state_rows": shape["synthesized_state_rows"],
        "measured_state_rows": shape["measured_state_rows"],
        "synthesized_action_rows": shape["synthesized_action_rows"],
        "measured_action_rows": shape["measured_action_rows"],
        "real_state_fraction": shape["real_state_fraction"],
        "real_action_fraction": shape["real_action_fraction"],
        "model_derived_frame_rows": shape["model_derived_frame_rows"],
        "raw_capture_frame_rows": shape["raw_capture_frame_rows"],
        "rights_metadata_frame_rows": shape["rights_metadata_frame_rows"],
    }
    write_json(root / "meta" / "info.json", info)
    write_json(root / "meta" / "modality.json", modality)
    write_json(root / "meta" / "stats.json", stats)
    write_json(root / "meta" / "relative_stats.json", stats)
    native_parquet = bool(episode_data_files) and all(
        item.get("native_parquet_written") for item in episode_data_files
    )
    complete = bool(frame_rows and all_episode_videos_materialized and native_parquet)
    provenance_passed = not state_action_provenance["blockers"]
    manifest = {
        "schema_version": "blueprint_gr00t_lerobot_export_manifest.v1",
        "generated_at": generated_at,
        "status": "written_native" if complete and provenance_passed else "written_degraded",
        "meta_paths": {
            "info": "meta/info.json",
            "episodes": "meta/episodes.jsonl",
            "tasks": "meta/tasks.jsonl",
            "modality": "meta/modality.json",
            "stats": "meta/stats.json",
            "relative_stats": "meta/relative_stats.json",
        },
        "data_files": episode_data_files,
        "video_files": video_files,
        "episode_count": len(episodes),
        "frame_count": len(frame_rows),
        "native_parquet_written": native_parquet,
        "materialized_video_count": len(video_files),
        "episode_video_count": len(video_files),
        "missing_video_episode_count": max(0, len(episodes) - len(video_files)),
        "all_episode_videos_materialized": all_episode_videos_materialized,
        "consumer_layout_complete": complete,
        "state_action_provenance": state_action_provenance,
        "blockers": [
            *(["gr00t_lerobot_native_parquet_not_written"] if not native_parquet else []),
            *(
                ["gr00t_lerobot_video_files_missing"]
                if not all_episode_videos_materialized
                else []
            ),
            *(["gr00t_lerobot_no_frame_rows"] if not frame_rows else []),
            *state_action_provenance["blockers"],
        ],
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "gr00t_layout_is_training_format_adapter": True,
            "modality_json_written": True,
            "synthesized_zero_state_rows": shape["synthesized_state_rows"],
            "synthesized_zero_state_is_not_robot_state_evidence": shape["synthesized_state_rows"] > 0,
            "real_state_fraction": state_action_provenance["real_state_fraction"],
            "real_action_fraction": state_action_provenance["real_action_fraction"],
            "measured_state_fraction_floor": state_action_provenance[
                "measured_state_fraction_floor"
            ],
            "measured_state_fraction_floor_passed": state_action_provenance[
                "measured_state_fraction_floor_passed"
            ],
            "observation_source_columns_written": True,
            "source_rights_metadata_columns_written": True,
        },
    }
    write_json(root / "gr00t_lerobot_export_manifest.json", manifest)
    return manifest


def _write_optional_exports(
    *,
    output_dir: Path,
    attempts: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    clips: Mapping[str, Any],
    generated_at: str,
    scene_id: str,
    capture_id: str,
    clip_source_roots: Sequence[Path] = (),
    package_rights_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    rows = _rows_for_optional_exports(
        attempts=attempts,
        label_rows=label_rows,
        metrics=metrics,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    files: Dict[str, str] = {}
    formats = _optional_export_formats()

    rlds_path = output_dir / "exports" / "rlds" / "episodes.jsonl"
    _write_jsonl(rlds_path, rows)
    files["rlds_episodes"] = _relative_to(output_dir, rlds_path)
    formats["rlds"] = {
        **formats["rlds"],
        "status": "written_jsonl",
        "format_written": True,
        "path": files["rlds_episodes"],
        "episode_count": len(rows),
        "native_package_required": False,
    }

    lerobot_rows = [
        {
            "episode_index": row.get("episode_index"),
            "episode_id": row.get("episode_id"),
            "task": row.get("task_id"),
            "scenario": row.get("scenario_id"),
            "observation": row.get("observations") or [],
            "action": row.get("actions") or [],
            "sc3_7d_delta_end_effector_action": row.get("sc3_7d_delta_end_effector_actions") or [],
            "reward_or_success": 1.0 if row.get("success") else 0.0,
            "metadata": {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "policy_id": row.get("policy_id"),
                "failure_labels": row.get("failure_labels") or [],
                "claim_boundary": dict(CLAIM_BOUNDARY),
            },
        }
        for row in rows
    ]
    lerobot_path = output_dir / "exports" / "lerobot" / "episodes.jsonl"
    _write_jsonl(lerobot_path, lerobot_rows)
    files["lerobot_episodes"] = _relative_to(output_dir, lerobot_path)
    formats["lerobot"] = {
        **formats["lerobot"],
        "status": "written_jsonl",
        "format_written": True,
        "path": files["lerobot_episodes"],
        "episode_count": len(lerobot_rows),
        "native_package_required": False,
    }

    hdf5_path = output_dir / "exports" / "hdf5" / "episodes.hdf5"
    if _write_native_hdf5(hdf5_path, rows):
        files["hdf5_episodes"] = _relative_to(output_dir, hdf5_path)
        formats["hdf5"] = {
            **formats["hdf5"],
            "status": "written_native",
            "format_written": True,
            "path": files["hdf5_episodes"],
            "episode_count": len(rows),
        }
    else:
        hdf5_fallback = output_dir / "exports" / "hdf5" / "episodes.hdf5.jsonl"
        _write_jsonl(hdf5_fallback, rows)
        files["hdf5_episodes"] = _relative_to(output_dir, hdf5_fallback)
        formats["hdf5"] = {
            **formats["hdf5"],
            "status": "written_jsonl_fallback",
            "format_written": True,
            "path": files["hdf5_episodes"],
            "episode_count": len(rows),
            "fallback_reason": "optional_dependency_h5py_missing",
        }

    parquet_path = output_dir / "exports" / "parquet" / "episodes.parquet"
    if _write_native_parquet(parquet_path, rows):
        files["parquet_episodes"] = _relative_to(output_dir, parquet_path)
        formats["parquet"] = {
            **formats["parquet"],
            "status": "written_native",
            "format_written": True,
            "path": files["parquet_episodes"],
            "episode_count": len(rows),
        }
    else:
        parquet_fallback = output_dir / "exports" / "parquet" / "episodes.parquet.jsonl"
        _write_jsonl(parquet_fallback, _flat_export_rows(rows))
        files["parquet_episodes"] = _relative_to(output_dir, parquet_fallback)
        formats["parquet"] = {
            **formats["parquet"],
            "status": "written_jsonl_fallback",
            "format_written": True,
            "path": files["parquet_episodes"],
            "episode_count": len(rows),
            "fallback_reason": "optional_dependency_pyarrow_or_pandas_missing",
        }

    video_bundle = _materialize_video_bundle(
        output_dir=output_dir,
        clips=clips,
        generated_at=generated_at,
        source_roots=clip_source_roots,
        package_rights_metadata=package_rights_metadata,
    )
    video_bundle_manifest = _mapping(video_bundle.get("manifest"))
    files["video_bundle_manifest"] = str(video_bundle.get("path") or "")
    formats["video_bundle"] = {
        **formats["video_bundle"],
        "status": video_bundle_manifest.get("status"),
        "format_written": True,
        "path": files["video_bundle_manifest"],
        "clip_count": int(video_bundle_manifest.get("clip_count") or 0),
        "materialized_clip_count": int(video_bundle_manifest.get("materialized_clip_count") or 0),
        "missing_clip_file_count": int(video_bundle_manifest.get("missing_clip_file_count") or 0),
        "all_declared_clips_materialized": bool(
            video_bundle_manifest.get("all_declared_clips_materialized")
        ),
    }
    materialized_clips = [
        dict(item)
        for item in video_bundle.get("materialized_clips") or []
        if isinstance(item, Mapping)
    ]
    lerobot_v3 = _write_lerobot_v3_export(
        output_dir=output_dir,
        rows=rows,
        materialized_clips=materialized_clips,
        generated_at=generated_at,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    files["lerobot_v3_manifest"] = "exports/lerobot_v3/lerobot_v3_export_manifest.json"
    formats["lerobot_v3"] = {
        **formats["lerobot_v3"],
        "status": lerobot_v3.get("status"),
        "format_written": True,
        "path": files["lerobot_v3_manifest"],
        "episode_count": lerobot_v3.get("episode_count"),
        "frame_count": lerobot_v3.get("frame_count"),
        "native_parquet_written": bool(lerobot_v3.get("native_parquet_written")),
        "materialized_video_count": int(lerobot_v3.get("materialized_video_count") or 0),
        "missing_video_episode_count": int(
            lerobot_v3.get("missing_video_episode_count") or 0
        ),
        "all_episode_videos_materialized": bool(
            lerobot_v3.get("all_episode_videos_materialized")
        ),
        "consumer_layout_complete": bool(lerobot_v3.get("consumer_layout_complete")),
        "state_action_provenance": {
            key: value
            for key, value in _mapping(lerobot_v3.get("state_action_provenance")).items()
            if key != "per_episode"
        },
        "blockers": _string_list(lerobot_v3.get("blockers")),
    }
    gr00t_lerobot = _write_gr00t_lerobot_export(
        output_dir=output_dir,
        rows=rows,
        materialized_clips=materialized_clips,
        generated_at=generated_at,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    files["gr00t_lerobot_manifest"] = "exports/gr00t_lerobot/gr00t_lerobot_export_manifest.json"
    files["gr00t_modality_json"] = "exports/gr00t_lerobot/meta/modality.json"
    formats["gr00t_lerobot"] = {
        **formats["gr00t_lerobot"],
        "status": gr00t_lerobot.get("status"),
        "format_written": True,
        "path": files["gr00t_lerobot_manifest"],
        "modality_json_path": files["gr00t_modality_json"],
        "episode_count": gr00t_lerobot.get("episode_count"),
        "frame_count": gr00t_lerobot.get("frame_count"),
        "native_parquet_written": bool(gr00t_lerobot.get("native_parquet_written")),
        "materialized_video_count": int(gr00t_lerobot.get("materialized_video_count") or 0),
        "missing_video_episode_count": int(
            gr00t_lerobot.get("missing_video_episode_count") or 0
        ),
        "all_episode_videos_materialized": bool(
            gr00t_lerobot.get("all_episode_videos_materialized")
        ),
        "consumer_layout_complete": bool(gr00t_lerobot.get("consumer_layout_complete")),
        "state_action_provenance": {
            key: value
            for key, value in _mapping(gr00t_lerobot.get("state_action_provenance")).items()
            if key != "per_episode"
        },
        "blockers": _string_list(gr00t_lerobot.get("blockers")),
    }

    # Round-trip validation: open each lerobot-format export the way a buyer
    # would (real lerobot loader when installed, spec-faithful hermetic reader
    # otherwise). The verdict gates the buyer readout's export-integrity
    # section; a package that cannot be loaded back must never read "ready".
    for format_name in ("lerobot_v3", "gr00t_lerobot"):
        export_root = output_dir / "exports" / format_name
        report = validate_lerobot_export(export_root, generated_at=generated_at)
        report_relpath = f"exports/{format_name}/round_trip_validation_report.json"
        write_json(output_dir / report_relpath, report)
        files[f"{format_name}_round_trip_validation_report"] = report_relpath
        formats[format_name]["round_trip_validation"] = round_trip_validation_summary(
            report, path=report_relpath
        )

    return {
        "schema_version": "post_training_data_package_optional_exports.v1",
        "generated_at": generated_at,
        "formats": formats,
        "files": files,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _clip_curation_summary(capture_root: Path) -> Dict[str, Any]:
    """Summarize clip curation + semantic dedup state for the package.

    Absence of the manifests is an explicit QA state, never silently green;
    coverage counts, when present, are post-dedup.
    """
    curation = _read_optional_mapping(
        capture_root / "derived" / "clip_curation" / "clip_curation_manifest.json"
    )
    rejections = _read_optional_mapping(
        capture_root / "derived" / "clip_curation" / "clip_rejection_manifest.json"
    )
    dedup = _read_optional_mapping(
        capture_root / "derived" / "semantic_dedup" / "semantic_dedup_manifest.json"
    )
    dedup_coverage = _mapping(dedup.get("coverage"))
    return {
        "curation_status": "run" if curation else "not_run",
        "dedup_status": "run" if dedup else "not_run",
        "accepted_clip_count": curation.get("accepted_clip_count"),
        "rejected_clip_count": (
            rejections.get("rejected_count")
            if rejections
            else curation.get("rejected_clip_count")
        ),
        "post_dedup_clip_count": dedup_coverage.get("kept_clip_count"),
        "dedup_dropped_clip_count": dedup_coverage.get("dropped_clip_count"),
        "embedding_provider": _mapping(dedup.get("embedding_provider")) or None,
        "qa_note": (
            "clip curation and semantic dedup manifests included"
            if curation and dedup
            else "clip curation/dedup not run for this bundle; coverage counts are uncurated"
        ),
    }


def _replay_review_instructions_markdown(*, scene_id: str, capture_id: str) -> str:
    return "\n".join(
        [
            "# Replay & Review Instructions",
            "",
            f"- Scene: {scene_id}",
            f"- Capture: {capture_id}",
            "",
            "## Verify integrity",
            "",
            "1. Read `package_index.json` for the full file inventory.",
            "2. Recompute SHA256 for every file and compare against `checksums.json`.",
            "",
            "## Review evidence",
            "",
            "3. Start with `dataset_card.json` for counts and the proof boundary.",
            "4. Read `data/attempts.jsonl` (one attempt per line) alongside",
            "   `data/failure_labels.jsonl`; failures are preserved, not filtered.",
            "5. Cross-check curation decisions in `curation_report.json` and",
            "   `semantic_dedup_report.json` before training on any clip.",
            "",
            "## Replay",
            "",
            "6. Replay attempts against the scenario definitions referenced by the",
            "   export manifest's `included_artifacts` (scenario_eval_matrix, task",
            "   cards). Attempt rows carry scenario/run ids for alignment.",
            "",
            "## Claim boundary",
            "",
            "- Nothing in this package is deployment approval, physical-robot proof,",
            "  or safety validation. Generated/model-derived media is labeled as such",
            "  and is never raw capture evidence.",
            "",
        ]
    )


def _write_package_files(
    *,
    output_dir: Path,
    included_artifacts: Mapping[str, str],
    trace: Mapping[str, Any],
    labels: Mapping[str, Any],
    metrics: Mapping[str, Any],
    clips: Mapping[str, Any],
    curation_report: Mapping[str, Any],
    semantic_dedup_report: Mapping[str, Any],
    sc3_action_report: Mapping[str, Any],
    generated_at: str,
    scene_id: str,
    capture_id: str,
    visual_augmentation_packet: Mapping[str, Any] | None = None,
    scaniverse_import: Mapping[str, Any] | None = None,
    rl_post_training_handoff: Mapping[str, Any] | None = None,
    clip_curation: Mapping[str, Any] | None = None,
    clip_source_roots: Sequence[Path] = (),
) -> Dict[str, Any]:
    data_dir = output_dir / "data"
    attempts = _rows(trace, "attempts")
    label_rows = _rows(labels, "labels")
    _write_jsonl(data_dir / "attempts.jsonl", attempts)
    _write_jsonl(data_dir / "failure_labels.jsonl", label_rows)
    write_json(
        data_dir / "metrics.json",
        dict(metrics)
        if metrics
        else {
            "schema_version": "post_training_package_metrics.v1",
            "generated_at": generated_at,
            "status": "missing_source_metrics",
            "attempt_count": len(attempts),
            "failure_count": len(label_rows),
        },
    )
    write_json(
        output_dir / "clips_manifest.json",
        dict(clips)
        if clips
        else {
            "schema_version": "post_training_package_clips_manifest.v1",
            "generated_at": generated_at,
            "status": "missing_source_clips",
            "clip_count": 0,
            "clips": [],
        },
    )
    write_json(output_dir / "curation_report.json", dict(curation_report))
    write_json(output_dir / "semantic_dedup_report.json", dict(semantic_dedup_report))
    write_json(output_dir / "sc3_action_normalization_report.json", dict(sc3_action_report))
    dataset_card = {
        "schema_version": "post_training_data_package_dataset_card.v1",
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "dataset_type": "real_site_robot_eval_post_training_package",
        "attempt_count": len(attempts),
        "failure_label_count": len(label_rows),
        "curated_clip_count": int(curation_report.get("accepted_clip_count") or 0),
        "semantic_dedup_status": semantic_dedup_report.get("status"),
        "sc3_action_contract_status": sc3_action_report.get("status"),
        # Upstream pipeline-stage curation/dedup (pre-export gating), distinct
        # from the in-export curation_report/semantic_dedup_report QA above.
        "clip_curation": dict(clip_curation or {"curation_status": "not_run", "dedup_status": "not_run"}),
        "source_artifacts": dict(included_artifacts),
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }
    consent_evidence_record = _read_optional_mapping(output_dir / "consent_evidence.json")
    revocation_takedown = _mapping(
        consent_evidence_record.get("revocation_takedown")
    ) or _read_optional_mapping(output_dir / "revocation_takedown_manifest.json")
    revocation_required = revocation_takedown.get("status") == "takedown_required"
    revenue_share_review = _mapping(
        consent_evidence_record.get("revenue_share_review")
    ) or _revenue_review_from_rights(
        generated_at=generated_at,
        rights_packet={},
        consent_source={},
    )
    data_processing_terms_review = _mapping(
        consent_evidence_record.get("data_processing_terms_review")
    ) or _data_processing_terms_review(
        generated_at=generated_at,
        rights_packet={},
        consent_source={},
    )
    package_rights_metadata = _package_clip_rights_metadata(
        consent_evidence=consent_evidence_record,
        revocation_takedown=revocation_takedown,
    )
    license_manifest = {
        "schema_version": "post_training_data_package_license_manifest.v1",
        "generated_at": generated_at,
        "status": "blocked_revocation_takedown_required"
        if revocation_required
        else "review_required",
        "rights_privacy_review_required": True,
        "commercial_use_requires_package_scope_clearance": True,
        "revocation_takedown": revocation_takedown
        or {
            "schema_version": "post_training_revocation_takedown_manifest.v1",
            "status": "not_available",
            "consent_revoked": False,
        },
        "revenue_share_review": revenue_share_review,
        "data_processing_terms_review": data_processing_terms_review,
        "included_artifacts": dict(included_artifacts),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "revenue_share_review.json", revenue_share_review)
    write_json(output_dir / "data_processing_terms_review.json", data_processing_terms_review)
    optional_exports = _write_optional_exports(
        output_dir=output_dir,
        attempts=attempts,
        label_rows=label_rows,
        metrics=metrics,
        clips=clips,
        generated_at=generated_at,
        scene_id=scene_id,
        capture_id=capture_id,
        clip_source_roots=clip_source_roots,
        package_rights_metadata=package_rights_metadata,
    )
    visual_augmentation_support: Dict[str, Any] | None = None
    if visual_augmentation_packet:
        packet_boundary = _mapping(visual_augmentation_packet.get("claim_boundary"))
        visual_augmentation_support = {
            "schema_version": "post_training_visual_augmentation_support_manifest.v1",
            "generated_at": generated_at,
            "status": "included_model_derived_support_packet",
            "source_packet_manifest": included_artifacts.get(
                "oscar_visual_augmentation_packet_manifest"
            ),
            "source_packet_status": visual_augmentation_packet.get("status"),
            "source_packet_type": visual_augmentation_packet.get("packet_type"),
            "variant_count": int(visual_augmentation_packet.get("variant_count") or 0),
            "generated_video_count": int(
                visual_augmentation_packet.get("generated_video_count") or 0
            ),
            "selected_backend_id": visual_augmentation_packet.get("selected_backend_id"),
            "requires_human_or_vlm_review_before_training_use": True,
            "generated_videos_model_derived": True,
            "raw_capture_evidence": False,
            "physical_robot_episode_evidence": False,
            "claim_boundary": {
                **packet_boundary,
                "artifact_purpose": "post_training_visual_augmentation_support",
                "included_in_post_training_data_package": True,
                "generated_videos_are_model_derived_support_assets": True,
                "generated_videos_are_raw_capture_evidence": False,
                "contact_physics_proven": False,
                "real_robot_readiness_proven": False,
                "deployment_safety_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        }
        write_json(
            output_dir / "visual_augmentation_support_manifest.json",
            visual_augmentation_support,
        )
    scaniverse_payload = _mapping(scaniverse_import)
    scaniverse_support: Dict[str, Any] | None = None
    if scaniverse_payload:
        scaniverse_boundary = _mapping(scaniverse_payload.get("claim_boundary"))
        scaniverse_support = {
            "schema_version": "post_training_scaniverse_support_manifest.v1",
            "generated_at": generated_at,
            "status": "included_external_derived_support_packet",
            "source_import_manifest": included_artifacts.get("scaniverse_import_manifest"),
            "source_proof_boundary": included_artifacts.get(
                "scaniverse_import_proof_boundary"
            ),
            "source_import_status": scaniverse_payload.get("status"),
            "asset_count": int(scaniverse_payload.get("asset_count") or 0),
            "asset_roles": sorted(
                {
                    str(_mapping(asset).get("asset_role") or "").strip()
                    for asset in scaniverse_payload.get("assets") or []
                    if str(_mapping(asset).get("asset_role") or "").strip()
                }
            ),
            "external_derived_support_asset": True,
            "raw_capture_evidence": False,
            "physical_robot_episode_evidence": False,
            "buyer_review_label": (
                "Scaniverse-derived support assets; raw Blueprint capture "
                "evidence remains authoritative."
            ),
            "isaac_handoff_candidacy": _mapping(
                scaniverse_payload.get("isaac_handoff_candidacy")
            ),
            "claim_boundary": {
                **scaniverse_boundary,
                "artifact_purpose": "post_training_scaniverse_support",
                "included_in_post_training_data_package": True,
                "external_derived_support_asset": True,
                "scaniverse_assets_are_raw_capture_evidence": False,
                "scaniverse_assets_are_task_success_evidence": False,
                "scaniverse_assets_are_physics_contact_evidence": False,
                "isaac_sim_execution_proven": False,
                "physics_contact_validated": False,
                "robot_policy_execution_proven": False,
                "deployment_readiness_proven": False,
                "public_claim_upgrade_allowed": False,
            },
        }
        write_json(
            output_dir / "scaniverse_support_asset_manifest.json",
            scaniverse_support,
        )
    rl_handoff_payload = dict(rl_post_training_handoff or {})
    if rl_handoff_payload:
        write_json(output_dir / "rl_post_training_handoff_packet.json", rl_handoff_payload)
    (output_dir / "replay_review_instructions.md").write_text(
        _replay_review_instructions_markdown(scene_id=scene_id, capture_id=capture_id),
        encoding="utf-8",
    )
    write_json(output_dir / "dataset_card.json", dataset_card)
    write_json(output_dir / "license_manifest.json", license_manifest)
    write_json(output_dir / "optional_export_manifest.json", optional_exports)
    package_file_index = {
        "attempts_jsonl": "data/attempts.jsonl",
        "failure_labels_jsonl": "data/failure_labels.jsonl",
        "metrics_json": "data/metrics.json",
        "clips_manifest": "clips_manifest.json",
        "curation_report": "curation_report.json",
        "semantic_dedup_report": "semantic_dedup_report.json",
        "sc3_action_normalization_report": "sc3_action_normalization_report.json",
        "dataset_card": "dataset_card.json",
        "license_manifest": "license_manifest.json",
        "optional_export_manifest": "optional_export_manifest.json",
        "replay_review_instructions": "replay_review_instructions.md",
        **dict(optional_exports.get("files") or {}),
    }
    if visual_augmentation_support:
        package_file_index["visual_augmentation_support_manifest"] = (
            "visual_augmentation_support_manifest.json"
        )
    if scaniverse_support:
        package_file_index["scaniverse_support_asset_manifest"] = (
            "scaniverse_support_asset_manifest.json"
        )
    if rl_handoff_payload:
        package_file_index["rl_post_training_handoff_packet"] = (
            "rl_post_training_handoff_packet.json"
        )
    if (output_dir / "consent_evidence.json").is_file():
        package_file_index["consent_evidence"] = "consent_evidence.json"
    if (output_dir / "revocation_takedown_manifest.json").is_file():
        package_file_index["revocation_takedown_manifest"] = (
            "revocation_takedown_manifest.json"
        )
    if (output_dir / "downstream_takedown_execution_ledger.json").is_file():
        package_file_index["downstream_takedown_execution_ledger"] = (
            "downstream_takedown_execution_ledger.json"
        )
    if (output_dir / "webapp_rights_privacy_takedown_notice.json").is_file():
        package_file_index["webapp_rights_privacy_takedown_notice"] = (
            "webapp_rights_privacy_takedown_notice.json"
        )
    if (output_dir / "hosted_session_takedown_request.json").is_file():
        package_file_index["hosted_session_takedown_request"] = (
            "hosted_session_takedown_request.json"
        )
    if (output_dir / "revenue_share_review.json").is_file():
        package_file_index["revenue_share_review"] = "revenue_share_review.json"
    if (output_dir / "data_processing_terms_review.json").is_file():
        package_file_index["data_processing_terms_review"] = (
            "data_processing_terms_review.json"
        )
    if (output_dir / "success_claim_ledger.json").is_file():
        package_file_index["success_claim_ledger"] = "success_claim_ledger.json"
    existing_export_paths = set(package_file_index.values())
    exports_dir = output_dir / "exports"
    if exports_dir.is_dir():
        for index, export_file in enumerate(sorted(exports_dir.rglob("*"))):
            if not export_file.is_file():
                continue
            relative = _relative_to(output_dir, export_file)
            if relative in existing_export_paths:
                continue
            key = f"export_file_{index:04d}_{_safe_path_component(export_file.stem, 'artifact')}"
            package_file_index[key] = relative
            existing_export_paths.add(relative)
    package_index = {
        "schema_version": "post_training_data_package_index.v1",
        "generated_at": generated_at,
        "status": "revoked_consent_takedown_required"
        if revocation_required
        else "created",
        "local_package_access_revoked": bool(revocation_required),
        "delivery_blocked_by_consent_revocation": bool(revocation_required),
        "signed_access_revoked_by_consent": bool(revocation_required),
        "revocation_takedown": revocation_takedown
        or {
            "schema_version": REVOCATION_TAKEDOWN_MANIFEST_SCHEMA_VERSION,
            "status": "not_available",
            "consent_revoked": False,
        },
        "files": package_file_index,
        "source_artifacts": dict(included_artifacts),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "package_index.json", package_index)
    package_files = {
        key: _artifact(output_dir, output_dir / path)
        for key, path in package_index["files"].items()
    }
    checksums = {
        "schema_version": "post_training_data_package_checksums.v1",
        "generated_at": generated_at,
        "files": package_files,
    }
    write_json(output_dir / "checksums.json", checksums)
    return {
        "dataset_card": dataset_card,
        "license_manifest": license_manifest,
        "visual_augmentation_support": visual_augmentation_support,
        "scaniverse_support": scaniverse_support,
        "rl_post_training_handoff": rl_handoff_payload,
        "package_index": package_index,
        "checksums": checksums,
        "package_files": package_files,
        "optional_exports": optional_exports,
    }


def _write_archive(output_dir: Path, generated_at: str) -> Dict[str, Any]:
    archive_dir = output_dir / "archives"
    ensure_dir(archive_dir)
    archive_path = archive_dir / "post_training_data_package.tar.gz"
    revocation_takedown = _read_optional_mapping(
        output_dir / "revocation_takedown_manifest.json"
    )
    revocation_required = revocation_takedown.get("status") == "takedown_required"
    archive_inputs = [
        output_dir / "data" / "attempts.jsonl",
        output_dir / "data" / "failure_labels.jsonl",
        output_dir / "data" / "metrics.json",
        output_dir / "clips_manifest.json",
        output_dir / "curation_report.json",
        output_dir / "semantic_dedup_report.json",
        output_dir / "sc3_action_normalization_report.json",
        output_dir / "customer_handoff_report.json",
        output_dir / "customer_handoff_report.md",
        output_dir / "delivery_manifest.json",
        output_dir / "signed_access_manifest.json",
        output_dir / "dataset_card.json",
        output_dir / "license_manifest.json",
        output_dir / "optional_export_manifest.json",
        output_dir / "replay_review_instructions.md",
        output_dir / "consent_evidence.json",
        output_dir / "revocation_takedown_manifest.json",
        output_dir / "downstream_takedown_execution_ledger.json",
        output_dir / "webapp_rights_privacy_takedown_notice.json",
        output_dir / "hosted_session_takedown_request.json",
        output_dir / "revenue_share_review.json",
        output_dir / "data_processing_terms_review.json",
        output_dir / "success_claim_ledger.json",
        output_dir / "visual_augmentation_support_manifest.json",
        output_dir / "scaniverse_support_asset_manifest.json",
        output_dir / "rl_post_training_handoff_packet.json",
        output_dir / "package_index.json",
        output_dir / "checksums.json",
    ]
    exports_dir = output_dir / "exports"
    if exports_dir.is_dir():
        archive_inputs.extend(
            path for path in sorted(exports_dir.rglob("*")) if path.is_file()
        )
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in archive_inputs:
            if path.is_file():
                tar.add(path, arcname=_relative_to(output_dir, path))
    archive_manifest = {
        "schema_version": "post_training_data_package_archive_manifest.v1",
        "generated_at": generated_at,
        "status": "created_revoked_consent_takedown_required"
        if revocation_required
        else "created",
        "local_package_access_revoked": bool(revocation_required),
        "delivery_blocked_by_consent_revocation": bool(revocation_required),
        "signed_access_revoked_by_consent": bool(revocation_required),
        "revocation_takedown": revocation_takedown
        or {
            "schema_version": REVOCATION_TAKEDOWN_MANIFEST_SCHEMA_VERSION,
            "status": "not_available",
            "consent_revoked": False,
        },
        "archive": _artifact(output_dir, archive_path),
        "included_files": [
            _relative_to(output_dir, path) for path in archive_inputs if path.is_file()
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "archive_manifest.json", archive_manifest)
    return archive_manifest


def build_post_training_data_package_export(
    *,
    capture_root: str | Path,
    job_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    resolved_job_dir = Path(job_dir).resolve() if job_dir else None
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else resolved_job_dir
        if resolved_job_dir
        else pipeline_dir / "post_training_data_package"
    )
    ensure_dir(resolved_output_dir)
    generated_at = utc_now_iso()

    included_artifacts: Dict[str, str] = {}
    if resolved_job_dir:
        for key, name in (
            ("job_request", "job_request.json"),
            ("normalized_attempt_trace", "normalized_attempt_trace.json"),
            ("failure_labels", "failure_labels.json"),
            ("policy_package_manifest", "policy_package_manifest.json"),
            ("task_eval_rl_post_training_handoff_packet", "rl_post_training_handoff_packet.json"),
            ("visual_review_ledger", "visual_review_ledger.json"),
            ("arena_eval_metrics", "arena_eval_metrics.json"),
            ("simulator_provider_adapter_manifest", "simulator_provider_adapter_manifest.json"),
            ("simulator_command_artifacts_manifest", "simulator_command_artifacts_manifest.json"),
            (
                "remote_cloud_execution_closure_manifest",
                "remote_cloud_execution_closure_manifest.json",
            ),
            (
                "robot_team_grade_eval_closure_manifest",
                "robot_team_grade_eval_closure_manifest.json",
            ),
            (
                "simulator_command_batch_trace_package_manifest",
                "simulator_command_batch_trace_package_manifest.json",
            ),
            (
                "simulator_command_batch_attempt_trace",
                "simulator_command_batch_attempt_trace.jsonl",
            ),
            (
                "simulator_command_batch_contact_stream",
                "simulator_command_batch_contact_stream.jsonl",
            ),
            (
                "simulator_command_batch_planner_state",
                "simulator_command_batch_planner_state.jsonl",
            ),
            (
                "simulator_command_batch_control_stream",
                "simulator_command_batch_control_stream.jsonl",
            ),
            (
                "simulator_command_batch_metrics",
                "simulator_command_batch_metrics.json",
            ),
            (
                "simulator_command_batch_failure_labels",
                "simulator_command_batch_failure_labels.json",
            ),
            (
                "simulator_command_batch_visual_media_coverage",
                "simulator_command_batch_visual_media_coverage.json",
            ),
            (
                "simulator_command_batch_visual_review_ledger",
                "simulator_command_batch_visual_review_ledger.json",
            ),
            (
                "simulator_command_digital_twin_fidelity_qa",
                "simulator_command_digital_twin_fidelity_qa.json",
            ),
            (
                "simulator_command_batch_artifact_checksums",
                "simulator_command_batch_artifact_checksums.json",
            ),
            (
                "simulator_command_batch_closure_manifest",
                "simulator_command_batch_closure_manifest.json",
            ),
            (
                "webapp_robot_eval_status_projection",
                "webapp_robot_eval_status_projection.json",
            ),
            ("scenario_eval_matrix", "scenario_eval_matrix.json"),
            ("robot_pov_observation_manifest", "robot_pov_observation_manifest.json"),
            ("robot_pov_observation_candidate_set", "robot_pov_observation_candidate_set.json"),
            ("selected_initial_policy_observation", "selected_initial_policy_observation.json"),
            ("robot_pov_observations", "robot_pov_observations.jsonl"),
            ("robot_pov_frame_sequence_manifest", "robot_pov_frame_sequence_manifest.json"),
            ("robot_pov_render_storyboard", "robot_pov_render_storyboard.json"),
            ("policy_execution_manifest", "policy_execution_manifest.json"),
            ("policy_execution_trace", "policy_execution_trace.json"),
            ("task_eval_run_report", "task_eval_run_report.json"),
            ("success_claim_ledger", "success_claim_ledger.json"),
            ("intervention_safety_ledger", "intervention_safety_ledger.json"),
            ("safety_events_ledger", "safety_events_ledger.json"),
            ("clips_manifest", "clips_manifest.json"),
            ("accepted_failure_labels", "accepted_failure_labels.json"),
            ("review_resolution_ledger", "review_resolution_ledger.json"),
            ("customer_handoff_report", "customer_handoff_report.json"),
            ("delivery_manifest", "delivery_manifest.json"),
            ("signed_access_manifest", "signed_access_manifest.json"),
            ("live_operator_ledger", "live_operator_ledger.json"),
            ("arena_rerun_plan", "arena_rerun_plan.json"),
            ("policy_adapter_manifest", "policy_adapter_manifest.json"),
            ("arena_result_ingest_ledger", "arena_result_ingest_ledger.json"),
            ("prediction_outcome_ledger", "prediction_outcome_ledger.json"),
            ("calibration_report", "calibration_report.json"),
            ("deployment_outcome_intake_manifest", "deployment_outcome_intake_manifest.json"),
            ("deployment_outcome_ledger", "deployment_outcome_ledger.json"),
            ("sim_vs_real_calibration_report", "sim_vs_real_calibration_report.json"),
            (
                "prediction_vs_actual_deployment_summary",
                "prediction_vs_actual_deployment_summary.json",
            ),
            (
                "real_world_validation_followup_plan",
                "real_world_validation_followup_plan.json",
            ),
            (
                "real_world_validation_followup_request_queue",
                "real_world_validation_followup_request_queue.json",
            ),
            ("live_eval_closure_manifest", "live_eval_closure_manifest.json"),
            ("live_eval_closure_evidence", "live_eval_closure_evidence.json"),
            ("breakage_library", "breakage_library.json"),
            ("evaluation_result", "evaluation_result.json"),
            ("robot_eval_report", "robot_eval_report.json"),
            ("robot_eval_report_markdown", "robot_eval_report.md"),
            ("proof_boundary", "proof_boundary.json"),
            (
                "policy_improvement_rl_post_training_handoff_packet",
                "policy_improvement_run/rl_post_training_handoff_packet.json",
            ),
            (
                "policy_autoresearch_report",
                "policy_autoresearch/policy_autoresearch_report.json",
            ),
            (
                "policy_candidate_package",
                "policy_autoresearch/policy_candidate_package.json",
            ),
            ("heldout_eval_result", "policy_autoresearch/heldout_eval_result.json"),
            (
                "oscar_visual_augmentation_packet_manifest",
                "oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json",
            ),
            (
                "oscar_visual_augmentation_variant_requests",
                "oscar_visual_augmentation_packet/visual_augmentation_variant_requests.jsonl",
            ),
            (
                "oscar_visual_augmentation_backend_registry",
                "oscar_visual_augmentation_packet/model_backend_registry.json",
            ),
            (
                "oscar_visual_distribution_shift_eval_protocol",
                "oscar_visual_augmentation_packet/visual_distribution_shift_eval_protocol.json",
            ),
            (
                "oscar_visual_augmentation_claim_boundary",
                "oscar_visual_augmentation_packet/claim_boundary.json",
            ),
            (
                "oscar_visual_augmentation_generation_run_manifest",
                "oscar_visual_augmentation_packet/visual_augmentation_generation_run_manifest.json",
            ),
            (
                "oscar_visual_augmentation_generation_results",
                "oscar_visual_augmentation_packet/visual_augmentation_generation_results.jsonl",
            ),
            (
                "oscar_visual_augmentation_generation_qa",
                "oscar_visual_augmentation_packet/visual_augmentation_generation_qa_manifest.json",
            ),
            (
                "oscar_visual_augmentation_training_readiness",
                "oscar_visual_augmentation_packet/visual_augmentation_training_readiness_manifest.json",
            ),
            (
                "oscar_visual_augmentation_training_dataset",
                "oscar_visual_augmentation_packet/visual_augmentation_training_dataset_manifest.json",
            ),
            (
                "oscar_visual_augmentation_training_episodes",
                "oscar_visual_augmentation_packet/exports/visual_augmentation/episodes.jsonl",
            ),
        ):
            value = _job_artifact(resolved_job_dir, name)
            if value:
                included_artifacts[key] = value

    for key, relative_path in (
        ("site_card", "robot_eval_dataset/site_card.json"),
        ("task_cards", "robot_eval_dataset/task_cards.json"),
        ("scenario_cards", "robot_eval_dataset/scenario_cards.json"),
        ("eval_cards", "robot_eval_dataset/eval_cards.json"),
        ("rights_packet", "robot_eval_dataset/rights_packet.json"),
        ("proof_boundaries", "robot_eval_dataset/proof_boundaries.json"),
        ("robot_eval_dataset_manifest", "robot_eval_dataset/robot_eval_dataset_manifest.json"),
        ("worldlabs_export_manifest", "worldlabs_export_manifest.json"),
        ("arena_environment_packet", "simulation_automation/arena_environment_packet.json"),
        ("gpu_handoff_packet", "simulation_automation/gpu_handoff_packet.json"),
        ("scaniverse_import_manifest", "scaniverse_assets/scaniverse_import_manifest.json"),
        (
            "scaniverse_import_proof_boundary",
            "scaniverse_assets/scaniverse_import_proof_boundary.json",
        ),
    ):
        value = _pipeline_artifact(pipeline_dir, relative_path)
        if value:
            included_artifacts[key] = _relative_to(resolved_output_dir, pipeline_dir / value)

    for key, relative_path in (
        ("clip_curation_manifest", "derived/clip_curation/clip_curation_manifest.json"),
        ("clip_rejection_manifest", "derived/clip_curation/clip_rejection_manifest.json"),
        ("semantic_dedup_manifest", "derived/semantic_dedup/semantic_dedup_manifest.json"),
    ):
        candidate = context.capture_root / relative_path
        if candidate.is_file():
            included_artifacts[key] = _relative_to(resolved_output_dir, candidate)

    # required/missing/status are computed further below (after trace/labels/
    # clips are read) since main's quality-gate blockers need those inputs.
    live_closure = (
        _read_optional_mapping(resolved_job_dir / "live_eval_closure_manifest.json")
        if resolved_job_dir
        else {}
    )
    proof_boundary = (
        _read_optional_mapping(resolved_job_dir / "proof_boundary.json")
        if resolved_job_dir
        else {}
    )
    trace = (
        _read_optional_mapping(resolved_job_dir / "normalized_attempt_trace.json")
        if resolved_job_dir
        else {}
    )
    labels = (
        _read_optional_mapping(resolved_job_dir / "failure_labels.json")
        if resolved_job_dir
        else {}
    )
    metrics = (
        _read_optional_mapping(resolved_job_dir / "arena_eval_metrics.json")
        if resolved_job_dir
        else {}
    )
    clips = (
        _read_optional_mapping(resolved_job_dir / "clips_manifest.json")
        if resolved_job_dir
        else {}
    )
    clip_source_roots = [
        *([resolved_job_dir] if resolved_job_dir else []),
        context.capture_root,
        context.pipeline_root,
        context.capture_root / "raw",
    ]
    sc3_action_report = _build_sc3_action_report(trace=trace, generated_at=generated_at)
    curation_report = _build_curation_report(
        clips=clips,
        trace=trace,
        sc3_action_report=sc3_action_report,
        generated_at=generated_at,
    )
    semantic_dedup_report = _build_semantic_dedup_report(
        clips=clips,
        curation_report=curation_report,
        generated_at=generated_at,
    )
    included_artifacts["curation_report"] = "curation_report.json"
    included_artifacts["semantic_dedup_report"] = "semantic_dedup_report.json"
    included_artifacts["sc3_action_normalization_report"] = "sc3_action_normalization_report.json"

    # LeRobot/GR00T-style per-episode export from the simulator batch streams.
    # Fail-closed inside its own manifest: missing streams block it without
    # blocking the wider package, and its status is surfaced in export_policy.
    lerobot_export: Dict[str, Any] = {}
    if resolved_job_dir and (
        resolved_job_dir / "simulator_command_batch_control_stream.jsonl"
    ).is_file():
        from .lerobot_episode_export import build_lerobot_episode_export

        job_request_for_export = _read_optional_mapping(
            resolved_job_dir / "job_request.json"
        )
        lerobot_export = build_lerobot_episode_export(
            job_dir=resolved_job_dir,
            output_dir=resolved_output_dir,
            robot_id=str(
                job_request_for_export.get("robot_id")
                or job_request_for_export.get("embodiment_id")
                or "unitree_g1"
            ),
            materialized_video_by_attempt=_lerobot_episode_materialized_video_map(
                clips=clips,
                source_roots=clip_source_roots,
            ),
            generated_at=generated_at,
        )
        included_artifacts["lerobot_episode_export_manifest"] = (
            "lerobot_episode_export/lerobot_episode_export_manifest.json"
        )
    required = (
        "normalized_attempt_trace",
        "failure_labels",
        "prediction_outcome_ledger",
        "calibration_report",
        "breakage_library",
        "site_card",
        "task_cards",
        "scenario_cards",
        "eval_cards",
        "proof_boundaries",
    )
    missing = [key for key in required if key not in included_artifacts]
    sc3_action_export_blockers = [
        blocker
        for blocker in _string_list(sc3_action_report.get("blockers"))
        if blocker.rsplit(":", 1)[-1] not in SC3_NO_ACTION_DATA_BLOCKERS
    ]
    quality_gate_blockers = [
        *[f"curation:{blocker}" for blocker in _string_list(curation_report.get("blockers"))],
        *[
            f"semantic_dedup:{blocker}"
            for blocker in _string_list(semantic_dedup_report.get("blockers"))
        ],
        *[f"sc3_action:{blocker}" for blocker in sc3_action_export_blockers],
    ]
    status = (
        "blocked_missing_inputs"
        if missing
        else "blocked_package_quality_gates"
        if quality_gate_blockers
        else "export_ready_review_required"
    )
    job_request = (
        _read_optional_mapping(resolved_job_dir / "job_request.json")
        if resolved_job_dir
        else {}
    )
    webapp_projection = (
        _read_optional_mapping(resolved_job_dir / "webapp_robot_eval_status_projection.json")
        if resolved_job_dir
        else {}
    )
    task_eval_run_report = (
        _read_optional_mapping(resolved_job_dir / "task_eval_run_report.json")
        if resolved_job_dir
        else {}
    )
    scenario_matrix = (
        _read_optional_mapping(resolved_job_dir / "scenario_eval_matrix.json")
        if resolved_job_dir
        else {}
    )
    evaluation_result = (
        _read_optional_mapping(resolved_job_dir / "evaluation_result.json")
        if resolved_job_dir
        else {}
    )
    policy_package = (
        _read_optional_mapping(resolved_job_dir / "policy_package_manifest.json")
        if resolved_job_dir
        else {}
    )
    policy_report = (
        _read_optional_mapping(
            resolved_job_dir / "policy_autoresearch" / "policy_autoresearch_report.json"
        )
        if resolved_job_dir
        else {}
    )
    candidate_package = (
        _read_optional_mapping(
            resolved_job_dir / "policy_autoresearch" / "policy_candidate_package.json"
        )
        if resolved_job_dir
        else {}
    )
    heldout_result = (
        _read_optional_mapping(resolved_job_dir / "policy_autoresearch" / "heldout_eval_result.json")
        if resolved_job_dir
        else {}
    )
    direct_success_claim_ledger = (
        _read_optional_mapping(resolved_job_dir / "success_claim_ledger.json")
        if resolved_job_dir
        else {}
    )
    policy_execution_trace = (
        _read_optional_mapping(resolved_job_dir / "policy_execution_trace.json")
        if resolved_job_dir
        else {}
    )
    safety_events = (
        _read_optional_mapping(resolved_job_dir / "intervention_safety_ledger.json")
        if resolved_job_dir
        else {}
    )
    if not safety_events and resolved_job_dir:
        safety_events = _read_optional_mapping(resolved_job_dir / "safety_events_ledger.json")
    success_claim_ledger = _success_claim_ledger_from_sources(
        direct_success_claim_ledger,
        task_eval_run_report,
        candidate_package,
        heldout_result,
    )
    if success_claim_ledger:
        write_json(resolved_output_dir / "success_claim_ledger.json", success_claim_ledger)
        included_artifacts["success_claim_ledger"] = "success_claim_ledger.json"
    product_handoff = _extract_product_handoff(webapp_projection, job_request)
    rights_packet = _read_optional_mapping(pipeline_dir / "robot_eval_dataset" / "rights_packet.json")
    consent_evidence = _build_consent_evidence_record(
        capture_root=context.capture_root,
        output_dir=resolved_output_dir,
        rights_packet=rights_packet,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        generated_at=generated_at,
    )
    included_artifacts["consent_evidence"] = "consent_evidence.json"
    included_artifacts["revocation_takedown_manifest"] = (
        "revocation_takedown_manifest.json"
    )
    for key, relative_path in _mapping(
        consent_evidence.get("downstream_takedown_artifacts")
    ).items():
        included_artifacts[key] = str(relative_path)
    consent_revoked_for_gate = _revocation_takedown_required(consent_evidence)
    consent_gate_blockers = [
        "consent:consent_revoked_takedown_required"
    ] if consent_revoked_for_gate else []
    quality_gate_blockers = [*quality_gate_blockers, *consent_gate_blockers]
    status = (
        "blocked_missing_inputs"
        if missing
        else "blocked_consent_revoked_takedown_required"
        if "consent:consent_revoked_takedown_required" in consent_gate_blockers
        else "blocked_package_quality_gates"
        if quality_gate_blockers
        else "export_ready_review_required"
    )
    visual_augmentation_packet = (
        _read_optional_mapping(
            resolved_job_dir
            / "oscar_visual_augmentation_packet"
            / "oscar_visual_augmentation_packet_manifest.json"
        )
        if resolved_job_dir
        else {}
    )
    scaniverse_import = _read_optional_mapping(
        pipeline_dir / "scaniverse_assets" / "scaniverse_import_manifest.json"
    )
    rl_post_training_handoff = build_rl_post_training_handoff_packet(
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        job_id=resolved_job_dir.name if resolved_job_dir else None,
        generated_at=generated_at,
        job_request=job_request,
        scenario_matrix=scenario_matrix,
        trace=trace,
        labels=labels,
        evaluation_result=evaluation_result,
        policy_package=policy_package,
        policy_report=policy_report,
        candidate_package=candidate_package,
        heldout_result=heldout_result,
        policy_execution_trace=policy_execution_trace,
        safety_events=safety_events,
        source_artifacts=included_artifacts,
    )
    package_files = _write_package_files(
        output_dir=resolved_output_dir,
        included_artifacts=included_artifacts,
        trace=trace,
        labels=labels,
        metrics=metrics,
        clips=clips,
        curation_report=curation_report,
        semantic_dedup_report=semantic_dedup_report,
        sc3_action_report=sc3_action_report,
        generated_at=generated_at,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        visual_augmentation_packet=visual_augmentation_packet,
        scaniverse_import=scaniverse_import,
        rl_post_training_handoff=rl_post_training_handoff,
        clip_curation=_clip_curation_summary(context.capture_root),
        clip_source_roots=clip_source_roots,
    )
    included_artifacts["replay_review_instructions"] = "replay_review_instructions.md"
    live_gate_references = {
        gate_id: _live_closure_gate_reference(live_closure, gate_id)
        for gate_id in (
            "webapp_upstream_truth",
            "rights_privacy_scope",
            "review_acceptance",
            "signed_delivery_access",
        )
    }
    handoff_payloads = _write_handoff_manifests(
        output_dir=resolved_output_dir,
        job_dir=resolved_job_dir,
        generated_at=generated_at,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
        export_ready=status == "export_ready_review_required",
        included_artifacts=included_artifacts,
        live_closure=live_closure,
        live_gate_references=live_gate_references,
        trace=trace,
        labels=labels,
        clips=clips,
        consent_evidence=consent_evidence,
    )
    included_artifacts["customer_handoff_report"] = "customer_handoff_report.json"
    included_artifacts["customer_handoff_report_markdown"] = "customer_handoff_report.md"
    included_artifacts["delivery_manifest"] = "delivery_manifest.json"
    included_artifacts["signed_access_manifest"] = "signed_access_manifest.json"
    archive_manifest = _write_archive(resolved_output_dir, generated_at)
    delivery_manifest = _mapping(handoff_payloads.get("delivery_manifest"))
    signed_access_manifest = _mapping(handoff_payloads.get("signed_access_manifest"))
    revocation_takedown_record = _mapping(consent_evidence.get("revocation_takedown"))
    manifest_consent_revoked = bool(
        _explicit_true(
            consent_evidence.get("consent_revoked"),
            revocation_takedown_record.get("consent_revoked"),
        )
        or _first_text(
            consent_evidence.get("consent_revoked_at"),
            revocation_takedown_record.get("consent_revoked_at"),
        )
        or consent_evidence.get("status")
        == "blocked_consent_revoked_takedown_required"
        or revocation_takedown_record.get("status") == "takedown_required"
    )
    handoff_records = {
        "proof_boundary_path": included_artifacts.get("proof_boundary"),
        "live_eval_closure_manifest_path": included_artifacts.get(
            "live_eval_closure_manifest"
        ),
        "live_eval_closure_evidence_path": included_artifacts.get(
            "live_eval_closure_evidence"
        ),
        "rights_packet_path": included_artifacts.get("rights_packet"),
        "review_resolution_ledger_path": included_artifacts.get(
            "review_resolution_ledger"
        ),
        "accepted_failure_labels_path": included_artifacts.get(
            "accepted_failure_labels"
        ),
        "customer_handoff_report_path": included_artifacts.get("customer_handoff_report"),
        "delivery_manifest_path": included_artifacts.get("delivery_manifest"),
        "signed_access_manifest_path": included_artifacts.get("signed_access_manifest"),
        "delivery_manifest_status": delivery_manifest.get("status"),
        "signed_access_manifest_status": signed_access_manifest.get("status"),
        "revocation_takedown_manifest_path": included_artifacts.get(
            "revocation_takedown_manifest"
        ),
        "revocation_takedown_status": _mapping(
            consent_evidence.get("revocation_takedown")
        ).get("status"),
        "local_package_access_revoked": manifest_consent_revoked,
        "post_training_package_export_ready": status == "export_ready_review_required",
        "customer_handoff_ready": bool(
            _mapping(handoff_payloads.get("summary")).get("customer_handoff_ready")
        ),
        "customer_handoff_blockers": _string_list(
            _mapping(handoff_payloads.get("summary")).get("blockers")
        ),
        "live_eval_closure_status": live_closure.get("status"),
        "proof_boundary_status": proof_boundary.get("status"),
        "live_closure_gate_references": live_gate_references,
    }
    manifest_claim_boundary = {
        **dict(CLAIM_BOUNDARY),
        "post_training_package_export_ready": status == "export_ready_review_required",
        "oscar_style_curation_filters_proven": curation_report.get("status") == "passed",
        "semantic_dedup_proven": semantic_dedup_report.get("status") == "passed",
        "sc3_7d_action_contract_proven": sc3_action_report.get("status") == "passed",
        "review_acceptance_proven": live_gate_references["review_acceptance"]["passed"],
        "rights_privacy_scope_proven": live_gate_references["rights_privacy_scope"][
            "passed"
        ],
        "signed_delivery_access_proven": live_gate_references["signed_delivery_access"][
            "passed"
        ],
        "customer_handoff_ready": bool(
            _mapping(handoff_payloads.get("summary")).get("customer_handoff_ready")
        ),
        "hosted_access_ready": bool(
            _mapping(handoff_payloads.get("summary")).get("customer_handoff_ready")
        ),
        "delivery_approval_proven": False,
        "delivery_access_is_deployment_approval": False,
        "package_delivery_is_deployment_approval": False,
        "deployment_approval_proven": False,
        "physical_robot_readiness_proven": False,
        "field_readiness_proven": False,
        "safety_validation_proven": False,
        "scaniverse_support_assets_included": bool(scaniverse_import),
        "scaniverse_assets_are_raw_capture_evidence": False,
        "scaniverse_assets_are_task_success_evidence": False,
        "scaniverse_assets_are_physics_contact_evidence": False,
        "scaniverse_assets_are_deployment_readiness_evidence": False,
        "consent_revocation_blocks_downstream_use": manifest_consent_revoked,
        "package_delivery_is_not_revenue_share_commitment": True,
    }

    optional_exports = _mapping(package_files.get("optional_exports"))
    optional_formats = _mapping(optional_exports.get("formats"))
    video_bundle_format = _mapping(optional_formats.get("video_bundle"))
    lerobot_v3_format = _mapping(optional_formats.get("lerobot_v3"))
    gr00t_lerobot_format = _mapping(optional_formats.get("gr00t_lerobot"))
    lerobot_state_action_provenance = _mapping(
        lerobot_v3_format.get("state_action_provenance")
    )
    manifest_claim_boundary["measured_state_fraction_floor_passed"] = (
        lerobot_state_action_provenance.get("measured_state_fraction_floor_passed")
        is True
    )
    manifest_claim_boundary["synthesized_state_rows_are_not_measured_state_evidence"] = (
        True
    )
    downstream_takedown_execution_ledger = _read_optional_mapping(
        resolved_output_dir / "downstream_takedown_execution_ledger.json"
    )

    manifest = {
        "schema_version": POST_TRAINING_DATA_PACKAGE_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "package_type": "post_training_data_package",
        "status": status,
        "blockers": [f"missing_{key}" for key in missing] + quality_gate_blockers,
        "included_artifacts": included_artifacts,
        "handoff_records": handoff_records,
        "product_handoff": product_handoff,
        "consent_evidence": {
            "path": "consent_evidence.json",
            "status": consent_evidence.get("status"),
            "consent_evidence_present": consent_evidence.get(
                "consent_evidence_present"
            )
            is True,
            "consent_revoked": manifest_consent_revoked,
            "consent_revoked_at": consent_evidence.get("consent_revoked_at"),
            "blockers": _string_list(consent_evidence.get("blockers")),
        },
        "revocation_takedown": {
            **revocation_takedown_record,
            "consent_revoked": manifest_consent_revoked,
            "path": "revocation_takedown_manifest.json",
        },
        "downstream_takedown_execution_ledger": downstream_takedown_execution_ledger,
        "downstream_takedown_artifacts": _mapping(
            consent_evidence.get("downstream_takedown_artifacts")
        ),
        "revenue_share_review": _mapping(
            _mapping(package_files.get("license_manifest")).get(
                "revenue_share_review"
            )
        )
        or _read_optional_mapping(resolved_output_dir / "revenue_share_review.json"),
        "data_processing_terms_review": _mapping(
            _mapping(package_files.get("license_manifest")).get(
                "data_processing_terms_review"
            )
        )
        or _read_optional_mapping(
            resolved_output_dir / "data_processing_terms_review.json"
        ),
        "success_claim_ledger_path": (
            "success_claim_ledger.json" if success_claim_ledger else None
        ),
        "manifest_counts": {
            "attempt_count": int(trace.get("attempt_count") or 0),
            "failure_label_count": int(labels.get("label_count") or 0),
            "clip_count": int(clips.get("clip_count") or 0),
            "materialized_clip_count": int(
                video_bundle_format.get("materialized_clip_count") or 0
            ),
            "missing_clip_file_count": int(
                video_bundle_format.get("missing_clip_file_count") or 0
            ),
            "curated_clip_count": int(curation_report.get("accepted_clip_count") or 0),
            "rejected_clip_count": int(curation_report.get("rejected_clip_count") or 0),
            "semantic_duplicate_group_count": int(
                semantic_dedup_report.get("duplicate_group_count") or 0
            ),
            "valid_sc3_7d_action_count": int(
                sc3_action_report.get("valid_sc3_7d_action_count") or 0
            ),
            "visual_augmentation_variant_count": int(
                visual_augmentation_packet.get("variant_count") or 0
            ),
            "visual_augmentation_generated_video_count": int(
                visual_augmentation_packet.get("generated_video_count") or 0
            ),
            "scaniverse_support_asset_count": int(
                scaniverse_import.get("asset_count") or 0
            ),
            "rl_handoff_recoverable_failure_label_count": int(
                _mapping(
                    rl_post_training_handoff.get("recoverable_failure_labels")
                ).get("label_count")
                or 0
            ),
            "rl_handoff_intervention_event_count": int(
                _mapping(
                    rl_post_training_handoff.get("intervention_safety_ledger")
                ).get("event_count")
                or 0
            ),
        },
        "export_policy": {
            "curated_robot_pov_clips_required_for_richer_exports": True,
            "robot_pov_observations_included": "robot_pov_observation_manifest"
            in included_artifacts,
            "scenario_eval_matrix_included": "scenario_eval_matrix" in included_artifacts,
            "policy_execution_trace_included": "policy_execution_trace" in included_artifacts,
            "normalized_eval_attempts_included": "normalized_attempt_trace" in included_artifacts,
            "failure_labels_included": "failure_labels" in included_artifacts,
            "visual_review_ledger_included": "visual_review_ledger" in included_artifacts
            or "simulator_command_batch_visual_review_ledger" in included_artifacts,
            "arena_metrics_included": bool(metrics),
            "clips_manifest_included": bool(clips),
            "consent_evidence_record_included": "consent_evidence" in included_artifacts,
            "consent_evidence_present": consent_evidence.get(
                "consent_evidence_present"
            )
            is True,
            "consent_revoked": manifest_consent_revoked,
            "revocation_takedown_manifest_included": (
                "revocation_takedown_manifest" in _mapping(package_files.get("package_index")).get("files", {})
            ),
            "webapp_rights_privacy_takedown_notice_included": (
                "webapp_rights_privacy_takedown_notice"
                in _mapping(package_files.get("package_index")).get("files", {})
            ),
            "hosted_session_takedown_request_included": (
                "hosted_session_takedown_request"
                in _mapping(package_files.get("package_index")).get("files", {})
            ),
            "downstream_takedown_execution_ledger_included": (
                "downstream_takedown_execution_ledger"
                in _mapping(package_files.get("package_index")).get("files", {})
            ),
            "revenue_share_review_included": (
                "revenue_share_review" in _mapping(package_files.get("package_index")).get("files", {})
            ),
            "data_processing_terms_review_included": (
                "data_processing_terms_review"
                in _mapping(package_files.get("package_index")).get("files", {})
            ),
            "oscar_style_curation_filters_passed": curation_report.get("status") == "passed",
            "semantic_dedup_passed": semantic_dedup_report.get("status") == "passed",
            "sc3_7d_action_contract_passed": sc3_action_report.get("status") == "passed",
            "calibration_included": "calibration_report" in included_artifacts,
            "simulator_provider_adapter_included": "simulator_provider_adapter_manifest"
            in included_artifacts,
            "simulator_command_batch_trace_streams_included": all(
                key in included_artifacts
                for key in (
                    "simulator_command_batch_attempt_trace",
                    "simulator_command_batch_contact_stream",
                    "simulator_command_batch_planner_state",
                    "simulator_command_batch_control_stream",
                )
            ),
            "lerobot_episode_export_included": bool(lerobot_export),
            "lerobot_episode_export_status": lerobot_export.get("status"),
            "lerobot_episode_export_episode_count": lerobot_export.get(
                "episode_count"
            ),
            "lerobot_gr00t_ready_episode_count": lerobot_export.get(
                "gr00t_ready_episode_count"
            ),
            "materialized_video_bundle_included": int(
                video_bundle_format.get("materialized_clip_count") or 0
            )
            > 0,
            "all_declared_clips_materialized": bool(
                video_bundle_format.get("all_declared_clips_materialized")
            ),
            "lerobot_v3_export_included": bool(lerobot_v3_format),
            "lerobot_v3_consumer_layout_complete": bool(
                lerobot_v3_format.get("consumer_layout_complete")
            ),
            "lerobot_real_state_fraction": lerobot_state_action_provenance.get(
                "real_state_fraction"
            ),
            "lerobot_real_action_fraction": lerobot_state_action_provenance.get(
                "real_action_fraction"
            ),
            "measured_state_fraction_floor": lerobot_state_action_provenance.get(
                "measured_state_fraction_floor"
            ),
            "measured_state_fraction_floor_passed": (
                lerobot_state_action_provenance.get(
                    "measured_state_fraction_floor_passed"
                )
                is True
            ),
            "gr00t_lerobot_export_included": bool(gr00t_lerobot_format),
            "gr00t_lerobot_consumer_layout_complete": bool(
                gr00t_lerobot_format.get("consumer_layout_complete")
            ),
            "gr00t_modality_json_included": bool(
                gr00t_lerobot_format.get("modality_json_path")
            ),
            "sim_vs_real_calibration_included": "sim_vs_real_calibration_report"
            in included_artifacts,
            "deployment_outcome_intake_included": "deployment_outcome_intake_manifest"
            in included_artifacts,
            "deployment_outcomes_included": "deployment_outcome_ledger" in included_artifacts,
            "real_world_validation_followup_plan_included": (
                "real_world_validation_followup_plan" in included_artifacts
            ),
            "real_world_validation_followup_queue_included": (
                "real_world_validation_followup_request_queue" in included_artifacts
            ),
            "live_eval_closure_included": "live_eval_closure_manifest"
            in included_artifacts,
            "breakage_library_included": "breakage_library" in included_artifacts,
            "robot_eval_report_included": "robot_eval_report" in included_artifacts,
            "visual_augmentation_packet_included": bool(visual_augmentation_packet),
            "visual_augmentation_is_model_derived_support": bool(
                visual_augmentation_packet
            ),
            "visual_augmentation_generated_videos_are_raw_capture_evidence": False,
            "scaniverse_support_assets_included": bool(scaniverse_import),
            "scaniverse_assets_are_external_derived_support": bool(scaniverse_import),
            "scaniverse_assets_are_raw_capture_evidence": False,
            "scaniverse_assets_are_task_success_evidence": False,
            "scaniverse_assets_are_physics_contact_evidence": False,
            "rl_post_training_handoff_included": bool(rl_post_training_handoff),
            "rl_sparse_reward_signal_included": bool(
                _mapping(rl_post_training_handoff.get("sparse_reward_signal"))
            ),
            "concurrent_baseline_ab_plan_included": bool(
                _mapping(rl_post_training_handoff.get("concurrent_baseline_ab"))
            ),
            "bottleneck_stage_detection_included": bool(
                _mapping(rl_post_training_handoff.get("bottleneck_stage_detection"))
            ),
            "speed_curriculum_plan_included": bool(
                _mapping(rl_post_training_handoff.get("speed_curriculum_plan"))
            ),
            "action_chunk_continuity_qa_included": bool(
                _mapping(rl_post_training_handoff.get("action_chunk_continuity_qa"))
            ),
            "intervention_safety_ledger_included": bool(
                _mapping(rl_post_training_handoff.get("intervention_safety_ledger"))
            ),
        },
        "optional_exports": optional_exports,
        "package_files": package_files["package_files"],
        "dataset_card_path": "dataset_card.json",
        "license_manifest_path": "license_manifest.json",
        "package_index_path": "package_index.json",
        "checksums_path": "checksums.json",
        "archive_manifest_path": "archive_manifest.json",
        "archive": archive_manifest["archive"],
        "curation_report_path": "curation_report.json",
        "replay_review_instructions_path": "replay_review_instructions.md",
        "semantic_dedup_report_path": "semantic_dedup_report.json",
        "sc3_action_normalization_report_path": "sc3_action_normalization_report.json",
        "optional_export_manifest_path": "optional_export_manifest.json",
        "visual_augmentation_support_manifest_path": (
            "visual_augmentation_support_manifest.json"
            if package_files.get("visual_augmentation_support")
            else None
        ),
        "scaniverse_support_asset_manifest_path": (
            "scaniverse_support_asset_manifest.json"
            if package_files.get("scaniverse_support")
            else None
        ),
        "rl_post_training_handoff_packet_path": "rl_post_training_handoff_packet.json",
        "rl_post_training_handoff_summary": {
            "concurrent_baseline_ab_status": _mapping(
                rl_post_training_handoff.get("concurrent_baseline_ab")
            ).get("status"),
            "dominant_bottleneck_stage": _mapping(
                rl_post_training_handoff.get("bottleneck_stage_detection")
            ).get("dominant_stage"),
            "speed_curriculum_status": _mapping(
                rl_post_training_handoff.get("speed_curriculum_plan")
            ).get("status"),
            "action_chunk_qa_status": _mapping(
                rl_post_training_handoff.get("action_chunk_continuity_qa")
            ).get("status"),
            "safety_validation_proven": False,
        },
        "claim_boundary": manifest_claim_boundary,
    }
    buyer_readout = build_buyer_package_readout(
        export_manifest=manifest,
        success_claim_ledger=success_claim_ledger,
        product_handoff=product_handoff,
    )
    write_json(resolved_output_dir / "buyer_package_readout.json", buyer_readout)
    (resolved_output_dir / "buyer_package_summary.md").write_text(
        render_buyer_package_readout_markdown(buyer_readout), encoding="utf-8"
    )
    manifest["buyer_package_readout_path"] = "buyer_package_readout.json"
    manifest["buyer_package_summary_path"] = "buyer_package_summary.md"
    manifest["buyer_readout_status"] = buyer_readout["status"]
    write_json(resolved_output_dir / "post_training_data_package_export_manifest.json", manifest)
    _annotate_live_closure_with_handoff(
        job_dir=resolved_job_dir,
        handoff_summary=_mapping(handoff_payloads.get("summary")),
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Post-Training Data Package export, checksum, and archive"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    result = build_post_training_data_package_export(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
    )
    default_output_dir = Path(args.capture_root) / "pipeline" / "post_training_data_package"
    manifest_dir = Path(args.output_dir or args.job_dir or default_output_dir)
    manifest_path = manifest_dir / "post_training_data_package_export_manifest.json"
    print(f"[post-training-data-package] manifest={manifest_path}")
    print(f"[post-training-data-package] status={result['status']}")
    return 0 if result["status"] == "export_ready_review_required" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
