"""Alpha-readiness validation and downstream WebApp sync helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .consent_normalization import strict_allow_bool
from .common import (
    optional_read_json,
    parse_bool,
    parse_gs_uri,
    read_json,
    to_pipeline_prefix,
    utc_now_iso,
    write_json,
)
from .webapp_sync import (
    WebappSyncError,
    derive_webapp_opportunity_state,
    derive_webapp_qualification_state,
    sync_webapp_pipeline_attachment,
    upstream_link_id_failures,
)


_COMMON_ENV_VARS = (
    "PIPELINE_PROJECT_ID",
    "PIPELINE_REGION",
    "PIPELINE_BUCKET",
    "GCS_ROOT",
    "PIPELINE_SYNC_WEBAPP_URL",
    "PIPELINE_SYNC_TOKEN",
)
_PRIVACY_ENV_VARS = (
    "PRIVACY_RUNNER_TOKEN",
    "PRIVACY_SAM3_URL",
    "PRIVACY_VIP_URL",
    "PRIVACY_DEEPPRIVACY2_URL",
)
_OPERATOR_LAUNCH_EVIDENCE_SCHEMA_VERSION = "operator_launch_evidence.v1"
_OPERATOR_LAUNCH_EVIDENCE_RELATIVE_PATH = "pipeline/operator_launch_evidence.json"
_OPERATOR_EVIDENCE_VERIFIED_STATUSES = {
    "approved",
    "completed",
    "executed",
    "passed",
    "ready",
    "recorded",
    "rotated",
    "settled",
    "signed",
    "verified",
}
_INDUSTRIAL_SITE_TYPE_MARKERS = {
    "warehouse",
    "manufacturing",
    "factory",
    "fulfillment",
    "industrial",
    "industrial_unknown",
    "brownfield",
    "plant",
    "distribution_center",
}
_INDUSTRIAL_AUTHORIZATION_CHECK_ID = "industrial_site_authorization_ehs_signoff"


def _read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _check(name: str, passed: bool, detail: str, *, category: str) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "category": category,
    }


def _check_file(path: Path, *, name: str, detail: str, category: str = "artifact") -> Dict[str, Any]:
    return _check(name, path.is_file(), detail if path.is_file() else f"{detail} missing", category=category)


def _string_list_or_default(*values: object) -> List[str]:
    out: List[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(str(item).strip() for item in value if str(item).strip())
    return out or ["unspecified evidence gaps"]


def _bool_env(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    return parse_bool(env.get(name), default=default)


def _env_check(env: Mapping[str, str], name: str, *, expected_value: Optional[str] = None) -> Dict[str, Any]:
    value = str(env.get(name) or "").strip()
    if expected_value is None:
        passed = bool(value)
        detail = f"{name} is configured" if passed else f"{name} is missing"
    else:
        passed = value.lower() == expected_value.lower()
        detail = (
            f"{name}={expected_value}"
            if passed
            else f"{name} must be {expected_value}, got {value or 'unset'}"
        )
    return _check(name.lower(), passed, detail, category="env")


def _mode_payload(descriptor_payload: Mapping[str, Any]) -> Dict[str, Any]:
    capture_mode = descriptor_payload.get("capture_mode")
    if isinstance(capture_mode, Mapping):
        return dict(capture_mode)
    metadata = descriptor_payload.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("capture_mode"), Mapping):
        return dict(metadata.get("capture_mode") or {})
    return {}


def _present_value(payload: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return None


def _text_from_mapping(payload: Mapping[str, Any], *keys: str) -> List[str]:
    values: List[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, (list, tuple)):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, Mapping):
            values.extend(
                str(item).strip()
                for item in value.values()
                if isinstance(item, str) and str(item).strip()
            )
    return values


def _site_type_candidates(
    *,
    descriptor: CaptureDescriptor,
    raw_manifest: Mapping[str, Any],
    rights_review: Mapping[str, Any],
) -> List[str]:
    candidates: List[str] = []
    descriptor_metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    for payload in (raw_manifest, descriptor_metadata, descriptor.quality, rights_review):
        if not isinstance(payload, Mapping):
            continue
        candidates.extend(
            _text_from_mapping(
                payload,
                "site_type",
                "siteType",
                "intended_space_type",
                "intendedSpaceType",
                "environment_type_hint",
                "environmentTypeHint",
                "scene_class",
                "sceneClass",
                "location_type",
                "locationType",
                "site_category",
                "siteCategory",
            )
        )
        nested = payload.get("metadata")
        if isinstance(nested, Mapping):
            candidates.extend(
                _text_from_mapping(
                    nested,
                    "site_type",
                    "siteType",
                    "intended_space_type",
                    "intendedSpaceType",
                    "environment_type_hint",
                    "environmentTypeHint",
                    "site_category",
                    "siteCategory",
                )
            )
    if descriptor.environment_type_hint:
        candidates.append(descriptor.environment_type_hint)
    return candidates


def _industrial_authorization_required(
    *,
    descriptor: CaptureDescriptor,
    raw_manifest: Mapping[str, Any],
    rights_review: Mapping[str, Any],
) -> bool:
    for value in _site_type_candidates(
        descriptor=descriptor,
        raw_manifest=raw_manifest,
        rights_review=rights_review,
    ):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _INDUSTRIAL_SITE_TYPE_MARKERS:
            return True
        if any(marker in normalized for marker in _INDUSTRIAL_SITE_TYPE_MARKERS):
            return True
    return False


def _uri(bucket: str, pipeline_prefix: str, relative_path: str) -> str:
    return f"gs://{bucket}/{pipeline_prefix}/{relative_path}"


def _latest_sync_payload(existing: Mapping[str, Any]) -> Dict[str, Any]:
    syncs = existing.get("syncs")
    if isinstance(syncs, Mapping):
        latest_stage = str(existing.get("latest_stage") or "").strip()
        latest = syncs.get(latest_stage)
        if isinstance(latest, Mapping):
            return dict(latest)
    return dict(existing)


_DEFAULT_SYNC_MAX_AGE_HOURS = 24.0


def _webapp_sync_verification(
    webapp_sync: Mapping[str, Any],
    *,
    env: Mapping[str, str],
) -> Dict[str, Any]:
    """Fail-closed truth check on the recorded WebApp sync result.

    A sync result only counts as verified launch evidence when it succeeded
    against real upstream WebApp records (no placeholder fallback) and is
    recent enough to still describe the WebApp's current state. Missing
    timestamps, placeholder fallbacks, and unverified upstream links all fail.
    """
    latest = _latest_sync_payload(webapp_sync)
    failures: list[str] = []
    status = str(latest.get("status") or webapp_sync.get("status") or "").strip().lower()
    if status != "succeeded":
        failures.append(f"status:{status or 'missing'}")
    attachment_payload = (
        latest.get("attachment_payload")
        if isinstance(latest.get("attachment_payload"), Mapping)
        else {}
    )
    if attachment_payload.get("upstream_links_verified") is not True:
        failures.append("upstream_links_not_verified")
    if bool(attachment_payload.get("placeholder_fallback_allowed")):
        failures.append("placeholder_fallback_enabled")
    synced_at_raw = str(latest.get("synced_at") or webapp_sync.get("latest_synced_at") or "").strip()
    max_age_hours = _DEFAULT_SYNC_MAX_AGE_HOURS
    raw_max_age = str(env.get("PIPELINE_SYNC_MAX_AGE_HOURS") or "").strip()
    if raw_max_age:
        try:
            max_age_hours = float(raw_max_age)
        except ValueError:
            pass
    if not synced_at_raw:
        failures.append("synced_at_missing")
    else:
        try:
            synced_at = datetime.fromisoformat(synced_at_raw.replace("Z", "+00:00"))
        except ValueError:
            failures.append("synced_at_unparseable")
        else:
            if synced_at.tzinfo is None:
                synced_at = synced_at.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - synced_at).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                failures.append(f"stale_sync_result:{age_hours:.1f}h_old")
    return {
        "verified": not failures,
        "failures": failures,
        "synced_at": synced_at_raw or None,
        "max_age_hours": max_age_hours,
    }


def _operator_evidence_entry(
    operator_evidence: Mapping[str, Any],
    check_id: str,
) -> Dict[str, Any]:
    checks = operator_evidence.get("checks")
    if isinstance(checks, Mapping):
        candidate = checks.get(check_id)
        if isinstance(candidate, Mapping):
            return dict(candidate)
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, Mapping) and str(item.get("id") or "") == check_id:
                return dict(item)

    required_checks = operator_evidence.get("operator_required_checks")
    if isinstance(required_checks, Mapping):
        candidate = required_checks.get(check_id)
        if isinstance(candidate, Mapping):
            return dict(candidate)
    if isinstance(required_checks, list):
        for item in required_checks:
            if isinstance(item, Mapping) and str(item.get("id") or "") == check_id:
                return dict(item)

    candidate = operator_evidence.get(check_id)
    return dict(candidate) if isinstance(candidate, Mapping) else {}


def _operator_evidence_has_ref(entry: Mapping[str, Any]) -> bool:
    if _present_value(
        entry,
        "evidence_uri",
        "evidence_ref",
        "proof_uri",
        "proof_ref",
        "document_uri",
        "signed_record_uri",
        "secret_version_ref",
        "stripe_event_id",
        "payment_intent_id",
        "checkout_session_id",
        "payout_id",
        "transfer_id",
        "provider_account_ref",
        "review_queue_uri",
        "buyer_session_ref",
        "recording_uri",
        "decision_record_uri",
        "industrial_authorization_record_uri",
        "site_authorization_record_uri",
        "ehs_signoff_uri",
        "safety_signoff_uri",
        "worker_pii_consent_posture_uri",
        "worker_consent_posture_uri",
        "works_council_review_uri",
        "nda_or_proprietary_data_terms_uri",
        "nda_attestation_uri",
        "restricted_zone_controls_uri",
        "data_residency_policy_uri",
        "us_only_beta_scope_uri",
        "signed_transfer_terms_uri",
        "signed_dpa_scc_uri",
        "standard_contractual_clauses_uri",
        "transfer_impact_assessment_uri",
        "subprocessor_residency_terms_uri",
        "data_residency_region_policy_uri",
    ):
        return True
    for key in ("evidence", "artifacts", "refs", "metadata"):
        value = entry.get(key)
        if isinstance(value, Mapping) and value:
            return True
        if isinstance(value, list) and value:
            return True
    return False


def _operator_evidence_file_errors(operator_evidence: Mapping[str, Any]) -> List[str]:
    if not operator_evidence:
        return []
    errors: List[str] = []
    schema_version = str(operator_evidence.get("schema_version") or "").strip()
    if schema_version != _OPERATOR_LAUNCH_EVIDENCE_SCHEMA_VERSION:
        errors.append("operator_launch_evidence_schema_version_invalid")
    checks = operator_evidence.get("checks")
    if not isinstance(checks, (Mapping, list)):
        errors.append("operator_launch_evidence_checks_missing")
    return errors


def _entry_has_any(entry: Mapping[str, Any], *keys: str) -> bool:
    if _present_value(entry, *keys):
        return True
    metadata = entry.get("metadata")
    if isinstance(metadata, Mapping) and _present_value(metadata, *keys):
        return True
    return False


def _entry_bool(entry: Mapping[str, Any], key: str) -> bool:
    value = entry.get(key)
    metadata = entry.get("metadata")
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get(key)
    return value is True


def _entry_list_empty(entry: Mapping[str, Any], *keys: str) -> bool:
    for key in keys:
        value = entry.get(key)
        metadata = entry.get("metadata")
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get(key)
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, str):
            return value.strip() in {"", "[]", "none", "no_open_requirements"}
    return False


def _entry_has_non_empty_list(entry: Mapping[str, Any], *keys: str) -> bool:
    for key in keys:
        value = entry.get(key)
        metadata = entry.get("metadata")
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _entry_string_values(entry: Mapping[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    metadata = entry.get("metadata")
    for key in keys:
        value = entry.get(key)
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get(key)
        if isinstance(value, str):
            candidates = [part.strip() for part in value.split(",")]
        elif isinstance(value, list | tuple | set):
            candidates = [str(item or "").strip() for item in value]
        else:
            candidates = []
        for candidate in candidates:
            if candidate and candidate not in values:
                values.append(candidate)
    return values


def _country_codes_are_us_only(values: list[str]) -> bool:
    normalized = {value.strip().upper().replace("-", "_") for value in values if value.strip()}
    return bool(normalized) and normalized.issubset({"US", "USA", "UNITED_STATES"})


def _entry_declares_us_only_beta_scope(entry: Mapping[str, Any]) -> bool:
    tester_countries = _entry_string_values(
        entry,
        "allowed_tester_countries",
        "allowed_participant_countries",
    )
    site_countries = _entry_string_values(
        entry,
        "allowed_site_countries",
        "allowed_capture_site_countries",
    )
    return bool(
        _entry_bool(entry, "non_us_participants_blocked")
        and _country_codes_are_us_only(tester_countries)
        and _country_codes_are_us_only(site_countries)
        and _entry_has_any(entry, "us_only_beta_scope_uri", "data_residency_policy_uri")
    )


def _entry_declares_signed_transfer_terms(entry: Mapping[str, Any]) -> bool:
    return bool(
        _entry_has_any(
            entry,
            "signed_transfer_terms_uri",
            "signed_dpa_scc_uri",
            "standard_contractual_clauses_uri",
        )
        and _entry_has_any(entry, "transfer_impact_assessment_uri")
        and _entry_has_any(
            entry,
            "subprocessor_residency_terms_uri",
            "data_residency_region_policy_uri",
        )
    )


def _entry_status_success(entry: Mapping[str, Any], *keys: str) -> bool:
    success_values = {"succeeded", "success", "ok", "passed", "verified", "200", "2xx", "http_200"}
    for key in keys:
        value = entry.get(key)
        metadata = entry.get("metadata")
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in success_values:
            return True
    return False


def _entry_declares_live_mode(entry: Mapping[str, Any]) -> bool:
    metadata = entry.get("metadata")
    livemode = entry.get("livemode")
    if livemode is None and isinstance(metadata, Mapping):
        livemode = metadata.get("livemode")
    if livemode is True:
        return True

    for key in ("provider_mode", "stripe_mode", "mode", "environment"):
        value = entry.get(key)
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get(key)
        if isinstance(value, str) and value.strip().lower() == "live":
            return True
    return False


def _operator_evidence_specific_failures(check_id: str, entry: Mapping[str, Any]) -> List[str]:
    failures: List[str] = []

    if check_id in {"legal_consent_posture_signoff", "operator_dpa_data_processing_terms"}:
        if not _entry_has_any(entry, "signed_record_uri", "document_uri"):
            failures.append("missing_signed_legal_or_dpa_record")
        if check_id == "operator_dpa_data_processing_terms":
            if not _entry_has_any(
                entry,
                "retention_policy_uri",
                "retention_policy_ref",
                "retention_policy_terms_uri",
                "retention_policy_schema",
            ):
                failures.append("missing_retention_policy_terms")
            if not (
                _entry_has_any(entry, "subprocessor_list_uri", "subprocessor_terms_uri")
                or _entry_has_non_empty_list(entry, "subprocessors", "subprocessor_list")
            ):
                failures.append("missing_subprocessor_list")
            if not _entry_has_any(
                entry,
                "access_audit_terms_uri",
                "access_audit_log_policy_uri",
                "access_audit_report_uri",
                "access_audit_ref",
            ):
                failures.append("missing_access_audit_terms")
    elif check_id == _INDUSTRIAL_AUTHORIZATION_CHECK_ID:
        if not _entry_has_any(
            entry,
            "signed_record_uri",
            "industrial_authorization_record_uri",
            "site_authorization_record_uri",
        ):
            failures.append("missing_industrial_site_authorization_record")
        if not _entry_has_any(entry, "site_authorizer_name", "authorizer_name"):
            failures.append("missing_site_authorizer_name")
        if not _entry_has_any(entry, "site_authorizer_role", "authorizer_role"):
            failures.append("missing_site_authorizer_role")
        if not _entry_has_any(entry, "ehs_signoff_uri", "safety_signoff_uri"):
            failures.append("missing_ehs_safety_signoff")
        if not _entry_has_any(
            entry,
            "worker_pii_consent_posture_uri",
            "worker_consent_posture_uri",
            "works_council_review_uri",
        ):
            failures.append("missing_worker_pii_or_works_council_posture")
        if not _entry_has_any(
            entry,
            "nda_or_proprietary_data_terms_uri",
            "nda_attestation_uri",
        ):
            failures.append("missing_nda_or_proprietary_data_terms")
        if not _entry_bool(entry, "ppe_requirements_acknowledged"):
            failures.append("ppe_requirements_not_acknowledged")
        if not _entry_bool(entry, "escort_requirements_acknowledged"):
            failures.append("escort_requirements_not_acknowledged")
        if not _entry_has_any(
            entry,
            "restricted_zone_controls_uri",
            "loto_forklift_restricted_zone_policy_uri",
        ):
            failures.append("missing_restricted_zone_controls")
    elif check_id == "cross_border_data_residency_posture":
        if not _entry_has_any(
            entry,
            "data_residency_policy_uri",
            "us_only_beta_scope_uri",
            "signed_transfer_terms_uri",
            "signed_dpa_scc_uri",
            "standard_contractual_clauses_uri",
        ):
            failures.append("missing_data_residency_or_transfer_record")
        if not (
            _entry_declares_us_only_beta_scope(entry)
            or _entry_declares_signed_transfer_terms(entry)
        ):
            failures.append("missing_us_only_scope_or_signed_transfer_terms")
    elif check_id == "paperclip_ops_relay_secret_rotation":
        if not _entry_has_any(entry, "secret_version_ref"):
            failures.append("missing_secret_version_ref")
        if not _entry_has_any(entry, "redeploy_evidence_uri", "redeploy_ref"):
            failures.append("missing_redeploy_evidence")
    elif check_id.endswith("_real_device_claim_flow"):
        if not _entry_has_any(entry, "recording_uri", "screen_recording_uri"):
            failures.append("missing_real_device_recording")
        if not _entry_has_any(entry, "capture_job_id"):
            failures.append("missing_capture_job_id_continuity")
    elif check_id == "buyer_payment_settlement":
        if not _entry_has_any(entry, "payment_intent_id", "checkout_session_id", "stripe_event_id"):
            failures.append("missing_live_payment_identifier")
        if not _entry_declares_live_mode(entry):
            failures.append("stripe_mode_not_live")
    elif check_id == "capturer_payout_settlement":
        if not _entry_has_any(entry, "payout_id", "transfer_id"):
            failures.append("missing_live_payout_or_transfer_identifier")
        if not _entry_has_any(entry, "webhook_reconciliation_uri", "creator_payout_ledger_ref", "ledger_entry_uri"):
            failures.append("missing_payout_webhook_or_ledger_reconciliation")
        if not _entry_declares_live_mode(entry):
            failures.append("stripe_mode_not_live")
    elif check_id == "stripe_connected_account_live_readiness":
        if not _entry_has_any(entry, "provider_account_ref", "stripe_account_id"):
            failures.append("missing_connected_account_ref")
        for key in ("provider_state_checked", "live_provider_ready", "payouts_enabled"):
            if not _entry_bool(entry, key):
                failures.append(f"{key}_not_true")
        metadata = entry.get("metadata")
        metadata_mode = metadata.get("provider_mode") if isinstance(metadata, Mapping) else ""
        mode = str(entry.get("provider_mode") or metadata_mode or "").strip().lower()
        if mode != "live":
            failures.append("provider_mode_not_live")
        if not _entry_list_empty(entry, "blocking_requirements", "requirements_currently_due"):
            failures.append("blocking_requirements_not_proven_empty")
    elif check_id == "payout_exception_monitor_live":
        if not _entry_has_any(entry, "monitor_uri", "query_uri", "alert_policy_uri", "dashboard_uri"):
            failures.append("missing_live_payout_exception_monitor_ref")
    elif check_id in {"identity_kyc_provider_decision", "background_check_provider_decision"}:
        if not _entry_has_any(entry, "decision_record_uri", "document_uri"):
            failures.append("missing_provider_decision_record")
    elif check_id == "human_finance_review_owner":
        if not _entry_has_any(entry, "finance_owner"):
            failures.append("missing_finance_owner")
        if not _entry_has_any(entry, "review_queue_uri", "review_queue_ref"):
            failures.append("missing_finance_review_queue")
    elif check_id == "buyer_artifact_access":
        if not _entry_has_any(entry, "buyer_session_ref"):
            failures.append("missing_authenticated_buyer_session_ref")
        if not _entry_has_any(entry, "artifact_access_log_uri"):
            failures.append("missing_artifact_access_log")
        if not _entry_status_success(entry, "signed_url_fetch_status", "authenticated_fetch_status"):
            failures.append("missing_executed_artifact_access_fetch")

    return failures


def _operator_evidence_verified(entry: Mapping[str, Any], check_id: str) -> bool:
    status = str(entry.get("status") or "").strip().lower()
    if not status and entry.get("passed") is True:
        status = "verified"
    has_time = bool(
        _present_value(
            entry,
            "verified_at",
            "completed_at",
            "signed_at",
            "rotated_at",
            "settled_at",
            "decided_at",
            "recorded_at",
        )
    )
    has_actor = bool(
        _present_value(
            entry,
            "verified_by",
            "signed_by",
            "operator_id",
            "owner",
            "finance_owner",
            "legal_owner",
            "ehs_owner",
            "site_authorizer_name",
            "authorizer_name",
            "security_owner",
        )
    )
    return (
        status in _OPERATOR_EVIDENCE_VERIFIED_STATUSES
        and _operator_evidence_has_ref(entry)
        and has_time
        and has_actor
        and not _operator_evidence_specific_failures(check_id, entry)
    )


def _operator_required_check(
    *,
    check_id: str,
    scope: str,
    required_evidence: str,
    operator_evidence: Mapping[str, Any],
    evidence_file_errors: List[str] | None = None,
) -> Dict[str, Any]:
    entry = _operator_evidence_entry(operator_evidence, check_id)
    validation_errors = list(
        evidence_file_errors
        if evidence_file_errors is not None
        else _operator_evidence_file_errors(operator_evidence)
    )
    if entry:
        validation_errors.extend(_operator_evidence_specific_failures(check_id, entry))
    verified = not validation_errors and _operator_evidence_verified(entry, check_id)
    status = str(entry.get("status") or "").strip().lower() if entry else "missing"
    return {
        "id": check_id,
        "scope": scope,
        "required_evidence": required_evidence,
        "passed": verified,
        "status": "verified" if verified else status or "unverified",
        "blocker": None if verified else f"{check_id}_evidence_missing_or_unverified",
        "evidence_validation_errors": validation_errors,
        "evidence": entry,
    }


def validate_operator_launch_evidence(
    operator_evidence: Mapping[str, Any],
    required_check_ids: List[str],
) -> Dict[str, Any]:
    """Validate launch operator evidence for a bounded list of evidence ids.

    This public helper shares the same per-id proof requirements used by the
    per-capture launch gate, so repo-level packets do not accidentally accept a
    generic ``status=verified`` record for live payments, payouts, device flows,
    or buyer artifact access.
    """

    normalized_ids = [str(item).strip() for item in required_check_ids if str(item).strip()]
    evidence_file_errors = _operator_evidence_file_errors(operator_evidence)
    checks = [
        _operator_required_check(
            check_id=check_id,
            scope="operator_evidence",
            required_evidence="live operator evidence for launch-readiness packet",
            operator_evidence=operator_evidence,
            evidence_file_errors=evidence_file_errors,
        )
        for check_id in normalized_ids
    ]
    verified_ids = [str(check["id"]) for check in checks if check["passed"]]
    remaining_ids = [str(check["id"]) for check in checks if not check["passed"]]
    blockers = [
        str(check["blocker"])
        for check in checks
        if not check["passed"] and check.get("blocker")
    ]
    return {
        "schema_version": _OPERATOR_LAUNCH_EVIDENCE_SCHEMA_VERSION,
        "status": "verified" if not remaining_ids else "blocked",
        "evidence_file_present": bool(operator_evidence),
        "schema_errors": evidence_file_errors,
        "required_count": len(checks),
        "verified_count": len(verified_ids),
        "verified_ids": verified_ids,
        "remaining_ids": remaining_ids,
        "blockers": blockers,
        "checks": checks,
        "claim_boundary": "operator_evidence_is_live_human_or_external_service_proof_not_automation",
    }


def _runtime_capability_payload(
    *,
    profile: str,
    runtime_launch_expected: bool,
    geometry_summary: Mapping[str, Any],
    site_world_spec: Mapping[str, Any],
    site_world_registration: Mapping[str, Any],
    site_world_health: Mapping[str, Any],
) -> Dict[str, Any]:
    has_site_world_bundle = bool(site_world_spec and site_world_registration and site_world_health)
    geometry_ready = bool(geometry_summary.get("ready_for_world_model"))
    geometry_live_ready = bool(geometry_summary.get("geometry_live_ready"))
    geometry_source = str(geometry_summary.get("geometry_source") or "missing").strip()
    fallback_used = bool(geometry_summary.get("fallback_used"))
    provider_native_result = bool(geometry_summary.get("provider_native_result"))
    site_frame_available = bool(geometry_summary.get("site_frame_available"))
    scale_resolved = bool(geometry_summary.get("scale_resolved"))
    runtime_launchable = bool(site_world_health.get("launchable"))
    runtime_status = str(site_world_health.get("status") or "missing").strip().lower()
    geometry_required = profile in {"meta_glasses", "android_video", "iphone_video_only"}

    blockers: List[str] = []
    if not has_site_world_bundle:
        blockers.append("missing_site_world_bundle")
    if geometry_required and not geometry_ready:
        blockers.append("geometry_not_ready")
    if geometry_required and (
        fallback_used
        or geometry_source != "video_to_world"
        or not geometry_live_ready
        or not provider_native_result
        or not site_frame_available
        or not scale_resolved
    ):
        blockers.append("geometry_not_live_video_to_world")
    if runtime_launch_expected and not runtime_launchable:
        blockers.append("runtime_not_launchable")
    if runtime_launch_expected and runtime_status in {"missing", "", "blocked", "failed"}:
        blockers.append("runtime_health_not_ready")

    return {
        "claim_scope": "native_runtime_capability_only",
        "status": "ready" if not blockers else "blocked",
        "launchable": runtime_launchable,
        "geometry_required": geometry_required,
        "geometry_ready": geometry_ready,
        "geometry_live_ready": geometry_live_ready,
        "geometry_source": geometry_source,
        "fallback_used": fallback_used,
        "provider_native_result": provider_native_result,
        "site_frame_available": site_frame_available,
        "scale_resolved": scale_resolved,
        "non_arkit_geometry_state": (
            "ready"
            if geometry_required
            and geometry_live_ready
            and provider_native_result
            and geometry_source == "video_to_world"
            else "degraded"
            if geometry_required
            and geometry_source == "local_sfm"
            and bool(geometry_summary.get("contract_ready_for_world_model"))
            else "not_applicable"
            if not geometry_required
            else "blocked"
        ),
        "site_world_bundle_ready": has_site_world_bundle,
        "runtime_health_status": runtime_status or "missing",
        "blockers": blockers,
    }


def write_pipeline_sync_result(
    *,
    pipeline_root: Path,
    stage: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    path = pipeline_root / "webapp_sync_result.json"
    existing = _read_json_object(path)
    syncs = existing.get("syncs") if isinstance(existing.get("syncs"), Mapping) else {}
    merged_syncs = {str(key): value for key, value in syncs.items()}
    if existing and not merged_syncs:
        legacy_stage = str(existing.get("latest_stage") or existing.get("stage") or "qualification").strip() or "qualification"
        merged_syncs[legacy_stage] = _latest_sync_payload(existing)
    stage_result = dict(result)
    stage_result.setdefault("synced_at", utc_now_iso())
    merged_syncs[stage] = stage_result
    payload = {
        "status": str(stage_result.get("status") or "unknown"),
        "latest_stage": stage,
        "latest_synced_at": stage_result["synced_at"],
        "syncs": merged_syncs,
    }
    write_json(path, payload)
    return payload


def build_alpha_readiness_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    resolved_env = dict(os.environ if env is None else env)
    descriptor_path = capture_root / "capture_descriptor.json"
    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"
    scene_memory_root = pipeline_root / "scene_memory"
    presentation_root = pipeline_root / "presentation_world"
    privacy_depth_root = pipeline_root / "privacy_depth"
    geometry_root = pipeline_root / "geometry"

    descriptor_payload = _read_json_object(descriptor_path)
    descriptor = (
        CaptureDescriptor.from_dict(descriptor_payload)
        if descriptor_payload
        else CaptureDescriptor.from_dict({})
    )
    mode_payload = _mode_payload(descriptor_payload)
    capture_mode_resolved = str(mode_payload.get("resolved_mode") or "").strip() or None
    if capture_mode_resolved is None:
        quality = descriptor_payload.get("quality") if isinstance(descriptor_payload.get("quality"), Mapping) else {}
        rights = descriptor_payload.get("metadata") if isinstance(descriptor_payload.get("metadata"), Mapping) else {}
        rights_block = rights.get("capture_rights") if isinstance(rights.get("capture_rights"), Mapping) else {}
        candidate = bool(
            descriptor.arkit_poses_uri
            or bool(quality.get("world_model_candidate"))
            or (
                bool(quality.get("geometry_ready"))
                and strict_allow_bool(
                    rights_block.get("derived_scene_generation_allowed")
                )
            )
        )
        capture_mode_resolved = "site_world_candidate" if candidate else "qualification_only"

    qa_report = _read_json_object(capture_root / "qa_report.json")
    gemini_review = _read_json_object(pipeline_root / "gemini_capture_fidelity_review.json")
    privacy_manifest = _read_json_object(pipeline_root / "privacy_processing_manifest.json")
    webapp_sync = _read_json_object(pipeline_root / "webapp_sync_result.json")
    geometry_summary = _read_json_object(geometry_root / "geometry_summary.json")
    site_world_spec = _read_json_object(eval_root / "site_world_spec.json")
    site_world_registration = _read_json_object(eval_root / "site_world_registration.json")
    site_world_health = _read_json_object(eval_root / "site_world_health.json")

    profile = "unsupported"
    if descriptor.capture_source == "iphone":
        profile = (
            "iphone_arkit_lidar"
            if descriptor.capture_modality == "iphone_arkit_lidar"
            else "iphone_video_only"
        )
    elif descriptor.capture_source == "glasses":
        profile = "meta_glasses"
    elif descriptor.capture_source == "android":
        profile = "android_video"

    requested_outputs = {
        str(value or "").strip().lower()
        for value in descriptor.requested_outputs
        if str(value or "").strip()
    }
    evaluation_requested = bool(
        requested_outputs.intersection({"deeper_evaluation", "evaluation_prep"})
        or eval_root.exists()
    )
    runtime_launch_expected = parse_bool(
        resolved_env.get("PIPELINE_ALPHA_EXPECT_HOSTED_RUNTIME"),
        default=evaluation_requested,
    )
    runtime_capability = _runtime_capability_payload(
        profile=profile,
        runtime_launch_expected=runtime_launch_expected,
        geometry_summary=geometry_summary,
        site_world_spec=site_world_spec,
        site_world_registration=site_world_registration,
        site_world_health=site_world_health,
    )

    env_checks: List[Dict[str, Any]] = [_env_check(resolved_env, name) for name in _COMMON_ENV_VARS]
    env_checks.append(_env_check(resolved_env, "PIPELINE_SYNC_REQUIRED", expected_value="true"))
    genai_present = bool(
        str(resolved_env.get("GOOGLE_GENAI_API_KEY") or "").strip()
        or str(resolved_env.get("GEMINI_API_KEY") or "").strip()
    )
    env_checks.append(
        _check(
            "gemini_api_key",
            genai_present,
            "GOOGLE_GENAI_API_KEY or GEMINI_API_KEY is configured"
            if genai_present
            else "GOOGLE_GENAI_API_KEY or GEMINI_API_KEY is missing",
            category="env",
        )
    )
    env_checks.append(_env_check(resolved_env, "PRIVACY_PIPELINE_ENABLED", expected_value="true"))
    env_checks.append(_env_check(resolved_env, "PRIVACY_FAIL_CLOSED", expected_value="true"))
    env_checks.extend(_env_check(resolved_env, name) for name in _PRIVACY_ENV_VARS)
    if runtime_launch_expected:
        env_checks.append(_env_check(resolved_env, "SITE_WORLD_RUNTIME_SERVICE_URL"))
        env_checks.append(_env_check(resolved_env, "SITE_WORLD_RUNTIME_SERVICE_API_KEY"))
    if profile in {"iphone_video_only", "meta_glasses", "android_video"}:
        env_checks.append(_env_check(resolved_env, "VIDEO_TO_WORLD_URL"))
        env_checks.append(_env_check(resolved_env, "VIDEO_TO_WORLD_RUNNER_TOKEN"))
    preview_provider = str(resolved_env.get("BLUEPRINT_PREVIEW_PROVIDER") or "world_labs").strip()
    if preview_provider == "world_labs":
        env_checks.append(_env_check(resolved_env, "WORLDLABS_API_KEY"))

    common_checks: List[Dict[str, Any]] = [
        _check_file(descriptor_path, name="capture_descriptor", detail="capture_descriptor.json exists"),
        _check_file(capture_root / "qa_report.json", name="qa_report", detail="qa_report.json exists"),
        _check(
            "gemini_review_succeeded",
            str(gemini_review.get("status") or "").strip().lower() == "succeeded",
            "Gemini capture fidelity review succeeded"
            if str(gemini_review.get("status") or "").strip().lower() == "succeeded"
            else f"Gemini review status is {gemini_review.get('status') or 'missing'}",
            category="status",
        ),
        _check(
            "privacy_completed",
            str(privacy_manifest.get("status") or "").strip().lower()
            in {
                "no_people_detected",
                "person_removed",
                "face_anonymized_fallback",
                "full_frame_redacted_local_proof",
            },
            "privacy produced buyer-safe walkthrough media"
            if str(privacy_manifest.get("status") or "").strip().lower()
            in {
                "no_people_detected",
                "person_removed",
                "face_anonymized_fallback",
                "full_frame_redacted_local_proof",
            }
            else f"privacy status is {privacy_manifest.get('status') or 'not_run'}",
            category="status",
        ),
        _check_file(pipeline_root / "qualification_summary.json", name="qualification_summary", detail="qualification_summary.json exists"),
        _check_file(pipeline_root / "capture_quality_summary.json", name="capture_quality_summary", detail="capture_quality_summary.json exists"),
        _check_file(pipeline_root / "rights_and_compliance_summary.json", name="rights_and_compliance_summary", detail="rights_and_compliance_summary.json exists"),
        _check_file(pipeline_root / "buyer_trust_score.json", name="buyer_trust_score", detail="buyer_trust_score.json exists"),
        _check_file(pipeline_root / "world_model_fit_summary.json", name="world_model_fit_summary", detail="world_model_fit_summary.json exists"),
        _check_file(pipeline_root / "provenance_summary.json", name="provenance_summary", detail="provenance_summary.json exists"),
        _check_file(pipeline_root / "gemini_capture_fidelity_review.json", name="gemini_capture_fidelity_review", detail="gemini_capture_fidelity_review.json exists"),
        _check_file(pipeline_root / "privacy_processing_manifest.json", name="privacy_processing_manifest", detail="privacy_processing_manifest.json exists"),
        _check_file(pipeline_root / "privacy_verification_report.json", name="privacy_verification_report", detail="privacy_verification_report.json exists"),
        _check_file(pipeline_root / "opportunity_handoff.json", name="opportunity_handoff", detail="opportunity_handoff.json exists"),
        _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists"),
        _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists"),
        _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists"),
        _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists"),
        _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists"),
        _check(
            "webapp_sync_succeeded",
            str(webapp_sync.get("status") or "").strip().lower() == "succeeded",
            "webapp sync succeeded"
            if str(webapp_sync.get("status") or "").strip().lower() == "succeeded"
            else f"webapp sync status is {webapp_sync.get('status') or 'missing'}",
            category="status",
        ),
    ]
    if runtime_launch_expected:
        blockers = {
            str(item).strip()
            for item in site_world_health.get("blockers", [])
            if str(item).strip()
        }
        common_checks.append(
            _check(
                "hosted_runtime_configured",
                "missing_runtime_service_url" not in blockers,
                "hosted runtime URL is configured"
                if "missing_runtime_service_url" not in blockers
                else "site world health is blocked by missing runtime service URL",
                category="status",
            )
        )

    common_passed = all(item["passed"] for item in common_checks if item["category"] != "env") and all(
        item["passed"] for item in env_checks
    )

    path_checks: List[Dict[str, Any]] = []
    external_alpha = {"status": "no_go", "reason": "unsupported_capture_path"}
    internal_alpha = {"status": "not_applicable", "reason": "not_meta_glasses"}

    if profile == "iphone_arkit_lidar":
        path_checks = [
            _check(
                "capture_source_iphone",
                descriptor.capture_source == "iphone",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check(
                "capture_modality_iphone_arkit_lidar",
                descriptor.capture_modality == "iphone_arkit_lidar",
                f"capture_modality is {descriptor.capture_modality or 'missing'}",
                category="path",
            ),
            _check(
                "arkit_bundle_complete",
                bool(descriptor.arkit_poses_uri and descriptor.arkit_intrinsics_uri and descriptor.arkit_depth_prefix_uri),
                "ARKit poses, intrinsics, and depth refs are present"
                if descriptor.arkit_poses_uri and descriptor.arkit_intrinsics_uri and descriptor.arkit_depth_prefix_uri
                else "ARKit bundle refs are incomplete",
                category="path",
            ),
            _check(
                "capture_mode_site_world_candidate",
                capture_mode_resolved == "site_world_candidate",
                f"capture_mode resolved to {capture_mode_resolved or 'missing'}",
                category="path",
            ),
            _check(
                "qa_report_passed",
                str(qa_report.get("status") or "").strip().lower() == "passed",
                f"qa_report status is {qa_report.get('status') or 'missing'}",
                category="path",
            ),
            _check_file(presentation_root / "presentation_bundle.json", name="presentation_bundle", detail="presentation_bundle.json exists", category="path"),
            _check_file(presentation_root / "presentation_world_manifest.json", name="presentation_world_manifest", detail="presentation_world_manifest.json exists", category="path"),
            _check_file(presentation_root / "runtime_demo_manifest.json", name="runtime_demo_manifest", detail="runtime_demo_manifest.json exists", category="path"),
            _check_file(eval_root / "hosted_session_runtime_manifest.json", name="hosted_session_runtime_manifest", detail="hosted_session_runtime_manifest.json exists", category="path"),
            _check_file(eval_root / "launchable_export_bundle.json", name="launchable_export_bundle", detail="launchable_export_bundle.json exists", category="path"),
            _check(
                "buyer_safe_walkthrough",
                bool(_present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")),
                "privacy produced buyer-safe walkthrough URI"
                if _present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")
                else "privacy did not produce buyer-safe walkthrough URI",
                category="path",
            ),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        external_alpha = {
            "status": "go" if common_passed and all(item["passed"] for item in path_checks) else "no_go",
            "reason": "all_common_and_iphone_checks_passed"
            if common_passed and all(item["passed"] for item in path_checks)
            else "iphone_alpha_requirements_not_met",
        }
    elif profile == "iphone_video_only":
        path_checks = [
            _check(
                "capture_source_iphone",
                descriptor.capture_source == "iphone",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check(
                "capture_modality_iphone_video_only",
                descriptor.capture_modality == "iphone_video_only",
                f"capture_modality is {descriptor.capture_modality or 'missing'}",
                category="path",
            ),
            _check_file(geometry_root / "geometry_manifest.json", name="geometry_manifest", detail="geometry_manifest.json exists", category="path"),
            _check_file(geometry_root / "geometry_summary.json", name="geometry_summary", detail="geometry_summary.json exists", category="path"),
            _check(
                "geometry_ready_for_world_model",
                bool(geometry_summary.get("ready_for_world_model")),
                "geometry is ready for native world-model conditioning"
                if bool(geometry_summary.get("ready_for_world_model"))
                else "geometry is not ready for native world-model conditioning",
                category="path",
            ),
            _check(
                "geometry_uses_real_video_to_world",
                str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used")),
                "geometry uses video_to_world without fallback"
                if str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used"))
                else "geometry fell back instead of using true video_to_world",
                category="path",
            ),
            _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists", category="path"),
            _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists", category="path"),
            _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists", category="path"),
            _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists", category="path"),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        external_alpha = {
            "status": "go" if common_passed and all(item["passed"] for item in path_checks) else "no_go",
            "reason": "all_common_and_iphone_video_only_checks_passed"
            if common_passed and all(item["passed"] for item in path_checks)
            else "iphone_video_only_alpha_requirements_not_met",
        }
    elif profile == "meta_glasses":
        path_checks = [
            _check(
                "capture_source_glasses",
                descriptor.capture_source == "glasses",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check_file(geometry_root / "geometry_manifest.json", name="geometry_manifest", detail="geometry_manifest.json exists", category="path"),
            _check_file(geometry_root / "geometry_summary.json", name="geometry_summary", detail="geometry_summary.json exists", category="path"),
            _check(
                "geometry_ready_for_world_model",
                bool(geometry_summary.get("ready_for_world_model")),
                "geometry is ready for world-model conditioning"
                if bool(geometry_summary.get("ready_for_world_model"))
                else "geometry is not ready for world-model conditioning",
                category="path",
            ),
            _check(
                "geometry_uses_real_video_to_world",
                str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used")),
                "geometry uses video_to_world without fallback"
                if str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used"))
                else "geometry fell back instead of using true video_to_world",
                category="path",
            ),
            _check_file(privacy_depth_root / "depth_manifest.json", name="privacy_depth_manifest", detail="privacy depth_manifest.json exists", category="path"),
            _check_file(privacy_depth_root / "confidence_manifest.json", name="privacy_confidence_manifest", detail="privacy confidence_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists", category="path"),
            _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists", category="path"),
            _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists", category="path"),
            _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists", category="path"),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        glasses_contract_ready = common_passed and all(item["passed"] for item in path_checks)
        external_alpha = {
            "status": "no_go",
            "reason": "glasses_requires_physical_device_and_operator_launch_evidence"
            if glasses_contract_ready
            else "glasses_external_alpha_requirements_not_met",
            "contract_status": "ready" if glasses_contract_ready else "blocked",
            "contract_reason": "all_common_and_glasses_checks_passed"
            if glasses_contract_ready
            else "glasses_contract_requirements_not_met",
        }
        internal_alpha = {
            "status": "go" if glasses_contract_ready else "no_go",
            "reason": "all_common_and_glasses_checks_passed"
            if glasses_contract_ready
            else "glasses_internal_alpha_requirements_not_met",
        }
    elif profile == "android_video":
        path_checks = [
            _check(
                "capture_source_android",
                descriptor.capture_source == "android",
                f"capture_source is {descriptor.capture_source or 'missing'}",
                category="path",
            ),
            _check_file(geometry_root / "geometry_manifest.json", name="geometry_manifest", detail="geometry_manifest.json exists", category="path"),
            _check_file(geometry_root / "geometry_summary.json", name="geometry_summary", detail="geometry_summary.json exists", category="path"),
            _check(
                "geometry_ready_for_world_model",
                bool(geometry_summary.get("ready_for_world_model")),
                "geometry is ready for native world-model conditioning"
                if bool(geometry_summary.get("ready_for_world_model"))
                else "geometry is not ready for native world-model conditioning",
                category="path",
            ),
            _check(
                "geometry_uses_real_video_to_world",
                str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used")),
                "geometry uses video_to_world without fallback"
                if str(geometry_summary.get("geometry_source") or "").strip() == "video_to_world"
                and not bool(geometry_summary.get("fallback_used"))
                else "geometry fell back instead of using true video_to_world",
                category="path",
            ),
            _check_file(scene_memory_root / "scene_memory_manifest.json", name="scene_memory_manifest", detail="scene_memory_manifest.json exists", category="path"),
            _check_file(scene_memory_root / "conditioning_bundle.json", name="conditioning_bundle", detail="conditioning_bundle.json exists", category="path"),
            _check_file(eval_root / "site_world_spec.json", name="site_world_spec", detail="site_world_spec.json exists", category="path"),
            _check_file(eval_root / "site_world_registration.json", name="site_world_registration", detail="site_world_registration.json exists", category="path"),
            _check_file(eval_root / "site_world_health.json", name="site_world_health", detail="site_world_health.json exists", category="path"),
            _check(
                "native_runtime_capability_ready",
                runtime_capability["status"] == "ready",
                "native runtime capability artifacts are ready"
                if runtime_capability["status"] == "ready"
                else f"native runtime capability is blocked: {', '.join(runtime_capability['blockers']) or 'unknown'}",
                category="path",
            ),
        ]
        android_contract_ready = common_passed and all(item["passed"] for item in path_checks)
        external_alpha = {
            "status": "no_go",
            "reason": "android_requires_physical_device_and_operator_launch_evidence"
            if android_contract_ready
            else "android_external_alpha_requirements_not_met",
            "contract_status": "ready" if android_contract_ready else "blocked",
            "contract_reason": "all_common_and_android_checks_passed"
            if android_contract_ready
            else "android_contract_requirements_not_met",
        }
        internal_alpha = {
            "status": "go" if android_contract_ready else "no_go",
            "reason": "all_common_and_android_checks_passed"
            if android_contract_ready
            else "android_internal_alpha_requirements_not_met",
        }

    external_alpha_go = str(external_alpha.get("status") or "").strip().lower() == "go"
    internal_alpha_go = str(internal_alpha.get("status") or "").strip().lower() == "go"
    if external_alpha_go:
        device_alpha_profile = {
            "status": "ready_for_external_alpha",
            "reason": external_alpha.get("reason"),
        }
    elif internal_alpha_go:
        device_alpha_profile = {
            "status": "internal_only",
            "reason": internal_alpha.get("reason"),
        }
    else:
        device_alpha_profile = {
            "status": "blocked",
            "reason": external_alpha.get("reason") or internal_alpha.get("reason"),
        }

    failed_checks = [
        item["name"]
        for item in [*env_checks, *common_checks, *path_checks]
        if not item["passed"]
    ]
    no_go_reasons = [item["detail"] for item in [*env_checks, *common_checks, *path_checks] if not item["passed"]]

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "capture_source": descriptor.capture_source,
        "capture_modality": descriptor.capture_modality,
        "capture_mode": capture_mode_resolved,
        "profile": profile,
        "runtime_launch_expected": runtime_launch_expected,
        "runtime_capability": runtime_capability,
        "launch_market_readiness": {
            "contract_ready": bool(external_alpha_go or internal_alpha_go),
            "internal_pilot_ready": bool(internal_alpha_go),
            "external_market_ready": bool(external_alpha_go),
            "site_faithful_market_ready": bool(external_alpha_go and profile == "iphone_arkit_lidar"),
            "claim_boundary": (
                "external_market"
                if external_alpha_go
                else "internal_or_blocked_until_live_operator_evidence"
            ),
        },
        "environment_checks": env_checks,
        "common_checks": common_checks,
        "path_checks": path_checks,
        "verdicts": {
            "external_alpha": external_alpha,
            "internal_experimental_alpha": internal_alpha,
        },
        "device_alpha_profile": device_alpha_profile,
        "common_status": "passed" if common_passed else "failed",
        "path_status": "passed" if path_checks and all(item["passed"] for item in path_checks) else "failed",
        "failed_checks": failed_checks,
        "no_go_reasons": no_go_reasons,
        "service_snapshot": {
            "webapp_sync_status": webapp_sync.get("status"),
            "privacy_status": privacy_manifest.get("status") or "not_run",
            "runtime_health_status": site_world_health.get("status") or "missing",
            "runtime_launchable": bool(site_world_health.get("launchable")),
        },
    }


def write_alpha_readiness_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    payload = build_alpha_readiness_summary(capture_root=capture_root, env=env)
    write_json(capture_root / "pipeline" / "alpha_readiness_summary.json", payload)
    return payload


def build_launch_gate_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    resolved_env = dict(os.environ if env is None else env)
    descriptor = CaptureDescriptor.from_dict(_read_json_object(capture_root / "capture_descriptor.json"))
    raw_manifest = _read_json_object(capture_root / "raw" / "manifest.json")
    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"

    alpha_summary = write_alpha_readiness_summary(capture_root=capture_root, env=resolved_env)
    opportunity_handoff = _read_json_object(pipeline_root / "opportunity_handoff.json")
    qualification_record = _read_json_object(pipeline_root / "qualification_record.json")
    scorecard = _read_json_object(pipeline_root / "capture_qa_scorecard.json")
    privacy_manifest = _read_json_object(pipeline_root / "privacy_processing_manifest.json")
    payout_recommendation = _read_json_object(pipeline_root / "capturer_payout_recommendation.json")
    launchable_export_bundle = _read_json_object(eval_root / "launchable_export_bundle.json")
    webapp_sync = _read_json_object(pipeline_root / "webapp_sync_result.json")
    operator_evidence = _read_json_object(pipeline_root / "operator_launch_evidence.json")
    rights_review = _read_json_object(pipeline_root / "rights_provenance_review.json")
    provenance_summary = _read_json_object(pipeline_root / "provenance_summary.json")
    recapture_requirements = _read_json_object(pipeline_root / "recapture_requirements.json")
    worldlabs_input_audit = _read_json_object(pipeline_root / "worldlabs_input_audit.json")
    authoritative_qualification_state = derive_webapp_qualification_state(
        readiness_state=qualification_record.get("readiness_state"),
        completeness_status=scorecard.get("completeness_status"),
    )

    site_submission_id = (
        descriptor.site_submission_id
        or str(opportunity_handoff.get("site_submission_id") or "").strip()
    )
    buyer_request_id = (
        descriptor.buyer_request_id
        or str(opportunity_handoff.get("buyer_request_id") or "").strip()
    )
    capture_job_id = (
        descriptor.capture_job_id
        or str(opportunity_handoff.get("capture_job_id") or "").strip()
    )
    # Payout readiness is a revenue-share hook: it must come from an explicit
    # recommendation decision, never inferred from the mere presence of a quote
    # or a recommended amount (that would fabricate payout readiness).
    payout_eligible = payout_recommendation.get("eligible_for_payout") is True
    upstream_id_failures = {
        key: reason
        for key, reason in upstream_link_id_failures(
            {
                "site_submission_id": site_submission_id,
                "buyer_request_id": buyer_request_id,
                "capture_job_id": capture_job_id,
                "scene_id": descriptor.scene_id,
                "capture_id": descriptor.capture_id,
            }
        ).items()
        if key in {"site_submission_id", "buyer_request_id", "capture_job_id"}
    }
    sync_verification = _webapp_sync_verification(webapp_sync, env=resolved_env)

    # Consent/rights, raw-bypass, provenance, and recapture evidence for the
    # buyer-ready verdict. Absent artifacts fail these checks: missing evidence
    # is never launch evidence.
    rights_review_status = str(rights_review.get("status") or "").strip().lower()
    rights_block = rights_review.get("rights") if isinstance(rights_review.get("rights"), Mapping) else {}
    rights_consent_status = str(rights_block.get("consent_status") or "").strip().lower()
    descriptor_metadata = descriptor.metadata if isinstance(descriptor.metadata, Mapping) else {}
    descriptor_labeling = (
        descriptor_metadata.get("worldlabs_input_labeling")
        if isinstance(descriptor_metadata.get("worldlabs_input_labeling"), Mapping)
        else {}
    )
    audit_labeling = (
        worldlabs_input_audit.get("input_labeling")
        if isinstance(worldlabs_input_audit.get("input_labeling"), Mapping)
        else {}
    )
    raw_worldlabs_bypass_used = bool(
        worldlabs_input_audit.get("raw_video_bypass_used")
        or audit_labeling.get("raw_video_bypass_used")
        or descriptor_labeling.get("raw_video_bypass_used")
    )
    provenance_record = (
        provenance_summary.get("record")
        if isinstance(provenance_summary.get("record"), Mapping)
        else {}
    )
    provenance_grounded = (
        str(provenance_summary.get("status") or "").strip().lower() == "grounded"
        and bool(provenance_record.get("canonical_truth"))
    )
    recapture_required = (
        bool(recapture_requirements.get("required")) if recapture_requirements else True
    )
    profile = str(alpha_summary.get("profile") or "unsupported")
    external_alpha = alpha_summary.get("verdicts", {}).get("external_alpha", {})
    internal_alpha = alpha_summary.get("verdicts", {}).get("internal_experimental_alpha", {})
    device_alpha_profile = alpha_summary.get("device_alpha_profile", {})
    runtime_capability = alpha_summary.get("runtime_capability", {})
    external_alpha_go = str(external_alpha.get("status") or "").strip().lower() == "go"
    internal_alpha_go = str(internal_alpha.get("status") or "").strip().lower() == "go"
    # Fail closed: a bundle without an explicit ready status is unproven
    # evidence, not a ready bundle. Legacy statusless bundle files must be
    # re-exported before they can pass the buyer-fulfillment gate.
    launchable_bundle_ready = bool(
        launchable_export_bundle
        and str(launchable_export_bundle.get("status") or "").strip().lower() in {"ready", "launch_ready"}
    )

    stage_checks = [
        _check(
            "inbound_request_linked",
            bool(site_submission_id) and "site_submission_id" not in upstream_id_failures,
            f"site_submission_id is {site_submission_id}"
            if site_submission_id and "site_submission_id" not in upstream_id_failures
            else (
                f"site_submission_id is not a real WebApp record: {upstream_id_failures.get('site_submission_id')}"
                if site_submission_id
                else "site_submission_id is missing from the captured opportunity handoff"
            ),
            category="launch_gate",
        ),
        _check(
            "approved_marketplace_capture_job_linked",
            bool(capture_job_id) and "capture_job_id" not in upstream_id_failures,
            f"capture_job_id is {capture_job_id}"
            if capture_job_id and "capture_job_id" not in upstream_id_failures
            else (
                f"capture_job_id is not a real WebApp record: {upstream_id_failures.get('capture_job_id')}"
                if capture_job_id
                else "capture_job_id is missing from the captured job linkage"
            ),
            category="launch_gate",
        ),
        _check(
            "buyer_request_linked",
            bool(buyer_request_id) and "buyer_request_id" not in upstream_id_failures,
            f"buyer_request_id is {buyer_request_id}"
            if buyer_request_id and "buyer_request_id" not in upstream_id_failures
            else (
                f"buyer_request_id is not a real WebApp record: {upstream_id_failures.get('buyer_request_id')}"
                if buyer_request_id
                else "buyer_request_id is missing from the buyer request linkage"
            ),
            category="launch_gate",
        ),
        _check(
            "mobile_claim_context_captured",
            bool(descriptor.capture_source and descriptor.quoted_payout_cents is not None),
            (
                f"capture source {descriptor.capture_source} retained quoted payout {descriptor.quoted_payout_cents}"
                if descriptor.capture_source and descriptor.quoted_payout_cents is not None
                else "capture descriptor is missing source or quoted payout context"
            ),
            category="launch_gate",
        ),
        _check_file(
            capture_root / "raw" / "capture_upload_complete.json",
            name="mobile_upload_completed",
            detail="raw/capture_upload_complete.json exists",
            category="launch_gate",
        ),
        _check(
            "qualification_authoritative",
            authoritative_qualification_state in {"qualified_ready", "qualified_risky"}
            or external_alpha_go
            or internal_alpha_go,
            (
                f"qualification_state is {authoritative_qualification_state or 'not_ready_yet'} and alpha verdict is enforced"
                if authoritative_qualification_state in {"qualified_ready", "qualified_risky"}
                or external_alpha_go
                or internal_alpha_go
                else "authoritative qualification_state did not reach a launchable verdict"
            ),
            category="launch_gate",
        ),
        _check(
            "privacy_safe_buyer_media_ready",
            bool(_present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")),
            "privacy manifest includes buyer-safe walkthrough media"
            if _present_value(privacy_manifest, "privacy_processed_video_uri", "world_model_video_uri")
            else "privacy-safe walkthrough media is missing",
            category="launch_gate",
        ),
        _check(
            "webapp_sync_completed",
            bool(sync_verification["verified"]),
            f"webapp sync succeeded against verified upstream records at {sync_verification['synced_at']}"
            if sync_verification["verified"]
            else "webapp sync result is not verified launch evidence: "
            + ", ".join(sync_verification["failures"]),
            category="launch_gate",
        ),
        _check(
            "buyer_fulfillment_bundle_ready",
            launchable_bundle_ready,
            "launchable_export_bundle.json is ready for buyer fulfillment"
            if launchable_bundle_ready
            else "launchable_export_bundle.json is missing or not ready",
            category="launch_gate",
        ),
        _check(
            "native_runtime_capability_ready",
            str(runtime_capability.get("status") or "").strip().lower() == "ready",
            "native runtime capability is ready"
            if str(runtime_capability.get("status") or "").strip().lower() == "ready"
            else f"native runtime capability is blocked: {', '.join(runtime_capability.get('blockers') or []) or 'unknown'}",
            category="launch_gate",
        ),
        _check(
            "capturer_payout_transition_ready",
            payout_eligible,
            "capturer payout recommendation explicitly marks this capture payout-eligible"
            if payout_eligible
            else "capturer payout recommendation is missing or does not explicitly mark payout eligibility",
            category="launch_gate",
        ),
        _check(
            "rights_provenance_review_cleared",
            rights_review_status == "cleared",
            f"rights provenance review is cleared (consent_status={rights_consent_status or 'unknown'})"
            if rights_review_status == "cleared"
            else f"rights provenance review status is {rights_review_status or 'missing'}; "
            "site rights and consent packet is not launch evidence",
            category="launch_gate",
        ),
        _check(
            "raw_worldlabs_bypass_not_used",
            not raw_worldlabs_bypass_used,
            "world-model input derives from privacy-safe media, not the raw walkthrough"
            if not raw_worldlabs_bypass_used
            else "raw World Labs bypass was used for this capture's world-model input; "
            "raw-derived outputs are never buyer-ready",
            category="launch_gate",
        ),
        _check(
            "provenance_summary_grounded",
            provenance_grounded,
            "provenance summary is grounded in canonical capture truth"
            if provenance_grounded
            else f"provenance summary status is {provenance_summary.get('status') or 'missing'}; "
            "package provenance is not grounded",
            category="launch_gate",
        ),
        _check(
            "recapture_not_required",
            not recapture_required,
            "recapture requirements are recorded and no recapture is required"
            if not recapture_required
            else (
                "recapture is required: "
                + ", ".join(
                    _string_list_or_default(
                        recapture_requirements.get("missing_evidence"),
                        recapture_requirements.get("recommendations"),
                    )
                )
                if recapture_requirements
                else "recapture_requirements.json is missing; recapture state is unknown"
            ),
            category="launch_gate",
        ),
    ]

    all_stage_checks_passed = all(item["passed"] for item in stage_checks)

    if all_stage_checks_passed and external_alpha_go:
        contract_status = "external_beta_contract_ready"
    elif all_stage_checks_passed and internal_alpha_go:
        contract_status = "internal_only_contract_ready"
    else:
        contract_status = "blocked"

    operator_evidence_file_errors = _operator_evidence_file_errors(operator_evidence)
    industrial_authorization_required = _industrial_authorization_required(
        descriptor=descriptor,
        raw_manifest=raw_manifest,
        rights_review=rights_review,
    )
    operator_required_checks = [
        _operator_required_check(
            check_id="legal_consent_posture_signoff",
            scope="legal",
            required_evidence=(
                "Legal/EHS signature over the current capture consent, rights, "
                "redaction, and delivery posture."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="operator_dpa_data_processing_terms",
            scope="legal_privacy_ops",
            required_evidence=(
                "Operator DPA or equivalent data-processing terms covering "
                "retention policy, subprocessor list, and access-audit terms "
                "for delivered packages and hosted review access."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="cross_border_data_residency_posture",
            scope="legal_privacy_ops",
            required_evidence=(
                "Either a signed US-only beta participant/site scope, or signed "
                "international-transfer terms with SCC/DPA, transfer-impact, "
                "subprocessor, and residency posture evidence."
            ),
            operator_evidence=operator_evidence,
        ),
        *(
            [
                _operator_required_check(
                    check_id=_INDUSTRIAL_AUTHORIZATION_CHECK_ID,
                    scope="legal_ehs_industrial_site",
                    required_evidence=(
                        "Industrial site authorization signed by a site "
                        "authorizer, EHS/safety sign-off, worker-PII or works "
                        "council posture, NDA/proprietary-data terms, PPE and "
                        "escort acknowledgement, and restricted-zone controls "
                        "for forklift lanes, LOTO, machine guards, and other "
                        "non-public industrial areas."
                    ),
                    operator_evidence=operator_evidence,
                )
            ]
            if industrial_authorization_required
            else []
        ),
        _operator_required_check(
            check_id="paperclip_ops_relay_secret_rotation",
            scope="ops_security",
            required_evidence=(
                "Cloud Secret Manager version or equivalent rotation record, "
                "plus redeploy evidence for the Paperclip ops relay secret."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id=f"{descriptor.capture_source or 'unknown'}_real_device_claim_flow",
            scope="device",
            required_evidence=(
                "Screenshot or screen recording showing discovery, claim, and "
                "upload completion for the same capture_job_id."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="buyer_payment_settlement",
            scope="payments",
            required_evidence=(
                "Stripe payment intent or checkout session proving a buyer "
                "purchase completed for the launch SKU."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="capturer_payout_settlement",
            scope="payouts",
            required_evidence=(
                "Live Stripe connected account state, live payout evidence, "
                "webhook reconciliation, and matching creator capture ledger "
                "entry for the approved capture."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="stripe_connected_account_live_readiness",
            scope="payouts",
            required_evidence=(
                "Backend /v1/stripe/account response showing "
                "provider_state_checked=true, provider_mode=live, "
                "live_provider_ready=true, payouts_enabled=true, and no "
                "blocking requirements."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="payout_exception_monitor_live",
            scope="ops",
            required_evidence=(
                "Live monitor or query evidence for payout.failed, "
                "payout.canceled, disbursement_failed, and overdue "
                "finance_review records."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="identity_kyc_provider_decision",
            scope="identity",
            required_evidence=(
                "Document whether Stripe Connect is the only near-term KYC "
                "path or provide account/env proof for Persona, Stripe "
                "Identity, or another identity provider."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="background_check_provider_decision",
            scope="background_checks",
            required_evidence=(
                "Document that no Checkr/background-check provider is "
                "integrated yet, or provide provider account/env proof before "
                "making screening claims."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="human_finance_review_owner",
            scope="ops",
            required_evidence=(
                "Named human finance owner and review queue/route for payout "
                "exceptions before any live payout execution flag is enabled."
            ),
            operator_evidence=operator_evidence,
        ),
        _operator_required_check(
            check_id="buyer_artifact_access",
            scope="buyer_access",
            required_evidence=(
                "Authenticated buyer session proving artifact or fulfillment "
                "access resolves after purchase."
            ),
            operator_evidence=operator_evidence,
        ),
    ]
    operator_evidence_blockers = [
        str(check["blocker"])
        for check in operator_required_checks
        if not check["passed"] and check.get("blocker")
    ]
    operator_evidence_verified = not operator_evidence_blockers
    external_beta_operator_evidence_required = contract_status == "external_beta_contract_ready"
    operator_evidence_status = {
        "schema_version": _OPERATOR_LAUNCH_EVIDENCE_SCHEMA_VERSION,
        "status": "verified" if operator_evidence_verified else "blocked",
        "required_for_external_beta": external_beta_operator_evidence_required,
        "evidence_file": _OPERATOR_LAUNCH_EVIDENCE_RELATIVE_PATH,
        "evidence_file_present": bool(operator_evidence),
        "schema_errors": operator_evidence_file_errors,
        "required_count": len(operator_required_checks),
        "verified_count": sum(1 for check in operator_required_checks if check["passed"]),
        "blockers": operator_evidence_blockers,
        "claim_boundary": (
            "operator_evidence_is_live_human_or_external_service_proof_not_automation"
        ),
    }

    if external_beta_operator_evidence_required:
        source_status = (
            "external_beta_live_evidence_ready"
            if operator_evidence_verified
            else "automated_contracts_passed_manual_ops_required"
        )
    else:
        source_status = contract_status

    justified_claims = [
        "Qualification and readiness remain enforced support gates; raw capture and package provenance remain authoritative.",
        "Privacy-safe walkthrough media is the buyer-facing artifact; runtime or world-model outputs stay downstream.",
    ]
    if all_stage_checks_passed:
        justified_claims.extend(
            [
                "Inbound request linkage, marketplace job linkage, upload completion, qualification, privacy processing, and WebApp sync are all contract-verified.",
                "Launchable export packaging exists for buyer fulfillment or buyer access flows.",
                "Capturer payout recommendation is contract-present; live Stripe/provider readiness remains an operator payment checklist item.",
            ]
        )
    if contract_status == "external_beta_contract_ready":
        justified_claims.append(
            "This source path is externally marketable for the paid marketplace beta at contract level once operator checks pass."
        )
    elif contract_status == "internal_only_contract_ready":
        justified_claims.append(
            "This source path is suitable for internal beta operations, qualification, privacy-safe previews, and workflow orchestration."
        )

    not_justified_claims = [
        "Do not claim runtime or world-model outputs can override raw capture, rights, privacy, provenance, or package truth.",
        "Do not claim strong site-faithful world-model quality; only native runtime capability and downstream packaging are proven here.",
        "Do not claim live buyer payments or live capturer payouts are proven until the operator payment checklist is completed.",
        "Do not claim Stripe, identity/KYC, background-check, instant-pay, or payout-timing readiness from backend URL, publishable key, or mocked tests.",
        "Do not claim real-device discovery and claim UX is proven in production until the device checklist is completed.",
    ]
    if not external_alpha_go:
        not_justified_claims.append(
            "Do not market this source as externally launch-ready while alpha readiness remains blocked."
        )

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "capture_source": descriptor.capture_source,
        "capture_modality": descriptor.capture_modality,
        "profile": profile,
        "overall_status": source_status,
        "device_alpha_profile": {
            "status": device_alpha_profile.get("status"),
            "reason": device_alpha_profile.get("reason"),
        },
        "runtime_capability": runtime_capability,
        "qualification_policy": {
            "authoritative_truth": True,
            "detail": "Raw capture, rights, privacy, provenance, and package artifacts are authoritative; qualification and readiness are enforced support gates.",
        },
        "stage_checks": stage_checks,
        "source_acceptance": {
            "status": source_status,
            "contract_status": contract_status,
            "operator_evidence_status": operator_evidence_status["status"],
            "industrial_authorization_required": industrial_authorization_required,
            "industrial_site_type_candidates": _site_type_candidates(
                descriptor=descriptor,
                raw_manifest=raw_manifest,
                rights_review=rights_review,
            ),
            "external_alpha_status": external_alpha.get("status"),
            "internal_alpha_status": internal_alpha.get("status"),
            "alpha_reason": external_alpha.get("reason") or internal_alpha.get("reason"),
        },
        "launch_claims": {
            "justified": justified_claims,
            "not_justified": not_justified_claims,
        },
        "operator_evidence_status": operator_evidence_status,
        "operator_required_checks": operator_required_checks,
    }


def write_launch_gate_summary(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    payload = build_launch_gate_summary(capture_root=capture_root, env=env)
    write_json(capture_root / "pipeline" / "launch_gate_summary.json", payload)
    return payload


def sync_webapp_evaluation_prep(
    *,
    capture_root: Path,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    resolved_env = dict(os.environ if env is None else env)
    descriptor_payload = _read_json_object(capture_root / "capture_descriptor.json")
    descriptor = CaptureDescriptor.from_dict(descriptor_payload)
    parsed = parse_gs_uri(str(descriptor.raw_prefix_uri))
    bucket = parsed.bucket
    pipeline_prefix = to_pipeline_prefix(descriptor.scene_id, descriptor.capture_id)
    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"
    opportunity_handoff = _read_json_object(pipeline_root / "opportunity_handoff.json")
    qualification_record = _read_json_object(pipeline_root / "qualification_record.json")
    scorecard = _read_json_object(pipeline_root / "capture_qa_scorecard.json")
    privacy_manifest = _read_json_object(pipeline_root / "privacy_processing_manifest.json")
    provider_run_manifest = _read_json_object(pipeline_root / "provider_run_manifest.json")
    site_world_health = _read_json_object(eval_root / "site_world_health.json")
    evaluation_prep_summary = _read_json_object(eval_root / "evaluation_prep_summary.json")
    rights_provenance_review = optional_read_json(pipeline_root / "rights_provenance_review.json") or {}
    site_package_manifest = optional_read_json(eval_root / "site_package_manifest.json") or {}
    proof_pack_manifest = optional_read_json(eval_root / "proof_pack_manifest.json") or {}
    hosted_review_readiness = optional_read_json(eval_root / "hosted_review_readiness.json") or {}
    proof_path_status = optional_read_json(eval_root / "proof_path_status.json") or {}
    delivery_manifest = optional_read_json(pipeline_root / "delivery_manifest.json") or {}
    signed_access_manifest = optional_read_json(pipeline_root / "signed_access_manifest.json") or {}
    alpha_summary = write_alpha_readiness_summary(capture_root=capture_root, env=resolved_env)
    launch_gate_summary = write_launch_gate_summary(capture_root=capture_root, env=resolved_env)
    site_submission_id = (
        descriptor.site_submission_id
        or str(opportunity_handoff.get("site_submission_id") or "").strip()
    )
    buyer_request_id = (
        descriptor.buyer_request_id
        or str(opportunity_handoff.get("buyer_request_id") or "").strip()
    )
    capture_job_id = (
        descriptor.capture_job_id
        or str(opportunity_handoff.get("capture_job_id") or "").strip()
    )

    qualification_state = derive_webapp_qualification_state(
        readiness_state=qualification_record.get("readiness_state"),
        completeness_status=scorecard.get("completeness_status"),
    )
    opportunity_state = derive_webapp_opportunity_state(qualification_state=qualification_state)

    def _artifact_if_exists(relative_path: str) -> Optional[str]:
        path = pipeline_root / relative_path
        if path.is_file():
            return _uri(bucket, pipeline_prefix, relative_path)
        return None

    delivery_artifact_uri = (
        _present_value(
            signed_access_manifest,
            "artifact_uri",
            "post_training_data_package_uri",
            "package_uri",
        )
        or _present_value(
            delivery_manifest,
            "artifact_uri",
            "post_training_data_package_uri",
            "package_uri",
        )
        or _artifact_if_exists("archives/post_training_data_package.tar.gz")
    )

    artifacts = {
        "qualification_summary_uri": _artifact_if_exists("qualification_summary.json"),
        "capture_quality_summary_uri": _artifact_if_exists("capture_quality_summary.json"),
        "rights_and_compliance_summary_uri": _artifact_if_exists("rights_and_compliance_summary.json"),
        "buyer_trust_score_uri": _artifact_if_exists("buyer_trust_score.json"),
        "capturer_payout_recommendation_uri": _artifact_if_exists("capturer_payout_recommendation.json"),
        "world_model_fit_summary_uri": _artifact_if_exists("world_model_fit_summary.json"),
        "provenance_summary_uri": _artifact_if_exists("provenance_summary.json"),
        "gemini_capture_fidelity_review_uri": _artifact_if_exists("gemini_capture_fidelity_review.json"),
        "privacy_processing_manifest_uri": _artifact_if_exists("privacy_processing_manifest.json"),
        "privacy_verification_report_uri": _artifact_if_exists("privacy_verification_report.json"),
        "webapp_sync_result_uri": _artifact_if_exists("webapp_sync_result.json"),
        "launch_gate_summary_uri": _artifact_if_exists("launch_gate_summary.json"),
        "post_training_data_package_uri": delivery_artifact_uri,
        "delivery_manifest_uri": _artifact_if_exists("delivery_manifest.json"),
        "signed_access_manifest_uri": _artifact_if_exists("signed_access_manifest.json"),
        "preview_manifest_uri": _artifact_if_exists("preview_manifest.json"),
        "worldlabs_request_manifest_uri": _artifact_if_exists("worldlabs_request_manifest.json"),
        "worldlabs_operation_manifest_uri": _artifact_if_exists("worldlabs_operation_manifest.json"),
        "worldlabs_world_manifest_uri": _artifact_if_exists("worldlabs_world_manifest.json"),
        "scene_memory_manifest_uri": _artifact_if_exists("scene_memory/scene_memory_manifest.json"),
        "conditioning_bundle_uri": _artifact_if_exists("scene_memory/conditioning_bundle.json"),
        "preview_simulation_manifest_uri": _artifact_if_exists("preview_simulation/preview_simulation_manifest.json"),
        "presentation_bundle_uri": _artifact_if_exists("presentation_world/presentation_bundle.json"),
        "presentation_world_manifest_uri": _artifact_if_exists("presentation_world/presentation_world_manifest.json"),
        "runtime_demo_manifest_uri": _artifact_if_exists("presentation_world/runtime_demo_manifest.json"),
        "authoritative_runtime_render_manifest_uri": _artifact_if_exists(
            "presentation_world/authoritative_runtime_render_manifest.json"
        ),
        "site_world_spec_uri": _artifact_if_exists("evaluation_prep/site_world_spec.json"),
        "site_world_registration_uri": _artifact_if_exists("evaluation_prep/site_world_registration.json"),
        "site_world_health_uri": _artifact_if_exists("evaluation_prep/site_world_health.json"),
        "hosted_session_runtime_manifest_uri": _artifact_if_exists("evaluation_prep/hosted_session_runtime_manifest.json"),
        "launchable_export_bundle_uri": _artifact_if_exists("evaluation_prep/launchable_export_bundle.json"),
        "evaluation_prep_manifest_uri": _artifact_if_exists("evaluation_prep/evaluation_prep_manifest.json"),
        "evaluation_prep_summary_uri": _artifact_if_exists("evaluation_prep/evaluation_prep_summary.json"),
        "site_package_manifest_uri": _artifact_if_exists("evaluation_prep/site_package_manifest.json"),
        "proof_pack_manifest_uri": _artifact_if_exists("evaluation_prep/proof_pack_manifest.json"),
        "hosted_review_readiness_uri": _artifact_if_exists("evaluation_prep/hosted_review_readiness.json"),
        "proof_path_status_uri": _artifact_if_exists("evaluation_prep/proof_path_status.json"),
        "rights_provenance_review_uri": _artifact_if_exists("rights_provenance_review.json"),
        "geometry_manifest_uri": _artifact_if_exists("geometry/geometry_manifest.json"),
        "geometry_summary_uri": _artifact_if_exists("geometry/geometry_summary.json"),
        "privacy_depth_manifest_uri": _artifact_if_exists("privacy_depth/depth_manifest.json"),
        "privacy_confidence_manifest_uri": _artifact_if_exists("privacy_depth/confidence_manifest.json"),
        "alpha_readiness_summary_uri": _artifact_if_exists("alpha_readiness_summary.json"),
        "worldlabs_launch_url": _present_value(
            provider_run_manifest,
            "worldlabs_launch_url",
            "preview_launch_url",
            "launch_url",
        ),
        "privacy_processed_video_uri": _present_value(privacy_manifest, "privacy_processed_video_uri"),
        "world_model_video_uri": _present_value(privacy_manifest, "world_model_video_uri"),
    }
    derived_assets = {
        key: value
        for key, value in {
            "scene_memory": {
                "status": str(_read_json_object(pipeline_root / "scene_memory" / "scene_memory_readiness.json").get("status") or "missing"),
                "manifest_uri": artifacts.get("scene_memory_manifest_uri"),
                "artifact_uri": artifacts.get("conditioning_bundle_uri"),
            }
            if artifacts.get("scene_memory_manifest_uri")
            else None,
            "presentation_world": {
                "status": str(_read_json_object(pipeline_root / "presentation_world" / "presentation_world_manifest.json").get("status") or "missing"),
                "manifest_uri": artifacts.get("presentation_world_manifest_uri"),
                "artifact_uri": artifacts.get("presentation_bundle_uri"),
            }
            if artifacts.get("presentation_world_manifest_uri")
            else None,
            "site_world_package": {
                "status": str(evaluation_prep_summary.get("site_world_status") or site_world_health.get("status") or "missing"),
                "manifest_uri": artifacts.get("evaluation_prep_manifest_uri"),
                "artifact_uri": artifacts.get("site_world_spec_uri"),
            }
            if artifacts.get("site_world_spec_uri")
            else None,
            "hosted_runtime": {
                "status": str(site_world_health.get("status") or "missing"),
                "manifest_uri": artifacts.get("hosted_session_runtime_manifest_uri"),
                "artifact_uri": artifacts.get("site_world_registration_uri"),
            }
            if artifacts.get("hosted_session_runtime_manifest_uri")
            else None,
        }.items()
        if value
    }
    evaluation_readiness = {
        "capture_source": descriptor.capture_source,
        "capture_modality": descriptor.capture_modality,
        "device_alpha_profile_status": alpha_summary.get("device_alpha_profile", {}).get("status"),
        "device_alpha_profile_reason": alpha_summary.get("device_alpha_profile", {}).get("reason"),
        "qualification_state": qualification_state,
        "opportunity_state": opportunity_state,
        "native_world_model_status": str(
            evaluation_prep_summary.get("native_world_model_status")
            or ("primary_ready" if artifacts.get("site_world_spec_uri") and artifacts.get("scene_memory_manifest_uri") else "not_ready")
        ),
        "native_world_model_primary": bool(
            evaluation_prep_summary.get("native_world_model_primary")
            if evaluation_prep_summary.get("native_world_model_primary") is not None
            else artifacts.get("site_world_spec_uri") and artifacts.get("scene_memory_manifest_uri")
        ),
        "provider_fallback_preview_status": (
            str(evaluation_prep_summary.get("provider_fallback_preview_status"))
            if evaluation_prep_summary.get("provider_fallback_preview_status") is not None
            else "fallback_available"
            if artifacts.get("preview_simulation_manifest_uri") or artifacts.get("world_model_video_uri")
            else "not_requested"
        ),
        "provider_fallback_only": bool(
            evaluation_prep_summary.get("provider_fallback_only")
            if evaluation_prep_summary.get("provider_fallback_only") is not None
            else not bool(
                evaluation_prep_summary.get("native_world_model_primary")
                if evaluation_prep_summary.get("native_world_model_primary") is not None
                else artifacts.get("site_world_spec_uri") and artifacts.get("scene_memory_manifest_uri")
            )
            and bool(artifacts.get("preview_simulation_manifest_uri") or artifacts.get("world_model_video_uri"))
        ),
        "runtime_health_status": site_world_health.get("status"),
        "runtime_launchable": bool(site_world_health.get("launchable")),
        "runtime_registration_status": site_world_health.get("runtime_registration_status"),
        "native_runtime_capability_state": alpha_summary.get("runtime_capability", {}).get("status"),
        "native_runtime_capability": alpha_summary.get("runtime_capability"),
        "evaluation_prep_summary": evaluation_prep_summary,
        "alpha_readiness": alpha_summary,
        "launch_gate_summary": launch_gate_summary,
        "rights_provenance_review": rights_provenance_review,
        # PIPE-02: surface the rights/privacy VERDICT so the WebApp gates
        # buyer/reviewer-facing progression on it rather than on artifact presence.
        "rights_review_status": (
            str(rights_provenance_review.get("status") or "").strip().lower() or None
        ),
        "site_package_manifest": site_package_manifest,
        "proof_pack_manifest": proof_pack_manifest,
        "hosted_review_readiness": hosted_review_readiness,
        "proof_path_status": proof_path_status,
        "proof_path_events": proof_path_status.get("event_statuses", []),
    }

    try:
        result = sync_webapp_pipeline_attachment(
            site_submission_id=site_submission_id,
            # PIPE-06: request_id == site_submission_id BY CONTRACT. The WebApp mints
            # the inbound request with site_submission_id = requestId, so this is an
            # intentional alias, not an independent fourth verification. The upstream-id
            # guard is effectively three independent links (site_submission_id /
            # buyer_request_id / capture_job_id) — treat it as such, not as four.
            request_id=site_submission_id,
            buyer_request_id=buyer_request_id,
            capture_job_id=capture_job_id,
            scene_id=descriptor.scene_id,
            capture_id=descriptor.capture_id,
            pipeline_prefix=pipeline_prefix,
            qualification_state=qualification_state,
            opportunity_state=opportunity_state,
            authoritative_state_update=True,
            artifacts={str(key): value for key, value in artifacts.items() if value},
            derived_assets=derived_assets,
            evaluation_readiness=evaluation_readiness,
            # Pass the capture root so the delivery-time consent-takedown gate
            # re-reads consent live and blocks the sync on an open revocation.
            capture_root=capture_root,
        )
    except (WebappSyncError, ValueError) as exc:
        result = {
            "status": "failed",
            "reason": str(exc),
            "blocker": "webapp_sync_requires_upstream_request_job_bootstrap",
        }
    return write_pipeline_sync_result(
        pipeline_root=pipeline_root,
        stage="evaluation_prep",
        result=result,
    )
