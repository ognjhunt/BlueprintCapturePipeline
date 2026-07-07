"""Helpers for syncing package and review metadata back into Blueprint-WebApp's control plane."""

from __future__ import annotations

import json
import hmac
import os
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from .common import parse_bool
from .launch_proof_policy import buyer_access_required, production_forces_false, production_forces_true


class WebappSyncError(RuntimeError):
    """Raised when pipeline-to-webapp sync is configured as required and fails."""


ROBOT_EVAL_WEBAPP_STATUS_PROJECTION_SCHEMA_VERSION = "webapp_robot_eval_status_projection.v1"

_PLACEHOLDER_ID_MARKERS = (
    "example",
    "placeholder",
    "replace_me",
    "sample",
    "todo",
    "tbd",
    "<",
    ">",
    "your-",
)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = [str(item) for item in value]
    else:
        candidates = []
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        text = candidate.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _string_env(name: str) -> str:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else ""


def _int_env(name: str, default: int) -> int:
    raw = _string_env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = _string_env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _safety_bool(value: Any) -> bool:
    return parse_bool(value, default=False)


def _revocation_bool(value: Any) -> bool:
    if isinstance(value, str) and value.strip().lower() in {
        "revoked",
        "withdrawn",
        "rescinded",
    }:
        return True
    return parse_bool(value, default=False)


def _artifact_uri_checksums(artifacts: Mapping[str, Any]) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for key, value in artifacts.items():
        if not value:
            continue
        text = str(value)
        checksums[str(key)] = sha256(text.encode("utf-8")).hexdigest()
    return checksums


def _safe_robot_eval_status_projection(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    scenario_batch = _mapping(value.get("scenario_batch"))
    trace_package = _mapping(value.get("trace_package"))
    task_metrics = _mapping(value.get("task_metrics"))
    batch_closure = _mapping(value.get("batch_closure"))
    digital_twin_fidelity = _mapping(value.get("digital_twin_fidelity"))
    policy_interface = _mapping(value.get("policy_interface"))
    closure_audit = _mapping(value.get("closure_audit"))
    remote_cloud_execution = _mapping(value.get("remote_cloud_execution"))
    robot_team_grade_eval_closure = _mapping(value.get("robot_team_grade_eval_closure"))
    proof_boundary = _mapping(value.get("proof_boundary"))
    artifact_paths = _mapping(value.get("artifact_paths"))
    rights_privacy = _mapping(value.get("rights_privacy"))
    product_handoff = _mapping(value.get("product_handoff"))
    buyer_display_guardrails = _mapping(value.get("buyer_display_guardrails"))
    must_not_display_as = _string_list(buyer_display_guardrails.get("must_not_display_as"))
    if not must_not_display_as:
        must_not_display_as = [
            "generated_world_rank_fidelity",
            "evaluation_readiness",
            "policy_quality_certification",
        ]
    # Success metrics are only as strong as the proven evaluation substrate. A bare
    # success rate with no substrate context reads as physical proof to a buyer, so the
    # substrate label is derived strictly from the proof boundary and defaults to the
    # weakest truthful description.
    real_world_proven = proof_boundary.get("real_world_outcome_proven") is True
    simulator_proven = proof_boundary.get("simulator_execution_proven") is True
    robot_policy_execution_proven = (
        proof_boundary.get("robot_policy_execution_proven") is True
    )
    physics_contact_validated = proof_boundary.get("physics_contact_validated") is True
    non_ranking_operational_claim_validated = (
        proof_boundary.get("non_ranking_operational_claim_validated") is True
    )
    rank_fidelity_result_proven = (
        proof_boundary.get("rank_fidelity_result_proven") is True
    )
    public_claim_upgrade_allowed = (
        proof_boundary.get("public_claim_upgrade_allowed") is True
    )
    if real_world_proven:
        success_rate_substrate = "real_robot_outcome"
    elif simulator_proven:
        success_rate_substrate = "simulator_execution"
    else:
        success_rate_substrate = "unproven_pipeline_output"
    rights_status = str(rights_privacy.get("rights_status") or "").strip().lower() or "missing"
    privacy_status = str(rights_privacy.get("privacy_status") or "").strip().lower() or "missing"
    _CLEARED_RIGHTS_STATUSES = {"approved", "verified", "cleared"}
    revocation_takedown = _mapping(
        value.get("revocation_takedown") or rights_privacy.get("revocation_takedown")
    )
    consent_revoked = (
        _revocation_bool(rights_privacy.get("consent_revoked"))
        or _revocation_bool(revocation_takedown.get("consent_revoked"))
        or revocation_takedown.get("status") == "takedown_required"
    )
    product_sku = str(product_handoff.get("product_sku") or "").strip() or None
    entitlement_id = str(product_handoff.get("entitlement_id") or "").strip() or None
    revenue_share_review = _mapping(
        product_handoff.get("revenue_share_review") or value.get("revenue_share_review")
    )
    buyer_readout_status = str(
        closure_audit.get("buyer_readout_status")
        or value.get("buyer_readout_status")
        or "missing"
    ).strip()
    buyer_readout_ready = buyer_readout_status == "buyer_readout_ready_review_required"
    readiness_claim_upgrade_allowed = public_claim_upgrade_allowed and buyer_readout_ready
    readiness_claim_upgrade_blockers = []
    if not buyer_readout_ready:
        readiness_claim_upgrade_blockers.append(
            f"buyer_readout_not_ready:{buyer_readout_status or 'missing'}"
        )
    return {
        "schema_version": ROBOT_EVAL_WEBAPP_STATUS_PROJECTION_SCHEMA_VERSION,
        "generated_at": str(value.get("generated_at") or "").strip() or None,
        "job_id": str(value.get("job_id") or "").strip(),
        "scene_id": str(value.get("scene_id") or "").strip(),
        "capture_id": str(value.get("capture_id") or "").strip(),
        "status": str(value.get("status") or "").strip(),
        "state": str(value.get("state") or "").strip(),
        "buyer_display_state": str(value.get("buyer_display_state") or "").strip(),
        "webapp_role": "display_status_and_proof_boundaries_only",
        "provider_complexity_hidden": True,
        "provider_details_exposed": False,
        "scenario_batch": {
            "status": scenario_batch.get("status"),
            "scenario_eval_run_count": scenario_batch.get("scenario_eval_run_count"),
            "target_scenario_eval_run_count": scenario_batch.get(
                "target_scenario_eval_run_count"
            ),
            "base_scenario_eval_run_count": scenario_batch.get(
                "base_scenario_eval_run_count"
            ),
            "scenario_eval_batch_expanded": bool(
                scenario_batch.get("scenario_eval_batch_expanded")
            ),
            "target_scenario_eval_run_count_satisfied": bool(
                scenario_batch.get("target_scenario_eval_run_count_satisfied")
            ),
            "episode_authoring_contract": _mapping(
                scenario_batch.get("episode_authoring_contract")
            ),
            "covered_scenario_eval_run_count": scenario_batch.get(
                "covered_scenario_eval_run_count"
            ),
            "missing_scenario_eval_run_count": scenario_batch.get(
                "missing_scenario_eval_run_count"
            ),
            "scenario_eval_run_coverage_complete": bool(
                scenario_batch.get("scenario_eval_run_coverage_complete")
            ),
            "scenario_eval_matrix_path": scenario_batch.get("scenario_eval_matrix_path"),
        },
        "trace_package": {
            "status": trace_package.get("status"),
            "machine_trace_package_complete": bool(
                trace_package.get("machine_trace_package_complete")
            ),
            "attempt_trace_path": trace_package.get("attempt_trace_path"),
            "robot_pov_observation_manifest_path": trace_package.get(
                "robot_pov_observation_manifest_path"
            ),
            "robot_pov_frame_sequence_manifest_path": trace_package.get(
                "robot_pov_frame_sequence_manifest_path"
            ),
            "third_person_video_manifest_path": trace_package.get(
                "third_person_video_manifest_path"
            ),
            "contact_stream_path": trace_package.get("contact_stream_path"),
        },
        "task_metrics": {
            "evaluation_status": task_metrics.get("evaluation_status"),
            "task_success_rate": task_metrics.get("task_success_rate"),
            "successful_attempt_count": task_metrics.get("successful_attempt_count"),
            "failed_attempt_count": task_metrics.get("failed_attempt_count"),
            "metric_coverage_complete": bool(task_metrics.get("metric_coverage_complete")),
            "failure_label_coverage_complete": bool(
                task_metrics.get("failure_label_coverage_complete")
            ),
            "success_rate_evaluation_substrate": success_rate_substrate,
            "success_rate_is_real_world_proof": real_world_proven,
            "success_rate_is_simulator_only": simulator_proven and not real_world_proven,
        },
        "batch_closure": {
            "status": batch_closure.get("status"),
            "batch_execution_status": batch_closure.get("batch_execution_status"),
            "machine_trace_package_complete": bool(
                batch_closure.get("machine_trace_package_complete")
            ),
            "robot_team_grade_package_complete": bool(
                batch_closure.get("robot_team_grade_package_complete")
            ),
            "robot_team_grade_blockers": _string_list(
                batch_closure.get("robot_team_grade_blockers")
            ),
            "batch_closure_manifest_path": batch_closure.get("batch_closure_manifest_path"),
            "batch_trace_package_manifest_path": batch_closure.get(
                "batch_trace_package_manifest_path"
            ),
        },
        "digital_twin_fidelity": {
            "status": digital_twin_fidelity.get("status"),
            "machine_fidelity_audit_complete": bool(
                digital_twin_fidelity.get("machine_fidelity_audit_complete")
            ),
            "robot_team_grade_fidelity_passed": bool(
                digital_twin_fidelity.get("robot_team_grade_fidelity_passed")
            ),
            "blockers": _string_list(digital_twin_fidelity.get("blockers")),
        },
        "policy_interface": {
            "status": policy_interface.get("status"),
            "selected_modalities": _string_list(policy_interface.get("selected_modalities")),
            "supported_modalities": _string_list(policy_interface.get("supported_modalities")),
            "observation_schema_id": policy_interface.get("observation_schema_id"),
            "action_schema_id": policy_interface.get("action_schema_id"),
            "reproducible_replay_required": bool(
                policy_interface.get("reproducible_replay_required")
            ),
            "robot_policy_execution_proven": bool(
                policy_interface.get("robot_policy_execution_proven")
            ),
        },
        "closure_audit": {
            "live_eval_closure_status": closure_audit.get("live_eval_closure_status"),
            "selected_scenario_coverage_closed": bool(
                closure_audit.get("selected_scenario_coverage_closed")
            ),
            "machine_trace_package_complete": bool(
                closure_audit.get("machine_trace_package_complete")
            ),
            "robot_team_grade_package_complete": bool(
                closure_audit.get("robot_team_grade_package_complete")
            ),
            "post_training_data_package_status": closure_audit.get(
                "post_training_data_package_status"
            ),
            "buyer_readout_status": buyer_readout_status,
            "buyer_readout_ready": buyer_readout_ready,
            "buyer_readout_required_for_buyer_visible_delivery": True,
            "no_readiness_claim_upgrade_without_evidence": bool(
                closure_audit.get("no_readiness_claim_upgrade_without_evidence")
            ),
        },
        "remote_cloud_execution": {
            "status": remote_cloud_execution.get("status"),
            "contract_ready_for_remote_runtime": bool(
                remote_cloud_execution.get("contract_ready_for_remote_runtime")
            ),
            "remote_cloud_execution_proven": bool(
                remote_cloud_execution.get("remote_cloud_execution_proven")
            ),
            "clean_shutdown_proven": bool(
                remote_cloud_execution.get("clean_shutdown_proven")
            ),
            "live_provider_calls_performed": bool(
                remote_cloud_execution.get("live_provider_calls_performed")
            ),
            "blockers": _string_list(remote_cloud_execution.get("blockers")),
            "closure_manifest_path": remote_cloud_execution.get("closure_manifest_path"),
        },
        "robot_team_grade_eval_closure": {
            "status": robot_team_grade_eval_closure.get("status"),
            "sim_only_beta_core_complete": bool(
                robot_team_grade_eval_closure.get("sim_only_beta_core_complete")
            ),
            "robot_team_grade_evaluation_complete": bool(
                robot_team_grade_eval_closure.get("robot_team_grade_evaluation_complete")
            ),
            "evaluation_readiness_complete": bool(
                robot_team_grade_eval_closure.get("evaluation_readiness_complete")
            ),
            "blocked_requirement_ids": _string_list(
                robot_team_grade_eval_closure.get("blocked_requirement_ids")
            ),
            "all_blocked_requirement_ids": _string_list(
                robot_team_grade_eval_closure.get("all_blocked_requirement_ids")
            ),
            "sim_only_beta_blocked_requirement_ids": _string_list(
                robot_team_grade_eval_closure.get("sim_only_beta_blocked_requirement_ids")
            ),
            "robot_team_grade_blocked_requirement_ids": _string_list(
                robot_team_grade_eval_closure.get("robot_team_grade_blocked_requirement_ids")
            ),
            "evaluation_readiness_blocked_requirement_ids": _string_list(
                robot_team_grade_eval_closure.get(
                    "evaluation_readiness_blocked_requirement_ids"
                )
            ),
            "closure_manifest_path": robot_team_grade_eval_closure.get(
                "closure_manifest_path"
            ),
        },
        "proof_boundary": {
            "simulator_execution_proven": simulator_proven,
            "robot_policy_execution_proven": robot_policy_execution_proven,
            "real_world_outcome_proven": real_world_proven,
            "physics_contact_validated": physics_contact_validated,
            "non_ranking_operational_claim_validated": (
                non_ranking_operational_claim_validated
            ),
            "rank_fidelity_result_proven": rank_fidelity_result_proven,
            "public_claim_upgrade_allowed": public_claim_upgrade_allowed,
            # Each proven flag lists the whitelisted manifests a buyer can audit it
            # against. An empty list means no synced artifact backs the flag.
            "evidence_manifest_paths": {
                "simulator_execution_proven": [
                    path
                    for path in (
                        artifact_paths.get("simulator_command_batch_closure_manifest"),
                        artifact_paths.get(
                            "simulator_command_batch_trace_package_manifest"
                        ),
                    )
                    if path
                ],
                "robot_policy_execution_proven": [
                    path
                    for path in (artifact_paths.get("policy_execution_manifest"),)
                    if path
                ],
                "real_world_outcome_proven": [],
            },
        },
        "rights_privacy": {
            "rights_status": rights_status,
            "privacy_status": privacy_status,
            "consent_scope_present": bool(_string_list(rights_privacy.get("consent_scope"))),
            "consent_revoked": consent_revoked,
            "revocation_takedown_status": str(
                revocation_takedown.get("status") or "not_available"
            ).strip(),
            "revocation_takedown_required": consent_revoked,
            "rights_privacy_blocking": not (
                rights_status in _CLEARED_RIGHTS_STATUSES
                and privacy_status in _CLEARED_RIGHTS_STATUSES
            )
            or consent_revoked,
        },
        "product_handoff": {
            "product_type": str(product_handoff.get("product_type") or "").strip() or None,
            "product_sku": product_sku,
            "entitlement_id": entitlement_id,
            "buyer_review_url": str(product_handoff.get("buyer_review_url") or "").strip()
            or None,
            "entitlement_wiring_present": bool(product_sku and entitlement_id),
            "pricing_is_out_of_band": True,
            "revenue_share_review": {
                "status": str(revenue_share_review.get("status") or "not_available").strip(),
                "required_before_paid_reuse_or_resale": _safety_bool(
                    revenue_share_review.get("required_before_paid_reuse_or_resale")
                ),
                "owner_revenue_share_record_present": _safety_bool(
                    revenue_share_review.get("owner_revenue_share_record_present")
                ),
                "revenue_share_commitment_made": revenue_share_review.get(
                    "revenue_share_commitment_made"
                )
                is True,
                "payout_commitment_allowed": revenue_share_review.get(
                    "payout_commitment_allowed"
                )
                is True,
            },
        },
        "artifact_paths": {
            str(key): item
            for key, item in artifact_paths.items()
            if key
            in {
                "scenario_eval_matrix",
                "simulator_command_batch_closure_manifest",
                "simulator_command_batch_trace_package_manifest",
                "normalized_attempt_trace",
                "failure_labels",
                "robot_pov_observation_manifest",
                "robot_pov_observation_candidate_set",
                "selected_initial_policy_observation",
                "robot_pov_frame_sequence_manifest",
                "policy_package_manifest",
                "policy_execution_manifest",
                "evaluation_result",
                "proof_boundary",
                "job_run_manifest",
                "post_training_data_package_export_manifest",
                "webapp_robot_eval_status_projection",
                "remote_cloud_execution_closure_manifest",
                "robot_team_grade_eval_closure_manifest",
            }
            and item
        },
        "buyer_display_guardrails": {
            "must_not_display_as": must_not_display_as,
            "provider_commands_exposed": False,
            "provider_credentials_exposed": False,
            "readiness_claim_upgrade_allowed": readiness_claim_upgrade_allowed,
            "readiness_claim_upgrade_blockers": readiness_claim_upgrade_blockers,
            "buyer_readout_ready": buyer_readout_ready,
        },
    }


def _contains_placeholder_id(value: str) -> bool:
    normalized = value.strip().lower()
    return any(marker in normalized for marker in _PLACEHOLDER_ID_MARKERS)


def _is_generated_capture_id(value: str, payload: Mapping[str, Any]) -> bool:
    scene_id = str(payload.get("scene_id") or "").strip()
    capture_id = str(payload.get("capture_id") or "").strip()
    generated_values = {capture_id}
    if scene_id and capture_id:
        generated_values.update(
            {
                f"{scene_id}:{capture_id}",
                f"{scene_id}/{capture_id}",
                f"{scene_id}/captures/{capture_id}",
            }
        )
    return bool(value.strip() and value.strip() in generated_values)


def upstream_link_id_failures(payload: Mapping[str, Any]) -> dict[str, str]:
    """Public fail-closed check: missing, placeholder, or capture-derived
    upstream WebApp ids keyed by field name. Empty dict means all ids are real."""
    return _upstream_link_failures(payload)


def _upstream_link_failures(payload: Mapping[str, Any]) -> dict[str, str]:
    failures: dict[str, str] = {}
    for key in ("site_submission_id", "request_id", "buyer_request_id", "capture_job_id"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            failures[key] = "missing"
            continue
        if _contains_placeholder_id(value):
            failures[key] = "placeholder upstream ids"
            continue
        if _is_generated_capture_id(value, payload):
            failures[key] = "generated capture ids"
    return failures


def _upstream_links_error(failures: Mapping[str, str]) -> ValueError:
    grouped: dict[str, list[str]] = {}
    for key, reason in failures.items():
        grouped.setdefault(reason, []).append(key)
    joined = ", ".join(failures.keys())
    reason_detail = "; ".join(
        f"{reason}: {', '.join(fields)}"
        for reason, fields in grouped.items()
    )
    return ValueError(
        "missing_upstream_pipeline_records: "
        f"{joined}. {reason_detail}. WebApp sync requires real WebApp request, buyer request, "
        "site submission, and capture job links before projecting hosted-review or buyer-access state."
    )


def _placeholder_sync_allowed() -> bool:
    return (
        production_forces_false("PIPELINE_SYNC_ALLOW_PLACEHOLDER_REQUESTS", default=False)
        or production_forces_false("PIPELINE_SYNC_ALLOW_PLACEHOLDER_FALLBACK", default=False)
    )


def _pipeline_sync_headers(sync_token: str, body: bytes) -> Dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    signature = hmac.new(
        sync_token.encode("utf-8"),
        f"{timestamp}.".encode("utf-8") + body,
        sha256,
    ).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Blueprint-Pipeline-Timestamp": timestamp,
        "X-Blueprint-Pipeline-Signature": f"sha256={signature}",
    }


def _extract_webapp_response_ids(response: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "id",
        "request_id",
        "attachment_id",
        "pipeline_attachment_id",
        "listing_id",
        "site_world_id",
        "artifact_id",
        "capture_job_id",
        "buyer_artifact_id",
        "entitlement_id",
        "entitlementId",
        "marketplace_entitlement_id",
        "marketplaceEntitlementId",
        "sku",
        "marketplace_sku",
        "marketplaceSku",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        value = response.get(key)
        if value:
            out[key] = value
    nested = response.get("attachment") if isinstance(response.get("attachment"), Mapping) else {}
    for key in keys:
        value = nested.get(key)
        if value and key not in out:
            out[key] = value
    for nested_key in ("marketplace_publication", "marketplacePublication", "pipeline"):
        nested_response = response.get(nested_key)
        if not isinstance(nested_response, Mapping):
            continue
        for key in keys:
            value = nested_response.get(key)
            if value and key not in out:
                out[key] = value
    return out


def _buyer_access_check_payload(response: Mapping[str, Any]) -> Dict[str, Any]:
    check_url = _string_env("PIPELINE_BUYER_ACCESS_CHECK_URL")
    check_token = _string_env("PIPELINE_BUYER_ACCESS_CHECK_TOKEN") or _string_env("PIPELINE_SYNC_TOKEN")
    if not check_url:
        return {
            "status": "blocked" if buyer_access_required() else "skipped",
            "buyer_access_checked": False,
            "reason": "buyer_access_check_not_configured",
            "blocker": "buyer_access_check_not_configured" if buyer_access_required() else None,
        }
    payload = {"webapp_response_ids": _extract_webapp_response_ids(response)}
    body = json.dumps(payload).encode("utf-8")
    headers = (
        _pipeline_sync_headers(check_token, body)
        if check_token
        else {"Content-Type": "application/json"}
    )
    request = urllib_request.Request(
        check_url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=max(1, _int_env("PIPELINE_BUYER_ACCESS_TIMEOUT_SECONDS", 10))) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        return {
            "status": "blocked",
            "buyer_access_checked": False,
            "reason": f"buyer_access_check_failed:{exc}",
            "blocker": "buyer_access_check_failed",
        }
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}
    allowed = bool(parsed.get("ok") or parsed.get("accessible") or parsed.get("buyer_accessible"))
    return {
        "status": "succeeded" if allowed else "blocked",
        "buyer_access_checked": True,
        "buyer_accessible": allowed,
        "response": parsed if isinstance(parsed, dict) else {},
        "blocker": None if allowed else "buyer_access_check_not_accessible",
    }


def derive_webapp_qualification_state(*, readiness_state: object, completeness_status: object) -> str:
    normalized_readiness = str(readiness_state or "").strip().lower()
    normalized_completeness = str(completeness_status or "").strip().lower()
    if normalized_completeness and normalized_completeness != "sufficient":
        return "needs_more_evidence"
    if normalized_readiness == "ready":
        return "qualified_ready"
    if normalized_readiness == "risky":
        return "qualified_risky"
    return "not_ready_yet"


def derive_webapp_opportunity_state(*, qualification_state: object) -> str:
    normalized = str(qualification_state or "").strip().lower()
    if normalized in {"qualified_ready", "qualified_risky"}:
        return "handoff_ready"
    return "not_applicable"


def build_webapp_pipeline_attachment_payload(
    *,
    site_submission_id: object,
    request_id: object = None,
    buyer_request_id: object = None,
    capture_job_id: object = None,
    scene_id: object,
    capture_id: object,
    pipeline_prefix: object,
    qualification_state: object,
    opportunity_state: object,
    artifacts: Mapping[str, Any],
    derived_assets: Optional[Mapping[str, Any]] = None,
    evaluation_readiness: Optional[Mapping[str, Any]] = None,
    robot_eval_status_projection: Optional[Mapping[str, Any]] = None,
    authoritative_state_update: bool = False,
) -> Dict[str, Any]:
    safe_robot_eval_projection = _safe_robot_eval_status_projection(
        robot_eval_status_projection
    )
    payload = {
        "schema_version": "v1",
        "site_submission_id": str(site_submission_id or "").strip(),
        "request_id": str(request_id or "").strip(),
        "buyer_request_id": str(buyer_request_id or "").strip(),
        "capture_job_id": str(capture_job_id or "").strip(),
        "scene_id": str(scene_id or "").strip(),
        "capture_id": str(capture_id or "").strip(),
        "pipeline_prefix": str(pipeline_prefix or "").strip(),
        "qualification_state": str(qualification_state or "").strip(),
        "opportunity_state": str(opportunity_state or "").strip(),
        "authoritative_state_update": bool(authoritative_state_update),
        "artifacts": {str(key): value for key, value in artifacts.items() if value},
        "derived_assets": (
            {str(key): value for key, value in derived_assets.items() if value}
            if isinstance(derived_assets, Mapping)
            else {}
        ),
        "evaluation_readiness": (
            {str(key): value for key, value in evaluation_readiness.items()}
            if isinstance(evaluation_readiness, Mapping)
            else None
        ),
        # evaluation_readiness is already in must_not_display_as; this sibling flag makes
        # the advisory scope explicit for consumers that only read the payload.
        "evaluation_readiness_is_advisory_only": True,
        "robot_eval_status_projection": safe_robot_eval_projection or None,
    }
    if not payload["site_submission_id"] and not payload["request_id"]:
        raise ValueError("site_submission_id or request_id is required")
    return payload


def sync_webapp_pipeline_attachment(
    *,
    site_submission_id: object,
    request_id: object = None,
    buyer_request_id: object = None,
    capture_job_id: object = None,
    scene_id: object,
    capture_id: object,
    pipeline_prefix: object,
    qualification_state: object,
    opportunity_state: object,
    artifacts: Mapping[str, Any],
    derived_assets: Optional[Mapping[str, Any]] = None,
    evaluation_readiness: Optional[Mapping[str, Any]] = None,
    robot_eval_status_projection: Optional[Mapping[str, Any]] = None,
    authoritative_state_update: bool = False,
    capture_root: Optional[Path] = None,
) -> Dict[str, Any]:
    # Rights are authoritative continuously: an open consent takedown on the
    # source capture must stop this sync before anything reaches the webapp.
    if capture_root is not None:
        from .consent_takedown import evaluate_delivery_time_takedown_gate

        gate = evaluate_delivery_time_takedown_gate(
            capture_root=capture_root, surface="webapp_sync"
        )
        if not gate.get("serve_allowed"):
            if _string_env("PIPELINE_SYNC_WEBAPP_URL") or _string_env(
                "PIPELINE_SYNC_TOKEN"
            ) or production_forces_true("PIPELINE_SYNC_REQUIRED", default=False):
                raise WebappSyncError(
                    f"open_consent_takedown:{gate.get('status')}"
                )
            return {
                "status": "failed",
                "reason": f"open_consent_takedown:{gate.get('status')}",
                "blocker": "open_consent_takedown",
                "attempts": 0,
                "takedown_gate": gate,
                "webapp_response_ids": {},
                "buyer_access_check": {
                    "status": "blocked",
                    "buyer_access_checked": False,
                    "reason": "open_consent_takedown",
                    "blocker": "open_consent_takedown",
                },
            }
    sync_url = _string_env("PIPELINE_SYNC_WEBAPP_URL")
    sync_token = _string_env("PIPELINE_SYNC_TOKEN")
    sync_required = production_forces_true("PIPELINE_SYNC_REQUIRED", default=False)
    placeholder_sync_allowed = _placeholder_sync_allowed()
    max_attempts = max(1, _int_env("PIPELINE_SYNC_MAX_ATTEMPTS", 3))
    retry_delay_ms = max(0, _int_env("PIPELINE_SYNC_RETRY_DELAY_MS", 500))
    payload = build_webapp_pipeline_attachment_payload(
        site_submission_id=site_submission_id,
        request_id=request_id,
        buyer_request_id=buyer_request_id,
        capture_job_id=capture_job_id,
        scene_id=scene_id,
        capture_id=capture_id,
        pipeline_prefix=pipeline_prefix,
        qualification_state=qualification_state,
        opportunity_state=opportunity_state,
        artifacts=artifacts,
        derived_assets=derived_assets,
        evaluation_readiness=evaluation_readiness,
        robot_eval_status_projection=robot_eval_status_projection,
        authoritative_state_update=authoritative_state_update,
    )
    payload["artifact_uri_checksums"] = _artifact_uri_checksums(payload.get("artifacts") or {})
    payload["placeholder_fallback_allowed"] = bool(placeholder_sync_allowed)
    upstream_link_failures = _upstream_link_failures(payload)
    missing_upstream_links = list(upstream_link_failures.keys())
    payload["upstream_links_verified"] = not bool(upstream_link_failures)
    payload["missing_upstream_links"] = missing_upstream_links
    payload["upstream_link_failures"] = upstream_link_failures
    payload["upstream_link_next_input"] = (
        None
        if not upstream_link_failures
        else "Provide real WebApp request, buyer_request_id, site_submission_id, and capture_job_id values from upstream bootstrap."
    )
    if upstream_link_failures and (sync_required or bool(sync_url) or bool(sync_token)):
        raise _upstream_links_error(upstream_link_failures)
    if upstream_link_failures:
        error = _upstream_links_error(upstream_link_failures)
        return {
            "status": "failed",
            "reason": str(error),
            "blocker": "missing_upstream_pipeline_records",
            "attempts": 0,
            "attachment_payload": payload,
            "artifact_uri_checksums": payload["artifact_uri_checksums"],
            "webapp_response_ids": {},
            "buyer_access_check": {
                "status": "blocked",
                "buyer_access_checked": False,
                "reason": "missing_upstream_pipeline_records",
                "blocker": "missing_upstream_pipeline_records",
            },
            "evaluation_readiness": (
                {str(key): value for key, value in evaluation_readiness.items()}
                if isinstance(evaluation_readiness, Mapping)
                else None
            ),
        }
    if not sync_url or not sync_token:
        result = {
            "status": "failed" if sync_required else "skipped",
            "reason": "sync_not_configured",
            "attempts": 0,
            "attachment_payload": payload,
            "artifact_uri_checksums": payload["artifact_uri_checksums"],
            "webapp_response_ids": {},
            "buyer_access_check": {
                "status": "blocked" if buyer_access_required() else "skipped",
                "buyer_access_checked": False,
                "reason": "sync_not_configured",
                "blocker": "sync_not_configured" if buyer_access_required() else None,
            },
            "evaluation_readiness": (
                {str(key): value for key, value in evaluation_readiness.items()}
                if isinstance(evaluation_readiness, Mapping)
                else None
            ),
        }
        if sync_required:
            raise WebappSyncError("sync_not_configured")
        return result

    timeout_seconds = max(1, _int_env("PIPELINE_SYNC_TIMEOUT_SECONDS", 10))
    last_reason = "sync_unknown_failure"

    for attempt in range(1, max_attempts + 1):
        request_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib_request.Request(
            sync_url,
            data=request_body,
            headers=_pipeline_sync_headers(sync_token, request_body),
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            last_reason = f"http_error:{exc.code}"
        except urllib_error.URLError as exc:
            last_reason = f"url_error:{exc.reason}"
        except (TimeoutError, ValueError) as exc:
            last_reason = exc.__class__.__name__.lower()
        else:
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                last_reason = "invalid_json"
            else:
                response_ids = _extract_webapp_response_ids(parsed if isinstance(parsed, dict) else {})
                buyer_access_check = _buyer_access_check_payload(parsed if isinstance(parsed, dict) else {})
                return {
                    "status": "succeeded",
                    "attempts": attempt,
                    "response": parsed if isinstance(parsed, dict) else {},
                    "webapp_response_ids": response_ids,
                    "artifact_uri_checksums": payload["artifact_uri_checksums"],
                    "buyer_access_check": buyer_access_check,
                    "attachment_payload": payload,
                    "evaluation_readiness": (
                        {str(key): value for key, value in evaluation_readiness.items()}
                        if isinstance(evaluation_readiness, Mapping)
                        else None
                    ),
                }

        if attempt < max_attempts and retry_delay_ms:
            time.sleep(retry_delay_ms / 1000)

    if sync_required:
        raise WebappSyncError(last_reason)
    return {
        "status": "failed",
        "reason": last_reason,
        "attempts": max_attempts,
        "webapp_response_ids": {},
        "artifact_uri_checksums": payload["artifact_uri_checksums"],
        "buyer_access_check": {
            "status": "blocked" if buyer_access_required() else "skipped",
            "buyer_access_checked": False,
            "reason": last_reason,
            "blocker": "sync_failed" if buyer_access_required() else None,
        },
        "attachment_payload": payload,
        "evaluation_readiness": (
            {str(key): value for key, value in evaluation_readiness.items()}
            if isinstance(evaluation_readiness, Mapping)
            else None
        ),
    }
