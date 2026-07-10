"""Deterministic fixture WAM evaluator for local robot-eval job tests."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .common import ensure_dir, read_json_any, utc_now_iso, write_json, write_text
from .failure_diagnosis_contract import (
    FAILURE_LABEL_PROOF_EFFECT,
    dedupe as _dedupe_refs,
    evidence_refs as _failure_evidence_refs,
    failure_root_cause_category as _failure_root_cause_category,
    frame_or_clip_refs as _failure_frame_or_clip_refs,
    remediation_candidate as _failure_remediation_candidate,
    review_status_for_failure_label as _failure_review_status,
)
from .success_claim_contracts import (
    build_artifact_freshness_evidence,
    build_contact_state_change_proof,
    build_media_validity,
    build_physical_readiness,
    build_policy_action_execution,
    build_review_task_success,
    build_simulator_execution,
    build_task_success_contract_result,
    coerce_strict_success,
    derive_task_proof_requirements,
)
from .task_eval_run_report import build_task_eval_run_report
from .wam_eval_substrate import (
    WAM_EVALUATION_SUBSTRATES,
    build_wam_eval_claim_boundary,
    build_wam_evaluation_request,
    normalize_evaluation_substrate,
    write_evaluation_substrate_registry,
)
from .sc3_fidelity_contracts import SC3_OOD_AXES
from .wam_provider_runtime import (
    classical_sim_cross_check_plan as _classical_sim_cross_check_plan,
    customer_validation_envelope as _customer_validation_envelope,
    live_provider_gate_blockers as _live_provider_gate_blockers,
    normalize_provider_rollouts as _normalize_provider_rollouts,
    policy_interface_binding as _policy_interface_binding,
    production_ops_manifest as _production_ops_manifest,
    provider_artifact_upload_proof as _provider_artifact_upload_proof,
    provider_auth_status as _provider_auth_status,
    provider_cost_ledger as _provider_cost_ledger,
    provider_execution_manifest as _provider_execution_manifest,
    provider_runtime_package as _provider_runtime_package,
    real_world_anchor_manifest as _real_world_anchor_manifest,
    run_provider_command as _run_provider_command,
    substrate_provider_command as _substrate_provider_command,
    vision_review_queue as _vision_review_queue,
)
from .wam_vision_success_judge import (
    FIXTURE_VISUAL_REVIEW_BLOCKER,
    FIXTURE_VISUAL_SMOKE_STATUS,
    build_fixture_vision_success_labels,
)


WAM_ROLLOUT_MANIFEST_SCHEMA_VERSION = "wam_rollout_manifest.v1"
WAM_ROLLOUT_RESULTS_SCHEMA_VERSION = "wam_rollout_results.v1"
POLICY_RANKING_SCORECARD_SCHEMA_VERSION = "policy_ranking_scorecard.v1"
POLICY_RANKING_TIE_BAND = 0.05
POLICY_RANKING_HIGH_UNCERTAINTY_THRESHOLD = 0.65
POLICY_RANKING_HIGH_OOD_RATE_THRESHOLD = 0.5
POLICY_RANKING_MIN_REPLICATES_PER_CONDITION = 20
POLICY_RANKING_MULTIPLICITY_ALPHA = 0.05
POLICY_RANKING_BOOTSTRAP_ITERATIONS = 4096
DECISION_AUTHORITY_REGISTRY_SCHEMA_VERSION = "blueprint.wam_decision_authority_registry.v1"
DECISION_EVIDENCE_SCHEMA_VERSIONS = {
    "execution": "blueprint.wam_replicate_execution_evidence.v1",
    "media": "blueprint.wam_replicate_media_evidence.v1",
    "outcome_label": "blueprint.wam_replicate_outcome_label_evidence.v1",
}
MATCHED_CONDITION_MANIFEST_SCHEMA_VERSION = "blueprint.wam_matched_condition_manifest.v1"
SHA256_REF_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
PINNED_DECISION_GOVERNANCE_PUBLIC_KEY_B64 = "UZqeANEDQXb26TVtFs/0EVxZHkLPa1pS77GwOxgOIV4="
PINNED_DECISION_GOVERNANCE_KEY_FINGERPRINT = (
    "sha256:d86af580d168c7dd2de471b979ab2f0f5e2cb68f99f4cf58cf6ce1ea7c2d4520"
)
REAL_WORLD_VALIDATION_FOLLOWUP_SCHEMA_VERSION = "real_world_validation_followup_request.v1"
SRCC_VALIDATION_PLAN_SCHEMA_VERSION = "srcc_validation_plan.v1"
NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION = "robot_eval_job_normalized_attempt_trace.v1"
FAILURE_LABELS_SCHEMA_VERSION = "robot_eval_job_failure_labels.v1"
PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION = "robot_eval_job_prediction_outcome_ledger.v1"
CALIBRATION_REPORT_SCHEMA_VERSION = "robot_eval_job_calibration_report.v1"
BREAKAGE_LIBRARY_SCHEMA_VERSION = "robot_eval_job_breakage_library.v1"
ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION = "accepted_real_world_anchor.v1"
ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS = (
    "scenario_eval_run_id",
    "policy_id",
    "task_id",
    "scenario_variation_instance_id",
)
CANDIDATE_SELECTION_REPORT_SCHEMA_VERSION = "wam_candidate_selection_report.v1"
CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN = 0.05
CANDIDATE_SELECTION_HIGH_UNCERTAINTY_THRESHOLD = 0.5
SHORT_VISUAL_SANITY_PASSED_STATUSES = {
    "passed_short_visual_sanity",
}
SHORT_VISUAL_SANITY_MANIFEST_KEYS = (
    "short_visual_sanity_manifest_path",
    "persistent_wam_short_visual_sanity_manifest_path",
    "review_quality_manifest_path",
)
SHORT_VISUAL_SANITY_INLINE_KEYS = (
    "short_visual_sanity_manifest",
    "persistent_wam_short_visual_sanity_manifest",
    "review_quality_manifest",
)
REVIEW_LABEL_REF_KEYS = (
    "review_label_ref",
    "review_label_refs",
    "review_label_path",
    "review_label_paths",
    "review_evidence_ref",
    "review_evidence_refs",
    "reviewer_id",
    "reviewed_by",
    "review_provenance",
    "label_provenance",
    "human_review_label_path",
    "vlm_review_result_path",
    "reviewed_visual_label_path",
)
REVIEW_PROVENANCE_REF_KEYS = (
    "source_policy_observation_visual_qa_path",
    "wam_rollout_visual_quality_report_path",
    "video_review_status_path",
    "review_video_path",
    "wam_rollout_frame_stats_path",
    "source_label_path",
    "evidence_path",
)
CONTACT_SHEET_REF_KEYS = (
    "wam_rollout_contact_sheet_path",
    "contact_sheet_path",
)
CONSISTENCY_SIGNAL_SUMMARY_SCHEMA_VERSION = "wam_forward_inverse_consistency_signal_summary.v1"
CONSISTENCY_SIGNAL_KEYS = (
    "forward_inverse_consistency_proven",
    "forward_dynamics_consistency_proven",
    "inverse_dynamics_consistency_proven",
    "forward_consistent",
    "inverse_consistent",
)
CONSISTENCY_OVERCLAIM_KEYS = (
    "evaluator_bounded_policy_ranking_upgraded_by_consistency",
    "policy_success_claimed_from_consistency",
    "task_success_claimed_from_consistency",
    "rank_fidelity_claimed_from_consistency",
    "deployment_readiness_claimed_from_consistency",
    "sensor_truth_claimed_from_consistency",
    "external_validation_claimed_from_consistency",
    "policy_success_proven",
    "task_success_proven",
    "rank_fidelity_result_proven",
    "deployment_readiness_proven",
    "sensor_truth_proven",
    "external_validation_proven",
    "public_claim_upgrade_allowed",
)

WAM_ARTIFACT_PATHS = {
    "evaluation_substrate_registry": "evaluation_substrate_registry.json",
    "wam_evaluation_request": "wam_evaluation_request.json",
    "wam_rollout_manifest": "wam_rollout_manifest.json",
    "wam_rollout_results": "wam_rollout_results.json",
    "vision_success_labels": "vision_success_labels.json",
    "normalized_attempt_trace": "normalized_attempt_trace.json",
    "task_eval_run_report": "task_eval_run_report.json",
    "failure_labels": "failure_labels.json",
    "prediction_outcome_ledger": "prediction_outcome_ledger.json",
    "calibration_report": "calibration_report.json",
    "breakage_library": "breakage_library.json",
    "policy_ranking_scorecard": "policy_ranking_scorecard.json",
    "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
    "real_world_validation_followup_request": "real_world_validation_followup_request.json",
    "srcc_validation_plan": "srcc_validation_plan.json",
    "wam_provider_runtime_package": "wam_provider_runtime_package.json",
    "wam_provider_execution_manifest": "wam_provider_execution_manifest.json",
    "wam_provider_cost_control_ledger": "wam_provider_cost_control_ledger.json",
    "wam_provider_artifact_upload_proof": "wam_provider_artifact_upload_proof.json",
    "wam_policy_interface_binding": "wam_policy_interface_binding.json",
    "wam_vision_success_review_queue": "wam_vision_success_review_queue.json",
    "wam_real_world_validation_anchor_manifest": "wam_real_world_validation_anchor_manifest.json",
    "wam_customer_validation_envelope": "wam_customer_validation_envelope.json",
    "wam_production_ops_manifest": "wam_production_ops_manifest.json",
    "wam_classical_sim_cross_check_plan": "wam_classical_sim_cross_check_plan.json",
    "candidate_selection_report": "candidate_selection_report.json",
    "candidate_selection_report_markdown": "candidate_selection_report.md",
    "visual_review_blocker_summary": "visual_review_blocker_summary.json",
    "customer_handoff_report": "customer_handoff_report.json",
    "customer_handoff_report_markdown": "customer_handoff_report.md",
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    return []


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return _string(value).lower() in {"1", "true", "yes", "y", "on", "passed"}


def _ordered_unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = _string(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _dedupe(values: Sequence[str]) -> list[str]:
    return _ordered_unique_strings(values)


def _consistency_support_signal_summary(
    *,
    labels: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    sources: list[Mapping[str, Any]] = [labels, *label_rows]
    signal_fields_present = sorted(
        {key for source in sources for key in CONSISTENCY_SIGNAL_KEYS if key in source}
    )
    overclaim_fields_present = sorted(
        {
            key
            for source in sources
            for key in CONSISTENCY_OVERCLAIM_KEYS
            if key in source and _truthy(source.get(key))
        }
    )
    proven_label_count = sum(
        1 for row in label_rows if any(_truthy(row.get(key)) for key in CONSISTENCY_SIGNAL_KEYS)
    )
    return {
        "schema_version": CONSISTENCY_SIGNAL_SUMMARY_SCHEMA_VERSION,
        "status": "support_signal_present" if signal_fields_present else "not_provided",
        "support_signal_only": True,
        "label_count_with_consistency_signal": proven_label_count,
        "signal_fields_present": signal_fields_present,
        "ignored_upgrade_fields_present": overclaim_fields_present,
        "ranking_inputs_unchanged": True,
        "task_success_labels_unchanged": True,
        "evaluator_bounded_policy_ranking_upgraded_by_consistency": False,
        "policy_success_claimed_from_consistency": False,
        "task_success_claimed_from_consistency": False,
        "rank_fidelity_claimed_from_consistency": False,
        "deployment_readiness_claimed_from_consistency": False,
        "sensor_truth_claimed_from_consistency": False,
        "external_validation_claimed_from_consistency": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            "forward_inverse_consistency_is_reliability_review_signal_only": True,
            "forward_inverse_consistency_does_not_upgrade_evaluator_bounded_policy_ranking": True,
            "forward_inverse_consistency_does_not_prove_policy_success": True,
            "forward_inverse_consistency_does_not_prove_task_success": True,
            "forward_inverse_consistency_does_not_prove_rank_fidelity": True,
            "forward_inverse_consistency_does_not_prove_deployment_readiness": True,
            "forward_inverse_consistency_does_not_prove_sensor_truth": True,
            "forward_inverse_consistency_is_not_external_validation": True,
        },
    }


def _safe_id(value: Any, *, fallback: str = "item") -> str:
    text = _string(value) or fallback
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _refs_from_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    refs: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            refs.extend(_string(item) for item in value.values() if _string(item))
        else:
            refs.extend(_string_list(value))
    return _dedupe(refs)


def _artifact_refs_with_markers(
    payload: Mapping[str, Any],
    markers: Sequence[str],
) -> list[str]:
    artifact_paths = _mapping(payload.get("artifact_paths") or payload.get("artifactPaths"))
    refs: list[str] = []
    for key, value in artifact_paths.items():
        key_text = _string(key).lower()
        if any(marker in key_text for marker in markers):
            refs.extend(_string_list(value))
            if not isinstance(value, (str, Sequence)) or isinstance(value, Mapping):
                refs.append(_string(value))
    return _dedupe(refs)


def _review_label_refs(payload: Mapping[str, Any]) -> list[str]:
    return _dedupe(
        [
            *_refs_from_keys(payload, REVIEW_LABEL_REF_KEYS),
            *_artifact_refs_with_markers(payload, ("review", "reviewed", "human", "vlm")),
        ]
    )


def _contact_sheet_refs(payload: Mapping[str, Any]) -> list[str]:
    return _dedupe(
        [
            *_refs_from_keys(payload, CONTACT_SHEET_REF_KEYS),
            *_artifact_refs_with_markers(payload, ("contact_sheet",)),
        ]
    )


def _review_provenance_refs(payload: Mapping[str, Any]) -> list[str]:
    return _dedupe(
        [
            *_refs_from_keys(payload, REVIEW_PROVENANCE_REF_KEYS),
            *_artifact_refs_with_markers(
                payload,
                ("visual_quality", "video_review", "frame_stats", "review_video"),
            ),
        ]
    )


def _inline_short_visual_sanity_manifest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    for key in SHORT_VISUAL_SANITY_INLINE_KEYS:
        manifest = _mapping(payload.get(key))
        if manifest:
            return manifest
    return {}


def _short_visual_sanity_manifest_paths(payload: Mapping[str, Any]) -> list[str]:
    return _dedupe(
        [
            *_refs_from_keys(payload, SHORT_VISUAL_SANITY_MANIFEST_KEYS),
            *_artifact_refs_with_markers(
                payload,
                ("short_visual_sanity", "persistent_wam_short_visual_sanity"),
            ),
        ]
    )


def _load_short_visual_sanity_manifest_from_ref(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.is_file():
        return {}
    return _read_optional_mapping(path)


def _short_visual_sanity_gate_from_labels(labels: Mapping[str, Any]) -> Dict[str, Any]:
    label_rows = [row for row in labels.get("labels", []) or [] if isinstance(row, Mapping)]
    sources = [labels, *label_rows]
    manifest: Dict[str, Any] = {}
    manifest_path = ""
    for source in sources:
        manifest = _inline_short_visual_sanity_manifest(source)
        paths = _short_visual_sanity_manifest_paths(source)
        if paths and not manifest_path:
            manifest_path = paths[0]
        if not manifest and paths:
            manifest = _load_short_visual_sanity_manifest_from_ref(paths[0])
        if manifest:
            manifest_path = (
                _string(manifest.get("short_visual_sanity_manifest_path"))
                or _string(manifest.get("manifest_path"))
                or manifest_path
            )
            break

    contact_sheet_refs = _dedupe(
        [
            *_contact_sheet_refs(labels),
            *[ref for row in label_rows for ref in _contact_sheet_refs(row)],
            *_contact_sheet_refs(manifest),
        ]
    )
    provenance_refs = _dedupe(
        [
            *_review_provenance_refs(labels),
            *[ref for row in label_rows for ref in _review_provenance_refs(row)],
            *_review_provenance_refs(manifest),
        ]
    )
    review_label_refs = _dedupe(
        [
            *_review_label_refs(labels),
            *[ref for row in label_rows for ref in _review_label_refs(row)],
        ]
    )
    missing_review_label_ids = [
        _string(row.get("label_id"))
        or _string(row.get("attempt_id"))
        or _string(row.get("rollout_id"))
        or f"label_{index:04d}"
        for index, row in enumerate(label_rows, start=1)
        if not _review_label_refs(row) and not _review_label_refs(labels)
    ]
    missing_review_grade_label_ids = [
        _string(row.get("label_id"))
        or _string(row.get("attempt_id"))
        or _string(row.get("rollout_id"))
        or f"label_{index:04d}"
        for index, row in enumerate(label_rows, start=1)
        if row.get("review_grade_success_label") is not True
    ]

    blockers: list[str] = []
    if not manifest:
        blockers.append("short_visual_sanity_manifest_missing_for_review_grade_ranking")
    else:
        manifest_status = _string(manifest.get("status"))
        if (
            manifest_status not in SHORT_VISUAL_SANITY_PASSED_STATUSES
            or manifest.get("short_visual_sanity_passed") is not True
        ):
            blockers.append("short_visual_sanity_manifest_not_passed")
        if _string(manifest.get("visual_profile")) != "review_quality":
            blockers.append("short_visual_sanity_manifest_not_review_quality")
        if manifest.get("visually_useful_rollout") is not True:
            blockers.append("short_visual_sanity_manifest_not_visually_useful")
        if _string(manifest.get("source_policy_observation_visual_qa_status")) and (
            _string(manifest.get("source_policy_observation_visual_qa_status"))
            != "passed_visual_quality_gate"
        ):
            blockers.append("short_visual_sanity_source_observation_qa_not_passed")
        blockers.extend(_string_list(manifest.get("blockers")))
    if not manifest_path:
        blockers.append("short_visual_sanity_manifest_ref_missing")
    if not contact_sheet_refs:
        blockers.append("short_visual_sanity_contact_sheet_ref_missing")
    if not provenance_refs:
        blockers.append("review_quality_provenance_refs_missing")
    if label_rows and missing_review_label_ids:
        blockers.append("review_grade_success_label_refs_missing")
    if label_rows and missing_review_grade_label_ids:
        blockers.append("review_grade_success_labels_missing_for_some_rollouts")

    blockers = sorted(set(blockers))
    return {
        "status": "passed" if not blockers else "blocked_visual_review_required",
        "passed": not blockers,
        "manifest_path": manifest_path or None,
        "manifest_status": _string(manifest.get("status")) or None,
        "short_visual_sanity_passed": manifest.get("short_visual_sanity_passed") is True,
        "visual_profile": _string(manifest.get("visual_profile")) or None,
        "visually_useful_rollout": manifest.get("visually_useful_rollout") is True,
        "contact_sheet_refs": contact_sheet_refs,
        "provenance_refs": provenance_refs,
        "review_label_refs": review_label_refs,
        "missing_review_label_ids": missing_review_label_ids,
        "missing_review_grade_success_label_ids": missing_review_grade_label_ids,
        "blockers": blockers,
        "claim_boundary": {
            "short_visual_sanity_is_review_quality_gate_not_task_success_proof": True,
            "generated_observations_are_support_artifacts_not_sensor_truth": True,
            "review_labels_required_for_success_label_use_in_ranking": True,
        },
    }


def _visual_review_gate_from_labels(labels: Mapping[str, Any]) -> Dict[str, Any]:
    label_rows = [row for row in labels.get("labels", []) or [] if isinstance(row, Mapping)]
    short_visual_sanity_gate = _short_visual_sanity_gate_from_labels(labels)
    visual_smoke_statuses = _string_list(labels.get("visual_smoke_statuses"))
    if not visual_smoke_statuses:
        visual_smoke_statuses = sorted(
            {
                _string(row.get("visual_smoke_status"))
                for row in label_rows
                if _string(row.get("visual_smoke_status"))
            }
        )
    visual_rollout_useful = bool(
        labels.get("visual_rollout_useful_for_task_success_review")
    ) or bool(
        label_rows
        and all(
            bool(row.get("visual_rollout_useful_for_task_success_review")) for row in label_rows
        )
    )
    fixture_only = bool(labels.get("fixture_evaluator_only")) or any(
        bool(row.get("fixture_evaluator_only")) for row in label_rows
    )
    review_grade_success_labels = bool(
        labels.get("review_grade_success_labels")
        and visual_rollout_useful
        and not fixture_only
        and short_visual_sanity_gate["passed"]
    )
    blockers = _string_list(labels.get("visual_review_blockers"))
    for row in label_rows:
        blockers.extend(_string_list(row.get("visual_review_blockers")))
    blockers.extend(_string_list(short_visual_sanity_gate.get("blockers")))
    if fixture_only and FIXTURE_VISUAL_REVIEW_BLOCKER not in blockers:
        blockers.append(FIXTURE_VISUAL_REVIEW_BLOCKER)
    if not visual_rollout_useful and not blockers:
        blockers.append("generated_rollout_visual_smoke_missing_or_failed")
    blockers = sorted(set(blockers))
    status = (
        "review_grade_success_labels_available"
        if review_grade_success_labels
        else "fixture_evaluator_only"
        if fixture_only
        else "blocked_visual_review_required"
    )
    return {
        "status": status,
        "visual_smoke_status": labels.get("visual_smoke_status")
        or (
            visual_smoke_statuses[0]
            if len(visual_smoke_statuses) == 1
            else "mixed_visual_smoke_statuses"
            if visual_smoke_statuses
            else FIXTURE_VISUAL_SMOKE_STATUS
        ),
        "visual_smoke_statuses": visual_smoke_statuses,
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
        "review_grade_visual_evidence_available": bool(
            labels.get("review_grade_visual_evidence_available") or visual_rollout_useful
        ),
        "review_grade_success_labels": review_grade_success_labels,
        "fixture_evaluator_only": fixture_only,
        "short_visual_sanity_gate": short_visual_sanity_gate,
        "review_quality_manifest_required_for_policy_ranking": True,
        "blockers": blockers,
    }


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _matrix_runs(matrix: Mapping[str, Any]) -> list[Dict[str, Any]]:
    runs = matrix.get("runs")
    if not isinstance(runs, list):
        return []
    normalized: list[Dict[str, Any]] = []
    for index, raw in enumerate(runs, start=1):
        if not isinstance(raw, Mapping):
            continue
        run = dict(raw)
        run_id = _string(run.get("scenario_eval_run_id") or run.get("scenarioEvalRunId"))
        if not run_id:
            run_id = f"scenario_eval_run_{index:04d}"
        run["scenario_eval_run_id"] = run_id
        normalized.append(run)
    return normalized


def _policy_candidates(
    *,
    request: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    wam_request = _mapping(request.get("wam_evaluation") or request.get("wamEvaluation"))
    raw = (
        request.get("policy_candidates")
        or request.get("policyCandidates")
        or request.get("policies")
        or request.get("checkpoints")
        or wam_request.get("policy_candidates")
        or wam_request.get("policyCandidates")
        or wam_request.get("policies")
        or wam_request.get("checkpoints")
    )
    candidates: list[Dict[str, Any]] = []
    if isinstance(raw, list):
        for index, item in enumerate(raw, start=1):
            payload = _mapping(item)
            policy_id = (
                _string(payload.get("policy_id") or payload.get("policyId"))
                or f"policy_candidate_{index:02d}"
            )
            capabilities = _string_list(
                payload.get("capabilities")
                or payload.get("policy_capabilities")
                or payload.get("policyCapabilities")
                or payload.get("supported_failure_modes")
                or payload.get("supportedFailureModes")
            )
            candidates.append(
                {
                    **payload,
                    "policy_id": policy_id,
                    "display_name": _string(payload.get("display_name") or payload.get("name"))
                    or policy_id,
                    "capabilities": sorted(set(capabilities)),
                }
            )
    if candidates:
        return candidates
    selected = _string_list(policy_manifest.get("selected_modalities"))
    policy_id = _string(policy_manifest.get("policy_id") or policy_manifest.get("policyId"))
    if not policy_id:
        policy_id = selected[0] if selected else "policy_package_candidate"
    return [
        {
            "policy_id": policy_id,
            "display_name": policy_id,
            "capabilities": _string_list(policy_manifest.get("policy_capabilities")),
            "source": "policy_package_manifest",
        }
    ]


def _risk_profile(run: Mapping[str, Any]) -> Dict[str, Any]:
    text = " ".join(
        _string(run.get(key))
        for key in (
            "scenario_eval_run_id",
            "scenario_id",
            "task_id",
            "variation_name",
        )
    ).lower()
    required: list[str] = []
    failures: list[str] = []
    if any(marker in text for marker in ("blocked", "obstacle", "narrow", "clearance")):
        required.append("clearance_aware_navigation")
        failures.append("blocked_path_or_clearance_failure")
    if any(marker in text for marker in ("human", "forklift", "crossing", "dynamic")):
        required.append("dynamic_obstacle_yield")
        failures.append("dynamic_agent_safety_failure")
    if any(marker in text for marker in ("occlusion", "glare", "missing_label", "wrong_object")):
        required.append("visual_recheck")
        failures.append("perception_ambiguity_failure")
    if any(marker in text for marker in ("grasp", "place", "object_rotation", "cart_shifted")):
        required.append("grasp_alignment_correction")
        failures.append("manipulation_alignment_failure")
    return {
        "required_capabilities": sorted(set(required)),
        "candidate_failure_modes": sorted(set(failures)),
    }


def _policy_supports(policy: Mapping[str, Any], capability: str) -> bool:
    capabilities = set(_string_list(policy.get("capabilities")))
    return "all" in capabilities or capability in capabilities


def _forced_failures(policy: Mapping[str, Any], run: Mapping[str, Any]) -> bool:
    profile = _mapping(policy.get("fixture_success_profile") or policy.get("success_profile"))
    fail_variations = set(_string_list(profile.get("fail_variation_names")))
    fail_runs = set(_string_list(profile.get("fail_scenario_eval_run_ids")))
    variation_name = _string(run.get("variation_name") or run.get("variationName"))
    run_id = _string(run.get("scenario_eval_run_id"))
    return variation_name in fail_variations or run_id in fail_runs


def _rollout_for_run(
    *,
    job_dir: Path,
    substrate: str,
    policy: Mapping[str, Any],
    run: Mapping[str, Any],
    index: int,
    generated_at: str,
) -> Dict[str, Any]:
    policy_id = _string(policy.get("policy_id")) or "policy"
    run_id = _string(run.get("scenario_eval_run_id"))
    risk = _risk_profile(run)
    missing = [
        capability
        for capability in _string_list(risk.get("required_capabilities"))
        if not _policy_supports(policy, capability)
    ]
    forced_failure = _forced_failures(policy, run)
    variation_name = _string(run.get("variation_name") or run.get("variationName"))
    registered_ood = _mapping(run.get("registered_ood") or run.get("registeredOod"))
    ood_flags = sorted(
        axis for axis, value in registered_ood.items() if axis in SC3_OOD_AXES and value is True
    )
    ood_registration_blockers = sorted(
        {
            *(["ood_axes_registration_missing"] if not registered_ood else []),
            *[f"unknown_ood_axis:{axis}" for axis in registered_ood if axis not in SC3_OOD_AXES],
            *[
                f"ood_axis_value_not_strict_boolean:{axis}"
                for axis, value in registered_ood.items()
                if axis in SC3_OOD_AXES and not isinstance(value, bool)
            ],
            *[
                f"ood_axis_registration_missing:{axis}"
                for axis in SC3_OOD_AXES
                if axis not in registered_ood
            ],
        }
    )
    registered_ood_axes_complete = not ood_registration_blockers and set(registered_ood) == set(
        SC3_OOD_AXES
    )
    uncertainty = min(
        0.95,
        round(
            0.12
            + 0.11 * len(missing)
            + 0.08 * len(_string_list(risk.get("required_capabilities")))
            + 0.18 * len(ood_flags),
            6,
        ),
    )
    success = not missing and not forced_failure and uncertainty < 0.75
    failure_modes = [] if success else _string_list(risk.get("candidate_failure_modes"))
    if forced_failure and "fixture_policy_failure" not in failure_modes:
        failure_modes.append("fixture_policy_failure")
    rollout_id = f"wam_{_safe_id(policy_id)}_{_safe_id(run_id)}"
    attempt_id = f"{rollout_id}_attempt"
    media_dir = job_dir / "wam_rollouts"
    ensure_dir(media_dir)
    support_manifest_path = media_dir / f"{rollout_id}.json"
    support_manifest = {
        "schema_version": "fixture_wam_rollout_support_manifest.v1",
        "generated_at": generated_at,
        "rollout_id": rollout_id,
        "evaluation_substrate": substrate,
        "policy_id": policy_id,
        "scenario_eval_run_id": run_id,
        "condition_id": _string(run.get("condition_id") or run.get("conditionId")) or None,
        "replicate_id": _string(run.get("replicate_id") or run.get("replicateId")) or None,
        "replicate_seed": run.get("replicate_seed")
        if run.get("replicate_seed") is not None
        else run.get("seed"),
        "generated_video_available": False,
        "deterministic_fixture_frames": [
            {
                "frame_index": 0,
                "description": "initial captured-site conditioned observation",
            },
            {
                "frame_index": 1,
                "description": "policy action applied through fixture WAM transition",
            },
            {
                "frame_index": 2,
                "description": "fixture outcome frame used by the vision success judge",
            },
        ],
        "claim_boundary": {
            "support_manifest_not_video_truth": True,
            "model_derived_support_artifact": True,
            "raw_capture_evidence": False,
        },
    }
    write_json(support_manifest_path, support_manifest)
    return {
        "rollout_id": rollout_id,
        "attempt_id": attempt_id,
        "generated_at": generated_at,
        "evaluation_substrate": substrate,
        "simulator_engine": substrate,
        "policy_id": policy_id,
        "policy_display_name": _string(policy.get("display_name")) or policy_id,
        "scenario_eval_run_id": run_id,
        "scenario_variation_instance_id": run.get("scenario_variation_instance_id")
        or run.get("scenarioVariationInstanceId"),
        "task_id": _string(run.get("task_id") or run.get("taskId")),
        "scenario_id": _string(run.get("scenario_id") or run.get("scenarioId")),
        "variation_name": variation_name or None,
        "rollout_index": index,
        "predicted_success": success,
        "required_policy_capabilities": _string_list(risk.get("required_capabilities")),
        "policy_capabilities": _string_list(policy.get("capabilities")),
        "missing_policy_capabilities": missing,
        "failure_mode_ids": failure_modes,
        "uncertainty_score": uncertainty,
        "ood_flags": ood_flags,
        "ood_registration_blockers": ood_registration_blockers,
        "registered_ood_axes_complete": registered_ood_axes_complete,
        "ood_axes_source": "frozen_registered_ood_mapping",
        "metrics": {
            "cycle_time_seconds": round(18.0 + index * 0.05 + len(missing) * 2.0, 6),
            "intervention_count": 0 if success else 1,
            "contact_event_count": 0 if success else int("clearance_aware_navigation" in missing),
            "safety_event_count": 0 if "dynamic_obstacle_yield" not in missing else 1,
            "world_model_uncertainty": uncertainty,
            "ood_flag_count": len(ood_flags),
        },
        "artifact_paths": {
            "rollout_support_manifest": str(support_manifest_path.relative_to(job_dir)),
        },
        "claim_boundary": {
            "model_derived_support_artifact": True,
            "raw_capture_evidence": False,
            "predicted_success_is_capability_prediction_not_task_success_proof": True,
            "task_success_proven": False,
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "real_world_outcome_proven": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _rollout_manifest(
    *,
    job_id: str,
    substrate: str,
    rollouts: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": WAM_ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "completed" if rollouts else "blocked_missing_rollouts",
        "evaluation_substrate": substrate,
        "rollout_count": len(rollouts),
        "rollouts": [dict(rollout) for rollout in rollouts],
        "artifact_dir": "wam_rollouts",
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _rollout_results(
    *,
    job_id: str,
    substrate: str,
    rollouts: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    success_count = sum(1 for rollout in rollouts if bool(rollout.get("predicted_success")))
    return {
        "schema_version": WAM_ROLLOUT_RESULTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "completed" if rollouts else "blocked_missing_rollouts",
        "evaluation_substrate": substrate,
        "rollout_count": len(rollouts),
        "predicted_success_count": success_count,
        "predicted_failure_count": len(rollouts) - success_count,
        "predicted_success_rate": round(success_count / len(rollouts), 6) if rollouts else 0.0,
        "failure_mode_ids": sorted(
            {mode for rollout in rollouts for mode in _string_list(rollout.get("failure_mode_ids"))}
        ),
        "ood_rollout_count": sum(
            1 for rollout in rollouts if _string_list(rollout.get("ood_flags"))
        ),
        "rollouts": [dict(rollout) for rollout in rollouts],
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _normalized_attempt_trace(
    *,
    substrate: str,
    labels: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    label_rows = [
        dict(label) for label in labels.get("labels", []) or [] if isinstance(label, Mapping)
    ]
    short_visual_sanity_gate = _short_visual_sanity_gate_from_labels(labels)
    shared_review_refs = _review_label_refs(labels)
    shared_contact_refs = _string_list(short_visual_sanity_gate.get("contact_sheet_refs"))
    shared_provenance_refs = _string_list(short_visual_sanity_gate.get("provenance_refs"))
    attempts: list[Dict[str, Any]] = []
    for label in label_rows:
        success_verdict = label.get("task_success")
        # Strict boolean only: a string like "true"/"1" or a missing field is a review
        # gap and must fail closed, never coerce to success.
        strict_boolean_verdict = isinstance(success_verdict, bool)
        success = success_verdict is True
        review_label_refs = _dedupe([*shared_review_refs, *_review_label_refs(label)])
        frame_or_clip_refs = _dedupe_refs(
            [
                *_failure_frame_or_clip_refs(label),
                *shared_contact_refs,
                *_contact_sheet_refs(label),
            ]
        )
        visual_review_blockers = _dedupe(
            [
                *_string_list(label.get("visual_review_blockers")),
                *_string_list(short_visual_sanity_gate.get("blockers")),
                *(["review_grade_label_refs_missing"] if not review_label_refs else []),
                *(["task_success_label_not_strict_boolean"] if not strict_boolean_verdict else []),
            ]
        )
        # The upstream label field is a claim; review-grade standing must be re-derived
        # from the gates it depends on, not passed through.
        review_grade_success_label = bool(
            label.get("review_grade_success_label")
            and label.get("review_grade_visual_evidence_available")
            and review_label_refs
            and strict_boolean_verdict
            and not visual_review_blockers
        )
        attempts.append(
            {
                "attempt_id": label.get("attempt_id"),
                "rollout_id": label.get("rollout_id"),
                "scenario_eval_run_id": label.get("scenario_eval_run_id"),
                "scenario_variation_instance_id": label.get("scenario_variation_instance_id"),
                "task_id": label.get("task_id"),
                "scenario_id": label.get("scenario_id"),
                "variation_name": label.get("variation_name"),
                "policy_id": label.get("policy_id"),
                "evaluation_substrate": substrate,
                "simulator_engine": substrate,
                "status": "completed" if success else "failed",
                "success": success,
                "task_success": success,
                "task_success_verdict_strict_boolean": strict_boolean_verdict,
                "failure_mode_ids": _string_list(label.get("failure_mode_ids")),
                "confidence": label.get("confidence"),
                "evidence_refs": _failure_evidence_refs(
                    label,
                    extra_refs=(
                        "vision_success_labels.json",
                        *_string_list(short_visual_sanity_gate.get("manifest_path")),
                        *shared_contact_refs,
                        *shared_provenance_refs,
                        *review_label_refs,
                    ),
                ),
                "source_trace_refs": _dedupe_refs(["vision_success_labels.json"]),
                "frame_or_clip_refs": frame_or_clip_refs,
                "visual_smoke_ref": label.get("visual_smoke_ref") or label.get("visualSmokeRef"),
                "visual_smoke_status": label.get("visual_smoke_status"),
                "visual_rollout_useful_for_task_success_review": bool(
                    label.get("visual_rollout_useful_for_task_success_review")
                ),
                "visual_review_blockers": visual_review_blockers,
                "fixture_evaluator_only": bool(label.get("fixture_evaluator_only")),
                "review_grade_visual_evidence_available": bool(
                    label.get("review_grade_visual_evidence_available")
                ),
                "review_grade_success_label": review_grade_success_label,
                "review_status": label.get("review_status") or label.get("review_label_status"),
                "review_label_refs": review_label_refs,
                "short_visual_sanity_gate": short_visual_sanity_gate,
                "generated_wam_rollout": True,
                "model_derived_support_artifact": True,
                "metrics": {
                    "world_model_uncertainty": _number(label.get("uncertainty_score")),
                    "intervention_count": 0 if success else 1,
                    "contact_event_count": 0,
                    "safety_event_count": 0,
                    "cycle_time_seconds": 20.0,
                },
                "artifact_paths": {"vision_success_label": "vision_success_labels.json"},
                "claim_boundary": {
                    "generated_wam_attempt": True,
                    "model_derived_support_artifact": True,
                    "visual_smoke_required_for_review_grade_success_label": True,
                    "short_visual_sanity_required_for_review_grade_success_label": True,
                    "review_label_refs_required_for_review_grade_success_label": True,
                    "visual_rollout_useful_for_task_success_review": bool(
                        label.get("visual_rollout_useful_for_task_success_review")
                    ),
                    "fixture_evaluator_only": bool(label.get("fixture_evaluator_only")),
                    "review_grade_success_label": review_grade_success_label,
                    "simulator_execution_proven": False,
                    "robot_policy_execution_proven": False,
                    "rank_fidelity_result_proven": False,
                },
            }
        )
    successful = [attempt for attempt in attempts if attempt["success"]]
    failed = [attempt for attempt in attempts if not attempt["success"]]
    run_ids = sorted(
        {
            _string(attempt.get("scenario_eval_run_id"))
            for attempt in attempts
            if attempt.get("scenario_eval_run_id")
        }
    )
    return {
        "schema_version": NORMALIZED_ATTEMPT_TRACE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if attempts else "blocked_missing_attempts",
        "runner": f"{substrate}_evaluator",
        "evaluation_substrate": substrate,
        "attempt_count": len(attempts),
        "successful_task_attempt_count": len(successful),
        "failed_task_attempt_count": len(failed),
        "task_success_rate": round(len(successful) / len(attempts), 6) if attempts else 0.0,
        "task_success_summary": {
            "attempt_count": len(attempts),
            "successful_attempt_count": len(successful),
            "failed_attempt_count": len(failed),
            "task_success_rate": round(len(successful) / len(attempts), 6) if attempts else 0.0,
        },
        "covered_scenario_eval_run_ids": run_ids,
        "missing_scenario_eval_run_ids": [],
        "scenario_eval_run_coverage_complete": bool(run_ids),
        "short_visual_sanity_gate": short_visual_sanity_gate,
        "attempts": attempts,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _failure_labels(
    *,
    substrate: str,
    trace: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    attempts = [item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)]
    failures = [attempt for attempt in attempts if not bool(attempt.get("success"))]
    labels: list[Dict[str, Any]] = []
    labels_missing_failure_modes: list[str] = []
    labels_missing_evidence_refs: list[str] = []
    labels_missing_review_status: list[str] = []
    nonreviewable_labels: list[str] = []
    visual_smoke_statuses: list[str] = []
    visual_review_blockers: list[str] = []
    for index, attempt in enumerate(failures, start=1):
        label_id = f"wam_failure_label_{index:04d}"
        failure_mode_ids = _string_list(attempt.get("failure_mode_ids"))
        frame_refs = _failure_frame_or_clip_refs(attempt)
        source_trace_refs = _dedupe_refs(
            [
                "normalized_attempt_trace.json",
                *_string_list(attempt.get("source_trace_refs")),
                "vision_success_labels.json",
            ]
        )
        evidence_refs = _failure_evidence_refs(
            attempt,
            extra_refs=tuple(source_trace_refs),
        )
        visual_smoke_ref = (
            _string(attempt.get("visual_smoke_ref") or attempt.get("visualSmokeRef")) or None
        )
        review_status = _failure_review_status(
            supplied_review_status=attempt.get("review_status"),
            supplied_status=attempt.get("status"),
            generated_rollout=True,
            frame_or_clip_ref_count=len(frame_refs),
        )
        root_cause_category = _failure_root_cause_category(
            failure_mode_ids,
            ood_flags=_string_list(attempt.get("ood_flags")),
            failure_reason="fixture_wam_predicted_task_failure",
        )
        unknown_when_evidence_weak = bool(
            not frame_refs
            or not evidence_refs
            or review_status == "non_reviewable_failure_hypothesis"
        )
        visual_smoke_status = (
            _string(attempt.get("visual_smoke_status")) or FIXTURE_VISUAL_SMOKE_STATUS
        )
        visual_rollout_useful = bool(attempt.get("visual_rollout_useful_for_task_success_review"))
        attempt_visual_blockers = _string_list(attempt.get("visual_review_blockers"))
        short_visual_sanity_gate = _mapping(attempt.get("short_visual_sanity_gate"))
        review_label_refs = _string_list(attempt.get("review_label_refs"))
        fixture_only = bool(attempt.get("fixture_evaluator_only"))
        if fixture_only and FIXTURE_VISUAL_REVIEW_BLOCKER not in attempt_visual_blockers:
            attempt_visual_blockers.append(FIXTURE_VISUAL_REVIEW_BLOCKER)
        attempt_visual_blockers.extend(_string_list(short_visual_sanity_gate.get("blockers")))
        if short_visual_sanity_gate.get("passed") is not True:
            attempt_visual_blockers.append("short_visual_sanity_gate_not_passed")
        if not review_label_refs:
            attempt_visual_blockers.append("review_grade_failure_label_refs_missing")
        if not visual_rollout_useful and not attempt_visual_blockers:
            attempt_visual_blockers.append("generated_rollout_visual_smoke_missing_or_failed")
        visual_smoke_statuses.append(visual_smoke_status)
        visual_review_blockers.extend(attempt_visual_blockers)
        label = {
            "label_id": label_id,
            "attempt_id": attempt.get("attempt_id"),
            "rollout_id": attempt.get("rollout_id"),
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "policy_id": attempt.get("policy_id"),
            "evaluation_substrate": substrate,
            "failure_mode_ids": failure_mode_ids,
            "failure_reason": "fixture_wam_predicted_task_failure",
            "source": "vision_success_labels",
            "evidence_refs": evidence_refs,
            "source_trace_refs": source_trace_refs,
            "frame_or_clip_refs": frame_refs,
            "visual_smoke_ref": visual_smoke_ref,
            "confidence": attempt.get("confidence"),
            "status": "review_required",
            "review_status": review_status,
            "reviewer_acceptance_required": True,
            "root_cause_category": root_cause_category,
            "remediation_candidate": _failure_remediation_candidate(
                root_cause_category,
                failure_mode_ids,
            ),
            "unknown_when_evidence_weak": unknown_when_evidence_weak,
            "non_reviewable_failure_hypothesis": (
                review_status == "non_reviewable_failure_hypothesis"
            ),
            "visual_smoke_status": visual_smoke_status,
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "visual_review_blockers": sorted(set(attempt_visual_blockers)),
            "short_visual_sanity_gate": short_visual_sanity_gate,
            "review_label_refs": review_label_refs,
            "fixture_evaluator_only": fixture_only,
            "review_grade_failure_diagnosis": False,
            "authoritative_failure_diagnosis": False,
            "generated_wam_rollout": True,
            "model_derived_support_artifact": True,
            "proof_effect": FAILURE_LABEL_PROOF_EFFECT,
        }
        if not failure_mode_ids:
            labels_missing_failure_modes.append(label_id)
        if not evidence_refs:
            labels_missing_evidence_refs.append(label_id)
        if not review_status:
            labels_missing_review_status.append(label_id)
        if review_status == "non_reviewable_failure_hypothesis":
            nonreviewable_labels.append(label_id)
        if fixture_only or not visual_rollout_useful:
            nonreviewable_labels.append(label_id)
        if attempt_visual_blockers:
            nonreviewable_labels.append(label_id)
        labels.append(label)
    coverage_blockers = []
    if labels_missing_failure_modes:
        coverage_blockers.append("failure_labels_missing_failure_mode_ids")
    if labels_missing_evidence_refs:
        coverage_blockers.append("failure_labels_missing_evidence_refs")
    if labels_missing_review_status:
        coverage_blockers.append("failure_labels_missing_review_status")
    deduped_nonreviewable_labels = sorted(set(nonreviewable_labels))
    visual_review_blockers = sorted(set(visual_review_blockers))
    visual_rollout_useful_for_review = (
        bool(failures)
        and not visual_review_blockers
        and all(
            bool(attempt.get("visual_rollout_useful_for_task_success_review"))
            for attempt in failures
        )
    )
    return {
        "schema_version": FAILURE_LABELS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_failures_labeled",
        "evaluation_substrate": substrate,
        "visual_smoke_status": visual_smoke_statuses[0]
        if len(set(visual_smoke_statuses)) == 1
        else "mixed_visual_smoke_statuses"
        if visual_smoke_statuses
        else FIXTURE_VISUAL_SMOKE_STATUS,
        "visual_smoke_statuses": sorted(set(visual_smoke_statuses)),
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful_for_review,
        "visual_review_blockers": visual_review_blockers,
        "fixture_evaluator_only": any(
            bool(attempt.get("fixture_evaluator_only")) for attempt in failures
        ),
        "review_grade_failure_diagnosis": False,
        "authoritative_failure_diagnosis": False,
        "label_count": len(failures),
        "failed_attempt_count": len(failures),
        "covered_failed_attempt_ids": sorted(
            _string(attempt.get("attempt_id")) for attempt in failures
        ),
        "missing_failed_attempt_ids": [],
        "covered_failed_scenario_eval_run_ids": sorted(
            {
                _string(attempt.get("scenario_eval_run_id"))
                for attempt in failures
                if attempt.get("scenario_eval_run_id")
            }
        ),
        "missing_failed_scenario_eval_run_ids": [],
        "failed_run_label_coverage_complete": True,
        "failure_diagnosis_coverage_complete": not coverage_blockers,
        "failure_diagnosis_review_complete": not deduped_nonreviewable_labels,
        "failure_diagnosis_complete": bool(
            failures and not coverage_blockers and not deduped_nonreviewable_labels
        )
        if failures
        else True,
        "failure_diagnosis_blockers": [
            *coverage_blockers,
            *visual_review_blockers,
            *(
                ["failure_labels_nonreviewable_failure_hypotheses"]
                if deduped_nonreviewable_labels
                else []
            ),
        ],
        "labels_missing_failure_mode_ids": labels_missing_failure_modes,
        "labels_missing_evidence_refs": labels_missing_evidence_refs,
        "labels_missing_review_status": labels_missing_review_status,
        "nonreviewable_failure_hypothesis_label_ids": deduped_nonreviewable_labels,
        "labels": labels,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _prediction_ledgers(
    *,
    substrate: str,
    trace: Mapping[str, Any],
    failure_labels: Mapping[str, Any] | None = None,
    generated_at: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    attempts = [item for item in trace.get("attempts", []) or [] if isinstance(item, Mapping)]
    records = [
        {
            "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": attempt.get("scenario_variation_instance_id"),
            "variation_name": attempt.get("variation_name"),
            "task_id": attempt.get("task_id"),
            "scenario_id": attempt.get("scenario_id"),
            "policy_id": attempt.get("policy_id"),
            "evaluation_substrate": attempt.get("evaluation_substrate"),
            "predicted_status": "passed" if attempt.get("success") else "failed",
            "predicted_success": bool(attempt.get("success")),
            "failure_mode_ids": _string_list(attempt.get("failure_mode_ids")),
            "world_model_uncertainty": _mapping(attempt.get("metrics")).get(
                "world_model_uncertainty"
            ),
            "actual_status": "needs_real_world_validation",
            "source": f"{substrate}_eval",
        }
        for attempt in attempts
    ]
    prediction = {
        "schema_version": PREDICTION_OUTCOME_LEDGER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if records else "not_available",
        "evaluation_substrate": substrate,
        "record_count": len(records),
        "records": records,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    calibration = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "not_measured",
        "evaluation_substrate": substrate,
        "record_count": len(records),
        "records": records,
        "accepted_anchor_schema": {
            "schema_version": ACCEPTED_REAL_WORLD_ANCHOR_SCHEMA_VERSION,
            "join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
            "required_prediction_fields": [
                *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                "predicted_success",
            ],
            "required_actual_fields": [
                *ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS,
                "actual_success",
                "owner_evidence_or_operator_attestation",
            ],
        },
        "accepted_anchor_count": 0,
        "sim_vs_real_calibration_score": None,
        "spearman_rank_correlation": None,
        "pearson_success_rate_correlation": None,
        "mean_maximum_rank_violation": None,
        "mmrv": None,
        "mean_absolute_success_rate_error": None,
        "confidence_intervals": {},
        "blockers": ["insufficient_anchor_count", "unmatched_prediction_rows"]
        if records
        else ["insufficient_anchor_count"],
        "srcc_validation_status": "not_measured",
        "customer_specific_srcc_claimed": False,
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    failures = [record for record in records if not record["predicted_success"]]
    label_rows = [
        dict(label)
        for label in (failure_labels or {}).get("labels", []) or []
        if isinstance(label, Mapping)
    ]
    labels_by_attempt = {
        _string(label.get("attempt_id")): label
        for label in label_rows
        if _string(label.get("attempt_id"))
    }
    labels_by_run = {
        _string(label.get("scenario_eval_run_id")): label
        for label in label_rows
        if _string(label.get("scenario_eval_run_id"))
    }
    aggregation_map: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    dominant_map: Dict[str, Dict[str, Any]] = {}
    for record in failures:
        label = labels_by_run.get(
            _string(record.get("scenario_eval_run_id"))
        ) or labels_by_attempt.get(_string(record.get("attempt_id")))
        failure_mode_ids = _string_list(
            (label or {}).get("failure_mode_ids") if label else record.get("failure_mode_ids")
        ) or ["unknown_failure_mode"]
        root_cause = _string(
            (label or {}).get("root_cause_category")
        ) or _failure_root_cause_category(
            failure_mode_ids,
            failure_reason=_string((label or {}).get("failure_reason")),
        )
        evidence_refs = _failure_evidence_refs(label or record)
        media_refs = _failure_frame_or_clip_refs(label or record)
        exemplar = {
            "attempt_id": (label or {}).get("attempt_id") or record.get("attempt_id"),
            "scenario_eval_run_id": record.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": record.get("scenario_variation_instance_id"),
            "variation_name": record.get("variation_name"),
            "policy_id": record.get("policy_id"),
            "task_id": record.get("task_id"),
            "scenario_id": record.get("scenario_id"),
            "failure_mode_ids": failure_mode_ids,
            "root_cause_category": root_cause,
            "evidence_refs": evidence_refs,
            "frame_or_clip_refs": media_refs,
            "visual_smoke_ref": (label or {}).get("visual_smoke_ref"),
            "review_status": (label or {}).get("review_status"),
        }
        for failure_mode_id in failure_mode_ids:
            key = (
                _string(record.get("policy_id")) or "unknown_policy",
                _string(record.get("task_id")) or "unknown_task",
                _string(record.get("scenario_id")) or "unknown_scenario",
                failure_mode_id,
                root_cause,
            )
            bucket = aggregation_map.setdefault(
                key,
                {
                    "policy_id": key[0],
                    "task_id": key[1],
                    "scenario_id": key[2],
                    "failure_mode_id": key[3],
                    "root_cause_category": key[4],
                    "failed_attempt_count": 0,
                    "scenario_eval_run_ids": [],
                    "exemplar_failed_attempts": [],
                    "media_refs": [],
                    "evidence_refs": [],
                },
            )
            bucket["failed_attempt_count"] += 1
            bucket["scenario_eval_run_ids"] = _dedupe_refs(
                [
                    *bucket["scenario_eval_run_ids"],
                    _string(record.get("scenario_eval_run_id")),
                ]
            )
            if len(bucket["exemplar_failed_attempts"]) < 3:
                bucket["exemplar_failed_attempts"].append(exemplar)
            bucket["media_refs"] = _dedupe_refs([*bucket["media_refs"], *media_refs])
            bucket["evidence_refs"] = _dedupe_refs([*bucket["evidence_refs"], *evidence_refs])
            dominant = dominant_map.setdefault(
                failure_mode_id,
                {
                    "failure_mode_id": failure_mode_id,
                    "failed_attempt_count": 0,
                    "root_cause_categories": [],
                    "exemplar_failed_attempts": [],
                    "media_refs": [],
                    "evidence_refs": [],
                },
            )
            dominant["failed_attempt_count"] += 1
            dominant["root_cause_categories"] = _dedupe_refs(
                [*dominant["root_cause_categories"], root_cause]
            )
            if len(dominant["exemplar_failed_attempts"]) < 3:
                dominant["exemplar_failed_attempts"].append(exemplar)
            dominant["media_refs"] = _dedupe_refs([*dominant["media_refs"], *media_refs])
            dominant["evidence_refs"] = _dedupe_refs([*dominant["evidence_refs"], *evidence_refs])
    aggregations = sorted(
        aggregation_map.values(),
        key=lambda row: (
            -int(row["failed_attempt_count"]),
            _string(row["policy_id"]),
            _string(row["task_id"]),
            _string(row["scenario_id"]),
            _string(row["failure_mode_id"]),
            _string(row["root_cause_category"]),
        ),
    )
    dominant_failure_modes = sorted(
        dominant_map.values(),
        key=lambda row: (-int(row["failed_attempt_count"]), _string(row["failure_mode_id"])),
    )
    breakage = {
        "schema_version": BREAKAGE_LIBRARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "review_required" if failures else "no_breakages_recorded",
        "evaluation_substrate": substrate,
        "record_count": len(failures),
        "records": failures,
        "aggregation_keys": [
            "policy_id",
            "task_id",
            "scenario_id",
            "failure_mode_id",
            "root_cause_category",
        ],
        "aggregation_count": len(aggregations),
        "aggregations": aggregations,
        "dominant_failure_modes": dominant_failure_modes,
        "dominant_failure_mode_id": dominant_failure_modes[0]["failure_mode_id"]
        if dominant_failure_modes
        else None,
        "source_failure_labels": "failure_labels.json",
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    return prediction, calibration, breakage


def _wam_report_attempts(trace: Mapping[str, Any]) -> list[Dict[str, Any]]:
    attempts: list[Dict[str, Any]] = []
    for index, attempt in enumerate(trace.get("attempts") or []):
        if not isinstance(attempt, Mapping):
            continue
        row = dict(attempt)
        strict_success = coerce_strict_success(
            row.get("success")
            if "success" in row
            else row.get("task_success")
            if "task_success" in row
            else _mapping(row.get("task_outcome")).get("task_success")
        )
        if strict_success is not None:
            row["success"] = strict_success
        row.setdefault("attempt_id", f"attempt_{index + 1:04d}")
        attempts.append(row)
    return attempts


def _wam_trace_task_success(attempts: Sequence[Mapping[str, Any]]) -> bool | None:
    if not attempts:
        return None
    verdicts: list[bool] = []
    for attempt in attempts:
        strict_success = coerce_strict_success(attempt.get("success"))
        if strict_success is None:
            return None
        verdicts.append(strict_success)
    return all(verdicts)


def _wam_task_metadata(
    *,
    request: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> Dict[str, Any]:
    for task in request.get("requested_tasks") or request.get("requestedTasks") or []:
        if isinstance(task, Mapping):
            return dict(task)
    for run in _matrix_runs(matrix):
        row = _mapping(run)
        if row:
            return {
                key: value
                for key, value in row.items()
                if key
                in {
                    "task_id",
                    "task_name",
                    "scenario_id",
                    "task_success_contract",
                    "success_contract",
                    "affordance_object_ids",
                    "target_object_ids",
                    "success_state_change",
                }
            }
    return {}


def _wam_review_verdicts(labels: Mapping[str, Any]) -> list[Dict[str, Any]]:
    verdicts: list[Dict[str, Any]] = []
    for row in labels.get("labels") or []:
        if not isinstance(row, Mapping):
            continue
        success = coerce_strict_success(row.get("task_success"))
        if success is None:
            continue
        verdicts.append(
            {
                "success": success,
                "reviewer": _string(row.get("reviewer") or row.get("source"))
                or "wam_vision_success_labels",
                "source_artifact": "vision_success_labels.json",
            }
        )
    return verdicts


def _wam_policy_id(
    *,
    policies: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> str | None:
    for row in [*attempts, *policies]:
        if isinstance(row, Mapping):
            policy_id = _string(row.get("policy_id") or row.get("policyId"))
            if policy_id:
                return policy_id
    return None


def _wam_task_eval_run_report(
    *,
    job_id: str,
    request: Mapping[str, Any],
    matrix: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
    substrate: str,
    labels: Mapping[str, Any],
    trace: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    provider_execution: Mapping[str, Any],
    policy_binding: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    attempts = _wam_report_attempts(trace)
    task_metadata = _wam_task_metadata(request=request, matrix=matrix)
    visual_review_ready = bool(
        scorecard.get("visual_rollout_useful_for_task_success_review")
        and not _string_list(scorecard.get("visual_review_blockers"))
    )
    freshness = build_artifact_freshness_evidence(
        artifact_run_id=job_id,
        current_run_id=job_id,
    )
    media_validity = build_media_validity(
        media_present=visual_review_ready,
        decodable=visual_review_ready,
        visual_stats={
            "blockers": _string_list(scorecard.get("visual_review_blockers")),
        },
        freshness=freshness if visual_review_ready else None,
    )
    review_task_success = build_review_task_success(
        media_validity=media_validity,
        reviewer_verdicts=_wam_review_verdicts(labels),
        camera_evidence={
            "robot_pov_camera_mode": "model_derived_wam_rollout",
            "visible_embodied_robot_action_evidence": visual_review_ready,
        },
    )
    task_success_contract = build_task_success_contract_result(
        task_metadata=task_metadata,
        trace_task_success=_wam_trace_task_success(attempts),
    )
    policy_id = _wam_policy_id(policies=policies, attempts=attempts)
    layers = {
        "media_validity": media_validity,
        "review_task_success": review_task_success,
        "task_success_contract": task_success_contract,
        "simulator_execution": build_simulator_execution(
            provider_runtime_status="blocked",
            output_artifacts_present=False,
            artifact_freshness=None,
            execution_log_present=False,
        ),
        "policy_action_execution": build_policy_action_execution(
            action_source="model_derived_wam_rollout",
            policy_id=policy_id,
            action_trace_present=False,
            actions_executed_in_simulator=False,
        ),
        "contact_state_change": build_contact_state_change_proof(
            proof_requirements=derive_task_proof_requirements(task_metadata),
            contact_reports=[],
            state_change_measurement=None,
        ),
        "physical_readiness": build_physical_readiness(
            real_robot_execution_evidence={"physical_robot_executed": False},
            deployment_approval={"approved": False},
        ),
    }
    rights_scope = _mapping(
        request.get("rights_privacy_scope") or request.get("rightsPrivacyScope")
    )
    rights_cleared = rights_scope.get("cleared") is True or _string(
        rights_scope.get("status")
    ).lower() in {"cleared", "pass", "passed"}
    return build_task_eval_run_report(
        job_id=job_id,
        scene_id=_string(matrix.get("scene_id") or request.get("scene_id")) or None,
        capture_id=_string(request.get("capture_id") or request.get("captureId")) or None,
        attempt_trace={**dict(trace), "attempts": attempts},
        task_metadata=task_metadata,
        success_claim_layers=layers,
        provider_execution={
            "wam_provider_execution_status": provider_execution.get("status"),
            "evaluation_substrate": substrate,
            "provider_runtime_success_is_not_task_success": True,
        },
        policy_binding={
            **_mapping(policy_binding),
            "policy_id": policy_id,
        },
        rights_privacy_gate={
            "status": "cleared" if rights_cleared else "not_cleared",
            "cleared": rights_cleared,
        },
        generated_at=generated_at,
    )


def _write_wam_artifacts(job_dir: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    for key, payload in payloads.items():
        write_json(job_dir / WAM_ARTIFACT_PATHS[key], payload)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _contained_evidence_file(
    *,
    evidence_root: Path | None,
    relative_path: object,
    max_bytes: int,
) -> tuple[Path | None, str | None]:
    if evidence_root is None:
        return None, "evidence_root_missing"
    text = _string(relative_path)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        return None, "evidence_path_invalid"
    root = evidence_root.resolve()
    candidate = root / path
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.stat()
    except OSError:
        return None, "evidence_file_missing_or_unreadable"
    if candidate.is_symlink() or not resolved.is_relative_to(root):
        return None, "evidence_path_symlink_or_escape"
    if not candidate.is_file() or metadata.st_size <= 0 or metadata.st_size > max_bytes:
        return None, "evidence_file_type_or_size_invalid"
    return candidate, None


def _load_hash_verified_json(
    *,
    evidence_root: Path | None,
    reference: object,
) -> tuple[dict[str, Any], str | None, str | None, list[str]]:
    ref = _mapping(reference)
    expected_digest = _string(ref.get("sha256")).lower()
    relative_path = _string(ref.get("path"))
    blockers: list[str] = []
    if SHA256_REF_PATTERN.fullmatch(expected_digest) is None:
        blockers.append("evidence_sha256_invalid")
    path, path_blocker = _contained_evidence_file(
        evidence_root=evidence_root,
        relative_path=relative_path,
        max_bytes=1024 * 1024,
    )
    if path_blocker:
        blockers.append(path_blocker)
    actual_digest: str | None = None
    payload: dict[str, Any] = {}
    if path is not None:
        try:
            actual_digest = _sha256_path(path)
            raw = read_json_any(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append("evidence_json_unreadable")
        else:
            payload = _mapping(raw)
            if not payload:
                blockers.append("evidence_json_not_object")
    if actual_digest is not None and actual_digest != expected_digest:
        blockers.append("evidence_sha256_mismatch")
    return payload, actual_digest, relative_path or None, sorted(set(blockers))


def _load_hash_verified_artifact(
    *,
    evidence_root: Path | None,
    reference: object,
) -> tuple[str | None, str | None, list[str]]:
    ref = _mapping(reference)
    expected_digest = _string(ref.get("sha256")).lower()
    relative_path = _string(ref.get("path"))
    blockers: list[str] = []
    if SHA256_REF_PATTERN.fullmatch(expected_digest) is None:
        blockers.append("artifact_sha256_invalid")
    path, path_blocker = _contained_evidence_file(
        evidence_root=evidence_root,
        relative_path=relative_path,
        max_bytes=512 * 1024 * 1024,
    )
    if path_blocker:
        blockers.append(path_blocker)
    actual_digest: str | None = None
    if path is not None:
        try:
            actual_digest = _sha256_path(path)
        except OSError:
            blockers.append("artifact_unreadable")
    if actual_digest is not None and actual_digest != expected_digest:
        blockers.append("artifact_sha256_mismatch")
    return actual_digest, relative_path or None, sorted(set(blockers))


def _canonical_signature_payload(
    payload: Mapping[str, Any],
    *,
    signature_field: str,
) -> bytes:
    unsigned = dict(payload)
    unsigned.pop(signature_field, None)
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _decode_ed25519_public_key(value: object) -> tuple[bytes | None, str | None]:
    try:
        raw = base64.b64decode(_string(value), validate=True)
    except (binascii.Error, ValueError):
        return None, "public_key_base64_invalid"
    if len(raw) != 32:
        return None, "public_key_length_invalid"
    return raw, None


def _public_key_fingerprint(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _verify_ed25519_signature(
    *,
    payload: Mapping[str, Any],
    signature_field: str,
    public_key_b64: object,
    expected_fingerprint: str,
) -> list[str]:
    signature = _mapping(payload.get(signature_field))
    blockers: list[str] = []
    if signature.get("algorithm") != "ed25519":
        blockers.append("signature_algorithm_invalid")
    if signature.get("key_fingerprint") != expected_fingerprint:
        blockers.append("signature_key_fingerprint_mismatch")
    raw_public_key, key_blocker = _decode_ed25519_public_key(public_key_b64)
    if key_blocker:
        blockers.append(key_blocker)
    elif _public_key_fingerprint(raw_public_key) != expected_fingerprint:
        blockers.append("public_key_fingerprint_mismatch")
    try:
        raw_signature = base64.b64decode(
            _string(signature.get("signature_base64")),
            validate=True,
        )
    except (binascii.Error, ValueError):
        blockers.append("signature_base64_invalid")
        raw_signature = b""
    if len(raw_signature) != 64:
        blockers.append("signature_length_invalid")
    if raw_public_key is not None and len(raw_signature) == 64:
        try:
            Ed25519PublicKey.from_public_bytes(raw_public_key).verify(
                raw_signature,
                _canonical_signature_payload(
                    payload,
                    signature_field=signature_field,
                ),
            )
        except (InvalidSignature, ValueError):
            blockers.append("signature_verification_failed")
    return sorted(set(blockers))


def _decision_authority_registry(
    *,
    substrate: str,
    evidence_root: Path | None,
    reference: object,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    payload, digest, path, blockers = _load_hash_verified_json(
        evidence_root=evidence_root,
        reference=reference,
    )
    if payload.get("schema_version") != DECISION_AUTHORITY_REGISTRY_SCHEMA_VERSION:
        blockers.append("authority_registry_schema_invalid")
    if payload.get("status") != "approved":
        blockers.append("authority_registry_not_approved")
    if payload.get("evaluation_substrate") != substrate:
        blockers.append("authority_registry_substrate_mismatch")
    if payload.get("fixture_evidence_decision_grade_allowed") is not False:
        blockers.append("authority_registry_fixture_boundary_invalid")
    if len(_string(payload.get("approved_by"))) < 3:
        blockers.append("authority_registry_approver_missing")
    if not _string(payload.get("registry_id")):
        blockers.append("authority_registry_id_missing")
    registry_version = payload.get("registry_version")
    if (
        isinstance(registry_version, bool)
        or not isinstance(registry_version, int)
        or registry_version < 1
    ):
        blockers.append("authority_registry_version_invalid")
    pinned_public_key, pinned_key_blocker = _decode_ed25519_public_key(
        PINNED_DECISION_GOVERNANCE_PUBLIC_KEY_B64
    )
    if pinned_key_blocker:
        blockers.append(f"pinned_governance_{pinned_key_blocker}")
    elif _public_key_fingerprint(pinned_public_key) != PINNED_DECISION_GOVERNANCE_KEY_FINGERPRINT:
        blockers.append("pinned_governance_key_fingerprint_invalid")
    blockers.extend(
        f"governance:{item}"
        for item in _verify_ed25519_signature(
            payload=payload,
            signature_field="governance_signature",
            public_key_b64=PINNED_DECISION_GOVERNANCE_PUBLIC_KEY_B64,
            expected_fingerprint=PINNED_DECISION_GOVERNANCE_KEY_FINGERPRINT,
        )
    )
    raw_authorities = payload.get("authorities")
    authorities: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_authorities, list):
        blockers.append("authority_registry_authorities_missing")
        raw_authorities = []
    for index, raw_authority in enumerate(raw_authorities):
        authority = _mapping(raw_authority)
        authority_id = _string(authority.get("authority_id"))
        role = _string(authority.get("role"))
        prefix = f"authority_registry_entry:{index}"
        if not authority_id:
            blockers.append(f"{prefix}:authority_id_missing")
            continue
        if authority_id in authorities:
            blockers.append(f"{prefix}:authority_id_duplicate")
        if role not in {"runtime", "reviewer"}:
            blockers.append(f"{prefix}:role_invalid")
        raw_key, key_blocker = _decode_ed25519_public_key(authority.get("public_key_base64"))
        if key_blocker:
            blockers.append(f"{prefix}:{key_blocker}")
        expected_fingerprint = _string(authority.get("public_key_fingerprint"))
        if (
            raw_key is None
            or not expected_fingerprint
            or _public_key_fingerprint(raw_key) != expected_fingerprint
        ):
            blockers.append(f"{prefix}:public_key_fingerprint_invalid")
        authorities[authority_id] = authority
    runtime_ids = {
        authority_id
        for authority_id, authority in authorities.items()
        if authority.get("role") == "runtime"
    }
    reviewer_ids = {
        authority_id
        for authority_id, authority in authorities.items()
        if authority.get("role") == "reviewer"
    }
    if not runtime_ids:
        blockers.append("trusted_runtime_authorities_missing")
    if not reviewer_ids:
        blockers.append("trusted_reviewer_authorities_missing")
    if runtime_ids & reviewer_ids:
        blockers.append("authority_registry_roles_not_separated")
    blockers = sorted(set(blockers))
    return (
        authorities,
        {
            "path": path,
            "sha256": digest,
            "registry_id": payload.get("registry_id"),
            "registry_version": payload.get("registry_version"),
            "governance_key_fingerprint": (PINNED_DECISION_GOVERNANCE_KEY_FINGERPRINT),
            "status": "verified" if not blockers else "blocked",
        },
        blockers,
    )


def _decision_grade_evidence_validation(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    substrate: str,
    evidence_root: Path | None,
    authority_registry_reference: object,
) -> dict[str, Any]:
    if substrate == "fixture_wam":
        return {
            "status": "inconclusive",
            "authority_registry": None,
            "validated_rows": [],
            "blockers": ["ranking_fixture_substrate_support_only"],
            "claim_boundary": {
                "fixture_evidence_is_support_only": True,
                "hash_verification_is_not_external_runtime_truth": True,
            },
        }
    authorities, registry, blockers = _decision_authority_registry(
        substrate=substrate,
        evidence_root=evidence_root,
        reference=authority_registry_reference,
    )
    seen_manifest_digests: dict[tuple[str, str], str] = {}
    seen_artifact_digests: dict[str, str] = {}
    seen_evidence_ids: dict[str, str] = {}
    seen_runtime_result_ids: dict[str, str] = {}
    validated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(label_rows):
        policy_id = _string(row.get("policy_id"))
        condition_id = _string(row.get("condition_id"))
        replicate_id = _string(row.get("replicate_id"))
        run_id = _string(row.get("scenario_eval_run_id"))
        seed = row.get("replicate_seed")
        result = row.get("task_success")
        matched_pair_id = _string(row.get("matched_pair_id"))
        task_id = _string(row.get("task_id"))
        scenario_id = _string(row.get("scenario_id"))
        criterion_id = _string(row.get("criterion_id"))
        initial_state_id = _string(row.get("initial_state_id"))
        runtime_result_id = _string(row.get("runtime_result_id"))
        row_id = f"{policy_id or 'missing'}:{condition_id or 'missing'}:{seed!s}"
        row_blockers: list[str] = []
        if not runtime_result_id:
            row_blockers.append("runtime_result_id_missing")
        else:
            first_row = seen_runtime_result_ids.get(runtime_result_id)
            if first_row is not None:
                row_blockers.append(f"runtime_result_id_reused_from:{first_row}")
            else:
                seen_runtime_result_ids[runtime_result_id] = row_id
        condition_manifest, condition_digest, condition_path, condition_blockers = (
            _load_hash_verified_json(
                evidence_root=evidence_root,
                reference=row.get("matched_condition_manifest"),
            )
        )
        row_blockers.extend(f"condition_manifest:{item}" for item in condition_blockers)
        if condition_manifest.get("schema_version") != MATCHED_CONDITION_MANIFEST_SCHEMA_VERSION:
            row_blockers.append("condition_manifest:schema_invalid")
        if condition_manifest.get("status") != "frozen":
            row_blockers.append("condition_manifest:status_invalid")
        condition_bindings = {
            "condition_id": condition_id,
            "replicate_seed": seed,
            "matched_pair_id": matched_pair_id,
            "replicate_id": replicate_id,
            "scenario_eval_run_id": run_id,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "criterion_id": criterion_id,
            "initial_state_id": initial_state_id,
        }
        for key, expected in condition_bindings.items():
            if condition_manifest.get(key) != expected:
                row_blockers.append(f"condition_manifest:binding_mismatch:{key}")
        evidence = _mapping(row.get("decision_grade_evidence"))
        validated_kinds: dict[str, Any] = {}
        for kind, schema_version in DECISION_EVIDENCE_SCHEMA_VERSIONS.items():
            payload, digest, path, ref_blockers = _load_hash_verified_json(
                evidence_root=evidence_root,
                reference=evidence.get(kind),
            )
            row_blockers.extend(f"{kind}:{item}" for item in ref_blockers)
            if digest is not None:
                digest_key = (kind, digest)
                first_row = seen_manifest_digests.get(digest_key)
                if first_row is not None:
                    row_blockers.append(f"{kind}:evidence_manifest_reused_from:{first_row}")
                else:
                    seen_manifest_digests[digest_key] = row_id
            if payload.get("schema_version") != schema_version:
                row_blockers.append(f"{kind}:schema_invalid")
            expected_status = {
                "execution": "completed",
                "media": "valid",
                "outcome_label": "accepted_reviewed_label",
            }[kind]
            if payload.get("status") != expected_status:
                row_blockers.append(f"{kind}:status_invalid")
            expected_bindings = {
                "policy_id": policy_id,
                "condition_id": condition_id,
                "replicate_id": replicate_id,
                "replicate_seed": seed,
                "scenario_eval_run_id": run_id,
                "matched_pair_id": matched_pair_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "criterion_id": criterion_id,
                "initial_state_id": initial_state_id,
                "runtime_result_id": runtime_result_id,
                "condition_manifest_sha256": condition_digest,
                "authority_registry_sha256": registry.get("sha256"),
                "result_task_success": result,
            }
            for key, expected in expected_bindings.items():
                if payload.get(key) != expected:
                    row_blockers.append(f"{kind}:binding_mismatch:{key}")
            evidence_id = _string(payload.get("evidence_id"))
            if not evidence_id:
                row_blockers.append(f"{kind}:evidence_id_missing")
            else:
                first_row = seen_evidence_ids.get(evidence_id)
                if first_row is not None:
                    row_blockers.append(f"{kind}:evidence_id_reused_from:{first_row}")
                else:
                    seen_evidence_ids[evidence_id] = row_id
            authority = _mapping(payload.get("authority"))
            required_role = "runtime" if kind == "execution" else "reviewer"
            authority_id = _string(authority.get("authority_id"))
            registered_authority = _mapping(authorities.get(authority_id))
            if authority.get("role") != required_role:
                row_blockers.append(f"{kind}:authority_role_invalid")
            if registered_authority.get("role") != required_role:
                row_blockers.append(f"{kind}:authority_untrusted")
            registered_fingerprint = _string(registered_authority.get("public_key_fingerprint"))
            if authority.get("public_key_fingerprint") != registered_fingerprint:
                row_blockers.append(f"{kind}:authority_fingerprint_mismatch")
            row_blockers.extend(
                f"{kind}:authority_signature:{item}"
                for item in _verify_ed25519_signature(
                    payload=payload,
                    signature_field="signature",
                    public_key_b64=registered_authority.get("public_key_base64"),
                    expected_fingerprint=registered_fingerprint,
                )
            )
            artifact_digest: str | None = None
            artifact_path: str | None = None
            artifact_digest, artifact_path, artifact_blockers = _load_hash_verified_artifact(
                evidence_root=evidence_root,
                reference=payload.get("artifact"),
            )
            row_blockers.extend(f"{kind}:{item}" for item in artifact_blockers)
            if artifact_digest is not None:
                first_row = seen_artifact_digests.get(artifact_digest)
                if first_row is not None:
                    row_blockers.append(f"{kind}:evidence_artifact_reused_from:{first_row}")
                else:
                    seen_artifact_digests[artifact_digest] = row_id
            validated_kinds[kind] = {
                "path": path,
                "sha256": digest,
                "artifact_path": artifact_path,
                "artifact_sha256": artifact_digest,
                "authority_id": authority_id or None,
            }
        row_blockers = sorted(set(row_blockers))
        blockers.extend(f"ranking_evidence:{row_id}:{item}" for item in row_blockers)
        validated_rows.append(
            {
                "row_index": index,
                "policy_id": policy_id or None,
                "condition_id": condition_id or None,
                "replicate_seed": seed,
                "matched_pair_id": matched_pair_id or None,
                "runtime_result_id": runtime_result_id or None,
                "condition_manifest": {
                    "path": condition_path,
                    "sha256": condition_digest,
                },
                "status": "verified" if not row_blockers else "blocked",
                "evidence": validated_kinds,
                "blockers": row_blockers,
            }
        )
    blockers = sorted(set(blockers))
    return {
        "status": "verified" if label_rows and not blockers else "inconclusive",
        "authority_registry": registry,
        "validated_rows": validated_rows,
        "blockers": blockers or ([] if label_rows else ["ranking_evidence_rows_missing"]),
        "claim_boundary": {
            "fixture_evidence_is_support_only": True,
            "hash_verification_is_not_external_runtime_truth": True,
            "nonfixture_authorities_must_be_registry_approved": True,
        },
    }


def _empirical_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _matched_cluster_bootstrap_pairwise(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    policy_ids: Sequence[str],
) -> dict[str, Any]:
    pairs = list(combinations(sorted(set(policy_ids)), 2))
    pair_count = len(pairs)
    if not pairs:
        return {
            "schema_version": "policy_ranking_matched_cluster_bootstrap.v1",
            "status": "inconclusive",
            "winner_policy_id": None,
            "pairwise_intervals": [],
            "blockers": ["paired_policy_comparisons_missing"],
        }
    lookup: dict[tuple[str, str], int] = {}
    pair_designs: dict[str, tuple[str, int]] = {}
    blockers: list[str] = []
    for row in label_rows:
        policy_id = _string(row.get("policy_id"))
        condition_id = _string(row.get("condition_id"))
        matched_pair_id = _string(row.get("matched_pair_id"))
        seed = row.get("replicate_seed")
        result = row.get("task_success")
        if (
            policy_id
            and condition_id
            and matched_pair_id
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(result, bool)
        ):
            key = (policy_id, matched_pair_id)
            if key in lookup:
                blockers.append(f"paired_runtime_result_duplicate:{policy_id}:{matched_pair_id}")
            lookup[key] = 1 if result else 0
            design = (condition_id, seed)
            existing_design = pair_designs.get(matched_pair_id)
            if existing_design is not None and existing_design != design:
                blockers.append(f"paired_design_identity_mismatch:{matched_pair_id}")
            else:
                pair_designs[matched_pair_id] = design
    alpha_per_pair = POLICY_RANKING_MULTIPLICITY_ALPHA / pair_count
    pairwise_intervals: list[dict[str, Any]] = []
    for policy_a, policy_b in pairs:
        differences_by_condition: dict[str, list[float]] = {}
        pair_ids_a = {pair_id for policy, pair_id in lookup if policy == policy_a}
        pair_ids_b = {pair_id for policy, pair_id in lookup if policy == policy_b}
        if pair_ids_a != pair_ids_b:
            blockers.append(f"paired_identity_set_incomplete:{policy_a}:{policy_b}")
        for pair_id in sorted(pair_ids_a & pair_ids_b):
            design = pair_designs.get(pair_id)
            if design is None:
                blockers.append(f"paired_design_missing:{pair_id}")
                continue
            condition, _seed = design
            differences_by_condition.setdefault(condition, []).append(
                float(lookup[(policy_a, pair_id)] - lookup[(policy_b, pair_id)])
            )
        if not differences_by_condition:
            blockers.append(f"paired_clusters_missing:{policy_a}:{policy_b}")
            continue
        condition_effects = [
            sum(values) / len(values) for values in differences_by_condition.values()
        ]
        observed_effect = sum(condition_effects) / len(condition_effects)
        seed_material = json.dumps(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "differences_by_condition": differences_by_condition,
                "iterations": POLICY_RANKING_BOOTSTRAP_ITERATIONS,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
        condition_names = sorted(differences_by_condition)
        bootstrap_effects: list[float] = []
        for _ in range(POLICY_RANKING_BOOTSTRAP_ITERATIONS):
            sampled_condition_effects: list[float] = []
            for _condition_index in condition_names:
                condition = condition_names[rng.randrange(len(condition_names))]
                values = differences_by_condition[condition]
                sampled = [values[rng.randrange(len(values))] for _ in values]
                sampled_condition_effects.append(sum(sampled) / len(sampled))
            bootstrap_effects.append(
                sum(sampled_condition_effects) / len(sampled_condition_effects)
            )
        lower = _empirical_quantile(bootstrap_effects, alpha_per_pair / 2.0)
        upper = _empirical_quantile(bootstrap_effects, 1.0 - alpha_per_pair / 2.0)
        pairwise_intervals.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "effect": "task_success_rate_difference_policy_a_minus_policy_b",
                "observed_effect": round(observed_effect, 6),
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "condition_count": len(differences_by_condition),
                "matched_seed_count": sum(
                    len(values) for values in differences_by_condition.values()
                ),
                "bootstrap_iterations": POLICY_RANKING_BOOTSTRAP_ITERATIONS,
                "familywise_alpha": POLICY_RANKING_MULTIPLICITY_ALPHA,
                "bonferroni_pair_alpha": alpha_per_pair,
            }
        )
    all_intervals_exclude_zero = bool(pairwise_intervals) and all(
        row["excludes_zero"] for row in pairwise_intervals
    )

    def candidate_beats(candidate: str, other: str) -> bool:
        for row in pairwise_intervals:
            if {row["policy_a"], row["policy_b"]} != {candidate, other}:
                continue
            if row["policy_a"] == candidate:
                return _number(row.get("lower")) > 0.0
            return _number(row.get("upper")) < 0.0
        return False

    winner_candidates = [
        candidate
        for candidate in sorted(set(policy_ids))
        if all(
            candidate_beats(candidate, other)
            for other in sorted(set(policy_ids))
            if other != candidate
        )
    ]
    winner_policy_id = (
        winner_candidates[0] if all_intervals_exclude_zero and len(winner_candidates) == 1 else None
    )
    if not all_intervals_exclude_zero:
        blockers.append("adjusted_paired_interval_includes_zero")
    if winner_policy_id is None:
        blockers.append("simultaneous_pairwise_winner_not_proven")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "policy_ranking_matched_cluster_bootstrap.v1",
        "status": "winner_proven" if not blockers else "inconclusive",
        "method": "deterministic_matched_condition_seed_hierarchical_cluster_bootstrap",
        "multiplicity_control": "bonferroni_all_policy_pairs",
        "familywise_alpha": POLICY_RANKING_MULTIPLICITY_ALPHA,
        "pair_count": pair_count,
        "all_adjusted_pairwise_intervals_exclude_zero": (all_intervals_exclude_zero),
        "winner_policy_id": winner_policy_id,
        "pairwise_intervals": pairwise_intervals,
        "blockers": blockers,
    }


def _blocked_matched_cluster_bootstrap_pairwise(
    *,
    policy_ids: Sequence[str],
    blockers: Sequence[str],
    submitted_row_count: int,
    verified_row_count: int,
) -> dict[str, Any]:
    """Return a non-decision surface when upstream evidence is not verified."""
    return {
        "schema_version": "policy_ranking_matched_cluster_bootstrap.v1",
        "status": "inconclusive",
        "method": "deterministic_matched_condition_seed_hierarchical_cluster_bootstrap",
        "multiplicity_control": "bonferroni_all_policy_pairs",
        "familywise_alpha": POLICY_RANKING_MULTIPLICITY_ALPHA,
        "pair_count": len(list(combinations(sorted(set(policy_ids)), 2))),
        "all_adjusted_pairwise_intervals_exclude_zero": False,
        "winner_policy_id": None,
        "pairwise_intervals": [],
        "submitted_row_count": submitted_row_count,
        "verified_row_count": verified_row_count,
        "blockers": sorted(
            set(
                [
                    "paired_inference_requires_decision_grade_verified_rows",
                    *blockers,
                ]
            )
        ),
    }


def _decision_grade_replicate_validation(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    policy_ids: Sequence[str],
    substrate: str,
    evidence_root: Path | None,
    authority_registry_reference: object,
) -> dict[str, Any]:
    blockers: list[str] = []
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    matched_designs: dict[tuple[str, int], dict[str, object]] = {}
    matched_policies: dict[tuple[str, int], set[str]] = {}
    pair_id_to_design: dict[str, tuple[str, int]] = {}
    for row in label_rows:
        policy_id = _string(row.get("policy_id")) or "policy"
        condition_id = _string(row.get("condition_id"))
        replicate_id = _string(row.get("replicate_id"))
        run_id = _string(row.get("scenario_eval_run_id"))
        seed = row.get("replicate_seed")
        matched_pair_id = _string(row.get("matched_pair_id"))
        task_id = _string(row.get("task_id"))
        scenario_id = _string(row.get("scenario_id"))
        criterion_id = _string(row.get("criterion_id"))
        initial_state_id = _string(row.get("initial_state_id"))
        condition_manifest_sha256 = _string(
            _mapping(row.get("matched_condition_manifest")).get("sha256")
        ).lower()
        if not isinstance(row.get("task_success"), bool):
            blockers.append("ranking_task_success_not_strict_boolean")
        if not condition_id:
            blockers.append("ranking_condition_id_missing")
        if not replicate_id:
            blockers.append("ranking_replicate_id_missing")
        if not run_id:
            blockers.append("ranking_scenario_eval_run_id_missing")
        if isinstance(seed, bool) or not isinstance(seed, int):
            blockers.append("ranking_replicate_seed_missing_or_invalid")
        for field, value in (
            ("matched_pair_id", matched_pair_id),
            ("task_id", task_id),
            ("scenario_id", scenario_id),
            ("criterion_id", criterion_id),
            ("initial_state_id", initial_state_id),
        ):
            if not value:
                blockers.append(f"ranking_{field}_missing")
        if SHA256_REF_PATTERN.fullmatch(condition_manifest_sha256) is None:
            blockers.append("ranking_condition_manifest_sha256_invalid")
        if condition_id and isinstance(seed, int) and not isinstance(seed, bool):
            design_key = (condition_id, seed)
            design = {
                "matched_pair_id": matched_pair_id,
                "replicate_id": replicate_id,
                "scenario_eval_run_id": run_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "criterion_id": criterion_id,
                "initial_state_id": initial_state_id,
                "condition_manifest_sha256": condition_manifest_sha256,
            }
            expected_design = matched_designs.get(design_key)
            if expected_design is None:
                matched_designs[design_key] = design
            else:
                for field, expected in expected_design.items():
                    if design.get(field) != expected:
                        blockers.append(
                            f"ranking_matched_design_mismatch:{condition_id}:{seed}:{field}"
                        )
            matched_policies.setdefault(design_key, set()).add(policy_id)
            if matched_pair_id:
                existing_key = pair_id_to_design.get(matched_pair_id)
                if existing_key is not None and existing_key != design_key:
                    blockers.append(f"ranking_matched_pair_id_reused:{matched_pair_id}")
                else:
                    pair_id_to_design[matched_pair_id] = design_key
        if condition_id:
            cells.setdefault((policy_id, condition_id), []).append(row)
    condition_ids = sorted({condition_id for _, condition_id in cells})
    cell_rows: list[dict[str, Any]] = []
    seed_sets: dict[tuple[str, str], set[int]] = {}
    for policy_id in policy_ids:
        for condition_id in condition_ids:
            rows = cells.get((policy_id, condition_id), [])
            replicate_ids = [_string(row.get("replicate_id")) for row in rows]
            seeds = {
                int(row["replicate_seed"])
                for row in rows
                if isinstance(row.get("replicate_seed"), int)
                and not isinstance(row.get("replicate_seed"), bool)
            }
            seed_sets[(policy_id, condition_id)] = seeds
            if len(rows) < POLICY_RANKING_MIN_REPLICATES_PER_CONDITION:
                blockers.append(
                    f"ranking_cell_replicates_lt_{POLICY_RANKING_MIN_REPLICATES_PER_CONDITION}:"
                    f"{policy_id}:{condition_id}"
                )
            if len(seeds) < POLICY_RANKING_MIN_REPLICATES_PER_CONDITION:
                blockers.append(
                    f"ranking_cell_unique_seeds_lt_{POLICY_RANKING_MIN_REPLICATES_PER_CONDITION}:"
                    f"{policy_id}:{condition_id}"
                )
            if len(set(replicate_ids)) != len(replicate_ids) or any(
                not value for value in replicate_ids
            ):
                blockers.append(f"ranking_replicate_ids_duplicate:{policy_id}:{condition_id}")
            if len(seeds) != len(rows):
                blockers.append(f"ranking_replicate_seeds_duplicate:{policy_id}:{condition_id}")
            cell_rows.append(
                {
                    "policy_id": policy_id,
                    "condition_id": condition_id,
                    "trial_count": len(rows),
                    "unique_seed_count": len(seeds),
                    "replicate_ids_unique": len(set(replicate_ids)) == len(replicate_ids),
                    "replicate_seeds_unique": len(seeds) == len(rows),
                }
            )
    evidence_validation = _decision_grade_evidence_validation(
        label_rows=label_rows,
        substrate=substrate,
        evidence_root=evidence_root,
        authority_registry_reference=authority_registry_reference,
    )
    blockers.extend(_string_list(evidence_validation.get("blockers")))
    expected_policies = set(policy_ids)
    for (condition_id, seed), observed_policies in matched_policies.items():
        if observed_policies != expected_policies:
            blockers.append(
                f"ranking_matched_pair_policy_coverage_incomplete:{condition_id}:{seed}"
            )
    for condition_id in condition_ids:
        expected: set[int] | None = None
        for policy_id in policy_ids:
            seeds = seed_sets.get((policy_id, condition_id), set())
            if expected is None:
                expected = seeds
            elif seeds != expected:
                blockers.append(f"ranking_matched_seed_set_mismatch:{condition_id}")
                break
    blockers = sorted(set(blockers))
    return {
        "schema_version": "policy_ranking_replicate_validation.v1",
        "status": "decision_grade" if cells and not blockers else "inconclusive",
        "minimum_replicates_per_policy_condition": (POLICY_RANKING_MIN_REPLICATES_PER_CONDITION),
        "condition_ids": condition_ids,
        "cells": cell_rows,
        "matched_seed_sets_required": True,
        "evidence_validation": evidence_validation,
        "blockers": blockers or ([] if cells else ["ranking_replicate_cells_missing"]),
    }


def _policy_scorecard(
    *,
    substrate: str,
    labels: Mapping[str, Any],
    generated_at: str,
    required_scenario_eval_run_ids: Sequence[str] = (),
    policy_ids: Sequence[str] = (),
    evidence_root: Path | None = None,
) -> Dict[str, Any]:
    label_rows = [
        dict(item) for item in labels.get("labels", []) or [] if isinstance(item, Mapping)
    ]
    visual_review_gate = _visual_review_gate_from_labels(labels)
    consistency_signal_summary = _consistency_support_signal_summary(
        labels=labels,
        label_rows=label_rows,
    )
    by_policy: Dict[str, list[Dict[str, Any]]] = {}
    for label in label_rows:
        by_policy.setdefault(_string(label.get("policy_id")) or "policy", []).append(label)
    required_run_ids = _ordered_unique_strings(required_scenario_eval_run_ids)
    if not required_run_ids:
        required_run_ids = _ordered_unique_strings(
            [
                label.get("scenario_eval_run_id")
                for label in label_rows
                if _string(label.get("scenario_eval_run_id"))
            ]
        )
    declared_policy_ids = _ordered_unique_strings(
        [
            *policy_ids,
            *[_string(label.get("policy_id")) or "policy" for label in label_rows],
        ]
    )
    rows: list[Dict[str, Any]] = []
    per_policy_coverage: list[Dict[str, Any]] = []
    missing_by_policy: Dict[str, list[str]] = {}
    extra_by_policy: Dict[str, list[str]] = {}
    attempt_count_by_policy: Dict[str, int] = {}
    duplicate_required_attempts_by_policy: Dict[str, list[str]] = {}
    required_run_set = set(required_run_ids)
    for policy_id in declared_policy_ids:
        policy_labels = by_policy.get(policy_id, [])
        observed_run_ids = _ordered_unique_strings(
            [label.get("scenario_eval_run_id") for label in policy_labels]
        )
        run_attempt_counts = {
            run_id: sum(
                1 for label in policy_labels if _string(label.get("scenario_eval_run_id")) == run_id
            )
            for run_id in observed_run_ids
        }
        covered_required_ids = [
            run_id for run_id in required_run_ids if run_id in set(observed_run_ids)
        ]
        missing_ids = [run_id for run_id in required_run_ids if run_id not in set(observed_run_ids)]
        extra_ids = sorted(set(observed_run_ids) - required_run_set) if required_run_ids else []
        duplicate_required_ids = [
            run_id
            for run_id, count in run_attempt_counts.items()
            if run_id in required_run_set and count > 1
        ]
        attempt_count = len(policy_labels)
        expected_attempt_count = len(required_run_ids)
        policy_coverage_complete = bool(
            required_run_ids
            and not missing_ids
            and not extra_ids
            and not duplicate_required_ids
            and attempt_count == expected_attempt_count
        )
        missing_by_policy[policy_id] = missing_ids
        extra_by_policy[policy_id] = extra_ids
        attempt_count_by_policy[policy_id] = attempt_count
        duplicate_required_attempts_by_policy[policy_id] = duplicate_required_ids
        per_policy_coverage.append(
            {
                "policy_id": policy_id,
                "required_scenario_eval_run_ids": list(required_run_ids),
                "covered_scenario_eval_run_ids": covered_required_ids,
                "missing_scenario_eval_run_ids": missing_ids,
                "extra_scenario_eval_run_ids": extra_ids,
                "attempt_count": attempt_count,
                "expected_attempt_count": expected_attempt_count,
                "duplicate_required_scenario_eval_run_ids": duplicate_required_ids,
                "coverage_complete": policy_coverage_complete,
            }
        )
        success_count = sum(1 for label in policy_labels if label.get("task_success") is True)
        uncertainties = [
            value
            for label in policy_labels
            for value in [_optional_number(label.get("uncertainty_score"))]
            if value is not None
        ]
        ood_flag_count = sum(1 for label in policy_labels if _string_list(label.get("ood_flags")))
        rows.append(
            {
                "policy_id": policy_id,
                "attempt_count": len(policy_labels),
                "predicted_success_count": success_count,
                "predicted_failure_count": len(policy_labels) - success_count,
                "predicted_success_rate": round(success_count / len(policy_labels), 6)
                if policy_labels
                else 0.0,
                "mean_uncertainty": round(sum(uncertainties) / len(uncertainties), 6)
                if uncertainties
                else None,
                "ood_flag_count": ood_flag_count,
                "ood_rate": round(ood_flag_count / len(policy_labels), 6) if policy_labels else 0.0,
                "failure_taxonomy": sorted(
                    {
                        mode
                        for label in policy_labels
                        for mode in _string_list(label.get("failure_mode_ids"))
                    }
                ),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -_number(row.get("predicted_success_rate")),
            _number(row.get("mean_uncertainty"), 1.0),
            _string(row.get("policy_id")),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    replicate_validation = _decision_grade_replicate_validation(
        label_rows=label_rows,
        policy_ids=declared_policy_ids,
        substrate=substrate,
        evidence_root=evidence_root,
        authority_registry_reference=labels.get("decision_grade_authority_registry"),
    )
    evidence_validation = _mapping(replicate_validation.get("evidence_validation"))
    evidence_row_results = evidence_validation.get("validated_rows")
    if not isinstance(evidence_row_results, list):
        evidence_row_results = []
    verified_row_indices = sorted(
        {
            int(row_result["row_index"])
            for row_result in evidence_row_results
            if isinstance(row_result, Mapping)
            and row_result.get("status") == "verified"
            and isinstance(row_result.get("row_index"), int)
            and not isinstance(row_result.get("row_index"), bool)
            and 0 <= int(row_result["row_index"]) < len(label_rows)
        }
    )
    all_rows_explicitly_verified = bool(label_rows) and (
        evidence_validation.get("status") == "verified"
        and verified_row_indices == list(range(len(label_rows)))
    )
    if replicate_validation.get("status") == "decision_grade" and all_rows_explicitly_verified:
        pairwise_inference = _matched_cluster_bootstrap_pairwise(
            label_rows=[label_rows[index] for index in verified_row_indices],
            policy_ids=declared_policy_ids,
        )
    else:
        inference_blockers = [
            *_string_list(replicate_validation.get("blockers")),
            *_string_list(evidence_validation.get("blockers")),
        ]
        if replicate_validation.get("status") != "decision_grade":
            inference_blockers.append("decision_grade_replicate_validation_not_passed")
        if not all_rows_explicitly_verified:
            inference_blockers.append("decision_grade_evidence_rows_not_all_verified")
        pairwise_inference = _blocked_matched_cluster_bootstrap_pairwise(
            policy_ids=declared_policy_ids,
            blockers=inference_blockers,
            submitted_row_count=len(label_rows),
            verified_row_count=len(verified_row_indices),
        )
    comparison_count = int(pairwise_inference.get("pair_count") or 0)
    score_range_blockers: list[str] = []
    ood_registration_contract_blockers: list[str] = []
    for label in label_rows:
        uncertainty = _optional_number(label.get("uncertainty_score"))
        if uncertainty is not None and not 0.0 <= uncertainty <= 1.0:
            score_range_blockers.append("uncertainty_score_out_of_range")
        confidence = _optional_number(label.get("confidence"))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            score_range_blockers.append("confidence_score_out_of_range")
        if label.get("registered_ood_axes_complete") is not True:
            ood_registration_contract_blockers.append("registered_ood_axes_incomplete")
        ood_registration_contract_blockers.extend(
            _string_list(label.get("ood_registration_blockers"))
        )
    score_ranges_valid = not score_range_blockers
    coverage_complete = bool(
        declared_policy_ids
        and required_run_ids
        and all(item["coverage_complete"] for item in per_policy_coverage)
    )
    top_policy_margin: float | None = None
    if len(ranked) >= 2:
        top_policy_margin = round(
            _number(ranked[0].get("predicted_success_rate"))
            - _number(ranked[1].get("predicted_success_rate")),
            6,
        )
    ranking_ambiguous = bool(
        len(ranked) >= 2
        and top_policy_margin is not None
        and top_policy_margin <= POLICY_RANKING_TIE_BAND
    )
    interval_winner_proven = bool(
        len(ranked) >= 2
        and replicate_validation.get("status") == "decision_grade"
        and pairwise_inference.get("status") == "winner_proven"
        and pairwise_inference.get("winner_policy_id") == ranked[0].get("policy_id")
    )
    if replicate_validation.get("status") == "decision_grade":
        ranking_ambiguous = not interval_winner_proven
    uncertainty_penalty_applied = any(
        _optional_number(row.get("mean_uncertainty")) is not None
        and _number(row.get("mean_uncertainty")) >= POLICY_RANKING_HIGH_UNCERTAINTY_THRESHOLD
        for row in ranked
    )
    ood_blockers = [
        f"policy:{row['policy_id']}:ood_rate_high"
        for row in ranked
        if _number(row.get("ood_rate")) >= POLICY_RANKING_HIGH_OOD_RATE_THRESHOLD
        and int(row.get("ood_flag_count") or 0) > 0
    ]
    comparison_blockers: list[str] = []
    if not label_rows:
        comparison_blockers.append("policy_labels_missing")
    if len(declared_policy_ids) < 2:
        comparison_blockers.append("policy_comparison_requires_at_least_two_candidates")
    if not required_run_ids:
        comparison_blockers.append("required_scenario_eval_run_ids_missing")
    if any(missing_by_policy.values()):
        comparison_blockers.append("policy_coverage_missing_required_scenario_eval_run_ids")
    if any(extra_by_policy.values()):
        comparison_blockers.append("policy_coverage_contains_unknown_scenario_eval_run_ids")
    if any(duplicate_required_attempts_by_policy.values()):
        comparison_blockers.append("policy_coverage_duplicate_required_scenario_attempts")
    if required_run_ids and any(
        count != len(required_run_ids) for count in attempt_count_by_policy.values()
    ):
        comparison_blockers.append("policy_attempt_count_not_equal_required_scenario_count")
    if not score_ranges_valid:
        comparison_blockers.extend(score_range_blockers)
    comparison_blockers = _dedupe(comparison_blockers)
    visual_review_blockers = visual_review_gate["blockers"]
    visual_review_required = bool(
        visual_review_blockers
        or not visual_review_gate["review_grade_success_labels"]
        or not visual_review_gate["visual_rollout_useful_for_task_success_review"]
        or not _mapping(visual_review_gate.get("short_visual_sanity_gate")).get("passed")
    )
    if comparison_blockers:
        status = "blocked_inconclusive_ranking"
    elif replicate_validation.get("status") != "decision_grade":
        status = "completed_inconclusive_insufficient_replicates"
    elif ood_registration_contract_blockers:
        status = "blocked_inconclusive_ood_registration"
    elif visual_review_required:
        status = "completed_visual_review_required"
    elif ranking_ambiguous:
        status = "completed_ambiguous_ranking"
    elif uncertainty_penalty_applied or ood_blockers:
        status = "completed_low_confidence_ranking"
    else:
        status = "completed"
    single_best_policy_claimed = bool(
        ranked
        and status == "completed"
        and not comparison_blockers
        and not ranking_ambiguous
        and not uncertainty_penalty_applied
        and not ood_blockers
        and interval_winner_proven
        and not ood_registration_contract_blockers
    )
    evaluator_top_policy_id = ranked[0]["policy_id"] if ranked else None
    confidence_level = "blocked"
    if not comparison_blockers:
        if replicate_validation.get("status") != "decision_grade":
            confidence_level = "inconclusive_insufficient_replicates"
        elif ranking_ambiguous:
            confidence_level = "ambiguous"
        elif uncertainty_penalty_applied or ood_blockers:
            confidence_level = "low"
        else:
            confidence_level = "decision_grade_evaluator_only"
    review_grade_policy_ranking = bool(
        status == "completed"
        and visual_review_gate["review_grade_success_labels"]
        and visual_review_gate["visual_rollout_useful_for_task_success_review"]
    )
    return {
        "schema_version": POLICY_RANKING_SCORECARD_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "evaluation_substrate": substrate,
        "ranking_basis": (
            "fixture_vision_success_labels_over_model_derived_wam_rollouts"
            if substrate == "fixture_wam"
            else "hash_verified_matched_nonfixture_execution_media_and_review_labels"
        ),
        "visual_smoke_status": visual_review_gate["visual_smoke_status"],
        "visual_smoke_statuses": visual_review_gate["visual_smoke_statuses"],
        "visual_rollout_useful_for_task_success_review": visual_review_gate[
            "visual_rollout_useful_for_task_success_review"
        ],
        "visual_review_blockers": visual_review_blockers,
        "fixture_evaluator_only": visual_review_gate["fixture_evaluator_only"],
        "review_grade_visual_evidence_available": visual_review_gate[
            "review_grade_visual_evidence_available"
        ],
        "review_grade_success_labels": visual_review_gate["review_grade_success_labels"],
        "review_grade_policy_ranking": review_grade_policy_ranking,
        "review_grade_policy_ranking_status": "completed"
        if review_grade_policy_ranking
        else "blocked_visual_review_required",
        "short_visual_sanity_gate": visual_review_gate["short_visual_sanity_gate"],
        "review_quality_manifest_required_for_policy_ranking": True,
        "comparison_contract": {
            "primary_eval_question": (
                "which policy_or_checkpoint performs better inside this configured evaluator"
            ),
            "comparison_scope": "configured_evaluator_only",
            "same_scenario_eval_matrix_required": True,
            "same_observation_protocol": True,
            "same_observation_protocol_id": "blueprint.robot_eval.observation.v1",
            "same_action_protocol": True,
            "same_action_protocol_id": "blueprint.robot_eval.action_trace.v1",
            "same_observation_and_label_protocol_required": True,
            "ranking_metrics": [
                "predicted_success_rate",
                "mean_uncertainty",
                "failure_taxonomy",
            ],
            "validation_metrics_when_real_anchors_exist": [
                "spearman_rank_correlation",
                "pearson_success_rate_correlation",
                "mean_maximum_rank_violation",
                "mean_absolute_success_rate_error",
            ],
            "traditional_sim_cross_check_optional": True,
            "evaluation_readiness_claimed": False,
            "external_deployment_grade_claimed": False,
            "forward_inverse_consistency_metrics_are_support_signals_only": True,
            "forward_inverse_consistency_does_not_upgrade_policy_ranking": True,
            "forward_inverse_consistency_does_not_prove_task_success": True,
            "forward_inverse_consistency_is_not_external_validation": True,
            "single_best_policy_claim_requires_margin_above_tie_band": True,
            "review_grade_policy_ranking_requires_passed_visual_smoke": True,
            "fixture_evaluator_only_ranking_is_not_review_grade": True,
            "review_grade_policy_ranking_requires_short_visual_sanity_manifest": True,
            "review_grade_policy_ranking_requires_review_label_refs": True,
            "decision_grade_requires_unique_hash_verified_replicate_evidence": True,
            "winner_requires_all_adjusted_paired_intervals_exclude_zero": True,
            "unpaired_pooled_wilson_intervals_used_for_winner_claim": False,
        },
        "policy_count": len(ranked),
        "scenario_attempt_count": len(label_rows),
        "required_scenario_eval_run_ids": list(required_run_ids),
        "per_policy_coverage": per_policy_coverage,
        "coverage_complete": coverage_complete,
        "missing_by_policy": missing_by_policy,
        "extra_by_policy": extra_by_policy,
        "attempt_count_by_policy": attempt_count_by_policy,
        "comparison_blockers": comparison_blockers,
        "decision_grade_replicate_validation": replicate_validation,
        "paired_cluster_bootstrap_inference": pairwise_inference,
        "interval_winner_proven": interval_winner_proven,
        "multiplicity_control": {
            "method": "bonferroni_all_policy_pairs_cluster_bootstrap",
            "familywise_alpha": POLICY_RANKING_MULTIPLICITY_ALPHA,
            "comparison_count": comparison_count,
        },
        "score_ranges_valid": score_ranges_valid,
        "score_range_blockers": _dedupe(score_range_blockers),
        "ood_registration_complete": not ood_registration_contract_blockers,
        "ood_registration_blockers": _dedupe(ood_registration_contract_blockers),
        "forward_inverse_consistency_signal_summary": consistency_signal_summary,
        "policy_rankings": ranked,
        "evaluator_top_policy_id": evaluator_top_policy_id,
        "top_policy_id": evaluator_top_policy_id if single_best_policy_claimed else None,
        "single_best_policy_claimed": single_best_policy_claimed,
        "ranking_confidence": {
            "top_policy_margin": top_policy_margin,
            "tie_band": POLICY_RANKING_TIE_BAND,
            "ranking_ambiguous": ranking_ambiguous,
            "uncertainty_penalty_applied": uncertainty_penalty_applied,
            "ood_blockers": ood_blockers,
            "confidence_level": confidence_level,
            "real_world_calibration_metrics": {
                "spearman_rank_correlation": "not_measured",
                "pearson_success_rate_correlation": "not_measured",
                "mean_maximum_rank_violation": "not_measured",
            },
        },
        "failure_taxonomy": sorted(
            {mode for label in label_rows for mode in _string_list(label.get("failure_mode_ids"))}
        ),
        "uncertainty_ood_summary": {
            "ood_label_count": sum(
                1 for label in label_rows if _string_list(label.get("ood_flags"))
            ),
            "mean_uncertainty": round(
                sum(_number(label.get("uncertainty_score")) for label in label_rows)
                / len(label_rows),
                6,
            )
            if label_rows
            else None,
        },
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "visual_smoke_required_for_review_grade_policy_ranking": True,
            "short_visual_sanity_required_for_review_grade_policy_ranking": True,
            "review_label_refs_required_for_review_grade_policy_ranking": True,
            "visual_rollout_useful_for_task_success_review": visual_review_gate[
                "visual_rollout_useful_for_task_success_review"
            ],
            "fixture_evaluator_only": visual_review_gate["fixture_evaluator_only"],
            "review_grade_policy_ranking": review_grade_policy_ranking,
            "fixture_evidence_cannot_be_decision_grade": True,
            "winner_claim_uses_paired_clustered_inference": True,
        },
    }


def _claim_boundary(*, substrate: str, generated_at: str) -> Dict[str, Any]:
    return build_wam_eval_claim_boundary(substrate=substrate, generated_at=generated_at)


def _real_world_validation_followup(
    *,
    job_id: str,
    substrate: str,
    scorecard: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": REAL_WORLD_VALIDATION_FOLLOWUP_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "requested_real_world_validation_anchors",
        "evaluation_substrate": substrate,
        "top_policy_id": scorecard.get("top_policy_id"),
        "requested_anchor_rollouts": [
            "real_world_rollouts_for_top_ranked_policy",
            "real_world_rollouts_for_low_ranked_policy",
            "real_world_rollouts_for_high_uncertainty_or_ood_scenarios",
        ],
        "minimum_validation_requirements": {
            "paired_real_outcome_records_required": True,
            "exact_scenario_eval_run_id_join_required": True,
            "policy_or_checkpoint_ids_required": True,
            "owner_evidence_or_operator_attestation_required": True,
        },
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _srcc_validation_plan(*, job_id: str, substrate: str, generated_at: str) -> Dict[str, Any]:
    return {
        "schema_version": SRCC_VALIDATION_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "requires_real_world_rollout_anchors",
        "evaluation_substrate": substrate,
        "metrics_to_compute_when_anchors_exist": [
            "spearman_rank_correlation",
            "pearson_success_rate_correlation",
            "mean_absolute_success_rate_error",
            "mean_maximum_rank_violation",
            "failure_mode_agreement",
        ],
        "accepted_anchor_join_keys": list(ACCEPTED_REAL_WORLD_ANCHOR_JOIN_KEYS),
        "customer_specific_srcc_claimed": False,
        "blocked_report_reason": "missing_paired_real_world_rollout_outcomes",
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }


def _policy_metadata(policies: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for policy in policies:
        policy_id = _string(policy.get("policy_id") or policy.get("policyId"))
        if not policy_id:
            continue
        metadata[policy_id] = {
            "policy_id": policy_id,
            "display_name": _string(policy.get("display_name") or policy.get("name")) or policy_id,
            "adapter_id": _string(
                policy.get("adapter_id")
                or policy.get("adapterId")
                or policy.get("policy_adapter_id")
                or policy.get("policyAdapterId")
            )
            or None,
            "checkpoint_id": _string(
                policy.get("checkpoint_id")
                or policy.get("checkpointId")
                or policy.get("checkpoint")
            )
            or None,
        }
    return metadata


def _policy_ranking_rows(scorecard: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = [
        dict(item)
        for item in scorecard.get("policy_rankings", []) or []
        if isinstance(item, Mapping)
    ]
    return sorted(
        rows,
        key=lambda row: (
            _number(row.get("rank"), 999999),
            -_number(row.get("predicted_success_rate")),
            _string(row.get("policy_id")),
        ),
    )


def _candidate_selection_summary(scorecard: Mapping[str, Any]) -> Dict[str, Any]:
    ranked = _policy_ranking_rows(scorecard)
    if not ranked:
        return {
            "status": "blocked_missing_ranking_evidence",
            "top_policy_id": None,
            "evaluator_top_policy_id": None,
            "runner_up_policy_id": None,
            "margin": None,
            "ranking_ambiguous": True,
            "tie_or_ambiguity_status": "no_candidate_ranking_available",
            "candidate_shortlist": [],
            "ambiguity_reasons": ["policy_ranking_scorecard_missing_or_empty"],
            "policy_rankings": [],
        }
    if len(ranked) == 1:
        only = ranked[0]
        return {
            "status": "single_candidate_no_comparative_ranking",
            "top_policy_id": None,
            "evaluator_top_policy_id": only.get("policy_id"),
            "runner_up_policy_id": None,
            "margin": None,
            "ranking_ambiguous": True,
            "tie_or_ambiguity_status": "single_candidate_no_comparison",
            "candidate_shortlist": [only],
            "ambiguity_reasons": ["only_one_policy_candidate_was_evaluated"],
            "policy_rankings": ranked,
        }

    top = ranked[0]
    runner_up = ranked[1]
    success_margin = round(
        _number(top.get("predicted_success_rate"))
        - _number(runner_up.get("predicted_success_rate")),
        6,
    )
    uncertainty_delta = None
    if top.get("mean_uncertainty") is not None and runner_up.get("mean_uncertainty") is not None:
        uncertainty_delta = round(
            _number(runner_up.get("mean_uncertainty")) - _number(top.get("mean_uncertainty")),
            6,
        )
    shortlist = [
        row
        for row in ranked
        if round(
            _number(top.get("predicted_success_rate")) - _number(row.get("predicted_success_rate")),
            6,
        )
        < CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN
    ]
    fallback_shortlist = shortlist
    if len(fallback_shortlist) < 2:
        fallback_shortlist = ranked[:2]
    ambiguous = success_margin < CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN
    scorecard_status = _string(scorecard.get("status"))
    comparison_blockers = _string_list(scorecard.get("comparison_blockers"))
    confidence = _mapping(scorecard.get("ranking_confidence"))
    visual_blockers = _string_list(scorecard.get("visual_review_blockers"))
    short_visual_sanity_gate = _mapping(scorecard.get("short_visual_sanity_gate"))
    visual_review_required = bool(
        scorecard_status == "completed_visual_review_required"
        or visual_blockers
        or short_visual_sanity_gate.get("passed") is False
        or not bool(scorecard.get("visual_rollout_useful_for_task_success_review"))
        or bool(scorecard.get("fixture_evaluator_only"))
        or not bool(scorecard.get("review_grade_success_labels"))
    )
    low_confidence_reasons = [
        *_string_list(confidence.get("ood_blockers")),
        *(
            ["uncertainty_penalty_applied"]
            if bool(confidence.get("uncertainty_penalty_applied"))
            else []
        ),
    ]
    low_confidence = bool(
        scorecard_status == "completed_low_confidence_ranking" or low_confidence_reasons
    )
    replicate_validation = _mapping(scorecard.get("decision_grade_replicate_validation"))
    replicate_inconclusive = bool(
        replicate_validation.get("status") != "decision_grade"
        or not scorecard.get("interval_winner_proven")
    )
    if comparison_blockers or scorecard_status == "blocked_inconclusive_ranking":
        return {
            "status": "blocked_inconclusive_candidate_selection",
            "top_policy_id": None,
            "evaluator_top_policy_id": top.get("policy_id"),
            "runner_up_policy_id": runner_up.get("policy_id"),
            "margin": {
                "predicted_success_rate": success_margin,
                "mean_uncertainty_advantage": uncertainty_delta,
                "ambiguity_threshold": CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN,
            },
            "ranking_ambiguous": True,
            "tie_or_ambiguity_status": "blocked_inconclusive",
            "candidate_shortlist": ranked,
            "ambiguity_reasons": [
                "policy_ranking_scorecard_blocked_or_inconclusive",
                *comparison_blockers,
            ],
            "policy_rankings": ranked,
        }
    return {
        "status": (
            "inconclusive_insufficient_replicates_candidate_shortlist"
            if replicate_inconclusive
            else "visual_review_required_candidate_shortlist"
            if visual_review_required
            else "ambiguous_candidate_shortlist"
            if ambiguous
            else "low_confidence_candidate_shortlist"
            if low_confidence
            else "clear_winner"
        ),
        "top_policy_id": None
        if replicate_inconclusive or ambiguous or visual_review_required or low_confidence
        else top.get("policy_id"),
        "evaluator_top_policy_id": top.get("policy_id"),
        "runner_up_policy_id": runner_up.get("policy_id"),
        "margin": {
            "predicted_success_rate": success_margin,
            "mean_uncertainty_advantage": uncertainty_delta,
            "ambiguity_threshold": CANDIDATE_SELECTION_AMBIGUITY_SUCCESS_RATE_MARGIN,
        },
        "ranking_ambiguous": bool(
            replicate_inconclusive or ambiguous or visual_review_required or low_confidence
        ),
        "tie_or_ambiguity_status": (
            "insufficient_replicates_or_interval_overlap"
            if replicate_inconclusive
            else "visual_review_required"
            if visual_review_required
            else "ambiguous"
            if ambiguous
            else "low_confidence"
            if low_confidence
            else "clear"
        ),
        "candidate_shortlist": fallback_shortlist
        if replicate_inconclusive or ambiguous or visual_review_required or low_confidence
        else [],
        "ambiguity_reasons": (
            [
                "decision_grade_replicate_or_interval_evidence_missing",
                *_string_list(replicate_validation.get("blockers")),
            ]
            if replicate_inconclusive
            else [
                "visual_review_blockers_or_fixture_only_labels_prevent_winner_claim",
                *visual_blockers,
                *_string_list(short_visual_sanity_gate.get("blockers")),
            ]
            if visual_review_required
            else ["top_two_success_rates_within_threshold"]
            if ambiguous
            else ["ranking_low_confidence", *low_confidence_reasons]
            if low_confidence
            else []
        ),
        "policy_rankings": ranked,
    }


def _scenario_metadata(matrix: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for run in _matrix_runs(matrix):
        run_id = _string(run.get("scenario_eval_run_id"))
        if not run_id:
            continue
        metadata[run_id] = {
            "scenario_eval_run_id": run_id,
            "scenario_variation_instance_id": run.get("scenario_variation_instance_id")
            or run.get("scenarioVariationInstanceId"),
            "task_id": _string(run.get("task_id") or run.get("taskId")) or None,
            "scenario_id": _string(run.get("scenario_id") or run.get("scenarioId")) or None,
            "variation_name": _string(run.get("variation_name") or run.get("variationName"))
            or None,
            "split": _string(run.get("split")) or None,
        }
    return metadata


def _label_rows(labels: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [dict(item) for item in labels.get("labels", []) or [] if isinstance(item, Mapping)]


def _label_evidence_ref(label: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "artifact_path": "vision_success_labels.json",
        "label_id": label.get("label_id"),
        "attempt_id": label.get("attempt_id"),
        "rollout_id": label.get("rollout_id"),
        "scenario_eval_run_id": label.get("scenario_eval_run_id"),
        "policy_id": label.get("policy_id"),
    }


def _scenario_matrix_coverage(
    *,
    matrix: Mapping[str, Any],
    labels: Mapping[str, Any],
    scorecard: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = _scenario_metadata(matrix)
    rows = _label_rows(labels)
    matrix_run_ids = sorted(metadata)
    covered_run_ids = sorted(
        {
            _string(label.get("scenario_eval_run_id"))
            for label in rows
            if label.get("scenario_eval_run_id")
        }
    )
    required_run_ids = matrix_run_ids or covered_run_ids
    missing_run_ids = sorted(set(required_run_ids) - set(covered_run_ids))
    policy_count = int(_number(scorecard.get("policy_count"), 0) or 0)
    expected_attempt_count = (
        len(required_run_ids) * policy_count if required_run_ids and policy_count else None
    )
    observed_attempt_count = len(rows)
    return {
        "scenario_eval_run_count": len(required_run_ids),
        "policy_count": policy_count,
        "expected_candidate_attempt_count": expected_attempt_count,
        "observed_candidate_attempt_count": observed_attempt_count,
        "coverage_complete": bool(
            required_run_ids
            and not missing_run_ids
            and (expected_attempt_count is None or observed_attempt_count >= expected_attempt_count)
        ),
        "required_scenario_eval_run_ids": required_run_ids,
        "covered_scenario_eval_run_ids": covered_run_ids,
        "missing_scenario_eval_run_ids": missing_run_ids,
        "coverage_source": "scenario_eval_matrix" if matrix_run_ids else "vision_success_labels",
    }


def _decisive_scenarios(
    *,
    matrix: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    metadata = _scenario_metadata(matrix)
    by_run: Dict[str, list[Dict[str, Any]]] = {}
    for label in _label_rows(labels):
        run_id = _string(label.get("scenario_eval_run_id"))
        if run_id:
            by_run.setdefault(run_id, []).append(label)
    decisive: list[Dict[str, Any]] = []
    for run_id, rows in sorted(by_run.items()):
        outcomes = []
        successes = []
        failures = []
        for label in sorted(rows, key=lambda item: _string(item.get("policy_id"))):
            policy_id = _string(label.get("policy_id")) or "policy"
            success = bool(label.get("task_success"))
            if success:
                successes.append(policy_id)
            else:
                failures.append(policy_id)
            outcomes.append(
                {
                    "policy_id": policy_id,
                    "task_success": success,
                    "uncertainty_score": _number(label.get("uncertainty_score")),
                    "failure_mode_ids": _string_list(label.get("failure_mode_ids")),
                    "ood_flags": _string_list(label.get("ood_flags")),
                    "evidence_ref": _label_evidence_ref(label),
                }
            )
        if successes and failures:
            decisive.append(
                {
                    **metadata.get(run_id, {"scenario_eval_run_id": run_id}),
                    "successful_policy_ids": successes,
                    "failed_policy_ids": failures,
                    "policy_outcomes": outcomes,
                    "failure_mode_ids": sorted(
                        {mode for outcome in outcomes for mode in outcome["failure_mode_ids"]}
                    ),
                    "exemplar_evidence_refs": [outcome["evidence_ref"] for outcome in outcomes[:4]],
                }
            )
    return decisive


def _high_uncertainty_scenarios(labels: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = [
        label
        for label in _label_rows(labels)
        if _number(label.get("uncertainty_score")) >= CANDIDATE_SELECTION_HIGH_UNCERTAINTY_THRESHOLD
    ]
    rows = sorted(
        rows,
        key=lambda label: (
            -_number(label.get("uncertainty_score")),
            _string(label.get("scenario_eval_run_id")),
            _string(label.get("policy_id")),
        ),
    )
    return [
        {
            "scenario_eval_run_id": label.get("scenario_eval_run_id"),
            "scenario_variation_instance_id": label.get("scenario_variation_instance_id"),
            "policy_id": label.get("policy_id"),
            "uncertainty_score": _number(label.get("uncertainty_score")),
            "ood_flags": _string_list(label.get("ood_flags")),
            "review_status": "needs_review",
            "evidence_ref": _label_evidence_ref(label),
        }
        for label in rows
    ]


def _ood_blockers(labels: Mapping[str, Any]) -> list[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for label in _label_rows(labels):
        for flag in _string_list(label.get("ood_flags")):
            row = grouped.setdefault(
                flag,
                {
                    "ood_flag": flag,
                    "count": 0,
                    "scenario_eval_run_ids": set(),
                    "affected_policy_ids": set(),
                    "exemplar_evidence_refs": [],
                },
            )
            row["count"] += 1
            if label.get("scenario_eval_run_id"):
                row["scenario_eval_run_ids"].add(_string(label.get("scenario_eval_run_id")))
            if label.get("policy_id"):
                row["affected_policy_ids"].add(_string(label.get("policy_id")))
            if len(row["exemplar_evidence_refs"]) < 3:
                row["exemplar_evidence_refs"].append(_label_evidence_ref(label))
    return [
        {
            **{
                key: value
                for key, value in row.items()
                if key not in {"scenario_eval_run_ids", "affected_policy_ids"}
            },
            "scenario_eval_run_ids": sorted(row["scenario_eval_run_ids"]),
            "affected_policy_ids": sorted(row["affected_policy_ids"]),
        }
        for row in sorted(grouped.values(), key=lambda item: (-item["count"], item["ood_flag"]))
    ]


def _failure_hook_template(failure_mode_id: str) -> Dict[str, list[str]]:
    templates = {
        "blocked_path_or_clearance_failure": {
            "data_to_collect": [
                "robot POV clips through blocked and narrow-clearance approaches",
                "depth, pose, near-miss, and contact annotations at obstacle boundaries",
            ],
            "scenario_variants_to_add": [
                "narrow aisle clearance sweeps",
                "partially blocked path variants",
                "movable obstacle offsets near the target approach",
            ],
        },
        "dynamic_agent_safety_failure": {
            "data_to_collect": [
                "robot POV and third-person clips with humans or carts crossing the route",
                "time-aligned agent trajectories and yield-distance labels",
            ],
            "scenario_variants_to_add": [
                "human crossing timing offsets",
                "forklift or cart crossing speed variants",
                "late-yield and stop-go interaction cases",
            ],
        },
        "perception_ambiguity_failure": {
            "data_to_collect": [
                "multi-angle robot POV clips for visually similar targets",
                "object identity, occlusion, glare, and missing-label annotations",
            ],
            "scenario_variants_to_add": [
                "glare and low-light target views",
                "partial occlusion variants",
                "wrong-object distractor placements",
            ],
        },
        "manipulation_alignment_failure": {
            "data_to_collect": [
                "hand-camera clips of grasp, place, and object-rotation attempts",
                "object pose, gripper pose, slip, and final-placement labels",
            ],
            "scenario_variants_to_add": [
                "object rotation variants",
                "shifted cart or bin target poses",
                "grasp approach angle sweeps",
            ],
        },
        "wam_ood_uncertain": {
            "data_to_collect": [
                "paired real rollout anchors for high-uncertainty generated scenarios",
                "operator review labels explaining whether the generated observation is usable",
            ],
            "scenario_variants_to_add": [
                "near-distribution versions of the OOD scenario",
                "single-factor OOD ablations for glare, occlusion, or target ambiguity",
            ],
        },
        "fixture_policy_failure": {
            "data_to_collect": [
                "policy command traces and robot POV clips around the forced failure case",
                "review notes confirming whether the fixture failure matches a real failure",
            ],
            "scenario_variants_to_add": [
                "direct regression case for the failing scenario_eval_run_id",
                "one-factor neighboring variants around the failing setup",
            ],
        },
        "unknown_needs_review": {
            "data_to_collect": [
                "human-reviewed rollout clips with failure reason annotations",
                "policy action traces, observations, and task-state snapshots near failure",
            ],
            "scenario_variants_to_add": [
                "minimal reproduction variants once the reviewer labels the failure",
                "neighboring scenario variants that isolate the suspected cause",
            ],
        },
    }
    return templates.get(failure_mode_id, templates["unknown_needs_review"])


def _retry_policy_refs(
    policy_ids: Sequence[str],
    policy_metadata: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    refs: list[Dict[str, Any]] = []
    for policy_id in _string_list(policy_ids):
        metadata = _mapping(policy_metadata.get(policy_id))
        refs.append(
            {
                "policy_id": policy_id,
                "display_name": metadata.get("display_name") or policy_id,
                "adapter_id": metadata.get("adapter_id"),
                "checkpoint_id": metadata.get("checkpoint_id"),
                "retry_reason": "rerun_after_failure_cluster_data_package_update",
            }
        )
    return refs


def _failure_clusters(
    *,
    failure_labels: Mapping[str, Any],
    selection: Mapping[str, Any],
    policy_metadata: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for label in [
        dict(item) for item in failure_labels.get("labels", []) or [] if isinstance(item, Mapping)
    ]:
        modes = _string_list(label.get("failure_mode_ids")) or ["unknown_needs_review"]
        for mode in modes:
            row = grouped.setdefault(
                mode,
                {
                    "failure_mode_id": mode,
                    "count": 0,
                    "affected_policy_ids": set(),
                    "scenario_eval_run_ids": set(),
                    "exemplar_evidence_refs": [],
                },
            )
            row["count"] += 1
            if label.get("policy_id"):
                row["affected_policy_ids"].add(_string(label.get("policy_id")))
            if label.get("scenario_eval_run_id"):
                row["scenario_eval_run_ids"].add(_string(label.get("scenario_eval_run_id")))
            if len(row["exemplar_evidence_refs"]) < 3:
                row["exemplar_evidence_refs"].append(
                    {
                        "artifact_path": "failure_labels.json",
                        "label_id": label.get("label_id"),
                        "attempt_id": label.get("attempt_id"),
                        "rollout_id": label.get("rollout_id"),
                        "scenario_eval_run_id": label.get("scenario_eval_run_id"),
                        "policy_id": label.get("policy_id"),
                    }
                )
    fallback_policy_ids = [
        _string(row.get("policy_id"))
        for row in selection.get("candidate_shortlist", []) or []
        if isinstance(row, Mapping) and _string(row.get("policy_id"))
    ]
    if not fallback_policy_ids and selection.get("top_policy_id"):
        fallback_policy_ids = [_string(selection.get("top_policy_id"))]
    clusters: list[Dict[str, Any]] = []
    for mode, row in sorted(grouped.items(), key=lambda item: (-item[1]["count"], item[0])):
        affected_policy_ids = sorted(row["affected_policy_ids"]) or fallback_policy_ids
        template = _failure_hook_template(mode)
        weak = mode == "unknown_needs_review"
        clusters.append(
            {
                "cluster_id": f"failure_cluster_{_safe_id(mode)}",
                "failure_mode_id": mode if not weak else None,
                "diagnosis": "unknown_needs_review"
                if weak
                else "failure_mode_observed_root_cause_needs_review",
                "evidence_strength": "weak" if weak else "label_only_needs_review",
                "count": row["count"],
                "affected_policy_ids": affected_policy_ids,
                "scenario_eval_run_ids": sorted(row["scenario_eval_run_ids"]),
                "exemplar_evidence_refs": row["exemplar_evidence_refs"],
                "post_training_data_package_hooks": {
                    **template,
                    "policy_adapter_or_checkpoint_to_retry": _retry_policy_refs(
                        affected_policy_ids,
                        policy_metadata,
                    ),
                },
            }
        )
    return clusters


def _dominant_failure_modes_from_clusters(
    clusters: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    if not clusters:
        return []
    return [
        {
            "failure_mode_id": cluster.get("failure_mode_id") or "unknown_needs_review",
            "count": int(_number(cluster.get("count"), 0) or 0),
            "diagnosis": cluster.get("diagnosis") or "unknown_needs_review",
            "evidence_strength": cluster.get("evidence_strength") or "weak",
        }
        for cluster in clusters
    ]


def _recommended_reruns(
    *,
    selection: Mapping[str, Any],
    high_uncertainty: Sequence[Mapping[str, Any]],
    ood_blockers: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    visual_gate: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    reruns: list[Dict[str, Any]] = []
    if selection.get("top_policy_id") is None:
        reruns.append(
            {
                "reason": "candidate_selection_not_decisive_enough_for_winner_claim",
                "status": "recommended",
                "policy_ids": [
                    _string(row.get("policy_id"))
                    for row in selection.get("candidate_shortlist", []) or []
                    if isinstance(row, Mapping) and _string(row.get("policy_id"))
                ],
                "scenario_eval_run_ids": [],
            }
        )
    blockers = _string_list(visual_gate.get("blockers"))
    if blockers:
        reruns.append(
            {
                "reason": "visual_review_blockers_present",
                "status": "required_before_review_grade_policy_ranking",
                "blockers": blockers,
                "policy_ids": [
                    _string(row.get("policy_id"))
                    for row in selection.get("candidate_shortlist", []) or []
                    if isinstance(row, Mapping) and _string(row.get("policy_id"))
                ],
                "scenario_eval_run_ids": [],
            }
        )
    if high_uncertainty:
        reruns.append(
            {
                "reason": "high_uncertainty_scenarios",
                "status": "recommended",
                "policy_ids": sorted(
                    {
                        _string(row.get("policy_id"))
                        for row in high_uncertainty
                        if _string(row.get("policy_id"))
                    }
                ),
                "scenario_eval_run_ids": sorted(
                    {
                        _string(row.get("scenario_eval_run_id"))
                        for row in high_uncertainty
                        if _string(row.get("scenario_eval_run_id"))
                    }
                ),
            }
        )
    if ood_blockers:
        reruns.append(
            {
                "reason": "ood_blockers_present",
                "status": "recommended",
                "policy_ids": sorted(
                    {
                        policy_id
                        for row in ood_blockers
                        for policy_id in _string_list(row.get("affected_policy_ids"))
                    }
                ),
                "scenario_eval_run_ids": sorted(
                    {
                        run_id
                        for row in ood_blockers
                        for run_id in _string_list(row.get("scenario_eval_run_ids"))
                    }
                ),
            }
        )
    for cluster in clusters[:5]:
        if not isinstance(cluster, Mapping):
            continue
        reruns.append(
            {
                "reason": "failure_cluster_regression",
                "status": "recommended",
                "failure_mode_id": cluster.get("failure_mode_id") or "unknown_needs_review",
                "policy_ids": _string_list(cluster.get("affected_policy_ids")),
                "scenario_eval_run_ids": _string_list(cluster.get("scenario_eval_run_ids")),
            }
        )
    return reruns


def _candidate_selection_report(
    *,
    job_id: str,
    substrate: str,
    matrix: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any],
    failure_labels: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    followup: Mapping[str, Any],
    anchor_manifest: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    selection = _candidate_selection_summary(scorecard)
    policy_meta = _policy_metadata(policies)
    decisive = _decisive_scenarios(matrix=matrix, labels=labels)
    clusters = _failure_clusters(
        failure_labels=failure_labels,
        selection=selection,
        policy_metadata=policy_meta,
    )
    coverage = _scenario_matrix_coverage(matrix=matrix, labels=labels, scorecard=scorecard)
    high_uncertainty = _high_uncertainty_scenarios(labels)
    ood_blockers = _ood_blockers(labels)
    visual_gate = {
        "status": scorecard.get("review_grade_policy_ranking_status")
        or "blocked_visual_review_required",
        "visual_smoke_status": scorecard.get("visual_smoke_status"),
        "visual_rollout_useful_for_task_success_review": bool(
            scorecard.get("visual_rollout_useful_for_task_success_review")
        ),
        "review_grade_policy_ranking": bool(scorecard.get("review_grade_policy_ranking")),
        "fixture_evaluator_only": bool(scorecard.get("fixture_evaluator_only")),
        "short_visual_sanity_gate": _mapping(scorecard.get("short_visual_sanity_gate")),
        "blockers": _string_list(scorecard.get("visual_review_blockers")),
    }
    consistency_signal_summary = _mapping(
        scorecard.get("forward_inverse_consistency_signal_summary")
    )
    exemplar_refs: list[Dict[str, Any]] = []
    for scenario in decisive[:4]:
        exemplar_refs.extend(
            ref
            for ref in scenario.get("exemplar_evidence_refs", []) or []
            if isinstance(ref, Mapping)
        )
    for cluster in clusters[:4]:
        exemplar_refs.extend(
            ref
            for ref in cluster.get("exemplar_evidence_refs", []) or []
            if isinstance(ref, Mapping)
        )
    claim_boundary = {
        **_claim_boundary(substrate=substrate, generated_at=generated_at),
        "boundary_statement": "sim-ranking handoff only; IRL validation is out of scope",
        "do_not_use_as_rank_fidelity_result": True,
        "rank_fidelity_result_claimed": False,
        "accepted_anchor_success_claimed": False,
        "best_policy_statement_scope": "configured_evaluator_only",
    }
    return {
        "schema_version": CANDIDATE_SELECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": selection["status"],
        "evaluation_substrate": substrate,
        "primary_eval_question": "which policy performed best in this evaluator, and what broke",
        "selection": selection,
        "top_policy_id": selection.get("top_policy_id"),
        "evaluator_top_policy_id": selection.get("evaluator_top_policy_id"),
        "runner_up_policy_id": selection.get("runner_up_policy_id"),
        "margin": selection.get("margin"),
        "tie_or_ambiguity_status": selection.get("tie_or_ambiguity_status"),
        "candidate_shortlist": selection.get("candidate_shortlist"),
        "recommendation": {
            "recommended_policy_id": selection.get("top_policy_id"),
            "evaluator_top_policy_id": selection.get("evaluator_top_policy_id"),
            "status": "recommended_in_configured_evaluator"
            if selection.get("top_policy_id")
            else "no_winner_claim_use_shortlist",
            "basis": "policy_ranking_scorecard.json",
            "why_not_recommended": _string_list(selection.get("ambiguity_reasons")),
        },
        "scenario_matrix_coverage": coverage,
        "decisive_scenarios": decisive,
        "high_uncertainty_scenarios": high_uncertainty,
        "ood_blockers": ood_blockers,
        "visual_reviewability_gate": visual_gate,
        "forward_inverse_consistency_signal_summary": consistency_signal_summary,
        "dominant_failure_modes": _dominant_failure_modes_from_clusters(clusters),
        "failure_clusters": clusters,
        "failure_evidence_status": "unknown_needs_review"
        if clusters and all(cluster.get("evidence_strength") == "weak" for cluster in clusters)
        else ("no_failures_observed_in_evaluator" if not clusters else "label_only_needs_review"),
        "exemplar_evidence_refs": exemplar_refs[:10],
        "recommended_reruns": _recommended_reruns(
            selection=selection,
            high_uncertainty=high_uncertainty,
            ood_blockers=ood_blockers,
            clusters=clusters,
            visual_gate=visual_gate,
        ),
        "artifact_paths": {
            "policy_ranking_scorecard": "policy_ranking_scorecard.json",
            "vision_success_labels": "vision_success_labels.json",
            "failure_labels": "failure_labels.json",
            "wam_rollout_results": "wam_rollout_results.json",
            "visual_review_blocker_summary": "visual_review_blocker_summary.json",
        },
        "claim_boundary": claim_boundary,
    }


def _candidate_selection_markdown(report: Mapping[str, Any]) -> str:
    selection = _mapping(report.get("selection"))
    margin = _mapping(report.get("margin"))
    recommended_policy = report.get("top_policy_id") or "none; use shortlist"
    evaluator_top_policy = report.get("evaluator_top_policy_id") or "none"
    shortlist = [
        _string(row.get("policy_id"))
        for row in report.get("candidate_shortlist", []) or []
        if isinstance(row, Mapping) and _string(row.get("policy_id"))
    ]
    lines = [
        "# WAM Candidate Selection Report",
        "",
        f"Status: `{report.get('status')}`",
        f"Evaluation substrate: `{report.get('evaluation_substrate')}`",
        f"Recommended policy: `{recommended_policy}`",
        f"Evaluator top policy: `{evaluator_top_policy}`",
        f"Runner-up: `{report.get('runner_up_policy_id')}`",
        f"Predicted success-rate margin: `{margin.get('predicted_success_rate')}`",
        f"Tie or ambiguity status: `{report.get('tie_or_ambiguity_status')}`",
        "",
        "Boundary: sim-ranking handoff only; IRL validation is out of scope.",
        "",
    ]
    if shortlist:
        lines.extend(
            [
                "## Candidate Shortlist",
                "",
                *[f"- `{policy_id}`" for policy_id in shortlist],
                "",
            ]
        )
    decisive = report.get("decisive_scenarios", []) or []
    lines.extend(["## Decisive Scenarios", ""])
    if decisive:
        for scenario in decisive[:8]:
            if not isinstance(scenario, Mapping):
                continue
            lines.append(
                f"- `{scenario.get('scenario_eval_run_id')}`: "
                f"passed={scenario.get('successful_policy_ids')} "
                f"failed={scenario.get('failed_policy_ids')}"
            )
    else:
        lines.append("- None found in this evaluator run.")
    lines.append("")
    clusters = report.get("failure_clusters", []) or []
    lines.extend(["## Failure Clusters", ""])
    if clusters:
        for cluster in clusters[:8]:
            if not isinstance(cluster, Mapping):
                continue
            lines.append(
                f"- `{cluster.get('cluster_id')}`: {cluster.get('diagnosis')} "
                f"({cluster.get('count')} labels)"
            )
    else:
        lines.append("- No failed evaluator labels were produced.")
    if selection.get("ambiguity_reasons"):
        lines.extend(
            [
                "## Ambiguity Reasons",
                "",
                *[f"- `{reason}`" for reason in _string_list(selection.get("ambiguity_reasons"))],
                "",
            ]
        )
    reruns = report.get("recommended_reruns", []) or []
    lines.extend(["## Recommended Reruns", ""])
    if reruns:
        for rerun in reruns[:8]:
            if not isinstance(rerun, Mapping):
                continue
            lines.append(f"- `{rerun.get('reason')}`: `{rerun.get('status')}`")
    else:
        lines.append("- None from this fixture handoff.")
    lines.append("")
    return "\n".join(lines)


def _visual_review_blocker_summary(
    *,
    job_id: str,
    substrate: str,
    labels: Mapping[str, Any],
    failure_labels: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    candidate_selection_report: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    label_blockers = _string_list(labels.get("visual_review_blockers"))
    failure_blockers = _string_list(failure_labels.get("visual_review_blockers"))
    scorecard_blockers = _string_list(scorecard.get("visual_review_blockers"))
    short_visual_sanity_gate = _mapping(scorecard.get("short_visual_sanity_gate"))
    blockers = _dedupe(
        [
            *label_blockers,
            *failure_blockers,
            *scorecard_blockers,
            *_string_list(short_visual_sanity_gate.get("blockers")),
        ]
    )
    return {
        "schema_version": "wam_visual_review_blocker_summary.v1",
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked_visual_review_required" if blockers else "no_visual_review_blockers",
        "evaluation_substrate": substrate,
        "blockers": blockers,
        "visual_smoke_status": scorecard.get("visual_smoke_status"),
        "visual_rollout_useful_for_task_success_review": bool(
            scorecard.get("visual_rollout_useful_for_task_success_review")
        ),
        "review_grade_policy_ranking": bool(scorecard.get("review_grade_policy_ranking")),
        "fixture_evaluator_only": bool(scorecard.get("fixture_evaluator_only")),
        "short_visual_sanity_gate": short_visual_sanity_gate,
        "candidate_selection_status": candidate_selection_report.get("status"),
        "recommended_policy_id": candidate_selection_report.get("top_policy_id"),
        "evaluator_top_policy_id": candidate_selection_report.get("evaluator_top_policy_id"),
        "source_artifacts": [
            "vision_success_labels.json",
            "failure_labels.json",
            "policy_ranking_scorecard.json",
            "candidate_selection_report.json",
            "persistent_wam_short_visual_sanity_manifest.json",
        ],
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "valid_mp4_or_provider_completion_is_not_visual_success": True,
            "visual_review_blockers_prevent_review_grade_policy_ranking": bool(blockers),
            "short_visual_sanity_required_for_review_grade_policy_ranking": True,
        },
    }


def _customer_handoff_markdown(report: Mapping[str, Any]) -> str:
    visual_gate = _mapping(report.get("visual_reviewability_gate"))
    blockers = _string_list(visual_gate.get("blockers"))
    return "\n".join(
        [
            "# WAM Policy Evaluation Handoff",
            "",
            f"Status: `{report.get('status')}`",
            f"Evaluation substrate: `{report.get('evaluation_substrate')}`",
            f"Top policy: `{report.get('top_policy_id')}`",
            f"Visual review gate: `{visual_gate.get('status')}`",
            f"Visual review blockers: `{', '.join(blockers) if blockers else 'none'}`",
            "",
            (
                "This ranks policies inside the configured evaluator. Generated rollouts "
                "and fixture labels are support artifacts for sim-ranking and failure triage."
            ),
            "",
        ]
    )


def _customer_handoff_report(
    *,
    job_id: str,
    substrate: str,
    scorecard: Mapping[str, Any],
    candidate_selection_report: Mapping[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    visual_gate = {
        "status": scorecard.get("review_grade_policy_ranking_status")
        or "blocked_visual_review_required",
        "visual_smoke_status": scorecard.get("visual_smoke_status") or FIXTURE_VISUAL_SMOKE_STATUS,
        "visual_rollout_useful_for_task_success_review": bool(
            scorecard.get("visual_rollout_useful_for_task_success_review")
        ),
        "review_grade_policy_ranking": bool(scorecard.get("review_grade_policy_ranking")),
        "fixture_evaluator_only": bool(scorecard.get("fixture_evaluator_only")),
        "short_visual_sanity_gate": _mapping(scorecard.get("short_visual_sanity_gate")),
        "blockers": _string_list(scorecard.get("visual_review_blockers")),
    }
    consistency_signal_summary = _mapping(
        scorecard.get("forward_inverse_consistency_signal_summary")
    )
    return {
        "schema_version": "wam_customer_handoff_report.v1",
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "generated",
        "evaluation_substrate": substrate,
        "top_policy_id": candidate_selection_report.get("top_policy_id"),
        "evaluator_top_policy_id": candidate_selection_report.get("evaluator_top_policy_id"),
        "candidate_selection_report_path": "candidate_selection_report.json",
        "candidate_selection_summary": {
            "status": candidate_selection_report.get("status"),
            "top_policy_id": candidate_selection_report.get("top_policy_id"),
            "evaluator_top_policy_id": candidate_selection_report.get("evaluator_top_policy_id"),
            "runner_up_policy_id": candidate_selection_report.get("runner_up_policy_id"),
            "margin": candidate_selection_report.get("margin"),
            "tie_or_ambiguity_status": candidate_selection_report.get("tie_or_ambiguity_status"),
            "candidate_shortlist": candidate_selection_report.get("candidate_shortlist"),
        },
        "legacy_scorecard_top_policy_id": scorecard.get("top_policy_id"),
        "visual_reviewability_gate": visual_gate,
        "forward_inverse_consistency_signal_summary": consistency_signal_summary,
        "artifact_paths": {
            key: value
            for key, value in WAM_ARTIFACT_PATHS.items()
            if key
            not in {
                "customer_handoff_report_markdown",
                "candidate_selection_report_markdown",
                "real_world_validation_followup_request",
                "wam_real_world_validation_anchor_manifest",
            }
        },
        "reader_boundary": (
            "Generated WAM rollouts and fixture-only labels are model-derived support "
            "artifacts for sim-ranking and failure triage. They become review-grade "
            "task-success evidence only when an explicit visual smoke artifact says "
            "the rollout is useful for task-success review."
        ),
        "claim_boundary": {
            **_claim_boundary(substrate=substrate, generated_at=generated_at),
            "visual_smoke_required_for_review_grade_policy_ranking": True,
            "short_visual_sanity_required_for_review_grade_policy_ranking": True,
            "review_label_refs_required_for_review_grade_policy_ranking": True,
            "fixture_evaluator_only": bool(scorecard.get("fixture_evaluator_only")),
            "review_grade_policy_ranking": bool(scorecard.get("review_grade_policy_ranking")),
        },
    }


def _blocked_wam_artifacts(
    *,
    job_dir: Path,
    job_id: str,
    substrate: str,
    generated_at: str,
    blockers: Sequence[str],
) -> Dict[str, Any]:
    registry = write_evaluation_substrate_registry(job_dir, generated_at=generated_at)
    matrix = _read_optional_mapping(job_dir / "scenario_eval_matrix.json")
    policy_manifest = _read_optional_mapping(job_dir / "policy_package_manifest.json")
    request_payload = _read_optional_mapping(job_dir / "job_request.json")
    policies = _policy_candidates(request=request_payload, policy_manifest=policy_manifest)
    policy_binding = _policy_interface_binding(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        policy_manifest=policy_manifest,
        policies=policies,
        generated_at=generated_at,
    )
    runtime_package = _provider_runtime_package(
        capture_root=job_dir.parents[2] if len(job_dir.parents) >= 3 else job_dir,
        job_dir=job_dir,
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scenario_eval_run_count=len(_matrix_runs(matrix)),
        policies=policies,
        generated_at=generated_at,
        artifact_output_uri=None,
        budget_usd=None,
    )
    provider_execution = _provider_execution_manifest(
        substrate=substrate,
        generated_at=generated_at,
        status="blocked",
        command_used=False,
        blockers=blockers,
    )
    provider_cost = _provider_cost_ledger(
        substrate=substrate,
        generated_at=generated_at,
        budget_usd=None,
        status="blocked",
    )
    provider_upload = _provider_artifact_upload_proof(
        substrate=substrate,
        generated_at=generated_at,
        artifact_output_uri=None,
    )
    request = build_wam_evaluation_request(
        job_id=job_id,
        substrate=substrate,
        generated_at=generated_at,
        status="blocked",
        blockers=blockers,
    )
    empty_rollout_manifest = {
        "schema_version": WAM_ROLLOUT_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked",
        "evaluation_substrate": substrate,
        "blockers": list(blockers),
        "rollout_count": 0,
        "rollouts": [],
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    empty_results = {
        "schema_version": WAM_ROLLOUT_RESULTS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "job_id": job_id,
        "status": "blocked",
        "evaluation_substrate": substrate,
        "blockers": list(blockers),
        "rollout_count": 0,
        "rollouts": [],
        "claim_boundary": _claim_boundary(substrate=substrate, generated_at=generated_at),
    }
    labels = build_fixture_vision_success_labels(
        rollout_results=empty_results,
        generated_at=generated_at,
    )
    trace = _normalized_attempt_trace(substrate=substrate, labels=labels, generated_at=generated_at)
    failure_labels = _failure_labels(
        substrate=substrate,
        trace=trace,
        generated_at=generated_at,
    )
    scorecard = _policy_scorecard(
        substrate=substrate,
        labels=labels,
        generated_at=generated_at,
        required_scenario_eval_run_ids=[
            _string(run.get("scenario_eval_run_id")) for run in _matrix_runs(matrix)
        ],
        policy_ids=[_string(policy.get("policy_id")) for policy in policies],
        evidence_root=job_dir,
    )
    claim_boundary = _claim_boundary(substrate=substrate, generated_at=generated_at)
    review_queue = _vision_review_queue(
        substrate=substrate,
        labels=labels,
        generated_at=generated_at,
    )
    followup = _real_world_validation_followup(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated_at,
    )
    srcc_plan = _srcc_validation_plan(job_id=job_id, substrate=substrate, generated_at=generated_at)
    anchor_manifest = _real_world_anchor_manifest(
        job_dir=job_dir,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated_at,
    )
    candidate_report = _candidate_selection_report(
        job_id=job_id,
        substrate=substrate,
        matrix=matrix,
        policies=policies,
        labels=labels,
        failure_labels=failure_labels,
        scorecard=scorecard,
        followup=followup,
        anchor_manifest=anchor_manifest,
        generated_at=generated_at,
    )
    validation_envelope = _customer_validation_envelope(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        anchor_manifest=anchor_manifest,
        generated_at=generated_at,
    )
    production_ops = _production_ops_manifest(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        provider_execution=provider_execution,
        generated_at=generated_at,
        artifact_output_uri=None,
        budget_usd=None,
    )
    cross_check_plan = _classical_sim_cross_check_plan(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        generated_at=generated_at,
    )
    handoff = _customer_handoff_report(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        candidate_selection_report=candidate_report,
        generated_at=generated_at,
    )
    task_eval_run_report = _wam_task_eval_run_report(
        job_id=job_id,
        request=request_payload,
        matrix=matrix,
        policies=policies,
        substrate=substrate,
        labels=labels,
        trace=trace,
        scorecard=scorecard,
        provider_execution=provider_execution,
        policy_binding=policy_binding,
        generated_at=generated_at,
    )
    visual_blocker_summary = _visual_review_blocker_summary(
        job_id=job_id,
        substrate=substrate,
        labels=labels,
        failure_labels=failure_labels,
        scorecard=scorecard,
        candidate_selection_report=candidate_report,
        generated_at=generated_at,
    )
    payloads = {
        "wam_provider_runtime_package": runtime_package,
        "wam_provider_execution_manifest": provider_execution,
        "wam_provider_cost_control_ledger": provider_cost,
        "wam_provider_artifact_upload_proof": provider_upload,
        "wam_policy_interface_binding": policy_binding,
        "wam_evaluation_request": request,
        "wam_rollout_manifest": empty_rollout_manifest,
        "wam_rollout_results": empty_results,
        "vision_success_labels": labels,
        "normalized_attempt_trace": trace,
        "task_eval_run_report": task_eval_run_report,
        "failure_labels": failure_labels,
        "policy_ranking_scorecard": scorecard,
        "wam_eval_claim_boundary": claim_boundary,
        "wam_vision_success_review_queue": review_queue,
        "real_world_validation_followup_request": followup,
        "srcc_validation_plan": srcc_plan,
        "wam_real_world_validation_anchor_manifest": anchor_manifest,
        "wam_customer_validation_envelope": validation_envelope,
        "wam_production_ops_manifest": production_ops,
        "wam_classical_sim_cross_check_plan": cross_check_plan,
        "candidate_selection_report": candidate_report,
        "visual_review_blocker_summary": visual_blocker_summary,
        "customer_handoff_report": handoff,
    }
    _write_wam_artifacts(job_dir, payloads)
    write_text(
        job_dir / WAM_ARTIFACT_PATHS["candidate_selection_report_markdown"],
        _candidate_selection_markdown(candidate_report),
    )
    write_text(
        job_dir / WAM_ARTIFACT_PATHS["customer_handoff_report_markdown"],
        _customer_handoff_markdown(handoff),
    )
    return {
        "status": "blocked",
        "blockers": list(blockers),
        "evaluation_substrate_registry": registry,
        **payloads,
        "artifact_paths": dict(WAM_ARTIFACT_PATHS),
    }


def run_wam_eval_job(
    *,
    capture_root: str | Path,
    job_dir: str | Path,
    evaluation_substrate: str = "fixture_wam",
    allow_live_provider: bool = False,
    provider_command: str | None = None,
    artifact_output_uri: str | None = None,
    budget_usd: float | None = None,
    max_retries: int = 0,
    timeout_seconds: int = 120,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Run the deterministic local WAM evaluator for an existing robot-eval job."""

    resolved_capture_root = Path(capture_root).resolve()
    resolved_job_dir = Path(job_dir).resolve()
    ensure_dir(resolved_job_dir)
    generated = generated_at or utc_now_iso()
    substrate = normalize_evaluation_substrate(evaluation_substrate)
    job_id = resolved_job_dir.name
    if substrate not in WAM_EVALUATION_SUBSTRATES:
        return _blocked_wam_artifacts(
            job_dir=resolved_job_dir,
            job_id=job_id,
            substrate=substrate,
            generated_at=generated,
            blockers=["evaluation_substrate_is_not_wam"],
        )

    matrix = _read_optional_mapping(resolved_job_dir / "scenario_eval_matrix.json")
    policy_manifest = _read_optional_mapping(resolved_job_dir / "policy_package_manifest.json")
    request_payload = _read_optional_mapping(resolved_job_dir / "job_request.json")
    runs = _matrix_runs(matrix)
    policies = _policy_candidates(request=request_payload, policy_manifest=policy_manifest)
    blockers: list[str] = []
    if not runs:
        blockers.append("scenario_eval_matrix_missing_or_empty")
    if not policies:
        blockers.append("policy_candidates_missing")
    if blockers:
        return _blocked_wam_artifacts(
            job_dir=resolved_job_dir,
            job_id=job_id,
            substrate=substrate,
            generated_at=generated,
            blockers=blockers,
        )

    registry = write_evaluation_substrate_registry(resolved_job_dir, generated_at=generated)
    policy_binding = _policy_interface_binding(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        policy_manifest=policy_manifest,
        policies=policies,
        generated_at=generated,
    )
    provider_command_text = _substrate_provider_command(substrate, provider_command)
    provider_runtime_package = _provider_runtime_package(
        capture_root=resolved_capture_root,
        job_dir=resolved_job_dir,
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scenario_eval_run_count=len(runs),
        policies=policies,
        generated_at=generated,
        artifact_output_uri=artifact_output_uri,
        budget_usd=budget_usd,
    )
    provider_runtime_package_path = (
        resolved_job_dir / WAM_ARTIFACT_PATHS["wam_provider_runtime_package"]
    )
    write_json(provider_runtime_package_path, provider_runtime_package)
    request = build_wam_evaluation_request(
        job_id=job_id,
        substrate=substrate,
        policy_ids=[_string(policy.get("policy_id")) for policy in policies],
        generated_at=generated,
    )
    rollouts: list[Dict[str, Any]] = []
    provider_execution_detail: Dict[str, Any] = {}
    provider_payload: Dict[str, Any] = {}
    provider_execution_status = "not_required_fixture"
    provider_execution_blockers: list[str] = []
    provider_command_used = False
    if substrate == "fixture_wam":
        for policy in policies:
            for run in runs:
                rollouts.append(
                    _rollout_for_run(
                        job_dir=resolved_job_dir,
                        substrate=substrate,
                        policy=policy,
                        run=run,
                        index=len(rollouts) + 1,
                        generated_at=generated,
                    )
                )
    else:
        provider_execution_blockers.extend(
            _live_provider_gate_blockers(allow_live_provider=allow_live_provider)
        )
        auth_status = _provider_auth_status(substrate)
        if not provider_command_text:
            provider_execution_blockers.append(
                f"{substrate}_provider_adapter_not_configured_for_local_run"
            )
        if not auth_status["auth_available"]:
            provider_execution_blockers.append(f"{substrate}_auth_env_missing")
        if provider_execution_blockers:
            return _blocked_wam_artifacts(
                job_dir=resolved_job_dir,
                job_id=job_id,
                substrate=substrate,
                generated_at=generated,
                blockers=provider_execution_blockers,
            )
        output_path = resolved_job_dir / "wam_provider" / "wam_provider_output.json"
        attempts = 0
        last_status = "blocked"
        last_payload: Any = {}
        last_detail: Dict[str, Any] = {}
        for attempt in range(max(0, max_retries) + 1):
            attempts = attempt + 1
            last_status, last_payload, last_detail = _run_provider_command(
                command_text=provider_command_text,
                runtime_package_path=provider_runtime_package_path,
                output_path=output_path,
                substrate=substrate,
                artifact_output_uri=artifact_output_uri,
                timeout_seconds=timeout_seconds,
            )
            rollouts = _normalize_provider_rollouts(
                payload=last_payload,
                substrate=substrate,
                generated_at=generated,
            )
            if last_status == "completed" and rollouts:
                break
        provider_command_used = True
        provider_execution_status = (
            "completed" if last_status == "completed" and rollouts else "blocked"
        )
        provider_payload = _mapping(last_payload)
        provider_execution_detail = {
            **last_detail,
            "normalized_rollout_count": len(rollouts),
            "attempt_count": attempts,
        }
        provider_execution_blockers.extend(_string_list(last_detail.get("blockers")))
        if not rollouts:
            provider_execution_blockers.append("wam_provider_output_missing_rollouts")
    rollout_manifest = _rollout_manifest(
        job_id=job_id,
        substrate=substrate,
        rollouts=rollouts,
        generated_at=generated,
    )
    rollout_results = _rollout_results(
        job_id=job_id,
        substrate=substrate,
        rollouts=rollouts,
        generated_at=generated,
    )
    labels = build_fixture_vision_success_labels(
        rollout_results=rollout_results,
        generated_at=generated,
    )
    trace = _normalized_attempt_trace(substrate=substrate, labels=labels, generated_at=generated)
    failure_labels = _failure_labels(
        substrate=substrate,
        trace=trace,
        generated_at=generated,
    )
    prediction, calibration, breakage = _prediction_ledgers(
        substrate=substrate,
        trace=trace,
        failure_labels=failure_labels,
        generated_at=generated,
    )
    scorecard = _policy_scorecard(
        substrate=substrate,
        labels=labels,
        generated_at=generated,
        required_scenario_eval_run_ids=[_string(run.get("scenario_eval_run_id")) for run in runs],
        policy_ids=[_string(policy.get("policy_id")) for policy in policies],
        evidence_root=resolved_job_dir,
    )
    claim_boundary = _claim_boundary(substrate=substrate, generated_at=generated)
    if provider_command_used and provider_execution_status == "completed":
        claim_boundary = {**claim_boundary, "live_provider_calls_performed": True}
    review_queue = _vision_review_queue(
        substrate=substrate,
        labels=labels,
        generated_at=generated,
    )
    followup = _real_world_validation_followup(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated,
    )
    srcc_plan = _srcc_validation_plan(job_id=job_id, substrate=substrate, generated_at=generated)
    anchor_manifest = _real_world_anchor_manifest(
        job_dir=resolved_job_dir,
        substrate=substrate,
        scorecard=scorecard,
        generated_at=generated,
    )
    candidate_report = _candidate_selection_report(
        job_id=job_id,
        substrate=substrate,
        matrix=matrix,
        policies=policies,
        labels=labels,
        failure_labels=failure_labels,
        scorecard=scorecard,
        followup=followup,
        anchor_manifest=anchor_manifest,
        generated_at=generated,
    )
    validation_envelope = _customer_validation_envelope(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        anchor_manifest=anchor_manifest,
        generated_at=generated,
    )
    provider_execution = _provider_execution_manifest(
        substrate=substrate,
        generated_at=generated,
        status=provider_execution_status
        if provider_execution_status != "not_required_fixture"
        else "not_required_fixture",
        command_used=provider_command_used,
        detail=provider_execution_detail,
        blockers=provider_execution_blockers,
        attempt_count=1
        if substrate == "fixture_wam"
        else int(provider_execution_detail.get("attempt_count") or 1),
        max_retries=max_retries,
    )
    provider_cost = _provider_cost_ledger(
        substrate=substrate,
        generated_at=generated,
        budget_usd=budget_usd,
        status=provider_execution_status,
        duration_seconds=_number(provider_execution_detail.get("duration_seconds"), None),
    )
    provider_upload = _provider_artifact_upload_proof(
        substrate=substrate,
        generated_at=generated,
        artifact_output_uri=artifact_output_uri,
        provider_payload=provider_payload,
    )
    production_ops = _production_ops_manifest(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        provider_execution=provider_execution,
        generated_at=generated,
        artifact_output_uri=artifact_output_uri,
        budget_usd=budget_usd,
    )
    cross_check_plan = _classical_sim_cross_check_plan(
        job_id=job_id,
        substrate=substrate,
        request=request_payload,
        scorecard=scorecard,
        generated_at=generated,
    )
    handoff = _customer_handoff_report(
        job_id=job_id,
        substrate=substrate,
        scorecard=scorecard,
        candidate_selection_report=candidate_report,
        generated_at=generated,
    )
    task_eval_run_report = _wam_task_eval_run_report(
        job_id=job_id,
        request=request_payload,
        matrix=matrix,
        policies=policies,
        substrate=substrate,
        labels=labels,
        trace=trace,
        scorecard=scorecard,
        provider_execution=provider_execution,
        policy_binding=policy_binding,
        generated_at=generated,
    )
    visual_blocker_summary = _visual_review_blocker_summary(
        job_id=job_id,
        substrate=substrate,
        labels=labels,
        failure_labels=failure_labels,
        scorecard=scorecard,
        candidate_selection_report=candidate_report,
        generated_at=generated,
    )
    payloads = {
        "wam_provider_runtime_package": provider_runtime_package,
        "wam_provider_execution_manifest": provider_execution,
        "wam_provider_cost_control_ledger": provider_cost,
        "wam_provider_artifact_upload_proof": provider_upload,
        "wam_policy_interface_binding": policy_binding,
        "wam_evaluation_request": request,
        "wam_rollout_manifest": rollout_manifest,
        "wam_rollout_results": rollout_results,
        "vision_success_labels": labels,
        "normalized_attempt_trace": trace,
        "task_eval_run_report": task_eval_run_report,
        "failure_labels": failure_labels,
        "prediction_outcome_ledger": prediction,
        "calibration_report": calibration,
        "breakage_library": breakage,
        "policy_ranking_scorecard": scorecard,
        "wam_eval_claim_boundary": claim_boundary,
        "wam_vision_success_review_queue": review_queue,
        "real_world_validation_followup_request": followup,
        "srcc_validation_plan": srcc_plan,
        "wam_real_world_validation_anchor_manifest": anchor_manifest,
        "wam_customer_validation_envelope": validation_envelope,
        "wam_production_ops_manifest": production_ops,
        "wam_classical_sim_cross_check_plan": cross_check_plan,
        "candidate_selection_report": candidate_report,
        "visual_review_blocker_summary": visual_blocker_summary,
        "customer_handoff_report": handoff,
    }
    _write_wam_artifacts(resolved_job_dir, payloads)
    write_text(
        resolved_job_dir / WAM_ARTIFACT_PATHS["candidate_selection_report_markdown"],
        _candidate_selection_markdown(candidate_report),
    )
    write_text(
        resolved_job_dir / WAM_ARTIFACT_PATHS["customer_handoff_report_markdown"],
        _customer_handoff_markdown(handoff),
    )
    return {
        "status": "completed" if rollouts else "blocked",
        "blockers": provider_execution_blockers,
        "evaluation_substrate": substrate,
        "evaluation_substrate_registry": registry,
        **payloads,
        "artifact_paths": dict(WAM_ARTIFACT_PATHS),
        "claim_boundary": claim_boundary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic fixture WAM eval job")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--evaluation-substrate", default="fixture_wam")
    parser.add_argument("--allow-live-provider", action="store_true")
    parser.add_argument("--provider-command")
    parser.add_argument("--artifact-output-uri")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    result = run_wam_eval_job(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        evaluation_substrate=args.evaluation_substrate,
        allow_live_provider=args.allow_live_provider,
        provider_command=args.provider_command,
        artifact_output_uri=args.artifact_output_uri,
        budget_usd=args.budget_usd,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[wam-eval] status={result['status']}")
    print(f"[wam-eval] job_dir={Path(args.job_dir).resolve()}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
