"""Shared failure diagnosis artifact helpers.

Generated and simulator-derived failure labels are support artifacts until a
human/VLM review accepts them or real-world validation is supplied.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


FAILURE_LABEL_PROOF_EFFECT = "none_until_review_accepted_or_real_world_validation_supplied"

ACCEPTED_FAILURE_REVIEW_STATUSES = {
    "accepted",
    "accepted_reviewed_failure_label",
    "human_reviewed_accepted",
    "vlm_reviewed_accepted",
    "owner_review_accepted",
    "real_world_validation_supplied",
}

REVIEWABLE_FAILURE_REVIEW_STATUSES = {
    "review_required",
    "pending_review",
    "ready_for_review",
    "reviewable_failure_hypothesis",
    "reviewable_generated_media",
    "available_for_human_audit_not_required_for_sim_only_metric",
}

NON_REVIEWABLE_FAILURE_REVIEW_STATUSES = {
    "non_reviewable_failure_hypothesis",
    "nonreviewable_failure_hypothesis",
    "blocked_nonreviewable_generated_rollout",
}


def mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def string(value: Any) -> str:
    return str(value or "").strip()


def string_list(value: Any) -> list[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def dedupe(values: Iterable[Any]) -> list[str]:
    return string_list(list(values))


def number_is_zero(value: Any) -> bool:
    try:
        if value is None or isinstance(value, bool):
            return False
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def refs_from_fields(payload: Mapping[str, Any], *field_names: str) -> list[str]:
    refs: list[str] = []
    for field_name in field_names:
        refs.extend(string_list(payload.get(field_name)))
    return dedupe(refs)


def frame_or_clip_refs(payload: Mapping[str, Any]) -> list[str]:
    refs = refs_from_fields(
        payload,
        "frame_or_clip_refs",
        "frameOrClipRefs",
        "media_refs",
        "mediaRefs",
        "clip_refs",
        "clipRefs",
        "frame_refs",
        "frameRefs",
    )
    refs.extend(
        string(payload.get(key))
        for key in (
            "generated_video_path",
            "generatedVideoPath",
            "generated_frame_path",
            "generatedFramePath",
            "robot_pov_video_path",
            "robotPovVideoPath",
            "review_video_path",
            "reviewVideoPath",
            "video_path",
            "videoPath",
            "clip_path",
            "clipPath",
            "contact_sheet_path",
            "contactSheetPath",
        )
        if string(payload.get(key))
    )
    artifact_paths = mapping(payload.get("artifact_paths") or payload.get("artifactPaths"))
    refs.extend(
        string(artifact_paths.get(key))
        for key in (
            "generated_video",
            "generated_video_path",
            "generated_frame",
            "generated_frame_path",
            "robot_pov_video",
            "review_video",
            "clip",
            "contact_sheet",
        )
        if string(artifact_paths.get(key))
    )
    return dedupe(refs)


def evidence_refs(payload: Mapping[str, Any], *, extra_refs: Sequence[str] = ()) -> list[str]:
    refs = refs_from_fields(payload, "evidence_refs", "evidenceRefs")
    refs.extend(frame_or_clip_refs(payload))
    refs.extend(refs_from_fields(payload, "source_trace_refs", "sourceTraceRefs"))
    refs.extend(string(ref) for ref in extra_refs if string(ref))
    artifact_paths = mapping(payload.get("artifact_paths") or payload.get("artifactPaths"))
    refs.extend(string(value) for value in artifact_paths.values() if string(value))
    return dedupe(refs)


def failure_root_cause_category(
    failure_mode_ids: Sequence[str],
    *,
    ood_flags: Sequence[str] = (),
    failure_reason: str | None = None,
) -> str:
    haystack = " ".join([*failure_mode_ids, *ood_flags, string(failure_reason)]).lower()
    if any(term in haystack for term in ("collision", "contact", "safety", "fall", "unsafe")):
        return "safety_or_contact_risk"
    if any(term in haystack for term in ("navigation", "blocked", "path", "target", "timeout")):
        return "navigation_or_path_planning"
    if any(term in haystack for term in ("grasp", "object", "placement", "manipulation")):
        return "manipulation_or_object_selection"
    if any(
        term in haystack
        for term in ("ood", "uncertain", "uncertainty", "glare", "visual", "reviewable", "quality")
    ):
        return "world_model_uncertainty"
    if failure_mode_ids:
        return "task_policy_failure"
    return "unknown"


def remediation_candidate(root_cause_category: str, failure_mode_ids: Sequence[str]) -> str:
    if root_cause_category == "navigation_or_path_planning":
        return "add blocked-path recovery examples and rerun scenario-family navigation checks"
    if root_cause_category == "manipulation_or_object_selection":
        return "add object-centric manipulation examples and verify target-object grounding"
    if root_cause_category == "safety_or_contact_risk":
        return "tighten clearance/contact constraints and review collision or near-miss traces"
    if root_cause_category == "world_model_uncertainty":
        return "collect stronger review media or real-world anchors before using this diagnosis"
    if failure_mode_ids:
        return "review failed attempt traces and add targeted policy-improvement examples"
    return "collect stronger evidence before proposing remediation"


def review_status_for_failure_label(
    *,
    supplied_review_status: Any = None,
    supplied_status: Any = None,
    generated_rollout: bool = True,
    frame_or_clip_ref_count: int = 0,
) -> str:
    review_status = string(supplied_review_status)
    if (
        generated_rollout
        and frame_or_clip_ref_count == 0
        and review_status
        and review_status not in ACCEPTED_FAILURE_REVIEW_STATUSES
        and review_status not in NON_REVIEWABLE_FAILURE_REVIEW_STATUSES
    ):
        return "non_reviewable_failure_hypothesis"
    if review_status:
        return review_status
    status = string(supplied_status)
    if (
        generated_rollout
        and frame_or_clip_ref_count == 0
        and status not in ACCEPTED_FAILURE_REVIEW_STATUSES
    ):
        return "non_reviewable_failure_hypothesis"
    if status in ACCEPTED_FAILURE_REVIEW_STATUSES | REVIEWABLE_FAILURE_REVIEW_STATUSES:
        return status
    return "review_required"


def review_status_is_accepted_or_reviewable(review_status: str) -> bool:
    normalized = string(review_status)
    return normalized in ACCEPTED_FAILURE_REVIEW_STATUSES | REVIEWABLE_FAILURE_REVIEW_STATUSES


def label_identifier(label: Mapping[str, Any], index: int) -> str:
    return (
        string(label.get("label_id") or label.get("labelId"))
        or string(label.get("attempt_id") or label.get("attemptId"))
        or string(label.get("scenario_eval_run_id") or label.get("scenarioEvalRunId"))
        or f"label_{index:04d}"
    )


def failed_attempt_rows(source_payload: Mapping[str, Any], source_name: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for attempt in source_payload.get("attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        if (
            attempt.get("success") is False
            or string(attempt.get("status") or attempt.get("result")).lower()
            in {"collision", "failed", "failure", "timeout", "unsafe"}
            or bool(string_list(attempt.get("failure_mode_ids") or attempt.get("failureModeIds")))
        ):
            row = dict(attempt)
            row["failure_label_source_trace"] = source_name
            rows.append(row)
    return rows


def label_matches_failed_attempt(
    label: Mapping[str, Any],
    *,
    failed_attempt_ids: set[str],
    failed_run_ids: set[str],
) -> bool:
    attempt_id = string(
        label.get("attempt_id")
        or label.get("attemptId")
        or label.get("policy_attempt_id")
        or label.get("policyAttemptId")
    )
    run_id = string(label.get("scenario_eval_run_id") or label.get("scenarioEvalRunId"))
    if not failed_attempt_ids and not failed_run_ids:
        return True
    return attempt_id in failed_attempt_ids or run_id in failed_run_ids


def generated_failure_label(label: Mapping[str, Any]) -> bool:
    source = string(label.get("source") or label.get("label_source") or label.get("labelSource"))
    substrate = string(label.get("evaluation_substrate") or label.get("simulator_engine"))
    return bool(
        label.get("generated_wam_rollout")
        or label.get("model_derived_support_artifact")
        or label.get("rollout_id")
        or label.get("visual_smoke_ref")
        or "wam" in source.lower()
        or "wam" in substrate.lower()
    )


def build_failure_diagnosis_audit(
    *,
    labels_payload: Mapping[str, Any],
    trace_payload: Mapping[str, Any],
    policy_trace_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    policy_trace = policy_trace_payload or {}
    simulator_failed_attempts = failed_attempt_rows(trace_payload, "normalized_attempt_trace")
    policy_failed_attempts = failed_attempt_rows(policy_trace, "policy_execution_trace")
    failed_attempts = [*simulator_failed_attempts, *policy_failed_attempts]
    label_rows = [
        dict(label)
        for label in labels_payload.get("labels", []) or []
        if isinstance(label, Mapping)
    ]
    required_attempt_ids = sorted(
        {
            string(attempt.get("attempt_id") or attempt.get("attemptId"))
            for attempt in failed_attempts
            if string(attempt.get("attempt_id") or attempt.get("attemptId"))
        }
    )
    required_run_ids = sorted(
        {
            string(attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId"))
            for attempt in failed_attempts
            if string(attempt.get("scenario_eval_run_id") or attempt.get("scenarioEvalRunId"))
        }
    )
    failed_attempt_id_set = set(required_attempt_ids)
    failed_run_id_set = set(required_run_ids)
    zero_failures_reviewed = (
        not failed_attempts
        and not label_rows
        and string(labels_payload.get("status"))
        in {"no_failures_labeled", "zero_failures_reviewed"}
        and number_is_zero(labels_payload.get("failed_attempt_count"))
    )
    labels_missing_failure_mode_ids: list[str] = []
    labels_missing_evidence_refs: list[str] = []
    labels_missing_review_status: list[str] = []
    labels_not_accepted_or_reviewable: list[str] = []
    nonreviewable_failure_hypothesis_label_ids: list[str] = []
    generated_missing_media_or_marker: list[str] = []
    coverage_label_rows: list[Dict[str, Any]] = []
    audited_label_ids: list[str] = []
    for index, label in enumerate(label_rows, start=1):
        if not label_matches_failed_attempt(
            label,
            failed_attempt_ids=failed_attempt_id_set,
            failed_run_ids=failed_run_id_set,
        ):
            continue
        label_id = label_identifier(label, index)
        audited_label_ids.append(label_id)
        failure_mode_ids = string_list(
            label.get("failure_mode_ids")
            or label.get("failureModeIds")
            or label.get("failure_modes")
            or label.get("failureModes")
        )
        if not failure_mode_ids:
            labels_missing_failure_mode_ids.append(label_id)
        else:
            coverage_label_rows.append(label)
        refs = string_list(label.get("evidence_refs") or label.get("evidenceRefs"))
        if not refs:
            labels_missing_evidence_refs.append(label_id)
        review_status = string(label.get("review_status") or label.get("reviewStatus"))
        if not review_status:
            labels_missing_review_status.append(label_id)
        if review_status and not review_status_is_accepted_or_reviewable(review_status):
            labels_not_accepted_or_reviewable.append(label_id)
        if review_status in NON_REVIEWABLE_FAILURE_REVIEW_STATUSES:
            nonreviewable_failure_hypothesis_label_ids.append(label_id)
        frame_refs = frame_or_clip_refs(label)
        is_generated = generated_failure_label(label)
        if (
            is_generated
            and not frame_refs
            and review_status not in NON_REVIEWABLE_FAILURE_REVIEW_STATUSES
        ):
            generated_missing_media_or_marker.append(label_id)

    covered_attempt_ids = sorted(
        {
            string(
                label.get("attempt_id")
                or label.get("attemptId")
                or label.get("policy_attempt_id")
                or label.get("policyAttemptId")
            )
            for label in coverage_label_rows
            if string(
                label.get("attempt_id")
                or label.get("attemptId")
                or label.get("policy_attempt_id")
                or label.get("policyAttemptId")
            )
        }
    )
    covered_run_ids = sorted(
        {
            string(label.get("scenario_eval_run_id") or label.get("scenarioEvalRunId"))
            for label in coverage_label_rows
            if string(label.get("scenario_eval_run_id") or label.get("scenarioEvalRunId"))
        }
    )
    missing_attempt_ids = sorted(set(required_attempt_ids) - set(covered_attempt_ids))
    missing_run_ids = sorted(set(required_run_ids) - set(covered_run_ids))
    coverage_blockers: list[str] = []
    if labels_missing_failure_mode_ids:
        coverage_blockers.append("failure_labels_missing_failure_mode_ids")
    if missing_attempt_ids or missing_run_ids:
        coverage_blockers.append("failure_labels_missing_failed_attempt_coverage")
    if labels_missing_evidence_refs:
        coverage_blockers.append("failure_labels_missing_evidence_refs")
    if labels_missing_review_status:
        coverage_blockers.append("failure_labels_missing_review_status")
    review_blockers: list[str] = []
    if labels_not_accepted_or_reviewable:
        review_blockers.append("failure_labels_not_accepted_or_reviewable")
    if nonreviewable_failure_hypothesis_label_ids:
        review_blockers.append("failure_labels_nonreviewable_failure_hypotheses")
    if generated_missing_media_or_marker:
        review_blockers.append(
            "failure_labels_generated_rollout_missing_reviewable_media_or_nonreviewable_marker"
        )
    return {
        "failed_attempt_count": len(failed_attempts),
        "failed_simulator_attempt_count": len(simulator_failed_attempts),
        "failed_policy_attempt_count": len(policy_failed_attempts),
        "required_failed_attempt_ids": required_attempt_ids,
        "covered_failed_attempt_ids": covered_attempt_ids,
        "missing_failed_attempt_ids": missing_attempt_ids,
        "required_failed_scenario_eval_run_ids": required_run_ids,
        "covered_failed_scenario_eval_run_ids": covered_run_ids,
        "missing_failed_scenario_eval_run_ids": missing_run_ids,
        "audited_failure_label_ids": audited_label_ids,
        "labels_missing_failure_mode_ids": labels_missing_failure_mode_ids,
        "labels_missing_evidence_refs": labels_missing_evidence_refs,
        "labels_missing_review_status": labels_missing_review_status,
        "labels_not_accepted_or_reviewable": labels_not_accepted_or_reviewable,
        "nonreviewable_failure_hypothesis_label_ids": (
            nonreviewable_failure_hypothesis_label_ids
        ),
        "generated_rollout_labels_missing_reviewable_media_or_nonreviewable_marker": (
            generated_missing_media_or_marker
        ),
        "failure_diagnosis_coverage_complete": not coverage_blockers,
        "failure_diagnosis_review_complete": not review_blockers,
        "failure_diagnosis_complete": not coverage_blockers and not review_blockers,
        "zero_failures_reviewed": zero_failures_reviewed,
        "coverage_blockers": coverage_blockers,
        "review_blockers": review_blockers,
        "blockers": [*coverage_blockers, *review_blockers],
    }
