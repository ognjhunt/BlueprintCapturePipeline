"""Deterministic fixture vision success judge for WAM rollout artifacts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .common import utc_now_iso


VISION_SUCCESS_LABELS_SCHEMA_VERSION = "vision_success_labels.v1"
FIXTURE_VISUAL_SMOKE_STATUS = "fixture_evaluator_only_no_visual_smoke"
FIXTURE_VISUAL_REVIEW_BLOCKER = "fixture_evaluator_only_no_review_grade_visual_evidence"


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


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_fixture_vision_success_labels(
    *,
    rollout_results: Mapping[str, Any],
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Build deterministic labels from fixture rollout results.

    The fixture judge intentionally labels generated rollout metadata, not raw
    video truth. It exists so local tests exercise the full WAM contract without
    using live VLM providers.
    """

    generated = generated_at or utc_now_iso()
    substrate = _string(rollout_results.get("evaluation_substrate")) or "fixture_wam"
    rows = [
        dict(item)
        for item in rollout_results.get("rollouts", []) or []
        if isinstance(item, Mapping)
    ]
    labels: list[Dict[str, Any]] = []
    for index, rollout in enumerate(rows, start=1):
        uncertainty = _float(rollout.get("uncertainty_score"), 0.0)
        task_success = bool(rollout.get("predicted_success"))
        ood_flags = _string_list(rollout.get("ood_flags"))
        confidence = max(0.05, min(0.99, round(1.0 - uncertainty, 6)))
        failure_modes = _string_list(rollout.get("failure_mode_ids"))
        if ood_flags and "wam_ood_uncertain" not in failure_modes:
            failure_modes.append("wam_ood_uncertain")
        visual_smoke_status = (
            _string(rollout.get("visual_smoke_status"))
            or _string(rollout.get("generated_rollout_visual_smoke_status"))
            or FIXTURE_VISUAL_SMOKE_STATUS
        )
        visual_rollout_useful = bool(
            rollout.get("visual_rollout_useful_for_task_success_review")
            or rollout.get("generated_rollout_visually_useful_for_success_review")
        )
        visual_review_blockers = _string_list(
            rollout.get("visual_review_blockers")
            or rollout.get("generated_rollout_visual_quality_blockers")
            or rollout.get("blockers")
        )
        if not visual_rollout_useful and FIXTURE_VISUAL_REVIEW_BLOCKER not in visual_review_blockers:
            visual_review_blockers.append(FIXTURE_VISUAL_REVIEW_BLOCKER)
        labels.append(
            {
                "label_id": f"fixture_wam_label_{index:04d}",
                "rollout_id": rollout.get("rollout_id"),
                "attempt_id": rollout.get("attempt_id"),
                "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
                "condition_id": rollout.get("condition_id"),
                "replicate_id": rollout.get("replicate_id"),
                "replicate_seed": rollout.get("replicate_seed"),
                "scenario_variation_instance_id": rollout.get(
                    "scenario_variation_instance_id"
                ),
                "task_id": rollout.get("task_id"),
                "scenario_id": rollout.get("scenario_id"),
                "variation_name": rollout.get("variation_name"),
                "policy_id": rollout.get("policy_id"),
                "evaluation_substrate": substrate,
                "task_success": task_success,
                "confidence": confidence,
                "uncertainty_score": uncertainty,
                "failure_mode_ids": failure_modes,
                "ood_flags": ood_flags,
                "ood_registration_blockers": rollout.get(
                    "ood_registration_blockers", []
                ),
                "registered_ood_axes_complete": rollout.get(
                    "registered_ood_axes_complete"
                )
                is True,
                "labeler": "fixture_vision_success_judge",
                "status": "labeled",
                "human_review_required": bool(ood_flags or uncertainty >= 0.5),
                "model_calls_performed": False,
                "visual_smoke_status": visual_smoke_status,
                "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
                "visual_review_blockers": visual_review_blockers,
                "fixture_evaluator_only": True,
                "review_grade_visual_evidence_available": visual_rollout_useful,
                "review_grade_success_label": False,
                "authoritative_task_success_label": False,
                "claim_boundary": {
                    "label_is_deterministic_fixture_output": True,
                    "label_is_not_human_or_live_vlm_review": True,
                    "generated_rollout_video_is_not_raw_capture_evidence": True,
                    "fixture_evaluator_only": True,
                    "visual_smoke_required_for_review_grade_success_label": True,
                    "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
                    "review_grade_success_label": False,
                    "rank_fidelity_result_proven": False,
                    "public_claim_upgrade_allowed": False,
                },
            }
        )
    successful = [label for label in labels if label["task_success"]]
    failed = [label for label in labels if not label["task_success"]]
    visual_rollout_useful = bool(labels) and all(
        label["visual_rollout_useful_for_task_success_review"] for label in labels
    )
    visual_smoke_statuses = sorted(
        {
            _string(label.get("visual_smoke_status"))
            for label in labels
            if _string(label.get("visual_smoke_status"))
        }
    )
    visual_review_blockers = sorted(
        {
            blocker
            for label in labels
            for blocker in _string_list(label.get("visual_review_blockers"))
        }
    )
    return {
        "schema_version": VISION_SUCCESS_LABELS_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if labels else "blocked_missing_rollouts",
        "labeler": "fixture_vision_success_judge",
        "evaluation_substrate": substrate,
        "model_calls_performed": False,
        "visual_smoke_status": "passed_visual_quality_smoke"
        if visual_rollout_useful
        else visual_smoke_statuses[0]
        if len(visual_smoke_statuses) == 1
        else "mixed_visual_smoke_statuses"
        if visual_smoke_statuses
        else FIXTURE_VISUAL_SMOKE_STATUS,
        "visual_smoke_statuses": visual_smoke_statuses,
        "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
        "visual_review_blockers": visual_review_blockers,
        "fixture_evaluator_only": True,
        "review_grade_visual_evidence_available": visual_rollout_useful,
        "review_grade_success_labels": False,
        "label_count": len(labels),
        "successful_label_count": len(successful),
        "failed_label_count": len(failed),
        "success_rate": round(len(successful) / len(labels), 6) if labels else 0.0,
        "failure_mode_ids": sorted(
            {mode for label in labels for mode in _string_list(label.get("failure_mode_ids"))}
        ),
        "ood_label_count": sum(1 for label in labels if _string_list(label.get("ood_flags"))),
        "labels": labels,
        "claim_boundary": {
            "vision_labels_are_model_derived_support_artifacts": True,
            "fixture_labeler_used_for_local_tests": True,
            "fixture_evaluator_only": True,
            "visual_smoke_required_for_review_grade_success_label": True,
            "visual_rollout_useful_for_task_success_review": visual_rollout_useful,
            "review_grade_success_labels": False,
            "live_vlm_or_human_review_proven": False,
            "customer_specific_srcc_claimed": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
