"""Deterministic fixture vision success judge for WAM rollout artifacts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .common import utc_now_iso


VISION_SUCCESS_LABELS_SCHEMA_VERSION = "vision_success_labels.v1"


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
        labels.append(
            {
                "label_id": f"fixture_wam_label_{index:04d}",
                "rollout_id": rollout.get("rollout_id"),
                "attempt_id": rollout.get("attempt_id"),
                "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
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
                "labeler": "fixture_vision_success_judge",
                "status": "labeled",
                "human_review_required": bool(ood_flags or uncertainty >= 0.5),
                "model_calls_performed": False,
                "claim_boundary": {
                    "label_is_deterministic_fixture_output": True,
                    "label_is_not_human_or_live_vlm_review": True,
                    "generated_rollout_video_is_not_raw_capture_evidence": True,
                    "robot_readiness_proven": False,
                    "public_claim_upgrade_allowed": False,
                },
            }
        )
    successful = [label for label in labels if label["task_success"]]
    failed = [label for label in labels if not label["task_success"]]
    return {
        "schema_version": VISION_SUCCESS_LABELS_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if labels else "blocked_missing_rollouts",
        "labeler": "fixture_vision_success_judge",
        "evaluation_substrate": substrate,
        "model_calls_performed": False,
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
            "live_vlm_or_human_review_proven": False,
            "customer_specific_srcc_claimed": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
