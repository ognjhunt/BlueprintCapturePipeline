"""SC3-style evaluator protocol contract for Blueprint robot evaluation jobs.

The protocol artifact is intentionally declarative. It describes the data and
proof gates required for an SC3-Eval-style evaluator without launching a model,
computing correlations from missing anchors, or upgrading generated media into
task success, deployment approval, safety validation, or physical readiness.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_normalization import (
    DEFAULT_ACTION_DIM,
    SC3_ACTION_ORDER,
    SC3_ACTION_REPRESENTATION,
    SC3_ACTION_UNITS,
)
from .robot_eval_calibration import (
    calibration_metrics_from_policy_summaries as _calibration_metrics_from_policy_summaries,
    policy_anchor_summaries as _policy_anchor_summaries,
)
from .sc3_fidelity_contracts import (
    validate_anchor_artifacts,
    validate_benchmark_cards,
    validate_checkpoint_attestation,
    validate_external_study,
    validate_horizon_execution_trace,
    validate_ood_registry,
    validate_synchronized_multiview,
)


SC3_EVAL_PROTOCOL_SCHEMA_VERSION = "sc3_eval_protocol.v2"
SC3_EVAL_PROTOCOL_ARTIFACT = "sc3_eval_protocol.json"

SC3_SOURCE_FACTS = {
    "paper_id": "arXiv:2606.18610v3",
    "source_url": "https://arxiv.org/abs/2606.18610",
    "html_url": "https://arxiv.org/html/2606.18610v3",
    "submitted_on": "2026-06-17",
    "current_version": "v3",
    "last_revised_on": "2026-06-26",
    "source_reverified_on": "2026-07-02",
    "title": "SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation",
    "initializes_from": "Cosmos3-Nano",
    "training_dataset": {
        "hours": 381,
        "physical_scene_count": 1,
        "object_category_count": 12,
        "camera_view_count": 3,
        "camera_views": ["third_person_camera_1", "third_person_camera_2", "wrist_camera"],
        "native_resolution": "480x640",
        "recorded_hz": 20,
        "action_representation": "7d_delta_end_effector_pose",
    },
    "evaluation_setup": {
        "policy_checkpoint_count": 7,
        "matched_initial_conditions_per_checkpoint": "36_to_37",
        "max_rollout_seconds": 20,
        "policy_action_chunk_count": 25,
        "executed_receding_horizon_action_count": 16,
        "world_model_prediction_horizon_frames": 24,
        "retained_execution_horizon_frames": 16,
    },
    "reported_metrics": {
        "headline_closed_loop": {"pearson": 0.929, "mmrv": 0.119},
        "in_distribution_online": {
            "sc3_eval_pearson": 0.984,
            "sc3_eval_mmrv": 0.022,
            "cosmos_predict25_pearson": 0.897,
            "cosmos_predict25_mmrv": 0.090,
        },
        "out_of_distribution_online": {
            "sc3_eval_pearson": 0.870,
            "sc3_eval_mmrv": 0.171,
            "cosmos_predict25_pearson": 0.871,
            "cosmos_predict25_mmrv": 0.195,
        },
    },
}

SC3_RELIABILITY_SIGNALS = [
    "forward_inverse_dynamics_consistency",
    "cross_view_consistency",
    "uncertainty_driven_early_termination",
]

SC3_REQUIRED_DATA = [
    "synchronized_multi_view_cameras",
    "robot_camera_profile",
    "action_chunks",
    "initial_observations",
    "generated_rollout_frames",
    "policy_requery_trace",
    "sc3_trained_checkpoint",
    "success_criteria",
    "failure_taxonomy",
    "accepted_anchor_joins",
    "frozen_ood_axes",
    "benchmark_card_separation",
]

SC3_REQUIRED_METRICS = [
    "pearson_success_rate_correlation",
    "spearman_rank_correlation",
    "srcc",
    "mean_maximum_rank_violation",
    "calibration_error",
    "confidence_abstention",
]

SUCCESS_CRITERIA_CONTRACT = {
    "source": "SC3-Eval v3 paper plus Blueprint task/eval cards",
    "sc3_reference_criteria": [
        "language_following",
        "object_lifting",
        "object_placing",
    ],
    "blueprint_requirement": (
        "Each job must keep task-specific success criteria explicit instead of "
        "letting generated-video labels become task-success truth."
    ),
}

FAILURE_TAXONOMY_CONTRACT = {
    "source": "SC3-Eval v3 paper plus Blueprint failure_labels.json",
    "sc3_reference_failure_modes": [
        "language_following_failure",
        "object_lifting_failure",
        "object_placing_failure",
    ],
    "blueprint_requirement": (
        "Failure labels remain review-required support evidence until accepted "
        "by a human, owner proof, or an explicit downstream closure gate."
    ),
}

CLAIM_BOUNDARY = {
    "sc3_protocol_artifact_is_not_model_execution": True,
    "self_consistency_is_reliability_support_only": True,
    "forward_inverse_consistency_does_not_label_task_success": True,
    "cross_view_consistency_does_not_label_task_success": True,
    "uncertainty_abstention_does_not_label_task_success": True,
    "generated_rollout_frames_are_model_derived_support_artifacts": True,
    "generated_video_success_label_is_separate_from_consistency": True,
    "correlation_requires_accepted_real_or_owner_anchors": True,
    "ninety_percent_or_better_blueprint_accuracy_claim_allowed": False,
    "deployment_approval_proven": False,
    "physical_robot_readiness_proven": False,
    "safety_validation_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _strict_bool(value: Any) -> bool:
    return value is True


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    text = _string(value)
    return [text] if text else []


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_sha256(value: Any) -> bool:
    text = _string(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_artifact_path(ref_value: Any, *, prefix: str) -> tuple[Path | None, list[str]]:
    ref = _mapping(ref_value)
    path_text = _string(ref.get("path"))
    digest = _string(ref.get("sha256")).lower()
    blockers: list[str] = []
    if not path_text:
        return None, [f"{prefix}_path_missing"]
    path = Path(path_text).expanduser()
    if not path.is_file():
        blockers.append(f"{prefix}_file_missing")
    if not _is_sha256(digest):
        blockers.append(f"{prefix}_sha256_invalid")
    elif path.is_file() and _file_sha256(path) != digest:
        blockers.append(f"{prefix}_sha256_mismatch")
    return (path if not blockers else None), blockers


def _validate_robot_profile_artifact(
    robot_pov_observation_manifest: Mapping[str, Any],
) -> list[str]:
    registry = _mapping(robot_pov_observation_manifest.get("camera_profile_registry"))
    path, blockers = _validated_artifact_path(
        robot_pov_observation_manifest.get("robot_camera_profile_artifact"),
        prefix="robot_camera_profile_artifact",
    )
    payload: dict[str, Any] = {}
    if path is not None:
        try:
            payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    if not (
        payload.get("schema_version") == "sc3_robot_camera_profile_evidence.v1"
        and _mapping(payload.get("camera_profile_registry")) == registry
    ):
        blockers.append("robot_camera_profile_artifact_content_mismatch")
    return sorted(set(blockers))


def _validate_initial_observation_artifacts(
    robot_pov_observation_manifest: Mapping[str, Any],
) -> tuple[int, list[str]]:
    refs = _list_of_mappings(robot_pov_observation_manifest.get("observation_artifacts"))
    blockers: list[str] = []
    observed_indices: list[int] = []
    seen_image_digests: set[str] = set()
    for index, ref in enumerate(refs):
        path, artifact_blockers = _validated_artifact_path(
            ref,
            prefix=f"initial_observation_artifact:{index}",
        )
        blockers.extend(artifact_blockers)
        payload: dict[str, Any] = {}
        if path is not None:
            try:
                payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        observed_index = payload.get("observation_index")
        if not (
            payload.get("schema_version") == "sc3_initial_observation_evidence.v1"
            and isinstance(observed_index, int)
            and not isinstance(observed_index, bool)
        ):
            blockers.append(f"initial_observation_artifact_content_invalid:{index}")
            continue
        observed_indices.append(observed_index)
        image_ref = _mapping(payload.get("image_artifact"))
        _, image_blockers = _validated_artifact_path(
            image_ref,
            prefix=f"initial_observation_image_artifact:{index}",
        )
        blockers.extend(image_blockers)
        image_digest = _string(image_ref.get("sha256")).lower()
        if image_digest in seen_image_digests:
            blockers.append(f"initial_observation_image_reused:{index}")
        seen_image_digests.add(image_digest)
    if observed_indices and sorted(observed_indices) != list(range(len(refs))):
        blockers.append("initial_observation_indices_not_contiguous")
    declared_count = robot_pov_observation_manifest.get("observation_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(refs)
    ):
        blockers.append("initial_observation_declared_count_mismatch")
    if not refs:
        blockers.append("initial_observation_artifacts_missing")
    return len(refs) if not blockers else 0, sorted(set(blockers))


def _validate_generated_rollout_artifact(
    wam_boundary: Mapping[str, Any],
) -> list[str]:
    path, blockers = _validated_artifact_path(
        wam_boundary.get("generated_rollout_artifact"),
        prefix="generated_rollout_artifact",
    )
    payload: dict[str, Any] = {}
    if path is not None:
        try:
            payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    frame_refs = _list_of_mappings(payload.get("generated_frame_artifacts"))
    if not (
        payload.get("schema_version") == "sc3_generated_rollout_evidence.v1"
        and payload.get("status") == "completed"
        and _string(payload.get("rollout_id"))
        and _is_sha256(payload.get("world_model_checkpoint_sha256"))
        and frame_refs
    ):
        blockers.append("generated_rollout_artifact_content_invalid")
    for index, frame_ref in enumerate(frame_refs):
        _, frame_blockers = _validated_artifact_path(
            frame_ref,
            prefix=f"generated_rollout_frame_artifact:{index}",
        )
        blockers.extend(frame_blockers)
    return sorted(set(blockers))


def _profile_camera_count(robot_pov_observation_manifest: Mapping[str, Any]) -> int:
    registry = _mapping(robot_pov_observation_manifest.get("camera_profile_registry"))
    profiles = _list_of_mappings(registry.get("profiles"))
    active_id = _string(registry.get("active_robot_profile_id"))
    if active_id:
        for profile in profiles:
            if _string(profile.get("robot_profile_id")) == active_id:
                cameras = _list_of_mappings(profile.get("cameras"))
                return len(cameras)
    if profiles:
        return max(len(_list_of_mappings(profile.get("cameras"))) for profile in profiles)
    profile = _mapping(robot_pov_observation_manifest.get("robot_profile"))
    return len(_list_of_mappings(profile.get("cameras") or profile.get("camera_rigs")))


def _accepted_anchor_count(
    calibration_report: Mapping[str, Any],
    prediction_outcome_correlation_ledger: Mapping[str, Any],
) -> int:
    direct = calibration_report.get("accepted_anchor_count")
    if isinstance(direct, int):
        return direct
    matched = prediction_outcome_correlation_ledger.get("matched_real_world_outcome_count")
    if isinstance(matched, int):
        return matched
    return 0


def _metric_status(
    *,
    metric_name: str,
    value: float | None,
    accepted_anchor_count: int,
    decision_grade: bool,
) -> dict[str, Any]:
    if accepted_anchor_count <= 0:
        return {
            "metric": metric_name,
            "status": "correlation_not_measured",
            "value": None,
            "blockers": ["accepted_real_or_owner_anchor_rows_missing"],
        }
    if not decision_grade:
        return {
            "metric": metric_name,
            "status": "inconclusive_insufficient_n",
            "value": None,
            "blockers": ["accepted_anchor_rows_below_matched_decision_grade_contract"],
        }
    if value is None:
        return {
            "metric": metric_name,
            "status": "blocked_metric_missing",
            "value": None,
            "blockers": [f"{metric_name}_missing_from_calibration_report"],
        }
    return {
        "metric": metric_name,
        "status": "measured",
        "value": value,
        "blockers": [],
    }


def _policy_adapter_review_contracts(
    *,
    policy_package_manifest: Mapping[str, Any],
    policy_execution_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    modalities = _mapping(policy_package_manifest.get("modalities"))
    execution_results = _mapping(policy_execution_manifest.get("modality_results"))
    contracts: list[dict[str, Any]] = []
    for modality, payload in modalities.items():
        modality_payload = _mapping(payload)
        if not _strict_bool(modality_payload.get("selected")):
            continue
        execution = _mapping(execution_results.get(modality))
        execution_performed = _strict_bool(execution.get("execution_performed"))
        execution_proven = _strict_bool(execution.get("robot_policy_execution_proven"))
        package_status = _string(modality_payload.get("status"))
        execution_status = _string(execution.get("status")) or "not_executed"
        contracts.append(
            {
                "modality": modality,
                "package_status": package_status,
                "execution_status": execution_status,
                "execution_performed": execution_performed,
                "robot_team_policy_execution_proven": execution_proven,
                "launch_reviewable_without_execution": (
                    package_status not in {"", "blocked", "not_selected"} and not execution_proven
                ),
                "same_observation_action_contract": True,
                "interface_contract": _mapping(modality_payload.get("interface_contract")),
                "claim_boundary": {
                    "reviewable_policy_adapter_pack_is_not_execution_proof": True,
                    "execution_proof_requires_policy_execution_manifest": True,
                    "rank_fidelity_result_proven": False,
                },
            }
        )
    return contracts


def _action_chunks_requirement(
    *,
    selected_modalities: list[str],
    policy_contracts: list[dict[str, Any]],
    action_norm: Mapping[str, Any],
) -> dict[str, Any]:
    """SC3 action-chunk readiness requires validated per-dimension normalization.

    Per the paper, actions must be 7-D delta-EE normalized per-dimension
    across the corpus; presence of a selected modality alone is not enough.
    """
    blockers: list[str] = []
    if not selected_modalities:
        blockers.append("policy_package_selected_modality_missing")
    norm_status = _string(action_norm.get("status"))
    if not action_norm:
        blockers.append("action_normalization_manifest_missing")
    elif norm_status != "validated":
        blockers.append("action_normalization_not_validated")
    else:
        if int(action_norm.get("declared_action_dim") or 0) != DEFAULT_ACTION_DIM:
            blockers.append("action_normalization_dim_not_7")
        if _string(action_norm.get("canonical_action_representation")) != (
            SC3_ACTION_REPRESENTATION
        ):
            blockers.append("action_normalization_representation_invalid")
        if list(action_norm.get("action_order") or []) != list(SC3_ACTION_ORDER):
            blockers.append("action_normalization_order_invalid")
        if list(action_norm.get("action_units") or []) != list(SC3_ACTION_UNITS):
            blockers.append("action_normalization_units_invalid")
        if action_norm.get("exact_consumed_trace_bound") is not True:
            blockers.append("action_normalization_not_bound_to_exact_consumed_trace")
        if action_norm.get("all_dimensions_nonzero_variance") is not True:
            blockers.append("action_normalization_zero_or_invalid_variance")
        if int(action_norm.get("rejected_episode_count") or 0) != 0:
            blockers.append("action_normalization_contains_rejected_episodes")
        if int(action_norm.get("accepted_episode_count") or 0) <= 0:
            blockers.append("action_normalization_has_no_accepted_episodes")
        for field_name in (
            "action_norm_stats_path",
            "action_norm_stats_sha256",
            "normalized_action_corpus_path",
            "normalized_action_corpus_sha256",
            "source_trace_sha256",
        ):
            if not _string(action_norm.get(field_name)):
                blockers.append(f"{field_name}_missing")
    return {
        "status": "reviewable" if not blockers else "blocked",
        "selected_policy_modalities": selected_modalities,
        "policy_adapter_pack_count": len(policy_contracts),
        "action_normalization_status": norm_status or "missing",
        "action_norm_stats_path": _string(action_norm.get("action_norm_stats_path")) or None,
        "action_norm_stats_sha256": _string(action_norm.get("action_norm_stats_sha256")) or None,
        "normalized_action_corpus_path": _string(action_norm.get("normalized_action_corpus_path"))
        or None,
        "normalized_action_corpus_sha256": _string(
            action_norm.get("normalized_action_corpus_sha256")
        )
        or None,
        "source_trace_sha256": _string(action_norm.get("source_trace_sha256")) or None,
        "blockers": blockers,
    }


def build_sc3_eval_protocol_artifact(
    *,
    generated_at: str,
    job_request: Mapping[str, Any],
    policy_package_manifest: Mapping[str, Any],
    policy_execution_manifest: Mapping[str, Any],
    robot_pov_observation_manifest: Mapping[str, Any],
    policy_ranking_scorecard: Mapping[str, Any] | None = None,
    prediction_outcome_correlation_ledger: Mapping[str, Any] | None = None,
    sim_vs_real_calibration_report: Mapping[str, Any] | None = None,
    wam_eval_claim_boundary: Mapping[str, Any] | None = None,
    action_normalization_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-provider SC3-compatible protocol readiness artifact."""

    scorecard = _mapping(policy_ranking_scorecard)
    action_norm = _mapping(action_normalization_manifest)
    correlation_ledger = _mapping(prediction_outcome_correlation_ledger)
    calibration_report = _mapping(sim_vs_real_calibration_report)
    wam_boundary = _mapping(wam_eval_claim_boundary)
    multiview_validation = validate_synchronized_multiview(
        _mapping(
            robot_pov_observation_manifest.get("synchronized_multiview")
            or job_request.get("sc3_synchronized_multiview")
        )
    )
    horizon_validation = validate_horizon_execution_trace(
        _mapping(
            policy_execution_manifest.get("sc3_horizon_execution_trace")
            or job_request.get("sc3_horizon_execution_trace")
        )
    )
    checkpoint_validation = validate_checkpoint_attestation(
        _mapping(
            wam_boundary.get("sc3_checkpoint_attestation")
            or job_request.get("sc3_checkpoint_attestation")
        )
    )
    ood_validation = validate_ood_registry(
        _mapping(
            calibration_report.get("frozen_ood_registry")
            or job_request.get("sc3_frozen_ood_registry")
        )
    )
    external_study_validation = validate_external_study(
        _mapping(
            calibration_report.get("external_sc3_study") or job_request.get("external_sc3_study")
        )
    )
    benchmark_validation = validate_benchmark_cards(_mapping(job_request.get("benchmark_cards")))
    raw_anchor_rows = _list_of_mappings(
        calibration_report.get("accepted_anchor_rows")
        or correlation_ledger.get("accepted_anchors")
        or correlation_ledger.get("joined_rows")
    )
    anchor_artifact_validation = validate_anchor_artifacts(raw_anchor_rows)
    validated_anchor_rows = _list_of_mappings(anchor_artifact_validation.get("valid_rows"))
    recomputed_anchor_summaries = _policy_anchor_summaries(validated_anchor_rows)
    recomputed_metrics = _calibration_metrics_from_policy_summaries(recomputed_anchor_summaries)
    policy_trace_path = _string(policy_execution_manifest.get("policy_execution_trace_path"))
    selected_modalities = _string_list(policy_package_manifest.get("selected_modalities"))
    policy_contracts = _policy_adapter_review_contracts(
        policy_package_manifest=policy_package_manifest,
        policy_execution_manifest=policy_execution_manifest,
    )
    camera_count = _profile_camera_count(robot_pov_observation_manifest)
    robot_profile_blockers = _validate_robot_profile_artifact(robot_pov_observation_manifest)
    initial_observation_count, initial_observation_blockers = (
        _validate_initial_observation_artifacts(robot_pov_observation_manifest)
    )
    generated_rollout_blockers = _validate_generated_rollout_artifact(wam_boundary)
    generated_rollout_available = not generated_rollout_blockers
    accepted_anchor_count = len(validated_anchor_rows)
    anchor_decision_grade = (
        anchor_artifact_validation.get("decision_grade_status") == "decision_grade"
    )
    ranking_status = _string(scorecard.get("status")) or "not_requested"
    ranking_blockers = _string_list(
        scorecard.get("comparison_blockers") or scorecard.get("blockers")
    )
    if ranking_status == "not_requested":
        ranking_interpretation = "not_requested"
    elif ranking_status in {"blocked_inconclusive_ranking", "completed_ambiguous_ranking"}:
        ranking_interpretation = ranking_status
    elif ranking_blockers:
        ranking_interpretation = "blocked_inconclusive_ranking"
    else:
        ranking_interpretation = ranking_status

    data_requirements = {
        "synchronized_multi_view_cameras": {
            "status": "ready" if multiview_validation.get("status") == "validated" else "blocked",
            "observed_camera_count": camera_count,
            "required_camera_count": 3,
            "validation": multiview_validation,
            "blockers": _string_list(multiview_validation.get("blockers")),
        },
        "robot_camera_profile": {
            "status": "ready" if not robot_profile_blockers else "blocked",
            "artifact": _mapping(
                robot_pov_observation_manifest.get("robot_camera_profile_artifact")
            ),
            "blockers": robot_profile_blockers,
            "claim_boundary": "camera profile is calibration/review support, not owner proof by itself",
        },
        "action_chunks": _action_chunks_requirement(
            selected_modalities=selected_modalities,
            policy_contracts=policy_contracts,
            action_norm=action_norm,
        ),
        "initial_observations": {
            "status": "ready" if initial_observation_count > 0 else "blocked",
            "observation_count": initial_observation_count,
            "artifacts": _list_of_mappings(
                robot_pov_observation_manifest.get("observation_artifacts")
            ),
            "blockers": initial_observation_blockers,
        },
        "generated_rollout_frames": {
            "status": "ready" if generated_rollout_available else "blocked",
            "artifact": _mapping(wam_boundary.get("generated_rollout_artifact")),
            "blockers": generated_rollout_blockers,
            "claim_boundary": "generated rollout frames are support artifacts only",
        },
        "policy_requery_trace": {
            "status": "ready" if horizon_validation.get("status") == "validated" else "blocked",
            "path": policy_trace_path or "policy_execution_trace.json",
            "robot_team_policy_execution_proven": _strict_bool(
                policy_execution_manifest.get("robot_team_policy_execution_proven")
            ),
            "horizon_execution_validation": horizon_validation,
            "blockers": _string_list(horizon_validation.get("blockers")),
        },
        "sc3_trained_checkpoint": {
            "status": "ready" if checkpoint_validation.get("status") == "validated" else "blocked",
            "validation": checkpoint_validation,
            "blockers": _string_list(checkpoint_validation.get("blockers")),
        },
        "success_criteria": {
            "status": "defined",
            **SUCCESS_CRITERIA_CONTRACT,
        },
        "failure_taxonomy": {
            "status": "defined",
            **FAILURE_TAXONOMY_CONTRACT,
        },
        "accepted_anchor_joins": {
            "status": (
                "ready"
                if anchor_decision_grade
                else "inconclusive_insufficient_n"
                if accepted_anchor_count > 0
                else "correlation_not_measured"
            ),
            "accepted_anchor_count": accepted_anchor_count,
            "join_keys": [
                "policy_id",
                "checkpoint_id",
                "policy_checkpoint_sha256",
                "criterion_id",
                "registered_split",
                "split_manifest_id",
                "split_manifest_sha256",
                "task_family",
                "task_id",
                "scenario_eval_run_id",
                "scenario_variation_instance_id",
                "condition_id",
                "condition_source_id",
                "replicate_id",
                "replicate_seed",
            ],
            "unit_of_analysis_fields": [
                "policy_id",
                "checkpoint_id",
                "criterion_id",
                "registered_split",
                "task_family",
            ],
            "public_rank_fidelity_requires_explicit_unit_of_analysis_fields": True,
            "blockers": (
                []
                if anchor_decision_grade
                else _string_list(anchor_artifact_validation.get("decision_grade_blockers"))
                if accepted_anchor_count > 0
                else ["accepted_anchor_rows_missing"]
            ),
            "artifact_validation": anchor_artifact_validation,
            "recomputed_unit_summaries": recomputed_anchor_summaries,
        },
        "frozen_ood_axes": {
            "status": "ready" if ood_validation.get("status") == "validated" else "blocked",
            "validation": ood_validation,
            "blockers": _string_list(ood_validation.get("blockers")),
        },
        "benchmark_card_separation": {
            "status": "ready" if benchmark_validation.get("status") == "validated" else "blocked",
            "validation": benchmark_validation,
            "blockers": _string_list(benchmark_validation.get("blockers")),
        },
    }

    calibration_error = _optional_number(recomputed_metrics.get("mean_absolute_success_rate_error"))
    metrics = {
        "pearson_success_rate_correlation": _metric_status(
            metric_name="pearson_success_rate_correlation",
            value=_optional_number(recomputed_metrics.get("pearson_success_rate_correlation")),
            accepted_anchor_count=accepted_anchor_count,
            decision_grade=anchor_decision_grade,
        ),
        "spearman_rank_correlation": _metric_status(
            metric_name="spearman_rank_correlation",
            value=_optional_number(recomputed_metrics.get("spearman_rank_correlation")),
            accepted_anchor_count=accepted_anchor_count,
            decision_grade=anchor_decision_grade,
        ),
        "srcc": _metric_status(
            metric_name="srcc",
            value=_optional_number(recomputed_metrics.get("spearman_rank_correlation")),
            accepted_anchor_count=accepted_anchor_count,
            decision_grade=anchor_decision_grade,
        ),
        "mean_maximum_rank_violation": _metric_status(
            metric_name="mean_maximum_rank_violation",
            value=_optional_number(recomputed_metrics.get("mean_maximum_rank_violation")),
            accepted_anchor_count=accepted_anchor_count,
            decision_grade=anchor_decision_grade,
        ),
        "calibration_error": _metric_status(
            metric_name="calibration_error",
            value=calibration_error,
            accepted_anchor_count=accepted_anchor_count,
            decision_grade=anchor_decision_grade,
        ),
        "confidence_abstention": {
            "metric": "confidence_abstention",
            "status": "defined",
            "signals": [
                "confidence",
                "uncertainty_score",
                "ood_flags",
                "early_termination",
            ],
            "claim_boundary": (
                "Confidence and abstention are reliability controls, not success labels."
            ),
        },
    }
    submitted_rank_fidelity_claim_eligibility = _mapping(
        calibration_report.get("rank_fidelity_claim_eligibility")
    )
    supplied_metric_mismatches: list[str] = []
    for metric_name in (
        "pearson_success_rate_correlation",
        "spearman_rank_correlation",
        "mean_maximum_rank_violation",
        "mean_absolute_success_rate_error",
    ):
        supplied_value = calibration_report.get(metric_name)
        supplied = _optional_number(supplied_value)
        recomputed = _optional_number(recomputed_metrics.get(metric_name))
        if metric_name in calibration_report and supplied is None:
            supplied_metric_mismatches.append(f"supplied_{metric_name}_missing_or_nonfinite")
        elif supplied is not None and (recomputed is None or abs(supplied - recomputed) > 1e-6):
            supplied_metric_mismatches.append(
                f"supplied_{metric_name}_does_not_match_validated_rows"
            )
    recomputed_claim_metrics_complete = bool(
        accepted_anchor_count > 0
        and anchor_decision_grade
        and all(
            _optional_number(recomputed_metrics.get(metric_name)) is not None
            for metric_name in (
                "pearson_success_rate_correlation",
                "spearman_rank_correlation",
                "mean_maximum_rank_violation",
                "mean_absolute_success_rate_error",
            )
        )
    )
    public_rank_fidelity_claim_eligible = bool(
        recomputed_claim_metrics_complete
        and not supplied_metric_mismatches
        and multiview_validation.get("status") == "validated"
        and horizon_validation.get("status") == "validated"
        and checkpoint_validation.get("status") == "validated"
        and ood_validation.get("status") == "validated"
        and benchmark_validation.get("status") == "validated"
        and external_study_validation.get("status") == "validated"
    )
    computed_eligibility_blockers: list[str] = []
    if accepted_anchor_count <= 0:
        computed_eligibility_blockers.append("validated_anchor_count_zero")
    if not recomputed_claim_metrics_complete:
        computed_eligibility_blockers.append("recomputed_claim_metrics_incomplete")
    if not anchor_decision_grade:
        computed_eligibility_blockers.append("accepted_anchor_decision_grade_contract_not_met")
    computed_eligibility_blockers.extend(supplied_metric_mismatches)
    for name, validation in (
        ("multiview", multiview_validation),
        ("horizon", horizon_validation),
        ("checkpoint", checkpoint_validation),
        ("ood", ood_validation),
        ("benchmark_cards", benchmark_validation),
        ("external_study", external_study_validation),
    ):
        if validation.get("status") != "validated":
            computed_eligibility_blockers.append(f"{name}_validation_not_ready")
    rank_fidelity_claim_eligibility = {
        "schema_version": "sc3_recomputed_rank_fidelity_claim_eligibility.v1",
        "status": "eligible" if public_rank_fidelity_claim_eligible else "ineligible",
        "public_rank_fidelity_claim_eligible": public_rank_fidelity_claim_eligible,
        "validated_anchor_count": accepted_anchor_count,
        "recomputed_metrics_complete": recomputed_claim_metrics_complete,
        "blockers": sorted(set(computed_eligibility_blockers)),
        "caller_supplied_eligibility_ignored": bool(submitted_rank_fidelity_claim_eligibility),
    }

    blocking_data_keys = [
        key
        for key, value in data_requirements.items()
        if _string(_mapping(value).get("status")) in {"blocked", "missing"}
    ]
    runtime_blocking_keys = [
        key
        for key in blocking_data_keys
        if key
        not in {
            "accepted_anchor_joins",
            "frozen_ood_axes",
            "benchmark_card_separation",
        }
    ]
    protocol_defined = True
    runtime_ready = not runtime_blocking_keys
    claim_ready = bool(runtime_ready and public_rank_fidelity_claim_eligible)
    if not runtime_ready:
        status = "blocked_runtime_inputs_missing_or_invalid"
    elif ranking_interpretation in {"blocked_inconclusive_ranking", "completed_ambiguous_ranking"}:
        status = ranking_interpretation
    elif not claim_ready:
        status = "runtime_ready_claim_blocked_external_or_fidelity_evidence"
    else:
        status = "claim_ready"

    return {
        "schema_version": SC3_EVAL_PROTOCOL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "source_facts": SC3_SOURCE_FACTS,
        "required_data": SC3_REQUIRED_DATA,
        "required_metrics": SC3_REQUIRED_METRICS,
        "data_requirements": data_requirements,
        "metrics": metrics,
        "recomputed_metrics": recomputed_metrics,
        "supplied_metric_mismatches": supplied_metric_mismatches,
        "readiness": {
            "protocol_defined": protocol_defined,
            "runtime_ready": runtime_ready,
            "claim_ready": claim_ready,
            "blocking_data_keys": blocking_data_keys,
            "runtime_blocking_data_keys": runtime_blocking_keys,
            "external_study_validation": external_study_validation,
        },
        "rank_fidelity_claim_eligibility": rank_fidelity_claim_eligibility,
        "submitted_rank_fidelity_claim_eligibility": (submitted_rank_fidelity_claim_eligibility),
        "public_rank_fidelity_claim_eligible": public_rank_fidelity_claim_eligible,
        "policy_adapter_pack_contracts": policy_contracts,
        "robot_embodiment_contract": {
            "robot_profile": _mapping(
                job_request.get("robot_profile") or job_request.get("robotProfile")
            ),
            "robot_pack_is_data_driven": True,
            "g1_is_reference_embodiment_not_customer_requirement": True,
        },
        "ranking_interpretation": {
            "status": ranking_interpretation,
            "source_policy_ranking_status": ranking_status,
            "blockers": ranking_blockers,
            "missing_symmetric_coverage_status": (
                "blocked_inconclusive_ranking"
                if ranking_interpretation == "blocked_inconclusive_ranking"
                else None
            ),
        },
        "reliability_signals": SC3_RELIABILITY_SIGNALS,
        "correlation_claim_status": (
            "correlation_not_measured"
            if accepted_anchor_count <= 0
            else "eligible_preregistered_external_rank_fidelity"
            if public_rank_fidelity_claim_eligible
            else "diagnostic_only_public_claim_ineligible"
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "artifact_paths": {
            "robot_camera_profile_registry": "robot_camera_profile_registry.json",
            "robot_camera_profile_launch_readiness": "robot_camera_profile_launch_readiness.json",
            "robot_pov_observation_manifest": "robot_pov_observation_manifest.json",
            "policy_package_manifest": "policy_package_manifest.json",
            "policy_execution_manifest": "policy_execution_manifest.json",
            "policy_execution_trace": "policy_execution_trace.json",
            "policy_ranking_scorecard": "policy_ranking_scorecard.json",
            "wam_eval_claim_boundary": "wam_eval_claim_boundary.json",
            "wam_prediction_outcome_correlation_ledger": (
                "wam_prediction_outcome_correlation_ledger.json"
            ),
            "sim_vs_real_calibration_report": "sim_vs_real_calibration_report.json",
            "action_validation_manifest": "action_validation_manifest.json",
        },
    }
