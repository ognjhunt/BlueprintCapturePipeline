"""SC3-style evaluator protocol contract for Blueprint robot evaluation jobs.

The protocol artifact is intentionally declarative. It describes the data and
proof gates required for an SC3-Eval-style evaluator without launching a model,
computing correlations from missing anchors, or upgrading generated media into
task success, deployment approval, safety validation, or physical readiness.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SC3_EVAL_PROTOCOL_SCHEMA_VERSION = "sc3_eval_protocol.v1"
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
    "success_criteria",
    "failure_taxonomy",
    "accepted_anchor_joins",
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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    text = _string(value)
    return [text] if text else []


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
) -> dict[str, Any]:
    if accepted_anchor_count <= 0:
        return {
            "metric": metric_name,
            "status": "correlation_not_measured",
            "value": None,
            "blockers": ["accepted_real_or_owner_anchor_rows_missing"],
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
        if not modality_payload.get("selected"):
            continue
        execution = _mapping(execution_results.get(modality))
        execution_performed = bool(execution.get("execution_performed"))
        execution_proven = bool(execution.get("robot_policy_execution_proven"))
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
                    package_status not in {"", "blocked", "not_selected"}
                    and not execution_proven
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
) -> dict[str, Any]:
    """Build a no-provider SC3-compatible protocol readiness artifact."""

    scorecard = _mapping(policy_ranking_scorecard)
    correlation_ledger = _mapping(prediction_outcome_correlation_ledger)
    calibration_report = _mapping(sim_vs_real_calibration_report)
    wam_boundary = _mapping(wam_eval_claim_boundary)
    policy_trace_path = _string(policy_execution_manifest.get("policy_execution_trace_path"))
    selected_modalities = _string_list(policy_package_manifest.get("selected_modalities"))
    policy_contracts = _policy_adapter_review_contracts(
        policy_package_manifest=policy_package_manifest,
        policy_execution_manifest=policy_execution_manifest,
    )
    camera_count = _profile_camera_count(robot_pov_observation_manifest)
    initial_observation_count = int(robot_pov_observation_manifest.get("observation_count") or 0)
    if initial_observation_count <= 0:
        initial_observation_count = len(
            _list_of_mappings(robot_pov_observation_manifest.get("observations"))
        )
    generated_rollout_available = bool(
        scorecard
        or correlation_ledger
        or _string(wam_boundary.get("wam_rollout_manifest_path"))
        or _string(wam_boundary.get("wam_rollout_results_path"))
    )
    accepted_anchor_count = _accepted_anchor_count(calibration_report, correlation_ledger)
    ranking_status = _string(scorecard.get("status")) or "not_requested"
    ranking_blockers = _string_list(scorecard.get("comparison_blockers") or scorecard.get("blockers"))
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
            "status": "ready" if camera_count >= 3 else "blocked",
            "observed_camera_count": camera_count,
            "required_camera_count": 3,
            "blockers": [] if camera_count >= 3 else ["synchronized_multi_view_camera_count_lt_3"],
        },
        "robot_camera_profile": {
            "status": _string(
                _mapping(
                    robot_pov_observation_manifest.get("robot_camera_profile_launch_readiness")
                ).get("status")
            )
            or "present",
            "path": "robot_camera_profile_launch_readiness.json",
            "claim_boundary": "camera profile is calibration/review support, not owner proof by itself",
        },
        "action_chunks": {
            "status": "reviewable" if selected_modalities else "blocked",
            "selected_policy_modalities": selected_modalities,
            "policy_adapter_pack_count": len(policy_contracts),
            "blockers": [] if selected_modalities else ["policy_package_selected_modality_missing"],
        },
        "initial_observations": {
            "status": "ready" if initial_observation_count > 0 else "blocked",
            "observation_count": initial_observation_count,
            "path": "robot_pov_observation_manifest.json",
            "blockers": [] if initial_observation_count > 0 else ["initial_observations_missing"],
        },
        "generated_rollout_frames": {
            "status": "available" if generated_rollout_available else "not_available",
            "claim_boundary": "generated rollout frames are support artifacts only",
        },
        "policy_requery_trace": {
            "status": _string(policy_execution_manifest.get("status")) or "not_available",
            "path": policy_trace_path or "policy_execution_trace.json",
            "robot_team_policy_execution_proven": bool(
                policy_execution_manifest.get("robot_team_policy_execution_proven")
            ),
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
            "status": "ready" if accepted_anchor_count > 0 else "correlation_not_measured",
            "accepted_anchor_count": accepted_anchor_count,
            "join_keys": [
                "scenario_eval_run_id",
                "policy_id",
                "task_id",
                "scenario_variation_instance_id",
            ],
            "blockers": [] if accepted_anchor_count > 0 else ["accepted_anchor_rows_missing"],
        },
    }

    calibration_error = _optional_number(
        calibration_report.get("mean_absolute_success_rate_error")
        or calibration_report.get("calibration_error")
    )
    metrics = {
        "pearson_success_rate_correlation": _metric_status(
            metric_name="pearson_success_rate_correlation",
            value=_optional_number(calibration_report.get("pearson_success_rate_correlation")),
            accepted_anchor_count=accepted_anchor_count,
        ),
        "spearman_rank_correlation": _metric_status(
            metric_name="spearman_rank_correlation",
            value=_optional_number(calibration_report.get("spearman_rank_correlation")),
            accepted_anchor_count=accepted_anchor_count,
        ),
        "srcc": _metric_status(
            metric_name="srcc",
            value=_optional_number(
                calibration_report.get("srcc")
                or calibration_report.get("spearman_rank_correlation")
            ),
            accepted_anchor_count=accepted_anchor_count,
        ),
        "mean_maximum_rank_violation": _metric_status(
            metric_name="mean_maximum_rank_violation",
            value=_optional_number(
                calibration_report.get("mean_maximum_rank_violation")
                or calibration_report.get("mmrv")
            ),
            accepted_anchor_count=accepted_anchor_count,
        ),
        "calibration_error": _metric_status(
            metric_name="calibration_error",
            value=calibration_error,
            accepted_anchor_count=accepted_anchor_count,
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

    blocking_data_keys = [
        key
        for key, value in data_requirements.items()
        if _string(_mapping(value).get("status")) in {"blocked", "missing"}
    ]
    if blocking_data_keys:
        status = "blocked_protocol_inputs_missing"
    elif ranking_interpretation in {"blocked_inconclusive_ranking", "completed_ambiguous_ranking"}:
        status = ranking_interpretation
    else:
        status = "protocol_defined"

    return {
        "schema_version": SC3_EVAL_PROTOCOL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "source_facts": SC3_SOURCE_FACTS,
        "required_data": SC3_REQUIRED_DATA,
        "required_metrics": SC3_REQUIRED_METRICS,
        "data_requirements": data_requirements,
        "metrics": metrics,
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
            else "correlation_metrics_require_current_report_values"
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
        },
    }
