from __future__ import annotations

from blueprint_pipeline.sc3_eval_protocol import (
    SC3_EVAL_PROTOCOL_SCHEMA_VERSION,
    build_sc3_eval_protocol_artifact,
)


def _robot_pov_manifest(camera_count: int = 3) -> dict[str, object]:
    return {
        "status": "completed",
        "observation_count": 2,
        "camera_profile_registry": {
            "active_robot_profile_id": "customer_bot",
            "profiles": [
                {
                    "robot_profile_id": "customer_bot",
                    "cameras": [
                        {"camera_id": f"cam_{index}"}
                        for index in range(camera_count)
                    ],
                }
            ],
        },
        "robot_camera_profile_launch_readiness": {
            "status": "smoke_only_owner_calibration_required"
        },
    }


def test_sc3_protocol_defines_required_data_and_blocks_correlation_without_anchors() -> None:
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={"robot_profile": {"robot_profile_id": "customer_bot"}},
        policy_package_manifest={
            "selected_modalities": ["policy_api_endpoint"],
            "modalities": {
                "policy_api_endpoint": {
                    "selected": True,
                    "status": "launch_ready_review_required",
                    "interface_contract": {"observation_schema": {"schema_id": "x"}},
                }
            },
        },
        policy_execution_manifest={
            "status": "blocked",
            "policy_execution_trace_path": "policy_execution_trace.json",
            "modality_results": {
                "policy_api_endpoint": {
                    "status": "blocked_policy_execution_gate",
                    "execution_performed": False,
                    "robot_policy_execution_proven": False,
                }
            },
        },
        robot_pov_observation_manifest=_robot_pov_manifest(),
    )

    assert artifact["schema_version"] == SC3_EVAL_PROTOCOL_SCHEMA_VERSION
    assert artifact["source_facts"]["paper_id"] == "arXiv:2606.18610v3"
    assert artifact["data_requirements"]["synchronized_multi_view_cameras"]["status"] == "ready"
    assert artifact["data_requirements"]["accepted_anchor_joins"]["status"] == (
        "correlation_not_measured"
    )
    assert artifact["metrics"]["pearson_success_rate_correlation"]["status"] == (
        "correlation_not_measured"
    )
    assert artifact["claim_boundary"]["ninety_percent_or_better_blueprint_accuracy_claim_allowed"] is False
    assert artifact["policy_adapter_pack_contracts"][0]["launch_reviewable_without_execution"] is True


def test_sc3_protocol_fails_closed_on_missing_multiview_and_preserves_ranking_status() -> None:
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={
            "selected_modalities": ["recorded_action_trace"],
            "modalities": {
                "recorded_action_trace": {
                    "selected": True,
                    "status": "launch_ready_review_required",
                }
            },
        },
        policy_execution_manifest={
            "status": "completed",
            "modality_results": {
                "recorded_action_trace": {
                    "status": "completed_reference_replay",
                    "execution_performed": False,
                    "robot_policy_execution_proven": False,
                }
            },
        },
        robot_pov_observation_manifest=_robot_pov_manifest(camera_count=1),
        policy_ranking_scorecard={
            "status": "blocked_inconclusive_ranking",
            "comparison_blockers": ["policy_coverage_missing_required_scenario_eval_run_ids"],
        },
    )

    assert artifact["status"] == "blocked_protocol_inputs_missing"
    assert artifact["data_requirements"]["synchronized_multi_view_cameras"]["status"] == "blocked"
    assert artifact["ranking_interpretation"]["status"] == "blocked_inconclusive_ranking"
    assert artifact["ranking_interpretation"]["missing_symmetric_coverage_status"] == (
        "blocked_inconclusive_ranking"
    )


def test_sc3_protocol_reports_metrics_only_when_anchor_values_exist() -> None:
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={"selected_modalities": ["high_level_skill_trace"]},
        policy_execution_manifest={"status": "completed", "modality_results": {}},
        robot_pov_observation_manifest=_robot_pov_manifest(),
        sim_vs_real_calibration_report={
            "accepted_anchor_count": 4,
            "pearson_success_rate_correlation": 0.91,
            "spearman_rank_correlation": 0.88,
            "mean_maximum_rank_violation": 0.12,
            "mean_absolute_success_rate_error": 0.07,
        },
    )

    assert artifact["metrics"]["pearson_success_rate_correlation"]["status"] == "measured"
    assert artifact["metrics"]["pearson_success_rate_correlation"]["value"] == 0.91
    assert artifact["metrics"]["spearman_rank_correlation"]["value"] == 0.88
    assert artifact["metrics"]["srcc"]["value"] == 0.88
    assert artifact["metrics"]["mean_maximum_rank_violation"]["value"] == 0.12
    assert artifact["metrics"]["calibration_error"]["value"] == 0.07
