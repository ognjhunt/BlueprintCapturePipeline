from __future__ import annotations

from blueprint_pipeline.wam_isaac_evaluation_hierarchy import (
    build_wam_isaac_disagreement_contract,
    manipulation_effector_progress_report,
)


def _accepted_frozen_benchmark() -> dict:
    return {"status": "accepted", "authority_kind": "frozen_benchmark"}


def test_manipulation_effector_gate_scales_for_eight_frame_prefix() -> None:
    frames = []
    for index in range(8):
        fraction = index / 7.0
        frames.append(
            {
                "landmarks": [
                    {
                        "name": "right_wrist_yaw_link",
                        "world_xyz": [0.5 - 0.01 * fraction, 0.0, 0.0],
                        "image_projection": {
                            "available": True,
                            "u_px": 100.0 + 6.5 * fraction,
                            "v_px": 200.0,
                        },
                    }
                ]
            }
        )

    result = manipulation_effector_progress_report(
        {
            "task_target_world_xyz_m": [0.0, 0.0, 0.0],
            "controller_fk_sequence": frames,
        }
    )

    assert result["capability_gate_passed"] is True
    assert result["observed_frame_count"] == 8
    assert result["reference_frame_count"] == 16
    assert result["duration_scale"] == 0.5
    assert result["minimum_required_progress_m"] == 0.0075
    assert result["minimum_required_projected_motion_px"] == 4.0


def test_manipulation_effector_gate_binds_nonzero_floors_for_short_prefix() -> None:
    frames = []
    for index in range(4):
        fraction = index / 3.0
        frames.append(
            {
                "landmarks": [
                    {
                        "name": "right_wrist_yaw_link",
                        "world_xyz": [0.5 - 0.006 * fraction, 0.0, 0.0],
                        "image_projection": {
                            "available": True,
                            "u_px": 100.0 + 3.0 * fraction,
                            "v_px": 200.0,
                        },
                    }
                ]
            }
        )

    result = manipulation_effector_progress_report(
        {
            "task_target_world_xyz_m": [0.0, 0.0, 0.0],
            "controller_fk_sequence": frames,
        }
    )

    assert result["capability_gate_passed"] is True
    assert result["observed_frame_count"] == 4
    assert result["duration_scale"] == 0.25
    assert result["minimum_required_progress_m"] == 0.005
    assert result["minimum_required_projected_motion_px"] == 2.0


def test_manipulation_effector_gate_preserves_sixteen_frame_thresholds() -> None:
    frames = []
    for index in range(16):
        fraction = index / 15.0
        frames.append(
            {
                "landmarks": [
                    {
                        "name": "right_wrist_yaw_link",
                        "world_xyz": [0.5 - 0.016 * fraction, 0.0, 0.0],
                        "image_projection": {
                            "available": True,
                            "u_px": 100.0 + 9.0 * fraction,
                            "v_px": 200.0,
                        },
                    }
                ]
            }
        )

    result = manipulation_effector_progress_report(
        {
            "task_target_world_xyz_m": [0.0, 0.0, 0.0],
            "controller_fk_sequence": frames,
        }
    )

    assert result["capability_gate_passed"] is True
    assert result["observed_frame_count"] == 16
    assert result["duration_scale"] == 1.0
    assert result["minimum_required_progress_m"] == 0.015
    assert result["minimum_required_projected_motion_px"] == 8.0


def test_contradiction_preserves_wam_score_but_caps_claims() -> None:
    score = {
        "generated_video_success_label_passed": True,
        "confidence": 0.91,
    }
    result = build_wam_isaac_disagreement_contract(
        trace_rows=[
            {
                "step_index": 3,
                "wam_predicted_stance_report": {"status": "upright"},
                "post_action_stance_report": {
                    "status": "unsafe",
                    "unsafe_stance_detected": True,
                },
            }
        ],
        primary_wam_score=score,
        independent_consistency_proven=True,
        accepted_calibration_authority=_accepted_frozen_benchmark(),
    )

    assert result["primary_wam_score"] == score
    assert result["primary_wam_score_preserved"] is True
    assert result["isaac_can_overwrite_or_terminate_wam_rollout"] is False
    assert result["disagreement_unresolved"] is True
    assert result["claim_ceiling"] == "uncalibrated_debug_evidence"
    assert result["task_success_claim_allowed"] is False
    assert result["rank_fidelity_claim_allowed"] is False
    assert result["categorical_contradictions"] == [
        {
            "step_index": 3,
            "kind": "stance_outcome_mismatch",
            "wam_status": "upright",
            "isaac_status": "unsafe",
        },
        {
            "step_index": 3,
            "kind": "wam_success_vs_isaac_unsafe_stance",
            "wam_status": "success",
            "isaac_status": "unsafe",
        },
    ]


def test_matching_diagnostics_still_require_independent_calibration() -> None:
    rows = [
        {
            "step_index": 1,
            "wam_predicted_stance_report": {"status": "upright"},
            "post_action_stance_report": {
                "status": "upright",
                "unsafe_stance_detected": False,
            },
        }
    ]
    uncalibrated = build_wam_isaac_disagreement_contract(
        trace_rows=rows,
        primary_wam_score={"generated_video_success_label_passed": True},
        independent_consistency_proven=True,
    )
    calibrated = build_wam_isaac_disagreement_contract(
        trace_rows=rows,
        primary_wam_score={"generated_video_success_label_passed": True},
        independent_consistency_proven=True,
        accepted_calibration_authority=_accepted_frozen_benchmark(),
    )

    assert uncalibrated["disagreement_unresolved"] is False
    assert uncalibrated["claim_ceiling"] == "uncalibrated_debug_evidence"
    assert uncalibrated["task_success_claim_allowed"] is False
    assert calibrated["claim_ceiling"] == "calibrated_sim_evaluation"
    assert calibrated["task_success_claim_allowed"] is True
    assert calibrated["rank_fidelity_claim_allowed"] is True


def test_missing_isaac_diagnostic_is_explicit_unresolved_evidence() -> None:
    result = build_wam_isaac_disagreement_contract(
        trace_rows=[{"step_index": 2}],
        primary_wam_score={"generated_video_success_label_passed": False},
        independent_consistency_proven=True,
        accepted_calibration_authority=_accepted_frozen_benchmark(),
    )

    assert result["missing_diagnostic_step_indices"] == [2]
    assert result["disagreement_unresolved"] is True
    assert "isaac_contradiction_check_incomplete" in result["blockers"]
    assert result["rank_fidelity_claim_allowed"] is False


def test_numeric_prefix_errors_are_aggregated_without_inventing_a_threshold() -> None:
    rows = [
        {
            "step_index": 1,
            "wam_predicted_stance_report": {"status": "upright"},
            "post_action_stance_report": {"status": "upright"},
            "wam_isaac_prefix_prediction_error": {
                "status": "measured",
                "mean_absolute_error_rad": 0.01,
                "maximum_absolute_error_rad": 0.05,
            },
        },
        {
            "step_index": 2,
            "wam_predicted_stance_report": {"status": "upright"},
            "post_action_stance_report": {"status": "upright"},
            "wam_isaac_prefix_prediction_error": {
                "status": "measured",
                "mean_absolute_error_rad": 0.03,
                "maximum_absolute_error_rad": 0.08,
            },
        },
    ]

    result = build_wam_isaac_disagreement_contract(
        trace_rows=rows,
        primary_wam_score={"generated_video_success_label_passed": False},
        independent_consistency_proven=False,
    )
    aggregate = result["numeric_prefix_prediction_error_aggregate"]

    assert aggregate["status"] == "measured"
    assert aggregate["measured_step_indices"] == [1, 2]
    assert aggregate["mean_of_step_mean_absolute_error_rad"] == 0.02
    assert aggregate["maximum_joint_absolute_error_rad"] == 0.08
    assert aggregate["cumulative_step_mean_absolute_error_rad"] == 0.04
    assert aggregate["threshold_status"] == "not_calibrated_numeric_error_is_advisory"
