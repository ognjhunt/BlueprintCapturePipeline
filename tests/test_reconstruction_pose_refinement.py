from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.reconstruction_pose_refinement import (
    POSE_REFINEMENT_EXECUTION_REQUEST_SCHEMA_VERSION,
    POSE_REFINEMENT_RESULT_SCHEMA_VERSION,
    build_pose_refinement_execution_request,
    build_pose_refinement_result,
)
from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    SupervisorContext,
    ToolRegistry,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


def _request() -> dict:
    return build_pose_refinement_execution_request(
        {
            "schema_version": POSE_REFINEMENT_EXECUTION_REQUEST_SCHEMA_VERSION,
            "stable_run_identity": "pose-refinement-1",
            "source_capture_digest": "sha256:" + "1" * 64,
            "capture_profile": "iphone_arkit_lidar",
            "reconstruction_dataset_digest": "sha256:" + "2" * 64,
            "frozen_split_digest": "sha256:" + "3" * 64,
            "camera_observation_digest": "sha256:" + "4" * 64,
            "camera_calibration_digest": "sha256:" + "5" * 64,
            "initial_pose_manifest_digest": "sha256:" + "6" * 64,
            "initial_pose_source": "verified_arkit_raw_contract_3_2",
            "method_id": "arkit_anchored_bundle_adjustment_v1",
            "drift_thresholds": {
                "maximum_translation_m": 0.05,
                "maximum_rotation_degrees": 2.0,
            },
            "thresholds_frozen_before_execution": True,
            "raw_arkit_poses_may_be_modified": False,
            "candidate_may_read_hidden_heldout": False,
            "coordinate_frame_declaration": {"frame": "arkit_world", "units": "meters"},
            "implementation_digest": "sha256:" + "7" * 64,
            "container_image_digest": "sha256:" + "8" * 64,
            "source_commit_sha": "9" * 40,
            "random_seed": 7,
            "resource_request": {"gpu_count": 0, "cpu_count": 4},
            "timeout_seconds": 600.0,
            "spend_cap_usd": 0.0,
            "authority_used": {"local_non_spend": True},
            "timestamp": "2026-07-30T21:00:00Z",
        }
    )


def _result(request: dict, *, max_translation: float = 0.02) -> dict:
    return build_pose_refinement_result(
        {
            "schema_version": POSE_REFINEMENT_RESULT_SCHEMA_VERSION,
            "source_capture_digest": request["source_capture_digest"],
            "pose_refinement_execution_request_digest": request[
                "pose_refinement_execution_request_digest"
            ],
            "frozen_split_digest": request["frozen_split_digest"],
            "camera_calibration_digest": request["camera_calibration_digest"],
            "initial_pose_manifest_digest": request["initial_pose_manifest_digest"],
            "implementation_digest": request["implementation_digest"],
            "container_image_digest": request["container_image_digest"],
            "status": "succeeded",
            "failure_code": None,
            "refined_pose_manifest_digest": "sha256:" + "a" * 64,
            "drift_metrics": {
                "maximum_translation_m": max_translation,
                "mean_translation_m": max_translation / 2,
                "maximum_rotation_degrees": 1.0,
                "mean_rotation_degrees": 0.5,
            },
            "registered_observation_ids": ["frame-1", "frame-2"],
            "rejected_observation_ids": [],
            "warnings": [],
            "blockers": [],
            "raw_arkit_poses_modified": False,
            "heldout_labels_included": False,
            "candidate_self_graded": False,
            "cost_usd": 0.0,
            "duration_seconds": 1.0,
            "proof_effect": "bounded_refined_trajectory_candidate_only",
            "claim_ceiling": "calibrated_camera_trajectory",
        }
    )


def test_registered_pose_refinement_enforces_frozen_arkit_drift_threshold(
    tmp_path: Path,
) -> None:
    request = _request()

    def runtime(*, request: dict, output_root: Path) -> dict:
        assert output_root.name == "pose_refinement"
        return _result(request)

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="pose-refinement-tool",
        customer_question="Refine poses while retaining ARKit anchors.",
        supervisor_output_dir=str(tmp_path / "run"),
        pose_refinement_request=request,
        pose_refiner=runtime,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[request["pose_refinement_execution_request_digest"]],
    ).to_mapping()
    binding = next(
        binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
        if binding.tool_id == "run_pose_refinement"
    )
    observation = binding.invoke(
        {
            "pose_refinement_execution_request_digest": request[
                "pose_refinement_execution_request_digest"
            ]
        }
    )

    assert observation["status"] == "completed"
    assert observation["typed_result"]["drift_within_frozen_thresholds"] is True
    assert observation["typed_result"]["raw_arkit_poses_modified"] is False
    assert observation["typed_result"]["heldout_labels_included"] is False
    assert observation["proof_effect"] == "none"


def test_registered_pose_refinement_refuses_success_above_frozen_drift_limit(
    tmp_path: Path,
) -> None:
    request = _request()

    def runtime(*, request: dict, output_root: Path) -> dict:
        return _result(request, max_translation=0.5)

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="pose-refinement-drift-refusal",
        customer_question="Do not accept excessive pose drift.",
        supervisor_output_dir=str(tmp_path / "run"),
        pose_refinement_request=request,
        pose_refiner=runtime,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[request["pose_refinement_execution_request_digest"]],
    ).to_mapping()
    binding = next(
        binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
        if binding.tool_id == "run_pose_refinement"
    )
    observation = binding.invoke(
        {
            "pose_refinement_execution_request_digest": request[
                "pose_refinement_execution_request_digest"
            ]
        }
    )

    assert observation["status"] == "refused"
    assert "drift_threshold_exceeded" in observation["typed_failure"]["reason"]
