from __future__ import annotations

import json
import sys
import subprocess
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import eval_ready_task_grounding as grounding
from blueprint_pipeline import oscar_cosmos_wam_evaluator as evaluator


_WAM_RUNTIME_ENV_VARS = (
    "BLUEPRINT_OSCAR_WAM_COMMAND",
    "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND",
    "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
    "BLUEPRINT_OSCAR_WAM_PROVIDER_USE_OBJECT_STORE",
    "BLUEPRINT_COSMOS_WAM_COMMAND",
    "BLUEPRINT_COSMOS_WAM_PROVIDER_COMMAND",
    "BLUEPRINT_COSMOS_WAM_CHECKPOINT",
    "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL",
    "BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER",
    "BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH",
    "BLUEPRINT_WAM_MODEL_CHECKPOINT",
)


def _clear_wam_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _WAM_RUNTIME_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_provider_output_zip(path: Path, *, include_runtime_result: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_path = path.parent / "oscar_generated_rollout_source.mp4"
    _write_test_video(video_path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(video_path, "oscar_generated_rollout.mp4")
        archive.writestr(
            "wam_provider_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "rollouts": [
                        {
                            "rollout_id": "oscar_wam_rollout_0001",
                            "policy_id": "oscar_wam_provider_runtime",
                            "scenario_eval_run_id": "run_1",
                            "generated_video_path": "/workspace/runtime_output/oscar_generated_rollout.mp4",
                        }
                    ],
                    "blockers": [],
                }
            ),
        )
        if include_runtime_result:
            archive.writestr(
                "wam_runtime_result.json",
                json.dumps(
                    {
                        "status": "completed",
                        "learned_wam_model_ran": True,
                        "truth_boundary": {
                            "generated_video_is_model_output": True,
                            "physical_robot_readiness_proven": False,
                            "deployment_readiness_proven": False,
                        },
                    }
                ),
            )
    return path


def _write_openvla_provider_smoke_job(job_dir: Path) -> Path:
    action = {"action_type": "waypoint", "waypoint": [0.36, -0.65, 0.79]}
    _write_json(
        job_dir / "openvla_policy_provider_smoke_summary.json",
        {
            "status": "completed",
            "openvla_model_executed": True,
            "openvla_policy_action_command_ran": True,
            "action": action,
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "openvla_provider_output" / "openvla_policy_provider_output.json",
        {
            "schema_version": "openvla_policy_provider_output.v1",
            "status": "completed",
            "openvla_model_executed": True,
            "openvla_model_loaded": True,
            "openvla_predict_action_invoked": True,
            "openvla_policy_action_command_ran": True,
            "action": action,
            "blockers": [],
        },
    )
    return job_dir


def _write_untrusted_openvla_provider_smoke_job(job_dir: Path) -> Path:
    action = {"action_type": "waypoint", "waypoint": [0.36, -0.65, 0.79]}
    _write_json(
        job_dir / "openvla_policy_provider_smoke_summary.json",
        {
            "status": "completed",
            "openvla_model_executed": True,
            "openvla_policy_action_command_ran": True,
            "action": action,
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "openvla_provider_output" / "openvla_policy_provider_output.json",
        {
            "status": "completed",
            "openvla_model_executed": True,
            "openvla_model_loaded": True,
            "openvla_predict_action_invoked": True,
            "openvla_policy_action_command_ran": True,
            "action": action,
            "blockers": [],
        },
    )
    return job_dir


def _write_test_video(path: Path, *, frame_count: int = 6, fps: float = 5.0) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    yy, xx = np.indices((48, 64))
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = (xx * 3 + index * 17) % 255
        frame[:, :, 1] = (yy * 5 + 90 + index * 11) % 255
        frame[:, :, 2] = ((xx + yy) * 2 + 40 + index * 23) % 255
        writer.write(frame)
    writer.release()


def _write_flat_dark_rollout(path: Path, *, frame_count: int = 10, fps: float = 5.0) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    first = np.zeros((48, 64, 3), dtype=np.uint8)
    first[:, :32] = (255, 0, 0)
    first[:, 32:] = (0, 255, 0)
    writer.write(first)
    for _ in range(max(0, frame_count - 1)):
        writer.write(np.full((48, 64, 3), 86, dtype=np.uint8))
    writer.release()


def _input_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "mujoco_job"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "matrix",
            "runs": [
                {
                    "scenario_eval_run_id": "run_1",
                    "task_id": "approach_target",
                    "spawn_id": "doorway",
                    "task_prompt": "Approach the target.",
                }
            ],
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "attempts",
            "attempts": [
                {
                    "attempt_id": "attempt_1",
                    "scenario_eval_run_id": "run_1",
                    "task_id": "approach_target",
                    "spawn_id": "doorway",
                    "success": True,
                }
            ],
        },
    )
    _write_jsonl(
        job_dir / "normalized_policy_action_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "normalized_action": {"action_type": "waypoint"},
                "policy_id": "endpoint_policy",
            }
        ],
    )
    _write_jsonl(
        job_dir / "g1_mujoco_locomotion_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "root_position": [0.0, 0.0, 0.79],
                "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        ],
    )
    _write_jsonl(
        job_dir / "g1_projected_skeleton_trace.jsonl",
        [
            {
                "schema_version": "blueprint.mujoco_g1.projected_upper_body_skeleton.v1",
                "status": "completed",
                "episode_id": "episode_1",
                "scenario_eval_run_id": "run_1",
                "projected_landmark_count": 2,
                "landmarks": [
                    {
                        "landmark_id": "left_hand",
                        "image_projection": {"available": True, "u_px": 24, "v_px": 24},
                    },
                    {
                        "landmark_id": "right_hand",
                        "image_projection": {"available": True, "u_px": 40, "v_px": 24},
                    },
                ],
                "segments": [{"from": "left_hand", "to": "right_hand"}],
            }
        ],
    )
    _write_json(
        job_dir / "g1_projected_skeleton_manifest.json",
        {
            "schema_version": "g1_projected_skeleton_trace_manifest.v1",
            "status": "completed",
            "row_count": 1,
            "projectable_row_count": 1,
            "claim_boundary": {
                "simulated_g1_arm_hand_state_available_for_wam_conditioning": True,
                "not_physical_robot_sensor_proof": True,
            },
        },
    )
    _write_json(
        job_dir / "review_video_selection_manifest.json",
        {
            "schema_version": "review_videos",
            "selected_review_videos": [
                {
                    "episode_id": "episode_1",
                    "camera": "head_pov",
                    "path": "/tmp/head_pov.mp4",
                    "egocentric_sensor_view": True,
                    "first_person_policy_observation_candidate": True,
                },
                {
                    "episode_id": "episode_1",
                    "camera": "third_person",
                    "path": "/tmp/review.mp4",
                    "egocentric_sensor_view": False,
                    "first_person_policy_observation_candidate": False,
                },
            ],
        },
    )
    return job_dir


def test_oscar_cosmos_wam_evaluator_writes_blocked_dry_run_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_wam_runtime_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HF_TOKEN_FILE", raising=False)
    monkeypatch.delenv("NGC_API_KEY_FILE", raising=False)
    input_job = _input_job(tmp_path)
    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_job",
        generated_at="now",
    )
    assert summary["status"] == "completed"
    assert summary["learned_wam_model_ran"] is False
    assert summary["http_endpoint_wrapper_available"] is True
    assert summary["model_http_wrapper_ready"] is False
    assert summary["real_model_endpoint_ready"] is False
    assert summary["real_model_endpoint_claim_blocked"] is True
    assert summary["forward_inverse_consistency_proven"] is False
    assert "blocked_missing_wam_runtime" in summary["blockers"]
    assert "blocked_missing_wam_model_checkpoint" in summary["blockers"]
    rollout_input = json.loads(
        (tmp_path / "wam_job" / "wam_rollout_input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert rollout_input["status"] == "ready_for_model"
    assert rollout_input["counts"]["wam_input_video_count"] == 1
    assert rollout_input["counts"]["diagnostic_review_video_count"] == 1
    assert rollout_input["counts"]["g1_projected_skeleton_row_count"] == 1
    assert rollout_input["counts"]["g1_projected_skeleton_projectable_row_count"] == 1
    assert rollout_input["inputs"]["g1_projected_skeleton_trace_jsonl"].endswith(
        "g1_projected_skeleton_trace.jsonl"
    )
    assert rollout_input["wam_input_videos"][0]["camera"] == "head_pov"
    assert rollout_input["diagnostic_review_videos"][0]["camera"] == "third_person"
    assert (
        rollout_input["wam_input_video_contract"][
            "third_person_overview_is_diagnostic_not_policy_observation"
        ]
        is True
    )
    required = [
        "wam_model_runtime_discovery.json",
        "wam_rollout_input_manifest.json",
        "wam_action_conditioning_manifest.json",
        "wam_generated_rollout_manifest.json",
        "wam_generated_rollout_results.json",
        "wam_generated_rollout_visual_smoke.json",
        "wam_consistency_checks.json",
        "wam_success_labels.json",
        "wam_policy_scorecard.json",
        "wam_policy_requery_manifest.json",
        "policy_requery_endpoint_readiness_manifest.json",
        "wam_policy_loop_manifest.json",
        "wam_manipulation_loop_readiness_manifest.json",
        "wam_evaluator_trace_binding.json",
        "wam_evaluator_truth_boundary.json",
        "policy_model_truth_boundary.json",
        "policy_model_endpoint_readiness_manifest.json",
        "policy_model_endpoint_creation_plan.json",
        "policy_model_endpoint_probe_results.json",
        "policy_cloud_gpu_setup_manifest.json",
        "local_model_source_tree_discovery.json",
    ]
    for filename in required:
        assert (tmp_path / "wam_job" / filename).is_file()
    action_conditioning = json.loads(
        (tmp_path / "wam_job" / "wam_action_conditioning_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "g1_projected_skeleton_trace.jsonl" in action_conditioning["conditioning_sources"]
    pose_encoding = action_conditioning["robot_pose_encoding"]
    assert pose_encoding["projected_g1_upper_body_skeleton_available"] is True
    assert pose_encoding["projected_skeleton_is_simulated_mujoco_state"] is True
    assert pose_encoding["projected_skeleton_is_not_physical_robot_proprioception"] is True
    assert pose_encoding["projected_skeleton_does_not_prove_wam_visual_usefulness"] is True
    trace_binding = json.loads(
        (tmp_path / "wam_job" / "wam_evaluator_trace_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert trace_binding["source_paths"]["g1_projected_skeleton_trace_jsonl"].endswith(
        "g1_projected_skeleton_trace.jsonl"
    )
    consistency = json.loads(
        (tmp_path / "wam_job" / "wam_consistency_checks.json").read_text(encoding="utf-8")
    )
    assert consistency["forward_inverse_consistency_proven"] is False
    assert consistency["action_conditioned_video_rollout_generated"] is False
    loop_manifest = json.loads(
        (tmp_path / "wam_job" / "wam_policy_loop_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert loop_manifest["actual_loop_mode"] == "offline_action_conditioned_wam_evaluator"
    assert loop_manifest["closed_loop_policy_wam_interaction"] is False
    assert loop_manifest["policy_observes_wam_generated_next_observation"] is False
    assert loop_manifest["closed_loop_manipulation_policy_wam_interaction_ready"] is False
    assert loop_manifest["wam_manipulation_loop_readiness_status"] == "blocked"
    assert (
        loop_manifest["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert loop_manifest["g1_robot_policy_selected_family"] is None
    assert loop_manifest["openvla_selected_as_g1_robot_policy"] is False


def test_oscar_cosmos_wam_evaluator_consumes_eval_ready_task_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_wam_runtime_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)
    _write_json(
        input_job / "raw" / "object_index.json",
        {
            "objects": [
                {
                    "object_id": "right_sink_handle_01",
                    "label": "right sink handle",
                    "source_prompt": "right sink handle",
                    "reference_crop": "object_index_artifacts/crops/right_sink_handle.png",
                    "all_crops": ["object_index_artifacts/crops/right_sink_handle.png"],
                    "keypoints": {"center": [322, 188]},
                    "mean_box_px": {"x": 302, "y": 168, "width": 40, "height": 40},
                    "mean_confidence": 0.91,
                }
            ]
        },
    )
    camera = input_job / "camera_calibration.json"
    _write_json(
        camera,
        {
            "fx": 800,
            "fy": 800,
            "cx": 320,
            "cy": 240,
            "width": 640,
            "height": 480,
            "reprojection_error_px": 1.2,
        },
    )
    scene = input_job / "kitchen.splat"
    scene.write_text("static 3dgs placeholder", encoding="utf-8")
    initial_frame = input_job / "initial.png"
    initial_frame.write_bytes(b"png")
    robot_model = input_job / "unitree_g1.xml"
    robot_model.write_text("<mujoco/>", encoding="utf-8")
    robot_state = input_job / "robot_state.json"
    _write_json(
        robot_state,
        {
            "right_end_effector_xyz": [0.005, -0.13, 2.0],
            "right_wrist_rotation_delta_deg": 22.0,
        },
    )
    grounding.build_eval_ready_task_grounding(
        capture_root=input_job,
        task_id="turn_on_sink_handle",
        task_text="turn on the sink right handle",
        target_label="right sink handle",
        scene_asset=scene,
        initial_frame=initial_frame,
        camera_calibration=camera,
        robot_model=robot_model,
        robot_state=robot_state,
        output_path=input_job / "eval_ready_task_grounding.json",
        articulated_handle_proxy=True,
    )

    grounded_job_dir = tmp_path / "wam_grounded_job"
    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=grounded_job_dir,
        generated_at="now",
    )

    assert summary["status"] == "completed"
    rollout_input = json.loads(
        (grounded_job_dir / "wam_rollout_input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert rollout_input["eval_ready_task_grounding"]["available"] is True
    assert rollout_input["eval_ready_task_grounding"]["learned_rollout_request_ready"] is True
    assert rollout_input["inputs"]["robot_fk_projected_skeleton_trace_jsonl"].endswith(
        "robot_fk_projected_skeleton_trace.jsonl"
    )
    assert rollout_input["task_prompts"][0]["selected_task_target"]["object_id"] == (
        "right_sink_handle_01"
    )
    assert "sink right handle" in rollout_input["task_prompts"][0]["target_prompts"]
    action_conditioning = json.loads(
        (grounded_job_dir / "wam_action_conditioning_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "robot_fk_projected_skeleton_trace.jsonl" in action_conditioning["conditioning_sources"]
    assert action_conditioning["robot_pose_encoding"]["generic_robot_fk_projection_available"] is True
    scorecard = json.loads(
        (grounded_job_dir / "wam_policy_scorecard.json").read_text(
            encoding="utf-8"
        )
    )
    assert scorecard["eval_ready_task_grounding_used"] is True
    assert scorecard["handle_proxy_state"] == "on_candidate"
    ledger = json.loads(
        (grounded_job_dir / "wam_prediction_outcome_correlation_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["status"] == "awaiting_real_world_outcomes"
    trace_binding = json.loads(
        (grounded_job_dir / "wam_evaluator_trace_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert trace_binding["source_paths"]["eval_ready_task_grounding"].endswith(
        "eval_ready_task_grounding.json"
    )
    loop_manifest = json.loads(
        (grounded_job_dir / "wam_policy_loop_manifest.json").read_text(encoding="utf-8")
    )
    assert loop_manifest["wam_rollout_selected_as_g1_robot_policy"] is False
    assert loop_manifest["unitree_hand_policy_required_for_g1_manipulation"] is True
    manipulation_loop = json.loads(
        (grounded_job_dir / "wam_manipulation_loop_readiness_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manipulation_loop["status"] == "blocked"
    assert manipulation_loop["manipulation_attempt_count"] == 0
    assert manipulation_loop["manipulation_contact_action_count"] == 0
    assert (
        manipulation_loop["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert manipulation_loop["g1_robot_policy_selected_family"] is None
    assert manipulation_loop["openvla_selected_as_g1_robot_policy"] is False
    assert manipulation_loop["wam_rollout_selected_as_g1_robot_policy"] is False
    assert manipulation_loop["unitree_hand_policy_required_for_g1_manipulation"] is True
    assert manipulation_loop["claim_boundary"][
        "wam_evaluator_is_test_bench_not_robot_manipulation_policy"
    ] is True
    assert manipulation_loop["claim_boundary"][
        "openvla_policy_is_not_selected_g1_robot_policy"
    ] is True
    assert manipulation_loop["claim_boundary"][
        "wam_rollout_is_not_selected_g1_robot_policy"
    ] is True
    requery_readiness = json.loads(
        (grounded_job_dir / "policy_requery_endpoint_readiness_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert requery_readiness["status"] == "blocked"
    assert requery_readiness["live_policy_requery_endpoint_ready"] is False
    assert "blocked_missing_live_policy_requery_endpoint_env_or_auth" in requery_readiness[
        "blockers"
    ]
    assert (
        requery_readiness["claim_boundary"][
            "source_endpoint_proof_is_not_current_live_requery_proof"
        ]
        is True
    )
    assert "blocked_missing_manipulation_contact_task_attempts" in manipulation_loop["blockers"]
    assert "blocked_missing_unitree_g1_hand_manipulation_policy" in manipulation_loop[
        "blockers"
    ]
    assert "blocked_missing_real_vla_or_unitree_hand_manipulation_policy" in manipulation_loop[
        "blockers"
    ]
    truth = json.loads(
        (grounded_job_dir / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["learned_wam_model_ran"] is False
    assert truth["http_endpoint_wrapper_available"] is True
    assert truth["model_http_wrapper_ready"] is False
    assert truth["real_model_endpoint_ready"] is False
    assert truth["model_endpoint_command_probe_passed"] is False
    assert truth["real_model_endpoint_claim_blocked"] is True
    assert truth["closed_loop_manipulation_policy_wam_interaction_ready"] is False
    assert truth["wam_manipulation_loop_readiness_status"] == "blocked"
    assert (
        truth["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert truth["g1_robot_policy_selected_family"] is None
    assert truth["openvla_selected_as_g1_robot_policy"] is False
    assert truth["wam_rollout_selected_as_g1_robot_policy"] is False
    assert truth["unitree_hand_policy_required_for_g1_manipulation"] is True
    assert truth["what_is_needed_to_make_false_flags_true"]["real_model_endpoint_ready"]
    assert "An HTTP endpoint without a runnable command" in " ".join(
        truth["why_cannot_just_create_endpoints"]
    )
    policy_truth = json.loads(
        (grounded_job_dir / "policy_model_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy_truth["schema_version"] == "policy_model_truth_boundary.v1"
    assert policy_truth["replaceable_model_adapter_boundary"] is True
    readiness = json.loads(
        (grounded_job_dir / "policy_model_endpoint_readiness_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["http_endpoint_wrapper_available"] is True
    assert readiness["real_model_ready_candidate_count"] == 0
    assert readiness["claim_boundary"]["endpoint_creation_is_not_model_execution_proof"] is True
    oscar = next(row for row in readiness["candidates"] if row["candidate_id"] == "oscar_wam")
    assert oscar["endpoint_wrapper_can_be_created"] is False
    assert "set_BLUEPRINT_OSCAR_WAM_COMMAND_to_runnable_adapter_command" in oscar[
        "what_is_needed_to_make_true"
    ]
    creation_plan = json.loads(
        (grounded_job_dir / "policy_model_endpoint_creation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert creation_plan["http_wrapper_binary_available"] is True
    assert creation_plan["endpoint_creation_modes"][0]["mode"] == "reference_endpoint_wrapper"
    assert creation_plan["endpoint_wrapper_missing_command_candidate_count"] >= 1
    assert creation_plan["can_create_real_model_endpoint_now"] is False
    assert creation_plan["claim_boundary"]["http_endpoint_creation_is_not_model_execution_proof"] is True
    assert "runnable adapter command" in " ".join(
        creation_plan["minimum_user_supplied_inputs"]
    )
    probe = json.loads(
        (grounded_job_dir / "policy_model_endpoint_probe_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert probe["status"] == "blocked"
    assert probe["probe_attempted_candidate_count"] == 0
    assert probe["can_claim_real_model_endpoint_after_probe"] is False
    assert "blocked_model_command_not_available" in probe["blockers"]
    assert "wrapped command to run" in " ".join(probe["why_cannot_just_create_endpoints"])
    source_discovery = json.loads(
        (grounded_job_dir / "local_model_source_tree_discovery.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        source_discovery["claim_boundary"]["source_tree_present_is_not_model_runtime_proof"]
        is True
    )


def test_oscar_cosmos_wam_evaluator_inherits_unitree_endpoint_hand_policy_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    input_job = _input_job(tmp_path)
    _write_json(
        input_job / "scenario_eval_matrix.json",
        {
            "schema_version": "matrix",
            "runs": [
                {
                    "scenario_eval_run_id": "run_1",
                    "task_id": "contact_or_push_light_object",
                    "spawn_id": "doorway",
                    "task_prompt": "Push the object.",
                }
            ],
        },
    )
    _write_json(
        input_job / "normalized_attempt_trace.json",
        {
            "schema_version": "attempts",
            "attempts": [
                {
                    "attempt_id": "attempt_1",
                    "scenario_eval_run_id": "run_1",
                    "task_id": "contact_or_push_light_object",
                    "spawn_id": "doorway",
                    "success": False,
                }
            ],
        },
    )
    _write_jsonl(
        input_job / "normalized_policy_action_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "normalized_action": {"action_type": "manipulation_contact"},
                "policy_id": "unitree_unifolm_vla_policy",
            }
        ],
    )
    _write_json(
        input_job / "manipulation_endpoint_task_report.json",
        {
            "schema_version": "manipulation_endpoint_task_report.v1",
            "manipulation_endpoint_path_used": True,
            "unitree_endpoint_hand_policy_used": True,
            "unitree_endpoint_fresh_policy_action_command_ran": True,
            "unitree_endpoint_provider_output_replay_used": False,
            "blockers": ["blocked_manipulation_contact_not_validated"],
        },
    )
    _write_json(
        input_job / "mujoco_g1_wam_vla_policy_endpoint_eval_summary.json",
        {
            "schema_version": "summary",
            "endpoint_policy_used": True,
            "fixture_policy_used": False,
            "unitree_endpoint_hand_policy_used": True,
            "unitree_endpoint_fresh_policy_action_command_ran": True,
            "unitree_endpoint_provider_output_replay_used": False,
        },
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_unitree_endpoint_status_job",
        generated_at="now",
    )

    assert summary["unitree_hand_manipulation_policy_used"] is True
    assert summary["g1_robot_policy_selected_family"] == "unitree_native_hand_manipulation_policy"
    assert summary["openvla_selected_as_g1_robot_policy"] is False
    assert summary["wam_rollout_selected_as_g1_robot_policy"] is False
    readiness = json.loads(
        (
            tmp_path
            / "wam_unitree_endpoint_status_job"
            / "wam_manipulation_loop_readiness_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert readiness["source_unitree_endpoint_hand_policy_used"] is True
    assert readiness["source_unitree_endpoint_fresh_policy_action_command_ran"] is True
    assert readiness["source_unitree_endpoint_provider_output_replay_used"] is False
    assert readiness["unitree_hand_manipulation_policy_scope"] == "endpoint_action_command"
    assert "blocked_missing_unitree_g1_hand_manipulation_policy" not in readiness["blockers"]
    assert "blocked_missing_wam_generated_rollout_for_manipulation_loop" in readiness["blockers"]
    requery_readiness = json.loads(
        (
            tmp_path
            / "wam_unitree_endpoint_status_job"
            / "policy_requery_endpoint_readiness_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert requery_readiness["source_unitree_endpoint_hand_policy_used"] is True
    assert requery_readiness["source_endpoint_proof_is_not_current_live_endpoint"] is True
    assert requery_readiness["live_policy_requery_endpoint_ready"] is False
    assert "source_endpoint_proof_exists_but_endpoint_not_currently_live_for_requery" in (
        requery_readiness["blockers"]
    )


def test_policy_requery_endpoint_readiness_requires_health_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_job = tmp_path / "source_job"
    input_job.mkdir()
    _write_json(
        input_job / "mujoco_g1_wam_vla_policy_endpoint_eval_summary.json",
        {
            "schema_version": "summary",
            "endpoint_policy_used": True,
            "fixture_policy_used": False,
            "unitree_endpoint_hand_policy_used": True,
            "unitree_endpoint_fresh_policy_action_command_ran": True,
        },
    )
    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token", encoding="utf-8")
    monkeypatch.setenv("TEAM_POLICY_ENDPOINT_URL", "http://127.0.0.1:8794/policy/action")
    monkeypatch.setenv("TEAM_POLICY_AUTH_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("VLA_POLICY_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("VLA_POLICY_AUTH_TOKEN_FILE", raising=False)
    monkeypatch.setattr(
        evaluator,
        "_probe_policy_requery_endpoint_health",
        lambda endpoint_url: {
            "status": "completed",
            "health_url": endpoint_url.replace("/policy/action", "/health"),
            "http_status": 200,
            "health_payload_redacted": {"status": "ready"},
            "raw_token_values_persisted": False,
            "raw_token_hashes_persisted": False,
        },
    )

    manifest = evaluator._build_policy_requery_endpoint_readiness_manifest(
        generated_at="now",
        input_dir=input_job,
        visual_rollout_useful=False,
        single_step_policy_requery_frame_useful=True,
        visual_smoke_status="failed_visual_quality_smoke",
    )

    assert manifest["status"] == "ready_for_policy_requery"
    assert manifest["policy_requery_endpoint_env_auth_configured"] is True
    assert manifest["live_policy_requery_endpoint_ready"] is True
    assert manifest["endpoint_candidates"][1]["health_probe"]["status"] == "completed"
    assert manifest["generated_rollout_visually_useful_for_policy_requery"] is True
    assert manifest["full_rollout_visually_useful_for_success_review"] is False
    assert manifest["blockers"] == []
    assert manifest["claim_boundary"]["raw_credentials_written_to_artifacts"] is False


def test_single_step_policy_requery_visual_candidate_allows_first_scene_frame_only() -> None:
    candidate = evaluator._single_step_policy_requery_visual_candidate(
        {
            "status": "failed_visual_quality_smoke",
            "blockers": ["generated_rollout_later_frames_lost_scene_structure"],
            "claim_boundary": {"visual_rollout_useful_for_task_success_review": False},
            "rollouts": [
                {
                    "visual_quality_flags": {
                        "first_frame_preserves_source_scene": True,
                        "later_frames_lost_scene_structure": True,
                    },
                    "sampled_frames": [
                        {
                            "frame_index": 0,
                            "edge_density": 0.026,
                            "luma_range": 254,
                        }
                    ],
                }
            ],
        }
    )

    assert candidate["status"] == "ready_for_single_step_policy_requery"
    assert candidate["single_step_policy_requery_frame_useful"] is True
    assert candidate["full_rollout_visually_useful_for_success_review"] is False
    assert candidate["blockers"] == []


def test_oscar_cosmos_wam_evaluator_blocks_third_person_only_wam_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)
    _write_json(
        input_job / "review_video_selection_manifest.json",
        {
            "schema_version": "review_videos",
            "selected_review_videos": [
                {
                    "episode_id": "episode_1",
                    "camera": "third_person",
                    "path": "/tmp/review.mp4",
                    "egocentric_sensor_view": False,
                }
            ],
        },
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "third_person_only_wam_job",
        generated_at="now",
    )

    assert summary["status"] == "blocked"
    assert "blocked_missing_egocentric_wam_input_video" in summary["blockers"]
    rollout_input = json.loads(
        (
            tmp_path
            / "third_person_only_wam_job"
            / "wam_rollout_input_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert rollout_input["status"] == "blocked_missing_egocentric_wam_input_video"
    assert rollout_input["counts"]["wam_input_video_count"] == 0
    assert rollout_input["counts"]["diagnostic_review_video_count"] == 1
    assert rollout_input["diagnostic_review_videos"][0]["camera"] == "third_person"


def test_oscar_cosmos_wam_evaluator_blocks_manipulation_loop_without_real_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)
    _write_json(
        input_job / "scenario_eval_matrix.json",
        {
            "schema_version": "matrix",
            "runs": [
                {
                    "scenario_eval_run_id": "run_1",
                    "task_id": "contact_or_push_light_object",
                    "spawn_id": "doorway",
                    "task_prompt": "Make contact with the light object.",
                }
            ],
        },
    )
    _write_json(
        input_job / "normalized_attempt_trace.json",
        {
            "schema_version": "attempts",
            "attempts": [
                {
                    "attempt_id": "attempt_1",
                    "scenario_eval_run_id": "run_1",
                    "task_id": "contact_or_push_light_object",
                    "spawn_id": "doorway",
                    "success": False,
                }
            ],
        },
    )
    _write_jsonl(
        input_job / "normalized_policy_action_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "normalized_action": {"action_type": "manipulation_contact"},
                "policy_id": "endpoint_policy",
            }
        ],
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_manipulation_blocked_job",
        generated_at="now",
    )

    assert summary["closed_loop_manipulation_policy_wam_interaction_ready"] is False
    assert (
        summary["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert summary["g1_robot_policy_selected_family"] is None
    assert summary["openvla_selected_as_g1_robot_policy"] is False
    assert summary["wam_rollout_selected_as_g1_robot_policy"] is False
    assert summary["unitree_hand_manipulation_policy_used"] is False
    assert summary["unitree_hand_policy_required_for_g1_manipulation"] is True
    manipulation_loop = json.loads(
        (
            tmp_path
            / "wam_manipulation_blocked_job"
            / "wam_manipulation_loop_readiness_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manipulation_loop["manipulation_attempt_count"] == 1
    assert manipulation_loop["manipulation_contact_action_count"] == 1
    assert (
        manipulation_loop["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert manipulation_loop["g1_robot_policy_selected_family"] is None
    assert manipulation_loop["openvla_selected_as_g1_robot_policy"] is False
    assert manipulation_loop["wam_rollout_selected_as_g1_robot_policy"] is False
    assert manipulation_loop["unitree_hand_manipulation_policy_used"] is False
    assert manipulation_loop["unitree_hand_policy_required_for_g1_manipulation"] is True
    assert "blocked_missing_manipulation_contact_task_attempts" not in manipulation_loop[
        "blockers"
    ]
    assert "blocked_no_manipulation_contact_actions_in_endpoint_trace" not in manipulation_loop[
        "blockers"
    ]
    assert "blocked_missing_unitree_g1_hand_manipulation_policy" in manipulation_loop[
        "blockers"
    ]
    assert "blocked_missing_real_vla_or_unitree_hand_manipulation_policy" in manipulation_loop[
        "blockers"
    ]
    assert "blocked_missing_wam_generated_rollout_for_manipulation_loop" in manipulation_loop[
        "blockers"
    ]


def test_oscar_cosmos_wam_evaluator_imports_openvla_provider_smoke_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    provider_job = _write_openvla_provider_smoke_job(
        tmp_path / "openvla_policy_provider_smoke_20260622T092432Z"
    )
    monkeypatch.setenv("BLUEPRINT_OPENVLA_PROVIDER_SMOKE_JOB_DIR", str(provider_job))
    input_job = _input_job(tmp_path)

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_openvla_bound_job",
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["wam_rollout_model_ran"] is False
    assert summary["oscar_cosmos_openvla_unitree_model_ran"] is False
    assert summary["policy_action_model_command_ran"] is False
    assert summary["openvla_policy_action_command_ran"] is False
    assert summary["openvla_policy_action_command_imported"] is True
    assert summary["endpoint_closed_loop_policy_proven"] is False
    assert summary["unitree_g1_dexterous_manipulation_proven"] is False

    proof = json.loads(
        (tmp_path / "wam_openvla_bound_job" / "openvla_provider_smoke_proof.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["status"] == "completed"
    assert proof["action"]["action_type"] == "waypoint"
    assert proof["model_execution_scope"] == (
        "provider_smoke_action_prediction_not_closed_loop_robot_control"
    )
    truth = json.loads(
        (tmp_path / "wam_openvla_bound_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["openvla_provider_smoke_model_executed"] is True
    assert truth["openvla_provider_smoke_imported"] is True
    assert truth["openvla_policy_action_command_imported"] is True
    assert truth["openvla_policy_action_command_ran"] is False
    assert truth["endpoint_closed_loop_policy_proven"] is False
    matrix = json.loads(
        (tmp_path / "wam_openvla_bound_job" / "policy_model_candidate_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    openvla_candidate = next(
        row for row in matrix["candidates"] if row["id"] == "openvla_policy"
    )
    assert openvla_candidate["provider_smoke_completed"] is True
    assert openvla_candidate["openvla_policy_action_command_imported"] is True
    assert openvla_candidate["openvla_policy_action_command_ran"] is False


def test_oscar_cosmos_wam_evaluator_rejects_schema_less_openvla_smoke_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.delenv("BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_SMOKE_JOB_DIR", raising=False)
    provider_job = _write_untrusted_openvla_provider_smoke_job(
        tmp_path / "openvla_policy_provider_smoke_schema_less"
    )
    monkeypatch.setenv("BLUEPRINT_OPENVLA_PROVIDER_SMOKE_JOB_DIR", str(provider_job))
    monkeypatch.setattr(evaluator, "_repo_root", lambda: tmp_path)
    input_job = _input_job(tmp_path)

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_openvla_untrusted_job",
        generated_at="now",
    )

    assert summary["policy_action_model_command_ran"] is False
    assert summary["openvla_policy_action_command_ran"] is False
    assert summary["oscar_cosmos_openvla_unitree_model_ran"] is False
    proof = json.loads(
        (
            tmp_path
            / "wam_openvla_untrusted_job"
            / "openvla_provider_smoke_proof.json"
        ).read_text(encoding="utf-8")
    )
    assert proof["status"] == "blocked"
    assert "openvla_provider_output_missing_trusted_runtime_proof" in proof["blockers"]


def test_oscar_cosmos_wam_evaluator_imports_unitree_unifolm_provider_smoke_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_OPENVLA_PROVIDER_SMOKE_JOB_DIR", raising=False)
    provider_job = tmp_path / "unitree_unifolm_policy_provider_smoke_20260622T131500Z"
    provider_job.mkdir()
    _write_json(
        provider_job / "unitree_unifolm_policy_provider_smoke_summary.json",
        {
            "schema_version": "unitree_unifolm_policy_provider_smoke.v1",
            "status": "completed",
            "mode": "vla",
            "unitree_unifolm_model_executed": True,
            "unitree_unifolm_policy_action_command_ran": True,
            "policy_action_model_command_ran": True,
            "action": {"action_type": "manipulation_contact"},
            "blockers": [],
        },
    )
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_SMOKE_JOB_DIR", str(provider_job))
    input_job = _input_job(tmp_path)

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_unitree_unifolm_bound_job",
        model_candidates=("oscar_wam", "unitree_unifolm_vla_policy"),
        generated_at="now",
    )

    assert summary["oscar_cosmos_openvla_unitree_model_ran"] is True
    assert summary["policy_action_model_command_ran"] is True
    assert summary["unitree_policy_action_command_ran"] is True
    assert summary["unitree_unifolm_policy_action_command_ran"] is True
    assert summary["openvla_policy_action_command_ran"] is False
    proof = json.loads(
        (
            tmp_path
            / "wam_unitree_unifolm_bound_job"
            / "unitree_unifolm_provider_smoke_proof.json"
        ).read_text(encoding="utf-8")
    )
    assert proof["status"] == "completed"
    assert proof["action"]["action_type"] == "manipulation_contact"
    openvla_proof = json.loads(
        (
            tmp_path
            / "wam_unitree_unifolm_bound_job"
            / "openvla_provider_smoke_proof.json"
        ).read_text(encoding="utf-8")
    )
    assert openvla_proof["status"] == "skipped"
    truth = json.loads(
        (
            tmp_path
            / "wam_unitree_unifolm_bound_job"
            / "wam_evaluator_truth_boundary.json"
        ).read_text(encoding="utf-8")
    )
    assert truth["unitree_policy_action_command_ran"] is True
    assert truth["openvla_policy_action_command_ran"] is False
    assert truth["unitree_g1_dexterous_manipulation_proven"] is False


def test_policy_model_candidate_matrix_names_unitree_unifolm_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)

    evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_unifolm_matrix_job",
        generated_at="now",
    )

    matrix = json.loads(
        (tmp_path / "wam_unifolm_matrix_job" / "policy_model_candidate_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    openvla_proof = json.loads(
        (tmp_path / "wam_unifolm_matrix_job" / "openvla_provider_smoke_proof.json").read_text(
            encoding="utf-8"
        )
    )
    assert openvla_proof["status"] == "skipped"
    assert openvla_proof["openvla_model_executed"] is False
    by_id = {row["id"]: row for row in matrix["candidates"]}
    assert by_id["unitree_unifolm_vla_policy"]["command_env"] == (
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND"
    )
    assert by_id["unitree_unifolm_vla_policy"]["checkpoint_env"] == (
        "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT"
    )
    assert by_id["unitree_unifolm_vla_policy"]["vlm_checkpoint_env"] == (
        "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT"
    )
    assert "unitreerobotics/UnifoLM-VLA-Base:checkpoints/pytorch_model.pt" in by_id[
        "unitree_unifolm_vla_policy"
    ]["known_public_checkpoint_files"]
    assert "unitreerobotics/UnifoLM-VLM-Base:<model repository root>" in by_id[
        "unitree_unifolm_vla_policy"
    ]["known_public_checkpoint_files"]
    assert by_id["unitree_unifolm_vla_policy"]["claim_boundary"][
        "checkpoint_presence_is_not_endpoint_execution"
    ] is True
    assert by_id["unitree_unifolm_wma_policy"]["command_env"] == (
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_COMMAND"
    )
    assert by_id["unitree_unifolm_wma_policy"]["checkpoint_env"] == (
        "BLUEPRINT_UNITREE_UNIFOLM_WMA_CHECKPOINT"
    )
    assert "unitreerobotics/UnifoLM-WMA-0-Dual:unifolm_wma_dual.ckpt" in by_id[
        "unitree_unifolm_wma_policy"
    ]["known_public_checkpoint_files"]
    assert by_id["unitree_unifolm_wma_policy"]["claim_boundary"][
        "world_model_action_stack_is_not_automatically_endpoint_ready"
    ] is True

    creation_plan = json.loads(
        (
            tmp_path
            / "wam_unifolm_matrix_job"
            / "policy_model_endpoint_creation_plan.json"
        ).read_text(encoding="utf-8")
    )
    readiness = json.loads(
        (
            tmp_path
            / "wam_unifolm_matrix_job"
            / "policy_model_endpoint_readiness_manifest.json"
        ).read_text(encoding="utf-8")
    )
    vla_readiness = next(
        row
        for row in readiness["candidates"]
        if row["candidate_id"] == "unitree_unifolm_vla_policy"
    )
    assert "BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT" in vla_readiness[
        "missing_required_checkpoint_envs"
    ]
    assert "set_BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT_to_local_checkpoint_path" in (
        vla_readiness["what_is_needed_to_make_true"]
    )
    assert creation_plan["unitree_unifolm_policy_ready_candidate_count"] == 0
    assert creation_plan["readiness_layer_summary"][
        "unitree_unifolm_policy_ready_candidate_count"
    ] == 0


def test_unitree_unifolm_vla_readiness_accepts_provider_checkpoint_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND", sys.executable)
    monkeypatch.delenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT", raising=False)
    policy_checkpoint = tmp_path / "UnifoLM-VLA-Base" / "checkpoints" / "pytorch_model.pt"
    policy_checkpoint.parent.mkdir(parents=True)
    policy_checkpoint.write_bytes(b"fake checkpoint")
    vlm_checkpoint = tmp_path / "UnifoLM-VLM-Base"
    vlm_checkpoint.mkdir()
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT", str(policy_checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT", str(vlm_checkpoint))

    readiness = evaluator.build_policy_model_endpoint_readiness_manifest(
        generated_at="now",
        candidates=("unitree_unifolm_vla_policy",),
    )

    row = readiness["candidates"][0]
    assert row["configured_checkpoint_env"] == "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT"
    assert row["checkpoint_requirement_satisfied"] is True
    checkpoint_status = row["required_checkpoint_envs"][0]
    assert checkpoint_status["env"] == "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT"
    assert checkpoint_status["configured_env"] == "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT"
    assert checkpoint_status["configured"] is True
    assert checkpoint_status["exists"] is True
    assert "BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT" in checkpoint_status[
        "accepted_envs"
    ]
    assert "BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT" not in row[
        "missing_required_checkpoint_envs"
    ]


def test_oscar_cosmos_wam_evaluator_imports_source_unitree_controller_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    input_job = _input_job(tmp_path)
    _write_json(
        input_job / "controller_truth_boundary.json",
        {
            "schema_version": "controller_truth_boundary.v1",
            "controller_backend": "unitree_rl_gym",
            "official_unitree_controller_used": True,
            "official_policy_execution_proven": True,
            "balanced_walking_controller_proven": True,
            "realistic_navigation_policy_used": True,
            "realistic_navigation_policy_used_for_endpoint_rollouts": True,
            "freejoint_proxy_used": False,
            "unitree_lower_body_locomotion_policy_used": True,
            "unitree_locomotion_policy_kind": "unitree_rl_gym_same_scene_lower_body_policy",
            "unitree_locomotion_policy_checkpoint_path": "/tmp/motion.pt",
            "unitree_hand_manipulation_policy_used": False,
            "unitree_lerobot_or_isaaclab_manipulation_policy_used": False,
        },
    )
    _write_json(
        input_job / "same_scene_unitree_controller_backend_manifest.json",
        {
            "schema_version": "same_scene_unitree_rl_gym_controller_backend.v1",
            "status": "completed",
            "controller_backend": "unitree_rl_gym",
            "backend_id": "unitree_rl_gym_same_scene_lower_body_policy",
            "official_unitree_controller_used": True,
            "balanced_walking_controller_proven": True,
            "freejoint_proxy_used_for_endpoint_rollouts": False,
            "realistic_navigation_policy_used_for_endpoint_rollouts": True,
        },
    )
    _write_jsonl(
        input_job / "g1_mujoco_locomotion_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "controller_backend": "unitree_rl_gym",
                "official_unitree_controller_used": True,
                "freejoint_proxy_used": False,
                "fall_detected": False,
            }
        ],
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_unitree_bound_job",
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["oscar_cosmos_openvla_unitree_model_ran"] is True
    assert summary["unitree_locomotion_policy_ran"] is True
    assert summary["official_unitree_controller_used"] is True
    assert summary["official_unitree_controller_proven"] is True
    assert summary["balanced_walking_controller_proven"] is True
    assert summary["freejoint_proxy_used"] is False
    assert summary["unitree_policy_action_command_ran"] is False
    assert summary["unitree_g1_dexterous_manipulation_proven"] is False
    proof = json.loads(
        (tmp_path / "wam_unitree_bound_job" / "source_unitree_controller_proof.json").read_text(
            encoding="utf-8"
        )
    )
    assert proof["unitree_locomotion_policy_ran"] is True
    assert proof["unitree_g1_dexterous_manipulation_proven"] is False
    truth = json.loads(
        (tmp_path / "wam_unitree_bound_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["official_unitree_controller_proven"] is True
    assert truth["unitree_locomotion_policy_kind"] == (
        "unitree_rl_gym_same_scene_lower_body_policy"
    )
    assert truth["unitree_g1_dexterous_manipulation_proven"] is False


def test_oscar_cosmos_wam_evaluator_does_not_count_unitree_trace_without_trusted_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    input_job = _input_job(tmp_path)
    _write_json(
        input_job / "controller_truth_boundary.json",
        {
            "schema_version": "controller_truth_boundary.v1",
            "controller_backend": "unitree_rl_gym",
            "official_unitree_controller_used": True,
            "official_policy_execution_proven": True,
            "blockers": ["controller_truth_incomplete"],
        },
    )
    _write_json(
        input_job / "same_scene_unitree_controller_backend_manifest.json",
        {
            "schema_version": "same_scene_unitree_rl_gym_controller_backend.v1",
            "status": "blocked",
            "controller_backend": "unitree_rl_gym",
            "backend_id": "unitree_rl_gym_same_scene_lower_body_policy",
            "official_unitree_controller_used": True,
            "blockers": ["same_scene_controller_not_completed"],
        },
    )
    _write_jsonl(
        input_job / "g1_mujoco_locomotion_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "controller_backend": "unitree_rl_gym",
                "official_unitree_controller_used": True,
                "freejoint_proxy_used": False,
                "fall_detected": False,
            }
        ],
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_unitree_untrusted_trace_job",
        generated_at="now",
    )

    assert summary["unitree_locomotion_policy_ran"] is False
    assert summary["official_unitree_controller_used"] is False
    assert summary["oscar_cosmos_openvla_unitree_model_ran"] is False
    proof = json.loads(
        (
            tmp_path
            / "wam_unitree_untrusted_trace_job"
            / "source_unitree_controller_proof.json"
        ).read_text(encoding="utf-8")
    )
    assert proof["trusted_artifact_checks"]["controller_truth_boundary_trusted"] is False
    assert proof["trusted_artifact_checks"]["trace_rows_are_supporting_evidence_only"] is True
    assert proof["trace_rows_with_unitree_controller"] == 1
    assert proof["unitree_locomotion_policy_ran"] is False


def test_oscar_cosmos_wam_evaluator_reports_file_auth_without_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    hf_file = tmp_path / "hf-token"
    ngc_file = tmp_path / "ngc-token"
    hf_file.write_text("hf-secret-value\n", encoding="utf-8")
    ngc_file.write_text("ngc-secret-value\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_FILE", str(hf_file))
    monkeypatch.setenv("NGC_API_KEY_FILE", str(ngc_file))

    discovery = evaluator.discover_wam_model_runtimes(generated_at="now")

    assert discovery["model_access_secret_status"]["huggingface"]["auth_ready"] is True
    assert discovery["model_access_secret_status"]["ngc"]["auth_ready"] is True
    serialized = json.dumps(discovery, sort_keys=True)
    assert "hf-secret-value" not in serialized
    assert "ngc-secret-value" not in serialized


def test_oscar_cosmos_wam_evaluator_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert evaluator._repo_root().name == "BlueprintCapturePipeline"
    assert evaluator._timestamp().endswith("Z")
    assert evaluator._string_list("one") == ["one"]
    assert evaluator._load_json(tmp_path / "missing.json") == {}
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('\n{"ok": 1}\n[]\n', encoding="utf-8")
    assert evaluator._read_jsonl(rows_path) == [{"ok": 1}]
    assert evaluator._read_jsonl(tmp_path / "missing.jsonl") == []
    assert evaluator._command_available("'unterminated") is False
    assert evaluator._command_available("   ") is False
    assert evaluator._relative_or_absolute(tmp_path / "a", tmp_path) == "a"
    assert evaluator._relative_or_absolute(Path("/tmp/outside-blueprint-test"), tmp_path).startswith("/")

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    assert evaluator._checkpoint_like_files(checkpoint)["checkpoint_files_found"][0]["relative_path"] == checkpoint.name
    oscar_checkpoint = tmp_path / "__0_0.distcp"
    oscar_checkpoint.write_bytes(b"x" * (51 * 1024 * 1024))
    oscar_scan = evaluator._checkpoint_like_files(oscar_checkpoint)
    assert oscar_scan["checkpoint_files_found"][0]["relative_path"] == oscar_checkpoint.name
    assert oscar_scan["checkpoint_files_found"][0]["large_enough_for_wam_or_vla_weights"] is True
    not_checkpoint = tmp_path / "not-checkpoint.txt"
    not_checkpoint.write_text("not weights", encoding="utf-8")
    assert evaluator._checkpoint_like_files(not_checkpoint)["checkpoint_files_found"] == []
    assert evaluator._checkpoint_like_files(tmp_path / "missing-root")["files_scanned"] == 0

    class OSErrorFile:
        suffix = ".pt"
        name = "bad.pt"

        def is_file(self) -> bool:
            return True

        def stat(self) -> object:
            raise OSError("no stat")

    assert evaluator._checkpoint_like_files(OSErrorFile())["checkpoint_files_found"][0]["size_bytes"] is None

    scan_root = tmp_path / "scan-root"
    scan_root.mkdir()
    for index in range(13):
        (scan_root / f"model-{index}.pt").write_bytes(b"x")
    scan = evaluator._checkpoint_like_files(scan_root)
    assert scan["truncated"] is True
    assert len(scan["checkpoint_files_found"]) == 12
    many_files = tmp_path / "many-files"
    many_files.mkdir()
    for index in range(3):
        (many_files / f"file-{index}.txt").write_text("x", encoding="utf-8")
    assert evaluator._checkpoint_like_files(many_files, max_files_scanned=2)["truncated"] is True

    bad_stat = tmp_path / "bad-stat-root"
    bad_stat.mkdir()
    bad_file = bad_stat / "bad.pt"
    bad_file.write_bytes(b"x")
    original_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object):
        if self == bad_file:
            raise OSError("bad stat")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    assert evaluator._checkpoint_like_files(bad_stat)["checkpoint_files_found"][0]["size_bytes"] is None


def test_oscar_cosmos_wam_evaluator_host_probe_and_plan_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        def __init__(self, returncode: int, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(evaluator.platform, "system", lambda: "Linux")
    monkeypatch.setattr(evaluator.shutil, "which", lambda _name: None)
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: Completed(0, '{"cuda": true}'))
    assert evaluator._local_host_probe()["torch_cuda_available"] is True
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: Completed(1, ""))
    assert evaluator._local_host_probe()["torch_probe_error_type"] == "torch_probe_subprocess_failed"
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe")))
    assert evaluator._local_host_probe()["torch_probe_error_type"] == "RuntimeError"

    monkeypatch.setattr(
        evaluator,
        "_source_roots_for_candidate",
        lambda _candidate: [
            {"label": "missing", "path": tmp_path / "missing-source", "configured_by_env": False}
        ],
    )
    assert "blocked_missing_local_model_source_tree" in evaluator._local_source_tree_probe("oscar_wam")["blockers"]
    readiness = evaluator.build_policy_model_endpoint_readiness_manifest(
        generated_at="now",
        candidates=("oscar_wam",),
        explicit_candidate_id="oscar_wam",
        explicit_command="definitely-not-a-real-command",
        explicit_checkpoint=tmp_path / "missing-checkpoint",
    )
    row = readiness["candidates"][0]
    assert "make_configured_model_command_executable_or_on_path" in row["what_is_needed_to_make_true"]
    assert "download_or_mount_configured_model_checkpoint_path" in row["what_is_needed_to_make_true"]
    plan = evaluator.build_policy_model_endpoint_creation_plan(
        generated_at="now",
        readiness_manifest={"candidates": ["bad", row]},
    )
    assert len(plan["candidate_creation_plans"]) == 1
    layer_summary = plan["readiness_layer_summary"]
    assert layer_summary["reference_endpoint_wrapper_ready"] is True
    assert layer_summary["closed_loop_wam_policy_endpoint_ready"] is False
    assert (
        "blocked_closed_loop_wam_policy_requery_not_yet_proven"
        in layer_summary["closed_loop_wam_policy_endpoint_blockers"]
    )
    assert layer_summary["claim_boundary"][
        "reference_endpoint_ready_is_not_vla_manipulation_proof"
    ] is True

    ready_checkpoint = tmp_path / "ready-checkpoint"
    ready_checkpoint.mkdir()
    ready_command = tmp_path / "ready-command.py"
    ready_command.write_text("print('ok')\n", encoding="utf-8")
    discovery = evaluator.discover_wam_model_runtimes(
        candidates=("oscar_wam", "cosmos_wam"),
        generated_at="now",
        explicit_candidate_id="oscar_wam",
        explicit_command=f"{sys.executable} {ready_command}",
        explicit_checkpoint=ready_checkpoint,
    )
    assert discovery["selected_candidate"] == "oscar_wam"
    assert discovery["selected_candidate_blockers"] == []
    assert discovery["blockers"] == []
    assert "blocked_missing_wam_runtime" in discovery["all_candidate_blockers"]
    assert "local_host_probe" in discovery

    monkeypatch.setattr(evaluator.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(evaluator.shutil, "which", lambda _name: None)
    host_blocked = evaluator.discover_wam_model_runtimes(
        candidates=("oscar_wam",),
        generated_at="now",
        explicit_candidate_id="oscar_wam",
        explicit_command=f"{sys.executable} -m blueprint_pipeline.oscar_wam_command_adapter",
        explicit_checkpoint=ready_checkpoint,
    )
    host_blocked_row = host_blocked["candidates"][0]
    assert host_blocked_row["configured_command_checkpoint_ready"] is True
    assert host_blocked_row["provider_or_linux_cuda_runtime_required"] is True
    assert "blocked_oscar_linux_cuda_runtime_required" in host_blocked_row[
        "official_adapter_host_preflight_blockers"
    ]

    provider_discovery = evaluator.discover_wam_model_runtimes(
        candidates=("oscar_wam",),
        generated_at="now",
        explicit_candidate_id="oscar_wam",
        explicit_command=(
            f"{sys.executable} -m blueprint_pipeline.oscar_wam_provider_command_adapter "
            "--mode vast-provider"
        ),
    )
    provider_row = provider_discovery["candidates"][0]
    assert provider_row["configured_command_checkpoint_ready"] is True
    assert provider_row["checkpoint_exists"] is False
    assert provider_row["checkpoint_requirement_satisfied"] is True
    assert provider_row["checkpoint_requirement_satisfied_by_provider_runtime"] is True
    assert provider_row["missing_required_checkpoint_envs"] == []
    assert "blocked_missing_wam_model_checkpoint" not in provider_row["blockers"]

    provider_readiness = evaluator.build_policy_model_endpoint_readiness_manifest(
        generated_at="now",
        candidates=("oscar_wam",),
        explicit_candidate_id="oscar_wam",
        explicit_command=(
            f"{sys.executable} -m blueprint_pipeline.oscar_wam_provider_command_adapter "
            "--mode vast-provider"
        ),
        local_model_gate_enabled_override=True,
    )
    readiness_row = provider_readiness["candidates"][0]
    assert readiness_row["real_model_runtime_ready"] is True
    assert readiness_row["checkpoint_exists"] is False
    assert readiness_row["checkpoint_requirement_satisfied"] is True
    assert readiness_row["checkpoint_requirement_satisfied_by_provider_runtime"] is True
    assert readiness_row["missing_required_checkpoint_envs"] == []
    assert "set_BLUEPRINT_OSCAR_WAM_CHECKPOINT_to_local_checkpoint_path" not in (
        readiness_row["what_is_needed_to_make_true"]
    )

    status_dir = tmp_path / "status-videos"
    _write_json(status_dir / "video_generation_status.json", {"videos": ["bad", {"path": "fallback.mp4"}]})
    assert evaluator._review_videos(status_dir) == [{"path": "fallback.mp4"}]


def test_oscar_cosmos_wam_command_and_rollout_edge_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_manifest = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    _write_json(input_manifest, {"schema_version": "input"})
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("cmd", 1)),
    )
    payload, detail = evaluator._run_local_wam_command(
        command="python missing.py",
        input_manifest_path=input_manifest,
        output_path=output_path,
        candidate_id="oscar_wam",
        checkpoint_path="checkpoint.pt",
        timeout_seconds=1,
    )
    assert payload == {}
    assert detail["blockers"][0].startswith("wam_model_command_failed:")

    class Completed:
        returncode = 0
        stdout = "not-json"
        stderr = ""

    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: Completed())
    payload, detail = evaluator._run_local_wam_command(
        command="python missing.py",
        input_manifest_path=input_manifest,
        output_path=output_path,
        candidate_id="oscar_wam",
        checkpoint_path=None,
        timeout_seconds=1,
    )
    assert payload == {}
    assert detail["blockers"] == ["wam_model_stdout_json_invalid"]

    class NonMappingCompleted:
        returncode = 0
        stdout = "[]"
        stderr = ""

    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: NonMappingCompleted())
    payload, detail = evaluator._run_local_wam_command(
        command="python missing.py",
        input_manifest_path=input_manifest,
        output_path=output_path,
        candidate_id="oscar_wam",
        checkpoint_path=None,
        timeout_seconds=1,
    )
    assert payload == {}
    assert detail["status"] == "completed"

    assert evaluator._wam_rollout_blocked_reason(["blocked_local_wam_model_run_not_enabled"]) == "blocked_local_wam_model_run_not_enabled"
    assert evaluator._wam_rollout_blocked_reason(["wam_model_command_failed:TimeoutExpired"]) == "blocked_wam_model_command_failed"
    assert evaluator._wam_rollout_blocked_reason(["blocked_missing_wam_model_checkpoint"]) == "blocked_missing_wam_model_checkpoint"
    assert evaluator._wam_rollout_blocked_reason(["blocked_missing_wam_runtime", "blocked_missing_wam_model_checkpoint"]) == "blocked_missing_wam_runtime_and_checkpoint"
    assert evaluator._wam_rollout_blocked_reason(["blocked_custom"]) == "blocked_custom"
    assert evaluator._wam_rollout_blocked_reason([]) == "blocked_missing_wam_model_runtime_or_checkpoint"
    assert evaluator._rollout_video_path({}, base_dir=tmp_path) is None
    assert evaluator._rollout_video_path({"generated_video_path": "video.mp4"}, base_dir=tmp_path) == tmp_path / "video.mp4"


def test_generated_rollout_visual_smoke_flags_flat_dark_rollout(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "generated.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    first = np.zeros((48, 64, 3), dtype=np.uint8)
    first[:, :32] = (255, 0, 0)
    first[:, 32:] = (0, 255, 0)
    writer.write(first)
    for _ in range(9):
        writer.write(np.full((48, 64, 3), 86, dtype=np.uint8))
    writer.release()

    smoke = evaluator._generated_rollout_visual_smoke(
        rollouts=[{"rollout_id": "rollout_1", "generated_video_path": str(video)}],
        output_dir=tmp_path,
        generated_at="now",
    )

    assert smoke["status"] == "failed_visual_quality_smoke"
    assert "generated_rollout_later_frames_flat_or_dark" in smoke["blockers"]
    assert smoke["claim_boundary"]["valid_mp4_file_generated"] is True
    assert smoke["claim_boundary"]["visual_rollout_useful_for_task_success_review"] is False


def test_generated_rollout_visual_smoke_rejects_non_scene_first_frame(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "generated_noise_after_blank.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    writer.write(np.full((48, 64, 3), 86, dtype=np.uint8))
    for index in range(9):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32] = (255, (20 + index * 20) % 255, 0)
        frame[:, 32:] = (0, 180, (220 - index * 10) % 255)
        writer.write(frame)
    writer.release()

    smoke = evaluator._generated_rollout_visual_smoke(
        rollouts=[{"rollout_id": "rollout_1", "generated_video_path": str(video)}],
        output_dir=tmp_path,
        generated_at="now",
    )

    assert smoke["status"] == "failed_visual_quality_smoke"
    assert "generated_rollout_first_frame_not_scene_like" in smoke["blockers"]
    assert (
        smoke["rollouts"][0]["visual_quality_flags"][
            "first_frame_preserves_source_scene"
        ]
        is False
    )
    assert (
        smoke["claim_boundary"]["visual_rollout_useful_for_task_success_review"]
        is False
    )


def test_generated_rollout_visual_smoke_rejects_scene_structure_loss(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "generated_scene_loss.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    first = np.zeros((48, 64, 3), dtype=np.uint8)
    first[:, :32] = (255, 0, 0)
    first[:, 32:] = (0, 255, 0)
    cv2.rectangle(first, (18, 8), (46, 38), (255, 255, 255), 2)
    for x in range(0, 64, 4):
        cv2.line(first, (x, 0), (x, 47), (255, 255, 255), 1)
    for y in range(0, 48, 4):
        cv2.line(first, (0, y), (63, y), (0, 0, 0), 1)
    writer.write(first)
    gradient = np.tile(np.linspace(80, 170, 64, dtype=np.uint8), (48, 1))
    for index in range(9):
        shifted = np.roll(gradient, index + 1, axis=1)
        frame = np.dstack((shifted, shifted, shifted))
        writer.write(frame)
    writer.release()

    smoke = evaluator._generated_rollout_visual_smoke(
        rollouts=[{"rollout_id": "rollout_1", "generated_video_path": str(video)}],
        output_dir=tmp_path,
        generated_at="now",
    )

    assert smoke["status"] == "failed_visual_quality_smoke"
    assert "generated_rollout_later_frames_lost_scene_structure" in smoke["blockers"]
    assert smoke["rollouts"][0]["visual_quality_flags"][
        "later_frames_lost_scene_structure"
    ] is True
    assert (
        smoke["claim_boundary"]["visual_rollout_useful_for_task_success_review"]
        is False
    )


def test_oscar_cosmos_wam_evaluator_blocks_labels_and_consistency_for_flat_dark_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_SUCCESS_LABELING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    wam_command = tmp_path / "wam_model_flat_dark_command.py"
    wam_command.write_text(
        """
import json
import os
from pathlib import Path

import cv2
import numpy as np

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
video = output.parent / "flat_dark_rollout.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
first = np.zeros((48, 64, 3), dtype=np.uint8)
first[:, :32] = (255, 0, 0)
first[:, 32:] = (0, 255, 0)
writer.write(first)
for _ in range(9):
    writer.write(np.full((48, 64, 3), 86, dtype=np.uint8))
writer.release()
payload = {
    "schema_version": "oscar_wam_command_adapter.v1",
    "status": "completed",
    "adapter_id": "blueprint_oscar_wam_command_adapter",
    "rollouts": [
        {
            "rollout_id": "rollout_flat_dark",
            "policy_id": "unit_test_wam",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
        }
    ],
    "fresh_model_command_executed_this_invocation": True,
    "fresh_model_run_claimed": True,
    "learned_wam_model_ran": True,
    "truth_boundary": {"generated_video_is_model_output": True},
}
output.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    label_marker = tmp_path / "success_label_command_ran.marker"
    label_command = tmp_path / "success_label_should_not_run.py"
    label_command.write_text(
        f"""
import json
import os
from pathlib import Path

Path({str(label_marker)!r}).write_text("ran", encoding="utf-8")
request = json.loads(Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_INPUT"]).read_text(encoding="utf-8"))
rollout = request["rollouts"][0]
payload = {{
    "schema_version": "wam_success_labels.command.v1",
    "provider": "fake-vlm",
    "labels": [
        {{
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "success": True,
            "confidence": 0.99,
            "rationale": "This command should not run for failed visual smoke.",
            "visual_evidence_used": True,
        }}
    ],
}}
Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    consistency_marker = tmp_path / "consistency_command_ran.marker"
    consistency_command = tmp_path / "consistency_should_not_run.py"
    consistency_command.write_text(
        f"""
import json
import os
from pathlib import Path

Path({str(consistency_marker)!r}).write_text("ran", encoding="utf-8")
request = json.loads(Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_INPUT"]).read_text(encoding="utf-8"))
rollout = request["rollouts"][0]
payload = {{
    "schema_version": "wam_episode_consistency.command.v1",
    "status": "completed",
    "provider": "fake-vlm-episode-consistency",
    "rollout_checks": [
        {{
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "forward_consistent": True,
            "inverse_consistent": True,
            "confidence": 0.99,
            "rationale": "This command should not run for failed visual smoke.",
            "visual_evidence_used": True,
            "action_trace_evidence_used": True,
        }}
    ],
}}
Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_flat_dark_visual_quality_job",
        model_candidates=("oscar_wam",),
        wam_model_command=f"{sys.executable} {wam_command}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        wam_success_label_command=f"{sys.executable} {label_command}",
        allow_wam_success_labeling=True,
        wam_consistency_command=f"{sys.executable} {consistency_command}",
        allow_wam_consistency_scoring=True,
        generated_at="now",
    )

    job_dir = tmp_path / "wam_flat_dark_visual_quality_job"
    assert summary["learned_wam_model_ran"] is True
    assert summary["wam_generated_rollout_status"] == "completed_visual_quality_failed"
    assert summary["generated_rollout_visually_useful_for_success_review"] is False
    assert summary["wam_success_label_judge_configured"] is True
    assert summary["wam_success_label_judge_ran"] is False
    assert summary["wam_success_label_from_generated_video"] is False
    assert summary["external_episode_consistency_scorer_blocked_by_visual_quality"] is True
    assert summary["external_episode_consistency_scorer_ran"] is False
    assert summary["forward_inverse_consistency_proven"] is False
    assert not label_marker.exists()
    assert not consistency_marker.exists()

    generated = json.loads(
        (job_dir / "wam_generated_rollout_manifest.json").read_text(encoding="utf-8")
    )
    assert generated["status"] == "completed_visual_quality_failed"
    assert generated["valid_reviewable_generated_video_available"] is False
    assert "blocked_generated_rollout_not_visually_useful_for_success_review" in generated[
        "blockers"
    ]
    assert "generated_rollout_later_frames_flat_or_dark" in generated["blockers"]
    loop_manifest = json.loads(
        (job_dir / "wam_policy_loop_manifest.json").read_text(encoding="utf-8")
    )
    assert loop_manifest["learned_wam_model_ran"] is True
    assert loop_manifest["wam_generated_rollout_status"] == "completed_visual_quality_failed"
    assert loop_manifest["action_conditioned_video_rollout_generated"] is True
    assert loop_manifest["policy_observes_wam_generated_next_observation"] is False
    assert "generated_rollout_later_frames_flat_or_dark" in loop_manifest[
        "generated_rollout_visual_quality_blockers"
    ]
    assert loop_manifest["single_step_policy_requery_frame_useful"] is True
    assert "blocked_missing_policy_requery_endpoint" in loop_manifest[
        "why_policy_requery_not_run"
    ]
    assert "generated_rollout_later_frames_flat_or_dark" in loop_manifest[
        "why_wam_success_label_not_run"
    ]
    success_request = json.loads(
        (job_dir / "wam_success_label_request.json").read_text(encoding="utf-8")
    )
    assert success_request["status"] == "blocked_generated_rollout_visual_quality"
    success_labels = json.loads(
        (job_dir / "wam_success_labels.json").read_text(encoding="utf-8")
    )
    assert success_labels["status"] == "blocked"
    assert success_labels["label_count"] == 0
    assert success_labels["command_result"] is None
    assert "blocked_generated_rollout_not_visually_useful_for_success_review" in success_labels[
        "blockers"
    ]
    consistency_request = json.loads(
        (job_dir / "wam_episode_consistency_request.json").read_text(encoding="utf-8")
    )
    assert consistency_request["status"] == "blocked_generated_rollout_visual_quality"
    consistency = json.loads(
        (job_dir / "wam_consistency_checks.json").read_text(encoding="utf-8")
    )
    assert consistency["status"] == "blocked_generated_rollout_visual_quality"
    assert consistency["external_episode_consistency_scorer_ran"] is False
    assert "blocked_generated_rollout_not_visually_useful_for_success_review" in consistency[
        "blockers"
    ]


def test_oscar_cosmos_wam_evaluator_reports_source_tree_without_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_wam_runtime_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source_root = tmp_path / "source-only-oscar"
    source_root.mkdir()
    (source_root / "README.md").write_text("source checkout only", encoding="utf-8")
    (source_root / "camera.pt").write_bytes(b"not model weights")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", str(source_root))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)

    discovery = evaluator.discover_wam_model_runtimes(
        generated_at="now",
        candidates=("oscar_wam",),
    )
    oscar = discovery["candidates"][0]
    source = oscar["local_source_discovery"]

    assert source["source_tree_present"] is True
    assert source["present_source_tree_count"] >= 1
    assert "blocked_source_tree_present_without_runnable_adapter_command" in source["blockers"]
    assert "blocked_source_tree_present_without_configured_checkpoint" in source["blockers"]
    assert "blocked_missing_wam_runtime" in oscar["blockers"]
    assert "blocked_missing_wam_model_checkpoint" in oscar["blockers"]


def test_oscar_cosmos_wam_evaluator_reports_checkpointed_missing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_wam_runtime_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "oscar-checkpoint"
    checkpoint.mkdir()

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_checkpointed_job",
        model_candidates=("oscar_wam",),
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["blockers"] == ["blocked_missing_wam_runtime"]
    assert summary["wam_generated_rollout_status"] == "blocked_missing_wam_runtime"
    generated = json.loads(
        (tmp_path / "wam_checkpointed_job" / "wam_generated_rollout_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["blocked_reason"] == "blocked_missing_wam_runtime"
    consistency = json.loads(
        (tmp_path / "wam_checkpointed_job" / "wam_consistency_checks.json").read_text(
            encoding="utf-8"
        )
    )
    assert consistency["generated_rollout_termination_reason"] == "blocked_missing_wam_runtime"


def test_oscar_cosmos_wam_evaluator_runs_configured_command_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", raising=False)
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_command.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

import cv2
import numpy as np

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
video = output.parent / "rollout_1.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
for index in range(6):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :32] = (255, 30 + index * 20, 0)
    frame[:, 32:] = (0, 180, 220 - index * 10)
    writer.write(frame)
writer.release()
payload = {
    "schema_version": "oscar_wam_command_adapter.v1",
    "status": "completed",
    "adapter_id": "blueprint_oscar_wam_command_adapter",
    "rollouts": [
        {
            "rollout_id": "rollout_1",
            "policy_id": "unit_test_wam",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
            "model_rollout_confidence": 0.42,
        }
    ],
    "fresh_model_command_executed_this_invocation": True,
    "fresh_model_run_claimed": True,
    "learned_wam_model_ran": True,
    "truth_boundary": {"generated_video_is_model_output": True},
}
output.write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({"status": "completed"}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_COMMAND", f"{sys.executable} {command_path}")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_model_job",
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is True
    assert summary["model_command_executed_this_invocation"] is True
    assert summary["model_http_wrapper_ready"] is True
    assert summary["model_endpoint_command_probe_passed"] is True
    assert summary["real_model_endpoint_probe_claim_ready"] is True
    assert summary["real_model_endpoint_claim_blocked"] is (
        not (
            summary["real_model_endpoint_ready"]
            and summary["model_endpoint_command_probe_passed"]
        )
    )
    assert summary["wam_generated_rollout_status"] == "completed"
    assert summary["forward_inverse_consistency_proven"] is False
    assert summary["external_episode_consistency_scorer_required"] is True
    creation_plan = json.loads(
        (tmp_path / "wam_model_job" / "policy_model_endpoint_creation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert creation_plan["endpoint_wrapper_ready_candidate_count"] >= 1
    assert creation_plan["can_create_real_model_endpoint_now"] is summary[
        "real_model_endpoint_ready"
    ]
    assert creation_plan["readiness_layer_summary"]["wam_rollout_provider_ready"] is summary[
        "real_model_endpoint_ready"
    ]
    assert creation_plan["readiness_layer_summary"][
        "closed_loop_wam_policy_endpoint_ready"
    ] is False
    probe = json.loads(
        (
            tmp_path
            / "wam_model_job"
            / "policy_model_endpoint_probe_results.json"
        ).read_text(encoding="utf-8")
    )
    assert probe["status"] == "completed"
    assert probe["probe_attempted_candidate_count"] == 1
    assert probe["probe_passed_candidate_count"] == 1
    assert probe["can_claim_real_model_endpoint_after_probe"] is True
    assert probe["candidates"][0]["blueprint_output_contract_valid"] is True
    generated = json.loads(
        (tmp_path / "wam_model_job" / "wam_generated_rollout_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["rollout_count"] == 1
    truth = json.loads(
        (tmp_path / "wam_model_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["action_conditioned_video_rollout_generated"] is True
    assert truth["oscar_cosmos_openvla_unitree_model_ran"] is True
    assert truth["wam_rollout_model_ran"] is True
    assert truth["real_model_endpoint_ready"] is summary["real_model_endpoint_ready"]
    assert truth["learned_wam_model_ran"] is True
    assert truth["forward_inverse_consistency_proven"] is False
    assert truth["external_episode_consistency_scorer_required"] is True
    request = json.loads(
        (tmp_path / "wam_model_job" / "wam_episode_consistency_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert request["claim_boundary"]["scorer_is_separate_from_wam_execution_and_evaluator"] is True
    loop_manifest = json.loads(
        (tmp_path / "wam_model_job" / "wam_policy_loop_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert loop_manifest["learned_wam_model_ran"] is True
    assert loop_manifest["learned_wam_model_ran_this_invocation"] is True
    assert loop_manifest["wam_generated_rollout_status"] == "completed"
    assert loop_manifest["closed_loop_policy_wam_interaction"] is False
    assert loop_manifest["policy_observes_wam_generated_next_observation"] is False
    assert loop_manifest["wam_policy_requery_status"] == "blocked_missing_policy_requery_endpoint"
    assert loop_manifest["why_policy_requery_not_run"] == [
        "blocked_missing_policy_requery_endpoint"
    ]
    requery = json.loads(
        (tmp_path / "wam_model_job" / "wam_policy_requery_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert requery["scheduler_implemented"] is True
    assert requery["status"] == "blocked_missing_policy_requery_endpoint"
    assert requery["full_closed_loop_episode_proven"] is False
    assert summary["artifact_paths"]["wam_policy_loop_manifest"].endswith(
        "wam_policy_loop_manifest.json"
    )
    assert summary["artifact_paths"]["wam_policy_requery_manifest"].endswith(
        "wam_policy_requery_manifest.json"
    )


def test_oscar_cosmos_wam_evaluator_does_not_count_generic_rollout_payload_as_model_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "generic_wam_payload.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

import cv2
import numpy as np

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
video = output.parent / "rollout_1.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
for index in range(6):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :32] = (255, 30 + index * 20, 0)
    frame[:, 32:] = (0, 180, 220 - index * 10)
    writer.write(frame)
writer.release()
payload = {
    "rollouts": [
        {
            "rollout_id": "rollout_1",
            "policy_id": "generic_fixture",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
        }
    ]
}
output.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "generic_wam_payload_job",
        model_candidates=("oscar_wam",),
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["model_command_executed_this_invocation"] is True
    assert summary["learned_wam_model_output_available"] is True
    assert summary["learned_wam_model_ran"] is False
    assert summary["wam_rollout_model_ran"] is False
    truth = json.loads(
        (tmp_path / "generic_wam_payload_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["learned_wam_model_output_available"] is True
    assert truth["learned_wam_model_ran"] is False


def test_wam_policy_requery_blocks_generic_endpoint_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "generated_frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(
        evaluator,
        "_policy_requery_endpoint_row",
        lambda: {
            "runtime": "team",
            "endpoint_env": "TEAM_POLICY_ENDPOINT_URL",
            "endpoint_url": "http://127.0.0.1:8765/policy/action",
            "auth_file_env": "TEAM_POLICY_AUTH_TOKEN_FILE",
            "auth_token_file_path": str(tmp_path / "token"),
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_extract_wam_requery_frame",
        lambda *, rollout, output_dir: (
            frame,
            {"status": "completed", "extracted_frame_path": str(frame)},
        ),
    )

    def _fake_requery_endpoint(**_kwargs):
        return (
            {
                "policy_id": "local_wam_vla_policy_command",
                "action": {"action_type": "waypoint", "waypoint": [0.4, 0.0, 0.79]},
                "endpoint_metadata": {
                    "raw_response_redacted": {
                        "policy_id": "g1_endpoint_reference_adapter",
                        "claim_boundary": {"reference_endpoint_is_not_real_policy": True},
                    }
                },
            },
            {"status": "completed", "endpoint_invoked": True},
        )

    monkeypatch.setattr(evaluator, "_call_policy_requery_endpoint", _fake_requery_endpoint)

    manifest = evaluator._run_wam_policy_requery(
        output_dir=tmp_path,
        generated_at="now",
        input_dir=tmp_path / "input_job",
        rollouts=[
            {
                "rollout_id": "rollout_1",
                "scenario_eval_run_id": "run_1",
                "task_id": "contact_or_push_light_object",
                "generated_video_path": str(tmp_path / "rollout.mp4"),
            }
        ],
        visual_rollout_useful=True,
        single_step_policy_requery_frame_useful=True,
        visual_smoke_status="passed",
        task_prompts=[{"scenario_eval_run_id": "run_1", "task_prompt": "Push the object."}],
        timeout_seconds=1,
    )

    assert manifest["endpoint_action_returned_for_wam_generated_next_observation"] is True
    assert manifest["policy_observes_wam_generated_next_observation"] is False
    assert manifest["single_step_wam_policy_requery_proven"] is False
    assert manifest["status"] == "blocked_policy_requery_endpoint_not_unitree_g1_hand_policy"
    assert (
        manifest["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert manifest["g1_robot_policy_selected_family"] is None
    assert manifest["openvla_selected_as_g1_robot_policy"] is False
    assert manifest["wam_rollout_selected_as_g1_robot_policy"] is False
    assert "blocked_policy_requery_endpoint_not_unitree_g1_hand_policy" in manifest["blockers"]
    assert (
        "blocked_policy_requery_endpoint_not_real_vla_or_unitree_hand_policy"
        in manifest["blockers"]
    )


def test_wam_policy_requery_accepts_unitree_lerobot_hand_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "generated_frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(
        evaluator,
        "_policy_requery_endpoint_row",
        lambda: {
            "runtime": "vla",
            "endpoint_env": "VLA_POLICY_ENDPOINT_URL",
            "endpoint_url": "http://127.0.0.1:8765/policy/action",
            "auth_file_env": "VLA_POLICY_AUTH_TOKEN_FILE",
            "auth_token_file_path": str(tmp_path / "token"),
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_extract_wam_requery_frame",
        lambda *, rollout, output_dir: (
            frame,
            {"status": "completed", "extracted_frame_path": str(frame)},
        ),
    )

    def _fake_requery_endpoint(**_kwargs):
        return (
            {
                "policy_id": "unitree_lerobot_g1_policy",
                "action": {
                    "action_type": "manipulation_contact",
                    "target_object_id": "blueprint_light_object",
                    "waypoint": [0.54, -0.65, 0.79],
                },
                "endpoint_metadata": {
                    "raw_response_redacted": {
                        "policy_id": "unitree_lerobot_g1_policy",
                        "policy_kind": "unitree_lerobot_g1_manipulation_policy",
                        "claim_boundary": {
                            "unitree_hand_manipulation_policy_used": True,
                            "unitree_lerobot_or_isaaclab_manipulation_policy_used": True,
                        },
                    }
                },
            },
            {"status": "completed", "endpoint_invoked": True},
        )

    monkeypatch.setattr(evaluator, "_call_policy_requery_endpoint", _fake_requery_endpoint)

    manifest = evaluator._run_wam_policy_requery(
        output_dir=tmp_path,
        generated_at="now",
        input_dir=tmp_path / "input_job",
        rollouts=[
            {
                "rollout_id": "rollout_1",
                "scenario_eval_run_id": "run_1",
                "task_id": "contact_or_push_light_object",
                "generated_video_path": str(tmp_path / "rollout.mp4"),
            }
        ],
        visual_rollout_useful=True,
        single_step_policy_requery_frame_useful=True,
        visual_smoke_status="passed",
        task_prompts=[{"scenario_eval_run_id": "run_1", "task_prompt": "Push the object."}],
        timeout_seconds=1,
    )

    assert manifest["status"] == "completed"
    assert manifest["endpoint_action_returned_for_wam_generated_next_observation"] is True
    assert manifest["policy_observes_wam_generated_next_observation"] is True
    assert manifest["single_step_wam_policy_requery_proven"] is True
    assert manifest["real_vla_or_unitree_hand_policy_endpoint_used"] is True
    assert manifest["unitree_g1_hand_policy_endpoint_used"] is True
    assert (
        manifest["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert manifest["g1_robot_policy_selected_family"] == "unitree_native_hand_policy_endpoint"
    assert manifest["openvla_selected_as_g1_robot_policy"] is False
    assert manifest["wam_rollout_selected_as_g1_robot_policy"] is False
    assert manifest["unitree_hand_policy_requery_used"] is True


def test_wam_policy_requery_unitree_provider_replay_is_not_fresh_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "generated_frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(
        evaluator,
        "_policy_requery_endpoint_row",
        lambda: {
            "runtime": "team",
            "endpoint_env": "TEAM_POLICY_ENDPOINT_URL",
            "endpoint_url": "http://127.0.0.1:8765/policy/action",
            "auth_file_env": "TEAM_POLICY_AUTH_TOKEN_FILE",
            "auth_token_file_path": str(tmp_path / "token"),
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_extract_wam_requery_frame",
        lambda *, rollout, output_dir: (
            frame,
            {"status": "completed", "extracted_frame_path": str(frame)},
        ),
    )

    def _fake_requery_endpoint(**_kwargs):
        return (
            {
                "policy_id": "unitree_unifolm_vla_policy_provider_replay",
                "action": {
                    "action_type": "manipulation_contact",
                    "target_object_id": "blueprint_light_object",
                    "waypoint": [0.54, -0.65, 0.79],
                },
                "endpoint_metadata": {
                    "raw_response_redacted": {
                        "policy_id": "unitree_unifolm_vla_policy_provider_replay",
                        "policy_kind": "unitree_unifolm_vla_policy_provider_replay",
                        "claim_boundary": {
                            "unitree_hand_manipulation_policy_used": True,
                            "provider_output_replay_used": True,
                        },
                    }
                },
            },
            {"status": "completed", "endpoint_invoked": True},
        )

    monkeypatch.setattr(evaluator, "_call_policy_requery_endpoint", _fake_requery_endpoint)

    manifest = evaluator._run_wam_policy_requery(
        output_dir=tmp_path,
        generated_at="now",
        input_dir=tmp_path / "input_job",
        rollouts=[
            {
                "rollout_id": "rollout_1",
                "scenario_eval_run_id": "run_1",
                "task_id": "contact_or_push_light_object",
                "generated_video_path": str(tmp_path / "rollout.mp4"),
            }
        ],
        visual_rollout_useful=False,
        single_step_policy_requery_frame_useful=True,
        visual_smoke_status="failed_visual_quality_smoke",
        task_prompts=[{"scenario_eval_run_id": "run_1", "task_prompt": "Push the object."}],
        timeout_seconds=1,
    )

    assert manifest["status"] == (
        "blocked_policy_requery_provider_replay_not_fresh_unitree_hand_policy"
    )
    assert manifest["endpoint_action_returned_for_wam_generated_next_observation"] is True
    assert manifest["unitree_g1_hand_policy_output_observed"] is True
    assert manifest["unitree_g1_hand_policy_endpoint_used"] is False
    assert manifest["fresh_unitree_hand_policy_requery_inference_proven"] is False
    assert manifest["policy_observes_wam_generated_next_observation"] is False
    assert manifest["single_step_wam_policy_requery_proven"] is False
    assert manifest["policy_requery_provider_replay_used"] is True
    assert manifest["g1_robot_policy_selected_family"] is None
    assert "blocked_policy_requery_provider_replay_not_fresh_unitree_hand_policy" in manifest[
        "blockers"
    ]


def test_oscar_cosmos_wam_evaluator_blocks_placeholder_generated_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_placeholder_command.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
video = output.parent / "rollout_1.mp4"
video.write_bytes(b"mp4-placeholder")
payload = {
    "rollouts": [
        {
            "rollout_id": "rollout_1",
            "policy_id": "unit_test_wam",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
        }
    ]
}
output.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_COMMAND", f"{sys.executable} {command_path}")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_placeholder_video_job",
        model_candidates=("oscar_wam",),
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["model_command_executed_this_invocation"] is True
    assert summary["learned_wam_model_ran"] is False
    assert summary["learned_wam_model_output_available"] is False
    assert "blocked_generated_rollout_video_not_reviewable" in summary["blockers"]
    generated = json.loads(
        (tmp_path / "wam_placeholder_video_job" / "wam_generated_rollout_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["generated_video_review_validations"][0]["status"] == "blocked"


def test_oscar_cosmos_wam_evaluator_uses_external_episode_consistency_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_consistent_command.py"
    command_path.write_text(
        f"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
work = output.parent / "consistent_oscar_payload"
work.mkdir(parents=True, exist_ok=True)
first_frame = work / "first_frame.png"
skeleton = work / "blueprint_proxy_skeleton_conditioning.mp4"
video = work / "oscar_generated_rollout.mp4"
frame = np.zeros((48, 64, 3), dtype=np.uint8)
frame[:, :32] = (255, 0, 0)
frame[:, 32:] = (0, 255, 0)
cv2.imwrite(str(first_frame), frame)
for target in (skeleton, video):
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    for index in range(6):
        next_frame = frame.copy()
        next_frame[:, :32, 1] = (20 + index * 30) % 255
        next_frame[:, 32:, 2] = (40 + index * 35) % 255
        writer.write(next_frame)
    writer.release()
input_package = {{
    "schema_version": "blueprint_oscar_wam_input_package.v1",
    "status": "completed",
    "num_frames": 6,
    "fps": 5.0,
    "height": 48,
    "width": 64,
    "scenario_eval_run_id": "run_1",
    "task_id": "approach_target",
    "spawn_id": "doorway",
    "first_frame": {{"path": str(first_frame), "height": 48, "width": 64}},
    "skeleton_video": {{
        "path": str(skeleton),
        "frame_count": 6,
        "fps": 5.0,
        "height": 48,
        "width": 64,
        "action_type_counts": [{{"action_type": "waypoint", "count": 6}}],
    }},
    "claim_boundary": {{
        "skeleton_conditioning_is_proxy_from_mujoco_trace": True,
        "true_robot_proprioceptive_skeleton_available": False,
        "generated_input_is_not_model_output": True,
    }},
}}
payload = {{
    "schema_version": "oscar_wam_command_adapter.v1",
    "status": "completed",
    "adapter_id": "blueprint_oscar_wam_command_adapter",
    "rollouts": [
        {{
            "rollout_id": "rollout_1",
            "policy_id": "unit_test_wam",
            "model_candidate": "oscar_wam",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
            "model_rollout_confidence": None,
        }}
    ],
    "model_provenance": {{
        "candidate": "oscar_wam",
        "checkpoint_path": {str(checkpoint)!r},
        "checkpoint_exists": True,
    }},
    "input_package": input_package,
    "blockers": [],
    "fresh_model_command_executed_this_invocation": True,
    "fresh_model_run_claimed": True,
    "learned_wam_model_ran": True,
    "truth_boundary": {{"generated_video_is_model_output": True}},
}}
output.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    consistency_command = tmp_path / "wam_consistency_command.py"
    consistency_command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_INPUT"]).read_text(encoding="utf-8"))
assert request["schema_version"] == "wam_episode_consistency_request.v1"
assert request["claim_boundary"]["scorer_is_separate_from_wam_execution_and_evaluator"] is True
rollout = request["rollouts"][0]
payload = {
    "schema_version": "wam_episode_consistency.command.v1",
    "status": "completed",
    "provider": "fake-vlm-episode-consistency",
    "model": "fake-vlm",
    "rollout_checks": [
        {
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "model_candidate": rollout.get("model_candidate"),
            "forward_consistent": True,
            "inverse_consistent": True,
            "confidence": 0.97,
            "rationale": "Visible motion follows the trace context.",
            "visible_action_alignment_evidence": ["video and trace both show a waypoint-conditioned rollout"],
            "inconsistency_evidence": [],
            "visual_evidence_used": True,
            "action_trace_evidence_used": True,
        }
    ],
}
Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_consistent_model_job",
        model_candidates=("oscar_wam",),
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        wam_consistency_command=f"{sys.executable} {consistency_command}",
        allow_wam_consistency_scoring=True,
        generated_at="now",
    )

    assert cv2  # keep the import visible to pytest's skip path
    assert summary["learned_wam_model_ran"] is True
    assert summary["forward_inverse_consistency_proven"] is True
    assert summary["external_episode_consistency_scorer_ran"] is True
    assert summary["external_episode_consistency_scorer_required"] is False
    assert summary["external_episode_consistency_scorer_id"] == "fake-vlm-episode-consistency"
    assert summary["artifact_paths"]["wam_episode_consistency_request"].endswith(
        "wam_episode_consistency_request.json"
    )
    consistency = json.loads(
        (tmp_path / "wam_consistent_model_job" / "wam_consistency_checks.json").read_text(
            encoding="utf-8"
        )
    )
    assert consistency["external_episode_consistency_scorer_id"] == "fake-vlm-episode-consistency"
    assert consistency["forward_dynamics_consistency_proven"] is True
    assert consistency["inverse_dynamics_consistency_proven"] is True
    assert consistency["what_is_needed_to_make_forward_inverse_consistency_true"] == []
    assert consistency["rollout_checks"][0]["visual_evidence_used"] is True
    assert consistency["rollout_checks"][0]["action_trace_evidence_used"] is True
    assert consistency["claim_boundary"][
        "forward_inverse_consistency_is_external_episode_label_not_wam_execution"
    ] is True
    assert consistency["claim_boundary"][
        "forward_inverse_consistency_does_not_prove_task_success"
    ] is True
    assert consistency["claim_boundary"][
        "forward_inverse_consistency_does_not_prove_physical_robot_readiness"
    ] is True
    checks = {row["check_id"]: row for row in consistency["checks"]}
    assert checks["forward_dynamics_consistency"]["status"] == "passed"
    assert checks["inverse_dynamics_consistency"]["status"] == "passed"
    truth = json.loads(
        (tmp_path / "wam_consistent_model_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["forward_inverse_consistency_proven"] is True
    assert truth["external_episode_consistency_scorer_ran"] is True
    assert truth["external_episode_consistency_scorer_required"] is False
    assert truth["external_episode_consistency_scorer_id"] == "fake-vlm-episode-consistency"
    assert truth["generated_outputs_are_raw_capture_evidence"] is False


def test_oscar_cosmos_wam_evaluator_runs_success_label_command_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_SUCCESS_LABELING", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    wam_command = tmp_path / "wam_model_command.py"
    wam_command.write_text(
        """
import json
import os
from pathlib import Path

import cv2
import numpy as np

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
video = output.parent / "rollout_1.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
for index in range(6):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :32] = (255, 30 + index * 20, 0)
    frame[:, 32:] = (0, 180, 220 - index * 10)
    writer.write(frame)
writer.release()
payload = {
    "schema_version": "oscar_wam_command_adapter.v1",
    "status": "completed",
    "adapter_id": "blueprint_oscar_wam_command_adapter",
    "rollouts": [
        {
            "rollout_id": "rollout_1",
            "policy_id": "unit_test_wam",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
            "model_rollout_confidence": 0.42,
        }
    ],
    "fresh_model_command_executed_this_invocation": True,
    "fresh_model_run_claimed": True,
    "learned_wam_model_ran": True,
    "truth_boundary": {"generated_video_is_model_output": True},
}
output.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    label_command = tmp_path / "success_label_command.py"
    label_command.write_text(
        """
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_INPUT"]).read_text(encoding="utf-8"))
rollout = request["rollouts"][0]
payload = {
    "schema_version": "wam_success_labels.command.v1",
    "provider": "fake-vlm",
    "model": "fake-video-judge",
    "labels": [
        {
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "success": True,
            "confidence": 0.91,
            "rationale": "The generated video reaches the target.",
            "task_completion_evidence": ["target reached"],
            "failure_modes": [],
            "visual_evidence_used": True,
        }
    ],
}
Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_success_label_job",
        model_candidates=("oscar_wam",),
        wam_model_command=f"{sys.executable} {wam_command}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        wam_success_label_command=f"{sys.executable} {label_command}",
        allow_wam_success_labeling=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is True
    assert summary["wam_success_label_from_generated_video"] is True
    assert summary["wam_success_label_judge_configured"] is True
    assert summary["wam_success_label_judge_ran"] is True
    assert summary["forward_inverse_consistency_proven"] is False
    labels = json.loads(
        (tmp_path / "wam_success_label_job" / "wam_success_labels.json").read_text(
            encoding="utf-8"
        )
    )
    assert labels["status"] == "completed"
    assert labels["label_count"] == 1
    assert labels["labels"][0]["success"] is True
    scorecard = json.loads(
        (tmp_path / "wam_success_label_job" / "wam_policy_scorecard.json").read_text(
            encoding="utf-8"
        )
    )
    assert scorecard["score_source"] == "vlm_judge_generated_video"
    assert scorecard["success_rate"] == 1.0
    truth = json.loads(
        (tmp_path / "wam_success_label_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["wam_success_label_from_generated_video"] is True
    assert truth["wam_success_label_judge_ran"] is True
    assert truth["forward_inverse_consistency_proven"] is False
    assert truth["external_episode_consistency_scorer_ran"] is False
    assert truth["external_episode_consistency_scorer_required"] is True


def test_oscar_cosmos_wam_evaluator_uses_env_command_and_blocks_missing_rollout_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_missing_video.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
payload = {"rollouts": ["bad", {"rollout_id": "rollout_missing", "generated_video_path": "missing.mp4"}]}
output.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_COMMAND", f"{sys.executable} {command_path}")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_missing_video_job",
        model_candidates=("oscar_wam",),
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert "blocked_generated_rollout_video_missing" in summary["blockers"]


def test_oscar_cosmos_wam_evaluator_runs_provider_command_without_local_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    input_job = _input_job(tmp_path)
    command_path = tmp_path / "provider_wam_model_command.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

import cv2
import numpy as np

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
video = output.parent / "provider_rollout_1.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
yy, xx = np.indices((48, 64))
for index in range(6):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = (xx * 3 + index * 17) % 255
    frame[:, :, 1] = (yy * 5 + 90 + index * 11) % 255
    frame[:, :, 2] = ((xx + yy) * 2 + 40 + index * 23) % 255
    writer.write(frame)
writer.release()
payload = {
    "schema_version": "oscar_wam_provider_command_adapter.v1",
    "status": "completed",
    "adapter_id": "blueprint_oscar_wam_provider_command_adapter",
    "mode": "vast_provider",
    "rollouts": [
        {
            "rollout_id": "provider_rollout_1",
            "policy_id": "unit_test_provider_wam",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
        }
    ],
    "fresh_model_command_executed_this_invocation": True,
    "fresh_model_run_claimed": True,
    "fresh_provider_model_run_claimed": True,
    "fresh_provider_launch_attempted": True,
    "learned_wam_model_ran": True,
    "truth_boundary": {"generated_video_is_model_output": True},
    "blockers": [],
}
output.write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({"status": "completed"}))
""".strip(),
        encoding="utf-8",
    )
    command = (
        f"{sys.executable} {command_path} "
        "--adapter blueprint_pipeline.oscar_wam_provider_command_adapter"
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_provider_command_job",
        model_candidates=("oscar_wam",),
        wam_model_command=command,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is True
    assert summary["model_command_executed_this_invocation"] is True
    assert summary["wam_generated_rollout_status"] == "completed"
    assert "blocked_missing_wam_model_checkpoint" not in summary["blockers"]
    discovery = json.loads(
        (tmp_path / "wam_provider_command_job" / "wam_model_runtime_discovery.json").read_text(
            encoding="utf-8"
        )
    )
    row = discovery["candidates"][0]
    assert row["checkpoint_exists"] is False
    assert row["checkpoint_requirement_satisfied"] is True
    assert row["checkpoint_requirement_satisfied_by_provider_runtime"] is True


def test_oscar_cosmos_wam_evaluator_imports_provider_output_without_fresh_run_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.setattr(evaluator, "normalize_model_access_env", lambda: None)
    monkeypatch.setattr(
        evaluator,
        "model_access_secret_status",
        lambda: {"huggingface": {"auth_ready": True}},
    )
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "provider-checkpoint-provenance"
    checkpoint.mkdir(parents=True)
    provider_job = tmp_path / "completed-provider-job"
    _write_provider_output_zip(provider_job / "vast_provider_runtime_output.zip")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_PROVIDER_COMPLETED_JOB_DIR", str(provider_job))

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_provider_replay_job",
        model_candidates=("oscar_wam",),
        wam_model_command=(
            f"{sys.executable} -m blueprint_pipeline.oscar_wam_provider_command_adapter "
            "--mode replay-existing-provider-output"
        ),
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["wam_generated_rollout_status"] == "completed"
    assert summary["learned_wam_model_output_available"] is True
    assert summary["learned_wam_model_ran"] is False
    assert summary["provider_output_replay_used"] is True
    assert summary["forward_inverse_consistency_proven"] is False
    assert summary["external_episode_consistency_scorer_ran"] is False
    generated = json.loads(
        (tmp_path / "wam_provider_replay_job" / "wam_generated_rollout_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["action_conditioned_video_rollout_available"] is True
    assert generated["action_conditioned_video_rollout_generated"] is True
    truth = json.loads(
        (tmp_path / "wam_provider_replay_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["provider_output_replay_used"] is True
    assert truth["learned_wam_model_output_available"] is True
    assert truth["learned_wam_model_ran"] is False
    assert truth["forward_inverse_consistency_proven"] is False
    assert truth["external_episode_consistency_scorer_ran"] is False


def test_oscar_cosmos_wam_evaluator_reports_imported_provider_model_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.setattr(evaluator, "normalize_model_access_env", lambda: None)
    monkeypatch.setattr(
        evaluator,
        "model_access_secret_status",
        lambda: {"huggingface": {"auth_ready": True}},
    )
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "provider-checkpoint-provenance"
    checkpoint.mkdir(parents=True)
    provider_job = tmp_path / "completed-provider-job"
    _write_provider_output_zip(
        provider_job / "vast_provider_runtime_output.zip",
        include_runtime_result=True,
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_PROVIDER_COMPLETED_JOB_DIR", str(provider_job))

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_provider_runtime_replay_job",
        model_candidates=("oscar_wam",),
        wam_model_command=(
            f"{sys.executable} -m blueprint_pipeline.oscar_wam_provider_command_adapter "
            "--mode replay-existing-provider-output"
        ),
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["provider_output_replay_used"] is True
    assert summary["fresh_model_command_executed_this_invocation"] is False
    assert summary["learned_wam_model_ran"] is True
    assert summary["provider_learned_wam_model_ran"] is True
    assert summary["provider_generated_video_is_model_output"] is True
    assert summary["oscar_cosmos_openvla_unitree_model_ran"] is True
    truth = json.loads(
        (
            tmp_path
            / "wam_provider_runtime_replay_job"
            / "wam_evaluator_truth_boundary.json"
        ).read_text(encoding="utf-8")
    )
    assert truth["learned_wam_model_ran"] is True
    assert truth["provider_output_replay_used"] is True
    assert truth["fresh_model_command_executed_this_invocation"] is False
    assert truth["action_conditioned_video_rollout_generated"] is True


def test_policy_endpoint_readiness_reports_missing_file_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = tmp_path / "wam-command"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.setattr(evaluator, "normalize_model_access_env", lambda: None)
    monkeypatch.setattr(
        evaluator,
        "model_access_secret_status",
        lambda: {"huggingface": {"auth_ready": False}},
    )

    manifest = evaluator.build_policy_model_endpoint_readiness_manifest(
        generated_at="now",
        candidates=("oscar_wam",),
        explicit_candidate_id="oscar_wam",
        explicit_command=str(command),
        explicit_checkpoint=checkpoint,
    )

    row = manifest["candidates"][0]
    assert row["status"] == "blocked"
    assert row["model_access_auth_ready"] == {"huggingface": False}
    assert "configure_file_based_huggingface_auth" in row["what_is_needed_to_make_true"]
    assert "configure_file_based_huggingface_auth" in manifest["blockers"]


def test_wam_policy_requery_records_unitree_lerobot_hand_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "generated_frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(
        evaluator,
        "_policy_requery_endpoint_row",
        lambda: {
            "runtime": "vla",
            "endpoint_env": "VLA_POLICY_ENDPOINT_URL",
            "endpoint_url": "http://127.0.0.1:8765/policy/action",
            "auth_token_file_path": str(tmp_path / "token"),
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_extract_wam_requery_frame",
        lambda *, rollout, output_dir: (
            frame,
            {"status": "completed", "extracted_frame_path": str(frame)},
        ),
    )

    def fake_call(*, endpoint_row, observation, timeout_seconds):
        return (
            {
                "policy_id": "unitree_lerobot_g1_policy",
                "action": {"action_type": "manipulation_contact"},
                "endpoint_metadata": {
                    "raw_response_redacted": {
                        "policy_id": "unitree_lerobot_g1_policy",
                        "policy_kind": "unitree_lerobot_g1_manipulation_policy",
                        "claim_boundary": {
                            "unitree_hand_manipulation_policy_used": True,
                            "unitree_lerobot_or_isaaclab_manipulation_policy_used": True,
                            "provider_output_replay_used": False,
                        },
                    }
                },
            },
            {"status": "completed", "endpoint_invoked": True},
        )

    monkeypatch.setattr(evaluator, "_call_policy_requery_endpoint", fake_call)

    result = evaluator._run_wam_policy_requery(
        output_dir=tmp_path,
        generated_at="now",
        input_dir=tmp_path / "source_job",
        rollouts=[
            {
                "rollout_id": "rollout_1",
                "scenario_eval_run_id": "run_1",
                "task_id": "contact_or_push_light_object",
                "generated_video_path": str(tmp_path / "generated.mp4"),
            }
        ],
        visual_rollout_useful=True,
        single_step_policy_requery_frame_useful=True,
        visual_smoke_status="passed",
        task_prompts=[
            {
                "scenario_eval_run_id": "run_1",
                "task_id": "contact_or_push_light_object",
                "task_prompt": "touch the object",
            }
        ],
        timeout_seconds=2,
    )

    assert result["status"] == "completed"
    assert result["single_step_wam_policy_requery_proven"] is True
    assert result["real_vla_or_unitree_hand_policy_endpoint_used"] is True
    assert result["unitree_g1_hand_policy_endpoint_used"] is True
    assert (
        result["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert result["g1_robot_policy_selected_family"] == "unitree_native_hand_policy_endpoint"
    assert result["openvla_selected_as_g1_robot_policy"] is False
    assert result["wam_rollout_selected_as_g1_robot_policy"] is False
    assert result["unitree_hand_policy_requery_used"] is True
    assert result["policy_requery_policy_id"] == "unitree_lerobot_g1_policy"
    assert result["claim_boundary"]["single_step_wam_policy_requery_is_not_task_success"] is True


def test_wam_policy_requery_reference_endpoint_does_not_satisfy_hand_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "generated_frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(
        evaluator,
        "_policy_requery_endpoint_row",
        lambda: {
            "runtime": "team",
            "endpoint_env": "TEAM_POLICY_ENDPOINT_URL",
            "endpoint_url": "http://127.0.0.1:8765/policy/action",
            "auth_token_file_path": str(tmp_path / "token"),
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_extract_wam_requery_frame",
        lambda *, rollout, output_dir: (
            frame,
            {"status": "completed", "extracted_frame_path": str(frame)},
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_call_policy_requery_endpoint",
        lambda *, endpoint_row, observation, timeout_seconds: (
            {
                "policy_id": "local_wam_vla_policy_command",
                "action": {"action_type": "waypoint", "waypoint": [0.0, 0.0, 0.79]},
                "endpoint_metadata": {
                    "raw_response_redacted": {
                        "policy_id": "reference_fixture_policy",
                        "claim_boundary": {
                            "unitree_hand_manipulation_policy_used": False,
                        },
                    }
                },
            },
            {"status": "completed", "endpoint_invoked": True},
        ),
    )

    result = evaluator._run_wam_policy_requery(
        output_dir=tmp_path,
        generated_at="now",
        input_dir=tmp_path / "source_job",
        rollouts=[
            {
                "rollout_id": "rollout_1",
                "scenario_eval_run_id": "run_1",
                "task_id": "contact_or_push_light_object",
                "generated_video_path": str(tmp_path / "generated.mp4"),
            }
        ],
        visual_rollout_useful=True,
        single_step_policy_requery_frame_useful=True,
        visual_smoke_status="passed",
        task_prompts=[],
        timeout_seconds=2,
    )

    assert result["status"] == "blocked_policy_requery_endpoint_not_unitree_g1_hand_policy"
    assert result["endpoint_action_returned_for_wam_generated_next_observation"] is True
    assert result["single_step_wam_policy_requery_proven"] is False
    assert result["real_vla_or_unitree_hand_policy_endpoint_used"] is False
    assert result["unitree_g1_hand_policy_endpoint_used"] is False
    assert (
        result["g1_robot_policy_selection_contract"]
        == "unitree_native_policy_required_for_g1_claims"
    )
    assert result["g1_robot_policy_selected_family"] is None
    assert result["openvla_selected_as_g1_robot_policy"] is False
    assert result["wam_rollout_selected_as_g1_robot_policy"] is False
    assert result["unitree_hand_policy_requery_used"] is False
    assert "blocked_policy_requery_endpoint_not_unitree_g1_hand_policy" in result["blockers"]
    assert "blocked_policy_requery_endpoint_not_real_vla_or_unitree_hand_policy" in result[
        "blockers"
    ]


def test_oscar_cosmos_wam_evaluator_uses_default_job_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_root=tmp_path / "jobs",
        model_candidates=("oscar_wam",),
        generated_at="now",
    )

    assert Path(summary["job_dir"]).parent == tmp_path / "jobs"


def test_oscar_cosmos_wam_evaluator_propagates_adapter_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_command.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
output.write_text(json.dumps({
    "status": "blocked",
    "blockers": ["blocked_oscar_requires_cuda_gpu_runtime"],
}), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_model_job",
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["wam_generated_rollout_status"] == "blocked_oscar_requires_cuda_gpu_runtime"
    assert "blocked_oscar_requires_cuda_gpu_runtime" in summary["blockers"]


def test_configured_wam_command_failure_is_not_reported_as_missing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "failing_wam_model_command.py"
    command_path.write_text(
        "import sys\nsys.stderr.write('cuda runtime unavailable\\n')\nsys.exit(7)\n",
        encoding="utf-8",
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_failed_runtime_job",
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["wam_generated_rollout_status"] == "blocked_wam_model_command_failed"
    assert "wam_model_command_nonzero_exit" in summary["blockers"]
    assert "blocked_missing_wam_runtime" not in summary["blockers"]
    assert "blocked_missing_wam_model_checkpoint" not in summary["blockers"]
    generated = json.loads(
        (tmp_path / "wam_failed_runtime_job" / "wam_generated_rollout_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["blocked_reason"] == "blocked_wam_model_command_failed"


def test_oscar_cosmos_wam_evaluator_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)
    exit_code = evaluator.main(
        ["--input-job-dir", str(input_job), "--job-dir", str(tmp_path / "cli_wam_job")]
    )
    assert exit_code == 0
    assert '"learned_wam_model_ran": false' in capsys.readouterr().out
