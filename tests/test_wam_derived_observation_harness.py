from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from blueprint_pipeline.wam_derived_observation_harness import (
    build_wam_derived_observation_step,
    run_wam_derived_observation_harness_step,
)
from blueprint_pipeline.wam_real_provider_validation_probe import (
    _validation_status,
    run_external_backend_from_env,
    run_probe,
)
from blueprint_pipeline.wam_sim_provider_e2e import run_sim_provider_e2e


def _write_frame(path: Path, *, size: tuple[int, int] = (640, 480), dark: bool = False) -> Path:
    if dark:
        image = Image.new("RGB", size, (5, 5, 5))
    else:
        width, height = size
        x_gradient = np.tile(np.linspace(40, 220, width, dtype=np.uint8), (height, 1))
        y_gradient = np.tile(np.linspace(30, 180, height, dtype=np.uint8), (width, 1)).T
        image = Image.fromarray(np.dstack((x_gradient, np.roll(x_gradient, 12, axis=1), y_gradient)), mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((260, 190, 380, 280), outline=(255, 255, 255), width=4)
        draw.ellipse((305, 215, 335, 245), fill=(230, 80, 45))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _target(
    *,
    object_id: str = "Sink054_handle",
    bbox: dict[str, int] | None = None,
    mask_path: str | None = None,
    width_m: float | None = None,
) -> dict[str, Any]:
    target = {
        "object_id": object_id,
        "track_id": object_id,
        "label": "right sink handle",
        "bbox": bbox or {"x": 260, "y": 190, "width": 120, "height": 90},
        "mask_path": mask_path or "fixture-mask.png",
        "confidence": 0.91,
        "source": "object_index_stage_fixture",
    }
    if width_m is not None:
        target["width_m"] = width_m
        target["size_source"] = "fixture_object_index"
    return target


def _grounding(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "eval_ready_task_grounding.v1",
        "status": "ready_for_learned_wam_rollout_request",
        "task": {
            "task_id": "turn_on_sink_handle",
            "target_prompts_for_object_index_backends": ["right sink handle"],
        },
        "selected_task_target": target,
        "readiness": {
            "learned_rollout_request_ready": True,
            "target_mask_or_keypoint_available": True,
            "blockers": [],
        },
    }


def _observation(frame: Path) -> dict[str, Any]:
    return {
        "schema_version": "blueprint_policy_observation.v1",
        "task_id": "turn_on_sink_handle",
        "task_prompt": "turn on the right sink handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {"camera_frame_path": str(frame), "camera_id": "head_pov"},
        "state": {"target_object_id": "Sink054_handle"},
        "unitree_g1_sonic_state": {"left_hand": [0.0, 0.1]},
        "safety_limits": {"max_joint_delta": 0.2},
    }


def test_harness_emits_derived_observation_artifacts_and_boundaries(tmp_path: Path) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    video = tmp_path / "generated.mp4"
    video.write_bytes(b"fixture-video")
    target = _target()

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_generated_video_path=video,
        source_policy_action={"action_type": "manipulation_contact", "gripper_command": "close"},
        action_history=[{"action_type": "manipulation_contact"}],
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        skeleton_conditioning={
            "projected_skeleton_trace_used": True,
            "projected_skeleton_trace_path": str(tmp_path / "skeleton.jsonl"),
        },
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={
            "rgb_only": True,
            "fields": ["camera_frame_path", "visual_observation"],
        },
    )

    paths = result["artifact_paths"]
    for key in (
        "wam_derived_observation_bundle",
        "wam_derived_observation_manifest",
        "wam_perception_harness_checks",
        "wam_policy_observation_adapter_report",
        "wam_derived_observation_steps",
    ):
        assert Path(paths[key]).is_file()

    step = result["step_record"]
    assert step["source_truth"]["capture_truth"] is False
    assert step["source_truth"]["physical_sensor_truth"] is False
    assert step["source_truth"]["derived_from_generated_pixels"] is True
    assert step["source_truth"]["model_derived_support_artifact"] is True
    assert step["harness_backend"]["kind"] == "fixture"
    assert step["harness_backend"]["real_sam_or_depth_model_ran"] is False
    assert step["robot_state"]["channel_truth"]["action_command_evaluator_controlled"] is True
    assert (
        step["robot_state"]["channel_truth"]["robot_state_inferred_by_sam_or_pixel_detector"]
        is False
    )
    assert step["objects"][0]["confidence"] > 0.8
    assert step["objects"][0]["mask_ref"]["physical_truth"] is False
    assert step["depth_estimates"][0]["sensor_depth"] is False
    assert step["depth_estimates"][0]["metric_depth_truth"] is False
    assert step["contact_likelihood"]["physical_contact_proven"] is False
    assert step["contact_likelihood"]["stable_grasp_proven"] is False
    assert step["consistency_checks"]["inverse_action_consistency"][
        "harness_does_not_prove_forward_inverse_consistency"
    ] is True
    assert result["checks"]["forward_inverse_consistency_proven"] is False

    adapter = result["policy_adapter_report"]
    assert adapter["adapter_status"] == "completed"
    assert "objects" in adapter["fields_withheld_due_to_contract"]
    assert "depth_estimates" in adapter["fields_withheld_due_to_contract"]
    assert "objects" not in adapter["adapted_policy_observation"]
    assert "depth_estimates" not in adapter["adapted_policy_observation"]


def test_mask_depth_capable_policy_receives_declared_enriched_fields(tmp_path: Path) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    target = _target()

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        policy_id="rgbd_mask_policy",
        declared_policy_observation_schema={
            "modalities": ["rgb", "depth", "mask", "state"],
            "fields": ["objects", "depth_estimates", "robot_state", "contact_likelihood"],
            "supports_depth": True,
            "supports_masks": True,
            "supports_state": True,
        },
    )

    adapted = result["adapted_policy_observation"]
    assert adapted["objects"][0]["object_id"] == "Sink054_handle"
    assert adapted["depth_estimates"][0]["sensor_depth"] is False
    assert adapted["robot_state"]["channel_truth"]["robot_state_inferred_by_sam_or_pixel_detector"] is False
    assert adapted["contact_likelihood"]["physical_contact_proven"] is False
    assert result["policy_adapter_report"]["safe_for_policy_requery"] is True


def test_low_confidence_lost_target_triggers_early_termination(tmp_path: Path) -> None:
    first_frame = _write_frame(tmp_path / "first.jpg")
    first_target = _target(object_id="Sink054_handle")
    first_step = build_wam_derived_observation_step(
        generated_at="now",
        step_index=1,
        source_generated_frame_path=first_frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(first_frame),
        object_index={"objects": [first_target]},
        eval_ready_task_grounding=_grounding(first_target),
    )

    second_frame = _write_frame(tmp_path / "second.jpg", dark=True)
    lost_target = _target(
        object_id="wrong_object",
        bbox={"x": 900, "y": 240, "width": 120, "height": 80},
    )
    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=2,
        source_generated_frame_path=second_frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(second_frame),
        object_index={"objects": [lost_target]},
        eval_ready_task_grounding=_grounding(lost_target),
        previous_steps=[first_step],
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={
            "rgb_only": True,
            "fields": ["camera_frame_path", "visual_observation"],
        },
    )

    step = result["step_record"]
    assert step["objects"][0]["temporal_stability"]["stable"] is False
    assert step["uncertainty"]["early_termination_recommended"] is True
    assert step["scoring_allowed"]["usable_for_policy_requery"] is False
    assert step["scoring_allowed"]["usable_for_success_scoring"] is False
    assert result["policy_adapter_report"]["safe_for_policy_requery"] is False
    assert result["checks"]["success_scoring_blocked"] is True
    assert "object_identity_lost_or_changed" in step["blockers"]


def test_external_backend_command_supplies_replaceable_perception_outputs(
    tmp_path: Path,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    script = tmp_path / "backend.py"
    script.write_text(
        "\n".join(
            [
                "import json, os",
                "output = os.environ['BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT']",
                "payload = {",
                "  'schema_version': 'wam_perception_backend_result.v1',",
                "  'status': 'completed',",
                "  'backend': {",
                "    'kind': 'external_command',",
                "    'status': 'completed',",
                "    'real_sam_or_depth_model_ran': True,",
                "  },",
                "  'objects': [{",
                "    'object_id': 'backend_target',",
                "    'track_id': 'backend_target',",
                "    'label': 'backend target',",
                "    'bbox': [120, 120, 220, 220],",
                "    'confidence': 0.88,",
                "    'source': 'fixture_external_detector'",
                "  }],",
                "  'depth_estimates': [{",
                "    'object_id': 'backend_target',",
                "    'relative_depth': 0.42,",
                "    'metric_depth': 0.7,",
                "    'confidence': 0.67",
                "  }],",
                "  'pose_estimates': [{",
                "    'object_id': 'backend_target',",
                "    'pose_2d': {'center_px': [170, 170]},",
                "    'confidence': 0.61",
                "  }],",
                "  'contact_likelihood': {'value': 0.53, 'confidence': 0.58}",
                "}",
                "open(output, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        backend_kind="external_command",
        backend_command=[sys.executable, str(script)],
        allow_external_backend=True,
        policy_id="rgbd_mask_policy",
        declared_policy_observation_schema={
            "modalities": ["rgb", "depth", "mask", "state"],
            "fields": ["objects", "depth_estimates", "pose_estimates", "contact_likelihood"],
            "supports_depth": True,
            "supports_masks": True,
            "supports_state": True,
        },
    )

    step = result["step_record"]
    assert step["harness_backend"]["kind"] == "external_command"
    assert step["harness_backend"]["real_sam_or_depth_model_ran"] is True
    assert Path(step["harness_backend"]["request_path"]).is_file()
    assert Path(step["harness_backend"]["result_path"]).is_file()
    assert step["objects"][0]["object_id"] == "backend_target"
    assert step["depth_estimates"][0]["metric_depth"] == 0.7
    assert step["depth_estimates"][0]["sensor_depth"] is False
    assert step["contact_likelihood"]["physical_contact_proven"] is False


def test_external_backend_fails_closed_without_gate(tmp_path: Path) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        backend_kind="sam3",
        backend_command=[sys.executable, "-c", "raise SystemExit(0)"],
        allow_external_backend=False,
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    step = result["step_record"]
    assert step["harness_backend"]["kind"] == "sam3"
    assert step["harness_backend"]["status"] == "blocked"
    assert "external_perception_backend_env_gate_not_enabled" in step["harness_backend"]["blockers"]
    assert step["harness_backend"]["real_sam_or_depth_model_ran"] is False


def test_multiview_calibration_validation_and_review_report_surfaces(
    tmp_path: Path,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    wrist = _write_frame(tmp_path / "wrist.jpg", size=(320, 240))
    target = _target(width_m=0.2)
    calibration = {
        "source": "capture_sim_fixture_calibration",
        "intrinsics": {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480,
        },
        "camera_calibration_quality_gate": {
            "schema_version": "camera_calibration_quality_gate.v1",
            "status": "passed",
            "confidence": 0.9,
        },
    }

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_generated_multiview_frame_paths={
            "head_pov": str(frame),
            "wrist": str(wrist),
        },
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        camera_calibration=calibration,
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
        validation_set=[
            {
                "step_index": 1,
                "expected_object_id": "Sink054_handle",
                "expected_target_visible": True,
                "expected_contact": True,
                "actual_success": True,
                "plain_video_success": True,
            }
        ],
    )

    step = result["step_record"]
    assert step["consistency_checks"]["multiview_consistent"]["status"] == "passed"
    assert step["depth_estimates"][0]["metric_depth"] is not None
    assert step["depth_estimates"][0]["metric_depth_truth"] is False
    assert step["pose_estimates"][0]["pose_3d"]["physical_pose_truth"] is False
    assert step["task_grounding"]["target_prompts"] == ["right sink handle"]
    assert result["validation_report"]["status"] == "completed"
    assert result["validation_report"]["metrics"]["object_id_accuracy"] == 1.0
    assert result["false_success_reduction_metrics"]["status"] == "completed"
    report_path = Path(result["artifact_paths"]["wam_perception_harness_review_report"])
    assert report_path.is_file()
    assert "Claim Boundary" in report_path.read_text(encoding="utf-8")


def test_review_acceptance_can_unblock_success_scoring_for_low_confidence_step(
    tmp_path: Path,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg", dark=True)
    target = _target(bbox={"x": 900, "y": 240, "width": 120, "height": 80})

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        review_acceptance={
            "accepted_for_success_scoring": True,
            "reviewer_id": "owner-reviewer",
            "evidence_refs": [str(frame)],
        },
        validation_set=[
            {
                "step_index": 1,
                "expected_object_id": "Sink054_handle",
                "expected_target_visible": False,
                "actual_success": False,
                "plain_video_success": True,
            }
        ],
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    step = result["step_record"]
    assert step["uncertainty"]["early_termination_recommended"] is True
    assert step["rollout_reviewability"]["review_acceptance"]["status"] == "accepted"
    assert step["scoring_allowed"]["success_scoring_review_accepted"] is True
    assert step["scoring_allowed"]["usable_for_success_scoring"] is True
    assert result["checks"]["success_scoring_blocked"] is False


def test_false_success_metrics_show_reduction_when_harness_blocks_bad_video_score(
    tmp_path: Path,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg", dark=True)
    target = _target(bbox={"x": 900, "y": 240, "width": 120, "height": 80})

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        validation_set=[
            {
                "step_index": 1,
                "expected_object_id": "Sink054_handle",
                "expected_target_visible": False,
                "actual_success": False,
                "plain_video_success": True,
            }
        ],
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    metrics = result["false_success_reduction_metrics"]
    assert metrics["plain_video_false_success_count"] == 1
    assert metrics["harness_false_success_after_gating_count"] == 0
    assert metrics["false_success_reduction_count"] == 1
    assert metrics["false_success_reduction_rate"] == 1.0


def test_real_provider_probe_validation_status_requires_capture_backed_labeled_rows(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "empty_validation.json"
    empty_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    empty_status = _validation_status(empty_path)
    assert empty_status["status"] == "blocked"
    assert "validation_set_rows_missing" in empty_status["blockers"]
    assert "capture_backed_validation_rows_missing" in empty_status["blockers"]

    fixture_path = tmp_path / "fixture_validation.json"
    fixture_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "actual_success": False,
                        "plain_video_success": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fixture_status = _validation_status(fixture_path)
    assert fixture_status["status"] == "blocked"
    assert "capture_backed_validation_rows_missing" in fixture_status["blockers"]
    assert "real_labeled_validation_rows_missing" in fixture_status["blockers"]

    source_less_path = tmp_path / "source_less_validation.json"
    source_less_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "actual_success": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_less_status = _validation_status(source_less_path)
    assert source_less_status["status"] == "blocked"
    assert "real_labeled_validation_source_missing" in source_less_status["blockers"]

    real_path = tmp_path / "real_validation.json"
    real_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "actual_success": False,
                        "source_capture_path": "captures/kitchen-run-001",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    real_status = _validation_status(real_path)
    assert real_status["status"] == "available"
    assert real_status["real_labeled_row_count"] == 1
    assert real_status["sourced_real_labeled_row_count"] == 1


def test_sim_provider_e2e_fixture_mode_runs_multistep_harness_and_adapter(
    tmp_path: Path,
) -> None:
    frame = _write_frame(tmp_path / "gpt_image2_start.jpg")
    args = argparse.Namespace(
        output_dir=tmp_path / "sim_e2e",
        generated_frame=frame,
        step_count=2,
        target_prompt="robot arm",
        policy_id="sim_rgbd_policy",
        policy_schema="rgbd_mask_pose",
        provider_mode="fixture",
        sam3_weights=None,
        sam3_confidence=0.01,
        pose_model="yolo11n-pose.pt",
        depth_provider="v2",
        depth_model_id="depth-anything/Depth-Anything-V2-Small-hf",
        da3_model_id="depth-anything/DA3-BASE",
        backend_timeout_seconds=60,
    )

    manifest = run_sim_provider_e2e(args)

    assert manifest["status"] == "completed"
    assert manifest["provider_mode"] == "fixture"
    assert manifest["step_count_completed"] == 2
    assert manifest["policy_requery_count"] == 2
    assert manifest["sim_only_provider_harness_e2e_completed"] is True
    assert manifest["perception_accuracy_validated"] is False
    assert manifest["claim_boundary"]["generated_frames_are_not_capture_truth"] is True
    assert manifest["claim_boundaries"]["generated_frames_are_not_capture_truth"] is True
    assert manifest["claim_boundary"]["physical_robot_readiness_proven"] is False
    assert Path(manifest["trace_path"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()
    assert Path(
        manifest["harness_artifact_paths"]["wam_derived_observation_steps"]
    ).is_file()
    trace_rows = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step_index"] for row in trace_rows] == [1, 2]
    assert all(row["safe_for_policy_requery"] for row in trace_rows)
    assert all(Path(row["generated_frame_path"]).is_file() for row in trace_rows)
    assert all(row["contact_likelihood"]["physical_contact_proven"] is False for row in trace_rows)
    assert all(row["uncertainty"]["early_termination_recommended"] is False for row in trace_rows)
    assert all(row["scoring_allowed"]["usable_for_policy_requery"] is True for row in trace_rows)


def test_real_provider_validation_probe_fails_closed_without_real_provider_inputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    for name in (
        "SAM3_WEIGHTS_PATH",
        "BLUEPRINT_SAM3_WEIGHTS_PATH",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_MODEL_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    frame = _write_frame(tmp_path / "generated.jpg")

    manifest = run_probe(
        output_dir=tmp_path / "proof",
        generated_frame_path=frame,
        validation_set_path=None,
        policy_id="rgbd_mask_pose_policy",
        policy_observation_schema={
            "modalities": ["rgb", "depth", "mask", "pose", "state"],
            "fields": [
                "camera_frame_path",
                "visual_observation",
                "objects",
                "depth_estimates",
                "pose_estimates",
            ],
        },
    )

    assert manifest["status"] == "blocked"
    assert manifest["proof_scope"]["real_sam3_depth_pose_proof_complete"] is False
    assert manifest["proof_scope"]["perception_accuracy_validated"] is False
    assert "sam3_weights_path_missing" in manifest["blockers"]
    assert "no_real_sam3_depth_or_pose_provider_ran" in manifest["blockers"]
    assert "validation_set_path_not_supplied" in manifest["blockers"]
    assert manifest["provider_readiness"]["sam3"]["hf_token_env_presence"]["HF_TOKEN"] is False
    assert Path(manifest["manifest_path"]).is_file()
    assert Path(
        manifest["harness_artifact_paths"]["wam_perception_harness_validation_report"]
    ).is_file()
    assert manifest["claim_boundary"]["inferred_depth_is_not_sensor_depth"] is True
    assert (
        manifest["claim_boundary"][
            "generated_rollout_or_harness_outputs_do_not_prove_real_world_success"
        ]
        is True
    )


def test_real_provider_backend_contract_records_missing_provider_blockers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    for name in (
        "SAM3_WEIGHTS_PATH",
        "BLUEPRINT_SAM3_WEIGHTS_PATH",
        "BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_MODEL_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    frame = _write_frame(tmp_path / "generated.jpg")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "backend_result.json"
    request_path.write_text(
        "{"
        f'"source_generated_frame_path": {str(frame)!r},'
        '"eval_ready_task_grounding": {"task": {"target_prompts_for_object_index_backends": ["red block"]}}'
        "}".replace("'", '"'),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_INPUT", str(request_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT", str(output_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_JOB_DIR", str(tmp_path))

    assert run_external_backend_from_env() == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["backend"]["kind"] == "real_provider_probe"
    assert payload["backend"]["real_sam_or_depth_model_ran"] is False
    assert "sam3_weights_path_missing" in payload["backend"]["blockers"]
    assert "depth_provider_command_not_configured" in payload["backend"]["blockers"]
    assert "no_real_sam3_depth_or_pose_provider_ran" in payload["backend"]["blockers"]
    assert payload["claim_boundary"]["estimated_depth_is_not_sensor_depth"] is True
