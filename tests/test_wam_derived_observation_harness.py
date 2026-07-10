from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


pytestmark = [pytest.mark.slow, pytest.mark.integration]
pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from blueprint_pipeline import wam_real_provider_validation_probe as real_probe
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


def test_external_backend_timeout_writes_blocked_artifacts(tmp_path: Path) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        backend_kind="external_command",
        backend_command=[
            sys.executable,
            "-c",
            "import time; print('started'); time.sleep(5)",
        ],
        allow_external_backend=True,
        backend_timeout_seconds=1,
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    backend = result["step_record"]["harness_backend"]
    assert backend["status"] == "blocked"
    assert "external_perception_backend_command_timed_out" in backend["blockers"]
    assert Path(backend["stdout_path"]).is_file()
    assert Path(backend["stderr_path"]).is_file()


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
                "capture_backed": True,
                "source_capture_path": "captures/kitchen-run-001",
                "reviewer_id": "operator-reviewer-1",
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


def test_multiview_duplicate_frame_paths_are_not_cross_view_proof(
    tmp_path: Path,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    target = _target(width_m=0.2)

    result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "harness",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_generated_multiview_frame_paths={
            "head_pov": str(frame),
            "wrist_alias": str(frame),
        },
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    multiview = result["step_record"]["consistency_checks"]["multiview_consistent"]
    assert multiview["status"] == "not_evaluated"
    assert multiview["passed"] is None
    assert multiview["real_multiview_available"] is False
    assert multiview["distinct_frame_path_count"] == 1
    assert "fewer_than_two_distinct_generated_views" in multiview["blockers"]
    assert multiview["claim_boundary"][
        "cross_view_consistency_requires_two_distinct_generated_camera_views"
    ] is True


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
                "capture_backed": True,
                "source_capture_path": "captures/kitchen-run-001",
                "reviewer_id": "operator-reviewer-1",
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
                "capture_backed": True,
                "source_capture_path": "captures/kitchen-run-001",
                "reviewer_id": "operator-reviewer-1",
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


def test_harness_validation_rows_without_labels_are_diagnostic_only(tmp_path: Path) -> None:
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
        validation_set=[{"step_index": 1}],
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    assert result["validation_report"]["status"] == "diagnostic_issues"
    assert result["validation_report"]["blockers"] == []
    assert "diagnostic_blockers" not in result["validation_report"]
    assert "validation_row_label_missing" in result["validation_report"]["diagnostic_issues"]
    assert result["false_success_reduction_metrics"]["status"] == "not_measured"
    assert result["false_success_reduction_metrics"]["blockers"] == []
    assert (
        "validation_row_label_missing"
        in result["false_success_reduction_metrics"]["diagnostic_issues"]
    )


def test_false_success_metrics_not_measured_without_accepted_labels(tmp_path: Path) -> None:
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
                "expected_target_visible": False,
                "actual_success": False,
                "plain_video_success": True,
            }
        ],
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    assert result["validation_report"]["status"] == "diagnostic_issues"
    assert result["validation_report"]["matched_step_count"] == 0
    assert "validation_row_not_capture_backed_or_accepted_anchor" in result[
        "validation_report"
    ]["diagnostic_issues"]
    assert result["false_success_reduction_metrics"]["status"] == "not_measured"
    assert result["false_success_reduction_metrics"]["plain_video_false_success_count"] == (
        "not_measured"
    )
    assert result["false_success_reduction_metrics"]["blockers"] == []


def test_external_consistency_requires_external_scorer_evidence(tmp_path: Path) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    target = _target()

    bare_claim = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "bare_claim",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        external_consistency={"forward_inverse_consistency_proven": True},
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    inverse = bare_claim["step_record"]["consistency_checks"]["inverse_action_consistency"]
    assert inverse["status"] == "separate_external_scorer_required"
    assert inverse["forward_inverse_consistency_proven"] is False
    assert "external_episode_consistency_scorer_run_not_proven" in inverse["blockers"]
    assert bare_claim["checks"]["forward_inverse_consistency_proven"] is False

    scorer_result = run_wam_derived_observation_harness_step(
        output_dir=tmp_path / "scorer_result",
        generated_at="now",
        step_index=1,
        source_generated_frame_path=frame,
        source_policy_action={"action_type": "manipulation_contact"},
        current_policy_observation=_observation(frame),
        object_index={"objects": [target]},
        eval_ready_task_grounding=_grounding(target),
        external_consistency={
            "forward_inverse_consistency_proven": True,
            "external_episode_consistency_scorer_ran": True,
            "external_episode_consistency_scorer_id": "external-vlm-consistency",
            "policy_success_claimed_from_consistency": True,
            "task_success_claimed_from_consistency": True,
            "rank_fidelity_claimed_from_consistency": True,
            "deployment_readiness_claimed_from_consistency": True,
            "sensor_truth_claimed_from_consistency": True,
            "external_validation_claimed_from_consistency": True,
            "public_claim_upgrade_allowed": True,
            "rollout_checks": [
                {
                    "rollout_id": "rollout-1",
                    "forward_consistent": True,
                    "inverse_consistent": True,
                    "visual_evidence_used": True,
                    "action_trace_evidence_used": True,
                }
            ],
        },
        policy_id="rgb_only_policy",
        declared_policy_observation_schema={"rgb_only": True},
    )

    inverse = scorer_result["step_record"]["consistency_checks"]["inverse_action_consistency"]
    assert inverse["status"] == "external_scorer_passed"
    assert inverse["forward_inverse_consistency_proven"] is True
    assert inverse["external_episode_consistency_scorer_id"] == "external-vlm-consistency"
    assert inverse["forward_inverse_consistency_is_reliability_review_signal_only"] is True
    assert inverse[
        "forward_inverse_consistency_does_not_upgrade_evaluator_bounded_policy_ranking"
    ] is True
    assert inverse["policy_success_claimed_from_consistency"] is False
    assert inverse["task_success_claimed_from_consistency"] is False
    assert inverse["rank_fidelity_claimed_from_consistency"] is False
    assert inverse["deployment_readiness_claimed_from_consistency"] is False
    assert inverse["sensor_truth_claimed_from_consistency"] is False
    assert inverse["external_validation_claimed_from_consistency"] is False
    assert inverse["claim_boundary"]["forward_inverse_consistency_is_not_external_validation"] is True
    assert scorer_result["checks"]["forward_inverse_consistency_proven"] is False
    assert scorer_result["checks"]["consistency_metrics_are_support_signals_only"] is True
    assert scorer_result["checks"]["task_success_claimed_from_consistency"] is False
    assert scorer_result["bundle"]["claim_boundary"][
        "forward_inverse_consistency_does_not_prove_sensor_truth"
    ] is True
    assert scorer_result["manifest"]["claim_boundary"][
        "deployment_readiness_claimed_from_consistency"
    ] is False


def test_real_provider_probe_validation_status_requires_capture_backed_labeled_rows(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "empty_validation.json"
    empty_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    empty_status = _validation_status(empty_path)
    assert empty_status["status"] == "diagnostic_issues"
    assert "diagnostic_blockers" not in empty_status
    assert "validation_set_rows_missing" in empty_status["diagnostic_issues"]
    assert "capture_backed_validation_rows_missing" in empty_status["diagnostic_issues"]

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
    assert fixture_status["status"] == "diagnostic_issues"
    assert "capture_backed_validation_rows_missing" in fixture_status["diagnostic_issues"]
    assert "real_labeled_validation_rows_missing" in fixture_status["diagnostic_issues"]

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
    assert source_less_status["status"] == "diagnostic_issues"
    assert "real_labeled_validation_source_missing" in source_less_status["diagnostic_issues"]

    real_path = tmp_path / "real_validation.json"
    real_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "target_prompt": "robot arm",
                        "actual_success": False,
                        "source_capture_path": "captures/kitchen-run-001",
                        "reviewer_id": "operator-reviewer-1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    real_status = _validation_status(real_path)
    assert real_status["status"] == "available"
    assert real_status["diagnostic_issues"] == []
    assert real_status["real_labeled_row_count"] == 1
    assert real_status["sourced_real_labeled_row_count"] == 1
    assert real_status["accepted_contract_row_count"] == 1


def test_real_provider_probe_validation_status_reports_bad_rows_as_diagnostics(
    tmp_path: Path,
) -> None:
    expected_frame = _write_frame(tmp_path / "probe_frame_001.jpg")

    missing_label_path = tmp_path / "missing_label_validation.json"
    missing_label_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "target_prompt": "robot arm",
                        "source_capture_path": "captures/kitchen-run-001",
                        "reviewer_id": "operator-reviewer-1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    missing_label_status = _validation_status(
        missing_label_path,
        expected_frame_path=expected_frame,
        target_prompts=["robot arm"],
    )
    assert missing_label_status["status"] == "diagnostic_issues"
    assert "diagnostic_blockers" not in missing_label_status
    assert "row_validation_label_missing" in missing_label_status["diagnostic_issues"]
    assert "blockers" not in missing_label_status["row_results"][0]
    assert "row_validation_label_missing" in missing_label_status["row_results"][0][
        "diagnostic_issues"
    ]

    mismatched_frame_path = tmp_path / "mismatched_frame_validation.json"
    mismatched_frame_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "target_prompt": "robot arm",
                        "actual_success": False,
                        "source_capture_path": "captures/kitchen-run-001",
                        "reviewer_id": "operator-reviewer-1",
                        "frame_id": "different_frame_999",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    mismatched_frame_status = _validation_status(
        mismatched_frame_path,
        expected_frame_path=expected_frame,
        target_prompts=["robot arm"],
    )
    assert mismatched_frame_status["status"] == "diagnostic_issues"
    assert "row_frame_id_or_path_mismatch" in mismatched_frame_status["diagnostic_issues"]
    assert mismatched_frame_status["accepted_contract_row_count"] == 0

    empty_target_path = tmp_path / "empty_target_validation.json"
    empty_target_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "target_prompt": " ",
                        "actual_success": False,
                        "source_capture_path": "captures/kitchen-run-001",
                        "reviewer_id": "operator-reviewer-1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    empty_target_status = _validation_status(
        empty_target_path,
        expected_frame_path=expected_frame,
        target_prompts=[],
    )
    assert empty_target_status["status"] == "diagnostic_issues"
    assert "validation_target_prompt_empty" in empty_target_status["diagnostic_issues"]
    assert "row_target_prompt_empty" in empty_target_status["diagnostic_issues"]

    provider_only_path = tmp_path / "provider_only_validation.json"
    provider_only_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "target_prompt": "robot arm",
                        "actual_success": False,
                        "source_artifact_path": "wam_perception_backend_result.json",
                        "reviewer_id": "operator-reviewer-1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provider_only_status = _validation_status(
        provider_only_path,
        expected_frame_path=expected_frame,
        target_prompts=["robot arm"],
    )
    assert provider_only_status["status"] == "diagnostic_issues"
    assert (
        "provider_only_validation_source_not_accepted"
        in provider_only_status["diagnostic_issues"]
    )
    assert "row_source_is_provider_only_output" in provider_only_status["diagnostic_issues"]

    bad_provenance_path = tmp_path / "bad_provenance_validation.json"
    bad_provenance_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "target_prompt": "robot arm",
                        "actual_success": False,
                        "source_capture_path": "captures/kitchen-run-001",
                        "reviewer_id": "<reviewer>",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bad_provenance_status = _validation_status(
        bad_provenance_path,
        expected_frame_path=expected_frame,
        target_prompts=["robot arm"],
    )
    assert bad_provenance_status["status"] == "diagnostic_issues"
    assert (
        "real_labeled_validation_provenance_missing"
        in bad_provenance_status["diagnostic_issues"]
    )
    assert (
        "row_reviewer_or_label_provenance_missing"
        in bad_provenance_status["diagnostic_issues"]
    )


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
    assert manifest["optional_truth_label_validation_requested"] is False
    assert manifest["claim_boundary"]["generated_frames_are_not_capture_truth"] is True
    assert manifest["claim_boundaries"]["generated_frames_are_not_capture_truth"] is True
    assert manifest["claim_boundary"]["generated_world_rank_fidelity_result_proven"] is False
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
        "BLUEPRINT_WAM_ALLOW_SAM3_ULTRALYTICS_AUTODOWNLOAD",
        "BLUEPRINT_WAM_SAM3_MODEL",
        "BLUEPRINT_WAM_SAM3_PROVIDER_KIND",
        "BLUEPRINT_WAM_ALLOW_SAM3_TRANSFORMERS_PROVIDER",
        "BLUEPRINT_WAM_SAM3_HF_MODEL_ID",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN_FILE",
        "HUGGINGFACE_HUB_TOKEN_FILE",
        "HUGGING_FACE_HUB_TOKEN_FILE",
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
    assert manifest["proof_scope"]["optional_labeled_validation_requested"] is False
    assert manifest["proof_scope"]["optional_labeled_validation_completed"] is False
    assert "sam3_weights_path_missing" in manifest["blockers"]
    assert "no_real_sam3_depth_or_pose_provider_ran" in manifest["blockers"]
    assert manifest["validation_set"]["status"] == "not_requested"
    assert manifest["validation_set"]["diagnostic_issues"] == []
    assert manifest["optional_validation_diagnostic_issues"] == []
    assert manifest["validation_report"]["diagnostic_issues"] == []
    assert manifest["false_success_reduction_metrics"]["diagnostic_issues"] == []
    assert "optional_validation_diagnostic_blockers" not in manifest
    assert not any("validation_set" in blocker for blocker in manifest["blockers"])
    assert manifest["provider_readiness"]["sam3"]["hf_token_env_presence"]["HF_TOKEN"] is False
    assert Path(manifest["manifest_path"]).is_file()
    assert Path(
        manifest["harness_artifact_paths"]["wam_perception_harness_validation_report"]
    ).is_file()
    assert manifest["claim_boundary"]["inferred_depth_is_not_sensor_depth"] is True
    assert (
        manifest["claim_boundary"][
            "generated_rollout_or_harness_outputs_do_not_prove_accepted_anchor_success"
        ]
        is True
    )


def test_real_provider_validation_probe_can_make_pose_optional(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    for name in (
        "SAM3_WEIGHTS_PATH",
        "BLUEPRINT_SAM3_WEIGHTS_PATH",
        "BLUEPRINT_WAM_ALLOW_SAM3_ULTRALYTICS_AUTODOWNLOAD",
        "BLUEPRINT_WAM_SAM3_MODEL",
        "BLUEPRINT_WAM_SAM3_PROVIDER_KIND",
        "BLUEPRINT_WAM_ALLOW_SAM3_TRANSFORMERS_PROVIDER",
        "BLUEPRINT_WAM_SAM3_HF_MODEL_ID",
        "BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_MODEL_PATH",
        "BLUEPRINT_ALLOW_WAM_AUTO_POSE_PROVIDER",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN_FILE",
        "HUGGINGFACE_HUB_TOKEN_FILE",
        "HUGGING_FACE_HUB_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    frame = _write_frame(tmp_path / "generated.jpg")

    manifest = run_probe(
        output_dir=tmp_path / "proof",
        generated_frame_path=frame,
        validation_set_path=None,
        target_prompts=["closed refrigerator door"],
        policy_id="rgbd_mask_policy",
        policy_observation_schema={
            "modalities": ["rgb", "depth", "mask", "state"],
            "fields": [
                "camera_frame_path",
                "visual_observation",
                "objects",
                "depth_estimates",
            ],
        },
        require_pose=False,
    )

    assert manifest["status"] == "blocked"
    assert manifest["proof_scope"]["real_pose_provider_required"] is False
    assert manifest["proof_scope"]["real_sam3_depth_proof_complete"] is False
    assert manifest["proof_scope"]["real_provider_requirement_completed"] is False
    assert "pose_provider_command_not_configured" not in manifest["blockers"]
    assert all(
        "pose_model" not in str(blocker)
        for blocker in manifest["harness_backend"]["blockers"]
    )


def test_real_provider_validation_probe_treats_bad_labels_as_diagnostics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    for name in (
        "SAM3_WEIGHTS_PATH",
        "BLUEPRINT_SAM3_WEIGHTS_PATH",
        "BLUEPRINT_WAM_ALLOW_SAM3_ULTRALYTICS_AUTODOWNLOAD",
        "BLUEPRINT_WAM_SAM3_MODEL",
        "BLUEPRINT_WAM_SAM3_PROVIDER_KIND",
        "BLUEPRINT_WAM_ALLOW_SAM3_TRANSFORMERS_PROVIDER",
        "BLUEPRINT_WAM_SAM3_HF_MODEL_ID",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN_FILE",
        "HUGGINGFACE_HUB_TOKEN_FILE",
        "HUGGING_FACE_HUB_TOKEN_FILE",
        "BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_PROVIDER_COMMAND",
        "BLUEPRINT_WAM_POSE_MODEL_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    frame = _write_frame(tmp_path / "generated.jpg")
    validation_path = tmp_path / "missing_label_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step_index": 1,
                        "capture_backed": True,
                        "target_prompt": "robot arm",
                        "source_capture_path": "captures/kitchen-run-001",
                        "reviewer_id": "operator-reviewer-1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = run_probe(
        output_dir=tmp_path / "proof",
        generated_frame_path=frame,
        validation_set_path=validation_path,
        target_prompts=["robot arm"],
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
    assert "sam3_weights_path_missing" in manifest["blockers"]
    assert "row_validation_label_missing" not in manifest["blockers"]
    assert "real_labeled_validation_rows_missing" not in manifest["blockers"]
    assert manifest["validation_set"]["status"] == "diagnostic_issues"
    assert "diagnostic_blockers" not in manifest["validation_set"]
    assert "row_validation_label_missing" in manifest["validation_set"]["diagnostic_issues"]
    assert (
        "row_validation_label_missing"
        in manifest["optional_validation_diagnostic_issues"]
    )
    assert manifest["false_success_reduction_metrics"]["status"] == "not_measured"


def test_real_provider_backend_contract_records_missing_provider_blockers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    for name in (
        "SAM3_WEIGHTS_PATH",
        "BLUEPRINT_SAM3_WEIGHTS_PATH",
        "BLUEPRINT_WAM_ALLOW_SAM3_ULTRALYTICS_AUTODOWNLOAD",
        "BLUEPRINT_WAM_SAM3_MODEL",
        "BLUEPRINT_WAM_SAM3_PROVIDER_KIND",
        "BLUEPRINT_WAM_ALLOW_SAM3_TRANSFORMERS_PROVIDER",
        "BLUEPRINT_WAM_SAM3_HF_MODEL_ID",
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


def test_real_provider_backend_records_ultralytics_sam3_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    weights = tmp_path / "sam3.pt"
    weights.write_bytes(b"fake-weights")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "backend_result.json"
    request_path.write_text(
        json.dumps(
            {
                "source_generated_frame_path": str(frame),
                "eval_ready_task_grounding": {
                    "task": {"target_prompts_for_object_index_backends": ["red block"]}
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeSAM3SemanticPredictor:
        def __init__(self, overrides: dict[str, Any]) -> None:
            self.overrides = overrides

        def set_image(self, path: str) -> None:
            assert Path(path).is_file()

        def __call__(self, text: list[str]) -> list[Any]:
            assert text == ["red block"]
            boxes = types.SimpleNamespace(
                xyxy=np.array([[1.0, 2.0, 30.0, 40.0]], dtype=np.float32),
                conf=np.array([0.82], dtype=np.float32),
            )
            return [types.SimpleNamespace(boxes=boxes)]

    ultralytics_module = types.ModuleType("ultralytics")
    models_module = types.ModuleType("ultralytics.models")
    sam_module = types.ModuleType("ultralytics.models.sam")
    sam_module.SAM3SemanticPredictor = FakeSAM3SemanticPredictor
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)
    monkeypatch.setitem(sys.modules, "ultralytics.models", models_module)
    monkeypatch.setitem(sys.modules, "ultralytics.models.sam", sam_module)
    monkeypatch.setattr(
        real_probe,
        "_module_available",
        lambda name: name == "ultralytics",
    )
    depth_code = (
        "import json, os; "
        "json.dump({'depth_estimates':[{'object_id':'generated_frame','relative_depth':0.4}]}, "
        "open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8'))"
    )
    monkeypatch.setenv("SAM3_WEIGHTS_PATH", str(weights))
    monkeypatch.setenv("BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND", f"{sys.executable} -c {depth_code!r}")
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_INPUT", str(request_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT", str(output_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_JOB_DIR", str(tmp_path))
    monkeypatch.delenv("BLUEPRINT_WAM_POSE_PROVIDER_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_WAM_AUTO_POSE_PROVIDER", raising=False)

    assert run_external_backend_from_env() == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    sam3_status = payload["backend"]["provider_statuses"][0]
    assert payload["status"] == "completed"
    assert sam3_status["provider"] == "sam3"
    assert sam3_status["kind"] == "sam3_semantic_segmentation"
    assert sam3_status["model_family"] == "sam3"
    assert sam3_status["runtime_package"] == "ultralytics.models.sam"
    assert sam3_status["runtime_class"] == "SAM3SemanticPredictor"
    assert sam3_status["ran"] is True
    assert sam3_status["blockers"] == []


def test_real_provider_backend_uses_ultralytics_sam3_autodownload_model_ref(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "backend_result.json"
    request_path.write_text(
        json.dumps(
            {
                "source_generated_frame_path": str(frame),
                "eval_ready_task_grounding": {
                    "task": {"target_prompts_for_object_index_backends": ["closed refrigerator door"]}
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeSAM3SemanticPredictor:
        def __init__(self, overrides: dict[str, Any]) -> None:
            assert overrides["model"] == "sam3.pt"
            self.overrides = overrides

        def set_image(self, path: str) -> None:
            assert Path(path).is_file()

        def __call__(self, text: list[str]) -> list[Any]:
            assert text == ["closed refrigerator door"]
            boxes = types.SimpleNamespace(
                xyxy=np.array([[4.0, 8.0, 48.0, 64.0]], dtype=np.float32),
                conf=np.array([0.74], dtype=np.float32),
            )
            return [types.SimpleNamespace(boxes=boxes)]

    ultralytics_module = types.ModuleType("ultralytics")
    models_module = types.ModuleType("ultralytics.models")
    sam_module = types.ModuleType("ultralytics.models.sam")
    sam_module.SAM3SemanticPredictor = FakeSAM3SemanticPredictor
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)
    monkeypatch.setitem(sys.modules, "ultralytics.models", models_module)
    monkeypatch.setitem(sys.modules, "ultralytics.models.sam", sam_module)
    monkeypatch.setattr(
        real_probe,
        "_module_available",
        lambda name: name == "ultralytics",
    )
    depth_code = (
        "import json, os; "
        "json.dump({'depth_estimates':[{'object_id':'generated_frame','relative_depth':0.42}]}, "
        "open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8'))"
    )
    monkeypatch.delenv("SAM3_WEIGHTS_PATH", raising=False)
    monkeypatch.delenv("BLUEPRINT_SAM3_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("BLUEPRINT_WAM_ALLOW_SAM3_ULTRALYTICS_AUTODOWNLOAD", "true")
    monkeypatch.setenv("BLUEPRINT_WAM_SAM3_MODEL", "sam3.pt")
    monkeypatch.setenv("BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND", f"{sys.executable} -c {depth_code!r}")
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_INPUT", str(request_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT", str(output_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_JOB_DIR", str(tmp_path))
    monkeypatch.delenv("BLUEPRINT_WAM_POSE_PROVIDER_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_WAM_AUTO_POSE_PROVIDER", raising=False)

    assert run_external_backend_from_env() == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    sam3_status = payload["backend"]["provider_statuses"][0]
    assert payload["status"] == "completed"
    assert sam3_status["provider"] == "sam3"
    assert sam3_status["ran"] is True
    assert sam3_status["blockers"] == []
    assert sam3_status["weights_path_present"] is False
    assert sam3_status["weights_file_exists"] is False
    assert sam3_status["autodownload_enabled"] is True
    assert sam3_status["model_ref"] == "sam3.pt"
    assert sam3_status["model_ref_source"] == "ultralytics_autodownload"


def test_real_provider_backend_records_transformers_sam3_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "backend_result.json"
    request_path.write_text(
        json.dumps(
            {
                "source_generated_frame_path": str(frame),
                "eval_ready_task_grounding": {
                    "task": {"target_prompts_for_object_index_backends": ["closed refrigerator door"]}
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeOriginalSizes:
        def tolist(self) -> list[list[int]]:
            return [[480, 640]]

    class FakeInputs(dict):
        def to(self, device: str) -> "FakeInputs":
            self["device"] = device
            return self

    class FakeSam3Model:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeSam3Model":
            assert model_id == "facebook/sam3"
            assert kwargs == {
                "revision": real_probe.DEFAULT_SAM3_HF_MODEL_REVISION,
                "trust_remote_code": False,
            }
            return cls()

        def to(self, device: str) -> "FakeSam3Model":
            assert device
            return self

        def __call__(self, **inputs: Any) -> dict[str, Any]:
            assert inputs["device"]
            return {"ok": True}

    class FakeSam3Processor:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> "FakeSam3Processor":
            assert model_id == "facebook/sam3"
            assert kwargs == {
                "revision": real_probe.DEFAULT_SAM3_HF_MODEL_REVISION,
                "trust_remote_code": False,
            }
            return cls()

        def __call__(self, *, images: Any, text: str, return_tensors: str) -> FakeInputs:
            assert images.size == (640, 480)
            assert text == "closed refrigerator door"
            assert return_tensors == "pt"
            return FakeInputs({"original_sizes": FakeOriginalSizes()})

        def post_process_instance_segmentation(
            self,
            outputs: Any,
            *,
            threshold: float,
            mask_threshold: float,
            target_sizes: list[list[int]],
        ) -> list[dict[str, Any]]:
            assert outputs == {"ok": True}
            assert threshold == 0.05
            assert mask_threshold == 0.05
            assert target_sizes == [[480, 640]]
            return [
                {
                    "boxes": np.array([[6.0, 9.0, 72.0, 88.0]], dtype=np.float32),
                    "scores": np.array([0.91], dtype=np.float32),
                    "masks": np.ones((1, 480, 640), dtype=np.uint8),
                }
            ]

    transformers_module = types.ModuleType("transformers")
    transformers_module.Sam3Model = FakeSam3Model
    transformers_module.Sam3Processor = FakeSam3Processor
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    class _FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: Any) -> bool:
            return False

    # the provider imports torch for no_grad inference; keep the test hermetic on hosts
    # (and CI runners) without the real torch wheel, matching the transformers fake above
    torch_module = types.ModuleType("torch")
    torch_module.no_grad = _FakeNoGrad
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(
        real_probe,
        "_module_available",
        lambda name: name == "transformers",
    )
    depth_code = (
        "import json, os; "
        "json.dump({'depth_estimates':[{'object_id':'generated_frame','relative_depth':0.42}]}, "
        "open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8'))"
    )
    monkeypatch.delenv("SAM3_WEIGHTS_PATH", raising=False)
    monkeypatch.delenv("BLUEPRINT_SAM3_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("BLUEPRINT_WAM_SAM3_PROVIDER_KIND", "transformers")
    monkeypatch.setenv("BLUEPRINT_WAM_ALLOW_SAM3_TRANSFORMERS_PROVIDER", "true")
    monkeypatch.setenv("BLUEPRINT_WAM_SAM3_HF_MODEL_ID", "facebook/sam3")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_FILE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN_FILE", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN_FILE", raising=False)
    monkeypatch.setenv("BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND", f"{sys.executable} -c {depth_code!r}")
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_INPUT", str(request_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT", str(output_path))
    monkeypatch.setenv("BLUEPRINT_WAM_PERCEPTION_BACKEND_JOB_DIR", str(tmp_path))
    monkeypatch.delenv("BLUEPRINT_WAM_POSE_PROVIDER_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_WAM_AUTO_POSE_PROVIDER", raising=False)

    assert run_external_backend_from_env() == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    sam3_status = payload["backend"]["provider_statuses"][0]
    assert payload["status"] == "completed"
    assert sam3_status["provider"] == "sam3"
    assert sam3_status["kind"] == "transformers_sam3"
    assert sam3_status["runtime_package"] == "transformers"
    assert sam3_status["runtime_class"] == "Sam3Model/Sam3Processor"
    assert sam3_status["ran"] is True
    assert sam3_status["blockers"] == []
    assert sam3_status["model_id"] == "facebook/sam3"
    assert sam3_status["model_revision"] == real_probe.DEFAULT_SAM3_HF_MODEL_REVISION
    assert sam3_status["model_remote_code_trusted"] is False
    assert sam3_status["transformers_provider_enabled"] is True
    assert payload["objects"][0]["source"] == "sam3_transformers_from_generated_pixels"
    assert Path(payload["objects"][0]["mask_path"]).is_file()


def test_transformers_sam3_rejects_unapproved_revision_before_model_load(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    frame = _write_frame(tmp_path / "generated.jpg")
    monkeypatch.setenv(real_probe.SAM3_TRANSFORMERS_ENV, "true")
    monkeypatch.setenv(real_probe.SAM3_HF_REVISION_ENV, "0" * 40)
    monkeypatch.setattr(real_probe, "_module_available", lambda name: name == "transformers")

    objects, status = real_probe._run_transformers_sam3_provider(
        {
            "source_generated_frame_path": str(frame),
            "target_prompts": ["object"],
        },
        tmp_path,
    )

    assert objects == []
    assert status["ran"] is False
    assert status["blockers"] == ["sam3_model_revision_not_approved"]
