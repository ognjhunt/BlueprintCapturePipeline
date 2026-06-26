from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from blueprint_pipeline.wam_generated_video_review import (
    assess_source_policy_observation_visual_qa,
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
    write_persistent_wam_visual_quality_artifacts,
)


def _write_good_frame(path: Path, *, size: tuple[int, int] = (320, 256)) -> Path:
    width, height = size
    x_gradient = np.tile(np.linspace(48, 210, width, dtype=np.uint8), (height, 1))
    y_gradient = np.tile(np.linspace(32, 120, height, dtype=np.uint8), (width, 1)).T
    frame = np.dstack((x_gradient, np.roll(x_gradient, 32, axis=1), y_gradient))
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 2 - 42, height // 2 - 30, width // 2 + 42, height // 2 + 30), outline=(255, 255, 255), width=4)
    draw.ellipse((width // 2 - 14, height // 2 - 14, width // 2 + 14, height // 2 + 14), fill=(230, 70, 55))
    for x in range(0, width, 24):
        draw.line((x, 0, x, height), fill=(20, 20, 20), width=1)
    for y in range(0, height, 24):
        draw.line((0, y, width, y), fill=(240, 240, 240), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_dark_frame(path: Path, *, size: tuple[int, int] = (320, 240)) -> Path:
    image = Image.new("RGB", size, (8, 8, 8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] // 2, size[1]), fill=(20, 22, 18))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_mask(path: Path, *, size: tuple[int, int], box: tuple[int, int, int, int]) -> Path:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _semantic_target(
    *,
    bbox: dict[str, int],
    crop: Path | None = None,
    mask: Path | None = None,
    keypoint: list[int] | None = None,
    synthetic_label: bool = False,
) -> dict[str, Any]:
    return {
        "object_id": "Sink054_handle" if not synthetic_label else "synthetic_sink_handle",
        "label": "right sink handle",
        "bbox": bbox,
        "reference_crop": str(crop) if crop else "",
        "all_crops": [str(crop)] if crop else [],
        "mask_path": str(mask) if mask else "",
        "keypoints": {"center": keypoint or [bbox["x"] + bbox["width"] // 2, bbox["y"] + bbox["height"] // 2]},
        "occlusion": "visible",
        "confidence": 0.94,
        "source": "synthetic_2d_label" if synthetic_label else "object_index_stage",
        "synthetic_label": synthetic_label,
    }


def _eval_ready_grounding(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "eval_ready_task_grounding.v1",
        "status": "ready_for_learned_wam_rollout_request",
        "selected_task_target": target,
        "readiness": {
            "learned_rollout_request_ready": True,
            "target_crop_available": bool(target.get("all_crops")),
            "target_mask_or_keypoint_available": bool(target.get("mask_path") or target.get("keypoints")),
            "blockers": [],
        },
    }


def _video_status(*, width: int, height: int, fps: str, frames: int) -> dict[str, object]:
    return {
        "status": "completed",
        "ffprobe_metadata": {
            "streams": [
                {
                    "width": width,
                    "height": height,
                    "avg_frame_rate": fps,
                    "r_frame_rate": fps,
                    "nb_frames": str(frames),
                    "duration": "6.0",
                }
            ],
            "format": {"duration": "6.0", "size": "1000"},
        },
    }


def test_source_policy_observation_visual_qa_good_frame_passes(tmp_path: Path) -> None:
    frame = _write_good_frame(tmp_path / "good.jpg")

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "passed_visual_quality_gate"
    assert qa["visual_success"] is True
    assert qa["target_visibility_status"] == "passed_visual_proxy"
    assert qa["blockers"] == []


def test_source_policy_observation_visual_qa_visible_semantic_target_passes(
    tmp_path: Path,
) -> None:
    frame = _write_good_frame(tmp_path / "visible.jpg", size=(640, 480))
    crop = _write_good_frame(tmp_path / "crop.jpg", size=(120, 90))
    mask = _write_mask(tmp_path / "mask.png", size=(640, 480), box=(260, 190, 380, 280))
    target = _semantic_target(
        bbox={"x": 260, "y": 190, "width": 120, "height": 90},
        crop=crop,
        mask=mask,
        keypoint=[320, 235],
    )

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        object_index={"objects": [target]},
        eval_ready_task_grounding=_eval_ready_grounding(target),
        semantic_artifact_base_dir=tmp_path,
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "passed_visual_quality_gate"
    assert qa["target_visibility_status"] == "passed_semantic_gate"
    semantic = qa["semantic_target_quality"]
    assert semantic["status"] == "passed"
    assert semantic["gates"]["target_object_visibility"]["passed"] is True
    assert semantic["gates"]["target_centering"]["passed"] is True
    assert semantic["gates"]["target_occlusion"]["passed"] is True
    assert semantic["gates"]["task_region_quality"]["passed"] is True


def test_source_policy_observation_visual_qa_offscreen_semantic_target_fails(
    tmp_path: Path,
) -> None:
    frame = _write_good_frame(tmp_path / "offscreen.jpg", size=(640, 480))
    target = _semantic_target(
        bbox={"x": 700, "y": 190, "width": 120, "height": 90},
        keypoint=[760, 235],
    )

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        object_index={"objects": [target]},
        eval_ready_task_grounding=_eval_ready_grounding(target),
        semantic_artifact_base_dir=tmp_path,
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "failed_visual_quality_gate"
    assert qa["target_visibility_status"] == "failed_semantic_gate"
    assert "target_object_offscreen_in_source_observation" in qa["blockers"]
    semantic = qa["semantic_target_quality"]
    assert semantic["gates"]["target_object_visibility"]["passed"] is False
    assert semantic["gates"]["target_centering"]["passed"] is False


def test_source_policy_observation_visual_qa_dark_flat_occluded_frame_fails(
    tmp_path: Path,
) -> None:
    frame = _write_dark_frame(tmp_path / "dark.jpg")

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "failed_visual_quality_gate"
    assert qa["visual_success"] is False
    assert "source_policy_observation_too_dark_for_review" in qa["blockers"]
    assert "target_object_visibility_failed_visual_proxy" in qa["blockers"]


def test_source_policy_observation_visual_qa_dark_frame_fails_semantic_task_region(
    tmp_path: Path,
) -> None:
    frame = _write_dark_frame(tmp_path / "semantic-dark.jpg", size=(640, 480))
    target = _semantic_target(
        bbox={"x": 260, "y": 190, "width": 120, "height": 90},
        keypoint=[320, 235],
    )

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        object_index={"objects": [target]},
        eval_ready_task_grounding=_eval_ready_grounding(target),
        semantic_artifact_base_dir=tmp_path,
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "failed_visual_quality_gate"
    assert "source_policy_observation_too_dark_for_review" in qa["blockers"]
    assert "target_task_region_too_dark_or_low_information" in qa["blockers"]
    assert qa["semantic_target_quality"]["gates"]["task_region_quality"]["passed"] is False


def test_source_policy_observation_visual_qa_cabinet_only_frame_fails_target_gate(
    tmp_path: Path,
) -> None:
    frame = _write_good_frame(tmp_path / "cabinet-only.jpg", size=(640, 480))
    cabinet = {
        "object_id": "cabinet_panel_001",
        "label": "cabinet panel",
        "bbox": {"x": 180, "y": 120, "width": 250, "height": 260},
        "keypoints": {"center": [305, 250]},
        "confidence": 0.92,
    }

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        object_index={"objects": [cabinet]},
        eval_ready_task_grounding={
            "schema_version": "eval_ready_task_grounding.v1",
            "status": "blocked",
            "readiness": {"blockers": ["missing_task_target_label_or_keypoint"]},
        },
        semantic_artifact_base_dir=tmp_path,
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "failed_visual_quality_gate"
    assert qa["target_visibility_status"] == "failed_semantic_gate"
    assert "target_object_not_found_in_semantic_index" in qa["blockers"]


def test_source_policy_observation_visual_qa_synthetic_labeled_frame_passes_with_boundary(
    tmp_path: Path,
) -> None:
    frame = _write_good_frame(tmp_path / "synthetic-labeled.jpg", size=(640, 480))
    target = _semantic_target(
        bbox={"x": 270, "y": 185, "width": 100, "height": 95},
        keypoint=[320, 232],
        synthetic_label=True,
    )

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        object_index={"objects": [target]},
        semantic_artifact_base_dir=tmp_path,
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "passed_visual_quality_gate"
    assert qa["target_visibility_status"] == "passed_semantic_gate"
    assert qa["semantic_target_quality"]["synthetic_label_evidence_used"] is True
    assert qa["claim_boundary"]["synthetic_labels_are_support_evidence_not_raw_capture_truth"] is True


def test_review_quality_profile_rejects_128px_media_while_smoke_marks_smoke_only(
    tmp_path: Path,
) -> None:
    source = _write_good_frame(tmp_path / "source.jpg")
    generated = _write_good_frame(tmp_path / "generated.jpg", size=(128, 128))

    review_report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "review-job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[generated],
        review_video_path=tmp_path / "review.mp4",
        video_status=_video_status(width=128, height=128, fps="4/1", frames=9),
        visual_profile="review_quality",
        requested_settings={"width": 128, "height": 128, "fps": 4, "num_frames": 9},
        provider_status="completed",
        live_wam_generation_success_count=1,
        learned_wam_model_success_count=1,
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
    )
    smoke_report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "smoke-job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[generated],
        review_video_path=tmp_path / "smoke.mp4",
        video_status=_video_status(width=128, height=128, fps="4/1", frames=9),
        visual_profile="smoke",
        requested_settings={"width": 128, "height": 128, "fps": 4, "num_frames": 9},
        provider_status="completed",
        live_wam_generation_success_count=1,
        learned_wam_model_success_count=1,
    )

    assert review_report["visual_success"] is False
    assert "review_quality_profile_media_below_minimum" in review_report["blockers"]
    assert review_report["provider_completed_visual_quality_failed"] is True
    assert review_report["claim_boundary"]["generated_observation_review_support_only"] is True
    assert review_report["claim_boundary"]["review_quality_gate_is_not_scale_up_approval"] is True
    assert smoke_report["profile_contract"]["smoke_only"] is True
    assert "review_quality_profile_media_below_minimum" not in smoke_report["blockers"]
    assert Path(str(smoke_report["contact_sheet_path"])).is_file()


def test_128px_4fps_mp4_is_valid_media_but_not_reviewable_success_evidence(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "low_res_valid.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (128, 128))
    assert writer.isOpened()
    for index in range(12):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[:, :64, 0] = (index * 17) % 255
        frame[:, 64:, 1] = (180 + index * 3) % 255
        frame[index % 64 : index % 64 + 32, 40:88, 2] = 255
        writer.write(frame)
    writer.release()

    validation = validate_generated_mp4_for_review(video)
    smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=[{"rollout_id": "rollout_low_res", "generated_video_path": str(video)}],
        output_dir=tmp_path,
        generated_at="now",
    )

    assert validation["status"] == "completed"
    assert validation["width"] == 128
    assert smoke["status"] == "failed_visual_quality_smoke"
    assert "generated_rollout_video_resolution_too_low_for_task_success_review" in smoke[
        "blockers"
    ]
    assert "generated_rollout_video_fps_too_low_for_task_success_review" in smoke["blockers"]
    assert smoke["claim_boundary"]["valid_mp4_file_generated"] is True
    assert (
        smoke["claim_boundary"]["visual_rollout_useful_for_task_success_review"]
        is False
    )
    assert smoke["claim_boundary"]["generated_observation_review_support_only"] is True


def test_provider_completed_but_visual_quality_fails_on_dark_generated_frame(
    tmp_path: Path,
) -> None:
    source = _write_good_frame(tmp_path / "source.jpg")
    dark_generated = _write_dark_frame(tmp_path / "generated-dark.jpg")

    report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[dark_generated],
        review_video_path=tmp_path / "review.mp4",
        video_status=_video_status(width=640, height=480, fps="15/1", frames=24),
        visual_profile="review_quality",
        requested_settings={"width": 640, "height": 480, "fps": 15, "num_frames": 24},
        provider_status="completed",
        live_wam_generation_success_count=1,
        learned_wam_model_success_count=1,
    )

    assert report["provider_completed"] is True
    assert report["live_wam_generation_success"] is True
    assert report["visual_success"] is False
    assert report["provider_completed_visual_quality_failed"] is True
    assert "wam_generated_frame_too_dark_for_review" in report["blockers"]


def test_generated_frame_drift_marks_visual_success_false(tmp_path: Path) -> None:
    source = _write_good_frame(tmp_path / "source.jpg")
    first = _write_good_frame(tmp_path / "generated-1.jpg")
    second = _write_dark_frame(tmp_path / "generated-2.jpg")

    report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[first, second],
        review_video_path=tmp_path / "review.mp4",
        video_status=_video_status(width=640, height=480, fps="15/1", frames=24),
        visual_profile="review_quality",
        requested_settings={"width": 640, "height": 480, "fps": 15, "num_frames": 24},
        provider_status="completed",
        live_wam_generation_success_count=2,
        learned_wam_model_success_count=2,
    )

    assert report["visual_success"] is False
    assert "wam_generated_frame_darkening_drift" in report["blockers"]
    assert (tmp_path / "job" / "wam_rollout_frame_stats.jsonl").is_file()
    rows = [
        json.loads(line)
        for line in (tmp_path / "job" / "wam_rollout_frame_stats.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 2
