from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.oscar_visual_augmentation_packet import (
    CLAIM_BOUNDARY,
    build_oscar_visual_augmentation_packet,
    main,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_bytes(path: Path, payload: bytes = b"asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "raw").mkdir(parents=True)
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    return capture_root


def test_oscar_visual_augmentation_packet_requires_camera_and_skeleton_provenance(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    output_dir = tmp_path / "packet"

    manifest = build_oscar_visual_augmentation_packet(
        capture_root=capture_root,
        output_dir=output_dir,
    )

    assert manifest["status"] == "blocked_missing_provenance_or_generation_inputs"
    assert "missing_first_frame_visual_context" in manifest["blockers"]
    assert "missing_skeleton_conditioning_video" in manifest["blockers"]
    assert "missing_camera_provenance" in manifest["blockers"]
    assert "missing_skeleton_provenance" in manifest["blockers"]
    assert manifest["claim_boundary"]["generated_videos_are_raw_capture_evidence"] is False
    assert manifest["claim_boundary"]["contact_physics_proven"] is False
    assert (output_dir / "claim_boundary.json").is_file()


def test_oscar_visual_augmentation_packet_writes_swappable_model_derived_packet(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"
    first_frame = job_dir / "first_frame.png"
    skeleton_video = job_dir / "skeleton.mp4"
    camera = job_dir / "camera_calibration_quality_gate.json"
    skeleton_trace = job_dir / "g1_projected_skeleton_trace.jsonl"
    generated_video = job_dir / "generated" / "kitchen.mp4"
    variants = tmp_path / "variants.json"
    output_dir = job_dir / "oscar_visual_augmentation_packet"
    _write_bytes(first_frame, b"png")
    _write_bytes(skeleton_video, b"mp4")
    _write_json(camera, {"schema_version": "camera_calibration_quality_gate.v1", "status": "ready"})
    skeleton_trace.parent.mkdir(parents=True, exist_ok=True)
    skeleton_trace.write_text('{"projected_landmark_count": 3}\n', encoding="utf-8")
    _write_bytes(generated_video, b"generated-mp4")
    _write_json(
        variants,
        {
            "variants": [
                {
                    "variant_id": "kitchen_counter",
                    "prompt": "same task in a realistic kitchen",
                    "environment_tags": ["kitchen"],
                }
            ]
        },
    )

    manifest = build_oscar_visual_augmentation_packet(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
        first_frame=first_frame,
        skeleton_video=skeleton_video,
        camera_provenance=camera,
        skeleton_provenance=skeleton_trace,
        variant_specs=variants,
        generated_videos=[
            {
                "variant_id": "kitchen_counter",
                "path": str(generated_video),
                "model_backend_id": "cosmos_wam",
            }
        ],
        selected_backend_id="cosmos_wam",
    )

    assert manifest["status"] == "completed_with_model_derived_generated_videos"
    assert manifest["variant_count"] == 1
    assert manifest["generated_video_count"] == 1
    assert manifest["selected_backend_id"] == "cosmos_wam"
    assert manifest["post_training_data_package_contract"][
        "generated_videos_must_remain_model_derived"
    ] is True
    assert manifest["generated_videos"][0]["model_derived"] is True
    assert manifest["generated_videos"][0]["raw_capture_evidence"] is False
    assert manifest["generated_videos"][0]["contact_physics_proven"] is False
    assert manifest["claim_boundary"] == CLAIM_BOUNDARY

    backend_registry = json.loads(
        (output_dir / "model_backend_registry.json").read_text(encoding="utf-8")
    )
    backend_ids = {row["backend_id"] for row in backend_registry["backends"]}
    assert {"oscar_wam", "cosmos_wam", "future_video_wam"} <= backend_ids
    variant_rows = [
        json.loads(line)
        for line in (output_dir / "visual_augmentation_variant_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert variant_rows[0]["resimulation_required"] is False
    assert variant_rows[0]["claim_boundary"]["deployment_safety_proven"] is False


def test_oscar_visual_augmentation_packet_cli_status_codes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    capture_root = _capture_root(tmp_path)
    blocked_output = tmp_path / "blocked"

    assert main(["--capture-root", str(capture_root), "--output-dir", str(blocked_output)]) == 1
    assert "status=blocked_missing_provenance_or_generation_inputs" in capsys.readouterr().out

    first_frame = tmp_path / "first_frame.png"
    skeleton_video = tmp_path / "skeleton.mp4"
    camera = tmp_path / "camera.json"
    skeleton_trace = tmp_path / "skeleton.jsonl"
    ready_output = tmp_path / "ready"
    _write_bytes(first_frame)
    _write_bytes(skeleton_video)
    _write_json(camera, {"status": "ready"})
    skeleton_trace.write_text("{}\n", encoding="utf-8")

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--output-dir",
                str(ready_output),
                "--first-frame",
                str(first_frame),
                "--skeleton-video",
                str(skeleton_video),
                "--camera-provenance",
                str(camera),
                "--skeleton-provenance",
                str(skeleton_trace),
            ]
        )
        == 0
    )
    assert "status=ready_for_model_backend_generation" in capsys.readouterr().out
