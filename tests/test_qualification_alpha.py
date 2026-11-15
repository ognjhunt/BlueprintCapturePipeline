from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.geometry_stage import build_geometry_stage_contract
from blueprint_pipeline.materialization import materialize_capture_bundle


def _build_staged_capture(
    tmp_path: Path,
    *,
    requested_outputs: list[str] | None = None,
) -> tuple[Path, str]:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    capture_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)

    manifest_payload = {
        "scene_id": scene_id,
        "capture_id": capture_id,
        "video_uri": "walkthrough.mov",
        "has_lidar": True,
        "intended_space_type": "warehouse",
        "width": 1920,
        "height": 1080,
        "device_model": "iPhone",
        "os_version": "18.0",
        "fps_source": 30.0,
        "capture_start_epoch_ms": 1_700_000_000_000,
        "capture_schema_version": "2.0.0",
        "capture_source": "iphone",
        "capture_tier_hint": "tier1_iphone",
        "requested_outputs": requested_outputs or ["qualification"],
        "disable_default_preview": requested_outputs is None,
        "capture_orientation": {
            "display_orientation": "landscape",
            "rotation_degrees": 0,
            "display_size": {
                "width": 1920,
                "height": 1080,
            },
        },
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "data_licensing_allowed": False,
            "capture_contributor_payout_eligible": True,
            "consent_status": "documented",
            "consent_scope": ["warehouse-a"],
            "permission_document_uri": "gs://bucket/rights/doc.pdf",
            "consent_notes": [],
        },
    }
    (raw_root / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
    (raw_root / "intake_packet.json").write_text(
        json.dumps(
            {
                "workflowName": "Inspect tote handoff",
                "taskSteps": ["Walk the tote handoff lane", "Capture approach and exit views"],
                "zone": "handoff aisle",
                "owner": "ops",
                "targetKPI": "trusted qualification record",
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "capture_context.json").write_text(
        json.dumps(
            {
                "sceneId": scene_id,
                "captureId": capture_id,
                "captureSource": "iphone",
                "captureModality": "iphone_arkit_lidar",
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )
    (raw_root / "task_hypothesis.json").write_text(
        json.dumps({"workflow_name": "Inspect tote handoff", "task_steps": ["Inspect lane"], "status": "accepted"}),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    arkit_root = raw_root / "arkit"
    (arkit_root / "depth").mkdir(parents=True)
    (arkit_root / "poses.jsonl").write_text(
        json.dumps({"frame_id": "000001", "t_device_sec": 0.0, "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}) + "\n",
        encoding="utf-8",
    )
    (arkit_root / "intrinsics.json").write_text(
        json.dumps({"width": 1920, "height": 1080, "fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0}),
        encoding="utf-8",
    )
    (arkit_root / "depth" / "000001.png").write_bytes(b"depth")

    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
    )
    return capture_root, str(materialized["descriptor_uri"])


def test_materialization_does_not_fabricate_missing_upstream_ids(tmp_path: Path) -> None:
    capture_root, _descriptor_uri = _build_staged_capture(tmp_path)
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))

    assert descriptor["site_submission_id"] is None
    assert descriptor["buyer_request_id"] is None
    assert descriptor["capture_job_id"] is None
    assert descriptor["upstream_link_truth_state"] == "blocked_missing_upstream_ids"
    assert descriptor["upstream_link_blockers"] == [
        "missing_site_submission_id",
        "missing_buyer_request_id",
        "missing_capture_job_id",
    ]
    assert descriptor["metadata"]["site_submission_id"] is None
    assert descriptor["metadata"]["buyer_request_id"] is None
    assert descriptor["metadata"]["capture_job_id"] is None


def _successful_capture_review() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "review_type": "gemini_multimodal_capture_review",
        "status": "succeeded",
        "generated_at": "2026-03-15T00:00:00+00:00",
        "provider_name": "gemini",
        "provider_model": "gemini-2.5-pro",
        "review_mode": "video_primary_frames_fallback",
        "confidence": 0.86,
        "summary": "Coverage is strong enough for qualification.",
        "scores": {
            "coverage": 0.86,
            "visual_clarity": 0.84,
            "lighting_stability": 0.8,
            "motion_stability": 0.78,
            "task_understanding": 0.82,
            "world_model_fitness": 0.79,
            "payout_quality": 0.76,
        },
        "bonus_signals": {
            "complete_coverage": {"score": 0.9, "reason": "All major task zones are visible."},
            "multi_pass": {"score": 0.7, "reason": "The walkthrough revisits key areas from multiple angles."},
            "lidar_depth": {"score": 1.0, "reason": "LiDAR/depth-backed capture quality is visible in the bundle."},
            "steady_walkthrough": {"score": 0.8, "reason": "Pacing and steadiness are strong enough for review."},
        },
        "assessments": {
            "blur": {"status": "good", "score": 0.84, "summary": "Video is sharp enough.", "impact": "low"},
            "lighting": {"status": "good", "score": 0.8, "summary": "Lighting is stable.", "impact": "low"},
            "motion_speed": {"status": "good", "score": 0.78, "summary": "Pacing is controlled.", "impact": "low"},
            "doubling_back": {"status": "good", "score": 0.72, "summary": "Revisits are productive.", "impact": "low"},
            "coverage_completeness": {"status": "good", "score": 0.86, "summary": "Scene coverage is complete.", "impact": "low"},
            "task_zone_completeness": {"status": "good", "score": 0.82, "summary": "Task zone is covered.", "impact": "low"},
            "occlusion_and_hidden_zone": {"status": "good", "score": 0.8, "summary": "Occlusion is limited.", "impact": "low"},
            "depth_and_spatial_conditioning": {"status": "good", "score": 0.9, "summary": "Spatial conditioning is strong.", "impact": "low"},
        },
        "findings": {
            "missing_views": [],
            "blur_observations": [],
            "lighting_observations": [],
            "occlusion_observations": [],
            "task_scope_notes": ["Task zone is visible."],
            "blocker_summaries": [],
            "recapture_recommendations": [],
        },
        "recommendations": {
            "world_model_recommendation": "good_candidate",
            "payout_recommendation": "baseline",
        },
        "provenance": {"provider_name": "gemini", "provider_model": "gemini-2.5-pro"},
    }


def _successful_privacy_processing() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "status": "person_removed",
        "mode": "removal",
        "fallback_used": False,
        "people_detected": 2,
        "people_removed": 2,
        "face_anonymized_segments": [],
        "raw_retained": True,
        "fail_closed": True,
        "depth_source": "depth_anything",
        "depth_conditioning": {
            "status": "succeeded",
            "source": "depth_anything",
            "provider": "depth_anything_3",
            "model_name": "da3metric-large",
            "depth_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_depth/depth",
            "confidence_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_depth/confidence",
            "depth_manifest_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_depth/depth_manifest.json",
            "confidence_manifest_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_depth/confidence_manifest.json",
            "frame_count": 24,
        },
        "privacy_processed_video_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/privacy/final_walkthrough.mov",
        "world_model_video_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/privacy/final_walkthrough.mov",
        "privacy_manifest_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_processing_manifest.json",
        "privacy_verification_report_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_verification_report.json",
    }


def _write_geometry_lane(monkeypatch, capture_root: Path) -> None:  # type: ignore[no-untyped-def]
    def _fake_provider(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        frames_dir = geometry_root / "frames" / "images"
        depth_dir = geometry_root / "depth"
        confidence_dir = geometry_root / "confidence"
        frames_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for frame_index in range(2):
            image_path = frames_dir / f"frame_{frame_index:06d}.npy"
            np.save(image_path, np.full((16, 24, 3), 60 + frame_index * 10, dtype=np.float32))
            depth_path = depth_dir / f"depth_{frame_index:06d}.npy"
            confidence_path = confidence_dir / f"confidence_{frame_index:06d}.npy"
            np.save(depth_path, np.full((16, 24), 1.2 + frame_index * 0.1, dtype=np.float32))
            np.save(confidence_path, np.full((16, 24), 0.85, dtype=np.float32))
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": float(frame_index),
                    "image_path": str(image_path),
                    "is_keyframe": True,
                    "blur_score": 0.1,
                    "overlap_hint": 0.9,
                    "world_from_camera": [
                        [1.0, 0.0, 0.0, frame_index * 0.2],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "camera_from_world": [
                        [1.0, 0.0, 0.0, -(frame_index * 0.2)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "pose_confidence": 0.92,
                    "depth_path": str(depth_path),
                    "depth_format": "npy",
                    "confidence_path": str(confidence_path),
                    "confidence_format": "npy",
                    "width": 24,
                    "height": 16,
                    "min_depth_m": 1.2,
                    "max_depth_m": 1.3,
                    "depth_unit": "meters",
                    "metric_depth_truth": True,
                    "depth_measurement_source": "provider_metric_reconstruction",
                    "confidence_range": [0.0, 1.0],
                }
            )
        return {
            "intrinsics": {
                "camera_model": "pinhole",
                "image_width": 24,
                "image_height": 16,
                "fx": 20.0,
                "fy": 20.0,
                "cx": 12.0,
                "cy": 8.0,
                "distortion": {"model": "none", "coefficients": []},
            },
            "frames": frames,
            "provider_metrics": {"backend": "test"},
            "provider_native_result": True,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 0.92,
            "p95_pose_delta_sec": 0.033,
            "provider_warnings": [],
            "provider_errors": [],
            "loop_closure_detected": False,
        }

    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")
    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _fake_provider)
    build_geometry_stage_contract(capture_root)


def test_materialized_capture_uses_beta_default_current_lanes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root, _descriptor_uri = _build_staged_capture(tmp_path, requested_outputs=[])
    raw_manifest_path = capture_root / "raw" / "manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_manifest.pop("requested_outputs", None)
    raw_manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")

    monkeypatch.setenv("BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL", "true")
    materialized = materialize_capture_bundle(
        bucket="local-blueprint",
        scene_id="scene-1",
        capture_id="capture-1",
        gcs_root=tmp_path,
    )

    assert materialized["descriptor"]["requested_outputs"] == [
        "qualification",
        "preview_simulation",
    ]
    assert materialized["descriptor"]["requested_lanes"] == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["requested_lanes"] == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]


# --- default geometry-stage routing must normalize capture-rights booleans ---

_RIGHTS_MISSING = object()
_RIGHTS_COERCION_FORMS = [
    pytest.param(False, False, id="bool-false"),
    pytest.param("false", False, id="str-false"),
    pytest.param("0", False, id="str-zero"),
    pytest.param("no", False, id="str-no"),
    pytest.param("off", False, id="str-off"),
    pytest.param("", False, id="empty-str"),
    pytest.param(_RIGHTS_MISSING, False, id="missing"),
    pytest.param(True, True, id="bool-true"),
    pytest.param("true", True, id="str-true"),
    pytest.param("1", True, id="str-one"),
    pytest.param("yes", True, id="str-yes"),
    pytest.param("on", True, id="str-on"),
]


def _rights_descriptor(rights_value: object) -> CaptureDescriptor:
    rights: dict[str, object] = {}
    if rights_value is not _RIGHTS_MISSING:
        rights["derived_scene_generation_allowed"] = rights_value
    return CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "capture_source": "meta_glasses",
            "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
            "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/frames.jsonl",
            "metadata": {"capture_rights": rights},
        }
    )


