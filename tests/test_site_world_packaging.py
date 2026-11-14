from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_contracts.site_world_contract import load_site_world_bundle, validate_site_world_bundle
from blueprint_pipeline.geometry_stage import build_geometry_stage_contract
from blueprint_pipeline.materialization import build_capture_bundle_records, materialize_capture_bundle


_PACKAGING_UPSTREAM_IDS = {
    "site_submission_id": "site-submission-packaging-office-001",
    "buyer_request_id": "buyer-request-packaging-office-001",
    "capture_job_id": "capture-job-packaging-office-001",
}


def _use_offline_webapp_sync(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in (
        "BLUEPRINT_LAUNCH_PROOF_MODE",
        "PIPELINE_SYNC_WEBAPP_URL",
        "PIPELINE_SYNC_TOKEN",
        "PIPELINE_SYNC_REQUIRED",
        "PIPELINE_BUYER_ACCESS_CHECK_URL",
        "PIPELINE_BUYER_ACCESS_CHECK_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def _successful_capture_review() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "review_type": "gemini_multimodal_capture_review",
        "status": "succeeded",
        "generated_at": "2026-03-15T00:00:00+00:00",
        "provider_name": "gemini",
        "provider_model": "gemini-2.5-pro",
        "review_mode": "video_primary_frames_fallback",
        "confidence": 0.88,
        "summary": "Capture supports downstream work.",
        "scores": {
            "coverage": 0.88,
            "visual_clarity": 0.84,
            "lighting_stability": 0.82,
            "motion_stability": 0.8,
            "task_understanding": 0.85,
            "world_model_fitness": 0.83,
            "payout_quality": 0.78,
        },
        "bonus_signals": {
            "complete_coverage": {"score": 0.9, "reason": "Coverage is complete."},
            "multi_pass": {"score": 0.7, "reason": "Multiple views are present."},
            "lidar_depth": {"score": 1.0, "reason": "Depth-backed capture quality is strong."},
            "steady_walkthrough": {"score": 0.85, "reason": "The walkthrough is steady."},
        },
        "findings": {
            "missing_views": [],
            "blur_observations": [],
            "lighting_observations": [],
            "occlusion_observations": [],
            "task_scope_notes": [],
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
        "people_detected": 1,
        "people_removed": 1,
        "face_anonymized_segments": [],
        "raw_retained": True,
        "fail_closed": True,
        "depth_source": "arkit",
        "depth_conditioning": {
            "status": "available",
            "source": "arkit",
            "provider": "arkit",
            "depth_prefix_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/raw/arkit/depth",
            "confidence_prefix_uri": None,
            "depth_manifest_uri": None,
            "confidence_manifest_uri": None,
        },
        "privacy_processed_video_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/privacy/final_walkthrough.mov",
        "world_model_video_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/privacy/final_walkthrough.mov",
        "privacy_manifest_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_processing_manifest.json",
        "privacy_verification_report_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_verification_report.json",
    }


class _HealthyRuntimeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def register_site_world_package(self, *, spec, registration, health):  # type: ignore[no-untyped-def]
        site_world_id = str(registration.get("site_world_id") or "siteworld-test")
        return {
            **dict(registration),
            "schema_version": "v1",
            "status": "ready",
            "site_world_id": site_world_id,
            "runtime_base_url": "http://runtime.test",
            "websocket_base_url": "ws://runtime.test",
            "runtime_capabilities": {
                "supports_step_rollout": True,
                "supports_batch_rollout": True,
                "supports_camera_views": True,
                "supports_stream": False,
                "supports_rlds_export": True,
                "supports_preview_render": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": True,
                "debug_render_outputs": True,
            },
            "health": {
                **dict(health),
                "schema_version": "v1",
                "site_world_id": site_world_id,
                "healthy": True,
                "launchable": True,
                "status": "healthy",
                "blockers": [],
                "warnings": [],
            },
        }

    def get_site_world_health(self, site_world_id: str):  # type: ignore[no-untyped-def]
        return {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "healthy": True,
            "launchable": True,
            "status": "healthy",
            "blockers": [],
            "warnings": [],
        }

    def create_session(self, site_world_id: str, **_kwargs):  # type: ignore[no-untyped-def]
        return {"site_world_id": site_world_id, "session_id": "session-test"}

    def reset_session(self, _session_id: str, **_kwargs):  # type: ignore[no-untyped-def]
        return {"status": "ok"}


def _write_backend_script(path: Path, *, mode: str) -> None:
    if mode == "success":
        body = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output_path = Path(sys.argv[2])
objects = [
    {
        "id": "cabinet_0001",
        "object_id": "cabinet_0001",
        "label": "cabinet",
        "boundingBox": {
            "center": [0.0, 0.0, 0.75],
            "extents": [0.8, 0.45, 0.9],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "mean_confidence": 0.95,
        "confidence": 0.95,
        "n_total_detections": 3,
        "n_frame_detections": 2,
        "reference_crop": "",
        "all_crops": [],
        "task_relevance": {"score": 0.9, "matched_terms": ["cabinet"]},
        "articulation_hints": {"interactive": True, "kind": "cabinet", "confidence": 0.82},
        "evidence_frames": [0, 1],
        "source_prompts": ["cabinet"],
        "provenance": {"grounding_level": "observed", "canonical_truth": True},
        "mean_box_px": {"area": 64000.0, "width": 240.0, "height": 260.0},
    },
    {
        "id": "aisle_0001",
        "object_id": "aisle_0001",
        "label": "aisle",
        "boundingBox": {
            "center": [0.0, 0.0, 0.0],
            "extents": [1.2, 1.2, 0.1],
            "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "orientationQuaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "mean_confidence": 0.9,
        "confidence": 0.9,
        "n_total_detections": 2,
        "n_frame_detections": 2,
        "reference_crop": "",
        "all_crops": [],
        "task_relevance": {"score": 0.1, "matched_terms": []},
        "articulation_hints": {"interactive": False, "kind": "static", "confidence": 0.3},
        "evidence_frames": [0, 1],
        "source_prompts": ["aisle"],
        "provenance": {"grounding_level": "observed", "canonical_truth": True},
        "mean_box_px": {"area": 72000.0, "width": 280.0, "height": 280.0},
    },
]
output_path.write_text(json.dumps({"backend_status": "ok", "objects": objects}, indent=2), encoding="utf-8")
"""
    elif mode == "runtime-missing":
        body = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "backend_status": "skipped",
            "reason": "ultralytics_missing:stubbed-for-test",
            "detections": [],
        },
        indent=2,
    ),
    encoding="utf-8",
)
"""
    else:
        body = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "backend_status": "skipped",
            "reason": "sam3_not_installed",
            "detections": [],
            "objects": [],
        },
        indent=2,
    ),
    encoding="utf-8",
)
"""
    path.write_text(body, encoding="utf-8")


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
            np.save(image_path, np.full((12, 18, 3), 80, dtype=np.float32))
            depth_path = depth_dir / f"depth_{frame_index:06d}.npy"
            confidence_path = confidence_dir / f"confidence_{frame_index:06d}.npy"
            np.save(depth_path, np.full((12, 18), 1.5, dtype=np.float32))
            np.save(confidence_path, np.full((12, 18), 0.8, dtype=np.float32))
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": float(frame_index) * 0.4,
                    "image_path": str(image_path),
                    "is_keyframe": True,
                    "blur_score": 0.1,
                    "overlap_hint": 0.9,
                    "world_from_camera": [
                        [1.0, 0.0, 0.0, frame_index * 0.15],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "camera_from_world": [
                        [1.0, 0.0, 0.0, -(frame_index * 0.15)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "pose_confidence": 0.9,
                    "depth_path": str(depth_path),
                    "depth_format": "npy",
                    "confidence_path": str(confidence_path),
                    "confidence_format": "npy",
                    "width": 18,
                    "height": 12,
                    "min_depth_m": 1.5,
                    "max_depth_m": 1.5,
                    "confidence_range": [0.0, 1.0],
                }
            )
        return {
            "intrinsics": {
                "camera_model": "pinhole",
                "image_width": 18,
                "image_height": 12,
                "fx": 16.0,
                "fy": 16.0,
                "cx": 9.0,
                "cy": 6.0,
                "distortion": {"model": "none", "coefficients": []},
            },
            "frames": frames,
            "provider_metrics": {"backend": "test"},
            "provider_warnings": [],
            "provider_errors": [],
            "loop_closure_detected": False,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _fake_provider)
    build_geometry_stage_contract(capture_root)


def _build_staged_capture(
    tmp_path: Path,
    *,
    manifest_overrides: dict[str, object] | None = None,
    context_overrides: dict[str, object] | None = None,
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
        "intended_space_type": "office",
        "width": 1920,
        "height": 1080,
        "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "consent_status": "documented",
            "permission_document_uri": "gs://local-blueprint/rights/consent-packet.pdf",
            "consent_scope": ["cabinet zone", "derived_generation", "robot_evaluation"],
        },
        **_PACKAGING_UPSTREAM_IDS,
        "upstream_handoff": dict(_PACKAGING_UPSTREAM_IDS),
    }
    if manifest_overrides:
        manifest_payload.update(manifest_overrides)
    (raw_root / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    (raw_root / "intake_packet.json").write_text(
        json.dumps(
            {
                "workflowName": "Open cabinet",
                "taskSteps": ["Walk to cabinet", "Open cabinet"],
                "zone": "cabinet zone",
                "owner": "ops",
                "targetKPI": "cabinet opened successfully",
            }
        ),
        encoding="utf-8",
    )
    context_payload = {
        "sceneId": scene_id,
        "captureId": capture_id,
        "captureSource": "iphone",
        "captureModality": "iphone_arkit_lidar",
        "hiddenZoneBound": 0.2,
        "captureOrientation": {
            "displayOrientation": "portrait",
            "displayWidth": 1080,
            "displayHeight": 1920,
            "rotationDegrees": 90,
        },
    }
    if context_overrides:
        context_payload.update(context_overrides)
    (raw_root / "capture_context.json").write_text(
        json.dumps(context_payload),
        encoding="utf-8",
    )
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    arkit_root = raw_root / "arkit"
    (arkit_root / "depth").mkdir(parents=True)
    (arkit_root / "poses.jsonl").write_text(
        json.dumps({"frame_id": "000001", "t_device_sec": 0.0, "T_world_camera": np.eye(4).tolist()}) + "\n",
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


def _assert_valid_production_bundle(eval_root: Path) -> dict[str, object]:
    registration_path = eval_root / "site_world_registration.json"
    bundle = load_site_world_bundle(registration_path, require_spec=True)
    errors = validate_site_world_bundle(bundle, production_mode=True)
    assert errors == []
    return bundle.spec


def test_materialization_capture_orientation_precedence(monkeypatch, tmp_path: Path) -> None:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    raw_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )

    def _write_inputs(manifest_payload: dict[str, object], context_payload: dict[str, object]) -> None:
        (raw_root / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
        (raw_root / "capture_context.json").write_text(json.dumps(context_payload), encoding="utf-8")

    class _Result:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    _write_inputs(
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "video_uri": "walkthrough.mov",
            "width": 1920,
            "height": 1080,
            "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
        },
        {
            "sceneId": scene_id,
            "captureId": capture_id,
            "captureOrientation": {"display_orientation": "portrait", "rotation_degrees": 90},
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.materialization.subprocess.run",
        lambda *args, **kwargs: _Result(
            json.dumps({"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "tags": {"rotate": "0"}}]})
        ),
    )
    explicit = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert explicit["descriptor"]["capture_orientation"]["display_orientation"] == "portrait"
    assert explicit["descriptor"]["capture_orientation"]["source"] == "capture_context"
    assert explicit["descriptor"]["capture_orientation"]["display_rotation_degrees"] == 90
    assert explicit["descriptor"]["capture_orientation"]["normalization_applied"] is True
    assert explicit["descriptor"]["capture_orientation"]["declared_capture_width"] == 1080
    assert explicit["descriptor"]["capture_orientation"]["declared_capture_height"] == 1920

    _write_inputs(
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "video_uri": "walkthrough.mov",
            "width": 1920,
            "height": 1080,
            "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
        },
        {"sceneId": scene_id, "captureId": capture_id},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.materialization.subprocess.run",
        lambda *args, **kwargs: _Result(
            json.dumps({"streams": [{"codec_type": "video", "width": 1920, "height": 1080, "tags": {"rotate": "90"}}]})
        ),
    )
    probed = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert probed["descriptor"]["capture_orientation"]["display_orientation"] == "portrait"
    assert probed["descriptor"]["capture_orientation"]["source"] == "video_metadata"
    assert probed["descriptor"]["capture_orientation"]["display_rotation_degrees"] == 90
    assert probed["descriptor"]["capture_orientation"]["normalization_applied"] is True

    _write_inputs(
        {
            "scene_id": scene_id,
            "capture_id": capture_id,
            "video_uri": "walkthrough.mov",
            "width": 720,
            "height": 1280,
            "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
        },
        {"sceneId": scene_id, "captureId": capture_id},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.materialization.subprocess.run",
        lambda *args, **kwargs: _Result("", returncode=1),
    )
    fallback = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
        write_frames_index=False,
    )
    assert fallback["descriptor"]["capture_orientation"]["display_orientation"] == "portrait"
    assert fallback["descriptor"]["capture_orientation"]["source"] == "inferred"
    assert fallback["descriptor"]["capture_orientation"]["display_rotation_degrees"] == 0
    assert fallback["descriptor"]["capture_orientation"]["normalization_applied"] is False


def test_materialization_capture_orientation_handles_missing_ffprobe(monkeypatch, tmp_path: Path) -> None:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    raw_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )
    (raw_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "video_uri": "walkthrough.mov",
                "width": 720,
                "height": 1280,
                "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "capture_context.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "blueprint_pipeline.materialization.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    records = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
        write_frames_index=False,
    )

    assert records["descriptor"]["capture_orientation"]["display_orientation"] == "portrait"
    assert records["descriptor"]["capture_orientation"]["source"] == "inferred"
    assert records["descriptor"]["capture_orientation"]["display_rotation_degrees"] == 0
    assert records["descriptor"]["capture_orientation"]["normalization_applied"] is False


