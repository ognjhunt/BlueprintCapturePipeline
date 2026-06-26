from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_contracts.site_world_contract import load_site_world_bundle, validate_site_world_bundle
from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline
from blueprint_pipeline.evaluation_prep_stage import _build_launchable_export_bundle, run_evaluation_prep_stage
from blueprint_pipeline.geometry_stage import build_geometry_stage_contract
from blueprint_pipeline.materialization import build_capture_bundle_records, materialize_capture_bundle
from blueprint_pipeline.qualification import _presentation_bundle_status, _presentation_primary_asset


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


def test_site_world_packaging_emits_launchable_bundle(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    success_backend = tmp_path / "success_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(success_backend, mode="success")
    _write_backend_script(sam3_backend, mode="sam3-skip")

    _use_offline_webapp_sync(monkeypatch)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL", "https://demo.example/internal")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL", "https://demo.example/public")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LEGACY_SIMREADY_EVAL_PREP", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LEGACY_MARBLE_EVAL_PREP", "true")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr("blueprint_pipeline.qualification.run_privacy_postprocess", lambda **_kwargs: _successful_privacy_processing())

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    pipeline_root_for_marble = capture_root / "pipeline"
    (pipeline_root_for_marble / "worldlabs_request_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "provider_name": "world_labs",
                "provider_model": "marble-1.1",
                "status": "ready_for_generation",
                "selected_video_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/privacy/final_walkthrough.mov",
                "source_manifest_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_processing_manifest.json",
                "worldlabs_input_audit_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input_audit.json",
                "selected_input_checksum_sha256": "selected-sha",
                "privacy_safe_input": True,
                "generation_request": {"model": "marble-1.1"},
            }
        ),
        encoding="utf-8",
    )
    (pipeline_root_for_marble / "worldlabs_operation_manifest.json").write_text(
        json.dumps({"operation_id": "op-fixture", "done": True, "status": "ready"}),
        encoding="utf-8",
    )
    (pipeline_root_for_marble / "worldlabs_world_manifest.json").write_text(
        json.dumps(
            {
                "world_id": "world-fixture",
                "world_marble_url": "https://marble.worldlabs.ai/worlds/world-fixture",
                "model": "marble-1.1",
                "updated_at": "2026-06-02T00:00:00Z",
                "assets": {
                    "mesh": {
                        "collider_mesh_url": "https://cdn.worldlabs.ai/world-fixture/collider.glb"
                    },
                    "splats": {
                        "spz_urls": {
                            "full": "https://cdn.worldlabs.ai/world-fixture/full.spz"
                        },
                        "semantics_metadata": {
                            "metric_scale_factor": 0.5,
                            "ground_plane_offset": 1.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation = run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    pipeline_root = capture_root / "pipeline"
    eval_root = pipeline_root / "evaluation_prep"
    manifest = json.loads((eval_root / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))
    health = json.loads((eval_root / "site_world_health.json").read_text(encoding="utf-8"))
    geometry = json.loads((eval_root / "object_geometry_manifest.json").read_text(encoding="utf-8"))
    webapp_sync = json.loads(
        (pipeline_root / "webapp_sync_result.json").read_text(encoding="utf-8")
    )

    assert evaluation["manifest_path"] == str(eval_root / "evaluation_prep_manifest.json")
    assert webapp_sync["status"] == "skipped"
    assert webapp_sync["latest_stage"] == "evaluation_prep"
    qualification_sync = webapp_sync["syncs"]["qualification"]
    assert qualification_sync["reason"] == "sync_not_configured"
    assert qualification_sync["attachment_payload"]["upstream_links_verified"] is True
    assert (
        qualification_sync["attachment_payload"]["site_submission_id"]
        == _PACKAGING_UPSTREAM_IDS["site_submission_id"]
    )
    assert (
        qualification_sync["attachment_payload"]["buyer_request_id"]
        == _PACKAGING_UPSTREAM_IDS["buyer_request_id"]
    )
    assert (
        qualification_sync["attachment_payload"]["capture_job_id"]
        == _PACKAGING_UPSTREAM_IDS["capture_job_id"]
    )
    assert evaluation["proof_path_status"]["event_statuses"][0]["event_name"] == "proof_pack_delivered"
    assert len(evaluation["proof_path_status"]["event_statuses"]) == 4
    assert (eval_root / "site_world_spec.json").is_file()
    assert (eval_root / "site_world_registration.json").is_file()
    assert (eval_root / "site_world_health.json").is_file()
    assert (pipeline_root / "presentation_world" / "presentation_bundle.json").is_file()
    assert (pipeline_root / "presentation_world" / "presentation_world_manifest.json").is_file()
    assert (pipeline_root / "presentation_world" / "runtime_demo_manifest.json").is_file()
    presentation_bundle = json.loads((pipeline_root / "presentation_world" / "presentation_bundle.json").read_text(encoding="utf-8"))
    runtime_demo_manifest = json.loads((pipeline_root / "presentation_world" / "runtime_demo_manifest.json").read_text(encoding="utf-8"))
    presentation_world_manifest = json.loads((pipeline_root / "presentation_world" / "presentation_world_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["presentation_bundle"] == "../presentation_world/presentation_bundle.json"
    assert manifest["artifacts"]["presentation_world_manifest"] == "../presentation_world/presentation_world_manifest.json"
    assert manifest["artifacts"]["runtime_demo_manifest"] == "../presentation_world/runtime_demo_manifest.json"
    assert presentation_bundle["authoritative_record"] is False
    assert presentation_bundle["canonical_source"]["scene_memory_manifest_uri"].endswith("/scene_memory/scene_memory_manifest.json")
    assert presentation_bundle["render_inputs"]["scene_memory_manifest_uri"].endswith("/scene_memory/scene_memory_manifest.json")
    assert presentation_bundle["render_inputs"]["conditioning_bundle_uri"].endswith("/scene_memory/conditioning_bundle.json")
    assert presentation_bundle["status"] == "missing"
    assert presentation_bundle["bundle_type"] == "gsplat_scene_v1"
    assert presentation_bundle["renderer_backend"] == "gsplat"
    assert presentation_bundle["fallback_policy"] == "canonical_only"
    assert presentation_world_manifest["presentation_bundle_uri"].endswith("/presentation_world/presentation_bundle.json")
    assert presentation_world_manifest["bundle_type"] == "gsplat_scene_v1"
    assert presentation_world_manifest["renderer_backend"] == "gsplat"
    assert presentation_world_manifest["readiness"]["bundle_status"] == "missing"
    assert presentation_world_manifest["orientation"]["display_orientation"] == "portrait"
    assert presentation_world_manifest["orientation"]["display_rotation_degrees"] == 90
    assert runtime_demo_manifest["ui_base_url"] == "https://demo.example/internal"
    assert runtime_demo_manifest["public_ui_base_url"] == "https://demo.example/public"
    assert runtime_demo_manifest["presentation_world_manifest_uri"].endswith("/presentation_world/presentation_world_manifest.json")
    assert runtime_demo_manifest["renderer_backend"] == "gsplat"
    assert runtime_demo_manifest["bundle_status"] == "missing"
    assert runtime_demo_manifest["fallback_policy"] == "canonical_only"
    assert runtime_demo_manifest["interactive_demo"]["readiness_state"] == "ready"
    assert runtime_demo_manifest["interactive_demo"]["render_inputs"]["site_world_spec_uri"].endswith("/evaluation_prep/site_world_spec.json")
    assert health["launchable"] is True
    registration = json.loads((eval_root / "site_world_registration.json").read_text(encoding="utf-8"))
    assert registration["default_backend"] == "site_world_runtime"
    assert "site_world_runtime" in registration["launchable_backends"]
    assert registration["backend_variants"]["site_world_runtime"]["launchable"] is True
    assert registration["backend_variants"]["cosmos_predict_lora_adapter"]["launchable"] is False
    assert len(geometry["objects"]) >= 1
    bundle = load_site_world_bundle(eval_root / "site_world_registration.json", require_spec=True)
    assert validate_site_world_bundle(bundle, production_mode=False) == []
    spec = bundle.spec
    runtime_eligibility = dict(spec["runtime_eligibility"])
    assert runtime_eligibility["readiness_state"] == "launchable"
    assert spec["canonical_output"]["authoritative_record"] is True
    assert spec["presentation_output"]["authoritative_record"] is False
    assert spec["primary_runtime_backend"] == "site_world_runtime"
    assert spec["canonical_world_model"]["world_model_backend"] == "site_world_runtime"
    assert spec["canonical_world_model"]["scene_representation"] == "pending_world_model_service"
    assert spec["runtime_render_source"] == "site_world_runtime_full_capture"
    assert spec["fallback_mode"] == "arkit_rgbd_last_resort"
    assert spec["canonical_world_model"]["primary_asset_path"] == ""
    assert spec["presentation"]["bundle_type"] == "gsplat_scene_v1"
    assert spec["presentation"]["renderer_backend"] == "gsplat"
    assert spec["presentation"]["bundle_status"] == "missing"
    assert spec["presentation"]["primary_asset_path"] == ""
    assert spec["presentation"]["orientation"]["display_orientation"] == "portrait"
    summary = json.loads((eval_root / "evaluation_prep_summary.json").read_text(encoding="utf-8"))
    launchable_export = json.loads((eval_root / "launchable_export_bundle.json").read_text(encoding="utf-8"))
    simready_scene_manifest = json.loads(
        (pipeline_root / "simready" / "simready_scene_manifest.json").read_text(encoding="utf-8")
    )
    simready_validation = json.loads(
        (pipeline_root / "simready" / "simready_validation.json").read_text(encoding="utf-8")
    )
    simready_prep_manifest = json.loads(
        (eval_root / "simready_prep_manifest.json").read_text(encoding="utf-8")
    )
    marble_bridge = json.loads(
        (pipeline_root / "marble_sim_assets" / "marble_simready_bridge.json").read_text(
            encoding="utf-8"
        )
    )
    marble_validation = json.loads(
        (pipeline_root / "marble_sim_assets" / "marble_asset_validation.json").read_text(
            encoding="utf-8"
        )
    )
    robot_eval_root = pipeline_root / "robot_eval_dataset"
    robot_eval_manifest = json.loads(
        (robot_eval_root / "robot_eval_dataset_manifest.json").read_text(encoding="utf-8")
    )
    robot_eval_site_card = json.loads(
        (robot_eval_root / "site_card.json").read_text(encoding="utf-8")
    )
    robot_eval_proof_boundaries = json.loads(
        (robot_eval_root / "proof_boundaries.json").read_text(encoding="utf-8")
    )
    assert summary["validation_gates"]["presentation_demo_ui_ready"]["passed"] is False
    assert launchable_export["bundles"]["presentation_demo_ui"]["launchable"] is False
    assert manifest["artifacts"]["simready_prep_manifest"] == "simready_prep_manifest.json"
    assert manifest["artifacts"]["marble_simready_bridge"] == "../marble_sim_assets/marble_simready_bridge.json"
    assert manifest["artifacts"]["marble_asset_validation"] == "../marble_sim_assets/marble_asset_validation.json"
    assert (
        manifest["artifacts"]["robot_eval_dataset_manifest"]
        == "../robot_eval_dataset/robot_eval_dataset_manifest.json"
    )
    assert manifest["artifacts"]["robot_eval_site_card"] == "../robot_eval_dataset/site_card.json"
    assert manifest["artifacts"]["robot_eval_task_cards"] == "../robot_eval_dataset/task_cards.json"
    assert manifest["artifacts"]["robot_eval_scenario_cards"] == "../robot_eval_dataset/scenario_cards.json"
    assert manifest["artifacts"]["robot_eval_cards"] == "../robot_eval_dataset/eval_cards.json"
    assert (
        manifest["artifacts"]["robot_eval_proof_boundaries"]
        == "../robot_eval_dataset/proof_boundaries.json"
    )
    assert (
        manifest["artifacts"]["robot_task_ontology_v1"]
        == "../robot_eval_dataset/task_ontology_v1.json"
    )
    assert (
        manifest["artifacts"]["robot_scenario_family_library"]
        == "../robot_eval_dataset/scenario_family_library.json"
    )
    assert (
        manifest["artifacts"]["robot_rights_packet"]
        == "../robot_eval_dataset/rights_packet.json"
    )
    assert (
        manifest["artifacts"]["recorded_trace_eval_report"]
        == "../robot_eval_dataset/recorded_trace_eval_report.json"
    )
    assert (
        manifest["artifacts"]["robot_team_test_submission_modalities"]
        == "../robot_eval_dataset/robot_team_test_submission_modalities.json"
    )
    assert simready_prep_manifest["scene_manifest_path"] == "../simready/simready_scene_manifest.json"
    assert simready_scene_manifest["framework_artifacts"]["isaac_sim"]["path"].endswith(
        "isaac_sim/site_scene.usda"
    )
    assert simready_scene_manifest["framework_artifacts"]["mujoco"]["path"].endswith(
        "mujoco/site_scene.xml"
    )
    assert simready_scene_manifest["framework_artifacts"]["pybullet"]["path"].endswith(
        "pybullet/site_scene.urdf"
    )
    assert simready_validation["claim_boundary"]["simulator_execution_proven"] is False
    assert simready_validation["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert summary["marble_sim_asset_lane_status"] == "review_ready_with_conversion_required"
    assert marble_bridge["evaluation_prep_summary"]["isaac_visual_conversion_required"] is True
    assert marble_validation["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert evaluation["marble_sim_assets"]["status"] == "review_ready_with_conversion_required"
    assert summary["robot_eval_dataset_status"] in {
        "capture_grounded_review_ready",
        "blocked",
    }
    assert robot_eval_manifest["schema_version"] == "real_site_robot_eval_dataset_manifest.v0.1"
    assert robot_eval_manifest["dataset_version"] == "0.1"
    assert (
        evaluation["site_package_manifest"]["artifacts"][
            "robot_team_test_submission_modalities_uri"
        ]
        == "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/robot_eval_dataset/robot_team_test_submission_modalities.json"
    )
    assert (
        evaluation["site_package_manifest"]["artifacts"]["robot_task_ontology_v1_uri"]
        == "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/robot_eval_dataset/task_ontology_v1.json"
    )
    assert (
        evaluation["site_package_manifest"]["artifacts"]["robot_rights_packet_uri"]
        == "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/robot_eval_dataset/rights_packet.json"
    )
    assert (
        evaluation["site_package_manifest"]["artifacts"]["recorded_trace_eval_report_uri"]
        == "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/robot_eval_dataset/recorded_trace_eval_report.json"
    )
    assert robot_eval_manifest["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert robot_eval_manifest["claim_boundary"]["simulator_execution_proven"] is False
    assert robot_eval_site_card["schema_version"] == "real_site_robot_eval_site_card.v0.1"
    assert robot_eval_proof_boundaries["robot_policy_execution_proven"] is False
    assert robot_eval_proof_boundaries["non_ranking_operational_claim_proven"] is False
    assert evaluation["robot_eval_dataset"]["manifest_path"].endswith(
        "/robot_eval_dataset/robot_eval_dataset_manifest.json"
    )
    assert health["runtime_smoke"]["status"] == "succeeded"
    assert health["runtime_smoke"]["session_created"] is True


def test_production_launchable_export_requires_runtime_session_smoke(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")

    bundle = _build_launchable_export_bundle(
        scene_memory_bundle_manifest={"geometry_summary_path": "../geometry/geometry_summary.json"},
        geometry_bundle_manifest={"status": "complete"},
        site_world_registration={"runtime_capabilities": {}},
        site_world_health={
            "launchable": False,
            "status": "degraded",
            "blockers": ["runtime_session_smoke_required"],
        },
        runtime_demo_manifest={},
        simready_prep_manifest_path=None,
    )

    assert bundle["status"] == "blocked"
    assert bundle["runtime_required"] is True
    assert "runtime_required_for_buyer_launch" in bundle["launch_blockers"]


def test_launchable_export_blocks_fallback_geometry_conditioning() -> None:
    bundle = _build_launchable_export_bundle(
        scene_memory_bundle_manifest={
            "geometry_summary_path": "../geometry/geometry_summary.json",
            "geometry_summary": {
                "status": "completed_with_fallback",
                "geometry_source": "fallback_geometry",
                "fallback_used": True,
                "ready_for_world_model": False,
                "geometry_live_ready": False,
                "site_faithful_market_ready": False,
                "launch_blockers": ["fallback_geometry_not_live_video_to_world"],
            },
        },
        geometry_bundle_manifest={"status": "missing"},
        site_world_registration={"runtime_capabilities": {}},
        site_world_health={"launchable": False, "status": "blocked", "blockers": []},
        runtime_demo_manifest={},
        simready_prep_manifest_path=None,
    )

    geometry_bundle = bundle["bundles"]["geometry_conditioning"]
    assert geometry_bundle["launchable"] is False
    assert geometry_bundle["geometry_source"] == "fallback_geometry"
    assert geometry_bundle["fallback_used"] is True
    assert "fallback_geometry_not_live_video_to_world" in geometry_bundle["blockers"]
    assert bundle["status"] == "partial"


def test_site_world_packaging_carries_geometry_conditioning(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    success_backend = tmp_path / "success_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(success_backend, mode="success")
    _write_backend_script(sam3_backend, mode="sam3-skip")
    _write_geometry_lane(monkeypatch, capture_root)

    _use_offline_webapp_sync(monkeypatch)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_UI_BASE_URL", "https://demo.example/internal")
    monkeypatch.setenv("BLUEPRINT_PRESENTATION_DEMO_PUBLIC_UI_BASE_URL", "https://demo.example/public")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr("blueprint_pipeline.qualification.run_privacy_postprocess", lambda **_kwargs: _successful_privacy_processing())

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    pipeline_root = capture_root / "pipeline"
    scene_memory_manifest = json.loads((pipeline_root / "scene_memory" / "scene_memory_manifest.json").read_text(encoding="utf-8"))
    conditioning_bundle = json.loads((pipeline_root / "scene_memory" / "conditioning_bundle.json").read_text(encoding="utf-8"))
    site_world_spec = json.loads((pipeline_root / "evaluation_prep" / "site_world_spec.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((pipeline_root / "evaluation_prep" / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))
    launchable_export = json.loads((pipeline_root / "evaluation_prep" / "launchable_export_bundle.json").read_text(encoding="utf-8"))

    assert scene_memory_manifest["geometry_conditioning"]["summary_uri"].endswith("/geometry/geometry_summary.json")
    assert conditioning_bundle["geometry"]["poses_uri"].endswith("/geometry/camera/poses.jsonl")
    assert site_world_spec["conditioning"]["geometry_summary_uri"].endswith("/geometry/geometry_summary.json")
    assert site_world_spec["geometry"]["geometry_summary_path"].endswith("/pipeline/geometry/geometry_summary.json")
    assert eval_manifest["artifacts"]["geometry_summary"].startswith("../geometry/")
    geometry_conditioning = launchable_export["bundles"]["geometry_conditioning"]
    assert geometry_conditioning["launchable"] is False
    assert geometry_conditioning["geometry_source"] == "local_sfm"
    assert geometry_conditioning["local_reference_ready"] is True
    assert geometry_conditioning["non_arkit_geometry_state"] == "degraded"
    assert "geometry_source_not_video_to_world:local_sfm" in geometry_conditioning["blockers"]
    assert eval_manifest["validation_gates"]["geometry_conditioning_ready"]["passed"] is False


def test_site_world_packaging_surfaces_runtime_missing_blockers(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    missing_backend = tmp_path / "missing_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(missing_backend, mode="runtime-missing")
    _write_backend_script(sam3_backend, mode="sam3-skip")

    _use_offline_webapp_sync(monkeypatch)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {missing_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {missing_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr("blueprint_pipeline.qualification.run_privacy_postprocess", lambda **_kwargs: _successful_privacy_processing())

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    pipeline_root = capture_root / "pipeline"
    raw_root = capture_root / "raw"
    eval_root = pipeline_root / "evaluation_prep"
    build_report = json.loads((raw_root / "object_index_build_report.json").read_text(encoding="utf-8"))
    geometry = json.loads((eval_root / "object_geometry_manifest.json").read_text(encoding="utf-8"))
    health = json.loads((eval_root / "site_world_health.json").read_text(encoding="utf-8"))
    manifest = json.loads((eval_root / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))

    assert build_report["empty_index_cause"] == "runtime_missing"
    assert geometry["status"] == "empty_object_index"
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in geometry["object_index_backend_blockers"]
    assert health["launchable"] is False
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" not in health["blockers"]
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in health["warnings"]
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in manifest["degradation_reasons"]
    bundle = load_site_world_bundle(eval_root / "site_world_registration.json", require_spec=True)
    assert validate_site_world_bundle(bundle, production_mode=False) == []
    spec = bundle.spec
    runtime_eligibility = dict(spec["runtime_eligibility"])
    assert runtime_eligibility["readiness_state"] == "launchable"
    assert "object_index_backend:yolo_world:ultralytics_missing:stubbed-for-test" in runtime_eligibility["warnings"]
    assert spec["canonical_output"]["authoritative_record"] is True
    assert spec["presentation_output"]["authoritative_record"] is False
    summary = json.loads((eval_root / "evaluation_prep_summary.json").read_text(encoding="utf-8"))
    runtime_demo_manifest = json.loads((pipeline_root / "presentation_world" / "runtime_demo_manifest.json").read_text(encoding="utf-8"))
    assert summary["validation_gates"]["presentation_demo_ui_ready"]["passed"] is False
    assert runtime_demo_manifest["status"] == "missing"
    assert runtime_demo_manifest["interactive_demo"]["readiness_state"] == "blocked"
    assert "missing_demo_ui_base_url" in runtime_demo_manifest["interactive_demo"]["blockers"]


def test_site_world_packaging_preserves_vertical_capture_orientation(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(
        tmp_path,
        context_overrides={
            "captureOrientation": {
                "display_orientation": "portrait",
                "rotation_degrees": 90,
                "display_size": {"width": 1080, "height": 1920},
            }
        },
    )
    success_backend = tmp_path / "success_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(success_backend, mode="success")
    _write_backend_script(sam3_backend, mode="sam3-skip")

    _use_offline_webapp_sync(monkeypatch)
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_GROUNDING_DINO_COMMAND", f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("OBJECT_INDEX_SAM3_COMMAND", f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setattr("blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient", _HealthyRuntimeClient)
    monkeypatch.setattr("blueprint_pipeline.qualification.infer_capture_fidelity_review", lambda **_kwargs: _successful_capture_review())
    monkeypatch.setattr("blueprint_pipeline.qualification.run_privacy_postprocess", lambda **_kwargs: _successful_privacy_processing())

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="scene_memory",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    run_evaluation_prep_stage(capture_root=capture_root, provider_name="manual")

    pipeline_root = capture_root / "pipeline"
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    scene_memory_manifest = json.loads((pipeline_root / "scene_memory" / "scene_memory_manifest.json").read_text(encoding="utf-8"))
    presentation_bundle = json.loads((pipeline_root / "presentation_world" / "presentation_bundle.json").read_text(encoding="utf-8"))
    runtime_demo_manifest = json.loads((pipeline_root / "presentation_world" / "runtime_demo_manifest.json").read_text(encoding="utf-8"))
    eval_manifest = json.loads((pipeline_root / "evaluation_prep" / "evaluation_prep_manifest.json").read_text(encoding="utf-8"))
    site_world_spec = json.loads((pipeline_root / "evaluation_prep" / "site_world_spec.json").read_text(encoding="utf-8"))
    hosted_runtime_manifest = json.loads((pipeline_root / "evaluation_prep" / "hosted_session_runtime_manifest.json").read_text(encoding="utf-8"))

    assert descriptor["capture_orientation"]["display_orientation"] == "portrait"
    assert scene_memory_manifest["capture_orientation"]["display_orientation"] == "portrait"
    assert presentation_bundle["capture_orientation"]["display_orientation"] == "portrait"
    assert runtime_demo_manifest["capture_orientation"]["display_orientation"] == "portrait"
    assert eval_manifest["capture_orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["capture_orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["presentation"]["orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["canonical_world_model"]["orientation"]["display_orientation"] == "portrait"
    assert site_world_spec["canonical_world_model"]["world_model_backend"] == "site_world_runtime"
    assert scene_memory_manifest["primary_runtime_backend"] == "site_world_runtime"
    assert scene_memory_manifest["canonical_world_model"]["scene_representation"] == "pending_world_model_service"
    assert hosted_runtime_manifest["primary_runtime_backend"] == "site_world_runtime"
    assert hosted_runtime_manifest["canonical_world_model"]["primary_asset_path"] == ""
    assert hosted_runtime_manifest["capture_orientation"]["display_orientation"] == "portrait"


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


def test_presentation_primary_asset_uses_advanced_geometry(tmp_path: Path) -> None:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    pipeline_dir = capture_root / "pipeline"
    advanced_dir = pipeline_dir / "advanced_geometry"
    advanced_dir.mkdir(parents=True)
    (advanced_dir / "3dgs_compressed.ply").write_text("advanced", encoding="utf-8")

    primary_asset = _presentation_primary_asset(
        pipeline_dir=pipeline_dir,
        bucket="local-blueprint",
        storage_root=tmp_path,
    )

    assert primary_asset is not None
    assert str(primary_asset["path"]).endswith("3dgs_compressed.ply")
    assert primary_asset["source_name"] == "advanced_geometry_3dgs"


def test_presentation_bundle_status_missing_without_primary_asset() -> None:
    status = _presentation_bundle_status(
        emit_presentation=True,
        primary_asset=None,
        render_inputs={"missing_inputs": []},
    )

    assert status == "missing"
