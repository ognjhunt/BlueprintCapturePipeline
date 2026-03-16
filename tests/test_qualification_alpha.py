from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_pipeline.capture_orchestrator import PipelineConfig, resolve_requested_lanes, run_capture_pipeline
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
            "provider_warnings": [],
            "provider_errors": [],
            "loop_closure_detected": False,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_da3_provider", _fake_provider)
    build_geometry_stage_contract(capture_root)


def test_qualification_completes_without_downstream_artifacts(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    sync_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "blueprint_pipeline.qualification.infer_capture_fidelity_review",
        lambda **_kwargs: _successful_capture_review(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.sync_webapp_pipeline_attachment",
        lambda **kwargs: sync_calls.append(kwargs) or None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _successful_privacy_processing(),
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    pipeline_root = capture_root / "pipeline"
    completion = json.loads((pipeline_root / ".qualification_pipeline_complete").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert completion["status"] == "completed"
    assert completion["alpha_scoring_status"] == "succeeded"
    assert "scene_memory_manifest" not in completion
    assert "preview_simulation_manifest" not in completion
    assert not (pipeline_root / "scene_memory").exists()
    assert (pipeline_root / "gemini_capture_fidelity_review.json").is_file()
    assert (pipeline_root / "world_model_fit_summary.json").is_file()
    assert (pipeline_root / "capturer_payout_recommendation.json").is_file()
    assert (pipeline_root / "provenance_summary.json").is_file()
    assert (pipeline_root / "buyer_trust_score.json").is_file()
    assert (pipeline_root / "provider_preview_status.json").is_file()
    assert sync_calls
    assert sync_calls[0]["artifacts"]["privacy_processed_video_uri"].endswith("/privacy/final_walkthrough.mov")
    assert sync_calls[0]["artifacts"]["world_model_video_uri"].endswith("/privacy/final_walkthrough.mov")
    assert "scene_memory_manifest_uri" not in sync_calls[0]["artifacts"]
    assert sync_calls[0]["derived_assets"] == {}
    payout = json.loads((pipeline_root / "capturer_payout_recommendation.json").read_text(encoding="utf-8"))
    quality = json.loads((pipeline_root / "capture_quality_summary.json").read_text(encoding="utf-8"))
    world_model_fit = json.loads((pipeline_root / "world_model_fit_summary.json").read_text(encoding="utf-8"))
    assert len(payout["bonus_breakdown"]) == 4
    assert payout["recommended_payout_cents"] >= payout["base_payout_cents"]
    assert quality["blur_assessment"]["status"] == "good"
    assert quality["coverage_completeness_assessment"]["status"] == "good"
    assert world_model_fit["status"] == "good_candidate"


def test_qualification_completes_when_preview_provider_fails(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path, requested_outputs=["preview_simulation"])

    monkeypatch.setattr(
        "blueprint_pipeline.qualification.infer_capture_fidelity_review",
        lambda **_kwargs: _successful_capture_review(),
    )
    monkeypatch.setenv("BLUEPRINT_PREVIEW_PROVIDER", "world_labs")
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.sync_webapp_pipeline_attachment",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _successful_privacy_processing(),
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    pipeline_root = capture_root / "pipeline"
    completion = json.loads((pipeline_root / ".qualification_pipeline_complete").read_text(encoding="utf-8"))
    provider_run = json.loads((pipeline_root / "provider_run_manifest.json").read_text(encoding="utf-8"))
    preview_status = json.loads((pipeline_root / "provider_preview_status.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert completion["status"] == "completed"
    assert provider_run["status"] == "failed"
    assert preview_status["status"] == "failed"
    assert completion["provider_run_manifest"].endswith("/provider_run_manifest.json")
    assert "scene_memory_manifest" not in completion


def test_qualification_persists_worldlabs_manifest_uris_when_preview_requested(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path, requested_outputs=["preview_simulation"])
    sync_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "blueprint_pipeline.qualification.infer_capture_fidelity_review",
        lambda **_kwargs: _successful_capture_review(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.sync_webapp_pipeline_attachment",
        lambda **kwargs: sync_calls.append(kwargs) or None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _successful_privacy_processing(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification._prepare_worldlabs_input_video",
        lambda **_kwargs: {
            "status": "ready",
            "manifest_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input_manifest.json",
            "output_video_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/worldlabs_input/worldlabs_input.mp4",
        },
    )
    monkeypatch.setenv("BLUEPRINT_PREVIEW_PROVIDER", "world_labs")

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    metadata = descriptor["metadata"]
    pipeline_root = capture_root / "pipeline"

    assert metadata["worldlabs_request_manifest_uri"].endswith("/pipeline/worldlabs_request_manifest.json")
    assert metadata["worldlabs_input_manifest_uri"].endswith("/pipeline/worldlabs_input/worldlabs_input_manifest.json")
    assert metadata["worldlabs_input_video_uri"].endswith("/pipeline/worldlabs_input/worldlabs_input.mp4")
    assert (pipeline_root / "worldlabs_request_manifest.json").is_file()
    assert sync_calls
    assert sync_calls[0]["artifacts"]["worldlabs_request_manifest_uri"].endswith("/pipeline/worldlabs_request_manifest.json")
    assert sync_calls[0]["artifacts"]["worldlabs_input_manifest_uri"].endswith("/pipeline/worldlabs_input/worldlabs_input_manifest.json")
    assert sync_calls[0]["artifacts"]["worldlabs_input_video_uri"].endswith("/pipeline/worldlabs_input/worldlabs_input.mp4")


def test_qualification_fail_closed_omits_buyer_safe_media(monkeypatch, tmp_path: Path) -> None:
    _capture_root, descriptor_uri = _build_staged_capture(tmp_path, requested_outputs=["preview_simulation"])
    sync_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "blueprint_pipeline.qualification.infer_capture_fidelity_review",
        lambda **_kwargs: _successful_capture_review(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.sync_webapp_pipeline_attachment",
        lambda **kwargs: sync_calls.append(kwargs) or None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: {
            "schema_version": "v1",
            "status": "failed_closed",
            "mode": "removal",
            "fallback_used": False,
            "people_detected": 1,
            "people_removed": 0,
            "face_anonymized_segments": [],
            "raw_retained": True,
            "fail_closed": True,
            "privacy_processed_video_uri": None,
            "world_model_video_uri": None,
            "privacy_manifest_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_processing_manifest.json",
            "privacy_verification_report_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline/privacy_verification_report.json",
        },
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert sync_calls
    assert "privacy_processed_video_uri" not in sync_calls[0]["artifacts"]
    assert "world_model_video_uri" not in sync_calls[0]["artifacts"]
    assert sync_calls[0]["deployment_readiness"]["privacy_processing"]["status"] == "failed_closed"


def test_qualification_ingests_geometry_summary_as_advisory(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    sync_calls: list[dict[str, object]] = []
    _write_geometry_lane(monkeypatch, capture_root)

    monkeypatch.setattr(
        "blueprint_pipeline.qualification.infer_capture_fidelity_review",
        lambda **_kwargs: _successful_capture_review(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.sync_webapp_pipeline_attachment",
        lambda **kwargs: sync_calls.append(kwargs) or None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.run_privacy_postprocess",
        lambda **_kwargs: _successful_privacy_processing(),
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    pipeline_root = capture_root / "pipeline"
    world_model_fit = json.loads((pipeline_root / "world_model_fit_summary.json").read_text(encoding="utf-8"))
    completion = json.loads((pipeline_root / ".qualification_pipeline_complete").read_text(encoding="utf-8"))

    assert world_model_fit["advisory_geometry"]["status"] == "completed"
    assert world_model_fit["advisory_geometry"]["ready_for_world_model"] is True
    assert world_model_fit["advisory_geometry"]["scale_status"] == "metric_trusted"
    assert completion["geometry_summary"].endswith("/geometry/geometry_summary.json")
    assert sync_calls
    assert sync_calls[0]["artifacts"]["geometry_summary_uri"].endswith("/geometry/geometry_summary.json")
    assert sync_calls[0]["deployment_readiness"]["advisory_geometry"]["status"] == "completed"


def test_resolve_requested_lanes_demotes_bridge_default_scene_memory(tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["requested_lanes"] = ["qualification", "scene_memory"]
    descriptor["requested_outputs"] = []
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri=descriptor_uri,
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification"]


def test_bad_video_review_forces_recapture_and_lower_world_model_fit(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    bad_review = _successful_capture_review()
    bad_review["scores"] = {
        **bad_review["scores"],
        "coverage": 0.4,
        "world_model_fitness": 0.3,
        "visual_clarity": 0.35,
        "lighting_stability": 0.4,
        "motion_stability": 0.3,
    }
    bad_review["assessments"] = {
        "blur": {"status": "poor", "score": 0.2, "summary": "Strong blur", "impact": "high"},
        "lighting": {"status": "poor", "score": 0.3, "summary": "Lighting changes", "impact": "high"},
        "motion_speed": {"status": "poor", "score": 0.2, "summary": "Too fast", "impact": "high"},
        "doubling_back": {"status": "review_required", "score": 0.3, "summary": "Inefficient rescans", "impact": "medium"},
        "coverage_completeness": {"status": "poor", "score": 0.4, "summary": "Missing areas", "impact": "high"},
        "task_zone_completeness": {"status": "review_required", "score": 0.45, "summary": "Task zone incomplete", "impact": "high"},
        "occlusion_and_hidden_zone": {"status": "review_required", "score": 0.35, "summary": "Occlusion present", "impact": "medium"},
        "depth_and_spatial_conditioning": {"status": "review_required", "score": 0.4, "summary": "Weak spatial conditioning", "impact": "high"},
    }
    bad_review["findings"] = {
        "missing_views": ["north aisle"],
        "blur_observations": ["motion blur in center aisle"],
        "lighting_observations": ["backlit doorway"],
        "occlusion_observations": ["stacked totes block corner"],
        "task_scope_notes": [],
        "blocker_summaries": ["capture quality is insufficient for world-model generation"],
        "recapture_recommendations": ["slow down and reshoot missing aisle coverage"],
    }

    monkeypatch.setattr(
        "blueprint_pipeline.qualification.infer_capture_fidelity_review",
        lambda **_kwargs: bad_review,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.sync_webapp_pipeline_attachment",
        lambda **_kwargs: None,
    )

    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="qualification",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    pipeline_root = capture_root / "pipeline"
    quality = json.loads((pipeline_root / "capture_quality_summary.json").read_text(encoding="utf-8"))
    world_model_fit = json.loads((pipeline_root / "world_model_fit_summary.json").read_text(encoding="utf-8"))
    recapture = json.loads((pipeline_root / "recapture_requirements.json").read_text(encoding="utf-8"))
    qualification = json.loads((pipeline_root / "qualification_record.json").read_text(encoding="utf-8"))

    assert quality["blur_assessment"]["status"] == "poor"
    assert quality["motion_speed_assessment"]["status"] == "poor"
    assert quality["coverage_completeness_assessment"]["status"] == "poor"
    assert world_model_fit["status"] == "review_required"
    assert recapture["required"] is True
    assert recapture["recommendations"]
    assert qualification["readiness_state"] == "not_ready_yet"
