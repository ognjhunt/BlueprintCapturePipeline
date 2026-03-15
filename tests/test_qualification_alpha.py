from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_orchestrator import PipelineConfig, resolve_requested_lanes, run_capture_pipeline
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
        "requested_outputs": requested_outputs or [],
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
    assert "scene_memory_manifest_uri" not in sync_calls[0]["artifacts"]
    assert sync_calls[0]["derived_assets"] == {}
    payout = json.loads((pipeline_root / "capturer_payout_recommendation.json").read_text(encoding="utf-8"))
    assert len(payout["bonus_breakdown"]) == 4
    assert payout["recommended_payout_cents"] >= payout["base_payout_cents"]


def test_qualification_completes_when_preview_provider_fails(monkeypatch, tmp_path: Path) -> None:
    capture_root, descriptor_uri = _build_staged_capture(tmp_path, requested_outputs=["preview_simulation"])

    monkeypatch.setattr(
        "blueprint_pipeline.qualification.infer_capture_fidelity_review",
        lambda **_kwargs: _successful_capture_review(),
    )
    monkeypatch.setenv("BLUEPRINT_PREVIEW_PROVIDER", "world_labs")
    monkeypatch.delenv("WORLDLABS_API_KEY", raising=False)
    monkeypatch.delenv("WORLDLABS_API_URL", raising=False)
    monkeypatch.setattr(
        "blueprint_pipeline.qualification.sync_webapp_pipeline_attachment",
        lambda **_kwargs: None,
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
