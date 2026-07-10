from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


pytestmark = pytest.mark.slow

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.capture_bridge import CaptureDescriptor  # noqa: E402
import blueprint_pipeline.geometry_stage as geometry_stage  # noqa: E402
from blueprint_pipeline.common import PipelineError  # noqa: E402
from blueprint_pipeline.geometry_stage import (  # noqa: E402
    GeometryStageResult,
    _build_canonical_geometry_artifacts,
    _build_dynamic_mask_manifest,
    _build_fallback_provider_result,
    _confidence_summary,
    _optional_json,
    _probe_video,
    _resolve_video_path,
    _run_geometry_provider,
    _summary_capture_source,
    _track_length_m,
    _write_ascii_pointcloud,
    assess_geometry_scale,
    build_geometry_stage_contract,
)
from blueprint_pipeline.materialization import materialize_capture_bundle  # noqa: E402


def _build_staged_capture(
    tmp_path: Path,
    *,
    manifest_overrides: dict[str, object] | None = None,
    context_overrides: dict[str, object] | None = None,
) -> Path:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    capture_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "scene_id": scene_id,
        "capture_id": capture_id,
        "video_uri": "walkthrough.mov",
        "capture_source": "glasses",
        "width": 1280,
        "height": 720,
        "requested_outputs": ["qualification"],
        "disable_default_preview": True,
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "consent_status": "documented",
            "permission_document_uri": "gs://local-blueprint/rights/consent-packet.pdf",
        },
    }
    if manifest_overrides:
        manifest_payload.update(manifest_overrides)
    (raw_root / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
    (raw_root / "intake_packet.json").write_text(
        json.dumps(
            {
                "workflowName": "Walk aisle",
                "taskSteps": ["Walk aisle"],
                "zone": "aisle-a",
                "owner": "ops",
            }
        ),
        encoding="utf-8",
    )
    context_payload = {
        "sceneId": scene_id,
        "captureId": capture_id,
        "captureSource": manifest_payload.get("capture_source", "glasses"),
        "captureModality": "glasses_video_only",
    }
    if context_overrides:
        context_payload.update(context_overrides)
    (raw_root / "capture_context.json").write_text(json.dumps(context_payload), encoding="utf-8")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")

    materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
    )
    return capture_root


def _write_frame_artifacts(base_dir: Path, *, frame_count: int = 3) -> list[dict[str, object]]:
    frames_dir = base_dir / "frames" / "images"
    depth_dir = base_dir / "depth"
    confidence_dir = base_dir / "confidence"
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for frame_index in range(frame_count):
        image_path = frames_dir / f"frame_{frame_index:06d}.npy"
        np.save(image_path, np.full((24, 32, 3), 40 + frame_index * 20, dtype=np.float32))
        depth_path = depth_dir / f"depth_{frame_index:06d}.npy"
        confidence_path = confidence_dir / f"confidence_{frame_index:06d}.npy"
        depth = np.full((24, 32), 1.0 + frame_index * 0.25, dtype=np.float32)
        confidence = np.full((24, 32), 0.9 - frame_index * 0.1, dtype=np.float32)
        np.save(depth_path, depth)
        np.save(confidence_path, confidence)
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": float(frame_index) * 0.5,
                "image_path": str(image_path),
                "is_keyframe": frame_index != 1,
                "blur_score": 0.1 + frame_index * 0.05,
                "overlap_hint": 0.9 - frame_index * 0.1,
                "world_from_camera": [
                    [1.0, 0.0, 0.0, frame_index * 0.2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "camera_from_world": [
                    [1.0, 0.0, 0.0, -(frame_index * 0.2)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "pose_confidence": 0.95 - frame_index * 0.05,
                "depth_path": str(depth_path),
                "depth_format": "npy",
                "confidence_path": str(confidence_path),
                "confidence_format": "npy",
                "width": 32,
                "height": 24,
                "min_depth_m": float(depth.min()),
                "max_depth_m": float(depth.max()),
                "depth_unit": "meters",
                "metric_depth_truth": True,
                "depth_measurement_source": "provider_metric_reconstruction",
                "confidence_range": [0.0, 1.0],
            }
        )
    return records


def test_build_geometry_stage_contract_writes_completed_outputs(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _fake_provider(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        return {
            "intrinsics": {
                "camera_model": "pinhole",
                "image_width": 32,
                "image_height": 24,
                "fx": 28.0,
                "fy": 29.0,
                "cx": 16.0,
                "cy": 12.0,
                "distortion": {"model": "none", "coefficients": []},
            },
            "frames": _write_frame_artifacts(geometry_root),
            "provider_metrics": {"backend": "test"},
            "provider_warnings": [],
            "provider_errors": [],
            "loop_closure_detected": False,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 0.92,
            "p95_pose_delta_sec": 0.033,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _fake_provider)

    result = build_geometry_stage_contract(capture_root)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    depth_manifest = json.loads(
        (capture_root / "pipeline" / "geometry" / "depth" / "depth_manifest.json").read_text(encoding="utf-8")
    )
    confidence_manifest = json.loads(
        (capture_root / "pipeline" / "geometry" / "confidence" / "confidence_manifest.json").read_text(encoding="utf-8")
    )

    assert result.status == "completed"
    assert manifest["status"] == "completed"
    assert summary["status"] == "completed"
    assert status["status"] == "completed"
    assert status["ready_for_world_model"] is True
    assert summary["ready_for_world_model"] is True
    assert summary["geometry_source"] == "video_to_world"
    assert summary["fallback_used"] is False
    assert summary["provider_native_result"] is True
    assert summary["geometry_live_ready"] is True
    assert summary["site_frame_available"] is True
    assert summary["scale_resolved"] is True
    assert summary["pose_track_count"] == 3
    assert summary["pose_match_rate"] == 0.92
    assert summary["p95_pose_delta_sec"] == 0.033
    assert summary["deliverables"]["pose_count"] == 3
    assert summary["deliverables"]["depth_frame_count"] == 3
    assert depth_manifest["frame_count"] == 3
    assert confidence_manifest["frame_count"] == 3
    assert summary["scale_assessment"]["status"] == "conditioning_only"
    assert (capture_root / "pipeline" / "geometry" / "camera" / "poses.jsonl").read_text(encoding="utf-8")
    assert (capture_root / "pipeline" / "geometry" / "alignment" / "canonical_pointcloud.ply").is_file()
    assert (capture_root / "pipeline" / "geometry" / "masks" / "dynamic_mask_manifest.json").is_file()


def test_build_geometry_stage_contract_records_failed_status(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _failing_provider(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _failing_provider)

    result = build_geometry_stage_contract(capture_root)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    provider_result = json.loads(
        (capture_root / "pipeline" / "geometry" / "logs" / "provider_result.json").read_text(encoding="utf-8")
    )
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))

    assert result.status == "completed_with_fallback"
    assert manifest["status"] == "completed_with_fallback"
    assert summary["status"] == "completed_with_fallback"
    assert summary["fallback_used"] is True
    assert summary["fallback_kind"] == "internal_synthetic_geometry"
    assert summary["geometry_source"] == "fallback_geometry"
    assert summary["ready_for_world_model"] is False
    assert summary["contract_ready_for_world_model"] is False
    assert summary["internal_fallback_ready"] is False
    assert summary["diagnostic_artifacts_shape_ready"] is True
    assert summary["synthetic_geometry_used"] is True
    assert summary["synthetic_artifacts_are_capture_truth"] is False
    assert summary["geometry_live_ready"] is False
    assert summary["site_faithful_market_ready"] is False
    assert "fallback_geometry_not_live_video_to_world" in summary["launch_blockers"]
    assert "synthetic_geometry_not_capture_truth" in summary["launch_blockers"]
    assert status["status"] == "completed_with_fallback"
    assert status["geometry_source"] == "fallback_geometry"
    assert status["fallback_used"] is True
    assert status["ready_for_world_model"] is False
    assert status["geometry_live_ready"] is False
    assert manifest["provider"]["fallback_used"] is True
    assert manifest["provider"]["provider_native_result"] is False
    assert manifest["world_model_contract"]["truth_label"] == "synthetic_diagnostic_not_capture_truth"
    assert provider_result["status"] == "provider_failed_synthetic_diagnostics_written"
    assert "boom" in provider_result["errors"]
    assert descriptor["geometry_ready"] is False
    assert descriptor["quality"]["geometry_ready"] is False
    assert descriptor["quality"]["geometry_live_ready"] is False
    assert descriptor["quality"]["fallback_used"] is True
    assert descriptor["world_model_candidate"] is False
    assert descriptor["metadata"]["geometry"]["ready_for_world_model"] is False
    assert descriptor["metadata"]["geometry"]["internal_fallback_ready"] is False


def test_local_sfm_writes_synthetic_diagnostic_geometry_summary(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"capture_source": "meta_glasses", "capture_profile_id": "glasses_pov"},
        context_overrides={"captureSource": "meta_glasses", "captureModality": "glasses_video_only"},
    )

    result = build_geometry_stage_contract(capture_root, provider="local_sfm", model="local-sfm-offline")

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    provider_result = json.loads(
        (capture_root / "pipeline" / "geometry" / "logs" / "provider_result.json").read_text(encoding="utf-8")
    )

    assert result.status == "completed_with_fallback"
    assert summary["geometry_source"] == "fallback_geometry"
    assert summary["capture_source"] == "meta_glasses"
    # local_sfm reuses the synthetic fabricator, so truth flags must say so.
    assert summary["fallback_used"] is True
    assert summary["fallback_kind"] == "internal_synthetic_geometry"
    assert summary["synthetic_geometry"] is True
    assert summary["provider_native_result"] is False
    assert summary["contract_ready_for_world_model"] is False
    assert summary["diagnostic_artifacts_shape_ready"] is True
    assert summary["ready_for_world_model"] is False
    assert summary["geometry_live_ready"] is False
    assert summary["intrinsics_available"] is True
    assert summary["site_frame_available"] is False
    assert summary["scale_resolved"] is False
    assert summary["pose_track_count"] > 0
    assert "provider_native_geometry_missing" in summary["blockers"]
    assert "synthetic_geometry_not_capture_truth" in summary["blockers"]
    assert "scale_not_proven" in summary["blockers"]
    assert provider_result["geometry_source"] == "fallback_geometry"
    assert provider_result["metrics"]["requested_backend"] == "local_sfm_offline"
    assert provider_result["metrics"]["real_sfm_runner_executed"] is False
    assert provider_result["provider_native_result"] is False


def test_video_to_world_missing_env_writes_provider_blocker_without_live_call(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.delenv("VIDEO_TO_WORLD_URL", raising=False)
    monkeypatch.delenv("VIDEO_TO_WORLD_RUNNER_TOKEN", raising=False)

    def _provider_should_not_run(**_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("video_to_world provider should be gated before live call")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _provider_should_not_run)

    result = build_geometry_stage_contract(capture_root, provider="video_to_world")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.status == "completed_with_fallback"
    assert summary["geometry_source"] == "fallback_geometry"
    assert summary["fallback_used"] is True
    assert summary["synthetic_geometry"] is True
    assert summary["provider_native_result"] is False
    assert summary["contract_ready_for_world_model"] is False
    assert summary["diagnostic_artifacts_shape_ready"] is True
    assert summary["geometry_live_ready"] is False
    assert "provider_native_geometry_missing" in summary["blockers"]
    assert "video_to_world_runner_not_configured" in summary["blockers"]
    assert "synthetic_geometry_not_capture_truth" in summary["blockers"]
    assert summary["provider_blocker"]["required_env"] == ["VIDEO_TO_WORLD_URL", "VIDEO_TO_WORLD_RUNNER_TOKEN"]
    assert "scripts/run_geometry_lane.py" in summary["provider_blocker"]["command"]


def test_production_provider_failure_blocks_without_fabricating_geometry(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _failing_provider(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _failing_provider)

    result = build_geometry_stage_contract(capture_root)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    geometry_root = capture_root / "pipeline" / "geometry"

    assert result.status == "blocked_geometry_unavailable"
    assert summary["status"] == "blocked_geometry_unavailable"
    assert summary["geometry_source"] == "unavailable"
    assert summary["fallback_used"] is False
    assert summary["synthetic_geometry"] is False
    assert summary["ready_for_world_model"] is False
    assert summary["external_market_ready"] is False
    assert "geometry_provider_unavailable" in summary["launch_blockers"]
    assert "synthetic_geometry_disallowed" in summary["launch_blockers"]
    assert status["status"] == "blocked_geometry_unavailable"
    assert status["ready_for_world_model"] is False
    # No fabricated tensors may exist anywhere in the geometry lane.
    assert not list((geometry_root / "depth").glob("*.npy"))
    assert not list((geometry_root / "confidence").glob("*.npy"))
    assert not list((geometry_root / "frames").rglob("*.npy"))
    assert not (geometry_root / "camera" / "poses.jsonl").exists()
    assert not (geometry_root / "camera" / "intrinsics.json").exists()


def test_production_local_sfm_request_blocks_without_fabricating_geometry(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")

    result = build_geometry_stage_contract(capture_root, provider="local_sfm", model="local-sfm-offline")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.status == "blocked_geometry_unavailable"
    assert "synthetic_geometry_disallowed" in summary["launch_blockers"]
    assert "local_sfm_synthetic_dev_requested" in summary["launch_blockers"]
    geometry_root = capture_root / "pipeline" / "geometry"
    assert not list((geometry_root / "depth").glob("*.npy"))
    assert not (geometry_root / "camera" / "poses.jsonl").exists()


def test_synthetic_geometry_disabled_env_blocks_dev_fallback(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.delenv("BLUEPRINT_LAUNCH_PROOF_MODE", raising=False)
    monkeypatch.setenv("BLUEPRINT_ALLOW_SYNTHETIC_GEOMETRY", "0")
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _failing_provider(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _failing_provider)

    result = build_geometry_stage_contract(capture_root)
    assert result.status == "blocked_geometry_unavailable"


def test_geometry_verification_rejects_empty_external_and_corrupt_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _invalid_provider(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        frames = _write_frame_artifacts(geometry_root)
        frames[0]["world_from_camera"] = []
        outside_depth = tmp_path / "outside_depth.npy"
        np.save(outside_depth, np.ones((24, 32), dtype=np.float32))
        frames[1]["depth_path"] = str(outside_depth)
        corrupt_confidence = Path(str(frames[2]["confidence_path"]))
        corrupt_confidence.write_bytes(b"not-a-decodable-array")
        return {
            "intrinsics": {
                "camera_model": "pinhole",
                "image_width": 32,
                "image_height": 24,
                "fx": 28.0,
                "fy": 29.0,
                "cx": 16.0,
                "cy": 12.0,
            },
            "frames": frames,
            "provider_native_result": True,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 1.0,
            "p95_pose_delta_sec": 0.01,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _invalid_provider)
    result = build_geometry_stage_contract(capture_root)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    report = json.loads(
        (capture_root / "pipeline" / "geometry" / "geometry_validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    reasons = {reason for rejection in report["rejections"] for reason in rejection["blockers"]}

    assert result.status == "completed_degraded"
    assert summary["verified_geometry_record_count"] == 0
    assert summary["scale_assessment"]["pose_coverage"] == 0.0
    assert summary["scale_assessment"]["depth_coverage"] == 0.0
    assert summary["ready_for_world_model"] is False
    assert "geometry_records_failed_verification" in summary["blockers"]
    assert "world_from_camera_missing_misshaped_or_nonfinite" in reasons
    assert "depth_tensor_missing_corrupt_or_shape_mismatch" in reasons
    assert "confidence_tensor_missing_corrupt_or_shape_mismatch" in reasons
    assert not (capture_root / "pipeline" / "geometry" / "alignment" / "canonical_pointcloud.ply").exists()


def test_blocked_rerun_quarantines_prior_tensors_and_clears_descriptor_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _valid_provider(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        return {
            "intrinsics": {
                "image_width": 32,
                "image_height": 24,
                "fx": 28.0,
                "fy": 29.0,
                "cx": 16.0,
                "cy": 12.0,
            },
            "frames": _write_frame_artifacts(geometry_root),
            "provider_native_result": True,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 1.0,
            "p95_pose_delta_sec": 0.01,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _valid_provider)
    first = build_geometry_stage_contract(capture_root)
    assert first.status == "completed"
    assert (capture_root / "pipeline" / "geometry" / "depth" / "depth_000000.npy").is_file()

    monkeypatch.setenv("BLUEPRINT_ALLOW_SYNTHETIC_GEOMETRY", "0")

    def _blocked_provider(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _blocked_provider)
    second = build_geometry_stage_contract(capture_root)
    summary = json.loads(second.summary_path.read_text(encoding="utf-8"))
    status = json.loads(second.status_path.read_text(encoding="utf-8"))
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    lineage = json.loads(
        (capture_root / "pipeline" / "geometry" / "previous_run_lineage.json").read_text(encoding="utf-8")
    )

    assert second.status == "blocked_geometry_unavailable"
    assert summary["current_usable_tensor_count"] == 0
    assert status["current_usable_tensor_count"] == 0
    assert not list((capture_root / "pipeline" / "geometry").rglob("*.npy"))
    assert descriptor["geometry_ready"] is False
    assert descriptor["geometry_live_ready"] is False
    assert lineage["previous_status"] == "completed"
    assert Path(lineage["previous_run_archive_path"]).is_dir()
    assert list(Path(lineage["previous_run_archive_path"]).rglob("*.npy"))


def test_assess_geometry_scale_respects_metric_policy() -> None:
    base = {
        "schema_version": "v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "capture_source": "glasses",
        "capture_tier": "tier2_glasses",
        "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
        "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/frames/index.jsonl",
        "capture_modality": "glasses_video_only",
        "evidence_tier": "pre_screen_video",
        "quality": {},
        "metadata": {},
    }

    descriptor = CaptureDescriptor.from_dict(base)
    assert assess_geometry_scale(descriptor)["status"] == "conditioning_only"

    descriptor = CaptureDescriptor.from_dict(
        {
            **base,
            "capture_modality": "glasses_plus_scaffolding",
            "scaffolding_validation": {"validated_metric_bundle": False},
        }
    )
    assert assess_geometry_scale(descriptor)["metric_trusted"] is False

    descriptor = CaptureDescriptor.from_dict(
        {
            **base,
            "capture_modality": "glasses_plus_scaffolding",
            "evidence_tier": "video_with_validated_scaffolding",
            "scaffolding_validation": {"validated_metric_bundle": True},
        }
    )
    assert assess_geometry_scale(descriptor)["status"] == "metric_trusted"

    descriptor = types.SimpleNamespace(
        capture_source="android",
        capture_modality="android_arcore_depth",
        evidence_tier="pre_screen_video",
        scaffolding_validation={},
    )
    assert assess_geometry_scale(descriptor)["reason"] == "raw_tracking_without_validated_scale"

    descriptor = types.SimpleNamespace(
        capture_source="drone",
        capture_modality="drone_video",
        evidence_tier="pre_screen_video",
        scaffolding_validation={},
    )
    assert assess_geometry_scale(descriptor)["status"] == "estimated_scale"

    descriptor = CaptureDescriptor.from_dict(
        {
            **base,
            "capture_source": "android",
            "capture_tier": "tier2_android",
            "capture_modality": "android_plus_scaffolding",
            "evidence_tier": "video_with_validated_scaffolding",
            "scaffolding_validation": {"validated_metric_bundle": True},
        }
    )
    assert assess_geometry_scale(descriptor)["status"] == "metric_trusted"

    descriptor = CaptureDescriptor.from_dict(
        {
            **base,
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "capture_modality": "iphone_arkit_lidar",
            "evidence_tier": "qualified_metric_capture",
        }
    )
    assert assess_geometry_scale(descriptor)["status"] == "metric_trusted"


def test_capture_descriptor_accepts_robot_eval_alias_requested_lanes() -> None:
    descriptor = CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
            "frames_index_uri": "gs://bucket/scenes/scene-1/captures/capture-1/frames/index.jsonl",
            "requested_lanes": ["robot_eval_dataset", "task_evaluation_run"],
        }
    )

    assert descriptor.requested_lanes == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]


def test_materialization_preserves_android_video_only_capture_source(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"capture_source": "android"},
        context_overrides={"captureSource": "android", "captureModality": "android_video_only"},
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["capture_source"] == "android"
    assert descriptor["capture_modality"] == "android_video_only"


def test_materialization_preserves_iphone_video_only_capture_modality(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"capture_source": "iphone", "has_lidar": False},
        context_overrides={"captureSource": "iphone", "captureModality": "iphone_video_only"},
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))

    assert descriptor["capture_source"] == "iphone"
    assert descriptor["capture_modality"] == "iphone_video_only"
    assert descriptor["evidence_tier"] == "pre_screen_video"


def test_materialization_attaches_route_anchor_sidecars(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"capture_source": "iphone", "has_lidar": False},
        context_overrides={"captureSource": "iphone", "captureModality": "iphone_video_only"},
    )
    raw_root = capture_root / "raw"
    (raw_root / "route_anchors.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "route_anchors": [
                    {
                        "anchor_id": "anchor_entry",
                        "anchor_type": "entry",
                        "label": "Entry",
                        "expected_observation": "pause_and_pan",
                        "required_in_primary_pass": True,
                        "required_in_revisit_pass": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "checkpoint_events.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "checkpoint_events": [
                    {
                        "anchor_id": "anchor_entry",
                        "pass_id": "pass-1",
                        "t_capture_sec": 0.25,
                        "hold_duration_sec": 1.1,
                        "completed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    materialize_capture_bundle(
        bucket="local-blueprint",
        scene_id="scene-1",
        capture_id="capture-1",
        gcs_root=tmp_path,
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    metadata = descriptor["metadata"]
    assert metadata["route_anchors"]["route_anchors"][0]["anchor_id"] == "anchor_entry"
    assert metadata["checkpoint_events"]["checkpoint_events"][0]["anchor_id"] == "anchor_entry"


def test_materialization_maps_preview_simulation_to_current_requested_lanes(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"requested_outputs": ["preview_simulation"]},
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["requested_lanes"] == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]


def test_materialization_maps_robot_eval_outputs_to_current_requested_lanes(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"requested_outputs": ["robot_eval_dataset"]},
    )
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["requested_lanes"] == ["qualification", "evaluation_prep"]

    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"requested_outputs": ["task_evaluation_run"]},
    )
    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["requested_lanes"] == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]


def test_materialization_keeps_explicit_scene_memory_as_legacy_requested_lane(
    tmp_path: Path,
) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"requested_outputs": ["scene_memory"]},
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["requested_lanes"] == ["qualification", "scene_memory"]


def test_materialization_defaults_preview_simulation_for_plain_uploads(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={
            "requested_outputs": None,
            "disable_default_preview": False,
        },
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["requested_outputs"] == ["qualification", "preview_simulation"]
    assert descriptor["requested_lanes"] == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]


def test_materialization_promotes_android_scaffolding_to_metric_ready_video(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={
            "capture_source": "android_phone",
            "capture_tier_hint": "tier2_android_phone",
            "scaffolding_used": ["checkerboard_calibration"],
            "calibration_assets": ["checkerboard_01.jpg"],
            "scaffolding_validation": {
                "validated_scale_m": 4.0,
                "validated_pose_coverage": 0.82,
                "hidden_zone_bound": 0.2,
            },
        },
        context_overrides={
            "captureSource": "android_phone",
            "captureModality": "android_plus_scaffolding",
            "scaleAnchorAssets": ["anchor_a"],
            "checkpointAssets": ["checkpoint_a"],
            "validatedScaleMeters": 4.0,
            "validatedPoseCoverage": 0.82,
            "hiddenZoneBound": 0.2,
        },
    )

    descriptor = json.loads((capture_root / "capture_descriptor.json").read_text(encoding="utf-8"))
    qa_report = json.loads((capture_root / "qa_report.json").read_text(encoding="utf-8"))

    assert descriptor["capture_source"] == "android"
    assert descriptor["capture_tier"] == "tier2_android"
    assert descriptor["capture_modality"] == "android_plus_scaffolding"
    assert descriptor["evidence_tier"] == "video_with_validated_scaffolding"
    # Metric scaffolding is accepted, but this legacy fixture deliberately has
    # no current v3 raw-integrity inventory, so the overall launch QA must stay
    # degraded rather than laundering legacy bytes into current proof.
    assert qa_report["status"] == "degraded"
    checks = {row["name"]: row["passed"] for row in qa_report["checks"]}
    assert checks["metric_geometry_present"] is True
    assert checks["scaffolding_validated"] is True
    assert checks["raw_bundle_integrity"] is False


def test_geometry_stage_helper_edges(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    context = geometry_stage.resolve_local_capture_context(capture_root)
    descriptor = CaptureDescriptor.from_dict(
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "capture_source": "iphone",
            "capture_tier": "tier1_iphone",
            "raw_prefix_uri": "gs://bucket/raw",
            "frames_index_uri": "gs://bucket/frames.jsonl",
        }
    )

    result = GeometryStageResult(
        capture_root=capture_root,
        geometry_root=capture_root / "pipeline" / "geometry",
        manifest_path=capture_root / "pipeline" / "geometry" / "geometry_manifest.json",
        summary_path=capture_root / "pipeline" / "geometry" / "geometry_summary.json",
        status_path=capture_root / "pipeline" / "geometry" / "geometry_status.json",
        status="completed",
    )
    assert result.to_dict()["status"] == "completed"
    assert _summary_capture_source(descriptor) == "iphone"
    assert _optional_json(tmp_path / "missing.json") == {}
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{bad-json", encoding="utf-8")
    assert _optional_json(invalid_json) == {}

    no_video_root = _build_staged_capture(tmp_path / "no-video")
    (no_video_root / "raw" / "walkthrough.mov").unlink()
    with pytest.raises(PipelineError, match="No walkthrough video"):
        _resolve_video_path(geometry_stage.resolve_local_capture_context(no_video_root))

    def _fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return types.SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "audio"},
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1920,
                            "height": 1080,
                            "pix_fmt": "yuv420p",
                            "avg_frame_rate": "30/1",
                        },
                    ],
                    "format": {"duration": "4.5", "bit_rate": "800000"},
                }
            )
        )

    monkeypatch.setattr(geometry_stage.subprocess, "run", _fake_run)
    probe = _probe_video(capture_root / "raw" / "walkthrough.mov")
    assert probe["probe_status"] == "ok"
    assert probe["width"] == 1920
    assert probe["duration_seconds"] == "4.5"

    assert _track_length_m([{"world_from_camera": "bad"}]) == 0.0
    assert _confidence_summary([]) == {"mean_pose_confidence": 0.0, "min_pose_confidence": 0.0}

    pointcloud_path = tmp_path / "pointcloud.ply"
    _write_ascii_pointcloud(
        pointcloud_path,
        [
            {"world_from_camera": "bad"},
            {"world_from_camera": [[1.0], [0.0], [0.0], [0.0]]},
            {
                "world_from_camera": [
                    [1.0, 0.0, 0.0, "bad"],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
        ],
    )
    # All-malformed poses yield an honest empty pointcloud, never a
    # fabricated origin vertex.
    pointcloud_text = pointcloud_path.read_text(encoding="utf-8")
    assert "element vertex 0" in pointcloud_text
    assert "0.000000 0.000000 0.000000" not in pointcloud_text

    fallback = _build_fallback_provider_result(
        video_path=capture_root / "raw" / "walkthrough.mov",
        geometry_root=tmp_path / "fallback-geometry",
        video_probe={"width": 320, "height": 240, "duration_seconds": 2.0},
        provider_error=RuntimeError("offline"),
    )
    assert fallback["intrinsics"]["fx"] == 32.0

    privacy_mask = capture_root / "privacy" / "masks" / "mask.png"
    privacy_mask.parent.mkdir(parents=True)
    privacy_mask.write_bytes(b"mask")
    (privacy_mask.parent / "nested").mkdir()
    context.pipeline_root.mkdir(parents=True, exist_ok=True)
    (context.pipeline_root / "privacy_processing_manifest.json").write_text("{}", encoding="utf-8")
    mask_manifest_path = _build_dynamic_mask_manifest(
        context=context,
        geometry_root=tmp_path / "geometry-masks",
    )
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    assert mask_manifest["mask_source"] == "privacy_processing"
    assert mask_manifest["artifacts"][0]["relative_path"] == "privacy/masks/mask.png"

    source_pointcloud = tmp_path / "canonical-geometry" / "provider" / "source.ply"
    source_pointcloud.parent.mkdir(parents=True)
    source_pointcloud.write_text("ply\ncopied\n", encoding="utf-8")
    canonical = _build_canonical_geometry_artifacts(
        context=context,
        geometry_root=tmp_path / "canonical-geometry",
        pose_records=[],
        geometry_source="video_to_world",
        fallback_used=False,
        coordinate_frame_session_id="session-1",
        canonical_pointcloud_source_path=str(source_pointcloud),
    )
    assert canonical["canonical_pointcloud_path"].read_text(encoding="utf-8") == "ply\ncopied\n"


def test_geometry_stage_da3_and_empty_frame_edges(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)

    def _fake_da3_fallback(**_kwargs):  # type: ignore[no-untyped-def]
        return {
            "provider_metrics": {"fallback_used": True},
            "provider_warnings": ["existing"],
        }

    monkeypatch.setattr(geometry_stage, "run_da3_provider", _fake_da3_fallback)
    da3_result = _run_geometry_provider(
        video_path=capture_root / "raw" / "walkthrough.mov",
        video_uri="gs://bucket/video.mov",
        geometry_root=tmp_path / "da3",
        dynamic_mask_manifest_path=tmp_path / "masks.json",
        dynamic_mask_manifest_uri="gs://bucket/masks.json",
        provider="da3",
        model="depth-anything",
        execution_mode="offline",
        video_probe={},
    )
    assert da3_result["fallback_used"] is True
    assert da3_result["fallback_kind"] == "local_da3_synthetic_depth"
    assert "local_da3_synthetic_depth_used" in da3_result["provider_warnings"]

    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _empty_provider(**_kwargs):  # type: ignore[no-untyped-def]
        return {"frames": []}

    monkeypatch.setattr(geometry_stage, "run_video_to_world_provider", _empty_provider)
    blocked = build_geometry_stage_contract(capture_root)
    assert blocked.status == "blocked_geometry_unavailable"
    blocked_summary = json.loads(blocked.summary_path.read_text(encoding="utf-8"))
    assert blocked_summary["current_usable_tensor_count"] == 0
    assert blocked_summary["blocked_reason"] == "provider_returned_zero_frame_records"

    def _da3_provider_without_intrinsics(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        return {
            "frames": _write_frame_artifacts(geometry_root, frame_count=1),
            "provider_metrics": {"backend": "da3"},
            "provider_warnings": [],
            "provider_errors": [],
            "provider_native_result": True,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 0.8,
            "p95_pose_delta_sec": 0.04,
        }

    monkeypatch.setattr(geometry_stage, "run_da3_provider", _da3_provider_without_intrinsics)
    da3_capture_root = _build_staged_capture(tmp_path / "da3-build")
    da3_build = build_geometry_stage_contract(
        da3_capture_root,
        provider="da3",
        model="depth-anything",
    )
    summary = json.loads(da3_build.summary_path.read_text(encoding="utf-8"))
    assert summary["geometry_source"] == "local_da3"
    assert "local_da3_not_live_video_to_world" in summary["launch_blockers"]
    assert "intrinsics_missing" in summary["launch_blockers"]


def test_local_sfm_builder_truth_flags_are_append_only(tmp_path: Path) -> None:
    local = geometry_stage._build_local_sfm_provider_result(
        video_path=tmp_path / "walkthrough.mp4",
        geometry_root=tmp_path / "geometry",
        video_probe={"width": 320, "height": 240, "duration_seconds": 1.0},
        provider_blocker=None,
    )
    # The local_sfm relabel must never clear the synthetic/fallback truth.
    assert local["fallback_used"] is True
    assert local["synthetic_geometry"] is True
    assert local["fallback_kind"] == "internal_synthetic_geometry"
    assert local["requested_geometry_source"] == "local_sfm"
    assert "synthetic_geometry_used" in local["provider_warnings"]


def test_reconcile_geometry_truth_flags_cannot_weaken() -> None:
    from blueprint_pipeline.geometry_sources import reconcile_geometry_truth_flags

    # A later relabel that resets fallback_used cannot launder synthetic truth.
    laundered = {
        "fallback_used": False,
        "fallback_kind": "local_sfm_synthetic_dev",
        "synthetic_geometry": False,
    }
    synthetic, fallback_used = reconcile_geometry_truth_flags(laundered)
    assert synthetic is True
    assert fallback_used is True

    honest_provider = {"fallback_used": False, "fallback_kind": None}
    synthetic, fallback_used = reconcile_geometry_truth_flags(honest_provider)
    assert synthetic is False
    assert fallback_used is False


def test_geometry_export_gate_blocks_synthetic_in_production(monkeypatch) -> None:
    from blueprint_pipeline.geometry_sources import (
        SyntheticGeometryExportError,
        geometry_export_gate,
    )

    synthetic_summary = {"fallback_used": True, "fallback_kind": "internal_synthetic_geometry"}

    with pytest.raises(SyntheticGeometryExportError):
        geometry_export_gate(
            synthetic_summary,
            export_name="cosmos_training_export",
            env={"BLUEPRINT_LAUNCH_PROOF_MODE": "production"},
        )

    stamp = geometry_export_gate(synthetic_summary, export_name="cosmos_training_export", env={})
    assert stamp["synthetic_geometry"] is True
    assert stamp["export_allowed_by"] == "synthetic_geometry_dev_allowance"

    clean = geometry_export_gate({"fallback_used": False}, env={"BLUEPRINT_LAUNCH_PROOF_MODE": "production"})
    assert clean["export_allowed_by"] == "provider_geometry"
