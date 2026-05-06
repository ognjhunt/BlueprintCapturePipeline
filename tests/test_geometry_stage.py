from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.capture_bridge import CaptureDescriptor  # noqa: E402
from blueprint_pipeline.geometry_stage import assess_geometry_scale, build_geometry_stage_contract  # noqa: E402
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
                "confidence_range": [0.0, 1.0],
            }
        )
    return records


def test_build_geometry_stage_contract_writes_completed_outputs(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)

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

    assert result.status == "completed_with_fallback"
    assert manifest["status"] == "completed"
    assert summary["status"] == "completed"
    assert summary["fallback_used"] is True
    assert summary["geometry_source"] == "fallback_geometry"
    assert status["status"] == "completed"
    assert status["ready_for_world_model"] is True
    assert provider_result["status"] == "completed"
    assert "boom" in provider_result["errors"]


def test_production_fallback_geometry_cannot_mark_launch_ready(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_capture(tmp_path)
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")

    def _failing_provider(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _failing_provider)

    result = build_geometry_stage_contract(capture_root)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    status = json.loads(result.status_path.read_text(encoding="utf-8"))

    assert summary["fallback_used"] is True
    assert summary["contract_ready_for_world_model"] is True
    assert summary["ready_for_world_model"] is False
    assert summary["external_market_ready"] is False
    assert summary["site_faithful_market_ready"] is False
    assert "fallback_geometry_not_launchable" in summary["launch_blockers"]
    assert status["ready_for_world_model"] is False


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


def test_materialization_maps_preview_simulation_to_scene_memory_requested_lanes(tmp_path: Path) -> None:
    capture_root = _build_staged_capture(
        tmp_path,
        manifest_overrides={"requested_outputs": ["preview_simulation"]},
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
    assert descriptor["requested_lanes"] == ["qualification", "scene_memory"]


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
    assert qa_report["status"] == "passed"
