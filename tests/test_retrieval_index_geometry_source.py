from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.slow

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.geometry_stage import build_geometry_stage_contract  # noqa: E402
from blueprint_pipeline.materialization import materialize_capture_bundle  # noqa: E402
from blueprint_pipeline.retrieval_index_stage import run_retrieval_index_stage  # noqa: E402


def _build_staged_glasses_capture(tmp_path: Path, *, with_privacy_video: bool = False) -> Path:
    bucket = "local-blueprint"
    scene_id = "scene-1"
    capture_id = "capture-1"
    capture_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "video_uri": "walkthrough.mov",
                "capture_source": "glasses",
                "capture_rights": {
                    "derived_scene_generation_allowed": True,
                    "consent_status": "documented",
                    "permission_document_uri": "gs://local-blueprint/rights/consent-packet.pdf",
                },
                "site_identity": {
                    "site_id": "site-1",
                    "site_id_source": "test",
                },
                "requested_outputs": ["qualification"],
                "disable_default_preview": True,
                "capture_mode": {
                    "requested_mode": "site_world_candidate",
                    "resolved_mode": "qualification_only",
                },
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "intake_packet.json").write_text(
        json.dumps({"workflowName": "walk", "taskSteps": ["walk"], "zone": "a"}),
        encoding="utf-8",
    )
    (raw_root / "capture_context.json").write_text(
        json.dumps({"captureSource": "glasses", "captureModality": "glasses_video_only"}),
        encoding="utf-8",
    )
    (raw_root / "capture_upload_complete.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (raw_root / "walkthrough.mov").write_bytes(b"not-a-real-video")
    materialize_capture_bundle(bucket=bucket, scene_id=scene_id, capture_id=capture_id, gcs_root=tmp_path)
    if with_privacy_video:
        privacy_root = capture_root / "privacy"
        privacy_root.mkdir(parents=True, exist_ok=True)
        (privacy_root / "final_walkthrough.mov").write_bytes(b"privacy-video")
    return capture_root


def test_retrieval_index_uses_pipeline_geometry_for_non_arkit(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_glasses_capture(tmp_path, with_privacy_video=True)
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")
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
                        "t_capture_sec": 1.0,
                        "hold_duration_sec": 1.0,
                        "completed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_provider(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        frames_dir = geometry_root / "frames" / "images"
        depth_dir = geometry_root / "depth"
        confidence_dir = geometry_root / "confidence"
        frames_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for frame_index in range(3):
            image_path = frames_dir / f"frame_{frame_index:06d}.npy"
            depth_path = depth_dir / f"depth_{frame_index:06d}.npy"
            confidence_path = confidence_dir / f"confidence_{frame_index:06d}.npy"
            checker = (np.indices((16, 24)).sum(axis=0) % 2).astype(np.float32) * 255.0
            image = np.repeat(checker[:, :, None], 3, axis=2)
            np.save(image_path, image)
            np.save(depth_path, np.full((16, 24), 1.0 + frame_index * 0.1, dtype=np.float32))
            np.save(confidence_path, np.full((16, 24), 0.8, dtype=np.float32))
            frames.append(
                {
                    "frame_index": frame_index,
                    "frame_id": str(frame_index).zfill(6),
                    "timestamp_seconds": float(frame_index),
                    "image_path": str(image_path),
                    "is_keyframe": True,
                    "blur_score": 0.0,
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
                    "pose_confidence": 0.9,
                    "depth_path": str(depth_path),
                    "confidence_path": str(confidence_path),
                    "depth_format": "npy",
                    "confidence_format": "npy",
                    "width": 24,
                    "height": 16,
                    "min_depth_m": 1.0,
                    "max_depth_m": 1.2,
                    "confidence_range": [0.0, 1.0],
                }
            )
        return {
            "intrinsics": {
                "camera_model": "pinhole",
                "image_width": 24,
                "image_height": 16,
                "fx": 18.0,
                "fy": 18.0,
                "cx": 12.0,
                "cy": 8.0,
                "distortion": {"model": "none", "coefficients": []},
            },
            "frames": frames,
            "provider_metrics": {},
            "provider_warnings": [],
            "provider_errors": [],
            "loop_closure_detected": False,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 0.92,
            "p95_pose_delta_sec": 0.033,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _fake_provider)
    monkeypatch.setattr(
        "blueprint_pipeline.retrieval_index_stage._generate_embeddings",
        lambda **_kwargs: [np.ones(1024, dtype=np.float32) for _ in _kwargs["image_paths"]],
    )

    build_geometry_stage_contract(capture_root)
    result = run_retrieval_index_stage(capture_root=capture_root, embedding_model=object())

    assert result["status"] == "completed"
    assert result["frames_included_in_index"] >= 1
    assert (capture_root / "world_model_export" / "dense_export_manifest.json").is_file()
    dense_index = (capture_root / "world_model_export" / "dense_index.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in dense_index if line.strip()]
    assert rows
    assert all(row["geometry_source"] == "video_to_world" for row in rows)
    assert all(row["privacy_source"] == "privacy/final_walkthrough.mov" for row in rows)
    assert all(row["depth_uri"] for row in rows)
    assert all("chunk_id" in row for row in rows)
    assert all("geometry_fingerprint" in row for row in rows)
    assert all("staticness_score" in row for row in rows)
    assert any("anchor_entry" in row["anchor_observations"] for row in rows)
    assert all("retrieval_signals" in row for row in rows)
    assert any((row["retrieval_signals"].get("anchor_observation_count") or 0) > 0 for row in rows)
    assert all("capture_confidence" in row["retrieval_signals"] for row in rows)
    assert Path(str(result["site_reference_index"])).is_file()
    site_root = capture_root.parents[3] / "sites" / "site-1" / "reference_memory"
    assert (site_root / "site_overlap_graph.json").is_file()
    assert (site_root / "indices" / "manifest.json").is_file()
    assert (site_root / "retrieval_validation.json").is_file()
    validation = json.loads((site_root / "retrieval_validation.json").read_text(encoding="utf-8"))
    assert validation["record_schema_valid"] is True
    assert validation["manifest_schema_valid"] is True
    assert validation["summary_projection_safe"] is True
    assert validation["privacy_safe_source_available"] is True
    assert validation["rights_lineage_present"] is True
    assert validation["provenance_lineage_present"] is True
    assert validation["retrieval_query_ready"] is True
    assert validation["runtime_adapter_consumption"]["local_contract_ready"] is True
    assert validation["readiness"]["retrieval"]["state"] == "ready"
    assert validation["readiness"]["non_arkit_geometry"]["state"] == "ready"
    assert validation["readiness"]["swm_world_model"]["state"] == "ready"
    assert validation["readiness"]["operational_live_provider_hosted"]["state"] == "blocked"


def test_retrieval_index_rejects_local_sfm_synthetic_diagnostics(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_glasses_capture(tmp_path, with_privacy_video=True)
    monkeypatch.setattr(
        "blueprint_pipeline.retrieval_index_stage._generate_embeddings",
        lambda **_kwargs: [np.ones(1024, dtype=np.float32) for _ in _kwargs["image_paths"]],
    )

    build_geometry_stage_contract(capture_root, provider="local_sfm", model="local-sfm-offline")
    summary = json.loads(
        (capture_root / "pipeline" / "geometry" / "geometry_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["geometry_source"] == "fallback_geometry"
    assert summary["fallback_used"] is True
    assert summary["synthetic_artifacts_are_capture_truth"] is False

    with pytest.raises(Exception, match="geometry_not_live_video_to_world:fallback_geometry"):
        run_retrieval_index_stage(capture_root=capture_root, embedding_model=object())


def test_retrieval_index_rejects_fallback_geometry_even_if_descriptor_is_stale(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_glasses_capture(tmp_path, with_privacy_video=True)
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")

    def _failing_provider(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("video_to_world_down")

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _failing_provider)
    build_geometry_stage_contract(capture_root)

    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["world_model_candidate"] = True
    descriptor["geometry_ready"] = True
    descriptor["quality"] = {
        **dict(descriptor.get("quality") or {}),
        "world_model_candidate": True,
        "geometry_ready": True,
    }
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(Exception, match="geometry_not_live_video_to_world"):
        run_retrieval_index_stage(capture_root=capture_root, embedding_model=object())


def test_retrieval_index_requires_privacy_safe_video_by_default(monkeypatch, tmp_path: Path) -> None:
    capture_root = _build_staged_glasses_capture(tmp_path, with_privacy_video=False)
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "http://video-to-world.local")
    monkeypatch.setenv("VIDEO_TO_WORLD_RUNNER_TOKEN", "test-token")
    monkeypatch.setenv("RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO", "true")

    def _fake_provider(**kwargs):  # type: ignore[no-untyped-def]
        geometry_root = Path(kwargs["geometry_root"])
        frames_dir = geometry_root / "frames" / "images"
        depth_dir = geometry_root / "depth"
        confidence_dir = geometry_root / "confidence"
        frames_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)
        image_path = frames_dir / "frame_000000.npy"
        depth_path = depth_dir / "depth_000000.npy"
        confidence_path = confidence_dir / "confidence_000000.npy"
        np.save(image_path, np.full((16, 24, 3), 100, dtype=np.float32))
        np.save(depth_path, np.full((16, 24), 1.0, dtype=np.float32))
        np.save(confidence_path, np.full((16, 24), 0.8, dtype=np.float32))
        return {
            "intrinsics": {
                "camera_model": "pinhole",
                "image_width": 24,
                "image_height": 16,
                "fx": 18.0,
                "fy": 18.0,
                "cx": 12.0,
                "cy": 8.0,
                "distortion": {"model": "none", "coefficients": []},
            },
            "frames": [
                {
                    "frame_index": 0,
                    "frame_id": "000000",
                    "timestamp_seconds": 0.0,
                    "image_path": str(image_path),
                    "is_keyframe": True,
                    "blur_score": 0.0,
                    "overlap_hint": 0.9,
                    "world_from_camera": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "camera_from_world": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "pose_confidence": 0.9,
                    "depth_path": str(depth_path),
                    "confidence_path": str(confidence_path),
                    "depth_format": "npy",
                    "confidence_format": "npy",
                    "width": 24,
                    "height": 16,
                    "min_depth_m": 1.0,
                    "max_depth_m": 1.0,
                    "confidence_range": [0.0, 1.0],
                }
            ],
            "provider_metrics": {},
            "provider_warnings": [],
            "provider_errors": [],
            "loop_closure_detected": False,
            "site_frame_available": True,
            "scale_resolved": True,
            "pose_match_rate": 0.92,
            "p95_pose_delta_sec": 0.033,
        }

    monkeypatch.setattr("blueprint_pipeline.geometry_stage.run_video_to_world_provider", _fake_provider)
    build_geometry_stage_contract(capture_root)

    with pytest.raises(Exception, match="privacy_safe_video_required"):
        run_retrieval_index_stage(capture_root=capture_root, embedding_model=object())
