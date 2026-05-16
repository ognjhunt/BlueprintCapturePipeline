from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.materialization import materialize_capture_bundle


def _write_capture(
    root: Path,
    *,
    frame_rows: list[dict[str, object]],
    pose_rows: list[dict[str, object]],
    include_site_identity: bool = True,
) -> dict[str, object]:
    bucket = "local-blueprint"
    scene_id = "scene-parity"
    capture_id = "capture-parity"
    raw_root = root / bucket / "scenes" / scene_id / "captures" / capture_id / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "arkit" / "depth").mkdir(parents=True)

    manifest = {
        "scene_id": scene_id,
        "capture_id": capture_id,
        "video_uri": "walkthrough.mov",
        "width": 1920,
        "height": 1080,
        "device_model": "iPhone 15 Pro",
        "os_version": "18.0",
        "fps_source": 30.0,
        "capture_start_epoch_ms": 1_700_000_000_000,
        "has_lidar": True,
        "capture_schema_version": "3.0.0",
        "capture_source": "iphone",
        "capture_tier_hint": "tier1_iphone",
        "requested_outputs": ["qualification", "preview_simulation", "deeper_evaluation"],
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "data_licensing_allowed": False,
            "capture_contributor_payout_eligible": True,
            "consent_status": "documented",
            "consent_scope": ["alpha-site"],
            "permission_document_uri": "gs://bucket/rights/doc.pdf",
            "consent_notes": [],
        },
        "capture_mode": {
            "requested_mode": "site_world_candidate",
            "resolved_mode": "site_world_candidate",
        },
    }
    if include_site_identity:
        manifest["site_identity"] = {
            "site_id": "site-parity",
            "site_id_source": "test_fixture",
        }
    (raw_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (raw_root / "intake_packet.json").write_text(
        json.dumps(
            {
                "workflowName": "Open capture",
                "taskSteps": ["Walk the site"],
                "zone": "main floor",
                "owner": "ops",
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
    (raw_root / "walkthrough.mov").write_bytes(b"video")
    (raw_root / "capture_upload_complete.json").write_text(
        json.dumps({"sceneId": scene_id, "captureId": capture_id}),
        encoding="utf-8",
    )
    (raw_root / "task_hypothesis.json").write_text(
        json.dumps({"workflow_name": "Open capture", "task_steps": ["Walk the site"], "status": "accepted"}),
        encoding="utf-8",
    )
    (raw_root / "arkit" / "frames.jsonl").write_text(
        "\n".join(json.dumps(row) for row in frame_rows) + "\n",
        encoding="utf-8",
    )
    (raw_root / "arkit" / "poses.jsonl").write_text(
        "\n".join(json.dumps(row) for row in pose_rows) + "\n",
        encoding="utf-8",
    )
    (raw_root / "arkit" / "intrinsics.json").write_text(
        json.dumps({"width": 1920, "height": 1080, "fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0}),
        encoding="utf-8",
    )
    (raw_root / "arkit" / "depth" / "000001.png").write_bytes(b"depth")
    return {"bucket": bucket, "scene_id": scene_id, "capture_id": capture_id}


def test_materialization_promotes_aligned_iphone_capture_to_site_world_candidate(tmp_path: Path) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0, "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            {"frame_id": "000002", "t_device_sec": 0.1, "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])

    assert payload["quality"]["pose_alignment_ok"] is True
    assert payload["quality"]["world_model_candidate"] is True
    assert payload["quality"]["pose_match_rate"] == 1.0
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "site_world_candidate"
    assert "site_id_present:True" in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]
    assert "pose_alignment_ok:True" in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]


def test_materialization_downgrades_site_world_candidate_without_stable_site_id(tmp_path: Path) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0, "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            {"frame_id": "000002", "t_device_sec": 0.1, "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
        include_site_identity=False,
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])

    assert payload["quality"]["pose_alignment_ok"] is True
    assert payload["quality"]["world_model_candidate"] is False
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "qualification_only"
    assert payload["metadata"]["capture_mode"]["downgrade_reason"] == "missing_site_id"
    assert "site_id_present:False" in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]


def test_materialization_preserves_meta_raw_video_descriptor_lineage(tmp_path: Path) -> None:
    bucket = "local-blueprint"
    scene_id = "scene-meta"
    capture_id = "capture-meta"
    raw_root = tmp_path / bucket / "scenes" / scene_id / "captures" / capture_id / "raw"
    glasses_root = raw_root / "glasses"
    glasses_root.mkdir(parents=True)
    (raw_root / "walkthrough.mov").write_bytes(b"video")
    (glasses_root / "frame_timestamps.jsonl").write_text(
        json.dumps({"frame_index": 0, "timestamp_seconds": 0.0}) + "\n",
        encoding="utf-8",
    )
    (glasses_root / "stream_metadata.json").write_text(
        json.dumps({"device_model": "Ray-Ban Meta", "fps": 30.0, "width": 1280, "height": 720}),
        encoding="utf-8",
    )
    (raw_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "video_uri": "walkthrough.mov",
                "capture_source": "meta_glasses",
                "source_device": "meta_glasses",
                "capture_profile_id": "glasses_pov",
                "width": 1280,
                "height": 720,
                "fps_source": 30.0,
                "capture_start_epoch_ms": 1_700_000_000_000,
                "site_identity": {"site_id": "site-meta", "site_id_source": "fixture"},
                "capture_topology": {
                    "capture_session_id": "session-meta-1",
                    "route_id": "route-main",
                    "pass_id": "pass-1",
                    "pass_index": 1,
                },
                "capture_rights": {
                    "derived_scene_generation_allowed": True,
                    "consent_status": "documented",
                    "permission_document_uri": "gs://bucket/rights/meta.pdf",
                },
                "privacy_lineage": {"status": "raw_unprocessed", "source": "field_capture"},
                "provenance_lineage": {"capture_app": "BlueprintCapture", "original_media_sha256": "abc"},
                "capture_mode": {
                    "requested_mode": "site_world_candidate",
                    "resolved_mode": "site_world_candidate",
                },
                "requested_outputs": ["qualification", "retrieval_index"],
                "disable_default_preview": True,
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "intake_packet.json").write_text(
        json.dumps({"workflowName": "Walk", "taskSteps": ["Walk route"], "zone": "main"}),
        encoding="utf-8",
    )
    (raw_root / "capture_context.json").write_text(
        json.dumps({"captureSource": "meta_glasses", "captureModality": "glasses_video_only"}),
        encoding="utf-8",
    )
    (raw_root / "capture_upload_complete.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=tmp_path,
    )
    payload = dict(materialized["descriptor"])

    assert payload["capture_source"] == "glasses"
    assert payload["source_device"] == "meta_glasses"
    assert payload["capture_modality"] == "glasses_pov"
    assert payload["raw_video_uri"].endswith("/raw/walkthrough.mov")
    assert payload["media_metadata"]["original_video_path"].endswith("raw/walkthrough.mov")
    assert payload["media_metadata"]["frame_timestamps_uri"].endswith("raw/glasses/frame_timestamps.jsonl")
    assert payload["media_metadata"]["stream_metadata_uri"].endswith("raw/glasses/stream_metadata.json")
    assert payload["metadata"]["capture_topology"]["capture_session_id"] == "session-meta-1"
    assert payload["metadata"]["capture_topology"]["route_id"] == "route-main"
    assert payload["metadata"]["capture_topology"]["pass_id"] == "pass-1"
    assert payload["metadata"]["privacy_lineage"]["status"] == "raw_unprocessed"
    assert payload["metadata"]["provenance_lineage"]["original_media_sha256"] == "abc"
    assert payload["quality"]["world_model_candidate"] is False
    assert payload["metadata"]["capture_mode"]["downgrade_reason"] == "awaiting_geometry_stage"


def test_materialization_downgrades_misaligned_iphone_capture_locally(tmp_path: Path) -> None:
    ids = _write_capture(
        tmp_path,
        frame_rows=[
            {"frame_id": "000001", "t_device_sec": 0.0},
            {"frame_id": "000002", "t_device_sec": 0.1},
        ],
        pose_rows=[
            {"frame_id": "001001", "t_device_sec": 2.0, "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            {"frame_id": "001002", "t_device_sec": 2.1, "T_world_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
    )

    materialized = materialize_capture_bundle(gcs_root=tmp_path, **ids)
    payload = dict(materialized["descriptor"])

    assert payload["quality"]["pose_alignment_ok"] is False
    assert payload["quality"]["world_model_candidate"] is False
    assert payload["quality"]["pose_match_rate"] == 1.0
    assert payload["quality"]["p95_pose_delta_sec"] > 0.2
    assert payload["metadata"]["capture_mode"]["resolved_mode"] == "qualification_only"
    assert payload["metadata"]["capture_mode"]["downgrade_reason"] == "insufficient_spatial_evidence"
    assert "pose_alignment_ok:False" in payload["metadata"]["scene_memory_capture"]["world_model_candidate_reasoning"]
