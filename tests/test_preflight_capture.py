from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.preflight_capture import build_capture_preflight_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_capture_root(tmp_path: Path, *, scene_id: str = "scene_preflight", capture_id: str = "cap_preflight") -> Path:
    return tmp_path / "bucket/scenes" / scene_id / "captures" / capture_id


def test_preflight_reports_metric_capture_when_raw_contract_is_present(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    raw_root = capture_root / "raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene_preflight",
            "capture_source": "iphone",
            "capture_tier_hint": "tier1_iphone",
            "has_lidar": True,
            "pose_match_rate": 0.96,
            "video_uri": "gs://bucket/scenes/scene_preflight/captures/cap_preflight/raw/walkthrough.mov",
        },
    )
    _write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": "Tote handoff",
            "taskSteps": ["Inbound", "Outbound"],
            "targetKPI": "cycle time",
            "zone": "dock_lane_a",
            "owner": "ops_manager",
        },
    )
    _write_json(raw_root / "capture_context.json", {"captureModality": "iphone_arkit_lidar"})
    _write_json(raw_root / "capture_upload_complete.json", {"scene_id": "scene_preflight", "capture_id": "cap_preflight"})
    (raw_root / "walkthrough.mov").write_bytes(b"mov")

    report = build_capture_preflight_report(capture_root)

    assert report["status"] == "ready_for_materialization"
    assert report["mode_decision"] == "qualified_metric_capture"
    assert report["missing_required_inputs"] == []


def test_preflight_marks_splat_as_supplemental_and_fails_closed_without_intake(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path, scene_id="scene_splat", capture_id="cap_splat")
    raw_root = capture_root / "raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene_splat",
            "capture_source": "iphone",
            "capture_tier_hint": "tier1_iphone",
            "has_lidar": True,
            "video_uri": "gs://bucket/scenes/scene_splat/captures/cap_splat/raw/walkthrough.mov",
        },
    )
    _write_json(raw_root / "capture_context.json", {"captureModality": "iphone_arkit_lidar"})
    _write_json(raw_root / "capture_upload_complete.json", {"scene_id": "scene_splat", "capture_id": "cap_splat"})
    (raw_root / "walkthrough.mov").write_bytes(b"mov")
    (raw_root / "splat.ply").write_bytes(b"ply")

    report = build_capture_preflight_report(capture_root)

    assert report["can_materialize"] is False
    assert "intake_packet" in report["missing_required_inputs"]
    assert report["splat_candidates"]
