from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.materialization import materialize_capture_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_materialize_capture_bundle_for_metric_iphone(tmp_path: Path) -> None:
    raw_root = tmp_path / "bucket/scenes/scene_a/captures/cap_a/raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene_a",
            "capture_source": "iphone",
            "capture_tier_hint": "tier1_iphone",
            "video_uri": "gs://bucket/scenes/scene_a/captures/cap_a/raw/walkthrough.mov",
            "has_lidar": True,
            "pose_match_rate": 0.96,
            "object_point_cloud_index": "arkit/objects/index.json",
            "intended_space_type": "warehouse",
        },
    )
    _write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": "Tote handoff",
            "taskSteps": ["Inbound", "Staging", "Outbound"],
            "targetKPI": "cycle time",
            "zone": "dock_lane_a",
            "owner": "ops_manager",
        },
    )
    _write_json(
        raw_root / "capture_context.json",
        {
            "captureModality": "iphone_arkit_lidar",
            "scaffoldingUsed": ["arkit_depth"],
            "coveragePlan": ["entry", "task zone"],
            "calibrationAssets": ["arkit/intrinsics.json"],
            "uncertaintyPriors": {"occlusion_risk": 0.2},
        },
    )
    (raw_root / "walkthrough.mov").write_bytes(b"mov")
    (raw_root / "arkit").mkdir(parents=True, exist_ok=True)
    (raw_root / "arkit/poses.jsonl").write_text("{}", encoding="utf-8")
    (raw_root / "arkit/intrinsics.json").write_text("{}", encoding="utf-8")

    result = materialize_capture_bundle(
        bucket="bucket",
        scene_id="scene_a",
        capture_id="cap_a",
        gcs_root=tmp_path,
    )

    assert result["descriptor"]["capture_modality"] == "iphone_arkit_lidar"
    assert result["descriptor"]["requested_lanes"] == ["qualification", "advanced_geometry"]
    assert result["qa_report"]["status"] == "passed"


def test_materialize_capture_bundle_for_video_only_glasses_stays_degraded(tmp_path: Path) -> None:
    raw_root = tmp_path / "bucket/scenes/scene_b/captures/cap_b/raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene_b",
            "capture_source": "glasses",
            "capture_tier_hint": "tier2_glasses",
            "video_uri": "gs://bucket/scenes/scene_b/captures/cap_b/raw/walkthrough.mov",
            "has_lidar": False,
            "intended_space_type": "warehouse",
        },
    )
    _write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": "Aisle walkthrough",
            "taskSteps": ["Aisle entry", "Shelf face"],
            "zone": "aisle_7",
        },
    )
    (raw_root / "walkthrough.mov").write_bytes(b"mov")

    result = materialize_capture_bundle(
        bucket="bucket",
        scene_id="scene_b",
        capture_id="cap_b",
        gcs_root=tmp_path,
    )

    assert result["descriptor"]["capture_modality"] == "glasses_video_only"
    assert result["descriptor"]["requested_lanes"] == ["qualification"]
    assert result["qa_report"]["status"] == "degraded"
    assert result["qa_report"]["escalation_recommendation"]["human_review_required"] is True


def test_materialize_capture_bundle_for_validated_glasses_scaffolding(tmp_path: Path) -> None:
    raw_root = tmp_path / "bucket/scenes/scene_c/captures/cap_c/raw"
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene_c",
            "capture_source": "glasses",
            "capture_tier_hint": "tier2_glasses",
            "video_uri": "gs://bucket/scenes/scene_c/captures/cap_c/raw/walkthrough.mov",
            "intended_space_type": "industrial_unknown",
        },
    )
    _write_json(
        raw_root / "intake_packet.json",
        {
            "workflowName": "Material handoff",
            "taskSteps": ["entry", "workcell", "handoff"],
            "zone": "cell_2",
            "owner": "ops_manager",
        },
    )
    _write_json(
        raw_root / "capture_context.json",
        {
            "captureModality": "glasses_plus_scaffolding",
            "scaffoldingUsed": ["scale_anchor", "checkpoint_stills", "companion_phone_pass"],
            "coveragePlan": ["entry", "workcell", "handoff"],
            "calibrationAssets": ["scaffolding/calibration_manifest.json"],
            "scaleAnchorAssets": ["scaffolding/scale_anchor_01.jpg"],
            "checkpointAssets": ["scaffolding/checkpoint_01.jpg"],
            "validatedScaleMeters": 1.0,
            "validatedPoseCoverage": 0.82,
            "hiddenZoneBound": 0.2,
        },
    )
    (raw_root / "walkthrough.mov").write_bytes(b"mov")

    result = materialize_capture_bundle(
        bucket="bucket",
        scene_id="scene_c",
        capture_id="cap_c",
        gcs_root=tmp_path,
    )

    assert result["descriptor"]["evidence_tier"] == "glasses_with_validated_scaffolding"
    assert result["descriptor"]["requested_lanes"] == ["qualification", "advanced_geometry"]
    assert result["qa_report"]["status"] == "passed"
