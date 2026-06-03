from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.simready_assets import build_simready_assets


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_capture_root(tmp_path: Path, *, site_id: str = "site-1") -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    _write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_identity": {"site_id": site_id, "site_id_source": "fixture"},
        },
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": site_id, "site_id_source": "fixture"}},
        },
    )
    site_root = tmp_path / "local-blueprint" / "sites" / site_id / "reference_memory"
    _write_json(
        site_root / "site_reference_manifest.json",
        {
            "schema_version": "site_reference_database.v1",
            "site_id": site_id,
            "total_reference_frames": 1,
            "capture_count": 1,
            "chunk_count": 1,
            "readiness": {"state": "ready"},
        },
    )
    _write_json(site_root / "retrieval_validation.json", {"status": "ready"})
    (site_root / "site_reference_index.jsonl").write_text(
        json.dumps(
            {
                "reference_id": "ref-1",
                "site_id": site_id,
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "frame_id": "frame-1",
                "zone_id": "zone-a",
                "chunk_id": "chunk-a",
                "geometry_source": "video_to_world",
                "privacy_source": "privacy/final_walkthrough.mov",
                "anchor_observations": [{"anchor_id": "cabinet-anchor"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return capture_root


def _object_geometry_manifest() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "objects": [
            {
                "object_id": "cabinet_0001",
                "label": "cabinet",
                "placement_bbox": {"center": [1.0, 0.5, 0.75], "extents": [0.8, 0.4, 1.2]},
                "collision_hulls": [{"kind": "box"}],
                "support_surfaces": [{"kind": "shelf"}],
                "provenance": {"grounding_level": "observed"},
            }
        ],
    }


def _task_anchor_manifest() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "tasks": [
            {
                "task_id": "open_cabinet",
                "task_text": "Open the cabinet",
                "task_category": "open_close",
                "target_object_ids": ["cabinet_0001"],
                "articulation_required_ids": ["cabinet_0001"],
                "start_zone": [0.0, 0.5, 0.0],
                "goal_zone": [1.0, 0.5, 0.75],
                "task_critical": True,
            }
        ],
    }


def _site_world_spec() -> dict[str, object]:
    return {
        "schema_version": "v1",
        "robot_profiles": [
            {
                "id": "mobile_manipulator_rgb_v1",
                "display_name": "Mobile manipulator",
                "embodiment_type": "mobile_manipulator",
                "action_space": {"name": "ee_delta_pose_gripper", "dim": 7},
            }
        ],
    }


def test_simready_assets_emit_framework_review_packet(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(
        capture_root / "pipeline" / "geometry" / "geometry_summary.json",
        {
            "geometry_source": "video_to_world",
            "fallback_used": False,
            "provider_native_result": True,
            "geometry_live_ready": True,
            "ready_for_world_model": True,
            "site_frame_available": True,
            "scale_resolved": True,
        },
    )

    result = build_simready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        site_world_spec=_site_world_spec(),
    )

    sim_root = capture_root / "pipeline" / "simready"
    manifest = json.loads((sim_root / "simready_scene_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((sim_root / "simready_validation.json").read_text(encoding="utf-8"))
    site_reference = json.loads((sim_root / "site_reference_summary.json").read_text(encoding="utf-8"))

    assert result["status"] == "prepared_for_review"
    assert validation["overall_status"] == "prepared_for_review"
    assert manifest["object_count"] == 1
    assert manifest["task_count"] == 1
    assert manifest["claim_boundary"]["simulator_execution_proven"] is False
    assert manifest["claim_boundary"]["robot_readiness_proven"] is False
    assert site_reference["status"] == "available"
    assert (sim_root / "isaac_sim" / "site_scene.usda").read_text(encoding="utf-8").startswith("#usda 1.0")
    assert "<mujoco" in (sim_root / "mujoco" / "site_scene.xml").read_text(encoding="utf-8")
    assert "<robot" in (sim_root / "pybullet" / "site_scene.urdf").read_text(encoding="utf-8")


def test_simready_assets_keep_fallback_geometry_review_only(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_json(
        capture_root / "pipeline" / "geometry" / "geometry_summary.json",
        {
            "geometry_source": "fallback_geometry",
            "fallback_used": True,
            "provider_native_result": False,
            "geometry_live_ready": False,
            "ready_for_world_model": False,
            "site_frame_available": False,
            "scale_resolved": False,
            "launch_blockers": ["fallback_geometry_not_live_video_to_world"],
        },
    )

    result = build_simready_assets(
        capture_root=capture_root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        site_world_spec=_site_world_spec(),
    )

    validation = json.loads(
        (capture_root / "pipeline" / "simready" / "simready_validation.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "degraded"
    assert validation["overall_status"] == "degraded"
    assert "fallback_geometry_review_only" in validation["warnings"]
    assert validation["claim_boundary"]["simulator_execution_proven"] is False
    assert validation["claim_boundary"]["robot_readiness_proven"] is False
