from __future__ import annotations

import hashlib
import math

import numpy as np
import trimesh

import blueprint_pipeline.external_scene_robot_placement as placement_module
from blueprint_pipeline.external_scene_robot_placement import (
    _footprint_overlap_counts,
    _triangle_footprint_overlap_count,
    propose_external_scene_robot_placement,
)
from blueprint_pipeline.scene_placement.types import StandPose


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_scene_placement_uses_official_franka_and_abstains_formally(
    tmp_path,
) -> None:
    # GLB source frame is Y-up; this is a flat 6x6 floor at Y=0.
    mesh = trimesh.Trimesh(
        vertices=np.asarray([[-3, 0, -3], [3, 0, -3], [3, 0, 3], [-3, 0, 3]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    glb = tmp_path / "scene.glb"
    glb.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    request = {
        "schema_version": "external_scene_robot_placement_request.v1",
        "robot_id": "franka_panda",
        "source_scene_digest": "sha256:" + "a" * 64,
        "target_analysis_digest": "sha256:" + "b" * 64,
        "target_binding_digest": "sha256:" + "c" * 64,
        "scene_frame_binding_digest": "sha256:" + "d" * 64,
        "collision_candidate_digest": "sha256:" + "e" * 64,
        "collision_source_digest": _digest(glb),
        "target_label": "fixture surface",
        "visual_confidence": 0.9,
        "target_position_collision_stage": [0.0, 0.0, 0.7],
        "target_spatial_uncertainty_stage_units": 0.25,
        "metric_scale_status": "provider_declared_not_independently_validated",
        "collision_status": "candidate_compiled",
        "candidate_may_self_authorize": False,
    }

    packet = propose_external_scene_robot_placement(
        collision_glb_path=glb,
        request=request,
    )

    placement = packet["placement"]
    assert placement["status"] == "runtime_visualization_candidate_only"
    assert placement["official_isaac_asset"].endswith("FrankaPanda/franka.usd")
    assert placement["mesh_vertex_overlap_probe_clear"] is True
    assert placement["mesh_triangle_aabb_overlap_probe_clear"] is True
    assert placement["analytic_reach_candidate"] is True
    assert placement["metric_reach_qualified"] is False
    assert placement["physical_execution_authorized"] is False
    assert "independent_metric_scale_missing" in placement["formal_gaps"]
    options = packet["render_options"]
    assert options["robot_id"] == "franka_panda"
    assert options["articulated_policy_trace_request"]["candidates"][1]["policy_id"] == (
        "franka-inspection-sweep-v1"
    )


def test_external_scene_placement_rescues_collision_clear_reach_candidate(
    tmp_path, monkeypatch
) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [[-3, 0, -3], [3, 0, -3], [3, 0, 3], [-3, 0, 3]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    glb = tmp_path / "scene.glb"
    glb.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    request = {
        "schema_version": "external_scene_robot_placement_request.v1",
        "robot_id": "franka_panda",
        "source_scene_digest": "sha256:" + "a" * 64,
        "target_analysis_digest": "sha256:" + "b" * 64,
        "target_binding_digest": "sha256:" + "c" * 64,
        "scene_frame_binding_digest": "sha256:" + "d" * 64,
        "collision_candidate_digest": "sha256:" + "e" * 64,
        "collision_source_digest": _digest(glb),
        "target_label": "high fixture surface",
        "visual_confidence": 0.9,
        "target_position_collision_stage": [0.0, 0.0, 0.7],
        "target_spatial_uncertainty_stage_units": 0.25,
        "metric_scale_status": "provider_declared_not_independently_validated",
        "collision_status": "candidate_compiled",
        "candidate_may_self_authorize": False,
    }
    candidates = iter(
        [
            StandPose((1.2, 0.0, 0.0), 0.0, "target", True, 0.15, "nominal"),
            StandPose((0.4, 0.0, 0.0), 0.0, "target", True, 0.05, "rescue"),
        ]
    )
    monkeypatch.setattr(
        placement_module,
        "ring_scan_stand_pose",
        lambda *_args, **_kwargs: next(candidates),
    )

    packet = propose_external_scene_robot_placement(
        collision_glb_path=glb,
        request=request,
    )

    placement = packet["placement"]
    assert placement["analytic_reach_candidate"] is True
    assert placement["standoff_stage_units"] == 0.05
    assert placement["placement_selection_strategy"] == (
        "collision_clear_analytic_reach_rescue_candidate"
    )
    assert "placement_below_nominal_standoff_range" in placement["formal_gaps"]


def test_triangle_crossing_footprint_is_not_missed_when_vertices_are_outside() -> None:
    # This elevated triangle crosses the 0.2 x 0.2 footprint, but all three
    # vertices lie outside it. The former vertex-only probe returned clear.
    vertices = np.asarray(
        [
            [-0.4, -0.4, 0.3],
            [0.4, -0.4, 0.3],
            [0.0, 0.4, 0.3],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    vertex_hits, triangle_hits = _footprint_overlap_counts(
        stage_vertices=vertices,
        faces=faces,
        position=(0.0, 0.0, 0.0),
        yaw=0.0,
        floor_z=0.0,
        half_extent_xy=(0.1, 0.1),
        probe_clearance=0.0,
        obstacle_height=0.72,
    )

    assert vertex_hits == 0
    assert triangle_hits == 1


def test_oriented_triangle_probe_is_identical_for_search_and_final_gate() -> None:
    triangles = np.asarray(
        [
            [
                [-0.4, -0.4],
                [0.4, -0.4],
                [0.0, 0.4],
            ]
        ],
        dtype=np.float64,
    )

    assert (
        _triangle_footprint_overlap_count(
            obstacle_triangles_xy=triangles,
            position=(0.0, 0.0, 0.79),
            yaw=math.pi / 3.0,
            half_extent_xy=(0.1, 0.1),
        )
        == 1
    )
