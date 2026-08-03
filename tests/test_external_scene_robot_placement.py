from __future__ import annotations

import hashlib
import math

import numpy as np
import trimesh

import blueprint_pipeline.external_scene_robot_placement as placement_module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_scene_robot_placement import (
    _footprint_overlap_counts,
    _infer_horizontal_support_surface,
    _select_supported_physics_probe,
    _triangle_footprint_overlap_count,
    propose_external_scene_robot_placement,
)
from blueprint_pipeline.scene_placement.types import StandPose


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _target_analysis(*, position: list[float]) -> dict:
    value = {
        "schema_version": "scene_task_target_analysis_result.v1",
        "status": "target_ready_for_bounded_sim",
        "scene_id": "test-scene",
        "source_scene_digest": "sha256:" + "a" * 64,
        "metric_scale_status": "provider_declared_not_independently_validated",
        "selected_target": {
            "proposal_id": "fixture-inspection-target",
            "object_label": "fixture surface",
            "task_family": "franka_fixture_inspection",
            "target_position_scene": position,
            "spatial_uncertainty_scene_units": 0.25,
        },
    }
    value["target_analysis_digest"] = canonical_digest(value, digest_field="target_analysis_digest")
    return value


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
        "target_analysis_digest": "",
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
    target_analysis = _target_analysis(position=[0.0, 0.0, 0.7])
    request["target_analysis_digest"] = target_analysis["target_analysis_digest"]

    packet = propose_external_scene_robot_placement(
        collision_glb_path=glb,
        request=request,
        target_analysis=target_analysis,
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
    assert "fixed_base_support_mount_not_physically_qualified" in placement["formal_gaps"]
    support = placement["fixed_base_support_mount"]
    assert support["schema_version"] == "fixed_base_support_mount_candidate.v1"
    assert support["status"] == "simulator_support_candidate_only"
    assert support["static_collision_required"] is True
    assert support["physical_load_capacity_qualified"] is False
    assert support["top_z_collision_stage"] == placement["robot_pose_xyzyaw_collision_stage"][2]
    assert (
        support["center_xyz_collision_stage"][:2]
        == placement["robot_pose_xyzyaw_collision_stage"][:2]
    )
    assert (
        support["center_xyz_collision_stage"][2] + 0.5 * support["height_stage_units"]
        == support["top_z_collision_stage"]
    )
    options = packet["render_options"]
    assert options["robot_id"] == "franka_panda"
    assert options["fixed_base_support_mount"] == support
    assert options["robot_ground_z"] == support["top_z_collision_stage"]
    cohort = options["articulated_policy_trace_request"]
    assert cohort["controller_id"] == "deterministic_franka_inspection_cohort.v1"
    assert [row["policy_id"] for row in cohort["candidates"]] == [
        "franka-inspection-center-hold-v1",
        "franka-inspection-left-narrow-v1",
        "franka-inspection-right-narrow-v1",
        "franka-inspection-left-wide-v1",
        "franka-inspection-right-wide-v1",
    ]
    assert (
        cohort["inspection_target_position_stage"] == (placement["target_position_collision_stage"])
    )
    outcome_contract = options["inspection_outcome_contract"]
    assert outcome_contract["placement_proposal_digest"] == placement["placement_proposal_digest"]
    assert outcome_contract["thresholds_frozen_before_candidate_execution"] is True


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
        "target_analysis_digest": "",
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
    target_analysis = _target_analysis(position=[0.0, 0.0, 0.7])
    request["target_analysis_digest"] = target_analysis["target_analysis_digest"]
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
        target_analysis=target_analysis,
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


def test_dominant_horizontal_floor_rejects_low_geometry_outlier() -> None:
    vertices = np.asarray(
        [
            [-3.0, -3.0, 0.0],
            [3.0, -3.0, 0.0],
            [3.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0],
            [0.0, 0.0, -2.0],
            [0.1, 0.0, -2.0],
            [0.0, 0.1, -2.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)

    floor_z, support_triangles, evidence = _infer_horizontal_support_surface(
        stage_vertices=vertices,
        faces=faces,
    )

    assert floor_z == 0.0
    assert support_triangles.shape[0] == 2
    assert evidence["global_minimum_vertex_z_rejected_as_floor"] == -2.0
    assert evidence["selection_method"] == "dominant_lower_horizontal_triangle_area_band"


def test_external_scene_placement_abstains_when_clear_space_has_no_floor_support(
    tmp_path,
) -> None:
    # A target well beyond this finite 2x2 floor used to appear collision-clear
    # because the probe only counted obstacles. It must now fail closed.
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    glb = tmp_path / "finite-floor.glb"
    glb.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    target_analysis = _target_analysis(position=[3.0, 0.0, 0.7])
    request = {
        "schema_version": "external_scene_robot_placement_request.v1",
        "robot_id": "franka_panda",
        "source_scene_digest": "sha256:" + "a" * 64,
        "target_analysis_digest": target_analysis["target_analysis_digest"],
        "target_binding_digest": "sha256:" + "c" * 64,
        "scene_frame_binding_digest": "sha256:" + "d" * 64,
        "collision_candidate_digest": "sha256:" + "e" * 64,
        "collision_source_digest": _digest(glb),
        "target_label": "unsupported fixture surface",
        "visual_confidence": 0.9,
        "target_position_collision_stage": [3.0, 0.0, 0.7],
        "target_spatial_uncertainty_stage_units": 0.25,
        "metric_scale_status": "unverified",
        "collision_status": "candidate_compiled",
        "candidate_may_self_authorize": False,
    }

    packet = propose_external_scene_robot_placement(
        collision_glb_path=glb,
        request=request,
        target_analysis=target_analysis,
    )

    placement = packet["placement"]
    assert placement["status"] == "abstained"
    assert placement["base_support_coverage"]["full_sample_support_candidate"] is False
    assert "robot_base_support_surface_missing" in placement["formal_gaps"]


def test_supported_stance_with_source_collision_conflict_requires_proxy_composition(
    tmp_path, monkeypatch
) -> None:
    # GLB is Y-up. The first four vertices are a floor; the last four map to a
    # vertical stage-X wall crossing the selected robot footprint.
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [-3, 0, -3],
                [3, 0, -3],
                [3, 0, 3],
                [-3, 0, 3],
                [0, 0, 0.5],
                [0, 1, 0.5],
                [0, 1, -0.5],
                [0, 0, -0.5],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64),
        process=False,
    )
    glb = tmp_path / "floor-with-wall.glb"
    glb.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    target_analysis = _target_analysis(position=[0.0, 0.0, 0.7])
    request = {
        "schema_version": "external_scene_robot_placement_request.v1",
        "robot_id": "franka_panda",
        "source_scene_digest": "sha256:" + "a" * 64,
        "target_analysis_digest": target_analysis["target_analysis_digest"],
        "target_binding_digest": "sha256:" + "c" * 64,
        "scene_frame_binding_digest": "sha256:" + "d" * 64,
        "collision_candidate_digest": "sha256:" + "e" * 64,
        "collision_source_digest": _digest(glb),
        "target_label": "fixture surface",
        "visual_confidence": 0.9,
        "target_position_collision_stage": [0.0, 0.0, 0.7],
        "target_spatial_uncertainty_stage_units": 0.25,
        "metric_scale_status": "unverified",
        "collision_status": "candidate_compiled",
        "candidate_may_self_authorize": False,
    }
    candidates = iter(
        [
            StandPose((0.0, 0.0, 0.0), 0.0, "target", False, 0.2, "source blocked"),
            StandPose((0.0, 0.0, 0.0), 0.0, "target", False, 0.2, "proxy blocked"),
            StandPose((0.0, 0.0, 0.0), 0.0, "target", True, 0.2, "support only"),
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
        target_analysis=target_analysis,
    )

    placement = packet["placement"]
    assert placement["status"] == "abstained"
    assert placement["base_support_coverage"]["full_sample_support_candidate"] is True
    assert placement["mesh_triangle_aabb_overlap_probe_hits"] > 0
    assert placement["bounded_floor_proxy"] is not None
    assert placement["placement_selection_strategy"] == (
        "proxy_composed_task_zone_candidate_required"
    )
    plan = placement["proxy_composed_evaluation_plan"]
    assert plan["status"] == "required_before_policy_evaluation"
    assert plan["source_collision_enabled_in_policy_lane"] is False
    assert plan["task_zone_simready_asset_required"] is False


def test_physics_probe_keeps_supported_minimum_conflict_candidate() -> None:
    support = np.asarray(
        [
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0]],
            [[-2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]],
        ],
        dtype=np.float64,
    )
    obstacle = np.asarray(
        [[[-2.0, -2.0], [2.0, -2.0], [0.0, 2.0]]],
        dtype=np.float64,
    )

    candidate = _select_supported_physics_probe(
        support_triangles=support,
        obstacle_triangles_xy=obstacle,
        position=(0.0, 0.0, 0.0),
        half_extent_xy=(0.18, 0.18),
        floor_z=0.0,
        height_tolerance=0.05,
    )

    assert candidate is not None
    assert candidate["source_surface_support_observed"] is True
    assert candidate["obstacle_overlap_probe_hits"] == 1
    assert candidate["probe_may_intersect_non_floor_source_geometry"] is True
