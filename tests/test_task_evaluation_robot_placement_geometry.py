from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.task_evaluation_robot_placement_geometry import (
    _parallel_geometry_gates,
    _parallel_trajectory_gates,
    _supported_sample_count,
    build_robot_placement_geometry_index,
    enumerate_robot_placement_geometry_candidates,
    render_robot_placement_geometry_previews,
    summarize_robot_placement_geometry,
    validate_robot_placement_geometry_candidate,
    validate_robot_placement_trajectory_position_ik,
)


def test_support_coverage_batches_samples_against_all_triangles() -> None:
    triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        ],
        dtype=np.float64,
    )
    samples = np.asarray(
        [[0.1, 0.1], [0.9, 0.1], [0.5, 0.5], [1.1, 0.5]],
        dtype=np.float64,
    )

    assert _supported_sample_count(samples, triangles) == 3


def _mesh(stage, path, points, faces):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    return mesh


def _box(stage, path, minimum, maximum):
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    points = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return _mesh(stage, path, points, faces)


def _assets(tmp_path):
    scene_path = tmp_path / "scene.usda"
    scene = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(scene, "/Scene")
    scene.SetDefaultPrim(root.GetPrim())
    _mesh(
        scene,
        "/Scene/Floor",
        [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)],
        [(0, 1, 2, 3)],
    )
    _box(scene, "/Scene/Obstacle", (1.2, -0.3, 0), (1.8, 0.3, 1.0))
    scene.GetRootLayer().Save()

    robot_path = tmp_path / "robot.usda"
    robot = Usd.Stage.CreateNew(str(robot_path))
    robot_root = UsdGeom.Xform.Define(robot, "/Robot")
    robot.SetDefaultPrim(robot_root.GetPrim())
    _box(robot, "/Robot/Body", (-0.15, -0.15, 0), (0.15, 0.15, 0.75))
    # Flattened simulator assets may retain environment geometry next to the
    # default robot prim. It must not become part of the placement preview.
    _box(robot, "/GroundPlane", (-25, -25, -0.01), (25, 25, 0,))
    robot.GetRootLayer().Save()
    return scene_path, robot_path


def _proposal(surface_id, position=(0.0, 0.0, 0.0), yaw=0.0):
    return {
        "candidate_id": "candidate",
        "support_surface_id": surface_id,
        "pose": {
            "position_world_m": list(position),
            "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)],
        },
    }


def test_geometry_gate_uses_batched_support_coverage(tmp_path, monkeypatch) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_geometry as module

    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(
        surface
        for surface in index.support_surfaces
        if surface.prim_path == "/Scene/Floor"
    )
    original = module._supported_sample_count
    calls = []

    def measured(samples, triangles):
        calls.append((samples.shape, triangles.shape))
        return original(samples, triangles)

    monkeypatch.setattr(module, "_supported_sample_count", measured)
    validate_robot_placement_geometry_candidate(
        index=index,
        proposal=_proposal(floor.surface_id),
        target_position_world_m=[0.8, 0.0, 0.5],
    )

    assert calls
    assert calls[0][0] == (5, 2)


def test_exact_geometry_gate_accepts_supported_clear_facing_pose(tmp_path) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    assert index.robot_triangles.shape == (12, 3, 3)
    assert index.robot_triangles[:, :, 0].min() >= -0.1501
    assert index.robot_triangles[:, :, 0].max() <= 0.1501
    assert float(abs(index.robot_triangles).max()) < 1.0
    floor = next(surface for surface in index.support_surfaces if surface.prim_path == "/Scene/Floor")
    target = [0.8, 0.0, 0.5]

    gate = validate_robot_placement_geometry_candidate(
        index=index,
        proposal=_proposal(floor.surface_id),
        target_position_world_m=target,
    )
    summary = summarize_robot_placement_geometry(index, target_position_world_m=target)

    assert gate["status"] == "passed"
    assert gate["supported_sample_count"] == 5
    assert gate["scene_overlap_triangle_count"] == 0
    assert gate["geometry_gate_digest"].startswith("sha256:")
    assert summary["geometry_summary_digest"].startswith("sha256:")
    candidates = enumerate_robot_placement_geometry_candidates(
        index=index,
        target_position_world_m=target,
        maximum_candidates=5,
    )
    assert len(candidates) == 5
    assert all(candidate["geometry_gate_digest"].startswith("sha256:") for candidate in candidates)


def test_geometry_gate_rejects_embedded_and_obstacle_overlapping_pose(tmp_path) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(surface for surface in index.support_surfaces if surface.prim_path == "/Scene/Floor")

    embedded = validate_robot_placement_geometry_candidate(
        index=index,
        proposal=_proposal(floor.surface_id, position=(0.0, 0.0, -0.4)),
        target_position_world_m=[0.8, 0.0, 0.5],
    )
    overlapping = validate_robot_placement_geometry_candidate(
        index=index,
        proposal=_proposal(floor.surface_id, position=(1.4, 0.0, 0.0), yaw=math.pi),
        target_position_world_m=[0.8, 0.0, 0.5],
    )

    assert "robot_root_not_on_declared_support_height" in embedded["blockers"]
    assert "robot_reset_bounds_overlap_scene_geometry" in overlapping["blockers"]


def test_geometry_previews_are_digest_bound_multimodal_inputs(tmp_path) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(surface for surface in index.support_surfaces if surface.prim_path == "/Scene/Floor")

    images = render_robot_placement_geometry_previews(
        index=index,
        proposal=_proposal(floor.surface_id),
        target_position_world_m=[0.8, 0.0, 0.5],
        image_size=(320, 240),
    )

    assert [image["label"] for image in images] == ["top_down_xy", "side_xz"]
    assert all(image["digest"].startswith("sha256:") for image in images)
    assert all(image["image_url"].startswith("data:image/png;base64,") for image in images)


def test_geometry_previews_show_full_tool_trajectory(tmp_path) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(
        surface for surface in index.support_surfaces if surface.prim_path == "/Scene/Floor"
    )
    proposal = _proposal(floor.surface_id)
    target = [0.8, 0.0, 0.5]
    point_only = render_robot_placement_geometry_previews(
        index=index,
        proposal=proposal,
        target_position_world_m=target,
        image_size=(320, 240),
    )
    trajectory = render_robot_placement_geometry_previews(
        index=index,
        proposal=proposal,
        target_position_world_m=target,
        trajectory_waypoints_world_m=[[0.45, 0.0, 0.5], target],
        image_size=(320, 240),
    )

    assert [row["digest"] for row in trajectory] != [
        row["digest"] for row in point_only
    ]


def test_empty_trajectory_preserves_target_only_geometry_compatibility(tmp_path) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(surface for surface in index.support_surfaces if surface.prim_path == "/Scene/Floor")
    gate = validate_robot_placement_geometry_candidate(
        index=index,
        proposal=_proposal(floor.surface_id),
        target_position_world_m=[0.8, 0.0, 0.5],
    )
    assert gate["status"] == "passed"
    assert gate["trajectory_position_ik_gate"]["status"] == "not_requested"


def test_enumeration_ranks_the_full_grid_before_applying_inventory_cap(
    tmp_path, monkeypatch
) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )

    def gate(*, proposal, **_kwargs):
        angle_index = int(str(proposal["candidate_id"]).rsplit("_", 1)[-1])
        trajectory_gate = {
            "minimum_manipulability": float(angle_index),
            "trajectory_position_ik_gate_digest": f"sha256:{angle_index:064x}",
        }
        return {
            "status": "passed",
            "geometry_gate_digest": f"sha256:{(angle_index + 100):064x}",
            "shoulder_to_target_distance_m": 0.5,
            "trajectory_position_ik_gate": trajectory_gate,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_robot_placement_geometry.validate_robot_placement_geometry_candidate",
        gate,
    )
    candidates = enumerate_robot_placement_geometry_candidates(
        index=index,
        target_position_world_m=[0.8, 0.0, 0.5],
        maximum_candidates=1,
    )
    assert candidates[0]["candidate_id"].endswith("_71")


def test_full_trajectory_failure_rejects_target_reachable_candidate(
    tmp_path, monkeypatch
) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(surface for surface in index.support_surfaces if surface.prim_path == "/Scene/Floor")

    def solve(**kwargs):
        target = list(kwargs["target_position_world_m"])
        return {
            "solved": target[0] < 1.0,
            "joint_positions": [0.1] * 7,
            "position_error_m": 0.0 if target[0] < 1.0 else 0.2,
            "manipulability": 0.12,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_robot_placement_geometry.solve_world_position_ik",
        solve,
    )
    gate = validate_robot_placement_geometry_candidate(
        index=index,
        proposal=_proposal(floor.surface_id),
        target_position_world_m=[0.8, 0.0, 0.5],
        trajectory_waypoints_world_m=[[0.8, 0.0, 0.5], [1.2, 0.0, 0.5]],
        trajectory_phase_ids=["target", "retreat"],
    )
    assert "task_target_outside_analytic_reach" not in gate["blockers"]
    assert "robot_trajectory_position_ik_unreached" in gate["blockers"]
    assert gate["status"] == "rejected"


def test_trajectory_ik_carries_sequential_seed_and_records_per_phase(monkeypatch) -> None:
    observed_seeds = []

    def solve(**kwargs):
        observed_seeds.append(kwargs["seed_joint_positions"])
        joint_positions = [float(len(observed_seeds))] * 7
        return {
            "solved": True,
            "joint_positions": joint_positions,
            "position_error_m": 5.0e-5,
            "manipulability": 0.14 - len(observed_seeds) * 0.01,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_robot_placement_geometry.solve_world_position_ik",
        solve,
    )
    gate = validate_robot_placement_trajectory_position_ik(
        proposal=_proposal("support"),
        trajectory_waypoints_world_m=[[0.4, 0.0, 0.5], [0.5, 0.0, 0.5]],
        trajectory_phase_ids=["precontact", "contact"],
        trajectory_orientations_world_xyzw=[
            [0.0, 0.70710678, 0.0, 0.70710678],
            [0.0, 0.70710678, 0.0, 0.70710678],
        ],
    )
    assert observed_seeds == [None, [1.0] * 7]
    assert [row["phase_id"] for row in gate["waypoints"]] == [
        "precontact",
        "contact",
    ]
    assert gate["waypoints"][1]["seed_joint_positions"] == [1.0] * 7
    assert gate["waypoints"][0]["orientation_world_xyzw"] == [
        0.0,
        0.70710678,
        0.0,
        0.70710678,
    ]
    assert gate["trajectory_position_ik_gate_digest"].startswith("sha256:")


def test_precomputed_trajectory_gate_preserves_exact_geometry_gate_bytes(
    tmp_path,
) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(
        surface
        for surface in index.support_surfaces
        if surface.prim_path == "/Scene/Floor"
    )
    proposal = _proposal(floor.surface_id)
    waypoints = [[0.6, 0.0, 0.5], [0.7, 0.0, 0.5]]
    phase_ids = ["precontact", "contact"]
    orientations = [
        [0.0, 0.70710678, 0.0, 0.70710678],
        [0.0, 0.70710678, 0.0, 0.70710678],
    ]
    direct = validate_robot_placement_geometry_candidate(
        index=index,
        proposal=proposal,
        target_position_world_m=[0.8, 0.0, 0.5],
        trajectory_waypoints_world_m=waypoints,
        trajectory_phase_ids=phase_ids,
        trajectory_orientations_world_xyzw=orientations,
    )
    trajectory_gate = validate_robot_placement_trajectory_position_ik(
        proposal=proposal,
        trajectory_waypoints_world_m=waypoints,
        trajectory_phase_ids=phase_ids,
        trajectory_orientations_world_xyzw=orientations,
    )
    precomputed = validate_robot_placement_geometry_candidate(
        index=index,
        proposal=proposal,
        target_position_world_m=[0.8, 0.0, 0.5],
        trajectory_gate_override=trajectory_gate,
    )

    assert precomputed == direct


def test_parallel_trajectory_workers_preserve_serial_order_and_digests(
    monkeypatch,
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_geometry as module

    proposals = [
        _proposal("support", position=(0.0, 0.0, 0.0)),
        {
            **_proposal("support", position=(0.05, 0.0, 0.0)),
            "candidate_id": "candidate-b",
        },
    ]
    kwargs = {
        "proposals": proposals,
        "trajectory_waypoints_world_m": [[0.5, 0.0, 0.5]],
        "trajectory_phase_ids": ["precontact"],
        "trajectory_orientations_world_xyzw": [
            [0.0, 0.70710678, 0.0, 0.70710678]
        ],
    }

    serial = _parallel_trajectory_gates(**kwargs, worker_count=1)

    executor_calls = []

    class RecordingExecutor:
        def __init__(self, *, max_workers, mp_context):
            executor_calls.append(
                {"max_workers": max_workers, "start_method": mp_context.get_start_method()}
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, work, *, chunksize):
            executor_calls[0]["chunksize"] = chunksize
            return map(function, work)

    monkeypatch.setattr(module, "ProcessPoolExecutor", RecordingExecutor)
    parallel = _parallel_trajectory_gates(**kwargs, worker_count=2)

    assert executor_calls == [
        {"max_workers": 2, "start_method": "spawn", "chunksize": 4}
    ]
    assert parallel == serial
    assert [
        row["trajectory_position_ik_gate_digest"] for row in parallel
    ] == [row["trajectory_position_ik_gate_digest"] for row in serial]


def test_parallel_geometry_workers_preserve_serial_order_and_gate_bytes(
    tmp_path, monkeypatch
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_geometry as module

    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    floor = next(
        surface
        for surface in index.support_surfaces
        if surface.prim_path == "/Scene/Floor"
    )
    proposals = [
        _proposal(floor.surface_id),
        {
            **_proposal(floor.surface_id, position=(0.05, 0.0, 0.0)),
            "candidate_id": "candidate-b",
        },
    ]
    kwargs = {
        "index": index,
        "proposals": proposals,
        "target_position_world_m": [0.8, 0.0, 0.5],
        "robot_id": "franka_panda",
    }
    serial = _parallel_geometry_gates(**kwargs, worker_count=1)
    executor_calls = []

    class RecordingExecutor:
        def __init__(self, *, max_workers, mp_context):
            executor_calls.append(
                {"max_workers": max_workers, "start_method": mp_context.get_start_method()}
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, work):
            return map(function, work)

    monkeypatch.setattr(module, "ProcessPoolExecutor", RecordingExecutor)
    parallel = _parallel_geometry_gates(**kwargs, worker_count=2)

    assert executor_calls == [{"max_workers": 2, "start_method": "spawn"}]
    assert parallel == serial
