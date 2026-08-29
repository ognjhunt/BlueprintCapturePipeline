from __future__ import annotations

import math

from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.task_evaluation_robot_placement_geometry import (
    build_robot_placement_geometry_index,
    enumerate_robot_placement_geometry_candidates,
    render_robot_placement_geometry_previews,
    summarize_robot_placement_geometry,
    validate_robot_placement_geometry_candidate,
)


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
