from __future__ import annotations

import math
import hashlib

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.task_evaluation_robot_placement_geometry import (
    _overlapping_scene_triangle_indices,
    _parallel_geometry_gates,
    _parallel_trajectory_gates,
    _supported_sample_count,
    _stage_triangles,
    _placed_task_asset_triangles,
    RobotPlacementGeometryError,
    build_robot_placement_geometry_index,
    enumerate_robot_placement_geometry_candidates,
    render_robot_placement_geometry_previews,
    summarize_robot_placement_geometry,
    validate_robot_placement_geometry_candidate,
    validate_robot_placement_trajectory_position_ik,
)


def test_spatial_collision_lookup_matches_exhaustive_boxes() -> None:
    random = np.random.default_rng(8062)
    for offset in (0.0, 1.0e6):
        robot_minimum = random.uniform(-2.0, 2.0, (350, 3)) + offset
        robot_maximum = robot_minimum + random.uniform(0.0, 0.4, (350, 3))
        scene_minimum = random.uniform(-3.0, 3.0, (180, 3)) + offset
        scene_maximum = scene_minimum + random.uniform(0.0, 0.8, (180, 3))
        # Include a touching point and a very elongated box, then retain a
        # non-sorted subset to check inclusive boundaries and result ordering.
        scene_minimum[0] = robot_maximum[0]
        scene_maximum[0] = robot_maximum[0]
        scene_minimum[1] = [-100.0 + offset, offset, offset]
        scene_maximum[1] = [100.0 + offset, offset, offset]
        selected = np.r_[0, 1, random.permutation(np.arange(2, 180))[:90]]
        expected = [
            int(index)
            for index in selected
            if np.any(
                np.all(
                    (robot_maximum >= scene_minimum[index])
                    & (robot_minimum <= scene_maximum[index]),
                    axis=1,
                )
            )
        ]

        assert _overlapping_scene_triangle_indices(
            scene_minimum=scene_minimum,
            scene_maximum=scene_maximum,
            scene_indices=selected,
            robot_minimum=robot_minimum,
            robot_maximum=robot_maximum,
        ) == expected
        assert 0 in expected


def test_spatial_collision_lookup_handles_empty_geometry() -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    point = np.zeros((1, 3), dtype=np.float64)
    for selected, robot in ((np.asarray([], dtype=int), point), (np.asarray([0]), empty)):
        assert _overlapping_scene_triangle_indices(
            scene_minimum=point,
            scene_maximum=point,
            scene_indices=selected,
            robot_minimum=robot,
            robot_maximum=robot,
        ) == []


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


def test_robot_preview_uses_default_root_and_includes_instanced_gripper(tmp_path):
    from blueprint_pipeline.task_evaluation_robot_placement_geometry import _stage_triangles, _robot_root
    points=[[0,0,0],[1,0,0],[0,1,0]]
    part=Usd.Stage.CreateNew(str(tmp_path/'part.usda'))
    part.SetDefaultPrim(UsdGeom.Xform.Define(part,'/Part').GetPrim())
    _mesh(part,'/Part/Geometry',points,[[0,1,2]])
    part.GetRootLayer().Save()
    robot=Usd.Stage.CreateInMemory()
    robot.SetDefaultPrim(UsdGeom.Xform.Define(robot,'/Robot').GetPrim())
    _mesh(robot,'/Robot/Arm',points,[[0,1,2]])
    proxy=UsdGeom.Xform.Define(robot,'/Robot/Gripper').GetPrim()
    proxy.GetReferences().AddReference(str(tmp_path/'part.usda'))
    proxy.SetInstanceable(True)
    _mesh(robot,'/DemoBowl',points,[[0,1,2]])
    triangles,paths=_stage_triangles(robot,root_prim=_robot_root(robot))
    assert len(triangles)==2
    assert set(paths)=={'/Robot/Arm','/Robot/Gripper/Geometry'}


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


@pytest.mark.parametrize("schema", [UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Capsule, UsdGeom.Sphere, UsdGeom.Cone])
def test_analytic_shapes_are_conservative_transformed_geometry(schema):
    stage = Usd.Stage.CreateInMemory()
    shape = schema.Define(stage, "/Shape")
    xform = UsdGeom.Xformable(shape)
    xform.AddTranslateOp().Set(Gf.Vec3d(3.0, -2.0, 1.0))
    xform.AddRotateZOp().Set(35.0)
    xform.AddScaleOp().Set(Gf.Vec3f(0.3, 0.5, 0.7))
    triangles, paths = _stage_triangles(stage)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(shape.GetPrim()).ComputeAlignedRange()
    assert len(triangles) == 12
    assert set(paths) == {"/Shape"}
    np.testing.assert_allclose(triangles.min(axis=(0, 1)), list(bounds.GetMin()), atol=1.e-7)
    np.testing.assert_allclose(triangles.max(axis=(0, 1)), list(bounds.GetMax()), atol=1.e-7)


def _task_asset(tmp_path):
    path = tmp_path / "task.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Task").GetPrim())
    panel = UsdGeom.Cube.Define(stage, "/Task/Panel")
    panel.CreateSizeAttr(0.5)
    stage.GetRootLayer().Save()
    return {
        "asset_id": "task-actor",
        "asset_usd_path": str(path),
        "asset_digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "pose": {"position_world_m": [0., 0., 0.4], "orientation_xyzw": [0., 0., 0., 1.]},
    }


def test_placed_task_actor_blocks_previously_clear_robot_pose(tmp_path):
    scene, robot = _assets(tmp_path)
    background = build_robot_placement_geometry_index(scene_collision_usd_path=scene, robot_asset_usd_path=robot)
    floor = next(s for s in background.support_surfaces if s.prim_path == "/Scene/Floor")
    arguments = dict(proposal=_proposal(floor.surface_id), target_position_world_m=[0.8, 0., 0.5])
    assert validate_robot_placement_geometry_candidate(index=background, **arguments)["status"] == "passed"
    actor = _task_asset(tmp_path)
    combined = build_robot_placement_geometry_index(scene_collision_usd_path=scene, robot_asset_usd_path=robot, task_assets=[actor])
    gate = validate_robot_placement_geometry_candidate(index=combined, **arguments)
    assert gate["collision_passed"] is False
    assert any(path.startswith("/TaskActors/") for path in gate["scene_overlap_prim_paths"])
    assert combined.support_surfaces == background.support_surfaces
    assert combined.scene_digest != background.scene_digest
    actor["pose"]["position_world_m"] = [-1., 0., 0.4]
    moved = build_robot_placement_geometry_index(scene_collision_usd_path=scene, robot_asset_usd_path=robot, task_assets=[actor])
    assert moved.scene_digest != combined.scene_digest
    assert validate_robot_placement_geometry_candidate(index=moved, **arguments)["status"] == "passed"


def test_task_actor_rejects_changed_source_bytes(tmp_path):
    scene, robot = _assets(tmp_path)
    actor = _task_asset(tmp_path)
    with open(actor["asset_usd_path"], "a") as stream:
        stream.write("\n# changed source\n")
    with pytest.raises(RobotPlacementGeometryError, match="task_asset_digest_mismatch"):
        build_robot_placement_geometry_index(scene_collision_usd_path=scene, robot_asset_usd_path=robot, task_assets=[actor])


def test_actor_pose_preserves_asset_root_transform_and_binds_referenced_layer(tmp_path):
    actor = _task_asset(tmp_path)
    path = actor["asset_usd_path"]
    stage = Usd.Stage.Open(path)
    root = UsdGeom.Xformable(stage.GetDefaultPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(0.2, 0., 0.))
    stage.GetRootLayer().Save()
    wrapper = tmp_path / "wrapper.usda"
    outer = Usd.Stage.CreateNew(str(wrapper))
    outer.SetDefaultPrim(UsdGeom.Xform.Define(outer, "/Wrapper").GetPrim())
    outer.GetDefaultPrim().GetReferences().AddReference(path)
    outer.GetRootLayer().Save()
    actor.update(asset_usd_path=str(wrapper), asset_digest="sha256:" + hashlib.sha256(wrapper.read_bytes()).hexdigest())
    actor["pose"] = {"position_world_m": [1., 2., 3.], "orientation_xyzw": [0., 0., math.sqrt(.5), math.sqrt(.5)]}
    triangles, _, bindings = _placed_task_asset_triangles([actor])
    np.testing.assert_allclose(triangles[0].mean(axis=(0, 1)), [1., 2.2, 3.], atol=1.e-7)
    assert {row["path"] for row in bindings[0]["collision_layers"]} == {path, str(wrapper)}


def test_enumeration_prefilter_preserves_every_passing_gate(tmp_path) -> None:
    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene, robot_asset_usd_path=robot
    )
    floor = next(s for s in index.support_surfaces if s.prim_path == "/Scene/Floor")
    proposals = [
        _proposal(floor.surface_id, position=position, yaw=yaw)
        for position in ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 0.0, 0.0), (0.0, 0.0, 0.1))
        for yaw in (0.0, math.pi / 2, math.pi)
    ]
    arguments = dict(index=index, proposals=proposals, target_position_world_m=[0.8, 0.0, 0.5],
                     robot_id="franka_panda", worker_count=1)
    complete = _parallel_geometry_gates(**arguments)
    filtered = _parallel_geometry_gates(**arguments, prefilter_nonpassing=True)
    assert [g for g in complete if g["status"] == "passed"]
    assert [g for g in filtered if g["status"] == "passed"] == [
        g for g in complete if g["status"] == "passed"
    ]
    assert any(len(g) == 1 for g in filtered)


@pytest.mark.parametrize("neighbor_height,expected", [(0.01, True), (0.04, False)])
def test_support_uses_adjacent_height_bins_only_within_declared_tolerance(
    tmp_path, neighbor_height, expected
) -> None:
    scene_path, robot_path = _assets(tmp_path)
    scene = Usd.Stage.Open(str(scene_path))
    scene.RemovePrim("/Scene/Floor")
    _mesh(
        scene, "/Scene/Floor",
        [(-2, -2, 0), (0, -2, 0), (0, 2, 0), (-2, 2, 0),
         (0, -2, neighbor_height), (2, -2, neighbor_height),
         (2, 2, neighbor_height), (0, 2, neighbor_height)],
        [(0, 1, 2, 3), (4, 5, 6, 7)],
    )
    scene.GetRootLayer().Save()
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene_path, robot_asset_usd_path=robot_path
    )
    floor = next(s for s in index.support_surfaces if s.prim_path == "/Scene/Floor" and s.height_m == 0.0)
    proposal = _proposal(floor.surface_id)
    gate = validate_robot_placement_geometry_candidate(
        index=index, proposal=proposal, target_position_world_m=[0.8, 0.0, 0.5]
    )
    assert gate["support_passed"] is expected
    assert (gate["supported_sample_count"] == 5) is expected
    prefiltered = _parallel_geometry_gates(
        index=index, proposals=[proposal], target_position_world_m=[0.8, 0.0, 0.5],
        robot_id="franka_panda", worker_count=1, prefilter_nonpassing=True,
    )
    assert (prefiltered[0]["status"] == "passed") is expected


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


def _shelf_scene(tmp_path):
    """A floor plus many tiny ledges sitting near the target height.

    The ledges are too small to support a robot base; the floor is the only
    real support. Ranking by height proximity puts every ledge ahead of the
    floor, which is what used to push the floor out of the search.
    """
    scene_path = tmp_path / "shelf-scene.usda"
    scene = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(scene, "/Scene")
    scene.SetDefaultPrim(root.GetPrim())
    _mesh(
        scene,
        "/Scene/Floor",
        [(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)],
        [(0, 1, 2, 3)],
    )
    for index in range(13):
        # Distinct heights just under the target so each ledge is its own
        # support surface and every one of them outranks the floor. They sit
        # clustered on +x, inside the xy prefilter but clear of the open -x
        # floor the robot can actually stand on.
        height = 0.38 + 0.005 * index
        y0 = -0.07 + 0.005 * index
        # Above the 0.015 m2 support-surface floor, well under the robot's
        # 0.30 x 0.30 m base footprint, so each ledge registers as a support
        # yet cannot actually carry the robot.
        _box(
            scene,
            f"/Scene/Ledge_{index:02d}",
            (0.70, y0, height - 0.01),
            (0.84, y0 + 0.14, height),
        )
    scene.GetRootLayer().Save()

    robot_path = tmp_path / "shelf-robot.usda"
    robot = Usd.Stage.CreateNew(str(robot_path))
    robot.SetDefaultPrim(UsdGeom.Xform.Define(robot, "/Robot").GetPrim())
    _box(robot, "/Robot/Body", (-0.15, -0.15, 0), (0.15, 0.15, 0.75))
    robot.GetRootLayer().Save()
    return scene_path, robot_path


def test_empty_initial_support_search_expands_to_other_observed_surfaces(tmp_path):
    scene_path, robot_path = _assets(tmp_path)
    scene = Usd.Stage.Open(str(scene_path))
    scene.RemovePrim("/Scene/Floor")
    scene.RemovePrim("/Scene/Obstacle")
    # The largest surfaces lie inside a solid obstruction. The surfaces
    # closest to task height are too small to support the base.
    _box(scene, "/Scene/Occupied", (-0.2, -0.9, 0.0), (1.3, 0.9, 2.0))
    for n, z in enumerate((0.1, 0.2, 0.3)):
        _mesh(scene, f"/Scene/Large_{n}",
              [(-0.2, -0.9, z), (1.3, -0.9, z), (1.3, 0.9, z), (-0.2, 0.9, z)],
              [(0, 1, 2, 3)])
    for n in range(13):
        z = 0.7 + 0.005 * n
        _mesh(scene, f"/Scene/Small_{n}",
              [(0.4, -0.07, z), (0.54, -0.07, z), (0.54, 0.07, z), (0.4, 0.07, z)],
              [(0, 1, 2, 3)])
    _mesh(scene, "/Scene/UsableShelf",
          [(-0.9, -0.35, 0.4), (-0.25, -0.35, 0.4), (-0.25, 0.35, 0.4), (-0.9, 0.35, 0.4)],
          [(0, 1, 2, 3)])
    scene.GetRootLayer().Save()
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene_path, robot_asset_usd_path=robot_path
    )
    diagnostics = {}
    candidates = enumerate_robot_placement_geometry_candidates(
        index=index, target_position_world_m=[0.0, 0.0, 0.9], maximum_candidates=3,
        diagnostics=diagnostics,
    )
    assert candidates
    assert all(p["support_surface_id"].startswith("/Scene/UsableShelf@") for p in candidates)
    assert diagnostics["support_search_pass_count"] == 2
    assert diagnostics["support_search_exhausted"] is True
    assert diagnostics["searched_support_surface_count"] > 15
    assert diagnostics["searched_support_surface_count"] == diagnostics["available_support_surface_count"]


def test_no_eligible_observed_support_returns_an_empty_inventory(tmp_path):
    scene_path, robot_path = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene_path, robot_asset_usd_path=robot_path
    )
    diagnostics = {}
    assert enumerate_robot_placement_geometry_candidates(
        index=index, target_position_world_m=[0.0, 0.0, 10.0], diagnostics=diagnostics
    ) == []
    assert diagnostics["proposal_count"] == 0
    assert diagnostics["available_support_surface_count"] == 0
    assert diagnostics["support_search_exhausted"] is True


def test_enumeration_keeps_the_largest_support_when_height_ranking_buries_it(
    tmp_path,
) -> None:
    """The floor must stay in the search even when it ranks last by height.

    Scene 840938 (2026-09-16): the floor ranked 18th of 19 viable supports at
    90.8 m2 and fell outside the height-ranked cap, so every evaluated
    candidate sat on a shelf as small as 0.029 m2 and the run blocked on
    robot_placement_geometry_candidate_inventory_empty.
    """
    import blueprint_pipeline.task_evaluation_robot_placement_geometry as module

    scene, robot = _shelf_scene(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    target = [0.0, 0.0, 0.45]

    floor = next(
        surface for surface in index.support_surfaces
        if surface.prim_path == "/Scene/Floor"
    )
    ledges = [
        surface for surface in index.support_surfaces
        if surface.prim_path.startswith("/Scene/Ledge_")
    ]
    # The premise: enough ledges rank ahead of the floor to fill the cap.
    assert len(ledges) >= module._HEIGHT_RANKED_SURFACE_LIMIT
    assert floor.area_m2 > max(surface.area_m2 for surface in ledges)

    candidates = enumerate_robot_placement_geometry_candidates(
        index=index,
        target_position_world_m=target,
        maximum_candidates=4,
    )
    assert candidates, "the largest support must stay in the search"
    assert all(
        candidate["support_surface_id"] == floor.surface_id
        for candidate in candidates
    )


def test_enumeration_still_prefers_surfaces_near_the_target_height(tmp_path) -> None:
    """Keeping the largest support must not displace the height ranking."""

    scene, robot = _assets(tmp_path)
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene,
        robot_asset_usd_path=robot,
    )
    target = [0.8, 0.0, 0.5]

    candidates = enumerate_robot_placement_geometry_candidates(
        index=index,
        target_position_world_m=target,
        maximum_candidates=5,
    )
    assert len(candidates) == 5


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


def test_thin_desk_in_gap_between_links_does_not_block_base(tmp_path) -> None:
    scene_path = tmp_path / "scene.usda"
    scene = Usd.Stage.CreateNew(str(scene_path))
    scene.SetDefaultPrim(UsdGeom.Xform.Define(scene, "/Scene").GetPrim())
    _mesh(scene, "/Scene/Floor", [(-2, -2, 0), (2, -2, 0),
                                 (2, 2, 0), (-2, 2, 0)], [(0, 1, 2, 3)])
    _box(scene, "/Scene/Desk", (-0.1, -0.2, 0.7), (0.1, 0.2, 0.75))
    scene.GetRootLayer().Save()
    robot_path = tmp_path / "robot.usda"
    robot = Usd.Stage.CreateNew(str(robot_path))
    robot.SetDefaultPrim(UsdGeom.Xform.Define(robot, "/Robot").GetPrim())
    _box(robot, "/Robot/Left", (-0.55, -0.05, 0), (-0.45, 0.05, 0.8))
    _box(robot, "/Robot/Right", (0.45, -0.05, 0), (0.55, 0.05, 0.8))
    robot.GetRootLayer().Save()
    index = build_robot_placement_geometry_index(
        scene_collision_usd_path=scene_path, robot_asset_usd_path=robot_path,
    )
    floor = next(row for row in index.support_surfaces if row.prim_path == "/Scene/Floor")
    gate = validate_robot_placement_geometry_candidate(
        index=index, proposal=_proposal(floor.surface_id),
        target_position_world_m=[0.8, 0.0, 0.5],
    )
    assert gate["status"] == "passed"
    assert gate["scene_overlap_triangle_count"] == 0


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

    assert [image["label"] for image in images] == ["top_down_xy", "side_task", "oblique"]
    assert all(image['render_provenance']['depth_buffer_shared_by_scene_and_robot'] for image in images)
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
