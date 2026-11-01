"""The stance search reads the scene's collision triangles exactly as before, without a per-point loop."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import team_embodiment_stance as stance  # noqa: E402


def _reference(path: Path, offset: list[float]) -> tuple[Any, Any]:
    """The per-point loop ``_world_triangles`` used before it was vectorized."""

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, counts = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        matrix, first, cursor = cache.GetLocalToWorldTransform(prim), len(vertices), 0
        vertices.extend([float(v) + offset[axis] for axis, v in enumerate(matrix.Transform(point))]
                        for point in points)
        for count in counts:
            faces.extend((first + indices[cursor], first + indices[cursor + k], first + indices[cursor + k + 1])
                         for k in range(1, int(count) - 1))
            cursor += int(count)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _scene(path: Path, *, projective: bool = False) -> Path:
    from pxr import Gf, Usd, UsdGeom

    generator = np.random.default_rng(7)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(0.37, -1.23, 0.011))
    root.AddRotateXYZOp().Set(Gf.Vec3f(3.0, -7.0, 41.0))
    root.AddScaleOp().Set(Gf.Vec3f(1.1033, 1.1033, 1.1033))
    group = UsdGeom.Xform.Define(stage, "/World/Group")
    group.AddTranslateOp().Set(Gf.Vec3d(-2.5, 0.75, 0.125))
    if projective:
        group.AddTransformOp().Set(Gf.Matrix4d(1, 0, 0, 0.01, 0, 1, 0, -0.02, 0, 0, 1, 0.005, 0, 0, 0, 1))
    # Triangles, quads, pentagons and degenerate faces (one or two vertices) in one mesh.
    sizes = generator.choice([1, 2, 3, 4, 5], size=400, p=[0.02, 0.03, 0.55, 0.3, 0.1]).astype(int)
    points = generator.uniform(-3.0, 3.0, size=(900, 3)).astype(np.float32)
    mixed = UsdGeom.Mesh.Define(stage, "/World/Group/Mixed")
    mixed.CreatePointsAttr([Gf.Vec3f(*map(float, row)) for row in points])
    mixed.CreateFaceVertexCountsAttr([int(value) for value in sizes])
    mixed.CreateFaceVertexIndicesAttr([int(value) for value in generator.integers(0, len(points), int(sizes.sum()))])
    # A second mesh, so face indices carry the earlier meshes' vertex count.
    plain = UsdGeom.Mesh.Define(stage, "/World/Plain")
    plain.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1e-7)])
    plain.CreateFaceVertexCountsAttr([4])
    plain.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    # An empty mesh is skipped, as before.
    UsdGeom.Mesh.Define(stage, "/World/Empty")
    stage.GetRootLayer().Save()
    return path


@pytest.mark.parametrize("projective", [False, True])
def test_vectorized_triangles_match_the_per_point_loop_bit_for_bit(tmp_path, projective) -> None:
    path = _scene(tmp_path / "scene.usda", projective=projective)
    offset = [0.125, -3.5, 1e-3]
    vertices, faces = stance._world_triangles(path, offset)
    expected_vertices, expected_faces = _reference(path, offset)
    assert vertices.dtype == expected_vertices.dtype and faces.dtype == expected_faces.dtype
    assert vertices.shape == expected_vertices.shape and faces.shape == expected_faces.shape
    assert np.array_equal(vertices, expected_vertices) and np.array_equal(faces, expected_faces)


def test_a_scene_without_any_triangle_is_still_refused_by_name(tmp_path) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "lines.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, "/Lines")
    mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([2])
    mesh.CreateFaceVertexIndicesAttr([0, 1])
    stage.GetRootLayer().Save()
    with pytest.raises(stance.TeamEmbodimentError) as refused:
        stance._world_triangles(tmp_path / "lines.usda", [0.0, 0.0, 0.0])
    assert refused.value.blockers == ["team_embodiment_scene_collision_unavailable"]


def test_a_face_reaching_past_its_indices_still_raises(tmp_path) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "short.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, "/Short")
    mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0])
    stage.GetRootLayer().Save()
    with pytest.raises(IndexError):
        stance._world_triangles(tmp_path / "short.usda", [0.0, 0.0, 0.0])


def test_the_stance_search_is_unchanged_on_a_room(tmp_path) -> None:
    from tests.test_team_embodiment import write_room_usd

    room = write_room_usd(tmp_path / "collision.usda")
    vertices, faces = stance._world_triangles(room, [0.0, 0.0, 0.0])
    expected_vertices, expected_faces = _reference(room, [0.0, 0.0, 0.0])
    assert np.array_equal(vertices, expected_vertices) and np.array_equal(faces, expected_faces)
    chosen = stance.ground_stance(collision_path=room, collision_sha256="sha256:" + "0" * 64, offset=[0.0, 0.0, 0.0],
                                  subject_center=[2.0, 0.0, 0.8], subject_minimum=[1.95, -0.05, 0.75],
                                  subject_maximum=[2.05, 0.05, 0.85], footprint_half_extent_m=[0.15, 0.15],
                                  root_world_z_m=0.75)
    assert math.isclose(chosen["floor_z_m"], 0.0, abs_tol=1e-9) and chosen["clear_stances"] > 0
