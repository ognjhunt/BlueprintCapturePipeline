from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.articulated_usd_depth_sweep import (
    ArticulatedUsdDepthSweepError,
    load_articulated_usd_triangles,
    materialize_articulated_usd_depth_sweep,
    rasterize_triangle_depth,
    rotate_triangles_about_axis,
)


def _triangle(stage: Usd.Stage, path: str, points: list[tuple[float, float, float]]) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])


def _fixture_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/Asset/static")
    _triangle(stage, "/Asset/static/triangle", [(-1, -1, 4), (1, -1, 4), (0, 1, 4)])
    UsdGeom.Xform.Define(stage, "/Asset/door")
    _triangle(stage, "/Asset/door/triangle", [(0, -1, 2), (1, -1, 2), (0, 1, 2)])
    stage.GetRootLayer().Save()
    return path


def _camera() -> dict[str, object]:
    return {
        "camera_id": "external",
        "T_world_camera_opencv": np.eye(4).tolist(),
        "intrinsics": {
            "fx": 40.0,
            "fy": 40.0,
            "cx": 32.0,
            "cy": 24.0,
            "width": 64,
            "height": 48,
        },
    }


def test_rotation_and_perspective_depth_are_geometric() -> None:
    triangles = np.array([[[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [1.0, 1.0, 2.0]]])
    rotated = rotate_triangles_about_axis(
        triangles, pivot=[0, 0, 0], axis=[0, 0, 1], angle_deg=90
    )
    assert rotated[0, 0] == pytest.approx([0.0, 1.0, 2.0], abs=1e-12)

    depth = rasterize_triangle_depth(
        triangles,
        T_world_camera_opencv=np.eye(4).tolist(),
        intrinsics=_camera()["intrinsics"],  # type: ignore[arg-type]
    )
    assert np.isfinite(depth).any()
    assert float(depth[np.isfinite(depth)].min()) == pytest.approx(2.0)


def test_actual_usd_depth_sweep_is_deterministic_and_binds_geometry(tmp_path: Path) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    static, moving = load_articulated_usd_triangles(usd, moving_link_path="/Asset/door")
    assert static.shape == (1, 3, 3)
    assert moving.shape == (1, 3, 3)

    manifests = []
    for name in ("first", "second"):
        manifests.append(
            materialize_articulated_usd_depth_sweep(
                usd_path=usd,
                cameras=[_camera()],
                door_angles_deg=[0.0, 45.0],
                moving_link_path="/Asset/door",
                hinge_origin_asset_m=[0.0, 0.0, 0.0],
                hinge_axis_asset=[0.0, 0.0, 1.0],
                T_world_asset=np.eye(4).tolist(),
                output_root=tmp_path / name,
                resolution_scale=0.5,
            )
        )
    assert manifests[0]["manifest_digest"] == manifests[1]["manifest_digest"]
    assert manifests[0]["actual_mesh_depth_rasterized"] is True
    assert manifests[0]["caller_supplied_coverage_mask"] is False
    assert manifests[0]["depth_dimensions"] == [32, 24]
    assert manifests[0]["finite_depth_pixel_count_by_cell"] != [0, 0]
    depth = np.load(tmp_path / "first/replacement_depth_sweep.npy")
    assert depth.shape == (2, 24, 32)


def test_depth_sweep_rejects_missing_moving_link(tmp_path: Path) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    with pytest.raises(ArticulatedUsdDepthSweepError) as exc:
        load_articulated_usd_triangles(usd, moving_link_path="/Asset/missing")
    assert exc.value.codes == ("articulated_depth_moving_link_missing",)
