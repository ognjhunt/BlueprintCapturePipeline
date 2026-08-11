from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_usd_depth_sweep import (
    ArticulatedUsdDepthSweepError,
    _primitive_points_and_faces,
    conservative_max_pool_alpha,
    evaluate_source_alpha_coverage,
    load_articulated_usd_triangles,
    load_usd_link_triangles,
    materialize_articulated_usd_depth_sweep,
    materialize_deleted_source_layer_replacement_coverage_qualification,
    materialize_replacement_usd_depth_sweep,
    materialize_reference_hybrid_review,
    materialize_source_layer_replacement_coverage_audit,
    materialize_target_core_replacement_coverage_audit,
    rasterize_triangle_depth,
    rotate_triangles_about_axis,
    seal_replacement_usd_depth_sweep_request,
    validate_replacement_usd_depth_sweep_request,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


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


def _primitive_graph_fixture_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    stage.SetDefaultPrim(root.GetPrim())
    links = {
        "root": ([0.0, 0.0, 4.0], "cube"),
        "door": ([0.45, 0.5, 4.0], "cube"),
        "latch": ([0.8, 0.5, 4.0], "cylinder"),
    }
    for link_id, (position, kind) in links.items():
        link = UsdGeom.Xform.Define(stage, f"/Asset/links/{link_id}")
        link.AddTranslateOp().Set(Gf.Vec3d(*position))
        if kind == "cube":
            geometry = UsdGeom.Cube.Define(
                stage, f"/Asset/links/{link_id}/geometry/shape"
            )
            geometry.CreateSizeAttr(0.35)
        else:
            geometry = UsdGeom.Cylinder.Define(
                stage, f"/Asset/links/{link_id}/geometry/shape"
            )
            geometry.CreateAxisAttr("X")
            geometry.CreateRadiusAttr(0.14)
            geometry.CreateHeightAttr(0.28)
    for joint_id, parent, child, parent_position in (
        ("hinge", "root", "door", [0.45, 0.5, 0.0]),
        ("coupler", "door", "latch", [0.35, 0.0, 0.0]),
    ):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/joints/{joint_id}")
        joint.CreateAxisAttr("X")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/Asset/links/{parent}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Asset/links/{child}")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*parent_position))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()
    return path


def _primitive_graph() -> dict:
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "root", "is_root": True, "semantic_role": "root"},
            {"link_id": "door", "is_root": False, "semantic_role": "target"},
            {"link_id": "latch", "is_root": False, "semantic_role": "dependent"},
        ],
        "joints": [
            {
                "joint_id": "hinge",
                "parent_link_id": "root",
                "child_link_id": "door",
                "joint_type": "revolute",
                "role": "target",
                "axis": [1.0, 0.0, 0.0],
                "limits": [0.0, 1.0],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 0.0,
                    "damping": 1.0,
                    "maximum_force": 10.0,
                },
                "dependency": None,
            },
            {
                "joint_id": "coupler",
                "parent_link_id": "door",
                "child_link_id": "latch",
                "joint_type": "revolute",
                "role": "dependent",
                "axis": [1.0, 0.0, 0.0],
                "limits": [0.0, 0.2],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 1.0,
                    "damping": 1.0,
                    "maximum_force": 10.0,
                },
                "dependency": {
                    "driver_joint_id": "hinge",
                    "multiplier": 0.2,
                    "offset": 0.0,
                    "tolerance": 0.001,
                },
            },
        ],
        "collision_pairs": [
            {"link_a": "root", "link_b": "door", "collision_enabled": True},
            {"link_a": "root", "link_b": "latch", "collision_enabled": True},
            {"link_a": "door", "link_b": "latch", "collision_enabled": False},
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"hinge": [0.5, 0.8]},
        },
    }


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


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _render_manifest(
    root: Path,
    *,
    background: str,
    image: np.ndarray,
    scene_id: str,
    inpainting_mask_input_authorized: bool = False,
) -> Path:
    root.mkdir()
    frames = root / "frames"
    frames.mkdir()
    frame = frames / "external.png"
    assert cv2.imwrite(str(frame), image)
    value = {
        "schema_version": "sealed_camera_render_manifest.v1",
        "status": "rendered_exact_cameras",
        "camera_set_label": f"{scene_id}_fixture",
        "render_count": 1,
        "splat_digest": "sha256:" + "a" * 64,
        "renderer_identity": {"background_rgb": background},
        "renders": [
            {
                "camera_id": "external",
                "relative_path": "frames/external.png",
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "digest": _sha256(frame),
            }
        ],
    }
    if inpainting_mask_input_authorized:
        value.update(
            {
                "authorization_class": "method_input",
                "calibrated_camera_file": {
                    "binding": "caller_file_exact_match",
                    "camera_count": 1,
                },
                "render_settings": {
                    "dimensions": {
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                    }
                },
            }
        )
    value["sealed_camera_render_manifest_digest"] = canonical_digest(
        value, digest_field="sealed_camera_render_manifest_digest"
    )
    path = root / "sealed_camera_render_manifest.v1.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


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


def test_general_depth_sweep_supports_primitives_and_complete_joint_state_cells(
    tmp_path: Path,
) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "primitive_graph.usda")
    graph = _primitive_graph()
    link_paths = {
        link_id: f"/Asset/links/{link_id}" for link_id in ("root", "door", "latch")
    }
    joint_paths = {
        joint_id: f"/Asset/joints/{joint_id}" for joint_id in ("hinge", "coupler")
    }
    local, _rest, type_counts = load_usd_link_triangles(
        usd, asset_prim_path="/Asset", link_paths=link_paths
    )
    assert set(local) == {"root", "door", "latch"}
    assert type_counts == {"Cube": 2, "Cylinder": 1}
    identity = np.eye(4).tolist()
    request = seal_replacement_usd_depth_sweep_request(
        asset_id="fixture_articulated_asset",
        task_kind="articulated_interaction",
        task_freeze_digest="sha256:" + "a" * 64,
        replacement_usd_sha256=_sha256(usd),
        replacement_usd_size_bytes=usd.stat().st_size,
        articulation_graph=graph,
        link_paths=link_paths,
        joint_paths=joint_paths,
        task_scoring_frame={
            "frame_id": "asset_root",
            "T_asset_task_scoring": identity,
        },
        camera_contract_digest="sha256:" + "b" * 64,
        cameras=[_camera()],
        joint_state_cells=[
            {
                "cell_id": "reset",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.0, "coupler": 0.0},
            },
            {
                "cell_id": "open",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.7, "coupler": 0.14},
            },
        ],
        resolution_scale=0.5,
    )
    result = materialize_replacement_usd_depth_sweep(
        usd_path=usd,
        request=request,
        output_root=tmp_path / "general_depth",
    )

    assert result["schema_version"] == "replacement_usd_depth_sweep.v2"
    assert result["asset_id"] == "fixture_articulated_asset"
    assert result["task_freeze_digest"] == "sha256:" + "a" * 64
    assert result["camera_contract_digest"] == "sha256:" + "b" * 64
    assert result["replacement_usd"]["size_bytes"] == usd.stat().st_size
    assert result["actual_usd_geometry_depth_rasterized"] is True
    assert result["actual_mesh_depth_rasterized"] is False
    assert result["geometry_type_counts"] == {"Cube": 2, "Cylinder": 1}
    assert result["asset_root_authored_transform_removed_before_placement"] is True
    assert result["T_world_asset_applied_exactly_once_per_cell"] is True
    assert result["finite_depth_pixel_count_by_cell"] != [0, 0]
    depths = np.load(tmp_path / "general_depth/replacement_depth_sweep.npy")
    assert depths.shape == (2, 24, 32)
    assert not np.array_equal(depths[0], depths[1])
    common_coverage = np.all(np.isfinite(depths) & (depths > 0.0), axis=0)
    mask = tmp_path / "target_core.png"
    assert cv2.imwrite(str(mask), common_coverage.astype(np.uint8) * 255)
    coverage = materialize_target_core_replacement_coverage_audit(
        target_core_mask_paths={"external": mask},
        depth_sweep_manifest_path=tmp_path
        / "general_depth/replacement_usd_depth_sweep.v2.json",
        output_root=tmp_path / "general_coverage",
        maximum_uncovered_fraction=0.1,
    )
    assert coverage["coverage_qualified"] is True
    assert coverage["state_cell_ids"] == ["reset", "open"]
    assert "door_state_angles_degrees" not in coverage
    assert "derived_from_all_door_cells" not in coverage[
        "residual_target_core_seam_masks"
    ][0]


def test_general_depth_excludes_hidden_proxy_and_guide_geometry(tmp_path: Path) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "visibility.usda")
    stage = Usd.Stage.Open(str(usd))
    hidden = UsdGeom.Cube.Define(stage, "/Asset/links/root/geometry/hidden")
    hidden.CreateSizeAttr(100.0)
    hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    proxy = UsdGeom.Cube.Define(stage, "/Asset/links/root/geometry/proxy")
    proxy.CreateSizeAttr(100.0)
    proxy.CreatePurposeAttr(UsdGeom.Tokens.proxy)
    guide = UsdGeom.Cube.Define(stage, "/Asset/links/root/geometry/guide")
    guide.CreateSizeAttr(100.0)
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    transparent = UsdGeom.Cube.Define(
        stage, "/Asset/links/root/geometry/transparent"
    )
    transparent.CreateSizeAttr(100.0)
    transparent.CreateDisplayOpacityAttr([0.25])
    stage.GetRootLayer().Save()

    _triangles, _rest, type_counts = load_usd_link_triangles(
        usd,
        asset_prim_path="/Asset",
        link_paths={
            link_id: f"/Asset/links/{link_id}"
            for link_id in ("root", "door", "latch")
        },
    )

    assert type_counts == {"Cube": 2, "Cylinder": 1}


def test_mesh_depth_rejects_ngons_instead_of_unsafe_fan_coverage(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateNew(str(tmp_path / "concave.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(2.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.5, 0.0),
            Gf.Vec3f(2.0, 2.0, 0.0),
            Gf.Vec3f(0.0, 2.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([5])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4])

    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="articulated_depth_mesh_topology_not_explicit_triangles",
    ):
        _primitive_points_and_faces(mesh.GetPrim())


def test_mesh_depth_rejects_out_of_range_or_degenerate_triangle_indices(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateNew(str(tmp_path / "invalid_indices.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 7])
    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="articulated_depth_face_indices_invalid",
    ):
        _primitive_points_and_faces(mesh.GetPrim())

    mesh.CreateFaceVertexIndicesAttr([0, 1, 1])
    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="articulated_depth_face_indices_invalid",
    ):
        _primitive_points_and_faces(mesh.GetPrim())


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_cylinder_tessellation_supports_all_openusd_axes(
    tmp_path: Path, axis: str
) -> None:
    stage = Usd.Stage.CreateNew(str(tmp_path / f"cylinder_{axis}.usda"))
    cylinder = UsdGeom.Cylinder.Define(stage, "/Cylinder")
    cylinder.CreateAxisAttr(axis)
    cylinder.CreateRadiusAttr(0.25)
    cylinder.CreateHeightAttr(0.8)
    stage.GetRootLayer().Save()

    points, faces = _primitive_points_and_faces(cylinder.GetPrim())

    assert points.shape == (130, 3)
    assert faces.shape == (256, 3)
    axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
    assert points[:, axis_index].min() == pytest.approx(-0.4)
    assert points[:, axis_index].max() == pytest.approx(0.4)


def test_legacy_840796_single_moving_link_adapter_accepts_primitives_and_one_placement(
    tmp_path: Path,
) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "legacy_graph.usda")
    manifest = materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0, 30.0],
        moving_link_path="/Asset/links/door",
        hinge_origin_asset_m=[0.0, 0.0, 4.0],
        hinge_axis_asset=[1.0, 0.0, 0.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=tmp_path / "legacy_depth",
        resolution_scale=0.5,
    )

    assert manifest["schema_version"] == "adp009b_articulated_usd_depth_sweep.v1"
    assert manifest["door_state_count"] == 2
    assert manifest["actual_usd_geometry_depth_rasterized"] is True
    assert manifest["actual_mesh_depth_rasterized"] is False
    assert manifest["legacy_single_moving_link_compatibility_adapter"] is True
    assert min(manifest["finite_depth_pixel_count_by_cell"]) > 0


def test_rigid_pose_cells_bind_nonidentity_task_scoring_frame_without_double_placement(
    tmp_path: Path,
) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "primitive_graph.usda")
    graph = _primitive_graph()
    graph["links"] = graph["links"][:2]
    graph["joints"] = graph["joints"][:1]
    graph["joints"][0]["role"] = "locked"
    graph["joints"][0]["drive"]["stiffness"] = 10.0
    graph["collision_pairs"] = graph["collision_pairs"][:1]
    graph["success_predicate"]["joint_intervals"] = {}
    scoring_offset = np.eye(4)
    scoring_offset[2, 3] = 0.5
    first_scoring_pose = scoring_offset.copy()
    second_scoring_pose = scoring_offset.copy()
    second_scoring_pose[0, 3] = 0.2
    request = seal_replacement_usd_depth_sweep_request(
        asset_id="fixture_rigid_asset",
        task_kind="rigid_object_manipulation",
        task_freeze_digest="sha256:" + "c" * 64,
        replacement_usd_sha256=_sha256(usd),
        replacement_usd_size_bytes=usd.stat().st_size,
        articulation_graph=graph,
        link_paths={"root": "/Asset/links/root", "door": "/Asset/links/door"},
        joint_paths={"hinge": "/Asset/joints/hinge"},
        task_scoring_frame={
            "frame_id": "observed_object_center",
            "T_asset_task_scoring": scoring_offset.tolist(),
        },
        camera_contract_digest="sha256:" + "d" * 64,
        cameras=[_camera()],
        joint_state_cells=[
            {
                "cell_id": "source_pose",
                "T_world_task_scoring": first_scoring_pose.tolist(),
                "joint_positions": {"hinge": 0.0},
            },
            {
                "cell_id": "destination_pose",
                "T_world_task_scoring": second_scoring_pose.tolist(),
                "joint_positions": {"hinge": 0.0},
            },
        ],
        resolution_scale=0.5,
    )
    result = materialize_replacement_usd_depth_sweep(
        usd_path=usd,
        request=request,
        output_root=tmp_path / "rigid_depth",
    )

    assert np.allclose(result["cells"][0]["T_world_asset"], np.eye(4))
    assert result["cells"][1]["T_world_asset"][0][3] == pytest.approx(0.2)
    # The authored /Asset translation of +10m is stripped; otherwise this
    # camera would see no geometry at either requested scoring pose.
    assert min(result["finite_depth_pixel_count_by_cell"]) > 0


def test_general_depth_request_rejects_incomplete_or_inconsistent_state_cells(
    tmp_path: Path,
) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "primitive_graph.usda")
    graph = _primitive_graph()
    identity = np.eye(4).tolist()
    request = seal_replacement_usd_depth_sweep_request(
        asset_id="fixture_invalid_state_asset",
        task_kind="articulated_interaction",
        task_freeze_digest="sha256:" + "e" * 64,
        replacement_usd_sha256=_sha256(usd),
        replacement_usd_size_bytes=usd.stat().st_size,
        articulation_graph=graph,
        link_paths={
            link_id: f"/Asset/links/{link_id}"
            for link_id in ("root", "door", "latch")
        },
        joint_paths={
            joint_id: f"/Asset/joints/{joint_id}"
            for joint_id in ("hinge", "coupler")
        },
        task_scoring_frame={
            "frame_id": "asset_root",
            "T_asset_task_scoring": identity,
        },
        camera_contract_digest="sha256:" + "f" * 64,
        cameras=[_camera()],
        joint_state_cells=[
            {
                "cell_id": "reset",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.0, "coupler": 0.0},
            },
            {
                "cell_id": "open",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.7, "coupler": 0.14},
            },
        ],
    )
    request["joint_state_cells"][1]["joint_positions"] = {
        "hinge": 0.7,
        "coupler": 0.0,
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="replacement_depth_cell_dependency_invalid:1:coupler",
    ):
        validate_replacement_usd_depth_sweep_request(request)

    request["joint_state_cells"][1]["joint_positions"]["coupler"] = 0.14
    request["camera_contract_digest"] = "not-a-digest"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="replacement_depth_camera_contract_digest_invalid",
    ):
        validate_replacement_usd_depth_sweep_request(request)


def test_general_depth_rejects_affine_camera_and_types_missing_dependency_driver(
    tmp_path: Path,
) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "primitive_graph.usda")
    graph = _primitive_graph()
    identity = np.eye(4).tolist()
    request = seal_replacement_usd_depth_sweep_request(
        asset_id="fixture_invalid_calibration_asset",
        task_kind="articulated_interaction",
        task_freeze_digest="sha256:" + "5" * 64,
        replacement_usd_sha256=_sha256(usd),
        replacement_usd_size_bytes=usd.stat().st_size,
        articulation_graph=graph,
        link_paths={
            link_id: f"/Asset/links/{link_id}"
            for link_id in ("root", "door", "latch")
        },
        joint_paths={
            joint_id: f"/Asset/joints/{joint_id}"
            for joint_id in ("hinge", "coupler")
        },
        task_scoring_frame={
            "frame_id": "asset_root",
            "T_asset_task_scoring": identity,
        },
        camera_contract_digest="sha256:" + "6" * 64,
        cameras=[_camera()],
        joint_state_cells=[
            {
                "cell_id": "reset",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.0, "coupler": 0.0},
            },
            {
                "cell_id": "open",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.7, "coupler": 0.14},
            },
        ],
    )
    request["cameras"][0]["T_world_camera_opencv"][0][0] = 2.0
    request["camera_rows_digest"] = canonical_digest(
        {"cameras": request["cameras"]}
    )
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="replacement_depth_camera_transform_invalid:0",
    ):
        validate_replacement_usd_depth_sweep_request(request)

    request["cameras"] = [_camera()]
    request["camera_rows_digest"] = canonical_digest(
        {"cameras": request["cameras"]}
    )
    request["joint_state_cells"][1]["joint_positions"]["hinge"] = "invalid"
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    with pytest.raises(ArticulatedUsdDepthSweepError) as excinfo:
        validate_replacement_usd_depth_sweep_request(request)
    assert "replacement_depth_cell_joint_position_invalid:1:hinge" in excinfo.value.codes
    assert (
        "replacement_depth_cell_dependency_driver_invalid:1:coupler"
        in excinfo.value.codes
    )


def test_general_depth_rejects_graph_axis_that_disagrees_with_usd_joint_frame(
    tmp_path: Path,
) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "primitive_graph.usda")
    graph = _primitive_graph()
    graph["joints"][0]["axis"] = [0.0, 0.0, 1.0]
    identity = np.eye(4).tolist()
    request = seal_replacement_usd_depth_sweep_request(
        asset_id="fixture_axis_mismatch_asset",
        task_kind="articulated_interaction",
        task_freeze_digest="sha256:" + "3" * 64,
        replacement_usd_sha256=_sha256(usd),
        replacement_usd_size_bytes=usd.stat().st_size,
        articulation_graph=graph,
        link_paths={
            link_id: f"/Asset/links/{link_id}"
            for link_id in ("root", "door", "latch")
        },
        joint_paths={
            joint_id: f"/Asset/joints/{joint_id}"
            for joint_id in ("hinge", "coupler")
        },
        task_scoring_frame={
            "frame_id": "asset_root",
            "T_asset_task_scoring": identity,
        },
        camera_contract_digest="sha256:" + "4" * 64,
        cameras=[_camera()],
        joint_state_cells=[
            {
                "cell_id": "reset",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.0, "coupler": 0.0},
            },
            {
                "cell_id": "open",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.7, "coupler": 0.14},
            },
        ],
    )

    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="replacement_depth_joint_axis_binding_mismatch:hinge:parent",
    ):
        materialize_replacement_usd_depth_sweep(
            usd_path=usd,
            request=request,
            output_root=tmp_path / "axis_mismatch",
        )


def test_general_depth_joins_joint_axis_and_reset_frames_through_link_rest_poses(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "rotated_links.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    for link_id in ("base", "door"):
        link = UsdGeom.Xform.Define(stage, f"/Asset/links/{link_id}")
        link.AddRotateYOp().Set(-90.0)
        cube = UsdGeom.Cube.Define(
            stage, f"/Asset/links/{link_id}/geometry/cube"
        )
        cube.CreateSizeAttr(0.25)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/hinge")
    joint.CreateAxisAttr("X")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/links/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/links/door")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()
    graph = {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "base", "is_root": True, "semantic_role": "root"},
            {"link_id": "door", "is_root": False, "semantic_role": "target"},
        ],
        "joints": [
            {
                "joint_id": "hinge",
                "parent_link_id": "base",
                "child_link_id": "door",
                "joint_type": "revolute",
                "role": "target",
                "axis": [0.0, 0.0, 1.0],
                "limits": [0.0, 1.0],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 0.0,
                    "damping": 1.0,
                    "maximum_force": 10.0,
                },
                "dependency": None,
            }
        ],
        "collision_pairs": [
            {"link_a": "base", "link_b": "door", "collision_enabled": True}
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"hinge": [0.5, 0.8]},
        },
    }
    identity = np.eye(4).tolist()

    def request() -> dict:
        return seal_replacement_usd_depth_sweep_request(
            asset_id="rotated_rest_fixture",
            task_kind="articulated_interaction",
            task_freeze_digest="sha256:" + "7" * 64,
            replacement_usd_sha256=_sha256(usd),
            replacement_usd_size_bytes=usd.stat().st_size,
            articulation_graph=graph,
            link_paths={
                "base": "/Asset/links/base",
                "door": "/Asset/links/door",
            },
            joint_paths={"hinge": "/Asset/joints/hinge"},
            task_scoring_frame={
                "frame_id": "asset_root",
                "T_asset_task_scoring": identity,
            },
            camera_contract_digest="sha256:" + "8" * 64,
            cameras=[_camera()],
            joint_state_cells=[
                {
                    "cell_id": "reset",
                    "T_world_task_scoring": identity,
                    "joint_positions": {"hinge": 0.0},
                },
                {
                    "cell_id": "open",
                    "T_world_task_scoring": identity,
                    "joint_positions": {"hinge": 0.7},
                },
            ],
        )

    result = materialize_replacement_usd_depth_sweep(
        usd_path=usd,
        request=request(),
        output_root=tmp_path / "rotated_rest",
    )
    assert result["finite_depth_pixel_count_by_cell"] != [0, 0]

    stage = Usd.Stage.Open(str(usd))
    authored = UsdPhysics.Joint(stage.GetPrimAtPath("/Asset/joints/hinge"))
    authored.GetLocalPos1Attr().Set(Gf.Vec3f(0.1, 0.0, 0.0))
    stage.GetRootLayer().Save()
    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="replacement_depth_joint_reset_frame_mismatch:hinge",
    ):
        materialize_replacement_usd_depth_sweep(
            usd_path=usd,
            request=request(),
            output_root=tmp_path / "reset_frame_mismatch",
        )


def test_general_depth_joint_frames_honor_nonzero_reset_coordinate(
    tmp_path: Path,
) -> None:
    reset = 0.4
    usd = tmp_path / "nonzero_reset.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/Asset/links/base")
    base.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))
    UsdGeom.Cube.Define(stage, "/Asset/links/base/geometry/cube").CreateSizeAttr(
        0.2
    )
    door = UsdGeom.Xform.Define(stage, "/Asset/links/door")
    door.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))
    door.AddRotateXOp().Set(-math.degrees(reset))
    door_geometry = UsdGeom.Cube.Define(
        stage, "/Asset/links/door/geometry/cube"
    )
    door_geometry.CreateSizeAttr(0.2)
    door_geometry.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.35, 0.0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/joints/hinge")
    joint.CreateAxisAttr("X")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/links/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/links/door")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()
    graph = {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "base", "is_root": True, "semantic_role": "root"},
            {"link_id": "door", "is_root": False, "semantic_role": "target"},
        ],
        "joints": [
            {
                "joint_id": "hinge",
                "parent_link_id": "base",
                "child_link_id": "door",
                "joint_type": "revolute",
                "role": "target",
                "axis": [1.0, 0.0, 0.0],
                "limits": [0.0, 1.0],
                "reset_position": reset,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 0.0,
                    "damping": 1.0,
                    "maximum_force": 10.0,
                },
                "dependency": None,
            }
        ],
        "collision_pairs": [
            {"link_a": "base", "link_b": "door", "collision_enabled": True}
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"hinge": [0.6, 0.8]},
        },
    }
    identity = np.eye(4).tolist()
    request = seal_replacement_usd_depth_sweep_request(
        asset_id="nonzero_reset_fixture",
        task_kind="articulated_interaction",
        task_freeze_digest="sha256:" + "9" * 64,
        replacement_usd_sha256=_sha256(usd),
        replacement_usd_size_bytes=usd.stat().st_size,
        articulation_graph=graph,
        link_paths={"base": "/Asset/links/base", "door": "/Asset/links/door"},
        joint_paths={"hinge": "/Asset/joints/hinge"},
        task_scoring_frame={
            "frame_id": "asset_root",
            "T_asset_task_scoring": identity,
        },
        camera_contract_digest="sha256:" + "a" * 64,
        cameras=[_camera()],
        joint_state_cells=[
            {
                "cell_id": "reset",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": reset},
            },
            {
                "cell_id": "moved",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.7},
            },
        ],
    )

    result = materialize_replacement_usd_depth_sweep(
        usd_path=usd,
        request=request,
        output_root=tmp_path / "nonzero_reset_depth",
    )

    depth = np.load(tmp_path / "nonzero_reset_depth/replacement_depth_sweep.npy")
    assert result["finite_depth_pixel_count_by_cell"] != [0, 0]
    assert not np.array_equal(depth[0], depth[1])


def test_general_depth_materializer_rejects_digest_bound_usd_size_mismatch(
    tmp_path: Path,
) -> None:
    usd = _primitive_graph_fixture_usd(tmp_path / "primitive_graph.usda")
    graph = _primitive_graph()
    identity = np.eye(4).tolist()
    request = seal_replacement_usd_depth_sweep_request(
        asset_id="fixture_size_binding_asset",
        task_kind="articulated_interaction",
        task_freeze_digest="sha256:" + "1" * 64,
        replacement_usd_sha256=_sha256(usd),
        replacement_usd_size_bytes=usd.stat().st_size,
        articulation_graph=graph,
        link_paths={
            link_id: f"/Asset/links/{link_id}"
            for link_id in ("root", "door", "latch")
        },
        joint_paths={
            joint_id: f"/Asset/joints/{joint_id}"
            for joint_id in ("hinge", "coupler")
        },
        task_scoring_frame={
            "frame_id": "asset_root",
            "T_asset_task_scoring": identity,
        },
        camera_contract_digest="sha256:" + "2" * 64,
        cameras=[_camera()],
        joint_state_cells=[
            {
                "cell_id": "reset",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.0, "coupler": 0.0},
            },
            {
                "cell_id": "open",
                "T_world_task_scoring": identity,
                "joint_positions": {"hinge": 0.7, "coupler": 0.14},
            },
        ],
    )
    request["replacement_usd_size_bytes"] += 1
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="replacement_depth_usd_bytes_changed",
    ):
        materialize_replacement_usd_depth_sweep(
            usd_path=usd,
            request=request,
            output_root=tmp_path / "blocked",
        )


def test_source_alpha_coverage_is_conservative_and_scene_neutral() -> None:
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[1, 1] = 0.75
    pooled = conservative_max_pool_alpha(
        alpha, output_height=2, output_width=2
    )
    assert pooled.tolist() == [[0.75, 0.0], [0.0, 0.0]]

    depth = np.full((2, 2, 2), np.inf, dtype=np.float32)
    depth[0, 0, 0] = 1.0
    rows = evaluate_source_alpha_coverage(
        pooled[None],
        depth,
        cells=[
            {
                "camera_id": "840313_external",
                "commanded_door_angle_deg": 0.0,
                "readback_door_angle_deg": 0.0,
            },
            {
                "camera_id": "840313_external",
                "commanded_door_angle_deg": 45.0,
                "readback_door_angle_deg": 45.0,
            },
        ],
        camera_ids=["840313_external"],
        coverage_margin_pixels=0,
    )
    assert rows[0]["uncovered_significant_pixel_count"] == 0
    assert rows[1]["uncovered_significant_pixel_count"] == 1


def _zero_residue_source_layer_audit(depth_path: Path, *, residue: int = 0) -> Path:
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    rows = []
    for index, cell in enumerate(depth["cells"]):
        rows.append(
            {
                "cell_index": index,
                "camera_id": cell["camera_id"],
                "commanded_door_angle_deg": cell["commanded_door_angle_deg"],
                "readback_door_angle_deg": cell["readback_door_angle_deg"],
                "uncovered_significant_pixel_count": residue,
                "largest_uncovered_component_pixels": residue,
                "uncovered_alpha_sum": float(residue),
                "uncovered_alpha_fraction": float(residue),
            }
        )
    audit = {
        "schema_version": "adp009b_source_layer_replacement_coverage_audit.v1",
        "status": "source_layer_coverage_measured",
        "source_layer_splat_digest": "sha256:" + "4" * 64,
        "depth_sweep_manifest": {
            "sha256": _sha256(depth_path),
            "manifest_digest": depth["manifest_digest"],
        },
        "camera_ids": ["external"],
        "significant_alpha_threshold": 1.0 / 255.0,
        "coverage_margin_pixels": 1,
        "cells": rows,
        "manifest_digest": "",
    }
    audit["manifest_digest"] = canonical_digest(audit, digest_field="manifest_digest")
    path = depth_path.parent / f"source-layer-audit-{residue}.json"
    path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_deleted_source_layer_coverage_requires_zero_residue(
    tmp_path: Path,
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=0.5,
    )
    depth_path = depth_root / "adp009b_articulated_usd_depth_sweep.v1.json"
    receipt = materialize_deleted_source_layer_replacement_coverage_qualification(
        source_layer_coverage_audit_path=_zero_residue_source_layer_audit(depth_path),
        depth_sweep_manifest_path=depth_path,
        output_path=tmp_path / "qualified-coverage.json",
    )

    assert receipt["coverage_qualified"] is True
    assert receipt["coverage_scope"] == "deleted_source_layer"
    assert receipt["all_deleted_source_contribution_occluded"] is True
    assert receipt["cells"][0]["residual_significant_pixels"] == 0

    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="source_layer_coverage_qualification_source_residue_observed",
    ):
        materialize_deleted_source_layer_replacement_coverage_qualification(
            source_layer_coverage_audit_path=_zero_residue_source_layer_audit(
                depth_path, residue=1
            ),
            depth_sweep_manifest_path=depth_path,
            output_path=tmp_path / "blocked-coverage.json",
        )


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_source_layer_coverage_audit_binds_render_pair_and_depth(
    tmp_path: Path, scene_id: str
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    depth = materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=0.5,
    )
    assert depth["actual_mesh_depth_rasterized"] is True

    alpha = np.zeros((48, 64), dtype=np.float32)
    alpha[15:30, 20:40] = 0.8
    foreground = np.zeros((48, 64, 3), dtype=np.float32)
    foreground[..., 1] = 120.0
    black = np.clip(foreground * alpha[..., None], 0, 255).astype(np.uint8)
    white = np.clip(
        foreground * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
        0,
        255,
    ).astype(np.uint8)
    black_manifest = _render_manifest(
        tmp_path / "black",
        background="#000000",
        image=black,
        scene_id=scene_id,
    )
    white_manifest = _render_manifest(
        tmp_path / "white",
        background="#ffffff",
        image=white,
        scene_id=scene_id,
    )
    receipt = materialize_source_layer_replacement_coverage_audit(
        black_render_manifest_path=black_manifest,
        white_render_manifest_path=white_manifest,
        depth_sweep_manifest_path=depth_root
        / "adp009b_articulated_usd_depth_sweep.v1.json",
        output_root=tmp_path / "audit",
        coverage_margin_pixels=0,
    )
    assert receipt["status"] == "source_layer_coverage_measured"
    assert receipt["summary"]["cell_count"] == 1
    assert receipt["coverage_qualified"] is False
    assert (tmp_path / "audit/source_alpha_by_camera.npy").is_file()
    assert len(receipt["review_contact_sheets"]) == 1
    assert (tmp_path / "audit/review_contact_sheets/external.png").is_file()
    assert len(receipt["uncovered_source_support_masks"]) == 1
    assert (
        tmp_path / "audit/uncovered_source_support_masks/external.png"
    ).is_file()
    assert receipt["uncovered_source_support_masks_are_inpainting_authority"] is False


def test_full_resolution_calibrated_residual_masks_can_bound_future_inpainting(
    tmp_path: Path,
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=1.0,
    )
    alpha = np.zeros((48, 64), dtype=np.float32)
    alpha[15:30, 20:40] = 0.8
    foreground = np.zeros((48, 64, 3), dtype=np.float32)
    foreground[..., 1] = 120.0
    black = np.clip(foreground * alpha[..., None], 0, 255).astype(np.uint8)
    white = np.clip(
        foreground * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
        0,
        255,
    ).astype(np.uint8)
    black_manifest = _render_manifest(
        tmp_path / "black",
        background="#000000",
        image=black,
        scene_id="840920",
        inpainting_mask_input_authorized=True,
    )
    white_manifest = _render_manifest(
        tmp_path / "white",
        background="#ffffff",
        image=white,
        scene_id="840920",
        inpainting_mask_input_authorized=True,
    )

    receipt = materialize_source_layer_replacement_coverage_audit(
        black_render_manifest_path=black_manifest,
        white_render_manifest_path=white_manifest,
        depth_sweep_manifest_path=depth_root
        / "adp009b_articulated_usd_depth_sweep.v1.json",
        output_root=tmp_path / "audit",
        coverage_margin_pixels=1,
    )

    assert receipt["uncovered_source_support_masks_are_inpainting_authority"] is True
    assert receipt["inpainting_mask_eligibility"] == {
        "full_resolution_source_frames": True,
        "full_resolution_replacement_depth": True,
        "calibrated_method_input_pair": True,
        "authorizes_only": "future_exact_mask_contained_multi_view_edit_input",
        "inpainting_result_qualified": False,
    }


def test_source_coverage_binds_co_present_depth_composition_identity(
    tmp_path: Path,
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    depth = materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=1.0,
    )
    composition_root = tmp_path / "composition"
    composition_root.mkdir()
    depth_array = np.load(depth_root / str(depth["arrays"]["relative_path"]), allow_pickle=False)
    composition_array = composition_root / "replacement_depth_composition.npy"
    np.save(composition_array, depth_array, allow_pickle=False)
    composition: dict[str, object] = {
        "schema_version": "public_scene_replacement_depth_composition.v1",
        "status": "co_present_replacement_depth_rasterized",
        "task_id": "fixture_task",
        "task_freeze_digest": "sha256:" + "1" * 64,
        "scored_task_asset_id": "fixture_asset_a",
        "replacement_asset_ids": ["fixture_asset_a", "fixture_asset_b"],
        "cells": depth["cells"],
        "arrays": {
            "relative_path": composition_array.name,
            "size_bytes": composition_array.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(composition_array.read_bytes()).hexdigest(),
        },
        "resolution_scale": 1.0,
        "actual_usd_geometry_depth_rasterized": True,
        "actual_composed_depth_rasterized": True,
        "caller_supplied_coverage_mask": False,
        "receipt_digest": "",
    }
    composition["receipt_digest"] = canonical_digest(
        composition, digest_field="receipt_digest"
    )
    composition_path = composition_root / "composition.json"
    composition_path.write_text(json.dumps(composition), encoding="utf-8")
    image = np.full((48, 64, 3), 96, dtype=np.uint8)
    black_manifest = _render_manifest(
        tmp_path / "black",
        background="#000000",
        image=image,
        scene_id="fixture",
        inpainting_mask_input_authorized=True,
    )
    white_manifest = _render_manifest(
        tmp_path / "white",
        background="#ffffff",
        image=image,
        scene_id="fixture",
        inpainting_mask_input_authorized=True,
    )

    receipt = materialize_source_layer_replacement_coverage_audit(
        black_render_manifest_path=black_manifest,
        white_render_manifest_path=white_manifest,
        depth_sweep_manifest_path=composition_path,
        output_root=tmp_path / "audit",
        coverage_margin_pixels=1,
    )

    assert receipt["task_id"] == "fixture_task"
    assert receipt["task_freeze_digest"] == composition["task_freeze_digest"]
    assert receipt["co_present_replacement_asset_ids"] == [
        "fixture_asset_a",
        "fixture_asset_b",
    ]
    assert receipt["replacement_depth_composition"]["receipt_digest"] == composition[
        "receipt_digest"
    ]


def test_full_resolution_review_only_residual_masks_cannot_authorize_inpainting(
    tmp_path: Path,
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=1.0,
    )
    alpha = np.zeros((48, 64), dtype=np.float32)
    alpha[15:30, 20:40] = 0.8
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    white = np.full((48, 64, 3), 255, dtype=np.uint8)
    black[alpha > 0.0, 1] = 96
    white[alpha > 0.0, 1] = 96
    black_manifest = _render_manifest(
        tmp_path / "black", background="#000000", image=black, scene_id="840920"
    )
    white_manifest = _render_manifest(
        tmp_path / "white", background="#ffffff", image=white, scene_id="840920"
    )

    receipt = materialize_source_layer_replacement_coverage_audit(
        black_render_manifest_path=black_manifest,
        white_render_manifest_path=white_manifest,
        depth_sweep_manifest_path=depth_root
        / "adp009b_articulated_usd_depth_sweep.v1.json",
        output_root=tmp_path / "audit",
        coverage_margin_pixels=1,
    )

    assert receipt["uncovered_source_support_masks_are_inpainting_authority"] is False
    assert receipt["inpainting_mask_eligibility"]["full_resolution_source_frames"] is True
    assert receipt["inpainting_mask_eligibility"]["calibrated_method_input_pair"] is False


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_reference_hybrid_review_changes_only_actual_usd_silhouette(
    tmp_path: Path, scene_id: str
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=0.5,
    )
    scene = np.full((48, 64, 3), [20, 40, 80], dtype=np.uint8)
    scene_manifest = _render_manifest(
        tmp_path / "scene",
        background="#0b0b10",
        image=scene,
        scene_id=scene_id,
    )
    receipt = materialize_reference_hybrid_review(
        retained_scene_render_manifest_path=scene_manifest,
        depth_sweep_manifest_path=depth_root
        / "adp009b_articulated_usd_depth_sweep.v1.json",
        output_root=tmp_path / "hybrid",
        replacement_rgb=(180, 190, 200),
    )

    depth = np.load(depth_root / "replacement_depth_sweep.npy")[0]
    finite = np.isfinite(depth) & (depth > 0.0)
    expected_scene = cv2.resize(scene, (32, 24), interpolation=cv2.INTER_AREA)
    rendered = cv2.imread(
        str(tmp_path / "hybrid/frames/external__door_000p000.png")
    )
    assert rendered is not None
    assert np.array_equal(rendered[~finite], expected_scene[~finite])
    assert np.any(rendered[finite] != expected_scene[finite])
    assert receipt["actual_usd_geometry_silhouette_used"] is True
    assert receipt["usd_materials_rendered"] is False
    assert receipt["native_isaac_or_rtx_render"] is False
    assert receipt["cell_count"] == 1
    assert len(receipt["contact_sheets"]) == 1
    assert receipt["manifest_digest"].startswith("sha256:")


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_target_core_coverage_uses_actual_usd_depth_for_both_fixtures(
    tmp_path: Path, scene_id: str
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=1.0,
    )
    depth = np.load(depth_root / "replacement_depth_sweep.npy")[0]
    target = (np.isfinite(depth) & (depth > 0.0)).astype(np.uint8) * 255
    # One measured fringe pixel remains outside the replacement silhouette.
    target[0, 0] = 255
    mask = tmp_path / f"{scene_id}.target_core.png"
    assert cv2.imwrite(str(mask), target)

    receipt = materialize_target_core_replacement_coverage_audit(
        target_core_mask_paths={"external": mask},
        depth_sweep_manifest_path=depth_root
        / "adp009b_articulated_usd_depth_sweep.v1.json",
        output_root=tmp_path / "coverage",
        maximum_uncovered_fraction=0.25,
    )

    assert receipt["schema_version"] == "articulated_excision_coverage.v1"
    assert receipt["coverage_qualified"] is True
    assert receipt["cells"][0]["residual_significant_pixels"] == 1
    assert receipt["cells"][0]["outside_mask_changed_pixels"] == 0
    assert receipt["residual_is_narrow_seam_candidate_not_inpainting_success"] is True
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert (tmp_path / "coverage/residual_target_core_seam_masks/external.png").is_file()


def test_target_core_coverage_fails_closed_on_camera_mask_mismatch(
    tmp_path: Path,
) -> None:
    usd = _fixture_usd(tmp_path / "fixture.usda")
    depth_root = tmp_path / "depth"
    materialize_articulated_usd_depth_sweep(
        usd_path=usd,
        cameras=[_camera()],
        door_angles_deg=[0.0],
        moving_link_path="/Asset/door",
        hinge_origin_asset_m=[0.0, 0.0, 0.0],
        hinge_axis_asset=[0.0, 0.0, 1.0],
        T_world_asset=np.eye(4).tolist(),
        output_root=depth_root,
        resolution_scale=1.0,
    )

    with pytest.raises(
        ArticulatedUsdDepthSweepError,
        match="target_core_coverage_camera_masks_mismatch",
    ):
        materialize_target_core_replacement_coverage_audit(
            target_core_mask_paths={},
            depth_sweep_manifest_path=depth_root
            / "adp009b_articulated_usd_depth_sweep.v1.json",
            output_root=tmp_path / "coverage",
            maximum_uncovered_fraction=0.05,
        )
