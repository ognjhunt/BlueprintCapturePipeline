from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from blueprint_pipeline.articulated_usd_depth_sweep import (
    ArticulatedUsdDepthSweepError,
    conservative_max_pool_alpha,
    evaluate_source_alpha_coverage,
    load_articulated_usd_triangles,
    materialize_articulated_usd_depth_sweep,
    materialize_reference_hybrid_review,
    materialize_source_layer_replacement_coverage_audit,
    rasterize_triangle_depth,
    rotate_triangles_about_axis,
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
    root: Path, *, background: str, image: np.ndarray, scene_id: str
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
