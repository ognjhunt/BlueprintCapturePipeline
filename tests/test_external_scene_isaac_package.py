from __future__ import annotations

import hashlib

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.external_scene_collision_candidate import (
    compile_external_scene_collision_candidate,
)
from blueprint_pipeline.external_scene_isaac_package import (
    compile_external_scene_isaac_package,
)
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_packages_registered_particlefield_and_static_collision_without_video(
    tmp_path,
) -> None:
    count = 32
    splat = SplatData(
        count=count,
        xyz=np.column_stack(
            [np.linspace(-0.2, 0.2, count), np.zeros(count), np.ones(count)]
        ).astype(np.float32),
        opacity=np.full(count, 8.0, dtype=np.float32),
        f_dc=np.zeros((count, 3), dtype=np.float32),
        scales=np.zeros((count, 3), dtype=np.float32),
        quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        properties=(),
    )
    ply = write_standard_3dgs_ply(splat, tmp_path / "appearance.ply")
    glb = tmp_path / "scene.glb"
    glb.write_bytes(trimesh.creation.box().export(file_type="glb"))
    collision = tmp_path / "collision.usda"
    collision_result = compile_external_scene_collision_candidate(
        source_path=glb,
        request={
            "schema_version": "external_scene_collision_compilation_request.v1",
            "source_asset_digest": _digest(glb),
            "source_format": "glb",
            "source_coordinate_frame": {"up_axis": "Y", "handedness": "right"},
            "metric_scale_status": "unverified",
            "source_video_available": False,
            "generated_fill_allowed": False,
            "collision_validated": False,
        },
        output_path=collision,
    )
    package = tmp_path / "scene.usdz"
    result = compile_external_scene_isaac_package(
        analysis_splat_path=ply,
        collision_usd_path=collision,
        output_path=package,
        request={
            "schema_version": "external_scene_isaac_package_request.v1",
            "appearance_scene_digest": "sha256:" + "a" * 64,
            "analysis_splat_digest": _digest(ply),
            "collision_candidate_digest": collision_result["collision_candidate_digest"],
            "collision_asset_digest": _digest(collision),
            "scene_frame_binding_digest": "sha256:" + "b" * 64,
            "source_to_collision_stage_matrix": [
                1,
                0,
                0,
                2,
                0,
                1,
                0,
                3,
                0,
                0,
                1,
                4,
                0,
                0,
                0,
                1,
            ],
            "metric_scale_status": "unverified",
            "collision_validated": False,
            "source_video_available": False,
            "generated_fill_allowed": False,
            "maximum_nonfinite_splat_fraction": 0.001,
        },
    )

    assert result["status"] == "candidate_packaged"
    assert result["source_video_required_for_candidate_packaging"] is False
    assert result["packaged_splat_count"] == count
    stage = Usd.Stage.Open(str(package))
    gaussian = stage.GetPrimAtPath("/World/BlueprintReconstruction/Appearance/Gaussians")
    collider = stage.GetPrimAtPath("/World/BlueprintReconstruction/Collision/ExternalSceneMesh")
    assert gaussian.GetTypeName() == "ParticleField3DGaussianSplat"
    assert collider.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(collider).ComputeVisibility() == UsdGeom.Tokens.invisible
    assert result["collision_geometry_render_hidden"] is True
    xform = UsdGeom.Xformable(
        stage.GetPrimAtPath("/World/BlueprintReconstruction/Appearance")
    ).GetLocalTransformation()
    assert xform.Transform(Gf.Vec3d(0, 0, 0)) == Gf.Vec3d(2, 3, 4)
