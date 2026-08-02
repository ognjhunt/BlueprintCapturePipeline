from __future__ import annotations

import hashlib

import numpy as np
import trimesh

from blueprint_pipeline.external_scene_frame_registration import (
    bind_same_source_splat_collision_frames,
    compose_registered_target_binding,
    register_external_scene_frames,
    transform_camera_specs_to_collision_stage,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _assets(tmp_path):
    first = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    first.apply_scale([2.3, 0.7, 1.1])
    first.apply_translation([1.7, -0.4, 0.8])
    second = trimesh.creation.icosphere(subdivisions=3, radius=0.55)
    second.apply_scale([0.6, 1.8, 0.9])
    second.apply_translation([-2.2, 1.3, -0.6])
    mesh = trimesh.util.concatenate([first, second])
    glb = tmp_path / "collision.glb"
    glb.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    collision_points = np.asarray(mesh.vertices, dtype=np.float64)
    rotation = np.diag([1.0, -1.0, -1.0])
    scale = 0.96
    translation = np.asarray([-0.3, 0.2, 1.1])
    appearance = ((collision_points - translation) / scale) @ rotation
    count = len(appearance)
    splat = SplatData(
        count=count,
        xyz=appearance.astype(np.float32),
        opacity=np.full(count, 8.0, dtype=np.float32),
        f_dc=np.zeros((count, 3), dtype=np.float32),
        scales=np.zeros((count, 3), dtype=np.float32),
        quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        properties=(),
    )
    ply = write_standard_3dgs_ply(splat, tmp_path / "appearance.ply")
    return ply, glb


def test_registers_right_handed_axis_convention_and_composes_target(tmp_path) -> None:
    ply, glb = _assets(tmp_path)
    request = {
        "schema_version": "external_scene_frame_registration_request.v1",
        "appearance_scene_digest": "sha256:" + "a" * 64,
        "analysis_splat_digest": _digest(ply),
        "collision_source_digest": _digest(glb),
        "appearance_up_axis": "Y",
        "collision_source_up_axis": "Y",
        "minimum_opacity": 0.3,
        "trim_fraction": 0.8,
        "sample_cap_per_asset": 10000,
        "maximum_trimmed_rmse_scene_units": 0.05,
        "minimum_runner_up_ratio": 1.1,
        "minimum_scale_ratio": 0.8,
        "maximum_scale_ratio": 1.2,
        "candidate_may_self_qualify": False,
    }
    result = register_external_scene_frames(
        analysis_splat_path=ply,
        collision_glb_path=glb,
        request=request,
    )

    assert result["status"] == "candidate_registered"
    assert result["selected_axis_rotation"] == [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    assert abs(result["estimated_scale_ratio"] - 0.96) < 1e-5
    assert result["metric_scale_proven"] is False
    target = {
        "status": "candidate_bound",
        "source_scene_digest": request["appearance_scene_digest"],
        "binding_evidence_digest": "sha256:" + "b" * 64,
        "position_scene": [0.0, 0.0, 0.0],
        "spatial_uncertainty_scene_units": 0.2,
    }
    composed = compose_registered_target_binding(
        target_binding=target,
        frame_registration=result,
    )
    np.testing.assert_allclose(
        composed["position_collision_stage"],
        [-0.3, -1.1, 0.2],
        atol=1e-5,
    )
    assert composed["metric_scale_proven"] is False

    cameras = transform_camera_specs_to_collision_stage(
        cameras=[
            {
                "id": "task_focus",
                "spec": {
                    "pos": [0.0, 0.0, 0.0],
                    "target": [1.0, 0.0, 0.0],
                    "fov": 52,
                    "up": [0.0, -1.0, 0.0],
                },
            }
        ],
        frame_registration=result,
    )
    assert cameras[0]["id"] == "task_focus"
    assert cameras[0]["spec"]["fov"] == 52.0
    assert len(cameras[0]["spec"]["pos"]) == 3
    assert abs(np.linalg.norm(cameras[0]["spec"]["up"]) - 1.0) < 1e-9


def test_binds_same_source_splat_transform_collision_without_icp() -> None:
    source_digest = "sha256:" + "a" * 64
    collision_digest = "sha256:" + "b" * 64
    generation = {
        "schema_version": "splat_transform_collision_candidate.v1",
        "status": "candidate_generated",
        "source_asset_digest": source_digest,
        "actions": {
            "coordinate_transform_applied": False,
            "global_decimation_applied": False,
        },
        "source_coordinate_frame": {"up_axis": "Y", "handedness": "right"},
        "output_coordinate_frame": {
            "up_axis": "Y",
            "handedness": "right",
            "basis": "source_preserved",
        },
        "artifacts": {"collision_glb": {"digest": collision_digest, "bytes": 100}},
    }
    generation["candidate_digest"] = canonical_digest(generation, digest_field="candidate_digest")
    candidate = {
        "schema_version": "external_scene_collision_candidate.v1",
        "status": "candidate_compiled",
        "source_asset_digest": collision_digest,
    }
    candidate["collision_candidate_digest"] = canonical_digest(
        candidate, digest_field="collision_candidate_digest"
    )

    result = bind_same_source_splat_collision_frames(
        source_scene_digest=source_digest,
        analysis_splat_digest="sha256:" + "c" * 64,
        collision_generation=generation,
        collision_candidate=candidate,
    )

    assert result["status"] == "candidate_registered"
    assert result["estimated_scale_ratio"] == 1.0
    assert result["source_to_collision_stage_matrix"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["metric_scale_proven"] is False
