from __future__ import annotations

import hashlib

import numpy as np

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.splat_bbox_target_binding import (
    SplatBBoxTargetBindingError,
    bind_splat_bbox_target,
)


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _scene(tmp_path):
    grid = np.linspace(-0.12, 0.12, 12)
    front = np.asarray([(x, y, 2.0) for x in grid for y in grid], dtype=np.float32)
    back = np.asarray([(x, y, 5.0) for x in grid for y in grid], dtype=np.float32)
    xyz = np.concatenate([front, back], axis=0)
    count = len(xyz)
    splat = SplatData(
        count=count,
        xyz=xyz,
        opacity=np.full(count, 8.0, dtype=np.float32),
        f_dc=np.zeros((count, 3), dtype=np.float32),
        scales=np.zeros((count, 3), dtype=np.float32),
        quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        properties=(),
    )
    return write_standard_3dgs_ply(splat, tmp_path / "analysis.ply")


def _request(path):
    return {
        "schema_version": "splat_bbox_target_binding_request.v1",
        "source_scene_digest": "sha256:" + "a" * 64,
        "analysis_splat_digest": _digest(path),
        "camera_spec_digest": "sha256:" + "b" * 64,
        "rgb_digest": "sha256:" + "c" * 64,
        "view_id": "center",
        "image_size": {"width": 640, "height": 480},
        "bbox_xyxy_pixels": [260, 180, 380, 300],
        "camera": {
            "pos": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, 1.0],
            "up": [0.0, 1.0, 0.0],
            "fov": 60.0,
        },
        "minimum_opacity": 0.18,
        "front_depth_fraction": 0.25,
        "minimum_projected_splats": 32,
        "binding_may_self_authorize": False,
    }


def test_bbox_binding_selects_front_surface_and_keeps_claim_boundary(tmp_path) -> None:
    scene = _scene(tmp_path)
    result = bind_splat_bbox_target(analysis_splat_path=scene, request=_request(scene))

    assert result["status"] == "candidate_bound"
    np.testing.assert_allclose(result["position_scene"], [0.0, 0.0, 2.0], atol=0.03)
    assert result["projected_splat_count"] == 288
    assert result["metric_scale_proven"] is False
    assert result["collision_support_proven"] is False
    assert result["binding_may_self_authorize"] is False


def test_bbox_binding_rejects_wrong_splat_digest(tmp_path) -> None:
    scene = _scene(tmp_path)
    request = _request(scene)
    request["analysis_splat_digest"] = "sha256:" + "d" * 64

    try:
        bind_splat_bbox_target(analysis_splat_path=scene, request=request)
    except SplatBBoxTargetBindingError as exc:
        assert "bbox_binding_splat_digest_mismatch" in exc.codes
    else:
        raise AssertionError("digest mismatch must fail closed")
