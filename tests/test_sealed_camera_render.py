from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.sealed_camera_render import (
    SealedCameraRenderError,
    render_splat_at_exact_cameras,
    transform_camera_into_provider_frame,
)


DIGEST = "sha256:" + "a" * 64


def _write_standard_3dgs_ply(path: Path, rows: list[tuple[float, float, float, float, float, float]]) -> None:
    properties = [
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rows)}\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    body = b""
    for x, y, z, r, g, b in rows:
        body += struct.pack(
            "<14f", x, y, z, r, g, b, 8.0, -3.4, -3.4, -3.4, 1.0, 0.0, 0.0, 0.0
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + body)


def test_transform_camera_into_provider_frame_inverts_alignment() -> None:
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale, translation = 2.0, np.array([1.0, -2.0, 0.5])
    alignment = {
        "estimated_scale_factor": scale,
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
    }
    candidate_pose = np.eye(4)
    candidate_pose[:3, 3] = [0.4, 0.7, 1.9]
    provider_pose = np.asarray(
        transform_camera_into_provider_frame(
            camera_to_world_candidate=candidate_pose.tolist(), alignment=alignment
        )
    )
    # Mapping the provider-frame camera center forward must recover the
    # candidate-frame center: x_c = s R x_p + t.
    recovered = scale * rotation @ provider_pose[:3, 3] + translation
    assert recovered == pytest.approx(candidate_pose[:3, 3], abs=1e-12)
    assert provider_pose[:3, :3] == pytest.approx(rotation.T @ candidate_pose[:3, :3])

    with pytest.raises(SealedCameraRenderError, match="pose_invalid"):
        transform_camera_into_provider_frame(
            camera_to_world_candidate=[[float("nan")] * 4] * 4, alignment=alignment
        )


@pytest.mark.parametrize("background_rgb", [-1, 0x1000000, True, "black"])
def test_exact_camera_render_rejects_unbound_background_rgb(
    tmp_path: Path, background_rgb: object
) -> None:
    splat = tmp_path / "scene.ply"
    _write_standard_3dgs_ply(splat, [(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)])

    with pytest.raises(SealedCameraRenderError) as exc:
        render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=[
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 32.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ],
            output_dir=tmp_path / "render",
            provider_splat_import_receipt_digest=DIGEST,
            alignment_digest=DIGEST,
            camera_set_label="fixture",
            background_rgb=background_rgb,  # type: ignore[arg-type]
        )

    assert exc.value.codes == ("render_background_rgb_invalid",)


@pytest.mark.slow
def test_exact_camera_render_places_known_gaussians_at_predicted_pixels(tmp_path: Path) -> None:
    splat = tmp_path / "scene.ply"
    bright = 1.77
    _write_standard_3dgs_ply(
        splat,
        [
            (0.0, 0.0, 2.0, bright, -1.0, -1.0),   # red at image center
            (0.4, 0.0, 2.0, -1.0, bright, -1.0),   # green right of center
            (0.0, 0.3, 2.0, -1.0, -1.0, bright),   # blue below center (OpenCV +y down)
        ],
    )
    fx = fy = 100.0
    cx, cy, width, height = 32.0, 24.0, 64, 48
    cameras = [
        {
            "camera_id": "sealed_check",
            "T_world_camera_provider_frame": np.eye(4).tolist(),
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "near": 0.05,
                "far": 100.0,
            },
        }
    ]
    manifest = render_splat_at_exact_cameras(
        splat_path=splat,
        cameras=cameras,
        output_dir=tmp_path / "render",
        provider_splat_import_receipt_digest=DIGEST,
        alignment_digest=DIGEST,
        camera_set_label="fixture_exact_check",
        background_rgb=0x102030,
    )
    assert manifest["status"] == "rendered_exact_cameras"
    assert manifest["render_count"] == 1
    assert manifest["renderer_identity"]["background_rgb"] == "#102030"
    frame = np.asarray(
        Image.open(tmp_path / "render" / manifest["renders"][0]["relative_path"]).convert("RGB")
    ).astype(np.float64)

    def peak_channel(u: float, v: float) -> int:
        patch = frame[int(v) - 4 : int(v) + 5, int(u) - 4 : int(u) + 5]
        return int(np.argmax(patch.reshape(-1, 3).max(axis=0)))

    # OpenCV projection: u = fx*x/z + cx, v = fy*y/z + cy.
    assert peak_channel(cx, cy) == 0  # red at center
    assert peak_channel(fx * 0.2 + cx, cy) == 1  # green at (52, 24)
    assert peak_channel(cx, fy * 0.15 + cy) == 2  # blue at (24, 39)
    background = frame[2:8, 2:8]
    assert float(background.max()) < 60.0
