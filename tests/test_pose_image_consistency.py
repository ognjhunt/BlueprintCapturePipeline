from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from blueprint_pipeline.pose_image_consistency import check_two_view_epipolar_consistency


WIDTH, HEIGHT = 320, 240
FX = FY = 260.0
CX, CY = WIDTH / 2.0, HEIGHT / 2.0
PLANE_DEPTH = 2.5


def _texture(rng: np.random.Generator) -> np.ndarray:
    blocks = rng.integers(0, 256, size=(HEIGHT // 2, WIDTH // 2)).astype(np.float64)
    return np.kron(blocks, np.ones((8, 8)))


def _render_plane(texture: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """Render a textured z=PLANE_DEPTH world plane through an OpenCV pinhole camera.

    The plane is supersampled 3x and box-downsampled so edges are antialiased
    like natural photographs, which the matcher's thresholds were tuned on.
    """

    supersample = 3
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    ys, xs = np.mgrid[0 : HEIGHT * supersample, 0 : WIDTH * supersample]
    xs = (xs + 0.5) / supersample - 0.5
    ys = (ys + 0.5) / supersample - 0.5
    directions_camera = np.stack(
        [(xs - CX) / FX, (ys - CY) / FY, np.ones_like(xs, dtype=np.float64)], axis=-1
    )
    directions_world = directions_camera @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (PLANE_DEPTH - translation[2]) / directions_world[..., 2]
    points = translation + scale[..., None] * directions_world
    texture_x = ((points[..., 0] + 1.6) / 3.2) * (texture.shape[1] - 2)
    texture_y = ((points[..., 1] + 1.2) / 2.4) * (texture.shape[0] - 2)
    valid = (
        (scale > 0)
        & (texture_x >= 0)
        & (texture_x < texture.shape[1] - 1)
        & (texture_y >= 0)
        & (texture_y < texture.shape[0] - 1)
    )
    tx0 = np.clip(texture_x.astype(int), 0, texture.shape[1] - 2)
    ty0 = np.clip(texture_y.astype(int), 0, texture.shape[0] - 2)
    fx = texture_x - tx0
    fy = texture_y - ty0
    sampled = (
        texture[ty0, tx0] * (1 - fx) * (1 - fy)
        + texture[ty0, tx0 + 1] * fx * (1 - fy)
        + texture[ty0 + 1, tx0] * (1 - fx) * fy
        + texture[ty0 + 1, tx0 + 1] * fx * fy
    )
    rendered = np.where(valid, sampled, 127.0)
    return rendered.reshape(HEIGHT, supersample, WIDTH, supersample).mean(axis=(1, 3))


def _observation(frame_id: str, camera_to_world: np.ndarray, relative: str) -> dict:
    return {
        "observation_id": frame_id,
        "image_relative_path": relative,
        "camera": {
            "T_world_camera": camera_to_world.tolist(),
            "rgb_intrinsics": {
                "width": WIDTH,
                "height": HEIGHT,
                "fx": FX,
                "fy": FY,
                "cx": CX,
                "cy": CY,
            },
        },
    }


def _yaw(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _write_pair(tmp_path: Path, *, flip_convention: bool) -> list[dict]:
    rng = np.random.default_rng(11)
    texture = _texture(rng)
    poses = []
    for index in range(3):
        camera_to_world = np.eye(4)
        # Distinct per-frame yaw makes the relative rotation non-identity, so a
        # convention flip cannot cancel out of the fundamental matrix.
        camera_to_world[:3, :3] = _yaw(1.6 * index - 1.6)
        camera_to_world[0, 3] = 0.022 * index
        camera_to_world[1, 3] = 0.006 * index
        poses.append(camera_to_world)
    observations = []
    for index, pose in enumerate(poses):
        image = _render_plane(texture, pose)
        relative = f"images/frame_{index:05d}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)
        declared = pose.copy()
        if flip_convention:
            declared[:3, :3] = declared[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        observations.append(_observation(f"frame_{index:05d}", declared, relative))
    return observations


def test_consistent_poses_pass_and_convention_flip_fails(tmp_path: Path) -> None:
    good = _write_pair(tmp_path / "good", flip_convention=False)
    report = check_two_view_epipolar_consistency(
        observations=good, image_root=tmp_path / "good"
    )
    assert report["status"] == "consistent", json.dumps(report, indent=2)
    assert report["aggregate_median_epipolar_px"] <= 1.5

    flipped = _write_pair(tmp_path / "flipped", flip_convention=True)
    report_flipped = check_two_view_epipolar_consistency(
        observations=flipped, image_root=tmp_path / "flipped"
    )
    assert report_flipped["status"] == "inconsistent", json.dumps(report_flipped, indent=2)


def test_untextured_or_tiny_input_is_inconclusive(tmp_path: Path) -> None:
    observations = []
    for index in range(3):
        relative = f"images/frame_{index:05d}.png"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (WIDTH, HEIGHT), 127).save(path)
        camera_to_world = np.eye(4)
        camera_to_world[0, 3] = 0.012 * index
        observations.append(_observation(f"frame_{index:05d}", camera_to_world, relative))
    report = check_two_view_epipolar_consistency(
        observations=observations, image_root=tmp_path
    )
    assert report["status"] == "inconclusive"

    report_single = check_two_view_epipolar_consistency(
        observations=observations[:1], image_root=tmp_path
    )
    assert report_single["status"] == "inconclusive"
