"""Two Atlas worlds of one room, aligned by the depth of their shared views: known truth.

A textured room corner is ray-cast for three cameras. The source pose sees each
view whole; the background pose sees the same physical cameras through a 1.6x
centre crop (an editor's earlier crop), in another Atlas world a known
similarity away, and one of its views was edited where an object was removed.
No provider call.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.atlas_shared_view_alignment import shared_view_alignment
from tests.test_atlas_source_geometry import quaternion_xyzw, rotation

SIZE = (360, 480)  # width, height: portrait like a phone walkthrough
SOURCE_K = {"width": SIZE[0], "height": SIZE[1], "fx": 300.0, "fy": 300.0, "cx": 180.0, "cy": 240.0}
CROP = 1.6
WORLD_SCALE = 1.7
WORLD_ROTATION = rotation(math.radians(25.0), math.radians(-4.0))
WORLD_TRANSLATION = np.array([0.5, -0.3, 1.2])
# OpenCV camera-to-world truth near the room's centre; an identity camera looks along +Z.
CAMERAS = [(rotation(0.0, 0.0), np.array([0.0, 0.2, 0.0])),
           (rotation(0.35, -0.1), np.array([0.6, 0.1, 0.3])),
           (rotation(-0.3, 0.12), np.array([-0.5, 0.3, -0.2]))]
# Room planes (normal, offset) with n . x = offset: floor, far wall and two side walls.
ROOM = [(np.array([0.0, 1.0, 0.0]), -1.2), (np.array([0.0, 0.0, 1.0]), 3.5),
        (np.array([1.0, 0.0, 0.0]), -2.4), (np.array([1.0, 0.0, 0.0]), 2.6)]


def _texture(seed):
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.random((512, 512, 3)).astype(np.float32), (0, 0), 2.5)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (noise * 255).astype(np.uint8)


TEXTURES = [_texture(seed) for seed in range(len(ROOM))]


def render(rotation_rub, position, k, planes=ROOM):
    """RGB and z-depth of the room for one RUB camera; colour belongs to the world point."""
    u = (np.arange(k["width"]) + 0.5 - k["cx"]) / k["fx"]
    v = (np.arange(k["height"]) + 0.5 - k["cy"]) / k["fy"]
    local = np.stack(np.broadcast_arrays(u[None, :], -v[:, None], -np.ones((k["height"], k["width"]))), axis=-1)
    rays = local @ rotation_rub.T
    depth = np.full(rays.shape[:2], np.inf)
    colour = np.zeros((*rays.shape[:2], 3), dtype=np.uint8)
    for index, (normal, offset) in enumerate(planes):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (offset - normal @ position) / (rays @ normal)
        hit = (t > 0) & (t < depth)
        point = position + rays[hit] * t[hit, None]
        axes = [axis for axis in range(3) if abs(normal[axis]) < 0.5]
        texel = np.mod(np.floor(point[:, axes] / 0.01).astype(np.int64), 512)
        colour[hit] = TEXTURES[index][texel[:, 1], texel[:, 0]]
        depth[hit] = t[hit]
    return colour, depth.astype(np.float32)


def camera(rotation_rub, position, k):
    return {"intrinsics": dict(k), "extrinsics": {"coordinateSystem": "rub", "position": list(map(float, position)),
                                                 "quaternion": quaternion_xyzw(rotation_rub)}}


def views(*, crop=1.0, scale=1.0, turn=np.eye(3), shift=np.zeros(3), planes=ROOM, edit=False, warped=None):
    """Posed frames of the room as one Atlas world returns them."""
    k = {**SOURCE_K, "fx": SOURCE_K["fx"] * crop, "fy": SOURCE_K["fy"] * crop}
    frames = []
    for index, (rotation_cv, position) in enumerate(CAMERAS):
        rotation_rub = rotation_cv @ np.diag([1.0, -1.0, -1.0])
        rgb, depth = render(rotation_rub, position, k, planes)
        if edit and index == 0:
            # The removed object's pixels were repainted and its depth replaced.
            rgb[260:420, 60:300] = 128
            depth[260:420, 60:300] *= 1.3
        if index == warped:
            depth *= 1.4  # One view's returned depth disagrees with the room.
        confidence = np.full(depth.shape, 200, np.uint8)
        frames.append(posed_frame(frame_id=f"decoded-{index:09d}", rgb=rgb, depth=depth * scale,
                                  confidence=confidence,
                                  camera=camera(turn @ rotation_rub, scale * turn @ position + shift, k)))
    return frames


def background(**options):
    return views(crop=CROP, scale=WORLD_SCALE, turn=WORLD_ROTATION, shift=WORLD_TRANSLATION, **options)


def assert_recovered(found):
    assert found is not None
    assert found["scale"] == pytest.approx(WORLD_SCALE, rel=5e-3)
    turn = math.degrees(math.acos(min(1.0, (np.trace(found["rotation"] @ WORLD_ROTATION.T) - 1) / 2)))
    assert turn < 0.3
    # A room point reaches the background world where the background sees it.
    point = np.array([1.0, -1.2, 3.5])
    moved = found["scale"] * found["rotation"] @ point + found["translation"]
    assert np.linalg.norm(moved - (WORLD_SCALE * WORLD_ROTATION @ point + WORLD_TRANSLATION)) < 0.02


def test_cropped_cleaned_views_recover_the_similarity_between_the_two_worlds():
    found = shared_view_alignment(views(), background(edit=True))
    assert_recovered(found)
    assert found["shared_frame_count"] == found["registered_frame_count"] == found["agreeing_frame_count"] == 3
    assert found["inlier_fraction"] > 0.8 and found["median_residual_fraction"] < 0.005


def test_a_view_whose_depth_disagrees_is_outvoted():
    found = shared_view_alignment(views(), background(warped=2))
    assert_recovered(found)
    assert found["registered_frame_count"] == 3 and found["agreeing_frame_count"] == 2


def test_point_pairs_on_one_wall_still_fix_the_similarity():
    wall = ROOM[1:2]
    assert_recovered(shared_view_alignment(views(planes=wall), background(planes=wall)))


def test_views_that_do_not_show_the_same_pixels_are_never_paired():
    unrelated = background()
    for frame in unrelated:
        frame["rgb"] = np.random.default_rng(3).integers(0, 255, frame["rgb"].shape, dtype=np.uint8)
    assert shared_view_alignment(views(), unrelated) is None


def test_one_shared_view_is_not_enough():
    assert shared_view_alignment(views()[:1], background()) is None
