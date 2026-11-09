"""A synthetic retained images2PosedRGBD result: a furnished room filmed on a walk, no provider call.

Frames have Atlas's real grid (1280x720 by default), half-float ZIP EXR depth in
9 mm plateaus with a low-frequency error of its own per view, and 8-bit
confidence and image PNGs, bound by digest into the result exactly as
``website_atlas_pose`` retains them. Used to measure the draft fusion's memory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from blueprint_pipeline import atlas_depth_splat as depth_splat
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_depth_splat import write_exr

# Axis-aligned room (x, y, z bounds) and two pieces of furniture inside it.
ROOM = ((-3.0, 3.0), (-1.4, 1.6), (-4.0, 4.0))
FURNITURE = (((-0.6, 0.6), (-1.4, -0.65), (-0.8, 0.0)),
             ((2.2, 3.0), (-1.4, 0.6), (1.0, 2.2)))


def room_camera(index: int, count: int, *, width: int = 1280, height: int = 720) -> dict:
    """A walk around the room's centre, looking inwards with a little yaw and pitch of its own."""
    angle = 2.0 * math.pi * index / count
    position = [1.6 * math.cos(angle), 0.05 * math.sin(3 * angle), 2.6 * math.sin(angle)]
    yaw = math.atan2(position[0], position[2]) + 0.35 * math.sin(5 * angle)
    pitch = -0.15 + 0.1 * math.cos(7 * angle)
    qy = (0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2))
    qx = (math.sin(pitch / 2), 0.0, 0.0, math.cos(pitch / 2))
    # Yaw then pitch: q = qy * qx (xyzw).
    x = qy[3] * qx[0] + qy[0] * qx[3] + qy[1] * qx[2] - qy[2] * qx[1]
    y = qy[3] * qx[1] - qy[0] * qx[2] + qy[1] * qx[3] + qy[2] * qx[0]
    z = qy[3] * qx[2] + qy[0] * qx[1] - qy[1] * qx[0] + qy[2] * qx[3]
    w = qy[3] * qx[3] - qy[0] * qx[0] - qy[1] * qx[1] - qy[2] * qx[2]
    focal = 0.78 * width
    return {"extrinsics": {"coordinateSystem": "rub", "position": position, "quaternion": [x, y, z, w]},
            "intrinsics": {"fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2,
                           "width": width, "height": height}}


def _box_entry(origin, rays, bounds):
    """Distance along each ray to where it enters an axis-aligned box from outside; inf if it misses."""
    near, far = np.full(rays.shape[:-1], -np.inf), np.full(rays.shape[:-1], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, (low, high) in enumerate(bounds):
            a, b = (low - origin[axis]) / rays[..., axis], (high - origin[axis]) / rays[..., axis]
            near, far = np.maximum(near, np.minimum(a, b)), np.minimum(far, np.maximum(a, b))
    return np.where((far >= near) & (near > 0), near, np.inf)


def room_frame(index: int, count: int, *, width: int = 1280, height: int = 720):
    """The camera, the image, Atlas-like terraced depth and the confidence of one view."""
    cam = room_camera(index, count, width=width, height=height)
    k, e = cam["intrinsics"], cam["extrinsics"]
    origin = np.asarray(e["position"], dtype=np.float64)
    u = (np.arange(width) + 0.5 - k["cx"]) / k["fx"]
    v = (np.arange(height) + 0.5 - k["cy"]) / k["fy"]
    # Planar rays (local z = -1), so the distance along them is the planar depth Atlas returns.
    rays = np.stack(np.broadcast_arrays(u[None, :], -v[:, None], -1.0), axis=-1)
    rays = rays @ depth_splat._rotation_matrix(e["quaternion"]).T
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = [np.where(rays[..., axis] > 0, (high - origin[axis]) / rays[..., axis],
                          (low - origin[axis]) / rays[..., axis]) for axis, (low, high) in enumerate(ROOM)]
    depth = np.minimum.reduce(exits)
    for box in FURNITURE:
        depth = np.minimum(depth, _box_entry(origin, rays, box))
    world = origin + depth[..., None] * rays
    rgb = np.clip(128 + 90 * np.sin(world * np.array([5.0, 7.0, 3.0]) + index * 0.01), 0, 255).astype(np.uint8)
    # Each view's own low-frequency error, within the few percent views share.
    rng = np.random.default_rng(index)
    tilt = rng.uniform(-0.008, 0.008, 3)
    error = 1.0 + tilt[0] + tilt[1] * (u[None, :] / u.max()) + tilt[2] * (v[:, None] / v.max())
    terraced = np.round(depth * error / 0.009) * 0.009
    confidence = np.full((height, width), 255, np.uint8)
    confidence[:, :4] = 0
    confidence[(np.indices((height, width)).sum(axis=0) // 37) % 11 == 0] = 140
    return cam, rgb, terraced.astype(np.float32), confidence


def write_room_pose_root(root: Path, count: int, *, width: int = 1280, height: int = 720,
                         views: int | None = None) -> Path:
    """Write the first ``views`` (all) of a ``count``-view walk as a digest-bound retained pose result."""
    import cv2

    cameras, usage = [], []
    for index in range(count if views is None else views):
        cam, rgb, depth, confidence = room_frame(index, count, width=width, height=height)
        folder = root / "atlas_pose" / "outputs" / f"{index:06d}"
        folder.mkdir(parents=True)
        write_exr(folder / "depth.bin", depth)
        for role, pixels in (("image", np.ascontiguousarray(rgb[..., ::-1])), ("confidence", confidence)):
            encoded = folder / f"{role}.png"
            assert cv2.imwrite(str(encoded), pixels)
            encoded.rename(folder / f"{role}.bin")
        outputs = [{"role": role, "path": str((folder / f"{role}.bin").relative_to(root)),
                    "sha256": depth_splat._sha256_file(folder / f"{role}.bin")}
                   for role in ("image", "depth", "confidence")]
        cameras.append({"camera": cam})
        usage.append({"frame_id": f"view-{index:03d}", "outputs": outputs})
    result = {"binding": {"source_video_digest": "sha256:" + "1" * 64}, "depth_unit": "provider_world_units",
              "scale_status": "unscaled_provider_world_units",
              "operation": {"id": "op_synthetic_room", "response": {"frames": cameras}},
              "operation_receipt": {"input_usage": {"frames": usage}}}
    result["digest"] = canonical_digest(result, digest_field="digest")
    (root / depth_splat.POSE_RESULT).write_text(json.dumps(result))
    return root
