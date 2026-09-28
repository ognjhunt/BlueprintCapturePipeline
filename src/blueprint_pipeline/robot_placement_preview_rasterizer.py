"""Opaque CPU depth previews of a bounded robot-placement neighborhood.

All local triangles share one depth buffer. No wireframe transparency or
robot-on-top compositing may turn occlusion into apparent collision/clearance.
These explicitly cropped geometry previews are never native execution evidence.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_release_identity import running_release_commit

RENDERER = "blueprint.cpu-placement-orthographic-zbuffer.v2"


def rasterize(
    *,
    triangles: np.ndarray,
    colours: np.ndarray,
    basis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    crop_low: np.ndarray,
    crop_high: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    pixels = np.full((height, width, 3), 247, dtype=np.uint8)
    depth = np.full((height, width), -np.inf, dtype=np.float64)
    projected = triangles @ basis.T
    scale = np.array([width - 1, height - 1]) / (high - low)
    xy = (projected[:, :, :2] - low) * scale
    xy[:, :, 1] = height - 1 - xy[:, :, 1]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    light = np.asarray([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    shade = 0.62 + 0.38 * np.abs((normals / np.maximum(norms[:, None], 1e-12)) @ light)
    shaded = np.clip(colours * shade[:, None], 0, 255).astype(np.uint8)
    for i, points in enumerate(xy):
        x0 = max(0, int(np.floor(points[:, 0].min())))
        x1 = min(width - 1, int(np.ceil(points[:, 0].max())))
        y0 = max(0, int(np.floor(points[:, 1].min())))
        y1 = min(height - 1, int(np.ceil(points[:, 1].max())))
        if x0 > x1 or y0 > y1:
            continue
        a, b, c = points
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-10:
            continue
        xx = np.arange(x0, x1 + 1)[None, :] + 0.5
        yy = np.arange(y0, y1 + 1)[:, None] + 0.5
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / den
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
        wc = 1 - wa - wb
        z = wa * projected[i, 0, 2] + wb * projected[i, 1, 2] + wc * projected[i, 2, 2]
        tile = depth[y0 : y1 + 1, x0 : x1 + 1]
        keep = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9) & (z > tile)
        # Clip fragments in world space, not just triangle centroids: a large
        # room wall must neither disappear locally nor project from far away.
        for axis in range(3):
            world = (
                wa * triangles[i, 0, axis] + wb * triangles[i, 1, axis] + wc * triangles[i, 2, axis]
            )
            keep &= (world >= crop_low[axis] - 1e-9) & (world <= crop_high[axis] + 1e-9)
        tile[keep] = z[keep]
        pixels[y0 : y1 + 1, x0 : x1 + 1][keep] = shaded[i]
    return pixels, depth


def render(
    *,
    index: Any,
    proposal: Mapping[str, Any],
    target_position_world_m: Sequence[float],
    trajectory_waypoints_world_m: Sequence[Sequence[float]] = (),
    image_size: tuple[int, int] = (1000, 720),
) -> list[dict[str, Any]]:
    from .task_evaluation_robot_placement_geometry import (
        RobotPlacementGeometryError,
        _yaw_from_quaternion,
    )

    pose = proposal.get("pose") or {}
    position = np.asarray(pose.get("position_world_m"), dtype=np.float64)
    quaternion = pose.get("orientation_xyzw")
    target = np.asarray(target_position_world_m, dtype=np.float64)
    path = np.asarray(trajectory_waypoints_world_m, dtype=np.float64)
    if path.size == 0:
        path = target.reshape(1, 3)
    if (
        position.shape != (3,)
        or target.shape != (3,)
        or path.ndim != 2
        or path.shape[1] != 3
        or not np.isfinite(np.concatenate([position, target, path.ravel()])).all()
        or not isinstance(quaternion, (list, tuple))
        or len(quaternion) != 4
        or not 120 <= min(image_size) <= max(image_size) <= 4096
    ):
        raise RobotPlacementGeometryError("robot_placement_preview_pose_invalid")
    yaw, _ = _yaw_from_quaternion(quaternion)
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    robot = index.robot_triangles @ rotation.T + position
    focus = np.concatenate([robot.reshape(-1, 3), path, target.reshape(1, 3)])
    crop_low, crop_high = focus.min(axis=0) - 0.30, focus.max(axis=0) + 0.30
    selected = np.flatnonzero(
        np.all(index.triangle_maximum >= crop_low, axis=1)
        & np.all(index.triangle_minimum <= crop_high, axis=1)
    )
    support = next(
        (s for s in index.support_surfaces if s.surface_id == proposal.get("support_surface_id")),
        None,
    )
    support_ids = set(support.triangle_indices) if support else set()
    colours = np.asarray(
        [(113, 164, 207) if int(i) in support_ids else (191, 197, 204) for i in selected],
        dtype=float,
    ).reshape(-1, 3)
    triangles = np.concatenate([index.triangles[selected], robot])
    colours = np.concatenate([colours, np.tile([218.0, 75.0, 57.0], (len(robot), 1))])
    forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    side_depth = np.cross(forward, up)
    oblique_depth = -forward + side_depth * 0.85 + up * 0.9
    oblique_depth /= np.linalg.norm(oblique_depth)
    oblique_right = np.cross(up, oblique_depth)
    oblique_right /= np.linalg.norm(oblique_right)
    cameras = [
        ("top_down_xy", np.eye(3)),
        ("side_task", np.asarray([forward, up, side_depth])),
        (
            "oblique",
            np.asarray([oblique_right, np.cross(oblique_depth, oblique_right), oblique_depth]),
        ),
    ]
    results = []
    for label, basis in cameras:
        bounds = np.asarray(list(product(*zip(crop_low, crop_high, strict=True)))) @ basis.T
        low, high = bounds[:, :2].min(axis=0), bounds[:, :2].max(axis=0)
        width, height = image_size
        canvas_height = height - 70
        # One metric scale in both dimensions; never distort the robot or gap.
        span = high - low
        ratio = width / canvas_height
        if span[0] / span[1] < ratio:
            extra = (span[1] * ratio - span[0]) / 2
            low[0] -= extra
            high[0] += extra
        else:
            extra = (span[0] / ratio - span[1]) / 2
            low[1] -= extra
            high[1] += extra
        pixels, _ = rasterize(
            triangles=triangles,
            colours=colours,
            basis=basis,
            low=low,
            high=high,
            crop_low=crop_low,
            crop_high=crop_high,
            image_size=(width, canvas_height),
        )
        image = Image.new("RGB", image_size, "white")
        image.paste(Image.fromarray(pixels), (0, 40))
        draw = ImageDraw.Draw(image)

        def point(world):
            projected = np.asarray(world) @ basis.T
            v = (projected[:2] - low) / (high - low)
            return (int(v[0] * (width - 1)), 40 + int((1 - v[1]) * (canvas_height - 1)))

        # Diagnostic overlays are explicitly named; no overlay is evidence of
        # camera visibility, collision clearance, or successful execution.
        for a, b in zip(path[:-1], path[1:], strict=True):
            draw.line([point(a), point(b)], fill=(15, 155, 184), width=2)
        x, y = point(target)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(20, 157, 74))
        draw.line([point(position), point(position + 0.22 * forward)], fill=(238, 148, 35), width=4)
        draw.text(
            (12, 8),
            f"{label} | opaque local geometry | red robot, blue support, gray scene",
            fill=(25, 30, 35),
        )
        draw.text(
            (12, height - 22),
            "Cropped region. Cyan trajectory / green target / orange facing are overlays. No native execution claim.",
            fill=(35, 40, 45),
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        provenance = {
            "renderer": RENDERER,
            "renderer_source_commit": running_release_commit(),
            "renderer_module_digest": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scene_digest": index.scene_digest,
            "robot_asset_digest": index.robot_asset_digest,
            "pose": dict(pose),
            "target_position_world_m": target.tolist(),
            "trajectory_world_m": path.tolist(),
            "camera_basis_world": basis.tolist(),
            "projection": "orthographic",
            "image_size": list(image_size),
            "crop_bounds_world_m": {"minimum": crop_low.tolist(), "maximum": crop_high.tolist()},
            "scene_triangle_count": len(index.triangles),
            "retained_scene_triangle_count": len(selected),
            "robot_triangle_count": len(robot),
            "robot_mesh_scope": "default_prim_with_instance_proxies",
            "depth_buffer_shared_by_scene_and_robot": True,
            "omitted_region": "outside_explicit_local_crop",
            "image_digest": digest,
            "native_execution_evidence": False,
        }
        provenance["render_digest"] = canonical_digest(provenance, digest_field="render_digest")
        results.append(
            {
                "label": label,
                "digest": digest,
                "image_url": "data:image/png;base64," + base64.b64encode(payload).decode(),
                "detail": "high",
                "render_provenance": provenance,
            }
        )
    return results
