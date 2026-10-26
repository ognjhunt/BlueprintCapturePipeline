"""CPU check that each policy camera sees the task's contact patch through the captured scene.

The GPU composition gate (``native_task_asset_composition_gate``) renders the scene
only after both policy servers have loaded. On 2026-10-25 it stopped a paid canary
($0.42) on splat occlusion that the saved splat already showed. This module reads the
packet's Gaussian splat and task asset and repeats the decisive part before any spend:

* the contact patch (the sealed articulated contact and its arrival/sweep radius, the
  region the GPU gate uses) must project inside every world-fixed policy camera;
* the task asset, ray cast through its colliders, must cover part of the patch;
* splat Gaussians in front of the asset may cover only a small share of the patch.

A legacy mechanism without a contact path is held to whole-asset coverage, as on the
GPU. Gaussians are discs of their largest scale and colliders stand in for render
meshes, so the GPU gate stays the authority and this check blocks only clear failures.
Review-only cameras and the asset's whole-body coverage are reported without blocking.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .scene_physics_preflight import (
    ScenePhysicsPreflightError,
    _file_sha256,
    _placed,
    _pose_matrix,
    body_poses,
    load_asset_model,
)

SCHEMA_VERSION = "native_task_scene_camera_visibility.v1"
PACKET_REPORT_FILENAME = "native_task_scene_camera_visibility.v1.json"
SPLAT_REPRESENTATION = "particlefield_3d_gaussian_splat"
DEPTH_TOLERANCE_M = 0.01
OCCLUDED_SHARE_LIMIT = 0.02
MINIMUM_OCCLUDED_PIXELS = 4
MINIMUM_ASSET_SHARE = 0.25
MINIMUM_OPACITY = 0.5
MAXIMUM_DISC_RADIUS_PX = 3
SILHOUETTE_MARGIN_PX = 2


class SceneCameraVisibilityError(ValueError):
    """The camera check cannot measure this packet."""


class SceneCameraVisibilityBlocked(SceneCameraVisibilityError):
    """A paid native run would start with its contact hidden from a policy camera."""

    def __init__(self, blockers: Sequence[str]):
        super().__init__(",".join(blockers))
        self.blockers = list(blockers)


def load_splat(usd_path: str | Path, prim_path: str,
               world_from_stage: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World centres, world disc radii and opacities of a Gaussian splat prim."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(prim_path) if stage else None
    if prim is None or not prim.IsValid():
        raise SceneCameraVisibilityError("scene_camera_splat_prim_missing")
    try:
        # float32 halves the memory of a multi-million Gaussian scan on the control plane.
        positions = np.asarray(prim.GetAttribute("positions").Get(), dtype=np.float32).reshape(-1, 3)
        scales = np.asarray(prim.GetAttribute("scales").Get(), dtype=np.float32).reshape(-1, 3)
        opacities = np.asarray(prim.GetAttribute("opacities").Get(), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise SceneCameraVisibilityError("scene_camera_splat_attributes_invalid") from exc
    if not (len(positions) == len(scales) == len(opacities)) or not len(positions):
        raise SceneCameraVisibilityError("scene_camera_splat_attributes_invalid")
    matrix = world_from_stage @ np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=float).T
    world = positions @ matrix[:3, :3].T.astype(np.float32) + matrix[:3, 3].astype(np.float32)
    linear = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
    return world, scales.max(axis=1) * np.float32(linear), opacities


def _camera(row: Mapping[str, Any]) -> tuple[np.ndarray, Mapping[str, float]]:
    world_from_camera = np.asarray(row["frame_from_camera_matrix"], dtype=float).reshape(4, 4)
    intrinsics = row["intrinsics"]
    if row.get("optical_convention", "opencv") != "opencv" or not np.isfinite(world_from_camera).all():
        raise SceneCameraVisibilityError("scene_camera_calibration_invalid:" + str(row.get("role")))
    return world_from_camera, intrinsics


def _ray_hits(colliders: Sequence[tuple[Any, np.ndarray]], origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Nearest ray parameter into any solid collider (camera depth when direction z is 1)."""
    best = np.full(len(directions), np.inf)
    for collider, world in colliders:
        rotation, translation = world[:3, :3], world[:3, 3]
        o = rotation.T @ (origin - translation)
        d = directions @ rotation
        with np.errstate(divide="ignore", invalid="ignore"):
            if collider.kind in {"box", "hull"}:
                if collider.kind == "box":
                    h = np.asarray(collider.half_extents, dtype=float)
                    normals = np.vstack([np.eye(3), -np.eye(3)])
                    offsets = np.concatenate([h, h])
                else:
                    normals, offsets = collider.planes[:, :3], collider.planes[:, 3]
                denominator = d @ normals.T
                numerator = offsets - normals @ o
                ratio = numerator / denominator
                entering = np.where(denominator < 0, ratio, -np.inf).max(axis=1)
                leaving = np.where(denominator > 0, ratio, np.inf).min(axis=1)
                parallel_outside = ((denominator == 0) & (numerator < 0)).any(axis=1)
                hit = (entering <= leaving) & (leaving > 0) & ~parallel_outside
                t = np.where(entering > 0, entering, leaving)
            elif collider.kind == "sphere":
                b = d @ o
                a = (d * d).sum(axis=1)
                disc = b * b - a * (o @ o - collider.radius ** 2)
                t = (-b - np.sqrt(np.maximum(disc, 0))) / a
                hit = disc >= 0
                t = np.where(t > 0, t, (-b + np.sqrt(np.maximum(disc, 0))) / a)
                hit &= t > 0
            else:  # cylinder about its local z axis
                r, hh = collider.radius, collider.half_height
                a = d[:, 0] ** 2 + d[:, 1] ** 2
                b = 2 * (o[0] * d[:, 0] + o[1] * d[:, 1])
                c = o[0] ** 2 + o[1] ** 2 - r * r
                disc = b * b - 4 * a * c
                side = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
                side_ok = (disc >= 0) & (side > 0) & (np.abs(o[2] + side * d[:, 2]) <= hh)
                caps = np.stack([(hh - o[2]) / d[:, 2], (-hh - o[2]) / d[:, 2]], axis=1)
                within = ((o[0] + caps * d[:, :1]) ** 2 + (o[1] + caps * d[:, 1:2]) ** 2) <= r * r
                caps = np.where(within & (caps > 0), caps, np.inf).min(axis=1)
                t = np.minimum(np.where(side_ok, side, np.inf), caps)
                hit = np.isfinite(t)
        best = np.where(hit & (t < best), t, best)
    return best


def _splat_depth(centres: np.ndarray, radii: np.ndarray, camera_from_world: np.ndarray,
                 intrinsics: Mapping[str, float], window: tuple[int, int, int, int]) -> np.ndarray:
    """Nearest Gaussian camera depth per pixel inside ``window`` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = window
    p = centres @ camera_from_world[:3, :3].T.astype(np.float32) + camera_from_world[:3, 3].astype(np.float32)
    front = p[:, 2] > 0.05
    p, s = p[front], radii[front]
    u = intrinsics["fx"] * p[:, 0] / p[:, 2] + intrinsics["cx"]
    v = intrinsics["fy"] * p[:, 1] / p[:, 2] + intrinsics["cy"]
    r = np.clip(np.round(intrinsics["fx"] * s / p[:, 2]), 0, MAXIMUM_DISC_RADIUS_PX).astype(np.int64)
    near = (u > x0 - MAXIMUM_DISC_RADIUS_PX - 1) & (u < x1 + MAXIMUM_DISC_RADIUS_PX + 1) \
        & (v > y0 - MAXIMUM_DISC_RADIUS_PX - 1) & (v < y1 + MAXIMUM_DISC_RADIUS_PX + 1)
    u, v, r, z = np.round(u[near]).astype(np.int64), np.round(v[near]).astype(np.int64), r[near], p[near, 2]
    width, height = x1 - x0, y1 - y0
    depth = np.full(width * height, np.inf, dtype=np.float32)
    for radius in range(MAXIMUM_DISC_RADIUS_PX + 1):
        chosen = r >= radius
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if max(abs(du), abs(dv)) != radius:
                    continue
                uu, vv = u[chosen] + du - x0, v[chosen] + dv - y0
                inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                np.minimum.at(depth, vv[inside] * width + uu[inside], z[chosen][inside])
    return depth.reshape(height, width)


def _erode(mask: np.ndarray, margin: int) -> np.ndarray:
    padded = np.pad(mask, margin, constant_values=False)
    interior = np.ones_like(mask)
    for dy in range(2 * margin + 1):
        for dx in range(2 * margin + 1):
            interior &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return interior


def camera_coverage(*, colliders: Sequence[tuple[Any, np.ndarray]], centres: np.ndarray, radii: np.ndarray,
                    camera: Mapping[str, Any], region: Mapping[str, Any] | None) -> dict[str, Any]:
    """Patch (or whole-asset) coverage of the placed asset through the splat in one camera."""
    world_from_camera, k = _camera(camera)
    camera_from_world = np.linalg.inv(world_from_camera)
    width, height = int(k["width"]), int(k["height"])
    corners = np.asarray([c for collider, world in colliders
                          for c in (collider.local_corners() @ world[:3, :3].T + world[:3, 3])])
    local = corners @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    row: dict[str, Any] = {"role": camera.get("role"), "policy_input": camera.get("policy_input") is True}
    if not (local[:, 2] > 0.05).any():
        return {**row, "asset_in_view": False}
    ahead = local[local[:, 2] > 0.05]
    us = k["fx"] * ahead[:, 0] / ahead[:, 2] + k["cx"]
    vs = k["fy"] * ahead[:, 1] / ahead[:, 2] + k["cy"]
    window = (max(0, int(us.min()) - 4), max(0, int(vs.min()) - 4),
              min(width, int(math.ceil(us.max())) + 5), min(height, int(math.ceil(vs.max())) + 5))
    if window[0] >= window[2] or window[1] >= window[3]:
        return {**row, "asset_in_view": False}
    xs, ys = np.meshgrid(np.arange(window[0], window[2]), np.arange(window[1], window[3]))
    rays = np.stack([(xs - k["cx"]) / k["fx"], (ys - k["cy"]) / k["fy"], np.ones_like(xs, dtype=float)], -1)
    asset = _ray_hits(colliders, world_from_camera[:3, 3], rays.reshape(-1, 3) @ world_from_camera[:3, :3].T)
    asset = asset.reshape(xs.shape)
    splat = _splat_depth(centres, radii, camera_from_world, k, window)
    interior = _erode(np.isfinite(asset), SILHOUETTE_MARGIN_PX)
    occluded = interior & (splat < asset - DEPTH_TOLERANCE_M)
    row.update(asset_in_view=True, asset_interior_pixels=int(interior.sum()),
               asset_share_occluded=float(occluded.sum() / max(1, interior.sum())))
    if region is None:
        row.update(scope="whole_asset", patch_pixels=int(interior.sum()), patch_asset_pixels=int(interior.sum()),
                   patch_occluded_pixels=int(occluded.sum()))
        return row
    centre = camera_from_world[:3, :3] @ np.asarray(region["center_world_m"]) + camera_from_world[:3, 3]
    radius = float(region["radius_m"])
    if centre[2] <= radius:
        return {**row, "scope": "contact_patch", "patch_in_view": False}
    uv = np.array([k["fx"] * centre[0] / centre[2] + k["cx"], k["fy"] * centre[1] / centre[2] + k["cy"]])
    # The GPU gate's projection: near side of the radius, widened for off-axis perspective.
    axes = np.array([k["fx"], k["fy"]]) * radius / (centre[2] - radius) * (1 + np.abs(centre[:2]) / centre[2])
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        return {**row, "scope": "contact_patch", "patch_in_view": False, "patch_centre_px": uv.tolist()}
    patch = ((xs - uv[0]) / axes[0]) ** 2 + ((ys - uv[1]) / axes[1]) ** 2 <= 1
    full_patch = math.pi * axes[0] * axes[1]
    row.update(scope="contact_patch", patch_in_view=True, patch_centre_px=uv.tolist(),
               patch_pixels=int(round(full_patch)), patch_asset_pixels=int((patch & interior).sum()),
               patch_occluded_pixels=int((patch & occluded).sum()))
    return row


def scene_camera_visibility(packet_dir: str | Path) -> dict[str, Any]:
    """Measure contact-patch visibility for every world-fixed camera of a packet's plan."""
    from .decision_evidence_contracts import canonical_digest
    from .native_task_asset_composition_gate import articulated_reset_contact_region

    root = Path(packet_dir)
    plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text())
    objects = [o for o in plan.get("objects", []) if isinstance(o, Mapping)]
    subjects = [o for o in objects if o.get("task_subject") is True or o.get("semantic_role") == "task_object"]
    appearance = [o for o in objects if o.get("semantic_role") == "scene_appearance"]
    alignment = plan.get("appearance_frame_alignment") or {}
    reason = None
    if plan.get("task_kind") != "articulated_open_close" or len(subjects) != 1 \
            or subjects[0].get("object_type") != "ARTICULATION":
        reason = "no_articulated_task_subject"
    elif len(appearance) != 1 or alignment.get("representation") != SPLAT_REPRESENTATION:
        reason = "appearance_not_a_gaussian_splat"
    base = {"schema_version": SCHEMA_VERSION, "arena_scene_plan_digest": plan.get("plan_digest"),
            "simulator_executed": False, "provider_mutation_performed": False}
    if reason is not None:
        result = {**base, "status": "not_applicable", "reason": reason, "blockers": [], "advisories": [],
                  "cameras": [], "summary": "Not checked on CPU: " + reason.replace("_", " ") + "."}
        result["report_digest"] = canonical_digest(result, digest_field="report_digest")
        return result

    def bound(row: Mapping[str, Any]) -> Path:
        path = (root / str(row.get("usd_path") or "")).resolve()
        if root.resolve() not in path.parents or not path.is_file() or _file_sha256(path) != row.get("sha256"):
            raise SceneCameraVisibilityError("scene_camera_packet_asset_unbound:" + str(row.get("semantic_role")))
        return path

    try:
        region = articulated_reset_contact_region(plan)
    except ValueError as exc:
        raise SceneCameraVisibilityError(str(exc)) from exc
    subject = subjects[0]
    reset = subject.get("reset_state") or {}
    try:
        model = load_asset_model(bound(subject))
    except ScenePhysicsPreflightError as exc:
        raise SceneCameraVisibilityError("scene_camera_asset_unreadable:" + str(exc)) from exc
    world_from_asset = _pose_matrix(reset.get("root_pose_world") or subject["pose_world"])
    poses = {b: world_from_asset @ p for b, p in body_poses(model, reset.get("joint_positions")).items()}
    colliders = list(_placed(model.colliders, poses))
    if not colliders:
        raise SceneCameraVisibilityError("scene_camera_asset_colliders_missing")
    centres, radii, opacity = load_splat(bound(appearance[0]), str(alignment.get("appearance_prim_path") or ""),
                                         _pose_matrix(appearance[0]["pose_world"]))
    keep = opacity >= MINIMUM_OPACITY
    centres, radii = centres[keep], radii[keep]
    cameras, blockers, advisories, notices = [], [], [], []
    for camera in plan.get("cameras") or []:
        role = str(camera.get("role"))
        if camera.get("pose_frame") != "world":
            notices.append(f"scene_camera_not_world_fixed:{role}")
            continue
        required = camera.get("policy_input") is True and camera.get("review_only") is not True
        row = camera_coverage(colliders=colliders, centres=centres, radii=radii, camera=camera, region=region)
        row["required"] = required
        cameras.append(row)
        problems = []
        if row.get("asset_in_view") is False or row.get("patch_in_view") is False:
            problems.append(f"scene_camera_contact_outside_view:{role}")
        else:
            patch, on_asset, occluded = row["patch_pixels"], row["patch_asset_pixels"], row["patch_occluded_pixels"]
            if on_asset < max(1.0, MINIMUM_ASSET_SHARE * patch):
                problems.append(f"scene_camera_contact_not_on_asset:{role}:{on_asset}/{patch}px")
            if occluded > max(MINIMUM_OCCLUDED_PIXELS, OCCLUDED_SHARE_LIMIT * max(1, on_asset)):
                problems.append(f"scene_camera_contact_occluded_by_scene:{role}:{100 * occluded / max(1, on_asset):.0f}%")
            if row["asset_share_occluded"] > 0:
                advisories.append(f"scene_camera_asset_partly_behind_scene:{role}:{100 * row['asset_share_occluded']:.0f}%")
        (blockers if required else advisories).extend(problems)
    if not any(row["required"] for row in cameras):
        blockers.append("scene_camera_policy_camera_missing")
    scope = "the handle contact patch" if region is not None else "the task asset"
    if blockers:
        summary = f"Blocked: a policy camera cannot see {scope} through the captured scene ({'; '.join(blockers)})."
    else:
        summary = f"Every world-fixed policy camera sees {scope} clear of the captured scene."
    result = {**base, "status": "blocked" if blockers else "passed", "blockers": blockers,
              "advisories": advisories, "notices": notices, "contact_region": region, "cameras": cameras,
              "splat_gaussians_considered": int(len(centres)), "summary": summary}
    result["report_digest"] = canonical_digest(result, digest_field="report_digest")
    return result


def require_scene_camera_admission(packet_dir: str | Path) -> dict[str, Any]:
    """Refuse a paid native run whose policy camera cannot see the contact, or that this cannot measure."""
    try:
        report = scene_camera_visibility(packet_dir)
    except (SceneCameraVisibilityError, ScenePhysicsPreflightError, OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SceneCameraVisibilityBlocked):
            raise
        raise SceneCameraVisibilityBlocked(["scene_camera_check_refused:" + str(exc)]) from exc
    if report["status"] == "blocked":
        raise SceneCameraVisibilityBlocked(report["blockers"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="CPU camera visibility of a native task arena packet.")
    parser.add_argument("--packet", required=True, help="native task arena packet directory")
    parser.add_argument("--output", help="write the full report here")
    args = parser.parse_args(argv)
    try:
        report = scene_camera_visibility(args.packet)
    except (SceneCameraVisibilityError, ScenePhysicsPreflightError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 1
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"], "blockers": report["blockers"],
                      "advisories": report["advisories"]}, sort_keys=True))
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
