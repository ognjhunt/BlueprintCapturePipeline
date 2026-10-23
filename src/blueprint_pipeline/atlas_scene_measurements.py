"""Automatic approximate scene/object sizes from paired Atlas source observations.

Scale priors stay named assumptions. Original cameras, meshes and depths are never
rescaled in place. The existing scene analyzer supplies robust room coverage.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image

from .atlas_depth_splat import load_posed_rgbd
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .splat_scene_analysis import _robust_minmax


def source_to_posed(source: Path, rgb: np.ndarray) -> dict:
    """Check pixel alignment; never assume the provider preserved the input crop."""
    with Image.open(source) as im:
        original = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    sh, sw = original.shape[:2]
    resized = cv2.resize(original, (w, h))
    error = float(np.abs(resized.astype(float) - rgb).mean())
    if abs(sw / sh - w / h) < 0.002 and error < 8:
        matrix = np.array([[w / sw, 0, 0], [0, h / sh, 0], [0, 0, 1.0]])
        return {
            "matrix": matrix.tolist(),
            "source_grid": [sw, sh],
            "method": "pixel_checked_resize",
            "mean_rgb_error": error,
        }
    detector = cv2.ORB_create(nfeatures=3000)
    a, da = detector.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), None)
    b, db = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    matches = (
        []
        if da is None or db is None
        else [
            x
            for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
            for x in pair[:1]
        ]
    )
    if len(matches) < 12:
        raise ValueError("atlas_source_to_posed_alignment_unavailable")
    p = np.array([a[m.queryIdx].pt for m in matches])
    q = np.array([b[m.trainIdx].pt for m in matches])
    affine, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=2)
    if affine is None or int(inliers.sum()) < 12 or np.linalg.det(affine[:, :2]) <= 0:
        raise ValueError("atlas_source_to_posed_alignment_unavailable")
    matrix = np.vstack([affine, [0, 0, 1.0]]) @ np.diag([w / sw, h / sh, 1])
    residual = float(
        np.median(
            np.linalg.norm(
                p[inliers[:, 0] > 0] @ affine[:, :2].T + affine[:, 2] - q[inliers[:, 0] > 0], axis=1
            )
        )
    )
    return {
        "matrix": matrix.tolist(),
        "source_grid": [sw, sh],
        "method": "source_rgb_feature_registration",
        "median_pixel_residual": residual,
        "inliers": int(inliers.sum()),
    }


def load_views(pose_root: Path, inputs: dict) -> tuple[dict, list[dict]]:
    from .atlas_context_fov import full_view_sources
    result, frames = load_posed_rgbd(pose_root)
    by_id = {r["frame_id"]: r for r in full_view_sources(inputs, pose_root)}
    usage = result["operation_receipt"]["input_usage"]["frames"]
    for frame, row in zip(frames, usage, strict=True):
        source = by_id[frame["frame_id"]]
        if _sha256_file(Path(source["image_path"])) != source["image_digest"]:
            raise ValueError("atlas_automation_source_changed")
        image = next(r for r in row["outputs"] if r["role"] == "image")
        frame.update(image_path=str(pose_root / image["path"]), image_digest=image["sha256"])
        try:
            frame["pixel_alignment"] = source_to_posed(Path(source["image_path"]), frame["rgb"])
        except ValueError as exc:
            # Other views can still supply useful dimensions. A lost alignment
            # never becomes a guessed source box or reference endpoint.
            frame["pixel_alignment"] = {"status": "unavailable", "reason": str(exc)}
    return result, frames


def mapped_pixels(frame: dict, normalized: list) -> np.ndarray:
    alignment = frame["pixel_alignment"]
    if "matrix" not in alignment:
        raise ValueError("atlas_source_to_posed_alignment_unavailable")
    xy = np.asarray(normalized, dtype=float) * alignment["source_grid"]
    return np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(alignment["matrix"]).T[:, :2]


def unproject(frame: dict, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
    k = frame["intrinsics"]
    local = np.column_stack(
        [
            (xy[:, 0] + 0.5 - k["cx"]) / k["fx"] * depth,
            -(xy[:, 1] + 0.5 - k["cy"]) / k["fy"] * depth,
            -depth,
        ]
    )
    return local @ frame["rotation"].T + frame["position"]


def reference_point(frame: dict, xy: np.ndarray) -> np.ndarray:
    x, y = np.rint(xy).astype(int)
    h, w = frame["depth"].shape
    if not 0 <= x < w or not 0 <= y < h:
        raise ValueError("atlas_reference_outside_returned_view")
    radius = max(1, round(min(w, h) * 0.005))
    sl = np.s_[
        max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)
    ]
    values = frame["depth"][sl]
    keep = np.isfinite(values) & (values > 0) & (frame["confidence"][sl] > 0)
    if not keep.any():
        raise ValueError("atlas_reference_depth_missing")
    return unproject(frame, np.array([[x, y]]), np.array([np.median(values[keep])]))[0]


def frame_points(frame: dict, mask=None) -> np.ndarray:
    depth = frame["depth"]
    keep = np.isfinite(depth) & (depth > 0) & (frame["confidence"] > 0)
    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError("atlas_measurement_mask_grid_mismatch")
        keep &= mask
    y, x = np.nonzero(keep)
    # Bound CPU/memory per view, not the number of input views.
    step = max(1, math.ceil(len(x) / 100_000))
    return unproject(frame, np.column_stack([x[::step], y[::step]]), depth[y[::step], x[::step]])


def estimate_sizes(
    *, frames: list[dict], automation: dict, pose_digest: str, output: Path, tracks: list[dict] = ()
) -> dict:
    binding = {
        "pose_digest": pose_digest,
        "automation": automation,
        "tracks_digest": canonical_digest({"tracks": tracks}),
    }
    if output.is_file():
        value = json.loads(output.read_text())
        if value.get("binding") != binding or value.get("digest") != canonical_digest(
            value, digest_field="digest"
        ):
            raise ValueError("atlas_measurements_binding_changed")
        return value
    by_id = {f["frame_id"]: f for f in frames}
    accepted, gaps = [], []
    for reference in automation.get("scale_references", []):
        try:
            f = by_id[reference["frame_id"]]
            pixels = mapped_pixels(f, reference["endpoints_normalized"])
            points = np.array([reference_point(f, xy) for xy in pixels])
            distance = float(np.linalg.norm(points[1] - points[0]))
            if not math.isfinite(distance) or distance <= 1e-6:
                raise ValueError("atlas_reference_span_invalid")
            accepted.append(
                {
                    **reference,
                    "world_endpoints": points.tolist(),
                    "span_provider_units": distance,
                    "meters_per_unit": reference["assumed_distance_m"] / distance,
                    "meters_per_unit_range": [
                        v / distance for v in reference["plausible_distance_range_m"]
                    ],
                    "basis": "model_assumed_reference",
                    "public_reference_readback": next(
                        (
                            r
                            for r in automation.get("reference_sources", [])
                            if r["frame_id"] == reference["frame_id"]
                            and r["label"] == reference["label"]
                        ),
                        None,
                    ),
                    "pixel_alignment": f["pixel_alignment"],
                }
            )
        except (KeyError, ValueError) as exc:
            gaps.append({"reference": reference, "reason": str(exc)})
    scale = float(np.median([r["meters_per_unit"] for r in accepted])) if accepted else None
    scale_range = (
        [
            min(r["meters_per_unit_range"][0] for r in accepted),
            max(r["meters_per_unit_range"][1] for r in accepted),
        ]
        if accepted
        else None
    )
    clouds = [frame_points(f) for f in frames]
    nonempty = [p for p in clouds if len(p)]
    points = np.concatenate(nonempty) if nonempty else np.empty((0, 3))

    def bounds(points):
        if not len(points):
            return {"status": "unavailable", "reason": "no_finite_observed_depth"}
        lo, hi = _robust_minmax(points, 1.0)
        extent = hi - lo
        return {
            "status": "estimated",
            "point_count": len(points),
            "bounds_provider_units": [lo.tolist(), hi.tolist()],
            "extent_xyz_provider_units": extent.tolist(),
            "extent_xyz_m_estimate": (extent * scale).tolist() if scale else None,
            "extent_xyz_m_range": [(extent * s).tolist() for s in scale_range] if scale else None,
            "extent_xyz_feet_estimate": (extent * scale / 0.3048).tolist() if scale else None,
            "coverage": "central_98_percent_visible_points_not_full_hidden_object_or_room",
        }

    objects = []
    from .website_task_masks import decode_track_mask

    for target in tracks:
        parts = [
            frame_points(by_id[o["source_frame_id"]], decode_track_mask(o))
            for o in target["track"]["observations"]
            if o["source_frame_id"] in by_id
        ]
        objects.append(
            {
                "target_id": target["target_id"],
                "label": target["label"],
                **bounds(np.concatenate(parts) if parts else np.empty((0, 3))),
            }
        )
    result = {
        "schema_version": "atlas_scene_measurements.v1",
        "binding": binding,
        "status": "estimated" if scale else "relative_sizes_only",
        "meters_per_unit_estimate": scale,
        "meters_per_unit_range": scale_range,
        "scale_references": accepted,
        "reference_gaps": gaps,
        "scene": bounds(points),
        "objects": objects,
        "scale_status": "estimated_from_named_assumptions"
        if scale
        else "unscaled_provider_world_units",
        "uncaptured_regions": automation.get("uncaptured_regions", []),
        "raw_geometry_rescaled": False,
        "metric_measurement_proven": False,
        "physical_evidence": False,
        "simulation_qualified": False,
        "claim_ceiling": "development_only",
    }
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(output, result)
    return result
