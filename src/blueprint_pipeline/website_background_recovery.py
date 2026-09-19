"""Recover visible background from original views using estimated cameras.

Copied pixels are observed colors; their placement still depends on inferred
geometry. Holes stay explicit for the existing image-editing worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .local_reconstruction_adapters import _sha256_file
from .website_task_masks import decode_track_mask


def removal_masks(task_masks: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    masks = {frame["frame_id"]: np.zeros((frame["height"], frame["width"]), dtype=bool) for frame in frames}
    for target in task_masks["targets"]:
        if target["task_effect"] != "manipulated" or target["disposition"] != "remove":
            continue
        for observation in target["track"]["observations"]:
            mask = decode_track_mask(observation)
            original = masks[observation["source_frame_id"]]
            if mask.shape != original.shape:
                raise ValueError("background_recovery_mask_shape_mismatch")
            original |= mask
    return {frame_id: binary_dilation(mask, iterations=1) if mask.any() else mask for frame_id, mask in masks.items()}


def _load(frame: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for path_key, digest_key in (("image_path", "image_digest"), ("geometry_path", "geometry_digest")):
        if _sha256_file(Path(frame[path_key])) != frame[digest_key]:
            raise ValueError("background_recovery_source_changed")
    with Image.open(frame["image_path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with np.load(frame["geometry_path"], allow_pickle=False) as data:
        depth, valid = data["depth_m"], data["valid_mask"]
    if rgb.shape[:2] != depth.shape or valid.shape != depth.shape:
        raise ValueError("background_recovery_geometry_shape_mismatch")
    return rgb, depth, valid


def recover_observed_background(*, frames: Sequence[Mapping[str, Any]], task_masks: Mapping[str, Any],
                                output_root: Path) -> list[dict[str, Any]]:
    masks = removal_masks(task_masks, frames)
    loaded = {frame["frame_id"]: _load(frame) for frame in frames}
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, target in enumerate(frames):
        target_id = target["frame_id"]
        original, target_depth, target_valid = loaded[target_id]
        edited = original.copy()
        target_mask = masks[target_id]
        recovered = np.zeros(target_mask.shape, dtype=bool)
        zbuffer = np.full(target_mask.shape, np.inf)
        height, width = target_mask.shape
        for source in frames:
            if source["frame_id"] == target_id or not target_mask.any():
                continue
            rgb, depth, valid = loaded[source["frame_id"]]
            ys, xs = np.nonzero(valid & ~masks[source["frame_id"]])
            if not len(xs):
                continue
            camera = (np.stack([xs, ys, np.ones_like(xs)], axis=1) @ np.linalg.inv(source["intrinsics"]).T) * depth[ys, xs, None]
            transform = np.asarray(target["camera_from_world"]) @ np.asarray(source["world_from_camera"])
            target_camera = camera @ transform[:3, :3].T + transform[:3, 3]
            projected = target_camera @ np.asarray(target["intrinsics"]).T
            front = np.isfinite(projected).all(axis=1) & (target_camera[:, 2] > 0)
            projected, target_camera, ys, xs = projected[front], target_camera[front], ys[front], xs[front]
            px = np.rint(projected[:, 0] / projected[:, 2]).astype(int)
            py = np.rint(projected[:, 1] / projected[:, 2]).astype(int)
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px, py, z, ys, xs = px[inside], py[inside], target_camera[inside, 2], ys[inside], xs[inside]
            # Only fill behind the original foreground. Missing target depth
            # cannot establish this visibility relationship and stays a hole.
            eligible = target_mask[py, px] & target_valid[py, px] & (z > target_depth[py, px] * 1.02)
            px, py, z, ys, xs = px[eligible], py[eligible], z[eligible], ys[eligible], xs[eligible]
            # Closest reprojected background sample wins each pixel.
            order = np.argsort(z)
            flat = py[order] * width + px[order]
            _, first = np.unique(flat, return_index=True)
            selected = order[first]
            px, py, z, ys, xs = px[selected], py[selected], z[selected], ys[selected], xs[selected]
            closer = z < zbuffer[py, px]
            px, py, z, ys, xs = px[closer], py[closer], z[closer], ys[closer], xs[closer]
            edited[py, px], zbuffer[py, px], recovered[py, px] = rgb[ys, xs], z, True
        remaining = target_mask & ~recovered
        attempted_recovery_count = int(recovered.sum())
        if remaining.any():
            # Sparse reprojected samples are not a coherent revealed surface.
            # Edit the whole removal region instead of locking those samples
            # into a checkerboard of observed and generated pixels.
            edited = original.copy()
            remaining = target_mask.copy()
            recovered[:] = False
        image_path, mask_path = output_root / f"{index:06d}.png", output_root / f"{index:06d}.mask.png"
        recovered_path = output_root / f"{index:06d}.recovered.png"
        Image.fromarray(edited).save(image_path)
        Image.fromarray((remaining * 255).astype(np.uint8)).save(mask_path)
        Image.fromarray((recovered * 255).astype(np.uint8)).save(recovered_path)
        prepared.append({"frame_id": target_id, "image_path": str(image_path), "image_digest": _sha256_file(image_path),
                         "remaining_mask_path": str(mask_path), "remaining_mask_digest": _sha256_file(mask_path),
                         "recovered_mask_path": str(recovered_path), "recovered_mask_digest": _sha256_file(recovered_path),
                         "original_image_digest": target["image_digest"], "recovered_pixel_count": int(recovered.sum()),
                         "attempted_recovery_pixel_count": attempted_recovery_count,
                         "remaining_pixel_count": int(remaining.sum()), "placement_basis": "model_estimated_geometry",
                         "generated_pixels_present": False, "metric_measurement_proven": False})
    return prepared
