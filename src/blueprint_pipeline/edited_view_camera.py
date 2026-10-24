"""Deterministic check that an edited view kept its source camera.

The image editor receives a letterboxed frame plus a reference view and is told
to keep the original camera. It sometimes reframes the scene to fill the canvas
or copies the reference view's composition. The model review passed three such
views of six in the 2026-10-22 dishwasher concept run. A copied camera makes
Atlas pose an invented viewpoint as if it were the recorded one.

An edit may repaint the removal mask and, for shadows and reflections, a margin
around it. Everything else must stay where the original camera put it, so most
of the original's keypoints there must reappear within a few pixels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .local_reconstruction_adapters import _sha256_file

CHECK_REVISION = "edited_view_camera_v1"
# Excludes the shadows and reflections an edit may change next to the object.
MASK_MARGIN_FRACTION = 0.02  # of the image diagonal
MATCH_TOLERANCE_PIXELS = 3.0
MAX_LONG_SIDE = 1600
# Too few features outside the mask to judge: leave the view to the model review.
MIN_VERIFIABLE_KEYPOINTS = 25
# Kept fraction of the original's outside-mask keypoints. Measured 2026-10-24 on
# the archived dishwasher edits (720x1280): faithful edits kept 0.61-0.67 with
# SIFT and 0.58-0.61 with ORB; reframed or copied views kept 0.000-0.012.
MIN_KEPT_FRACTION = 0.25


def _detector():
    import cv2

    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=2000), cv2.NORM_L2
    return cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING


def edited_view_camera(original_path: Path, edited_path: Path, mask_path: Path) -> dict[str, Any]:
    """``kept``, ``moved`` or ``unverifiable``, from keypoints outside the dilated edit mask."""
    import cv2

    original, edited, mask = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                              for path in (original_path, edited_path, mask_path))
    result: dict[str, Any] = {"revision": CHECK_REVISION}
    if original is None or edited is None or mask is None:
        return {**result, "status": "unverifiable", "reason": "unreadable_image"}
    if edited.shape != original.shape or mask.shape != original.shape:
        return {**result, "status": "unverifiable", "reason": "size_mismatch"}
    scale = min(1.0, MAX_LONG_SIDE / max(original.shape))
    if scale < 1.0:
        size = (round(original.shape[1] * scale), round(original.shape[0] * scale))
        original, edited = (cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in (original, edited))
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    radius = max(3, round(MASK_MARGIN_FRACTION * math.hypot(*original.shape)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    keep = np.where(cv2.dilate((mask > 127).astype(np.uint8), kernel) == 0, 255, 0).astype(np.uint8)
    detector, norm = _detector()
    original_points, original_descriptors = detector.detectAndCompute(original, keep)
    result["keypoints"] = len(original_points)
    if len(original_points) < MIN_VERIFIABLE_KEYPOINTS:
        return {**result, "status": "unverifiable", "reason": "too_few_features_outside_mask"}
    edited_points, edited_descriptors = detector.detectAndCompute(edited, keep)
    kept = 0
    if edited_descriptors is not None and len(edited_points) >= 2:
        for pair in cv2.BFMatcher(norm).knnMatch(original_descriptors, edited_descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                a, b = original_points[pair[0].queryIdx].pt, edited_points[pair[0].trainIdx].pt
                kept += math.hypot(a[0] - b[0], a[1] - b[1]) < MATCH_TOLERANCE_PIXELS
    fraction = kept / len(original_points)
    return {**result, "status": "kept" if fraction >= MIN_KEPT_FRACTION else "moved",
            "kept_matches": kept, "kept_fraction": round(fraction, 4)}


def audit_prepared_background(path: Path) -> list[dict[str, Any]]:
    """Re-check every edited view a retained background approved. Reads files only."""
    saved = json.loads(Path(path).read_text())
    rows = []
    for frame in saved.get("background", saved)["frames"]:
        if not frame.get("generated_pixels_present") or not frame.get("remaining_mask_path"):
            rows.append({"frame_id": frame["frame_id"], "status": "not_edited"})
            continue
        paths = (Path(frame["original_image_path"]), Path(frame["image_path"]), Path(frame["remaining_mask_path"]))
        expected = (frame.get("original_image_digest"), frame.get("image_digest"), frame.get("remaining_mask_digest"))
        if any(digest and (not path.is_file() or _sha256_file(path) != digest) for path, digest in zip(paths, expected)):
            rows.append({"frame_id": frame["frame_id"], "status": "unverifiable", "reason": "retained_bytes_changed"})
            continue
        rows.append({"frame_id": frame["frame_id"], **edited_view_camera(*paths)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-check a retained prepared background's edited views "
                                                 "for a changed camera. No provider calls.")
    parser.add_argument("prepared_background", type=Path, help="prepared_background.json")
    rows = audit_prepared_background(parser.parse_args(argv).prepared_background)
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 2 if any(row["status"] == "moved" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
