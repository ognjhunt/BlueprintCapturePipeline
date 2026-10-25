"""Detect a coherent camera shift outside an object's removal mask.

Missing texture matches alone do not establish camera movement: an editor may
repaint wood grain while retaining cabinet edges. A distributed, robust feature
transform supplies positive movement evidence. Inconclusive views still need
the independent visual review, including rejection of a copied reference view.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .local_reconstruction_adapters import _sha256_file

CHECK_REVISION = "edited_view_camera_v2"
# Excludes the shadows and reflections an edit may change next to the object.
MASK_MARGIN_FRACTION = 0.02  # of the image diagonal
MATCH_TOLERANCE_PIXELS = 3.0
MAX_LONG_SIDE = 1600
# Too few features outside the mask to judge: leave the view to the model review.
MIN_VERIFIABLE_KEYPOINTS = 25
# Enough unchanged texture can establish a retained camera directly. Too little
# cannot establish movement without a coherent, spatially distributed transform.
MIN_KEPT_FRACTION = 0.25
MIN_TRANSFORM_INLIERS = 12
MIN_INLIER_FRACTION = 0.5
MIN_FEATURE_SPAN_FRACTION = 0.25
MAX_CAMERA_SHIFT_FRACTION = 0.025  # image diagonal; small local distortions may pass review


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
    matches = []
    if edited_descriptors is not None and len(edited_points) >= 2:
        for pair in cv2.BFMatcher(norm).knnMatch(original_descriptors, edited_descriptors, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                matches.append(pair[0])
    # Repeated grain and multiple SIFT orientations must not multiply one
    # correspondence into several independent pieces of geometric evidence.
    unique = []
    seen_original, seen_edited = set(), set()
    for match in sorted(matches, key=lambda item: item.distance):
        a, b = original_points[match.queryIdx].pt, edited_points[match.trainIdx].pt
        a_key, b_key = tuple(round(v) for v in a), tuple(round(v) for v in b)
        if a_key not in seen_original and b_key not in seen_edited:
            unique.append((a, b))
            seen_original.add(a_key)
            seen_edited.add(b_key)
    kept = sum(math.dist(a, b) < MATCH_TOLERANCE_PIXELS for a, b in unique)
    fraction = kept / len(original_points)
    result.update(kept_matches=kept, kept_fraction=round(fraction, 4), matched_features=len(unique))
    if fraction >= MIN_KEPT_FRACTION:
        return {**result, "status": "kept", "reason": "unchanged_features"}
    if len(unique) < MIN_TRANSFORM_INLIERS:
        return {**result, "status": "unverifiable", "reason": "too_few_shared_features"}
    source, destination = (np.asarray(points, dtype=np.float32) for points in zip(*unique))
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC, ransacReprojThreshold=MATCH_TOLERANCE_PIXELS)
    if transform is None or inlier_mask is None or not np.isfinite(transform).all():
        return {**result, "status": "unverifiable", "reason": "no_coherent_transform"}
    inliers = source[inlier_mask.ravel().astype(bool)]
    result["transform_inliers"] = len(inliers)
    if len(inliers) < MIN_TRANSFORM_INLIERS or len(inliers) / len(unique) < MIN_INLIER_FRACTION:
        return {**result, "status": "unverifiable", "reason": "insufficient_transform_inliers"}
    height, width = original.shape
    span = np.ptp(inliers, axis=0) / np.asarray([width, height])
    result["feature_span_fraction"] = [round(float(value), 4) for value in span]
    if float(span.min()) < MIN_FEATURE_SPAN_FRACTION:
        return {**result, "status": "unverifiable", "reason": "localized_shared_features"}
    grid = np.asarray([(x, y) for x in (0, width / 2, width - 1)
                       for y in (0, height / 2, height - 1)], dtype=np.float64)
    shifted = grid @ transform[:, :2].T + transform[:, 2]
    displacement = float(np.median(np.linalg.norm(shifted - grid, axis=1))) / math.hypot(height, width)
    result["camera_shift_fraction"] = round(displacement, 4)
    moved = displacement > MAX_CAMERA_SHIFT_FRACTION
    return {**result, "status": "moved" if moved else "kept",
            "reason": "coherent_camera_shift" if moved else "aligned_shared_features"}


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
    return 2 if any(row["status"] in {"moved", "unverifiable"} for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
