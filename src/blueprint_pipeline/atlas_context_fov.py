"""Verify the pixels retained by pose, before buying generated scene views.

A requested output aspect does not prove that the provider retained the input
field of view. Registration here measures the actual returned image against
the exact source; it never changes the native camera, depth or source bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest

MIN_SOURCE_AREA_RETAINED = .99


def _audit_input_binding(inputs, sources, frames):
    """Exact pixels and implementation behind a reusable, passed CPU audit."""
    import cv2
    from PIL import __version__ as pillow_version

    if any(not isinstance(frame.get("rgb"), np.ndarray) or frame["rgb"].ndim != 3
           or frame["rgb"].shape[2] != 3 for frame in frames):
        raise ValueError("atlas_context_rgb_grid_invalid")
    return {"schema_version": "atlas_context_fov_inputs.v1",
            "source_pose_inputs_digest": inputs["digest"],
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {"opencv": cv2.__version__, "numpy": np.__version__, "pillow": pillow_version},
            "minimum_retained_source_area_fraction": MIN_SOURCE_AREA_RETAINED,
            "sources": [{"frame_id": row["frame_id"], "image_digest": row["image_digest"]} for row in sources],
            "returned": [{"frame_id": frame["frame_id"], "shape": list(frame["rgb"].shape),
                          "dtype": frame["rgb"].dtype.str,
                          "pixel_sha256": hashlib.sha256(np.ascontiguousarray(frame["rgb"]).tobytes()).hexdigest()}
                         for frame in frames]}


def square_pad(source: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep every source pixel; add neutral padding rather than crop to a grid."""
    height, width = source.shape[:2]
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    padded = np.full((side, side, 3), 127, np.uint8)
    padded[top:top + height, left:left + width] = source
    return padded, {"source_rect": [left, top, width, height], "padded_size": [side, side],
                    "source_pixels_unchanged": True}


def full_view_sources(inputs: dict, root: Path) -> list:
    """Verify retained sources and the exact padding bought by this pose input."""
    from PIL import Image
    from .local_reconstruction_adapters import _sha256_file

    sources = inputs.get("full_view_sources")
    if sources is None:
        return inputs["frames"]
    if ([r.get("frame_id") for r in sources] != [r["frame_id"] for r in inputs["frames"]]):
        raise ValueError("atlas_context_source_binding_invalid")
    for source, padded in zip(sources, inputs["frames"], strict=True):
        path = Path(source["image_path"])
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or _sha256_file(path) != source["image_digest"]):
            raise ValueError("atlas_context_source_changed")
        with Image.open(path) as image:
            expected, transform = square_pad(np.asarray(image.convert("RGB")))
        with Image.open(padded["image_path"]) as image:
            actual = np.asarray(image.convert("RGB"))
        if source.get("padding") != transform or not np.array_equal(actual, expected):
            raise ValueError("atlas_context_padding_changed")
    return sources


def image_coverage(source: np.ndarray, returned: np.ndarray) -> dict:
    """Measure a same-axis scale/crop, refusing an unsupported pixel mapping."""
    import cv2

    if source.ndim != 3 or returned.ndim != 3 or source.shape[2] != 3 or returned.shape[2] != 3:
        raise ValueError("atlas_context_rgb_grid_invalid")
    direct = (float(np.abs(source.astype(float) - returned.astype(float)).mean())
              if source.shape == returned.shape else math.inf)
    if direct <= 2:
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
        inliers, residual = None, 0.
    else:
        detector = cv2.SIFT_create(nfeatures=4000)
        ka, da = detector.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), None)
        kb, db = detector.detectAndCompute(cv2.cvtColor(returned, cv2.COLOR_RGB2GRAY), None)
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            raise ValueError("atlas_context_pixel_mapping_unproven")
        pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
        if len(good) < 8:
            raise ValueError("atlas_context_pixel_mapping_unproven")
        a = np.float32([ka[m.queryIdx].pt for m in good])
        b = np.float32([kb[m.trainIdx].pt for m in good])
        cv2.setRNGSeed(0)
        matrix, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
            ransacReprojThreshold=2, maxIters=10000, confidence=.999)
        if matrix is None or mask is None or int(mask.sum()) < 8:
            raise ValueError("atlas_context_pixel_mapping_unproven")
        angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        if abs(angle) > .1:
            raise ValueError("atlas_context_pixel_axes_changed")
        chosen = mask.ravel().astype(bool)
        residual = float(np.median(np.linalg.norm(cv2.transform(a[None], matrix)[0] - b, axis=1)[chosen]))
        inliers = int(chosen.sum())
        if residual > 1:
            raise ValueError("atlas_context_pixel_mapping_unproven")
    height, width = returned.shape[:2]
    reprojection = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones(source.shape[:2], np.uint8), matrix, (width, height),
                           flags=cv2.INTER_NEAREST).astype(bool)
    if not valid.any():
        raise ValueError("atlas_context_pixel_mapping_unproven")
    error = float(np.abs(reprojection.astype(float) - returned.astype(float))[valid].mean())
    # Pose legitimately down/up samples RGB. Compare band-limited colour for
    # content agreement; registration and retained bounds independently check
    # framing. Raw differences remain in the receipt rather than being hidden.
    blurred_a = cv2.GaussianBlur(reprojection.astype(np.float32), (0, 0), 1.5)
    blurred_b = cv2.GaussianBlur(returned.astype(np.float32), (0, 0), 1.5)
    content_error = float(np.abs(blurred_a - blurred_b)[valid].mean())
    if content_error > 12:
        raise ValueError("atlas_context_pixels_changed")
    corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
    retained = cv2.transform(corners, cv2.invertAffineTransform(matrix))[0]
    sh, sw = source.shape[:2]
    source_bounds = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    area, _ = cv2.intersectConvexConvex(retained, source_bounds)
    fraction = min(1., max(0., float(area) / (sw * sh)))
    return {"source_shape": list(source.shape), "returned_shape": list(returned.shape),
            "source_to_returned_affine": matrix.tolist(), "inlier_count": inliers,
            "median_registration_error_pixels": residual, "mean_pixel_difference": error,
            "low_pass_mean_pixel_difference": content_error,
            "retained_source_polygon": retained.tolist(), "retained_source_area_fraction": fraction,
            "full_source_field_of_view_preserved": fraction >= MIN_SOURCE_AREA_RETAINED}


def verify_pose_context_fov(*, pose_root: Path, frames: list, output_root: Path) -> dict:
    """Retain the audit even on refusal. No uploads, model calls or pixel edits."""
    from PIL import Image
    from .website_atlas_pose import load_pending_atlas_pose_inputs

    inputs = load_pending_atlas_pose_inputs(pose_root)
    # The canonical loader already verifies every input/source byte and padding
    # transform. Do not decode and compare the same padded images a second time.
    sources = inputs.get("full_view_sources") or inputs["frames"]
    by_id = {row["frame_id"]: row for row in sources}
    path = output_root / "context-fov-audit.json"
    previous, previous_bytes = None, None
    if path.is_symlink():
        raise ValueError("atlas_context_audit_invalid")
    if path.is_file():
        previous_bytes = path.read_bytes()
        previous = json.loads(previous_bytes)
        if (not isinstance(previous, dict)
                or previous.get("schema_version") != "atlas_context_fov_audit.v1"
                or previous.get("digest") != canonical_digest(previous, digest_field="digest")):
            raise ValueError("atlas_context_audit_invalid")
    rows = []
    binding = None
    try:
        if [frame["frame_id"] for frame in frames] != list(by_id):
            raise ValueError("atlas_context_source_binding_invalid")
        binding = _audit_input_binding(inputs, sources, frames)
        if previous and previous.get("status") == "passed" and previous.get("input_binding") == binding:
            prior_rows = previous.get("frames") or []
            if (previous.get("source_pose_inputs_digest") != inputs["digest"]
                    or type(previous.get("provider_calls")) is not int
                    or previous.get("provider_calls") != 0
                    or previous.get("minimum_retained_source_area_fraction") != MIN_SOURCE_AREA_RETAINED
                    or [row.get("frame_id") for row in prior_rows] != list(by_id)
                    or any(row.get("source_image_digest") != by_id[row["frame_id"]]["image_digest"]
                           or row.get("full_source_field_of_view_preserved") is not True
                           or type(row.get("retained_source_area_fraction")) not in (int, float)
                           or not math.isfinite(row["retained_source_area_fraction"])
                           or row["retained_source_area_fraction"] < MIN_SOURCE_AREA_RETAINED
                           for row in prior_rows)):
                raise ValueError("atlas_context_audit_invalid")
            return previous
        for frame in frames:
            source = by_id[frame["frame_id"]]
            with Image.open(source["image_path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            rows.append({"frame_id": frame["frame_id"], "source_image_digest": source["image_digest"],
                         **image_coverage(rgb, frame["rgb"])})
        passed = bool(rows) and all(row["full_source_field_of_view_preserved"] for row in rows)
        report = {"schema_version": "atlas_context_fov_audit.v1", "status": "passed" if passed else "refused",
                  "source_pose_inputs_digest": inputs["digest"], "frames": rows,
                  "minimum_retained_source_area_fraction": MIN_SOURCE_AREA_RETAINED,
                  "provider_calls": 0}
    except ValueError as error:
        report = {"schema_version": "atlas_context_fov_audit.v1", "status": "refused", "reason": str(error),
                  "source_pose_inputs_digest": inputs["digest"], "frames": rows, "provider_calls": 0}
    if binding is not None:
        report["input_binding"] = binding
    report["digest"] = canonical_digest(report, digest_field="digest")
    if previous_bytes is not None:
        # A legacy audit has no posed-pixel binding: recompute it once, retaining
        # its exact original bytes rather than retroactively asserting one.
        from .launch_immutable_input_writer import write_exclusive_private_bytes
        history = output_root / "context-fov-audit-history" / (hashlib.sha256(previous_bytes).hexdigest() + ".json")
        write_exclusive_private_bytes(history, previous_bytes)
    write_json(path, report)
    if report["status"] != "passed":
        raise ValueError("atlas_context_field_of_view_lost")
    return report
