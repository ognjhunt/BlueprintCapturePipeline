"""Use exactly the observed SAM silhouette as the editable region.

Calibrated projections remain provenance, not a substitute object silhouette.
No guard band, shadow extension, or projected-box fallback is added. Missing
or full-frame SAM masks need correction before any image-edit request.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageChops

from .decision_evidence_contracts import canonical_digest

REPAIR_SUPPORT_POLICY = "sam_silhouette_exact_no_fallback_v4"
GUARD_BAND_PIXELS = 0
CALIBRATED_REACH_PIXELS = 0


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def _binary(path: Path) -> Image.Image:
    with Image.open(path) as image:
        result = image.convert("L")
    if any(result.histogram()[1:255]):
        raise ValueError("scene_configuration_repair_support_mask_nonbinary")
    return result


def materialize_repair_support(*, calibrated_mask_path: Path, sam_mask_path: Path,
                               source_frame_path: Path, calibration_digest: str,
                               output_root: Path) -> dict:
    calibrated, sam = _binary(calibrated_mask_path), _binary(sam_mask_path)
    with Image.open(source_frame_path) as image:
        size = image.size
    if calibrated.size != size or sam.size != size:
        raise ValueError("scene_configuration_repair_support_shape_invalid")
    count = sam.histogram()[255]
    if not count:
        raise ValueError("scene_configuration_repair_support_sam_core_missing")
    if count == size[0] * size[1]:
        raise ValueError("scene_configuration_repair_support_sam_core_full_frame")
    if output_root.exists():
        raise ValueError("scene_configuration_repair_support_output_exists")
    output_root.mkdir(parents=True)
    core_path, support_path = output_root / "object-core.png", output_root / "repair-support.png"
    sam.save(core_path)
    sam.save(support_path)
    provenance = {
        "policy": REPAIR_SUPPORT_POLICY,
        "source_frame": _record(source_frame_path),
        "calibrated_object_mask": _record(calibrated_mask_path),
        "raw_sam_mask": _record(sam_mask_path),
        "calibration_digest": calibration_digest,
        "guard_band_pixels": 0, "calibrated_reach_pixels": 0,
        "sam_silhouette_pixel_count": count,
        "calibrated_pixels_beyond_reach_dropped": ImageChops.subtract(calibrated, sam).histogram()[255],
        "guard_band_pixels_offered": 0, "guard_band_pixels_admitted": 0,
        "calibrated_projection_fallback_used": False,
        "observed_segmentation_truth": False,
        "sam_ownership_evidence_modified": False,
        "whole_object_coverage_visually_qualified": False,
        "protected_object_preservation_visually_qualified": False,
    }
    return {
        "repair_object_core": {**_record(core_path), "provenance": provenance},
        "repair_support_mask": {**_record(support_path), "provenance": provenance},
        "derivation_digest": canonical_digest(provenance, digest_field="derivation_digest"),
    }
