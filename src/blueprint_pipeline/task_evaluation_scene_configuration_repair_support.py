"""Derive candidate 2D repair support without changing SAM ownership evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from scipy.ndimage import maximum_filter

from .decision_evidence_contracts import canonical_digest

REPAIR_SUPPORT_POLICY = "calibrated_object_plus_sam_core_bounded_guard_band_v1"
GUARD_BAND_PIXELS = 32


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
    """Bind a candidate object core and margin to existing calibrated inputs.

    The margin is a repair allowance, not segmentation truth. Independent
    visual review must still establish complete removal and no collateral edit.
    """
    calibrated = _binary(calibrated_mask_path)
    sam = _binary(sam_mask_path)
    with Image.open(source_frame_path) as image:
        size = image.size
    if calibrated.size != size or sam.size != size:
        raise ValueError("scene_configuration_repair_support_shape_invalid")
    core = ImageChops.lighter(calibrated, sam)
    if core.getbbox() is None:
        raise ValueError("scene_configuration_repair_support_core_missing")
    if output_root.exists():
        raise ValueError("scene_configuration_repair_support_output_exists")
    support = Image.fromarray(maximum_filter(np.asarray(core), size=2 * GUARD_BAND_PIXELS + 1, mode="constant", cval=0))
    output_root.mkdir(parents=True)
    core_path = output_root / "object-core.png"
    support_path = output_root / "repair-support.png"
    core.save(core_path)
    support.save(support_path)
    provenance = {
        "policy": REPAIR_SUPPORT_POLICY,
        "source_frame": _record(source_frame_path),
        "calibrated_object_mask": _record(calibrated_mask_path),
        "raw_sam_mask": _record(sam_mask_path),
        "calibration_digest": calibration_digest,
        "guard_band_pixels": GUARD_BAND_PIXELS,
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
