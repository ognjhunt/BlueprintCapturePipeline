"""Derive candidate 2D repair support without changing SAM ownership evidence.

The support is what the image editor may repaint and what the locality seal
keeps from its output. Attempt #20 of InteriorGS 840938 (2026-09-13) showed
the previous rule handing the editor the neighbour: the calibrated object mask
is the projection of the padded 3D cutout, three times the observed SAM
silhouette for the vase in view source-08, and it reached across the bottle
standing beside it; a flat 32-pixel guard band then grew that region further
with no regard for what the pixels were. The seal pasted generated pixels over
part of the bottle and the reviewer rejected the frame for exactly that.

Two rules now bound the support:

* calibrated coverage counts only within ``CALIBRATED_REACH_PIXELS`` of the
  observed silhouette (it exists to catch the parts SAM missed at the object's
  own boundary, never to annex what stands nearby); a view SAM missed entirely
  keeps the whole calibrated mask, as before;
* the guard band admits a pixel only when it looks like the supporting surface
  under the object's contact shadow (luminance inside a band around the
  surface reference measured on the band itself, darker allowed, brighter
  barely) and it connects to the object core. A pale neighbour is neither.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from scipy.ndimage import label, maximum_filter

from .decision_evidence_contracts import canonical_digest

REPAIR_SUPPORT_POLICY = "calibrated_object_near_sam_core_surface_band_guard_v2"
GUARD_BAND_PIXELS = 32
CALIBRATED_REACH_PIXELS = 24
SHADOW_LUMINANCE_FLOOR_FRACTION = 0.62
HIGHLIGHT_LUMINANCE_CEILING_FRACTION = 1.06


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return maximum_filter(mask, size=2 * radius + 1, mode="constant", cval=0)


def _constrain_guard_band(*, core: np.ndarray, guard: np.ndarray, luminance: np.ndarray) -> tuple[np.ndarray, dict]:
    """Admit guard-band pixels that look like the object's supporting surface and touch the core."""
    ring = guard & ~core
    if not ring.any():
        return core.copy(), {"guard_band_rule": "supporting_surface_luminance_band_connected_to_core",
                             "surface_reference_luminance": None, "guard_band_pixels_offered": 0,
                             "guard_band_pixels_admitted": 0}
    reference = float(np.median(luminance[ring]))
    floor = reference * SHADOW_LUMINANCE_FLOOR_FRACTION
    ceiling = reference * HIGHLIGHT_LUMINANCE_CEILING_FRACTION
    admitted = ring & (luminance >= floor) & (luminance <= ceiling)
    labelled, _ = label(admitted | core)
    touching = np.unique(labelled[core])
    reachable = np.isin(labelled, touching[touching > 0]) & admitted
    support = core | reachable
    return support, {"guard_band_rule": "supporting_surface_luminance_band_connected_to_core",
                     "surface_reference_luminance": round(reference, 3), "luminance_floor": round(floor, 3),
                     "luminance_ceiling": round(ceiling, 3), "guard_band_pixels_offered": int(ring.sum()),
                     "guard_band_pixels_admitted": int(reachable.sum())}


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
        luminance = np.asarray(image.convert("L"), dtype=np.float64)
    if calibrated.size != size or sam.size != size:
        raise ValueError("scene_configuration_repair_support_shape_invalid")
    if ImageChops.lighter(calibrated, sam).getbbox() is None:
        raise ValueError("scene_configuration_repair_support_core_missing")
    if output_root.exists():
        raise ValueError("scene_configuration_repair_support_output_exists")
    sam_array = np.asarray(sam) > 0
    calibrated_array = np.asarray(calibrated) > 0
    if sam_array.any():
        near = _dilate(sam_array.astype(np.uint8), CALIBRATED_REACH_PIXELS) > 0
        core_array = sam_array | (calibrated_array & near)
    else:  # SAM missed this view entirely: the calibrated projection is the only object evidence
        core_array = calibrated_array
    calibrated_dropped = int((calibrated_array & ~core_array).sum())
    core = Image.fromarray(np.where(core_array, 255, 0).astype(np.uint8), mode="L")
    guard = _dilate(core_array.astype(np.uint8), GUARD_BAND_PIXELS) > 0
    support_array, guard_record = _constrain_guard_band(core=core_array, guard=guard, luminance=luminance)
    support = Image.fromarray(np.where(support_array, 255, 0).astype(np.uint8), mode="L")
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
        "calibrated_reach_pixels": CALIBRATED_REACH_PIXELS,
        "calibrated_pixels_beyond_reach_dropped": calibrated_dropped,
        **guard_record,
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
