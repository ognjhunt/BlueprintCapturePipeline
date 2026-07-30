"""Adapter: tracked 2D detections + qualified depth -> perception-backend ``views``.

This closes the gap between the repo's existing perception models and the multi-view
perception path. SAM3 (``scripts/sam3_detect.py`` / ``object_index_sam3_runner.py``) emits
per-frame detection records; a geometry worker may emit per-frame depth arrays
(``depth_*.npy``, often downsampled vs the render). A depth array is diagnostic until its
source, exact payload, timing, calibration, pose, metric scale, and validation evidence pass
the qualification seam below. This module turns
those two raw outputs, for each rendered view, into the ``{detections, depth_provider, camera}``
dicts that :class:`MultiViewPerceptionSceneSpatialIndex` consumes — so:

    bounds -> view_ring_for_bounds -> render each view -> SAM3 + DA3 per view
           -> build_perception_views(...)  <-- THIS MODULE
           -> MultiViewPerceptionSceneSpatialIndex -> fused 3D catalog -> task->target->placement

The format-bridging is PURE and tested with synthetic records/maps — no torch, no GPU, no
network. The actual model inference is NOT baked in: it is *injected* (``detect`` / ``depth``
callables) because SAM3/DA3 invocation is worker-specific (which weights, which device). The
caller hands us callables that run their models; we handle the rest. The two converters
(:func:`detections_from_sam3`, :func:`depth_provider_from_map`) are the load-bearing,
fully-unit-tested geometry/format glue.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from typing import Callable, Dict, List, Mapping, Optional, Sequence

Camera = Mapping[str, object]

_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_METRIC_DEPTH_AUTHORITY_PROFILES = {
    "iphone_arkit_lidar",
    "calibrated_rgbd",
    "validated_metric_reconstruction",
}


def qualify_metric_depth_evidence(
    depth_evidence: Optional[Mapping[str, object]],
    *,
    observed_depth_payload_digest: Optional[str] = None,
) -> Dict[str, object]:
    """Validate the evidence that permits a depth map to drive metric placement.

    A numeric array is not metric authority by itself.  This seam requires the map,
    retained frame, source capture, camera calibration, pose, timing, scale, and
    validation metrics to be bound explicitly.  Unqualified maps remain useful for
    diagnostic visualization, but cannot silently produce placement geometry.
    """

    blockers: List[str] = []
    evidence = dict(depth_evidence or {})

    def require_digest(name: str) -> None:
        value = str(evidence.get(name) or "").strip().lower()
        if not _SHA256_REF.fullmatch(value):
            blockers.append(f"{name}_missing_or_invalid")

    for digest_field in (
        "source_capture_digest",
        "retained_frame_digest",
        "depth_digest",
        "camera_calibration_digest",
        "camera_pose_digest",
        "sync_map_row_digest",
    ):
        require_digest(digest_field)

    declared_payload_digest = str(evidence.get("depth_payload_digest") or "").strip().lower()
    if not _SHA256_REF.fullmatch(declared_payload_digest):
        blockers.append("depth_payload_digest_missing_or_invalid")
    observed_payload_digest = str(observed_depth_payload_digest or "").strip().lower()
    if not _SHA256_REF.fullmatch(observed_payload_digest):
        blockers.append("depth_payload_digest_unverified")
    elif declared_payload_digest != observed_payload_digest:
        blockers.append("depth_payload_digest_mismatch")

    profile = str(evidence.get("depth_authority_profile") or "").strip().lower()
    if profile not in _METRIC_DEPTH_AUTHORITY_PROFILES:
        blockers.append("depth_authority_profile_not_metric")
    if str(evidence.get("units") or "").strip().lower() != "meters":
        blockers.append("depth_units_not_meters")
    if str(evidence.get("depth_semantics") or "").strip().lower() != "z_depth":
        blockers.append("depth_semantics_not_z_depth")
    if evidence.get("metric_scale_verified") is not True:
        blockers.append("metric_scale_not_verified")

    for ref_field in ("frame_id", "camera_calibration_ref", "camera_pose_ref"):
        if not str(evidence.get(ref_field) or "").strip():
            blockers.append(f"{ref_field}_missing")

    try:
        pts = float(evidence.get("decoded_pts_seconds"))
        if not math.isfinite(pts) or pts < 0.0:
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("decoded_pts_seconds_missing_or_invalid")

    try:
        uncertainty = float(evidence.get("translation_uncertainty_m"))
        if not math.isfinite(uncertainty) or not 0.0 <= uncertainty <= 0.25:
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("translation_uncertainty_out_of_bounds")

    try:
        reprojection = float(evidence.get("reprojection_error_px"))
        if not math.isfinite(reprojection) or not 0.0 <= reprojection <= 5.0:
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("reprojection_error_out_of_bounds")

    if profile == "iphone_arkit_lidar" and not str(
        evidence.get("depth_confidence_ref") or ""
    ).strip():
        blockers.append("depth_confidence_ref_missing")
    if profile == "iphone_arkit_lidar":
        require_digest("depth_confidence_digest")
    if profile == "validated_metric_reconstruction" and not str(
        evidence.get("validation_report_ref") or ""
    ).strip():
        blockers.append("validation_report_ref_missing")
    if profile == "validated_metric_reconstruction":
        require_digest("validation_report_digest")

    return {
        "metric_placement_authorized": not blockers,
        "blockers": sorted(set(blockers)),
        "depth_authority_profile": profile or "undeclared",
        "source_capture_digest": str(evidence.get("source_capture_digest") or ""),
        "retained_frame_digest": str(evidence.get("retained_frame_digest") or ""),
        "depth_digest": str(evidence.get("depth_digest") or ""),
        "depth_payload_digest": declared_payload_digest,
        "observed_depth_payload_digest": observed_payload_digest,
        "frame_id": str(evidence.get("frame_id") or ""),
        "decoded_pts_seconds": evidence.get("decoded_pts_seconds"),
        "camera_calibration_ref": str(evidence.get("camera_calibration_ref") or ""),
        "camera_pose_ref": str(evidence.get("camera_pose_ref") or ""),
        "claim_boundary": {
            "two_dimensional_segmentation_is_not_metric_geometry": True,
            "numeric_depth_without_qualification_is_diagnostic_only": True,
            "physics_ready": False,
            "physical_task_success": False,
        },
    }


def depth_payload_digest(depth_map) -> str:
    """Return a deterministic digest of the exact depth payload consumed by this seam.

    Array-like values bind shape, dtype, and C-order bytes. Dependency-free nested sequences
    use canonical JSON with non-finite values rejected. This digest complements the retained
    file digest: it proves the loaded values used for geometry are the declared values.
    """

    shape = getattr(depth_map, "shape", None)
    dtype = getattr(depth_map, "dtype", None)
    tobytes = getattr(depth_map, "tobytes", None)
    if shape is not None and dtype is not None and callable(tobytes):
        header = json.dumps(
            {"shape": [int(value) for value in shape], "dtype": str(dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = tobytes(order="C")
        return "sha256:" + hashlib.sha256(header + b"\n" + payload).hexdigest()
    encoded = json.dumps(
        depth_map,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _cam_get(camera: Camera, name: str):
    if isinstance(camera, Mapping):
        return camera.get(name)
    return getattr(camera, name, None)


def _first(record: Mapping[str, object], *names: str):
    """First present, non-None value among ``names`` (tolerates SAM3 field-name variants)."""
    for n in names:
        if n in record and record[n] is not None:
            return record[n]
    return None


def detections_from_sam3(
    records: Sequence[Mapping[str, object]],
    *,
    width: int,
    height: int,
) -> List[Dict[str, object]]:
    """Convert SAM3 detection records to perception ``{label, bbox_px, confidence}`` dicts.

    Robust to SAM3's field-name + coordinate variants (the runner just passes records through
    verbatim, so the schema isn't fixed):
      * box under ``bbox_px`` / ``bbox_xyxy`` / ``bbox`` / ``box`` / ``xyxy`` (x0,y0,x1,y1);
      * coordinates either in PIXELS or NORMALIZED [0,1] — detected by whether every component
        is within [0,1], then scaled by ``width``/``height``;
      * label under ``label`` / ``prompt`` / ``class`` / ``name``;
      * score under ``confidence`` / ``score`` / ``conf`` (defaults to 1.0).
    Boxes are normalized to (min_x, min_y, max_x, max_y) so a flipped detection still bounds
    correctly. Malformed records (missing/!=4-length box) are skipped, never fatal.
    """
    out: List[Dict[str, object]] = []
    for i, r in enumerate(records):
        if not isinstance(r, Mapping):
            continue
        box = _first(r, "bbox_px", "bbox_xyxy", "bbox", "box", "xyxy")
        if box is None:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in box)  # type: ignore[misc]
        except (TypeError, ValueError):
            continue
        # Normalized -> pixels when every coord sits within the unit square.
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0 + 1e-9:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        bbox_px = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        label = _first(r, "label", "prompt", "class", "name")
        conf = _first(r, "confidence", "score", "conf")
        det: Dict[str, object] = {
            "label": str(label or ""),
            "bbox_px": bbox_px,
            "confidence": float(conf) if conf is not None else 1.0,
        }
        rid = _first(r, "id", "object_id", "track_id")
        if rid is not None:
            det["id"] = str(rid)
        cat = _first(r, "category", "class")
        if cat is not None:
            det["category"] = str(cat)
        out.append(det)
    return out


def _map_dims(depth_map) -> tuple:
    """(rows, cols) of a depth map — numpy array (``.shape``) or list-of-lists."""
    shape = getattr(depth_map, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    rows = len(depth_map)
    cols = len(depth_map[0]) if rows else 0
    return rows, cols


def depth_provider_from_map(
    depth_map,
    *,
    cam_width: Optional[int] = None,
    cam_height: Optional[int] = None,
) -> Callable[[float, float], float]:
    """Wrap a HxW numeric depth array as a sampler for the backend.

    Nearest-neighbour sample with clamping. CRUCIALLY handles a depth map whose resolution
    differs from the render the detections came from (DA3 frequently runs downsampled): camera
    pixel ``(px, py)`` is scaled by ``(cols/cam_width, rows/cam_height)`` before indexing, so the
    box pixels and the depth lattice line up regardless of resolution. ``depth_map`` may be a
    numpy array or a list-of-lists. Qualification is intentionally outside this low-level
    sampler: only :func:`build_perception_view` may authorize values as metric z-depth after
    checking source-bound evidence. Out-of-range pixels clamp to the edge.
    """
    rows, cols = _map_dims(depth_map)

    def provider(px: float, py: float) -> float:
        if rows == 0 or cols == 0:
            raise ValueError("empty depth map")
        sx = cols / float(cam_width) if cam_width else 1.0
        sy = rows / float(cam_height) if cam_height else 1.0
        col = int(round(px * sx))
        row = int(round(py * sy))
        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)
        return float(depth_map[row][col])

    return provider


def build_perception_view(
    camera: Camera,
    sam3_records: Sequence[Mapping[str, object]],
    depth_map,
    *,
    samples_per_axis: int = 3,
    depth_evidence: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Assemble one ``{detections, depth_provider, camera}`` view from raw SAM3 + DA3 outputs.

    Reads ``width``/``height`` from ``camera`` (the render resolution) to convert boxes + scale
    depth sampling. ``samples_per_axis`` is carried through for the single-view backend's robust
    median depth (passed via the view's camera-agnostic plumbing by the caller if needed).
    """
    width = int(_cam_get(camera, "width") or 0)
    height = int(_cam_get(camera, "height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("camera must carry positive 'width'/'height' (the render resolution)")
    observed_depth_payload_digest = depth_payload_digest(depth_map)
    qualification = qualify_metric_depth_evidence(
        depth_evidence,
        observed_depth_payload_digest=observed_depth_payload_digest,
    )
    return {
        "detections": detections_from_sam3(sam3_records, width=width, height=height),
        "depth_provider": depth_provider_from_map(depth_map, cam_width=width, cam_height=height),
        "camera": camera,
        "samples_per_axis": samples_per_axis,
        "metric_placement_authorized": qualification["metric_placement_authorized"],
        "depth_qualification": qualification,
    }


def build_perception_views(
    cameras: Sequence[Camera],
    sam3_records_per_view: Sequence[Sequence[Mapping[str, object]]],
    depth_maps: Sequence,
    *,
    samples_per_axis: int = 3,
    depth_evidence_per_view: Optional[Sequence[Mapping[str, object]]] = None,
) -> List[Dict[str, object]]:
    """Zip per-view cameras + SAM3 records + DA3 depth maps into the fusion ``views`` list.

    The three sequences must be equal length (one entry per rendered view); a mismatch is a
    caller bug and raises rather than silently dropping views.
    """
    if not (len(cameras) == len(sam3_records_per_view) == len(depth_maps)):
        raise ValueError(
            "cameras, sam3_records_per_view, and depth_maps must have equal length "
            f"(got {len(cameras)}, {len(sam3_records_per_view)}, {len(depth_maps)})"
        )
    if depth_evidence_per_view is not None and len(depth_evidence_per_view) != len(cameras):
        raise ValueError(
            "depth_evidence_per_view must be absent or have one entry per camera "
            f"({len(depth_evidence_per_view)} != {len(cameras)})"
        )
    return [
        build_perception_view(cameras[i], sam3_records_per_view[i], depth_maps[i],
                              samples_per_axis=samples_per_axis,
                              depth_evidence=(
                                  depth_evidence_per_view[i]
                                  if depth_evidence_per_view is not None
                                  else None
                              ))
        for i in range(len(cameras))
    ]


def build_perception_views_from_frames(
    frames: Sequence,
    cameras: Sequence[Camera],
    *,
    detect: Callable[[object], Sequence[Mapping[str, object]]],
    depth: Callable[[object], object],
    depth_evidence: Optional[Callable[[object, object], Mapping[str, object]]] = None,
    samples_per_axis: int = 3,
) -> List[Dict[str, object]]:
    """Run injected SAM3 + DA3 callables over each rendered frame, then assemble fusion views.

    ``detect(frame) -> sam3_records`` and ``depth(frame) -> depth_map`` are INJECTED — they wrap
    whatever SAM3/DA3 the worker has (weights, device). Keeping them injected is what lets this
    module stay pure + unit-tested (tests pass deterministic fakes) while real runs pass GPU model
    calls. ``frames`` and ``cameras`` must be equal length (one rendered frame per view camera).
    """
    if len(frames) != len(cameras):
        raise ValueError(f"frames and cameras must have equal length ({len(frames)} != {len(cameras)})")
    records = [list(detect(f)) for f in frames]
    maps = [depth(f) for f in frames]
    evidence = (
        [depth_evidence(frames[index], maps[index]) for index in range(len(frames))]
        if depth_evidence is not None
        else None
    )
    return build_perception_views(
        cameras,
        records,
        maps,
        samples_per_axis=samples_per_axis,
        depth_evidence_per_view=evidence,
    )


__all__ = [
    "detections_from_sam3",
    "depth_provider_from_map",
    "depth_payload_digest",
    "build_perception_view",
    "build_perception_views",
    "build_perception_views_from_frames",
    "qualify_metric_depth_evidence",
]
