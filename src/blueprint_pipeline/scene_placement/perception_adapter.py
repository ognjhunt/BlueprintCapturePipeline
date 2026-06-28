"""Adapter: real SAM3 detections + DA3 depth maps -> perception-backend ``views``.

This closes the gap between the repo's existing perception models and the multi-view
perception path. SAM3 (``scripts/sam3_detect.py`` / ``object_index_sam3_runner.py``) emits
per-frame detection records; DA3 (``geometry_stage.py``) emits per-frame metric depth maps
(``depth_*.npy``, HxW float32 meters, often downsampled vs the render). This module turns
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

from typing import Callable, Dict, List, Mapping, Optional, Sequence

Camera = Mapping[str, object]


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
    """Wrap a HxW metric depth map as a ``depth_provider(px, py) -> meters`` for the backend.

    Nearest-neighbour sample with clamping. CRUCIALLY handles a depth map whose resolution
    differs from the render the detections came from (DA3 frequently runs downsampled): camera
    pixel ``(px, py)`` is scaled by ``(cols/cam_width, rows/cam_height)`` before indexing, so the
    box pixels and the depth lattice line up regardless of resolution. ``depth_map`` may be a
    numpy array or a list-of-lists; values are metric meters along the optical axis (DA3 metric
    semantics — matching :func:`unproject`'s contract). Out-of-range pixels clamp to the edge.
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
    return {
        "detections": detections_from_sam3(sam3_records, width=width, height=height),
        "depth_provider": depth_provider_from_map(depth_map, cam_width=width, cam_height=height),
        "camera": camera,
        "samples_per_axis": samples_per_axis,
    }


def build_perception_views(
    cameras: Sequence[Camera],
    sam3_records_per_view: Sequence[Sequence[Mapping[str, object]]],
    depth_maps: Sequence,
    *,
    samples_per_axis: int = 3,
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
    return [
        build_perception_view(cameras[i], sam3_records_per_view[i], depth_maps[i],
                              samples_per_axis=samples_per_axis)
        for i in range(len(cameras))
    ]


def build_perception_views_from_frames(
    frames: Sequence,
    cameras: Sequence[Camera],
    *,
    detect: Callable[[object], Sequence[Mapping[str, object]]],
    depth: Callable[[object], object],
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
    return build_perception_views(cameras, records, maps, samples_per_axis=samples_per_axis)


__all__ = [
    "detections_from_sam3",
    "depth_provider_from_map",
    "build_perception_view",
    "build_perception_views",
    "build_perception_views_from_frames",
]
