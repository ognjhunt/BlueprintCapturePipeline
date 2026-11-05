"""Cut a subject the geometry-only route left in the reconstruction out of the collider.

A geometry-only website job (``website_atlas_pipeline.ROUTE_ENV``) keeps the scene
as filmed, so the subject's own surfaces are in the collider, and the surface under
it was never seen: the subject hid it. Before compile looks anything up on the
collider, the subject's box, widened by a margin, is cut out from just below the
surface around it to above its top, and the cut is filled at that surface's height.
Support lookups and the robot fit check then meet the surface the subject stands
on, never the subject.

The box comes from the video track and the height from the faces around it: both
are estimates, and the subject is not modelled. Like collider conditioning
(``website_collider_conditioning``), the cut collider is written once, bound by
digest and sealed into the base scene before compile binds
``collision_mesh_digest``; a retry reuses it. When no surface is found around the
subject, nothing is cut, and the fit check is not run on a collider that still
holds the subject.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core.handoff_lease import HandoffLeaseError, handoff_effect
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

REVISION = "website_subject_excision.v1"
COLLIDER_NAME, RECEIPT_NAME = "collider_subject_excised.glb", "subject_excision.json"
#: Sideways, metres: the box is a video estimate.
SIDE_MARGIN_M = 0.03
#: Cut above the box's top, metres.
TOP_MARGIN_M = 0.05
#: Cut below the surrounding surface, metres, so no bump of it under the subject
#: stands above the fill (Atlas counters undulate 1-3 cm).
BELOW_SUPPORT_M = 0.03
#: Width of the ring around the box whose faces give the surface's height, metres.
RING_M = 0.10
#: Faces steeper than this are walls, never the surface.
RING_MAX_TILT_DEGREES = 30.0
#: With fewer ring faces than this, no surface is assumed and nothing is cut.
RING_MIN_FACES = 3


def surrounding_support_height(triangles: np.ndarray, *, low: Sequence[float], high: Sequence[float], up: int,
                               up_sign: int, bottom_height: float, window: float,
                               meters_per_unit: float) -> tuple[float, int] | None:
    """The height (along ``up_sign``) of the surface around a footprint, and the faces it rests on.

    Nearly level faces in a ring around the footprint, within ``window`` of its
    estimated bottom; like a support lookup (``website_support_geometry``), the
    band of the highest surface reached, so a counter's lower edge never wins.
    """
    horizontal = [axis for axis in range(3) if axis != up]
    centres = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    level = np.zeros(len(triangles), dtype=bool)
    nonzero = lengths > 0
    level[nonzero] = np.abs(normals[nonzero, up]) / lengths[nonzero] >= np.cos(np.deg2rad(RING_MAX_TILT_DEGREES))
    ring = RING_M / meters_per_unit
    near, inside = np.ones(len(triangles), dtype=bool), np.ones(len(triangles), dtype=bool)
    for axis in horizontal:
        near &= (centres[:, axis] >= low[axis] - ring) & (centres[:, axis] <= high[axis] + ring)
        inside &= (centres[:, axis] >= low[axis]) & (centres[:, axis] <= high[axis])
    heights = up_sign * centres[:, up]
    seed = level & near & ~inside & (np.abs(heights - bottom_height) <= window)
    if int(seed.sum()) < RING_MIN_FACES:
        return None
    top = np.percentile(heights[seed], 90)
    band = seed & (np.abs(heights - top) <= BELOW_SUPPORT_M / meters_per_unit)
    return float(np.median(heights[band])), int(band.sum())


def _fill(pieces: np.ndarray, *, low: np.ndarray, high: np.ndarray, up: int, up_sign: int,
          height: float, band: float) -> np.ndarray:
    """Triangles closing the cut at ``height``: a fan from the centre to where the surface meets the walls.

    The fan reuses the clipped pieces' own vertices on the box walls, so the
    filled patch is one connected surface with the surface around it.
    """
    horizontal = [axis for axis in range(3) if axis != up]
    span = float(np.max(high - low))
    eps = 1e-9 * max(span, 1.0)
    points = pieces.reshape(-1, 3)
    on_wall = np.zeros(len(points), dtype=bool)
    for first, second in (horizontal, horizontal[::-1]):
        on_wall |= ((np.abs(points[:, first] - low[first]) <= eps) | (np.abs(points[:, first] - high[first]) <= eps)) & (
            (points[:, second] >= low[second] - eps) & (points[:, second] <= high[second] + eps))
    boundary = points[on_wall & (np.abs(up_sign * points[:, up] - height) <= band)]
    rows: dict[tuple[float, float], np.ndarray] = {}
    for point in boundary:
        # One vertex per wall position: the one nearest the surface's height.
        key = (round(float(point[horizontal[0]]), 9), round(float(point[horizontal[1]]), 9))
        if key not in rows or abs(up_sign * point[up] - height) < abs(up_sign * rows[key][up] - height):
            rows[key] = point
    for x in (low[horizontal[0]], high[horizontal[0]]):
        for y in (low[horizontal[1]], high[horizontal[1]]):
            if not any(abs(key[0] - x) <= eps and abs(key[1] - y) <= eps for key in rows):
                corner = np.zeros(3)
                corner[horizontal[0]], corner[horizontal[1]], corner[up] = x, y, up_sign * height
                rows[(float(x), float(y))] = corner
    ring = np.asarray(list(rows.values()), dtype=float)
    centre = np.zeros(3)
    centre[horizontal] = (low[horizontal] + high[horizontal]) / 2
    centre[up] = up_sign * height
    angles = np.arctan2(ring[:, horizontal[1]] - centre[horizontal[1]], ring[:, horizontal[0]] - centre[horizontal[0]])
    ring = ring[np.argsort(angles)]
    fan = np.stack([np.repeat(centre[None], len(ring), axis=0), ring, np.roll(ring, -1, axis=0)], axis=1)
    normal = np.cross(fan[:, 1] - fan[:, 0], fan[:, 2] - fan[:, 0])[:, up] * up_sign
    # Wind every triangle to face up the scene.
    flip = normal < 0
    fan[flip] = fan[flip][:, [0, 2, 1]]
    return fan


def excise_subject(*, base_scene: Mapping[str, Any], collider: Any, subject_min: Sequence[float],
                   subject_max: Sequence[float], up: int, up_sign: int, meters_per_unit: float,
                   placement_residual_m: float | None, output_root: Path) -> dict[str, Any]:
    """The base scene with the subject's box cut out of its collider, filled and sealed in.

    Returns ``excised`` with the sealed base scene, the cut collider and the record,
    or ``unchanged`` and the reason nothing was cut.
    """
    from .website_native_background import exclude_box_interior
    from .website_support_geometry import search_window_m

    def unchanged(reason: str, **fields: Any) -> dict[str, Any]:
        return {"status": "unchanged", "record": {"schema_version": REVISION, "status": "unchanged",
                                                  "reason": reason, **fields}}

    low, high = np.asarray(subject_min, dtype=float), np.asarray(subject_max, dtype=float)
    if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or not np.all(low < high)
            or up not in (0, 1, 2) or up_sign not in (-1, 1)
            or not np.isfinite(meters_per_unit) or meters_per_unit <= 0):
        return unchanged("subject_box_invalid")
    side = SIDE_MARGIN_M / meters_per_unit
    low, high = low - side, high + side
    # Heights run along up_sign, so the subject stands on its lowest height either way.
    bottom, top = sorted((up_sign * float(subject_min[up]), up_sign * float(subject_max[up])))
    vertices = np.asarray(collider.vertices, dtype=np.float64)
    faces = np.asarray(collider.faces, dtype=np.int64)
    found = surrounding_support_height(vertices[faces], low=low, high=high, up=up, up_sign=up_sign,
                                       bottom_height=bottom, window=search_window_m(placement_residual_m) / meters_per_unit,
                                       meters_per_unit=meters_per_unit)
    if found is None:
        return unchanged("surrounding_support_not_found")
    height, ring_faces = found
    if top <= height:
        return unchanged("subject_below_surrounding_support")
    below, cut_top = height - BELOW_SUPPORT_M / meters_per_unit, top + TOP_MARGIN_M / meters_per_unit
    low[up], high[up] = (below, cut_top) if up_sign > 0 else (-cut_top, -below)
    binding = {"revision": REVISION, "source_collision_mesh_digest": base_scene["collision_mesh_digest"],
               "up": up, "up_sign": up_sign, "meters_per_unit": float(meters_per_unit),
               "placement_residual_m": placement_residual_m, "box_minimum_runtime": low.tolist(),
               "box_maximum_runtime": high.tolist(), "support_height_runtime_units": up_sign * height,
               "margins_m": {"side": SIDE_MARGIN_M, "top": TOP_MARGIN_M, "below_support": BELOW_SUPPORT_M,
                             "ring": RING_M}}
    root = Path(output_root) / "collider" / ("subject-excision-" + canonical_digest(binding)[7:31])
    receipt_path = root / RECEIPT_NAME
    try:
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text())
            output = Path(prior["output"]["path"])
            if (prior.get("binding") == binding and prior.get("digest") == canonical_digest(prior, digest_field="digest")
                    and output.parent == root and output.is_file() and _sha256_file(output) == prior["output"]["sha256"]):
                return _sealed(base_scene, prior, receipt_path)
        kept, pieces, cut = exclude_box_interior(vertices[faces], minimum=low, maximum=high)
        if cut["fully_interior_faces"] == 0 and cut["clipped_faces"] == 0:
            return unchanged("nothing_inside_subject_box", binding=binding)
        fill = _fill(pieces, low=low, high=high, up=up, up_sign=up_sign, height=height,
                     band=BELOW_SUPPORT_M / meters_per_unit)
        # Untouched faces keep their shared vertices, so the surfaces around the cut keep their shape.
        added = np.concatenate([pieces, fill]).reshape(-1, 3)
        cut_vertices = np.concatenate([vertices, added])
        cut_faces = np.concatenate([faces[kept], len(vertices) + np.arange(len(added)).reshape(-1, 3)])
        import trimesh
        root.mkdir(parents=True, exist_ok=True)
        path, staged = root / COLLIDER_NAME, root / (COLLIDER_NAME + ".partial")
        mesh = trimesh.Trimesh(vertices=cut_vertices, faces=cut_faces, process=False)
        # The subject's own vertices go with its faces; face order is kept.
        mesh.remove_unreferenced_vertices()
        mesh.export(str(staged), file_type="glb")
        with handoff_effect("shared_artifact_promotion"):
            os.replace(staged, path)
        receipt = {"schema_version": REVISION, "binding": binding,
                   "output": {"path": str(path), "sha256": _sha256_file(path)},
                   "cut": {key: value for key, value in cut.items() if key != "replacement_source_faces"},
                   "fill": {"faces": len(fill), "ring_faces": ring_faces,
                            "basis": "surrounding_surface_height", "observed": False},
                   "subject_modelled": False, "physical_measurement": False}
        receipt["digest"] = canonical_digest(receipt, digest_field="digest")
        from .common import write_json
        with handoff_effect("shared_artifact_promotion"):
            write_json(receipt_path, receipt)
        return _sealed(base_scene, receipt, receipt_path)
    except (HandoffLeaseError, OSError):
        raise
    except Exception as exc:  # noqa: BLE001 - recorded; the fit check never runs on an uncut subject
        return unchanged("excision_failed", error=type(exc).__name__, binding=binding)


def _sealed(base_scene: Mapping[str, Any], receipt: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    import trimesh
    digest = receipt["output"]["sha256"]
    record = {"schema_version": REVISION, "status": "excised", "collision_mesh_digest": digest,
              "source_collision_mesh_digest": receipt["binding"]["source_collision_mesh_digest"],
              "receipt_path": str(receipt_path), "receipt_digest": receipt["digest"],
              "box_minimum_runtime": receipt["binding"]["box_minimum_runtime"],
              "box_maximum_runtime": receipt["binding"]["box_maximum_runtime"],
              "support_height_runtime_units": receipt["binding"]["support_height_runtime_units"],
              "cut": receipt["cut"], "fill": receipt["fill"]}
    sealed = {**base_scene, "collision_mesh_path": receipt["output"]["path"], "collision_mesh_digest": digest,
              "collision_binding_id": "website-collider-" + digest[7:39],
              "subject_excision": {
                  "schema_version": REVISION, "receipt_path": str(receipt_path), "receipt_digest": receipt["digest"],
                  "source_collision_mesh_path": base_scene["collision_mesh_path"],
                  "source_collision_mesh_digest": base_scene["collision_mesh_digest"],
                  "source_collision_binding_id": base_scene.get("collision_binding_id")}}
    collider = trimesh.load(receipt["output"]["path"], force="mesh", process=False)
    return {"status": "excised", "base_scene": sealed, "collider": collider, "record": record}
