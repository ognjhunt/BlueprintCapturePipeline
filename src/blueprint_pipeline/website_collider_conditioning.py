"""Make a website collider physically usable where objects are placed.

Beta journey step: scene preparation, before the collider is sealed. An Atlas
collider is a splat-derived mesh, and its surfaces are lumpy: the dining table
of capture-447ae4f4 stood 6 mm RMS (26 mm at worst) off its fitted plane with a
median face tilt of 7.7 degrees, so a 31 x 52 cm tray rested on its highest
bumps and could rock. Compile finds each surface an object starts or ends on by
fitting it (``fitted_surface_under_footprint``). This module flattens exactly
those surfaces into one level patch at the fitted top before
``collision_mesh_digest`` is computed, so every later reader (support fit,
region admission, native bind, relocation) reads the same sealed triangles.

Level, not tilted with the fit: every consumer models a support as level at its
top (``top_runtime_units``, the support plane's ``top_z_m``), and a tilted
planar patch reads as a different band of level faces under each footprint
(capture-447ae4f4 flattened at its 0.6 degree fit: the one table read as three
surfaces). The fit's tilt is recorded, and a surface whose tilt would carry the
level patch out of its fit band is kept raw.

Only a fitted surface's own interior vertices within its measured fit band
move, and only along gravity: the surface's rim (any vertex another face also
uses), every other face, a level-face, measured or authored surface, and the
splat stay exactly as they were. Nothing here blocks: when a surface cannot be
flattened safely the raw collider is kept and the reason recorded. Taubin
smoothing of the rest is off unless a caller supplies its parameters. Every
value is an estimate, never a measurement.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core.handoff_lease import HandoffLeaseError, handoff_effect
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

REVISION = "website_collider_conditioning.v1"
#: The one support basis that is a fit to bumpy triangles. Level faces are already
#: flat; floors, container floors and authored surfaces are not this.
FLATTENED_BASIS = "fitted_surface_under_footprint"
#: A vertex moves only within this many fit RMS of its surface's fit, or within the
#: registration's placement uncertainty near the task when that is larger.
BAND_RMS_MULTIPLE = 3.0
#: support_under admits no steeper fit (website_support_geometry.FITTED_MAX_TILT_DEGREES).
MAX_FIT_TILT_DEGREES = 5.0
#: A surface with more of its interior outside the band than this is not flattened:
#: a patch of moved and kept vertices would be no flatter than the captured one.
MAX_OUT_OF_BAND_SHARE = 0.05
#: Faces steeper than this are walls, edges and doorways: smoothing never moves them.
WALL_MIN_TILT_DEGREES = 60.0
COLLIDER_NAME, RECEIPT_NAME = "collider_conditioned.glb", "collider_conditioning.json"
#: Volume-preserving Taubin smoothing of everything no object rests on; a caller
#: enables it by passing ``smoothing`` (these values, or its own).
TAUBIN_DEFAULTS = {"lamb": 0.5, "nu": 0.53, "iterations": 10, "weights": "uniform",
                   "max_displacement_m": 0.01, "max_aabb_change_m": 0.005,
                   "start_footprint_margin_m": 0.05}
#: What compile passes. Off: placement is fixed by flattening, and smoothing the
#: rest of a 1.5 M-face Atlas collider moves floors and counters no task uses.
DEFAULT_SMOOTHING: Mapping[str, Any] | None = None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _surface_stats(vertices: np.ndarray, faces: np.ndarray, *, up: int,
                   meters_per_unit: float) -> dict[str, float]:
    """Flatness of one surface: its vertices' residual to their own plane, and face tilt from gravity."""
    points = np.unique(vertices[np.unique(faces.ravel())], axis=0)
    middle = points.mean(axis=0)
    normal = np.linalg.svd(points - middle, full_matrices=False)[2][-1]
    residual = np.abs((points - middle) @ normal) * meters_per_unit
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normals, axis=1)
    usable = length > 0
    tilt = np.degrees(np.arccos(np.clip(np.abs(normals[usable, up]) / length[usable], 0.0, 1.0)))
    return {"vertex_residual_rms_m": float(np.sqrt(np.mean(residual ** 2))),
            "vertex_residual_p95_m": float(np.percentile(residual, 95)),
            "vertex_residual_max_m": float(residual.max()),
            "face_tilt_median_degrees": float(np.median(tilt)) if len(tilt) else 0.0,
            "face_tilt_p90_degrees": float(np.percentile(tilt, 90)) if len(tilt) else 0.0}


def _surfaces(placements: Sequence[Mapping[str, Any]], face_count: int
              ) -> tuple[list[list[Mapping[str, Any]]], list[dict[str, Any]]]:
    """Fitted supports grouped into surfaces (one surface when two fits name it), and the rest."""
    from .website_support_geometry import same_support_surface

    surfaces: list[list[Mapping[str, Any]]] = []
    untouched: list[dict[str, Any]] = []
    for row in placements:
        support = row.get("support")
        if not isinstance(support, Mapping):
            continue
        faces = support.get("face_indices") or []
        reason = ("measured_surface" if support.get("physical_measurement") is True
                  else None if support.get("basis") == FLATTENED_BASIS else "not_a_fitted_surface")
        if reason is None and (not faces or min(faces) < 0 or max(faces) >= face_count):
            reason = "support_faces_not_in_collider"
        if reason is not None:
            untouched.append({"role": row["role"], "basis": support.get("basis"), "reason": reason})
            continue
        for members in surfaces:
            if any(same_support_surface(dict(member["support"]), dict(support)) for member in members):
                members.append(row)
                break
        else:
            surfaces.append([row])
    return surfaces, untouched


def flatten_placement_surfaces(vertices: np.ndarray, faces: np.ndarray,
                               placements: Sequence[Mapping[str, Any]], *, up: int, up_sign: int,
                               meters_per_unit: float, placement_uncertainty_m: float | None
                               ) -> tuple[np.ndarray, dict[str, Any]]:
    """Each fitted surface's interior vertices within its fit band, moved along gravity to its top.

    ``placements`` rows are ``{"role", "support"}`` in runtime units, as compile's
    support lookups return them; two rows naming one surface share one patch,
    at the higher of their tops. A vertex moves only when every face using it
    (by position, so unshared Atlas vertices count) is a face of this surface,
    and when its residual to the surface's fit is within the band
    ``max(3 x fit RMS, placement uncertainty)``; the level top itself lies within
    that band of the fit everywhere on the surface. Moving along gravity keeps
    every face's plan footprint and winding, so no face turns over. Returns the
    new vertex array (same layout, same faces) and the record.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    plan = [axis for axis in range(3) if axis != up]
    surfaces, untouched = _surfaces(placements, len(faces))
    uncertainty = _finite(placement_uncertainty_m)
    upright = vertices.copy()
    upright[:, up] *= up_sign
    claim = np.full(len(vertices), -1, dtype=np.int64)
    conflict = np.zeros(len(vertices), dtype=bool)
    heights = np.zeros(len(vertices))
    records: list[tuple[dict[str, Any], np.ndarray]] = []
    for index, members in enumerate(surfaces):
        face_ids = np.unique(np.concatenate([np.asarray(row["support"]["face_indices"], dtype=np.int64)
                                             for row in members]))
        # The fit support_under made: a plane through the surface's face centres.
        centres = upright[faces[face_ids]].mean(axis=1)
        middle = centres.mean(axis=0)
        normal = np.linalg.svd(centres - middle, full_matrices=False)[2][-1]
        normal = normal if normal[up] >= 0 else -normal
        rms = float(np.sqrt(np.mean(((centres - middle) @ normal) ** 2)))
        tilt = float(np.degrees(np.arccos(min(1.0, abs(float(normal[up]))))))
        top = max(up_sign * float(row["support"]["top_runtime_units"]) for row in members)
        band_m = max(BAND_RMS_MULTIPLE * rms * meters_per_unit, uncertainty or 0.0)
        record: dict[str, Any] = {
            "roles": [row["role"] for row in members], "face_count": int(len(face_ids)),
            "face_indices": face_ids.tolist(),
            "fitted_plane": {"normal_upright": normal.tolist(), "point_upright_runtime_units": middle.tolist(),
                             "rms_m": rms * meters_per_unit, "tilt_degrees": tilt},
            "plane": {"basis": "level_at_fitted_top", "up_axis_index": up, "up_sign": up_sign,
                      "top_runtime_units": up_sign * top},
            "band_m": band_m,
            "band_basis": ("registration_placement_uncertainty" if band_m == uncertainty
                           else f"{BAND_RMS_MULTIPLE:g}x_fit_rms"),
            "before": _surface_stats(vertices, faces[face_ids], up=up, meters_per_unit=meters_per_unit)}
        records.append((record, face_ids))
        if tilt > MAX_FIT_TILT_DEGREES:
            record.update(status="kept_raw", reason="fitted_plane_not_level")
            continue
        # Every vertex index at a position this surface's faces use: an unshared
        # copy of it under another face pins that position too.
        used = np.unique(faces[face_ids].ravel())
        low, high = vertices[used].min(axis=0), vertices[used].max(axis=0)
        near = np.flatnonzero(np.all((vertices >= low) & (vertices <= high), axis=1))
        keys, inverse = np.unique(vertices[near], axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        key_of = np.full(len(vertices), -1, dtype=np.int64)
        key_of[near] = inverse
        touched = np.flatnonzero((key_of[faces] >= 0).any(axis=1))
        member = np.zeros(len(faces), dtype=bool)
        member[face_ids] = True
        rows = key_of[faces[touched]]
        own = rows[member[touched]]
        total = np.bincount(rows[rows >= 0], minlength=len(keys))
        inside = np.bincount(own[own >= 0], minlength=len(keys))
        interior = (total > 0) & (total == inside)
        upright_keys = keys.copy()
        upright_keys[:, up] *= up_sign
        fit = middle[up] - ((upright_keys[:, plan[0]] - middle[plan[0]]) * normal[plan[0]]
                            + (upright_keys[:, plan[1]] - middle[plan[1]]) * normal[plan[1]]) / normal[up]
        residual = (upright_keys[:, up] - fit) * meters_per_unit
        move = (top - upright_keys[:, up]) * meters_per_unit
        record.update(rim_vertex_count=int(np.count_nonzero((inside > 0) & (total > inside))),
                      level_to_fit_max_m=(float(np.abs(top - fit[interior]).max() * meters_per_unit)
                                          if interior.any() else 0.0))
        if record["level_to_fit_max_m"] > band_m:
            # A level patch would step off the fitted surface somewhere: keep it as captured.
            record.update(status="kept_raw", reason="fitted_tilt_exceeds_band")
            continue
        eligible = interior & (np.abs(residual) <= band_m)
        record["out_of_band_vertex_count"] = int(np.count_nonzero(interior & ~eligible))
        if not eligible.any() or record["out_of_band_vertex_count"] > MAX_OUT_OF_BAND_SHARE * np.count_nonzero(interior):
            # Too much of it lies off its own fit to be one level patch: keep it as captured.
            record.update(status="kept_raw", reason="surface_outside_its_fit_band")
            continue
        selected = near[eligible[inverse]]
        conflict[selected[claim[selected] >= 0]] = True
        claim[selected] = index
        heights[selected] = top
        record.update(status="flattened", moved_vertex_count=int(np.count_nonzero(eligible)),
                      max_displacement_m=float(np.abs(move[eligible]).max()),
                      rms_displacement_m=float(np.sqrt(np.mean(move[eligible] ** 2))))
    output = vertices.copy()
    # A position two distinct surfaces would both move stays where it is.
    final = (claim >= 0) & ~conflict
    output[final, up] = up_sign * heights[final]
    for record, face_ids in records:
        if record["status"] == "flattened":
            moved = np.any(output != vertices, axis=1)
            used = np.unique(faces[face_ids].ravel())
            kept = used[~moved[used]]
            above = (up_sign * vertices[kept, up] - up_sign * record["plane"]["top_runtime_units"]) * meters_per_unit
            record["after"] = {**_surface_stats(output, faces[face_ids], up=up, meters_per_unit=meters_per_unit),
                               # The share of the surface now one level patch, and what of its
                               # rim (kept as captured) still stands above it.
                               "level_face_fraction": float(np.mean(moved[faces[face_ids]].all(axis=1))),
                               "kept_vertices_above_patch_count": int(np.count_nonzero(above > 0.001)),
                               "kept_vertex_max_above_patch_m": float(max(above.max(initial=0.0), 0.0))}
    return output, {"surfaces": [record for record, _ in records], "untouched_supports": untouched,
                    "positions_kept_between_surfaces": int(np.count_nonzero(conflict))}


def smooth_unplaced(vertices: np.ndarray, faces: np.ndarray, *, pinned: np.ndarray, up: int,
                    meters_per_unit: float, parameters: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Taubin-smooth every position no object rests on, walls and doorways pinned.

    Positions are merged first (an Atlas collider shares few vertices, and an
    unshared one has no neighbours to average). ``pinned`` vertex indices, and
    every vertex of a face steeper than ``WALL_MIN_TILT_DEGREES``, stay exactly
    where they are. The result is kept only when no position moved, and the
    bounding box changed, by more than the recorded bounds.
    """
    from scipy.sparse import coo_matrix

    settings = {**TAUBIN_DEFAULTS, **dict(parameters)}
    lamb, nu, iterations = float(settings["lamb"]), float(settings["nu"]), int(settings["iterations"])
    if not (0 < lamb < nu and iterations >= 0) or settings["weights"] != "uniform":
        raise ValueError("website_collider_smoothing_parameters_invalid")
    vertices = np.asarray(vertices, dtype=np.float64)
    keys, inverse = np.unique(vertices, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    merged = inverse[faces]
    corners = keys[merged]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normals, axis=1)
    steep = np.abs(normals[:, up]) < math.cos(math.radians(WALL_MIN_TILT_DEGREES)) * length
    fixed = np.zeros(len(keys), dtype=bool)
    fixed[inverse[np.flatnonzero(pinned)]] = True
    fixed[merged[steep].ravel()] = True
    edges = merged[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    adjacency = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]],
                                                     np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(keys),) * 2).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    free = (~fixed & (degree > 0)).astype(np.float64)[:, None]
    scale = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)[:, None]
    smoothed = keys.copy()
    for _ in range(iterations):
        for factor in (lamb, -nu):
            smoothed = smoothed + factor * free * (scale * (adjacency @ smoothed) - smoothed)
    displacement = np.linalg.norm(smoothed - keys, axis=1) * meters_per_unit
    aabb_change = float(np.abs(np.r_[smoothed.min(axis=0) - keys.min(axis=0),
                                     smoothed.max(axis=0) - keys.max(axis=0)]).max() * meters_per_unit)
    record = {"method": "taubin", **{key: settings[key] for key in TAUBIN_DEFAULTS},
              "merged_position_count": int(len(keys)), "pinned_position_count": int(np.count_nonzero(fixed)),
              "moved_position_count": int(np.count_nonzero(displacement > 0)),
              "measured_max_displacement_m": float(displacement.max()) if len(displacement) else 0.0,
              "measured_aabb_change_m": aabb_change}
    if (record["measured_max_displacement_m"] > float(settings["max_displacement_m"])
            or aabb_change > float(settings["max_aabb_change_m"])):
        return vertices, {**record, "status": "kept_unsmoothed", "reason": "smoothing_bound_exceeded"}
    return smoothed[inverse], {**record, "status": "smoothed"}


def _footprint_column(vertices: np.ndarray, footprint: Sequence[Sequence[float]], *, up: int,
                      margin: float) -> np.ndarray:
    lower, upper = (np.asarray(value, dtype=np.float64) for value in footprint)
    inside = np.ones(len(vertices), dtype=bool)
    for axis in range(3):
        if axis != up:
            inside &= (vertices[:, axis] >= lower[axis] - margin) & (vertices[:, axis] <= upper[axis] + margin)
    return inside


def condition_placement_collider(*, base_scene: Mapping[str, Any], collider: Any,
                                 placements: Sequence[Mapping[str, Any]], up: int, up_sign: int,
                                 meters_per_unit: float, placement_uncertainty_m: float | None,
                                 output_root: Path, smoothing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The base scene with a conditioned collider sealed in, or the raw one kept and why.

    ``placements`` rows are ``{"role", "support", "footprint": (lower, upper)}``,
    found on ``collider`` (the base scene's own) exactly as compile finds them.
    Returns ``{"status": "unchanged"}`` when nothing applies, ``kept_raw`` with
    the record, or ``conditioned`` with the sealed base scene and the record.
    The conditioned copy keeps the collider's vertex and face layout, so every
    support's ``face_indices`` still names the same triangles; retries reuse it.
    """
    candidates = [row for row in placements if isinstance(row.get("support"), Mapping)]
    if smoothing is None and not any(row["support"].get("basis") == FLATTENED_BASIS for row in candidates):
        return {"status": "unchanged"}
    source_digest = base_scene["collision_mesh_digest"]
    record: dict[str, Any] = {"schema_version": REVISION, "source_collision_mesh_digest": source_digest}
    try:
        binding = {"revision": REVISION, "source_collision_mesh_digest": source_digest, "up": up,
                   "up_sign": up_sign, "meters_per_unit": float(meters_per_unit),
                   "placement_uncertainty_m": _finite(placement_uncertainty_m),
                   "smoothing": None if smoothing is None else {**TAUBIN_DEFAULTS, **dict(smoothing)},
                   "placements": [{"role": row["role"], "support": dict(row["support"]),
                                   "footprint": [np.asarray(v, dtype=float).tolist() for v in row["footprint"]]}
                                  for row in candidates]}
        root = Path(output_root) / "collider" / canonical_digest(binding)[7:31]
        receipt_path = root / RECEIPT_NAME
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text())
            output = Path(prior["output"]["path"])
            if (prior.get("binding") == binding and prior.get("digest") == canonical_digest(prior, digest_field="digest")
                    and output.parent == root and output.is_file()
                    and _sha256_file(output) == prior["output"]["sha256"]):
                return _sealed(base_scene, prior, receipt_path)
        vertices = np.asarray(collider.vertices, dtype=np.float64)
        faces = np.asarray(collider.faces, dtype=np.int64)
        conditioned, flattening = flatten_placement_surfaces(vertices, faces, candidates, up=up, up_sign=up_sign,
            meters_per_unit=meters_per_unit, placement_uncertainty_m=placement_uncertainty_m)
        record["flattening"] = flattening
        if smoothing is not None:
            pinned = np.zeros(len(vertices), dtype=bool)
            for surface in flattening["surfaces"]:
                pinned[faces[surface["face_indices"]].ravel()] = True
            margin = float({**TAUBIN_DEFAULTS, **dict(smoothing)}["start_footprint_margin_m"]) / meters_per_unit
            for row in candidates:
                if row["role"] == "start_support":
                    pinned |= _footprint_column(vertices, row["footprint"], up=up, margin=margin)
            conditioned, record["smoothing"] = smooth_unplaced(conditioned, faces, pinned=pinned, up=up,
                meters_per_unit=meters_per_unit, parameters=smoothing)
        else:
            record["smoothing"] = {"status": "not_requested"}
        if np.array_equal(conditioned, vertices):
            reasons = sorted({row["reason"] for row in flattening["surfaces"] if row.get("reason")}
                             | {row["reason"] for row in flattening["untouched_supports"]}
                             | ({record["smoothing"]["reason"]} if "reason" in record["smoothing"] else set()))
            return {"status": "kept_raw", "record": {**record, "status": "kept_raw",
                                                     "reasons": reasons or ["nothing_to_condition"]}}
        import trimesh
        root.mkdir(parents=True, exist_ok=True)
        path = root / COLLIDER_NAME
        staged = root / (COLLIDER_NAME + ".partial")
        trimesh.Trimesh(vertices=conditioned, faces=faces, visual=collider.visual,
                        process=False).export(str(staged), file_type="glb")
        with handoff_effect("shared_artifact_promotion"):
            os.replace(staged, path)
        receipt = {"schema_version": REVISION, "binding": binding,
                   "output": {"path": str(path), "sha256": _sha256_file(path)}, **record,
                   "splat_unchanged": True, "physical_measurement": False}
        receipt["digest"] = canonical_digest(receipt, digest_field="digest")
        from .common import write_json
        with handoff_effect("shared_artifact_promotion"):
            write_json(receipt_path, receipt)
        return _sealed(base_scene, receipt, receipt_path)
    except HandoffLeaseError:
        raise
    except Exception as exc:  # noqa: BLE001 - geometry is never a hard stop: keep the raw collider, say why
        return {"status": "kept_raw", "record": {**record, "status": "kept_raw",
                                                 "reasons": [f"conditioning_failed:{type(exc).__name__}:{exc}"]}}


def _sealed(base_scene: Mapping[str, Any], receipt: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    digest = receipt["output"]["sha256"]
    record = {key: receipt[key] for key in ("schema_version", "source_collision_mesh_digest", "flattening",
                                            "smoothing")}
    record.update(status="conditioned", collision_mesh_digest=digest, receipt_path=str(receipt_path),
                  receipt_digest=receipt["digest"])
    sealed = {**base_scene, "collision_mesh_path": receipt["output"]["path"], "collision_mesh_digest": digest,
              "collision_binding_id": "website-collider-" + digest[7:39],
              "collider_conditioning": {
                  "schema_version": REVISION, "receipt_path": str(receipt_path), "receipt_digest": receipt["digest"],
                  "source_collision_mesh_path": base_scene["collision_mesh_path"],
                  "source_collision_mesh_digest": base_scene["collision_mesh_digest"],
                  "source_collision_binding_id": base_scene.get("collision_binding_id")}}
    return {"status": "conditioned", "base_scene": sealed, "record": record}
