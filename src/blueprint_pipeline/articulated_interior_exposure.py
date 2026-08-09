"""Check that opening the task door actually reveals the asset's interior.

A replacement derived from a collision mesh inherits a solid carcass: the
source mesh describes the appliance's outer shell, so the support link owns a
continuous front face. Authoring an inset interior behind that face produces an
asset that passes every geometric gate we had - one articulation root, correct
joints and limits, colliders that do not span the seam, a required generated
interior - and still shows a flat wall when the door swings open, because the
cavity is sealed inside the shell it was placed in.

This gate casts rays inward through the aperture the open door leaves behind
and asks what they hit first. Interior geometry means the cavity is reachable;
support-link geometry means the carcass is walling it off. It is a visibility
and reachability check, not a claim that the modelled interior matches the real
appliance - the interior was never observed and stays labelled candidate
geometry either way.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest


INTERIOR_EXPOSURE_SCHEMA_VERSION = "articulated_interior_exposure.v1"


class ArticulatedInteriorExposureError(ValueError):
    """Stable, sorted interior-exposure failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _interval(value: Any, error: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ArticulatedInteriorExposureError([error])
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ArticulatedInteriorExposureError([error])
        number = float(item)
        if not math.isfinite(number):
            raise ArticulatedInteriorExposureError([error])
        out.append(number)
    if out[0] >= out[1]:
        raise ArticulatedInteriorExposureError([error])
    return out


def _triangles(stage, prim, cache) -> np.ndarray:
    """World-space triangles of one mesh prim."""

    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return np.zeros((0, 3, 3))
    transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
    local = np.array([[float(v) for v in p] for p in points])
    homo = np.concatenate([local, np.ones((local.shape[0], 1))], axis=1)
    world = (transform @ homo.T).T[:, :3]
    out: list[list[list[float]]] = []
    cursor = 0
    for count in counts:
        face = [int(v) for v in indices[cursor : cursor + int(count)]]
        cursor += int(count)
        for offset in range(1, len(face) - 1):
            out.append(
                [
                    world[face[0]].tolist(),
                    world[face[offset]].tolist(),
                    world[face[offset + 1]].tolist(),
                ]
            )
    return np.array(out) if out else np.zeros((0, 3, 3))


def _ray_hits(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> float:
    """Nearest positive Moller-Trumbore intersection distance, or infinity."""

    if tris.shape[0] == 0:
        return math.inf
    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(det) < 1e-12
    safe = np.where(parallel, 1.0, det)
    tvec = origin - tris[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) / safe
    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", direction, qvec) / safe
    t = np.einsum("ij,ij->i", edge2, qvec) / safe
    hit = (~parallel) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
    return float(t[hit].min()) if hit.any() else math.inf


def evaluate_interior_exposure(
    *,
    replacement_usd_path: str | Path,
    support_link_path: str,
    task_door_link_path: str,
    interior_prim_paths: Sequence[str],
    aperture_plane_y_m: float,
    aperture_x_interval_m: Sequence[float],
    aperture_z_interval_m: Sequence[float],
    samples_per_axis: int = 11,
    minimum_exposed_fraction: float = 0.75,
    inward_axis: Sequence[float] = (0.0, -1.0, 0.0),
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Ray-cast through the open door's aperture and report what is reachable."""

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedInteriorExposureError(
            ["articulated_interior_openusd_runtime_missing"]
        ) from exc

    asset = Path(replacement_usd_path).expanduser().resolve()
    if not asset.is_file():
        raise ArticulatedInteriorExposureError(["articulated_interior_asset_missing"])
    x_interval = _interval(
        aperture_x_interval_m, "articulated_interior_aperture_interval_invalid"
    )
    z_interval = _interval(
        aperture_z_interval_m, "articulated_interior_aperture_interval_invalid"
    )
    if not isinstance(samples_per_axis, int) or samples_per_axis < 2:
        raise ArticulatedInteriorExposureError(
            ["articulated_interior_samples_per_axis_invalid"]
        )

    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise ArticulatedInteriorExposureError(["articulated_interior_asset_unreadable"])
    errors: list[str] = []
    support = stage.GetPrimAtPath(str(support_link_path))
    door = stage.GetPrimAtPath(str(task_door_link_path))
    if not support.IsValid():
        errors.append(f"articulated_interior_support_link_missing:{support_link_path}")
    if not door.IsValid():
        errors.append(
            f"articulated_interior_task_door_link_missing:{task_door_link_path}"
        )
    interior_paths = [str(path) for path in interior_prim_paths]
    if not interior_paths:
        errors.append("articulated_interior_interior_prim_paths_missing")
    for path in interior_paths:
        if not stage.GetPrimAtPath(path).IsValid():
            errors.append(f"articulated_interior_interior_prim_missing:{path}")
    if errors:
        raise ArticulatedInteriorExposureError(errors)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    interior_set = set(interior_paths)
    interior_tris: list[np.ndarray] = []
    support_tris: dict[str, np.ndarray] = {}
    for prim in Usd.PrimRange(support):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        tris = _triangles(stage, prim, cache)
        if tris.shape[0] == 0:
            continue
        if path in interior_set:
            interior_tris.append(tris)
        else:
            support_tris[path] = tris
    for path in interior_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsA(UsdGeom.Mesh) and not str(prim.GetPath()).startswith(
            str(support_link_path)
        ):
            interior_tris.append(_triangles(stage, prim, cache))
    interior = (
        np.concatenate(interior_tris) if interior_tris else np.zeros((0, 3, 3))
    )

    direction = np.asarray(inward_axis, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    xs = np.linspace(x_interval[0], x_interval[1], samples_per_axis)
    zs = np.linspace(z_interval[0], z_interval[1], samples_per_axis)
    hit_interior = 0
    hit_support = 0
    hit_nothing = 0
    occluders: dict[str, int] = {}
    for x in xs:
        for z in zs:
            origin = np.array([float(x), float(aperture_plane_y_m), float(z)])
            interior_distance = _ray_hits(origin, direction, interior)
            nearest_support = math.inf
            nearest_path = ""
            for path, tris in support_tris.items():
                distance = _ray_hits(origin, direction, tris)
                if distance < nearest_support:
                    nearest_support, nearest_path = distance, path
            if not math.isfinite(interior_distance) and not math.isfinite(
                nearest_support
            ):
                hit_nothing += 1
            elif nearest_support < interior_distance:
                hit_support += 1
                occluders[nearest_path] = occluders.get(nearest_path, 0) + 1
            else:
                hit_interior += 1

    total = int(samples_per_axis * samples_per_axis)
    exposed_fraction = hit_interior / total if total else 0.0
    exposed = exposed_fraction >= float(minimum_exposed_fraction)
    blockers: list[str] = []
    if not exposed:
        blockers.append(
            "articulated_interior_occluded_by_support_link"
            if hit_support
            else "articulated_interior_not_reachable_through_aperture"
        )

    receipt: dict[str, Any] = {
        "schema_version": INTERIOR_EXPOSURE_SCHEMA_VERSION,
        "status": "interior_exposed" if exposed else "interior_not_exposed",
        "replacement_usd_path": str(asset),
        "replacement_usd_sha256": _sha256(asset),
        "support_link_path": str(support_link_path),
        "task_door_link_path": str(task_door_link_path),
        "interior_prim_paths": sorted(interior_paths),
        "aperture": {
            "plane_y_m": float(aperture_plane_y_m),
            "x_interval_m": x_interval,
            "z_interval_m": z_interval,
            "inward_axis": [float(value) for value in direction],
        },
        "samples": {
            "per_axis": int(samples_per_axis),
            "total": total,
            "hit_interior": hit_interior,
            "hit_support": hit_support,
            "hit_nothing": hit_nothing,
        },
        "exposed_fraction": round(exposed_fraction, 6),
        "minimum_exposed_fraction": float(minimum_exposed_fraction),
        "interior_exposed": bool(exposed),
        "occluding_prim_paths": sorted(occluders),
        "occluding_sample_counts": dict(sorted(occluders.items())),
        "blockers": blockers,
        "claim_boundary": {
            "interior_is_generated_candidate_geometry": True,
            "interior_matches_real_appliance": False,
            "native_simulator_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if destination is not None:
        write_json(Path(destination).expanduser().resolve(), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "INTERIOR_EXPOSURE_SCHEMA_VERSION",
    "ArticulatedInteriorExposureError",
    "evaluate_interior_exposure",
]
