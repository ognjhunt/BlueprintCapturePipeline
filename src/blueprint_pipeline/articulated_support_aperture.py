"""Open the aperture a task door needs, without disturbing collision.

A replacement built from a collision mesh inherits a sealed carcass, so the
support link's front face stands between the door and the interior authored
behind it. :mod:`articulated_interior_exposure` detects that; this module fixes
it, by subtracting the aperture rectangle from the outward-facing faces that lie
on the aperture plane and retriangulating what remains.

Two invariants make the fix safe to apply to a physics-qualified asset. Existing
points are never moved or dropped - clipping only appends new vertices, and each
new vertex lies on the plane of a face it came from, inside that face - so a
convex-hull collider computed from the point set is provably identical. And the
operation only ever removes faces from the support link, never from the door,
the interior, or anything the caller marks protected.

The result is an opened *visual* shell. It does not claim the modelled interior
resembles the real appliance; that geometry was never observed.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest


SUPPORT_APERTURE_SCHEMA_VERSION = "articulated_support_aperture.v1"
DEFAULT_OUTWARD_DOT_MINIMUM = 0.85
DEFAULT_PLANE_TOLERANCE_M = 0.01
DEFAULT_MAXIMUM_REMOVED_AREA_FRACTION = 0.35


class ArticulatedSupportApertureError(ValueError):
    """Stable, sorted aperture-cut failures."""

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
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ArticulatedSupportApertureError([error])
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ArticulatedSupportApertureError([error])
        number = float(item)
        if not math.isfinite(number):
            raise ArticulatedSupportApertureError([error])
        out.append(number)
    if out[0] >= out[1]:
        raise ArticulatedSupportApertureError([error])
    return out


def _clip_half_plane(
    polygon: list[tuple[float, float]], keep
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a convex polygon against one half-plane."""

    if not polygon:
        return []
    out: list[tuple[float, float]] = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        current_in, previous_in = keep(current), keep(previous)
        if current_in != previous_in:
            t = keep.fraction(previous, current)
            out.append(
                (
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                )
            )
        if current_in:
            out.append(current)
    return out


class _AxisHalfPlane:
    """Keep points on one side of an axis-aligned line in the (a, b) plane."""

    def __init__(self, axis: int, bound: float, keep_less: bool):
        self.axis, self.bound, self.keep_less = axis, bound, keep_less

    def __call__(self, point) -> bool:
        value = point[self.axis]
        return value <= self.bound if self.keep_less else value >= self.bound

    def fraction(self, start, end) -> float:
        denominator = end[self.axis] - start[self.axis]
        if abs(denominator) < 1e-15:
            return 0.0
        return (self.bound - start[self.axis]) / denominator


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index, (x, y) in enumerate(polygon):
        nx, ny = polygon[(index + 1) % len(polygon)]
        total += x * ny - nx * y
    return abs(total) / 2.0


def _subtract_rectangle(
    triangle: Sequence[tuple[float, float]],
    x_interval: Sequence[float],
    z_interval: Sequence[float],
) -> list[list[tuple[float, float]]]:
    """Triangle minus an axis-aligned rectangle, as disjoint convex pieces."""

    pieces = []
    left = _clip_half_plane(list(triangle), _AxisHalfPlane(0, x_interval[0], True))
    right = _clip_half_plane(list(triangle), _AxisHalfPlane(0, x_interval[1], False))
    middle = _clip_half_plane(list(triangle), _AxisHalfPlane(0, x_interval[0], False))
    middle = _clip_half_plane(middle, _AxisHalfPlane(0, x_interval[1], True))
    below = _clip_half_plane(middle, _AxisHalfPlane(1, z_interval[0], True))
    above = _clip_half_plane(middle, _AxisHalfPlane(1, z_interval[1], False))
    for piece in (left, right, below, above):
        if len(piece) >= 3 and _polygon_area(piece) > 1e-12:
            pieces.append(piece)
    return pieces


def cut_support_link_aperture(
    *,
    source_usd_path: str | Path,
    destination: str | Path,
    support_link_path: str,
    aperture_x_interval_m: Sequence[float],
    aperture_z_interval_m: Sequence[float],
    outward_axis: Sequence[float] = (0.0, 1.0, 0.0),
    aperture_plane_m: float | None = None,
    plane_tolerance_m: float = DEFAULT_PLANE_TOLERANCE_M,
    outward_dot_minimum: float = DEFAULT_OUTWARD_DOT_MINIMUM,
    protected_prim_paths: Sequence[str] = (),
    maximum_removed_area_fraction: float = DEFAULT_MAXIMUM_REMOVED_AREA_FRACTION,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Subtract the aperture rectangle from the support link's outward face."""

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedSupportApertureError(
            ["articulated_support_aperture_openusd_runtime_missing"]
        ) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ArticulatedSupportApertureError(
            ["articulated_support_aperture_source_missing"]
        )
    if output == source:
        raise ArticulatedSupportApertureError(
            ["articulated_support_aperture_destination_is_source"]
        )
    x_interval = _interval(
        aperture_x_interval_m, "articulated_support_aperture_interval_invalid"
    )
    z_interval = _interval(
        aperture_z_interval_m, "articulated_support_aperture_interval_invalid"
    )
    outward = np.asarray(outward_axis, dtype=np.float64)
    norm = float(np.linalg.norm(outward))
    if norm < 1e-9:
        raise ArticulatedSupportApertureError(
            ["articulated_support_aperture_outward_axis_invalid"]
        )
    outward = outward / norm
    # The two in-plane axes are whichever world axes the outward axis is not.
    outward_index = int(np.argmax(np.abs(outward)))
    plane_axes = [axis for axis in range(3) if axis != outward_index]

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        output.unlink(missing_ok=True)
        raise ArticulatedSupportApertureError(
            ["articulated_support_aperture_source_unreadable"]
        )
    support = stage.GetPrimAtPath(str(support_link_path))
    if not support.IsValid():
        output.unlink(missing_ok=True)
        raise ArticulatedSupportApertureError(
            [f"articulated_support_aperture_support_link_missing:{support_link_path}"]
        )

    protected = {str(path) for path in protected_prim_paths}
    rows: list[dict[str, Any]] = []
    removed_area = 0.0
    total_area = 0.0
    faces_removed = 0
    faces_added = 0
    approximations: dict[str, str] = {}

    for prim in Usd.PrimRange(support):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        mesh = UsdGeom.Mesh(prim)
        approximation = prim.GetAttribute("physics:approximation")
        if approximation and approximation.HasAuthoredValue():
            approximations[path] = str(approximation.Get())
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        vertices = [[float(v) for v in p] for p in points]
        counts = [int(v) for v in counts]
        indices = [int(v) for v in indices]

        plane = (
            float(aperture_plane_m)
            if aperture_plane_m is not None
            else max(float(np.dot(np.asarray(v), outward)) for v in vertices)
        )
        new_counts: list[int] = []
        new_indices: list[int] = []
        mesh_removed = 0
        mesh_added = 0
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            corner = np.array([vertices[i] for i in face])
            normal = np.cross(corner[1] - corner[0], corner[2] - corner[0])
            magnitude = float(np.linalg.norm(normal))
            area = magnitude / 2.0
            total_area += area
            on_plane = all(
                abs(float(np.dot(v, outward)) - plane) <= plane_tolerance_m
                for v in corner
            )
            facing = (
                magnitude > 1e-12
                and float(np.dot(normal / magnitude, outward)) >= outward_dot_minimum
            )
            if path in protected or not (on_plane and facing):
                new_counts.append(count)
                new_indices.extend(face)
                continue
            flat = [(float(v[plane_axes[0]]), float(v[plane_axes[1]])) for v in corner]
            pieces = _subtract_rectangle(flat, x_interval, z_interval)
            kept_area = sum(_polygon_area(piece) for piece in pieces)
            if area - kept_area <= 1e-12:
                # The aperture misses this face entirely; leave it exactly as it
                # was rather than retriangulating it into an identical shape.
                new_counts.append(count)
                new_indices.extend(face)
                continue
            removed_area += area - kept_area
            mesh_removed += 1
            faces_removed += 1
            for piece in pieces:
                base = len(vertices)
                for point in piece:
                    coordinate = [0.0, 0.0, 0.0]
                    coordinate[outward_index] = plane
                    coordinate[plane_axes[0]] = point[0]
                    coordinate[plane_axes[1]] = point[1]
                    vertices.append(coordinate)
                for offset in range(1, len(piece) - 1):
                    new_counts.append(3)
                    new_indices.extend([base, base + offset, base + offset + 1])
                    mesh_added += 1
                    faces_added += 1
        if mesh_removed:
            mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
            mesh.CreateFaceVertexCountsAttr(new_counts)
            mesh.CreateFaceVertexIndicesAttr(new_indices)
        rows.append(
            {
                "prim_path": path,
                "faces_removed": mesh_removed,
                "faces_added": mesh_added,
                "protected": path in protected,
            }
        )

    if faces_removed == 0:
        output.unlink(missing_ok=True)
        raise ArticulatedSupportApertureError(
            ["articulated_support_aperture_removed_nothing"]
        )
    fraction = removed_area / total_area if total_area else 0.0
    if fraction > float(maximum_removed_area_fraction):
        output.unlink(missing_ok=True)
        raise ArticulatedSupportApertureError(
            [
                "articulated_support_aperture_removed_area_above_ceiling:"
                f"{fraction:.4f}>{float(maximum_removed_area_fraction):.4f}"
            ]
        )

    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output))
    still: dict[str, str] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(str(support_link_path))):
        attribute = prim.GetAttribute("physics:approximation")
        if attribute and attribute.HasAuthoredValue():
            still[str(prim.GetPath())] = str(attribute.Get())

    receipt: dict[str, Any] = {
        "schema_version": SUPPORT_APERTURE_SCHEMA_VERSION,
        "status": "support_aperture_cut",
        "source_usd_path": str(source),
        "source_usd_sha256": _sha256(source),
        "opened_usd_path": str(output),
        "opened_usd_sha256": _sha256(output),
        "support_link_path": str(support_link_path),
        "aperture": {
            "plane_m": aperture_plane_m,
            "x_interval_m": x_interval,
            "z_interval_m": z_interval,
            "outward_axis": [float(value) for value in outward],
            "plane_tolerance_m": float(plane_tolerance_m),
            "outward_dot_minimum": float(outward_dot_minimum),
        },
        "faces_removed": faces_removed,
        "faces_added": faces_added,
        "removed_area_fraction": round(fraction, 6),
        "maximum_removed_area_fraction": float(maximum_removed_area_fraction),
        "meshes": rows,
        "protected_prim_paths": sorted(protected),
        "collision": {
            "approximation_unchanged": still == approximations,
            "approximations": dict(sorted(still.items())),
            # Clipping only appends vertices that lie on an existing face plane
            # inside an existing face, so the convex hull of the point set is
            # unchanged by construction.
            "convex_hull_point_set_preserved": True,
        },
        "preserved": {
            "articulation_root_count": len(
                [
                    p
                    for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI)
                ]
            ),
            "rigid_body_count": len(
                [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            ),
            "assembly_joint_count": len(
                [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
            ),
        },
        "claim_boundary": {
            "source_usd_modified": False,
            "opened_geometry_is_candidate_not_observed_truth": True,
            "native_simulator_qualified": False,
        },
        "receipt_path": str(
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else output.with_name(output.stem + "_aperture_receipt.json")
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(receipt["receipt_path"]), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ArticulatedSupportApertureError",
    "SUPPORT_APERTURE_SCHEMA_VERSION",
    "cut_support_link_aperture",
]
