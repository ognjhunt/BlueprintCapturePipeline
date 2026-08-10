"""Author the interior a replaced appliance needs, as real cavity geometry.

No Content Agent creates geometry. The Joint Agent is topology-only by
NVIDIA's own contract; the Material and Texture agents dress whatever surfaces
already exist; the Physics Agent assigns properties. So when the 840796 twin
opened onto a featureless block, that was not an agent failing - the agents
faithfully textured the placeholder Blueprint had authored. A single six-face
box with every normal facing outward is a solid inset panel: it reads as a
recessed wall, and a gripper could not place anything in it.

This module authors the cavity instead: liner walls with a real hollow between
them, evenly spaced shelves, and optional door bins on the task door. Every
part is parametric from the cavity interval, so the same call serves an oven,
a cabinet, or a drawer, and every part carries the generated-candidate
provenance tag. The appliance interior was never observed - it is closed in
every frame of the scan - so this is plausible structure, never a claim about
what is actually inside.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


INTERIOR_AUTHORING_SCHEMA_VERSION = "articulated_interior_authoring.v1"
PROVENANCE_ATTRIBUTE = "blueprint:articulatedReplacement:provenance"
GENERATED_VALUE = "generated_candidate_geometry"
MINIMUM_SHELF_SPACING_M = 0.08


class ArticulatedInteriorAuthoringError(ValueError):
    """Stable, sorted interior-authoring failures."""

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
        raise ArticulatedInteriorAuthoringError([error])
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ArticulatedInteriorAuthoringError([error])
        number = float(item)
        if not math.isfinite(number):
            raise ArticulatedInteriorAuthoringError([error])
        out.append(number)
    if out[0] >= out[1]:
        raise ArticulatedInteriorAuthoringError([error])
    return out


def author_articulated_interior(
    *,
    source_usd_path: str | Path,
    destination: str | Path,
    support_link_path: str,
    replace_prim_paths: Sequence[str],
    cavity_x_interval_m: Sequence[float],
    cavity_y_interval_m: Sequence[float],
    cavity_z_interval_m: Sequence[float],
    wall_thickness_m: float = 0.012,
    shelf_count: int = 3,
    shelf_thickness_m: float = 0.010,
    shelf_inset_m: float = 0.004,
    door_bin_count: int = 0,
    door_link_path: str | None = None,
    door_bin_y_interval_m: Sequence[float] | None = None,
    door_bin_height_m: float = 0.09,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replace a placeholder interior with liner walls, shelves and door bins."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedInteriorAuthoringError(
            ["articulated_interior_openusd_runtime_missing"]
        ) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ArticulatedInteriorAuthoringError(["articulated_interior_source_missing"])
    if output == source:
        raise ArticulatedInteriorAuthoringError(
            ["articulated_interior_destination_is_source"]
        )
    x = _interval(cavity_x_interval_m, "articulated_interior_cavity_interval_invalid")
    y = _interval(cavity_y_interval_m, "articulated_interior_cavity_interval_invalid")
    z = _interval(cavity_z_interval_m, "articulated_interior_cavity_interval_invalid")
    wall = float(wall_thickness_m)
    shelf_t = float(shelf_thickness_m)
    if wall <= 0.0 or shelf_t <= 0.0 or not math.isfinite(wall):
        raise ArticulatedInteriorAuthoringError(
            ["articulated_interior_thickness_invalid"]
        )
    inner = (
        [x[0] + wall, x[1] - wall],
        [y[0] + wall, y[1]],
        [z[0] + wall, z[1] - wall],
    )
    if any(low >= high for low, high in inner):
        raise ArticulatedInteriorAuthoringError(
            ["articulated_interior_cavity_too_small"]
        )
    if not isinstance(shelf_count, int) or shelf_count < 0:
        raise ArticulatedInteriorAuthoringError(["articulated_interior_shelf_count_invalid"])
    if shelf_count:
        spacing = (inner[2][1] - inner[2][0]) / (shelf_count + 1)
        if spacing < MINIMUM_SHELF_SPACING_M:
            raise ArticulatedInteriorAuthoringError(
                [
                    "articulated_interior_shelf_spacing_too_small:"
                    f"{spacing:.4f}<{MINIMUM_SHELF_SPACING_M}"
                ]
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        output.unlink(missing_ok=True)
        raise ArticulatedInteriorAuthoringError(
            ["articulated_interior_source_unreadable"]
        )
    support = stage.GetPrimAtPath(str(support_link_path))
    if not support.IsValid():
        output.unlink(missing_ok=True)
        raise ArticulatedInteriorAuthoringError(
            [f"articulated_interior_support_link_missing:{support_link_path}"]
        )

    removed: list[str] = []
    for path in replace_prim_paths:
        if stage.GetPrimAtPath(str(path)).IsValid():
            stage.RemovePrim(str(path))
            removed.append(str(path))

    parts: list[dict[str, Any]] = []
    door_inner_face: float | None = None

    def box(prim_path: str, role: str, xs, ys, zs) -> None:
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(a, b, c)
                for a in (xs[0], xs[1])
                for b in (ys[0], ys[1])
                for c in (zs[0], zs[1])
            ]
        )
        quads = [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ]
        counts, indices = [], []
        for quad in quads:
            counts.extend([3, 3])
            indices.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            "convexHull"
        )
        prim.CreateAttribute(PROVENANCE_ATTRIBUTE, Sdf.ValueTypeNames.String).Set(
            GENERATED_VALUE
        )
        parts.append(
            {
                "prim_path": prim_path,
                "role": role,
                "world_aabb_min_m": [xs[0], ys[0], zs[0]],
                "world_aabb_max_m": [xs[1], ys[1], zs[1]],
            }
        )

    scope = f"{support_link_path}/generated_interior"
    UsdGeom.Xform.Define(stage, scope)
    box(f"{scope}/liner_back", "liner_back", x, [y[0], y[0] + wall], z)
    box(f"{scope}/liner_left", "liner_left", [x[0], x[0] + wall], y, z)
    box(f"{scope}/liner_right", "liner_right", [x[1] - wall, x[1]], y, z)
    box(f"{scope}/liner_floor", "liner_floor", x, y, [z[0], z[0] + wall])
    box(f"{scope}/liner_ceiling", "liner_ceiling", x, y, [z[1] - wall, z[1]])
    for index in range(shelf_count):
        height = inner[2][0] + (index + 1) * (inner[2][1] - inner[2][0]) / (
            shelf_count + 1
        )
        box(
            f"{scope}/shelf_{index:02d}",
            "shelf",
            [inner[0][0] + shelf_inset_m, inner[0][1] - shelf_inset_m],
            [inner[1][0], inner[1][1] - shelf_inset_m],
            [height - shelf_t / 2.0, height + shelf_t / 2.0],
        )

    if door_bin_count:
        if not door_link_path or door_bin_y_interval_m is None:
            output.unlink(missing_ok=True)
            raise ArticulatedInteriorAuthoringError(
                ["articulated_interior_door_bin_binding_missing"]
            )
        if not stage.GetPrimAtPath(str(door_link_path)).IsValid():
            output.unlink(missing_ok=True)
            raise ArticulatedInteriorAuthoringError(
                [f"articulated_interior_door_link_missing:{door_link_path}"]
            )
        bin_y = _interval(
            door_bin_y_interval_m, "articulated_interior_door_bin_interval_invalid"
        )
        # A bin inside the door's own thickness is not a bin. Measure the
        # door's existing geometry and require the bins to sit inboard of its
        # inner face, reaching into the cavity the way a real door pocket does.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        door_low, door_high = float("inf"), float("-inf")
        for existing in Usd.PrimRange(stage.GetPrimAtPath(str(door_link_path))):
            if not existing.IsA(UsdGeom.Mesh):
                continue
            bounds = cache.ComputeWorldBound(existing).ComputeAlignedRange()
            door_low = min(door_low, float(bounds.GetMin()[1]))
            door_high = max(door_high, float(bounds.GetMax()[1]))
        door_inner_face = door_low if door_low <= door_high else None
        if door_low <= door_high:
            overlap = min(door_high, bin_y[1]) - max(door_low, bin_y[0])
            if overlap > 1e-9:
                output.unlink(missing_ok=True)
                raise ArticulatedInteriorAuthoringError(
                    [
                        "articulated_interior_door_bin_intersects_door:"
                        f"overlap_m={overlap:.4f}"
                    ]
                )
        bin_scope = f"{door_link_path}/generated_door_bins"
        UsdGeom.Xform.Define(stage, bin_scope)
        span = (z[1] - z[0]) / (door_bin_count + 1)
        for index in range(door_bin_count):
            base = z[0] + (index + 1) * span - float(door_bin_height_m) / 2.0
            box(
                f"{bin_scope}/door_bin_{index:02d}",
                "door_bin",
                [x[0] + wall, x[1] - wall],
                bin_y,
                [base, base + float(door_bin_height_m)],
            )

    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output))

    free_volume = (
        (inner[0][1] - inner[0][0])
        * (inner[1][1] - inner[1][0])
        * (inner[2][1] - inner[2][0])
    ) - shelf_count * (
        (inner[0][1] - inner[0][0] - 2 * shelf_inset_m)
        * (inner[1][1] - inner[1][0] - shelf_inset_m)
        * shelf_t
    )

    receipt: dict[str, Any] = {
        "schema_version": INTERIOR_AUTHORING_SCHEMA_VERSION,
        "status": "articulated_interior_authored",
        "source_usd_path": str(source),
        "source_usd_sha256": _sha256(source),
        "interior_usd_path": str(output),
        "interior_usd_sha256": _sha256(output),
        "support_link_path": str(support_link_path),
        "removed_prim_paths": sorted(removed),
        "cavity": {
            "x_interval_m": x,
            "y_interval_m": y,
            "z_interval_m": z,
            "wall_thickness_m": wall,
            "inner_x_interval_m": list(inner[0]),
            "inner_y_interval_m": list(inner[1]),
            "inner_z_interval_m": list(inner[2]),
        },
        "shelves": {
            "count": shelf_count,
            "thickness_m": shelf_t,
            "inset_m": float(shelf_inset_m),
        },
        "door_bins": {
            "count": int(door_bin_count),
            "door_inner_face_m": door_inner_face,
        },
        "parts": parts,
        "free_volume_m3": round(float(free_volume), 6),
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
            "interior_never_observed": True,
            "matches_real_appliance_interior": False,
            "all_parts_tagged_generated_candidate_geometry": True,
            "source_usd_modified": False,
            "native_simulator_qualified": False,
        },
        "receipt_path": str(
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else output.with_name(output.stem + "_interior_receipt.json")
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(receipt["receipt_path"]), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ArticulatedInteriorAuthoringError",
    "GENERATED_VALUE",
    "INTERIOR_AUTHORING_SCHEMA_VERSION",
    "PROVENANCE_ATTRIBUTE",
    "author_articulated_interior",
]
