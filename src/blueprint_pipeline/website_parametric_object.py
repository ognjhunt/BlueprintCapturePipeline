"""Free parametric shapes for a job's object, and simulated variation of them (owner direction 2026-11-23).

The build plan (``website_build_plan``) names the sourcing of each object; for a plain carton, tote, open bin, tray or
can it is a parametric shape made from its dimensions, at no cost, before any paid generated model. This module makes
that shape deterministically from the dimensions the video gives (never a measurement), as the kind of asset the build
consumes from ARTISAN: a rigid body with its collision, its mass and its bounds.

``part_boxes`` and ``mesh`` are the geometry: a carton is one closed box; a tote, an open bin and a tray are a floor
and four walls (a tote also a rim around its top); a can is a closed cylinder. ``write_candidate`` writes it where the
object stood, as construction's source candidate, so the existing no-spend authoring (``provided_mesh_rigid_authoring``)
makes the replacement asset from it, exactly as it does a designed object's. ``author_asset`` writes the asset itself
(each part its own collider, so an open container stays open), for a build that has no construction stage.

``variants`` gives N deterministic variants of a shape within the plan's size and mass ranges, seeded by the job, each
labelled simulated variation, never a measurement. A range the plan does not give is not varied.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest

GENERATOR = "website_parametric_object.v1"
SOURCE = "parametric"
SHAPES = ("carton", "tote", "open_bin", "tray", "cylinder")
OPEN_CONTAINERS = ("tote", "open_bin", "tray")
#: An open container's wall: this share of its smaller footprint side, within these bounds (metres).
WALL_FRACTION = 0.04
WALL_BOUNDS_M = (0.002, 0.012)
#: A tote's rim: the wall's thickness out and down from its top edge, all inside the observed footprint.
RIM_DEPTH_WALLS = 3
CYLINDER_SECTIONS = 48
DIMENSION_BOUNDS_M = (0.01, 3.0)
MAX_VARIANTS = 32
VARIATION_LABEL = "simulated_variation"
CLAIM = ("A plain shape sized from the video's estimated dimensions, not a measurement; its variants are simulated "
         "variation within the plan's ranges.")


def design(shape: str, dimensions_m: Sequence[float]) -> dict[str, Any]:
    """The shape and its outer dimensions (x, y footprint; z height), checked; a can's diameter is the footprint's
    mean side."""
    dims = [float(v) for v in dimensions_m]
    if (shape not in SHAPES or len(dims) != 3 or any(isinstance(v, bool) for v in dimensions_m)
            or not all(math.isfinite(v) and DIMENSION_BOUNDS_M[0] <= v <= DIMENSION_BOUNDS_M[1] for v in dims)):
        raise ValueError("website_parametric_object_design_invalid")
    if shape == "cylinder":
        diameter = (dims[0] + dims[1]) / 2
        dims = [diameter, diameter, dims[2]]
    return {"shape": shape, "dimensions_m": [round(v, 6) for v in dims], "generator": GENERATOR}


def wall_m(chosen: Mapping[str, Any]) -> float:
    x, y, _z = chosen["dimensions_m"]
    return float(min(max(min(x, y) * WALL_FRACTION, WALL_BOUNDS_M[0]), WALL_BOUNDS_M[1]))


def part_boxes(chosen: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The shape's convex parts, centred on the object's middle (Z up, metres): each a box, or the can's cylinder."""
    x, y, z = chosen["dimensions_m"]
    if chosen["shape"] == "cylinder":
        return [{"part": "body", "kind": "cylinder", "center_m": [0.0, 0.0, 0.0], "extents_m": [x, y, z]}]
    if chosen["shape"] == "carton":
        return [{"part": "body", "kind": "box", "center_m": [0.0, 0.0, 0.0], "extents_m": [x, y, z]}]
    wall = wall_m(chosen)
    rim = wall if chosen["shape"] == "tote" else 0.0
    inner_x, inner_y = x - 2 * rim, y - 2 * rim  # the body's footprint, inside the rim
    low = -z / 2
    parts = [{"part": "floor", "kind": "box", "center_m": [0.0, 0.0, low + wall / 2],
              "extents_m": [inner_x, inner_y, wall]}]
    height = z - wall
    centre_z = low + wall + height / 2
    for name, cx, cy, ex, ey in (("wall_x_low", -(inner_x - wall) / 2, 0.0, wall, inner_y - 2 * wall),
                                 ("wall_x_high", (inner_x - wall) / 2, 0.0, wall, inner_y - 2 * wall),
                                 ("wall_y_low", 0.0, -(inner_y - wall) / 2, inner_x, wall),
                                 ("wall_y_high", 0.0, (inner_y - wall) / 2, inner_x, wall)):
        parts.append({"part": name, "kind": "box", "center_m": [cx, cy, centre_z], "extents_m": [ex, ey, height]})
    if rim:
        depth = min(RIM_DEPTH_WALLS * wall, height)
        top = z / 2 - depth / 2
        for name, cx, cy, ex, ey in (("rim_x_low", -(x - rim) / 2, 0.0, rim, y),
                                     ("rim_x_high", (x - rim) / 2, 0.0, rim, y),
                                     ("rim_y_low", 0.0, -(y - rim) / 2, inner_x, rim),
                                     ("rim_y_high", 0.0, (y - rim) / 2, inner_x, rim)):
            parts.append({"part": name, "kind": "box", "center_m": [cx, cy, top], "extents_m": [ex, ey, depth]})
    return [{**row, "center_m": [round(v, 6) for v in row["center_m"]],
             "extents_m": [round(v, 6) for v in row["extents_m"]]} for row in parts]


def _part_mesh(row: Mapping[str, Any]) -> Any:
    import trimesh
    if row["kind"] == "cylinder":
        part = trimesh.creation.cylinder(radius=row["extents_m"][0] / 2, height=row["extents_m"][2],
                                         sections=CYLINDER_SECTIONS)
    else:
        part = trimesh.creation.box(extents=row["extents_m"])
    part.apply_translation(row["center_m"])
    return part


def mesh(chosen: Mapping[str, Any]) -> Any:
    """The whole shape as one mesh, centred on the object's middle (Z up, metres)."""
    import trimesh
    return trimesh.util.concatenate([_part_mesh(row) for row in part_boxes(chosen)])


def _define_mesh(stage: Any, path: str, part: Any, offset: Sequence[float]) -> Any:
    from pxr import UsdGeom
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr((np.asarray(part.vertices) + np.asarray(offset, dtype=float)).astype(np.float32).tolist())
    prim.CreateFaceVertexCountsAttr([3] * len(part.faces))
    prim.CreateFaceVertexIndicesAttr(np.asarray(part.faces).ravel().tolist())
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return prim


def write_candidate(chosen: Mapping[str, Any], *, center_xyz_m: Sequence[float], path: Path) -> dict[str, Any]:
    """The shape as construction's source candidate, where the object stood (Z up, metres)."""
    from pxr import Usd, UsdGeom
    shape = mesh(chosen)
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    UsdGeom.Xform.Define(stage, "/Root/SourceObjectCandidate")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _define_mesh(stage, "/Root/SourceObjectCandidate/Parametric", shape, center_xyz_m)
    stage.GetRootLayer().Save()
    return {"source": SOURCE, "generator": GENERATOR, "design": dict(chosen), "face_count": int(len(shape.faces)),
            "part_count": len(part_boxes(chosen)), "authoring_cost_usd": 0.0}


def _checked_bounds(physics_bounds: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    keys = ("mass_kg", "static_friction", "dynamic_friction", "restitution")
    if set(physics_bounds) != set(keys):
        raise ValueError("website_parametric_object_physics_invalid")
    bounds = {key: [float(v) for v in physics_bounds[key]] for key in keys}
    if (any(len(row) != 2 or not all(math.isfinite(v) for v in row) or not 0 <= row[0] <= row[1]
            for row in bounds.values()) or bounds["mass_kg"][0] <= 0 or bounds["restitution"][1] > 1):
        raise ValueError("website_parametric_object_physics_invalid")
    return bounds


def author_asset(chosen: Mapping[str, Any], *, physics_bounds: Mapping[str, Sequence[float]],
                 output_root: Path) -> dict[str, Any]:
    """The rigid-body asset (``parametric_object.usda``): a visual mesh, one collider per convex part, the middle of the
    mass range as its mass, and its bounds; with its receipt. Deterministic: the same design writes the same bytes."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    bounds = _checked_bounds(physics_bounds)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "parametric_object.usda"
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(asset)
    UsdPhysics.RigidBodyAPI.Apply(asset)
    mass = (bounds["mass_kg"][0] + bounds["mass_kg"][1]) / 2
    UsdPhysics.MassAPI.Apply(asset).CreateMassAttr(mass)
    UsdGeom.Xform.Define(stage, "/Asset/Geometry")
    _define_mesh(stage, "/Asset/Geometry/Visual", mesh(chosen), (0.0, 0.0, 0.0))
    UsdGeom.Xform.Define(stage, "/Asset/Geometry/Collision")
    parts = part_boxes(chosen)
    for row in parts:
        prim = _define_mesh(stage, f"/Asset/Geometry/Collision/{row['part']}", _part_mesh(row), (0.0, 0.0, 0.0))
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("convexHull")
    material = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, "/Asset/PhysicsMaterial").GetPrim())
    material.CreateStaticFrictionAttr(sum(bounds["static_friction"]) / 2)
    material.CreateDynamicFrictionAttr(sum(bounds["dynamic_friction"]) / 2)
    material.CreateRestitutionAttr(sum(bounds["restitution"]) / 2)
    extent = [chosen["dimensions_m"][0] / 2, chosen["dimensions_m"][1] / 2, chosen["dimensions_m"][2] / 2]
    UsdGeom.Boundable(stage.GetPrimAtPath("/Asset/Geometry/Visual")).CreateExtentAttr(
        [Gf.Vec3f(*(-v for v in extent)), Gf.Vec3f(*extent)])
    stage.GetRootLayer().Save()
    data = path.read_bytes()
    receipt = {"schema_version": "website_parametric_object_asset.v1", "source": SOURCE, "generator": GENERATOR,
               "design": dict(chosen), "collision_parts": parts, "mass_kg": mass, "physics_bounds": bounds,
               "bounds_m": {"minimum": [-v for v in extent], "maximum": extent},
               "asset": {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)},
               "authoring_cost_usd": 0.0, "model_called": False, "physical_measurement": False, "claim": CLAIM}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    return receipt


def _unit(seed: str, index: int, quantity: str) -> float:
    digest = hashlib.sha256(f"{GENERATOR}|{seed}|{index}|{quantity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def variants(chosen: Mapping[str, Any], plan_row: Mapping[str, Any], *, seed: str, count: int) -> dict[str, Any]:
    """``count`` variants of the shape within the plan's ranges, the same for the same job (``seed``): the largest side
    scaled into the plan's ``size_m`` range (every side by the same factor), the mass drawn from its ``mass_kg`` range.
    A range the plan does not give is not varied, and nothing is ever called a measurement."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_VARIANTS or not seed:
        raise ValueError("website_parametric_object_variants_invalid")
    size, mass = plan_row.get("size_m"), plan_row.get("mass_kg")
    largest = max(chosen["dimensions_m"])
    rows = []
    for index in range(count):
        row: dict[str, Any] = {"index": index, "label": VARIATION_LABEL}
        if isinstance(size, Mapping):
            side = size["low"] + (size["high"] - size["low"]) * _unit(seed, index, "size")
            row["design"] = design(chosen["shape"], [v * side / largest for v in chosen["dimensions_m"]])
        else:
            row["design"] = dict(chosen)
        if isinstance(mass, Mapping):
            row["mass_kg"] = round(mass["low"] + (mass["high"] - mass["low"]) * _unit(seed, index, "mass"), 6)
        rows.append(row)
    varied = [name for name, value in (("size_m", size), ("mass_kg", mass)) if isinstance(value, Mapping)]
    value = {"schema_version": "website_parametric_object_variants.v1", "generator": GENERATOR, "seed": seed,
             "label": VARIATION_LABEL, "varied": varied,
             "ranges": {name: dict(plan_row[name]) for name in varied}, "variants": rows,
             "physical_measurement": False, "claim": CLAIM}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


__all__ = ["CLAIM", "GENERATOR", "OPEN_CONTAINERS", "SHAPES", "SOURCE", "VARIATION_LABEL", "author_asset", "design",
           "mesh", "part_boxes", "variants", "write_candidate"]
