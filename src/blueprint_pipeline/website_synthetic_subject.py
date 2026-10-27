"""A designed object a task adds to a captured scene, ADP-030 day 28.

Nothing about the object was observed. Its design is the task analysis's
proposal (``synthetic_design_choice``); its geometry is a deterministic
primitive; its start is the clear spot on the observed support the analysis
named, closest to the task's destination, with the column above it free for a
top-down approach. Construction authors it exactly as it does an
owner-provided mesh (``provided_mesh_rigid_authoring``): the same static,
native-import and assembly qualification follow, and no provider is called.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ORIGIN = "synthetic_design"
AUTHORING_METHOD = "exact_provided_mesh_with_bounded_simulation_physics"
AUTHORING_ADAPTER = "provided_mesh_rigid_authoring"
#: The support region is searched on this grid. A spot keeps this clearance from
#: anything else and, for the hand coming down to it, prefers open space this high
#: above it (not under a desk); only when no spot is open does a short column do.
START_SEARCH_STEP_M = 0.05
START_CLEARANCE_M = 0.02
APPROACH_COLUMNS_M = (1.0, 0.30)
#: Support lookups copy the collider, so only this many clear spots are tried.
MAXIMUM_SUPPORT_LOOKUPS = 24
#: Ranges for an object nobody measured; the mass keeps the proposal within 10 %.
STATIC_FRICTION = (0.5, 0.8)
DYNAMIC_FRICTION = (0.4, 0.7)
RESTITUTION = (0.1, 0.4)
MASS_RELATIVE_RANGE = 0.1
SPHERE_SUBDIVISIONS = 4
CYLINDER_SECTIONS = 48


def design_physics(design: Mapping[str, Any]) -> dict[str, Any]:
    """The bounded simulation priors of the design, in ``screen_physics``'s shape."""
    dimensions = [float(v) for v in design["dimensions_m"]]
    mass = float(design["mass_kg"])
    return {"basis": ORIGIN, "dimensions_m": dimensions, "envelope_volume_m3": float(np.prod(dimensions)),
            "bounds": {"mass_kg": [mass * (1 - MASS_RELATIVE_RANGE), mass * (1 + MASS_RELATIVE_RANGE)],
                       "static_friction": list(STATIC_FRICTION), "dynamic_friction": list(DYNAMIC_FRICTION),
                       "restitution": list(RESTITUTION)},
            "priors": {"design_basis": "synthetic_design_choice", "physical_measurement": False},
            "sensitivity": "awaiting_robot_team_selection", "measurement_escalation": None}


def primitive_mesh(design: Mapping[str, Any]) -> Any:
    """The design centred at the origin, Z up, metres."""
    import trimesh

    dimensions = [float(v) for v in design["dimensions_m"]]
    if design["shape"] == "sphere":
        return trimesh.creation.icosphere(subdivisions=SPHERE_SUBDIVISIONS, radius=dimensions[0] / 2.0)
    if design["shape"] == "cylinder":
        return trimesh.creation.cylinder(radius=dimensions[0] / 2.0, height=dimensions[2],
                                         sections=CYLINDER_SECTIONS)
    return trimesh.creation.box(extents=dimensions)


def write_candidate(design: Mapping[str, Any], *, center_xyz_m: Sequence[float], path: Path) -> dict[str, Any]:
    """The design as construction's source candidate, where it starts (Z up, metres)."""
    from pxr import Usd, UsdGeom

    mesh = primitive_mesh(design)
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    UsdGeom.Xform.Define(stage, "/Root/SourceObjectCandidate")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Mesh.Define(stage, "/Root/SourceObjectCandidate/Designed")
    prim.CreatePointsAttr((np.asarray(mesh.vertices) + np.asarray(center_xyz_m, dtype=float)).astype(np.float32).tolist())
    prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    prim.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).ravel().tolist())
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return {"shape": design["shape"], "face_count": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
            "sphere_subdivisions": SPHERE_SUBDIVISIONS, "cylinder_sections": CYLINDER_SECTIONS}


def choose_start(*, collider: Any, region: Sequence[Sequence[float]], attractor: Sequence[float],
                 design: Mapping[str, Any], up: int, up_sign: int, meters_per_unit: float,
                 support_under: Callable[..., dict[str, Any] | None]
                 ) -> tuple[list[float] | None, list[float] | None, dict[str, Any] | None, dict[str, Any]]:
    """The clear spot on the region's support closest to ``attractor``, in runtime units.

    Returns the object's bounds resting on the support, the support, and a report;
    the bounds and support are ``None`` when the region has no clear spot.
    """
    unit = 1.0 / meters_per_unit
    horizontal = [axis for axis in range(3) if axis != up]
    half = [float(design["dimensions_m"][0]) * unit / 2.0, float(design["dimensions_m"][1]) * unit / 2.0]
    height = float(design["dimensions_m"][2]) * unit
    low, high = np.asarray(region[0], dtype=float), np.asarray(region[1], dtype=float)
    # Heights along the scene's up direction, whichever way its axis points.
    triangles = np.asarray(collider.triangles, dtype=float)
    span_low, span_high = triangles.min(axis=1), triangles.max(axis=1)
    lift = lambda value: up_sign * value  # noqa: E731
    region_top, region_bottom = max(lift(low[up]), lift(high[up])), min(lift(low[up]), lift(high[up]))
    margin = half[0] + half[1] + START_CLEARANCE_M * unit
    near = np.all([(span_high[:, a] >= low[a] - margin) & (span_low[:, a] <= high[a] + margin) for a in horizontal],
                  axis=0)
    nearby_low, nearby_high = span_low[near], span_high[near]
    step = START_SEARCH_STEP_M * unit
    axes = [np.arange(low[a] + half[i], high[a] - half[i] + 1e-9, step) if high[a] - low[a] > 2 * half[i]
            else np.array([(low[a] + high[a]) / 2.0]) for i, a in enumerate(horizontal)]
    spots = sorted(((x, y) for x in axes[0] for y in axes[1]),
                   key=lambda s: (math.hypot(s[0] - attractor[horizontal[0]], s[1] - attractor[horizontal[1]]), s))
    report = {"rule": "closest_clear_spot_on_named_support_to_destination", "candidate_spots": len(spots),
              "clearance_m": START_CLEARANCE_M, "approach_columns_m": list(APPROACH_COLUMNS_M),
              "search_step_m": START_SEARCH_STEP_M, "support_lookups": 0, "clear_spots_seen": 0}

    def box(spot: Sequence[float], bottom: float, top: float) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = np.zeros(3), np.zeros(3)
        for i, axis in enumerate(horizontal):
            lower[axis], upper[axis] = spot[i] - half[i], spot[i] + half[i]
        lower[up], upper[up] = (bottom, top) if up_sign == 1 else (-top, -bottom)
        return lower, upper

    for column, surface in ((c, s) for c in APPROACH_COLUMNS_M for s in dict.fromkeys((region_top, region_bottom))):
        for spot in spots:
            lower, upper = box(spot, surface + START_CLEARANCE_M * unit, surface + height + column * unit)
            grown = [lower - START_CLEARANCE_M * unit, upper + START_CLEARANCE_M * unit]
            for axis in (up,):
                grown[0][axis], grown[1][axis] = lower[axis], upper[axis]
            if np.any(np.all((nearby_high >= grown[0]) & (nearby_low <= grown[1]), axis=1)):
                continue
            report["clear_spots_seen"] += 1
            if report["support_lookups"] >= MAXIMUM_SUPPORT_LOOKUPS:
                break
            report["support_lookups"] += 1
            lower, upper = box(spot, surface, surface + height)
            support = support_under(collider, lower, upper, up=up, meters_per_unit=meters_per_unit, up_sign=up_sign)
            if support is None:
                continue
            lower, upper = box(spot, lift(support["top_runtime_units"]), lift(support["top_runtime_units"]) + height)
            report.update(surface_guess_runtime_units=float(up_sign * surface), clear_column_m=column,
                          spot_runtime_units=[float(v) for v in spot])
            return lower.tolist(), upper.tolist(), support, report
    return None, None, None, report
