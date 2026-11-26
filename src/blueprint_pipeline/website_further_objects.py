"""Several objects per job, stage 3: each free further object authored on the CPU and placed where it was filmed.

A job's build plan ranks up to eight objects (``website_build_plan.built_objects``); the first is the job's own object,
the one the success rule scores (``task.subject``, rank 1). Each one after it is a further object, ranked 2 to 8
(``website_build_plan.built_rank``), with the free source it could use (``website_extra_objects``). This module turns
those into objects of the job's scene, beside the scored one:

- ``further_inputs``: compile's input, one row per further object, read from what is already sealed (the free source,
  and for a library model its sealed pick); a caller passes it as ``further_objects``. On the live path the handoff
  reads it for a decision sealed under v3 only (stage 5, ``sealed_inputs``): the job's Atlas decision took out of the
  scan, with the job's object and before the room was reconstructed, each further object sure of a free source
  (``website_build_plan.further_removal``, marked ``BUILT_EXTRA`` on the removal manifest), and kept every other one
  in the scan (``NOT_BUILT``, with why). A decision sealed earlier has no further object.
- ``scored_entries``: the removal manifest's moved objects compile reads as the job's own. A further object composition
  may take is not one of them, so it never holds the job as a several-object one (``MULTI_OBJECT_GAP``), in compile
  and in the handoff alike; any other several-object job is held as before.
- ``place``: each further object at its own filmed pose, through the scored object's registration (its box from its
  own masks) and the same support snap (it rests on whatever surface is under it in the scan), authored as free
  supplied geometry: its library model when that model is sealed, fetched, still allowed and credited, else the plan's
  parametric shape to fall back on; or the plan's parametric shape that is not a can. Each authored asset passes a
  static check (``static_check``) and is seated by its own file's lowest point on that surface (``seat``): a model
  taller or shorter than the estimate, within the tolerance, neither sinks into it nor floats. The intake request
  names each one placed (``task.additional_objects``).
- ``compose``: the further objects the intake request took (``usable_additional_objects``, never the raw field), each
  with its own replacement id and compose-back record (``website_preparation_request``), as the preparation's
  ``further_objects`` record. Each gets the prim ``/World/Object_02`` onward in the order composed, the runtime name
  ``job_object_02`` onward, and the role ``job_object`` (``ROLE``) in the scene half and its plan.
- ``for_revision`` and ``rows``: what a scene half (``team_scene_packet``) adds after its task row, read on the host
  from the preparation the half's revision was built from; ``refused_rows_dropped``: a packet writer that refuses the
  extra rows has them dropped, flagged ``further_objects_left_out``, and the half goes on.

A paid further object (one that needs ARTISAN, the paid generated model) the decision reserved on its own row of the
job's ledger and took out of the scan (stage 6) is placed as a free one is and authored with ARTISAN under that
reservation alone, then settled at what it cost (``website_further_artisan``). A further object the decision kept in
the scan (one that needs ARTISAN where no reservation was made, or one a limit on the job's paid work refused) is
left in the scan as filmed: named with its reason on the record (``left_in_scan``) and flagged
``scene_estimate:further_object_left_in_scan``. One taken out of the scan (its removal manifest row ``BUILT_EXTRA``)
that has no track, no registered scene or no surface under it, that overlaps the scored object, its destination or
another further object, whose authoring or static check fails, or that the request does not take, is in neither the
scan nor the simulation: named with its reason (``removed_not_built``) and flagged
``scene_estimate:further_object_removed_not_built``, never said to be in the scan. Neither is ever a stop. Only
privacy, rights and money stop a job: a library model whose rights cannot be confirmed now is never used. Nothing here
reserves, asks a model or fetches beyond what the library step already pinned; a paid object spends only under the
reservation the Atlas job made for it. The success rule still scores only the scored object (``scene_package_stage``).
"""
from __future__ import annotations

import hashlib
import math
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "website_further_objects.v1"
REVISION = "website_further_objects.r1"
#: A composed further object's role row in a scene half and its plan, after the task row (``scene_package_stage``).
ROLE = "job_object"
#: The scored object's prim in the company's scene; each further object's is ``/World/Object_02`` onward.
SCORED_PRIM = "/World/Object"
DIRECTORY = "further_objects"
LIBRARY, PARAMETRIC = "library", "parametric"
#: The flag a further object left in the scan as filmed raises (``atlas_scene_estimates``); the job goes on.
LEFT_IN_SCAN = "scene_estimate:further_object_left_in_scan"
#: The flag a further object taken out of the scan and then not built raises: it is in neither; the job goes on.
REMOVED_NOT_BUILT = "scene_estimate:further_object_removed_not_built"
#: An input row's reason it is not to be built (``sealed_inputs``): kept in the scan by the decision, or not read as
#: the decision read it. ``place`` names it with that reason and builds nothing for it.
NOT_BUILT = "not_built"
#: ``place``'s mark on a row left unbuilt whose object was taken out of the scan; ``compose`` reads it.
_TAKEN_OUT = "taken_out_of_scan"
#: A scene half whose packet writer refused the extra rows: they are dropped, and the half goes on.
LEFT_OUT = "further_objects_left_out"
#: Why a further object is left in the scan as filmed; a ``:`` adds the detail (the S1 reason, what it overlaps, the
#: authoring's own code, the check that failed). None stops the job.
INPUT_INVALID = "further_object_input_invalid"
NEEDS_ARTISAN = "further_object_needs_artisan"
NOT_TRACKED = "further_object_not_tracked"
NO_SCENE = "further_object_scene_not_registered"
SUPPORT_NOT_FOUND = "further_object_support_not_found"
OVERLAPS = "further_object_overlaps"
AUTHORING_FAILED = "further_object_authoring_failed"
STATIC_CHECK_FAILED = "further_object_static_check_failed"
#: Anything else that failed for one further object, by its error's type: that object alone stays in the scan.
FAILED = "further_object_failed"
#: The intake request left the further objects out (``website_preparation_request.intake_request``).
NOT_TAKEN = "website_preparation_further_objects_left_out"
LIBRARY_NOT_PICKED = "website_extra_object_library_not_picked"
#: Two boxes that share less than this on every axis (metres) only touch.
TOUCH_M = 1e-6
#: The authored file's bounds are float32 points: this much slack beside the size tolerance (metres).
FLOAT_SLACK_M = 1e-5
ASSET_FILE = "library_object.usda"
CLAIM = ("A further object of the job, authored as free supplied geometry sized from the video's estimated bounds and "
         "placed where the video shows it; never a measurement. The success rule does not score it.")


def _left(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {"rank": row.get("rank"), "target_id": str(row.get("target_id")), "reason": str(reason)[:300]}


def further_inputs(plan: Mapping[str, Any] | None, *, entries: Sequence[Mapping[str, Any]],
                   editor_objects: Sequence[Mapping[str, Any]], task_context: Mapping[str, Any], library_root: Path,
                   catalog_entries: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Compile's ``further_objects``: each further object of the plan (``website_extra_objects.extra_object_sources``),
    by rank, with its free source and, for a library model, its sealed pick (``library_subject``). Reads only."""
    from .website_extra_objects import extra_object_sources, object_root, sealed_library_model
    record = extra_object_sources(plan, entries=entries, editor_objects=editor_objects, task_context=task_context,
                                  library_root=library_root, catalog_entries=catalog_entries)
    rows = {row["target_id"]: row for row in (plan or {}).get("objects") or []}
    found = []
    for row in record["objects"]:
        item = {key: row[key] for key in ("rank", "target_id", "name", "source", "parametric_shape", "fallback_shape",
                                          "needs_artisan") if key in row}
        if row["source"] == LIBRARY:
            item["library_subject"], _ = sealed_library_model(
                rows[row["target_id"]], root=object_root(library_root, row["target_id"]), task_context=task_context,
                catalog_entries=catalog_entries)
        found.append(item)
    return found


def sealed_inputs(decided: Mapping[str, Any] | None, *, entries: Sequence[Mapping[str, Any]],
                  task_context: Mapping[str, Any], library_root: Path,
                  catalog_entries: Sequence[Mapping[str, Any]] | None = None,
                  budget_root: Path | None = None) -> list[dict[str, Any]]:
    """Compile's ``further_objects`` for a decision sealed under v3 (``website_build_plan_notes.further_rows``), else
    none. ``further_inputs`` is read as the decision read the plan: the editor's rows of the objects it would build and
    the clean plate's own ``entries`` (before the handoff marks them), one row per further object the decision names,
    by the rank it sealed. One it took out of the scan comes with its free source; one it kept in the scan is marked
    ``NOT_BUILT`` with the reason it sealed, so composition never builds an object still in the scan; one taken out
    that is no longer read as it was (another rank, no row, a fault reading it) is marked ``NOT_BUILT`` too
    (``INPUT_INVALID``, ``FAILED``). A paid one the decision reserved (stage 6, ``website_build_budget.reserve_at_removal``)
    carries its reservation (``paid``, with ``budget_root``, the job's build budget), built or not, so compile authors it
    under that reservation alone and settles it either way (``website_further_artisan``). Reads only; never a stop."""
    from .hard_stops import PRIVACY_RIGHTS, wall_kind
    from .website_build_plan_notes import further_rows
    sealed = further_rows(decided)
    if not sealed or not isinstance((decided or {}).get("built"), Mapping):
        return []
    editor = [{"target_id": item, "disposition": "remove"}
              for row in (decided["built"], *sealed) for item in row.get("editor_object_ids") or []]
    failed = None
    try:
        read = {row["target_id"]: row for row in further_inputs(
            ((decided.get("build_plan") or {}).get("plan")), entries=entries, editor_objects=editor,
            task_context=task_context, library_root=library_root, catalog_entries=catalog_entries)}
    except Exception as exc:  # noqa: BLE001 - a further object never stops the scored object's job
        if wall_kind(exc) == PRIVACY_RIGHTS:
            raise
        read, failed = {}, f"{FAILED}:{type(exc).__name__}"
    found = []
    for row in sealed:
        named = {"rank": row["rank"], "target_id": row["target_id"], "name": row.get("name")}
        item = read.get(row["target_id"])
        paid = ({"paid": {**row["paid"], "budget_root": str(budget_root)}}
                if isinstance(row.get("paid"), Mapping) and budget_root is not None else {})
        if row.get("removed") is not True:
            found.append({**named, NOT_BUILT: str(row.get("kept_in_scan_reason") or INPUT_INVALID), **paid})
        elif item is None or item.get("rank") != row["rank"]:
            found.append({**named, NOT_BUILT: failed or INPUT_INVALID, **paid})
        else:
            found.append({**item, **paid})
    return found


def _problem(rows: Sequence[Mapping[str, Any]], *, strategy: str, origin: str) -> str | None:
    """Why the intake request would not take these further objects (``additional_objects_problem``), checked before
    anything is placed: their ranks, the task's strategy and their geometry's origin; each target named once."""
    from .task_evaluation_scene_intent_contracts import ADDITIONAL_OBJECTS_FIELD, additional_objects_problem
    targets = [row.get("target_id") for row in rows]
    if any(not isinstance(target, str) or not target for target in targets) or len(set(targets)) != len(targets):
        return INPUT_INVALID
    return additional_objects_problem({"strategy": strategy, ADDITIONAL_OBJECTS_FIELD: [
        {"rank": row.get("rank"), "subject": {"geometry_origin": origin}} for row in rows]})


def _rows(further_objects: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(row) for row in further_objects or () if isinstance(row, Mapping)]


def scored_entries(entries: Sequence[Mapping[str, Any]],
                   further_objects: Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    """The removal manifest's ``manipulated`` rows compile reads as the job's own object: every one but a further
    object composition may take (pick and place, ranks as the contract has them). Without further objects, exactly the
    ``manipulated`` rows, so any other several-object job is held as before."""
    rows = _rows(further_objects)
    taken = set() if not rows or _problem(rows, strategy="pick_and_place", origin="removed_before_reconstruction") \
        else {str(row["target_id"]) for row in rows}
    return [row for row in entries if row.get("task_effect") == "manipulated" and str(row.get("target_id")) not in taken]


def _box(low: Sequence[float], high: Sequence[float], matrix: Any) -> tuple[list[float], list[float]]:
    """A runtime box in another frame (``matrix``), enclosing its moved corners."""
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
                       dtype=float)
    moved = corners @ np.asarray(matrix, dtype=float)[:3, :3].T + np.asarray(matrix, dtype=float)[:3, 3]
    return moved.min(axis=0).tolist(), moved.max(axis=0).tolist()


def _overlap(first: tuple[Sequence[float], Sequence[float]], second: tuple[Sequence[float], Sequence[float]]) -> bool:
    return all(min(first[1][i], second[1][i]) - max(first[0][i], second[0][i]) > TOUCH_M for i in range(3))


def _destination_box(destination: Mapping[str, Any] | None, low: Sequence[float],
                     high: Sequence[float]) -> tuple[list[float], list[float]] | None:
    """The scored object's box where its destination sets it down (simulator frame, Z up), when it has one."""
    position = (destination or {}).get("position_world_m")
    if not (isinstance(position, (list, tuple)) and len(position) == 3):
        return None
    half = [(float(high[i]) - float(low[i])) / 2 for i in range(2)]
    return ([float(position[0]) - half[0], float(position[1]) - half[1], float(position[2])],
            [float(position[0]) + half[0], float(position[1]) + half[1],
             float(position[2]) + float(high[2]) - float(low[2])])


def _library_design(row: Mapping[str, Any], dimensions: Sequence[float], tolerance: float,
                    root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The sealed library model sized to the object, or None and why: rights are read as they stand now (the deployed
    catalog), and a model whose credit cannot be read is never used."""
    subject = row.get("library_subject")
    if not isinstance(subject, Mapping):
        return None, LIBRARY_NOT_PICKED
    from .website_library_catalog import rights_now
    refused = rights_now(subject.get("attribution"))
    if refused:
        return None, refused
    from .website_library_asset import library_design
    from .website_library_notes import credit
    design, why = library_design(subject, estimated_m=dimensions, tolerance=tolerance, output_root=root / "library_model")
    if design is None:
        return None, why
    credited = credit({"library_design": design}, root)
    return (None, str(credited["library_refusal"])) if "library_refusal" in credited else (design, None)


def _mesh_prim(stage: Any, path: str, points: Any, faces: Any) -> Any:
    from pxr import UsdGeom
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr(np.asarray(points, dtype=np.float32).tolist())
    prim.CreateFaceVertexCountsAttr([3] * len(faces))
    prim.CreateFaceVertexIndicesAttr(np.asarray(faces, dtype=np.int64).ravel().tolist())
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return prim


def author_library_asset(design: Mapping[str, Any], *, physics_bounds: Mapping[str, Sequence[float]],
                         output_root: Path) -> dict[str, Any]:
    """The rigid-body asset of a library model (``library_object.usda``): the model's own mesh as its look, centred on
    its middle and turned as its sizing says, its convex hull as its collider, the middle of the mass range as its mass;
    with its receipt. The model's own file is checked first. Deterministic."""
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from .website_parametric_object import _checked_bounds
    bounds = _checked_bounds(physics_bounds)
    model = design["model_file"]
    if hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest() != model["sha256"]:
        raise ValueError("library_model_file_changed")
    source = Usd.Stage.Open(str(model["path"]))
    mesh = UsdGeom.Mesh(source.GetPrimAtPath("/LibraryModel/Mesh"))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float) - np.asarray(design["model_centre_m"], dtype=float)
    if design["sizing"]["quarter_turns_about_up"] % 2:
        points = np.stack([-points[:, 1], points[:, 0], points[:, 2]], axis=1)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    hull = trimesh.convex.convex_hull(points)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ASSET_FILE
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
    visual = _mesh_prim(stage, "/Asset/Geometry/Visual", points, faces)
    low, high = points.min(axis=0), points.max(axis=0)
    UsdGeom.Boundable(visual.GetPrim()).CreateExtentAttr([Gf.Vec3f(*low.tolist()), Gf.Vec3f(*high.tolist())])
    UsdGeom.Xform.Define(stage, "/Asset/Geometry/Collision")
    collider = _mesh_prim(stage, "/Asset/Geometry/Collision/hull", hull.vertices, hull.faces)
    UsdGeom.Imageable(collider).CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
    material = UsdPhysics.MaterialAPI.Apply(UsdGeom.Scope.Define(stage, "/Asset/PhysicsMaterial").GetPrim())
    material.CreateStaticFrictionAttr(sum(bounds["static_friction"]) / 2)
    material.CreateDynamicFrictionAttr(sum(bounds["dynamic_friction"]) / 2)
    material.CreateRestitutionAttr(sum(bounds["restitution"]) / 2)
    stage.GetRootLayer().Save()
    data = path.read_bytes()
    receipt = {"schema_version": "website_further_object_library_asset.v1", "source": LIBRARY,
               "entry_id": design["entry_id"], "model_file_sha256": model["sha256"], "mass_kg": mass,
               "physics_bounds": bounds, "bounds_m": {"minimum": low.tolist(), "maximum": high.tolist()},
               "asset": {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)},
               "authoring_cost_usd": 0.0, "model_called": False, "physical_measurement": False, "claim": CLAIM}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    return receipt


def author(row: Mapping[str, Any], *, dimensions: Sequence[float], tolerance: float, physics_bounds: Mapping[str, Any],
           output_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """One further object authored as free supplied geometry, or None and why: its library model first (with what it
    falls back on when that model cannot be used, said why), else its parametric shape, which must meet its estimated
    bounds within ``tolerance`` (a can on an oblong footprint does not)."""
    from .website_parametric_object import author_asset, design
    root = Path(output_root)
    not_used = None
    if row.get("source") == LIBRARY:
        chosen, not_used = _library_design(row, dimensions, tolerance, root)
        if chosen is not None:
            try:
                receipt = author_library_asset(chosen, physics_bounds=physics_bounds, output_root=root)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                not_used = "library_asset_unwritten:" + type(exc).__name__
            else:
                return {"source": LIBRARY, "library_design": chosen, "receipt": receipt, "path": root / ASSET_FILE}, None
        shape = row.get("fallback_shape")
    else:
        shape = row.get("parametric_shape")
    try:
        shaped = design(shape, dimensions)
        if any(abs(made - seen) > tolerance * seen + 1e-6 for made, seen in zip(shaped["dimensions_m"], dimensions)):
            raise ValueError("website_parametric_object_outside_estimated_bounds")
        receipt = author_asset(shaped, physics_bounds=physics_bounds, output_root=root)
    except (TypeError, ValueError) as exc:
        why = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return None, f"{AUTHORING_FAILED}:{why}" + (f";library_not_used:{not_used}" if not_used else "")
    return {"source": PARAMETRIC, "parametric_design": shaped, "receipt": receipt,
            "path": root / receipt["asset"]["path"], **({"library_not_used": not_used} if not_used else {})}, None


def static_check(path: Path, *, dimensions_m: Sequence[float], tolerance: float) -> str | None:
    """Why an authored further object fails the static check, or None: one self-contained file whose default prim is
    its one rigid body, with a positive finite mass, at least one collider, no joint, finite points and faces that
    index them, and bounds within ``tolerance`` of the object's estimated size. Reads only."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(str(path))
        if [layer for layer in stage.GetUsedLayers() if layer != stage.GetSessionLayer()] != [stage.GetRootLayer()]:
            return "not_self_contained"
        root = stage.GetDefaultPrim()
        prims = list(stage.Traverse())
        if not root or [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)] != [root]:
            return "not_one_rigid_body"
        mass = UsdPhysics.MassAPI(root).GetMassAttr().Get()
        if not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0:
            return "mass_invalid"
        if not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims):
            return "no_collider"
        if any(prim.IsA(UsdPhysics.Joint) for prim in prims):
            return "has_joint"
        for prim in prims:
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], dtype=float)
                indices = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
                if (not len(points) or not np.isfinite(points).all() or not len(indices) or indices.min() < 0
                        or indices.max() >= len(points)):
                    return "mesh_invalid"
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
        box = cache.ComputeWorldBound(root).ComputeAlignedRange()
        made = [float(box.GetMax()[i] - box.GetMin()[i]) for i in range(3)]
    except Exception as exc:  # noqa: BLE001 - a file that cannot be read fails the check, by its error's type
        return "unreadable:" + type(exc).__name__
    if any(abs(size - seen) > tolerance * seen + FLOAT_SLACK_M for size, seen in zip(made, dimensions_m)):
        return "bounds_outside_estimate"
    return None


def seat(asset_bounds: Mapping[str, Sequence[float]], *, estimated: tuple[Sequence[float], Sequence[float]],
         runtime_to_sim: Any) -> dict[str, Any]:
    """Where an authored further object stands (simulator frame, Z up, metres): its file's origin over the middle of
    its estimated footprint, and its file's own lowest point (``asset_bounds``, its receipt's bounds about that origin)
    on the support the estimate was snapped to, the estimate's bottom. So a library model inside the size tolerance
    but taller or shorter than the video's estimate neither sinks into the surface nor floats above it (Codex review of
    #3755). ``position``: the origin, the scene's pose; ``bounds``: the file's box there; ``runtime``: that box about
    the origin in runtime units, whose centre is the compose-back record's position."""
    low, high = [float(v) for v in asset_bounds["minimum"]], [float(v) for v in asset_bounds["maximum"]]
    position = [(float(estimated[0][i]) + float(estimated[1][i])) / 2 for i in range(2)] + [
        float(estimated[0][2]) - low[2]]
    half = [(high[i] - low[i]) / 2 for i in range(3)]
    runtime = _box([position[i] - half[i] for i in range(3)], [position[i] + half[i] for i in range(3)],
                   np.linalg.inv(np.asarray(runtime_to_sim, dtype=float)))
    return {"position": position, "bounds": ([position[i] + low[i] for i in range(3)],
                                             [position[i] + high[i] for i in range(3)]),
            "runtime": runtime, "asset_bottom_m": low[2]}


def place(further_objects: Sequence[Mapping[str, Any]] | None, *, entries: Sequence[Mapping[str, Any]],
          targets: Mapping[str, Mapping[str, Any]], frames: Sequence[Mapping[str, Any]], matrix: Any,
          runtime_to_sim: Any, residual: float | None, collider: Any, up: int, up_sign: int, mpu: float,
          scored: tuple[Sequence[float], Sequence[float]], destination: Mapping[str, Any] | None, strategy: str,
          origin: str, tolerance: float, output_root: Path,
          task_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Each further object placed and authored (``objects``), each one not built with why (``left_in_scan``; marked
    for ``compose`` when its removal manifest row says it was taken out of the scan, ``BUILT_EXTRA``), and the records
    the intake request names (``requested``, ``website_preparation_request.additional_objects``). A row marked
    ``NOT_BUILT`` is named with its reason and nothing is authored for it.

    ``scored``: the scored object's box at its start (runtime units), ``destination`` its destination (simulator frame);
    ``matrix``, ``runtime_to_sim``, ``residual``, ``up``, ``up_sign`` and ``mpu`` are compile's own registration and
    frame, and ``collider`` the sealed collider it placed the scored object on (None without a registered scene).

    A paid further object (its row's ``paid`` reservation, stage 6) is placed as a free one is and authored with ARTISAN
    under that reservation alone (``website_further_artisan.author``, with ``task_context``), one at a time in rank
    order; whatever becomes of it, its reservation is settled at what its authoring cost, released when nothing was
    spent (``close``). A privacy or rights stop on its path is raised; any other refusal (the account's month among
    them) leaves that object not built, never the scored one."""
    rows = _rows(further_objects)
    empty: dict[str, Any] = {"objects": [], "left_in_scan": [], "requested": []}
    if not rows:
        return empty
    from .website_build_plan import BUILT_EXTRA
    by_target = {str(row.get("target_id")): row for row in entries if isinstance(row, Mapping)}

    def unbuilt(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
        # Taken out of the scan (``BUILT_EXTRA``) and not built: ``compose`` names it so, never as in the scan.
        taken = (by_target.get(str(row.get("target_id"))) or {}).get("task_effect") == BUILT_EXTRA
        return {**_left(row, reason), **({_TAKEN_OUT: True} if taken else {})}
    problem = _problem(rows, strategy=strategy, origin=origin)
    if problem is not None:
        reason = problem if problem == INPUT_INVALID else "scene_intake_" + problem
        _close_paid(rows, task_context)
        return {**empty, "left_in_scan": [unbuilt(row, row.get(NOT_BUILT) or reason) for row in rows]}
    from .website_support_geometry import nearest_support_under, support_under
    from .website_task_masks import estimate_target_bounds
    from .website_task_preparation import screen_physics
    start = _box(*scored, runtime_to_sim)
    boxes = [("scored_object_start", start)]
    end = _destination_box(destination, *start)
    if end is not None:
        boxes.append(("scored_object_destination", end))
    placed, left = [], []

    def one(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        rank, target_id = int(row["rank"]), str(row["target_id"])
        target = targets.get(target_id)
        if row.get(NOT_BUILT):
            return None, str(row[NOT_BUILT])
        if row.get("source") not in (LIBRARY, PARAMETRIC) and not isinstance(row.get("paid"), Mapping):
            return None, NEEDS_ARTISAN + ":" + str((row.get("needs_artisan") or {}).get("reason") or "")
        if collider is None:
            return None, NO_SCENE
        try:
            bounds = estimate_target_bounds(target["track"], list(frames), source_to_target=np.asarray(matrix))
        except (KeyError, TypeError, ValueError, IndexError):
            return None, NOT_TRACKED
        low, high = [float(v) for v in bounds["minimum"]], [float(v) for v in bounds["maximum"]]
        options = {"up": up, "meters_per_unit": mpu, "up_sign": up_sign, "placement_residual_m": residual}
        support = support_under(collider, low, high, **options)
        basis = "support_under_its_footprint"
        if support is None:
            support, basis = nearest_support_under(collider, low, high, **options), "nearest_surface_under_its_footprint"
        if support is None:
            return None, SUPPORT_NOT_FOUND
        snap = float(support["top_runtime_units"]) - (low if up_sign == 1 else high)[up]
        low[up], high[up] = low[up] + snap, high[up] + snap
        sim_min, sim_max = _box(low, high, runtime_to_sim)
        met = next((name for name, box in boxes if _overlap((sim_min, sim_max), box)), None)
        if met is not None:
            return None, f"{OVERLAPS}:{met}"
        dimensions = [sim_max[i] - sim_min[i] for i in range(3)]
        folder = Path(output_root) / DIRECTORY / f"rank-{rank:02d}"
        if row.get("source") in (LIBRARY, PARAMETRIC):
            authored, why = author(row, dimensions=dimensions, tolerance=tolerance,
                                   physics_bounds=screen_physics(dimensions)["bounds"], output_root=folder)
        else:  # its own reservation's ARTISAN run (stage 6)
            from .website_further_artisan import author as generate
            authored, why = generate(row, dimensions=dimensions, tolerance=tolerance,
                                     physics_bounds=screen_physics(dimensions)["bounds"], task_context=task_context,
                                     target=target, frames=frames, output_root=folder)
        if authored is None:
            return None, why
        failed = static_check(authored["path"], dimensions_m=dimensions, tolerance=tolerance)
        if failed is not None:
            return None, f"{STATIC_CHECK_FAILED}:{failed}"
        seated = seat(authored["receipt"]["bounds_m"], estimated=(sim_min, sim_max), runtime_to_sim=runtime_to_sim)
        met = next((name for name, box in boxes if _overlap(seated["bounds"], box)), None)
        if met is not None:  # the file as authored, which may be larger than the estimate, meets another box
            return None, f"{OVERLAPS}:{met}"
        boxes.append((f"further_object_rank_{rank}", seated["bounds"]))
        entry = by_target.get(target_id) or {"target_id": target_id,
                                              **({"semantic_label": target["semantic_label"]}
                                                 if isinstance(target.get("semantic_label"), str) else {})}
        return {"rank": rank, "target_id": target_id, "entry": dict(entry), "authored": authored,
                "runtime_min": seated["runtime"][0], "runtime_max": seated["runtime"][1],
                "sim_min": seated["bounds"][0], "sim_max": seated["bounds"][1], "position": seated["position"],
                "estimated": {"minimum": sim_min, "maximum": sim_max}, "snap": snap,
                "support": {"top_runtime_units": float(support["top_runtime_units"]), "basis": basis,
                            "snap_m": snap * mpu, "snap_basis": "estimate",
                            "seated_on": "authored_asset_bottom", "asset_bottom_m": seated["asset_bottom_m"]}}, None

    from .hard_stops import wall_kind
    from .job_retry_policy import PRIVACY_RIGHTS
    for row in sorted(rows, key=lambda item: item["rank"]):
        try:
            item, why = one(row)
        except Exception as exc:  # noqa: BLE001 - a further object never stops the scored object's job
            if isinstance(row.get("paid"), Mapping) and wall_kind(exc) == PRIVACY_RIGHTS:
                raise  # the site withdrew, or its rights changed: nothing more is sent, its hold kept
            item, why = None, f"{FAILED}:{type(exc).__name__}"
        if item is None:
            left.append(unbuilt(row, str(why)))
        else:
            placed.append(item)
    _close_paid(rows, task_context)
    requested = [{"rank": item["rank"], "entry": item["entry"], "aabb_min": item["sim_min"], "aabb_max": item["sim_max"],
                  "origin": origin, "complete": False} for item in placed]
    return {"objects": placed, "left_in_scan": left, "requested": requested}


def _close_paid(rows: Sequence[Mapping[str, Any]], task_context: Mapping[str, Any] | None) -> None:
    """Settle each paid further object's reservation once compile decided on it (``website_further_artisan.close``):
    an authored one already is; any other at what its authoring spent, $0 when none started. One that cannot be settled
    now keeps its whole hold (over, never under) and is settled by a later pass; never a stop."""
    if task_context is None:
        return
    from .website_further_artisan import close
    for row in rows:
        if isinstance(row.get("paid"), Mapping):
            try:
                close(row["paid"], task_context=task_context)
            except Exception:  # noqa: BLE001 - an unsettled row keeps its whole quote
                continue


def runtime_id(replacement: str) -> str:
    """A replacement id as the native runtime names an asset (``native_task_subject_identity``)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", replacement)


def compose(task: Mapping[str, Any], placed: Mapping[str, Any], *, task_context: Mapping[str, Any],
            base_scene: Mapping[str, Any], mpu: float, registration_path: Path,
            output_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The preparation's ``further_objects`` record and each composed object's compose-back record by its target, for
    the further objects the intake request ``task`` took (``usable_additional_objects``). Every other one is named with
    why: one taken out of the scan under ``removed_not_built`` (in neither the scan nor the simulation), any other
    under ``left_in_scan`` (as filmed); each flagged. Empty for a job with no further object, so its preparation is as
    before; ``removed_not_built`` is written only when it names one, so a record without one is as stage 3 wrote it."""
    if not placed.get("objects") and not placed.get("left_in_scan"):
        return {}, {}
    from .task_evaluation_scene_intent_contracts import usable_additional_objects
    from .website_build_plan import BUILT_EXTRA
    from .website_preparation_request import compose_back_record, replacement_id
    taken = {row["rank"] for row in usable_additional_objects(task)}
    objects: list[dict[str, Any]] = []
    left = list(placed.get("left_in_scan") or [])
    backs: dict[str, dict[str, Any]] = {}
    for item in placed.get("objects") or []:
        if item["rank"] not in taken:
            removed = (item.get("entry") or {}).get("task_effect") == BUILT_EXTRA
            left.append({**_left(item, NOT_TAKEN), **({_TAKEN_OUT: True} if removed else {})})
            continue
        slot = len(objects) + 2
        back = compose_back_record(task_context=task_context, subject_min=item["runtime_min"],
                                   subject_max=item["runtime_max"], base_scene=base_scene, meters_per_unit=mpu,
                                   snap=item["snap"], registration_path=registration_path, rank=item["rank"])
        backs[item["target_id"]] = back
        authored = item["authored"]
        receipt = authored["receipt"]
        objects.append({
            "rank": item["rank"], "target_id": item["target_id"], "prim_path": f"/World/Object_{slot:02d}",
            "runtime_name": f"{ROLE}_{slot:02d}", "replacement_asset_id": replacement_id(task_context, item["rank"]),
            "runtime_asset_id": runtime_id(replacement_id(task_context, item["rank"])), "source": authored["source"],
            **({"library_design": authored["library_design"]} if authored["source"] == LIBRARY else
               {"parametric_design": authored["parametric_design"]} if authored["source"] == PARAMETRIC else
               {"artisan_authoring": authored["artisan_authoring"]}),
            **({"library_not_used": authored["library_not_used"]} if "library_not_used" in authored else {}),
            "asset": {"relative_path": Path(authored["path"]).resolve().relative_to(Path(output_root).resolve()).as_posix(),
                      "sha256": "sha256:" + receipt["asset"]["sha256"], "size_bytes": receipt["asset"]["size_bytes"]},
            "asset_receipt": receipt, "static_check": "passed",
            # Its file's box where it stands, seated on its support (``seat``); the video's estimate beside it.
            "bounds_world_m": {"minimum": item["sim_min"], "maximum": item["sim_max"]},
            "estimated_bounds_world_m": item["estimated"],
            "pose_world": {"position_world_m": item["position"], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "support": item["support"], "compose_back": back})
    left.sort(key=lambda row: (row["rank"] if isinstance(row["rank"], int) else 99, row["target_id"]))
    gone = [{key: value for key, value in row.items() if key != _TAKEN_OUT} for row in left if row.get(_TAKEN_OUT)]
    left = [row for row in left if not row.get(_TAKEN_OUT)]
    record = {"schema_version": SCHEMA_VERSION, "revision": REVISION, "scored_object": SCORED_PRIM,
              "objects": objects, "left_in_scan": left, **({"removed_not_built": gone} if gone else {}),
              "flags": [*([LEFT_IN_SCAN] if left else []), *([REMOVED_NOT_BUILT] if gone else [])],
              "claim_boundary": CLAIM}
    record["digest"] = canonical_digest(record, digest_field="digest")
    return {"further_objects": record}, backs


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def for_revision(capture_root: Path, revision: Mapping[str, Any]) -> dict[str, Any] | None:
    """The further objects a website revision's preparation composed, read on the host from the preparation its source
    names by its bytes (``website_half_preparation``): each with its asset's path, when that file is inside the capture
    and has the bytes the record names; one whose file does not is left out, said why. None without any."""
    from .website_half_preparation import binding, half_preparation
    bound = binding(revision)
    if not bound:
        return None
    root = Path(capture_root)
    preparation, path = half_preparation(root, bound)
    record = (preparation or {}).get("further_objects")
    if (path is None or not isinstance(record, Mapping)
            or record.get("digest") != canonical_digest(record, digest_field="digest") or not record.get("objects")):
        return None
    objects, missing = [], []
    for row in record["objects"]:
        asset = (path.parent / str(row["asset"]["relative_path"])).resolve()
        if (not asset.is_relative_to(root.resolve()) or asset.is_symlink() or not asset.is_file()
                or _sha256(asset) != row["asset"]["sha256"]):
            missing.append(str(row["prim_path"]))
            continue
        objects.append({**row, "asset_path": str(asset)})
    return {"record_digest": record["digest"], "objects": objects,
            **({LEFT_OUT: {"reason": "further_object_asset_changed", "prims": missing}} if missing else {})}


def rows(further: Mapping[str, Any] | None, *, root: Path,
         evidence_asset_row: Callable[..., dict[str, Any]]) -> list[dict[str, Any]]:
    """A scene half's ``job_object`` rows, after its task row: each further object's asset copied under the half's
    ``root`` by its runtime name, at its filmed pose, as one rigid body with its own runtime name and asset ids."""
    found = []
    for row in (further or {}).get("objects") or []:
        copied = Path(root) / "further-objects" / f"{row['runtime_name']}.usda"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["asset_path"], copied)
        if _sha256(copied) != row["asset"]["sha256"]:
            raise ValueError("team_scene_further_object_copy_mismatch")
        pose = {"position_world_m": [float(v) for v in row["pose_world"]["position_world_m"]],
                "orientation_xyzw": [float(v) for v in row["pose_world"]["orientation_xyzw"]]}
        found.append({**evidence_asset_row(copied, root=Path(root), semantic_role=ROLE, pose_world=pose),
                      "name": row["runtime_name"], "asset_id": row["runtime_asset_id"],
                      "source_asset_id": row["replacement_asset_id"], "object_type": "RIGID",
                      "reset_state": {"root_pose_world": pose, "joint_positions": {}}})
    return found


def _codes(error: BaseException) -> list[str]:
    errors = getattr(error, "errors", None)
    if isinstance(errors, (list, tuple)) and errors:
        return [str(item) for item in errors]
    return [code for code in str(error).replace(",", ";").split(";") if code]


def refused_rows_dropped(write: Callable[[Mapping[str, Any]], Any], request: Mapping[str, Any], *,
                         output_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``write(request)`` (a packet writer) with the request's ``job_object`` rows; when it refuses and a code it
    gives names one of them, the request without them is written instead, the partial output removed first, and why
    is returned (``LEFT_OUT``): a further object never stops the scored object's half. Any other refusal is raised."""
    further = [row for row in request.get("assets") or [] if row.get("semantic_role") == ROLE]
    try:
        write(request)
        return dict(request), None
    except ValueError as exc:
        codes = _codes(exc)
        if not further or not any(ROLE in code for code in codes):
            raise
    shutil.rmtree(output_dir, ignore_errors=True)
    kept = {**request, "assets": [row for row in request["assets"] if row.get("semantic_role") != ROLE],
            "request_digest": ""}
    kept["request_digest"] = canonical_digest(kept, digest_field="request_digest")
    write(kept)
    return kept, {"reason": ",".join(sorted(code for code in codes if ROLE in code))[:400],
                  "runtime_names": [str(row.get("name")) for row in further]}


def half_record(further: Mapping[str, Any] | None, rows_written: Sequence[Mapping[str, Any]],
                left_out: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a scene half's sealed record says of its further objects: the runtime names it carries, the preparation
    record they come from, and any left out and why. Empty without further objects, so the record is as before."""
    if not further:
        return {}
    dropped = {**dict(further.get(LEFT_OUT) or {}), **dict(left_out or {})}
    return {"further_objects": {"record_digest": further.get("record_digest"),
                                "composed": [str(row.get("name")) for row in rows_written],
                                **({LEFT_OUT: dropped} if dropped else {})}}


__all__ = ["DIRECTORY", "LEFT_IN_SCAN", "LEFT_OUT", "NEEDS_ARTISAN", "NOT_BUILT", "NOT_TAKEN", "OVERLAPS",
           "REMOVED_NOT_BUILT", "ROLE", "SCHEMA_VERSION", "SCORED_PRIM", "STATIC_CHECK_FAILED", "author",
           "author_library_asset", "compose", "for_revision", "further_inputs", "half_record", "place",
           "refused_rows_dropped", "rows", "runtime_id", "scored_entries", "seat", "sealed_inputs", "static_check"]
