"""Normalize provided mesh coordinates and retain an exact object mapping.

This is a deterministic format/frame adapter, not reconstruction. Original
bytes and their declarations remain the authority; the resulting USD is a
development-only runtime candidate with separately authored static colliders.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from .decision_evidence_contracts import canonical_digest
from .provided_scene_mesh import inspect_mesh
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require, sha


@contextmanager
def mesh_filename(path: Path, filename: str):
    suffix = Path(filename).suffix.lower()
    if path.suffix.lower() == suffix:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="completed-mesh-") as temporary:
        alias = Path(temporary).resolve() / ("source" + suffix)
        try:
            os.link(path, alias)
        except OSError:
            shutil.copyfile(path, alias)
        yield alias


def _usd_copy(source: Path, output: Path, *, scale: float, axis: str) -> dict[str, str]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    original = Usd.Stage.Open(str(source), Usd.Stage.LoadNone)
    require(original is not None, "completed_mesh_open_failed")
    source_layer = original.Flatten()
    stage = Usd.Stage.CreateNew(str(output))
    parent = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(parent.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    transform = Gf.Matrix4d().SetScale(scale)
    if axis == "Y":
        transform = transform * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
    parent.AddTransformOp().Set(transform)
    for prim in original.GetPseudoRoot().GetChildren():
        require(Sdf.CopySpec(source_layer, prim.GetPath(), stage.GetRootLayer(),
                             Sdf.Path("/Root" + str(prim.GetPath()))), "completed_mesh_copy_failed")
    mapping = {}
    for prim in stage.Traverse():
        require(not prim.IsA(UsdPhysics.Joint) and not prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                "completed_mesh_articulation_not_supported")
        # Preserve visual relationships when the source roots acquire the
        # normalization parent. External dependencies were refused by inspection.
        source_prim = original.GetPrimAtPath(str(prim.GetPath()).removeprefix("/Root")) if str(prim.GetPath()) != "/Root" else None
        for relation in prim.GetRelationships():
            source_relation = source_prim.GetRelationship(relation.GetName()) if source_prim else None
            targets = source_relation.GetTargets() if source_relation else []
            if targets:
                relation.SetTargets([Sdf.Path("/Root" + str(p)) if p.IsAbsolutePath() else p for p in targets])
        for attribute in prim.GetAttributes():
            source_attribute = source_prim.GetAttribute(attribute.GetName()) if source_prim else None
            targets = source_attribute.GetConnections() if source_attribute else []
            if targets:
                attribute.SetConnections([Sdf.Path("/Root" + str(p)) if p.IsAbsolutePath() else p for p in targets])
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI,
                    UsdPhysics.MeshCollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        for name in prim.GetPropertyNames():
            if str(name).startswith(("physics:", "physx")):
                prim.RemoveProperty(name)
        if prim.IsA(UsdGeom.Mesh):
            path = str(prim.GetPath())
            mapping[path.removeprefix("/Root")] = path
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    stage.GetRootLayer().Save()
    return mapping


def _triangle_copy(source: Path, output: Path, *, suffix: str, scale: float, axis: str) -> dict[str, str]:
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

    original = trimesh.load(source, file_type=suffix[1:], force="scene", process=False)
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mapping = {}
    for index, node in enumerate(sorted(original.graph.nodes_geometry)):
        transform, name = original.graph[node]
        source_mesh = original.geometry[name]
        require(isinstance(source_mesh, trimesh.Trimesh), "completed_mesh_geometry_invalid")
        # Do not silently discard embedded textures and call the result the
        # supplied appearance. Texture-bearing glTF needs an admitted converter.
        require(source_mesh.visual.kind != "texture", "completed_textured_gltf_not_supported")
        points = trimesh.transform_points(source_mesh.vertices, transform) * scale
        if axis == "Y":
            points = points[:, [0, 2, 1]] * [1, -1, 1]
        path = f"/Root/mesh_{index:04d}_" + re.sub(r"[^A-Za-z0-9_]", "_", str(node))[:80]
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(source_mesh.faces))
        mesh.CreateFaceVertexIndicesAttr(source_mesh.faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if source_mesh.visual.kind in {"vertex", "face"}:
            colors = np.asarray(source_mesh.visual.vertex_colors, dtype=float) / 255.0
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(colors[:, :3].tolist())
            mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex).Set(colors[:, 3].tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        mapping[str(node)] = path
    stage.GetRootLayer().Save()
    return mapping


def normalize_completed_mesh(*, source: Path, original_filename: str, coordinate_frame: dict,
                             output_root: Path) -> dict:
    """Read, convert, and independently verify every source object's bounds."""
    inspection = inspect_mesh(source, original_filename=original_filename,
                              coordinate_frame_declaration=coordinate_frame)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    receipt_path = output_root / "mesh_normalization.v1.json"
    if receipt_path.exists():
        prior = read(receipt_path, digest_field="normalization_digest")
        require(prior.get("source_digest") == inspection["asset_digest"]
                and prior.get("source_inspection_digest") == inspection["inspection_digest"],
                "completed_mesh_normalization_conflict")
        checked_file(output_root / prior["output"]["relative_path"], prior["output"])
        return prior
    output = output_root / "normalized_scene.usda"
    require(not output.exists(), "completed_mesh_partial_normalization_requires_replay")
    scale, axis = float(coordinate_frame["meters_per_unit"]), coordinate_frame["up_axis"]
    with mesh_filename(source, original_filename) as named:
        suffix = Path(original_filename).suffix.lower()
        if suffix in {".usd", ".usda", ".usdc"}:
            mapping = _usd_copy(named, output, scale=scale, axis=axis)
        else:
            mapping = _triangle_copy(named, output, suffix=suffix, scale=scale, axis=axis)
    observed = inspect_mesh(output, original_filename=output.name,
                            coordinate_frame_declaration={"meters_per_unit": 1.0, "up_axis": "Z"})
    actual = {row["source_object_id"]: row for row in observed["objects"]}
    require(len(inspection["objects"]) == len(actual) == len(mapping), "completed_mesh_object_count_changed")
    for row in inspection["objects"]:
        found = actual[mapping[row["source_object_id"]]]
        require(all(found[key] == row[key] for key in ("point_count", "face_count"))
                and all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                        for key in ("world_aabb_min_m", "world_aabb_max_m")
                        for a, b in zip(row[key], found[key], strict=True)),
                "completed_mesh_geometry_changed")
    require(sha(source) == inspection["asset_digest"], "completed_mesh_source_changed")
    value = {"schema_version": "task_evaluation_completed_mesh_normalization.v1",
        "source_digest": inspection["asset_digest"], "source_inspection_digest": inspection["inspection_digest"],
        "source_original_filename": original_filename, "declared_coordinate_frame": coordinate_frame,
        "output": {"relative_path": output.name, "sha256": sha(output), "size_bytes": output.stat().st_size},
        "object_mapping": mapping, "normalized_inspection": observed,
        "runtime_frame": {"meters_per_unit": 1.0, "up_axis": "Z"},
        "source_bytes_unchanged": True, "reconstruction_performed": False,
        "physical_scale_measured": False, "physics_qualified": False,
        "claim_scope": "development_only"}
    value["normalization_digest"] = canonical_digest(value, digest_field="normalization_digest")
    receipt_path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return value
