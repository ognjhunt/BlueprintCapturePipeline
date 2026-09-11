"""Separate publisher mesh components without changing or discarding source faces.

ADP-009D/day-28: InteriorGS instances can share a SAGE furniture mesh. Whole-prim
deletion would remove the table too. This CPU producer partitions that retained
mesh by connected components and re-runs the existing object-identity checks.
Its output is a derived collision candidate, never a new publisher source or a
claim that PhysX cooking/contact behavior has been qualified.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .sage_collision_identity import _box_metrics, inspect_sage_collision_identity
from .scene_placement.interiorgs_index import load_interiorgs_labels
from .validation_file_digests import scoped_measurement

SCHEMA = "interiorgs_sage_collision_partition.v1"
CONTAINMENT_TOLERANCE_M = 0.003
WELD_DECIMALS = 7


def _require(condition, code):
    if not condition:
        raise ValueError("sage_collision_partition_" + code)


def _record(path):
    path = Path(path)
    _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "file_invalid")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": "sha256:" + digest.hexdigest()}


def _stage(path):
    from pxr import Sdf, Usd, UsdGeom
    layer = Sdf.Layer.FindOrOpen(str(path))
    _require(layer is not None and not layer.subLayerPaths and not layer.GetExternalReferences(),
             "external_dependency_forbidden")
    stage = Usd.Stage.Open(layer)
    _require(UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0,
             "metric_frame_invalid")
    return stage


def _meshes(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    result = {}
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        _require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
                 and np.isfinite(points).all() and counts and min(counts) >= 3
                 and sum(counts) == len(indices) and min(indices) >= 0 and max(indices) < len(points),
                 "mesh_topology_invalid")
        _require(not any(a.GetNumTimeSamples() for a in (
            mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr())),
            "animated_mesh_forbidden")
        matrix = cache.GetLocalToWorldTransform(prim)
        world = np.asarray([matrix.Transform(Gf.Vec3d(*map(float, p))) for p in points])
        faces, offset = [], 0
        for count in counts:
            faces.append(indices[offset:offset + count])
            offset += count
        result[str(prim.GetPath())] = {"points": points, "world": world, "faces": faces,
            "face_vertex_indices": np.asarray(indices),
            "face_counts": np.asarray(counts), "face_offsets": np.r_[0, np.cumsum(counts)],
            "face_world": world[indices],
            "collision": prim.HasAPI(UsdPhysics.CollisionAPI), "prim": prim,
            "collision_enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if prim.HasAPI(UsdPhysics.CollisionAPI) else None}
    return result


class _FaceRows:
    """Read-only row access without copying millions of Python vertex integers."""

    def __init__(self, indices, offsets):
        self.indices, self.offsets = indices, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.indices[self.offsets[index]:self.offsets[index + 1]].tolist()


def _measured_meshes(stage, source_digest):
    # The caller has reopened and hashed the USD, including its no-external-
    # dependencies check. Keep only numeric geometry in this operation's cache;
    # live USD prims and their physics attributes are always reopened below.
    def measure():
        return {path: {key: value for key, value in row.items() if key not in {"prim", "faces"}}
                for path, row in _meshes(stage).items()}
    meshes = scoped_measurement(("sage_partition_mesh_geometry", source_digest), measure)
    for path, row in meshes.items():
        row["prim"] = stage.GetPrimAtPath(path)
        row["faces"] = _FaceRows(row["face_vertex_indices"], row["face_offsets"])
    return meshes


def _selected_face_world(mesh, indices):
    """Flatten selected faces without changing their order or coordinates."""
    starts = mesh["face_offsets"][indices]
    counts = mesh["face_counts"][indices]
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    flat = np.repeat(starts - offsets, counts) + np.arange(int(counts.sum()))
    return mesh["face_world"][flat]


def _components(mesh):
    points, faces = mesh["world"], mesh["faces"]
    parent = list(range(len(points)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    welded = {}
    for index, point in enumerate(points):
        key = tuple(np.round(point, WELD_DECIMALS))
        if key in welded:
            join(index, welded[key])
        else:
            welded[key] = index
    for face in faces:
        for vertex in face[1:]:
            join(face[0], vertex)
    groups = defaultdict(list)
    for index, face in enumerate(faces):
        groups[find(face[0])].append(index)
    result = []
    for face_ids in groups.values():
        vertices = sorted({v for i in face_ids for v in faces[i]})
        selected = points[vertices]
        result.append({"faces": face_ids, "minimum": selected.min(axis=0), "maximum": selected.max(axis=0)})
    return result


def _selection(meshes, objects):
    components = {path: _components(mesh) for path, mesh in meshes.items()
                  if mesh["collision"] and mesh["collision_enabled"] is True}
    selected = {}
    for obj in objects:
        candidates = []
        lo, hi = np.asarray(obj.bbox_min), np.asarray(obj.bbox_max)
        for path, parts in components.items():
            # A single connected component may match despite small publisher
            # box residuals. Multipart furniture is admitted only as a union of
            # completely contained components; no face is clipped at an OBB.
            whole = [part for part in parts if all(a >= b for a, b in zip(
                _box_metrics(lo, hi, part["minimum"], part["maximum"]), (.85, .9, .9), strict=True))]
            contained = [part for part in parts
                         if np.all(part["minimum"] >= lo - CONTAINMENT_TOLERANCE_M)
                         and np.all(part["maximum"] <= hi + CONTAINMENT_TOLERANCE_M)]
            options = [[part] for part in whole]
            if contained and not whole:
                options.append(contained)
            for option in options:
                minimum = np.min([p["minimum"] for p in option], axis=0)
                maximum = np.max([p["maximum"] for p in option], axis=0)
                metrics = _box_metrics(lo, hi, minimum, maximum)
                if all(a >= b for a, b in zip(metrics, (.85, .9, .9), strict=True)):
                    candidates.append({"source_prim": path,
                        "source_face_indices": sorted(i for part in option for i in part["faces"]),
                        "component_count": len(option), "identity_metrics": list(metrics)})
        _require(len(candidates) == 1, "component_match_not_unique:" + obj.id)
        selected[obj.id] = candidates[0]
    occupied = defaultdict(set)
    for item in selected.values():
        faces = set(item["source_face_indices"])
        _require(not occupied[item["source_prim"]].intersection(faces), "selected_objects_share_faces")
        occupied[item["source_prim"]].update(faces)
    return selected


def _set_faces(prim, source, face_ids):
    from pxr import UsdGeom, Vt
    mesh = UsdGeom.Mesh(prim)
    vertices = sorted({v for i in face_ids for v in source["faces"][i]})
    remap = {old: new for new, old in enumerate(vertices)}
    points = source["points"][vertices]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set([len(source["faces"][i]) for i in face_ids])
    mesh.GetFaceVertexIndicesAttr().Set([remap[v] for i in face_ids for v in source["faces"][i]])
    mesh.CreateExtentAttr([tuple(map(float, points.min(axis=0))), tuple(map(float, points.max(axis=0)))])
    # The derivative is collision geometry. Vertex/face-varying render arrays
    # and material subsets cannot be copied with their old element indices.
    for attr in list(prim.GetAttributes()):
        if attr.GetName() == "normals" or attr.GetName().startswith("primvars:"):
            prim.RemoveProperty(attr.GetName())
    for child in list(prim.GetChildren()):
        _require(child.IsA(UsdGeom.Subset), "mesh_child_not_supported")
        prim.GetStage().RemovePrim(child.GetPath())


def validate_partition(receipt, *, source_path, labels_path, output_path):
    """Reopen geometry and verify the face partition, not just its self-report."""
    from pxr import UsdGeom
    _require(receipt.get("schema_version") == SCHEMA and receipt.get("status") == "geometry_partitioned_pending_native_validation"
             and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest"),
             "receipt_invalid")
    reopened = {}
    for name, path in (("source", source_path), ("labels", labels_path), ("output", output_path)):
        actual = _record(path)
        _require(all(receipt.get(name, {}).get(k) == actual[k] for k in ("sha256", "size_bytes")), "bytes_changed")
        reopened[name] = actual["sha256"]
    source_stage, output_stage = _stage(source_path), _stage(output_path)
    original = _measured_meshes(source_stage, reopened["source"])
    derived = _measured_meshes(output_stage, reopened["output"])
    objects = load_interiorgs_labels(labels_path)
    requested = receipt.get("selected_instance_ids")
    _require(isinstance(requested, list) and len(requested) == len(set(requested)) == 2,
             "selection_invalid")
    selected = scoped_measurement(("sage_partition_component_selection", reopened["source"],
        reopened["labels"], tuple(requested), WELD_DECIMALS, CONTAINMENT_TOLERANCE_M),
        lambda: _selection(original, [next(o for o in objects if o.id == oid) for oid in requested]))
    _require(selected == receipt.get("component_selection"), "selection_changed")
    coverage = defaultdict(list)
    paths = set()
    for row in receipt["face_partitions"]:
        source, dest = row["source_prim"], row["output_prim"]
        indices = row["source_face_indices"]
        _require(source in original and dest in derived and dest not in paths
                 and isinstance(indices, list) and indices == sorted(set(indices))
                 and indices and min(indices) >= 0 and max(indices) < len(original[source]["faces"]),
                 "face_map_invalid")
        paths.add(dest)
        old, new = original[source], derived[dest]
        _require(old["collision"] == new["collision"] and old["collision_enabled"] == new["collision_enabled"]
                 and len(indices) == len(new["faces"]), "collision_state_changed")
        _require(np.array_equal(old["face_counts"][indices], new["face_counts"])
                 and np.array_equal(_selected_face_world(old, indices), new["face_world"]),
                 "source_face_geometry_changed")
        # Keep authored collision approximation. Splitting changes cooking
        # inputs, so the receipt explicitly still requires native qualification.
        for attr in old["prim"].GetAttributes():
            if attr.GetName().startswith(("physics:", "physx")):
                _require(attr.Get() == new["prim"].GetAttribute(attr.GetName()).Get(), "collision_configuration_changed")
        coverage[source].extend(indices)
    _require(paths == set(derived), "unaccounted_output_mesh")
    _require(set(coverage) == set(original) and all(sorted(coverage[path]) == list(range(len(mesh["faces"])))
             for path, mesh in original.items()), "source_faces_missing_or_duplicated")
    for oid, selection in selected.items():
        matching = [r for r in receipt["face_partitions"] if r.get("instance_id") == oid]
        _require(len(matching) == 1 and matching[0]["source_prim"] == selection["source_prim"]
                 and matching[0]["source_face_indices"] == selection["source_face_indices"], "object_face_map_changed")
        identity = inspect_sage_collision_identity(labels_path=labels_path, target_instance_id=oid,
                                                   sage_collision_usd_path=output_path)
        _require(identity["whole_object_collision_identity_passed"]
                 and identity["whole_object_matches"][0]["prim_path"] == matching[0]["output_prim"],
                 "derived_object_identity_invalid")
    _require(not any(p.IsA(UsdGeom.Mesh) and not p.IsActive() for p in output_stage.TraverseAll()),
             "hidden_mesh_forbidden")
    return receipt


def materialize_sage_collision_partition(*, source_path, labels_path, instance_ids, output_root):
    from pxr import Sdf, Usd
    source_path, labels_path, output_root = map(Path, (source_path, labels_path, output_root))
    source_record, labels_record = _record(source_path), _record(labels_path)
    _require(len(instance_ids) == len(set(instance_ids)) == 2, "selection_invalid")
    objects = load_interiorgs_labels(labels_path)
    _require(all(sum(o.id == oid for o in objects) == 1 for oid in instance_ids), "instance_missing")
    source_stage = _stage(source_path)
    original = _meshes(source_stage)
    selected = _selection(original, [next(o for o in objects if o.id == oid) for oid in instance_ids])
    output = output_root / "partitioned_collision.usd"
    receipt_path = output_root / "collision_partition.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        _require(receipt.get("selected_instance_ids") == list(instance_ids), "output_conflict")
        return validate_partition(receipt, source_path=source_path, labels_path=labels_path, output_path=output)
    _require(not output_root.exists() and not any(p.is_symlink() for p in (output_root, *output_root.parents)),
             "output_conflict")
    output_root.mkdir(parents=True, mode=0o750)
    source_stage.GetRootLayer().Export(str(output))
    stage = Usd.Stage.Open(str(output))
    partitions, occupied = [], defaultdict(set)
    for oid, item in selected.items():
        path = Sdf.Path(item["source_prim"])
        name = "bcp_instance_" + hashlib.sha256(oid.encode()).hexdigest()[:16]
        destination = path.GetParentPath().AppendChild(name)
        _require(not stage.GetPrimAtPath(destination), "output_prim_conflict")
        Sdf.CopySpec(source_stage.GetRootLayer(), path, stage.GetRootLayer(), destination)
        _set_faces(stage.GetPrimAtPath(destination), original[str(path)], item["source_face_indices"])
        occupied[str(path)].update(item["source_face_indices"])
        partitions.append({"instance_id": oid, "source_prim": str(path), "output_prim": str(destination),
                           "source_face_indices": item["source_face_indices"]})
    for path, mesh in original.items():
        remaining = sorted(set(range(len(mesh["faces"]))) - occupied[path])
        if not remaining:
            stage.RemovePrim(path)
        else:
            if occupied[path]:
                _set_faces(stage.GetPrimAtPath(path), mesh, remaining)
            partitions.append({"source_prim": path, "output_prim": path, "source_face_indices": remaining})
    stage.GetRootLayer().Save()
    receipt = {"schema_version": SCHEMA, "status": "geometry_partitioned_pending_native_validation",
        "source": source_record, "labels": labels_record, "output": _record(output),
        "selected_instance_ids": list(instance_ids), "component_selection": selected,
        "face_partitions": partitions, "containment_tolerance_m": CONTAINMENT_TOLERANCE_M,
        "weld_decimals": WELD_DECIMALS, "source_faces_deleted": 0,
        "source_bytes_changed": False, "native_collision_cooking_qualified": False,
        "claim_ceiling": "development_only_source_geometry_partition"}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    validate_partition(receipt, source_path=source_path, labels_path=labels_path, output_path=output)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
