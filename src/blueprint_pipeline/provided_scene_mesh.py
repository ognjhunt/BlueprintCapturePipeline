"""Inspect supplied geometry without pretending it is an observed capture.

No geometry is generated here. Units are owner declarations cross-checked with
authored USD metadata; native collision/task physics remain separate gates.
External USD/glTF dependencies are refused before scene composition/loading.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .reconstruction_capability import (
    ReconstructionContractError, build_reconstruction_method_profile, normalize_reconstruction_result,
)

ADAPTER = "local://provided-scene-mesh-import-v1"
MAX_INSPECTION_BYTES = 256 * 1024 * 1024
MAX_VERTICES = 10_000_000


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ReconstructionContractError(["provided_mesh:" + code])


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def method_profile(*, execution_authorized: bool = False) -> dict[str, Any]:
    return build_reconstruction_method_profile({
        "method_id": "local_provided_scene_mesh_import", "version": "1",
        "implementation_digest": _sha(Path(__file__)), "method_kind": "provided_scene_mesh_import",
        "provider_identity": "local", "execution_mode": "hermetic_local", "adapter_reference": ADAPTER,
        "outputs": ["appearance_layer"], "required_capture_authority_profiles": ["provided_scene_mesh"],
        "required_claim_ceiling_flags": [], "qualified_claim_types": ["appearance_review"],
        "execution_authorized": execution_authorized, "qualification_status": "debug_only",
        "expected_cost_usd": 0, "provider_constraints": {"external_processing": False},
        "rights_constraints": {"requires_local_processing_allowed": True},
        "failure_modes": ["asset_digest_mismatch", "units_missing", "external_dependency_forbidden",
                          "geometry_invalid", "inspection_size_limit"],
    })


def _row(identity: str, points: Any, faces: Any, *, scale: float, axis: str) -> dict[str, Any]:
    import numpy as np

    vertices = np.asarray(points, dtype=float)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3 and 3 <= len(vertices) <= MAX_VERTICES
             and np.isfinite(vertices).all(), "geometry_invalid")
    _require(len(faces) > 0, "mesh_faces_missing")
    vertices = vertices * scale
    if axis == "Y":
        vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    _require(np.isfinite(lower).all() and np.isfinite(upper).all(), "geometry_invalid")
    return {"source_object_id": identity, "point_count": len(vertices), "face_count": len(faces),
            "world_aabb_min_m": lower.tolist(), "world_aabb_max_m": upper.tolist()}


def inspect_mesh(path: Path, *, original_filename: str,
                 coordinate_frame_declaration: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "asset_missing")
    _require(0 < path.stat().st_size <= MAX_INSPECTION_BYTES, "inspection_size_limit")
    scale = coordinate_frame_declaration.get("meters_per_unit")
    axis = coordinate_frame_declaration.get("up_axis")
    _require(not isinstance(scale, bool) and isinstance(scale, (int, float))
             and math.isfinite(scale) and 0 < scale <= 1000 and axis in {"Y", "Z"}, "units_missing")
    suffix = Path(original_filename).suffix.lower()
    if suffix in {".usd", ".usda", ".usdc"} and path.suffix.lower() != suffix:
        # Content-addressed blobs have no extension. USD selects its reader by
        # filename; a bounded temporary hard link preserves the exact bytes.
        with tempfile.TemporaryDirectory(prefix="provided-mesh-inspection-") as scratch:
            alias = Path(scratch).resolve() / ("source" + suffix)
            try:
                os.link(path, alias)
            except OSError:
                shutil.copyfile(path, alias)
            return inspect_mesh(alias, original_filename=original_filename,
                                coordinate_frame_declaration=coordinate_frame_declaration)
    objects = []
    if suffix in {".usd", ".usda", ".usdc"}:
        from pxr import Sdf, Usd, UsdGeom
        layer = Sdf.Layer.FindOrOpen(str(path))
        _require(layer is not None, "usd_unreadable")
        _require(not layer.subLayerPaths and not layer.GetExternalReferences()
                 and not layer.GetExternalAssetDependencies(), "external_dependency_forbidden")
        stage = Usd.Stage.Open(layer, Usd.Stage.LoadNone)
        _require(stage is not None, "usd_unreadable")
        if stage.HasAuthoredMetadata("metersPerUnit"):
            _require(math.isclose(UsdGeom.GetStageMetersPerUnit(stage), scale, abs_tol=1e-12), "units_conflict")
        if stage.HasAuthoredMetadata("upAxis"):
            _require(str(UsdGeom.GetStageUpAxis(stage)) == axis, "up_axis_conflict")
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = mesh.GetFaceVertexIndicesAttr().Get() or []
            _require(sum(counts) == len(indices) and all(n >= 3 for n in counts)
                     and all(0 <= i < len(points) for i in indices), "mesh_topology_invalid")
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world = [list(transform.Transform(point)) for point in points]
            objects.append(_row(str(prim.GetPath()), world, counts, scale=scale, axis=axis))
    elif suffix in {".glb", ".ply"}:
        if suffix == ".glb":
            with path.open("rb") as stream:
                header = stream.read(20)
                _require(len(header) == 20, "glb_invalid")
                magic, version, total, length, kind = struct.unpack("<4sIIII", header)
                _require(magic == b"glTF" and version == 2 and total == path.stat().st_size
                         and kind == 0x4E4F534A and length <= 16 * 1024 * 1024, "glb_invalid")
                document = json.loads(stream.read(length))
            def external(value: Any) -> bool:
                if isinstance(value, dict):
                    return "uri" in value or any(external(v) for v in value.values())
                return isinstance(value, list) and any(external(v) for v in value)
            _require(not external(document), "external_dependency_forbidden")
        import trimesh
        scene = trimesh.load(path, file_type=suffix[1:], force="scene", process=False, skip_materials=True)
        for node in sorted(scene.graph.nodes_geometry):
            transform, name = scene.graph[node]
            mesh = scene.geometry[name]
            _require(isinstance(mesh, trimesh.Trimesh), "mesh_faces_missing")
            objects.append(_row(str(node), trimesh.transform_points(mesh.vertices, transform),
                                mesh.faces, scale=scale, axis=axis))
    else:
        raise ReconstructionContractError(["provided_mesh:format_not_supported"])
    _require(bool(objects), "mesh_objects_missing")
    report = {"schema_version": "provided_scene_mesh_inspection.v1", "asset_digest": _sha(path),
        "objects": objects, "source_kind": "provided_scene_mesh",
        "coordinate_system": {"declared_meters_per_unit": scale, "declared_up_axis": axis,
                              "output_up_axis": "Z", "physical_scale_measured": False},
        "raw_capture_authority": False, "collision_qualified": False, "physics_qualified": False}
    report["inspection_digest"] = canonical_digest(report, digest_field="inspection_digest")
    return report


class ProvidedSceneMeshImportAdapter:
    adapter_reference = ADAPTER

    def execute(self, *, intake_id: str, capture_digest: str, capture_root: Path,
                asset_relative_path: str, original_filename: str, output_root: Path,
                rights_and_retention: Mapping[str, Any], coordinate_frame_declaration: Mapping[str, Any],
                source_capture_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
        relative = Path(asset_relative_path)
        _require(not relative.is_absolute() and ".." not in relative.parts, "asset_path_invalid")
        path = capture_root / relative
        _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "asset_missing")
        _require(_sha(path) == capture_digest, "asset_digest_mismatch")
        try:
            report = inspect_mesh(path, original_filename=original_filename,
                                  coordinate_frame_declaration=coordinate_frame_declaration)
        except ReconstructionContractError:
            raise
        except Exception as exc:
            raise ReconstructionContractError(["provided_mesh:geometry_unreadable"]) from exc
        profile = method_profile(execution_authorized=True)
        root = output_root / capture_digest.removeprefix("sha256:") / "provided_scene_mesh_v1"
        _require(not any(p.is_symlink() for p in (root, *root.parents)), "output_unsafe")
        root.mkdir(parents=True, exist_ok=True)
        target = root / "inspection.json"
        if target.exists():
            _require(not target.is_symlink() and json.loads(target.read_text()) == report, "output_conflict")
        else:
            from .task_evaluation_launch_preparation_queue import _write_launch_preparation_record_exclusive_locked
            _write_launch_preparation_record_exclusive_locked(target, report)
        return normalize_reconstruction_result({
            "result_id": "provided-mesh-" + report["inspection_digest"][7:23], "intake_id": intake_id,
            "capture_digest": capture_digest, "method_id": profile["method_id"], "method_version": "1",
            "method_profile_digest": profile["method_profile_digest"],
            "implementation_digest": profile["implementation_digest"], "provider_identity": "local",
            "runtime_identity": "provided-scene-mesh-inspection-v1", "runtime_digest": profile["implementation_digest"],
            "outputs": ["appearance_layer"], "source_frames": {"status": "not_supplied"},
            "camera_solution": {"status": "not_supplied", "calibrated": False},
            "coordinate_system": report["coordinate_system"],
            "asset_references": {"provided_scene_mesh": {"uri": "content-addressed-capture://sha256/" + capture_digest[7:],
                                                        "digest": capture_digest},
                                 "mesh_inspection": {"uri": "local-reconstruction://" + _sha(target)[7:],
                                                     "digest": _sha(target)}},
            "coverage_map": {"status": "not_observed"}, "observed_regions": [], "generated_regions": [],
            "invalid_regions": [], "uncertainty_map": {"source_observation_coverage": "unknown"},
            "validation_metrics": report, "cost_usd": 0, "duration_seconds": 0,
            "rights_and_retention": dict(rights_and_retention), "provider_receipt": None, "deletion_evidence": None,
            "claim_ceiling": {"provided_mesh_imported": True, "raw_capture_authority": False,
                              "captured_observation": False, "metric_geometry": False,
                              "collision_geometry": False, "physics": False, "physical_task_success": False,
                              "comparative_policy_ranking_verdict": "thesis_not_supported"},
        })
