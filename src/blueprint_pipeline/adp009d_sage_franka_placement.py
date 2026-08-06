"""Derive an ADP-009D Franka placement candidate from sealed SAGE USD bytes.

The adapter performs a deterministic OpenUSD-to-GLB geometry projection, then
reuses :mod:`external_scene_robot_placement` for the actual collision/support
search.  The derived GLB is a placement-analysis intermediate; it never
replaces the exact SAGE USD in simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .external_scene_robot_placement import (
    build_external_scene_robot_placement_request,
    propose_external_scene_robot_placement,
)


CONVERSION_SCHEMA_VERSION = "adp009d_sage_collision_analysis_glb.v1"
PLACEMENT_PACKET_SCHEMA_VERSION = "adp009d_sage_franka_placement_packet.v1"
SEALED_SCENE_RECEIPT_DIGEST = (
    "sha256:b259532be614098a3830aa9945770a96371968f9c68e8087eb21a2ca00e3c3e3"
)
SEALED_AURA_DIGEST = (
    "sha256:cbb05fc8e6da6ecdb72464f3b115f63e8747e2b67e97c309b4e40952b33000bd"
)
TARGET_BINDING_DIGEST = (
    "sha256:a25e43f695e6eea3dff0c18ad1ee5001bcb7162600bade996d7e08b7c01d5d8b"
)


class SageFrankaPlacementError(ValueError):
    """Stable SAGE placement materialization errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _triangulate(counts: Sequence[int], indices: Sequence[int]) -> np.ndarray:
    face_counts = np.asarray(counts, dtype=np.int64)
    face_indices = np.asarray(indices, dtype=np.int64)
    if face_counts.size and bool(np.all(face_counts == 3)):
        if face_indices.size != 3 * face_counts.size:
            raise SageFrankaPlacementError(["sage_collision_topology_invalid"])
        return face_indices.reshape((-1, 3))
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for raw_count in counts:
        count = int(raw_count)
        face = [int(item) for item in indices[offset : offset + count]]
        offset += count
        if count < 3:
            continue
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, count - 1))
    if offset != len(indices) or not triangles:
        raise SageFrankaPlacementError(["sage_collision_topology_invalid"])
    return np.asarray(triangles, dtype=np.int64)


def materialize_sage_collision_analysis_glb(
    *,
    sage_usd_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Project all active SAGE collision meshes into one digest-bound GLB."""

    try:
        import trimesh
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise SageFrankaPlacementError(["sage_collision_openusd_runtime_missing"]) from exc

    source = Path(sage_usd_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise SageFrankaPlacementError(["sage_collision_usd_missing"])
    if destination.exists() and any(destination.iterdir()):
        raise SageFrankaPlacementError(["sage_collision_analysis_output_not_empty"])
    destination.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if stage is None:
        raise SageFrankaPlacementError(["sage_collision_usd_open_failed"])
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise SageFrankaPlacementError(["sage_collision_stage_not_z_up"])
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-12:
        raise SageFrankaPlacementError(["sage_collision_stage_not_meter_units"])

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    mesh_records: list[dict[str, Any]] = []
    vertex_offset = 0
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
        indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default()) or []
        if not points or not counts or not indices:
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        local_points = np.asarray(points, dtype=np.float64)
        homogeneous = np.column_stack(
            (local_points, np.ones(len(local_points), dtype=np.float64))
        )
        transformed = homogeneous @ matrix
        stage_points = transformed[:, :3] / transformed[:, 3, None]
        local_faces = _triangulate(counts, indices)
        # external_scene_robot_placement applies GLB Y-up -> stage Z-up as
        # [x, -z, y].  This exact inverse preserves the SAGE stage coordinates.
        glb_points = np.column_stack(
            (stage_points[:, 0], stage_points[:, 2], -stage_points[:, 1])
        )
        vertices.append(glb_points)
        faces.append(local_faces + vertex_offset)
        vertex_offset += len(glb_points)
        mesh_records.append(
            {
                "prim_path": str(prim.GetPath()),
                "point_count": len(points),
                "triangle_count": len(local_faces),
                "collision_api_applied": prim.HasAPI(UsdPhysics.CollisionAPI),
            }
        )
    if not vertices:
        raise SageFrankaPlacementError(["sage_collision_meshes_missing"])
    combined_vertices = np.concatenate(vertices, axis=0)
    combined_faces = np.concatenate(faces, axis=0)
    mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False,
        validate=False,
    )
    exported = mesh.export(file_type="glb")
    if not isinstance(exported, bytes) or not exported:
        raise SageFrankaPlacementError(["sage_collision_glb_export_failed"])
    glb_path = destination / "sage_840313_collision_analysis.glb"
    glb_path.write_bytes(exported)
    receipt: dict[str, Any] = {
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "status": "materialized_analysis_intermediate",
        "source_sage_usd": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "meters_per_unit": meters_per_unit,
            "up_axis": "Z",
        },
        "analysis_glb": {
            "path": str(glb_path),
            "size_bytes": glb_path.stat().st_size,
            "sha256": _sha256(glb_path),
            "coordinate_conversion": "sage_z_up_to_glb_y_up_inverse_of_existing_placement_adapter",
        },
        "mesh_count": len(mesh_records),
        "vertex_count": int(len(combined_vertices)),
        "triangle_count": int(len(combined_faces)),
        "meshes": mesh_records,
        "simulation_asset_replacement": False,
        "source_usd_mutated": False,
        "claim_ceiling": "deterministic_collision_geometry_analysis_intermediate",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(destination / "adp009d_sage_collision_analysis_glb.v1.json", receipt)
    return receipt


def materialize_sage_franka_placement_packet(
    *,
    conversion_receipt: Mapping[str, Any],
    output_dir: str | Path,
    target_position_m: Sequence[float] = (3.4681748, -3.3100837, 0.6096791),
) -> dict[str, Any]:
    """Run the existing placement engine against one admitted SAGE conversion."""

    conversion = json.loads(json.dumps(dict(conversion_receipt)))
    if (
        conversion.get("schema_version") != CONVERSION_SCHEMA_VERSION
        or conversion.get("status") != "materialized_analysis_intermediate"
        or conversion.get("receipt_digest")
        != canonical_digest(conversion, digest_field="receipt_digest")
    ):
        raise SageFrankaPlacementError(["sage_collision_conversion_receipt_invalid"])
    glb_record = conversion.get("analysis_glb") or {}
    glb_path = Path(str(glb_record.get("path") or "")).expanduser().resolve()
    if (
        not glb_path.is_file()
        or glb_path.stat().st_size != glb_record.get("size_bytes")
        or _sha256(glb_path) != glb_record.get("sha256")
    ):
        raise SageFrankaPlacementError(["sage_collision_analysis_glb_changed"])
    if len(target_position_m) != 3 or not all(
        np.isfinite(float(item)) for item in target_position_m
    ):
        raise SageFrankaPlacementError(["sage_franka_target_position_invalid"])
    target_analysis: dict[str, Any] = {
        "schema_version": "adp009d_registered_pick_place_target_analysis.v1",
        "status": "derived_from_sealed_target_obb",
        "selected_target": {
            "target_id": "840313_ins160_approved_can",
            "task_family": "rigid_opaque_pick_place",
            "position_m": [float(item) for item in target_position_m],
        },
        "sealed_scene_receipt_digest": SEALED_SCENE_RECEIPT_DIGEST,
        "target_binding_digest": TARGET_BINDING_DIGEST,
        "outcomes_observed_before_selection": False,
    }
    target_analysis["target_analysis_digest"] = canonical_digest(
        target_analysis, digest_field="target_analysis_digest"
    )
    request = build_external_scene_robot_placement_request(
        {
            "schema_version": "external_scene_robot_placement_request.v1",
            "robot_id": "franka_panda",
            "source_scene_digest": SEALED_AURA_DIGEST,
            "target_analysis_digest": target_analysis["target_analysis_digest"],
            "target_binding_digest": TARGET_BINDING_DIGEST,
            "scene_frame_binding_digest": SEALED_SCENE_RECEIPT_DIGEST,
            "collision_candidate_digest": conversion["receipt_digest"],
            "collision_source_digest": glb_record["sha256"],
            "target_position_collision_stage": [
                float(item) for item in target_position_m
            ],
            "target_spatial_uncertainty_stage_units": 0.0311,
            "target_label": "approved canned beverage",
            "visual_confidence": 1.0,
            "metric_scale_status": "validated",
            "collision_status": "candidate_compiled",
            "candidate_may_self_authorize": False,
        }
    )
    packet = propose_external_scene_robot_placement(
        collision_glb_path=glb_path,
        request=request,
        target_analysis=target_analysis,
    )
    result: dict[str, Any] = {
        "schema_version": PLACEMENT_PACKET_SCHEMA_VERSION,
        "status": (
            "placement_candidate_materialized"
            if packet["placement"].get("status") == "runtime_visualization_candidate_only"
            else "blocked"
        ),
        "conversion_receipt_digest": conversion["receipt_digest"],
        "request": request,
        "target_analysis": target_analysis,
        "placement": packet["placement"],
        "render_options": packet["render_options"],
        "native_contact_reachability_qualified": False,
        "policy_execution_authorized": False,
        "blockers": [
            "franka_robotiq_native_reset_contact_reachability_missing",
            "robot_base_and_camera_calibration_native_probe_missing",
        ],
    }
    result["packet_digest"] = canonical_digest(result, digest_field="packet_digest")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SageFrankaPlacementError(["sage_franka_placement_output_not_empty"])
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "adp009d_sage_franka_placement_packet.v1.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-usd", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    conversion = materialize_sage_collision_analysis_glb(
        sage_usd_path=args.sage_usd,
        output_dir=args.output_dir / "conversion",
    )
    packet = materialize_sage_franka_placement_packet(
        conversion_receipt=conversion,
        output_dir=args.output_dir / "placement",
    )
    print(json.dumps(packet, sort_keys=True))
    return 0 if packet["status"] == "placement_candidate_materialized" else 2


__all__ = [
    "CONVERSION_SCHEMA_VERSION",
    "PLACEMENT_PACKET_SCHEMA_VERSION",
    "SageFrankaPlacementError",
    "materialize_sage_collision_analysis_glb",
    "materialize_sage_franka_placement_packet",
]


if __name__ == "__main__":
    raise SystemExit(main())
