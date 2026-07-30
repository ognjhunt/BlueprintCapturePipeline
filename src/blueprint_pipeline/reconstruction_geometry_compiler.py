"""Deterministic observed-surface metric geometry and collider compilation.

This module intentionally implements a conservative baseline.  It accepts a
source-bound JSON surface whose vertices and faces cite captured observations,
removes low-confidence support, and emits the remaining observed triangles.
It never closes holes, reconstructs unseen surfaces, or treats appearance as
geometry.  The collider compiler is likewise a no-fill baseline: it copies the
qualified observed surface and measures its topology while leaving collision
validation to an independent qualifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_collider_candidate_manifest,
    build_collider_qualification_report,
    build_metric_geometry_manifest,
)


SOURCE_SCHEMA = "observed_surface_mesh.v1"
COMPILER_SCHEMA = "metric_geometry_compilation_request.v1"
COMPILER_METHOD = "blueprint.observed_surface_confidence_filter"
COMPILER_VERSION = "1.0.0"
COLLIDER_METHOD = "blueprint.observed_surface_collider_baseline"
COLLIDER_VERSION = "1.0.0"
QUALIFICATION_REQUEST_SCHEMA = "collider_qualification_request.v1"
QUALIFICATION_MEASUREMENT_SCHEMA = "collider_qualification_measurements.v1"
QUALIFICATION_METHOD = "blueprint.independent_collider_measurement_evaluator"
QUALIFICATION_VERSION = "1.0.0"
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_VERTICES = 2_000_000
MAX_FACES = 4_000_000

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_UPPER_LIMITS = {
    "scale_error_fraction",
    "gravity_alignment_error_deg",
    "floor_height_residual_m",
    "wall_offset_residual_m",
    "visual_to_collider_disagreement_m",
    "clearance_error_m",
}
_LOWER_LIMITS = {"mesh_coverage_fraction", "minimum_obstacle_thickness_m"}
_QUALIFICATION_METRICS = _UPPER_LIMITS | _LOWER_LIMITS


class ReconstructionGeometryCompilerError(ValueError):
    """Typed fail-closed geometry compilation error."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_source_path(root: Path, relative_path: Any) -> Path:
    text = str(relative_path or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ReconstructionGeometryCompilerError(["source_asset_relative_path_unsafe"])
    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ReconstructionGeometryCompilerError(["source_asset_symlink_forbidden"])
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReconstructionGeometryCompilerError(["source_asset_missing"]) from exc
    if root_resolved not in resolved.parents or not resolved.is_file():
        raise ReconstructionGeometryCompilerError(["source_asset_escape_or_not_file"])
    return resolved


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ReconstructionGeometryCompilerError(["output_root_outside_artifact_root"]) from exc


def _prepare_output_root(output_root: str | Path, artifact_root: str | Path) -> Path:
    root = Path(artifact_root).resolve(strict=True)
    output = Path(output_root)
    if output.is_symlink():
        raise ReconstructionGeometryCompilerError(["output_root_symlink_forbidden"])
    try:
        resolved = output.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ReconstructionGeometryCompilerError(["output_root_invalid"]) from exc
    if resolved != root and root not in resolved.parents:
        raise ReconstructionGeometryCompilerError(["output_root_outside_artifact_root"])
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or output.resolve(strict=True) != resolved:
        raise ReconstructionGeometryCompilerError(["output_root_symlink_forbidden"])
    return output


def _finite_number(value: Any, *, minimum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        return None
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ReconstructionGeometryCompilerError(["source_asset_oversized"])

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ReconstructionGeometryCompilerError(
                    [f"source_asset_duplicate_json_key:{key}"]
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except ReconstructionGeometryCompilerError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconstructionGeometryCompilerError(["source_asset_malformed_json"]) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionGeometryCompilerError(["source_asset_not_object"])
    return dict(value)


def _validate_request_digest(request: Mapping[str, Any]) -> None:
    supplied = request.get("source_artifact_digest")
    if not isinstance(supplied, str) or supplied != canonical_digest(
        request, digest_field="source_artifact_digest"
    ):
        raise ReconstructionGeometryCompilerError(["source_artifact_digest_mismatch"])
    if request.get("schema_version") != COMPILER_SCHEMA:
        raise ReconstructionGeometryCompilerError(["compilation_request_schema_invalid"])


def _validate_metric_scale(request: Mapping[str, Any]) -> None:
    status = request.get("metric_scale_status")
    if status not in {"validated", "sensor_metric_unvalidated", "anchor_required"}:
        raise ReconstructionGeometryCompilerError(["metric_scale_status_invalid"])
    if status != "validated":
        return
    result = request.get("metric_scale_validation")
    if not isinstance(result, Mapping):
        raise ReconstructionGeometryCompilerError(["metric_scale_validation_missing"])
    supplied = result.get("metric_scale_validation_result_digest")
    if (
        result.get("status") != "validated"
        or not isinstance(supplied, str)
        or not _DIGEST.fullmatch(supplied)
        or supplied
        != canonical_digest(result, digest_field="metric_scale_validation_result_digest")
    ):
        raise ReconstructionGeometryCompilerError(["metric_scale_validation_invalid"])


def _parse_observed_surface(
    value: Mapping[str, Any], *, minimum_confidence: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    errors: list[str] = []
    if value.get("schema_version") != SOURCE_SCHEMA:
        errors.append("observed_surface_schema_invalid")
    frame = value.get("coordinate_frame_declaration")
    if not isinstance(frame, Mapping) or frame.get("units") != "meters" or frame.get(
        "up_axis"
    ) != "Z":
        errors.append("observed_surface_metric_z_up_frame_required")
    vertices_value = value.get("vertices")
    faces_value = value.get("faces")
    if not isinstance(vertices_value, list) or len(vertices_value) > MAX_VERTICES:
        errors.append("observed_surface_vertices_invalid")
        vertices_value = []
    if not isinstance(faces_value, list) or len(faces_value) > MAX_FACES:
        errors.append("observed_surface_faces_invalid")
        faces_value = []

    vertices: dict[str, dict[str, Any]] = {}
    rejected_vertices: list[dict[str, str]] = []
    for row in vertices_value:
        if not isinstance(row, Mapping):
            errors.append("observed_surface_vertex_not_object")
            continue
        vertex_id = str(row.get("vertex_id") or "")
        position = row.get("position_m")
        confidence = _finite_number(row.get("confidence"), minimum=0.0)
        region_id = str(row.get("region_id") or "")
        observations = row.get("source_observation_ids")
        if (
            _ID.fullmatch(vertex_id) is None
            or vertex_id in vertices
            or not isinstance(position, list)
            or len(position) != 3
            or any(_finite_number(item) is None for item in position)
            or confidence is None
            or confidence > 1.0
            or _ID.fullmatch(region_id) is None
            or not isinstance(observations, list)
            or not observations
            or any(_ID.fullmatch(str(item)) is None for item in observations)
            or row.get("generated") is not False
        ):
            errors.append(f"observed_surface_vertex_invalid:{vertex_id or 'unknown'}")
            continue
        normalized = {
            "vertex_id": vertex_id,
            "position_m": [float(item) for item in position],
            "confidence": float(confidence),
            "region_id": region_id,
            "source_observation_ids": sorted(set(str(item) for item in observations)),
        }
        if confidence < minimum_confidence:
            rejected_vertices.append(
                {"vertex_id": vertex_id, "reason": "below_minimum_confidence"}
            )
        vertices[vertex_id] = normalized

    faces: list[dict[str, Any]] = []
    rejected_faces: list[dict[str, str]] = []
    seen_faces: set[str] = set()
    low_confidence = {row["vertex_id"] for row in rejected_vertices}
    for row in faces_value:
        if not isinstance(row, Mapping):
            errors.append("observed_surface_face_not_object")
            continue
        face_id = str(row.get("face_id") or "")
        vertex_ids = row.get("vertex_ids")
        region_id = str(row.get("region_id") or "")
        if (
            _ID.fullmatch(face_id) is None
            or face_id in seen_faces
            or not isinstance(vertex_ids, list)
            or len(vertex_ids) != 3
            or len(set(str(item) for item in vertex_ids)) != 3
            or any(str(item) not in vertices for item in vertex_ids)
            or _ID.fullmatch(region_id) is None
            or row.get("observed") is not True
            or row.get("generated") is not False
        ):
            errors.append(f"observed_surface_face_invalid:{face_id or 'unknown'}")
            continue
        seen_faces.add(face_id)
        normalized_ids = [str(item) for item in vertex_ids]
        if any(item in low_confidence for item in normalized_ids):
            rejected_faces.append({"face_id": face_id, "reason": "low_confidence_support"})
            continue
        faces.append(
            {"face_id": face_id, "vertex_ids": normalized_ids, "region_id": region_id}
        )
    if errors:
        raise ReconstructionGeometryCompilerError(errors)
    if not faces:
        raise ReconstructionGeometryCompilerError(["insufficient_observed_surface"])

    used_ids = {item for face in faces for item in face["vertex_ids"]}
    selected = sorted((vertices[item] for item in used_ids), key=lambda row: row["vertex_id"])
    faces.sort(key=lambda row: row["face_id"])
    stats = {
        "input_vertex_count": len(vertices),
        "selected_vertex_count": len(selected),
        "rejected_vertex_count": len(rejected_vertices),
        "input_face_count": len(faces) + len(rejected_faces),
        "selected_face_count": len(faces),
        "rejected_face_count": len(rejected_faces),
        "rejected_vertices": sorted(rejected_vertices, key=lambda row: row["vertex_id"]),
        "rejected_faces": sorted(rejected_faces, key=lambda row: row["face_id"]),
    }
    return selected, faces, stats


def _write_ascii_ply(path: Path, vertices: list[dict[str, Any]], faces: list[dict[str, Any]]) -> None:
    indexes = {row["vertex_id"]: index for index, row in enumerate(vertices)}
    lines = [
        "ply",
        "format ascii 1.0",
        "comment Blueprint observed surface; no generated fill",
        f"element vertex {len(vertices)}",
        "property double x",
        "property double y",
        "property double z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for row in vertices:
        lines.append(" ".join(format(value, ".17g") for value in row["position_m"]))
    for row in faces:
        lines.append("3 " + " ".join(str(indexes[item]) for item in row["vertex_ids"]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _lineage(request: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "stable_run_identity",
        "source_capture_identity",
        "source_capture_digest",
        "original_file_references",
        "source_commit_sha",
        "deterministic_configuration_digest",
        "train_heldout_split_digest",
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "authority_used",
        "timestamp",
    )
    return {key: json.loads(json.dumps(request.get(key))) for key in keys}


def compile_metric_geometry(
    *, source_artifact: Mapping[str, Any], output_root: str | Path, artifact_root: str | Path
) -> dict[str, Any]:
    """Filter a source-bound observed surface into a metric reference candidate."""

    request = json.loads(json.dumps(dict(source_artifact)))
    _validate_request_digest(request)
    _validate_metric_scale(request)
    if request.get("generated_fill_used") is not False:
        raise ReconstructionGeometryCompilerError(["generated_or_unseen_fill_forbidden"])
    if request.get("appearance_asset_used_as_geometry_truth") is not False:
        raise ReconstructionGeometryCompilerError(["appearance_cannot_be_geometry_truth"])
    minimum = _finite_number(request.get("minimum_confidence"), minimum=0.0)
    if minimum is None or minimum > 1.0:
        raise ReconstructionGeometryCompilerError(["minimum_confidence_invalid"])
    source_binding = request.get("source_asset")
    if not isinstance(source_binding, Mapping):
        raise ReconstructionGeometryCompilerError(["source_asset_binding_missing"])
    root = Path(artifact_root)
    source_path = _safe_source_path(root, source_binding.get("relative_path"))
    if source_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ReconstructionGeometryCompilerError(["source_asset_oversized"])
    source_digest = _sha256(source_path)
    if source_binding.get("digest") != source_digest:
        raise ReconstructionGeometryCompilerError(["source_asset_digest_mismatch"])
    original_digests = {
        str(row.get("digest") or "")
        for row in request.get("original_file_references") or []
        if isinstance(row, Mapping)
    }
    if source_digest not in original_digests:
        raise ReconstructionGeometryCompilerError(["source_asset_provenance_binding_missing"])
    source = _load_json_object(source_path)
    if source.get("coordinate_frame_declaration") != request.get(
        "coordinate_frame_declaration"
    ):
        raise ReconstructionGeometryCompilerError(["source_coordinate_frame_binding_mismatch"])
    vertices, faces, filter_stats = _parse_observed_surface(
        source, minimum_confidence=float(minimum)
    )

    output = _prepare_output_root(output_root, root)
    output_path = output / "observed_metric_surface.ply"
    _write_ascii_ply(output_path, vertices, faces)
    output_digest = _sha256(output_path)
    observed_regions = sorted({row["region_id"] for row in faces})
    declared_values = request.get("declared_region_ids")
    unsupported_values = request.get("unsupported_region_ids")
    if (
        not isinstance(declared_values, list)
        or not isinstance(unsupported_values, list)
        or any(_ID.fullmatch(str(item)) is None for item in declared_values + unsupported_values)
    ):
        raise ReconstructionGeometryCompilerError(["region_ledger_invalid"])
    declared_regions = {str(item) for item in declared_values}
    unsupported_regions = sorted(
        declared_regions - set(observed_regions) | set(str(item) for item in unsupported_values)
    )
    asset_reference = _relative_to_root(output_path, root)
    value = {
        **_lineage(request),
        "producing_method": COMPILER_METHOD,
        "implementation_version": COMPILER_VERSION,
        "input_digests": [
            {"artifact_id": "metric_geometry_compilation_request", "digest": request["source_artifact_digest"]},
            {"artifact_id": "observed_surface_source", "digest": source_digest},
        ],
        "output_digests": [{"artifact_id": "observed_metric_surface", "digest": output_digest}],
        "metric_scale_status": request["metric_scale_status"],
        "units": "meters",
        "provider_runtime_identity": {"provider": "local", "runtime": "python"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "warnings": list(request.get("warnings") or []),
        "blockers": list(request.get("blockers") or []),
        "parent_artifact_or_event": {"digest": request["source_artifact_digest"]},
        "generated_fill_used": False,
        "appearance_asset_used_as_geometry_truth": False,
        "observed_region_ids": observed_regions,
        "unsupported_region_ids": unsupported_regions,
        "confidence_filter": {
            "minimum_confidence": float(minimum),
            "missing_depth_preserved": True,
            "low_confidence_preserved_as_rejected_evidence": True,
            **filter_stats,
        },
        "geometry_asset_reference": asset_reference,
        "geometry_asset_digest": output_digest,
        "topology": {
            "vertex_count": len(vertices),
            "triangle_count": len(faces),
            "holes_closed": 0,
            "unseen_surfaces_created": 0,
        },
        "proof_effect": "metric_reference_candidate_only",
        "claim_ceiling": "metric_reference_geometry",
    }
    return build_metric_geometry_manifest(value)


def _read_ascii_ply(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReconstructionGeometryCompilerError(["metric_geometry_asset_not_ascii_ply"]) from exc
    if not lines or lines[0] != "ply" or "format ascii 1.0" not in lines[:3]:
        raise ReconstructionGeometryCompilerError(["metric_geometry_asset_not_ascii_ply"])
    try:
        vertex_count = int(next(line.split()[2] for line in lines if line.startswith("element vertex ")))
        face_count = int(next(line.split()[2] for line in lines if line.startswith("element face ")))
        header_end = lines.index("end_header")
    except (StopIteration, ValueError, IndexError) as exc:
        raise ReconstructionGeometryCompilerError(["metric_geometry_ply_header_invalid"]) from exc
    vertex_lines = lines[header_end + 1 : header_end + 1 + vertex_count]
    face_lines = lines[header_end + 1 + vertex_count :]
    if len(vertex_lines) != vertex_count or len(face_lines) != face_count:
        raise ReconstructionGeometryCompilerError(["metric_geometry_ply_count_mismatch"])
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    try:
        for line in vertex_lines:
            values = tuple(float(item) for item in line.split())
            if len(values) != 3 or not all(math.isfinite(item) for item in values):
                raise ValueError
            vertices.append(values)
        for line in face_lines:
            values = [int(item) for item in line.split()]
            if len(values) != 4 or values[0] != 3 or len(set(values[1:])) != 3:
                raise ValueError
            face = tuple(values[1:])
            if min(face) < 0 or max(face) >= vertex_count:
                raise ValueError
            faces.append(face)
    except ValueError as exc:
        raise ReconstructionGeometryCompilerError(["metric_geometry_ply_payload_invalid"]) from exc
    return vertices, faces


def _topology(faces: list[tuple[int, int, int]]) -> tuple[dict[str, Any], dict[str, Any]]:
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces[vertex].append(face_index)
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[tuple(sorted((start, end)))] += 1
    neighbors: dict[int, set[int]] = defaultdict(set)
    for indexes in vertex_to_faces.values():
        for left in indexes:
            neighbors[left].update(item for item in indexes if item != left)
    unseen = set(range(len(faces)))
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            for neighbor in neighbors[queue.popleft()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    boundary_graph: dict[int, set[int]] = defaultdict(set)
    for start, end in boundary_edges:
        boundary_graph[start].add(end)
        boundary_graph[end].add(start)
    boundary_vertices = set(boundary_graph)
    boundary_components = 0
    while boundary_vertices:
        boundary_components += 1
        queue = deque([boundary_vertices.pop()])
        while queue:
            for neighbor in boundary_graph[queue.popleft()]:
                if neighbor in boundary_vertices:
                    boundary_vertices.remove(neighbor)
                    queue.append(neighbor)
    return (
        {
            "count": components,
            "disconnected_count": max(0, components - 1),
            "face_count": len(faces),
        },
        {
            "count": boundary_components,
            "boundary_edge_count": len(boundary_edges),
            "nonmanifold_edge_count": nonmanifold_edges,
            "area_m2": None,
            "area_measurement_status": "not_measured_by_topology_baseline",
        },
    )


def compile_collision_candidate(
    *, source_artifact: Mapping[str, Any], output_root: str | Path, artifact_root: str | Path
) -> dict[str, Any]:
    """Emit a no-fill collision candidate from a metric geometry manifest."""

    try:
        metric = build_metric_geometry_manifest(source_artifact)
    except ReconstructionGeometryContractError as exc:
        raise ReconstructionGeometryCompilerError(exc.codes) from exc
    reference = metric.get("geometry_asset_reference")
    source_path = _safe_source_path(Path(artifact_root), reference)
    if _sha256(source_path) != metric.get("geometry_asset_digest"):
        raise ReconstructionGeometryCompilerError(["metric_geometry_asset_digest_mismatch"])
    vertices, faces = _read_ascii_ply(source_path)
    if not vertices or not faces:
        raise ReconstructionGeometryCompilerError(["metric_geometry_surface_empty"])
    components, holes = _topology(faces)
    output = _prepare_output_root(output_root, artifact_root)
    output_path = output / "observed_surface_collider_candidate.ply"
    shutil.copyfile(source_path, output_path)
    output_digest = _sha256(output_path)
    root = Path(artifact_root)
    value = {
        **_lineage(metric),
        "producing_method": COLLIDER_METHOD,
        "implementation_version": COLLIDER_VERSION,
        "input_digests": [
            {
                "artifact_id": "metric_geometry_manifest",
                "digest": metric["metric_geometry_manifest_digest"],
            },
            {"artifact_id": "metric_geometry_asset", "digest": metric["geometry_asset_digest"]},
        ],
        "output_digests": [{"artifact_id": "collider_candidate", "digest": output_digest}],
        "metric_geometry_manifest_digest": metric["metric_geometry_manifest_digest"],
        "metric_scale_status": metric["metric_scale_status"],
        "units": "meters",
        "provider_runtime_identity": {"provider": "local", "runtime": "python"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "warnings": list(metric.get("warnings") or []),
        "blockers": list(metric.get("blockers") or []),
        "parent_artifact_or_event": {"digest": metric["metric_geometry_manifest_digest"]},
        "collider_asset_reference": _relative_to_root(output_path, root),
        "collider_asset_digest": output_digest,
        "unobserved_regions_filled": False,
        "collision_validated": False,
        "component_statistics": {**components, "vertex_count": len(vertices)},
        "hole_statistics": holes,
        "unsupported_region_ids": list(metric.get("unsupported_region_ids") or []),
        "candidate_operation": "exact_observed_surface_copy_no_decimation_no_hole_fill",
        "proof_effect": "collision_candidate_only",
        "claim_ceiling": "collision_geometry_candidate",
    }
    return build_collider_candidate_manifest(value)


def _validated_metric_map(value: Any, *, code: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != _QUALIFICATION_METRICS:
        raise ReconstructionGeometryCompilerError([code])
    normalized: dict[str, float] = {}
    for key in sorted(_QUALIFICATION_METRICS):
        number = _finite_number(value.get(key), minimum=0.0)
        if number is None:
            raise ReconstructionGeometryCompilerError([code])
        normalized[key] = float(number)
    return normalized


def qualify_collision_candidate(
    *,
    source_artifact: Mapping[str, Any],
    output_root: str | Path,
    artifact_root: str | Path,
    qualification_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Qualify a collider from independent, frozen measurement evidence."""

    del output_root  # The supervisor owns report persistence.
    try:
        candidate = build_collider_candidate_manifest(source_artifact)
    except ReconstructionGeometryContractError as exc:
        raise ReconstructionGeometryCompilerError(exc.codes) from exc
    request = json.loads(json.dumps(dict(qualification_request)))
    request_digest = request.get("collider_qualification_request_digest")
    if (
        request.get("schema_version") != QUALIFICATION_REQUEST_SCHEMA
        or not isinstance(request_digest, str)
        or request_digest
        != canonical_digest(request, digest_field="collider_qualification_request_digest")
    ):
        raise ReconstructionGeometryCompilerError(["collider_qualification_request_invalid"])
    if request.get("collider_candidate_manifest_digest") != candidate.get(
        "collider_candidate_manifest_digest"
    ):
        raise ReconstructionGeometryCompilerError(["collider_qualification_candidate_mismatch"])
    for key in (
        "source_capture_digest",
        "train_heldout_split_digest",
        "coordinate_frame_declaration",
        "camera_calibration_binding",
    ):
        if request.get(key) != candidate.get(key):
            raise ReconstructionGeometryCompilerError(
                [f"collider_qualification_lineage_mismatch:{key}"]
            )
    if request.get("metric_scale_status") != candidate.get("metric_scale_status"):
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_metric_scale_binding_mismatch"]
        )

    thresholds = _validated_metric_map(
        request.get("thresholds"), code="collider_qualification_thresholds_invalid"
    )
    threshold_digest = canonical_digest(thresholds, digest_field="qa_thresholds_digest")
    if request.get("qa_thresholds_digest") != threshold_digest:
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_thresholds_digest_mismatch"]
        )
    evaluator = request.get("independent_evaluator")
    if (
        not isinstance(evaluator, Mapping)
        or evaluator.get("candidate_method_independent") is not True
        or not isinstance(evaluator.get("method_id"), str)
        or not evaluator.get("method_id")
        or evaluator.get("method_id") == candidate.get("producing_method")
    ):
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_independent_evaluator_required"]
        )
    measurement_binding = request.get("measurement_artifact")
    if not isinstance(measurement_binding, Mapping):
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_binding_missing"]
        )
    root = Path(artifact_root)
    measurement_path = _safe_source_path(root, measurement_binding.get("relative_path"))
    if measurement_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_oversized"]
        )
    measurement_digest = _sha256(measurement_path)
    if measurement_binding.get("digest") != measurement_digest:
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_digest_mismatch"]
        )
    original_digests = {
        str(row.get("digest") or "")
        for row in request.get("original_file_references") or []
        if isinstance(row, Mapping)
    }
    if measurement_digest not in original_digests:
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_provenance_missing"]
        )
    measurement = _load_json_object(measurement_path)
    if measurement.get("schema_version") != QUALIFICATION_MEASUREMENT_SCHEMA:
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_schema_invalid"]
        )
    if measurement.get("collider_candidate_manifest_digest") != candidate.get(
        "collider_candidate_manifest_digest"
    ) or measurement.get("collider_asset_digest") != candidate.get("collider_asset_digest"):
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_candidate_binding_mismatch"]
        )
    if (
        measurement.get("candidate_self_graded") is not False
        or measurement.get("thresholds_modified_after_measurement") is not False
        or measurement.get("generated_geometry_promoted_to_collision_truth") is not False
        or measurement.get("evaluator") != evaluator
    ):
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_independence_invalid"]
        )
    measurements = _validated_metric_map(
        measurement.get("measurements"),
        code="collider_qualification_measurements_invalid",
    )
    requested_regions = request.get("task_region_ids")
    measured_regions = measurement.get("evaluated_task_region_ids")
    if (
        not isinstance(requested_regions, list)
        or not requested_regions
        or any(_ID.fullmatch(str(item)) is None for item in requested_regions)
        or sorted(set(str(item) for item in measured_regions or []))
        != sorted(set(str(item) for item in requested_regions))
    ):
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_task_region_coverage_invalid"]
        )
    robot_checked = measurement.get("robot_footprint_navigability_checked") is True
    measurement_blockers = measurement.get("blockers")
    if not isinstance(measurement_blockers, list):
        raise ReconstructionGeometryCompilerError(
            ["collider_qualification_measurement_blockers_invalid"]
        )
    passed = (
        candidate.get("metric_scale_status") == "validated"
        and robot_checked
        and not measurement_blockers
    )
    for key in sorted(_QUALIFICATION_METRICS):
        if key in _UPPER_LIMITS and measurements[key] > thresholds[key]:
            passed = False
        if key in _LOWER_LIMITS and measurements[key] < thresholds[key]:
            passed = False
    decision = "accepted_bounded_navigation" if passed else "rejected"
    failed_thresholds = sorted(
        key
        for key in _QUALIFICATION_METRICS
        if (key in _UPPER_LIMITS and measurements[key] > thresholds[key])
        or (key in _LOWER_LIMITS and measurements[key] < thresholds[key])
    )
    value = {
        **_lineage(request),
        "producing_method": QUALIFICATION_METHOD,
        "implementation_version": QUALIFICATION_VERSION,
        "input_digests": [
            {
                "artifact_id": "collider_candidate_manifest",
                "digest": candidate["collider_candidate_manifest_digest"],
            },
            {
                "artifact_id": "independent_collider_measurements",
                "digest": measurement_digest,
            },
            {"artifact_id": "qa_thresholds", "digest": threshold_digest},
            {"artifact_id": "qualification_request", "digest": request_digest},
        ],
        "output_digests": [],
        "units": "meters",
        "provider_runtime_identity": {"provider": "local", "runtime": "python"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "warnings": list(request.get("warnings") or []),
        "blockers": sorted(set(str(item) for item in measurement_blockers)),
        "parent_artifact_or_event": {
            "digest": candidate["collider_candidate_manifest_digest"]
        },
        "collider_candidate_manifest_digest": candidate[
            "collider_candidate_manifest_digest"
        ],
        "collider_asset_digest": candidate["collider_asset_digest"],
        "collider_qualification_request_digest": request_digest,
        "measurement_artifact_digest": measurement_digest,
        "qa_thresholds_digest": threshold_digest,
        "measurements": measurements,
        "thresholds": thresholds,
        "failed_threshold_ids": failed_thresholds,
        "metric_scale_status": candidate["metric_scale_status"],
        "robot_footprint_navigability_checked": robot_checked,
        "task_region_ids": sorted(set(str(item) for item in requested_regions)),
        "independent_evaluator": dict(evaluator),
        "candidate_self_graded": False,
        "decision": decision,
        "unsupported_claims": [
            "grasping",
            "articulation",
            "contact_force",
            "deployment",
            "physical_success",
        ],
        "proof_effect": "bounded_navigation_collision_qualification",
        "claim_ceiling": "bounded_navigation_simulation",
    }
    return build_collider_qualification_report(value)


__all__ = [
    "COLLIDER_METHOD",
    "COLLIDER_VERSION",
    "COMPILER_METHOD",
    "COMPILER_SCHEMA",
    "COMPILER_VERSION",
    "QUALIFICATION_MEASUREMENT_SCHEMA",
    "QUALIFICATION_METHOD",
    "QUALIFICATION_REQUEST_SCHEMA",
    "QUALIFICATION_VERSION",
    "ReconstructionGeometryCompilerError",
    "SOURCE_SCHEMA",
    "compile_collision_candidate",
    "compile_metric_geometry",
    "qualify_collision_candidate",
]
