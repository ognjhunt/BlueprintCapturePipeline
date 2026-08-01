"""Qualify an externally captured point cloud as COLMAP initialization points.

The compiler binds a strictly imported provider point cloud (for example the
MuSHRoom Polycam export) into the frozen candidate camera frame using only
candidate-frame camera centers from two published trajectories of the same
frames.  It estimates a proper-rotation similarity transform, applies frozen
residual/bounds gates, and emits an exporter-compatible vertices artifact plus
a digest-bound compilation result.  It never reads hidden held-out pixels or
hidden camera poses, never fabricates geometry, and cannot upgrade metric,
collision, Isaac, task, physical, or deployment claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .external_reconstruction_import import (
    ExternalReconstructionImportError,
    build_external_reconstruction_import_receipt,
)


REQUEST_SCHEMA_VERSION = "external_pointcloud_initialization_request.v1"
RESULT_SCHEMA_VERSION = "external_pointcloud_initialization_result.v1"
VERTICES_SCHEMA_VERSION = "external_pointcloud_initialization_vertices.v1"
COMPILED_STATUS = "compiled_external_initialization_points"

MAX_PLY_HEADER_BYTES = 65_536
MAX_PLY_VERTEX_COUNT = 50_000_000
MIN_ALIGNMENT_PAIR_COUNT = 8
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLY_SCALAR_SIZES = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}
_PLY_NUMPY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


class ExternalPointcloudInitializationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_file(root: Path, relative_path: str, *, label: str) -> Path:
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if (
        not relative_path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "evaluator_hidden" in relative.parts
        or "held_out" in relative.parts
    ):
        raise ExternalPointcloudInitializationError([f"{label}_path_unsafe_or_hidden"])
    resolved_root = root.resolve()
    path = (resolved_root / Path(*relative.parts)).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ExternalPointcloudInitializationError([f"{label}_path_escape"])
    if not path.is_file() or path.is_symlink():
        raise ExternalPointcloudInitializationError([f"{label}_missing_or_symlink"])
    return path


def _write_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ExternalPointcloudInitializationError(
                ["external_pointcloud_immutable_conflict"]
            )
        return "sha256:" + hashlib.sha256(payload).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ExternalPointcloudInitializationError(
                    ["external_pointcloud_immutable_conflict"]
                )
    finally:
        temporary.unlink(missing_ok=True)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def read_pointcloud_ply_positions(path: Path) -> np.ndarray:
    """Read x/y/z positions from a strict binary little-endian PLY file.

    Only scalar vertex properties are supported; list properties, ASCII, and
    big-endian layouts are refused rather than guessed at.
    """

    with path.open("rb") as stream:
        header_bytes = b""
        while b"end_header" not in header_bytes:
            chunk = stream.read(4096)
            if not chunk or len(header_bytes) > MAX_PLY_HEADER_BYTES:
                raise ExternalPointcloudInitializationError(["ply_header_invalid"])
            header_bytes += chunk
        header_end = header_bytes.index(b"end_header")
        newline = header_bytes.index(b"\n", header_end)
        header_text = header_bytes[:newline].decode("ascii", errors="strict")
        body_offset = newline + 1
        lines = [line.strip() for line in header_text.splitlines() if line.strip()]
        if not lines or lines[0] != "ply":
            raise ExternalPointcloudInitializationError(["ply_magic_invalid"])
        if "format binary_little_endian 1.0" not in lines:
            raise ExternalPointcloudInitializationError(["ply_format_unsupported"])
        vertex_count: int | None = None
        properties: list[tuple[str, str]] = []
        active_element: str | None = None
        for line in lines[1:]:
            if (
                line.startswith("comment")
                or line.startswith("obj_info")
                or line.startswith("format ")
            ):
                continue
            if line.startswith("element "):
                parts = line.split()
                if len(parts) != 3:
                    raise ExternalPointcloudInitializationError(["ply_header_invalid"])
                active_element = parts[1]
                if active_element == "vertex":
                    if vertex_count is not None:
                        raise ExternalPointcloudInitializationError(["ply_header_invalid"])
                    try:
                        vertex_count = int(parts[2])
                    except ValueError as exc:
                        raise ExternalPointcloudInitializationError(
                            ["ply_header_invalid"]
                        ) from exc
                elif parts[2] not in {"0"}:
                    raise ExternalPointcloudInitializationError(
                        ["ply_non_vertex_elements_unsupported"]
                    )
                continue
            if line.startswith("property "):
                if active_element != "vertex":
                    continue
                parts = line.split()
                if len(parts) == 3 and parts[1] in _PLY_SCALAR_SIZES:
                    properties.append((parts[2], parts[1]))
                else:
                    raise ExternalPointcloudInitializationError(
                        ["ply_vertex_property_unsupported"]
                    )
                continue
            if line == "end_header":
                break
            raise ExternalPointcloudInitializationError(["ply_header_invalid"])
        if (
            vertex_count is None
            or vertex_count <= 0
            or vertex_count > MAX_PLY_VERTEX_COUNT
        ):
            raise ExternalPointcloudInitializationError(["ply_vertex_count_invalid"])
        names = [name for name, _ in properties]
        if len(set(names)) != len(names) or not {"x", "y", "z"} <= set(names):
            raise ExternalPointcloudInitializationError(["ply_position_properties_missing"])
        stride = sum(_PLY_SCALAR_SIZES[kind] for _, kind in properties)
        expected = vertex_count * stride
        stream.seek(0, os.SEEK_END)
        actual = stream.tell() - body_offset
        if actual < expected:
            raise ExternalPointcloudInitializationError(["ply_body_truncated"])
        stream.seek(body_offset)
        dtype = np.dtype(
            [(name, "<" + _PLY_NUMPY_TYPES[kind]) for name, kind in properties]
        )
        rows = np.fromfile(stream, dtype=dtype, count=vertex_count)
    if rows.shape[0] != vertex_count:
        raise ExternalPointcloudInitializationError(["ply_body_truncated"])
    positions = np.stack(
        [rows["x"].astype(np.float64), rows["y"].astype(np.float64), rows["z"].astype(np.float64)],
        axis=1,
    )
    if not np.isfinite(positions).all():
        raise ExternalPointcloudInitializationError(["ply_positions_nonfinite"])
    return positions


def _load_pose_centers(
    document: Mapping[str, Any], *, candidate_ids: set[str], label: str
) -> dict[str, np.ndarray]:
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ExternalPointcloudInitializationError([f"{label}_frames_missing"])
    centers: dict[str, np.ndarray] = {}
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ExternalPointcloudInitializationError([f"{label}_frame_invalid"])
        frame_id = Path(str(frame.get("file_path") or "").replace("\\", "/")).stem
        if frame_id not in candidate_ids:
            continue
        if frame_id in centers:
            raise ExternalPointcloudInitializationError([f"{label}_frame_duplicate"])
        try:
            matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalPointcloudInitializationError([f"{label}_pose_invalid"]) from exc
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
        ):
            raise ExternalPointcloudInitializationError([f"{label}_pose_invalid"])
        centers[frame_id] = matrix[:3, 3]
    if set(centers) != candidate_ids:
        raise ExternalPointcloudInitializationError([f"{label}_candidate_coverage_incomplete"])
    return centers


def _estimate_similarity(
    source_centers: np.ndarray, target_centers: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, bool]:
    """Umeyama similarity (scale, rotation, translation) with proper rotation.

    Returns (scale, rotation, translation, reflection_preferred).  The rotation
    is always proper (determinant +1); ``reflection_preferred`` reports whether
    an improper transform would have fit better, which callers must fail on.
    """

    source_mean = source_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_centered = source_centers - source_mean
    target_centered = target_centers - target_mean
    source_variance = float(np.mean(np.sum(np.square(source_centered), axis=1)))
    if source_variance <= 0.0:
        raise ExternalPointcloudInitializationError(["alignment_source_degenerate"])
    covariance = (target_centered.T @ source_centered) / source_centers.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection_preferred = bool(np.linalg.det(u @ vt) < 0.0)
    correction = np.ones(3)
    if reflection_preferred:
        correction[2] = -1.0
    rotation = u @ np.diag(correction) @ vt
    scale = float(np.sum(singular_values * correction) / source_variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ExternalPointcloudInitializationError(["alignment_scale_invalid"])
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation, reflection_preferred


def compile_external_pointcloud_initialization(
    *,
    source_artifact: Mapping[str, Any],
    import_receipt: Mapping[str, Any],
    import_output_root: str | Path,
    source_trajectory_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Transform a strictly imported point cloud into the candidate camera frame.

    ``source_artifact`` is an ``external_pointcloud_initialization_request.v1``
    mapping.  The alignment uses camera centers of candidate frames only; the
    hidden held-out frames and pixels are never read.
    """

    request = json.loads(canonical_json(dict(source_artifact)))
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("external_pointcloud_request_schema_invalid")
    for key in ("source_capture_digest", "frozen_split_digest", "camera_observation_digest"):
        if not _is_digest(request.get(key)):
            errors.append(f"external_pointcloud_request_{key}_invalid")
    commit = str(request.get("source_commit_sha") or "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        errors.append("external_pointcloud_request_source_commit_invalid")
    candidate_ids = request.get("candidate_observation_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) < MIN_ALIGNMENT_PAIR_COUNT
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(not isinstance(item, str) or not item for item in candidate_ids)
    ):
        errors.append("external_pointcloud_request_candidate_ids_invalid")
    thresholds = request.get("alignment_thresholds")
    threshold_fields = {
        "maximum_rms_residual",
        "maximum_max_residual",
        "minimum_in_bounds_ratio",
        "bounds_inflation_factor",
        "minimum_bounds_margin",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != threshold_fields:
        errors.append("external_pointcloud_request_thresholds_invalid")
    else:
        for key in threshold_fields:
            number = thresholds.get(key)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) <= 0.0
            ):
                errors.append(f"external_pointcloud_request_threshold_invalid:{key}")
    if request.get("thresholds_frozen_before_alignment") is not True:
        errors.append("external_pointcloud_request_thresholds_not_frozen")
    if request.get("hidden_heldout_access_requested") is not False:
        errors.append("external_pointcloud_request_hidden_access_not_false")
    maximum_points = request.get("maximum_points")
    if (
        isinstance(maximum_points, bool)
        or not isinstance(maximum_points, int)
        or maximum_points <= 0
    ):
        errors.append("external_pointcloud_request_maximum_points_invalid")
    for key in ("pointcloud_asset_id", "source_trajectory_relative_path", "target_trajectory_relative_path", "timestamp", "stable_run_identity"):
        if not str(request.get(key) or "").strip():
            errors.append(f"external_pointcloud_request_{key}_missing")
    if not isinstance(request.get("coordinate_frame_declaration"), Mapping):
        errors.append("external_pointcloud_request_coordinate_frame_invalid")
    if not isinstance(request.get("authority_used"), Mapping):
        errors.append("external_pointcloud_request_authority_missing")
    supplied_digest = request.pop("external_pointcloud_initialization_request_digest", None)
    request_digest = canonical_digest(
        request, digest_field="external_pointcloud_initialization_request_digest"
    )
    request["external_pointcloud_initialization_request_digest"] = request_digest
    if supplied_digest is not None and supplied_digest != request_digest:
        errors.append("external_pointcloud_request_digest_mismatch")
    if errors:
        raise ExternalPointcloudInitializationError(errors)

    try:
        receipt = build_external_reconstruction_import_receipt(dict(import_receipt))
    except ExternalReconstructionImportError as exc:
        raise ExternalPointcloudInitializationError(
            [f"external_pointcloud_import_receipt_invalid:{code}" for code in exc.codes]
        ) from exc
    if receipt.get("source_capture_digest") != request["source_capture_digest"]:
        raise ExternalPointcloudInitializationError(
            ["external_pointcloud_import_capture_binding_mismatch"]
        )
    asset_rows = [
        asset
        for asset in receipt["imported_assets"]
        if asset.get("asset_id") == request["pointcloud_asset_id"]
    ]
    if len(asset_rows) != 1 or str(asset_rows[0].get("format") or "") != ".ply":
        raise ExternalPointcloudInitializationError(["external_pointcloud_asset_binding_invalid"])
    asset = asset_rows[0]
    import_root = Path(import_output_root).resolve()
    ply_path = _safe_file(import_root, str(asset["relative_path"]), label="pointcloud_asset")
    if _sha256_file(ply_path) != asset["digest"]:
        raise ExternalPointcloudInitializationError(["external_pointcloud_asset_digest_mismatch"])

    trajectory_root = Path(source_trajectory_root).resolve()
    source_trajectory_path = _safe_file(
        trajectory_root,
        str(request["source_trajectory_relative_path"]),
        label="source_trajectory",
    )
    target_trajectory_path = _safe_file(
        trajectory_root,
        str(request["target_trajectory_relative_path"]),
        label="target_trajectory",
    )
    source_trajectory_digest = _sha256_file(source_trajectory_path)
    target_trajectory_digest = _sha256_file(target_trajectory_path)
    for key, observed in (
        ("source_trajectory_digest", source_trajectory_digest),
        ("target_trajectory_digest", target_trajectory_digest),
    ):
        expected = request.get(key)
        if expected is not None and expected != observed:
            raise ExternalPointcloudInitializationError(
                [f"external_pointcloud_{key}_mismatch"]
            )
    try:
        source_document = json.loads(source_trajectory_path.read_text(encoding="utf-8"))
        target_document = json.loads(target_trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalPointcloudInitializationError(
            ["external_pointcloud_trajectory_unreadable"]
        ) from exc
    candidate_set = set(candidate_ids)
    source_centers_by_id = _load_pose_centers(
        source_document, candidate_ids=candidate_set, label="source_trajectory"
    )
    target_centers_by_id = _load_pose_centers(
        target_document, candidate_ids=candidate_set, label="target_trajectory"
    )
    ordered_ids = sorted(candidate_set)
    source_centers = np.stack([source_centers_by_id[i] for i in ordered_ids])
    target_centers = np.stack([target_centers_by_id[i] for i in ordered_ids])

    scale, rotation, translation, reflection_preferred = _estimate_similarity(
        source_centers, target_centers
    )
    if reflection_preferred:
        raise ExternalPointcloudInitializationError(
            ["external_pointcloud_handedness_reflection_detected"]
        )
    residual_vectors = target_centers - (scale * (source_centers @ rotation.T) + translation)
    residuals = np.linalg.norm(residual_vectors, axis=1)
    rms_residual = float(np.sqrt(np.mean(np.square(residuals))))
    max_residual = float(np.max(residuals))
    if (
        rms_residual > float(thresholds["maximum_rms_residual"])
        or max_residual > float(thresholds["maximum_max_residual"])
    ):
        raise ExternalPointcloudInitializationError(
            ["external_pointcloud_alignment_residual_threshold_exceeded"]
        )

    positions = read_pointcloud_ply_positions(ply_path)
    transformed = scale * (positions @ rotation.T) + translation
    if not np.isfinite(transformed).all():
        raise ExternalPointcloudInitializationError(["external_pointcloud_transform_nonfinite"])
    bounds_min = target_centers.min(axis=0)
    bounds_max = target_centers.max(axis=0)
    bounds_center = (bounds_min + bounds_max) / 2.0
    half_extent = (bounds_max - bounds_min) / 2.0
    margin = np.maximum(
        half_extent * float(thresholds["bounds_inflation_factor"]),
        float(thresholds["minimum_bounds_margin"]),
    )
    in_bounds = np.all(np.abs(transformed - bounds_center) <= margin, axis=1)
    in_bounds_ratio = float(np.mean(in_bounds))
    if in_bounds_ratio < float(thresholds["minimum_in_bounds_ratio"]):
        raise ExternalPointcloudInitializationError(
            ["external_pointcloud_out_of_bounds_ratio_exceeded"]
        )

    total_points = int(transformed.shape[0])
    stride = max(1, math.ceil(total_points / int(maximum_points)))
    sampled = np.round(transformed[::stride], 6)
    vertices = [{"position_m": [float(v[0]), float(v[1]), float(v[2])]} for v in sampled]

    output = Path(output_root).resolve()
    vertices_artifact = {
        "schema_version": VERTICES_SCHEMA_VERSION,
        "source_capture_digest": request["source_capture_digest"],
        "train_heldout_split_digest": request["frozen_split_digest"],
        "camera_observation_digest": request["camera_observation_digest"],
        "external_import_receipt_digest": receipt["external_import_receipt_digest"],
        "external_pointcloud_initialization_request_digest": request_digest,
        "generated_fill_used": False,
        "units_declaration": str(request.get("units") or "target_frame_units_not_independently_validated"),
        "vertex_count": len(vertices),
        "vertices": vertices,
    }
    vertices_relative = "initialization/external_pointcloud_initialization_vertices.v1.json"
    vertices_digest = _write_immutable(
        output / Path(*PurePosixPath(vertices_relative).parts),
        (canonical_json(vertices_artifact) + "\n").encode("utf-8"),
    )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "stable_run_identity": request["stable_run_identity"],
        "status": COMPILED_STATUS,
        "source_capture_digest": request["source_capture_digest"],
        "source_commit_sha": commit,
        "train_heldout_split_digest": request["frozen_split_digest"],
        "camera_observation_binding": {
            "camera_observation_digest": request["camera_observation_digest"],
            "candidate_observation_count": len(ordered_ids),
        },
        "external_import_receipt_digest": receipt["external_import_receipt_digest"],
        "external_pointcloud_initialization_request_digest": request_digest,
        "pointcloud_asset": {
            "asset_id": asset["asset_id"],
            "digest": asset["digest"],
            "size_bytes": asset["size_bytes"],
        },
        "source_trajectory_digest": source_trajectory_digest,
        "target_trajectory_digest": target_trajectory_digest,
        "alignment": {
            "method": "umeyama_similarity_candidate_camera_centers.v1",
            "pair_count": len(ordered_ids),
            "estimated_scale_factor": round(scale, 12),
            "rotation_matrix": [[round(float(v), 12) for v in row] for row in rotation],
            "translation": [round(float(v), 12) for v in translation],
            "rms_residual": round(rms_residual, 9),
            "max_residual": round(max_residual, 9),
            "in_bounds_ratio": round(in_bounds_ratio, 9),
            "thresholds": {key: float(thresholds[key]) for key in sorted(threshold_fields)},
        },
        "surface_asset": {
            "relative_path": vertices_relative,
            "digest": vertices_digest,
        },
        "total_source_points": total_points,
        "emitted_point_count": len(vertices),
        "subsample_stride": stride,
        "coordinate_frame_declaration": dict(request["coordinate_frame_declaration"]),
        "units": str(request.get("units") or "target_frame_units_not_independently_validated"),
        "metric_scale_status": str(
            request.get("metric_scale_status") or "not_independently_validated"
        ),
        "hidden_heldout_observations_accessed": False,
        "generated_fill_used": False,
        "raw_input_poses_modified": False,
        "alignment_used_candidate_frames_only": True,
        "reflection_preferred_by_alignment": False,
        "alignment_residual_gates_passed": True,
        "authority_used": dict(request["authority_used"]),
        "cost_usd": 0.0,
        "warnings": [
            "vertex_key_position_m_is_exporter_format_units_remain_declared_not_proven"
        ],
        "blockers": [],
        "proof_effect": "initialization_geometry_candidate_binding_only",
        "claim_ceiling": "reconstruction_training_request",
        "parent_artifact_or_event": {
            "external_import_receipt_digest": receipt["external_import_receipt_digest"],
            "request_digest": request_digest,
        },
        "timestamp": request["timestamp"],
    }
    result["external_pointcloud_initialization_result_digest"] = canonical_digest(
        result, digest_field="external_pointcloud_initialization_result_digest"
    )
    _write_immutable(
        output / "initialization" / "external_pointcloud_initialization_result.v1.json",
        (canonical_json(result) + "\n").encode("utf-8"),
    )
    return result


__all__ = [
    "COMPILED_STATUS",
    "ExternalPointcloudInitializationError",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "VERTICES_SCHEMA_VERSION",
    "compile_external_pointcloud_initialization",
    "read_pointcloud_ply_positions",
]
