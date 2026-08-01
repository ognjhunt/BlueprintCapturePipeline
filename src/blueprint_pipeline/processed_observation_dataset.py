"""Compile processed public RGB-D/pose sequences without inventing raw authority.

This module is for datasets such as MuSHRoom that publish captured RGB images,
aligned depth, and camera transforms but do not publish the original retained
video or Blueprint encoder/sensor ledgers.  It freezes candidate and evaluator
views, verifies every consumed byte, and emits the existing candidate/held-out
manifests used by reconstruction consumers.  It deliberately does not claim
Raw Contract authority, decoded video timing, metric verification, physics, or
physical task success.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
COMPILER_VERSION = "processed_observation_dataset_compiler.v1"
REQUEST_SCHEMA_VERSION = "processed_observation_dataset_compile_request.v1"
PROCESSED_DATASET_SCHEMA_VERSION = "processed_observation_dataset_manifest.v1"
PROCESSED_SPLIT_SCHEMA_VERSION = "processed_observation_split_manifest.v1"
PROCESSED_CANDIDATE_SCHEMA_VERSION = "processed_candidate_dataset_manifest.v1"
PROCESSED_HELDOUT_SCHEMA_VERSION = "processed_hidden_heldout_manifest.v1"
CAMERA_OBSERVATION_SCHEMA_VERSION = "processed_camera_observation_manifest.v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "processed_observation_source_manifest.v1"
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_HELDOUT_IDS_BYTES = 4 * 1024 * 1024
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class ProcessedObservationDatasetError(ValueError):
    """Stable fail-closed error for processed observation compilation."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def build_processed_observation_dataset_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and digest-bind the untrusted processed-dataset request."""

    request = json.loads(canonical_json(dict(value)))
    allowed = {
        "schema_version",
        "dataset_id",
        "scene_id",
        "source_bundle_digest",
        "source_bundle_size_bytes",
        "source_bundle_uri",
        "license_id",
        "long_transformations_relative_path",
        "declared_heldout_ids_relative_path",
        "independent_transformations_relative_path",
        "source_commit_sha",
        "authority_used",
        "coordinate_frame_declaration",
        "timestamp",
        "processed_observation_dataset_compile_request_digest",
    }
    errors = [f"processed_request_unknown_field:{key}" for key in request if key not in allowed]
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("processed_request_schema_invalid")
    for key in ("dataset_id", "scene_id"):
        if not _IDENTIFIER.fullmatch(str(request.get(key) or "")):
            errors.append(f"processed_request_{key}_invalid")
    if not _is_digest(request.get("source_bundle_digest")):
        errors.append("processed_request_source_bundle_digest_invalid")
    size = request.get("source_bundle_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        errors.append("processed_request_source_bundle_size_invalid")
    source_uri = str(request.get("source_bundle_uri") or "")
    parsed = urlsplit(source_uri)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        errors.append("processed_request_source_bundle_uri_invalid")
    if not str(request.get("license_id") or "").strip():
        errors.append("processed_request_license_id_missing")
    for key in (
        "long_transformations_relative_path",
        "declared_heldout_ids_relative_path",
        "independent_transformations_relative_path",
    ):
        try:
            _safe_relative(request.get(key), field=f"processed_request_{key}")
        except ProcessedObservationDatasetError as exc:
            errors.extend(exc.codes)
    commit = str(request.get("source_commit_sha") or "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        errors.append("processed_request_source_commit_invalid")
    authority = request.get("authority_used")
    if not isinstance(authority, Mapping):
        errors.append("processed_request_authority_invalid")
    elif (
        authority.get("local_processing_allowed") is not True
        or authority.get("external_provider_upload_allowed") is not False
        or not str(authority.get("privacy_scope") or "").strip()
    ):
        errors.append("processed_request_authority_not_fail_closed")
    coordinate_frame = request.get("coordinate_frame_declaration")
    if not isinstance(coordinate_frame, Mapping) or not coordinate_frame:
        errors.append("processed_request_coordinate_frame_invalid")
    if not str(request.get("timestamp") or "").strip():
        errors.append("processed_request_timestamp_missing")
    supplied_digest = request.pop(
        "processed_observation_dataset_compile_request_digest", None
    )
    request["processed_observation_dataset_compile_request_digest"] = canonical_digest(
        request,
        digest_field="processed_observation_dataset_compile_request_digest",
    )
    if supplied_digest is not None and supplied_digest != request[
        "processed_observation_dataset_compile_request_digest"
    ]:
        errors.append("processed_request_digest_mismatch")
    if errors:
        raise ProcessedObservationDatasetError(errors)
    return request


def compile_bound_processed_observation_dataset(
    *,
    source_artifact: Mapping[str, Any],
    source_bundle: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Execute a validated request against explicitly supplied local roots."""

    request = build_processed_observation_dataset_request(source_artifact)
    _verify_source_bundle(
        source_bundle,
        expected_digest=request["source_bundle_digest"],
        expected_size=request["source_bundle_size_bytes"],
    )
    return compile_processed_observation_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        dataset_id=request["dataset_id"],
        scene_id=request["scene_id"],
        source_bundle_digest=request["source_bundle_digest"],
        source_bundle_size_bytes=request["source_bundle_size_bytes"],
        source_bundle_uri=request["source_bundle_uri"],
        license_id=request["license_id"],
        long_transformations_relative_path=request[
            "long_transformations_relative_path"
        ],
        declared_heldout_ids_relative_path=request[
            "declared_heldout_ids_relative_path"
        ],
        independent_transformations_relative_path=request[
            "independent_transformations_relative_path"
        ],
        source_commit_sha=request["source_commit_sha"],
        authority_used=request["authority_used"],
        coordinate_frame_declaration=request["coordinate_frame_declaration"],
        timestamp=request["timestamp"],
    )


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verify_source_bundle(
    value: str | Path,
    *,
    expected_digest: str,
    expected_size: int,
) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ProcessedObservationDatasetError(
            ["processed_source_bundle_missing_or_symlink"]
        )
    path = path.resolve()
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ProcessedObservationDatasetError(
            ["processed_source_bundle_size_mismatch"]
        )
    if _sha256_file(path) != expected_digest:
        raise ProcessedObservationDatasetError(
            ["processed_source_bundle_digest_mismatch"]
        )
    return path


def _safe_relative(value: Any, *, field: str) -> PurePosixPath:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ProcessedObservationDatasetError([f"{field}_path_unsafe"])
    return path


def _path_has_symlink(root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _bound_file(
    root: Path,
    value: Any,
    *,
    field: str,
    maximum_bytes: int | None = None,
) -> tuple[Path, str]:
    relative = _safe_relative(value, field=field)
    unresolved = root / Path(*relative.parts)
    if _path_has_symlink(root, unresolved):
        raise ProcessedObservationDatasetError([f"{field}_missing_or_symlink"])
    path = unresolved.resolve()
    if path != root and root not in path.parents:
        raise ProcessedObservationDatasetError([f"{field}_path_escape"])
    if path.is_symlink() or not path.is_file():
        raise ProcessedObservationDatasetError([f"{field}_missing_or_symlink"])
    size = path.stat().st_size
    if size <= 0 or (maximum_bytes is not None and size > maximum_bytes):
        raise ProcessedObservationDatasetError([f"{field}_size_invalid"])
    return path, _sha256_file(path)


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProcessedObservationDatasetError([f"{field}_json_invalid"]) from exc
    if not isinstance(value, Mapping):
        raise ProcessedObservationDatasetError([f"{field}_json_not_object"])
    return json.loads(canonical_json(dict(value)))


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ProcessedObservationDatasetError(["processed_dataset_immutable_conflict"])
        return normalized
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
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ProcessedObservationDatasetError(
                    ["processed_dataset_immutable_conflict"]
                )
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _materialize(source: Path, destination: Path, *, digest: str) -> None:
    if source.is_symlink() or not source.is_file() or _sha256_file(source) != digest:
        raise ProcessedObservationDatasetError(["processed_source_digest_mismatch"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or _sha256_file(destination) != digest
        ):
            raise ProcessedObservationDatasetError(
                ["processed_materialized_artifact_conflict"]
            )
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if _sha256_file(destination) != digest:
        destination.unlink(missing_ok=True)
        raise ProcessedObservationDatasetError(
            ["processed_materialized_artifact_digest_mismatch"]
        )


def _frame_id(relative_path: str, *, trajectory: str) -> str:
    stem = PurePosixPath(relative_path).stem
    if not stem or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in stem):
        raise ProcessedObservationDatasetError(["processed_frame_id_invalid"])
    return f"{trajectory}:{stem}"


def _camera_binding(row: Mapping[str, Any], *, frame_id: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        width = int(row.get("w"))
        height = int(row.get("h"))
        fx = float(row.get("fl_x"))
        fy = float(row.get("fl_y"))
        cx = float(row.get("cx"))
        cy = float(row.get("cy"))
        transform = np.asarray(row.get("transform_matrix"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProcessedObservationDatasetError(
            [f"processed_camera_binding_invalid:{frame_id}"]
        ) from exc
    if width <= 0 or height <= 0:
        errors.append(f"processed_image_dimensions_invalid:{frame_id}")
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy)):
        errors.append(f"processed_focal_length_invalid:{frame_id}")
    if not all(math.isfinite(value) for value in (cx, cy)) or not (
        0.0 <= cx <= width and 0.0 <= cy <= height
    ):
        errors.append(f"processed_principal_point_invalid:{frame_id}")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        errors.append(f"processed_transform_invalid:{frame_id}")
    else:
        rotation = transform[:3, :3]
        if not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
            errors.append(f"processed_transform_homogeneous_row_invalid:{frame_id}")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            errors.append(f"processed_rotation_not_orthonormal:{frame_id}")
        determinant = float(np.linalg.det(rotation))
        if not math.isfinite(determinant) or abs(determinant - 1.0) > 1e-4:
            errors.append(f"processed_rotation_not_proper:{frame_id}")
    if errors:
        raise ProcessedObservationDatasetError(errors)
    return {
        "rgb_intrinsics": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "T_world_camera": transform.tolist(),
    }


def _trajectory_rows(
    *,
    dataset_root: Path,
    transformations_relative_path: str,
    trajectory: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformations_path, transformations_digest = _bound_file(
        dataset_root,
        transformations_relative_path,
        field=f"{trajectory}_transformations",
        maximum_bytes=_MAX_JSON_BYTES,
    )
    document = _read_json(transformations_path, field=f"{trajectory}_transformations")
    camera_model = str(document.get("camera_model") or "").strip()
    if camera_model not in {"PINHOLE", "OPENCV"}:
        raise ProcessedObservationDatasetError(
            [f"processed_camera_model_unsupported:{trajectory}"]
        )
    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ProcessedObservationDatasetError(
            [f"processed_frames_missing:{trajectory}"]
        )
    base = transformations_path.parent
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping):
            raise ProcessedObservationDatasetError(
                [f"processed_frame_not_object:{trajectory}:{index}"]
            )
        image_relative = _safe_relative(raw.get("file_path"), field="processed_image")
        depth_relative = _safe_relative(raw.get("depth_file_path"), field="processed_depth")
        unresolved_image = base / Path(*image_relative.parts)
        unresolved_depth = base / Path(*depth_relative.parts)
        if _path_has_symlink(dataset_root, unresolved_image) or _path_has_symlink(
            dataset_root, unresolved_depth
        ):
            raise ProcessedObservationDatasetError(
                [f"processed_frame_artifact_missing_or_symlink:{trajectory}:{index}"]
            )
        image = unresolved_image.resolve()
        depth = unresolved_depth.resolve()
        if any(
            path != dataset_root and dataset_root not in path.parents
            for path in (image, depth)
        ):
            raise ProcessedObservationDatasetError(["processed_frame_path_escape"])
        if any(path.is_symlink() or not path.is_file() or path.stat().st_size <= 0 for path in (image, depth)):
            raise ProcessedObservationDatasetError(
                [f"processed_frame_artifact_missing_or_symlink:{trajectory}:{index}"]
            )
        frame_id = _frame_id(image_relative.as_posix(), trajectory=trajectory)
        if frame_id in seen:
            raise ProcessedObservationDatasetError(
                [f"processed_frame_id_duplicate:{frame_id}"]
            )
        seen.add(frame_id)
        camera = _camera_binding(raw, frame_id=frame_id)
        rows.append(
            {
                "frame_id": frame_id,
                "trajectory": trajectory,
                "trajectory_index": index,
                "source_frame_name": image_relative.stem,
                "image_source_path": image,
                "depth_source_path": depth,
                "image_digest": _sha256_file(image),
                "depth_digest": _sha256_file(depth),
                "image_size_bytes": image.stat().st_size,
                "depth_size_bytes": depth.stat().st_size,
                "camera": camera,
            }
        )
    return rows, {
        "relative_path": transformations_path.relative_to(dataset_root).as_posix(),
        "digest": transformations_digest,
        "camera_model": camera_model,
        "frame_count": len(rows),
    }


def _heldout_source_names(dataset_root: Path, relative_path: str) -> tuple[set[str], dict[str, Any]]:
    path, digest = _bound_file(
        dataset_root,
        relative_path,
        field="declared_heldout_ids",
        maximum_bytes=_MAX_HELDOUT_IDS_BYTES,
    )
    try:
        names = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except (OSError, UnicodeDecodeError) as exc:
        raise ProcessedObservationDatasetError(["declared_heldout_ids_invalid"]) from exc
    if not names or any("/" in name or "\\" in name or name in {".", ".."} for name in names):
        raise ProcessedObservationDatasetError(["declared_heldout_ids_invalid"])
    return names, {
        "relative_path": path.relative_to(dataset_root).as_posix(),
        "digest": digest,
        "count": len(names),
    }


def _artifact_reference(path: Path, value: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "digest": "sha256:"
        + hashlib.sha256((canonical_json(dict(value)) + "\n").encode("utf-8")).hexdigest(),
    }


def compile_processed_observation_dataset(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    dataset_id: str,
    scene_id: str,
    source_bundle_digest: str,
    source_bundle_size_bytes: int,
    source_bundle_uri: str,
    license_id: str,
    long_transformations_relative_path: str,
    declared_heldout_ids_relative_path: str,
    independent_transformations_relative_path: str,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    coordinate_frame_declaration: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Freeze processed RGB-D/pose observations into candidate/evaluator lanes."""

    source_root = Path(dataset_root).expanduser()
    output_base = Path(output_root).expanduser()
    if source_root.is_symlink() or output_base.is_symlink():
        raise ProcessedObservationDatasetError(["processed_dataset_root_symlink_forbidden"])
    source_root = source_root.resolve()
    output_base = output_base.resolve()
    errors: list[str] = []
    if not source_root.is_dir():
        errors.append("processed_dataset_root_missing")
    if not dataset_id.strip() or not scene_id.strip():
        errors.append("processed_dataset_identity_missing")
    if not _is_digest(source_bundle_digest) or source_bundle_size_bytes <= 0:
        errors.append("processed_source_bundle_binding_invalid")
    if not source_bundle_uri.strip() or not license_id.strip():
        errors.append("processed_source_rights_binding_missing")
    if len(source_commit_sha) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit_sha
    ):
        errors.append("processed_source_commit_invalid")
    if not isinstance(authority_used, Mapping) or not authority_used:
        errors.append("processed_authority_missing")
    if not isinstance(coordinate_frame_declaration, Mapping) or not coordinate_frame_declaration:
        errors.append("processed_coordinate_frame_declaration_missing")
    if not timestamp.strip():
        errors.append("processed_timestamp_missing")
    if errors:
        raise ProcessedObservationDatasetError(errors)

    long_rows, long_source = _trajectory_rows(
        dataset_root=source_root,
        transformations_relative_path=long_transformations_relative_path,
        trajectory="long",
    )
    independent_rows, independent_source = _trajectory_rows(
        dataset_root=source_root,
        transformations_relative_path=independent_transformations_relative_path,
        trajectory="independent",
    )
    heldout_names, heldout_source = _heldout_source_names(
        source_root, declared_heldout_ids_relative_path
    )
    long_names = {str(row["source_frame_name"]) for row in long_rows}
    if not heldout_names.issubset(long_names):
        raise ProcessedObservationDatasetError(
            ["declared_heldout_ids_not_subset_of_long_trajectory"]
        )

    long_candidate = [row for row in long_rows if row["source_frame_name"] not in heldout_names]
    ranked = sorted(
        long_candidate,
        key=lambda row: hashlib.sha256(
            f"{source_bundle_digest}\0{row['frame_id']}\0{row['image_digest']}".encode()
        ).hexdigest(),
    )
    validation_count = max(1, round(len(ranked) * 0.1)) if len(ranked) > 1 else 0
    validation_ids = {str(row["frame_id"]) for row in ranked[:validation_count]}
    for row in long_rows:
        if row["source_frame_name"] in heldout_names:
            row["split"] = "held_out"
            row["heldout_reason"] = "dataset_declared_long_trajectory_test_view"
        elif row["frame_id"] in validation_ids:
            row["split"] = "validation"
            row["heldout_reason"] = None
        else:
            row["split"] = "training"
            row["heldout_reason"] = None
    for row in independent_rows:
        row["split"] = "held_out"
        row["heldout_reason"] = "independent_evaluation_trajectory"

    all_rows = sorted(
        [*long_rows, *independent_rows],
        key=lambda row: (str(row["trajectory"]), int(row["trajectory_index"])),
    )
    source_binding_digest = canonical_digest(
        {
            "source_bundle_digest": source_bundle_digest,
            "long_transformations": long_source,
            "declared_heldout_ids": heldout_source,
            "independent_transformations": independent_source,
            "frames": [
                {
                    "frame_id": row["frame_id"],
                    "image_digest": row["image_digest"],
                    "depth_digest": row["depth_digest"],
                    "camera": row["camera"],
                    "split": row["split"],
                }
                for row in all_rows
            ],
        }
    )
    configuration = {
        "compiler_version": COMPILER_VERSION,
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "source_bundle_digest": source_bundle_digest,
        "source_binding_digest": source_binding_digest,
        "source_commit_sha": source_commit_sha,
        "authority_digest": canonical_digest(dict(authority_used)),
        "coordinate_frame_declaration_digest": canonical_digest(
            dict(coordinate_frame_declaration)
        ),
        "split_rule": "dataset_declared_test_plus_independent_trajectory_with_digest_ranked_validation_v1",
    }
    configuration_digest = canonical_digest(configuration)
    root = output_base / f"processed_dataset_{configuration_digest[7:23]}"

    split_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    candidate_observations: list[dict[str, Any]] = []
    for row in all_rows:
        split_rows.append(
            {
                "frame_id": row["frame_id"],
                "trajectory": row["trajectory"],
                "trajectory_index": row["trajectory_index"],
                "frame_digest": row["image_digest"],
                "depth_digest": row["depth_digest"],
                "split": row["split"],
            }
        )
        suffix = row["image_source_path"].suffix.lower()
        depth_suffix = row["depth_source_path"].suffix.lower()
        if row["split"] == "held_out":
            image_relative = Path("evaluator_hidden") / "images" / f"{row['trajectory']}_{row['source_frame_name']}{suffix}"
            depth_relative = Path("evaluator_hidden") / "depth" / f"{row['trajectory']}_{row['source_frame_name']}{depth_suffix}"
            _materialize(row["image_source_path"], root / image_relative, digest=row["image_digest"])
            _materialize(row["depth_source_path"], root / depth_relative, digest=row["depth_digest"])
            heldout_rows.append(
                {
                    "frame_id": row["frame_id"],
                    "trajectory": row["trajectory"],
                    "trajectory_index": row["trajectory_index"],
                    "frame_digest": row["image_digest"],
                    "depth_digest": row["depth_digest"],
                    "evaluator_relative_path": image_relative.as_posix(),
                    "evaluator_depth_relative_path": depth_relative.as_posix(),
                    "camera": row["camera"],
                    "heldout_reason": row["heldout_reason"],
                }
            )
            continue
        image_relative = Path("candidate_dataset") / row["split"] / f"{row['source_frame_name']}{suffix}"
        depth_relative = Path("candidate_dataset") / row["split"] / "depth" / f"{row['source_frame_name']}{depth_suffix}"
        _materialize(row["image_source_path"], root / image_relative, digest=row["image_digest"])
        _materialize(row["depth_source_path"], root / depth_relative, digest=row["depth_digest"])
        candidate_row = {
            "frame_id": row["frame_id"],
            "trajectory": row["trajectory"],
            "trajectory_index": row["trajectory_index"],
            "frame_digest": row["image_digest"],
            "depth_digest": row["depth_digest"],
            "split": row["split"],
            "candidate_relative_path": image_relative.as_posix(),
            "candidate_depth_relative_path": depth_relative.as_posix(),
            "image_metadata": {
                "width": row["camera"]["rgb_intrinsics"]["width"],
                "height": row["camera"]["rgb_intrinsics"]["height"],
                "pixel_orientation": "dataset_processed_orientation",
            },
            "quality_signals": {},
        }
        candidate_rows.append(candidate_row)
        candidate_observations.append(
            {
                "observation_id": row["frame_id"],
                "split": row["split"],
                "image_relative_path": image_relative.as_posix(),
                "image_digest": row["image_digest"],
                "depth_relative_path": depth_relative.as_posix(),
                "depth_digest": row["depth_digest"],
                "camera": row["camera"],
            }
        )

    split = {
        "schema_version": PROCESSED_SPLIT_SCHEMA_VERSION,
        "frozen": True,
        "capture_digest": source_bundle_digest,
        "deterministic_configuration_digest": configuration_digest,
        "split_seed_digest": source_binding_digest,
        "assignments": split_rows,
        "candidate_can_change_assignments": False,
        "hidden_heldout_access": "independent_evaluator_only",
    }
    split["split_digest"] = canonical_digest(split, digest_field="split_digest")
    candidate = {
        "schema_version": PROCESSED_CANDIDATE_SCHEMA_VERSION,
        "capture_digest": source_bundle_digest,
        "split_digest": split["split_digest"],
        "training_and_validation_only": True,
        "heldout_pixels_included": False,
        "frames": candidate_rows,
    }
    candidate["candidate_dataset_digest"] = canonical_digest(
        candidate, digest_field="candidate_dataset_digest"
    )
    heldout = {
        "schema_version": PROCESSED_HELDOUT_SCHEMA_VERSION,
        "capture_digest": source_bundle_digest,
        "split_digest": split["split_digest"],
        "access_scope": "independent_evaluator_only",
        "candidate_method_access_allowed": False,
        "frames": heldout_rows,
    }
    heldout["hidden_heldout_digest"] = canonical_digest(
        heldout, digest_field="hidden_heldout_digest"
    )
    observations = {
        "schema_version": CAMERA_OBSERVATION_SCHEMA_VERSION,
        "source_capture_digest": source_bundle_digest,
        "split_digest": split["split_digest"],
        "source_representation": "processed_public_rgbd_pose_sequence",
        "candidate_splits_only": True,
        "hidden_heldout_pixels_included": False,
        "calibration_status": "dataset_provided_not_blueprint_raw_authority",
        "observations": candidate_observations,
    }
    observations["camera_observation_digest"] = canonical_digest(
        observations, digest_field="camera_observation_digest"
    )
    source_manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "source_bundle_uri": source_bundle_uri,
        "source_bundle_digest": source_bundle_digest,
        "source_bundle_size_bytes": source_bundle_size_bytes,
        "license_id": license_id,
        "long_trajectory": long_source,
        "declared_heldout_ids": heldout_source,
        "independent_trajectory": independent_source,
        "source_binding_digest": source_binding_digest,
        "raw_capture_bundle_available": False,
        "original_video_available": False,
        "decoded_video_pts_available": False,
        "encoder_retention_map_available": False,
        "imu_available": False,
        "tracking_relocalization_log_available": False,
    }
    source_manifest["source_manifest_digest"] = canonical_digest(
        source_manifest, digest_field="source_manifest_digest"
    )

    paths = {
        "source": root / "processed_observation_source_manifest.json",
        "split": root / "frozen_split_manifest.json",
        "candidate": root / "candidate_dataset_manifest.json",
        "heldout": root / "evaluator_hidden" / "hidden_heldout_manifest.json",
        "observations": root / "candidate_camera_observation_manifest.json",
    }
    source_manifest = _write_immutable(paths["source"], source_manifest)
    split = _write_immutable(paths["split"], split)
    candidate = _write_immutable(paths["candidate"], candidate)
    heldout = _write_immutable(paths["heldout"], heldout)
    observations = _write_immutable(paths["observations"], observations)
    artifact_references = {
        name: _artifact_reference(path, value, root=output_base)
        for name, path, value in (
            ("processed_observation_source_manifest", paths["source"], source_manifest),
            ("frozen_split_manifest", paths["split"], split),
            ("candidate_dataset_manifest", paths["candidate"], candidate),
            ("hidden_heldout_evaluator_manifest", paths["heldout"], heldout),
            ("candidate_camera_observation_manifest", paths["observations"], observations),
        )
    }
    dataset = {
        "schema_version": PROCESSED_DATASET_SCHEMA_VERSION,
        "stable_run_identity": f"processed-dataset-{configuration_digest[7:31]}",
        "source_capture_identity": f"public-{dataset_id}-{scene_id}",
        "source_capture_digest": source_bundle_digest,
        "original_file_references": [
            {
                "source_uri": source_bundle_uri,
                "digest": source_bundle_digest,
                "size_bytes": source_bundle_size_bytes,
            }
        ],
        "producing_method": COMPILER_VERSION,
        "implementation_version": "1.0.0",
        "container_image_digest": None,
        "source_commit_sha": source_commit_sha,
        "deterministic_configuration": configuration,
        "deterministic_configuration_digest": configuration_digest,
        "input_digests": {
            "source_bundle_digest": source_bundle_digest,
            "source_binding_digest": source_binding_digest,
            "authority_digest": configuration["authority_digest"],
        },
        "output_digests": {
            "candidate_dataset_digest": candidate["candidate_dataset_digest"],
            "hidden_heldout_digest": heldout["hidden_heldout_digest"],
            "camera_observation_digest": observations["camera_observation_digest"],
        },
        "train_heldout_split_digest": split["split_digest"],
        "camera_calibration_binding": {
            "camera_observation_digest": observations["camera_observation_digest"],
            "status": "dataset_provided_not_blueprint_raw_authority",
        },
        "coordinate_frame_declaration": dict(coordinate_frame_declaration),
        "units": "dataset_declared_units",
        "metric_scale_status": "dataset_declared_not_independently_verified",
        "capture_authority_profile": "public_processed_rgbd_pose_sequence",
        "stream_metadata": {
            "long_trajectory_frames": len(long_rows),
            "independent_trajectory_frames": len(independent_rows),
            "candidate_frames": len(candidate_rows),
            "hidden_heldout_frames": len(heldout_rows),
        },
        "provider_runtime_identity": {"provider": "local", "runtime_identity": "python_numpy"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "duration_accounting": "not_measured_for_local_deterministic_compilation",
        "authority_used": dict(authority_used),
        "warnings": [
            "processed_dataset_is_not_raw_capture_authority",
            "camera_poses_intrinsics_depth_are_dataset_provided_not_independently_verified",
            "local_compilation_duration_not_measured",
        ],
        "blockers": [
            "raw_capture_bundle_missing",
            "decoded_video_pts_missing",
            "encoder_retention_map_missing",
            "imu_and_tracking_relocalization_missing",
            "metric_scale_not_independently_verified",
        ],
        "proof_effect": "processed_captured_observation_availability_only",
        "claim_ceiling": "captured_observation_review_from_processed_public_dataset",
        "parent_artifact_or_event": {
            "source_manifest_digest": source_manifest["source_manifest_digest"]
        },
        "timestamp": timestamp,
        "relative_path": root.relative_to(output_base).as_posix(),
        "artifact_references": artifact_references,
        "candidate_dataset_contains_hidden_heldout_pixels": False,
        "candidate_can_modify_split": False,
        "raw_capture_bytes_remain_authoritative": False,
        "source_dataset_bundle_remains_authoritative": True,
        "claim_flags": {
            "processed_captured_observation": True,
            "raw_capture_authority": False,
            "decoded_video_timing": False,
            "metric_scale_verified": False,
            "collision_geometry": False,
            "physics": False,
            "physical_task_success": False,
            "deployment_readiness": False,
            "safety_certification": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
        "dataset_manifest_digest": None,
    }
    dataset["dataset_manifest_digest"] = canonical_digest(
        dataset, digest_field="dataset_manifest_digest"
    )
    return _write_immutable(root / "reconstruction_dataset_manifest.json", dataset)


__all__ = [
    "CAMERA_OBSERVATION_SCHEMA_VERSION",
    "COMPILER_VERSION",
    "PROCESSED_CANDIDATE_SCHEMA_VERSION",
    "PROCESSED_DATASET_SCHEMA_VERSION",
    "PROCESSED_HELDOUT_SCHEMA_VERSION",
    "PROCESSED_SPLIT_SCHEMA_VERSION",
    "ProcessedObservationDatasetError",
    "REQUEST_SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_processed_observation_dataset_request",
    "compile_bound_processed_observation_dataset",
    "compile_processed_observation_dataset",
]
