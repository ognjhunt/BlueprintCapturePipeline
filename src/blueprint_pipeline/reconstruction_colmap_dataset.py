"""Deterministically materialize candidate-only COLMAP text datasets.

The exporter converts already-bound camera-to-world observations into the
world-to-camera convention consumed by COLMAP/3DGRUT.  It never estimates or
refines poses and it cannot read evaluator-held-out pixels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA_VERSION = "colmap_training_dataset_export_request.v1"
RESULT_SCHEMA_VERSION = "colmap_training_dataset_export_result.v1"


class ColmapTrainingDatasetError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


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


def _safe_file(root: Path, relative_path: str, *, label: str) -> Path:
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if (
        not relative_path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "evaluator_hidden" in relative.parts
        or "held_out" in relative.parts
    ):
        raise ColmapTrainingDatasetError([f"{label}_path_unsafe_or_hidden"])
    resolved_root = root.resolve()
    path = (resolved_root / Path(*relative.parts)).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ColmapTrainingDatasetError([f"{label}_path_escape"])
    if not path.is_file() or path.is_symlink():
        raise ColmapTrainingDatasetError([f"{label}_missing_or_symlink"])
    return path


def _write_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ColmapTrainingDatasetError(["colmap_export_immutable_conflict"])
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
                raise ColmapTrainingDatasetError(["colmap_export_immutable_conflict"])
    finally:
        temporary.unlink(missing_ok=True)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return deterministic COLMAP (qw, qx, qy, qz) from a proper rotation."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            values = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            values = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            values = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    norm = math.sqrt(sum(value * value for value in values))
    normalized = tuple(value / norm for value in values)
    return tuple(-value for value in normalized) if normalized[0] < 0.0 else normalized


def _camera_line(observation: Mapping[str, Any], camera_id: int) -> tuple[str, str]:
    camera = observation.get("camera")
    camera = camera if isinstance(camera, Mapping) else observation
    intrinsics = camera.get("rgb_intrinsics")
    matrix = camera.get("T_world_camera")
    if not isinstance(intrinsics, Mapping) or not isinstance(matrix, list):
        raise ColmapTrainingDatasetError(["colmap_camera_binding_missing"])
    blueprint_native_matrix = observation.get("T_world_camera")
    if blueprint_native_matrix is not None and canonical_json(
        blueprint_native_matrix
    ) != canonical_json(matrix):
        raise ColmapTrainingDatasetError(["colmap_camera_projection_pose_mismatch"])
    try:
        transform = np.asarray(matrix, dtype=np.float64)
        rotation_camera_world = transform[:3, :3]
        translation_camera_world = transform[:3, 3]
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError
        if not np.allclose(rotation_camera_world.T @ rotation_camera_world, np.eye(3), atol=1e-4):
            raise ValueError
        rotation_world_camera = rotation_camera_world.T
        translation_world_camera = -(rotation_world_camera @ translation_camera_world)
        quaternion = _rotation_to_quaternion(rotation_world_camera)
        width, height = int(intrinsics["width"]), int(intrinsics["height"])
        fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
        cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ColmapTrainingDatasetError(["colmap_camera_binding_invalid"]) from exc
    if min(width, height) <= 0 or not all(math.isfinite(v) and v > 0 for v in (fx, fy)):
        raise ColmapTrainingDatasetError(["colmap_intrinsics_invalid"])
    camera_row = f"{camera_id} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}"
    pose = " ".join(f"{value:.17g}" for value in (*quaternion, *translation_world_camera))
    return camera_row, pose


def bind_colmap_initialization_surface(
    *,
    source_artifact: Mapping[str, Any],
    surface_compilation_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an accepted observed ARKit surface to a frozen COLMAP request.

    The returned request keeps the same candidate pixels and poses. It changes
    only the initialization-geometry binding and its own digest; generated fill
    or a cross-capture/split/calibration surface is refused.
    """

    request = json.loads(canonical_json(dict(source_artifact)))
    supplied_request_digest = request.get("colmap_training_dataset_export_request_digest")
    if (
        request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or supplied_request_digest
        != canonical_digest(
            request,
            digest_field="colmap_training_dataset_export_request_digest",
        )
    ):
        raise ColmapTrainingDatasetError(["colmap_surface_binding_request_invalid"])
    surface_result = json.loads(canonical_json(dict(surface_compilation_result)))
    result_digest = surface_result.get("arkit_depth_surface_compilation_result_digest")
    if (
        surface_result.get("schema_version")
        != "arkit_depth_surface_compilation_result.v1"
        or result_digest
        != canonical_digest(
            surface_result,
            digest_field="arkit_depth_surface_compilation_result_digest",
        )
        or surface_result.get("status") != "compiled_observed_surface_candidate"
    ):
        raise ColmapTrainingDatasetError(["colmap_surface_binding_result_invalid"])
    surface_asset = surface_result.get("surface_asset")
    calibration = request.get("camera_calibration_manifest")
    calibration_binding = surface_result.get("camera_calibration_binding")
    surface_calibration_digest = None
    if isinstance(calibration_binding, Mapping):
        surface_calibration_digest = calibration_binding.get(
            "calibration_digest"
        ) or calibration_binding.get("digest")
    if (
        not isinstance(surface_asset, Mapping)
        or not _is_digest(surface_asset.get("digest"))
        or not str(surface_asset.get("relative_path") or "")
        or surface_result.get("source_capture_digest") != request.get("source_capture_digest")
        or surface_result.get("train_heldout_split_digest")
        != request.get("frozen_split_digest")
        or not isinstance(calibration, Mapping)
        or surface_calibration_digest != calibration.get("calibration_digest")
        or canonical_json(surface_result.get("coordinate_frame_declaration"))
        != canonical_json(request.get("coordinate_frame_declaration"))
    ):
        raise ColmapTrainingDatasetError(["colmap_surface_binding_lineage_mismatch"])
    if (
        surface_result.get("hidden_heldout_observations_accessed") is not False
        or surface_result.get("generated_fill_used") is not False
        or surface_result.get("raw_arkit_poses_modified") is not False
    ):
        raise ColmapTrainingDatasetError(["colmap_surface_binding_truth_boundary_invalid"])
    prior_request_digest = supplied_request_digest
    request["initialization_surface"] = {
        "relative_path": str(surface_asset["relative_path"]),
        "digest": surface_asset["digest"],
    }
    request["initialization_surface_compilation_result_digest"] = result_digest
    request["parent_colmap_training_dataset_export_request_digest"] = prior_request_digest
    request["blockers"] = sorted(
        blocker
        for blocker in request.get("blockers") or []
        if blocker != "initialization_surface_not_bound"
    )
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request,
        digest_field="colmap_training_dataset_export_request_digest",
    )
    return request


def export_colmap_training_dataset(
    *,
    source_artifact: Mapping[str, Any],
    artifact_root: str | Path,
    output_root: str | Path,
    initialization_artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    request = json.loads(canonical_json(dict(source_artifact)))
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        errors.append("colmap_export_request_schema_invalid")
    for key in ("source_capture_digest", "reconstruction_dataset_digest", "frozen_split_digest"):
        if not _is_digest(request.get(key)):
            errors.append(f"colmap_export_{key}_invalid")
    source_commit = str(request.get("source_commit_sha") or "")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        errors.append("colmap_export_source_commit_invalid")
    observations = request.get("camera_observation_manifest")
    calibration = request.get("camera_calibration_manifest")
    candidate = request.get("candidate_dataset_manifest")
    if not isinstance(observations, Mapping) or not isinstance(candidate, Mapping):
        errors.append("colmap_export_manifests_missing")
    if not isinstance(request.get("authority_used"), Mapping):
        errors.append("colmap_export_authority_missing")
    if not str(request.get("timestamp") or "").strip():
        errors.append("colmap_export_timestamp_missing")
    supplied_digest = request.pop("colmap_training_dataset_export_request_digest", None)
    request_digest = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    request["colmap_training_dataset_export_request_digest"] = request_digest
    if supplied_digest is not None and supplied_digest != request_digest:
        errors.append("colmap_export_request_digest_mismatch")
    if errors:
        raise ColmapTrainingDatasetError(errors)
    assert isinstance(observations, Mapping) and isinstance(candidate, Mapping)
    if observations.get("camera_observation_digest") != canonical_digest(
        observations, digest_field="camera_observation_digest"
    ):
        raise ColmapTrainingDatasetError(["colmap_export_observation_manifest_digest_invalid"])
    if calibration is not None:
        if not isinstance(calibration, Mapping) or calibration.get(
            "calibration_digest"
        ) != canonical_digest(calibration, digest_field="calibration_digest"):
            raise ColmapTrainingDatasetError(["colmap_export_calibration_manifest_invalid"])
        if (
            calibration.get("capture_digest") != request["source_capture_digest"]
            or calibration.get("camera_model") != "PINHOLE"
            or not isinstance(calibration.get("intrinsics"), Mapping)
        ):
            raise ColmapTrainingDatasetError(["colmap_export_calibration_binding_invalid"])
    if (
        observations.get("hidden_heldout_pixels_included") is not False
        or candidate.get("heldout_pixels_included") is not False
    ):
        raise ColmapTrainingDatasetError(["colmap_export_hidden_heldout_exposed"])
    if candidate.get("candidate_dataset_digest") != canonical_digest(
        candidate, digest_field="candidate_dataset_digest"
    ):
        raise ColmapTrainingDatasetError(["colmap_export_candidate_manifest_digest_invalid"])
    observation_capture_digest = observations.get("source_capture_digest") or observations.get(
        "capture_digest"
    )
    if (
        any(
            value != request["source_capture_digest"]
            for value in (
                observation_capture_digest,
                candidate.get("capture_digest"),
            )
        )
        or candidate.get("split_digest") != request["frozen_split_digest"]
    ):
        raise ColmapTrainingDatasetError(["colmap_export_capture_or_split_binding_mismatch"])
    candidate_rows = {str(row.get("frame_id")): row for row in candidate.get("frames", [])}
    observation_rows = observations.get("observations")
    if not isinstance(observation_rows, list) or not observation_rows:
        raise ColmapTrainingDatasetError(["colmap_export_observations_missing"])

    configuration_digest = canonical_digest(
        {"request_digest": request_digest, "exporter": RESULT_SCHEMA_VERSION}
    )
    root = Path(output_root).resolve() / f"colmap_dataset_{configuration_digest[7:23]}"
    image_digests: list[dict[str, str]] = []
    observation_ids: list[str] = []
    camera_lines = ["# Camera list with one line of data per camera:"]
    image_lines = ["# Image list with two lines of data per image:"]
    source_root = Path(artifact_root).resolve()
    for image_id, observation in enumerate(observation_rows, start=1):
        if not isinstance(observation, Mapping):
            raise ColmapTrainingDatasetError(["colmap_export_observation_invalid"])
        frame_id = str(observation.get("observation_id") or "")
        if not frame_id or frame_id in observation_ids:
            raise ColmapTrainingDatasetError(["colmap_export_observation_id_invalid_or_duplicate"])
        candidate_row = candidate_rows.get(frame_id)
        split = observation.get("split")
        if not isinstance(candidate_row, Mapping) or split not in {"training", "validation"}:
            raise ColmapTrainingDatasetError(["colmap_export_observation_not_candidate"])
        if candidate_row.get("split") != split or candidate_row.get(
            "frame_digest"
        ) != observation.get("image_digest"):
            raise ColmapTrainingDatasetError(["colmap_export_observation_manifest_mismatch"])
        if calibration is not None:
            camera = observation.get("camera")
            if (
                observation.get("calibration_digest") != calibration.get("calibration_digest")
                or not isinstance(camera, Mapping)
                or canonical_json(camera.get("rgb_intrinsics"))
                != canonical_json(calibration.get("intrinsics"))
            ):
                raise ColmapTrainingDatasetError(
                    ["colmap_export_observation_calibration_projection_mismatch"]
                )
        source = _safe_file(
            source_root, str(observation.get("image_relative_path") or ""), label="image"
        )
        if _sha256_file(source) != observation.get("image_digest"):
            raise ColmapTrainingDatasetError(["colmap_export_image_digest_mismatch"])
        suffix = source.suffix.lower()
        name = f"{image_id:06d}_{frame_id}{suffix}"
        digest = _write_immutable(root / "images" / name, source.read_bytes())
        image_digests.append({"artifact_id": name, "digest": digest})
        observation_ids.append(frame_id)
        camera_row, pose = _camera_line(observation, image_id)
        camera_lines.append(camera_row)
        image_lines.extend([f"{image_id} {pose} {image_id} {name}", ""])

    surface_ref = request.get("initialization_surface")
    point_lines = ["# 3D point list with one line of data per point:"]
    initialization_digest = None
    if isinstance(surface_ref, Mapping):
        surface_root = (
            Path(initialization_artifact_root).resolve()
            if initialization_artifact_root is not None
            else source_root
        )
        surface_path = _safe_file(
            surface_root,
            str(surface_ref.get("relative_path") or ""),
            label="initialization_surface",
        )
        initialization_digest = _sha256_file(surface_path)
        if initialization_digest != surface_ref.get("digest"):
            raise ColmapTrainingDatasetError(["colmap_export_initialization_digest_mismatch"])
        surface = json.loads(surface_path.read_text(encoding="utf-8"))
        if (
            surface.get("source_capture_digest") != request["source_capture_digest"]
            or surface.get("train_heldout_split_digest") != request["frozen_split_digest"]
            or surface.get("generated_fill_used") is not False
        ):
            raise ColmapTrainingDatasetError(["colmap_export_initialization_lineage_invalid"])
        maximum = int(request.get("maximum_initialization_points") or 100000)
        vertices = surface.get("vertices") if isinstance(surface.get("vertices"), list) else []
        stride = max(1, math.ceil(len(vertices) / maximum)) if vertices else 1
        for point_id, vertex in enumerate(vertices[::stride], start=1):
            position = vertex.get("position_m") if isinstance(vertex, Mapping) else None
            if not isinstance(position, list) or len(position) != 3:
                raise ColmapTrainingDatasetError(["colmap_export_initialization_point_invalid"])
            xyz = [float(value) for value in position]
            if not all(math.isfinite(value) for value in xyz):
                raise ColmapTrainingDatasetError(["colmap_export_initialization_point_nonfinite"])
            point_lines.append(
                f"{point_id} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} 128 128 128 0"
            )

    artifacts = {}
    for name, lines in (
        ("cameras.txt", camera_lines),
        ("images.txt", image_lines),
        ("points3D.txt", point_lines),
    ):
        artifacts[name] = _write_immutable(
            root / "sparse" / "0" / name, ("\n".join(lines) + "\n").encode("utf-8")
        )
    dataset_digest = canonical_digest(
        {"images": image_digests, "sparse": artifacts, "request_digest": request_digest}
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "stable_run_identity": request.get("stable_run_identity"),
        "status": "exported_candidate_only_colmap_text_dataset",
        "source_capture_digest": request["source_capture_digest"],
        "source_commit_sha": source_commit,
        "reconstruction_dataset_digest": request["reconstruction_dataset_digest"],
        "frozen_split_digest": request["frozen_split_digest"],
        "camera_observation_digest": observations.get("camera_observation_digest"),
        "initialization_surface_digest": initialization_digest,
        "colmap_training_dataset_digest": dataset_digest,
        "producing_method": "blueprint.candidate_observations_to_colmap_text",
        "implementation_version": "1.0.0",
        "container_image_digest": None,
        "deterministic_configuration_digest": configuration_digest,
        "input_digests": [
            request["reconstruction_dataset_digest"],
            request["frozen_split_digest"],
            observations.get("camera_observation_digest"),
            candidate.get("candidate_dataset_digest"),
            initialization_digest,
        ],
        "output_digests": [dataset_digest, *artifacts.values()],
        "camera_calibration_binding": {
            "camera_observation_digest": observations.get("camera_observation_digest")
        },
        "coordinate_frame_declaration": dict(request.get("coordinate_frame_declaration") or {}),
        "units": request.get("units"),
        "metric_scale_status": request.get("metric_scale_status"),
        "provider_runtime_identity": {"provider": "local", "runtime": "python_numpy"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": dict(request["authority_used"]),
        "relative_path": root.relative_to(Path(output_root).resolve()).as_posix(),
        "image_count": len(image_digests),
        "observation_ids": observation_ids,
        "rejected_observation_ids": [],
        "initialization_point_count": len(point_lines) - 1,
        "hidden_heldout_pixels_included": False,
        "raw_input_poses_modified": False,
        "pose_refinement_executed": False,
        "trainer_self_grading_permitted": False,
        "proof_effect": "trainer_input_materialization_only",
        "claim_ceiling": "reconstruction_training_request",
        "warnings": ["colmap_text_export_does_not_qualify_poses_scale_or_geometry"],
        "blockers": list(request.get("blockers") or []),
        "parent_artifact_or_event": {
            "request_digest": request_digest,
            "reconstruction_dataset_digest": request["reconstruction_dataset_digest"],
        },
        "timestamp": request["timestamp"],
    }
    result["colmap_training_dataset_export_result_digest"] = canonical_digest(
        result, digest_field="colmap_training_dataset_export_result_digest"
    )
    _write_immutable(
        root / "colmap_training_dataset_export_result.json",
        (canonical_json(result) + "\n").encode("utf-8"),
    )
    return result


__all__ = [
    "bind_colmap_initialization_surface",
    "ColmapTrainingDatasetError",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "export_colmap_training_dataset",
]
