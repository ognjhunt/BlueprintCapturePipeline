"""Deterministic retained-frame selection and frozen reconstruction splits.

This module is deliberately downstream of media decoding.  It accepts only
decoded observations already bound to one retained source video, freezes the
selection and train/validation/held-out assignment, and materializes two
separate views:

* ``candidate_dataset`` contains training and validation pixels only.
* ``evaluator_hidden`` contains held-out pixels for an independent evaluator.

The source video remains authoritative and complete.  The compiled PNGs and
split manifests are derived, replaceable artifacts and never establish camera
calibration, metric scale, geometry, collision, physics, or physical success.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


DATASET_SCHEMA_VERSION = "reconstruction_dataset_manifest.v1"
SELECTION_SCHEMA_VERSION = "retained_frame_selection_manifest.v1"
SPLIT_SCHEMA_VERSION = "frozen_reconstruction_split_manifest.v1"
CANDIDATE_SCHEMA_VERSION = "candidate_reconstruction_dataset_manifest.v1"
HELDOUT_SCHEMA_VERSION = "hidden_heldout_evaluator_manifest.v1"
COMPILER_VERSION = "deterministic_retained_frame_compiler.v1"


class ReconstructionFrameDatasetError(ValueError):
    """Stable fail-closed error for reconstruction frame compilation."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _safe_relative(value: Any) -> str:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReconstructionFrameDatasetError(["frame_artifact_relative_path_unsafe"])
    return path.as_posix()


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReconstructionFrameDatasetError(["immutable_dataset_artifact_invalid"]) from exc
        if canonical_json(existing) != canonical_json(normalized):
            raise ReconstructionFrameDatasetError(["immutable_dataset_artifact_conflict"])
        return dict(existing)
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
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(existing) != canonical_json(normalized):
                raise ReconstructionFrameDatasetError(["immutable_dataset_artifact_conflict"])
            return dict(existing)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _materialize_bound_frame(source: Path, destination: Path, expected_digest: str) -> None:
    if source.is_symlink() or not source.is_file() or _sha256_file(source) != expected_digest:
        raise ReconstructionFrameDatasetError(["selected_frame_source_digest_mismatch"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != expected_digest:
            raise ReconstructionFrameDatasetError(["materialized_split_frame_conflict"])
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if _sha256_file(destination) != expected_digest:
        destination.unlink(missing_ok=True)
        raise ReconstructionFrameDatasetError(["materialized_split_frame_digest_mismatch"])


def _normalized_frames(
    *, artifact_root: Path, selected_frames: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    frame_ids: set[str] = set()
    frame_indexes: set[tuple[str, int]] = set()
    presentation_times: set[tuple[str, float]] = set()
    group_camera_bindings: set[tuple[str, str]] = set()
    for ordinal, raw in enumerate(selected_frames):
        frame_id = str(raw.get("frame_id") or "").strip()
        try:
            frame_index = int(raw.get("decoded_frame_index"))
            presentation_time = float(raw.get("t_video_sec"))
        except (TypeError, ValueError):
            errors.append(f"selected_frame_numeric_binding_invalid:{ordinal}")
            continue
        digest = str(raw.get("digest") or "")
        source_camera_identity = str(raw.get("source_camera_identity") or "").strip()
        observation_group_id = str(raw.get("observation_group_id") or "").strip()
        grouped = bool(source_camera_identity or observation_group_id)
        if grouped and not (source_camera_identity and observation_group_id):
            errors.append(f"selected_frame_group_binding_incomplete:{ordinal}")
        group_camera_binding = (observation_group_id, source_camera_identity)
        if grouped and group_camera_binding in group_camera_bindings:
            errors.append(f"selected_frame_group_camera_duplicate:{ordinal}")
        elif grouped:
            group_camera_bindings.add(group_camera_binding)
        uniqueness_camera = source_camera_identity or "__single_camera__"
        try:
            relative_path = _safe_relative(raw.get("artifact_relative_path"))
        except ReconstructionFrameDatasetError as exc:
            errors.extend(exc.codes)
            continue
        if not frame_id or frame_id in frame_ids:
            errors.append(f"selected_frame_id_missing_or_duplicate:{ordinal}")
        frame_index_key = (uniqueness_camera, frame_index)
        presentation_time_key = (uniqueness_camera, presentation_time)
        if frame_index < 0 or frame_index_key in frame_indexes:
            errors.append(f"selected_frame_index_invalid_or_duplicate:{ordinal}")
        if presentation_time < 0 or presentation_time_key in presentation_times:
            errors.append(f"selected_frame_pts_invalid_or_duplicate:{ordinal}")
        if not _is_digest(digest):
            errors.append(f"selected_frame_digest_invalid:{ordinal}")
        source = (artifact_root / relative_path).resolve()
        if artifact_root != source and artifact_root not in source.parents:
            errors.append(f"selected_frame_path_escape:{ordinal}")
        elif source.is_symlink() or not source.is_file() or _sha256_file(source) != digest:
            errors.append(f"selected_frame_source_invalid:{ordinal}")
        frame_ids.add(frame_id)
        frame_indexes.add(frame_index_key)
        presentation_times.add(presentation_time_key)
        row = {
                "frame_id": frame_id,
                "decoded_frame_index": frame_index,
                "t_video_sec": round(presentation_time, 9),
                "source_pts_seconds": raw.get("source_pts_seconds"),
                "source_dts_seconds": raw.get("source_dts_seconds"),
                "duration_seconds": raw.get("duration_seconds"),
                "key_frame": bool(raw.get("key_frame")),
                "artifact_relative_path": relative_path,
                "digest": digest,
                "image_metadata": dict(raw.get("image_metadata") or {}),
                "quality_signals": dict(raw.get("quality_signals") or {}),
            }
        if grouped:
            row["source_camera_identity"] = source_camera_identity
            row["observation_group_id"] = observation_group_id
        rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("source_camera_identity") or ""),
            row["decoded_frame_index"],
            row["frame_id"],
        )
    )
    rows_by_camera: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_camera.setdefault(
            str(row.get("source_camera_identity") or "__single_camera__"), []
        ).append(row)
    if any(
        camera_rows[index]["t_video_sec"] >= camera_rows[index + 1]["t_video_sec"]
        for camera_rows in rows_by_camera.values()
        for index in range(len(camera_rows) - 1)
    ):
        errors.append("selected_frame_pts_not_strictly_increasing")
    if not rows:
        errors.append("selected_frames_missing")
    if errors:
        raise ReconstructionFrameDatasetError(errors)
    return rows


def _split_assignments(
    frames: Sequence[Mapping[str, Any]], *, split_seed_digest: str
) -> tuple[dict[str, str], list[str]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    grouped_mode = any(row.get("observation_group_id") for row in frames)
    for row in frames:
        group_id = str(row.get("observation_group_id") or row["frame_id"])
        groups.setdefault(group_id, []).append(row)
    count = len(groups)
    if count < 3:
        return ({str(row["frame_id"]): "training" for row in frames}, [
            (
                "insufficient_selected_observation_groups_for_disjoint_"
                "validation_and_hidden_heldout"
                if grouped_mode
                else "insufficient_selected_frames_for_disjoint_validation_and_hidden_heldout"
            )
        ])
    heldout_count = max(1, round(count * 0.2))
    validation_count = max(1, round(count * 0.1))
    while heldout_count + validation_count >= count:
        if heldout_count > 1:
            heldout_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            break
    ranked: list[tuple[str, list[Mapping[str, Any]], str]] = []
    for group_id, group_rows in groups.items():
        if len(group_rows) == 1 and not group_rows[0].get("observation_group_id"):
            rank_digest = str(group_rows[0]["digest"])
        else:
            rank_digest = canonical_digest(
                {
                    "group_id": group_id,
                    "frames": sorted(
                        (
                            {
                                "frame_id": row["frame_id"],
                                "source_camera_identity": row.get(
                                    "source_camera_identity"
                                ),
                                "digest": row["digest"],
                            }
                            for row in group_rows
                        ),
                        key=lambda row: row["frame_id"],
                    ),
                }
            )
        rank = hashlib.sha256(
            f"{split_seed_digest}\0{group_id}\0{rank_digest}".encode("utf-8")
        ).hexdigest()
        ranked.append((group_id, group_rows, rank))
    ranked.sort(key=lambda row: row[2])
    group_assignments: dict[str, str] = {}
    for group_id, _rows, _rank in ranked[:heldout_count]:
        group_assignments[group_id] = "held_out"
    for group_id, _rows, _rank in ranked[
        heldout_count : heldout_count + validation_count
    ]:
        group_assignments[group_id] = "validation"
    for group_id, _rows, _rank in ranked[heldout_count + validation_count :]:
        group_assignments[group_id] = "training"
    assignments = {
        str(row["frame_id"]): group_assignments[
            str(row.get("observation_group_id") or row["frame_id"])
        ]
        for row in frames
    }
    return assignments, []


def _artifact_reference(path: Path, value: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "digest": "sha256:" + hashlib.sha256(
            (canonical_json(dict(value)) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def _validated_existing_dataset(
    path: Path, *, root: Path, configuration_digest: str
) -> dict[str, Any]:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionFrameDatasetError(["existing_dataset_manifest_invalid"]) from exc
    errors: list[str] = []
    if not isinstance(existing, Mapping):
        raise ReconstructionFrameDatasetError(["existing_dataset_manifest_not_object"])
    existing = dict(existing)
    if existing.get("deterministic_configuration_digest") != configuration_digest:
        errors.append("existing_dataset_configuration_mismatch")
    if existing.get("dataset_manifest_digest") != canonical_digest(
        existing, digest_field="dataset_manifest_digest"
    ):
        errors.append("existing_dataset_manifest_digest_mismatch")
    references = existing.get("artifact_references")
    if not isinstance(references, Mapping):
        errors.append("existing_dataset_artifact_references_missing")
        references = {}
    for name in (
        "retained_frame_selection_manifest",
        "frozen_split_manifest",
        "candidate_dataset_manifest",
        "hidden_heldout_evaluator_manifest",
    ):
        reference = references.get(name)
        if not isinstance(reference, Mapping) or not _is_digest(reference.get("digest")):
            errors.append(f"existing_dataset_artifact_reference_invalid:{name}")
            continue
        try:
            relative_path = _safe_relative(reference.get("relative_path"))
        except ReconstructionFrameDatasetError:
            errors.append(f"existing_dataset_artifact_reference_unsafe:{name}")
            continue
        artifact = (root / relative_path).resolve()
        if root != artifact and root not in artifact.parents:
            errors.append(f"existing_dataset_artifact_reference_escape:{name}")
        elif artifact.is_symlink() or not artifact.is_file() or _sha256_file(artifact) != reference.get(
            "digest"
        ):
            errors.append(f"existing_dataset_artifact_digest_mismatch:{name}")
    if errors:
        raise ReconstructionFrameDatasetError(errors)
    return existing


def compile_frozen_frame_dataset(
    *,
    artifact_root: str | Path,
    intake_id: str,
    capture_digest: str,
    capture_authority_profile: str,
    source_video_relative_path: str,
    source_video_digest: str,
    decoded_frame_count: int,
    selected_frames: Sequence[Mapping[str, Any]],
    stream_metadata: Mapping[str, Any],
    runtime_identity: str,
    runtime_digest: str,
    implementation_digest: str,
    source_commit_sha: str,
    rights_and_retention: Mapping[str, Any],
    selection_rule: str = "evenly_spaced_actual_decoded_pts_with_endpoints_v1",
    parent_artifact: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
    camera_calibration_binding: Mapping[str, Any] | None = None,
    coordinate_frame_declaration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one content-bound dataset and isolate its hidden held-out pixels."""

    root = Path(artifact_root).expanduser().resolve()
    errors: list[str] = []
    try:
        normalized_source_video_relative_path = _safe_relative(source_video_relative_path)
    except ReconstructionFrameDatasetError:
        normalized_source_video_relative_path = ""
        errors.append("dataset_source_video_relative_path_unsafe")
    if not str(intake_id).strip():
        errors.append("dataset_intake_id_missing")
    for label, value in (
        ("capture_digest", capture_digest),
        ("source_video_digest", source_video_digest),
        ("runtime_digest", runtime_digest),
        ("implementation_digest", implementation_digest),
    ):
        if not _is_digest(value):
            errors.append(f"dataset_{label}_invalid")
    if decoded_frame_count <= 0:
        errors.append("dataset_decoded_frame_count_invalid")
    if not str(runtime_identity).strip():
        errors.append("dataset_runtime_identity_missing")
    if not str(selection_rule).strip():
        errors.append("dataset_selection_rule_missing")
    if camera_calibration_binding is not None and not isinstance(
        camera_calibration_binding, Mapping
    ):
        errors.append("dataset_camera_calibration_binding_invalid")
    if coordinate_frame_declaration is not None and not isinstance(
        coordinate_frame_declaration, Mapping
    ):
        errors.append("dataset_coordinate_frame_declaration_invalid")
    if len(source_commit_sha) != 40 or any(character not in "0123456789abcdef" for character in source_commit_sha):
        errors.append("dataset_source_commit_sha_invalid")
    if errors:
        raise ReconstructionFrameDatasetError(errors)
    frames = _normalized_frames(artifact_root=root, selected_frames=selected_frames)
    selected_frame_binding_digest = canonical_digest({"frames": frames})
    stream_metadata_digest = canonical_digest(dict(stream_metadata))
    authority_digest = canonical_digest(dict(rights_and_retention))
    parent_artifact_digest = canonical_digest(dict(parent_artifact or {}))
    camera_calibration_binding_digest = (
        canonical_digest(dict(camera_calibration_binding))
        if camera_calibration_binding is not None
        else None
    )
    coordinate_frame_declaration_digest = (
        canonical_digest(dict(coordinate_frame_declaration))
        if coordinate_frame_declaration is not None
        else None
    )
    grouped_observations = any(row.get("observation_group_id") for row in frames)
    config = {
        "compiler_version": COMPILER_VERSION,
        "source_capture_identity": intake_id,
        "source_capture_digest": capture_digest,
        "capture_authority_profile": capture_authority_profile,
        "selection_rule": str(selection_rule),
        "split_rule": (
            "digest_ranked_disjoint_observation_group_train_validation_hidden_heldout_v1"
            if grouped_observations
            else "digest_ranked_disjoint_train_validation_hidden_heldout_v1"
        ),
        "grouped_observation_splits": grouped_observations,
        "decoded_frame_count": decoded_frame_count,
        "selected_frame_count": len(frames),
        "source_video_relative_path": normalized_source_video_relative_path,
        "source_video_digest": source_video_digest,
        "selected_frame_binding_digest": selected_frame_binding_digest,
        "stream_metadata_digest": stream_metadata_digest,
        "authority_digest": authority_digest,
        "parent_artifact_digest": parent_artifact_digest,
        "runtime_digest": runtime_digest,
        "implementation_digest": implementation_digest,
        "source_commit_sha": source_commit_sha,
    }
    if camera_calibration_binding_digest is not None:
        config["camera_calibration_binding_digest"] = (
            camera_calibration_binding_digest
        )
    if coordinate_frame_declaration_digest is not None:
        config["coordinate_frame_declaration_digest"] = (
            coordinate_frame_declaration_digest
        )
    configuration_digest = canonical_digest(config)
    dataset_root = root / f"frozen_dataset_{configuration_digest[7:23]}"
    existing_path = dataset_root / "reconstruction_dataset_manifest.json"
    if existing_path.is_file():
        return _validated_existing_dataset(
            existing_path, root=root, configuration_digest=configuration_digest
        )
    split_seed = canonical_digest(
        {
            "capture_digest": capture_digest,
            "configuration_digest": configuration_digest,
            "selected_frame_binding_digest": selected_frame_binding_digest,
        }
    )
    assignments, blockers = _split_assignments(frames, split_seed_digest=split_seed)
    split_rows = [
        {
            "frame_id": row["frame_id"],
            "decoded_frame_index": row["decoded_frame_index"],
            "t_video_sec": row["t_video_sec"],
            "frame_digest": row["digest"],
            "split": assignments[row["frame_id"]],
        }
        | (
            {
                "source_camera_identity": row["source_camera_identity"],
                "observation_group_id": row["observation_group_id"],
            }
            if row.get("observation_group_id")
            else {}
        )
        for row in frames
    ]
    split_binding = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "frozen": True,
        "capture_digest": capture_digest,
        "deterministic_configuration_digest": configuration_digest,
        "split_seed_digest": split_seed,
        "assignments": split_rows,
        "candidate_can_change_assignments": False,
        "hidden_heldout_access": "independent_evaluator_only",
    }
    split_binding["split_digest"] = canonical_digest(split_binding, digest_field="split_digest")
    candidate_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    for row in frames:
        split = assignments[row["frame_id"]]
        source = root / row["artifact_relative_path"]
        if split == "held_out":
            relative = Path("evaluator_hidden") / "held_out" / f"{row['frame_id']}.png"
            _materialize_bound_frame(source, dataset_root / relative, row["digest"])
            heldout_rows.append(
                {
                    "frame_id": row["frame_id"],
                    "decoded_frame_index": row["decoded_frame_index"],
                    "t_video_sec": row["t_video_sec"],
                    "frame_digest": row["digest"],
                    "evaluator_relative_path": relative.as_posix(),
                }
                | (
                    {
                        "source_camera_identity": row["source_camera_identity"],
                        "observation_group_id": row["observation_group_id"],
                    }
                    if row.get("observation_group_id")
                    else {}
                )
            )
        else:
            relative = Path("candidate_dataset") / split / f"{row['frame_id']}.png"
            _materialize_bound_frame(source, dataset_root / relative, row["digest"])
            candidate_rows.append(
                {
                    "frame_id": row["frame_id"],
                    "decoded_frame_index": row["decoded_frame_index"],
                    "t_video_sec": row["t_video_sec"],
                    "frame_digest": row["digest"],
                    "split": split,
                    "candidate_relative_path": relative.as_posix(),
                    "image_metadata": row["image_metadata"],
                    "quality_signals": row["quality_signals"],
                }
                | (
                    {
                        "source_camera_identity": row["source_camera_identity"],
                        "observation_group_id": row["observation_group_id"],
                    }
                    if row.get("observation_group_id")
                    else {}
                )
            )
    compiled_at = timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "source_video_digest": source_video_digest,
        "decoded_frame_count": decoded_frame_count,
        "selected_frame_count": len(frames),
        "selection_rule": config["selection_rule"],
        "complete_retained_source_preserved": True,
        "source_video_relative_path": normalized_source_video_relative_path,
        "frames": frames,
    }
    selection["selection_digest"] = canonical_digest(selection, digest_field="selection_digest")
    candidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "split_digest": split_binding["split_digest"],
        "training_and_validation_only": True,
        "heldout_pixels_included": False,
        "frames": candidate_rows,
    }
    candidate["candidate_dataset_digest"] = canonical_digest(
        candidate, digest_field="candidate_dataset_digest"
    )
    heldout = {
        "schema_version": HELDOUT_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "split_digest": split_binding["split_digest"],
        "access_scope": "independent_evaluator_only",
        "candidate_method_access_allowed": False,
        "frames": heldout_rows,
    }
    heldout["hidden_heldout_digest"] = canonical_digest(
        heldout, digest_field="hidden_heldout_digest"
    )
    paths = {
        "selection": dataset_root / "retained_frame_selection_manifest.json",
        "split": dataset_root / "frozen_split_manifest.json",
        "candidate": dataset_root / "candidate_dataset_manifest.json",
        "heldout": dataset_root / "evaluator_hidden" / "hidden_heldout_manifest.json",
    }
    selection = _write_immutable(paths["selection"], selection)
    split_binding = _write_immutable(paths["split"], split_binding)
    candidate = _write_immutable(paths["candidate"], candidate)
    heldout = _write_immutable(paths["heldout"], heldout)
    artifact_refs = {
        name: _artifact_reference(path, value, root=root)
        for name, path, value in (
            ("retained_frame_selection_manifest", paths["selection"], selection),
            ("frozen_split_manifest", paths["split"], split_binding),
            ("candidate_dataset_manifest", paths["candidate"], candidate),
            ("hidden_heldout_evaluator_manifest", paths["heldout"], heldout),
        )
    }
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "stable_run_identity": f"frame-dataset-{configuration_digest[7:31]}",
        "source_capture_identity": intake_id,
        "source_capture_digest": capture_digest,
        "original_file_references": [
            {
                "relative_path": normalized_source_video_relative_path,
                "digest": source_video_digest,
            }
        ],
        "producing_method": COMPILER_VERSION,
        "implementation_version": implementation_digest,
        "container_image_digest": None,
        "source_commit_sha": source_commit_sha,
        "deterministic_configuration": config,
        "deterministic_configuration_digest": configuration_digest,
        "input_digests": {
            "source_video_digest": source_video_digest,
            "selected_frame_binding_digest": selected_frame_binding_digest,
            "stream_metadata_digest": stream_metadata_digest,
            "authority_digest": authority_digest,
            "parent_artifact_digest": parent_artifact_digest,
        }
        | (
            {
                "camera_calibration_binding_digest": camera_calibration_binding_digest
            }
            if camera_calibration_binding_digest is not None
            else {}
        )
        | (
            {
                "coordinate_frame_declaration_digest": coordinate_frame_declaration_digest
            }
            if coordinate_frame_declaration_digest is not None
            else {}
        ),
        "output_digests": {
            "candidate_dataset_digest": candidate["candidate_dataset_digest"],
            "hidden_heldout_digest": heldout["hidden_heldout_digest"],
        },
        "train_heldout_split_digest": split_binding["split_digest"],
        "camera_calibration_binding": (
            dict(camera_calibration_binding)
            if camera_calibration_binding is not None
            else None
        ),
        "coordinate_frame_declaration": (
            dict(coordinate_frame_declaration)
            if coordinate_frame_declaration is not None
            else {"status": "not_established_by_frame_dataset_compiler"}
        ),
        "units": "source_pixels_and_seconds",
        "metric_scale_status": "not_established",
        "capture_authority_profile": capture_authority_profile,
        "stream_metadata": dict(stream_metadata),
        "provider_runtime_identity": {
            "provider": "local",
            "runtime_identity": runtime_identity,
            "runtime_digest": runtime_digest,
        },
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "duration_accounting": "not_measured_for_local_deterministic_compilation",
        "authority_used": dict(rights_and_retention),
        "warnings": blockers + ["local_compilation_duration_not_measured"],
        "blockers": blockers,
        "proof_effect": "decoded_observation_availability_only",
        "claim_ceiling": "decoded_observation_availability",
        "parent_artifact_or_event": dict(parent_artifact or {}),
        "timestamp": compiled_at,
        "artifact_references": artifact_refs,
        "candidate_dataset_contains_hidden_heldout_pixels": False,
        "candidate_can_modify_split": False,
        "raw_capture_bytes_remain_authoritative": True,
        "dataset_manifest_digest": None,
    }
    dataset["dataset_manifest_digest"] = canonical_digest(
        dataset, digest_field="dataset_manifest_digest"
    )
    return _write_immutable(existing_path, dataset)


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "COMPILER_VERSION",
    "DATASET_SCHEMA_VERSION",
    "HELDOUT_SCHEMA_VERSION",
    "ReconstructionFrameDatasetError",
    "SELECTION_SCHEMA_VERSION",
    "SPLIT_SCHEMA_VERSION",
    "compile_frozen_frame_dataset",
]
