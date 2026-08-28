"""Seal and reopen a non-qualifying scene-configuration retry checkpoint.

The production lane deliberately requires all six stages to execute inside one
parent provider run.  This checkpoint is a separate diagnostic artifact: it
retains the already-paid provider render and semantic-teacher candidates so a
later diagnostic allocation can exercise the downstream implementation without
claiming (or publishing) a completed configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_disclosure import MATERIALIZED_STATUS
from .task_evaluation_scene_configuration_render_inputs import (
    RESULT_SCHEMA_VERSION as RENDER_RESULT_SCHEMA_VERSION,
)


SCHEMA_VERSION = "task_evaluation_scene_configuration_diagnostic_checkpoint.v1"
STATUS = "render_and_semantic_teacher_completed_diagnostic_checkpoint"
_SEMANTIC_REQUEST_SCHEMA = "semantic_teacher_image_edit_runtime_request.v1"
_SEMANTIC_RESULT_SCHEMA = "semantic_teacher_image_edit_runtime_result.v1"
_SEMANTIC_RECEIPT_SCHEMA = "public_scene_whole_frame_semantic_teacher_candidates.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "api_key_value",
        "authorization_header",
        "bearer_token",
        "credential",
        "credential_value",
        "openai_api_key",
        "password",
        "secret",
        "secret_value",
        "token",
    }
)


class TaskEvaluationSceneConfigurationDiagnosticCheckpointError(RuntimeError):
    """A diagnostic checkpoint was incomplete, mutable, or mismatched."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(code)
    return dict(value)


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _SECRET_FIELDS
            or _contains_secret_material(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_material(child) for child in value)
    return isinstance(value, str) and value.startswith(("sk-", "Bearer "))


def _bound_file(value: Any, *, code: str, digest_key: str = "digest") -> Path:
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(code)
    unresolved = Path(str(value.get("path") or "")).expanduser()
    path = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get(digest_key)
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(code)
    return path


def _copy_inventory_file(
    *, source: Path, root: Path, relative: str, role: str
) -> dict[str, Any]:
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_path_invalid"
        )
    destination = root.joinpath(*relative_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_duplicate_path"
        )
    shutil.copyfile(source, destination)
    source_mode = stat.S_IMODE(source.stat().st_mode)
    os.chmod(destination, source_mode)
    if destination.stat().st_size != source.stat().st_size or _sha256(
        destination
    ) != _sha256(source):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_copy_mismatch"
        )
    return {
        "role": role,
        "relative_path": relative_path.as_posix(),
        "digest": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "mode": source_mode,
    }


def _normalized_semantic_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove locations/provenance while retaining scientific bytes and choices."""

    omitted = {
        "path",
        "relative_path",
        "source_commit_sha",
        "source_packet_digest",
        "request_digest",
    }

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): normalize(child)
                for key, child in item.items()
                if str(key) not in omitted
            }
        if isinstance(item, list):
            return [normalize(child) for child in item]
        return item

    return normalize(value)


def _portable_render_template(render_inputs: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(render_inputs)))
    source = value.get("source_appearance") or {}
    source.pop("path", None)
    source["checkpoint_role"] = None
    value["source_appearance"] = source
    for field, role in (
        ("camera_calibration", "camera_calibration"),
        ("render_manifest", "render_manifest"),
    ):
        row = value.get(field) or {}
        row.pop("path", None)
        row["checkpoint_role"] = role
        value[field] = row
    for frame in value.get("derived_frames") or []:
        camera_id = str(frame.get("camera_id") or "")
        frame.pop("path", None)
        frame["checkpoint_role"] = f"raw_frame:{camera_id}"
        mask = frame.get("source_object_mask") or {}
        mask.pop("path", None)
        mask["checkpoint_role"] = f"source_object_mask:{camera_id}"
        frame["source_object_mask"] = mask
    cutout = value.get("derived_gaussian_cutout") or {}
    retained = cutout.get("retained_scene_without_source_object") or {}
    retained.pop("path", None)
    retained["checkpoint_role"] = "retained_scene_without_source_object"
    cutout["retained_scene_without_source_object"] = retained
    candidate = cutout.get("source_object_candidate")
    if isinstance(candidate, dict):
        candidate.pop("path", None)
        candidate["checkpoint_role"] = None
    value["derived_gaussian_cutout"] = cutout
    value["source_checkpoint_render_result_digest"] = value.get("result_digest")
    value["result_digest"] = ""
    return value


def _scientific_bindings(
    *, stage_input: Mapping[str, Any], render_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    envelope = stage_input.get("construction_envelope")
    configuration = stage_input.get("configuration")
    stage = stage_input.get("stage")
    renderer_runtime = render_inputs.get("renderer_runtime")
    disclosure = render_inputs.get("disclosure_decision")
    source_appearance = render_inputs.get("source_appearance")
    calibration = render_inputs.get("camera_calibration")
    if not all(
        isinstance(value, Mapping)
        for value in (
            envelope,
            configuration,
            renderer_runtime,
            disclosure,
            source_appearance,
            calibration,
            stage,
        )
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_binding_invalid"
        )
    stage_id = str(stage.get("stage_id") or "")
    configuration_rows = envelope.get("stage_configuration_references")
    stage_sequence = (envelope.get("recipe") or {}).get("stage_sequence")
    if isinstance(stage_sequence, list):
        matching_stage_indexes = [
            index
            for index, row in enumerate(stage_sequence)
            if isinstance(row, Mapping) and row.get("stage_id") == stage_id
        ]
        if (
            not isinstance(configuration_rows, list)
            or len(configuration_rows) != len(stage_sequence)
            or len(matching_stage_indexes) != 1
        ):
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                "scene_configuration_diagnostic_checkpoint_configuration_mismatch"
            )
        stage_index = matching_stage_indexes[0]
        configuration_row = configuration_rows[stage_index]
        row_identity_valid = isinstance(configuration_row, Mapping) and (
            configuration_row.get("stage_id") == stage_id
            or configuration_row.get("contract_path")
            == f"construction.recipe.stage_sequence.{stage_index}.configuration"
        )
    else:
        matching_configuration_rows = (
            [
                row
                for row in configuration_rows
                if isinstance(row, Mapping) and row.get("stage_id") == stage_id
            ]
            if isinstance(configuration_rows, list)
            else []
        )
        row_identity_valid = len(matching_configuration_rows) == 1
        configuration_row = (
            matching_configuration_rows[0]
            if row_identity_valid
            else {}
        )
    if not row_identity_valid:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_configuration_mismatch"
        )
    unresolved_configuration = Path(
        str(
            configuration_row.get("materialized_path")
            or configuration_row.get("path")
            or ""
        )
    ).expanduser()
    configuration_path = unresolved_configuration.resolve()
    materialized_configuration = _read(
        configuration_path,
        code="scene_configuration_diagnostic_checkpoint_configuration_mismatch",
    )
    configuration_digest = str(configuration_row.get("digest") or "")
    if (
        unresolved_configuration.is_symlink()
        or not configuration_path.is_file()
        or configuration_path.stat().st_size != configuration_row.get("size_bytes")
        or _sha256(configuration_path) != configuration_digest
        or stage_input.get("configuration_sha256") != configuration_digest
        or materialized_configuration != dict(configuration)
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_configuration_mismatch"
        )
    source_splat_digest = str(render_inputs.get("source_splat_digest") or "")
    calibration_digest = str(calibration.get("digest") or "")
    if (
        _DIGEST.fullmatch(source_splat_digest) is None
        or _DIGEST.fullmatch(calibration_digest) is None
        or source_appearance.get("digest") != source_splat_digest
        or not isinstance(source_appearance.get("size_bytes"), int)
        or source_appearance.get("size_bytes", 0) <= 0
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_binding_invalid"
        )
    value: dict[str, Any] = {
        "source_splat_digest": source_splat_digest,
        "source_splat_size_bytes": source_appearance["size_bytes"],
        "camera_calibration_digest": calibration_digest,
        "stage_configuration_digest": configuration_digest,
        "renderer_runtime": dict(renderer_runtime),
        "renderer_runtime_digest": canonical_digest(renderer_runtime),
        "disclosure_decision": dict(disclosure),
        "disclosure_decision_digest": canonical_digest(disclosure),
        "binding_digest": "",
    }
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    return value


def _camera_rows(
    *, render_inputs: Mapping[str, Any], calibration_path: Path
) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]]]:
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_calibration_invalid"
        ) from exc
    frames = render_inputs.get("derived_frames")
    if not isinstance(calibration, list) or not isinstance(frames, list) or len(frames) != 8:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_camera_set_invalid"
        )
    cameras: list[dict[str, Any]] = []
    frame_rows: list[Mapping[str, Any]] = []
    observed: set[str] = set()
    for index, (camera, frame) in enumerate(zip(calibration, frames, strict=True)):
        spec = camera.get("spec") if isinstance(camera, Mapping) else None
        pose = spec.get("pose") if isinstance(spec, Mapping) else None
        intrinsics = spec.get("intrinsics") if isinstance(spec, Mapping) else None
        camera_id = str(camera.get("id") or "") if isinstance(camera, Mapping) else ""
        if (
            not camera_id
            or camera_id in observed
            or not isinstance(frame, Mapping)
            or frame.get("camera_id") != camera_id
            or not isinstance(pose, Mapping)
            or not isinstance(pose.get("T_world_camera_opencv"), list)
            or not isinstance(intrinsics, Mapping)
        ):
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                "scene_configuration_diagnostic_checkpoint_camera_set_invalid"
            )
        observed.add(camera_id)
        cameras.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                "pose": dict(pose),
                "intrinsics": dict(intrinsics),
            }
        )
        frame_rows.append(frame)
    return cameras, frame_rows


def materialize_scene_configuration_diagnostic_checkpoint(
    *,
    stage_production_input_path: str | Path,
    render_inputs_result_path: str | Path,
    semantic_runtime_request_path: str | Path,
    semantic_runtime_result_path: str | Path,
    semantic_teacher_receipt_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Seal one complete 8/8 render+semantic prefix for diagnostic reuse only."""

    stage_path = Path(stage_production_input_path).expanduser().resolve()
    render_path = Path(render_inputs_result_path).expanduser().resolve()
    request_path = Path(semantic_runtime_request_path).expanduser().resolve()
    result_path = Path(semantic_runtime_result_path).expanduser().resolve()
    receipt_path = Path(semantic_teacher_receipt_path).expanduser().resolve()
    stage_input = _read(stage_path, code="scene_configuration_diagnostic_checkpoint_stage_input_invalid")
    render_inputs = _read(render_path, code="scene_configuration_diagnostic_checkpoint_render_invalid")
    semantic_request = _read(request_path, code="scene_configuration_diagnostic_checkpoint_semantic_request_invalid")
    semantic_result = _read(result_path, code="scene_configuration_diagnostic_checkpoint_semantic_result_invalid")
    teacher_receipt = _read(receipt_path, code="scene_configuration_diagnostic_checkpoint_semantic_receipt_invalid")
    if any(
        _contains_secret_material(value)
        for value in (
            stage_input,
            render_inputs,
            semantic_request,
            semantic_result,
            teacher_receipt,
        )
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_secret_material_forbidden"
        )
    envelope = stage_input.get("construction_envelope")
    source_commit = str(stage_input.get("source_commit") or "")
    if (
        stage_input.get("schema_version")
        != "task_evaluation_scene_configuration_stage_production_input.v1"
        or _COMMIT.fullmatch(source_commit) is None
        or not isinstance(envelope, Mapping)
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or envelope.get("expected_production_commit") != source_commit
        or render_inputs.get("schema_version") != RENDER_RESULT_SCHEMA_VERSION
        or render_inputs.get("status") != MATERIALIZED_STATUS
        or render_inputs.get("result_digest")
        != canonical_digest(render_inputs, digest_field="result_digest")
        or render_inputs.get("derived_frame_count") != 8
        or render_inputs.get("render_completed_on_provider") is not True
        or semantic_request.get("schema_version") != _SEMANTIC_REQUEST_SCHEMA
        or semantic_request.get("request_digest")
        != canonical_digest(semantic_request, digest_field="request_digest")
        or semantic_result.get("schema_version") != _SEMANTIC_RESULT_SCHEMA
        or semantic_result.get("status")
        != "completed_unreviewed_semantic_teacher_candidates"
        or semantic_result.get("result_digest")
        != canonical_digest(semantic_result, digest_field="result_digest")
        or semantic_result.get("source_runtime_request_digest")
        != semantic_request.get("request_digest")
        or semantic_result.get("request_count") != 8
        or semantic_result.get("successful_request_count") != 8
        or semantic_result.get("failed_request_count") != 0
        or semantic_result.get("raw_secret_values_recorded") is not False
        or teacher_receipt.get("schema_version") != _SEMANTIC_RECEIPT_SCHEMA
        or teacher_receipt.get("status")
        != "whole_frame_semantic_teacher_candidates_unreviewed"
        or teacher_receipt.get("receipt_digest")
        != canonical_digest(teacher_receipt, digest_field="receipt_digest")
        or teacher_receipt.get("frame_count") != 8
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_prefix_incomplete"
        )
    calibration = _bound_file(
        render_inputs.get("camera_calibration"),
        code="scene_configuration_diagnostic_checkpoint_calibration_invalid",
    )
    cameras, frames = _camera_rows(
        render_inputs=render_inputs, calibration_path=calibration
    )
    request_tasks = semantic_request.get("tasks")
    result_tasks = semantic_result.get("tasks")
    teacher_frames = teacher_receipt.get("frames")
    if (
        not isinstance(request_tasks, list)
        or len(request_tasks) != 1
        or not isinstance(result_tasks, list)
        or len(result_tasks) != 1
        or not isinstance(teacher_frames, list)
        or len(teacher_frames) != 8
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_semantic_set_invalid"
        )
    request_frames = request_tasks[0].get("frames")
    result_frames = result_tasks[0].get("frames")
    if not isinstance(request_frames, list) or not isinstance(result_frames, list):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_semantic_set_invalid"
        )
    # The semantic packet deliberately interleaves elevations for review
    # diversity while the calibration lists them elevation-major: run
    # ...-15c1ade8-...-191412Z carried [e0-a0, e1-a0, e1-a1, e0-a1, ...]
    # against calibration [e0-a0..e0-a3, e1-a0..e1-a3] -- the same eight
    # cameras. Identity is the camera SET with no duplicates; ordering is each
    # producer's own, so requiring order equality refused a correct pass.
    expected_camera_ids = sorted(row["camera_id"] for row in cameras)
    observed_sets = [
        [str(row.get("camera_id") or "") for row in rows if isinstance(row, Mapping)]
        for rows in (request_frames, result_frames, teacher_frames)
    ]
    if any(
        len(ids) != len(set(ids)) or sorted(ids) != expected_camera_ids
        for ids in observed_sets
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_semantic_camera_mismatch"
        )

    root = Path(output_root).expanduser().resolve()
    if root.is_symlink() or root.exists():
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_output_invalid"
        )
    root.mkdir(parents=True, mode=0o750)
    inventory: list[dict[str, Any]] = []
    try:
        inventory.append(
            _copy_inventory_file(
                source=calibration,
                root=root,
                relative="render/camera_calibration.json",
                role="camera_calibration",
            )
        )
        render_manifest = _bound_file(
            render_inputs.get("render_manifest"),
            code="scene_configuration_diagnostic_checkpoint_render_manifest_invalid",
        )
        inventory.append(
            _copy_inventory_file(
                source=render_manifest,
                root=root,
                relative="render/render_manifest.json",
                role="render_manifest",
            )
        )
        retained = _bound_file(
            (render_inputs.get("derived_gaussian_cutout") or {}).get(
                "retained_scene_without_source_object"
            ),
            code="scene_configuration_diagnostic_checkpoint_retained_scene_invalid",
        )
        inventory.append(
            _copy_inventory_file(
                source=retained,
                root=root,
                relative="render/retained_scene.ply",
                role="retained_scene_without_source_object",
            )
        )
        checkpoint_frames: list[dict[str, Any]] = []
        for index, (camera, frame, teacher) in enumerate(
            zip(cameras, frames, teacher_frames, strict=True)
        ):
            raw = _bound_file(
                frame,
                code="scene_configuration_diagnostic_checkpoint_frame_invalid",
            )
            mask = _bound_file(
                frame.get("source_object_mask"),
                code="scene_configuration_diagnostic_checkpoint_mask_invalid",
            )
            edit = _bound_file(
                teacher.get("whole_frame_semantic_teacher"),
                code="scene_configuration_diagnostic_checkpoint_semantic_frame_invalid",
                digest_key="sha256",
            )
            raw_row = _copy_inventory_file(
                source=raw,
                root=root,
                relative=f"render/frames/{index:05d}{raw.suffix.lower() or '.bin'}",
                role=f"raw_frame:{camera['camera_id']}",
            )
            mask_row = _copy_inventory_file(
                source=mask,
                root=root,
                relative=f"render/masks/{index:05d}{mask.suffix.lower() or '.bin'}",
                role=f"source_object_mask:{camera['camera_id']}",
            )
            edit_row = _copy_inventory_file(
                source=edit,
                root=root,
                relative=f"semantic/frames/{index:05d}{edit.suffix.lower() or '.bin'}",
                role=f"semantic_teacher_frame:{camera['camera_id']}",
            )
            inventory.extend((raw_row, mask_row, edit_row))
            checkpoint_frames.append(
                {
                    **camera,
                    "raw_frame_role": raw_row["role"],
                    "source_object_mask_role": mask_row["role"],
                    "semantic_teacher_frame_role": edit_row["role"],
                }
            )
        for source, relative, role in (
            (request_path, "semantic/runtime_request.json", "semantic_runtime_request"),
            (result_path, "semantic/runtime_result.json", "semantic_runtime_result"),
            (receipt_path, "semantic/teacher_receipt.json", "semantic_teacher_receipt"),
        ):
            inventory.append(
                _copy_inventory_file(
                    source=source, root=root, relative=relative, role=role
                )
            )
        bindings = _scientific_bindings(
            stage_input=stage_input, render_inputs=render_inputs
        )
        source_bundle_input_digest = str(
            envelope.get("portable_envelope_digest")
            or envelope.get("envelope_digest")
            or ""
        )
        source_construction_envelope_digest = str(
            envelope.get("control_plane_envelope_digest")
            or envelope.get("envelope_digest")
            or ""
        )
        source_toolchain_digest = str(stage_input.get("toolchain_digest") or "")
        if any(
            _DIGEST.fullmatch(value) is None
            for value in (
                source_bundle_input_digest,
                source_construction_envelope_digest,
                source_toolchain_digest,
            )
        ):
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                "scene_configuration_diagnostic_checkpoint_provenance_invalid"
            )
        checkpoint: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "executed_inside_one_parent_provider_run": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "source_commit_provenance": source_commit,
            "source_run_id_provenance": stage_input.get("run_id"),
            "source_bundle_input_digest_provenance": source_bundle_input_digest,
            "source_construction_envelope_digest_provenance": (
                source_construction_envelope_digest
            ),
            "source_toolchain_digest_provenance": source_toolchain_digest,
            "scientific_bindings": bindings,
            "render_inputs_template": _portable_render_template(render_inputs),
            "camera_count": 8,
            "cameras": checkpoint_frames,
            "semantic_teacher": {
                "backend_id": semantic_result.get("backend_id"),
                "backend_entry_digest": semantic_result.get("backend_entry_digest"),
                "adapter_id": semantic_result.get("adapter_id"),
                "model_snapshot": semantic_result.get("model_snapshot"),
                "runtime_request_digest": semantic_request["request_digest"],
                "scientific_request_digest": canonical_digest(
                    _normalized_semantic_request(semantic_request)
                ),
                "runtime_result_digest": semantic_result["result_digest"],
                "teacher_receipt_digest": teacher_receipt["receipt_digest"],
                "requested_frame_count": 8,
                "completed_frame_count": 8,
                "failed_frame_count": 0,
                "status": "completed_8_of_8_unreviewed_candidates",
            },
            "inventory": sorted(inventory, key=lambda row: row["relative_path"]),
            "completed_stage_prefix_count": 0,
            "completed_stage_results": [],
            "raw_secret_values_recorded": False,
            "checkpoint_digest": "",
        }
        checkpoint["checkpoint_digest"] = canonical_digest(
            checkpoint, digest_field="checkpoint_digest"
        )
        manifest_path = root / f"{SCHEMA_VERSION}.json"
        manifest_path.write_text(canonical_json(checkpoint) + "\n", encoding="utf-8")
        return checkpoint
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def validate_scene_configuration_diagnostic_checkpoint(
    *, checkpoint_root: str | Path, expected_scientific_binding_digest: str | None = None
) -> dict[str, Any]:
    """Reopen every byte and enforce the diagnostic-only claim boundary."""

    unresolved_root = Path(checkpoint_root).expanduser()
    root = unresolved_root.resolve()
    manifest_path = root / f"{SCHEMA_VERSION}.json"
    value = _read(
        manifest_path,
        code="scene_configuration_diagnostic_checkpoint_invalid",
    )
    bindings = value.get("scientific_bindings")
    render_template = value.get("render_inputs_template")
    semantic = value.get("semantic_teacher")
    inventory = value.get("inventory")
    cameras = value.get("cameras")
    completed_stage_results = value.get("completed_stage_results")
    if (
        unresolved_root.is_symlink()
        or not root.is_dir()
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("diagnostic_only") is not True
        or value.get("qualification_eligible") is not False
        or value.get("executed_inside_one_parent_provider_run") is not False
        or value.get("configured_revision_publication_permitted") is not False
        or value.get("offering_publication_permitted") is not False
        or value.get("terminal_e2e_completion_permitted") is not False
        or value.get("raw_secret_values_recorded") is not False
        or _contains_secret_material(value)
        or value.get("checkpoint_digest")
        != canonical_digest(value, digest_field="checkpoint_digest")
        or not isinstance(bindings, Mapping)
        or not isinstance(render_template, Mapping)
        or bindings.get("binding_digest")
        != canonical_digest(bindings, digest_field="binding_digest")
        or (
            expected_scientific_binding_digest is not None
            and bindings.get("binding_digest") != expected_scientific_binding_digest
        )
        or not isinstance(semantic, Mapping)
        or semantic.get("status") != "completed_8_of_8_unreviewed_candidates"
        or semantic.get("requested_frame_count") != 8
        or semantic.get("completed_frame_count") != 8
        or semantic.get("failed_frame_count") != 0
        or value.get("camera_count") != 8
        or not isinstance(cameras, list)
        or len(cameras) != 8
        or not isinstance(inventory, list)
        or not inventory
        or not isinstance(completed_stage_results, list)
        or value.get("completed_stage_prefix_count")
        != len(completed_stage_results)
        or not 0 <= len(completed_stage_results) <= 6
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_invalid"
        )
    expected_paths: set[str] = set()
    roles: set[str] = set()
    for row in inventory:
        relative = str(row.get("relative_path") or "") if isinstance(row, Mapping) else ""
        role = str(row.get("role") or "") if isinstance(row, Mapping) else ""
        posix = PurePosixPath(relative)
        path = root.joinpath(*posix.parts).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                "scene_configuration_diagnostic_checkpoint_inventory_invalid"
            ) from exc
        if (
            not isinstance(row, Mapping)
            or not role
            or role in roles
            or not relative
            or relative in expected_paths
            or posix.is_absolute()
            or ".." in posix.parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("digest")
            or stat.S_IMODE(path.stat().st_mode) != row.get("mode")
        ):
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                "scene_configuration_diagnostic_checkpoint_inventory_invalid"
            )
        expected_paths.add(relative)
        roles.add(role)
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed_paths != expected_paths or any(path.is_symlink() for path in root.rglob("*")):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_inventory_invalid"
        )
    expected_camera_ids = [str(row.get("camera_id") or "") for row in cameras]
    if (
        len(set(expected_camera_ids)) != 8
        or any(
            row.get("frame_index") != index
            or row.get("raw_frame_role") not in roles
            or row.get("source_object_mask_role") not in roles
            or row.get("semantic_teacher_frame_role") not in roles
            for index, row in enumerate(cameras)
            if isinstance(row, Mapping)
        )
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_camera_set_invalid"
        )
    return value


def _stage_configuration_digest(configuration_path: Path) -> str:
    if configuration_path.is_symlink() or not configuration_path.is_file():
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_stage_configuration_invalid"
        )
    return _sha256(configuration_path)


def _portable_stage_result(
    *,
    result: Mapping[str, Any],
    stage_id: str,
    configuration_path: Path,
    checkpoint_root: Path,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    if (
        result.get("schema_version")
        != "task_evaluation_scene_configuration_stage_result.v1"
        or result.get("status") != "completed"
        or result.get("stage_id") != stage_id
        or result.get("configuration_digest")
        != _stage_configuration_digest(configuration_path)
        or result.get("diagnostic_only") is not True
        or result.get("qualification_eligible") is not False
        or result.get("executed_inside_one_parent_provider_run") is not False
        or result.get("configured_revision_publication_permitted") is not False
        or result.get("offering_publication_permitted") is not False
        or result.get("terminal_e2e_completion_permitted") is not False
        or result.get("stage_result_digest")
        != canonical_digest(result, digest_field="stage_result_digest")
        or not isinstance(result.get("output_artifacts"), list)
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            f"scene_configuration_diagnostic_completed_stage_invalid:{stage_id}"
        )
    portable = json.loads(json.dumps(dict(result)))
    portable_artifacts = portable["output_artifacts"]
    for index, (source_row, target_row) in enumerate(
        zip(result["output_artifacts"], portable_artifacts, strict=True)
    ):
        role = str(source_row.get("role") or "") if isinstance(source_row, Mapping) else ""
        source = _bound_file(
            source_row,
            code=f"scene_configuration_diagnostic_completed_stage_artifact_invalid:{stage_id}",
        )
        checkpoint_role = f"completed_stage:{stage_id}:{role}"
        inventory.append(
            _copy_inventory_file(
                source=source,
                root=checkpoint_root,
                relative=(
                    f"completed_stages/{stage_id}/{index:04d}"
                    f"{source.suffix.lower() or '.bin'}"
                ),
                role=checkpoint_role,
            )
        )
        target_row.pop("path", None)
        target_row["checkpoint_role"] = checkpoint_role
    portable["source_stage_result_digest"] = portable["stage_result_digest"]
    portable["stage_result_digest"] = canonical_digest(
        portable, digest_field="stage_result_digest"
    )
    return portable


def advance_scene_configuration_diagnostic_checkpoint(
    *,
    checkpoint_root: str | Path,
    stage_results: list[Mapping[str, Any]],
    stage_sequence: list[Mapping[str, Any]],
    configurations: Mapping[str, tuple[Mapping[str, Any], Path]],
    output_root: str | Path,
) -> dict[str, Any]:
    """Advance a diagnostic checkpoint through one contiguous valid stage prefix."""

    source_root = Path(checkpoint_root).expanduser().resolve()
    source = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=source_root
    )
    if (
        len(stage_sequence) != 6
        or len(stage_results) > 6
        or any(not isinstance(row, Mapping) for row in stage_sequence)
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_completed_stage_prefix_invalid"
        )
    expected_ids = [str(row.get("stage_id") or "") for row in stage_sequence]
    observed_ids = [str(row.get("stage_id") or "") for row in stage_results]
    if observed_ids != expected_ids[: len(observed_ids)]:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_completed_stage_prefix_invalid"
        )
    carried_count = int(source["completed_stage_prefix_count"])
    if len(stage_results) < carried_count:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_completed_stage_regression"
        )
    root = Path(output_root).expanduser().resolve()
    if root.is_symlink() or root.exists():
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_output_invalid"
        )
    root.mkdir(parents=True, mode=0o750)
    inventory: list[dict[str, Any]] = []
    try:
        for row in source["inventory"]:
            source_path = source_root / str(row["relative_path"])
            inventory.append(
                _copy_inventory_file(
                    source=source_path,
                    root=root,
                    relative=str(row["relative_path"]),
                    role=str(row["role"]),
                )
            )
        portable_results = json.loads(
            json.dumps(source["completed_stage_results"])
        )
        for index in range(carried_count, len(stage_results)):
            stage_id = expected_ids[index]
            configuration = configurations.get(stage_id)
            if configuration is None:
                raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                    "scene_configuration_diagnostic_stage_configuration_invalid"
                )
            portable_results.append(
                _portable_stage_result(
                    result=stage_results[index],
                    stage_id=stage_id,
                    configuration_path=configuration[1],
                    checkpoint_root=root,
                    inventory=inventory,
                )
            )
        advanced = json.loads(json.dumps(source))
        advanced["status"] = STATUS
        advanced["inventory"] = sorted(
            inventory, key=lambda row: row["relative_path"]
        )
        advanced["completed_stage_prefix_count"] = len(portable_results)
        advanced["completed_stage_results"] = portable_results
        advanced["checkpoint_digest"] = canonical_digest(
            advanced, digest_field="checkpoint_digest"
        )
        manifest = root / f"{SCHEMA_VERSION}.json"
        manifest.write_text(canonical_json(advanced) + "\n", encoding="utf-8")
        return advanced
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def hydrate_scene_configuration_diagnostic_completed_stages(
    *,
    checkpoint_root: str | Path,
    stage_sequence: list[Mapping[str, Any]],
    configurations: Mapping[str, tuple[Mapping[str, Any], Path]],
) -> list[dict[str, Any]]:
    """Reopen the contiguous carried dependency results and every artifact."""

    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_root
    )
    carried = checkpoint["completed_stage_results"]
    if (
        len(stage_sequence) != 6
        or len(carried) != checkpoint["completed_stage_prefix_count"]
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_completed_stage_prefix_invalid"
        )
    root = Path(checkpoint_root).expanduser().resolve()
    by_role = {
        str(row["role"]): root / str(row["relative_path"])
        for row in checkpoint["inventory"]
    }
    results: list[dict[str, Any]] = []
    for index, portable in enumerate(carried):
        stage_id = str(stage_sequence[index].get("stage_id") or "")
        configuration = configurations.get(stage_id)
        value = json.loads(json.dumps(portable))
        if (
            configuration is None
            or value.get("stage_id") != stage_id
            or value.get("configuration_digest")
            != _stage_configuration_digest(configuration[1])
            or value.get("diagnostic_only") is not True
            or value.get("qualification_eligible") is not False
            or value.get("stage_result_digest")
            != canonical_digest(value, digest_field="stage_result_digest")
        ):
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                f"scene_configuration_diagnostic_completed_stage_invalid:{stage_id}"
            )
        for artifact in value.get("output_artifacts") or []:
            role = artifact.pop("checkpoint_role", None)
            path = by_role.get(str(role or ""))
            if path is None:
                raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                    f"scene_configuration_diagnostic_completed_stage_artifact_invalid:{stage_id}"
                )
            artifact["path"] = str(path.resolve())
        value["stage_result_digest"] = canonical_digest(
            value, digest_field="stage_result_digest"
        )
        results.append(value)
    return results


def hydrate_scene_configuration_diagnostic_render_inputs(
    *,
    checkpoint_root: str | Path,
    expected_scientific_binding_digest: str,
) -> dict[str, Any]:
    """Resolve checkpoint roles to immutable local bytes for resumed stage 1."""

    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_root,
        expected_scientific_binding_digest=expected_scientific_binding_digest,
    )
    root = Path(checkpoint_root).expanduser().resolve()
    by_role = {
        str(row["role"]): root / str(row["relative_path"])
        for row in checkpoint["inventory"]
    }
    render = json.loads(json.dumps(checkpoint["render_inputs_template"]))

    def hydrate(row: dict[str, Any], *, required: bool = True) -> None:
        role = row.pop("checkpoint_role", None)
        if role is None and not required:
            return
        path = by_role.get(str(role or ""))
        if path is None:
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                "scene_configuration_diagnostic_checkpoint_role_missing"
            )
        row["path"] = str(path.resolve())

    hydrate(render["source_appearance"], required=False)
    hydrate(render["camera_calibration"])
    hydrate(render["render_manifest"])
    for frame in render["derived_frames"]:
        hydrate(frame)
        hydrate(frame["source_object_mask"])
    cutout = render["derived_gaussian_cutout"]
    hydrate(cutout["retained_scene_without_source_object"])
    candidate = cutout.get("source_object_candidate")
    if isinstance(candidate, dict):
        hydrate(candidate, required=False)
    render["result_digest"] = canonical_digest(render, digest_field="result_digest")
    return render


def hydrate_scene_configuration_diagnostic_semantic_outputs(
    *,
    checkpoint_root: str | Path,
    current_semantic_runtime_request: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Copy exactly eight sealed edits after checking the rebuilt request."""

    checkpoint = validate_scene_configuration_diagnostic_checkpoint(
        checkpoint_root=checkpoint_root
    )
    if (
        current_semantic_runtime_request.get("schema_version")
        != _SEMANTIC_REQUEST_SCHEMA
        or current_semantic_runtime_request.get("request_digest")
        != canonical_digest(
            current_semantic_runtime_request, digest_field="request_digest"
        )
        or canonical_digest(
            _normalized_semantic_request(current_semantic_runtime_request)
        )
        != checkpoint["semantic_teacher"]["scientific_request_digest"]
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_semantic_request_mismatch"
        )
    tasks = current_semantic_runtime_request.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_semantic_request_mismatch"
        )
    task_id = str(tasks[0].get("task_id") or "")
    request_frames = tasks[0].get("frames")
    camera_ids = [row["camera_id"] for row in checkpoint["cameras"]]
    if (
        not task_id
        or not isinstance(request_frames, list)
        or [str(row.get("camera_id") or "") for row in request_frames]
        != camera_ids
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_semantic_request_mismatch"
        )
    root = Path(checkpoint_root).expanduser().resolve()
    source_by_role = {
        str(row["role"]): root / str(row["relative_path"])
        for row in checkpoint["inventory"]
    }
    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or output.exists():
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_semantic_output_invalid"
        )
    task_root = output / "tasks" / task_id
    task_root.mkdir(parents=True, mode=0o750)
    for index, camera_id in enumerate(camera_ids):
        source = source_by_role[f"semantic_teacher_frame:{camera_id}"]
        destination = task_root / f"{index:05d}.png"
        shutil.copyfile(source, destination)
        if destination.stat().st_size != source.stat().st_size or _sha256(
            destination
        ) != _sha256(source):
            raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
                "scene_configuration_diagnostic_checkpoint_semantic_copy_mismatch"
            )
    semantic = checkpoint["semantic_teacher"]
    result: dict[str, Any] = {
        "schema_version": _SEMANTIC_RESULT_SCHEMA,
        "status": "completed_unreviewed_semantic_teacher_candidates",
        "source_runtime_request_digest": current_semantic_runtime_request[
            "request_digest"
        ],
        "backend_id": semantic["backend_id"],
        "backend_entry_digest": semantic["backend_entry_digest"],
        "adapter_id": semantic["adapter_id"],
        "model_snapshot": semantic["model_snapshot"],
        "request_count": 8,
        "successful_request_count": 8,
        "failed_request_count": 0,
        "tasks": [{"task_id": task_id, "camera_count": 8}],
        "raw_secret_values_recorded": False,
        "diagnostic_checkpoint_reused": True,
        "provider_calls_performed": 0,
        "source_checkpoint_runtime_result_digest": semantic[
            "runtime_result_digest"
        ],
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def diagnostic_checkpoint_scientific_binding_digest(
    *, stage_input: Mapping[str, Any], render_inputs: Mapping[str, Any]
) -> str:
    """Compute the exact reuse key without binding to source commit provenance."""

    return str(
        _scientific_bindings(
            stage_input=stage_input, render_inputs=render_inputs
        )["binding_digest"]
    )


__all__ = [
    "SCHEMA_VERSION",
    "STATUS",
    "TaskEvaluationSceneConfigurationDiagnosticCheckpointError",
    "advance_scene_configuration_diagnostic_checkpoint",
    "diagnostic_checkpoint_scientific_binding_digest",
    "hydrate_scene_configuration_diagnostic_completed_stages",
    "hydrate_scene_configuration_diagnostic_render_inputs",
    "hydrate_scene_configuration_diagnostic_semantic_outputs",
    "materialize_scene_configuration_diagnostic_checkpoint",
    "validate_scene_configuration_diagnostic_checkpoint",
]


def main(argv: list[str] | None = None) -> int:
    """Seal a checkpoint post hoc from retained immutable provider outputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-production-input", required=True)
    parser.add_argument("--render-inputs-result", required=True)
    parser.add_argument("--semantic-runtime-request", required=True)
    parser.add_argument("--semantic-runtime-result", required=True)
    parser.add_argument("--semantic-teacher-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = materialize_scene_configuration_diagnostic_checkpoint(
        stage_production_input_path=args.stage_production_input,
        render_inputs_result_path=args.render_inputs_result,
        semantic_runtime_request_path=args.semantic_runtime_request,
        semantic_runtime_result_path=args.semantic_runtime_result,
        semantic_teacher_receipt_path=args.semantic_teacher_receipt,
        output_root=args.output_root,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
