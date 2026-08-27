"""Seal and reopen a non-qualifying scene-configuration retry checkpoint.

The production lane deliberately requires all six stages to execute inside one
parent provider run.  This checkpoint is a separate diagnostic artifact: it
retains the already-paid provider render and semantic-teacher candidates so a
later diagnostic allocation can exercise the downstream implementation without
claiming (or publishing) a completed configuration.
"""

from __future__ import annotations

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
    """Remove provenance-only fields while retaining every scientific choice."""

    normalized = json.loads(json.dumps(dict(value)))
    normalized.pop("source_commit_sha", None)
    normalized.pop("request_digest", None)
    return normalized


def _canonical_json_file_digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        (canonical_json(value) + "\n").encode("utf-8")
    ).hexdigest()


def _scientific_bindings(
    *, stage_input: Mapping[str, Any], render_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    envelope = stage_input.get("construction_envelope")
    configuration = stage_input.get("configuration")
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
        )
    ):
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_binding_invalid"
        )
    configuration_digest = _canonical_json_file_digest(configuration)
    if stage_input.get("configuration_sha256") != configuration_digest:
        raise TaskEvaluationSceneConfigurationDiagnosticCheckpointError(
            "scene_configuration_diagnostic_checkpoint_configuration_mismatch"
        )
    bundle_input_digest = str(
        envelope.get("portable_envelope_digest")
        or envelope.get("envelope_digest")
        or ""
    )
    construction_digest = str(
        envelope.get("control_plane_envelope_digest")
        or envelope.get("envelope_digest")
        or ""
    )
    recipe = envelope.get("recipe")
    recipe_digest = str(
        envelope.get("recipe_digest")
        or ((recipe or {}).get("recipe_digest") if isinstance(recipe, Mapping) else "")
        or ""
    )
    toolchain_digest = str(stage_input.get("toolchain_digest") or "")
    source_splat_digest = str(render_inputs.get("source_splat_digest") or "")
    calibration_digest = str(calibration.get("digest") or "")
    if (
        _DIGEST.fullmatch(bundle_input_digest) is None
        or _DIGEST.fullmatch(construction_digest) is None
        or _DIGEST.fullmatch(recipe_digest) is None
        or _DIGEST.fullmatch(toolchain_digest) is None
        or _DIGEST.fullmatch(source_splat_digest) is None
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
        "source_bundle_input_digest": bundle_input_digest,
        "source_construction_envelope_digest": construction_digest,
        "recipe_digest": recipe_digest,
        "stage_configuration_digest": configuration_digest,
        "toolchain_digest": toolchain_digest,
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
    expected_camera_ids = [row["camera_id"] for row in cameras]
    observed_sets = [
        [str(row.get("camera_id") or "") for row in rows if isinstance(row, Mapping)]
        for rows in (request_frames, result_frames, teacher_frames)
    ]
    if any(ids != expected_camera_ids for ids in observed_sets):
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
            "scientific_bindings": bindings,
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
    semantic = value.get("semantic_teacher")
    inventory = value.get("inventory")
    cameras = value.get("cameras")
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
        or value.get("checkpoint_digest")
        != canonical_digest(value, digest_field="checkpoint_digest")
        or not isinstance(bindings, Mapping)
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
    "diagnostic_checkpoint_scientific_binding_digest",
    "materialize_scene_configuration_diagnostic_checkpoint",
    "validate_scene_configuration_diagnostic_checkpoint",
]
