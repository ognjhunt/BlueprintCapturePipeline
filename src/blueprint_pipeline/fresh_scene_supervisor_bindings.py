"""Bind host-resident fresh-scene inputs to the production Agents SDK supervisor.

The model never receives filesystem paths or implementations.  This module
validates a control-plane-owned manifest, rehashes every referenced request,
checks the request's own canonical digest, and returns only the trusted
``SupervisorContext`` mappings needed by registered deterministic tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .host_resident_launch_inputs import (
    configured_launch_input_roots,
    path_is_host_resident,
)
from .task_evaluation_supervisor.lifecycle import (
    run_capture_reconstruction_supervisor_continuation,
)


SCHEMA_VERSION = "fresh_scene_supervisor_bindings.v1"
STATUS_SCHEMA = "fresh_scene_paired_target_preparation.v1"
SAM_REQUEST_SCHEMA = "fresh_scene_sam31_task_input_tool_request.v1"
MASK_REQUEST_SCHEMA = "fresh_scene_calibrated_mask_tool_request.v1"
REMOVAL_FREEZE_REQUEST_SCHEMA = "fresh_scene_removal_freeze_tool_request.v1"
SEGMENT_CUTOUT_REQUEST_SCHEMA = "fresh_scene_segment_cutout_tool_request.v1"
ARTIFIXER_CANDIDATE_REQUEST_SCHEMA = (
    "fresh_scene_artifixer_candidate_preparation_request.v1"
)
SEMANTIC_TEACHER_EDIT_REQUEST_SCHEMA = (
    "fresh_scene_semantic_teacher_image_edit_request.v1"
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_BOUND_INPUT_FILES = 1024
MAX_BOUND_INPUT_BYTES = 2 * 1024**3


class FreshSceneSupervisorBindingError(ValueError):
    """A fresh-scene supervisor binding is unsafe, stale, or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSceneSupervisorBindingError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise FreshSceneSupervisorBindingError(code)
    return value


def _resident_path(
    value: str | Path,
    *,
    roots: Sequence[Path],
    kind: str,
    code: str,
) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise FreshSceneSupervisorBindingError(code)
    resolved = unresolved.resolve()
    exists = resolved.is_file() if kind == "file" else resolved.is_dir()
    if not exists or not path_is_host_resident(resolved, roots):
        raise FreshSceneSupervisorBindingError(code)
    return resolved


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _request_input_paths(
    request: Mapping[str, Any], *, schema: str, roots: Sequence[Path]
) -> list[Path]:
    paths: list[Path] = []
    if schema == SAM_REQUEST_SCHEMA:
        for key in (
            "calibrated_view_receipt_path",
            "task_freeze_path",
            "provider_profile_path",
            "prompts_path",
        ):
            paths.append(
                _resident_path(
                    str(request.get(key) or ""),
                    roots=roots,
                    kind="file",
                    code=f"fresh_scene_tool_request_input_not_host_resident:{key}",
                )
            )
        if request.get("ffmpeg_executable") is not None:
            paths.append(
                _resident_path(
                    str(request["ffmpeg_executable"]),
                    roots=roots,
                    kind="file",
                    code=(
                        "fresh_scene_tool_request_input_not_host_resident:"
                        "ffmpeg_executable"
                    ),
                )
            )
    elif schema == MASK_REQUEST_SCHEMA:
        for value in request.get("task_freeze_paths") or []:
            paths.append(
                _resident_path(
                    str(value),
                    roots=roots,
                    kind="file",
                    code="fresh_scene_tool_request_input_not_host_resident:task_freeze",
                )
            )
        for raw in (request.get("task_inputs") or {}).values():
            for key in ("source_track_result_path", "camera_contract_path"):
                paths.append(
                    _resident_path(
                        str(raw.get(key) or ""),
                        roots=roots,
                        kind="file",
                        code=f"fresh_scene_tool_request_input_not_host_resident:{key}",
                    )
                )
            image_root = _resident_path(
                str(raw.get("source_image_root") or ""),
                roots=roots,
                kind="directory",
                code="fresh_scene_tool_request_input_not_host_resident:source_image_root",
            )
            paths.extend(sorted(image_root.rglob("*.png")))
        paths.append(
            _resident_path(
                str(request.get("reviewed_track_selection_receipt_path") or ""),
                roots=roots,
                kind="file",
                code=(
                    "fresh_scene_tool_request_input_not_host_resident:"
                    "reviewed_track_selection_receipt_path"
                ),
            )
        )
    elif schema == REMOVAL_FREEZE_REQUEST_SCHEMA:
        removal_inputs: dict[str, Path] = {}
        for key in (
            "source_standard_splat_path",
            "source_collision_path",
            "registered_frame_receipt_path",
            "calibrated_mask_set_receipt_path",
        ):
            resident = _resident_path(
                str(request.get(key) or ""),
                roots=roots,
                kind="file",
                code=f"fresh_scene_tool_request_input_not_host_resident:{key}",
            )
            paths.append(resident)
            removal_inputs[key] = resident
        for task in (request.get("tasks") or {}).values():
            if isinstance(task, Mapping) and task.get("render_input_receipt_path"):
                paths.append(
                    _resident_path(
                        str(task["render_input_receipt_path"]),
                        roots=roots,
                        kind="file",
                        code=(
                            "fresh_scene_tool_request_input_not_host_resident:"
                            "render_input_receipt_path"
                        ),
                    )
                )
        mask_receipt_path = removal_inputs["calibrated_mask_set_receipt_path"]
        mask_receipt = _read_object(
            mask_receipt_path, code="fresh_scene_removal_mask_receipt_invalid"
        )
        mask_tasks = mask_receipt.get("tasks")
        if not isinstance(mask_tasks, list) or not 1 <= len(mask_tasks) <= 5:
            raise FreshSceneSupervisorBindingError(
                "fresh_scene_removal_mask_receipt_invalid"
            )
        for task in mask_tasks:
            if not isinstance(task, Mapping):
                raise FreshSceneSupervisorBindingError(
                    "fresh_scene_removal_mask_receipt_invalid"
                )
            for record in (
                task.get("task_freeze"),
                task.get("source_track_result"),
                task.get("camera_contract"),
            ):
                if not isinstance(record, Mapping):
                    raise FreshSceneSupervisorBindingError(
                        "fresh_scene_removal_mask_receipt_invalid"
                    )
                relative = record.get("relative_path")
                referenced = (
                    mask_receipt_path.parent / str(relative)
                    if relative
                    else Path(str(record.get("path") or ""))
                )
                paths.append(
                    _resident_path(
                        referenced,
                        roots=roots,
                        kind="file",
                        code="fresh_scene_removal_transitive_input_not_host_resident",
                    )
                )
            for collection, key in (
                (task.get("source_images"), "image"),
                (task.get("masks"), "mask"),
            ):
                if not isinstance(collection, list):
                    raise FreshSceneSupervisorBindingError(
                        "fresh_scene_removal_mask_receipt_invalid"
                    )
                for row in collection:
                    record = row.get(key) if isinstance(row, Mapping) else None
                    if not isinstance(record, Mapping):
                        raise FreshSceneSupervisorBindingError(
                            "fresh_scene_removal_mask_receipt_invalid"
                        )
                    relative = str(record.get("relative_path") or "")
                    paths.append(
                        _resident_path(
                            mask_receipt_path.parent / relative,
                            roots=roots,
                            kind="file",
                            code=(
                                "fresh_scene_removal_transitive_input_not_host_resident"
                            ),
                        )
                    )
    elif schema == SEGMENT_CUTOUT_REQUEST_SCHEMA:
        paths.append(
            _resident_path(
                str(request.get("source_standard_splat_path") or ""),
                roots=roots,
                kind="file",
                code=(
                    "fresh_scene_tool_request_input_not_host_resident:"
                    "source_standard_splat_path"
                ),
            )
        )
        for key in (
            "task_freeze_paths",
            "sweep_freeze_paths_by_task",
            "contribution_manifest_paths_by_task",
        ):
            raw = request.get(key)
            values = raw if isinstance(raw, list) else (raw or {}).values()
            for value in values:
                paths.append(
                    _resident_path(
                        str(value),
                        roots=roots,
                        kind="file",
                        code=f"fresh_scene_tool_request_input_not_host_resident:{key}",
                    )
                )
        manifests = request.get("contribution_manifest_paths_by_task") or {}
        for manifest_value in manifests.values():
            manifest_path = Path(str(manifest_value)).expanduser().resolve()
            manifest = _read_object(
                manifest_path, code="fresh_scene_segment_cutout_manifest_invalid"
            )
            repetitions = manifest.get("repetitions")
            if not isinstance(repetitions, list) or len(repetitions) < 2:
                raise FreshSceneSupervisorBindingError(
                    "fresh_scene_segment_cutout_manifest_invalid"
                )
            for record in repetitions:
                if not isinstance(record, Mapping):
                    raise FreshSceneSupervisorBindingError(
                        "fresh_scene_segment_cutout_manifest_invalid"
                    )
                paths.append(
                    _resident_path(
                        manifest_path.parent
                        / str(record.get("relative_path") or ""),
                        roots=roots,
                        kind="file",
                        code="fresh_scene_segment_cutout_array_not_host_resident",
                    )
                )
    elif schema == ARTIFIXER_CANDIDATE_REQUEST_SCHEMA:
        for key in ("segment_cutout_set_path", "execution_authority_path"):
            paths.append(
                _resident_path(
                    str(request.get(key) or ""),
                    roots=roots,
                    kind="file",
                    code=f"fresh_scene_tool_request_input_not_host_resident:{key}",
                )
            )
    elif schema == SEMANTIC_TEACHER_EDIT_REQUEST_SCHEMA:
        for key in (
            "source_candidate_inputs_receipt_path",
            "backend_registry_path",
            "rights_attestation_path",
        ):
            paths.append(
                _resident_path(
                    str(request.get(key) or ""),
                    roots=roots,
                    kind="file",
                    code=f"fresh_scene_tool_request_input_not_host_resident:{key}",
                )
            )
        candidate_path = paths[0]
        candidate = _read_object(
            candidate_path, code="fresh_scene_semantic_teacher_candidate_invalid"
        )
        tasks = candidate.get("tasks")
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= 5:
            raise FreshSceneSupervisorBindingError(
                "fresh_scene_semantic_teacher_candidate_invalid"
            )
        for task in tasks:
            frames = task.get("frames") if isinstance(task, Mapping) else None
            if not isinstance(frames, list) or not frames:
                raise FreshSceneSupervisorBindingError(
                    "fresh_scene_semantic_teacher_candidate_invalid"
                )
            for frame in frames:
                if not isinstance(frame, Mapping):
                    raise FreshSceneSupervisorBindingError(
                        "fresh_scene_semantic_teacher_candidate_invalid"
                    )
                for key in ("input_retained_frame", "input_exact_repair_mask"):
                    record = frame.get(key)
                    if not isinstance(record, Mapping):
                        raise FreshSceneSupervisorBindingError(
                            "fresh_scene_semantic_teacher_candidate_invalid"
                        )
                    paths.append(
                        _resident_path(
                            str(record.get("path") or ""),
                            roots=roots,
                            kind="file",
                            code=(
                                "fresh_scene_semantic_teacher_transitive_input_"
                                "not_host_resident"
                            ),
                        )
                    )
        for value in request.get("object_absent_reference_receipt_paths") or []:
            paths.append(
                _resident_path(
                    str(value),
                    roots=roots,
                    kind="file",
                    code=(
                        "fresh_scene_tool_request_input_not_host_resident:"
                        "object_absent_reference_receipt_paths"
                    ),
                )
            )
    else:  # pragma: no cover - every accepted schema is handled above
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
    unique = sorted(set(paths))
    if (
        not unique
        or len(unique) > MAX_BOUND_INPUT_FILES
        or any(path.is_symlink() or not path.is_file() for path in unique)
        or sum(path.stat().st_size for path in unique) > MAX_BOUND_INPUT_BYTES
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_input_set_invalid")
    return unique


def _request_input_inventory(
    request: Mapping[str, Any], *, schema: str, roots: Sequence[Path]
) -> list[dict[str, Any]]:
    return [_record(path) for path in _request_input_paths(request, schema=schema, roots=roots)]


def _validate_status(value: Mapping[str, Any]) -> dict[str, Any]:
    status = dict(value)
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or _DIGEST.fullmatch(str(status.get("status_digest") or "")) is None
        or status.get("status_digest")
        != canonical_digest(status, digest_field="status_digest")
        or not isinstance(status.get("task_ids"), list)
        or not 1 <= len(status["task_ids"]) <= 5
        or status.get("task_count") != len(status["task_ids"])
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_status_binding_invalid")
    return status


def _validate_request(
    value: Mapping[str, Any],
    *,
    schema: str,
    roots: Sequence[Path],
) -> dict[str, Any]:
    request = dict(value)
    if (
        request.get("schema_version") != schema
        or _DIGEST.fullmatch(str(request.get("request_digest") or "")) is None
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
    if schema == SAM_REQUEST_SCHEMA:
        _request_input_paths(request, schema=schema, roots=roots)
    elif schema == MASK_REQUEST_SCHEMA:
        freezes = request.get("task_freeze_paths")
        task_inputs = request.get("task_inputs")
        selected = request.get("selected_track_ids_by_task")
        review_path = request.get("reviewed_track_selection_receipt_path")
        if (
            not isinstance(freezes, list)
            or not 1 <= len(freezes) <= 5
            or not isinstance(task_inputs, Mapping)
            or not isinstance(selected, Mapping)
            or not str(review_path or "").strip()
            or set(task_inputs) != set(selected)
            or not 1 <= len(task_inputs) <= 5
        ):
            raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
        for task_id, raw in task_inputs.items():
            if not isinstance(raw, Mapping) or not str(task_id).strip():
                raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
        _request_input_paths(request, schema=schema, roots=roots)
    elif schema == REMOVAL_FREEZE_REQUEST_SCHEMA:
        tasks = request.get("tasks")
        if not isinstance(tasks, Mapping) or not 1 <= len(tasks) <= 5:
            raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
        for task_id, raw in tasks.items():
            if (
                not str(task_id).strip()
                or not isinstance(raw, Mapping)
                or not str(raw.get("target_collision_prim_path") or "").startswith("/")
                or not isinstance(raw.get("scene"), Mapping)
                or not isinstance(raw.get("policy"), Mapping)
                or not isinstance(raw.get("historical_baseline"), Mapping)
            ):
                raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
        _request_input_paths(request, schema=schema, roots=roots)
    elif schema == SEGMENT_CUTOUT_REQUEST_SCHEMA:
        task_freezes = request.get("task_freeze_paths")
        sweeps = request.get("sweep_freeze_paths_by_task")
        manifests = request.get("contribution_manifest_paths_by_task")
        if (
            not isinstance(task_freezes, list)
            or not 1 <= len(task_freezes) <= 5
            or not isinstance(sweeps, Mapping)
            or not isinstance(manifests, Mapping)
            or set(sweeps) != set(manifests)
            or len(sweeps) != len(task_freezes)
        ):
            raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
        _request_input_paths(request, schema=schema, roots=roots)
    elif schema == ARTIFIXER_CANDIDATE_REQUEST_SCHEMA:
        selected = request.get("selected_task_ids")
        references = request.get("object_absent_reference_receipt_paths")
        if (
            selected is not None
            and (not isinstance(selected, list) or not 1 <= len(selected) <= 5)
        ) or not isinstance(references, list):
            raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
        _request_input_paths(request, schema=schema, roots=roots)
    elif schema == SEMANTIC_TEACHER_EDIT_REQUEST_SCHEMA:
        selected = request.get("selected_task_ids")
        if (
            not str(request.get("backend_id") or "").strip()
            or not str(request.get("prompt_policy") or "").strip()
            or request.get("output_format") != "png"
            or request.get("retry_count") != 0
            or (
                selected is not None
                and (not isinstance(selected, list) or not 1 <= len(selected) <= 5)
            )
        ):
            raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
        _request_input_paths(request, schema=schema, roots=roots)
    else:  # pragma: no cover - every accepted schema is handled above
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
    return request


def _selected_task_ids(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    selected = [str(value).strip() for value in values]
    if (
        not 1 <= len(selected) <= 5
        or any(not value for value in selected)
        or len(set(selected)) != len(selected)
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
    return sorted(selected)


def _request_path(
    value: str | Path,
    *,
    roots: Sequence[Path],
    field: str,
) -> str:
    return str(
        _resident_path(
            value,
            roots=roots,
            kind="file",
            code=f"fresh_scene_tool_request_input_not_host_resident:{field}",
        )
    )


def _write_tool_request(
    *,
    value: Mapping[str, Any],
    schema: str,
    output_path: str | Path,
    roots: Sequence[Path] | None,
) -> dict[str, Any]:
    resident_roots = tuple(
        configured_launch_input_roots() if roots is None else roots
    )
    request = dict(value)
    request["schema_version"] = schema
    request["request_digest"] = ""
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request = _validate_request(request, schema=schema, roots=resident_roots)
    destination = Path(output_path).expanduser().resolve()
    if (
        destination.exists()
        or destination.is_symlink()
        or not path_is_host_resident(destination, resident_roots)
    ):
        raise FreshSceneSupervisorBindingError(
            "fresh_scene_tool_request_output_invalid"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(request) + "\n", encoding="utf-8")
    return request


def materialize_fresh_scene_removal_freeze_request(
    *,
    source_standard_splat_path: str | Path,
    source_collision_path: str | Path,
    registered_frame_receipt_path: str | Path,
    calibrated_mask_set_receipt_path: str | Path,
    tasks: Mapping[str, Any],
    output_path: str | Path,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Seal the reviewed mask-to-removal inputs; never select Gaussians."""

    resident_roots = tuple(
        configured_launch_input_roots() if roots is None else roots
    )
    if (
        not 1 <= len(tasks) <= 5
        or any(not isinstance(task_id, str) or not task_id.strip() for task_id in tasks)
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
    normalized_tasks: dict[str, Any] = {}
    for task_id in sorted(tasks):
        raw = tasks[task_id]
        if not isinstance(task_id, str) or not isinstance(raw, Mapping):
            raise FreshSceneSupervisorBindingError(
                "fresh_scene_tool_request_invalid"
            )
        task = {
            "target_collision_prim_path": raw.get("target_collision_prim_path"),
            "scene": raw.get("scene"),
            "policy": raw.get("policy"),
            "historical_baseline": raw.get("historical_baseline"),
        }
        if raw.get("render_input_receipt_path") is not None:
            task["render_input_receipt_path"] = _request_path(
                str(raw["render_input_receipt_path"]),
                roots=resident_roots,
                field="render_input_receipt_path",
            )
        normalized_tasks[task_id] = task
    return _write_tool_request(
        value={
            "source_standard_splat_path": _request_path(
                source_standard_splat_path,
                roots=resident_roots,
                field="source_standard_splat_path",
            ),
            "source_collision_path": _request_path(
                source_collision_path,
                roots=resident_roots,
                field="source_collision_path",
            ),
            "registered_frame_receipt_path": _request_path(
                registered_frame_receipt_path,
                roots=resident_roots,
                field="registered_frame_receipt_path",
            ),
            "calibrated_mask_set_receipt_path": _request_path(
                calibrated_mask_set_receipt_path,
                roots=resident_roots,
                field="calibrated_mask_set_receipt_path",
            ),
            "tasks": normalized_tasks,
        },
        schema=REMOVAL_FREEZE_REQUEST_SCHEMA,
        output_path=output_path,
        roots=resident_roots,
    )


def materialize_fresh_scene_segment_cutout_request(
    *,
    source_standard_splat_path: str | Path,
    task_freeze_paths: Sequence[str | Path],
    sweep_freeze_paths_by_task: Mapping[str, str | Path],
    contribution_manifest_paths_by_task: Mapping[str, str | Path],
    output_path: str | Path,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Seal completed contribution evidence for deterministic cutout."""

    resident_roots = tuple(
        configured_launch_input_roots() if roots is None else roots
    )
    if (
        any(
            not isinstance(task_id, str) or not task_id.strip()
            for task_id in sweep_freeze_paths_by_task
        )
        or any(
            not isinstance(task_id, str) or not task_id.strip()
            for task_id in contribution_manifest_paths_by_task
        )
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
    task_freezes = sorted(
        _request_path(
            value,
            roots=resident_roots,
            field="task_freeze_paths",
        )
        for value in task_freeze_paths
    )
    if len(set(task_freezes)) != len(task_freezes):
        raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_invalid")
    sweeps = {
        str(task_id): _request_path(
            value,
            roots=resident_roots,
            field="sweep_freeze_paths_by_task",
        )
        for task_id, value in sorted(sweep_freeze_paths_by_task.items())
    }
    manifests = {
        str(task_id): _request_path(
            value,
            roots=resident_roots,
            field="contribution_manifest_paths_by_task",
        )
        for task_id, value in sorted(contribution_manifest_paths_by_task.items())
    }
    return _write_tool_request(
        value={
            "source_standard_splat_path": _request_path(
                source_standard_splat_path,
                roots=resident_roots,
                field="source_standard_splat_path",
            ),
            "task_freeze_paths": task_freezes,
            "sweep_freeze_paths_by_task": sweeps,
            "contribution_manifest_paths_by_task": manifests,
        },
        schema=SEGMENT_CUTOUT_REQUEST_SCHEMA,
        output_path=output_path,
        roots=resident_roots,
    )


def materialize_fresh_scene_artifixer_candidate_request(
    *,
    segment_cutout_set_path: str | Path,
    execution_authority_path: str | Path,
    output_path: str | Path,
    selected_task_ids: Sequence[str] | None = None,
    object_absent_reference_receipt_paths: Sequence[str | Path] = (),
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Seal reviewed cutout inputs without executing an appearance model."""

    resident_roots = tuple(
        configured_launch_input_roots() if roots is None else roots
    )
    return _write_tool_request(
        value={
            "segment_cutout_set_path": _request_path(
                segment_cutout_set_path,
                roots=resident_roots,
                field="segment_cutout_set_path",
            ),
            "execution_authority_path": _request_path(
                execution_authority_path,
                roots=resident_roots,
                field="execution_authority_path",
            ),
            "selected_task_ids": _selected_task_ids(selected_task_ids),
            "object_absent_reference_receipt_paths": sorted(
                _request_path(
                    value,
                    roots=resident_roots,
                    field="object_absent_reference_receipt_paths",
                )
                for value in object_absent_reference_receipt_paths
            ),
        },
        schema=ARTIFIXER_CANDIDATE_REQUEST_SCHEMA,
        output_path=output_path,
        roots=resident_roots,
    )


def materialize_fresh_scene_semantic_teacher_edit_request(
    *,
    source_candidate_inputs_receipt_path: str | Path,
    backend_registry_path: str | Path,
    backend_id: str,
    rights_attestation_path: str | Path,
    output_path: str | Path,
    selected_task_ids: Sequence[str] | None = None,
    object_absent_reference_receipt_paths: Sequence[str | Path] = (),
    prompt_policy: str = "generic_masked_object_absent_background_completion_v2",
    output_format: str = "png",
    retry_count: int = 0,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Seal whole-frame teacher inputs without upload, inference, or spend."""

    resident_roots = tuple(
        configured_launch_input_roots() if roots is None else roots
    )
    return _write_tool_request(
        value={
            "source_candidate_inputs_receipt_path": _request_path(
                source_candidate_inputs_receipt_path,
                roots=resident_roots,
                field="source_candidate_inputs_receipt_path",
            ),
            "backend_registry_path": _request_path(
                backend_registry_path,
                roots=resident_roots,
                field="backend_registry_path",
            ),
            "backend_id": backend_id.strip(),
            "rights_attestation_path": _request_path(
                rights_attestation_path,
                roots=resident_roots,
                field="rights_attestation_path",
            ),
            "selected_task_ids": _selected_task_ids(selected_task_ids),
            "object_absent_reference_receipt_paths": sorted(
                _request_path(
                    value,
                    roots=resident_roots,
                    field="object_absent_reference_receipt_paths",
                )
                for value in object_absent_reference_receipt_paths
            ),
            "prompt_policy": prompt_policy,
            "output_format": output_format,
            "retry_count": retry_count,
        },
        schema=SEMANTIC_TEACHER_EDIT_REQUEST_SCHEMA,
        output_path=output_path,
        roots=resident_roots,
    )


def materialize_fresh_scene_supervisor_bindings(
    *,
    preparation_status_path: str | Path,
    output_path: str | Path,
    sam31_task_input_request_path: str | Path | None = None,
    calibrated_mask_request_path: str | Path | None = None,
    removal_freeze_request_path: str | Path | None = None,
    segment_cutout_request_path: str | Path | None = None,
    artifixer_candidate_request_path: str | Path | None = None,
    semantic_teacher_edit_request_path: str | Path | None = None,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Seal the exact non-spend tool inputs available on this host."""

    resident_roots = tuple(configured_launch_input_roots() if roots is None else roots)
    status_path = _resident_path(
        preparation_status_path,
        roots=resident_roots,
        kind="file",
        code="fresh_scene_status_not_host_resident",
    )
    status = _validate_status(
        _read_object(status_path, code="fresh_scene_status_binding_invalid")
    )
    context_fields: dict[str, dict[str, Any]] = {
        "fresh_scene_preparation_status": status
    }
    requested_tool_ids = ["inspect_fresh_scene_preparation"]
    request_records: dict[str, dict[str, Any]] = {}
    for field, tool_id, schema, raw_path in (
        (
            "fresh_scene_sam31_task_input_request",
            "materialize_sam31_task_inputs",
            SAM_REQUEST_SCHEMA,
            sam31_task_input_request_path,
        ),
        (
            "fresh_scene_calibrated_mask_request",
            "materialize_calibrated_object_masks",
            MASK_REQUEST_SCHEMA,
            calibrated_mask_request_path,
        ),
        (
            "fresh_scene_removal_freeze_request",
            "materialize_fresh_scene_removal_freezes",
            REMOVAL_FREEZE_REQUEST_SCHEMA,
            removal_freeze_request_path,
        ),
        (
            "fresh_scene_segment_cutout_request",
            "materialize_fresh_scene_segment_cutout",
            SEGMENT_CUTOUT_REQUEST_SCHEMA,
            segment_cutout_request_path,
        ),
        (
            "fresh_scene_artifixer_candidate_request",
            "materialize_fresh_scene_artifixer_candidate",
            ARTIFIXER_CANDIDATE_REQUEST_SCHEMA,
            artifixer_candidate_request_path,
        ),
        (
            "fresh_scene_semantic_teacher_edit_request",
            "materialize_fresh_scene_semantic_teacher_edit_packet",
            SEMANTIC_TEACHER_EDIT_REQUEST_SCHEMA,
            semantic_teacher_edit_request_path,
        ),
    ):
        if raw_path is None:
            continue
        path = _resident_path(
            raw_path,
            roots=resident_roots,
            kind="file",
            code="fresh_scene_tool_request_not_host_resident",
        )
        request = _validate_request(
            _read_object(path, code="fresh_scene_tool_request_invalid"),
            schema=schema,
            roots=resident_roots,
        )
        context_fields[field] = request
        request_records[field] = {
            **_record(path),
            "request_digest": request["request_digest"],
            "input_inventory": _request_input_inventory(
                request, schema=schema, roots=resident_roots
            ),
        }
        requested_tool_ids.append(tool_id)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "host_resident_non_spend_bindings_prepared",
        "task_count": status["task_count"],
        "task_ids": list(status["task_ids"]),
        "preparation_status": {
            **_record(status_path),
            "status_digest": status["status_digest"],
        },
        "tool_requests": request_records,
        "requested_tool_ids": sorted(requested_tool_ids),
        "context_field_names": sorted(context_fields),
        "agent_receives_paths": False,
        "paid_execution_authorized": False,
        "provider_mutations_performed": 0,
        "proof_effect": "none",
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FreshSceneSupervisorBindingError("fresh_scene_binding_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def compile_fresh_scene_supervisor_bindings(
    manifest_path: str | Path,
    *,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Reopen and rehash a sealed manifest into trusted supervisor bindings."""

    resident_roots = tuple(configured_launch_input_roots() if roots is None else roots)
    path = _resident_path(
        manifest_path,
        roots=resident_roots,
        kind="file",
        code="fresh_scene_binding_manifest_not_host_resident",
    )
    manifest = _read_object(path, code="fresh_scene_binding_manifest_invalid")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "host_resident_non_spend_bindings_prepared"
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or manifest.get("agent_receives_paths") is not False
        or manifest.get("paid_execution_authorized") is not False
        or manifest.get("provider_mutations_performed") != 0
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_binding_manifest_invalid")
    status_record = manifest.get("preparation_status")
    if not isinstance(status_record, Mapping):
        raise FreshSceneSupervisorBindingError("fresh_scene_binding_manifest_invalid")
    status_path = _resident_path(
        str(status_record.get("path") or ""),
        roots=resident_roots,
        kind="file",
        code="fresh_scene_status_not_host_resident",
    )
    if (
        status_path.stat().st_size != status_record.get("size_bytes")
        or _sha256(status_path) != status_record.get("sha256")
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_status_bytes_changed")
    status = _validate_status(
        _read_object(status_path, code="fresh_scene_status_binding_invalid")
    )
    if status.get("status_digest") != status_record.get("status_digest"):
        raise FreshSceneSupervisorBindingError("fresh_scene_status_bytes_changed")
    context: dict[str, dict[str, Any]] = {"fresh_scene_preparation_status": status}
    schema_by_field = {
        "fresh_scene_sam31_task_input_request": SAM_REQUEST_SCHEMA,
        "fresh_scene_calibrated_mask_request": MASK_REQUEST_SCHEMA,
        "fresh_scene_removal_freeze_request": REMOVAL_FREEZE_REQUEST_SCHEMA,
        "fresh_scene_segment_cutout_request": SEGMENT_CUTOUT_REQUEST_SCHEMA,
        "fresh_scene_artifixer_candidate_request": ARTIFIXER_CANDIDATE_REQUEST_SCHEMA,
        "fresh_scene_semantic_teacher_edit_request": (
            SEMANTIC_TEACHER_EDIT_REQUEST_SCHEMA
        ),
    }
    records = manifest.get("tool_requests")
    if not isinstance(records, Mapping) or set(records) - set(schema_by_field):
        raise FreshSceneSupervisorBindingError("fresh_scene_binding_manifest_invalid")
    tool_by_field = {
        "fresh_scene_sam31_task_input_request": "materialize_sam31_task_inputs",
        "fresh_scene_calibrated_mask_request": "materialize_calibrated_object_masks",
        "fresh_scene_removal_freeze_request": "materialize_fresh_scene_removal_freezes",
        "fresh_scene_segment_cutout_request": "materialize_fresh_scene_segment_cutout",
        "fresh_scene_artifixer_candidate_request": (
            "materialize_fresh_scene_artifixer_candidate"
        ),
        "fresh_scene_semantic_teacher_edit_request": (
            "materialize_fresh_scene_semantic_teacher_edit_packet"
        ),
    }
    expected_tools = {"inspect_fresh_scene_preparation"}
    for field, raw in records.items():
        if not isinstance(raw, Mapping):
            raise FreshSceneSupervisorBindingError("fresh_scene_binding_manifest_invalid")
        request_path = _resident_path(
            str(raw.get("path") or ""),
            roots=resident_roots,
            kind="file",
            code="fresh_scene_tool_request_not_host_resident",
        )
        if (
            request_path.stat().st_size != raw.get("size_bytes")
            or _sha256(request_path) != raw.get("sha256")
        ):
            raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_bytes_changed")
        request = _validate_request(
            _read_object(request_path, code="fresh_scene_tool_request_invalid"),
            schema=schema_by_field[field],
            roots=resident_roots,
        )
        if request["request_digest"] != raw.get("request_digest"):
            raise FreshSceneSupervisorBindingError("fresh_scene_tool_request_bytes_changed")
        if raw.get("input_inventory") != _request_input_inventory(
            request, schema=schema_by_field[field], roots=resident_roots
        ):
            raise FreshSceneSupervisorBindingError(
                "fresh_scene_tool_request_input_bytes_changed"
            )
        context[field] = request
        expected_tools.add(tool_by_field[field])
    if (
        manifest.get("requested_tool_ids") != sorted(expected_tools)
        or manifest.get("context_field_names") != sorted(context)
    ):
        raise FreshSceneSupervisorBindingError("fresh_scene_binding_manifest_invalid")
    return {
        "requested_tool_ids": sorted(expected_tools),
        "context_bindings": context,
        "manifest_digest": manifest["manifest_digest"],
    }


def _cli_mapping(path: str | Path, *, code: str) -> dict[str, Any]:
    roots = configured_launch_input_roots()
    source = _resident_path(
        path, roots=roots, kind="file", code=f"{code}_not_host_resident"
    )
    value = _read_object(source, code=code)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    removal = commands.add_parser("request-removal-freezes")
    removal.add_argument("--source-standard-splat", required=True)
    removal.add_argument("--source-collision", required=True)
    removal.add_argument("--registered-frame-receipt", required=True)
    removal.add_argument("--calibrated-mask-set-receipt", required=True)
    removal.add_argument("--tasks-json", required=True)
    removal.add_argument("--output", required=True)
    cutout = commands.add_parser("request-segment-cutout")
    cutout.add_argument("--source-standard-splat", required=True)
    cutout.add_argument("--task-freeze", action="append", required=True)
    cutout.add_argument("--sweep-freezes-json", required=True)
    cutout.add_argument("--contribution-manifests-json", required=True)
    cutout.add_argument("--output", required=True)
    candidate = commands.add_parser("request-artifixer-candidate")
    candidate.add_argument("--segment-cutout-set", required=True)
    candidate.add_argument("--execution-authority", required=True)
    candidate.add_argument("--selected-task-id", action="append")
    candidate.add_argument(
        "--object-absent-reference-receipt", action="append", default=[]
    )
    candidate.add_argument("--output", required=True)
    teacher = commands.add_parser("request-semantic-teacher-edit")
    teacher.add_argument("--source-candidate-inputs-receipt", required=True)
    teacher.add_argument("--backend-registry", required=True)
    teacher.add_argument("--backend-id", required=True)
    teacher.add_argument("--rights-attestation", required=True)
    teacher.add_argument("--selected-task-id", action="append")
    teacher.add_argument(
        "--object-absent-reference-receipt", action="append", default=[]
    )
    teacher.add_argument(
        "--prompt-policy",
        default="generic_masked_object_absent_background_completion_v2",
    )
    teacher.add_argument("--output-format", default="png")
    teacher.add_argument("--retry-count", type=int, default=0)
    teacher.add_argument("--output", required=True)
    build = commands.add_parser("build")
    build.add_argument("--preparation-status", required=True)
    build.add_argument("--sam31-task-input-request")
    build.add_argument("--calibrated-mask-request")
    build.add_argument("--removal-freeze-request")
    build.add_argument("--segment-cutout-request")
    build.add_argument("--artifixer-candidate-request")
    build.add_argument("--semantic-teacher-edit-request")
    build.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--binding-manifest", required=True)
    run.add_argument("--capture-root", required=True)
    run.add_argument("--control-plane-inspection", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--allow-live-agents-sdk", action="store_true")
    run.add_argument("--agent-inference-budget-usd", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "request-removal-freezes":
        result = materialize_fresh_scene_removal_freeze_request(
            source_standard_splat_path=args.source_standard_splat,
            source_collision_path=args.source_collision,
            registered_frame_receipt_path=args.registered_frame_receipt,
            calibrated_mask_set_receipt_path=args.calibrated_mask_set_receipt,
            tasks=_cli_mapping(
                args.tasks_json, code="fresh_scene_removal_tasks_invalid"
            ),
            output_path=args.output,
        )
    elif args.command == "request-segment-cutout":
        result = materialize_fresh_scene_segment_cutout_request(
            source_standard_splat_path=args.source_standard_splat,
            task_freeze_paths=args.task_freeze,
            sweep_freeze_paths_by_task=_cli_mapping(
                args.sweep_freezes_json,
                code="fresh_scene_segment_sweep_inputs_invalid",
            ),
            contribution_manifest_paths_by_task=_cli_mapping(
                args.contribution_manifests_json,
                code="fresh_scene_segment_manifest_inputs_invalid",
            ),
            output_path=args.output,
        )
    elif args.command == "request-artifixer-candidate":
        result = materialize_fresh_scene_artifixer_candidate_request(
            segment_cutout_set_path=args.segment_cutout_set,
            execution_authority_path=args.execution_authority,
            selected_task_ids=args.selected_task_id,
            object_absent_reference_receipt_paths=(
                args.object_absent_reference_receipt
            ),
            output_path=args.output,
        )
    elif args.command == "request-semantic-teacher-edit":
        result = materialize_fresh_scene_semantic_teacher_edit_request(
            source_candidate_inputs_receipt_path=(
                args.source_candidate_inputs_receipt
            ),
            backend_registry_path=args.backend_registry,
            backend_id=args.backend_id,
            rights_attestation_path=args.rights_attestation,
            selected_task_ids=args.selected_task_id,
            object_absent_reference_receipt_paths=(
                args.object_absent_reference_receipt
            ),
            prompt_policy=args.prompt_policy,
            output_format=args.output_format,
            retry_count=args.retry_count,
            output_path=args.output,
        )
    elif args.command == "build":
        result = materialize_fresh_scene_supervisor_bindings(
            preparation_status_path=args.preparation_status,
            sam31_task_input_request_path=args.sam31_task_input_request,
            calibrated_mask_request_path=args.calibrated_mask_request,
            removal_freeze_request_path=args.removal_freeze_request,
            segment_cutout_request_path=args.segment_cutout_request,
            artifixer_candidate_request_path=args.artifixer_candidate_request,
            semantic_teacher_edit_request_path=args.semantic_teacher_edit_request,
            output_path=args.output,
        )
    else:
        bindings = compile_fresh_scene_supervisor_bindings(args.binding_manifest)
        inspection = _read_object(
            Path(args.control_plane_inspection).expanduser().resolve(),
            code="fresh_scene_control_plane_inspection_invalid",
        )
        result = run_capture_reconstruction_supervisor_continuation(
            capture_root=args.capture_root,
            control_plane_inspection=inspection,
            requested_tool_ids=bindings["requested_tool_ids"],
            context_bindings=bindings["context_bindings"],
            allow_live_agents_sdk=args.allow_live_agents_sdk,
            agent_inference_budget_usd=args.agent_inference_budget_usd,
            source_commit_sha=args.source_commit,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "FreshSceneSupervisorBindingError",
    "compile_fresh_scene_supervisor_bindings",
    "materialize_fresh_scene_artifixer_candidate_request",
    "materialize_fresh_scene_removal_freeze_request",
    "materialize_fresh_scene_segment_cutout_request",
    "materialize_fresh_scene_semantic_teacher_edit_request",
    "materialize_fresh_scene_supervisor_bindings",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
