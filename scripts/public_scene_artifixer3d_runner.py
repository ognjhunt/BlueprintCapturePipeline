#!/usr/bin/env python3
"""Execute one sealed object-free ArtiFixer/3D/3D+ candidate packet."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from blueprint_pipeline.image_editor_backend_registry import registered_backend_ids
import time
from typing import Any, Mapping, Sequence
import zipfile


MANIFEST_SCHEMA = "public_scene_artifixer3d_bundle.v1"
REQUEST_SCHEMA = "public_scene_artifixer3d_runtime_request.v1"
RESULT_SCHEMA = "public_scene_artifixer3d_runtime_result.v1"
TASK_PROGRESS_SCHEMA = "public_scene_artifixer3d_task_progress.v1"
TASK_PROGRESS_FILENAME = "public_scene_artifixer3d_task_progress.json"
LEGACY_INPUT_SCHEMA = "public_scene_artifixer3d_candidate_inputs.v3"
LEGACY_INPUT_FILENAME = f"{LEGACY_INPUT_SCHEMA}.json"
DUAL_TARGET_INPUT_SCHEMA = "public_scene_artifixer3d_dual_target_inputs.v1"
DUAL_TARGET_INPUT_FILENAME = f"{DUAL_TARGET_INPUT_SCHEMA}.json"
DUAL_TARGET_PIPELINE_MODE = "dual_target_artifixer3d_only"
DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE = "dual_target_artifixer3d_render_only"
CHECKPOINT_REUSE_SCHEMA = "public_scene_artifixer3d_checkpoint_reuse.v1"
NATIVE_APPEARANCE_EXPORT_SCHEMA = "public_scene_artifixer3d_native_appearance_export.v1"
DUAL_TARGET_PHASES = [
    "dual_target_input_validation",
    "artifixer3d_distillation",
    "artifixer3d_review_render",
    "native_appearance_export",
    "external_visual_and_multiview_review",
]
DUAL_TARGET_RENDER_ONLY_PHASES = [
    "reused_checkpoint_validation",
    "deterministic_distillation_input_replay",
    "artifixer3d_review_render",
    "native_appearance_export",
    "external_visual_and_multiview_review",
]
DUAL_TARGET_LOSS_OVERRIDES = {
    "loss.use_ssim": False,
    "loss.use_l1": True,
    "loss.lambda_l1": 1.0,
    "loss.use_l2": False,
    "loss.use_lpips_override": True,
    "loss.lambda_lpips_override": 0.1,
    "loss.lambda_reconlosses_override": 0.0,
}
# Read from the registry rather than a second copy of the same literals: this
# module and the bundle module each had their own set, so admitting a backend in
# one and not the other was a silent disagreement waiting to happen.
DIRECT_EDITOR_BACKENDS = set(registered_backend_ids())
SEMANTIC_EDITOR_PROMPT = (
    "Reconstruct the natural empty background where the solid black masked hole "
    "appears. Continue the surrounding floor, wall, cabinet, desk, curtain, and "
    "their edges, texture, lighting, reflections, and perspective as appropriate. "
    "The removed foreground object is absent. Add no replacement object, furniture, "
    "text, decoration, silhouette, patch, blank panel, or solid-color shape. Preserve "
    "the rest of the photograph exactly."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = json.loads(json.dumps(dict(value)))
    payload.pop(field, None)
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _task_progress(
    *,
    base: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    expected_task_count: int,
) -> dict[str, Any]:
    completed_task_ids = [str(task["task_id"]) for task in tasks]
    result: dict[str, Any] = {
        "schema_version": TASK_PROGRESS_SCHEMA,
        "status": (
            "all_tasks_completed_unreviewed"
            if len(tasks) == expected_task_count
            else "partial_tasks_completed_unreviewed"
        ),
        "runtime_request_digest": base["runtime_request_digest"],
        "manifest_digest": base["manifest_digest"],
        "candidate_input_receipt_digest": base["candidate_input_receipt_digest"],
        "expected_task_count": expected_task_count,
        "completed_task_count": len(tasks),
        "completed_task_ids": completed_task_ids,
        "tasks": list(tasks),
        "semantic_object_free_review_passed": False,
        "multiview_consistency_review_passed": False,
        "physical_or_deployment_evidence": False,
        "claim_boundary": "completed_task_outputs_are_unreviewed_generated_candidates",
    }
    result["progress_digest"] = _canonical_digest(result, "progress_digest")
    return result


def _read_task_progress(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        result = _read(path, "artifixer3d_task_progress_unreadable")
    except ValueError:
        return None
    tasks = result.get("tasks")
    completed_task_ids = result.get("completed_task_ids")
    if (
        result.get("schema_version") != TASK_PROGRESS_SCHEMA
        or result.get("progress_digest") != _canonical_digest(result, "progress_digest")
        or not isinstance(tasks, list)
        or not isinstance(completed_task_ids, list)
        or result.get("completed_task_count") != len(tasks)
        or completed_task_ids
        != [str(task.get("task_id")) for task in tasks if isinstance(task, Mapping)]
        or any(not _completed_task_is_bound(task) for task in tasks)
        or result.get("semantic_object_free_review_passed") is not False
        or result.get("multiview_consistency_review_passed") is not False
        or result.get("physical_or_deployment_evidence") is not False
    ):
        return None
    return result


def _completed_task_is_bound(task: Any) -> bool:
    if not isinstance(task, Mapping):
        return False
    pipeline_mode = task.get("pipeline_mode")
    if pipeline_mode in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }:
        dual_bound = (
            task.get("outside_support_invariance_status") == "deferred_until_final_soft_composite"
            and task.get("outside_exact_support_invariance_proven") is False
            and task.get("outside_support_changed_pixels_total") is None
        )
        if pipeline_mode == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE:
            return (
                dual_bound
                and task.get("checkpoint_reused") is True
                and task.get("training_executed") is False
                and task.get("direct_artifixer_executed") is False
                and task.get("artifixer3d_plus_executed") is False
                and str(task.get("checkpoint_reuse_digest") or "").startswith("sha256:")
            )
        return dual_bound
    return task.get("outside_support_changed_pixels_total") == 0


def _bound(root: Path, record: Any, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    relative = Path(str(record.get("relative_path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(code)
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or root.resolve() not in path.resolve().parents
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _dual_target_request_is_bound(request: Mapping[str, Any]) -> bool:
    artifixer3d = request.get("artifixer3d")
    return (
        request.get("pipeline_mode") == DUAL_TARGET_PIPELINE_MODE
        and request.get("direct_editor_backend") == "none"
        and request.get("semantic_editor") is None
        and request.get("semantic_editor_only") is False
        and request.get("model") is None
        and request.get("wan_base") is None
        and request.get("direct_inference") is None
        and request.get("direct_model_weights_required") is False
        and request.get("phases") == DUAL_TARGET_PHASES
        and request.get("outside_exact_support_changed_pixels_permitted")
        == "unconstrained_for_raw_representation_review"
        and request.get("outside_support_invariance_gate") == "deferred_until_final_soft_composite"
        and isinstance(artifixer3d, Mapping)
        and artifixer3d.get("loss_overrides") == DUAL_TARGET_LOSS_OVERRIDES
        and artifixer3d.get("anchor_mask_reduction") == "full_frame_mean"
        and isinstance(artifixer3d.get("steps"), int)
        and not isinstance(artifixer3d.get("steps"), bool)
        and artifixer3d["steps"] > 0
        and isinstance(artifixer3d.get("config_name"), str)
        and bool(artifixer3d["config_name"])
    )


def _same_file_record(left: Any, right: Any) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("size_bytes") == right.get("size_bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _render_only_request_is_bound(
    request: Mapping[str, Any], *, candidate: Mapping[str, Any], input_root: Path
) -> bool:
    """Validate the zero-closed checkpoint lineage and every reused byte."""

    artifixer3d = request.get("artifixer3d")
    reuse = artifixer3d.get("checkpoint_reuse") if isinstance(artifixer3d, Mapping) else None
    if (
        request.get("pipeline_mode") != DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
        or request.get("direct_editor_backend") != "none"
        or request.get("semantic_editor") is not None
        or request.get("semantic_editor_only") is not False
        or request.get("model") is not None
        or request.get("wan_base") is not None
        or request.get("direct_inference") is not None
        or request.get("direct_model_weights_required") is not False
        or request.get("phases") != DUAL_TARGET_RENDER_ONLY_PHASES
        or request.get("outside_exact_support_changed_pixels_permitted")
        != "unconstrained_for_raw_representation_review"
        or request.get("outside_support_invariance_gate") != "deferred_until_final_soft_composite"
        or not isinstance(artifixer3d, Mapping)
        or artifixer3d.get("loss_overrides") != DUAL_TARGET_LOSS_OVERRIDES
        or artifixer3d.get("anchor_mask_reduction") != "full_frame_mean"
        or artifixer3d.get("training_permitted") is not False
        or artifixer3d.get("distillation_input_replay_only") is not True
        or not isinstance(artifixer3d.get("steps"), int)
        or isinstance(artifixer3d.get("steps"), bool)
        or artifixer3d["steps"] <= 0
        or not isinstance(artifixer3d.get("config_name"), str)
        or not artifixer3d["config_name"]
        or not isinstance(reuse, Mapping)
        or reuse.get("schema_version") != CHECKPOINT_REUSE_SCHEMA
        or reuse.get("reuse_digest") != _canonical_digest(reuse, "reuse_digest")
        or reuse.get("source_pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
        or reuse.get("source_candidate_input_receipt_digest") != candidate.get("receipt_digest")
        or reuse.get("training_reexecution_permitted") is not False
        or reuse.get("direct_inference_permitted") is not False
        or reuse.get("artifixer3d_plus_permitted") is not False
        or reuse.get("provider_zero_confirmed_before_reuse") is not True
    ):
        return False

    try:
        authority_path = _bound(
            input_root,
            reuse.get("source_attempt_authority"),
            "artifixer3d_checkpoint_reuse_authority_unbound",
        )
        attempt_path = _bound(
            input_root,
            reuse.get("source_attempt_result"),
            "artifixer3d_checkpoint_reuse_attempt_unbound",
        )
        zero_path = _bound(
            input_root,
            reuse.get("source_provider_zero"),
            "artifixer3d_checkpoint_reuse_zero_unbound",
        )
        runtime_path = _bound(
            input_root,
            reuse.get("source_runtime_result"),
            "artifixer3d_checkpoint_reuse_runtime_unbound",
        )
        authority = _read(authority_path, "artifixer3d_checkpoint_reuse_authority_invalid")
        attempt = _read(attempt_path, "artifixer3d_checkpoint_reuse_attempt_invalid")
        zero = _read(zero_path, "artifixer3d_checkpoint_reuse_zero_invalid")
        runtime = _read(runtime_path, "artifixer3d_checkpoint_reuse_runtime_invalid")
    except ValueError:
        return False

    authority_digest = authority.get("authorization_digest")
    runtime_tasks = runtime.get("tasks")
    task_ids = [str(task.get("task_id") or "") for task in candidate["tasks"]]
    if (
        authority.get("schema_version") != "public_scene_artifixer3d_paid_attempt_authority.v1"
        or authority_digest != _canonical_digest(authority, "authorization_digest")
        or reuse.get("source_attempt_authority_digest") != authority_digest
        or attempt.get("schema_version") != "public_scene_artifixer3d_vast_run.v1"
        or attempt.get("status") not in {"blocked", "completed"}
        or reuse.get("source_attempt_terminal_status") != attempt.get("status")
        or attempt.get("authorization_consumption", {}).get("status") != "consumed"
        or attempt.get("authorization_consumption", {}).get("authorization_digest")
        != authority_digest
        or attempt.get("continuing_spend_from_this_run") is not False
        or attempt.get("all_staged_objects_absent") is not True
        or zero.get("schema_version") != "artifixer3d_postblocked_provider_zero.v1"
        or zero.get("receipt_digest") != _canonical_digest(zero, "receipt_digest")
        or zero.get("attempt_authority_digest") != authority_digest
        or zero.get("attempt_terminal_status") != attempt.get("status")
        or zero.get("provider_zero_confirmed") is not True
        or zero.get("continuing_spend_from_attempt") is not False
        or zero.get("all_staged_objects_absent") is not True
        or zero.get("inventory", {}).get("api_confirmed") is not True
        or zero.get("inventory", {}).get("live_resource_count") != 0
        or not _same_file_record(
            zero.get("attempt_authority"),
            {
                "size_bytes": authority_path.stat().st_size,
                "sha256": _sha256(authority_path),
            },
        )
        or not _same_file_record(
            zero.get("attempt_result"),
            {
                "size_bytes": attempt_path.stat().st_size,
                "sha256": _sha256(attempt_path),
            },
        )
        or runtime.get("schema_version") != RESULT_SCHEMA
        or runtime.get("status")
        != "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
        or runtime.get("pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
        or runtime.get("candidate_input_receipt_digest") != candidate.get("receipt_digest")
        or runtime.get("task_ids") != task_ids
        or runtime.get("artifixer3d_distillation_executed") is not True
        or runtime.get("artifixer_direct_inference_executed") is not False
        or runtime.get("semantic_editor_inference_executed") is not False
        or runtime.get("artifixer3d_plus_inference_executed") is not False
        or runtime.get("manifest_digest") != reuse.get("source_manifest_digest")
        or runtime.get("runtime_request_digest") != reuse.get("source_runtime_request_digest")
        or attempt.get("manifest_digest") != runtime.get("manifest_digest")
        or attempt.get("runtime_request_digest") != runtime.get("runtime_request_digest")
        or not isinstance(runtime_tasks, list)
        or len(runtime_tasks) != len(task_ids)
    ):
        return False

    checkpoints = reuse.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(task_ids):
        return False
    for task_id, source_task, row in zip(task_ids, runtime_tasks, checkpoints):
        if not isinstance(source_task, Mapping) or not isinstance(row, Mapping):
            return False
        source_checkpoint = source_task.get("artifixer3d_checkpoint")
        try:
            checkpoint = _bound(
                input_root,
                row.get("checkpoint"),
                "artifixer3d_checkpoint_reuse_checkpoint_unbound",
            )
        except ValueError:
            return False
        if (
            source_task.get("task_id") != task_id
            or source_task.get("pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
            or row.get("task_id") != task_id
            or row.get("steps") != artifixer3d.get("steps")
            or not isinstance(row.get("source_provider_zip_member"), str)
            or not row["source_provider_zip_member"]
            or not _same_file_record(source_checkpoint, row.get("checkpoint"))
            or checkpoint.stat().st_size <= 0
        ):
            return False
    return True


def _dual_target_candidate_is_bound(candidate: Mapping[str, Any]) -> bool:
    tasks = candidate.get("tasks")
    if (
        candidate.get("schema_version") != DUAL_TARGET_INPUT_SCHEMA
        or candidate.get("status") != "paired_target_inputs_prepared_no_model_no_execution"
        or candidate.get("pipeline_mode") != DUAL_TARGET_PIPELINE_MODE
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= 5
        or candidate.get("replacement_object_count") != len(tasks)
    ):
        return False
    for task in tasks:
        if not isinstance(task, Mapping):
            return False
        physical_count = task.get("physical_camera_count")
        training_count = task.get("training_record_count")
        frames = task.get("frames")
        selected = task.get("selected_anchor_indices")
        teachers = task.get("semantic_teacher_indices")
        if (
            not isinstance(physical_count, int)
            or isinstance(physical_count, bool)
            or physical_count <= 0
            or training_count != 2 * physical_count
            or not isinstance(frames, list)
            or len(frames) != physical_count
            or not isinstance(selected, list)
            or not isinstance(teachers, list)
            or len(selected) != physical_count
            or len(teachers) != physical_count
        ):
            return False
        frame_anchor_indices: list[int] = []
        frame_teacher_indices: list[int] = []
        for physical_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                return False
            anchor_index = frame.get("anchor_training_index")
            teacher_index = frame.get("semantic_teacher_training_index")
            if (
                frame.get("physical_camera_index") != physical_index
                or anchor_index != 2 * physical_index
                or teacher_index != 2 * physical_index + 1
                or not isinstance(frame.get("camera_id"), str)
                or any(
                    not isinstance(frame.get(field), Mapping)
                    for field in (
                        "anchor_rgb",
                        "exact_repair_mask",
                        "anchor_loss_mask",
                        "semantic_teacher_rgb",
                        "semantic_teacher_override_rgb",
                    )
                )
                or frame.get("teacher_loss_mask_materialized") is not False
                or frame.get("pair_pose_and_intrinsics_exactly_equal") is not True
            ):
                return False
            frame_anchor_indices.append(anchor_index)
            frame_teacher_indices.append(teacher_index)
        if (
            selected != frame_anchor_indices
            or teachers != frame_teacher_indices
            or sorted(selected + teachers) != list(range(training_count))
            or not isinstance(task.get("selected_anchor_indices_file"), Mapping)
            or not isinstance(task.get("review_trajectory"), Mapping)
        ):
            return False
    return True


def _validate_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = root / "provider_runtime"
    manifest = _read(
        runtime / "artifixer3d_bundle_manifest.json", "artifixer3d_manifest_unreadable"
    )
    request = _read(runtime / "artifixer3d_runtime_request.json", "artifixer3d_request_unreadable")
    render_only = request.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    dual_target = request.get("pipeline_mode") in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }
    candidate_filename = DUAL_TARGET_INPUT_FILENAME if dual_target else LEGACY_INPUT_FILENAME
    expected_candidate_schema = DUAL_TARGET_INPUT_SCHEMA if dual_target else LEGACY_INPUT_SCHEMA
    candidate = _read(
        runtime / "input" / candidate_filename,
        "artifixer3d_candidate_unreadable",
    )
    attestation = _read(
        runtime / "artifixer3d_use_attestation.json",
        "artifixer3d_use_attestation_unreadable",
    )
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("manifest_digest") != _canonical_digest(manifest, "manifest_digest")
        or request.get("schema_version") != REQUEST_SCHEMA
        or request.get("runtime_request_digest")
        != _canonical_digest(request, "runtime_request_digest")
        or candidate.get("schema_version") != expected_candidate_schema
        or candidate.get("receipt_digest") != _canonical_digest(candidate, "receipt_digest")
        or manifest.get("runtime_request", {}).get("runtime_request_digest")
        != request["runtime_request_digest"]
        or manifest.get("candidate_input_receipt", {}).get("receipt_digest")
        != candidate["receipt_digest"]
        or request.get("candidate_input_receipt_digest") != candidate["receipt_digest"]
        or manifest.get("contains_raw_dataset_bytes") is not False
        or manifest.get("contains_model_weights") is not render_only
        or manifest.get("contains_reused_private_derived_3dgrut_checkpoint", False)
        is not render_only
        or manifest.get("contains_released_direct_model_weights", False) is not False
        or request.get("source_object_restoration_permitted") is not False
        or (
            request.get("pipeline_mode") == DUAL_TARGET_PIPELINE_MODE
            and not _dual_target_request_is_bound(request)
        )
        or (
            render_only
            and not _render_only_request_is_bound(
                request,
                candidate=candidate,
                input_root=runtime / "input",
            )
        )
        or (
            not dual_target
            and (
                request.get("outside_exact_support_changed_pixels_permitted") != 0
                or request.get("direct_editor_backend") not in DIRECT_EDITOR_BACKENDS
            )
        )
        or (dual_target and not _dual_target_candidate_is_bound(candidate))
        or manifest.get("blueprint_source_identity") != request.get("blueprint_source_identity")
        or attestation.get("attestation_digest")
        != _canonical_digest(attestation, "attestation_digest")
        or manifest.get("use_attestation", {}).get("attestation_digest")
        != attestation.get("attestation_digest")
        or request.get("use_attestation", {}).get("attestation_digest")
        != attestation.get("attestation_digest")
        or attestation.get("internal_noncommercial_research_and_development_only") is not True
        or attestation.get("private_derived_input_upload_authorized") is not True
        or attestation.get("raw_dataset_bytes_upload_authorized") is not False
        or attestation.get("provider_training_authorized") is not False
        or attestation.get("commercial_use_authorized") is not False
        or attestation.get("redistribution_authorized") is not False
        or attestation.get("publication_authorized") is not False
    ):
        raise ValueError("artifixer3d_bundle_binding_invalid")
    backend = request["direct_editor_backend"]
    if manifest.get("direct_editor_backend") != backend:
        raise ValueError("artifixer3d_bundle_binding_invalid")
    semantic = request.get("semantic_editor")
    if dual_target:
        if (
            backend != "none"
            or manifest.get("pipeline_mode") != request.get("pipeline_mode")
            or manifest.get("direct_editor_backend") != "none"
            or manifest.get("semantic_editor_model_identity") is not None
            or (
                render_only
                and manifest.get("checkpoint_reuse")
                != request.get("artifixer3d", {}).get("checkpoint_reuse")
            )
            or (not render_only and manifest.get("checkpoint_reuse") is not None)
        ):
            raise ValueError("artifixer3d_dual_target_binding_invalid")
    elif backend != "artifixer":
        if (
            not isinstance(semantic, Mapping)
            or semantic.get("backend") != backend
            or semantic.get("license") != "Apache-2.0"
            or semantic.get("output_must_be_exact_support_composited") is not True
            or manifest.get("semantic_editor_model_identity") != semantic
        ):
            raise ValueError("artifixer3d_semantic_editor_binding_invalid")
        if backend == "vibe_image_edit" and (
            request.get("semantic_editor_only") is not True
            or semantic.get("enable_model_cpu_offload") is not False
            or not isinstance(semantic.get("source"), Mapping)
        ):
            raise ValueError("artifixer3d_semantic_editor_binding_invalid")
        if not isinstance(request.get("semantic_editor_only"), bool):
            raise ValueError("artifixer3d_semantic_editor_binding_invalid")
    elif semantic is not None or manifest.get("semantic_editor_model_identity") is not None:
        raise ValueError("artifixer3d_semantic_editor_binding_invalid")
    for row in manifest.get("candidate_files") or []:
        _bound(runtime / "input", row, "artifixer3d_candidate_file_invalid")
    for row in manifest.get("source_files") or []:
        _bound(runtime / "ArtiFixer_official", row, "artifixer3d_source_file_invalid")
    if (
        _bound(
            root,
            manifest.get("use_attestation"),
            "artifixer3d_use_attestation_unbound",
        )
        != runtime / "artifixer3d_use_attestation.json"
    ):
        raise ValueError("artifixer3d_use_attestation_unbound")
    return manifest, request, candidate


def _verify_inventory(root: Path, rows: Sequence[Mapping[str, Any]], code: str) -> None:
    for row in rows:
        relative = Path(str(row.get("path") or ""))
        path = root / relative
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ValueError(code + ":" + relative.as_posix())


def _download_models(request: Mapping[str, Any], cache: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    model = request["model"]
    wan = request["wan_base"]
    checkpoint_dir = cache / "artifixer"
    wan_dir = cache / "wan"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wan_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        hf_hub_download(
            repo_id=model["repository"],
            revision=model["revision"],
            filename=model["files"][0]["path"],
            local_dir=checkpoint_dir,
        )
    )
    snapshot_download(
        repo_id=wan["repository"],
        revision=wan["revision"],
        allow_patterns=[row["path"] for row in wan["files"]],
        local_dir=wan_dir,
    )
    _verify_inventory(checkpoint_dir, model["files"], "artifixer3d_checkpoint_invalid")
    _verify_inventory(wan_dir, wan["files"], "artifixer3d_wan_runtime_invalid")
    return checkpoint, wan_dir


def _download_semantic_editor(request: Mapping[str, Any], cache: Path) -> Path | None:
    backend = request["direct_editor_backend"]
    if backend == "artifixer":
        return None
    from huggingface_hub import snapshot_download

    semantic = request["semantic_editor"]
    output = cache / backend
    snapshot_download(
        repo_id=semantic["repository"],
        revision=semantic["revision"],
        local_dir=output,
    )
    _verify_inventory(
        output,
        semantic["large_files"],
        "artifixer3d_semantic_editor_model_invalid",
    )
    return output


def _zero_prompt(path: Path) -> None:
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("unconditioned", data=np.zeros((1, 4096), dtype=np.uint16))
        dataset.attrs["caption"] = ""


def _run(command: Sequence[str], *, cwd: Path, log: Path, timeout: int) -> None:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "COMMAND "
        + " ".join(command)
        + "\n"
        + f"DURATION_SECONDS {time.monotonic() - started:.6f}\n"
        + (completed.stdout or "")
        + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ValueError(
            f"artifixer3d_command_failed:{Path(command[1]).name if len(command) > 1 else 'command'}"
        )


def _materialize_split(template_path: Path, output_path: Path) -> dict[str, Any]:
    template = _read(template_path, "artifixer3d_split_template_invalid")
    split = template.get("upstream_split")
    if not isinstance(split, Mapping) or set(split) != {"test"}:
        raise ValueError("artifixer3d_split_template_invalid")
    _write(output_path, split)
    return dict(split)


def _prediction_dir(save_root: Path, task_id: str) -> Path:
    matches = sorted(save_root.glob(f"**/{task_id}/frames/batch_0000/pred"))
    if len(matches) != 1 or not matches[0].is_dir():
        raise ValueError("artifixer3d_prediction_directory_invalid")
    return matches[0]


def _exact_composite(
    *, retained: Path, mask: Path, prediction: Path, output: Path
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    with Image.open(retained) as image:
        before = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask) as image:
        support = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    with Image.open(prediction) as image:
        generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if before.shape != generated.shape or before.shape[:2] != support.shape:
        raise ValueError("artifixer3d_composite_shape_invalid")
    composite = before.copy()
    composite[support] = generated[support]
    outside = ~support
    outside_changes = int(np.count_nonzero(np.any(composite[outside] != before[outside], axis=1)))
    if outside_changes != 0:
        raise ValueError("artifixer3d_outside_support_change")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite, mode="RGB").save(output)
    return {
        "path": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "repair_pixel_count": int(np.count_nonzero(support)),
        "outside_support_changed_pixels": outside_changes,
    }


def _copy_scene(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError("artifixer3d_repaired_scene_exists")
    shutil.copytree(source, destination, symlinks=False)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


class _CheckpointExportModel:
    """Minimal CPU adapter for the pinned 3DGRUT exporter interface."""

    _TENSOR_FIELDS = (
        "positions",
        "rotation",
        "scale",
        "density",
        "features_albedo",
        "features_specular",
    )

    def __init__(self, checkpoint: Mapping[str, Any]) -> None:
        missing = [name for name in self._TENSOR_FIELDS if name not in checkpoint]
        if missing:
            raise ValueError("artifixer3d_native_export_checkpoint_fields_missing")
        for name in self._TENSOR_FIELDS:
            tensor = checkpoint[name]
            if not hasattr(tensor, "shape") or not hasattr(tensor, "detach"):
                raise ValueError("artifixer3d_native_export_checkpoint_tensor_invalid")
            setattr(self, name, tensor)
        count = int(self.positions.shape[0])
        if (
            count <= 0
            or tuple(self.positions.shape) != (count, 3)
            or tuple(self.rotation.shape) != (count, 4)
            or tuple(self.scale.shape) != (count, 3)
            or tuple(self.density.shape) != (count, 1)
            or tuple(self.features_albedo.shape) != (count, 3)
            or int(self.features_specular.shape[0]) != count
        ):
            raise ValueError("artifixer3d_native_export_checkpoint_shape_invalid")
        self.max_n_features = int(checkpoint.get("max_n_features", -1))
        self.n_active_features = int(checkpoint.get("n_active_features", -1))
        expected_specular = ((self.max_n_features + 1) ** 2 - 1) * 3
        if (
            not 0 <= self.n_active_features <= self.max_n_features
            or self.max_n_features < 0
            or tuple(self.features_specular.shape) != (count, expected_specular)
        ):
            raise ValueError("artifixer3d_native_export_checkpoint_features_invalid")

    def get_positions(self):
        return self.positions

    def get_max_n_features(self) -> int:
        return self.max_n_features

    def get_n_active_features(self) -> int:
        return self.n_active_features

    def get_scale(self, preactivation: bool = False):
        if not preactivation:
            raise ValueError("artifixer3d_native_export_activation_ambiguous")
        return self.scale

    def get_rotation(self, preactivation: bool = False):
        if not preactivation:
            raise ValueError("artifixer3d_native_export_activation_ambiguous")
        return self.rotation

    def get_density(self, preactivation: bool = False):
        if not preactivation:
            raise ValueError("artifixer3d_native_export_activation_ambiguous")
        return self.density

    def get_features_albedo(self):
        return self.features_albedo

    def get_features_specular(self):
        return self.features_specular


def _export_checkpoint_native_appearance(*, checkpoint: Path, task_output: Path) -> dict[str, Any]:
    """Serialize one bound checkpoint to standard PLY and Isaac-ready USDZ.

    The trained coordinates are retained verbatim.  In particular, the pinned
    USDZ exporter's camera-derived recenter/upright transform is disabled.
    """

    import torch
    from threedgrut.export.ply_exporter import PLYExporter
    from threedgrut.export.usdz_exporter import USDZExporter

    output_root = task_output / "native_appearance"
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("artifixer3d_native_export_destination_exists")
    output_root.mkdir(parents=True)
    try:
        checkpoint_value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("artifixer3d_native_export_checkpoint_unreadable") from exc
    if not isinstance(checkpoint_value, Mapping):
        raise ValueError("artifixer3d_native_export_checkpoint_invalid")
    config = checkpoint_value.get("config")
    try:
        config.export_usdz.apply_normalizing_transform = False
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("artifixer3d_native_export_config_invalid") from exc
    model = _CheckpointExportModel(checkpoint_value)
    ply_path = output_root / "repaired_scene.ply"
    usdz_path = output_root / "repaired_scene.usdz"
    PLYExporter().export(model, ply_path, dataset=None, conf=config)
    USDZExporter().export(model, usdz_path, dataset=None, conf=config)
    usdz_members = _align_and_validate_usdz(usdz_path)
    if any(
        path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        for path in (ply_path, usdz_path)
    ):
        raise ValueError("artifixer3d_native_export_output_invalid")
    result = {
        "schema_version": NATIVE_APPEARANCE_EXPORT_SCHEMA,
        "status": "native_appearance_candidates_exported_pending_native_import_and_multiview_review",
        "source_checkpoint": _file_record(checkpoint),
        "gaussian_count": int(model.positions.shape[0]),
        "coordinate_contract": {
            "source_gaussian_tensor_coordinates_preserved": True,
            "camera_derived_normalizing_transform_applied": False,
            "standard_gaussian_ply_transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "isaac_nurec_usdz_wrapper_transform_matrix": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "usdz_wrapper_transform_role": ("fixed_pinned_3dgrut_to_usd_axis_convention_only"),
        },
        "standard_gaussian_ply": _file_record(ply_path),
        "isaac_nurec_usdz": _file_record(usdz_path),
        "isaac_nurec_usdz_archive_contract": {
            "compression": "stored",
            "payload_alignment_bytes": 64,
            "all_payload_offsets_aligned": True,
            "nurec_gzip_mtime_normalized_to_zero": True,
            "members": usdz_members,
        },
        "usdz_tensor_precision": "float16_pinned_upstream_exporter",
        "generated_output_is_capture_or_physical_evidence": False,
        "native_import_qualified": False,
    }
    result["export_digest"] = _canonical_digest(result, "export_digest")
    del checkpoint_value
    return result


def _align_and_validate_usdz(path: Path) -> list[dict[str, Any]]:
    """Repack a USDZ with stored, 64-byte-aligned member payloads."""

    try:
        with zipfile.ZipFile(path, "r") as source:
            infos = source.infolist()
            names = [info.filename for info in infos]
            if (
                not infos
                or len(names) != len(set(names))
                or any(info.is_dir() or info.flag_bits & 0x1 for info in infos)
            ):
                raise ValueError
            members = [(info.filename, source.read(info)) for info in infos]
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("artifixer3d_native_export_usdz_invalid") from exc
    temporary = path.with_name(path.name + ".aligned.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("artifixer3d_native_export_usdz_temporary_exists")
    try:
        with temporary.open("wb") as handle:
            with zipfile.ZipFile(
                handle,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                for name, body in members:
                    if name.endswith(".nurec"):
                        if len(body) < 10 or body[:3] != b"\x1f\x8b\x08":
                            raise ValueError("artifixer3d_native_export_nurec_invalid")
                        body = body[:4] + b"\0\0\0\0" + body[8:]
                    info = zipfile.ZipInfo(name)
                    info.compress_type = zipfile.ZIP_STORED
                    header_size = 30 + len(name.encode("utf-8"))
                    padding = (-(handle.tell() + header_size)) % 64
                    if padding:
                        if padding < 4:
                            padding += 64
                        info.extra = struct.pack("<HH", 0x1986, padding - 4) + b"\0" * (padding - 4)
                    archive.writestr(info, body)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    rows: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle, zipfile.ZipFile(path, "r") as archive:
            for info in archive.infolist():
                handle.seek(info.header_offset)
                header = handle.read(30)
                if len(header) != 30:
                    raise ValueError
                fields = struct.unpack("<IHHHHHIIIHH", header)
                data_offset = info.header_offset + 30 + fields[-2] + fields[-1]
                if info.compress_type != zipfile.ZIP_STORED or data_offset % 64:
                    raise ValueError
                rows.append(
                    {
                        "filename": info.filename,
                        "size_bytes": info.file_size,
                        "data_offset_bytes": data_offset,
                        "sha256": "sha256:" + hashlib.sha256(archive.read(info)).hexdigest(),
                    }
                )
    except (OSError, ValueError, zipfile.BadZipFile, struct.error) as exc:
        raise ValueError("artifixer3d_native_export_usdz_alignment_invalid") from exc
    return rows


def _hydra_value(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return str(value)


def _prepare_dual_target_teacher_frames(
    *, task: Mapping[str, Any], staged_task: Path, task_output: Path
) -> tuple[Path, list[dict[str, Any]]]:
    """Bind and stage only the seamless teacher frames for odd training records."""
    from PIL import Image

    teacher_root = task_output / "paired_semantic_teachers"
    if teacher_root.exists():
        raise ValueError("artifixer3d_dual_target_teacher_root_exists")
    teacher_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for frame in task["frames"]:
        physical_index = int(frame["physical_camera_index"])
        teacher_index = int(frame["semantic_teacher_training_index"])
        original = _bound(
            staged_task,
            frame["anchor_rgb"],
            "artifixer3d_dual_target_original_unbound",
        )
        teacher = _bound(
            staged_task,
            frame["semantic_teacher_override_rgb"],
            "artifixer3d_dual_target_teacher_unbound",
        )
        teacher_copy = _bound(
            staged_task,
            frame["semantic_teacher_rgb"],
            "artifixer3d_dual_target_teacher_unbound",
        )
        if _sha256(teacher) != _sha256(teacher_copy):
            raise ValueError("artifixer3d_dual_target_teacher_copy_mismatch")
        with Image.open(original) as original_image, Image.open(teacher) as teacher_image:
            if original_image.size != teacher_image.size:
                raise ValueError("artifixer3d_dual_target_teacher_shape_invalid")
            image_size = list(original_image.size)
        output = teacher_root / f"{teacher_index:05d}.png"
        shutil.copyfile(teacher, output)
        rows.append(
            {
                "frame_index": physical_index,
                "camera_id": frame["camera_id"],
                "semantic_teacher_training_index": teacher_index,
                "image_size": image_size,
                "source": _file_record(teacher),
                "staged": _file_record(output),
            }
        )
    if sorted(path.name for path in teacher_root.iterdir()) != [
        f"{int(index):05d}.png" for index in task["semantic_teacher_indices"]
    ]:
        raise ValueError("artifixer3d_dual_target_teacher_coverage_invalid")
    return teacher_root, rows


def _stage_dual_target_anchor_masks(
    *, task: Mapping[str, Any], staged_task: Path, distillation_input_dir: Path
) -> list[dict[str, Any]]:
    """Stage exact-sized binary trust masks next to selected anchor images."""
    import numpy as np
    from PIL import Image

    image_root = distillation_input_dir / "images"
    rows: list[dict[str, Any]] = []
    for frame in task["frames"]:
        anchor_index = int(frame["anchor_training_index"])
        matches = [
            path
            for path in image_root.glob(f"frame_{anchor_index:05d}.*")
            if not path.name.endswith("_mask.png")
        ]
        if len(matches) != 1 or matches[0].is_symlink() is False:
            raise ValueError("artifixer3d_dual_target_anchor_image_invalid")
        anchor = matches[0]
        mask = _bound(
            staged_task,
            frame["anchor_loss_mask"],
            "artifixer3d_dual_target_anchor_mask_unbound",
        )
        exact_mask = _bound(
            staged_task,
            frame["exact_repair_mask"],
            "artifixer3d_dual_target_exact_mask_unbound",
        )
        with Image.open(anchor) as anchor_image:
            anchor_size = anchor_image.size
        with Image.open(mask) as mask_image:
            if mask_image.mode != "L" or mask_image.size != anchor_size:
                raise ValueError("artifixer3d_dual_target_anchor_mask_shape_invalid")
            trust = np.asarray(mask_image, dtype=np.uint8)
        with Image.open(exact_mask) as exact_image:
            if exact_image.size != anchor_size:
                raise ValueError("artifixer3d_dual_target_exact_mask_shape_invalid")
            exact_support = np.asarray(exact_image.convert("L"), dtype=np.uint8) > 0
        if not np.all((trust == 0) | (trust == 255)):
            raise ValueError("artifixer3d_dual_target_anchor_mask_not_binary")
        if np.any(trust[exact_support] != 0):
            raise ValueError("artifixer3d_dual_target_anchor_mask_misses_exact_support")
        output = anchor.with_name(anchor.stem + "_mask.png")
        if output.exists() or output.is_symlink():
            raise ValueError("artifixer3d_dual_target_anchor_mask_destination_exists")
        shutil.copyfile(mask, output)
        rows.append(
            {
                "physical_camera_index": int(frame["physical_camera_index"]),
                "camera_id": frame["camera_id"],
                "anchor_training_index": anchor_index,
                "trusted_pixel_count": int(np.count_nonzero(trust)),
                "excluded_pixel_count": int(trust.size - np.count_nonzero(trust)),
                "source": _file_record(mask),
                "staged": _file_record(output),
            }
        )
    staged_masks = sorted(image_root.glob("frame_*_mask.png"))
    if len(staged_masks) != len(task["selected_anchor_indices"]):
        raise ValueError("artifixer3d_dual_target_anchor_mask_coverage_invalid")
    if any(
        (image_root / f"frame_{int(index):05d}_mask.png").exists()
        for index in task["semantic_teacher_indices"]
    ):
        raise ValueError("artifixer3d_dual_target_teacher_mask_forbidden")
    return rows


def _normalize_dual_target_review_frames(
    *,
    task: Mapping[str, Any],
    staged_task: Path,
    review_dir: Path,
    task_output: Path,
) -> list[dict[str, Any]]:
    """Byte-copy the native 3DGRUT renders into the provider-retained layout."""
    from PIL import Image

    frames = task["frames"]
    camera_count = int(task["physical_camera_count"])
    if len(frames) != camera_count or [
        int(frame["physical_camera_index"]) for frame in frames
    ] != list(range(camera_count)):
        raise ValueError("artifixer3d_dual_target_review_camera_order_invalid")

    render_root = review_dir / "renders"
    if review_dir.is_symlink() or render_root.is_symlink() or not render_root.is_dir():
        raise ValueError("artifixer3d_dual_target_review_frame_set_invalid")
    expected_names = [f"{index:05d}.png" for index in range(camera_count)]
    rendered_entries = sorted(render_root.iterdir(), key=lambda path: path.name)
    if [path.name for path in rendered_entries] != expected_names or any(
        path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        for path in rendered_entries
    ):
        raise ValueError("artifixer3d_dual_target_review_frame_set_invalid")

    normalized_root = task_output / "artifixer3d_review_frames"
    if normalized_root.exists() or normalized_root.is_symlink():
        raise ValueError("artifixer3d_dual_target_review_destination_exists")
    normalized_root.mkdir()
    rows: list[dict[str, Any]] = []
    for frame, rendered in zip(frames, rendered_entries):
        index = int(frame["physical_camera_index"])
        anchor = _bound(
            staged_task,
            frame["anchor_rgb"],
            "artifixer3d_dual_target_anchor_rgb_unbound",
        )
        try:
            with Image.open(anchor) as anchor_image:
                expected_size = anchor_image.size
            with Image.open(rendered) as rendered_image:
                if rendered_image.format != "PNG":
                    raise ValueError
                rendered_size = rendered_image.size
                rendered_image.verify()
        except (OSError, SyntaxError, ValueError) as exc:
            raise ValueError("artifixer3d_dual_target_review_frame_invalid") from exc
        if rendered_size != expected_size:
            raise ValueError("artifixer3d_dual_target_review_frame_size_invalid")

        normalized = normalized_root / f"{index:05d}.png"
        shutil.copyfile(rendered, normalized)
        if (
            normalized.is_symlink()
            or not normalized.is_file()
            or normalized.stat().st_size != rendered.stat().st_size
            or _sha256(normalized) != _sha256(rendered)
        ):
            raise ValueError("artifixer3d_dual_target_review_frame_copy_invalid")
        rows.append(
            {
                "frame_index": index,
                "camera_id": frame["camera_id"],
                **_file_record(normalized),
            }
        )
    if (
        len(rows) != camera_count
        or sorted(path.name for path in normalized_root.iterdir()) != expected_names
    ):
        raise ValueError("artifixer3d_dual_target_review_coverage_invalid")
    return rows


def _prepare_dual_target_distillation_replay(
    *,
    task: Mapping[str, Any],
    input_root: Path,
    task_output: Path,
    request: Mapping[str, Any],
    log: Path,
) -> dict[str, Any]:
    """Replay the deterministic camera/input preparation shared by train and render."""

    from data_processing import artifixer3d

    task_id = str(task["task_id"])
    staged_task = input_root / task_id
    if task_output.exists():
        raise ValueError("artifixer3d_dual_target_task_output_exists")
    log.parent.mkdir(parents=True)
    _zero_prompt(staged_task / "captions" / "unconditioned_zero_prompt.h5")
    teacher_root, teacher_rows = _prepare_dual_target_teacher_frames(
        task=task,
        staged_task=staged_task,
        task_output=task_output,
    )
    transforms_path = _bound(
        staged_task,
        task["transforms"],
        "artifixer3d_dual_target_transforms_unbound",
    )
    selected_path = _bound(
        staged_task,
        task["selected_anchor_indices_file"],
        "artifixer3d_dual_target_selected_indices_unbound",
    )
    try:
        selected_values = json.loads(selected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("artifixer3d_dual_target_selected_indices_invalid") from exc
    if selected_values != task["selected_anchor_indices"]:
        raise ValueError("artifixer3d_dual_target_selected_indices_invalid")
    review_trajectory = _bound(
        staged_task,
        task["review_trajectory"],
        "artifixer3d_dual_target_review_trajectory_unbound",
    )
    split_path = staged_task / "split.dual_target_distill.json"
    _write(
        split_path,
        {
            "test": {
                task_id: {
                    "transforms_path": transforms_path.relative_to(staged_task).as_posix(),
                    "image_root": ".",
                    "selected_indices_path": selected_path.relative_to(staged_task).as_posix(),
                    "prompt_path": "captions/unconditioned_zero_prompt.h5",
                    "camera_scale": 1.0,
                    "has_gt": False,
                }
            }
        },
    )
    artifixer3d_root = task_output / "artifixer3d"
    scene = artifixer3d.load_prepared_scene(staged_task, split_path, task_id)
    steps = int(request["artifixer3d"]["steps"])
    paths = artifixer3d.artifixer3d_paths(scene, artifixer3d_root, None, steps)
    with log.open("w", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            artifixer3d.materialize_distillation_input(scene, paths, teacher_root)
    anchor_mask_rows = _stage_dual_target_anchor_masks(
        task=task,
        staged_task=staged_task,
        distillation_input_dir=paths.distillation_input_dir,
    )
    return {
        "task_id": task_id,
        "staged_task": staged_task,
        "teacher_rows": teacher_rows,
        "review_trajectory": review_trajectory,
        "scene": scene,
        "steps": steps,
        "paths": paths,
        "anchor_mask_rows": anchor_mask_rows,
    }


def _dual_target_task_runtime(
    *,
    task: Mapping[str, Any],
    input_root: Path,
    source_root: Path,
    output_root: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one paired-target 3DGRUT candidate and stop at raw review renders."""
    from data_processing import artifixer3d
    from data_processing import threedgrut_training

    task_id = str(task["task_id"])
    task_output = output_root / "tasks" / task_id
    log = task_output / "logs" / "artifixer3d_dual_target.log"
    prepared = _prepare_dual_target_distillation_replay(
        task=task,
        input_root=input_root,
        task_output=task_output,
        request=request,
        log=log,
    )
    staged_task = prepared["staged_task"]
    scene = prepared["scene"]
    steps = prepared["steps"]
    paths = prepared["paths"]
    overrides = [
        f"path={paths.distillation_input_dir}",
        f"out_dir={paths.run_root}",
        f"selected_indices_file={paths.distillation_selected_indices_path}",
        f"image_path_override={paths.override_image_dir.name}",
        "test_last=False",
        "export_ingp.enabled=False",
        f"experiment_name={scene.scene_id}",
        f"n_iterations={steps}",
        "use_wandb=False",
        f"checkpoint.iterations=[{steps}]",
    ]
    overrides.extend(
        f"{name}={_hydra_value(value)}"
        for name, value in request["artifixer3d"]["loss_overrides"].items()
    )
    with log.open("a", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            threedgrut_training.train_3dgrut(
                request["artifixer3d"]["config_name"],
                overrides,
                threedgrut_training.DEFAULT_THREEDGRUT_CONFIG_DIR,
            )
    checkpoint = artifixer3d.artifixer3d_checkpoint(scene, paths, steps)
    if not checkpoint.is_file():
        raise ValueError("artifixer3d_checkpoint_missing_or_ambiguous")
    with log.open("a", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            review_dir = artifixer3d.render_artifixer3d(
                scene,
                paths,
                checkpoint=checkpoint,
                checkpoint_reused=False,
                replace=False,
                render_trajectory_path=prepared["review_trajectory"],
            )
    review_rows = _normalize_dual_target_review_frames(
        task=task,
        staged_task=staged_task,
        review_dir=review_dir,
        task_output=task_output,
    )
    if len(review_rows) != task["physical_camera_count"]:
        raise ValueError("artifixer3d_dual_target_review_coverage_invalid")
    native_appearance = _export_checkpoint_native_appearance(
        checkpoint=checkpoint, task_output=task_output
    )
    return {
        "task_id": task_id,
        "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
        "training_record_count": task["training_record_count"],
        "selected_anchor_indices": task["selected_anchor_indices"],
        "semantic_teacher_indices": task["semantic_teacher_indices"],
        "semantic_teacher_frames": prepared["teacher_rows"],
        "anchor_loss_masks": prepared["anchor_mask_rows"],
        "anchor_mask_reduction": request["artifixer3d"]["anchor_mask_reduction"],
        "loss_overrides": request["artifixer3d"]["loss_overrides"],
        "artifixer3d_checkpoint": _file_record(checkpoint),
        "artifixer3d_log_sha256": _sha256(log),
        "artifixer3d_plus_log_sha256": None,
        "artifixer3d_review_frames": review_rows,
        "native_appearance": native_appearance,
        "final_candidate_frames": review_rows,
        "raw_representation_review_only": True,
        "outside_support_invariance_status": "deferred_until_final_soft_composite",
        "outside_exact_support_invariance_proven": False,
        "outside_support_changed_pixels_total": None,
        "semantic_object_free_review_passed": False,
        "multiview_consistency_review_passed": False,
    }


def _dual_target_render_only_task_runtime(
    *,
    task: Mapping[str, Any],
    input_root: Path,
    output_root: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Render one sealed checkpoint without invoking any training or polish pass."""

    from data_processing import artifixer3d

    task_id = str(task["task_id"])
    task_output = output_root / "tasks" / task_id
    log = task_output / "logs" / "artifixer3d_render_only.log"
    prepared = _prepare_dual_target_distillation_replay(
        task=task,
        input_root=input_root,
        task_output=task_output,
        request=request,
        log=log,
    )
    reuse = request["artifixer3d"]["checkpoint_reuse"]
    checkpoint_rows = [row for row in reuse["checkpoints"] if row.get("task_id") == task_id]
    if len(checkpoint_rows) != 1:
        raise ValueError("artifixer3d_checkpoint_reuse_task_ambiguous")
    checkpoint = _bound(
        input_root,
        checkpoint_rows[0]["checkpoint"],
        "artifixer3d_checkpoint_reuse_checkpoint_unbound",
    )
    with log.open("a", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            review_dir = artifixer3d.render_artifixer3d(
                prepared["scene"],
                prepared["paths"],
                checkpoint=checkpoint,
                checkpoint_reused=True,
                replace=False,
                render_trajectory_path=prepared["review_trajectory"],
            )
    review_rows = _normalize_dual_target_review_frames(
        task=task,
        staged_task=prepared["staged_task"],
        review_dir=review_dir,
        task_output=task_output,
    )
    if len(review_rows) != task["physical_camera_count"]:
        raise ValueError("artifixer3d_dual_target_review_coverage_invalid")
    native_appearance = _export_checkpoint_native_appearance(
        checkpoint=checkpoint, task_output=task_output
    )
    return {
        "task_id": task_id,
        "pipeline_mode": DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
        "source_pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
        "checkpoint_reused": True,
        "checkpoint_reuse_digest": reuse["reuse_digest"],
        "training_executed": False,
        "direct_artifixer_executed": False,
        "artifixer3d_plus_executed": False,
        "training_record_count": task["training_record_count"],
        "selected_anchor_indices": task["selected_anchor_indices"],
        "semantic_teacher_indices": task["semantic_teacher_indices"],
        "semantic_teacher_frames": prepared["teacher_rows"],
        "anchor_loss_masks": prepared["anchor_mask_rows"],
        "anchor_mask_reduction": request["artifixer3d"]["anchor_mask_reduction"],
        "loss_overrides": request["artifixer3d"]["loss_overrides"],
        "artifixer3d_checkpoint": _file_record(checkpoint),
        "artifixer3d_log_sha256": _sha256(log),
        "artifixer3d_plus_log_sha256": None,
        "artifixer3d_review_frames": review_rows,
        "native_appearance": native_appearance,
        "final_candidate_frames": review_rows,
        "raw_representation_review_only": True,
        "outside_support_invariance_status": "deferred_until_final_soft_composite",
        "outside_exact_support_invariance_proven": False,
        "outside_support_changed_pixels_total": None,
        "semantic_object_free_review_passed": False,
        "multiview_consistency_review_passed": False,
    }


def _semantic_editor_predictions(
    *,
    task: Mapping[str, Any],
    staged_task: Path,
    task_output: Path,
    model_root: Path,
    request: Mapping[str, Any],
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    import torch
    from PIL import Image

    semantic = request["semantic_editor"]
    backend = request["direct_editor_backend"]
    if backend == "vibe_image_edit":
        from vibe.editor import ImageEditor

        editor = ImageEditor(
            checkpoint_path=str(model_root),
            image_guidance_scale=float(semantic["image_guidance_scale"]),
            guidance_scale=float(semantic["guidance_scale"]),
            num_inference_steps=int(semantic["num_inference_steps"]),
            device="cuda:0",
            local_files_only=True,
        )
    else:
        raise ValueError("artifixer3d_semantic_editor_backend_invalid")
    predictions: dict[int, Path] = {}
    rows: list[dict[str, Any]] = []
    output_root = task_output / "semantic_editor" / "predictions"
    for frame in task["frames"]:
        index = int(frame["frame_index"])
        source = staged_task / frame["masked_reference_rgb"]["relative_path"]
        with Image.open(source) as image:
            condition = image.convert("RGB")
        seed = int(request["random_seed"]) + index
        generated = editor.generate_edited_image(
            SEMANTIC_EDITOR_PROMPT,
            conditioning_image=condition,
            randomize_seed=False,
            seed=seed,
            num_images_per_prompt=1,
            do_revert_resize=True,
        )[0].convert("RGB")
        if generated.size != condition.size:
            generated = generated.resize(condition.size, Image.Resampling.LANCZOS)
        output = output_root / f"{index:05d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        generated.save(output)
        predictions[index] = output
        rows.append(
            {
                "frame_index": index,
                "camera_id": frame["camera_id"],
                "prediction": {
                    "path": str(output),
                    "size_bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                },
            }
        )
    del editor
    torch.cuda.empty_cache()
    return predictions, rows


def _task_runtime(
    *,
    task: Mapping[str, Any],
    input_root: Path,
    source_root: Path,
    output_root: Path,
    checkpoint: Path,
    wan_root: Path,
    semantic_editor_root: Path | None,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if request.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE:
        return _dual_target_render_only_task_runtime(
            task=task,
            input_root=input_root,
            output_root=output_root,
            request=request,
        )
    if request.get("pipeline_mode") == DUAL_TARGET_PIPELINE_MODE:
        return _dual_target_task_runtime(
            task=task,
            input_root=input_root,
            source_root=source_root,
            output_root=output_root,
            request=request,
        )
    task_id = str(task["task_id"])
    staged_task = input_root / task_id
    task_output = output_root / "tasks" / task_id
    logs = task_output / "logs"
    python = sys.executable
    prompt = staged_task / "captions" / "unconditioned_zero_prompt.h5"
    if request.get("semantic_editor_only") is not True:
        _zero_prompt(prompt)
    fold_predictions: dict[int, Path] = {}
    direct_rows: list[dict[str, Any]] = []
    if request["direct_editor_backend"] != "artifixer":
        if semantic_editor_root is None:
            raise ValueError("artifixer3d_semantic_editor_model_missing")
        fold_predictions, direct_rows = _semantic_editor_predictions(
            task=task,
            staged_task=staged_task,
            task_output=task_output,
            model_root=semantic_editor_root,
            request=request,
        )
    for fold in (
        task["direct_inference_folds"] if request["direct_editor_backend"] == "artifixer" else []
    ):
        fold_id = str(fold["fold_id"])
        template = staged_task / Path(fold["split_template"]["path"]).name
        split_path = staged_task / f"split.direct_{fold_id}.json"
        _materialize_split(template, split_path)
        save_dir = task_output / "direct" / fold_id
        command = [
            python,
            "-m",
            "model_eval.run_inference",
            "--evalset",
            "reconstructed_colmap",
            "--checkpoint_pt",
            str(checkpoint),
            "--model_id",
            str(wan_root),
            "--save_dir",
            str(save_dir),
            "--split_path",
            str(split_path),
            "--render_trajectory",
            "trajectory",
            "--num_views",
            str(len(fold["selected_indices"])),
            "--neighbor_selection_mode",
            request["direct_inference"]["neighbor_selection_mode"],
            "--num_inference_steps",
            str(request["direct_inference"]["num_inference_steps"]),
            "--frames_per_block",
            str(request["direct_inference"]["frames_per_block"]),
            "--max_neighbors_per_encode",
            "1",
            "--save_frame_outputs_only",
            "--log_with",
            "none",
            "--seed",
            str(request["random_seed"]),
        ]
        _run(command, cwd=source_root, log=logs / f"direct_{fold_id}.log", timeout=3600)
        predictions = _prediction_dir(save_dir, task_id)
        for index in fold["target_indices"]:
            prediction = predictions / f"{int(index):05d}.png"
            if not prediction.is_file() or int(index) in fold_predictions:
                raise ValueError("artifixer3d_direct_prediction_coverage_invalid")
            fold_predictions[int(index)] = prediction
        direct_rows.append(
            {
                "fold_id": fold_id,
                "target_indices": fold["target_indices"],
                "prediction_directory": str(predictions),
                "log_sha256": _sha256(logs / f"direct_{fold_id}.log"),
            }
        )
    expected = set(task["direct_prediction_coverage_indices"])
    if set(fold_predictions) != expected:
        raise ValueError("artifixer3d_direct_prediction_coverage_invalid")

    repaired_scene = task_output / "repaired_scene"
    _copy_scene(staged_task, repaired_scene)
    composite_rows: list[dict[str, Any]] = []
    for frame in task["frames"]:
        index = int(frame["frame_index"])
        retained = staged_task / frame["rendered_rgb"]["relative_path"]
        mask = staged_task / frame["exact_repair_mask"]["relative_path"]
        output = repaired_scene / "images" / f"{index:05d}.png"
        row = _exact_composite(
            retained=retained,
            mask=mask,
            prediction=fold_predictions[index],
            output=output,
        )
        row.update(frame_index=index, camera_id=frame["camera_id"])
        composite_rows.append(row)

    if request.get("semantic_editor_only") is True:
        return {
            "task_id": task_id,
            "direct_editor_backend": request["direct_editor_backend"],
            "direct_folds": direct_rows,
            "direct_exact_composite_frames": composite_rows,
            "artifixer3d_checkpoint": None,
            "artifixer3d_log_sha256": None,
            "artifixer3d_plus_log_sha256": None,
            "final_candidate_frames": composite_rows,
            "outside_support_changed_pixels_total": sum(
                row["outside_support_changed_pixels"] for row in composite_rows
            ),
            "semantic_object_free_review_passed": False,
            "multiview_consistency_review_passed": False,
        }

    distill_split = repaired_scene / "split.distill.json"
    _write(
        distill_split,
        {
            "test": {
                task_id: {
                    "transforms_path": "transforms.json",
                    "image_root": ".",
                    "render_dir": "renders",
                    "opacity_dir": "opacity",
                    "selected_indices_path": "selected_indices.json",
                    "prompt_path": "captions/unconditioned_zero_prompt.h5",
                    "camera_scale": 1.0,
                    "has_gt": False,
                }
            }
        },
    )
    artifixer3d_root = task_output / "artifixer3d"
    command = [
        python,
        "-m",
        "data_processing.run_artifixer3d",
        "--scene_root",
        str(repaired_scene),
        "--artifixer_frames_dir",
        str(repaired_scene / "images"),
        "--split_path",
        str(distill_split),
        "--output_root",
        str(artifixer3d_root),
        "--artifixer3d_steps",
        str(request["artifixer3d"]["steps"]),
        "--config_name",
        request["artifixer3d"]["config_name"],
        "--phases",
        "distill,render,prepare_artifixer3d_plus",
        "--no-use_wandb",
    ]
    _run(command, cwd=source_root, log=logs / "artifixer3d.log", timeout=10_800)
    plus_split = repaired_scene / "split_artifixer3d_plus.json"
    if not plus_split.is_file():
        raise ValueError("artifixer3d_plus_split_missing")
    plus_save = task_output / "artifixer3d_plus"
    command = [
        python,
        "-m",
        "model_eval.run_inference",
        "--evalset",
        "reconstructed_colmap",
        "--checkpoint_pt",
        str(checkpoint),
        "--model_id",
        str(wan_root),
        "--save_dir",
        str(plus_save),
        "--split_path",
        str(plus_split),
        "--render_trajectory",
        "all_frames",
        "--num_views",
        str(len(task["artifixer3d_distillation"]["selected_anchor_indices"])),
        "--neighbor_selection_mode",
        request["direct_inference"]["neighbor_selection_mode"],
        "--num_inference_steps",
        str(request["direct_inference"]["num_inference_steps"]),
        "--frames_per_block",
        str(request["direct_inference"]["frames_per_block"]),
        "--max_neighbors_per_encode",
        "1",
        "--save_frame_outputs_only",
        "--log_with",
        "none",
        "--seed",
        str(request["random_seed"]),
    ]
    _run(command, cwd=source_root, log=logs / "artifixer3d_plus.log", timeout=3600)
    plus_predictions = _prediction_dir(plus_save, task_id)
    final_root = task_output / "final_candidate_frames"
    final_rows: list[dict[str, Any]] = []
    for frame in task["frames"]:
        index = int(frame["frame_index"])
        prediction = plus_predictions / f"{index:05d}.png"
        if not prediction.is_file():
            raise ValueError("artifixer3d_plus_prediction_missing")
        row = _exact_composite(
            retained=staged_task / frame["rendered_rgb"]["relative_path"],
            mask=staged_task / frame["exact_repair_mask"]["relative_path"],
            prediction=prediction,
            output=final_root / f"{index:05d}.png",
        )
        row.update(frame_index=index, camera_id=frame["camera_id"])
        final_rows.append(row)
    checkpoints = sorted(artifixer3d_root.glob("**/ckpt_*.pt"))
    if len(checkpoints) != 1:
        raise ValueError("artifixer3d_checkpoint_missing_or_ambiguous")
    return {
        "task_id": task_id,
        "direct_editor_backend": request["direct_editor_backend"],
        "direct_folds": direct_rows,
        "direct_exact_composite_frames": composite_rows,
        "artifixer3d_checkpoint": {
            "path": str(checkpoints[0]),
            "size_bytes": checkpoints[0].stat().st_size,
            "sha256": _sha256(checkpoints[0]),
        },
        "artifixer3d_log_sha256": _sha256(logs / "artifixer3d.log"),
        "artifixer3d_plus_log_sha256": _sha256(logs / "artifixer3d_plus.log"),
        "final_candidate_frames": final_rows,
        "outside_support_changed_pixels_total": sum(
            row["outside_support_changed_pixels"] for row in final_rows
        ),
        "semantic_object_free_review_passed": False,
        "multiview_consistency_review_passed": False,
    }


def execute(*, bundle_root: Path, output_root: Path, rehearsal: bool) -> dict[str, Any]:
    manifest, request, candidate = _validate_bundle(bundle_root)
    render_only = request.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    dual_target = request.get("pipeline_mode") in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }
    base: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "runtime_request_digest": request["runtime_request_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "replacement_object_count": candidate["replacement_object_count"],
        "task_ids": request["task_ids"],
        "source_object_restoration_permitted": False,
        "pipeline_mode": request.get("pipeline_mode"),
        "phases": request.get("phases"),
        "outside_exact_support_changed_pixels_permitted": (
            "unconstrained_for_raw_representation_review" if dual_target else 0
        ),
        "outside_support_invariance_gate": (
            "deferred_until_final_soft_composite" if dual_target else None
        ),
        "provider_zero_required_after_return": True,
        "physical_or_deployment_evidence": False,
        "checkpoint_reuse_digest": (
            request.get("artifixer3d", {}).get("checkpoint_reuse", {}).get("reuse_digest")
            if render_only
            else None
        ),
    }
    if rehearsal:
        return {
            "schema_version": "provider_bundle_rehearsal.v1",
            "status": "passed",
            "bundle_manifest_digest": manifest["manifest_digest"],
            "runtime_request_digest": request["runtime_request_digest"],
            "candidate_input_receipt_digest": candidate["receipt_digest"],
            "replacement_object_count": candidate["replacement_object_count"],
            "task_ids": request["task_ids"],
            "pipeline_mode": request.get("pipeline_mode"),
            "phases": request.get("phases"),
            "paid_inference_performed": False,
            "gpu_runtime_started": False,
            "provider_mutations_performed": 0,
            "blockers": [],
        }
    cache = bundle_root.parent / "artifixer3d_model_cache"
    if dual_target:
        semantic_editor_root = None
        checkpoint = Path("unused-dual-target-artifixer3d-only")
        wan_root = Path("unused-dual-target-artifixer3d-only")
    else:
        semantic_editor_root = _download_semantic_editor(request, cache)
    if not dual_target and request.get("semantic_editor_only") is True:
        checkpoint = Path("unused-semantic-editor-only")
        wan_root = Path("unused-semantic-editor-only")
    elif not dual_target:
        checkpoint, wan_root = _download_models(request, cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source_root = bundle_root / "provider_runtime" / "ArtiFixer_official"
    input_root = bundle_root / "provider_runtime" / "input"
    tasks: list[dict[str, Any]] = []
    expected_task_count = len(candidate["tasks"])
    for task in candidate["tasks"]:
        completed = _task_runtime(
            task=task,
            input_root=input_root,
            source_root=source_root,
            output_root=output_root,
            checkpoint=checkpoint,
            wan_root=wan_root,
            semantic_editor_root=semantic_editor_root,
            request=request,
        )
        tasks.append(completed)
        _write(
            output_root / TASK_PROGRESS_FILENAME,
            _task_progress(
                base=base,
                tasks=tasks,
                expected_task_count=expected_task_count,
            ),
        )
    if dual_target:
        if any(not _completed_task_is_bound(task) for task in tasks):
            raise ValueError("artifixer3d_dual_target_deferred_invariance_invalid")
    elif any(task["outside_support_changed_pixels_total"] != 0 for task in tasks):
        raise ValueError("artifixer3d_outside_support_change")
    return {
        **base,
        "status": (
            "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
            if dual_target
            else "candidate_completed_requires_visual_and_multiview_review"
        ),
        "tasks": tasks,
        "model_loaded": True,
        "artifixer_direct_inference_executed": (
            not dual_target and request["direct_editor_backend"] == "artifixer"
        ),
        "semantic_editor_inference_executed": (
            not dual_target and request["direct_editor_backend"] != "artifixer"
        ),
        "artifixer3d_distillation_executed": (
            not render_only and (dual_target or request.get("semantic_editor_only") is not True)
        ),
        "artifixer3d_checkpoint_reused": render_only,
        "artifixer3d_plus_inference_executed": (
            not dual_target and request.get("semantic_editor_only") is not True
        ),
        "outside_exact_support_invariance_proven": False if dual_target else True,
        "provider_mutations_performed": 1,
        "blockers": [
            "semantic_object_free_visual_review_required",
            "multiview_consistency_review_required",
            "appearance_repair_not_yet_qualified",
        ],
        "claim_boundary": "generated_candidate_appearance_not_capture_or_physical_evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rehearsal", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / (
        "provider_bundle_rehearsal.json"
        if args.rehearsal
        else "public_scene_artifixer3d_runtime_result.json"
    )
    try:
        result = execute(
            bundle_root=args.bundle_root.resolve(),
            output_root=output,
            rehearsal=args.rehearsal,
        )
    except Exception as exc:  # preserve the typed terminal runtime failure
        progress = _read_task_progress(output / TASK_PROGRESS_FILENAME)
        completed_tasks = list(progress["tasks"]) if progress is not None else []
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "blocked",
            "tasks": completed_tasks,
            "completed_task_count": len(completed_tasks),
            "completed_task_ids": [task["task_id"] for task in completed_tasks],
            "partial_task_evidence_preserved": bool(completed_tasks),
            "task_progress_digest": (progress["progress_digest"] if progress is not None else None),
            "model_loaded": False,
            "artifixer_direct_inference_executed": False,
            "semantic_editor_inference_executed": False,
            "artifixer3d_distillation_executed": False,
            "artifixer3d_plus_inference_executed": False,
            "provider_mutations_performed": 0 if args.rehearsal else 1,
            "blockers": [f"artifixer3d_runtime_exception:{type(exc).__name__}", str(exc)],
            "provider_zero_required_after_return": True,
            "physical_or_deployment_evidence": False,
            "claim_boundary": "runtime_failure_only",
        }
    _write(result_path, result)
    print(_canonical_json(result), flush=True)
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
