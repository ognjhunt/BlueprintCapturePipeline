"""Bound one semantic-teacher repair round to an exact rejected review.

The production ArtiFixer reviewer can reject a candidate even when object
absence, orientation, and local plausibility all pass.  This module recognizes
only the remaining locality-only failure, stages a stricter request for those
exact cameras, and merges the returned semantic targets without modifying the
already sealed render or first-pass semantic outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .semantic_teacher_image_edit_worker import (
    RUNTIME_REQUEST_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_artifixer_ai_visual_review import (
    DUAL_TARGET_REVIEW_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION as REVIEW_EXECUTION_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_semantic_locality import (
    SEMANTIC_LOCALITY_SCHEMA_VERSION,
    seal_semantic_teacher_frame,
)


PLAN_SCHEMA_VERSION = "task_evaluation_artifixer_selective_repair_plan.v1"
MERGE_SCHEMA_VERSION = "task_evaluation_artifixer_selective_repair_merge.v1"
STRICT_LOCALITY_PROMPT_POLICY = (
    "generic_masked_object_absent_background_completion_strict_locality_v1"
)
STRICT_LOCALITY_PROMPT = (
    "Remove only the object inside the supplied edit mask. Reconstruct the "
    "continuing tabletop or background inside that mask while matching the "
    "immediately adjacent material, color, texture, lighting, perspective, and "
    "camera viewpoint. Preserve every pixel and every material feature outside "
    "the supplied mask: do not redraw, restyle, regrain, revein, recolor, "
    "relight, smooth, or otherwise change the surrounding tabletop or scene. "
    "Do not add an object, silhouette, patch, panel, text, watermark, or robot."
)
MAX_REPAIR_ROUNDS = 1
MAX_SELECTIVE_REPAIR_FRAMES = 2


class TaskEvaluationArtifixerSelectiveRepairError(RuntimeError):
    """The rejected review was not eligible for a bounded locality repair."""


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
        raise TaskEvaluationArtifixerSelectiveRepairError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationArtifixerSelectiveRepairError(code)
    return dict(value)


def _bound_file(
    value: Any, *, root: Path | None = None, code: str
) -> Path:
    if not isinstance(value, Mapping):
        raise TaskEvaluationArtifixerSelectiveRepairError(code)
    raw = value.get("path") if root is None else value.get("relative_path")
    unresolved = Path(str(raw or "")).expanduser()
    if root is not None:
        relative = PurePosixPath(str(raw or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise TaskEvaluationArtifixerSelectiveRepairError(code)
        unresolved = root.joinpath(*relative.parts)
    resolved = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_size != value.get("size_bytes")
        or _sha256(resolved) != value.get("sha256")
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(code)
    return resolved


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if root is None:
        value["path"] = str(path.resolve())
    else:
        value["relative_path"] = path.relative_to(root).as_posix()
    return value


def _validate_exact_mask_binding(
    *,
    review_input_path: Path,
    review_frame: Mapping[str, Any],
    request_root: Path,
    request_frame: Mapping[str, Any],
    mask_encoding: str,
) -> None:
    source = _bound_file(
        review_frame.get("source_frame"),
        code="scene_configuration_artifixer_selective_repair_source_invalid",
    )
    exact_mask = _bound_file(
        review_frame.get("exact_repair_mask"),
        code="scene_configuration_artifixer_selective_repair_mask_invalid",
    )
    staged_source = _bound_file(
        request_frame.get("input_rgb"),
        root=request_root,
        code="scene_configuration_artifixer_selective_repair_source_invalid",
    )
    staged_mask = _bound_file(
        request_frame.get("edit_mask"),
        root=request_root,
        code="scene_configuration_artifixer_selective_repair_mask_invalid",
    )
    try:
        with Image.open(source) as image:
            source_rgb = image.convert("RGB")
            source_bytes = source_rgb.tobytes()
            source_size = source_rgb.size
        with Image.open(staged_source) as image:
            staged_rgb = image.convert("RGB")
            staged_source_bytes = staged_rgb.tobytes()
            staged_source_size = staged_rgb.size
        with Image.open(exact_mask) as image:
            original_mask = image.convert("L")
            original_values = original_mask.tobytes()
            original_size = original_mask.size
        with Image.open(staged_mask) as image:
            if mask_encoding == "rgba_alpha_zero_edit_region_png":
                staged_values = image.convert("RGBA").getchannel("A").tobytes()
                staged_edit = bytes(value == 0 for value in staged_values)
            elif mask_encoding == "binary_white_edit_region_png":
                staged_values = image.convert("L").tobytes()
                staged_edit = bytes(value > 0 for value in staged_values)
            elif mask_encoding == "binary_black_edit_region_png":
                staged_values = image.convert("L").tobytes()
                staged_edit = bytes(value == 0 for value in staged_values)
            else:
                raise ValueError
            staged_size = image.size
    except (OSError, ValueError) as exc:
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_mask_invalid"
        ) from exc
    original_edit = bytes(value > 0 for value in original_values)
    if (
        source_size != staged_source_size
        or source_bytes != staged_source_bytes
        or original_size != staged_size
        or original_edit != staged_edit
        or not any(original_edit)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_mask_invalid"
        )
    # The path is part of the review receipt's digest. Resolve it here so a
    # caller cannot pass an equivalent-looking row from another receipt.
    if not review_input_path.is_file():
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_review_invalid"
        )


def materialize_selective_repair_request(
    *,
    review_input_path: str | Path,
    review_execution_path: str | Path,
    semantic_runtime_request_path: str | Path,
    semantic_runtime_result: Mapping[str, Any],
    semantic_locality_receipt_path: str | Path,
    expected_request_cost_usd: float,
    maximum_stage_cost_usd: float,
    output_root: str | Path,
) -> dict[str, Any]:
    """Stage one exact-mask repair request for locality-only rejections."""

    review_path = Path(review_input_path).expanduser().resolve()
    execution_path = Path(review_execution_path).expanduser().resolve()
    request_path = Path(semantic_runtime_request_path).expanduser().resolve()
    locality_path = Path(semantic_locality_receipt_path).expanduser().resolve()
    review = _read(
        review_path,
        code="scene_configuration_artifixer_selective_repair_review_invalid",
    )
    execution = _read(
        execution_path,
        code="scene_configuration_artifixer_selective_repair_review_invalid",
    )
    request = _read(
        request_path,
        code="scene_configuration_artifixer_selective_repair_request_invalid",
    )
    locality = _read(
        locality_path,
        code="scene_configuration_artifixer_selective_repair_locality_invalid",
    )
    semantic_result = dict(semantic_runtime_result)
    review_tasks = review.get("tasks")
    execution_rows = execution.get("frames")
    request_tasks = request.get("tasks")
    locality_rows = locality.get("frames")
    if (
        review.get("schema_version") != DUAL_TARGET_REVIEW_SCHEMA_VERSION
        or review.get("receipt_digest")
        != canonical_digest(review, digest_field="receipt_digest")
        or not isinstance(review_tasks, list)
        or len(review_tasks) != 1
        or not isinstance(review_tasks[0], Mapping)
        or execution.get("schema_version") != REVIEW_EXECUTION_SCHEMA_VERSION
        or execution.get("status") != "completed"
        or execution.get("decision") != "rejected"
        or execution.get("execution_digest")
        != canonical_digest(execution, digest_field="execution_digest")
        or execution.get("final_composite_receipt_digest")
        != review.get("receipt_digest")
        or execution.get("provider_called") is not True
        or execution.get("response_store") is not False
        or execution.get("tracing_disabled") is not True
        or execution.get("raw_secret_values_recorded") is not False
        or not isinstance(execution_rows, list)
        or request.get("schema_version") != RUNTIME_REQUEST_SCHEMA_VERSION
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
        or not isinstance(request_tasks, list)
        or len(request_tasks) != 1
        or not isinstance(request_tasks[0], Mapping)
        or semantic_result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or semantic_result.get("status")
        != "completed_unreviewed_semantic_teacher_candidates"
        or semantic_result.get("source_runtime_request_digest")
        != request.get("request_digest")
        or semantic_result.get("result_digest")
        != canonical_digest(semantic_result, digest_field="result_digest")
        or locality.get("schema_version") != SEMANTIC_LOCALITY_SCHEMA_VERSION
        or locality.get("status")
        != "semantic_teacher_exact_support_locality_sealed"
        or locality.get("receipt_digest")
        != canonical_digest(locality, digest_field="receipt_digest")
        or locality.get("source_runtime_request_digest")
        != request.get("request_digest")
        or locality.get("source_runtime_result_digest")
        != semantic_result.get("result_digest")
        or locality.get("all_non_target_source_pixels_preserved_exactly") is not True
        or not isinstance(locality_rows, list)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_review_invalid"
        )
    review_task = review_tasks[0]
    request_task = request_tasks[0]
    review_frames = review_task.get("frames")
    request_frames = request_task.get("frames")
    task_id = str(review_task.get("task_id") or "")
    if (
        not task_id
        or execution.get("task_id") != task_id
        or request_task.get("task_id") != task_id
        or not isinstance(review_frames, list)
        or not isinstance(request_frames, list)
        or len(review_frames) != len(request_frames)
        or len(execution_rows) != len(review_frames)
        or len(locality_rows) != len(review_frames)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_review_invalid"
        )
    request_by_camera = {
        str(row.get("camera_id") or ""): row
        for row in request_frames
        if isinstance(row, Mapping)
    }
    review_by_camera = {
        str(row.get("camera_id") or ""): row
        for row in review_frames
        if isinstance(row, Mapping)
    }
    if (
        len(request_by_camera) != len(request_frames)
        or len(review_by_camera) != len(review_frames)
        or set(request_by_camera) != set(review_by_camera)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_review_invalid"
        )
    locality_by_camera = {
        str(row.get("camera_id") or ""): row
        for row in locality_rows
        if isinstance(row, Mapping)
    }
    if (
        len(locality_by_camera) != len(locality_rows)
        or set(locality_by_camera) != set(review_by_camera)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_locality_invalid"
        )
    selected_by_camera: dict[str, dict[str, Any]] = {}
    observed: set[str] = set()
    for row in execution_rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationArtifixerSelectiveRepairError(
                "scene_configuration_artifixer_selective_repair_review_invalid"
            )
        camera_id = str(row.get("camera_id") or "")
        review_frame = review_by_camera.get(camera_id)
        final = review_frame.get("final_frame") if isinstance(review_frame, Mapping) else None
        if (
            not camera_id
            or camera_id in observed
            or not isinstance(final, Mapping)
            or row.get("task_id") != task_id
            or row.get("frame_sha256") != final.get("sha256")
            or not str(row.get("rationale") or "").strip()
        ):
            raise TaskEvaluationArtifixerSelectiveRepairError(
                "scene_configuration_artifixer_selective_repair_review_invalid"
            )
        observed.add(camera_id)
        accepted = (
            row.get("decision") == "accepted"
            and row.get("orientation_is_upright") is True
            and row.get("source_object_absent") is True
            and row.get("repair_is_locally_plausible") is True
            and row.get("preserves_non_target_content") is True
        )
        locality_only = (
            row.get("decision") == "rejected"
            and row.get("orientation_is_upright") is True
            and row.get("source_object_absent") is True
            and row.get("repair_is_locally_plausible") is True
            and row.get("preserves_non_target_content") is False
        )
        if locality_only:
            selected_by_camera[camera_id] = {
                "review_row": row,
                "selection_reasons": ["independent_visual_review_preservation_rejection"],
                "feedback": [str(row["rationale"])],
            }
        elif not accepted:
            raise TaskEvaluationArtifixerSelectiveRepairError(
                "scene_configuration_artifixer_selective_repair_ineligible_rejection"
            )
    if observed != set(review_by_camera):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_ineligible_rejection"
        )
    for camera_id, locality_row in locality_by_camera.items():
        request_frame = request_by_camera[camera_id]
        if (
            locality_row.get("source_frame", {}).get("sha256")
            != request_frame.get("input_rgb", {}).get("sha256")
            or locality_row.get("exact_edit_mask", {}).get("sha256")
            != request_frame.get("edit_mask", {}).get("sha256")
        ):
            raise TaskEvaluationArtifixerSelectiveRepairError(
                "scene_configuration_artifixer_selective_repair_locality_invalid"
            )
        if locality_row.get("deterministic_selective_repair_required") is True:
            selected = selected_by_camera.setdefault(
                camera_id,
                {
                    "review_row": next(
                        row
                        for row in execution_rows
                        if row.get("camera_id") == camera_id
                    ),
                    "selection_reasons": [],
                    "feedback": [],
                },
            )
            selected["selection_reasons"].append(
                "deterministic_gross_outside_mask_change"
            )
            selected["feedback"].append(
                str(locality_row.get("deterministic_repair_feedback") or "").strip()
            )
    rejected = [
        selected_by_camera[camera_id]
        for camera_id in review_by_camera
        if camera_id in selected_by_camera
    ]
    if not 1 <= len(rejected) <= MAX_SELECTIVE_REPAIR_FRAMES:
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_ineligible_rejection"
        )
    if (
        isinstance(expected_request_cost_usd, bool)
        or not isinstance(expected_request_cost_usd, (int, float))
        or not math.isfinite(float(expected_request_cost_usd))
        or float(expected_request_cost_usd) <= 0
        or isinstance(maximum_stage_cost_usd, bool)
        or not isinstance(maximum_stage_cost_usd, (int, float))
        or not math.isfinite(float(maximum_stage_cost_usd))
        or float(maximum_stage_cost_usd) <= 0
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_cost_invalid"
        )
    base_cost = semantic_result.get("computed_editor_cost_usd")
    if (
        isinstance(base_cost, bool)
        or not isinstance(base_cost, (int, float))
        or not math.isfinite(float(base_cost))
        or float(base_cost) < 0
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_cost_invalid"
        )
    projected_repair_cost = float(expected_request_cost_usd) * len(rejected)
    remaining_cost = float(maximum_stage_cost_usd) - float(base_cost)
    if remaining_cost + 1e-9 < projected_repair_cost:
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_cost_insufficient"
        )
    mask_encoding = str(((request.get("backend") or {}).get("execution") or {}).get("mask_encoding") or "")
    selected_rows: list[dict[str, Any]] = []
    repair_frames: list[dict[str, Any]] = []
    feedback_lines: list[str] = []
    for repair_index, selected in enumerate(rejected):
        row = selected["review_row"]
        camera_id = str(row["camera_id"])
        review_frame = review_by_camera[camera_id]
        request_frame = request_by_camera[camera_id]
        _validate_exact_mask_binding(
            review_input_path=review_path,
            review_frame=review_frame,
            request_root=request_path.parent,
            request_frame=request_frame,
            mask_encoding=mask_encoding,
        )
        original_index = int(review_frame["frame_index"])
        selected_rows.append(
            {
                "repair_index": repair_index,
                "original_frame_index": original_index,
                "camera_id": camera_id,
                "rejected_frame_sha256": row["frame_sha256"],
                "review_rationale": str(row["rationale"]),
                "selection_reasons": list(selected["selection_reasons"]),
                "repair_feedback": list(dict.fromkeys(selected["feedback"])),
                "source_rgb_sha256": request_frame["input_rgb"]["sha256"],
                "exact_edit_mask_sha256": request_frame["edit_mask"]["sha256"],
                "failed_semantic_teacher_sha256": locality_by_camera[camera_id][
                    "raw_semantic_teacher"
                ]["sha256"],
            }
        )
        feedback_lines.append(
            f"Camera {camera_id}: "
            + " ".join(dict.fromkeys(selected["feedback"]))
        )
        repair_frames.append(
            {
                **dict(request_frame),
                "frame_index": repair_index,
                "prior_failed_semantic_teacher": locality_by_camera[camera_id][
                    "raw_semantic_teacher"
                ],
            }
        )
    repair_request: dict[str, Any] = {
        **request,
        "prompt_policy": STRICT_LOCALITY_PROMPT_POLICY,
        "prompt": STRICT_LOCALITY_PROMPT + " Failed-candidate feedback: " + " ".join(feedback_lines),
        "tasks": [
            {
                **dict(request_task),
                "camera_count": len(repair_frames),
                "frames": repair_frames,
            }
        ],
        "max_parallel_requests": min(len(repair_frames), 2),
        "maximum_cost_usd": round(remaining_cost, 9),
        "expected_request_cost_usd": float(expected_request_cost_usd),
        "retry_count": 0,
        "request_digest": "",
    }
    repair_request["request_digest"] = canonical_digest(
        repair_request, digest_field="request_digest"
    )
    root = Path(output_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_output_exists"
        )
    root.mkdir(parents=True, mode=0o700)
    repair_request_path = root / f"{RUNTIME_REQUEST_SCHEMA_VERSION}.json"
    repair_request_path.write_text(
        canonical_json(repair_request) + "\n", encoding="utf-8"
    )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "one_bounded_locality_repair_ready",
        "repair_round": 1,
        "maximum_repair_rounds": MAX_REPAIR_ROUNDS,
        "maximum_repair_frames": MAX_SELECTIVE_REPAIR_FRAMES,
        "task_id": task_id,
        "selected_frames": selected_rows,
        "selected_frame_count": len(selected_rows),
        "review_input_digest": review["receipt_digest"],
        "review_execution_digest": execution["execution_digest"],
        "semantic_locality_receipt_digest": locality["receipt_digest"],
        "source_semantic_request_digest": request["request_digest"],
        "source_semantic_result_digest": semantic_result["result_digest"],
        "repair_request_digest": repair_request["request_digest"],
        "strict_locality_prompt_policy": STRICT_LOCALITY_PROMPT_POLICY,
        "exact_source_edit_masks_reused_without_dilation": True,
        "base_computed_editor_cost_usd": float(base_cost),
        "projected_repair_cost_usd": projected_repair_cost,
        "remaining_stage_cost_usd": remaining_cost,
        "additional_provider_request_cap": len(selected_rows),
        "second_repair_round_permitted": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    plan_path = root / f"{PLAN_SCHEMA_VERSION}.json"
    plan_path.write_text(canonical_json(plan) + "\n", encoding="utf-8")
    return {
        "plan": plan,
        "plan_path": str(plan_path),
        "repair_request": repair_request,
        "repair_request_path": str(repair_request_path),
    }


def merge_selective_repair_outputs(
    *,
    plan_path: str | Path,
    semantic_runtime_request_path: str | Path,
    semantic_locality_receipt_path: str | Path,
    source_semantic_output_root: str | Path,
    source_semantic_result: Mapping[str, Any],
    repair_output_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create an immutable eight-frame set with only selected cameras replaced."""

    plan_file = Path(plan_path).expanduser().resolve()
    plan = _read(
        plan_file,
        code="scene_configuration_artifixer_selective_repair_plan_invalid",
    )
    request_path = Path(semantic_runtime_request_path).expanduser().resolve()
    request = _read(
        request_path,
        code="scene_configuration_artifixer_selective_repair_request_invalid",
    )
    locality_path = Path(semantic_locality_receipt_path).expanduser().resolve()
    locality = _read(
        locality_path,
        code="scene_configuration_artifixer_selective_repair_locality_invalid",
    )
    source_root = Path(source_semantic_output_root).expanduser().resolve()
    repair_root = Path(repair_output_root).expanduser().resolve()
    repair_result_path = repair_root / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json"
    repair_result = _read(
        repair_result_path,
        code="scene_configuration_artifixer_selective_repair_result_invalid",
    )
    source_result = dict(source_semantic_result)
    selected = plan.get("selected_frames")
    source_tasks = source_result.get("tasks")
    repair_tasks = repair_result.get("tasks")
    request_tasks = request.get("tasks")
    locality_rows = locality.get("frames")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != "one_bounded_locality_repair_ready"
        or plan.get("repair_round") != 1
        or plan.get("maximum_repair_rounds") != 1
        or plan.get("second_repair_round_permitted") is not False
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
        or not isinstance(selected, list)
        or len(selected) != plan.get("selected_frame_count")
        or not 1 <= len(selected) <= MAX_SELECTIVE_REPAIR_FRAMES
        or source_result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or source_result.get("status")
        != "completed_unreviewed_semantic_teacher_candidates"
        or source_result.get("result_digest") != plan.get("source_semantic_result_digest")
        or source_result.get("result_digest")
        != canonical_digest(source_result, digest_field="result_digest")
        or repair_result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or repair_result.get("status")
        != "completed_unreviewed_semantic_teacher_candidates"
        or repair_result.get("source_runtime_request_digest")
        != plan.get("repair_request_digest")
        or repair_result.get("result_digest")
        != canonical_digest(repair_result, digest_field="result_digest")
        or repair_result.get("request_count") != len(selected)
        or repair_result.get("attempted_request_count") != len(selected)
        or repair_result.get("successful_request_count") != len(selected)
        or repair_result.get("failed_request_count") != 0
        or repair_result.get("billing_qualified") is not True
        or repair_result.get("retry_count") != 0
        or not isinstance(source_tasks, list)
        or len(source_tasks) != 1
        or not isinstance(repair_tasks, list)
        or len(repair_tasks) != 1
        or request.get("schema_version") != RUNTIME_REQUEST_SCHEMA_VERSION
        or request.get("request_digest") != plan.get("source_semantic_request_digest")
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
        or not isinstance(request_tasks, list)
        or len(request_tasks) != 1
        or locality.get("schema_version") != SEMANTIC_LOCALITY_SCHEMA_VERSION
        or locality.get("receipt_digest")
        != plan.get("semantic_locality_receipt_digest")
        or locality.get("receipt_digest")
        != canonical_digest(locality, digest_field="receipt_digest")
        or locality.get("all_non_target_source_pixels_preserved_exactly") is not True
        or not isinstance(locality_rows, list)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_result_invalid"
        )
    task_id = str(plan.get("task_id") or "")
    source_task = source_tasks[0]
    repair_task = repair_tasks[0]
    request_task = request_tasks[0]
    source_frames = source_task.get("frames") if isinstance(source_task, Mapping) else None
    repair_frames = repair_task.get("frames") if isinstance(repair_task, Mapping) else None
    request_frames = request_task.get("frames") if isinstance(request_task, Mapping) else None
    if (
        source_task.get("task_id") != task_id
        or repair_task.get("task_id") != task_id
        or not isinstance(source_frames, list)
        or not source_frames
        or not isinstance(repair_frames, list)
        or len(repair_frames) != len(selected)
        or not isinstance(request_frames, list)
        or len(request_frames) != len(source_frames)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_result_invalid"
        )
    selected_by_original = {
        int(row["original_frame_index"]): row for row in selected
    }
    repair_by_index = {
        int(row.get("frame_index", -1)): row
        for row in repair_frames
        if isinstance(row, Mapping)
    }
    request_by_camera = {
        str(row.get("camera_id") or ""): row
        for row in request_frames
        if isinstance(row, Mapping)
    }
    locality_by_camera = {
        str(row.get("camera_id") or ""): row
        for row in locality_rows
        if isinstance(row, Mapping)
    }
    if (
        len(request_by_camera) != len(request_frames)
        or len(locality_by_camera) != len(locality_rows)
        or set(locality_by_camera) != set(request_by_camera)
    ):
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_request_invalid"
        )
    mask_encoding = str(
        ((request.get("backend") or {}).get("execution") or {}).get(
            "mask_encoding"
        )
        or ""
    )
    root = Path(output_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise TaskEvaluationArtifixerSelectiveRepairError(
            "scene_configuration_artifixer_selective_repair_output_exists"
        )
    frames_root = root / "tasks" / task_id
    frames_root.mkdir(parents=True, mode=0o700)
    inventory: list[dict[str, Any]] = []
    try:
        for expected_index, source_frame in enumerate(source_frames):
            if (
                not isinstance(source_frame, Mapping)
                or source_frame.get("frame_index") != expected_index
                or not isinstance(source_frame.get("semantic_teacher_frame"), Mapping)
            ):
                raise TaskEvaluationArtifixerSelectiveRepairError(
                    "scene_configuration_artifixer_selective_repair_source_invalid"
                )
            selected_row = selected_by_original.get(expected_index)
            role = "reused_first_pass_semantic_frame"
            if selected_row is not None:
                repair_index = int(selected_row["repair_index"])
                repair_frame = repair_by_index.get(repair_index)
                repair_record = (
                    repair_frame.get("semantic_teacher_frame")
                    if isinstance(repair_frame, Mapping)
                    else None
                )
                if (
                    not isinstance(repair_frame, Mapping)
                    or repair_frame.get("camera_id") != selected_row.get("camera_id")
                    or repair_frame.get("terminal_state")
                    != "completed_unreviewed_candidate"
                    or not isinstance(repair_record, Mapping)
                ):
                    raise TaskEvaluationArtifixerSelectiveRepairError(
                        "scene_configuration_artifixer_selective_repair_result_invalid"
                    )
                source = _bound_file(
                    repair_record,
                    root=repair_root,
                    code="scene_configuration_artifixer_selective_repair_result_invalid",
                )
                role = "selectively_repaired_semantic_frame"
            else:
                source = _bound_file(
                    locality_by_camera[str(source_frame["camera_id"])].get(
                        "sealed_semantic_teacher"
                    ),
                    root=source_root,
                    code="scene_configuration_artifixer_selective_repair_source_invalid",
                )
            destination = frames_root / f"{expected_index:05d}.png"
            if selected_row is None:
                shutil.copyfile(source, destination)
                if _sha256(destination) != _sha256(source):
                    raise TaskEvaluationArtifixerSelectiveRepairError(
                        "scene_configuration_artifixer_selective_repair_copy_mismatch"
                    )
            else:
                request_frame = request_by_camera.get(str(source_frame["camera_id"]))
                if not isinstance(request_frame, Mapping):
                    raise TaskEvaluationArtifixerSelectiveRepairError(
                        "scene_configuration_artifixer_selective_repair_request_invalid"
                    )
                original = _bound_file(
                    request_frame.get("input_rgb"),
                    root=request_path.parent,
                    code="scene_configuration_artifixer_selective_repair_source_invalid",
                )
                exact_mask = _bound_file(
                    request_frame.get("edit_mask"),
                    root=request_path.parent,
                    code="scene_configuration_artifixer_selective_repair_mask_invalid",
                )
                seal_semantic_teacher_frame(
                    source_path=original,
                    mask_path=exact_mask,
                    raw_teacher_path=source,
                    mask_encoding=mask_encoding,
                    output_path=destination,
                )
            inventory.append(
                {
                    "frame_index": expected_index,
                    "camera_id": source_frame["camera_id"],
                    "role": role,
                    **_record(destination, root=root),
                }
            )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    receipt: dict[str, Any] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "status": "one_bounded_locality_repair_merged",
        "repair_round": 1,
        "maximum_repair_rounds": 1,
        "task_id": task_id,
        "source_semantic_result_digest": source_result["result_digest"],
        "repair_result_digest": repair_result["result_digest"],
        "repair_plan_digest": plan["plan_digest"],
        "strict_locality_prompt_policy": STRICT_LOCALITY_PROMPT_POLICY,
        "exact_source_edit_masks_reused_without_dilation": True,
        "frame_count": len(inventory),
        "repaired_frame_count": len(selected),
        "reused_frame_count": len(inventory) - len(selected),
        "frame_inventory": inventory,
        "second_repair_round_permitted": False,
        "all_non_target_source_pixels_preserved_exactly": True,
        "generated_output_is_capture_or_physical_evidence": False,
        "merge_digest": "",
    }
    receipt["merge_digest"] = canonical_digest(receipt, digest_field="merge_digest")
    receipt_path = root / f"{MERGE_SCHEMA_VERSION}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_path),
        "semantic_teacher_frames_root": str(frames_root),
    }


__all__ = [
    "MAX_REPAIR_ROUNDS",
    "MAX_SELECTIVE_REPAIR_FRAMES",
    "MERGE_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "STRICT_LOCALITY_PROMPT",
    "STRICT_LOCALITY_PROMPT_POLICY",
    "TaskEvaluationArtifixerSelectiveRepairError",
    "materialize_selective_repair_request",
    "merge_selective_repair_outputs",
]
