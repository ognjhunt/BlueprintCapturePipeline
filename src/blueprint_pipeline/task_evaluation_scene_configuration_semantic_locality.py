"""Seal semantic-teacher candidates to the exact repair support.

Image editors are permitted to synthesize the pixels inside an admitted edit
mask.  They are not authoritative for the rest of the rendered observation.
This module composites every unreviewed teacher candidate onto its exact source
frame before ArtiFixer training, then records and reads back the invariant that
every pixel outside the mask is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from .decision_evidence_contracts import canonical_digest, canonical_json
from .semantic_teacher_image_edit_worker import (
    RUNTIME_REQUEST_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
)


SEMANTIC_LOCALITY_SCHEMA_VERSION = (
    "task_evaluation_artifixer_semantic_locality_seal.v1"
)
SEMANTIC_LOCALITY_POLICY = (
    "exact_edit_support_source_preservation_inner_feather_v2"
)
MAX_LOCALITY_SEAL_FRAMES = 8
MAX_INNER_FEATHER_RADIUS_PIXELS = 16
GROSS_OUTSIDE_CHANGE_CHANNEL_DELTA = 32
GROSS_OUTSIDE_CHANGE_FRACTION = 0.25


class TaskEvaluationSceneConfigurationSemanticLocalityError(RuntimeError):
    """A semantic candidate could not be bound to exact source pixels."""


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
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(code)
    return dict(value)


def _bound_file(
    value: Any,
    *,
    root: Path,
    code: str,
) -> Path:
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(code)
    relative = PurePosixPath(str(value.get("relative_path") or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(code)
    unresolved = root.joinpath(*relative.parts)
    resolved = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_size != value.get("size_bytes")
        or _sha256(resolved) != value.get("sha256")
    ):
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(code)
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


def _edit_support(*, mask_path: Path, encoding: str) -> Image.Image:
    try:
        with Image.open(mask_path) as image:
            if encoding == "rgba_alpha_zero_edit_region_png":
                alpha = image.convert("RGBA").getchannel("A")
                support = alpha.point(lambda value: 255 if value == 0 else 0)
            elif encoding == "binary_white_edit_region_png":
                luminance = image.convert("L")
                support = luminance.point(lambda value: 255 if value > 0 else 0)
            elif encoding == "binary_black_edit_region_png":
                luminance = image.convert("L")
                support = luminance.point(lambda value: 255 if value == 0 else 0)
            else:
                raise ValueError
    except (OSError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_mask_invalid"
        ) from exc
    if support.getbbox() is None:
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_mask_empty"
        )
    return support


def _outside_difference_present(
    *, source: Image.Image, candidate: Image.Image, support: Image.Image
) -> bool:
    difference = ImageChops.difference(source, candidate)
    black = Image.new("RGB", source.size, color=(0, 0, 0))
    outside_difference = Image.composite(black, difference, support)
    return outside_difference.getbbox() is not None


def _outside_high_delta_fraction(
    *, source: Image.Image, candidate: Image.Image, support: Image.Image
) -> float:
    channels = ImageChops.difference(source, candidate).split()
    maximum = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    black = Image.new("L", source.size, color=0)
    outside_maximum = Image.composite(black, maximum, support)
    histogram = outside_maximum.histogram()
    support_histogram = support.histogram()
    outside_pixels = source.width * source.height - sum(support_histogram[1:])
    if outside_pixels <= 0:
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_mask_invalid"
        )
    high_delta_pixels = sum(
        histogram[GROSS_OUTSIDE_CHANGE_CHANNEL_DELTA + 1 :]
    )
    return high_delta_pixels / outside_pixels


def _inner_feather_alpha(support: Image.Image) -> tuple[Image.Image, int]:
    """Fade a generated fill toward source pixels without leaving support."""

    bounds = support.getbbox()
    if bounds is None:
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_mask_empty"
        )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    radius = min(
        MAX_INNER_FEATHER_RADIUS_PIXELS,
        max(1, min(width, height) // 8),
    )
    eroded = support.filter(ImageFilter.MinFilter(radius * 2 + 1))
    if eroded.getbbox() is None:
        # Extremely narrow supports cannot be feathered without deleting the
        # admitted edit region. Keep their existing exact-support behavior.
        return support, 0
    blurred = eroded.filter(ImageFilter.GaussianBlur(radius=radius / 2))
    return ImageChops.multiply(support, blurred), radius


def seal_semantic_teacher_frame(
    *,
    source_path: str | Path,
    mask_path: str | Path,
    raw_teacher_path: str | Path,
    mask_encoding: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one teacher frame with exact source pixels outside the mask."""

    source_file = Path(source_path).expanduser().resolve()
    mask_file = Path(mask_path).expanduser().resolve()
    raw_teacher_file = Path(raw_teacher_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_output_exists"
        )
    support = _edit_support(mask_path=mask_file, encoding=mask_encoding)
    try:
        with Image.open(source_file) as image:
            source = image.convert("RGB")
        with Image.open(raw_teacher_file) as image:
            raw_teacher = image.convert("RGB")
    except OSError as exc:
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_image_invalid"
        ) from exc
    if source.size != raw_teacher.size or source.size != support.size:
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_shape_invalid"
        )
    raw_changed_outside = _outside_difference_present(
        source=source,
        candidate=raw_teacher,
        support=support,
    )
    high_delta_fraction = _outside_high_delta_fraction(
        source=source,
        candidate=raw_teacher,
        support=support,
    )
    inner_feather, feather_radius = _inner_feather_alpha(support)
    sealed = Image.composite(raw_teacher, source, inner_feather)
    if _outside_difference_present(
        source=source,
        candidate=sealed,
        support=support,
    ):
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_readback_failed"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sealed.save(destination, format="PNG")
    return {
        "raw_teacher_changed_outside_exact_support": raw_changed_outside,
        "outside_exact_support_high_delta_pixel_fraction": high_delta_fraction,
        "deterministic_selective_repair_required": (
            high_delta_fraction > GROSS_OUTSIDE_CHANGE_FRACTION
        ),
        "inner_feather_radius_pixels": feather_radius,
        "outside_exact_support_preserved_after_inner_feather": True,
    }


def materialize_semantic_locality_seal(
    *,
    semantic_runtime_request_path: str | Path,
    semantic_runtime_result: Mapping[str, Any],
    semantic_output_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Preserve every source pixel outside every exact edit mask.

    The source, encoded mask, and raw teacher bytes are all digest-bound by the
    runtime request/result.  The returned frames contain teacher pixels only on
    the exact edit support and rendered source pixels everywhere else.
    """

    request_path = Path(semantic_runtime_request_path).expanduser().resolve()
    request = _read(
        request_path,
        code="scene_configuration_artifixer_semantic_locality_request_invalid",
    )
    result = dict(semantic_runtime_result)
    request_tasks = request.get("tasks")
    result_tasks = result.get("tasks")
    if (
        request.get("schema_version") != RUNTIME_REQUEST_SCHEMA_VERSION
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
        or result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or result.get("status")
        != "completed_unreviewed_semantic_teacher_candidates"
        or result.get("source_runtime_request_digest") != request.get("request_digest")
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
        or not isinstance(request_tasks, list)
        or len(request_tasks) != 1
        or not isinstance(result_tasks, list)
        or len(result_tasks) != 1
        or not isinstance(request_tasks[0], Mapping)
        or not isinstance(result_tasks[0], Mapping)
    ):
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_binding_invalid"
        )
    request_task = request_tasks[0]
    result_task = result_tasks[0]
    task_id = str(request_task.get("task_id") or "")
    request_frames = request_task.get("frames")
    result_frames = result_task.get("frames")
    if (
        not task_id
        or result_task.get("task_id") != task_id
        or not isinstance(request_frames, list)
        or not isinstance(result_frames, list)
        or not 1 <= len(request_frames) <= MAX_LOCALITY_SEAL_FRAMES
        or len(result_frames) != len(request_frames)
    ):
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_inventory_invalid"
        )
    result_by_camera = {
        str(row.get("camera_id") or ""): row
        for row in result_frames
        if isinstance(row, Mapping)
    }
    if len(result_by_camera) != len(result_frames):
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_inventory_invalid"
        )
    encoding = str(
        ((request.get("backend") or {}).get("execution") or {}).get(
            "mask_encoding"
        )
        or ""
    )
    semantic_root = Path(semantic_output_root).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_output_exists"
        )
    frames_root = root / "tasks" / task_id
    frames_root.mkdir(parents=True, mode=0o700)
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    try:
        for expected_index, request_frame in enumerate(request_frames):
            if not isinstance(request_frame, Mapping):
                raise TaskEvaluationSceneConfigurationSemanticLocalityError(
                    "scene_configuration_artifixer_semantic_locality_inventory_invalid"
                )
            camera_id = str(request_frame.get("camera_id") or "")
            result_frame = result_by_camera.get(camera_id)
            if (
                not camera_id
                or camera_id in observed
                or request_frame.get("frame_index") != expected_index
                or not isinstance(result_frame, Mapping)
                or result_frame.get("terminal_state")
                != "completed_unreviewed_candidate"
                or result_frame.get("source_rgb_sha256")
                != (request_frame.get("input_rgb") or {}).get("sha256")
                or result_frame.get("edit_mask_sha256")
                != (request_frame.get("edit_mask") or {}).get("sha256")
            ):
                raise TaskEvaluationSceneConfigurationSemanticLocalityError(
                    "scene_configuration_artifixer_semantic_locality_inventory_invalid"
                )
            observed.add(camera_id)
            source_path = _bound_file(
                request_frame.get("input_rgb"),
                root=request_path.parent,
                code="scene_configuration_artifixer_semantic_locality_source_invalid",
            )
            mask_path = _bound_file(
                request_frame.get("edit_mask"),
                root=request_path.parent,
                code="scene_configuration_artifixer_semantic_locality_mask_invalid",
            )
            raw_teacher_path = _bound_file(
                result_frame.get("semantic_teacher_frame"),
                root=semantic_root,
                code="scene_configuration_artifixer_semantic_locality_teacher_invalid",
            )
            destination = frames_root / f"{expected_index:05d}.png"
            frame_seal = seal_semantic_teacher_frame(
                source_path=source_path,
                mask_path=mask_path,
                raw_teacher_path=raw_teacher_path,
                mask_encoding=encoding,
                output_path=destination,
            )
            raw_changed_outside = frame_seal[
                "raw_teacher_changed_outside_exact_support"
            ]
            high_delta_fraction = frame_seal[
                "outside_exact_support_high_delta_pixel_fraction"
            ]
            deterministic_repair_required = frame_seal[
                "deterministic_selective_repair_required"
            ]
            rows.append(
                {
                    "frame_index": expected_index,
                    "camera_id": camera_id,
                    "source_frame": _record(source_path),
                    "exact_edit_mask": _record(mask_path),
                    "raw_semantic_teacher": _record(raw_teacher_path),
                    "raw_teacher_changed_outside_exact_support": (
                        raw_changed_outside
                    ),
                    "outside_exact_support_high_delta_channel_threshold": (
                        GROSS_OUTSIDE_CHANGE_CHANNEL_DELTA
                    ),
                    "outside_exact_support_high_delta_pixel_fraction": round(
                        high_delta_fraction, 9
                    ),
                    "gross_outside_change_fraction_threshold": (
                        GROSS_OUTSIDE_CHANGE_FRACTION
                    ),
                    "deterministic_selective_repair_required": (
                        deterministic_repair_required
                    ),
                    "deterministic_repair_feedback": (
                        "The failed candidate made gross changes outside the exact "
                        "edit mask. Preserve every non-target object, surface, "
                        "material feature, and pixel outside the supplied mask."
                        if deterministic_repair_required
                        else None
                    ),
                    "sealed_semantic_teacher": _record(destination, root=root),
                    "inner_feather_radius_pixels": frame_seal[
                        "inner_feather_radius_pixels"
                    ],
                    "outside_exact_support_preserved_after_inner_feather": (
                        frame_seal[
                            "outside_exact_support_preserved_after_inner_feather"
                        ]
                    ),
                    "outside_exact_support_changed_pixels_after_seal": 0,
                    "non_target_source_pixels_preserved_exactly": True,
                }
            )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    if observed != set(result_by_camera):
        shutil.rmtree(root, ignore_errors=True)
        raise TaskEvaluationSceneConfigurationSemanticLocalityError(
            "scene_configuration_artifixer_semantic_locality_inventory_invalid"
        )
    receipt: dict[str, Any] = {
        "schema_version": SEMANTIC_LOCALITY_SCHEMA_VERSION,
        "status": "semantic_teacher_exact_support_locality_sealed",
        "policy": SEMANTIC_LOCALITY_POLICY,
        "task_id": task_id,
        "source_runtime_request_digest": request["request_digest"],
        "source_runtime_result_digest": result["result_digest"],
        "frame_count": len(rows),
        "raw_outside_support_change_frame_count": sum(
            1 for row in rows if row["raw_teacher_changed_outside_exact_support"]
        ),
        "deterministic_selective_repair_frame_count": sum(
            1
            for row in rows
            if row["deterministic_selective_repair_required"]
        ),
        "gross_outside_change_channel_delta": GROSS_OUTSIDE_CHANGE_CHANNEL_DELTA,
        "gross_outside_change_fraction_threshold": GROSS_OUTSIDE_CHANGE_FRACTION,
        "maximum_locality_seal_frames": MAX_LOCALITY_SEAL_FRAMES,
        "frames": rows,
        "all_non_target_source_pixels_preserved_exactly": True,
        "semantic_object_absence_review_passed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = root / f"{SEMANTIC_LOCALITY_SCHEMA_VERSION}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_path),
        "semantic_teacher_frames_root": str(frames_root),
    }


__all__ = [
    "GROSS_OUTSIDE_CHANGE_CHANNEL_DELTA",
    "GROSS_OUTSIDE_CHANGE_FRACTION",
    "MAX_INNER_FEATHER_RADIUS_PIXELS",
    "MAX_LOCALITY_SEAL_FRAMES",
    "SEMANTIC_LOCALITY_POLICY",
    "SEMANTIC_LOCALITY_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationSemanticLocalityError",
    "materialize_semantic_locality_seal",
    "seal_semantic_teacher_frame",
]
