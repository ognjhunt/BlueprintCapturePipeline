"""Bridge source-bound SAM tracks into calibrated public-scene object masks.

SAM 3.1 deliberately emits compact, provider-neutral RLE tracks.  Gaussian
excision deliberately consumes one exact binary PNG per calibrated camera.
This module is the missing deterministic bridge: it selects an explicitly
declared track union for each of 1--5 preregistered task objects, verifies the
source-frame and camera bindings, and materializes portable image/mask inputs.

The selected masks remain inferred candidate support.  They do not establish
object identity, geometry, removal quality, or physical truth; downstream
FlashSplat contribution accounting and human visual review remain mandatory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import validate_task_freeze_set
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .scene_placement.semantic_source_track_import import (
    MASK_ENCODING,
    RESULT_SCHEMA_VERSION as SOURCE_TRACK_SCHEMA_VERSION,
)


SCHEMA_VERSION = "public_scene_calibrated_object_mask_set.v1"
MAX_CAMERAS = 128


class CalibratedObjectMaskError(ValueError):
    """Stable fail-closed errors for source-track mask materialization."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    value["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return value


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibratedObjectMaskError([code]) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise CalibratedObjectMaskError([code])
    return value


def _file(path: str | Path, *, code: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise CalibratedObjectMaskError([code])
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise CalibratedObjectMaskError([code])
    return resolved


def _camera_rows(path: Path) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibratedObjectMaskError(["calibrated_masks_camera_contract_invalid"]) from exc
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_CAMERAS:
        raise CalibratedObjectMaskError(["calibrated_masks_camera_contract_invalid"])
    rows: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, Mapping):
            raise CalibratedObjectMaskError(["calibrated_masks_camera_contract_invalid"])
        row = dict(raw)
        camera_id = str(row.get("camera_id") or "").strip()
        intrinsics = row.get("intrinsics")
        if (
            not camera_id
            or camera_id in rows
            or not isinstance(intrinsics, Mapping)
            or isinstance(intrinsics.get("width"), bool)
            or not isinstance(intrinsics.get("width"), int)
            or isinstance(intrinsics.get("height"), bool)
            or not isinstance(intrinsics.get("height"), int)
            or intrinsics["width"] <= 0
            or intrinsics["height"] <= 0
        ):
            raise CalibratedObjectMaskError(["calibrated_masks_camera_contract_invalid"])
        rows[camera_id] = row
    return rows


def _verified_source_tracks(path: Path) -> dict[str, Any]:
    value = _read(path, code="calibrated_masks_source_tracks_invalid")
    bindings = value.get("bindings")
    tracks = value.get("track_registry")
    frame_masks = value.get("frame_masks")
    if (
        value.get("schema_version") != SOURCE_TRACK_SCHEMA_VERSION
        or value.get("status") != "completed"
        or value.get("result_digest") != canonical_json_digest(
            {key: item for key, item in value.items() if key != "result_digest"}
        )
        or not isinstance(bindings, Mapping)
        or not isinstance(tracks, list)
        or not tracks
        or not isinstance(frame_masks, list)
        or not frame_masks
        or bindings.get("track_registry_digest") != canonical_json_digest(tracks)
        or bindings.get("frame_masks_digest") != canonical_json_digest(frame_masks)
    ):
        raise CalibratedObjectMaskError(["calibrated_masks_source_tracks_invalid"])
    return value


def _track_map(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in value.get("track_registry") or []:
        if not isinstance(raw, Mapping):
            raise CalibratedObjectMaskError(["calibrated_masks_track_registry_invalid"])
        row = dict(raw)
        track_id = str(row.get("track_id") or "").strip()
        if not track_id or track_id in result:
            raise CalibratedObjectMaskError(["calibrated_masks_track_registry_invalid"])
        result[track_id] = row
    return result


def _frame_map(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in value.get("frame_masks") or []:
        if not isinstance(raw, Mapping):
            raise CalibratedObjectMaskError(["calibrated_masks_frame_registry_invalid"])
        row = dict(raw)
        frame_id = str(row.get("source_frame_id") or "").strip()
        width = row.get("width")
        height = row.get("height")
        if (
            not frame_id
            or frame_id in result
            or isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or row.get("mask_encoding") != MASK_ENCODING
            or not isinstance(row.get("track_masks"), list)
        ):
            raise CalibratedObjectMaskError(["calibrated_masks_frame_registry_invalid"])
        result[frame_id] = row
    return result


def _decode_union(
    frame: Mapping[str, Any], *, selected_track_ids: set[str], code: str
) -> np.ndarray:
    width = int(frame["width"])
    height = int(frame["height"])
    output = np.zeros(width * height, dtype=np.uint8)
    observed: set[str] = set()
    for raw in frame["track_masks"]:
        if not isinstance(raw, Mapping):
            raise CalibratedObjectMaskError([code])
        track_id = str(raw.get("track_id") or "")
        if track_id not in selected_track_ids:
            continue
        observed.add(track_id)
        previous_end = 0
        for run in raw.get("runs") or []:
            if not isinstance(run, Mapping):
                raise CalibratedObjectMaskError([code])
            start = run.get("start")
            length = run.get("length")
            probability = run.get("probability")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or start < previous_end
                or length <= 0
                or start + length > len(output)
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not 0.0 < float(probability) <= 1.0
            ):
                raise CalibratedObjectMaskError([code])
            output[start : start + length] = 255
            previous_end = start + length
    if observed != selected_track_ids or not np.any(output):
        raise CalibratedObjectMaskError([code])
    return output.reshape((height, width))


def materialize_calibrated_object_mask_set(
    *,
    task_freeze_paths: Sequence[str | Path],
    task_inputs: Mapping[str, Mapping[str, Any]],
    selected_track_ids_by_task: Mapping[str, Sequence[str]],
    reviewed_track_selection_receipt_path: str | Path | None = None,
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize task-local calibrated images and inferred object masks."""

    output = Path(output_root).expanduser().resolve()
    if (
        output.is_symlink()
        or (output.exists() and any(output.iterdir()))
    ):
        raise CalibratedObjectMaskError(["calibrated_masks_input_or_output_root_invalid"])
    task_paths = [
        _file(path, code="calibrated_masks_task_freeze_missing")
        for path in task_freeze_paths
    ]
    try:
        tasks = [
            _read(path, code="calibrated_masks_task_freeze_invalid")
            for path in task_paths
        ]
        task_set = validate_task_freeze_set(tasks)
    except ValueError as exc:
        raise CalibratedObjectMaskError(["calibrated_masks_task_freeze_invalid"]) from exc
    task_ids = sorted(str(task["task_id"]) for task in tasks)
    if (
        set(selected_track_ids_by_task) != set(task_ids)
        or set(task_inputs) != set(task_ids)
        or any(not isinstance(task_inputs[task_id], Mapping) for task_id in task_ids)
    ):
        raise CalibratedObjectMaskError(["calibrated_masks_task_track_map_invalid"])
    if reviewed_track_selection_receipt_path is None:
        raise CalibratedObjectMaskError(["calibrated_masks_review_receipt_missing"])
    try:
        from .public_scene_sam31_track_selection_review import (
            validate_sam31_track_selection_review,
        )

        review = validate_sam31_track_selection_review(
            receipt_path=reviewed_track_selection_receipt_path,
            task_freeze_paths=task_paths,
            task_inputs=task_inputs,
            selected_track_ids_by_task=selected_track_ids_by_task,
        )
    except ValueError as exc:
        raise CalibratedObjectMaskError(
            ["calibrated_masks_review_receipt_invalid"]
        ) from exc

    output.mkdir(parents=True)

    task_rows: list[dict[str, Any]] = []
    task_path_by_id = {
        str(task["task_id"]): path for task, path in zip(tasks, task_paths, strict=True)
    }
    for task_id in task_ids:
        task_input = task_inputs[task_id]
        source_track_path = _file(
            str(task_input.get("source_track_result_path") or ""),
            code=f"calibrated_masks_source_tracks_missing:{task_id}",
        )
        camera_path = _file(
            str(task_input.get("camera_contract_path") or ""),
            code=f"calibrated_masks_camera_contract_missing:{task_id}",
        )
        image_root = Path(str(task_input.get("source_image_root") or "")).expanduser().resolve()
        raw_camera_frame_map = task_input.get("camera_frame_map")
        if image_root.is_symlink() or not image_root.is_dir() or not isinstance(
            raw_camera_frame_map, Mapping
        ):
            raise CalibratedObjectMaskError(
                [f"calibrated_masks_task_inputs_invalid:{task_id}"]
            )
        source_tracks = _verified_source_tracks(source_track_path)
        tracks = _track_map(source_tracks)
        frames = _frame_map(source_tracks)
        cameras = _camera_rows(camera_path)
        camera_frame_map = {
            str(camera_id).strip(): str(frame_id).strip()
            for camera_id, frame_id in raw_camera_frame_map.items()
        }
        if (
            set(camera_frame_map) != set(cameras)
            or any(not frame_id for frame_id in camera_frame_map.values())
            or len(set(camera_frame_map.values())) != len(cameras)
            or set(camera_frame_map.values()) != set(frames)
        ):
            raise CalibratedObjectMaskError(
                [f"calibrated_masks_camera_frame_set_mismatch:{task_id}"]
            )
        selected = tuple(sorted(set(str(item) for item in selected_track_ids_by_task[task_id])))
        if not selected or any(not item or item not in tracks for item in selected):
            raise CalibratedObjectMaskError(
                [f"calibrated_masks_selected_tracks_invalid:{task_id}"]
            )
        task_root = output / "tasks" / task_id
        image_output = task_root / "images"
        mask_root = task_root / "masks"
        image_output.mkdir(parents=True)
        mask_root.mkdir(parents=True)
        camera_copy = task_root / "cameras.v1.json"
        shutil.copy2(camera_path, camera_copy)
        image_rows: list[dict[str, Any]] = []
        masks: list[dict[str, Any]] = []
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            source_frame_id = camera_frame_map[camera_id]
            frame = frames[source_frame_id]
            source_image = image_root / f"{camera_id}.png"
            if (
                source_image.is_symlink()
                or not source_image.is_file()
                or _sha256(source_image) != frame.get("source_frame_digest")
                or canonical_json_digest(camera) != frame.get("camera_record_digest")
            ):
                raise CalibratedObjectMaskError(
                    [f"calibrated_masks_camera_source_binding_invalid:{task_id}:{camera_id}"]
                )
            with Image.open(source_image) as image:
                image.load()
                expected = (
                    int(camera["intrinsics"]["width"]),
                    int(camera["intrinsics"]["height"]),
                )
                if image.format != "PNG" or image.size != expected or image.size != (
                    int(frame["width"]),
                    int(frame["height"]),
                ):
                    raise CalibratedObjectMaskError(
                        [f"calibrated_masks_camera_source_binding_invalid:{task_id}:{camera_id}"]
                    )
            image_destination = image_output / f"{camera_id}.png"
            shutil.copy2(source_image, image_destination)
            image_rows.append(
                {
                    "camera_id": camera_id,
                    "source_frame_id": source_frame_id,
                    "source_frame_digest": frame["source_frame_digest"],
                    "camera_record_digest": frame["camera_record_digest"],
                    "image": _record(image_destination, root=output),
                }
            )
            mask = _decode_union(
                frame,
                selected_track_ids=set(selected),
                code=f"calibrated_masks_selected_track_missing:{task_id}:{camera_id}",
            )
            destination = mask_root / f"{camera_id}.png"
            Image.fromarray(mask, mode="L").save(destination, format="PNG", optimize=False)
            masks.append(
                {
                    "camera_id": camera_id,
                    "foreground_pixel_count": int(np.count_nonzero(mask)),
                    "mask": _record(destination, root=output),
                }
            )
        task = next(row for row in tasks if row["task_id"] == task_id)
        task_rows.append(
            {
                "task_id": task_id,
                "source_object_instance_id": task["source_object"]["instance_id"],
                "source_object_semantic_label": task["source_object"]["semantic_label"],
                "task_freeze": {
                    **_record(task_path_by_id[task_id]),
                    "task_freeze_digest": task["task_freeze_digest"],
                },
                "selected_track_ids": list(selected),
                "selected_track_labels": [tracks[item]["label"] for item in selected],
                "source_track_result": {
                    **_record(source_track_path),
                    "result_digest": source_tracks["result_digest"],
                },
                "camera_contract": _record(camera_copy, root=output),
                "camera_frame_map": camera_frame_map,
                "camera_count": len(cameras),
                "source_images_root": str(image_output),
                "source_images": image_rows,
                "mask_root": str(mask_root),
                "masks": masks,
            }
        )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "calibrated_inferred_object_masks_materialized_pending_review",
        "task_count": len(task_rows),
        "camera_count_total": sum(row["camera_count"] for row in task_rows),
        "task_freeze_set_digest": task_set["set_digest"],
        "tasks": task_rows,
        "selection_authority": {
            "track_ids_explicitly_declared": True,
            "review_receipt_path": str(
                Path(reviewed_track_selection_receipt_path).expanduser().resolve()
            ),
            "review_receipt_digest": review["receipt_digest"],
            "all_selected_tracks_human_review_accepted": True,
            "cross_prompt_instance_deduplication_inferred": False,
            "mask_dilation_pixels": 0,
            "mask_values": [0, 255],
        },
        "claim_boundary": {
            "masks_are_model_inferred_candidates": True,
            "object_identity_qualified": False,
            "gaussian_ownership_qualified": False,
            "removal_qualified": False,
            "physical_evidence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / f"{SCHEMA_VERSION}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def materialize_calibrated_object_mask_set_from_tool_request(
    *, request: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Execute the registered Agents SDK tool request after digest validation."""

    if (
        request.get("schema_version")
        != "fresh_scene_calibrated_mask_tool_request.v1"
        or request.get("request_digest")
        != canonical_digest(dict(request), digest_field="request_digest")
        or not isinstance(request.get("task_freeze_paths"), list)
        or not isinstance(request.get("task_inputs"), Mapping)
        or not isinstance(request.get("selected_track_ids_by_task"), Mapping)
        or not str(request.get("reviewed_track_selection_receipt_path") or "").strip()
    ):
        raise CalibratedObjectMaskError(["calibrated_masks_tool_request_invalid"])
    return materialize_calibrated_object_mask_set(
        task_freeze_paths=[str(path) for path in request["task_freeze_paths"]],
        task_inputs=request["task_inputs"],
        selected_track_ids_by_task=request["selected_track_ids_by_task"],
        reviewed_track_selection_receipt_path=str(
            request["reviewed_track_selection_receipt_path"]
        ),
        output_root=output_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-freeze", action="append", required=True)
    parser.add_argument(
        "--task-inputs-json",
        required=True,
        help=(
            "JSON mapping task_id to source_track_result_path, camera_contract_path, "
            "source_image_root, and camera_frame_map."
        ),
    )
    parser.add_argument(
        "--selected-tracks-json",
        required=True,
        help="JSON object mapping each task_id to one or more exact SAM track IDs.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reviewed-track-selection-receipt", required=True)
    args = parser.parse_args(argv)
    selected_path = _file(
        args.selected_tracks_json, code="calibrated_masks_selected_track_map_missing"
    )
    selected = _read(
        selected_path, code="calibrated_masks_selected_track_map_invalid"
    )
    task_inputs_path = _file(
        args.task_inputs_json, code="calibrated_masks_task_inputs_missing"
    )
    task_inputs = _read(
        task_inputs_path, code="calibrated_masks_task_inputs_invalid"
    )
    materialize_calibrated_object_mask_set(
        task_freeze_paths=args.task_freeze,
        task_inputs=task_inputs,
        selected_track_ids_by_task=selected,
        reviewed_track_selection_receipt_path=args.reviewed_track_selection_receipt,
        output_root=args.output_root,
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "CalibratedObjectMaskError",
    "materialize_calibrated_object_mask_set",
    "materialize_calibrated_object_mask_set_from_tool_request",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
