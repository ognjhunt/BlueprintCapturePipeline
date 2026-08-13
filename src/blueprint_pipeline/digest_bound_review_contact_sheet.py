"""Materialize lossless, digest-bound review contact sheets.

The source result remains the authority for task order, frame order, camera
identity, and file bytes.  This module adds labels in dedicated header bands and
pastes every decoded source image at integer coordinates without resizing.  It
then reopens the PNG and proves every sheet crop is pixel-identical to its
source frame before sealing the review manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "digest_bound_review_contact_sheets.v1"
SOURCE_SCHEMA_VERSION = "public_scene_artifixer3d_raw_result.v1"
FRAME_FIELDS = ("artifixer3d_review_frames", "final_candidate_frames")
COLUMNS = 4
LABEL_HEIGHT = 48
MAX_TASKS = 5
MAX_FRAMES_PER_TASK = 64
MAX_SHEET_PIXELS = 200_000_000
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ReviewContactSheetError(ValueError):
    """The source result, frame bytes, or lossless sheet proof is invalid."""


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewContactSheetError(code) from exc
    if not isinstance(value, dict):
        raise ReviewContactSheetError(code)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _pixel_sha256(image: Image.Image) -> str:
    return "sha256:" + hashlib.sha256(image.tobytes()).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_path(result_path: Path, record: Any) -> Path:
    if not isinstance(record, Mapping):
        raise ReviewContactSheetError("review_frame_record_invalid")
    raw = Path(str(record.get("path") or "")).expanduser()
    if not raw.is_absolute():
        raw = result_path.parent / raw
    if raw.is_symlink():
        raise ReviewContactSheetError("review_frame_bytes_unbound")
    path = raw.resolve()
    if (
        not path.is_file()
        or isinstance(record.get("size_bytes"), bool)
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ReviewContactSheetError("review_frame_bytes_unbound")
    return path


def _task_frames(task: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]]]:
    present = [
        field
        for field in FRAME_FIELDS
        if isinstance(task.get(field), list) and bool(task.get(field))
    ]
    if len(present) != 1:
        raise ReviewContactSheetError("review_frame_field_ambiguous")
    rows = task[present[0]]
    if not 1 <= len(rows) <= MAX_FRAMES_PER_TASK or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise ReviewContactSheetError("review_frame_count_invalid")
    return present[0], list(rows)


def _load_frames(
    *, result_path: Path, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    camera_ids: set[str] = set()
    expected_size: tuple[int, int] | None = None
    expected_mode: str | None = None
    for expected_index, row in enumerate(rows):
        frame_index = row.get("frame_index")
        camera_id = str(row.get("camera_id") or "")
        if (
            isinstance(frame_index, bool)
            or frame_index != expected_index
            or not camera_id
            or camera_id in camera_ids
        ):
            raise ReviewContactSheetError("review_frame_order_or_camera_invalid")
        source = _source_path(result_path, row)
        try:
            with Image.open(source) as opened:
                if opened.format != "PNG":
                    raise ReviewContactSheetError("review_frame_png_required")
                opened.load()
                image = opened.copy()
        except (OSError, SyntaxError) as exc:
            raise ReviewContactSheetError("review_frame_png_invalid") from exc
        if image.mode not in {"RGB", "RGBA"}:
            raise ReviewContactSheetError("review_frame_mode_invalid")
        if expected_size is None:
            expected_size = image.size
            expected_mode = image.mode
        elif image.size != expected_size or image.mode != expected_mode:
            raise ReviewContactSheetError("review_frame_geometry_or_mode_mismatch")
        camera_ids.add(camera_id)
        loaded.append(
            {
                "frame_index": expected_index,
                "camera_id": camera_id,
                "source": source,
                "image": image,
                "source_record": _file_record(source),
                "source_decoded_pixel_sha256": _pixel_sha256(image),
            }
        )
    return loaded


def _sheet_mode_color(mode: str, rgb: tuple[int, int, int]) -> tuple[int, ...]:
    return (*rgb, 255) if mode == "RGBA" else rgb


def _materialize_task_sheet(
    *, task_id: str, frames: Sequence[dict[str, Any]], output: Path
) -> dict[str, Any]:
    first = frames[0]["image"]
    frame_width, frame_height = first.size
    row_count = math.ceil(len(frames) / COLUMNS)
    sheet_size = (COLUMNS * frame_width, row_count * (LABEL_HEIGHT + frame_height))
    if sheet_size[0] * sheet_size[1] > MAX_SHEET_PIXELS:
        raise ReviewContactSheetError("review_contact_sheet_pixel_cap_exceeded")
    sheet = Image.new(
        first.mode,
        sheet_size,
        _sheet_mode_color(first.mode, (20, 20, 20)),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    placements: list[dict[str, Any]] = []
    for position, frame in enumerate(frames):
        column = position % COLUMNS
        row = position // COLUMNS
        x = column * frame_width
        label_y = row * (LABEL_HEIGHT + frame_height)
        image_y = label_y + LABEL_HEIGHT
        label = f"frame {frame['frame_index']:05d} | {frame['camera_id']}"
        draw.text(
            (x + 10, label_y + 10),
            label,
            font=font,
            fill=_sheet_mode_color(first.mode, (255, 255, 255)),
        )
        sheet.paste(frame["image"], (x, image_y))
        placements.append(
            {
                "frame_index": frame["frame_index"],
                "camera_id": frame["camera_id"],
                "label": label,
                "sheet_crop_xyxy": [
                    x,
                    image_y,
                    x + frame_width,
                    image_y + frame_height,
                ],
                "source": frame["source_record"],
                "source_decoded_pixel_sha256": frame[
                    "source_decoded_pixel_sha256"
                ],
            }
        )

    sheet_path = output / f"{task_id}_review_contact_sheet_4col.png"
    temporary = sheet_path.with_suffix(".tmp.png")
    sheet.save(temporary, format="PNG", optimize=False)
    try:
        with Image.open(temporary) as reopened:
            if reopened.format != "PNG" or reopened.mode != first.mode:
                raise ReviewContactSheetError("review_contact_sheet_round_trip_invalid")
            reopened.load()
            for placement, frame in zip(placements, frames, strict=True):
                crop = reopened.crop(tuple(placement["sheet_crop_xyxy"]))
                crop_digest = _pixel_sha256(crop)
                placement["sheet_crop_decoded_pixel_sha256"] = crop_digest
                placement["pixel_identical"] = (
                    crop.mode == frame["image"].mode
                    and crop.size == frame["image"].size
                    and crop_digest == frame["source_decoded_pixel_sha256"]
                    and crop.tobytes() == frame["image"].tobytes()
                )
                if placement["pixel_identical"] is not True:
                    raise ReviewContactSheetError(
                        "review_contact_sheet_crop_pixel_identity_failed"
                    )
    except (OSError, SyntaxError) as exc:
        raise ReviewContactSheetError("review_contact_sheet_round_trip_invalid") from exc
    temporary.replace(sheet_path)
    return {
        "task_id": task_id,
        "frame_count": len(frames),
        "columns": COLUMNS,
        "rows": row_count,
        "label_height_pixels": LABEL_HEIGHT,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_mode": first.mode,
        "contact_sheet_width": sheet_size[0],
        "contact_sheet_height": sheet_size[1],
        "contact_sheet": _file_record(sheet_path),
        "frames": placements,
        "all_sheet_crops_pixel_identical": True,
        "resampling_operations": 0,
        "display_paths": [
            str(sheet_path.resolve()),
            *(str(frame["source"].resolve()) for frame in frames),
        ],
    }


def materialize_digest_bound_review_contact_sheets(
    *, raw_result_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Create full-resolution contact sheets for every task in one raw result."""

    unresolved_result_path = Path(raw_result_path).expanduser()
    if unresolved_result_path.is_symlink():
        raise ReviewContactSheetError("review_raw_result_invalid")
    result_path = unresolved_result_path.resolve()
    unresolved_output = Path(output_root).expanduser()
    if unresolved_output.is_symlink():
        raise ReviewContactSheetError("review_output_not_empty")
    output = unresolved_output.resolve()
    raw = _read(result_path, code="review_raw_result_unreadable")
    tasks = raw.get("tasks")
    if (
        result_path.is_symlink()
        or raw.get("schema_version") != SOURCE_SCHEMA_VERSION
        or raw.get("result_digest")
        != canonical_digest(raw, digest_field="result_digest")
        or raw.get("appearance_repair_qualified") is not False
        or raw.get("generated_output_is_capture_or_physical_evidence") is not False
        or not isinstance(tasks, list)
        or not 1 <= len(tasks) <= MAX_TASKS
        or raw.get("replacement_object_count") != len(tasks)
    ):
        raise ReviewContactSheetError("review_raw_result_invalid")
    if output.exists() and (
        output.is_symlink() or not output.is_dir() or any(output.iterdir())
    ):
        raise ReviewContactSheetError("review_output_not_empty")
    output.mkdir(parents=True, exist_ok=True)

    task_results: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ReviewContactSheetError("review_task_invalid")
        task_id = str(task.get("task_id") or "")
        if not _TASK_ID.fullmatch(task_id) or task_id in task_ids:
            raise ReviewContactSheetError("review_task_id_invalid")
        frame_field, rows = _task_frames(task)
        frames = _load_frames(result_path=result_path, rows=rows)
        task_result = _materialize_task_sheet(
            task_id=task_id,
            frames=frames,
            output=output,
        )
        task_result["source_frame_field"] = frame_field
        task_results.append(task_result)
        task_ids.add(task_id)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "lossless_contact_sheets_materialized_pending_human_review",
        "source_raw_result": {
            **_file_record(result_path),
            "result_digest": raw["result_digest"],
            "pipeline_mode": raw.get("pipeline_mode"),
        },
        "layout": {
            "columns": COLUMNS,
            "label_height_pixels": LABEL_HEIGHT,
            "image_placement": "integer_coordinate_direct_paste",
            "resampling_operations": 0,
            "font": "Pillow.load_default()",
            "pillow_version": PILLOW_VERSION,
        },
        "task_count": len(task_results),
        "tasks": task_results,
        "all_sheet_crops_pixel_identical": all(
            task["all_sheet_crops_pixel_identical"] for task in task_results
        ),
        "human_visual_review": "pending",
        "appearance_repair_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "claim_boundary": (
            "Derived review layout only; pixel identity does not establish "
            "semantic quality, multiview consistency, hidden-background truth, "
            "simulation evidence, or physical evidence."
        ),
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = output / "digest_bound_review_contact_sheets.v1.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = materialize_digest_bound_review_contact_sheets(
        raw_result_path=args.raw_result,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_path": result["manifest_path"],
                "manifest_digest": result["manifest_digest"],
                "display_paths": [
                    path
                    for task in result["tasks"]
                    for path in task["display_paths"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ReviewContactSheetError",
    "SCHEMA_VERSION",
    "materialize_digest_bound_review_contact_sheets",
]
