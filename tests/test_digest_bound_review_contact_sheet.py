from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.digest_bound_review_contact_sheet import (
    ReviewContactSheetError,
    materialize_digest_bound_review_contact_sheets,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path, *, frame_index: int, camera_id: str) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "camera_id": camera_id,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_frame(
    path: Path,
    *,
    task_index: int,
    frame_index: int,
    size: tuple[int, int] = (73, 41),
) -> None:
    image = Image.new(
        "RGB",
        size,
        (
            (task_index * 41 + frame_index * 13) % 256,
            (task_index * 29 + frame_index * 31) % 256,
            (task_index * 17 + frame_index * 47) % 256,
        ),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (frame_index % 11, task_index % 7, size[0] - 2, size[1] - 2),
        outline=(255 - frame_index, 127 + task_index, frame_index + task_index),
        width=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _write_raw_result(
    root: Path,
    *,
    task_count: int = 1,
    frames_per_task: int = 8,
) -> tuple[Path, dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task_index in range(task_count):
        rows: list[dict[str, Any]] = []
        for frame_index in range(frames_per_task):
            path = (
                root
                / "runtime_output"
                / f"task_{task_index}"
                / "artifixer3d_review_frames"
                / f"{frame_index:05d}.png"
            )
            _write_frame(
                path,
                task_index=task_index,
                frame_index=frame_index,
            )
            rows.append(
                _file_record(
                    path,
                    frame_index=frame_index,
                    camera_id=f"task_{task_index}_camera_{frame_index:05d}",
                )
            )
        tasks.append(
            {
                "task_id": f"task_{task_index}",
                "artifixer3d_review_frames": rows,
            }
        )
    raw: dict[str, Any] = {
        "schema_version": "public_scene_artifixer3d_raw_result.v1",
        "status": "completed_unqualified_candidate",
        "pipeline_mode": "fixture_render_only",
        "replacement_object_count": task_count,
        "tasks": tasks,
        "appearance_repair_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "result_digest": "",
    }
    raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
    path = root / "public_scene_artifixer3d_raw_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, raw


def _reseal(path: Path, raw: dict[str, Any]) -> None:
    raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materializes_full_resolution_4x2_sheet_with_pixel_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path, _raw = _write_raw_result(tmp_path / "source")

    def _resampling_forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("review materializer must not resample source pixels")

    monkeypatch.setattr(Image.Image, "resize", _resampling_forbidden)
    monkeypatch.setattr(Image.Image, "thumbnail", _resampling_forbidden)
    output = tmp_path / "review"
    result = materialize_digest_bound_review_contact_sheets(
        raw_result_path=raw_path,
        output_root=output,
    )

    assert result["task_count"] == 1
    assert result["all_sheet_crops_pixel_identical"] is True
    assert result["layout"]["resampling_operations"] == 0
    task = result["tasks"][0]
    assert task["frame_count"] == 8
    assert task["columns"] == 4
    assert task["rows"] == 2
    assert task["contact_sheet_width"] == 4 * 73
    assert task["contact_sheet_height"] == 2 * (48 + 41)
    assert task["resampling_operations"] == 0
    assert Path(task["display_paths"][0]).is_absolute()
    assert all(Path(path).is_absolute() for path in task["display_paths"])

    with Image.open(task["contact_sheet"]["path"]) as sheet:
        sheet.load()
        for frame in task["frames"]:
            with Image.open(frame["source"]["path"]) as source:
                source.load()
                crop = sheet.crop(tuple(frame["sheet_crop_xyxy"]))
                assert crop.mode == source.mode
                assert crop.size == source.size
                assert crop.tobytes() == source.tobytes()
                assert frame["pixel_identical"] is True

    manifest_path = Path(result["manifest_path"])
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["manifest_digest"] == canonical_digest(
        on_disk, digest_field="manifest_digest"
    )
    assert on_disk["source_raw_result"]["sha256"] == _sha256(raw_path)


def test_supports_five_tasks_with_eight_frames_each(tmp_path: Path) -> None:
    raw_path, _raw = _write_raw_result(
        tmp_path / "source",
        task_count=5,
        frames_per_task=8,
    )
    result = materialize_digest_bound_review_contact_sheets(
        raw_result_path=raw_path,
        output_root=tmp_path / "review",
    )

    assert result["task_count"] == 5
    assert [task["task_id"] for task in result["tasks"]] == [
        f"task_{index}" for index in range(5)
    ]
    assert all(task["frame_count"] == 8 for task in result["tasks"])
    assert all(task["rows"] == 2 for task in result["tasks"])
    assert all(
        task["all_sheet_crops_pixel_identical"] is True
        for task in result["tasks"]
    )
    assert len(list((tmp_path / "review").glob("*_4col.png"))) == 5


def test_accepts_final_composite_frames_without_weakening_qualification(
    tmp_path: Path,
) -> None:
    raw_path, raw = _write_raw_result(tmp_path / "source")
    raw["schema_version"] = "public_scene_artifixer3d_final_composite.v1"
    raw["tasks"][0]["frames"] = raw["tasks"][0].pop(
        "artifixer3d_review_frames"
    )
    raw.pop("result_digest")
    raw["receipt_digest"] = ""
    raw["receipt_digest"] = canonical_digest(raw, digest_field="receipt_digest")
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")

    result = materialize_digest_bound_review_contact_sheets(
        raw_result_path=raw_path,
        output_root=tmp_path / "review",
    )

    assert result["tasks"][0]["source_frame_field"] == "frames"
    assert result["appearance_repair_qualified"] is False
    assert result["all_sheet_crops_pixel_identical"] is True


def test_fails_closed_for_source_byte_and_raw_digest_tamper(tmp_path: Path) -> None:
    raw_path, raw = _write_raw_result(tmp_path / "source")
    first_frame = Path(raw["tasks"][0]["artifixer3d_review_frames"][0]["path"])
    first_frame.write_bytes(first_frame.read_bytes() + b"tamper")
    with pytest.raises(ReviewContactSheetError, match="review_frame_bytes_unbound"):
        materialize_digest_bound_review_contact_sheets(
            raw_result_path=raw_path,
            output_root=tmp_path / "byte_tamper_output",
        )

    raw_path, raw = _write_raw_result(tmp_path / "digest_source")
    raw["pipeline_mode"] = "unsealed_tamper"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReviewContactSheetError, match="review_raw_result_invalid"):
        materialize_digest_bound_review_contact_sheets(
            raw_result_path=raw_path,
            output_root=tmp_path / "digest_tamper_output",
        )


def test_fails_closed_for_path_tamper_and_ambiguous_frame_fields(
    tmp_path: Path,
) -> None:
    raw_path, raw = _write_raw_result(tmp_path / "path_source")
    raw["tasks"][0]["artifixer3d_review_frames"][0]["path"] = str(
        tmp_path / "not_the_bound_frame.png"
    )
    _reseal(raw_path, raw)
    with pytest.raises(ReviewContactSheetError, match="review_frame_bytes_unbound"):
        materialize_digest_bound_review_contact_sheets(
            raw_result_path=raw_path,
            output_root=tmp_path / "path_tamper_output",
        )

    raw_path, raw = _write_raw_result(tmp_path / "ambiguous_source")
    raw["tasks"][0]["final_candidate_frames"] = list(
        raw["tasks"][0]["artifixer3d_review_frames"]
    )
    _reseal(raw_path, raw)
    with pytest.raises(ReviewContactSheetError, match="review_frame_field_ambiguous"):
        materialize_digest_bound_review_contact_sheets(
            raw_result_path=raw_path,
            output_root=tmp_path / "ambiguous_output",
        )


def test_fails_closed_for_nonempty_output_and_frame_geometry_mismatch(
    tmp_path: Path,
) -> None:
    raw_path, _raw = _write_raw_result(tmp_path / "nonempty_source")
    output = tmp_path / "nonempty_output"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(ReviewContactSheetError, match="review_output_not_empty"):
        materialize_digest_bound_review_contact_sheets(
            raw_result_path=raw_path,
            output_root=output,
        )
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "preserve me"

    raw_path, raw = _write_raw_result(tmp_path / "geometry_source")
    row = raw["tasks"][0]["artifixer3d_review_frames"][-1]
    last_frame = Path(row["path"])
    _write_frame(last_frame, task_index=0, frame_index=7, size=(74, 41))
    row["size_bytes"] = last_frame.stat().st_size
    row["sha256"] = _sha256(last_frame)
    _reseal(raw_path, raw)
    with pytest.raises(
        ReviewContactSheetError,
        match="review_frame_geometry_or_mode_mismatch",
    ):
        materialize_digest_bound_review_contact_sheets(
            raw_result_path=raw_path,
            output_root=tmp_path / "geometry_output",
        )
