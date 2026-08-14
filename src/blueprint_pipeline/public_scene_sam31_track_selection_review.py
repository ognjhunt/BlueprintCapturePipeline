"""Render and seal human review of SAM track selections for 1--5 tasks.

The candidate packet is deterministic visual support.  A separate acceptance
receipt binds the exact task freezes, normalized SAM results, selected track
IDs, and review media.  Downstream mask materialization must reopen that
receipt; a naked digest or an agent's prose is not selection authority.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from .decision_evidence_contracts import canonical_digest, canonical_json
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest


CANDIDATE_SCHEMA_VERSION = "public_scene_sam31_track_selection_review_candidate.v1"
RECEIPT_SCHEMA_VERSION = "public_scene_sam31_track_selection_review.v1"


class Sam31TrackSelectionReviewError(ValueError):
    """A review candidate or acceptance receipt is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value: dict[str, Any] = {
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }
    if root is not None:
        try:
            value["relative_path"] = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            value["path"] = str(resolved)
    else:
        value["path"] = str(resolved)
    return value


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sam31TrackSelectionReviewError(code) from exc
    if unresolved.is_symlink() or not resolved.is_file() or not isinstance(value, dict):
        raise Sam31TrackSelectionReviewError(code)
    return resolved, value


def _record_path(record: Mapping[str, Any], *, root: Path) -> Path:
    relative = record.get("relative_path")
    absolute = record.get("path")
    if bool(relative) == bool(absolute):
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    path = (root / str(relative)).resolve() if relative else Path(str(absolute)).resolve()
    if relative:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise Sam31TrackSelectionReviewError(
                "sam31_review_media_record_invalid"
            ) from exc
    return path


def _verify_record(record: object, *, root: Path) -> Path:
    if not isinstance(record, Mapping):
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    path = _record_path(record, root=root)
    if path.is_symlink() or not path.is_file():
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    expected = _record(path, root=root if record.get("relative_path") else None)
    if any(record.get(key) != value for key, value in expected.items()):
        raise Sam31TrackSelectionReviewError("sam31_review_media_record_invalid")
    return path


def _validate_candidate_file(candidate_path: Path, candidate: Mapping[str, Any]) -> None:
    if (
        candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or candidate.get("status")
        != "selected_track_overlays_materialized_pending_human_review"
        or candidate.get("candidate_digest")
        != canonical_digest(candidate, digest_field="candidate_digest")
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    root = candidate_path.parent.resolve()
    bindings = candidate.get("selection_bindings")
    review_media = candidate.get("review_media")
    if (
        not isinstance(bindings, list)
        or not isinstance(review_media, list)
        or not 1 <= len(bindings) <= 5
        or candidate.get("task_count") != len(bindings)
        or candidate.get("task_count") != len(review_media)
        or Path(str(candidate.get("candidate_masks_root") or "")).resolve()
        != root / "candidate_masks"
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    task_ids = [str(row.get("task_id") or "") for row in bindings if isinstance(row, Mapping)]
    if len(task_ids) != len(bindings) or not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    for binding in bindings:
        _verify_record(binding.get("task_freeze"), root=root)
        _verify_record(binding.get("source_track_result"), root=root)
        _verify_record(binding.get("camera_contract"), root=root)
    if [str(row.get("task_id") or "") for row in review_media] != task_ids:
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    for task in review_media:
        frames = task.get("frames")
        if (
            not isinstance(frames, list)
            or not frames
            or task.get("camera_count") != len(frames)
            or len({str(row.get("camera_id") or "") for row in frames}) != len(frames)
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
        for frame in frames:
            source_path = _verify_record(frame.get("source_image"), root=root)
            mask_path = _verify_record(frame.get("selected_mask"), root=root)
            overlay_path = _verify_record(frame.get("overlay"), root=root)
            with (
                Image.open(source_path) as source_file,
                Image.open(mask_path) as mask_file,
                Image.open(overlay_path) as overlay_file,
            ):
                source = source_file.convert("RGB")
                mask = mask_file.convert("L")
                overlay = overlay_file.convert("RGB")
                histogram = mask.histogram()
                foreground = sum(histogram[1:])
                if (
                    source_file.format != "PNG"
                    or mask_file.format != "PNG"
                    or overlay_file.format != "PNG"
                    or source.size != mask.size
                    or source.size != overlay.size
                    or foreground != frame.get("foreground_pixel_count")
                    or foreground != histogram[255]
                ):
                    raise Sam31TrackSelectionReviewError(
                        "sam31_review_media_content_invalid"
                    )
                color = Image.new("RGB", source.size, (255, 0, 160))
                alpha = mask.point(lambda value: 128 if value else 0)
                expected_overlay = Image.composite(color, source, alpha)
                if ImageChops.difference(expected_overlay, overlay).getbbox() is not None:
                    raise Sam31TrackSelectionReviewError(
                        "sam31_review_media_content_invalid"
                    )


def _selection_bindings(
    *,
    task_freeze_paths: Sequence[str | Path],
    task_inputs: Mapping[str, Mapping[str, Any]],
    selected_track_ids_by_task: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for freeze_value in task_freeze_paths:
        freeze_path, freeze = _read(freeze_value, code="sam31_review_task_freeze_invalid")
        task_id = str(freeze.get("task_id") or "")
        raw_input = task_inputs.get(task_id)
        selected = sorted(set(str(item) for item in selected_track_ids_by_task.get(task_id, [])))
        if (
            not task_id
            or not isinstance(raw_input, Mapping)
            or not selected
            or freeze.get("task_freeze_digest")
            != canonical_digest(freeze, digest_field="task_freeze_digest")
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_selection_binding_invalid")
        tracks_path, tracks = _read(
            str(raw_input.get("source_track_result_path") or ""),
            code="sam31_review_source_tracks_invalid",
        )
        cameras_path = Path(str(raw_input.get("camera_contract_path") or "")).expanduser().resolve()
        image_root = Path(str(raw_input.get("source_image_root") or "")).expanduser().resolve()
        camera_frame_map = raw_input.get("camera_frame_map")
        if (
            cameras_path.is_symlink()
            or not cameras_path.is_file()
            or image_root.is_symlink()
            or not image_root.is_dir()
            or not isinstance(camera_frame_map, Mapping)
            or not camera_frame_map
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_task_inputs_invalid")
        available = {
            str(row.get("track_id") or "")
            for row in tracks.get("track_registry") or []
            if isinstance(row, Mapping)
        }
        if (
            tracks.get("schema_version") != "semantic_source_track_import_result.v1"
            or tracks.get("status") != "completed"
            or tracks.get("result_digest")
            != canonical_json_digest(
                {key: value for key, value in tracks.items() if key != "result_digest"}
            )
            or any(item not in available for item in selected)
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_source_tracks_invalid")
        bindings.append(
            {
                "task_id": task_id,
                "task_freeze": {
                    **_record(freeze_path),
                    "task_freeze_digest": freeze["task_freeze_digest"],
                },
                "source_track_result": {
                    **_record(tracks_path),
                    "result_digest": tracks["result_digest"],
                },
                "camera_contract": _record(cameras_path),
                "source_image_root": str(image_root),
                "camera_frame_map": {
                    str(camera_id): str(frame_id)
                    for camera_id, frame_id in sorted(camera_frame_map.items())
                },
                "selected_track_ids": selected,
                "selected_track_labels": sorted(
                    str(row.get("label") or "")
                    for row in tracks["track_registry"]
                    if row.get("track_id") in selected
                ),
            }
        )
    if (
        not 1 <= len(bindings) <= 5
        or len({row["task_id"] for row in bindings}) != len(bindings)
        or set(task_inputs) != {row["task_id"] for row in bindings}
        or set(selected_track_ids_by_task) != {row["task_id"] for row in bindings}
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_task_set_invalid")
    return sorted(bindings, key=lambda row: row["task_id"])


def _write_overlay(*, image_path: Path, mask_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as source_image, Image.open(mask_path) as source_mask:
        image = source_image.convert("RGB")
        mask = source_mask.convert("L")
        if image.size != mask.size:
            raise Sam31TrackSelectionReviewError("sam31_review_overlay_dimensions_invalid")
        color = Image.new("RGB", image.size, (255, 0, 160))
        alpha = mask.point(lambda value: 128 if value else 0)
        overlay = Image.composite(color, image, alpha)
        overlay.save(output_path, format="PNG", optimize=False)


def materialize_sam31_track_selection_review_candidate(
    *,
    task_freeze_paths: Sequence[str | Path],
    task_inputs: Mapping[str, Mapping[str, Any]],
    selected_track_ids_by_task: Mapping[str, Sequence[str]],
    output_root: str | Path,
) -> dict[str, Any]:
    """Render exact selected masks over calibrated frames, pending human review."""

    from .public_scene_calibrated_object_masks import (
        _camera_rows,
        _decode_union,
        _frame_map,
        _track_map,
        _verified_source_tracks,
    )

    output = Path(output_root).expanduser().resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise Sam31TrackSelectionReviewError("sam31_review_output_not_empty")
    bindings = _selection_bindings(
        task_freeze_paths=task_freeze_paths,
        task_inputs=task_inputs,
        selected_track_ids_by_task=selected_track_ids_by_task,
    )
    masks_root = output / "candidate_masks"
    masks_root.mkdir(parents=True)
    review_rows: list[dict[str, Any]] = []
    for binding in bindings:
        task_id = binding["task_id"]
        task_input = task_inputs[task_id]
        tracks_path = Path(str(task_input["source_track_result_path"])).expanduser().resolve()
        cameras_path = Path(str(task_input["camera_contract_path"])).expanduser().resolve()
        image_root = Path(str(task_input["source_image_root"])).expanduser().resolve()
        source_tracks = _verified_source_tracks(tracks_path)
        tracks = _track_map(source_tracks)
        frames = _frame_map(source_tracks)
        cameras = _camera_rows(cameras_path)
        camera_frame_map = {
            str(camera_id): str(frame_id)
            for camera_id, frame_id in task_input["camera_frame_map"].items()
        }
        selected = set(binding["selected_track_ids"])
        if (
            set(camera_frame_map) != set(cameras)
            or set(camera_frame_map.values()) != set(frames)
            or any(item not in tracks for item in selected)
        ):
            raise Sam31TrackSelectionReviewError("sam31_review_camera_frame_set_invalid")
        media_root = output / "review_media" / task_id
        media_root.mkdir(parents=True)
        mask_task_root = masks_root / task_id
        mask_task_root.mkdir(parents=True)
        frame_rows: list[dict[str, Any]] = []
        for camera_id in sorted(cameras):
            source_frame_id = camera_frame_map[camera_id]
            frame = frames[source_frame_id]
            image_path = image_root / f"{camera_id}.png"
            if (
                not image_path.is_file()
                or image_path.is_symlink()
                or _sha256(image_path) != frame.get("source_frame_digest")
                or canonical_json_digest(cameras[camera_id])
                != frame.get("camera_record_digest")
            ):
                raise Sam31TrackSelectionReviewError("sam31_review_source_image_invalid")
            with Image.open(image_path) as image:
                expected_size = (
                    int(cameras[camera_id]["intrinsics"]["width"]),
                    int(cameras[camera_id]["intrinsics"]["height"]),
                )
                if (
                    image.format != "PNG"
                    or image.size != expected_size
                    or image.size != (int(frame["width"]), int(frame["height"]))
                ):
                    raise Sam31TrackSelectionReviewError(
                        "sam31_review_source_image_invalid"
                    )
            mask = _decode_union(
                frame,
                selected_track_ids=selected,
                code=f"sam31_review_selected_track_missing:{task_id}:{camera_id}",
            )
            mask_path = mask_task_root / f"{camera_id}.png"
            Image.fromarray(mask, mode="L").save(mask_path, format="PNG", optimize=False)
            overlay_path = media_root / f"{camera_id}.png"
            _write_overlay(image_path=image_path, mask_path=mask_path, output_path=overlay_path)
            frame_rows.append(
                {
                    "camera_id": camera_id,
                    "source_image": _record(image_path, root=output),
                    "selected_mask": _record(mask_path, root=output),
                    "overlay": _record(overlay_path, root=output),
                    "foreground_pixel_count": int((mask != 0).sum()),
                }
            )
        review_rows.append(
            {"task_id": task_id, "camera_count": len(frame_rows), "frames": frame_rows}
        )
    candidate: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "status": "selected_track_overlays_materialized_pending_human_review",
        "task_count": len(bindings),
        "selection_bindings": bindings,
        "candidate_masks_root": str(masks_root),
        "review_media": review_rows,
        "overlay_policy": {
            "selected_pixels_rgb": [255, 0, 160],
            "selected_pixels_alpha_255": 128,
            "source_pixels_resampled": False,
        },
        "claim_boundary": {
            "human_review_completed": False,
            "object_identity_qualified": False,
            "gaussian_ownership_qualified": False,
            "physical_evidence": False,
        },
        "candidate_digest": "",
    }
    candidate["candidate_digest"] = canonical_digest(
        candidate, digest_field="candidate_digest"
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{CANDIDATE_SCHEMA_VERSION}.json"
    destination.write_text(canonical_json(candidate) + "\n", encoding="utf-8")
    _validate_candidate_file(destination, candidate)
    return candidate


def seal_sam31_track_selection_review(
    *,
    candidate_path: str | Path,
    reviewed_by: str,
    reviewed_on: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Accept every selected task track after inspecting the rendered overlays."""

    candidate_file, candidate = _read(candidate_path, code="sam31_review_candidate_invalid")
    _validate_candidate_file(candidate_file, candidate)
    if not str(reviewed_by).strip() or not str(reviewed_on).strip():
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "selected_tracks_human_review_accepted",
        "candidate": {
            **_record(candidate_file),
            "candidate_digest": candidate["candidate_digest"],
        },
        "selection_bindings": candidate["selection_bindings"],
        "task_count": candidate["task_count"],
        "reviewed_by": str(reviewed_by).strip(),
        "reviewed_on": str(reviewed_on).strip(),
        "all_selected_tracks_accepted": True,
        "agent_selected_tracks_without_human_review": False,
        "claim_boundary": {
            "track_selection_reviewed": True,
            "object_identity_qualified": False,
            "gaussian_ownership_qualified": False,
            "physical_evidence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise Sam31TrackSelectionReviewError("sam31_review_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def validate_sam31_track_selection_review(
    *,
    receipt_path: str | Path,
    task_freeze_paths: Sequence[str | Path],
    task_inputs: Mapping[str, Mapping[str, Any]],
    selected_track_ids_by_task: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Reopen a human receipt and prove it accepts these exact selection inputs."""

    _path, receipt = _read(receipt_path, code="sam31_review_receipt_invalid")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "selected_tracks_human_review_accepted"
        or receipt.get("all_selected_tracks_accepted") is not True
        or receipt.get("agent_selected_tracks_without_human_review") is not False
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_receipt_invalid")
    candidate_record = receipt.get("candidate")
    if not isinstance(candidate_record, Mapping):
        raise Sam31TrackSelectionReviewError("sam31_review_receipt_invalid")
    candidate_path, candidate = _read(
        str(candidate_record.get("path") or ""), code="sam31_review_candidate_invalid"
    )
    if (
        _record(candidate_path) != {
            key: candidate_record.get(key) for key in ("path", "size_bytes", "sha256")
        }
        or candidate.get("candidate_digest") != candidate_record.get("candidate_digest")
        or candidate.get("candidate_digest")
        != canonical_digest(candidate, digest_field="candidate_digest")
    ):
        raise Sam31TrackSelectionReviewError("sam31_review_candidate_invalid")
    _validate_candidate_file(candidate_path, candidate)
    expected = _selection_bindings(
        task_freeze_paths=task_freeze_paths,
        task_inputs=task_inputs,
        selected_track_ids_by_task=selected_track_ids_by_task,
    )
    if receipt.get("selection_bindings") != expected or candidate.get(
        "selection_bindings"
    ) != expected:
        raise Sam31TrackSelectionReviewError("sam31_review_selection_mismatch")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--task-freeze", action="append", required=True)
    candidate.add_argument("--task-inputs", required=True)
    candidate.add_argument("--selected-tracks", required=True)
    candidate.add_argument("--output-root", required=True)
    accept = commands.add_parser("accept")
    accept.add_argument("--candidate", required=True)
    accept.add_argument("--reviewed-by", required=True)
    accept.add_argument("--reviewed-on", required=True)
    accept.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "candidate":
        _task_inputs_path, task_inputs = _read(
            args.task_inputs, code="sam31_review_task_inputs_invalid"
        )
        _selected_path, selected_tracks = _read(
            args.selected_tracks, code="sam31_review_selected_tracks_invalid"
        )
        materialize_sam31_track_selection_review_candidate(
            task_freeze_paths=args.task_freeze,
            task_inputs=task_inputs,
            selected_track_ids_by_task=selected_tracks,
            output_root=args.output_root,
        )
    else:
        seal_sam31_track_selection_review(
            candidate_path=args.candidate,
            reviewed_by=args.reviewed_by,
            reviewed_on=args.reviewed_on,
            output_path=args.output,
        )
    return 0


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "Sam31TrackSelectionReviewError",
    "materialize_sam31_track_selection_review_candidate",
    "seal_sam31_track_selection_review",
    "validate_sam31_track_selection_review",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
