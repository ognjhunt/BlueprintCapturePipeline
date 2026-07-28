"""Fail-closed crop/layout audit for the public OSCAR/RoboArena episodes.

The source videos place generated OSCAR pixels beside third-party physical
ground-truth pixels.  This module materializes the exact label-blind image bytes
that a later evaluator may consume.  Provider code must consume these audited
files, not recrop the comparison video independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np
from PIL import Image, ImageDraw

from .common import write_json
from .policy_ranking_roboarena_calibration import (
    build_phase_a_inventory,
    canonical_sha256,
    file_sha256,
)


SCHEMA_VERSION = "policy_ranking_roboarena_crop_layout_audit.v1"
EXPECTED_SOURCE_WIDTH = 1280
EXPECTED_SOURCE_HEIGHT = 480
GENERATED_CROP = (0, 0, 640, 480)
SAMPLED_FRAME_COUNT = 32
JPEG_QUALITY = 90
MIN_MEAN_HALF_DIFFERENCE = 0.005
MAX_LOCAL_DECODE_WORKERS = 8
EXPECTED_ROLLOUT_README_SHA256 = (
    "f94076393ecbfa0b9373241a701b068e76a4fc5d8cab542cda13de31f313b34e"
)
LEFT_ANCHOR = "**Left** — OSCAR world-model rollout"
RIGHT_ANCHOR = "**Right** — the real-robot policy rollout (ground truth)."


class CropAuditError(RuntimeError):
    """The comparison layout or locally materialized crop was ambiguous."""


def _frame_indices(frame_count: int, count: int = SAMPLED_FRAME_COUNT) -> list[int]:
    if frame_count <= 0:
        return []
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def _overlay_fraction(frame: np.ndarray) -> float:
    """Supporting OSCAR palette signal; never used as the layout authority."""

    blue, green, red = cv2.split(frame)
    yellow = (red > 175) & (green > 175) & (blue < 120)
    red_line = (red > 175) & (green < 155) & (blue < 155)
    green_line = (green > 170) & (red < 190) & (blue < 190)
    violet = (red > 110) & (blue > 110) & (green < 190)
    return float(np.mean(yellow | red_line | green_line | violet))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _audit_one(
    request: Mapping[str, Any],
    *,
    rollout_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    relative_path = str(request["relative_path"])
    source = (rollout_root / relative_path).resolve()
    expected_source = rollout_root.resolve()
    if source != expected_source and expected_source not in source.parents:
        raise CropAuditError("source_path_escaped_rollout_root")
    if not source.is_file():
        raise CropAuditError("source_video_missing")
    source_sha256 = file_sha256(source)
    if source_sha256 != request.get("video_sha256"):
        raise CropAuditError("source_video_sha256_mismatch")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise CropAuditError("source_video_open_failed")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = _frame_indices(frame_count)
    if len(indices) != SAMPLED_FRAME_COUNT:
        capture.release()
        raise CropAuditError("insufficient_sampled_frame_positions")

    request_id = str(request["request_id"])
    request_dir = output_root / "requests" / request_id
    frame_rows: list[dict[str, Any]] = []
    decoded_by_index: dict[int, dict[str, Any]] = {}
    wanted_indices = set(indices)
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise CropAuditError(f"source_frame_decode_failed:{frame_index}")
            if frame_index not in wanted_indices:
                continue
            height, width = frame.shape[:2]
            if (width, height) != (EXPECTED_SOURCE_WIDTH, EXPECTED_SOURCE_HEIGHT):
                raise CropAuditError(
                    f"unexpected_source_geometry:{width}x{height}:expected_1280x480"
                )
            left = np.ascontiguousarray(frame[:, : GENERATED_CROP[2]])
            right = np.ascontiguousarray(frame[:, GENERATED_CROP[2] :])
            if left.shape != (GENERATED_CROP[3], GENERATED_CROP[2], 3):
                raise CropAuditError("generated_crop_geometry_mismatch")
            encoded_ok, encoded = cv2.imencode(
                ".jpg", left, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not encoded_ok:
                raise CropAuditError(f"generated_crop_jpeg_encode_failed:{frame_index}")
            payload = encoded.tobytes()
            decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != left.shape:
                raise CropAuditError(f"generated_crop_round_trip_failed:{frame_index}")
            decoded_by_index[frame_index] = {
                "payload": payload,
                "raw_crop_bgr_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
                "encoded_jpeg_sha256": hashlib.sha256(payload).hexdigest(),
                "encoded_width": int(decoded.shape[1]),
                "encoded_height": int(decoded.shape[0]),
                "half_difference": float(
                    np.mean(np.abs(left.astype(float) - right)) / 255.0
                ),
                "left_overlay_fraction": _overlay_fraction(left),
                "right_overlay_fraction": _overlay_fraction(right),
            }
    finally:
        capture.release()

    if set(decoded_by_index) != wanted_indices:
        raise CropAuditError("sampled_source_frame_set_incomplete")
    for sample_position, frame_index in enumerate(indices):
        decoded = decoded_by_index[frame_index]
        frame_path = request_dir / "frames" / f"{sample_position:02d}_{frame_index:06d}.jpg"
        _write_bytes(frame_path, decoded["payload"])
        frame_rows.append(
            {
                "sample_position": sample_position,
                "frame_index": frame_index,
                "relative_output_path": frame_path.relative_to(output_root).as_posix(),
                "raw_crop_bgr_sha256": decoded["raw_crop_bgr_sha256"],
                "encoded_jpeg_sha256": decoded["encoded_jpeg_sha256"],
                "encoded_width": decoded["encoded_width"],
                "encoded_height": decoded["encoded_height"],
            }
        )

    mean_half_difference = float(
        np.mean([decoded_by_index[index]["half_difference"] for index in indices])
    )
    if mean_half_difference < MIN_MEAN_HALF_DIFFERENCE:
        raise CropAuditError("comparison_halves_not_materially_distinct")
    output_identity = {
        "request_id": request_id,
        "source_video_sha256": source_sha256,
        "crop_xyxy": list(GENERATED_CROP),
        "jpeg_quality": JPEG_QUALITY,
        "frames": frame_rows,
    }
    return {
        "request_id": request_id,
        "session_id": request["session_id"],
        "policy_id_internal_only": request["policy_id_internal_only"],
        "source_relative_path": relative_path,
        "source_video_sha256": source_sha256,
        "source_width": EXPECTED_SOURCE_WIDTH,
        "source_height": EXPECTED_SOURCE_HEIGHT,
        "source_frame_count": frame_count,
        "source_fps": fps,
        "generated_crop_xyxy": list(GENERATED_CROP),
        "physical_right_half_x_range": [GENERATED_CROP[2], EXPECTED_SOURCE_WIDTH],
        "physical_right_half_pixels_encoded": False,
        "sampled_frame_count": len(frame_rows),
        "unique_sampled_frame_count": len(wanted_indices),
        "repeated_sample_count": len(indices) - len(wanted_indices),
        "short_episode_source": frame_count < SAMPLED_FRAME_COUNT,
        "sampled_frames": frame_rows,
        "cropped_output_sha256": canonical_sha256(output_identity),
        "mean_normalized_left_right_pixel_difference": mean_half_difference,
        "overlay_palette_supporting_signal": {
            "authority": False,
            "left_mean_fraction": float(
                np.mean([decoded_by_index[index]["left_overlay_fraction"] for index in indices])
            ),
            "right_mean_fraction": float(
                np.mean([decoded_by_index[index]["right_overlay_fraction"] for index in indices])
            ),
        },
    }


def _representative_request_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_id_internal_only"])].append(row)
    selected: list[str] = []
    for policy_id in sorted(by_policy):
        policy_rows = sorted(
            by_policy[policy_id], key=lambda row: (str(row["session_id"]), str(row["request_id"]))
        )
        selected.append(str(policy_rows[0]["request_id"]))
        if len(policy_rows) > 1:
            selected.append(str(policy_rows[-1]["request_id"]))
    return selected


def _contact_sheet(
    rows: Sequence[Mapping[str, Any]], *, output_root: Path, output_path: Path
) -> dict[str, Any]:
    by_request = {str(row["request_id"]): row for row in rows}
    selected = _representative_request_ids(rows)
    tiles: list[tuple[Image.Image, str, int]] = []
    for request_id in selected:
        row = by_request[request_id]
        frames = list(row["sampled_frames"])
        for frame in (frames[0], frames[len(frames) // 2], frames[-1]):
            path = output_root / str(frame["relative_output_path"])
            image = Image.open(path).convert("RGB")
            image.thumbnail((256, 192), Image.Resampling.LANCZOS)
            label = (
                f"{row['policy_id_internal_only']} | {str(row['session_id'])[:8]} | "
                f"f{frame['frame_index']}"
            )
            tiles.append((image.copy(), label, int(frame["frame_index"])))
    columns = 3
    tile_width, tile_height, label_height = 256, 192, 28
    rows_count = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows_count * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (tile, label, _) in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + tile_height + 4), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=False)
    return {
        "path": str(output_path.resolve()),
        "sha256": file_sha256(output_path),
        "width": sheet.width,
        "height": sheet.height,
        "representative_request_count": len(selected),
        "representative_policy_count": len(
            {str(by_request[request_id]["policy_id_internal_only"]) for request_id in selected}
        ),
        "frame_count": len(tiles),
        "selection_rule": "lexicographically_first_and_last_session_per_policy_then_first_middle_last_sample",
        "manual_visual_review": "pending",
    }


def audit_and_materialize_generated_crops(
    inventory: Mapping[str, Any],
    *,
    rollout_root: str | Path,
    rollout_readme_path: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Audit every inventory video and materialize only generated-frame bytes."""

    root = Path(rollout_root).resolve()
    output = Path(output_root).resolve()
    readme = Path(rollout_readme_path).resolve()
    manifest = Path(manifest_path).resolve()
    blockers: list[str] = []
    readme_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    readme_sha256 = file_sha256(readme) if readme.is_file() else None
    if readme_sha256 != EXPECTED_ROLLOUT_README_SHA256:
        blockers.append("rollout_readme_sha256_mismatch")
    if LEFT_ANCHOR not in readme_text or RIGHT_ANCHOR not in readme_text:
        blockers.append("rollout_readme_layout_anchor_missing")
    requests = list(inventory.get("requests") or [])
    if inventory.get("status") != "ready" or len(requests) != 441:
        blockers.append("phase_a_inventory_not_ready_441")

    audited_rows: list[dict[str, Any]] = []
    if not blockers:
        rows_by_request: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=MAX_LOCAL_DECODE_WORKERS) as executor:
            futures = {
                executor.submit(_audit_one, request, rollout_root=root, output_root=output): request
                for request in requests
            }
            for future in as_completed(futures):
                request = futures[future]
                try:
                    row = future.result()
                    rows_by_request[str(row["request_id"])] = row
                except Exception as exc:
                    blockers.append(
                        f"crop_audit_failed:{request.get('request_id')}:"
                        f"{type(exc).__name__}:{exc}"
                    )
        audited_rows = [
            rows_by_request[str(request["request_id"])]
            for request in requests
            if str(request["request_id"]) in rows_by_request
        ]
    contact_sheet: dict[str, Any] | None = None
    if not blockers and len(audited_rows) == 441:
        contact_sheet = _contact_sheet(
            audited_rows,
            output_root=output,
            output_path=output / "review" / "representative_generated_crops.png",
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": inventory.get("experiment_id"),
        "status": "ready_for_manual_visual_review" if not blockers else "blocked",
        "inventory_sha256": inventory.get("inventory_sha256"),
        "layout_authority": {
            "source": "official_OSCAR_policy_rollout_README",
            "path": str(readme),
            "sha256": readme_sha256,
            "left_semantics": "OSCAR world-model rollout with skeleton conditioning overlaid",
            "right_semantics": "real-robot policy rollout ground truth",
        },
        "deterministic_layout_contract": {
            "source_geometry": [EXPECTED_SOURCE_WIDTH, EXPECTED_SOURCE_HEIGHT],
            "generated_crop_xyxy": list(GENERATED_CROP),
            "physical_right_half_x_range": [GENERATED_CROP[2], EXPECTED_SOURCE_WIDTH],
            "minimum_mean_normalized_left_right_pixel_difference": MIN_MEAN_HALF_DIFFERENCE,
            "sampled_frames_per_video": SAMPLED_FRAME_COUNT,
            "sampling_rule": "even_positions_with_replacement_for_sources_under_32_frames",
            "jpeg_quality": JPEG_QUALITY,
            "local_decode_workers": MAX_LOCAL_DECODE_WORKERS,
        },
        "request_count": len(requests),
        "audited_request_count": len(audited_rows),
        "all_physical_right_half_pixels_excluded": bool(audited_rows)
        and len(audited_rows) == len(requests)
        and all(not row["physical_right_half_pixels_encoded"] for row in audited_rows),
        "output_root": str(output),
        "requests": audited_rows,
        "contact_sheet": contact_sheet,
        "blockers": blockers,
        "provider_called": False,
        "data_uploaded": False,
        "outcome_labels_accessed": False,
        "claim_boundary": "Pixel/layout and local crop-byte audit only; representative manual review must pass before provider transport.",
    }
    result["audit_sha256"] = canonical_sha256(result)
    write_json(manifest, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--roboarena-root", required=True)
    parser.add_argument("--rollout-readme", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    inventory = build_phase_a_inventory(
        rollout_root=args.rollout_root, roboarena_root=args.roboarena_root
    )
    result = audit_and_materialize_generated_crops(
        inventory,
        rollout_root=args.rollout_root,
        rollout_readme_path=args.rollout_readme,
        output_root=args.output_root,
        manifest_path=args.manifest,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "requests"}, indent=2))
    return 0 if result["status"] == "ready_for_manual_visual_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
