"""Build one task-local SAM 3.1 packet from calibrated public-scene renders.

This is a deterministic adapter between the exact-camera render receipt and
the existing SAM 3.1 paid lane. It removes the historical need to hand-author
frame registries, camera joins, JPEG derivatives, or retained-sequence digests.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import validate_task_freeze
from .scene_placement.sam31_source_track_provider import RUN_REQUEST_SCHEMA_VERSION
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest


SCHEMA_VERSION = "public_scene_sam31_task_input_packet.v1"


class PublicSceneSam31InputError(ValueError):
    """Fail-closed public-scene SAM packet error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicSceneSam31InputError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise PublicSceneSam31InputError(code)
    return value


def _file(path: str | Path, *, code: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise PublicSceneSam31InputError(code)
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise PublicSceneSam31InputError(code)
    return resolved


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    value["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return value


def _encode_lossless_sequence(
    *, frame_root: Path, frame_rate_hz: int, output_path: Path, ffmpeg: str
) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-framerate",
        str(frame_rate_hz),
        "-i",
        str(frame_root / "%06d.png"),
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-threads",
        "1",
        "-pix_fmt",
        "rgb24",
        str(output_path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicSceneSam31InputError("sam31_task_input_lossless_encode_failed") from exc
    if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise PublicSceneSam31InputError("sam31_task_input_lossless_encode_failed")
    return command


def materialize_public_scene_sam31_task_inputs(
    *,
    calibrated_view_receipt_path: str | Path,
    task_freeze_path: str | Path,
    provider_profile_path: str | Path,
    prompts_path: str | Path,
    output_root: str | Path,
    frame_rate_hz: int = 1,
    ffmpeg_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize a portable, source-bound SAM request without execution."""

    receipt_path = _file(
        calibrated_view_receipt_path, code="sam31_task_input_calibrated_receipt_missing"
    )
    freeze_path = _file(task_freeze_path, code="sam31_task_input_task_freeze_missing")
    profile_path = _file(provider_profile_path, code="sam31_task_input_profile_missing")
    prompt_path = _file(prompts_path, code="sam31_task_input_prompts_missing")
    output = Path(output_root).expanduser().resolve()
    if (
        output.is_symlink()
        or (output.exists() and any(output.iterdir()))
        or isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or not 1 <= frame_rate_hz <= 60
    ):
        raise PublicSceneSam31InputError("sam31_task_input_output_or_rate_invalid")
    receipt = _read(receipt_path, code="sam31_task_input_calibrated_receipt_invalid")
    freeze = validate_task_freeze(
        _read(freeze_path, code="sam31_task_input_task_freeze_invalid")
    )
    profile = _read(profile_path, code="sam31_task_input_profile_invalid")
    prompts_value = json.loads(prompt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "public_scene_interiorgs_edit_input_receipt.v2"
        or receipt.get("status") != "render_derived_input_packet_materialized"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("scene", {}).get("task_id") != freeze["task_id"]
        or profile.get("profile_digest")
        != canonical_json_digest(
            {key: value for key, value in profile.items() if key != "profile_digest"}
        )
        or not isinstance(prompts_value, list)
        or not prompts_value
    ):
        raise PublicSceneSam31InputError("sam31_task_input_contract_invalid")
    artifacts = receipt.get("derived_artifacts")
    if not isinstance(artifacts, Mapping):
        raise PublicSceneSam31InputError("sam31_task_input_artifacts_invalid")
    camera_record = artifacts.get("cameras")
    image_records = artifacts.get("images")
    if not isinstance(camera_record, Mapping) or not isinstance(image_records, list):
        raise PublicSceneSam31InputError("sam31_task_input_artifacts_invalid")
    source_root = receipt_path.parent
    camera_path = source_root / str(camera_record.get("relative_path") or "")
    if (
        camera_path.is_symlink()
        or not camera_path.is_file()
        or camera_path.stat().st_size != camera_record.get("size_bytes")
        or _sha256(camera_path) != camera_record.get("sha256")
    ):
        raise PublicSceneSam31InputError("sam31_task_input_camera_bytes_changed")
    cameras_value = json.loads(camera_path.read_text(encoding="utf-8"))
    if not isinstance(cameras_value, list) or len(cameras_value) != len(image_records):
        raise PublicSceneSam31InputError("sam31_task_input_camera_set_invalid")
    cameras = {str(row.get("camera_id") or ""): row for row in cameras_value}
    images = {str(row.get("camera_id") or ""): row for row in image_records}
    if not cameras or set(cameras) != set(images) or "" in cameras:
        raise PublicSceneSam31InputError("sam31_task_input_camera_set_invalid")

    output.mkdir(parents=True)
    png_root = output / "lossless_frames"
    jpeg_root = output / "analysis_jpegs"
    png_root.mkdir()
    jpeg_root.mkdir()
    ordered_camera_ids = sorted(cameras)
    source_rows: list[dict[str, Any]] = []
    for index, camera_id in enumerate(ordered_camera_ids):
        row = images[camera_id]
        source = source_root / str(row.get("relative_path") or "")
        camera = cameras[camera_id]
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != row.get("size_bytes")
            or _sha256(source) != row.get("sha256")
        ):
            raise PublicSceneSam31InputError(
                f"sam31_task_input_image_bytes_changed:{camera_id}"
            )
        with Image.open(source) as image:
            image.load()
            if image.format != "PNG":
                raise PublicSceneSam31InputError(
                    f"sam31_task_input_image_format_invalid:{camera_id}"
                )
            rgb = image.convert("RGB")
            width, height = rgb.size
            png = png_root / f"{index:06d}.png"
            shutil.copy2(source, png)
            jpeg = jpeg_root / f"{index:06d}.jpg"
            rgb.save(jpeg, format="JPEG", quality=95, subsampling=0, optimize=False)
        source_rows.append(
            {
                "index": index,
                "camera_id": camera_id,
                "camera": camera,
                "camera_record_digest": canonical_json_digest(camera),
                "source_png": png,
                "source_png_digest": _sha256(png),
                "analysis_jpeg": jpeg,
                "analysis_jpeg_digest": _sha256(jpeg),
                "width": width,
                "height": height,
            }
        )
    dimensions = {(row["width"], row["height"]) for row in source_rows}
    if len(dimensions) != 1:
        raise PublicSceneSam31InputError("sam31_task_input_frame_dimensions_disagree")
    ffmpeg = (
        str(Path(ffmpeg_executable).expanduser().resolve())
        if ffmpeg_executable is not None
        else str(shutil.which("ffmpeg") or "")
    )
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise PublicSceneSam31InputError("sam31_task_input_ffmpeg_missing")
    retained_sequence = output / "retained_calibrated_sequence.mkv"
    encode_command = _encode_lossless_sequence(
        frame_root=png_root,
        frame_rate_hz=frame_rate_hz,
        output_path=retained_sequence,
        ffmpeg=ffmpeg,
    )
    retained_digest = _sha256(retained_sequence)
    camera_solution_digest = canonical_json_digest(cameras_value)
    frame_registry: list[dict[str, Any]] = []
    frame_artifacts: list[dict[str, Any]] = []
    camera_frame_map: dict[str, str] = {}
    for row in source_rows:
        frame_id = f"{freeze['task_id']}:{row['camera_id']}"
        pts = row["index"] / frame_rate_hz
        sync_digest = canonical_json_digest(
            {
                "source_frame_id": frame_id,
                "camera_id": row["camera_id"],
                "model_frame_index": row["index"],
                "decoded_pts_seconds": pts,
                "camera_record_digest": row["camera_record_digest"],
            }
        )
        frame_registry.append(
            {
                "source_frame_id": frame_id,
                "model_frame_index": row["index"],
                "source_frame_digest": row["source_png_digest"],
                "retained_video_digest": retained_digest,
                "decoded_pts_seconds": pts,
                "sync_map_row_digest": sync_digest,
                "camera_record_digest": row["camera_record_digest"],
                "encoder_retained": True,
                "width": row["width"],
                "height": row["height"],
                "analysis_jpeg_digest": row["analysis_jpeg_digest"],
            }
        )
        frame_artifacts.append(
            {
                "source_frame_id": frame_id,
                "path": str(row["analysis_jpeg"]),
                "media_type": "image/jpeg",
                "sha256": row["analysis_jpeg_digest"],
                "size_bytes": row["analysis_jpeg"].stat().st_size,
            }
        )
        camera_frame_map[row["camera_id"]] = frame_id
    run_request = {
        "schema_version": RUN_REQUEST_SCHEMA_VERSION,
        "bindings": {
            "capture_digest": receipt["receipt_digest"],
            "retained_video_digest": retained_digest,
            "camera_solution_digest": camera_solution_digest,
            "frame_registry_digest": canonical_json_digest(frame_registry),
        },
        "frame_registry": frame_registry,
        "frame_artifacts": frame_artifacts,
        "provider_profile": profile,
        "prompts": prompts_value,
        "allowed_evidence_uses": ["semantic_analysis"],
    }
    request_path = output / "semantic_sam31_source_track_run_request.v1.json"
    request_path.write_text(canonical_json(run_request) + "\n", encoding="utf-8")
    packet: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_upload_no_execution",
        "task_id": freeze["task_id"],
        "calibrated_view_receipt": {
            **_record(receipt_path),
            "receipt_digest": receipt["receipt_digest"],
        },
        "task_freeze": {
            **_record(freeze_path),
            "task_freeze_digest": freeze["task_freeze_digest"],
        },
        "provider_profile": {
            **_record(profile_path),
            "profile_digest": profile["profile_digest"],
        },
        "prompts": prompts_value,
        "camera_frame_map": camera_frame_map,
        "camera_count": len(source_rows),
        "retained_sequence": _record(retained_sequence, root=output),
        "lossless_frame_bytes_retained": True,
        "analysis_jpegs_are_model_derivatives": True,
        "run_request": {
            **_record(request_path, root=output),
            "request_digest": canonical_json_digest(run_request),
        },
        "encode_command": encode_command,
        "paid_execution_started": False,
        "provider_mutations_performed": 0,
        "claim_ceiling": "calibrated_render_derived_sam_input_only",
        "receipt_digest": "",
    }
    packet["receipt_digest"] = canonical_digest(packet, digest_field="receipt_digest")
    packet_path = output / f"{SCHEMA_VERSION}.json"
    packet_path.write_text(canonical_json(packet) + "\n", encoding="utf-8")
    return packet


def materialize_public_scene_sam31_task_inputs_from_tool_request(
    *, request: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Execute the registered Agents SDK tool request after digest validation."""

    if (
        request.get("schema_version") != "fresh_scene_sam31_task_input_tool_request.v1"
        or request.get("request_digest")
        != canonical_digest(dict(request), digest_field="request_digest")
    ):
        raise PublicSceneSam31InputError("sam31_task_input_tool_request_invalid")
    return materialize_public_scene_sam31_task_inputs(
        calibrated_view_receipt_path=str(request.get("calibrated_view_receipt_path") or ""),
        task_freeze_path=str(request.get("task_freeze_path") or ""),
        provider_profile_path=str(request.get("provider_profile_path") or ""),
        prompts_path=str(request.get("prompts_path") or ""),
        output_root=output_root,
        frame_rate_hz=int(request.get("frame_rate_hz") or 1),
        ffmpeg_executable=(
            str(request["ffmpeg_executable"])
            if request.get("ffmpeg_executable") is not None
            else None
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-view-receipt", required=True)
    parser.add_argument("--task-freeze", required=True)
    parser.add_argument("--provider-profile", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-rate-hz", type=int, default=1)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args(argv)
    materialize_public_scene_sam31_task_inputs(
        calibrated_view_receipt_path=args.calibrated_view_receipt,
        task_freeze_path=args.task_freeze,
        provider_profile_path=args.provider_profile,
        prompts_path=args.prompts,
        output_root=args.output_root,
        frame_rate_hz=args.frame_rate_hz,
        ffmpeg_executable=args.ffmpeg,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
