"""Review-media probing and bounded MP4 repair for Isaac provider output."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .g1_kitchen_semantic_review import run_full_episode_semantic_review


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def admit_full_ordered_episode(
    *,
    camera_frames: dict[str, Sequence[str | Path]],
    frame_semantics: dict[str, dict[str, Any]],
    semantic_review: dict[str, Any] | None,
    expected_frame_count: int,
    min_robot_occupancy: float = 0.03,
    min_target_occupancy: float = 0.01,
) -> dict[str, Any]:
    """Fail-closed admission over every ordered overview and robot-POV frame."""
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return {
            "schema_version": "isaac_full_ordered_episode_media_admission.v1",
            "status": "blocked",
            "blockers": ["review_media_numpy_or_pillow_missing"],
        }
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    required_roles = ("overview", "robot_pov")
    role_checksums: dict[str, list[str]] = {}
    for role in required_roles:
        paths = [Path(item) for item in camera_frames.get(role, ())]
        if len(paths) != int(expected_frame_count):
            blockers.append(
                f"{role}:ordered_frame_count_mismatch:{len(paths)}!={int(expected_frame_count)}"
            )
        checksums: list[str] = []
        previous_rgb = None
        for index, path in enumerate(paths):
            row_blockers: list[str] = []
            if not path.is_file():
                row_blockers.append("frame_missing")
                rows.append(
                    {
                        "camera_role": role,
                        "frame_index": index,
                        "path": str(path),
                        "status": "blocked",
                        "blockers": row_blockers,
                    }
                )
                blockers.append(f"{role}:{index}:frame_missing")
                continue
            try:
                rgb = np.asarray(Image.open(path).convert("RGB"), dtype="float32")
            except Exception:
                row_blockers.append("frame_decode_failed")
                rgb = None
            checksum = _file_sha256(path)
            checksums.append(checksum)
            dark = bright = flat = clipped = temporal_delta = None
            if rgb is not None:
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    row_blockers.append("frame_wrong_shape")
                elif not np.isfinite(rgb).all():
                    row_blockers.append("frame_nonfinite")
                else:
                    luma = rgb.mean(axis=2)
                    dark = float(np.mean(luma <= 4.0))
                    bright = float(np.mean(luma >= 251.0))
                    flat = float(np.std(luma))
                    clipped = float(np.mean((rgb <= 1.0) | (rgb >= 254.0)))
                    if dark >= 0.995:
                        row_blockers.append("frame_blank_black")
                    if bright >= 0.995:
                        row_blockers.append("frame_blank_white")
                    if flat < 1.0:
                        row_blockers.append("frame_flat")
                    if clipped >= 0.98:
                        row_blockers.append("frame_excessively_clipped")
                    if previous_rgb is not None and previous_rgb.shape == rgb.shape:
                        temporal_delta = float(np.mean(np.abs(rgb - previous_rgb)))
                        if temporal_delta <= 1e-6:
                            row_blockers.append("frame_stale_checksum_or_pixels")
                    previous_rgb = rgb
            semantics = dict(frame_semantics.get(str(path)) or {})
            if semantics.get("sha256") != checksum:
                row_blockers.append("semantic_review_frame_sha256_mismatch")
            robot_occupancy = semantics.get("robot_pixel_occupancy")
            target_occupancy = semantics.get("target_pixel_occupancy")
            if role == "overview":
                for field in (
                    "g1_visible",
                    "target_visible",
                    "floor_support_visible",
                    "orientation_visible",
                    "clearance_visible",
                ):
                    if semantics.get(field) is not True:
                        row_blockers.append(f"overview_{field}_not_proven")
                if (
                    not isinstance(robot_occupancy, (int, float))
                    or float(robot_occupancy) < min_robot_occupancy
                ):
                    row_blockers.append("overview_robot_pixel_occupancy_too_low")
                if (
                    not isinstance(target_occupancy, (int, float))
                    or float(target_occupancy) < min_target_occupancy
                ):
                    row_blockers.append("overview_target_pixel_occupancy_too_low")
            else:
                if semantics.get("target_visible") is not True:
                    row_blockers.append("robot_pov_target_not_visible")
                if semantics.get("active_hand_wrist_chain_visible") is not True:
                    row_blockers.append("robot_pov_active_hand_wrist_chain_not_visible")
            rows.append(
                {
                    "camera_role": role,
                    "frame_index": index,
                    "path": str(path),
                    "sha256": checksum,
                    "status": "passed" if not row_blockers else "blocked",
                    "blockers": sorted(set(row_blockers)),
                    "robot_pixel_occupancy": robot_occupancy,
                    "target_pixel_occupancy": target_occupancy,
                    "temporal_delta_mean_abs": temporal_delta,
                    "dark_fraction": dark,
                    "bright_fraction": bright,
                    "luma_std": flat,
                    "clipped_channel_fraction": clipped,
                }
            )
            blockers.extend(f"{role}:{index}:{item}" for item in row_blockers)
        role_checksums[role] = checksums
    review = dict(semantic_review or {})
    if (
        review.get("status") != "passed"
        or review.get("full_ordered_episode_reviewed") is not True
    ):
        blockers.append("full_ordered_episode_semantic_review_missing_or_blocked")
    if review.get("abstained") is True:
        blockers.append("full_ordered_episode_semantic_review_abstained")
    if not str(review.get("review_runtime_id") or ""):
        blockers.append("full_ordered_episode_semantic_review_runtime_id_missing")
    if review.get("review_source") != "external_semantic_review_api":
        blockers.append("full_ordered_episode_semantic_review_source_invalid")
    for field in ("request_sha256", "response_sha256"):
        digest = str(review.get(field) or "").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            blockers.append(f"full_ordered_episode_semantic_review_{field}_invalid")
    return {
        "schema_version": "isaac_full_ordered_episode_media_admission.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "expected_frame_count_per_required_camera": int(expected_frame_count),
        "required_camera_roles": list(required_roles),
        "frame_rows": rows,
        "ordered_camera_frame_sha256s": role_checksums,
        "semantic_review": review or None,
        "schematic_topdown_debugging_only": True,
        "full_ordered_episode_admitted": not blockers,
    }


def admit_collected_scenario_episode(
    *, scenario_dir: str | Path, expected_frame_count: int
) -> dict[str, Any]:
    """Load worker-emitted semantic sidecars and persist collected media admission."""
    root = Path(scenario_dir)
    frames = root / "frames"
    semantics_path = root / "full_episode_frame_semantics.json"
    review_path = root / "full_episode_semantic_review.json"
    if not semantics_path.is_file() or not review_path.is_file():
        run_full_episode_semantic_review(
            scenario_dir=root,
            expected_frame_count=int(expected_frame_count),
        )
    try:
        raw_semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
        semantics = dict(raw_semantics.get("frames") or raw_semantics)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        semantics = {}
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if not isinstance(review, dict):
            review = None
    except (OSError, json.JSONDecodeError):
        review = None
    result = admit_full_ordered_episode(
        camera_frames={
            "overview": sorted(frames.glob("overview_[0-9][0-9][0-9][0-9].png")),
            "robot_pov": sorted(frames.glob("robot_pov_[0-9][0-9][0-9][0-9].png")),
        },
        frame_semantics=semantics,
        semantic_review=review,
        expected_frame_count=int(expected_frame_count),
    )
    output = root / "full_ordered_episode_media_admission.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "path": str(output)}

def _int_or_none(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _probe_video_file(path: str | Path) -> dict:
    """Best-effort local media metadata for collected provider videos."""
    video_path = Path(path)
    if not video_path.is_file():
        return {
            "status": "missing",
            "path": str(video_path),
            "blockers": ["video_file_missing"],
        }
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {
            "status": "unavailable",
            "path": str(video_path),
            "tool": "ffprobe",
            "blockers": ["ffprobe_not_found"],
        }
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "status": "failed",
            "path": str(video_path),
            "tool": ffprobe,
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-400:],
            "blockers": ["ffprobe_failed"],
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "path": str(video_path),
            "tool": ffprobe,
            "blockers": ["ffprobe_output_not_json"],
        }
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_stream = next(
        (
            item
            for item in streams
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ),
        {},
    )
    format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    fps = None
    rate = str(video_stream.get("r_frame_rate") or "").strip()
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = _float_or_none(den)
        if den_f:
            fps = (_float_or_none(num) or 0.0) / den_f
    else:
        fps = _float_or_none(rate)
    return {
        "status": "ready",
        "path": str(video_path),
        "tool": ffprobe,
        "width": _int_or_none(video_stream.get("width")),
        "height": _int_or_none(video_stream.get("height")),
        "frame_count": _int_or_none(video_stream.get("nb_frames")),
        "fps": fps,
        "duration_seconds": (
            _float_or_none(video_stream.get("duration"))
            or _float_or_none(format_info.get("duration"))
        ),
        "codec_name": video_stream.get("codec_name") or None,
    }


REVIEW_MP4_FRAME_PATTERNS = {
    "overview": "overview_[0-9][0-9][0-9][0-9].png",
    "robot_pov": "robot_pov_[0-9][0-9][0-9][0-9].png",
    "placement_topdown": "placement_topdown_[0-9][0-9][0-9][0-9].png",
}

REVIEW_MP4_FRAME_SEQUENCES = {
    "overview": "overview_%04d.png",
    "robot_pov": "robot_pov_%04d.png",
    "placement_topdown": "placement_topdown_%04d.png",
}


def _ffmpeg_mp4_command(
    *,
    ffmpeg: str,
    frames_sequence: str,
    fps: int,
    out_path: str,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-framerate",
        str(int(fps)),
        "-start_number",
        "0",
        "-i",
        frames_sequence,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_path,
    ]


def _repair_collected_review_mp4s(
    *,
    render_out_dir: Path,
    result: dict,
    fps: int,
    optional_videos: Sequence[str] = (),
    expected_frame_count: int | None = None,
) -> dict:
    """Assemble missing collected MP4s locally when the provider image lacked ffmpeg.

    When ``expected_frame_count`` is known, a repair over fewer frames is labeled
    ``repaired_truncated`` with a blocker instead of ``repaired`` — a locally
    assembled MP4 must never make a partially-uploaded provider render read as a
    complete one. The truncated video is still written for human review.
    """
    scenarios = result.get("scenarios", []) if isinstance(result, dict) else []
    ffmpeg = shutil.which("ffmpeg")
    repairs: list[dict] = []
    optional_video_set = {str(name) for name in optional_videos}
    expected = int(expected_frame_count) if expected_frame_count else None
    for sc in scenarios:
        sid = sc.get("scenario_id")
        sdir = render_out_dir / str(sid)
        frames_dir = sdir / "frames"
        for name, pattern in REVIEW_MP4_FRAME_PATTERNS.items():
            out_path = sdir / f"{name}.mp4"
            frame_paths = sorted(frames_dir.glob(pattern))
            rec: dict = {
                "scenario_id": sid,
                "video": name,
                "path": str(out_path),
                "frame_pattern": pattern,
                "frame_count": len(frame_paths),
                "expected_frame_count": expected,
            }
            truncated = expected is not None and 0 < len(frame_paths) < expected
            if out_path.is_file():
                rec["status"] = "already_present"
            elif not frame_paths:
                if name in optional_video_set:
                    rec["status"] = "skipped_optional"
                    rec["optional"] = True
                else:
                    rec["status"] = "missing_frames"
                    rec["blockers"] = ["video_frames_missing"]
            elif not ffmpeg:
                rec["status"] = "unavailable"
                rec["blockers"] = ["local_ffmpeg_not_found"]
            else:
                cmd = _ffmpeg_mp4_command(
                    ffmpeg=ffmpeg,
                    frames_sequence=str(frames_dir / REVIEW_MP4_FRAME_SEQUENCES[name]),
                    fps=int(fps),
                    out_path=str(out_path),
                )
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                rec["tool"] = ffmpeg
                rec["exit_code"] = proc.returncode
                if proc.returncode == 0 and out_path.is_file():
                    if truncated:
                        rec["status"] = "repaired_truncated"
                        rec["blockers"] = [
                            "mp4_repair_truncated_frames:"
                            f"{name}:{len(frame_paths)}<{expected}"
                        ]
                    else:
                        rec["status"] = "repaired"
                else:
                    rec["status"] = "failed"
                    rec["blockers"] = ["local_ffmpeg_mp4_repair_failed"]
                    rec["stderr_tail"] = proc.stderr[-500:]
            repairs.append(rec)
    blockers = sorted({
        str(blocker)
        for rec in repairs
        for blocker in (rec.get("blockers") or [])
    })
    return {
        "schema_version": "isaac_collected_review_mp4_repair.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "ffmpeg": ffmpeg,
        "fps": int(fps),
        "repairs": repairs,
        "claim_boundary": (
            "Local MP4 repair assembles already-collected provider PNG frames. It does not alter "
            "rendered frames, task outcomes, or evaluator success labels."
        ),
    }
