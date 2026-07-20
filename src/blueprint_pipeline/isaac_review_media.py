"""Review-media probing and bounded MP4 repair for Isaac provider output."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from .g1_kitchen_semantic_review import run_full_episode_semantic_review


CAMERA_MOTION_MODELS = {
    "overview": "task_framed_third_person_review",
    "robot_pov": "rigid_head_local_transform",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_index(path: Path) -> int | None:
    suffix = path.stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _step_binding_blockers(
    step_bindings: Sequence[Mapping[str, Any]] | None, expected_frame_count: int
) -> list[str]:
    """Validate the attempt-bound per-step horizon before any frame is trusted."""
    if step_bindings is None:
        return ["episode_step_bindings_missing"]
    bindings = [dict(item) for item in step_bindings]
    blockers: list[str] = []
    indices = [int(item.get("step_index") or 0) for item in bindings]
    if len(bindings) != int(expected_frame_count) or indices != list(
        range(int(expected_frame_count))
    ):
        blockers.append(
            "episode_step_bindings_count_mismatch:"
            f"{len(bindings)}!={int(expected_frame_count)}"
        )
    previous_after: int | None = None
    for item in bindings:
        for field in ("source_action_sha256", "stage_id", "simulator_session_id"):
            if not str(item.get(field) or "").strip():
                blockers.append(
                    f"episode_step_binding_incomplete:{item.get('step_index')}:{field}"
                )
        try:
            before = int(str(item.get("before_timestamp")))
            after = int(str(item.get("after_timestamp")))
        except (TypeError, ValueError):
            blockers.append(
                "episode_step_bindings_timestamps_not_ordered:"
                f"{item.get('step_index')}:invalid"
            )
            continue
        if after <= before or (previous_after is not None and before < previous_after):
            blockers.append(
                "episode_step_bindings_timestamps_not_ordered:"
                f"{item.get('step_index')}"
            )
        previous_after = after
    return blockers


def admit_full_ordered_episode(
    *,
    camera_frames: dict[str, Sequence[str | Path]],
    frame_semantics: dict[str, dict[str, Any]],
    semantic_review: dict[str, Any] | None,
    expected_frame_count: int,
    step_bindings: Sequence[Mapping[str, Any]] | None,
    frame_step_bindings: Mapping[str, Mapping[str, Any]] | None,
    min_robot_occupancy: float = 0.03,
    min_target_occupancy: float = 0.01,
) -> dict[str, Any]:
    """Fail-closed admission over every ordered overview and robot-POV frame.

    ``expected_frame_count`` must come from the immutable attempt/task/executor
    manifest — never from the frames that happened to arrive. ``step_bindings``
    carries the attested per-step action SHA / stage / session / timestamp rows,
    and ``frame_step_bindings`` is the renderer-emitted frame-to-step sidecar;
    both are required, so equally truncated or foreign camera streams block.
    """
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
    blockers.extend(_step_binding_blockers(step_bindings, expected_frame_count))
    bindings_by_name = (
        {str(name): dict(value) for name, value in frame_step_bindings.items()}
        if frame_step_bindings is not None
        else None
    )
    if bindings_by_name is None:
        blockers.append("episode_frame_step_bindings_missing")
    steps_by_index = {
        int(item.get("step_index")): dict(item)
        for item in (step_bindings or [])
        if isinstance(item.get("step_index"), int)
        and not isinstance(item.get("step_index"), bool)
    }
    for role in required_roles:
        paths = [Path(item) for item in camera_frames.get(role, ())]
        if len(paths) != int(expected_frame_count):
            blockers.append(
                f"{role}:ordered_frame_count_mismatch:{len(paths)}!={int(expected_frame_count)}"
            )
        observed_indices = [_frame_index(path) for path in paths]
        if paths and observed_indices != list(range(int(expected_frame_count))):
            blockers.append(
                f"{role}:frame_indices_not_contiguous:"
                + ",".join(str(item) for item in observed_indices)
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
            if bindings_by_name is not None:
                frame_binding = bindings_by_name.get(path.name)
                if frame_binding is None:
                    row_blockers.append("frame_step_binding_missing")
                else:
                    if str(frame_binding.get("sha256") or "") != checksum:
                        row_blockers.append("frame_step_binding_sha256_mismatch")
                    if str(frame_binding.get("camera_role") or "") != role:
                        row_blockers.append("frame_step_binding_role_mismatch")
                    if (
                        str(frame_binding.get("camera_motion_model") or "")
                        != CAMERA_MOTION_MODELS[role]
                    ):
                        row_blockers.append("frame_step_binding_camera_motion_mismatch")
                    if int(frame_binding.get("step_index", -1)) != index:
                        row_blockers.append("frame_step_binding_index_mismatch")
                    expected_step = steps_by_index.get(index, {})
                    for field in (
                        "source_action_sha256",
                        "stage_id",
                        "simulator_session_id",
                        "before_timestamp",
                        "after_timestamp",
                    ):
                        if (
                            not str(expected_step.get(field) or "")
                            or str(frame_binding.get(field) or "")
                            != str(expected_step.get(field) or "")
                        ):
                            row_blockers.append(f"frame_step_binding_{field}_mismatch")
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
        seen: dict[str, int] = {}
        for index, checksum in enumerate(checksums):
            if checksum in seen:
                blockers.append(
                    f"{role}:frame_duplicate_sha256_across_steps:{seen[checksum]},{index}"
                )
            else:
                seen[checksum] = index
    review = dict(semantic_review or {})
    if semantic_review is not None and int(
        review.get("frame_review_count") or -1
    ) != 2 * int(expected_frame_count):
        blockers.append(
            "semantic_review_coverage_count_mismatch:"
            f"{review.get('frame_review_count')}!={2 * int(expected_frame_count)}"
        )
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


FRAME_STEP_BINDINGS_SCHEMA_VERSION = "isaac_review_frame_step_bindings.v1"


def record_frame_step_bindings(
    *, frames_dir: str | Path, artifacts: Sequence[Mapping[str, Any]]
) -> Path:
    """Persist the renderer's frame-to-step binding sidecar, merging per step."""
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    path = Path(frames_dir) / "frame_step_bindings.json"
    frames: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("frames"), dict):
                frames = dict(existing["frames"])
        except (OSError, json.JSONDecodeError):
            frames = {}
    for artifact in artifacts:
        detail = dict(artifact)
        name = Path(str(detail.get("path") or "")).name
        if not name:
            continue
        frames[name] = {
            "camera_role": detail.get("camera_role"),
            "step_index": int(detail.get("frame_index") or 0),
            "sha256": detail.get("sha256"),
            **{
                field: detail.get(field)
                for field in (
                    "source_action_sha256",
                    "simulator_session_id",
                    "stage_id",
                    "before_timestamp",
                    "after_timestamp",
                    "attempt_id",
                    "launch_nonce",
                    "allocation_launch_session_id",
                    "qualification_attempt_bound",
                    "qualification_attempt_sequence",
                    "qualification_attempt_nonce_sha256",
                    "control_frame_global_index",
                    "physics_step_count_before",
                    "physics_step_count_after",
                    "physics_step_delta",
                    "simulation_time_before_seconds",
                    "simulation_time_after_seconds",
                    "simulation_time_delta_seconds",
                    "outer_source_step_index",
                    "horizon_frame_index",
                    "controller_frame_index",
                    "source_action_frame_sha256",
                    "task_joint_value_rad",
                    "registered_transition_passed",
                    "semantic_terminal_frame",
                    "captured_at_ns",
                    "camera_motion_model",
                )
            },
        }
    path.write_text(
        json.dumps(
            {"schema_version": FRAME_STEP_BINDINGS_SCHEMA_VERSION, "frames": frames},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _load_frame_step_bindings(frames_dir: Path) -> dict[str, dict[str, Any]] | None:
    path = frames_dir / "frame_step_bindings.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != FRAME_STEP_BINDINGS_SCHEMA_VERSION
        or not isinstance(payload.get("frames"), dict)
    ):
        return None
    return {str(name): dict(value) for name, value in payload["frames"].items()}


def admit_collected_scenario_episode(
    *,
    scenario_dir: str | Path,
    expected_frame_count: int | None,
    step_bindings: Sequence[Mapping[str, Any]] | None,
    attestation_pins: Mapping[str, Any] | None = None,
    identity_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load worker-emitted semantic sidecars and persist collected media admission."""
    root = Path(scenario_dir)
    frames = root / "frames"
    overview_paths = sorted(frames.glob("overview_[0-9][0-9][0-9][0-9].png"))
    robot_pov_paths = sorted(frames.glob("robot_pov_[0-9][0-9][0-9][0-9].png"))
    effective_expected = int(expected_frame_count or 0)
    if effective_expected <= 0 and len(overview_paths) == len(robot_pov_paths):
        # Dynamic episodes bind validation to the terminal trace's observed
        # complete camera horizon instead of a predeclared frame count.
        effective_expected = len(overview_paths)
    semantics_path = root / "full_episode_frame_semantics.json"
    review_path = root / "full_episode_semantic_review.json"
    if (
        attestation_pins is not None
        or not semantics_path.is_file()
        or not review_path.is_file()
    ):
        run_full_episode_semantic_review(
            scenario_dir=root,
            expected_frame_count=effective_expected,
            attestation_pins=attestation_pins,
            identity_binding=identity_binding,
            step_bindings=step_bindings,
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
            "overview": overview_paths,
            "robot_pov": robot_pov_paths,
        },
        frame_semantics=semantics,
        semantic_review=review,
        expected_frame_count=effective_expected,
        step_bindings=step_bindings,
        frame_step_bindings=_load_frame_step_bindings(frames),
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
