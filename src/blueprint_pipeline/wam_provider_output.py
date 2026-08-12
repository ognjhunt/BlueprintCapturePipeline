"""Provider-neutral inspection of asynchronous WAM output artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from .core.common import ensure_dir


VideoProbe = Callable[[Path], dict[str, Any]]

RUNTIME_RESULT_FILENAMES = (
    "isaac_runtime_result.json",
    "wam_runtime_result.json",
    "evaluator_runtime_result.json",
    "unitree_unifolm_policy_provider_output.json",
    "unitree_groot_n17_sonic_policy_provider_output.json",
    "unitree_groot_n17_sonic_wam_persistent_session_output.json",
    "adp_simpler_closed_loop_execution.json",
    "adp_content_agents_vast_result.json",
    "adp_joint_agent_result.json",
    "adp_aura_author_smoke_result.json",
    "adp_aura_interiorgs_result.json",
    "adp_inpaint360_interiorgs_result.json",
    "adp009d_native_microcheck.json",
    "adp009d_ovrtx_live_camera_result.json",
    "adp009d_aura_native_live_camera_result.json",
    "adp009d_retained_scene_gpu_render_result.v1.json",
    "adp009b_gaussian_excision_result.json",
    "native_task_arena_construction_result.v1.json",
    "native_task_arena_control_result.v1.json",
    "native_task_arena_policy_result.v1.json",
)
ENTRYPOINT_DIAGNOSTIC_FILENAME = "provider_entrypoint_diagnostic.json"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def probe_mp4_video(path: Path) -> dict[str, Any]:
    """Validate one extracted MP4 with ffprobe without making task claims."""

    blockers: list[str] = []
    if not path.is_file():
        return {
            "status": "blocked",
            "path": str(path),
            "blockers": ["mp4_file_missing"],
        }
    size = path.stat().st_size
    if size <= 0:
        blockers.append("mp4_file_empty")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        blockers.append("ffprobe_not_available")
        return {
            "status": "blocked",
            "path": str(path),
            "size_bytes": size,
            "ffprobe_available": False,
            "duration_seconds": None,
            "frame_count": None,
            "blockers": blockers,
        }
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=duration,nb_frames,nb_read_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "path": str(path),
            "size_bytes": size,
            "ffprobe_available": True,
            "duration_seconds": None,
            "frame_count": None,
            "blockers": [f"ffprobe_failed:{type(exc).__name__}"],
        }
    parse_error = None
    payload: dict[str, Any] = {}
    try:
        parsed = json.loads(completed.stdout or "{}")
        payload = dict(parsed) if isinstance(parsed, Mapping) else {}
    except Exception as exc:
        parse_error = f"{type(exc).__name__}:{str(exc)[:200]}"
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    stream = streams[0] if streams and isinstance(streams[0], Mapping) else {}
    fmt = _mapping(payload.get("format"))
    duration = _number(stream.get("duration")) or _number(fmt.get("duration"))
    frame_count = _number(stream.get("nb_read_frames")) or _number(
        stream.get("nb_frames")
    )
    if completed.returncode != 0:
        blockers.append(f"ffprobe_returncode:{completed.returncode}")
    if parse_error:
        blockers.append("ffprobe_json_parse_failed")
    if duration is None or duration <= 0:
        blockers.append("mp4_duration_not_positive")
    if frame_count is None or frame_count <= 0:
        blockers.append("mp4_frame_count_not_positive")
    return {
        "status": "completed" if not blockers else "blocked",
        "path": str(path),
        "size_bytes": size,
        "ffprobe_available": True,
        "ffprobe_returncode": completed.returncode,
        "duration_seconds": duration,
        "frame_count": int(frame_count) if frame_count is not None else None,
        "parse_error": parse_error,
        "stderr_preview": (completed.stderr or "")[:500],
        "blockers": blockers,
    }


def summarize_runtime_result(
    runtime_result: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep only provider-runtime evidence needed by orchestration surfaces."""

    if not runtime_result:
        return None
    checkpoint_detail = _mapping(runtime_result.get("checkpoint_detail"))
    cuda_probe = _mapping(runtime_result.get("cuda_probe"))
    subprocess_result = _mapping(runtime_result.get("subprocess"))
    generated_path = _string(runtime_result.get("generated_rollout_video_path"))
    task_success = runtime_result.get("task_success")
    summary = {
        "status": _string(runtime_result.get("status")) or None,
        "blockers": _string_list(runtime_result.get("blockers")),
        "action_conditioned_video_rollout_generated": bool(
            runtime_result.get("action_conditioned_video_rollout_generated")
        ),
        "generated_rollout_video_present": bool(generated_path),
        "generated_rollout_video_filename": (
            Path(generated_path).name if generated_path else None
        ),
        "repeated_policy_calls_count": runtime_result.get("repeated_policy_calls_count"),
        "generated_next_observation_count": runtime_result.get(
            "generated_next_observation_count"
        ),
        "live_wam_generation_success_count": runtime_result.get(
            "live_wam_generation_success_count"
        ),
        "learned_wam_model_success_count": runtime_result.get(
            "learned_wam_model_success_count"
        ),
        "policy_observes_wam_generated_next_observation": runtime_result.get(
            "policy_observes_wam_generated_next_observation"
        ),
        "provider_instance_reused_for_policy_and_wam_loop": runtime_result.get(
            "provider_instance_reused_for_policy_and_wam_loop"
        ),
        "checkpoint_status": _string(checkpoint_detail.get("status")) or None,
        "cuda_probe_status": _string(cuda_probe.get("status")) or None,
        "torch_cuda_available": (
            _mapping(cuda_probe.get("payload")).get("torch_cuda_available")
            if cuda_probe
            else None
        ),
        "cuda_device_count": (
            _mapping(cuda_probe.get("payload")).get("cuda_device_count")
            if cuda_probe
            else None
        ),
        "subprocess_status": _string(subprocess_result.get("status")) or None,
        "task_success": task_success if isinstance(task_success, bool) else None,
        "raw_secret_values_recorded": False,
    }
    if _string(runtime_result.get("claim_class")):
        summary.update(
            {
                "evaluator_result_count": runtime_result.get("result_count"),
                "evaluator_error_count": runtime_result.get("error_count"),
                "evaluator_model": _string(runtime_result.get("model")) or None,
                "claim_class": _string(runtime_result.get("claim_class")),
            }
        )
    return summary


def inspect_provider_runtime_output_zip(
    path: Path | None,
    *,
    video_extract_dir: Path | None = None,
    expected_video_count: int | None = None,
    video_probe: VideoProbe = probe_mp4_video,
) -> dict[str, Any]:
    """Inspect one provider output without promoting runtime success to task success."""

    if path is None:
        return {
            "status": "not_configured",
            "zip_path": None,
            "zip_present": False,
            "zip_size_bytes": 0,
            "video_smoke_proven": False,
        }
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {
            "status": "missing",
            "zip_path": str(resolved),
            "zip_present": False,
            "zip_size_bytes": 0,
            "video_smoke_proven": False,
        }
    names: list[str] = []
    runtime_result: dict[str, Any] | None = None
    entrypoint_diagnostic: dict[str, Any] | None = None
    json_parse_errors: list[str] = []
    mp4s: list[str] = []
    mp4_validation_rows: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(resolved) as archive:
            names = sorted(archive.namelist())
            mp4s = [name for name in names if name.lower().endswith(".mp4")]
            for candidate in names:
                if candidate.endswith(RUNTIME_RESULT_FILENAMES):
                    try:
                        parsed = json.loads(archive.read(candidate).decode("utf-8"))
                        if isinstance(parsed, Mapping):
                            runtime_result = dict(parsed)
                        break
                    except Exception as exc:
                        json_parse_errors.append(
                            f"{candidate}:{type(exc).__name__}"
                        )
            for candidate in names:
                if candidate.endswith(ENTRYPOINT_DIAGNOSTIC_FILENAME):
                    try:
                        parsed = json.loads(archive.read(candidate).decode("utf-8"))
                        if isinstance(parsed, Mapping):
                            entrypoint_diagnostic = dict(parsed)
                        break
                    except Exception as exc:
                        json_parse_errors.append(f"{candidate}:{type(exc).__name__}")
            if video_extract_dir and mp4s:
                ensure_dir(video_extract_dir)
                for index, member in enumerate(mp4s):
                    local_path = video_extract_dir / f"{index:03d}_{Path(member).name}"
                    local_path.write_bytes(archive.read(member))
                    row = video_probe(local_path)
                    row["zip_member"] = member
                    mp4_validation_rows.append(row)
    except Exception as exc:
        return {
            "status": "blocked",
            "zip_path": str(resolved),
            "zip_present": True,
            "zip_size_bytes": resolved.stat().st_size,
            "blockers": [f"provider_runtime_output_zip_invalid:{type(exc).__name__}"],
            "video_smoke_proven": False,
        }
    expected_count = expected_video_count if expected_video_count is not None else 0
    mp4_count_matches_expected = expected_count > 0 and len(mp4s) >= expected_count
    all_validated = bool(mp4_validation_rows) and all(
        row.get("status") == "completed" for row in mp4_validation_rows
    )
    validation_blockers: list[str] = []
    if expected_count and len(mp4s) < expected_count:
        validation_blockers.append("mp4_count_below_expected_video_smoke_camera_count")
    if mp4s and video_extract_dir and not all_validated:
        validation_blockers.append("ffprobe_validation_failed_for_one_or_more_mp4s")
    if mp4s and not video_extract_dir:
        validation_blockers.append("mp4_ffprobe_validation_not_requested")
    video_smoke_proven = mp4_count_matches_expected and all_validated
    runtime_result_summary = summarize_runtime_result(runtime_result)
    return {
        "status": "completed",
        "zip_path": str(resolved),
        "zip_present": True,
        "zip_size_bytes": resolved.stat().st_size,
        "zip_member_count": len(names),
        "zip_members_preview": names[:50],
        "runtime_result_present": runtime_result is not None,
        "runtime_result": runtime_result_summary,
        "runtime_result_status": (
            runtime_result_summary.get("status") if runtime_result_summary else None
        ),
        "runtime_result_blockers": (
            _string_list(runtime_result_summary.get("blockers"))
            if runtime_result_summary
            else []
        ),
        "entrypoint_diagnostic_present": entrypoint_diagnostic is not None,
        "entrypoint_diagnostic": entrypoint_diagnostic,
        "mp4_count": len(mp4s),
        "mp4_members": mp4s[:25],
        "video_smoke_expected_video_count": expected_count or None,
        "video_smoke_proven": video_smoke_proven,
        "mp4_validation": {
            "status": (
                "completed"
                if video_smoke_proven
                else ("not_applicable_no_mp4_members" if not mp4s else "blocked")
            ),
            "expected_video_count": expected_count or None,
            "mp4_count": len(mp4s),
            "validated_mp4_count": sum(
                1
                for row in mp4_validation_rows
                if row.get("status") == "completed"
            ),
            "blockers": validation_blockers,
            "files": mp4_validation_rows,
        },
        "json_parse_errors": json_parse_errors,
    }
