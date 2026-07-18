"""Build distinct OSCAR-prediction and same-session Isaac episode reviews.

The final execution review is intentionally fail closed: it contains only
ordered overview and robot-POV frames rendered by the persistent Isaac task
session and bound to exact action measurements. OSCAR/WAM clips remain useful
model-derived support media, but are assembled and labeled separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "groot_oscar_episode_review_validation.v1"
WAM_PREDICTION_SCHEMA_VERSION = "groot_oscar_wam_prediction_review_validation.v1"
TRACE_NAME = "oscar_isaac_closed_loop_trace.jsonl"
OUTPUT_NAME = "final_review.mp4"
VALIDATION_NAME = "final_review_validation.json"
WAM_PREDICTION_OUTPUT_NAME = "wam_prediction_review.mp4"
WAM_PREDICTION_VALIDATION_NAME = "wam_prediction_review_validation.json"
ISAAC_STATE_DIR_NAME = "isaac_task_state"
ISAAC_FRAME_BINDINGS_SCHEMA_VERSION = "isaac_review_frame_step_bindings.v1"
ISAAC_INITIAL_FRAME_BINDINGS_SCHEMA_VERSION = "isaac_initial_review_frame_bindings.v1"
ISAAC_INITIAL_FRAME_BINDINGS_NAME = "initial_frame_bindings.json"
ISAAC_CAMERA_ROLES = ("overview", "robot_pov")
ISAAC_CAMERA_MOTION_MODELS = {
    "overview": "task_framed_third_person_review",
    "robot_pov": "rigid_head_local_transform",
}
ISAAC_ROLE_OUTPUT_NAMES = {
    "overview": "isaac_overview_review.mp4",
    "robot_pov": "isaac_robot_pov_review.mp4",
}
ISAAC_CONTROLLER_HZ = 50.0
ISAAC_REVIEW_FPS = 10
ISAAC_REVIEW_CONTROLLER_FRAME_STRIDE = 5
ISAAC_ACTION_FRAME_START_INDEX = 1
WAM_PREDICTION_REVIEW_FPS = 15
MINIMUM_WIDTH = 640
MINIMUM_HEIGHT = 480


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validation(
    episode_dir: Path,
    *,
    blockers: Sequence[str],
    output_name: str = WAM_PREDICTION_OUTPUT_NAME,
    validation_name: str = WAM_PREDICTION_VALIDATION_NAME,
    schema_version: str = WAM_PREDICTION_SCHEMA_VERSION,
    review_source: str = "oscar_wam_predicted_rollout_clips",
    trace_step_count: int = 0,
    ordered_step_indices: Sequence[int] = (),
    ordered_clip_count: int = 0,
    concat_mode: str | None = None,
    codec: str | None = None,
    width: int = 0,
    height: int = 0,
    frame_count: int = 0,
    duration_seconds: float = 0.0,
    executed_prefix_duration_seconds_by_step: Sequence[float] = (),
) -> dict[str, Any]:
    output = episode_dir / output_name
    unique_blockers = sorted({str(value) for value in blockers if str(value)})
    passed = not unique_blockers
    result = {
        "schema_version": schema_version,
        "status": "passed" if passed else "blocked",
        "blockers": unique_blockers,
        "path": str(output) if passed and output.is_file() else None,
        "sha256": _sha256(output) if passed and output.is_file() else None,
        "codec": codec,
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "duration_seconds": float(duration_seconds),
        "trace_step_count": int(trace_step_count),
        "ordered_clip_count": int(ordered_clip_count),
        "ordered_step_indices": [int(value) for value in ordered_step_indices],
        "episode_order_verified": bool(
            passed
            and trace_step_count > 0
            and ordered_clip_count == trace_step_count
            and list(ordered_step_indices) == list(range(1, trace_step_count + 1))
        ),
        "concat_mode": concat_mode,
        "video_frame_count_mode": "dynamic_from_executed_controller_duration",
        "prediction_review_timeline_mode": "executed_control_prefix_per_decision",
        "executed_prefix_duration_seconds_by_step": [
            float(value) for value in executed_prefix_duration_seconds_by_step
        ],
        "expected_executed_timeline_duration_seconds": float(
            sum(executed_prefix_duration_seconds_by_step)
        ),
        "full_prediction_horizons_preserved_in_source_clips": True,
        "overlapping_unexecuted_prediction_tails_excluded": True,
        "minimum_resolution": {"width": MINIMUM_WIDTH, "height": MINIMUM_HEIGHT},
        "review_source": review_source,
        "execution_truth": False,
        "same_session_isaac_frames": False,
        "claim_boundary": {
            "contains_oscar_wam_model_predictions": True,
            "is_not_same_session_isaac_execution_media": True,
            "is_not_task_success_proof": True,
        },
    }
    _write_json(episode_dir / validation_name, result)
    return result


def _ordered_clips(episode_dir: Path) -> tuple[list[Path], list[int], int, list[str]]:
    trace_path = episode_dir / TRACE_NAME
    if not trace_path.is_file():
        return [], [], 0, ["closed_loop_trace_missing"]
    lines = [line for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return [], [], 0, ["closed_loop_trace_empty"]

    blockers: list[str] = []
    clips: list[Path] = []
    indices: list[int] = []
    seen_paths: set[Path] = set()
    for position, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            blockers.append(f"closed_loop_trace_json_invalid:{position}")
            continue
        if not isinstance(row, Mapping):
            blockers.append(f"closed_loop_trace_row_not_object:{position}")
            continue
        try:
            step_index = int(row.get("step_index"))
        except (TypeError, ValueError):
            blockers.append(f"closed_loop_trace_step_index_invalid:{position}")
            continue
        indices.append(step_index)
        raw_path = str(row.get("wam_generated_video") or "").strip()
        if not raw_path:
            blockers.append(f"closed_loop_step_video_path_missing:{position}")
            continue
        clip = Path(raw_path).expanduser()
        if not clip.is_absolute():
            clip = episode_dir / clip
        clip = clip.resolve()
        if clip in seen_paths:
            blockers.append(f"closed_loop_step_video_path_duplicate:{position}")
            continue
        seen_paths.add(clip)
        if not clip.is_file() or clip.stat().st_size <= 0:
            blockers.append(f"closed_loop_step_video_file_missing_or_empty:{position}")
            continue
        clips.append(clip)

    expected = list(range(1, len(lines) + 1))
    if indices != expected:
        blockers.append("closed_loop_episode_step_order_not_contiguous")
    if len(clips) != len(lines):
        blockers.append("closed_loop_episode_clip_count_does_not_match_trace")
    return clips, indices, len(lines), blockers


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _executed_prefix_durations(
    episode_dir: Path,
    step_indices: Sequence[int],
) -> tuple[list[float], list[str]]:
    state_dir = episode_dir.parent / ISAAC_STATE_DIR_NAME
    durations: list[float] = []
    blockers: list[str] = []
    for step_index in step_indices:
        measurement_path = state_dir / f"task_measurement_{step_index:04d}.json"
        measurement = _read_json_mapping(measurement_path)
        if measurement is None:
            blockers.append(
                f"wam_review_execution_measurement_missing_or_invalid:{step_index}"
            )
            continue

        duration: float | None = None
        contract = measurement.get("controller_execution_contract")
        if isinstance(contract, Mapping):
            try:
                declared_duration = float(
                    contract.get("declared_execution_duration_seconds")
                )
            except (TypeError, ValueError):
                declared_duration = 0.0
            if math.isfinite(declared_duration) and declared_duration > 0:
                duration = declared_duration

        if duration is None:
            controller_measurements = measurement.get("controller_frame_measurements")
            if isinstance(controller_measurements, list) and controller_measurements:
                frame_deltas: list[float] = []
                for row in controller_measurements:
                    if not isinstance(row, Mapping):
                        frame_deltas = []
                        break
                    try:
                        delta = float(row.get("simulation_time_delta_seconds"))
                    except (TypeError, ValueError):
                        frame_deltas = []
                        break
                    if not math.isfinite(delta) or delta <= 0:
                        frame_deltas = []
                        break
                    frame_deltas.append(delta)
                if frame_deltas:
                    duration = sum(frame_deltas)

        if duration is None:
            blockers.append(
                f"wam_review_executed_prefix_duration_missing_or_invalid:{step_index}"
            )
            continue
        durations.append(duration)
    return durations, blockers


def _concat_executed_prefixes(
    clips: Sequence[Path],
    durations_seconds: Sequence[float],
    output: Path,
) -> bool:
    if len(clips) != len(durations_seconds) or not clips:
        return False
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        command.extend(["-i", str(clip)])
    filters = [
        (
            f"[{index}:v]trim=duration={durations_seconds[index]:.9f},"
            f"setpts=PTS-STARTPTS,scale={MINIMUM_WIDTH}:{MINIMUM_HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={MINIMUM_WIDTH}:{MINIMUM_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"format=yuv420p,setsar=1,settb=AVTB[v{index}]"
        )
        for index in range(len(clips))
    ]
    filters.append(
        "".join(f"[v{index}]" for index in range(len(clips)))
        + f"concat=n={len(clips)}:v=1:a=0[vjoined]"
    )
    filters.append(f"[vjoined]fps={WAM_PREDICTION_REVIEW_FPS}[vout]")
    command.extend(
        [
            "-filter_complex", ";".join(filters), "-map", "[vout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", str(output),
        ]
    )
    completed = _run(command)
    return completed.returncode == 0 and output.is_file() and output.stat().st_size > 0


def _probe(output: Path) -> tuple[str | None, int, int, int, float, list[str]]:
    completed = _run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames,nb_frames,duration",
            "-show_entries", "format=duration", "-of", "json", str(output),
        ]
    )
    blockers: list[str] = []
    codec: str | None = None
    width = height = frame_count = 0
    duration = 0.0
    if completed.returncode != 0:
        blockers.append(f"ffprobe_failed:{completed.returncode}")
    try:
        metadata = json.loads(completed.stdout)
        stream = metadata["streams"][0]
        codec = str(stream.get("codec_name") or "") or None
        width = int(stream["width"])
        height = int(stream["height"])
        frames_value = stream.get("nb_read_frames") or stream.get("nb_frames") or 0
        frame_count = int(frames_value)
        duration = float(stream.get("duration") or metadata.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError, KeyError, IndexError, json.JSONDecodeError):
        blockers.append("ffprobe_metadata_invalid")
    if width < MINIMUM_WIDTH or height < MINIMUM_HEIGHT:
        blockers.append(f"review_resolution_below_640x480:{width}x{height}")
    if frame_count < 1 or duration <= 0:
        blockers.append("review_video_empty_or_zero_duration")
    return codec, width, height, frame_count, duration, blockers


def build_wam_prediction_review(episode_dir: str | Path) -> dict[str, Any]:
    """Assemble OSCAR/WAM model-predicted clips as non-execution review media."""

    resolved_dir = Path(episode_dir).expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    output = resolved_dir / WAM_PREDICTION_OUTPUT_NAME
    output.unlink(missing_ok=True)
    clips, indices, trace_count, blockers = _ordered_clips(resolved_dir)
    if blockers:
        return _validation(
            resolved_dir,
            blockers=blockers,
            trace_step_count=trace_count,
            ordered_step_indices=indices,
            ordered_clip_count=len(clips),
        )

    executed_durations, duration_blockers = _executed_prefix_durations(
        resolved_dir, indices
    )
    if duration_blockers:
        return _validation(
            resolved_dir,
            blockers=duration_blockers,
            trace_step_count=trace_count,
            ordered_step_indices=indices,
            ordered_clip_count=len(clips),
            executed_prefix_duration_seconds_by_step=executed_durations,
        )

    if not _concat_executed_prefixes(clips, executed_durations, output):
        output.unlink(missing_ok=True)
        return _validation(
            resolved_dir,
            blockers=["ffmpeg_executed_prefix_concat_failed"],
            trace_step_count=trace_count,
            ordered_step_indices=indices,
            ordered_clip_count=len(clips),
            executed_prefix_duration_seconds_by_step=executed_durations,
        )

    codec, width, height, frame_count, duration, probe_blockers = _probe(output)
    expected_duration = sum(executed_durations)
    duration_tolerance = 2.0 / WAM_PREDICTION_REVIEW_FPS
    if abs(duration - expected_duration) > duration_tolerance:
        probe_blockers.append(
            "wam_review_duration_does_not_match_executed_controller_timeline:"
            f"expected={expected_duration:.6f}:actual={duration:.6f}"
        )
    if probe_blockers:
        output.unlink(missing_ok=True)
    return _validation(
        resolved_dir,
        blockers=probe_blockers,
        trace_step_count=trace_count,
        ordered_step_indices=indices,
        ordered_clip_count=len(clips),
        concat_mode="executed_control_prefix_reencode",
        codec=codec,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=duration,
        executed_prefix_duration_seconds_by_step=executed_durations,
    )


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _isaac_state_directory(episode_dir: Path) -> tuple[Path | None, list[str]]:
    candidates = [
        episode_dir.parent / ISAAC_STATE_DIR_NAME,
        episode_dir / ISAAC_STATE_DIR_NAME,
    ]
    existing = [candidate.resolve() for candidate in candidates if candidate.is_dir()]
    if not existing:
        return None, ["same_session_isaac_state_directory_missing"]
    if len(set(existing)) != 1:
        return None, ["same_session_isaac_state_directory_ambiguous"]
    return existing[0], []


def _collect_isaac_execution_frames(
    episode_dir: Path,
    *,
    trace_step_count: int,
) -> dict[str, Any]:
    """Bind every execution frame to its same-session action measurement."""

    blockers: list[str] = []
    if trace_step_count < 1:
        blockers.append("same_session_isaac_trace_step_count_invalid")
    state_dir, directory_blockers = _isaac_state_directory(episode_dir)
    blockers.extend(directory_blockers)
    if state_dir is None:
        return {
            "status": "blocked",
            "blockers": sorted(set(blockers)),
            "state_dir": None,
            "frames_dir": None,
            "paths_by_role": {role: [] for role in ISAAC_CAMERA_ROLES},
            "bound_steps": [],
        }

    frames_dir = state_dir / "frames"
    bindings_path = frames_dir / "frame_step_bindings.json"
    bindings_payload = _read_json_mapping(bindings_path)
    if (
        bindings_payload is None
        or bindings_payload.get("schema_version") != ISAAC_FRAME_BINDINGS_SCHEMA_VERSION
        or not isinstance(bindings_payload.get("frames"), Mapping)
    ):
        blockers.append("same_session_isaac_frame_bindings_missing_or_invalid")
        bindings: dict[str, Any] = {}
    else:
        bindings = dict(bindings_payload["frames"])

    expected_measurement_names = {
        f"task_measurement_{index:04d}.json"
        for index in range(1, trace_step_count + 1)
    }
    observed_measurement_names = {
        path.name
        for path in state_dir.glob("task_measurement_[0-9][0-9][0-9][0-9].json")
    }
    if observed_measurement_names != expected_measurement_names:
        blockers.append("same_session_isaac_task_measurement_horizon_mismatch")

    paths_by_role: dict[str, list[Path]] = {role: [] for role in ISAAC_CAMERA_ROLES}
    bound_steps: list[dict[str, Any]] = []
    expected_action_frame_names: set[str] = set()
    ordered_execution_frame_indices: list[int] = []
    ordered_control_frame_indices: list[int] = []
    terminal_execution_frame_indices: list[int] = []
    session_ids: set[str] = set()
    stage_ids: set[str] = set()
    attempt_ids: set[str] = set()
    launch_nonces: set[str] = set()
    baseline_digests: set[str] = set()
    previous_after_timestamp: int | None = None
    previous_control_frame_global_index = 0
    previous_physics_step_count_after: int | None = None
    previous_simulation_time_after: float | None = None

    for source_step_index in range(1, trace_step_count + 1):
        measurement_path = state_dir / f"task_measurement_{source_step_index:04d}.json"
        measurement = _read_json_mapping(measurement_path)
        if measurement is None:
            blockers.append(
                f"same_session_isaac_task_measurement_missing_or_invalid:{source_step_index}"
            )
            measurement = {}
        if measurement.get("schema_version") != "task_transition_measurement.v1":
            blockers.append(
                f"same_session_isaac_task_measurement_schema_invalid:{source_step_index}"
            )
        if measurement.get("source_step_index") != source_step_index:
            blockers.append(
                f"same_session_isaac_task_measurement_source_step_mismatch:{source_step_index}"
            )
        if measurement.get("evidence_step_index") != source_step_index:
            blockers.append(
                f"same_session_isaac_task_measurement_evidence_step_mismatch:{source_step_index}"
            )
        action_sha256 = str(measurement.get("source_action_sha256") or "").lower()
        if not _is_sha256(action_sha256):
            blockers.append(
                f"same_session_isaac_task_measurement_action_sha256_invalid:{source_step_index}"
            )
        binding_fields = {
            field: str(measurement.get(field) or "")
            for field in (
                "source_action_sha256",
                "simulator_session_id",
                "stage_id",
                "before_timestamp",
                "after_timestamp",
                "attempt_id",
                "launch_nonce",
            )
        }
        if any(not value for value in binding_fields.values()):
            blockers.append(
                f"same_session_isaac_task_measurement_binding_incomplete:{source_step_index}"
            )
        for optional_field in (
            "allocation_launch_session_id",
            "qualification_attempt_bound",
            "qualification_attempt_sequence",
            "qualification_attempt_nonce_sha256",
        ):
            if measurement.get(optional_field) is not None:
                binding_fields[optional_field] = str(measurement.get(optional_field))
        session_ids.add(binding_fields["simulator_session_id"])
        stage_ids.add(binding_fields["stage_id"])
        attempt_ids.add(binding_fields["attempt_id"])
        launch_nonces.add(binding_fields["launch_nonce"])
        baseline_digests.add(str(measurement.get("episode_baseline_digest") or ""))
        try:
            before_timestamp = int(binding_fields["before_timestamp"])
            after_timestamp = int(binding_fields["after_timestamp"])
        except ValueError:
            blockers.append(
                f"same_session_isaac_task_measurement_timestamps_invalid:{source_step_index}"
            )
            before_timestamp = after_timestamp = 0
        if after_timestamp <= before_timestamp or (
            previous_after_timestamp is not None
            and before_timestamp < previous_after_timestamp
        ):
            blockers.append(
                f"same_session_isaac_task_measurement_timestamps_not_ordered:{source_step_index}"
            )
        previous_after_timestamp = after_timestamp

        raw_indices = measurement.get("controller_review_frame_indices")
        review_indices = (
            [
                int(value)
                for value in raw_indices
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            if isinstance(raw_indices, Sequence)
            and not isinstance(raw_indices, (str, bytes))
            else []
        )
        executed_count = int(
            measurement.get("controller_horizon_executed_frame_count") or 0
        )
        controller_measurements = measurement.get("controller_frame_measurements")
        controller_measurements = (
            list(controller_measurements)
            if isinstance(controller_measurements, Sequence)
            and not isinstance(controller_measurements, (str, bytes))
            else []
        )
        if len(controller_measurements) != executed_count or executed_count < 1:
            blockers.append(
                f"same_session_isaac_controller_measurement_count_invalid:{source_step_index}"
            )
        terminal_review_index = measurement.get("controller_terminal_review_frame_index")
        terminated_on_success = (
            measurement.get("controller_horizon_terminated_on_semantic_success") is True
        )
        sampled_indices: list[int] = []
        controller_frames: list[dict[str, Any]] = []
        for horizon_position, raw_controller_measurement in enumerate(
            controller_measurements
        ):
            controller_measurement = (
                dict(raw_controller_measurement)
                if isinstance(raw_controller_measurement, Mapping)
                else {}
            )
            control_frame_global_index = int(
                controller_measurement.get("control_frame_global_index") or 0
            )
            expected_global_index = previous_control_frame_global_index + 1
            previous_control_frame_global_index = control_frame_global_index
            try:
                physics_step_count_before = int(
                    controller_measurement.get("physics_step_count_before")
                )
                physics_step_count_after = int(
                    controller_measurement.get("physics_step_count_after")
                )
                physics_step_delta = int(controller_measurement.get("physics_step_delta"))
                simulation_time_before = float(
                    controller_measurement.get("simulation_time_before_seconds")
                )
                simulation_time_after = float(
                    controller_measurement.get("simulation_time_after_seconds")
                )
                simulation_time_delta = float(
                    controller_measurement.get("simulation_time_delta_seconds")
                )
            except (TypeError, ValueError):
                physics_step_count_before = physics_step_count_after = -1
                physics_step_delta = -1
                simulation_time_before = simulation_time_after = -1.0
                simulation_time_delta = -1.0
            physics_step_valid = (
                physics_step_delta == 1
                and physics_step_count_after - physics_step_count_before == 1
                and (
                    previous_physics_step_count_after is None
                    or physics_step_count_before == previous_physics_step_count_after
                )
            )
            simulation_time_valid = (
                math.isclose(
                    simulation_time_delta,
                    1.0 / ISAAC_CONTROLLER_HZ,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    simulation_time_after - simulation_time_before,
                    1.0 / ISAAC_CONTROLLER_HZ,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and (
                    previous_simulation_time_after is None
                    or math.isclose(
                        simulation_time_before,
                        previous_simulation_time_after,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            )
            previous_physics_step_count_after = physics_step_count_after
            previous_simulation_time_after = simulation_time_after
            terminal_frame = controller_measurement.get("semantic_terminal_frame") is True
            scheduled_frame = (
                control_frame_global_index > 0
                and control_frame_global_index
                % ISAAC_REVIEW_CONTROLLER_FRAME_STRIDE
                == 0
            )
            sampled_for_review = scheduled_frame or terminal_frame
            review_index_value = controller_measurement.get("review_frame_index")
            review_index = (
                int(review_index_value)
                if isinstance(review_index_value, int)
                and not isinstance(review_index_value, bool)
                else None
            )
            if (
                control_frame_global_index != expected_global_index
                or not physics_step_valid
                or not simulation_time_valid
                or controller_measurement.get("horizon_frame_index") != horizon_position
                or controller_measurement.get("registered_transition_passed") is not terminal_frame
                or controller_measurement.get("scheduled_review_frame") is not scheduled_frame
                or controller_measurement.get("sampled_for_review") is not sampled_for_review
                or sampled_for_review is not (review_index is not None)
                or (terminal_frame and horizon_position != len(controller_measurements) - 1)
            ):
                blockers.append(
                    "same_session_isaac_controller_frame_measurement_invalid:"
                    f"{source_step_index}:{horizon_position}"
                )
            if not sampled_for_review or review_index is None:
                continue
            expected_review_index = (
                len(ordered_execution_frame_indices)
                + len(sampled_indices)
                + ISAAC_ACTION_FRAME_START_INDEX
            )
            if review_index != expected_review_index:
                blockers.append(
                    "same_session_isaac_review_frame_order_invalid:"
                    f"{source_step_index}:{horizon_position}"
                )
            sampled_indices.append(review_index)
            ordered_control_frame_indices.append(control_frame_global_index)
            role_rows: dict[str, Any] = {}
            artifact_rows = controller_measurement.get("review_frame_artifacts")
            artifact_rows = (
                list(artifact_rows)
                if isinstance(artifact_rows, Sequence)
                and not isinstance(artifact_rows, (str, bytes))
                else []
            )
            artifacts_by_role = {
                str(item.get("camera_role") or ""): dict(item)
                for item in artifact_rows
                if isinstance(item, Mapping)
            }
            for role in ISAAC_CAMERA_ROLES:
                frame_name = f"{role}_{review_index:04d}.png"
                expected_action_frame_names.add(frame_name)
                frame_path = frames_dir / frame_name
                binding_value = bindings.get(frame_name)
                binding = dict(binding_value) if isinstance(binding_value, Mapping) else {}
                if (
                    frame_path.is_symlink()
                    or not frame_path.is_file()
                    or frame_path.stat().st_size <= 0
                ):
                    blockers.append(
                        f"same_session_isaac_execution_frame_missing_or_unsafe:{frame_name}"
                    )
                    continue
                frame_sha256 = _sha256(frame_path)
                artifact_row = artifacts_by_role.get(role, {})
                if (
                    binding.get("camera_role") != role
                    or binding.get("camera_motion_model")
                    != ISAAC_CAMERA_MOTION_MODELS[role]
                    or binding.get("step_index") != review_index
                    or binding.get("control_frame_global_index")
                    != control_frame_global_index
                    or binding.get("physics_step_count_before")
                    != physics_step_count_before
                    or binding.get("physics_step_count_after")
                    != physics_step_count_after
                    or binding.get("physics_step_delta") != 1
                    or not math.isclose(
                        float(binding.get("simulation_time_before_seconds") or -1.0),
                        simulation_time_before,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        float(binding.get("simulation_time_after_seconds") or -1.0),
                        simulation_time_after,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        float(binding.get("simulation_time_delta_seconds") or -1.0),
                        1.0 / ISAAC_CONTROLLER_HZ,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or binding.get("outer_source_step_index") != source_step_index
                    or binding.get("horizon_frame_index") != horizon_position
                    or binding.get("controller_frame_index")
                    != controller_measurement.get("controller_frame_index")
                    or binding.get("source_action_frame_sha256")
                    != controller_measurement.get("source_action_frame_sha256")
                    or binding.get("semantic_terminal_frame") is not terminal_frame
                    or str(binding.get("sha256") or "").lower() != frame_sha256
                    or artifact_row.get("frame_index") != review_index
                    or artifact_row.get("control_frame_global_index")
                    != control_frame_global_index
                    or str(artifact_row.get("sha256") or "").lower() != frame_sha256
                ):
                    blockers.append(f"same_session_isaac_frame_binding_invalid:{frame_name}")
                for field, expected in binding_fields.items():
                    if str(binding.get(field)) != expected:
                        blockers.append(
                            f"same_session_isaac_frame_binding_{field}_mismatch:{frame_name}"
                        )
                paths_by_role[role].append(frame_path)
                role_rows[role] = {
                    "path": str(frame_path),
                    "sha256": frame_sha256,
                    "binding": binding,
                }
            controller_frames.append(
                {
                    "review_frame_index": review_index,
                    "control_frame_global_index": control_frame_global_index,
                    "horizon_frame_index": horizon_position,
                    "semantic_terminal_frame": terminal_frame,
                    "frames": role_rows,
                }
            )
        if (
            review_indices != sampled_indices
            or int(measurement.get("controller_review_frame_count") or 0)
            != len(sampled_indices)
        ):
            blockers.append(
                f"same_session_isaac_controller_review_frame_indices_invalid:{source_step_index}"
            )
        ordered_execution_frame_indices.extend(sampled_indices)
        terminal_sampled_indices = [
            int(row["review_frame_index"])
            for row in controller_frames
            if row.get("semantic_terminal_frame") is True
        ]
        if terminated_on_success:
            if terminal_sampled_indices != [terminal_review_index]:
                blockers.append(
                    f"same_session_isaac_terminal_review_frame_invalid:{source_step_index}"
                )
            elif terminal_review_index is not None:
                terminal_execution_frame_indices.append(int(terminal_review_index))
        elif terminal_review_index is not None or terminal_sampled_indices:
            blockers.append(
                f"same_session_isaac_terminal_review_frame_unexpected:{source_step_index}"
            )
        bound_steps.append(
            {
                "source_step_index": source_step_index,
                "measurement_path": str(measurement_path),
                "measurement_sha256": _sha256(measurement_path)
                if measurement_path.is_file()
                else None,
                "source_action_sha256": action_sha256 or None,
                "controller_frames": controller_frames,
                "terminal_review_frame_index": terminal_review_index,
            }
        )

    expected_initial_frame_names = {f"{role}_0000.png" for role in ISAAC_CAMERA_ROLES}
    expected_frame_names = expected_initial_frame_names | expected_action_frame_names
    observed_frame_names = {
        path.name
        for role in ISAAC_CAMERA_ROLES
        for path in frames_dir.glob(f"{role}_[0-9][0-9][0-9][0-9].png")
    }
    if observed_frame_names != expected_frame_names:
        blockers.append("same_session_isaac_execution_frame_horizon_mismatch")
    if bindings and set(bindings) != expected_action_frame_names:
        blockers.append("same_session_isaac_frame_binding_horizon_mismatch")
    initial_bindings_path = frames_dir / ISAAC_INITIAL_FRAME_BINDINGS_NAME
    initial_bindings_payload = _read_json_mapping(initial_bindings_path)
    if (
        initial_bindings_payload is None
        or initial_bindings_payload.get("schema_version")
        != ISAAC_INITIAL_FRAME_BINDINGS_SCHEMA_VERSION
        or not isinstance(initial_bindings_payload.get("frames"), Mapping)
    ):
        blockers.append("same_session_isaac_initial_frame_bindings_missing_or_invalid")
        initial_bindings: dict[str, Any] = {}
    else:
        initial_bindings = dict(initial_bindings_payload["frames"])
    if set(initial_bindings) != expected_initial_frame_names:
        blockers.append("same_session_isaac_initial_frame_binding_horizon_mismatch")
    initial_frame_evidence: dict[str, Any] = {}
    for role in ISAAC_CAMERA_ROLES:
        name = f"{role}_0000.png"
        initial_path = frames_dir / name
        binding_value = initial_bindings.get(name)
        binding = dict(binding_value) if isinstance(binding_value, Mapping) else {}
        if (
            initial_path.is_symlink()
            or not initial_path.is_file()
            or initial_path.stat().st_size <= 0
        ):
            blockers.append(f"same_session_isaac_initial_frame_missing_or_unsafe:{name}")
            continue
        initial_sha256 = _sha256(initial_path)
        if (
            binding.get("camera_role") != role
            or binding.get("camera_motion_model")
            != ISAAC_CAMERA_MOTION_MODELS[role]
            or binding.get("step_index") != 0
            or binding.get("review_frame_index") != 0
            or binding.get("control_frame_global_index") != 0
            or binding.get("initial_frame") is not True
            or str(binding.get("sha256") or "").lower() != initial_sha256
            or str(binding.get("simulator_session_id") or "") not in session_ids
            or str(binding.get("stage_id") or "") not in stage_ids
            or str(binding.get("attempt_id") or "") not in attempt_ids
            or str(binding.get("launch_nonce") or "") not in launch_nonces
            or str(binding.get("episode_baseline_digest") or "")
            not in baseline_digests
        ):
            blockers.append(f"same_session_isaac_initial_frame_binding_invalid:{name}")
        paths_by_role[role].insert(0, initial_path)
        initial_frame_evidence[role] = {
            "path": str(initial_path),
            "sha256": initial_sha256,
            "binding": binding,
        }

    for label, values in (
        ("simulator_session_id", session_ids),
        ("stage_id", stage_ids),
        ("attempt_id", attempt_ids),
        ("launch_nonce", launch_nonces),
        ("episode_baseline_digest", baseline_digests),
    ):
        nonempty = {value for value in values if value}
        if len(nonempty) != 1:
            blockers.append(f"same_session_isaac_{label}_not_unique")

    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "state_dir": str(state_dir),
        "frames_dir": str(frames_dir),
        "frame_bindings_path": str(bindings_path),
        "frame_bindings_sha256": _sha256(bindings_path)
        if bindings_path.is_file()
        else None,
        "initial_frame_bindings_path": str(initial_bindings_path),
        "initial_frame_bindings_sha256": _sha256(initial_bindings_path)
        if initial_bindings_path.is_file()
        else None,
        "simulator_session_id": next(iter(session_ids)) if len(session_ids) == 1 else None,
        "stage_id": next(iter(stage_ids)) if len(stage_ids) == 1 else None,
        "attempt_id": next(iter(attempt_ids)) if len(attempt_ids) == 1 else None,
        "launch_nonce": next(iter(launch_nonces)) if len(launch_nonces) == 1 else None,
        "ordered_execution_frame_indices": ordered_execution_frame_indices,
        "ordered_execution_frame_count": len(ordered_execution_frame_indices),
        "ordered_review_frame_indices": [0, *ordered_execution_frame_indices],
        "ordered_review_control_frame_indices": [0, *ordered_control_frame_indices],
        "ordered_review_frame_count": len(ordered_execution_frame_indices) + 1,
        "terminal_execution_frame_indices": terminal_execution_frame_indices,
        "initial_frame_evidence": initial_frame_evidence,
        "paths_by_role": {
            role: [str(path) for path in paths]
            for role, paths in paths_by_role.items()
        },
        "bound_steps": bound_steps,
    }


def _encode_isaac_role_video(
    *,
    frames_dir: Path,
    role: str,
    frame_count: int,
    control_frame_indices: Sequence[int],
    output: Path,
) -> dict[str, Any]:
    output.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(ISAAC_REVIEW_FPS),
        "-start_number",
        "0",
        "-i",
        str(frames_dir / f"{role}_%04d.png"),
        "-frames:v",
        str(frame_count),
        "-vf",
        f"scale={MINIMUM_WIDTH}:{MINIMUM_HEIGHT},setsar=1,format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    completed = _run(command)
    blockers: list[str] = []
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        blockers.append(f"same_session_isaac_{role}_video_encode_failed")
        output.unlink(missing_ok=True)
        codec = None
        width = height = observed_frames = 0
        duration = 0.0
    else:
        codec, width, height, observed_frames, duration, probe_blockers = _probe(output)
        blockers.extend(f"{role}:{blocker}" for blocker in probe_blockers)
        if observed_frames != frame_count:
            blockers.append(
                f"same_session_isaac_{role}_video_frame_count_mismatch:"
                f"{observed_frames}!={frame_count}"
            )
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "camera_role": role,
        "path": str(output) if output.is_file() else None,
        "sha256": _sha256(output) if output.is_file() else None,
        "codec": codec,
        "width": width,
        "height": height,
        "frame_count": observed_frames,
        "duration_seconds": duration,
        "fps": ISAAC_REVIEW_FPS,
        "source_control_hz": ISAAC_CONTROLLER_HZ,
        "source_control_frame_indices": [int(value) for value in control_frame_indices],
        "source_control_frame_deltas": [
            int(control_frame_indices[index]) - int(control_frame_indices[index - 1])
            for index in range(1, len(control_frame_indices))
        ],
        "semantic_terminal_frame_always_included": True,
        "camera_motion_model": (
            ISAAC_CAMERA_MOTION_MODELS[role]
        ),
        "source": "hash_bound_same_session_isaac_png_frames",
    }


def _encode_primary_robot_pov_review(
    *,
    robot_pov_video: Path,
    frame_count: int,
    output: Path,
) -> tuple[str | None, int, int, int, float, list[str]]:
    output.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(robot_pov_video),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    completed = _run(command)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        output.unlink(missing_ok=True)
        return None, 0, 0, 0, 0.0, [
            "same_session_isaac_robot_pov_primary_review_encode_failed"
        ]
    codec, width, height, observed_frames, duration, blockers = _probe(output)
    if observed_frames != frame_count:
        blockers.append(
            "same_session_isaac_robot_pov_primary_review_frame_count_mismatch:"
            f"{observed_frames}!={frame_count}"
        )
    if width != MINIMUM_WIDTH or height != MINIMUM_HEIGHT:
        blockers.append(
            "same_session_isaac_robot_pov_primary_review_resolution_invalid:"
            f"{width}x{height}"
        )
    return codec, width, height, observed_frames, duration, blockers


def _final_validation(
    episode_dir: Path,
    *,
    blockers: Sequence[str],
    trace_step_count: int,
    frame_evidence: Mapping[str, Any] | None = None,
    role_videos: Mapping[str, Any] | None = None,
    wam_prediction_review: Mapping[str, Any] | None = None,
    codec: str | None = None,
    width: int = 0,
    height: int = 0,
    frame_count: int = 0,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    output = episode_dir / OUTPUT_NAME
    unique_blockers = sorted({str(value) for value in blockers if str(value)})
    passed = not unique_blockers and output.is_file() and output.stat().st_size > 0
    ordered_indices = list(range(1, trace_step_count + 1))
    evidence = dict(frame_evidence or {})
    ordered_review_frame_count = int(evidence.get("ordered_review_frame_count") or 0)
    ordered_review_frame_indices = [
        int(value) for value in evidence.get("ordered_review_frame_indices") or []
    ]
    ordered_control_frame_indices = [
        int(value)
        for value in evidence.get("ordered_review_control_frame_indices") or []
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "blockers": unique_blockers,
        "path": str(output) if passed else None,
        "candidate_path": str(output) if output.is_file() else None,
        "sha256": _sha256(output) if passed else None,
        "codec": codec,
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "duration_seconds": float(duration_seconds),
        "trace_step_count": int(trace_step_count),
        "ordered_clip_count": int(trace_step_count) if evidence.get("status") == "passed" else 0,
        "ordered_step_indices": ordered_indices,
        "ordered_execution_frame_count": int(
            evidence.get("ordered_execution_frame_count") or 0
        ),
        "ordered_review_frame_count": ordered_review_frame_count,
        "ordered_review_frame_indices": ordered_review_frame_indices,
        "ordered_review_control_frame_indices": ordered_control_frame_indices,
        "terminal_execution_frame_indices": list(
            evidence.get("terminal_execution_frame_indices") or []
        ),
        "episode_order_verified": bool(
            passed
            and trace_step_count > 0
            and ordered_review_frame_count > 1
            and ordered_review_frame_indices == list(range(ordered_review_frame_count))
            and len(ordered_control_frame_indices) == ordered_review_frame_count
        ),
        "concat_mode": "primary_same_session_isaac_robot_pov_only",
        "primary_camera_role": "robot_pov",
        "overview_excluded_from_primary_review": True,
        "video_frame_count_mode": "dynamic_from_actual_episode_duration",
        "review_sampling": {
            "source_control_hz": ISAAC_CONTROLLER_HZ,
            "nominal_review_hz": ISAAC_REVIEW_FPS,
            "controller_frame_stride": ISAAC_REVIEW_CONTROLLER_FRAME_STRIDE,
            "initial_frame_included": True,
            "semantic_terminal_frame_always_included": True,
            "terminal_off_stride_interval_preserved_in_bindings": True,
        },
        "minimum_resolution": {"width": MINIMUM_WIDTH, "height": MINIMUM_HEIGHT},
        "review_source": "persistent_same_session_isaac_execution_frames",
        "execution_truth": True,
        "same_session_isaac_frames": bool(evidence.get("status") == "passed"),
        "required_camera_roles": list(ISAAC_CAMERA_ROLES),
        "isaac_frame_evidence": evidence,
        "isaac_role_videos": dict(role_videos or {}),
        "wam_prediction_review": dict(wam_prediction_review or {}),
        "claim_boundary": {
            "contains_only_same_session_isaac_execution_frames": True,
            "oscar_wam_prediction_pixels_excluded_from_final_review": True,
            "separate_wam_prediction_review_is_model_derived_support": True,
            "video_is_review_media_not_semantic_success_attestation": True,
            "robot_pov_camera_motion_model": "rigid_head_local_transform",
            "robot_pov_camera_inherits_head_translation_and_rotation": True,
            "robot_pov_camera_reaims_at_task_each_frame": False,
            "primary_review_is_robot_head_egocentric_only": True,
            "overview_available_only_as_separate_diagnostic": True,
        },
    }
    _write_json(episode_dir / VALIDATION_NAME, result)
    return result


def build_episode_review(episode_dir: str | Path) -> dict[str, Any]:
    """Build separate WAM-prediction and same-session Isaac execution reviews."""

    resolved_dir = Path(episode_dir).expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (*ISAAC_ROLE_OUTPUT_NAMES.values(), OUTPUT_NAME):
        (resolved_dir / stale_name).unlink(missing_ok=True)
    wam_review = build_wam_prediction_review(resolved_dir)
    trace_count = int(wam_review.get("trace_step_count") or 0)
    blockers: list[str] = []
    if wam_review.get("status") != "passed":
        blockers.append("separate_wam_prediction_review_not_passed")

    frame_evidence = _collect_isaac_execution_frames(
        resolved_dir,
        trace_step_count=trace_count,
    )
    blockers.extend(frame_evidence.get("blockers") or [])
    role_videos: dict[str, Any] = {}
    if frame_evidence.get("status") == "passed":
        frames_dir = Path(str(frame_evidence["frames_dir"]))
        review_frame_count = int(frame_evidence.get("ordered_review_frame_count") or 0)
        control_frame_indices = [
            int(value)
            for value in frame_evidence.get("ordered_review_control_frame_indices") or []
        ]
        for role in ISAAC_CAMERA_ROLES:
            role_result = _encode_isaac_role_video(
                frames_dir=frames_dir,
                role=role,
                frame_count=review_frame_count,
                control_frame_indices=control_frame_indices,
                output=resolved_dir / ISAAC_ROLE_OUTPUT_NAMES[role],
            )
            role_videos[role] = role_result
            blockers.extend(role_result.get("blockers") or [])

    codec: str | None = None
    width = height = observed_frames = 0
    duration = 0.0
    if role_videos and all(
        dict(role_videos.get(role) or {}).get("status") == "passed"
        for role in ISAAC_CAMERA_ROLES
    ):
        codec, width, height, observed_frames, duration, primary_blockers = (
            _encode_primary_robot_pov_review(
                robot_pov_video=resolved_dir / ISAAC_ROLE_OUTPUT_NAMES["robot_pov"],
                frame_count=int(
                    frame_evidence.get("ordered_review_frame_count") or 0
                ),
                output=resolved_dir / OUTPUT_NAME,
            )
        )
        blockers.extend(primary_blockers)
    else:
        (resolved_dir / OUTPUT_NAME).unlink(missing_ok=True)

    return _final_validation(
        resolved_dir,
        blockers=blockers,
        trace_step_count=trace_count,
        frame_evidence=frame_evidence,
        role_videos=role_videos,
        wam_prediction_review=wam_review,
        codec=codec,
        width=width,
        height=height,
        frame_count=observed_frames,
        duration_seconds=duration,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("usage: python -m blueprint_pipeline.groot_oscar_episode_review EPISODE_DIR", file=sys.stderr)
        return 2
    episode_dir = Path(args[0]).expanduser().resolve()
    try:
        result = build_episode_review(episode_dir)
    except Exception as exc:  # noqa: BLE001 - CLI must persist a terminal failure when possible
        episode_dir.mkdir(parents=True, exist_ok=True)
        result = _final_validation(
            episode_dir,
            blockers=[f"episode_review_builder_exception:{type(exc).__name__}"],
            trace_step_count=0,
        )
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
