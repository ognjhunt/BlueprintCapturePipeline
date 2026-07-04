"""LeRobot/GR00T-style per-episode export from simulator batch streams.

Two deliverables, both fail-closed:

1. ``build_modality_config`` — a GR00T ``modality.json`` generated from a
   :class:`RobotProfile`: named index slices into the concatenated action
   (and, when available, state) vectors, plus video keys from the profile's
   camera rigs. The action layout is the SC3 7D delta end-effector pose the
   pipeline already validates (``delta_position_m`` [0:3],
   ``delta_rotation_axis_angle`` [3:6], ``gripper`` [6:7]).

2. ``build_lerobot_episode_export`` — maps
   ``simulator_command_batch_control_stream.jsonl`` (+ the attempt trace) into
   per-episode rows: one row per control action, LeRobot field names
   (``action``, ``observation.state``, ``timestamp``, ``episode_index``,
   ``frame_index``), with ``meta/info.json``, ``meta/episodes.jsonl``,
   ``meta/tasks.jsonl``, ``meta/stats.json``, and ``meta/modality.json``.

Fail-closed rules (never zero-filled, never synthesized):

- missing control stream file -> export status ``blocked``;
- an attempt with no control rows, or any action that does not parse as a
  valid SC3 7D vector -> that episode is EXCLUDED with a per-episode blocker;
- per-step state or timestamps absent from the stream -> the fields are
  omitted (not zero-filled) and the episode is flagged; ``gr00t_ready`` stays
  False until state, timestamps, and materialized video all exist;
- parquet requires pyarrow; without it the canonical JSONL is still written
  and ``parquet_status`` says ``blocked_missing_pyarrow`` instead of silently
  pretending.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .scene_placement.robot_profile import RobotProfile, get_robot_profile

LEROBOT_EPISODE_EXPORT_SCHEMA_VERSION = "lerobot_episode_export.v1"
MODALITY_CONFIG_SCHEMA_VERSION = "gr00t_modality_config.v1"

CONTROL_STREAM_FILENAME = "simulator_command_batch_control_stream.jsonl"
ATTEMPT_TRACE_FILENAME = "simulator_command_batch_attempt_trace.jsonl"

# SC3 7D delta end-effector pose layout — the action contract the pipeline
# already validates in post_training_data_package._sc3_action_vector.
SC3_ACTION_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("delta_position_m", 0, 3),
    ("delta_rotation_axis_angle", 3, 6),
    ("gripper", 6, 7),
)
SC3_ACTION_DIM = 7

# Default per-step state layout when a stream carries robot state. Base pose
# as position + wxyz quaternion. States are only exported when the stream rows
# actually carry them; this layout is a declaration, not a promise.
DEFAULT_STATE_LAYOUTS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "humanoid": (
        ("base_position_m", 0, 3),
        ("base_orientation_quat_wxyz", 3, 7),
    ),
}

_STATE_VECTOR_KEYS = ("base_pose_7d", "state_vector", "robot_state_vector")
_TIMESTAMP_KEYS = ("timestamp", "time_s", "sim_time_s")

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "lerobot_episode_export_support",
    "episodes_are_simulator_traces_not_physical_robot_data": True,
    "attempt_success_labels_are_simulator_criteria_labels_not_task_success_proof": True,
    "gr00t_ready_requires_state_timestamps_and_materialized_video": True,
    "absent_fields_are_omitted_never_zero_filled": True,
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _float_vector(value: Any, dim: int) -> List[float] | None:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == dim
    ):
        values = [_finite_float(item) for item in value]
        if all(item is not None for item in values):
            return [float(item) for item in values if item is not None]
    return None


# ---------------------------------------------------------------------------
# 1. modality.json from RobotProfile
# ---------------------------------------------------------------------------


def _slices(layout: Sequence[tuple[str, int, int]]) -> Dict[str, Dict[str, int]]:
    return {name: {"start": start, "end": end} for name, start, end in layout}


def build_modality_config(
    profile: RobotProfile,
    *,
    state_layout: Sequence[tuple[str, int, int]] | None = None,
) -> Dict[str, Any]:
    """GR00T-style modality config: named slices into state/action vectors.

    The action block is always emitted (the SC3 7D contract is validated
    elsewhere in the pipeline). The state block is a declared layout; whether
    episodes actually carry state is reported per-episode by the export.
    """
    resolved_state_layout = tuple(
        state_layout
        if state_layout is not None
        else DEFAULT_STATE_LAYOUTS.get(profile.embodiment_type, ())
    )
    video = {}
    for rig in profile.camera_rigs:
        rig_map = _mapping(rig)
        camera_id = _string(rig_map.get("camera_id"))
        if camera_id:
            video[camera_id] = {
                "original_key": f"observation.images.{camera_id}",
                "mount": rig_map.get("mount"),
                "modalities": rig_map.get("modalities"),
            }
    return {
        "schema_version": MODALITY_CONFIG_SCHEMA_VERSION,
        "robot_id": profile.robot_id,
        "embodiment_type": profile.embodiment_type,
        "state": _slices(resolved_state_layout),
        "state_dim": max((end for _, _, end in resolved_state_layout), default=0),
        "action": {
            "sc3_7d_delta_end_effector_pose": {
                "start": 0,
                "end": SC3_ACTION_DIM,
                "absolute": False,
                "fields": _slices(SC3_ACTION_LAYOUT),
            }
        },
        "action_dim": SC3_ACTION_DIM,
        "video": video,
        "annotation": {"human.task_description": {"original_key": "task"}},
        "source_action_interface": dict(profile.action_interface),
        "source_observation_schema": dict(profile.observation_schema),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


# ---------------------------------------------------------------------------
# 2. per-episode rows from the simulator batch streams
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _sc3_vector_from_action(action: Any) -> List[float] | None:
    vector = _float_vector(action, SC3_ACTION_DIM)
    if vector is not None:
        return vector
    payload = _mapping(action)
    if not payload:
        return None
    for key in (
        "sc3_7d_delta_ee_pose",
        "sc3_action_vector",
        "action_vector_7d",
        "delta_end_effector_pose_7d",
        "delta_ee_pose_7d",
    ):
        vector = _float_vector(payload.get(key), SC3_ACTION_DIM)
        if vector is not None:
            return vector
    normalized = _mapping(payload.get("normalized_action"))
    if normalized:
        return _sc3_vector_from_action(normalized)
    position = _float_vector(
        payload.get("delta_position_m") or payload.get("translation_delta_m"), 3
    )
    rotation = _float_vector(
        payload.get("delta_rotation_axis_angle")
        or payload.get("rotation_delta_axis_angle"),
        3,
    )
    gripper = _finite_float(
        payload.get("gripper_delta")
        if payload.get("gripper_delta") is not None
        else payload.get("gripper")
    )
    if position is not None and rotation is not None and gripper is not None:
        return [*position, *rotation, gripper]
    return None


def _state_from_payload(payload: Mapping[str, Any], state_dim: int) -> List[float] | None:
    if state_dim <= 0:
        return None
    for key in _STATE_VECTOR_KEYS:
        vector = _float_vector(payload.get(key), state_dim)
        if vector is not None:
            return vector
    return None


def _timestamp_from_payload(payload: Mapping[str, Any]) -> float | None:
    for key in _TIMESTAMP_KEYS:
        value = _finite_float(payload.get(key))
        if value is not None:
            return value
    return None


def _vector_stats(vectors: Sequence[Sequence[float]]) -> Dict[str, List[float]] | None:
    if not vectors:
        return None
    dim = len(vectors[0])
    count = len(vectors)
    means = [sum(vector[i] for vector in vectors) / count for i in range(dim)]
    stds = [
        math.sqrt(
            sum((vector[i] - means[i]) ** 2 for vector in vectors) / count
        )
        for i in range(dim)
    ]
    return {
        "mean": [round(v, 9) for v in means],
        "std": [round(v, 9) for v in stds],
        "min": [min(vector[i] for vector in vectors) for i in range(dim)],
        "max": [max(vector[i] for vector in vectors) for i in range(dim)],
        "count": [count] * dim,
    }


def _try_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        return "blocked_missing_pyarrow"
    columns: Dict[str, List[Any]] = {}
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns[key] = []
                keys.append(key)
    for row in rows:
        for key in keys:
            columns[key].append(row.get(key))
    pq.write_table(pa.table(columns), str(path))
    return "written"


def build_lerobot_episode_export(
    *,
    job_dir: str | Path,
    output_dir: str | Path,
    robot_id: str | None = None,
    robot_profile: RobotProfile | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Map simulator batch streams into per-episode LeRobot-style rows.

    One episode per attempt in the attempt trace. Episodes fail closed: a
    missing control stream blocks the export; an attempt whose control rows
    are missing or whose actions do not parse as SC3 7D vectors is excluded
    with a blocker, never padded.
    """
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    export_root = Path(output_dir).expanduser().resolve() / "lerobot_episode_export"
    stamp = generated_at or utc_now_iso()
    profile = robot_profile
    if profile is None and robot_id:
        profile = get_robot_profile(robot_id)
    modality = build_modality_config(profile) if profile else None
    state_dim = int(modality["state_dim"]) if modality else 0

    blockers: List[str] = []
    control_path = resolved_job_dir / CONTROL_STREAM_FILENAME
    trace_path = resolved_job_dir / ATTEMPT_TRACE_FILENAME
    if not control_path.is_file():
        blockers.append("control_stream_missing")
    if not trace_path.is_file():
        blockers.append("attempt_trace_missing")
    if profile is None:
        blockers.append("robot_profile_missing")

    manifest: Dict[str, Any] = {
        "schema_version": LEROBOT_EPISODE_EXPORT_SCHEMA_VERSION,
        "generated_at": stamp,
        "job_dir": str(resolved_job_dir),
        "export_dir": str(export_root),
        "robot_id": profile.robot_id if profile else None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    if blockers:
        manifest.update(
            {
                "status": "blocked",
                "blockers": sorted(blockers),
                "episode_count": 0,
                "excluded_episode_count": 0,
            }
        )
        ensure_dir(export_root)
        write_json(export_root / "lerobot_episode_export_manifest.json", manifest)
        return manifest

    control_rows = _read_jsonl(control_path)
    attempts = _read_jsonl(trace_path)
    actions_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
    for row in control_rows:
        if row.get("stream_type") != "control_action":
            continue
        attempt_id = _string(row.get("attempt_id"))
        if attempt_id:
            actions_by_attempt.setdefault(attempt_id, []).append(row)

    data_dir = export_root / "data"
    meta_dir = export_root / "meta"
    ensure_dir(data_dir)
    ensure_dir(meta_dir)

    tasks: List[str] = []
    task_indices: Dict[str, int] = {}
    episodes_meta: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    all_action_vectors: List[List[float]] = []
    all_state_vectors: List[List[float]] = []
    all_timestamps: List[float] = []
    episode_index = 0
    global_index = 0

    for attempt in attempts:
        attempt_id = _string(attempt.get("attempt_id"))
        episode_blockers: List[str] = []
        control = sorted(
            actions_by_attempt.get(attempt_id, []),
            key=lambda row: int(row.get("action_index") or 0),
        )
        if not attempt_id:
            episode_blockers.append("attempt_id_missing")
        if not control:
            episode_blockers.append("control_rows_missing_for_attempt")

        rows: List[Dict[str, Any]] = []
        state_present = bool(control) and state_dim > 0
        timestamps_present = bool(control)
        task_text = _string(attempt.get("task_id")) or "unspecified_task"
        if task_text not in task_indices:
            task_indices[task_text] = len(tasks)
            tasks.append(task_text)

        for frame_index, control_row in enumerate(control):
            payload = _mapping(control_row.get("action"))
            vector = _sc3_vector_from_action(
                control_row.get("action") if not payload else payload
            )
            if vector is None:
                episode_blockers.append(
                    f"sc3_7d_action_invalid_at_index:{control_row.get('action_index')}"
                )
                continue
            row: Dict[str, Any] = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "index": global_index + frame_index,
                "task_index": task_indices[task_text],
                "task": task_text,
                "action": vector,
                "attempt_id": attempt_id,
                "episode_id": attempt.get("episode_id"),
                "scenario_id": attempt.get("scenario_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
            }
            state = _state_from_payload(payload, state_dim)
            if state is not None:
                row["observation.state"] = state
            else:
                state_present = False
            timestamp = _timestamp_from_payload(payload)
            if timestamp is not None:
                row["timestamp"] = timestamp
            else:
                timestamps_present = False
            rows.append(row)

        if episode_blockers:
            excluded.append(
                {
                    "attempt_id": attempt_id or None,
                    "blockers": sorted(set(episode_blockers)),
                }
            )
            continue

        episode_file = data_dir / f"episode_{episode_index:06d}.jsonl"
        episode_file.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        parquet_status = _try_write_parquet(
            data_dir / f"episode_{episode_index:06d}.parquet", rows
        )
        all_action_vectors.extend(row["action"] for row in rows)
        all_state_vectors.extend(
            row["observation.state"] for row in rows if "observation.state" in row
        )
        all_timestamps.extend(row["timestamp"] for row in rows if "timestamp" in row)
        episodes_meta.append(
            {
                "episode_index": episode_index,
                "attempt_id": attempt_id,
                "length": len(rows),
                "task": task_text,
                "task_index": task_indices[task_text],
                "state_present": state_present,
                "timestamps_present": timestamps_present,
                "video_present": False,
                "gr00t_ready": False,
                "gr00t_ready_missing": [
                    item
                    for item, present in (
                        ("per_step_state", state_present),
                        ("per_step_timestamps", timestamps_present),
                        ("materialized_video", False),
                    )
                    if not present
                ],
                "attempt_success_label": attempt.get("success")
                if isinstance(attempt.get("success"), bool)
                else None,
                "parquet_status": parquet_status,
                "file": f"data/episode_{episode_index:06d}.jsonl",
            }
        )
        global_index += len(rows)
        episode_index += 1

    fps = None
    if len(all_timestamps) >= 2:
        deltas = [
            later - earlier
            for earlier, later in zip(all_timestamps, all_timestamps[1:])
            if later > earlier
        ]
        if deltas:
            fps = round(1.0 / (sum(deltas) / len(deltas)), 3)

    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in episodes_meta)
        + ("\n" if episodes_meta else ""),
        encoding="utf-8",
    )
    (meta_dir / "tasks.jsonl").write_text(
        "\n".join(
            json.dumps({"task_index": index, "task": task})
            for index, task in enumerate(tasks)
        )
        + ("\n" if tasks else ""),
        encoding="utf-8",
    )
    stats = {
        "action": _vector_stats(all_action_vectors),
        "observation.state": _vector_stats(all_state_vectors),
    }
    write_json(meta_dir / "stats.json", stats)
    if modality:
        write_json(meta_dir / "modality.json", modality)
    write_json(
        meta_dir / "info.json",
        {
            "schema_version": LEROBOT_EPISODE_EXPORT_SCHEMA_VERSION,
            "robot_id": profile.robot_id if profile else None,
            "fps": fps,
            "total_episodes": len(episodes_meta),
            "total_frames": global_index,
            "features": {
                "action": {"dtype": "float32", "shape": [SC3_ACTION_DIM]},
                **(
                    {"observation.state": {"dtype": "float32", "shape": [state_dim]}}
                    if state_dim
                    else {}
                ),
            },
        },
    )

    if not episodes_meta:
        blockers.append("no_exportable_episodes")
    manifest.update(
        {
            "status": "completed_review_required" if episodes_meta else "blocked",
            "blockers": sorted(blockers),
            "episode_count": len(episodes_meta),
            "total_frame_count": global_index,
            "excluded_episode_count": len(excluded),
            "excluded_episodes": excluded,
            "fps": fps,
            "gr00t_ready_episode_count": sum(
                1 for row in episodes_meta if row["gr00t_ready"]
            ),
            "modality_config_path": "meta/modality.json" if modality else None,
            "parquet_status": (
                episodes_meta[0]["parquet_status"] if episodes_meta else None
            ),
        }
    )
    write_json(export_root / "lerobot_episode_export_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robot-id", default="unitree_g1")
    args = parser.parse_args(argv)
    manifest = build_lerobot_episode_export(
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        robot_id=args.robot_id,
    )
    print(json.dumps({"status": manifest.get("status"), "episode_count": manifest.get("episode_count")}))
    return 0 if manifest.get("status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
