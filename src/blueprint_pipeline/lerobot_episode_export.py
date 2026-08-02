"""LeRobot/GR00T-style per-episode export from simulator batch streams.

Two deliverables, both fail-closed:

1. ``build_modality_config`` — a GR00T ``modality.json`` generated from a
   :class:`RobotProfile`: named index slices into the concatenated action
   (and, when available, state) vectors, plus video keys from the profile's
   camera rigs. The preferred action layout comes from the robot profile. A
   legacy SC3 7D delta end-effector layout remains available only as an
   explicit compatibility fallback.

2. ``build_lerobot_episode_export`` — maps
   ``simulator_command_batch_control_stream.jsonl`` (+ the attempt trace) into
   per-episode rows: one row per control action, LeRobot field names
   (``action``, ``observation.state``, ``timestamp``, ``episode_index``,
   ``frame_index``), with ``meta/info.json``, ``meta/episodes.jsonl``,
   ``meta/tasks.jsonl``, ``meta/stats.json``, and ``meta/modality.json``.

Fail-closed rules (never zero-filled, never synthesized):

- missing control stream file -> export status ``blocked``;
- an attempt with no control rows, or any action that does not parse as a
  valid vector for the selected robot-profile action layout -> that episode is
  EXCLUDED with a per-episode blocker;
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
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .scene_placement.robot_profile import DEFAULT_ROBOT_ID, RobotProfile, get_robot_profile

LEROBOT_EPISODE_EXPORT_SCHEMA_VERSION = "lerobot_episode_export.v2"
MODALITY_CONFIG_SCHEMA_VERSION = "gr00t_modality_config.v2"
ACTION_LAYOUT_SCHEMA_VERSION = "blueprint_lerobot_action_layout.v1"

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
SC3_ACTION_LAYOUT_ID = "sc3_7d_delta_end_effector_pose"
GENERIC_ACTION_VECTOR_KEYS = (
    "action_vector",
    "actions",
    "action_values",
    "policy_action_vector",
)

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
    "observation_source_columns_are_metadata_not_rights_clearance": True,
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "model_derived"}
    return False


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


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _sc3_action_layout(*, source: str = "legacy_sc3_default") -> Dict[str, Any]:
    return {
        "schema_version": ACTION_LAYOUT_SCHEMA_VERSION,
        "layout_id": SC3_ACTION_LAYOUT_ID,
        "action_dim": SC3_ACTION_DIM,
        "absolute": False,
        "segments": [
            {
                "name": name,
                "start": start,
                "end": end,
                "source_keys": [name],
            }
            for name, start, end in SC3_ACTION_LAYOUT
        ],
        "vector_keys": [
            "sc3_7d_delta_ee_pose",
            "sc3_action_vector",
            "action_vector_7d",
            "delta_end_effector_pose_7d",
            "delta_ee_pose_7d",
        ],
        "layout_source": source,
        "claim_boundary": (
            "Legacy single-end-effector SC3 adapter layout; it does not claim "
            "bimanual, whole-body, mobile-base, or physical robot policy coverage."
        ),
    }


def _segments_from_layout_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, Sequence) and not isinstance(raw_segments, (str, bytes, bytearray)):
        for raw in raw_segments:
            segment = _mapping(raw)
            name = _string(segment.get("name"))
            start = _non_negative_int(segment.get("start"))
            end = _non_negative_int(segment.get("end"))
            if not name or start is None or end is None or end <= start:
                continue
            source_keys = [
                _string(key)
                for key in (
                    segment.get("source_keys")
                    if isinstance(segment.get("source_keys"), Sequence)
                    and not isinstance(segment.get("source_keys"), (str, bytes, bytearray))
                    else []
                )
                if _string(key)
            ]
            segments.append(
                {
                    "name": name,
                    "start": start,
                    "end": end,
                    "source_keys": source_keys or [name],
                }
            )
    return sorted(segments, key=lambda item: int(item["start"]))


def _layout_dim(layout: Mapping[str, Any]) -> int:
    value = _non_negative_int(layout.get("action_dim"))
    if value:
        return value
    segments = _segments_from_layout_payload(layout)
    return max((int(segment["end"]) for segment in segments), default=0)


def _layout_id(layout: Mapping[str, Any]) -> str:
    return _string(layout.get("layout_id")) or SC3_ACTION_LAYOUT_ID


def _profile_action_layout(profile: RobotProfile) -> Dict[str, Any]:
    profile_payload = _mapping(profile.action_interface.get("lerobot_export"))
    if not profile_payload:
        return _sc3_action_layout()
    layout_id = _string(profile_payload.get("layout_id"))
    action_dim = _non_negative_int(profile_payload.get("action_dim"))
    segments = _segments_from_layout_payload(profile_payload)
    max_segment_end = max((int(segment["end"]) for segment in segments), default=0)
    if not layout_id or not action_dim or not segments or max_segment_end != action_dim:
        return _sc3_action_layout(source="legacy_sc3_default_profile_layout_invalid")
    vector_keys = [
        _string(key)
        for key in (
            profile_payload.get("vector_keys")
            if isinstance(profile_payload.get("vector_keys"), Sequence)
            and not isinstance(profile_payload.get("vector_keys"), (str, bytes, bytearray))
            else []
        )
        if _string(key)
    ]
    legacy_supported_layouts = [
        _string(key)
        for key in (
            profile_payload.get("legacy_supported_layouts")
            if isinstance(profile_payload.get("legacy_supported_layouts"), Sequence)
            and not isinstance(profile_payload.get("legacy_supported_layouts"), (str, bytes, bytearray))
            else []
        )
        if _string(key)
    ]
    return {
        "schema_version": ACTION_LAYOUT_SCHEMA_VERSION,
        "layout_id": layout_id,
        "action_dim": action_dim,
        "absolute": bool(profile_payload.get("absolute", False)),
        "segments": segments,
        "vector_keys": vector_keys,
        "legacy_supported_layouts": legacy_supported_layouts,
        "layout_source": "robot_profile.action_interface.lerobot_export",
        "claim_boundary": profile_payload.get("claim_boundary"),
    }


def _candidate_action_layouts(profile: RobotProfile) -> List[Dict[str, Any]]:
    preferred = _profile_action_layout(profile)
    layouts = [preferred]
    legacy_supported = set(preferred.get("legacy_supported_layouts") or [])
    if _layout_id(preferred) != SC3_ACTION_LAYOUT_ID and SC3_ACTION_LAYOUT_ID in legacy_supported:
        layouts.append(_sc3_action_layout(source="robot_profile.legacy_supported_layouts"))
    return layouts


def _layout_segments(layout: Mapping[str, Any]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            _string(segment.get("name")),
            int(segment.get("start")),
            int(segment.get("end")),
        )
        for segment in _segments_from_layout_payload(layout)
    )


def _manifest_action_layout(layout: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": ACTION_LAYOUT_SCHEMA_VERSION,
        "layout_id": _layout_id(layout),
        "action_dim": _layout_dim(layout),
        "absolute": bool(layout.get("absolute", False)),
        "segments": _segments_from_layout_payload(layout),
        "vector_keys": list(layout.get("vector_keys") or []),
        "layout_source": layout.get("layout_source"),
        "legacy_supported_layouts": list(layout.get("legacy_supported_layouts") or []),
        "claim_boundary": layout.get("claim_boundary"),
    }


# ---------------------------------------------------------------------------
# 1. modality.json from RobotProfile
# ---------------------------------------------------------------------------


def _slices(layout: Sequence[tuple[str, int, int]]) -> Dict[str, Dict[str, int]]:
    return {name: {"start": start, "end": end} for name, start, end in layout}


def build_modality_config(
    profile: RobotProfile,
    *,
    state_layout: Sequence[tuple[str, int, int]] | None = None,
    action_layout: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """GR00T-style modality config: named slices into state/action vectors.

    The action block is emitted from the robot profile's LeRobot export layout
    unless the caller selects a compatibility layout for a specific stream. The
    state block is a declared layout; whether episodes actually carry state is
    reported per-episode by the export.
    """
    resolved_state_layout = tuple(
        state_layout
        if state_layout is not None
        else DEFAULT_STATE_LAYOUTS.get(profile.embodiment_type, ())
    )
    resolved_action_layout = dict(action_layout or _profile_action_layout(profile))
    action_layout_id = _layout_id(resolved_action_layout)
    action_dim = _layout_dim(resolved_action_layout)
    action_segments = _layout_segments(resolved_action_layout)
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
            action_layout_id: {
                "start": 0,
                "end": action_dim,
                "absolute": bool(resolved_action_layout.get("absolute", False)),
                "fields": _slices(action_segments),
                "vector_keys": list(resolved_action_layout.get("vector_keys") or []),
                "layout_source": resolved_action_layout.get("layout_source"),
            }
        },
        "action_dim": action_dim,
        "action_layout_id": action_layout_id,
        "action_layout_schema_version": ACTION_LAYOUT_SCHEMA_VERSION,
        "action_layout": _manifest_action_layout(resolved_action_layout),
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


def _segment_vector_from_payload(
    payload: Mapping[str, Any],
    segment: Mapping[str, Any],
) -> List[float] | None:
    dim = int(segment.get("end") or 0) - int(segment.get("start") or 0)
    if dim <= 0:
        return None
    candidate_keys = [
        _string(key)
        for key in (
            segment.get("source_keys")
            if isinstance(segment.get("source_keys"), Sequence)
            and not isinstance(segment.get("source_keys"), (str, bytes, bytearray))
            else []
        )
        if _string(key)
    ]
    segment_name = _string(segment.get("name"))
    if segment_name and segment_name not in candidate_keys:
        candidate_keys.append(segment_name)
    for key in candidate_keys:
        vector = _float_vector(payload.get(key), dim)
        if vector is not None:
            return vector
    return None


def _segment_vector_from_action(
    action: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> List[float] | None:
    segments = _segments_from_layout_payload(layout)
    if not segments:
        return None
    values: List[float] = []
    cursor = 0
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if start != cursor:
            return None
        segment_values = _segment_vector_from_payload(action, segment)
        if segment_values is None or len(segment_values) != end - start:
            return None
        values.extend(segment_values)
        cursor = end
    return values if len(values) == _layout_dim(layout) else None


def _action_vector_from_action(
    action: Any,
    layout: Mapping[str, Any],
) -> List[float] | None:
    layout_id = _layout_id(layout)
    action_dim = _layout_dim(layout)
    if layout_id == SC3_ACTION_LAYOUT_ID:
        return _sc3_vector_from_action(action)
    if action_dim <= 0:
        return None
    vector = _float_vector(action, action_dim)
    if vector is not None:
        return vector
    payload = _mapping(action)
    if not payload:
        return None
    vector_keys = [
        _string(key)
        for key in (
            layout.get("vector_keys")
            if isinstance(layout.get("vector_keys"), Sequence)
            and not isinstance(layout.get("vector_keys"), (str, bytes, bytearray))
            else []
        )
        if _string(key)
    ]
    for key in [*vector_keys, *GENERIC_ACTION_VECTOR_KEYS]:
        vector = _float_vector(payload.get(key), action_dim)
        if vector is not None:
            return vector
    normalized = _mapping(payload.get("normalized_action"))
    if normalized:
        vector = _action_vector_from_action(normalized, layout)
        if vector is not None:
            return vector
    nested_action = _mapping(payload.get("action"))
    if nested_action:
        vector = _action_vector_from_action(nested_action, layout)
        if vector is not None:
            return vector
    return _segment_vector_from_action(payload, layout)


def _select_action_layout(
    *,
    profile: RobotProfile,
    control_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    layouts = _candidate_action_layouts(profile)
    for layout in layouts:
        for control_row in control_rows:
            if control_row.get("stream_type") != "control_action":
                continue
            payload = _mapping(control_row.get("action"))
            action = control_row.get("action") if not payload else payload
            if _action_vector_from_action(action, layout) is not None:
                return layout
    return layouts[0]


def _invalid_action_blocker(layout: Mapping[str, Any], index: Any) -> str:
    if _layout_id(layout) == SC3_ACTION_LAYOUT_ID:
        return f"sc3_7d_action_invalid_at_index:{index}"
    return f"{_layout_id(layout)}_action_invalid_at_index:{index}"


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


def _video_source_from_mapping(
    value: Any,
    *,
    job_dir: Path,
) -> Dict[str, Any]:
    payload = _mapping(value)
    if not payload and isinstance(value, str):
        payload = {"path": value}
    source_text = _string(
        payload.get("path")
        or payload.get("source_path")
        or payload.get("materialized_path")
        or payload.get("video_path")
    )
    if not source_text:
        return {}
    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = job_dir / source_path
    if not source_path.is_file():
        return {
            **payload,
            "path": str(source_path),
            "present": False,
            "missing_reason": "materialized_video_file_not_found",
        }
    return {
        **payload,
        "path": str(source_path),
        "present": True,
    }


def _observation_source_metadata(
    *,
    attempt: Mapping[str, Any],
    control_row: Mapping[str, Any],
    video_source: Mapping[str, Any],
) -> Dict[str, Any]:
    source_text = _string(
        control_row.get("observation_source")
        or control_row.get("source_kind")
        or attempt.get("observation_source")
        or attempt.get("source_kind")
        or video_source.get("observation_source")
        or video_source.get("source_kind")
    )
    detail = _string(
        control_row.get("observation_source_detail")
        or attempt.get("observation_source_detail")
        or video_source.get("observation_source_detail")
        or video_source.get("source_path")
        or video_source.get("path")
    )
    source_lower = source_text.lower()
    model_derived = (
        _boolish(control_row.get("model_derived"))
        or _boolish(attempt.get("model_derived"))
        or _boolish(video_source.get("model_derived"))
        or source_lower in {"generated", "model_derived", "synthetic"}
    )
    raw_capture = source_lower in {
        "raw_capture",
        "source_capture",
        "physical_capture",
        "physical_robot_capture",
    }
    if model_derived and not source_text:
        source_text = "model_derived"
    elif not source_text:
        source_text = "simulator_trace"
    simulator_trace = (
        not model_derived
        and not raw_capture
        and source_text.lower() in {"simulator_trace", "simulator", "simulation"}
    )
    return {
        "observation_source": source_text,
        "observation_source_detail": detail or None,
        "observation_source_is_model_derived": model_derived,
        "observation_source_is_raw_capture_evidence": bool(
            raw_capture and not model_derived
        ),
        "observation_source_is_simulator_trace": simulator_trace,
    }


def _episode_video_key(modality: Mapping[str, Any] | None) -> str:
    video = _mapping(_mapping(modality).get("video"))
    for camera_id in video:
        text = _string(camera_id)
        if text:
            return f"observation.images.{text}"
    return "observation.images.ego_view"


def _copy_episode_video(
    *,
    source: Mapping[str, Any],
    export_root: Path,
    video_key: str,
    episode_index: int,
) -> Dict[str, Any]:
    if source.get("present") is not True:
        return {}
    source_path = Path(str(source.get("path") or "")).expanduser()
    if not source_path.is_file():
        return {}
    suffix = source_path.suffix.lower() or ".mp4"
    destination = (
        export_root
        / "videos"
        / video_key
        / "chunk-000"
        / f"file-{episode_index:06d}{suffix}"
    )
    ensure_dir(destination.parent)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return {
        "video_key": video_key,
        "path": destination.relative_to(export_root).as_posix(),
        "source_path": str(source_path),
        "clip_id": source.get("clip_id"),
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
    materialized_video_by_attempt: Mapping[str, Any] | None = None,
    accepted_attempt_ids: Sequence[str] | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Map simulator batch streams into per-episode LeRobot-style rows.

    One episode per attempt in the attempt trace. Episodes fail closed: a
    missing control stream blocks the export; an attempt whose control rows
    are missing or whose actions do not parse as the selected robot-profile
    action layout is excluded with a blocker, never padded.
    """
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    export_root = Path(output_dir).expanduser().resolve() / "lerobot_episode_export"
    stamp = generated_at or utc_now_iso()
    profile = robot_profile
    if profile is None and robot_id:
        profile = get_robot_profile(robot_id)

    blockers: List[str] = []
    control_path = resolved_job_dir / CONTROL_STREAM_FILENAME
    trace_path = resolved_job_dir / ATTEMPT_TRACE_FILENAME
    if not control_path.is_file():
        blockers.append("control_stream_missing")
    if not trace_path.is_file():
        blockers.append("attempt_trace_missing")
    if profile is None:
        blockers.append("robot_profile_missing")

    control_rows = _read_jsonl(control_path) if control_path.is_file() else []
    selected_action_layout = (
        _select_action_layout(profile=profile, control_rows=control_rows)
        if profile is not None
        else _sc3_action_layout()
    )
    modality = (
        build_modality_config(profile, action_layout=selected_action_layout)
        if profile
        else None
    )
    state_dim = int(modality["state_dim"]) if modality else 0
    action_dim = _layout_dim(selected_action_layout)
    video_key = _episode_video_key(modality)
    video_sources = {
        _string(attempt_id): _video_source_from_mapping(value, job_dir=resolved_job_dir)
        for attempt_id, value in _mapping(materialized_video_by_attempt).items()
        if _string(attempt_id)
    }
    canonical_attempt_filter = (
        {_string(item) for item in accepted_attempt_ids if _string(item)}
        if accepted_attempt_ids is not None
        else None
    )

    manifest: Dict[str, Any] = {
        "schema_version": LEROBOT_EPISODE_EXPORT_SCHEMA_VERSION,
        "generated_at": stamp,
        "job_dir": str(resolved_job_dir),
        "export_dir": str(export_root),
        "robot_id": profile.robot_id if profile else None,
        "action_layout_id": _layout_id(selected_action_layout),
        "action_dim": action_dim,
        "action_layout": _manifest_action_layout(selected_action_layout),
        "canonical_attempt_filter_applied": canonical_attempt_filter is not None,
        "canonical_accepted_attempt_ids": sorted(canonical_attempt_filter or []),
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

    attempts = _read_jsonl(trace_path)
    actions_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
    for row in control_rows:
        if row.get("stream_type") != "control_action":
            continue
        attempt_id = _string(row.get("attempt_id"))
        if attempt_id and (
            canonical_attempt_filter is None or attempt_id in canonical_attempt_filter
        ):
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
    model_derived_frame_count = 0
    raw_capture_frame_count = 0
    simulator_trace_frame_count = 0
    episode_index = 0
    global_index = 0

    for attempt in attempts:
        attempt_id = _string(attempt.get("attempt_id"))
        if (
            canonical_attempt_filter is not None
            and attempt_id not in canonical_attempt_filter
        ):
            excluded.append(
                {
                    "attempt_id": attempt_id or None,
                    "blockers": ["excluded_by_canonical_training_quality_pipeline"],
                }
            )
            continue
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
        episode_video_source = video_sources.get(attempt_id, {})

        for frame_index, control_row in enumerate(control):
            payload = _mapping(control_row.get("action"))
            stream_payload = dict(control_row)
            stream_payload.update(payload)
            vector = _action_vector_from_action(
                control_row.get("action") if not payload else payload,
                selected_action_layout,
            )
            if vector is None:
                episode_blockers.append(
                    _invalid_action_blocker(
                        selected_action_layout,
                        control_row.get("action_index"),
                    )
                )
                continue
            source = _observation_source_metadata(
                attempt=attempt,
                control_row=control_row,
                video_source=episode_video_source,
            )
            row: Dict[str, Any] = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "index": global_index + frame_index,
                "task_index": task_indices[task_text],
                "task": task_text,
                "action": vector,
                "action_layout_id": _layout_id(selected_action_layout),
                "attempt_id": attempt_id,
                "episode_id": attempt.get("episode_id"),
                "scenario_id": attempt.get("scenario_id"),
                "scenario_eval_run_id": attempt.get("scenario_eval_run_id"),
                "observation_source": source["observation_source"],
                "observation_source_detail": source["observation_source_detail"],
                "observation_source_is_model_derived": source[
                    "observation_source_is_model_derived"
                ],
                "observation_source_is_raw_capture_evidence": source[
                    "observation_source_is_raw_capture_evidence"
                ],
                "observation_source_is_simulator_trace": source[
                    "observation_source_is_simulator_trace"
                ],
            }
            state = _state_from_payload(stream_payload, state_dim)
            if state is not None:
                row["observation.state"] = state
            else:
                state_present = False
            timestamp = _timestamp_from_payload(stream_payload)
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

        video_info = _copy_episode_video(
            source=episode_video_source,
            export_root=export_root,
            video_key=video_key,
            episode_index=episode_index,
        )
        video_present = bool(video_info)
        for row in rows:
            if video_present:
                row[video_key] = video_info["path"]
                row["video_path"] = video_info["path"]

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
        episode_model_derived_frame_count = sum(
            1 for row in rows if row.get("observation_source_is_model_derived") is True
        )
        episode_raw_capture_frame_count = sum(
            1
            for row in rows
            if row.get("observation_source_is_raw_capture_evidence") is True
        )
        episode_simulator_trace_frame_count = sum(
            1
            for row in rows
            if row.get("observation_source_is_simulator_trace") is True
        )
        model_derived_frame_count += episode_model_derived_frame_count
        raw_capture_frame_count += episode_raw_capture_frame_count
        simulator_trace_frame_count += episode_simulator_trace_frame_count
        source_values = sorted(
            {
                _string(row.get("observation_source"))
                for row in rows
                if _string(row.get("observation_source"))
            }
        )
        episodes_meta.append(
            {
                "episode_index": episode_index,
                "attempt_id": attempt_id,
                "length": len(rows),
                "task": task_text,
                "task_index": task_indices[task_text],
                "state_present": state_present,
                "timestamps_present": timestamps_present,
                "video_present": video_present,
                "video_key": video_key if video_present else None,
                "video_path": video_info.get("path") if video_present else None,
                "source_video_path": video_info.get("source_path") if video_present else None,
                "source_clip_id": video_info.get("clip_id") if video_present else None,
                "observation_source": source_values[0]
                if len(source_values) == 1
                else "mixed",
                "observation_source_values": source_values,
                "model_derived_frame_count": episode_model_derived_frame_count,
                "raw_capture_frame_count": episode_raw_capture_frame_count,
                "simulator_trace_frame_count": episode_simulator_trace_frame_count,
                "gr00t_ready": bool(state_present and timestamps_present and video_present),
                "gr00t_ready_missing": [
                    item
                    for item, present in (
                        ("per_step_state", state_present),
                        ("per_step_timestamps", timestamps_present),
                        ("materialized_video", video_present),
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
    materialized_video_count = sum(
        1 for row in episodes_meta if row.get("video_present") is True
    )

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
                "action": {"dtype": "float32", "shape": [action_dim]},
                **(
                    {"observation.state": {"dtype": "float32", "shape": [state_dim]}}
                    if state_dim
                    else {}
                ),
                **(
                    {video_key: {"dtype": "video", "shape": [0, 0, 3]}}
                    if materialized_video_count
                    else {}
                ),
                "observation_source": {"dtype": "string", "shape": [1]},
                "observation_source_detail": {"dtype": "string", "shape": [1]},
                "observation_source_is_model_derived": {
                    "dtype": "bool",
                    "shape": [1],
                },
                "observation_source_is_raw_capture_evidence": {
                    "dtype": "bool",
                    "shape": [1],
                },
                "observation_source_is_simulator_trace": {
                    "dtype": "bool",
                    "shape": [1],
                },
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
            "action_layout_id": _layout_id(selected_action_layout),
            "action_dim": action_dim,
            "action_layout": _manifest_action_layout(selected_action_layout),
            "excluded_episode_count": len(excluded),
            "excluded_episodes": excluded,
            "fps": fps,
            "materialized_video_count": materialized_video_count,
            "observation_source_columns_written": True,
            "model_derived_frame_count": model_derived_frame_count,
            "raw_capture_frame_count": raw_capture_frame_count,
            "simulator_trace_frame_count": simulator_trace_frame_count,
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
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
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
