"""Round-trip validation gate for LeRobot-style export directories.

The datasets this pipeline sells (``exports/lerobot_v3``, ``exports/gr00t_lerobot``,
and the per-episode LeRobot export) are written but were never loaded back before
delivery. A buyer's first action is ``LeRobotDataset(path)``; if episode indexing,
fps, video<->parquet frame alignment, or state/action dims are off by one, they
churn on day one. ``validate_lerobot_export`` opens the export the way a consumer
would — with the real ``lerobot`` loader when it is installed, otherwise with a
spec-faithful hermetic reader — and fails closed.

Fail-closed rules:

- anything the reader cannot open or prove (missing meta files, native parquet
  with no pyarrow available and no jsonl mirror, zero frames) is a blocker,
  never a quiet pass;
- per-episode timestamps must exist, be strictly monotonic (no duplicates), and
  match the declared fps; frame indices must be gapless; state/action dims must
  be stable across episodes and match ``meta/info.json``;
- every ``task_index`` referenced by a frame or episode must resolve to a task
  row (referential integrity), and declared per-episode video spans must imply
  exactly the episode's row count;
- a passing report is a loadability claim only — never a data-quality or task
  success claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import utc_now_iso, write_json

LEROBOT_EXPORT_VALIDATION_SCHEMA_VERSION = "lerobot_export_round_trip_validation.v1"

_FPS_RELATIVE_TOLERANCE = 0.02

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "lerobot_export_round_trip_validation",
    "validation_scope_is_loadability_and_alignment_only": True,
    "validation_passed_is_not_data_quality_or_success_claim": True,
    "validation_passed_is_not_deployment_approval": True,
    "hermetic_reader_used_when_lerobot_not_installed": True,
}

CHECK_NAMES: tuple[str, ...] = (
    "dataset_files_present",
    "dataset_rows_readable",
    "lerobot_native_load",
    "timestamps_monotonic_per_episode",
    "fps_consistent",
    "frame_index_sequential",
    "global_index_contiguous",
    "feature_dims_stable",
    "action_space_declared_consistent",
    "tasks_schema_valid",
    "task_index_referential_integrity",
    "episode_metadata_consistent",
    "video_files_present",
    "video_frame_alignment_declared",
    "video_frame_alignment_decoded",
)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
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


def _read_parquet_rows(path: Path) -> tuple[List[Dict[str, Any]] | None, str | None]:
    if importlib.util.find_spec("pyarrow") is None:
        return None, "missing_pyarrow"
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        return [dict(row) for row in pq.read_table(str(path)).to_pylist()], None
    except Exception as exc:  # noqa: BLE001 - any read failure is a blocker
        return None, f"unreadable_{type(exc).__name__}"


_ROW_FILE_SUFFIX_PRIORITY = (".parquet", ".parquet.jsonl", ".jsonl")


def _logical_row_files(directory: Path) -> Dict[str, Dict[str, Path]]:
    """Group row files by logical name so parquet + jsonl mirrors read once."""
    logical: Dict[str, Dict[str, Path]] = {}
    for candidate in sorted(directory.rglob("*")):
        if not candidate.is_file():
            continue
        name = candidate.name
        suffix = next(
            (item for item in _ROW_FILE_SUFFIX_PRIORITY if name.endswith(item)), None
        )
        if suffix is None:
            continue
        stem = name[: -len(suffix)]
        key = str(candidate.parent.relative_to(directory) / stem)
        logical.setdefault(key, {})[suffix] = candidate
    return logical


def _read_logical_rows(
    variants: Mapping[str, Path],
) -> tuple[List[Dict[str, Any]] | None, bool, str | None]:
    """Read one logical row file: native parquet first, jsonl mirrors after.

    Returns (rows, native_parquet_read, blocker_suffix).
    """
    parquet_error: str | None = None
    parquet_path = variants.get(".parquet")
    if parquet_path is not None:
        rows, parquet_error = _read_parquet_rows(parquet_path)
        if rows is not None:
            return rows, True, None
    for suffix in (".parquet.jsonl", ".jsonl"):
        mirror = variants.get(suffix)
        if mirror is not None:
            return _read_jsonl_rows(mirror), False, None
    return None, False, parquet_error or "row_file_missing"


def _task_rows_from_payload(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        name = row.get("task")
        if name is None:
            # pandas writes the task name as the frame index; pyarrow surfaces it
            # as __index_level_0__.
            name = row.get("__index_level_0__") or row.get("name")
        out.append({"task_index": row.get("task_index"), "task": name})
    return out


def _decoded_video_frame_count(path: Path) -> tuple[int | None, str | None]:
    if importlib.util.find_spec("cv2") is None:
        return None, "missing_cv2"
    try:
        import cv2  # type: ignore[import-not-found]

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return None, "video_open_failed"
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            return (count, None) if count > 0 else (None, "video_frame_count_missing")
        finally:
            capture.release()
    except Exception as exc:  # noqa: BLE001 - undecodable media must fail closed
        return None, f"video_decode_failed:{type(exc).__name__}"


def _find_video_file(
    videos_dir: Path,
    video_key: str,
    chunk_index: int,
    file_index: int,
    *,
    layout: str,
) -> Path | None:
    chunk_patterns = (f"chunk-{chunk_index:03d}", f"chunk-{chunk_index:06d}")
    file_patterns = (
        f"file-{file_index:03d}",
        f"file-{file_index:06d}",
        f"episode_{file_index:06d}",
        f"episode_{file_index:03d}",
    )
    candidate_dirs: list[Path] = []
    for chunk_pattern in chunk_patterns:
        if layout == "gr00t_lerobot":
            candidate_dirs.append(videos_dir / chunk_pattern / video_key)
        candidate_dirs.append(videos_dir / video_key / chunk_pattern)
    for candidate_dir in candidate_dirs:
        if not candidate_dir.is_dir():
            continue
        for file_pattern in file_patterns:
            matches = sorted(candidate_dir.glob(f"{file_pattern}.*"))
            if matches:
                return matches[0]
    return None


def _modality_action_dim(modality: Mapping[str, Any]) -> int | None:
    """Declared action dim from a GR00T ``modality.json``.

    Reads the top-level ``action_dim`` when present, otherwise the widest
    ``end`` across the declared action blocks. Returns ``None`` when the
    modality does not declare an action space (older exports) so the check
    is skipped rather than failing closed on absence.
    """
    dim = _int_or_none(modality.get("action_dim"))
    if dim is not None:
        return dim
    ends: List[int] = []
    for block in _mapping(modality.get("action")).values():
        end = _int_or_none(_mapping(block).get("end"))
        if end is not None:
            ends.append(end)
    return max(ends) if ends else None


def _gr00t_modality_video_keys(modality: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for channel, config in _mapping(modality.get("video")).items():
        original_key = str(_mapping(config).get("original_key") or "").strip()
        if original_key:
            keys.add(original_key)
        elif isinstance(channel, str) and channel.strip():
            keys.add(channel.strip())
    return keys


def _try_lerobot_native_load(export_dir: Path) -> str | None:
    """Open the export with the real lerobot loader. Returns a blocker or None."""
    try:
        from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-not-found]
            LeRobotDataset,
        )
    except Exception as exc:  # noqa: BLE001 - a broken install must not pass silently
        return f"lerobot_loader_failed:import_{type(exc).__name__}"
    try:
        dataset = LeRobotDataset(
            repo_id="blueprint/round-trip-validation",
            root=export_dir,
            download_videos=False,
        )
        len(dataset)
    except Exception as exc:  # noqa: BLE001 - loader rejection is the finding
        return f"lerobot_loader_failed:{type(exc).__name__}"
    return None


def validate_lerobot_export(
    export_dir: str | Path,
    *,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Round-trip validate a LeRobot-style export directory. Fails closed."""
    root = Path(export_dir).expanduser()
    stamp = generated_at or utc_now_iso()
    blockers: List[str] = []
    checks: Dict[str, str] = {name: "skipped" for name in CHECK_NAMES}
    counts: Dict[str, int] = {
        "episode_count": 0,
        "frame_count": 0,
        "task_count": 0,
        "video_file_count": 0,
    }
    layout = "unknown"
    loader = "unavailable"
    native_parquet_read = False

    def _report() -> Dict[str, Any]:
        return {
            "schema_version": LEROBOT_EXPORT_VALIDATION_SCHEMA_VERSION,
            "generated_at": stamp,
            "export_dir": str(root),
            "layout": layout,
            "loader": loader,
            "native_parquet_read": native_parquet_read,
            "status": "blocked" if blockers else "passed",
            "blockers": sorted(set(blockers)),
            "checks": checks,
            "counts": counts,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }

    if not root.is_dir():
        blockers.append("export_dir_missing")
        checks["dataset_files_present"] = "failed"
        return _report()

    meta_dir = root / "meta"
    data_dir = root / "data"
    info_path = meta_dir / "info.json"
    if (meta_dir / "episodes.jsonl").is_file():
        layout = "gr00t_lerobot"
    elif (meta_dir / "episodes").is_dir():
        layout = "lerobot_v3"

    info: Dict[str, Any] = {}
    if not info_path.is_file():
        blockers.append("info_json_missing")
    else:
        try:
            info = _mapping(json.loads(info_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            blockers.append("info_json_invalid")
    modality: Dict[str, Any] = {}
    modality_path = meta_dir / "modality.json"
    if layout == "gr00t_lerobot":
        if not modality_path.is_file():
            blockers.append("gr00t_modality_json_missing")
        else:
            try:
                modality = _mapping(json.loads(modality_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                blockers.append("gr00t_modality_json_invalid")

    # --- data rows -----------------------------------------------------------
    frame_rows: List[Dict[str, Any]] = []
    data_files = _logical_row_files(data_dir) if data_dir.is_dir() else {}
    if not data_files:
        blockers.append("data_files_missing")
    parquet_files_read = 0
    for key in sorted(data_files):
        rows, native, error = _read_logical_rows(data_files[key])
        if rows is None:
            prefix = (
                "parquet_unreadable_missing_pyarrow"
                if error == "missing_pyarrow"
                else "data_file_unreadable"
            )
            blockers.append(f"{prefix}:{key}")
            continue
        parquet_files_read += 1 if native else 0
        frame_rows.extend(rows)
    native_parquet_read = bool(data_files) and parquet_files_read == len(data_files)

    # --- episodes metadata ---------------------------------------------------
    episode_meta_rows: List[Dict[str, Any]] | None = None
    if layout == "gr00t_lerobot":
        episode_meta_rows = _read_jsonl_rows(meta_dir / "episodes.jsonl")
    elif layout == "lerobot_v3":
        episode_meta_rows = []
        for key in sorted(episode_files := _logical_row_files(meta_dir / "episodes")):
            rows, _, error = _read_logical_rows(episode_files[key])
            if rows is None:
                prefix = (
                    "parquet_unreadable_missing_pyarrow"
                    if error == "missing_pyarrow"
                    else "episodes_metadata_unreadable"
                )
                blockers.append(f"{prefix}:meta/episodes/{key}")
                continue
            episode_meta_rows.extend(rows)
    if episode_meta_rows is None:
        blockers.append("episodes_metadata_missing")
        episode_meta_rows = []

    # --- tasks ---------------------------------------------------------------
    tasks: List[Dict[str, Any]] | None = None
    tasks_parquet = meta_dir / "tasks.parquet"
    if tasks_parquet.is_file():
        rows, error = _read_parquet_rows(tasks_parquet)
        if rows is not None:
            tasks = _task_rows_from_payload(rows)
    if tasks is None:
        for name in ("tasks.parquet.jsonl", "tasks.jsonl"):
            candidate = meta_dir / name
            if candidate.is_file():
                tasks = _task_rows_from_payload(_read_jsonl_rows(candidate))
                break
    if tasks is None:
        if tasks_parquet.is_file():
            blockers.append("parquet_unreadable_missing_pyarrow:meta/tasks.parquet")
        else:
            blockers.append("tasks_file_missing")
        tasks = []

    checks["dataset_files_present"] = "failed" if blockers else "passed"
    checks["dataset_rows_readable"] = (
        "failed"
        if any("unreadable" in blocker for blocker in blockers)
        else ("passed" if data_files else "skipped")
    )
    if blockers:
        # Unreadable or structurally missing exports: nothing below is provable.
        return _report()

    if not frame_rows:
        blockers.append("dataset_empty_no_frames")
        return _report()

    counts["frame_count"] = len(frame_rows)
    counts["task_count"] = len(tasks)

    # --- tasks schema + referential integrity --------------------------------
    task_indices: set[int] = set()
    tasks_schema_ok = True
    for task in tasks:
        task_index = _int_or_none(task.get("task_index"))
        name = str(task.get("task") or "").strip()
        if task_index is None or task_index < 0:
            blockers.append("tasks_schema_invalid:non_integer_task_index")
            tasks_schema_ok = False
            continue
        if not name:
            blockers.append(f"tasks_schema_invalid:empty_task_name:{task_index}")
            tasks_schema_ok = False
        if task_index in task_indices:
            blockers.append(f"tasks_schema_invalid:duplicate_task_index:{task_index}")
            tasks_schema_ok = False
        task_indices.add(task_index)
    if not tasks:
        blockers.append("tasks_schema_invalid:no_tasks")
        tasks_schema_ok = False
    checks["tasks_schema_valid"] = "passed" if tasks_schema_ok else "failed"

    # --- group frames by episode ----------------------------------------------
    rows_by_episode: Dict[int, List[Dict[str, Any]]] = {}
    for row in frame_rows:
        episode_index = _int_or_none(row.get("episode_index"))
        if episode_index is None:
            blockers.append("frame_row_missing_episode_index")
            continue
        rows_by_episode.setdefault(episode_index, []).append(row)
    counts["episode_count"] = len(rows_by_episode)

    declared_fps = _finite_float(info.get("fps"))
    if declared_fps is not None and declared_fps <= 0:
        declared_fps = None
    if layout == "lerobot_v3" and declared_fps is None:
        blockers.append("fps_missing_in_info")

    timestamps_ok = True
    fps_ok = True
    fps_checked = False
    frame_index_ok = True
    referential_ok = True
    estimated_fps: float | None = None

    for episode_index in sorted(rows_by_episode):
        episode_rows = sorted(
            rows_by_episode[episode_index],
            key=lambda row: (_int_or_none(row.get("frame_index")) or 0),
        )

        frame_indices = [_int_or_none(row.get("frame_index")) for row in episode_rows]
        if frame_indices != list(range(len(episode_rows))):
            blockers.append(f"frame_index_not_sequential:episode_{episode_index}")
            frame_index_ok = False

        timestamps = [_finite_float(row.get("timestamp")) for row in episode_rows]
        if any(timestamp is None for timestamp in timestamps):
            blockers.append(f"timestamps_missing:episode_{episode_index}")
            timestamps_ok = False
        else:
            values = [float(timestamp) for timestamp in timestamps if timestamp is not None]
            if any(later <= earlier for earlier, later in zip(values, values[1:])):
                blockers.append(f"timestamps_not_monotonic:episode_{episode_index}")
                timestamps_ok = False
            deltas = [later - earlier for earlier, later in zip(values, values[1:])]
            if deltas:
                expected_delta = (
                    1.0 / declared_fps if declared_fps is not None else deltas[0]
                )
                fps_checked = True
                tolerance = abs(expected_delta) * _FPS_RELATIVE_TOLERANCE + 1e-9
                if any(abs(delta - expected_delta) > tolerance for delta in deltas):
                    blockers.append(f"fps_inconsistent:episode_{episode_index}")
                    fps_ok = False
                elif estimated_fps is None and expected_delta > 0:
                    estimated_fps = 1.0 / expected_delta

        episode_task_refs = {
            _int_or_none(row.get("task_index"))
            for row in episode_rows
            if row.get("task_index") is not None
        }
        if any(
            reference is None or reference not in task_indices
            for reference in episode_task_refs
        ):
            blockers.append(f"task_index_dangling:episode_{episode_index}")
            referential_ok = False

    checks["timestamps_monotonic_per_episode"] = (
        "passed" if timestamps_ok else "failed"
    )
    checks["fps_consistent"] = (
        ("passed" if fps_ok else "failed") if fps_checked else "skipped"
    )
    checks["frame_index_sequential"] = "passed" if frame_index_ok else "failed"

    # --- global index contiguity ----------------------------------------------
    global_indices = [_int_or_none(row.get("index")) for row in frame_rows]
    if all(index is not None for index in global_indices):
        if set(global_indices) != set(range(len(frame_rows))):
            blockers.append("global_index_not_contiguous")
            checks["global_index_contiguous"] = "failed"
        else:
            checks["global_index_contiguous"] = "passed"

    # --- feature dims -----------------------------------------------------------
    features = _mapping(info.get("features"))
    dims_ok = True
    for field, unstable_blocker, mismatch_blocker in (
        ("action", "action_dim_unstable", "action_dim_mismatch_with_info"),
        (
            "observation.state",
            "state_dim_unstable",
            "state_dim_mismatch_with_info",
        ),
    ):
        lengths = {
            len(row[field])
            for row in frame_rows
            if isinstance(row.get(field), (list, tuple))
        }
        if not lengths:
            continue
        if len(lengths) > 1:
            blockers.append(unstable_blocker)
            dims_ok = False
            continue
        declared_shape = _mapping(features.get(field)).get("shape")
        if (
            isinstance(declared_shape, (list, tuple))
            and len(declared_shape) == 1
            and _int_or_none(declared_shape[0]) is not None
            and _int_or_none(declared_shape[0]) != next(iter(lengths))
        ):
            blockers.append(mismatch_blocker)
            dims_ok = False
    checks["feature_dims_stable"] = "passed" if dims_ok else "failed"

    # --- declared action-space consistency (GR00T modality) -------------------
    # R080: the export declares its action space (single_arm_7d | bimanual_14d |
    # whole_body | mobile_base_arm) in modality.json. When present, the declared
    # dim must agree with info.json's action feature shape and the actual row
    # width so a mislabeled action contract fails closed instead of shipping.
    modality_action_dim = (
        _modality_action_dim(modality) if layout == "gr00t_lerobot" else None
    )
    if modality_action_dim is not None:
        action_row_dims = {
            len(row["action"])
            for row in frame_rows
            if isinstance(row.get("action"), (list, tuple))
        }
        info_action_shape = _mapping(features.get("action")).get("shape")
        declared_info_dim = None
        if (
            isinstance(info_action_shape, (list, tuple))
            and len(info_action_shape) == 1
        ):
            declared_info_dim = _int_or_none(info_action_shape[0])
        action_space_ok = True
        if action_row_dims and modality_action_dim not in action_row_dims:
            blockers.append("action_space_dim_mismatch_rows")
            action_space_ok = False
        if declared_info_dim is not None and declared_info_dim != modality_action_dim:
            blockers.append("action_space_dim_mismatch_info")
            action_space_ok = False
        checks["action_space_declared_consistent"] = (
            "passed" if action_space_ok else "failed"
        )

    # --- episode metadata consistency ------------------------------------------
    episodes_ok = True
    meta_by_index: Dict[int, Dict[str, Any]] = {}
    for meta_row in episode_meta_rows:
        episode_index = _int_or_none(meta_row.get("episode_index"))
        if episode_index is None:
            blockers.append("episode_metadata_missing_episode_index")
            episodes_ok = False
            continue
        meta_by_index[episode_index] = meta_row
    if set(meta_by_index) != set(rows_by_episode):
        blockers.append("episode_metadata_index_mismatch")
        episodes_ok = False
    for episode_index, meta_row in sorted(meta_by_index.items()):
        row_count = len(rows_by_episode.get(episode_index, []))
        declared_length = _int_or_none(meta_row.get("length"))
        if declared_length is not None and declared_length != row_count:
            blockers.append(f"episode_length_mismatch:episode_{episode_index}")
            episodes_ok = False
        task_refs = meta_row.get("tasks")
        if task_refs is None and meta_row.get("task_index") is not None:
            task_refs = [meta_row.get("task_index")]
        for reference in task_refs or []:
            if _int_or_none(reference) not in task_indices:
                blockers.append(f"task_index_dangling:episode_{episode_index}")
                referential_ok = False
    checks["episode_metadata_consistent"] = "passed" if episodes_ok else "failed"
    checks["task_index_referential_integrity"] = (
        "passed" if referential_ok else "failed"
    )

    # --- video alignment ---------------------------------------------------------
    declared_video_keys = sorted(
        key
        for key, feature in features.items()
        if _mapping(feature).get("dtype") == "video"
    )
    if layout == "gr00t_lerobot":
        declared_video_keys = sorted(
            set(declared_video_keys) | _gr00t_modality_video_keys(modality)
        )
    videos_dir = root / "videos"
    counts["video_file_count"] = (
        sum(1 for item in videos_dir.rglob("*") if item.is_file())
        if videos_dir.is_dir()
        else 0
    )
    alignment_fps = declared_fps if declared_fps is not None else estimated_fps
    videos_present_ok = True
    videos_present_checked = False
    declared_alignment_ok = True
    declared_alignment_checked = False
    decoded_alignment_ok = True
    decoded_alignment_checked = False
    for episode_index, meta_row in sorted(meta_by_index.items()):
        row_count = len(rows_by_episode.get(episode_index, []))
        meta_video_keys = {
            key.split("/", 2)[1]
            for key in meta_row
            if isinstance(key, str) and key.startswith("videos/") and key.count("/") >= 2
        }
        if layout == "gr00t_lerobot":
            declared_video_keys = sorted(set(declared_video_keys) | meta_video_keys)
        for video_key in sorted(set(declared_video_keys) | meta_video_keys):
            from_ts = _finite_float(meta_row.get(f"videos/{video_key}/from_timestamp"))
            to_ts = _finite_float(meta_row.get(f"videos/{video_key}/to_timestamp"))
            if from_ts is not None and to_ts is not None and alignment_fps:
                declared_alignment_checked = True
                declared_frames = round((to_ts - from_ts) * alignment_fps)
                if declared_frames != row_count:
                    blockers.append(
                        f"video_frame_count_mismatch:episode_{episode_index}"
                    )
                    declared_alignment_ok = False
            if video_key not in declared_video_keys:
                continue
            videos_present_checked = True
            chunk_index = _int_or_none(
                meta_row.get(f"videos/{video_key}/chunk_index")
            ) or 0
            file_index = _int_or_none(meta_row.get(f"videos/{video_key}/file_index"))
            if file_index is None:
                file_index = episode_index
            video_path = _find_video_file(
                videos_dir,
                video_key,
                chunk_index,
                file_index,
                layout=layout,
            )
            if video_path is None:
                blockers.append(f"video_file_missing:episode_{episode_index}")
                videos_present_ok = False
                continue
            decoded, decode_error = _decoded_video_frame_count(video_path)
            if decoded is not None:
                decoded_alignment_checked = True
                if decoded != row_count:
                    blockers.append(
                        f"video_frame_count_mismatch_decoded:episode_{episode_index}"
                    )
                    decoded_alignment_ok = False
            elif decode_error != "missing_cv2":
                decoded_alignment_checked = True
                blockers.append(f"video_file_undecodable:episode_{episode_index}")
                decoded_alignment_ok = False
    if videos_present_checked:
        checks["video_files_present"] = "passed" if videos_present_ok else "failed"
    if declared_alignment_checked:
        checks["video_frame_alignment_declared"] = (
            "passed" if declared_alignment_ok else "failed"
        )
    if decoded_alignment_checked:
        checks["video_frame_alignment_decoded"] = (
            "passed" if decoded_alignment_ok else "failed"
        )

    # --- real loader, when available ------------------------------------------
    loader = "hermetic_parquet" if native_parquet_read else "hermetic_jsonl_fallback"
    if layout == "lerobot_v3" and importlib.util.find_spec("lerobot") is not None:
        loader_blocker = _try_lerobot_native_load(root)
        loader = "lerobot_native+hermetic"
        if loader_blocker:
            blockers.append(loader_blocker)
            checks["lerobot_native_load"] = "failed"
        else:
            checks["lerobot_native_load"] = "passed"

    return _report()


def round_trip_validation_summary(
    report: Mapping[str, Any], *, path: str | None = None
) -> Dict[str, Any]:
    """Compact verdict for embedding in export manifests and buyer readouts."""
    payload = _mapping(report)
    return {
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "blockers": list(payload.get("blockers") or []),
        "loader": payload.get("loader"),
        "layout": payload.get("layout"),
        "checks": _mapping(payload.get("checks")),
        "counts": _mapping(payload.get("counts")),
        "path": path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="write round_trip_validation_report.json inside the export dir",
    )
    args = parser.parse_args(argv)
    report = validate_lerobot_export(args.export_dir)
    if args.write_report:
        write_json(
            Path(args.export_dir) / "round_trip_validation_report.json", report
        )
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
