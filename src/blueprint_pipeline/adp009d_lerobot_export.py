"""Export ADP-009D episode receipts as a LeRobot v2.1 dataset.

LeRobot's on-disk layout is the interchange robotics teams actually consume:
openpi fine-tunes from it, GR00T reads the same schema, and every lab tool
that speaks "dataset" speaks this tree.  Exporting it turns an episode run
from "here is our receipt JSON" into "here is a dataset your loader already
reads", with nothing invented in between:

* ``observation.state`` -- the pre-step observed joints plus measured gripper
  width, exactly the step trace's rows.
* ``action`` -- the clipped executed DROID row (seven joint velocities plus
  absolute gripper), not the policy's unexecuted plan.
* per-camera H.264 videos at the true 15 Hz control rate, copied from the
  dataset capture streams when every episode has them.

Receipts that predate step-trace retention cannot be exported honestly --
their per-step record was discarded at run time -- so the export refuses them
by name rather than synthesizing rows.

This is a local analysis-side tool: it is not part of the provider runtime
bundle and may depend on pyarrow.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LEROBOT_CODEBASE_VERSION = "v2.1"
LEROBOT_EXPORT_SCHEMA_VERSION = "adp009d_lerobot_export.v1"

DEFAULT_ROBOT_TYPE = "franka_panda"
CHUNKS_SIZE = 1000

STATE_NAMES = [f"panda_joint{index}" for index in range(1, 8)] + ["gripper_width_m"]
ACTION_NAMES = [
    f"panda_joint{index}_velocity_rad_s" for index in range(1, 8)
] + ["gripper_droid_absolute"]

BLOCKER_NO_RECEIPTS = "lerobot_export_no_receipts"
BLOCKER_STEP_TRACE_MISSING = "lerobot_export_step_trace_missing"
BLOCKER_FPS_INCONSISTENT = "lerobot_export_control_hz_inconsistent"
BLOCKER_CAPTURE_INCONSISTENT = "lerobot_export_capture_inconsistent"
BLOCKER_CAPTURE_MEDIA_ROOT = "lerobot_export_capture_requires_media_root"
BLOCKER_VIDEO_MISSING = "lerobot_export_capture_video_missing"
BLOCKER_TOO_MANY_EPISODES = "lerobot_export_exceeds_single_chunk"
BLOCKER_OUTPUT_EXISTS = "lerobot_export_output_dir_not_empty"
BLOCKER_PROMPT_MISSING = "lerobot_export_prompt_missing"


class LeRobotExportError(ValueError):
    """Fail-closed export contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row)) + "\n" for row in rows), encoding="utf-8"
    )


def _stats(vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def _episode_rows(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = receipt.get("step_trace")
    if not isinstance(trace, Mapping) or not trace.get("rows"):
        raise LeRobotExportError([BLOCKER_STEP_TRACE_MISSING])
    rows = []
    for row in trace["rows"]:
        state = [float(value) for value in row["observation_joint_position_rad"]]
        width = row.get("object_sample", {}).get("gripper_width_m")
        state.append(float(width) if width is not None else 0.0)
        rows.append(
            {
                "state": state,
                "action": [float(value) for value in row["action_droid"]],
                "timestamp": float(row["sim_time_s"]),
            }
        )
    return rows


def _capture_streams(
    receipt: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]] | None:
    capture = receipt.get("dataset_capture")
    if not isinstance(capture, Mapping) or not capture.get("streams"):
        return None
    return dict(capture["streams"])


def _video_feature(stream: Mapping[str, Any], fps: float) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [int(stream["height"]), int(stream["width"]), 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": float(fps),
            "video.height": int(stream["height"]),
            "video.width": int(stream["width"]),
            "video.channels": 3,
            "video.codec": str(stream["video"]["codec"]),
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def export_lerobot_dataset(
    *,
    episode_receipts: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    media_root: str | Path | None = None,
    robot_type: str = DEFAULT_ROBOT_TYPE,
) -> dict[str, Any]:
    """Write one LeRobot v2.1 dataset tree from episode receipts.

    ``media_root`` is the directory the receipts' dataset-capture relative
    paths resolve against; it is required exactly when captures are present.
    Either every episode carries a capture or none does -- a mixed export
    would silently ship a dataset whose video coverage varies by episode.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    receipts = list(episode_receipts)
    if not receipts:
        raise LeRobotExportError([BLOCKER_NO_RECEIPTS])
    if len(receipts) > CHUNKS_SIZE:
        raise LeRobotExportError([f"{BLOCKER_TOO_MANY_EPISODES}:{len(receipts)}"])

    control_rates = set()
    for receipt in receipts:
        trace = receipt.get("step_trace")
        if not isinstance(trace, Mapping) or "control_hz" not in trace:
            raise LeRobotExportError([BLOCKER_STEP_TRACE_MISSING])
        control_rates.add(int(trace["control_hz"]))
    if len(control_rates) != 1:
        raise LeRobotExportError(
            [f"{BLOCKER_FPS_INCONSISTENT}:{sorted(control_rates)}"]
        )
    fps = control_rates.pop()

    captures = [_capture_streams(receipt) for receipt in receipts]
    if any(capture is not None for capture in captures) and not all(
        capture is not None for capture in captures
    ):
        raise LeRobotExportError([BLOCKER_CAPTURE_INCONSISTENT])
    has_video = captures[0] is not None
    if has_video:
        stream_sets = {tuple(sorted(capture)) for capture in captures}
        if len(stream_sets) != 1:
            raise LeRobotExportError([BLOCKER_CAPTURE_INCONSISTENT])
        if media_root is None:
            raise LeRobotExportError([BLOCKER_CAPTURE_MEDIA_ROOT])

    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise LeRobotExportError([f"{BLOCKER_OUTPUT_EXISTS}:{root}"])
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    prompts: list[str] = []
    task_index_by_prompt: dict[str, int] = {}
    for receipt in receipts:
        prompt = str(receipt.get("prompt") or "")
        if not prompt:
            raise LeRobotExportError([BLOCKER_PROMPT_MISSING])
        if prompt not in task_index_by_prompt:
            task_index_by_prompt[prompt] = len(prompts)
            prompts.append(prompt)

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [len(ACTION_NAMES)],
            "names": list(ACTION_NAMES),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    video_streams_exported = 0
    if has_video:
        for stream_id, stream in sorted(captures[0].items()):
            features[f"observation.images.{stream_id}"] = _video_feature(
                stream, float(fps)
            )

    episodes_meta: list[dict[str, Any]] = []
    stats_meta: list[dict[str, Any]] = []
    global_index = 0
    total_frames = 0
    for episode_index, receipt in enumerate(receipts):
        rows = _episode_rows(receipt)
        length = len(rows)
        total_frames += length
        prompt = str(receipt["prompt"])
        task_index = task_index_by_prompt[prompt]

        flat_state = [value for row in rows for value in row["state"]]
        flat_action = [value for row in rows for value in row["action"]]
        state_array = pa.FixedSizeListArray.from_arrays(
            pa.array(flat_state, type=pa.float32()), len(STATE_NAMES)
        )
        action_array = pa.FixedSizeListArray.from_arrays(
            pa.array(flat_action, type=pa.float32()), len(ACTION_NAMES)
        )
        table = pa.table(
            {
                "observation.state": state_array,
                "action": action_array,
                "timestamp": pa.array(
                    [row["timestamp"] for row in rows], type=pa.float32()
                ),
                "frame_index": pa.array(range(length), type=pa.int64()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "index": pa.array(
                    range(global_index, global_index + length), type=pa.int64()
                ),
                "task_index": pa.array([task_index] * length, type=pa.int64()),
                "next.done": pa.array(
                    [step == length - 1 for step in range(length)], type=pa.bool_()
                ),
            }
        )
        pq.write_table(
            table, root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        global_index += length

        if has_video:
            for stream_id, stream in sorted(captures[episode_index].items()):
                source = Path(media_root).expanduser().resolve() / str(
                    stream["video"]["relative_path"]
                )
                if not source.is_file():
                    raise LeRobotExportError(
                        [f"{BLOCKER_VIDEO_MISSING}:{source.name}"]
                    )
                destination = (
                    root
                    / "videos"
                    / "chunk-000"
                    / f"observation.images.{stream_id}"
                    / f"episode_{episode_index:06d}.mp4"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                video_streams_exported += 1

        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [prompt],
                "length": length,
            }
        )
        stats_meta.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": _stats([row["state"] for row in rows]),
                    "action": _stats([row["action"] for row in rows]),
                    "timestamp": _stats([[row["timestamp"]] for row in rows]),
                },
            }
        )

    info: dict[str, Any] = {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": str(robot_type),
        "total_episodes": len(receipts),
        "total_frames": total_frames,
        "total_tasks": len(prompts),
        "total_videos": video_streams_exported,
        "total_chunks": 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": int(fps),
        "splits": {"train": f"0:{len(receipts)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
    }
    if has_video:
        info["video_path"] = (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [
            {"task_index": index, "task": prompt}
            for index, prompt in enumerate(prompts)
        ],
    )
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes_meta)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats_meta)

    return {
        "schema_version": LEROBOT_EXPORT_SCHEMA_VERSION,
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "output_dir": root.as_posix(),
        "episodes_exported": len(receipts),
        "total_frames": total_frames,
        "fps": int(fps),
        "video_streams_exported": video_streams_exported,
        "stats_features": ["observation.state", "action", "timestamp"],
    }


__all__ = [
    "ACTION_NAMES",
    "LEROBOT_CODEBASE_VERSION",
    "LEROBOT_EXPORT_SCHEMA_VERSION",
    "STATE_NAMES",
    "LeRobotExportError",
    "export_lerobot_dataset",
]
