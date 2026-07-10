"""Canonical, one-to-one temporal alignment for capture sensor streams."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

_TIMESTAMP_FIELDS: dict[str, tuple[str, str]] = {
    "t_device_sec": ("seconds", "capture_start"),
    "tCaptureSec": ("seconds", "capture_start"),
    "timestamp_sec": ("seconds", "capture_start"),
    "timestamp_seconds": ("seconds", "capture_start"),
    "t_device_ms": ("milliseconds", "capture_start"),
    "timestamp_ms": ("milliseconds", "capture_start"),
    "timestamp_epoch_ms": ("milliseconds", "unix_epoch"),
    "timestamp_epoch_sec": ("seconds", "unix_epoch"),
}
_ID_FIELDS = ("frame_id", "frameId", "frame_index", "frameIndex")
_SAFE_TEXT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class _Sample:
    row_index: int
    canonical_id: str | None
    timestamp_sec: float
    timestamp_field: str
    declared_unit: str
    origin: str
    timebase_id: str


def canonical_stream_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return f"frame-{int(text):012d}"
    return text if _SAFE_TEXT_ID.fullmatch(text) else None


def _timestamp_candidates(row: Mapping[str, Any]) -> list[tuple[str, float, str, str]]:
    candidates: list[tuple[str, float, str, str]] = []
    for key, (unit, origin) in _TIMESTAMP_FIELDS.items():
        if row.get(key) is None:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        scale = 0.001 if unit == "milliseconds" else 1.0
        candidates.append((key, value * scale, unit, origin))
    if row.get("timestamp") is not None:
        unit = str(row.get("timestamp_unit") or "").strip().lower()
        origin = str(row.get("timestamp_origin") or "").strip().lower()
        if unit in {"seconds", "milliseconds"} and origin in {
            "capture_start",
            "unix_epoch",
        }:
            try:
                value = float(row["timestamp"])
            except (TypeError, ValueError):
                value = float("nan")
            scale = 0.001 if unit == "milliseconds" else 1.0
            candidates.append(("timestamp", value * scale, unit, origin))
    return candidates


def _parse_stream(
    rows: Sequence[Mapping[str, Any]], *, stream_name: str
) -> tuple[list[_Sample], list[str]]:
    samples: list[_Sample] = []
    blockers: list[str] = []
    seen_ids: set[str] = set()
    seen_timestamps: set[float] = set()
    declared_units: set[str] = set()
    origins: set[str] = set()
    timebase_ids: set[str] = set()
    for index, row in enumerate(rows):
        candidates = _timestamp_candidates(row)
        if not candidates:
            blockers.append(f"{stream_name}:row_{index}:timestamp_missing_or_ambiguous")
            continue
        normalized_values = {round(item[1], 9) for item in candidates if math.isfinite(item[1])}
        if len(candidates) > 1 and len(normalized_values) != 1:
            blockers.append(f"{stream_name}:row_{index}:conflicting_timestamp_fields")
            continue
        field, timestamp_sec, unit, origin = candidates[0]
        if not math.isfinite(timestamp_sec):
            blockers.append(f"{stream_name}:row_{index}:timestamp_nonfinite")
            continue
        declared_units.add(unit)
        origins.add(origin)
        timebase_id = str(row.get("timebase_id") or row.get("clock_id") or "").strip()
        if not timebase_id:
            timebase_id = f"canonical:{origin}"
        timebase_ids.add(timebase_id)
        raw_id = next((row.get(key) for key in _ID_FIELDS if row.get(key) is not None), None)
        canonical_id = canonical_stream_id(raw_id)
        if raw_id is not None and canonical_id is None:
            blockers.append(f"{stream_name}:row_{index}:canonical_id_invalid")
        if canonical_id is not None:
            if canonical_id in seen_ids:
                blockers.append(f"{stream_name}:duplicate_canonical_id:{canonical_id}")
            seen_ids.add(canonical_id)
        rounded_timestamp = round(timestamp_sec, 9)
        if rounded_timestamp in seen_timestamps:
            blockers.append(f"{stream_name}:duplicate_timestamp:{rounded_timestamp}")
        seen_timestamps.add(rounded_timestamp)
        samples.append(
            _Sample(
                row_index=index,
                canonical_id=canonical_id,
                timestamp_sec=timestamp_sec,
                timestamp_field=field,
                declared_unit=unit,
                origin=origin,
                timebase_id=timebase_id,
            )
        )
    if len(declared_units) > 1:
        blockers.append(f"{stream_name}:mixed_timestamp_units")
    if len(origins) > 1:
        blockers.append(f"{stream_name}:mixed_timestamp_origins")
    if len(timebase_ids) > 1:
        blockers.append(f"{stream_name}:mixed_timebase_ids")
    timestamps = [sample.timestamp_sec for sample in samples]
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        blockers.append(f"{stream_name}:timestamps_not_strictly_monotonic")
    return samples, sorted(set(blockers))


def align_frame_pose_streams(
    frame_rows: Sequence[Mapping[str, Any]],
    pose_rows: Sequence[Mapping[str, Any]],
    *,
    max_delta_sec: float = 0.2,
) -> dict[str, Any]:
    if not math.isfinite(max_delta_sec) or max_delta_sec < 0.0:
        raise ValueError("max_delta_sec must be finite and non-negative")
    frames, frame_blockers = _parse_stream(frame_rows, stream_name="frames")
    poses, pose_blockers = _parse_stream(pose_rows, stream_name="poses")
    blockers = [*frame_blockers, *pose_blockers]
    frame_timebases = {(sample.timebase_id, sample.origin) for sample in frames}
    pose_timebases = {(sample.timebase_id, sample.origin) for sample in poses}
    if frame_timebases and pose_timebases and frame_timebases != pose_timebases:
        blockers.append("frame_pose_timebase_or_origin_mismatch")

    pose_by_id = {
        sample.canonical_id: sample
        for sample in poses
        if sample.canonical_id is not None
    }
    used_pose_rows: set[int] = set()
    joins: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for frame in frames:
        selected: _Sample | None = None
        method = ""
        id_candidate = pose_by_id.get(frame.canonical_id) if frame.canonical_id else None
        if id_candidate is not None and id_candidate.row_index not in used_pose_rows:
            delta = abs(frame.timestamp_sec - id_candidate.timestamp_sec)
            if delta <= max_delta_sec:
                selected = id_candidate
                method = "canonical_id_and_delta"
            else:
                drops.append(
                    {
                        "stream": "frames",
                        "row_index": frame.row_index,
                        "canonical_id": frame.canonical_id,
                        "reason": "id_match_delta_exceeded",
                        "delta_sec": delta,
                    }
                )
                continue
        if selected is None:
            available = [pose for pose in poses if pose.row_index not in used_pose_rows]
            if available:
                nearest = min(
                    available,
                    key=lambda pose: (
                        abs(frame.timestamp_sec - pose.timestamp_sec),
                        pose.row_index,
                    ),
                )
                if abs(frame.timestamp_sec - nearest.timestamp_sec) <= max_delta_sec:
                    selected = nearest
                    method = "nearest_time_one_to_one"
        if selected is None:
            drops.append(
                {
                    "stream": "frames",
                    "row_index": frame.row_index,
                    "canonical_id": frame.canonical_id,
                    "reason": "no_pose_within_delta",
                }
            )
            continue
        used_pose_rows.add(selected.row_index)
        joins.append(
            {
                "frame_row_index": frame.row_index,
                "pose_row_index": selected.row_index,
                "frame_canonical_id": frame.canonical_id,
                "pose_canonical_id": selected.canonical_id,
                "frame_timestamp_sec": frame.timestamp_sec,
                "pose_timestamp_sec": selected.timestamp_sec,
                "delta_sec": abs(frame.timestamp_sec - selected.timestamp_sec),
                "method": method,
            }
        )
    for pose in poses:
        if pose.row_index not in used_pose_rows:
            drops.append(
                {
                    "stream": "poses",
                    "row_index": pose.row_index,
                    "canonical_id": pose.canonical_id,
                    "reason": "pose_not_joined",
                }
            )
    deltas = [float(join["delta_sec"]) for join in joins]
    match_rate = len(joins) / len(frames) if frames else 0.0
    metrics = {
        "matched_count": len(joins),
        "frame_count": len(frames),
        "pose_count": len(poses),
        "match_rate": match_rate,
        "delta_p50_sec": float(np.percentile(deltas, 50)) if deltas else None,
        "delta_p95_sec": float(np.percentile(deltas, 95)) if deltas else None,
        "delta_max_sec": max(deltas) if deltas else None,
    }
    return {
        "schema_version": "blueprint.temporal_alignment.v1",
        "status": "verified" if frames and poses and not blockers else "blocked",
        "timebase": {
            "frame_timebases": sorted([list(item) for item in frame_timebases]),
            "pose_timebases": sorted([list(item) for item in pose_timebases]),
            "canonical_unit": "seconds",
        },
        "metrics": metrics,
        "joins": joins,
        "drop_ledger": drops,
        "blockers": sorted(set(blockers)),
    }


__all__ = ["align_frame_pose_streams", "canonical_stream_id"]
