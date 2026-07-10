"""Per-dimension action normalization and validation for SC3-style eval.

SC3-Eval (arXiv 2606.18610) requires 7-dimensional delta end-effector actions
normalized per-dimension across the training corpus, with per-chunk temporal
alignment between actions and frames. This module computes and persists those
statistics and validates action streams before they can feed evaluation or
training export.

Doctrine: raw action streams are capture truth and are never normalized in
place — normalized copies live alongside the raw stream, and records that fail
validation are rejected into a manifest, never silently zero-filled.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import ensure_dir, utc_now_iso, write_json

ACTION_NORMALIZATION_SCHEMA_VERSION = "action_normalization.v2"

# SC3-Eval action layout: delta translation (3), delta rotation (3), gripper (1).
DEFAULT_ACTION_DIM = 7
SC3_ACTION_REPRESENTATION = "7d_delta_end_effector_pose"
SC3_ACTION_ORDER = (
    "delta_x_m",
    "delta_y_m",
    "delta_z_m",
    "delta_roll_rad",
    "delta_pitch_rad",
    "delta_yaw_rad",
    "gripper_normalized",
)
SC3_ACTION_UNITS = ("m", "m", "m", "rad", "rad", "rad", "normalized")
SC3_ACTION_REPRESENTATION_ALIASES = {
    SC3_ACTION_REPRESENTATION,
    "sc3_7d_delta_end_effector_pose",
    "ee_delta_pose_gripper",
}
MIN_NORMALIZATION_STD = 1e-12


@dataclass(frozen=True)
class ActionValidationConfig:
    """Physical sanity bounds for delta end-effector action streams.

    Defaults are deliberately generous upper bounds for tabletop
    manipulation at 10 Hz (SC3-Eval's control rate); tighten per robot
    profile via config.
    """

    expected_dim: int = DEFAULT_ACTION_DIM
    max_abs_translation_delta_m: float = 0.30
    max_abs_rotation_delta_rad: float = 1.0
    gripper_min: float = -1.5
    gripper_max: float = 1.5
    chunk_alignment_tolerance_sec: float = 0.05

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionValidationConfig":
        known = {f: payload[f] for f in cls.__dataclass_fields__ if f in payload}
        return cls(**known)


@dataclass
class ActionValidationResult:
    valid: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "reasons": list(self.reasons)}


def _as_float_rows(actions: Sequence[Sequence[Any]]) -> Optional[List[List[float]]]:
    rows: List[List[float]] = []
    for row in actions:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return None
        try:
            values = [float(v) for v in row]
        except (TypeError, ValueError):
            return None
        rows.append(values)
    return rows


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_action_stream(
    actions: Sequence[Sequence[Any]],
    *,
    config: ActionValidationConfig,
) -> ActionValidationResult:
    """Validate one episode's action stream against the declared contract.

    Checks dimensionality, finiteness, translation/rotation delta bounds, and
    gripper range. Returns typed reasons; never mutates or repairs the stream.
    """
    reasons: List[str] = []
    rows = _as_float_rows(actions)
    if rows is None or not rows:
        return ActionValidationResult(False, ["action_stream_missing_or_non_numeric"])

    for index, row in enumerate(rows):
        if len(row) != config.expected_dim:
            reasons.append(f"action_dim_mismatch:step_{index}:got_{len(row)}_want_{config.expected_dim}")
            continue
        if any(v != v or v in (float("inf"), float("-inf")) for v in row):
            reasons.append(f"action_non_finite:step_{index}")
            continue
        translation = row[0:3]
        rotation = row[3:6]
        gripper = row[6] if len(row) > 6 else None
        if any(abs(v) > config.max_abs_translation_delta_m for v in translation):
            reasons.append(f"translation_delta_out_of_bounds:step_{index}")
        if any(abs(v) > config.max_abs_rotation_delta_rad for v in rotation):
            reasons.append(f"rotation_delta_out_of_bounds:step_{index}")
        if gripper is not None and not (config.gripper_min <= gripper <= config.gripper_max):
            reasons.append(f"gripper_out_of_bounds:step_{index}")

    return ActionValidationResult(not reasons, reasons)


def validate_chunk_alignment(
    *,
    chunk_start_times_sec: Sequence[float],
    frame_times_sec: Sequence[float],
    config: ActionValidationConfig,
) -> ActionValidationResult:
    """Verify each action chunk start aligns with an observed frame timestamp."""
    if not chunk_start_times_sec:
        return ActionValidationResult(False, ["chunk_timestamps_missing"])
    if not frame_times_sec:
        return ActionValidationResult(False, ["frame_timestamps_missing"])
    try:
        chunks = [float(t) for t in chunk_start_times_sec]
        frames = sorted(float(t) for t in frame_times_sec)
    except (TypeError, ValueError):
        return ActionValidationResult(False, ["action_or_frame_timestamps_non_numeric"])
    if any(not math.isfinite(value) for value in [*chunks, *frames]):
        return ActionValidationResult(False, ["action_or_frame_timestamps_non_finite"])
    reasons: List[str] = []
    for index, start_f in enumerate(chunks):
        nearest = min(frames, key=lambda t: abs(t - start_f))
        if abs(nearest - start_f) > config.chunk_alignment_tolerance_sec:
            reasons.append(
                f"chunk_frame_misaligned:chunk_{index}:delta_{abs(nearest - start_f):.4f}s"
            )
    return ActionValidationResult(not reasons, reasons)


def compute_normalization_stats(
    episodes: Mapping[str, Sequence[Sequence[Any]]],
    *,
    expected_dim: int = DEFAULT_ACTION_DIM,
) -> Optional[Dict[str, Any]]:
    """Compute per-dimension mean/std/min/max across the accepted corpus.

    Returns None when no valid rows exist — stats are never fabricated.
    """
    columns: List[List[float]] = [[] for _ in range(expected_dim)]
    episode_count = 0
    for _episode_id, actions in episodes.items():
        rows = _as_float_rows(actions)
        if not rows:
            continue
        used = False
        for row in rows:
            if len(row) != expected_dim:
                continue
            for dim, value in enumerate(row):
                columns[dim].append(value)
            used = True
        if used:
            episode_count += 1
    if not columns[0]:
        return None
    stats: List[Dict[str, float]] = []
    for values in columns:
        count = len(values)
        mean = sum(values) / count
        variance = sum((v - mean) ** 2 for v in values) / count
        stats.append(
            {
                "mean": mean,
                "std": variance ** 0.5,
                "min": min(values),
                "max": max(values),
                "count": count,
            }
        )
    return {
        "schema_version": ACTION_NORMALIZATION_SCHEMA_VERSION,
        "expected_dim": expected_dim,
        "episode_count": episode_count,
        "per_dimension": stats,
    }


def normalize_actions(
    actions: Sequence[Sequence[Any]],
    *,
    stats: Mapping[str, Any],
) -> List[List[float]]:
    """Return a normalized copy, failing on incomplete or zero-variance stats."""
    per_dim = list(stats.get("per_dimension") or [])
    expected_dim = int(stats.get("expected_dim") or 0)
    rows = _as_float_rows(actions)
    if rows is None or not rows:
        raise ValueError("action_stream_missing_or_non_numeric")
    if expected_dim != DEFAULT_ACTION_DIM or len(per_dim) != expected_dim:
        raise ValueError("normalization_stats_dimension_contract_invalid")
    normalized: List[List[float]] = []
    for row in rows:
        if len(row) != expected_dim:
            raise ValueError("action_dim_mismatch_for_normalization")
        out: List[float] = []
        for dim, value in enumerate(row):
            dim_stats = per_dim[dim]
            mean = float(dim_stats["mean"])
            std = float(dim_stats["std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std <= MIN_NORMALIZATION_STD:
                raise ValueError(f"normalization_std_invalid:dimension_{dim}")
            out.append((value - mean) / std)
        normalized.append(out)
    return normalized


def build_action_normalization_manifest(
    *,
    output_dir: str | Path,
    episodes: Mapping[str, Mapping[str, Any]],
    action_space: Mapping[str, Any] | None = None,
    corpus_provenance: Mapping[str, Any] | None = None,
    config: ActionValidationConfig | None = None,
) -> Dict[str, Any]:
    """Validate a corpus of episodes and persist normalization statistics.

    ``episodes`` maps episode_id -> {"actions": [[...7 floats...], ...],
    "chunk_start_times_sec": [...], "frame_times_sec": [...]} (timestamp
    fields required, along with ``control_rate_hz``. ``action_space`` must
    declare the exact 7-D order and units. ``corpus_provenance`` binds the
    statistics to the exact trace consumed by the evaluator.

    Writes ``action_norm_stats.json`` and ``action_validation_manifest.json``
    into ``output_dir`` and returns the manifest. Status is ``validated``
    only when every episode passes, provenance is complete, and every
    normalization dimension has non-zero variance; otherwise it fails closed.
    """
    out_root = Path(output_dir)
    ensure_dir(out_root)
    action_contract = dict(action_space or {})
    try:
        declared_dim = int(action_contract.get("dim") or 0)
    except (TypeError, ValueError):
        declared_dim = 0
    cfg = config or ActionValidationConfig(expected_dim=declared_dim)
    if cfg.expected_dim != declared_dim:
        cfg = ActionValidationConfig(**{**cfg.__dict__, "expected_dim": declared_dim})

    contract_blockers: List[str] = []
    representation = str(
        action_contract.get("representation")
        or action_contract.get("name")
        or action_contract.get("layout_id")
        or ""
    ).strip()
    declared_order = tuple(action_contract.get("order") or [])
    declared_units = tuple(action_contract.get("units") or [])
    if not action_contract:
        contract_blockers.append("action_space_contract_missing")
    if declared_dim != DEFAULT_ACTION_DIM:
        contract_blockers.append("action_space_dim_must_equal_7")
    if representation not in SC3_ACTION_REPRESENTATION_ALIASES:
        contract_blockers.append("action_representation_not_sc3_7d_delta_end_effector")
    if declared_order != SC3_ACTION_ORDER:
        contract_blockers.append("action_dimension_order_missing_or_invalid")
    if declared_units != SC3_ACTION_UNITS:
        contract_blockers.append("action_dimension_units_missing_or_invalid")

    provenance = dict(corpus_provenance or {})
    provenance_required = (
        "source_trace_path",
        "source_trace_sha256",
        "trace_schema_version",
        "consumed_by",
    )
    for field_name in provenance_required:
        if not str(provenance.get(field_name) or "").strip():
            contract_blockers.append(f"corpus_provenance_{field_name}_missing")
    trace_sha = str(provenance.get("source_trace_sha256") or "").strip().lower()
    if trace_sha and (len(trace_sha) != 64 or any(char not in "0123456789abcdef" for char in trace_sha)):
        contract_blockers.append("corpus_provenance_source_trace_sha256_invalid")

    accepted: Dict[str, Sequence[Sequence[Any]]] = {}
    results: Dict[str, Dict[str, Any]] = {}
    for episode_id, payload in episodes.items():
        actions = payload.get("actions") if isinstance(payload, Mapping) else None
        stream_result = validate_action_stream(list(actions or []), config=cfg)
        reasons = list(stream_result.reasons)
        chunk_times = (
            list(payload.get("chunk_start_times_sec") or [])
            if isinstance(payload, Mapping)
            else []
        )
        frame_times = (
            list(payload.get("frame_times_sec") or [])
            if isinstance(payload, Mapping)
            else []
        )
        control_rate_hz = (
            payload.get("control_rate_hz") if isinstance(payload, Mapping) else None
        )
        if len(chunk_times) != len(list(actions or [])):
            reasons.append("chunk_timestamp_count_must_equal_action_count")
        alignment = validate_chunk_alignment(
            chunk_start_times_sec=chunk_times,
            frame_times_sec=frame_times,
            config=cfg,
        )
        reasons.extend(alignment.reasons)
        try:
            rate = float(control_rate_hz)
        except (TypeError, ValueError):
            rate = 0.0
        if not math.isfinite(rate) or rate <= 0.0:
            reasons.append("control_rate_hz_missing_or_invalid")
        try:
            chunk_values = [float(value) for value in chunk_times]
        except (TypeError, ValueError):
            chunk_values = []
        if len(chunk_values) > 1:
            if any(later <= earlier for earlier, later in zip(chunk_values, chunk_values[1:])):
                reasons.append("chunk_timestamps_not_strictly_increasing")
            elif rate > 0.0:
                expected_period = 1.0 / rate
                tolerance = max(cfg.chunk_alignment_tolerance_sec, expected_period * 0.1)
                if any(
                    abs((later - earlier) - expected_period) > tolerance
                    for earlier, later in zip(chunk_values, chunk_values[1:])
                ):
                    reasons.append("chunk_timestamps_do_not_match_control_rate")
        valid = not reasons
        results[str(episode_id)] = {
            "valid": valid,
            "reasons": sorted(set(reasons)),
            "action_count": len(list(actions or [])),
            "chunk_timestamp_count": len(chunk_times),
            "frame_timestamp_count": len(frame_times),
            "control_rate_hz": rate if rate > 0.0 else None,
        }
        if valid:
            accepted[str(episode_id)] = list(actions or [])

    stats = compute_normalization_stats(accepted, expected_dim=cfg.expected_dim) if accepted else None
    variance_blockers: List[str] = []
    if stats is not None:
        for dimension, dimension_stats in enumerate(stats.get("per_dimension") or []):
            std = float(dimension_stats.get("std") or 0.0)
            if not math.isfinite(std) or std <= MIN_NORMALIZATION_STD:
                variance_blockers.append(
                    f"normalization_zero_or_invalid_variance:dimension_{dimension}"
                )
    blockers = list(contract_blockers)
    if not episodes:
        blockers.append("action_episode_corpus_missing")
    if len(accepted) != len(episodes):
        blockers.append("one_or_more_action_episodes_rejected")
    if stats is None:
        blockers.append("no_valid_action_episodes")
    blockers.extend(variance_blockers)
    blockers = sorted(set(blockers))

    stats_path = out_root / "action_norm_stats.json"
    normalized_path = out_root / "normalized_action_corpus.json"
    stats_sha256: str | None = None
    normalized_sha256: str | None = None
    if stats is not None and not blockers:
        stats_payload = {
            **stats,
            "action_representation": SC3_ACTION_REPRESENTATION,
            "action_order": list(SC3_ACTION_ORDER),
            "action_units": list(SC3_ACTION_UNITS),
            "source_trace_sha256": trace_sha,
            "generated_at": utc_now_iso(),
        }
        write_json(stats_path, stats_payload)
        stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        normalized_episodes: Dict[str, Dict[str, Any]] = {}
        for episode_id, actions in accepted.items():
            episode_payload = dict(episodes[episode_id])
            normalized_episodes[episode_id] = {
                "normalized_actions": normalize_actions(actions, stats=stats_payload),
                "chunk_start_times_sec": list(
                    episode_payload.get("chunk_start_times_sec") or []
                ),
                "frame_times_sec": list(episode_payload.get("frame_times_sec") or []),
                "control_rate_hz": episode_payload.get("control_rate_hz"),
            }
        normalized_payload = {
            "schema_version": "normalized_action_corpus.v1",
            "source_trace_sha256": trace_sha,
            "action_norm_stats_sha256": stats_sha256,
            "episodes": normalized_episodes,
        }
        write_json(normalized_path, normalized_payload)
        normalized_sha256 = hashlib.sha256(normalized_path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": ACTION_NORMALIZATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "validated" if not blockers else "blocked",
        "blockers": blockers,
        "declared_action_dim": declared_dim,
        "action_representation": representation or None,
        "canonical_action_representation": SC3_ACTION_REPRESENTATION,
        "action_order": list(declared_order),
        "action_units": list(declared_units),
        "config": {
            "expected_dim": cfg.expected_dim,
            "max_abs_translation_delta_m": cfg.max_abs_translation_delta_m,
            "max_abs_rotation_delta_rad": cfg.max_abs_rotation_delta_rad,
            "gripper_min": cfg.gripper_min,
            "gripper_max": cfg.gripper_max,
            "chunk_alignment_tolerance_sec": cfg.chunk_alignment_tolerance_sec,
        },
        "episode_count": len(episodes),
        "accepted_episode_count": len(accepted),
        "rejected_episode_count": len(episodes) - len(accepted),
        "episode_results": results,
        "action_norm_stats_path": str(stats_path.resolve()) if stats_sha256 else None,
        "action_norm_stats_sha256": stats_sha256,
        "normalized_action_corpus_path": (
            str(normalized_path.resolve()) if normalized_sha256 else None
        ),
        "normalized_action_corpus_sha256": normalized_sha256,
        "corpus_provenance": provenance,
        "source_trace_sha256": trace_sha or None,
        "exact_consumed_trace_bound": bool(trace_sha and not contract_blockers),
        "all_dimensions_nonzero_variance": not variance_blockers and stats is not None,
        "raw_actions_untouched": True,
    }
    write_json(out_root / "action_validation_manifest.json", manifest)
    return manifest


def _explicit_action_vector(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    if not isinstance(value, Mapping):
        return None
    for key in (
        "delta_end_effector_pose_7d",
        "sc3_7d_delta_end_effector_pose",
        "action_vector_7d",
        "action",
    ):
        candidate = value.get(key)
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            return candidate
    return None


def build_action_normalization_from_trace(
    *,
    output_dir: str | Path,
    trace: Mapping[str, Any],
    source_trace_path: str | Path,
    consumed_by: str,
    action_space: Mapping[str, Any] | None = None,
    config: ActionValidationConfig | None = None,
) -> Dict[str, Any]:
    """Build the manifest from the exact persisted policy-action trace.

    Only explicit 7-D vectors are admitted. Semantic commands are not projected
    into vectors, and absent timestamps/control rate remain blockers.
    """

    source_path = Path(source_trace_path).expanduser().resolve()
    source_sha256 = (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_path.is_file()
        else _canonical_sha256(dict(trace))
    )
    episodes: Dict[str, Dict[str, Any]] = {}
    for index, attempt in enumerate(trace.get("attempts", []) or [], start=1):
        if not isinstance(attempt, Mapping):
            continue
        raw_actions = (
            attempt.get("sc3_7d_delta_end_effector_actions")
            or attempt.get("action_trace")
            or attempt.get("actions")
            or []
        )
        actions: List[List[Any]] = []
        entry_timestamps: List[Any] = []
        for entry in raw_actions if isinstance(raw_actions, Sequence) else []:
            vector = _explicit_action_vector(entry)
            if vector is not None:
                actions.append(list(vector))
                if isinstance(entry, Mapping):
                    timestamp = entry.get("timestamp_sec")
                    if timestamp is None:
                        timestamp = entry.get("timestamp_seconds")
                    if timestamp is None:
                        timestamp = entry.get("chunk_start_time_sec")
                    if timestamp is not None:
                        entry_timestamps.append(timestamp)
        chunk_times = list(
            attempt.get("chunk_start_times_sec")
            or attempt.get("action_timestamps_sec")
            or entry_timestamps
            or []
        )
        frame_times = list(
            attempt.get("frame_times_sec")
            or attempt.get("observation_timestamps_sec")
            or []
        )
        episode_id = str(
            attempt.get("episode_id")
            or attempt.get("attempt_id")
            or f"trace_attempt_{index:04d}"
        )
        episodes[episode_id] = {
            "actions": actions,
            "chunk_start_times_sec": chunk_times,
            "frame_times_sec": frame_times,
            "control_rate_hz": attempt.get("control_rate_hz")
            or trace.get("control_rate_hz"),
        }
    trace_action_space = trace.get("action_space")
    resolved_action_space = (
        dict(action_space)
        if isinstance(action_space, Mapping)
        else dict(trace_action_space)
        if isinstance(trace_action_space, Mapping)
        else {}
    )
    return build_action_normalization_manifest(
        output_dir=output_dir,
        episodes=episodes,
        action_space=resolved_action_space,
        corpus_provenance={
            "source_trace_path": str(source_path),
            "source_trace_sha256": source_sha256,
            "trace_schema_version": trace.get("schema_version"),
            "consumed_by": consumed_by,
            "source_trace_file_present": source_path.is_file(),
        },
        config=config,
    )


def load_action_normalization_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}
