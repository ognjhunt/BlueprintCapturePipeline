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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import ensure_dir, utc_now_iso, write_json

ACTION_NORMALIZATION_SCHEMA_VERSION = "action_normalization.v1"

# SC3-Eval action layout: delta translation (3), delta rotation (3), gripper (1).
DEFAULT_ACTION_DIM = 7


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
    frames = sorted(float(t) for t in frame_times_sec)
    reasons: List[str] = []
    for index, start in enumerate(chunk_start_times_sec):
        start_f = float(start)
        nearest = min(frames, key=lambda t: abs(t - start_f))
        if abs(nearest - start_f) > config.chunk_alignment_tolerance_sec:
            reasons.append(f"chunk_frame_misaligned:chunk_{index}:delta_{abs(nearest - start_f):.4f}s")
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
    epsilon: float = 1e-8,
) -> List[List[float]]:
    """Return a normalized COPY of the action stream (raw stays untouched)."""
    per_dim = list(stats.get("per_dimension") or [])
    rows = _as_float_rows(actions) or []
    normalized: List[List[float]] = []
    for row in rows:
        out: List[float] = []
        for dim, value in enumerate(row):
            dim_stats = per_dim[dim] if dim < len(per_dim) else {}
            mean = float(dim_stats.get("mean") or 0.0)
            std = float(dim_stats.get("std") or 0.0)
            out.append((value - mean) / (std + epsilon))
        normalized.append(out)
    return normalized


def build_action_normalization_manifest(
    *,
    output_dir: str | Path,
    episodes: Mapping[str, Mapping[str, Any]],
    action_space: Mapping[str, Any] | None = None,
    config: ActionValidationConfig | None = None,
) -> Dict[str, Any]:
    """Validate a corpus of episodes and persist normalization statistics.

    ``episodes`` maps episode_id -> {"actions": [[...7 floats...], ...],
    "chunk_start_times_sec": [...], "frame_times_sec": [...]} (timestamp
    fields optional; when present, per-chunk alignment is validated).

    Writes ``action_norm_stats.json`` and ``action_validation_manifest.json``
    into ``output_dir`` and returns the manifest. Status is ``validated``
    only when at least one episode passes and none were silently repaired;
    otherwise ``blocked`` with per-episode typed reasons.
    """
    out_root = Path(output_dir)
    ensure_dir(out_root)
    declared_dim = int((action_space or {}).get("dim") or DEFAULT_ACTION_DIM)
    cfg = config or ActionValidationConfig(expected_dim=declared_dim)
    if cfg.expected_dim != declared_dim:
        cfg = ActionValidationConfig(**{**cfg.__dict__, "expected_dim": declared_dim})

    accepted: Dict[str, Sequence[Sequence[Any]]] = {}
    results: Dict[str, Dict[str, Any]] = {}
    for episode_id, payload in episodes.items():
        actions = payload.get("actions") if isinstance(payload, Mapping) else None
        stream_result = validate_action_stream(list(actions or []), config=cfg)
        reasons = list(stream_result.reasons)
        chunk_times = list(payload.get("chunk_start_times_sec") or []) if isinstance(payload, Mapping) else []
        frame_times = list(payload.get("frame_times_sec") or []) if isinstance(payload, Mapping) else []
        if chunk_times or frame_times:
            alignment = validate_chunk_alignment(
                chunk_start_times_sec=chunk_times,
                frame_times_sec=frame_times,
                config=cfg,
            )
            reasons.extend(alignment.reasons)
        valid = not reasons
        results[str(episode_id)] = {"valid": valid, "reasons": reasons}
        if valid:
            accepted[str(episode_id)] = list(actions or [])

    stats = compute_normalization_stats(accepted, expected_dim=cfg.expected_dim) if accepted else None
    stats_path = out_root / "action_norm_stats.json"
    if stats is not None:
        write_json(stats_path, {**stats, "generated_at": utc_now_iso()})

    manifest = {
        "schema_version": ACTION_NORMALIZATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "validated" if stats is not None else "blocked",
        "blockers": [] if stats is not None else ["no_valid_action_episodes"],
        "declared_action_dim": declared_dim,
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
        "action_norm_stats_path": str(stats_path.resolve()) if stats is not None else None,
        "raw_actions_untouched": True,
    }
    write_json(out_root / "action_validation_manifest.json", manifest)
    return manifest


def load_action_normalization_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}
