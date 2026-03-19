"""Bounded future-anchor / lookahead re-grounding from real future references."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..common import utc_now_iso

DEFAULT_FUTURE_ANCHOR_REGROUNDING_POLICY: Dict[str, Any] = {
    "policy_version": "v1",
    "re_grounding_mode": "bounded_future_anchor_support",
    "support_source_mode": "real_future_references_only",
    "require_grounded_protected_regions": True,
    "require_task_targets": True,
    "min_lookahead_gap_sec": 0.4,
    "max_lookahead_gap_sec": 6.0,
    "min_anchor_gain": 1,
    "min_checkpoint_improvement_sec": 0.05,
    "max_future_candidates_per_target": 2,
}


def build_future_anchor_regrounding_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    selection_entries: Sequence[Mapping[str, Any]],
    task_anchor_manifest: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    regrounding_name: str,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_policy = resolve_future_anchor_regrounding_policy(policy)
    record_by_frame_id = {
        str(record.get("frame_id") or "").strip(): dict(record)
        for record in records
        if str(record.get("frame_id") or "").strip()
    }

    grounded = (
        str(protected_regions_manifest.get("grounding_status") or "grounded").strip().lower() == "grounded"
    )
    task_target_count = sum(
        len(list(task.get("target_object_ids") or []))
        for task in list(task_anchor_manifest.get("tasks") or [])
        if isinstance(task, Mapping)
    )

    entries: List[Dict[str, Any]] = []
    regrounded_target_count = 0

    for selection in selection_entries:
        target_frame_id = str(selection.get("target_frame_id") or "").strip()
        target_record = record_by_frame_id.get(target_frame_id)
        if target_record is None:
            continue
        entry = _reground_target(
            target_record=target_record,
            selection=selection,
            grounded=grounded,
            task_target_count=task_target_count,
            record_by_frame_id=record_by_frame_id,
            policy=resolved_policy,
        )
        if entry["status"] == "re_grounded":
            regrounded_target_count += 1
        entries.append(entry)

    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "regrounding_name": regrounding_name,
        "policy": resolved_policy,
        "grounded_protected_regions": grounded,
        "task_target_count": task_target_count,
        "record_count": len(records),
        "target_count": len(selection_entries),
        "re_grounded_target_count": regrounded_target_count,
        "skipped_target_count": max(0, len(entries) - regrounded_target_count),
        "entries": entries,
    }


def resolve_future_anchor_regrounding_policy(
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = dict(DEFAULT_FUTURE_ANCHOR_REGROUNDING_POLICY)
    if not isinstance(policy, Mapping):
        return resolved
    for key, value in policy.items():
        resolved[str(key)] = value
    return resolved


def _reground_target(
    *,
    target_record: Mapping[str, Any],
    selection: Mapping[str, Any],
    grounded: bool,
    task_target_count: int,
    record_by_frame_id: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    target_frame_id = str(target_record.get("frame_id") or "").strip()
    if bool(policy.get("require_grounded_protected_regions", True)) and not grounded:
        return _skipped_entry(target_frame_id=target_frame_id, selection=selection, reason="protected_regions_ungrounded")
    if bool(policy.get("require_task_targets", True)) and task_target_count <= 0:
        return _skipped_entry(target_frame_id=target_frame_id, selection=selection, reason="no_task_targets")

    target_time = _optional_float(target_record.get("t_capture_sec"))
    target_anchor_count = len(_anchor_ids(target_record.get("anchor_observations")))
    target_checkpoint_proximity = _signal_float(target_record, "checkpoint_proximity_sec")

    if target_time is None:
        return _skipped_entry(target_frame_id=target_frame_id, selection=selection, reason="missing_target_time")

    min_lookahead_gap_sec = float(policy.get("min_lookahead_gap_sec") or 0.4)
    max_lookahead_gap_sec = float(policy.get("max_lookahead_gap_sec") or 6.0)
    min_anchor_gain = int(policy.get("min_anchor_gain") or 1)
    min_checkpoint_improvement_sec = float(policy.get("min_checkpoint_improvement_sec") or 0.05)
    max_future_candidates_per_target = int(policy.get("max_future_candidates_per_target") or 2)

    future_candidates: List[Dict[str, Any]] = []
    for frame_id in list(selection.get("selected_reference_frame_ids") or []):
        candidate = record_by_frame_id.get(str(frame_id or "").strip())
        if candidate is None:
            continue
        candidate_time = _optional_float(candidate.get("t_capture_sec"))
        if candidate_time is None:
            continue
        lookahead_gap_sec = candidate_time - target_time
        if lookahead_gap_sec < min_lookahead_gap_sec or lookahead_gap_sec > max_lookahead_gap_sec:
            continue
        candidate_anchor_count = len(_anchor_ids(candidate.get("anchor_observations")))
        anchor_gain = candidate_anchor_count - target_anchor_count
        candidate_checkpoint_proximity = _signal_float(candidate, "checkpoint_proximity_sec")
        checkpoint_improvement_sec = (
            target_checkpoint_proximity - candidate_checkpoint_proximity
            if target_checkpoint_proximity is not None and candidate_checkpoint_proximity is not None
            else None
        )
        if anchor_gain < min_anchor_gain and (
            checkpoint_improvement_sec is None or checkpoint_improvement_sec < min_checkpoint_improvement_sec
        ):
            continue
        future_candidates.append(
            {
                "reference_id": str(candidate.get("reference_id") or "").strip() or None,
                "reference_frame_id": str(candidate.get("frame_id") or "").strip() or None,
                "lookahead_gap_sec": round(lookahead_gap_sec, 4),
                "anchor_gain": int(anchor_gain),
                "checkpoint_improvement_sec": round(checkpoint_improvement_sec, 4) if checkpoint_improvement_sec is not None else None,
                "anchor_observation_count": candidate_anchor_count,
                "anchor_observations": _anchor_ids(candidate.get("anchor_observations")),
            }
        )

    future_candidates.sort(
        key=lambda item: (
            -int(item["anchor_gain"]),
            -float(item["checkpoint_improvement_sec"] if item["checkpoint_improvement_sec"] is not None else -1.0),
            float(item["lookahead_gap_sec"]),
            str(item["reference_frame_id"] or ""),
        )
    )
    future_candidates = future_candidates[:max_future_candidates_per_target]

    if not future_candidates:
        return _skipped_entry(target_frame_id=target_frame_id, selection=selection, reason="no_bounded_future_anchor_support")

    return {
        "target_frame_id": target_frame_id,
        "target_reference_decoupling_mode": (
            (selection.get("decoupling") or {})
            if isinstance(selection.get("decoupling"), Mapping)
            else {}
        ).get("mode"),
        "future_anchor_context_id": f"future-anchor-{target_frame_id}",
        "status": "re_grounded",
        "reason": None,
        "future_anchor_reference_ids": [item["reference_id"] for item in future_candidates if item.get("reference_id")],
        "future_anchor_frame_ids": [item["reference_frame_id"] for item in future_candidates if item.get("reference_frame_id")],
        "future_anchor_count": len(future_candidates),
        "future_anchor_candidates": future_candidates,
    }


def _skipped_entry(
    *,
    target_frame_id: str,
    selection: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "target_frame_id": target_frame_id,
        "target_reference_decoupling_mode": (
            (selection.get("decoupling") or {})
            if isinstance(selection.get("decoupling"), Mapping)
            else {}
        ).get("mode"),
        "future_anchor_context_id": f"future-anchor-{target_frame_id or 'unknown'}",
        "status": "skipped",
        "reason": reason,
        "future_anchor_reference_ids": [],
        "future_anchor_frame_ids": [],
        "future_anchor_count": 0,
        "future_anchor_candidates": [],
    }


def _anchor_ids(raw_value: Any) -> List[str]:
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw_value:
        if isinstance(item, Mapping):
            text = str(item.get("anchor_id") or item.get("anchorId") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _signal_float(record: Mapping[str, Any], key: str) -> Optional[float]:
    signals = dict(record.get("retrieval_signals") or {}) if isinstance(record.get("retrieval_signals"), Mapping) else {}
    return _optional_float(signals.get(key))


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
