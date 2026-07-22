"""Provider-neutral state transitions for hosted native-runtime rollouts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def ensure_rollout_state(
    state: Dict[str, Any],
    *,
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge persisted rollout state with current defaults without aliasing lists."""

    rollout = dict(state.get("rollout") or {})
    merged = {**dict(defaults), **rollout}
    merged["control_intent"] = {
        **dict(defaults.get("control_intent") or {}),
        **dict(rollout.get("control_intent") or {}),
    }
    for key in (
        "trajectory_horizon",
        "grounding_reference_set",
        "queued_chunk_ids",
        "buffered_chunk_ids",
        "chunks",
        "refined_chunk_ids",
    ):
        merged[key] = list(rollout.get(key) or [])
    merged["world_state"] = {
        **dict(defaults.get("world_state") or {}),
        **dict(rollout.get("world_state") or {}),
    }
    state["rollout"] = merged
    return merged


def remaining_ready_chunks(rollout: Mapping[str, Any]) -> int:
    """Count buffered chunks strictly after the currently active chunk."""

    buffered = [
        str(item)
        for item in list(rollout.get("buffered_chunk_ids") or [])
        if str(item)
    ]
    active_chunk_id = str(rollout.get("active_chunk_id") or "").strip()
    active_index = buffered.index(active_chunk_id) if active_chunk_id in buffered else -1
    return len(buffered) - max(active_index + 1, 0)


def should_queue_more_chunks(
    rollout: Mapping[str, Any],
    *,
    default_target_ready_chunks: int,
) -> bool:
    """Return whether a rollout needs another generation request."""

    if list(rollout.get("queued_chunk_ids") or []):
        return False
    target = max(
        1,
        int(rollout.get("target_ready_chunks") or default_target_ready_chunks),
    )
    return remaining_ready_chunks(rollout) < target


def chunk_record(
    rollout: Mapping[str, Any],
    chunk_id: str,
) -> Optional[Dict[str, Any]]:
    """Return a detached chunk record by identifier."""

    for chunk in list(rollout.get("chunks") or []):
        if str(chunk.get("chunk_id") or "") == chunk_id:
            return dict(chunk)
    return None


def replace_chunk(
    rollout: Dict[str, Any],
    chunk_payload: Mapping[str, Any],
    *,
    retained_chunk_limit: int = 6,
) -> None:
    """Insert or replace a chunk while retaining the bounded playback window."""

    chunk_id = str(chunk_payload.get("chunk_id") or "")
    chunks = [
        dict(item)
        for item in list(rollout.get("chunks") or [])
        if str(item.get("chunk_id") or "") != chunk_id
    ]
    chunks.append(dict(chunk_payload))
    chunks.sort(key=lambda item: int(item.get("chunk_index") or 0))
    rollout["chunks"] = chunks[-max(1, retained_chunk_limit) :]


def current_chunk(rollout: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the active chunk, if the rollout points at one."""

    active_chunk_id = str(rollout.get("active_chunk_id") or "").strip()
    return chunk_record(rollout, active_chunk_id) if active_chunk_id else None


def refresh_rollout_playback(
    state: Dict[str, Any],
    *,
    defaults: Mapping[str, Any],
    now_ms: int,
) -> None:
    """Advance playback or mark buffering/underrun from persisted chunk state."""

    rollout = ensure_rollout_state(state, defaults=defaults)
    active = current_chunk(rollout)
    if active:
        activated_at_ms = int(active.get("activated_at_ms") or 0)
        duration_ms = int(
            active.get("duration_ms") or rollout.get("chunk_duration_ms") or 0
        )
        if activated_at_ms > 0 and duration_ms > 0 and now_ms - activated_at_ms >= duration_ms:
            buffer_ids = [
                str(item)
                for item in list(rollout.get("buffered_chunk_ids") or [])
                if str(item)
            ]
            try:
                index = buffer_ids.index(str(active.get("chunk_id") or ""))
            except ValueError:
                index = -1
            next_chunk_id = (
                buffer_ids[index + 1]
                if index >= 0 and index + 1 < len(buffer_ids)
                else None
            )
            if next_chunk_id:
                next_chunk = chunk_record(rollout, next_chunk_id)
                if next_chunk is not None:
                    next_chunk["activated_at_ms"] = now_ms
                    replace_chunk(rollout, next_chunk)
                    rollout["active_chunk_id"] = next_chunk_id
                    rollout["status"] = "playing"
                    rollout["underrun"] = False
                    rollout["current_media_type"] = next_chunk.get("media_type")
                    rollout["current_render_source"] = next_chunk.get("render_source")
            else:
                rollout["status"] = "underrun"
                rollout["underrun"] = True

    if rollout.get("active_chunk_id"):
        return
    buffer_ids = [
        str(item)
        for item in list(rollout.get("buffered_chunk_ids") or [])
        if str(item)
    ]
    if buffer_ids:
        first_chunk = chunk_record(rollout, buffer_ids[0])
        if first_chunk is not None:
            first_chunk["activated_at_ms"] = now_ms
            replace_chunk(rollout, first_chunk)
            rollout["active_chunk_id"] = str(first_chunk.get("chunk_id") or "")
            rollout["status"] = "playing"
            rollout["underrun"] = False
            rollout["current_media_type"] = first_chunk.get("media_type")
            rollout["current_render_source"] = first_chunk.get("render_source")
    elif rollout.get("queued_chunk_ids"):
        rollout["status"] = "buffering"
    elif rollout.get("chunk_count", 0) > 0:
        rollout["status"] = "underrun"
