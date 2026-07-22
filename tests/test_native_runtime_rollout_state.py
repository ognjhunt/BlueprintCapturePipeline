from __future__ import annotations

from blueprint_pipeline.native_runtime_rollout_state import (
    current_chunk,
    ensure_rollout_state,
    refresh_rollout_playback,
    remaining_ready_chunks,
    replace_chunk,
    should_queue_more_chunks,
)


def _defaults() -> dict:
    return {
        "status": "idle",
        "chunk_duration_ms": 100,
        "target_ready_chunks": 2,
        "control_intent": {"seq": 0, "vx": 0.0},
        "trajectory_horizon": [],
        "grounding_reference_set": [],
        "queued_chunk_ids": [],
        "buffered_chunk_ids": [],
        "chunks": [],
        "refined_chunk_ids": [],
        "world_state": {"state_version": 0},
    }


def test_rollout_state_merge_preserves_nested_defaults_without_aliasing() -> None:
    state = {
        "rollout": {
            "status": "playing",
            "control_intent": {"seq": 3},
            "buffered_chunk_ids": ["chunk-1"],
            "world_state": {"state_version": 2},
        }
    }
    defaults = _defaults()

    rollout = ensure_rollout_state(state, defaults=defaults)

    assert rollout["status"] == "playing"
    assert rollout["control_intent"] == {"seq": 3, "vx": 0.0}
    assert rollout["world_state"] == {"state_version": 2}
    rollout["buffered_chunk_ids"].append("chunk-2")
    assert defaults["buffered_chunk_ids"] == []


def test_rollout_queue_decision_counts_only_chunks_after_active() -> None:
    rollout = {
        "active_chunk_id": "chunk-1",
        "buffered_chunk_ids": ["chunk-0", "chunk-1", "chunk-2"],
        "queued_chunk_ids": [],
        "target_ready_chunks": 2,
    }

    assert remaining_ready_chunks(rollout) == 1
    assert should_queue_more_chunks(rollout, default_target_ready_chunks=1) is True
    rollout["queued_chunk_ids"] = ["chunk-3"]
    assert should_queue_more_chunks(rollout, default_target_ready_chunks=1) is False


def test_chunk_replacement_is_sorted_bounded_and_detached() -> None:
    rollout: dict = {"chunks": []}
    for index in range(8):
        replace_chunk(
            rollout,
            {"chunk_id": f"chunk-{index}", "chunk_index": index},
        )
    replacement = {"chunk_id": "chunk-7", "chunk_index": 7, "status": "ready"}
    replace_chunk(rollout, replacement)
    replacement["status"] = "mutated-after-call"
    rollout["active_chunk_id"] = "chunk-7"

    assert [row["chunk_index"] for row in rollout["chunks"]] == list(range(2, 8))
    assert current_chunk(rollout)["status"] == "ready"


def test_playback_refresh_advances_then_reports_underrun() -> None:
    state = {
        "rollout": {
            "status": "playing",
            "active_chunk_id": "chunk-0",
            "buffered_chunk_ids": ["chunk-0", "chunk-1"],
            "chunks": [
                {
                    "chunk_id": "chunk-0",
                    "chunk_index": 0,
                    "activated_at_ms": 100,
                    "duration_ms": 100,
                },
                {
                    "chunk_id": "chunk-1",
                    "chunk_index": 1,
                    "media_type": "video/mp4",
                    "render_source": "runtime",
                },
            ],
        }
    }

    refresh_rollout_playback(state, defaults=_defaults(), now_ms=250)
    rollout = state["rollout"]
    assert rollout["active_chunk_id"] == "chunk-1"
    assert rollout["current_media_type"] == "video/mp4"

    refresh_rollout_playback(state, defaults=_defaults(), now_ms=400)
    assert state["rollout"]["status"] == "underrun"
    assert state["rollout"]["underrun"] is True
