from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.policy_ranking_label_free_chunk_selection import (
    action_chunk_motion_metrics,
    select_first_frame_high_motion_pair,
)
from blueprint_pipeline.policy_ranking_successor_cosmos import droid_action_stream


def _stream(translation_m: float, *, rotation_rad: float = 0.0, gripper_change: bool = False):
    rows: list[list[float]] = []
    cosine = float(np.cos(rotation_rad))
    sine = float(np.sin(rotation_rad))
    for index in range(16):
        gripper = 1.0 if gripper_change and index >= 8 else 0.0
        rows.append(
            [
                translation_m,
                0.0,
                0.0,
                cosine,
                sine,
                0.0,
                -sine,
                cosine,
                0.0,
                gripper,
            ]
        )
    return droid_action_stream(rows)


def _candidate(session: str, policy: str, translation: float, **kwargs: Any) -> dict[str, Any]:
    return {
        "session_id": session,
        "policy_id": policy,
        "start_index": 0,
        "action_stream": _stream(translation, **kwargs),
    }


def test_motion_metrics_make_weak_and_strong_chunks_explicit() -> None:
    weak = action_chunk_motion_metrics(_stream(0.0009))
    strong = action_chunk_motion_metrics(_stream(0.0067, gripper_change=True))

    assert weak["translation_mean_per_step_m"] == pytest.approx(0.0009)
    assert weak["nontrivial_translation_step_count_1mm"] == 0
    assert strong["translation_mean_per_step_m"] == pytest.approx(0.0067)
    assert strong["gripper_transition_count"] == 1


def test_selector_preserves_first_frame_and_uses_distinct_real_policy_swap() -> None:
    result = select_first_frame_high_motion_pair(
        [
            _candidate("session-a", "policy-a", 0.001),
            _candidate("session-a", "policy-b", 0.003),
            _candidate("session-b", "policy-a", 0.007),
            _candidate("session-b", "policy-b", 0.005),
        ]
    )

    assert result["recorded"]["session_id_internal_only"] == "session-b"
    assert result["recorded"]["policy_id_internal_only"] == "policy-a"
    assert result["policy_swapped"]["session_id_internal_only"] == "session-b"
    assert result["policy_swapped"]["policy_id_internal_only"] == "policy-b"
    assert result["recorded"]["start_index"] == 0
    assert result["label_seal"]["outcome_labels_accessed"] is False


def test_selector_rejects_nonfirst_chunk_and_label_fields() -> None:
    shifted = _candidate("session-a", "policy-a", 0.004)
    shifted["start_index"] = 12
    with pytest.raises(ValueError, match="start_index_zero"):
        select_first_frame_high_motion_pair([shifted, _candidate("session-a", "policy-b", 0.003)])

    leaked = _candidate("session-a", "policy-a", 0.004)
    leaked["success_rate"] = 0.9
    with pytest.raises(ValueError, match="label_or_prediction_field_forbidden"):
        select_first_frame_high_motion_pair([leaked, _candidate("session-a", "policy-b", 0.003)])


def test_selector_fails_when_selected_session_has_no_real_swap() -> None:
    with pytest.raises(ValueError, match="no_distinct_real_policy_swap"):
        select_first_frame_high_motion_pair(
            [
                _candidate("session-a", "only-policy", 0.01),
                _candidate("session-b", "other-policy", 0.009),
            ]
        )
