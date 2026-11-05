from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_rtx_streaming_guard import (
    RtxStreamingGuardError,
    configure_post_gate_rtx_streaming_wait,
)


def _gate() -> dict:
    value = {
        "schema_version": "policy_canary_runtime_observation_integrity_gate.v1",
        "status": "passed",
        "policy_observation_integrity_passed": True,
        "candidate_policy_loaded": False,
        "candidate_policy_queried": False,
        "blockers": [],
        "gate_digest": "",
    }
    value["gate_digest"] = canonical_digest(value, digest_field="gate_digest")
    return value


def test_stale_busy_renderer_is_bounded_only_after_passed_gate(tmp_path: Path) -> None:
    renderer = SimpleNamespace(
        _STREAMING_WAIT_TIMEOUT_S=30.0,
        _get_stage_streaming_busy=lambda: True,
    )
    output = tmp_path / "guard.json"

    receipt = configure_post_gate_rtx_streaming_wait(
        observation_gate=_gate(), output_path=output, renderer_utils=renderer
    )

    assert receipt["status"] == "configured"
    assert receipt["streaming_busy_after_gate"] is True
    assert receipt["previous_wait_timeout_seconds"] == 30.0
    assert receipt["maximum_wait_timeout_seconds"] == 1.0
    assert receipt["camera_qualification_skipped"] is False
    assert renderer._STREAMING_WAIT_TIMEOUT_S == 1.0
    assert json.loads(output.read_text(encoding="utf-8")) == receipt


def test_guard_refuses_unqualified_or_post_query_gate(tmp_path: Path) -> None:
    gate = _gate()
    gate["candidate_policy_queried"] = True
    gate["gate_digest"] = canonical_digest(gate, digest_field="gate_digest")

    with pytest.raises(RtxStreamingGuardError, match="observation_gate_invalid"):
        configure_post_gate_rtx_streaming_wait(
            observation_gate=gate,
            output_path=tmp_path / "guard.json",
            renderer_utils=SimpleNamespace(
                _STREAMING_WAIT_TIMEOUT_S=30.0,
                _get_stage_streaming_busy=lambda: True,
            ),
        )


def _flagged_gate(**changes) -> dict:
    value = {**_gate(), "status": "blocked", "frame_structure_passed": True,
             "policy_observation_integrity_passed": False,
             "blockers": ["droid_wrist_task_pixels_below_threshold"], **changes}
    value["gate_digest"] = canonical_digest(value, digest_field="gate_digest")
    return value


def _renderer() -> SimpleNamespace:
    return SimpleNamespace(_STREAMING_WAIT_TIMEOUT_S=30.0, _get_stage_streaming_busy=lambda: False)


def test_a_flagged_beta_gate_is_bounded_only_when_its_caller_accepts_flags(tmp_path: Path) -> None:
    with pytest.raises(RtxStreamingGuardError, match="observation_gate_invalid"):
        configure_post_gate_rtx_streaming_wait(observation_gate=_flagged_gate(), output_path=tmp_path / "a.json",
                                               renderer_utils=_renderer())
    receipt = configure_post_gate_rtx_streaming_wait(
        observation_gate=_flagged_gate(), output_path=tmp_path / "b.json", renderer_utils=_renderer(),
        accept_observation_quality_flags=True)
    assert receipt["status"] == "configured" and receipt["observation_quality_flagged"] is True
    # A passed gate's receipt is unchanged by the option.
    passed = configure_post_gate_rtx_streaming_wait(
        observation_gate=_gate(), output_path=tmp_path / "c.json", renderer_utils=_renderer(),
        accept_observation_quality_flags=True)
    assert "observation_quality_flagged" not in passed


@pytest.mark.parametrize("changes", [
    {"frame_structure_passed": False},  # Unusable frames are never flagged past.
    {"status": "passed"},  # Integrity not passed under a passed status is malformed.
    {"blockers": []},
    {"candidate_policy_queried": True},
])
def test_accepting_flags_still_refuses_unusable_frames_or_a_queried_gate(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(RtxStreamingGuardError, match="observation_gate_invalid"):
        configure_post_gate_rtx_streaming_wait(
            observation_gate=_flagged_gate(**changes), output_path=tmp_path / "guard.json",
            renderer_utils=_renderer(), accept_observation_quality_flags=True)


def test_accepting_flags_still_refuses_a_forged_gate(tmp_path: Path) -> None:
    forged = {**_flagged_gate(), "blockers": ["something_else"]}  # edited after sealing
    with pytest.raises(RtxStreamingGuardError, match="observation_gate_invalid"):
        configure_post_gate_rtx_streaming_wait(observation_gate=forged, output_path=tmp_path / "guard.json",
                                               renderer_utils=_renderer(), accept_observation_quality_flags=True)
