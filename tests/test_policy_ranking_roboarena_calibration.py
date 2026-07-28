from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_roboarena_calibration import (
    CalibrationContractError,
    EXECUTED_PREFIX_SECONDS,
    IDENTITY_ROT6D,
    build_action_controls_v2,
    build_phase_a_inventory,
    no_motion_action_chunk,
    preregistered_protocol,
    validate_preregistered_protocol,
)


def _actions(*, phase: float = 0.0) -> list[list[float]]:
    rows: list[list[float]] = []
    for index in range(16):
        angle = phase + index * 0.002
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rows.append(
            [
                0.0001 * (index + 1),
                -0.0002 * index,
                0.0003 * index,
                cosine,
                sine,
                0.0,
                -sine,
                cosine,
                0.0,
                1.0 if index < 8 else 0.0,
            ]
        )
    return rows


def _trace(trace_id: str, *, phase: float = 0.0) -> dict[str, object]:
    return {
        "trace_kind": "recorded_real_candidate_policy",
        "source_trace_id": trace_id,
        "actions": _actions(phase=phase),
    }


def test_no_motion_control_uses_valid_rot6d_identity_and_explicit_gripper_hold() -> None:
    actions = no_motion_action_chunk(gripper_hold_value=1.0)
    assert len(actions) == 16
    assert all(row[:3] == [0.0, 0.0, 0.0] for row in actions)
    assert all(row[3:9] == list(IDENTITY_ROT6D) for row in actions)
    assert all(row[-1] == 1.0 for row in actions)
    assert actions[0] != [0.0] * 10


def test_action_controls_require_real_distinct_policy_trace() -> None:
    controls = build_action_controls_v2(
        _trace("session-a:policy-a"),
        _trace("session-a:policy-b", phase=0.1),
        gripper_hold_value=1.0,
        shuffle_seed=20260728,
    )
    assert set(controls["conditions"]) == {
        "recorded",
        "no_motion",
        "shuffled",
        "reversed",
        "policy_swapped",
    }
    assert len(set(controls["action_sha256_by_condition"].values())) == 5
    assert controls["synthetic_policy_swapped_forbidden"] is True

    synthetic = _trace("synthetic", phase=0.2)
    synthetic["trace_kind"] = "synthetic_constant_control"
    with pytest.raises(CalibrationContractError, match="must_be_real_candidate_policy_trace"):
        build_action_controls_v2(
            _trace("session-a:policy-a"),
            synthetic,
            gripper_hold_value=1.0,
            shuffle_seed=20260728,
        )


def test_literal_zero_rot6d_is_rejected() -> None:
    invalid = _trace("session-a:policy-a")
    invalid["actions"] = [[0.0] * 10 for _ in range(16)]
    with pytest.raises(CalibrationContractError, match="rot6d_columns_not_orthonormal"):
        build_action_controls_v2(
            invalid,
            _trace("session-a:policy-b", phase=0.1),
            gripper_hold_value=1.0,
            shuffle_seed=20260728,
        )


def test_protocol_requires_full_episode_then_disjoint_closed_loop_before_site() -> None:
    protocol = preregistered_protocol()
    validation = validate_preregistered_protocol(protocol)
    assert validation["status"] == "passed"
    assert validation["serial_wam_chain_forbidden"] is True
    assert protocol["phases"][0]["full_episode_required"] is True
    assert protocol["phases"][1]["executed_prefix_seconds"] == EXECUTED_PREFIX_SECONDS
    assert protocol["phases"][2]["admission"] == "phase_B_all_gates_passed"


def test_protocol_rejects_serial_oscar_cosmos_chain() -> None:
    protocol = preregistered_protocol()
    mutated = copy.deepcopy(protocol)
    mutated["backend_graph"]["forbidden_edges"] = []
    mutated_without_hash = dict(mutated)
    mutated_without_hash.pop("protocol_sha256")
    from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256

    mutated["protocol_sha256"] = canonical_sha256(mutated_without_hash)
    with pytest.raises(CalibrationContractError, match="serial_wam_chain_not_forbidden"):
        validate_preregistered_protocol(mutated)


def test_protocol_lock_matches_executable_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = json.loads(
        (
            root
            / "docs/experiments/policy_ranking_roboarena_full_stack_calibration_20260728/protocol_lock.json"
        ).read_text(encoding="utf-8")
    )
    protocol = preregistered_protocol()
    assert lock["protocol_sha256"] == protocol["protocol_sha256"]
    assert lock["frozen_endpoint_gates"] == protocol["endpoint_gates"]
    assert lock["phase_order"] == [row["phase"] for row in protocol["phases"]]
    assert lock["paid_execution_admitted"] is False

    preflight = json.loads(
        (
            root
            / "docs/experiments/policy_ranking_roboarena_full_stack_calibration_20260728/phase_a_inventory_preflight.json"
        ).read_text(encoding="utf-8")
    )
    assert preflight["protocol_sha256"] == protocol["protocol_sha256"]
    assert preflight["matrix"]["full_episode_request_count"] == 63 * 7
    assert preflight["provider_called"] is False
    assert preflight["data_uploaded"] is False
    assert preflight["secure_key_destination_approved"] is False


def test_phase_a_inventory_requires_complete_full_episode_matrix(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts"
    roboarena = tmp_path / "roboarena"
    (rollouts / ".git").mkdir(parents=True)
    for session in ("session-a", "session-b"):
        metadata = roboarena / "evaluation_sessions" / session / "metadata.yaml"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            "language_instruction: Put the object in the tray\n"
            "preference: A\n"
            "longform_feedback: hidden answer\n",
            encoding="utf-8",
        )
        for policy in ("policy-a", "policy-b"):
            video = rollouts / session / policy / "left" / "compare_overlay_vs_gt.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(f"full-episode:{session}:{policy}".encode())

    inventory = build_phase_a_inventory(
        rollout_root=rollouts,
        roboarena_root=roboarena,
        expected_session_count=2,
        expected_policy_count=2,
    )
    assert inventory["status"] == "ready"
    assert inventory["request_count"] == 4
    assert inventory["claim_class"] == "reproduction_only_not_independent_confirmation"
    assert inventory["outcome_fields_loaded_into_inventory"] is False
    assert all(row["full_episode_source"] is True for row in inventory["requests"])
    assert all("preference" not in row for row in inventory["requests"])
    assert all("longform_feedback" not in row for row in inventory["requests"])
    assert inventory["evaluator"]["model"] == "gpt-5-mini-2025-08-07"
