from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    PolicyCanaryEpisodeFailure,
    PolicyCanarySessionError,
    build_session_authority,
    consume_session_authority_once,
    execute_paired_session,
    validate_runtime_input_manifest,
    validate_session_result,
)
from tests.test_task_evaluation_policy_canary_setup import _setup as public_setup


def _record(path: Path) -> dict[str, object]:
    import hashlib

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _activation() -> dict[str, object]:
    setup = public_setup()
    units = []
    for index in range(10):
        units.append(
            {
                "campaign_unit_id": f"unit-{index}",
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "candidate_ids": list(CANDIDATE_IDS),
            }
        )
    value: dict[str, object] = {
        "schema_version": "task_evaluation_policy_campaign_activation.v1",
        "run_id": "scene-839873-canary-1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(CANDIDATE_IDS),
        "task_success_contract": setup["task_success_contract"],
        "task_success_contract_digest": setup["task_success_contract_digest"],
        "campaign_unit_count": 10,
        "campaign_units": units,
        "activation_digest": "",
    }
    value["activation_digest"] = canonical_digest(
        value, digest_field="activation_digest"
    )
    return value


def _runtime_inputs(tmp_path: Path, activation: dict[str, object]) -> dict[str, object]:
    records = {}
    for name in ("base-packet", "runtime-source", "construction"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        records[name] = _record(path)
    cells = []
    for index in range(10):
        scenario = {"factor": "canonical" if index < 2 else "variation", "index": index}
        cells.append(
            {
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "cell_spec_digest": "sha256:" + f"{index:064x}",
                "family": "canonical_anchor" if index < 2 else "placement_approach",
                "resolved_scenario": scenario,
                "resolved_scenario_digest": canonical_digest(scenario),
                "control_diagnostic": {
                    "mode": "nonblocking_diagnostic_pending",
                    "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False,
                },
            }
        )
    value: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(CANDIDATE_IDS),
        "task_success_contract": activation["task_success_contract"],
        "task_success_contract_digest": activation[
            "task_success_contract_digest"
        ],
        "activation_digest": activation["activation_digest"],
        "configuration_digest": "sha256:" + "1" * 64,
        "plan_digest": "sha256:" + "2" * 64,
        "base_native_packet": records["base-packet"],
        "runtime_source": records["runtime-source"],
        "construction_result": records["construction"],
        "cells": cells,
        "execution_authority": {
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
            "single_warm_provider_session_required": True,
            "caller_surviving_watchdog_required": True,
            "billing_teardown_provider_zero_required": True,
        },
        "runtime_inputs_digest": "",
    }
    value["runtime_inputs_digest"] = canonical_digest(
        value, digest_field="runtime_inputs_digest"
    )
    return value


def _authority(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    activation = _activation()
    activation_path = tmp_path / "activation.json"
    activation_path.write_text(json.dumps(activation) + "\n", encoding="utf-8")
    inputs = _runtime_inputs(tmp_path, activation)
    inputs_path = tmp_path / "runtime-inputs.json"
    inputs_path.write_text(json.dumps(inputs) + "\n", encoding="utf-8")
    authority = build_session_authority(
        activation_manifest=activation,
        activation_record=_record(activation_path),
        runtime_inputs=inputs,
        runtime_input_record=_record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-839873-0123456789abcdef",
        hard_cap_usd=4.0,
        hard_ttl_seconds=14_400,
    )
    return authority, inputs


def test_session_authority_binds_one_allocation_and_typed_controls_gaps(
    tmp_path: Path,
) -> None:
    authority, inputs = _authority(tmp_path)

    assert validate_runtime_input_manifest(inputs)["runtime_inputs_digest"] == authority[
        "runtime_inputs_digest"
    ]
    assert authority["maximum_provider_allocations"] == 1
    assert authority["retry_cap"] == 0
    assert authority["caller_surviving_watchdog_required"] is True
    assert authority["official_ranking_authorized"] is False


def test_session_rejects_missing_resolved_scenario(tmp_path: Path) -> None:
    _authority_value, inputs = _authority(tmp_path)
    inputs["cells"][0].pop("resolved_scenario")  # type: ignore[index]
    inputs["runtime_inputs_digest"] = canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    )

    with pytest.raises(PolicyCanarySessionError, match="runtime_input_cell_invalid"):
        validate_runtime_input_manifest(inputs)


def test_runtime_inputs_reject_task_success_contract_tamper(tmp_path: Path) -> None:
    _authority_value, inputs = _authority(tmp_path)
    inputs["task_success_contract"]["criteria"]["orientation"]["mode"] = (
        "required"
    )
    inputs["runtime_inputs_digest"] = canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    )

    with pytest.raises(
        PolicyCanarySessionError,
        match="rigid_task_success_contract_digest_mismatch",
    ):
        validate_runtime_input_manifest(inputs)


def test_runtime_inputs_reject_unconfirmed_agent_contract(tmp_path: Path) -> None:
    activation = _activation()
    inputs = _runtime_inputs(tmp_path, activation)
    published = inputs["task_success_contract"]
    proposal = seal_rigid_task_success_contract(
        task_spec={},
        site_id=published["scope"]["site_id"],
        task_id=published["scope"]["task_id"],
        author_source="agent_proposal",
        author_id="agent:criteria-drafter",
        confirmation_status="proposal_only",
        criteria=published["criteria"],
    )
    inputs["task_success_contract"] = proposal
    inputs["task_success_contract_digest"] = proposal["contract_digest"]
    inputs["runtime_inputs_digest"] = canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    )

    with pytest.raises(
        PolicyCanarySessionError,
        match="rigid_task_success_contract_unconfirmed",
    ):
        validate_runtime_input_manifest(inputs)


def test_runtime_inputs_reject_unknown_fields(tmp_path: Path) -> None:
    activation = _activation()
    inputs = _runtime_inputs(tmp_path, activation)
    inputs["unreviewed_grading_override"] = True
    inputs["runtime_inputs_digest"] = canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    )

    with pytest.raises(
        PolicyCanarySessionError,
        match="policy_canary_runtime_input_identity_invalid",
    ):
        validate_runtime_input_manifest(inputs)


def test_executor_opens_one_session_loads_each_policy_once_and_runs_twenty(
    tmp_path: Path,
) -> None:
    authority, inputs = _authority(tmp_path)
    calls = {"open": 0, "close": 0, "loads": [], "episodes": []}

    def open_session(_inputs):
        calls["open"] += 1
        return {"session": "warm", "provider_allocations_observed": 1}

    def load_policy(_session, candidate_id):
        calls["loads"].append(candidate_id)
        return {
            "candidate_id": candidate_id,
            "checkpoint_digest": "sha256:" + (
                "c" if candidate_id == "pi05_droid" else "d"
            ) * 64,
            "runtime_identity_digest": "sha256:" + (
                "e" if candidate_id == "pi05_droid" else "f"
            ) * 64,
        }

    def run_episode(_session, policy, context):
        calls["episodes"].append((policy["candidate_id"], context["cell_id"]))
        return {
            "status": "completed",
            "candidate_policy_queried": True,
            "actions_reached_robot": True,
            "arm_moved": True,
            "observation_support_qualified": (
                policy["candidate_id"] == "pi05_droid"
            ),
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
            "lossless_frame_manifest_digest": "sha256:" + "a" * 64,
            "review_video_digest": "sha256:" + "b" * 64,
            "returned_action_sequence_digest": "sha256:" + "1" * 64,
            "action_delivery_readback_digest": "sha256:" + "2" * 64,
            "state_trace_digest": "sha256:" + "3" * 64,
            "contact_force_digest": "sha256:" + "4" * 64,
            "task_object_trajectory_digest": "sha256:" + "5" * 64,
            "deterministic_score_digest": "sha256:" + "6" * 64,
            "scoring_authority": "deterministic_simulator_state",
        }

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=open_session,
        load_policy=load_policy,
        run_episode=run_episode,
        close_policy=lambda _policy: None,
        close_session=lambda _session: calls.__setitem__("close", calls["close"] + 1)
        or {
            "status": "closed",
            "provider_allocations_observed": 1,
            "teardown_completed": True,
            "provider_zero_confirmed": True,
        },
    )

    assert result["status"] == "completed_unqualified"
    assert calls == {
        "open": 1,
        "close": 1,
        "loads": list(CANDIDATE_IDS),
        "episodes": [
            (candidate, f"cell-{index}")
            for candidate in CANDIDATE_IDS
            for index in range(10)
        ],
    }
    assert len(result["episodes"]) == 20
    assert all(row["ranking_eligible"] is False for row in result["episodes"])
    assert all(
        row["policy_outcome_interpretable"] is True
        for row in result["episodes"]
        if row["candidate_id"] == "pi05_droid"
    )
    assert all(
        row["policy_outcome_interpretable"] is False
        for row in result["episodes"]
        if row["candidate_id"] == "groot_n17_droid"
    )
    assert result["provider_allocations_observed"] == 1
    assert validate_session_result(result)["result_digest"] == result["result_digest"]

    legacy = json.loads(json.dumps(result))
    legacy.pop("task_success_contract")
    legacy.pop("task_success_contract_digest")
    legacy["result_digest"] = canonical_digest(legacy, digest_field="result_digest")
    with pytest.raises(
        PolicyCanarySessionError,
        match="policy_canary_session_result_task_success_contract_missing",
    ):
        validate_session_result(legacy)
    assert (
        validate_session_result(
            legacy, allow_legacy_missing_task_success_contract=True
        )["result_digest"]
        == legacy["result_digest"]
    )


def test_executor_can_isolate_one_cell_for_fresh_isaac_process(
    tmp_path: Path,
) -> None:
    authority, inputs = _authority(tmp_path)
    observed: list[tuple[str, str]] = []

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=lambda _inputs: {"simulation_app": "cell"},
        load_policy=lambda _session, candidate_id: {
            "candidate_id": candidate_id,
            "checkpoint_digest": "sha256:" + "c" * 64,
            "runtime_identity_digest": "sha256:" + "d" * 64,
        },
        run_episode=lambda _session, policy, context: observed.append(
            (policy["candidate_id"], context["cell_id"])
        )
        or {
            "status": "blocked",
            "candidate_policy_queried": False,
            "actions_reached_robot": False,
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
        },
        close_policy=lambda _policy: None,
        close_session=lambda _session: {
            "status": "runtime_close_committed_after_result_seal",
            "runtime_closed": True,
            "provider_closeout_pending": True,
        },
        provider_closeout_pending=True,
        selected_cell_index=4,
    )

    assert observed == [
        ("pi05_droid", "cell-4"),
        ("groot_n17_droid", "cell-4"),
    ]
    assert result["selected_cell_index"] == 4
    assert len(result["episodes"]) == 2
    assert result["status"] == "runtime_selected_cell_completed_pending_aggregation"


def test_episode_failure_is_preserved_and_does_not_cancel_remaining_rollouts(
    tmp_path: Path,
) -> None:
    authority, inputs = _authority(tmp_path)
    observed = []

    def run_episode(_session, policy, context):
        observed.append((context["candidate_id"], context["cell_id"]))
        if len(observed) == 1:
            raise RuntimeError("fixture failure")
        return {
            "status": "completed",
            "candidate_policy_queried": True,
            "actions_reached_robot": True,
            "arm_moved": True,
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
            "lossless_frame_manifest_digest": "sha256:" + "a" * 64,
            "review_video_digest": "sha256:" + "b" * 64,
            "returned_action_sequence_digest": "sha256:" + "1" * 64,
            "action_delivery_readback_digest": "sha256:" + "2" * 64,
            "state_trace_digest": "sha256:" + "3" * 64,
            "contact_force_digest": "sha256:" + "4" * 64,
            "task_object_trajectory_digest": "sha256:" + "5" * 64,
            "deterministic_score_digest": "sha256:" + "6" * 64,
            "scoring_authority": "deterministic_simulator_state",
        }

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=lambda _inputs: {"provider_allocations_observed": 1},
        load_policy=lambda _session, candidate_id: {
            "candidate_id": candidate_id,
            "checkpoint_digest": "sha256:" + (
                "c" if candidate_id == "pi05_droid" else "d"
            ) * 64,
            "runtime_identity_digest": "sha256:" + (
                "e" if candidate_id == "pi05_droid" else "f"
            ) * 64,
        },
        run_episode=run_episode,
        close_policy=lambda _policy: None,
        close_session=lambda _session: {
            "status": "closed",
            "provider_allocations_observed": 1,
            "teardown_completed": True,
            "provider_zero_confirmed": True,
        },
    )

    assert len(observed) == 20
    assert result["episodes"][0]["typed_harness_failure"] == "RuntimeError"
    assert result["episodes"][0]["visual_evidence"]["media_gap"]["type"] == (
        "before_first_observation"
    )
    assert result["episodes"][0]["reset_state_digest"] == canonical_digest(
        {
            "resolved_scenario": inputs["cells"][0]["resolved_scenario"],
            "seed": inputs["cells"][0]["seed"],
            "execution_performed": False,
        }
    )
    assert result["episodes"][1]["status"] == "completed"
    assert result["status"] == "blocked"
    assert validate_session_result(result)["status"] == "blocked"


def test_action_rejection_retains_policy_truth_and_validates_as_uninterpretable(
    tmp_path: Path,
) -> None:
    authority, inputs = _authority(tmp_path)
    rejection = {
        "schema_version": "policy_canary_action_delivery_rejection.v1",
        "status": "rejected_before_robot",
        "reason": "hard_joint_limit_violation",
        "violations": [
            "candidate_action_joint_position_bounds_invalid:count=7:first_row=1:"
            "first_dimension=3:value=-0.06750612128169087:"
            "bounds=[-3.0717999935150146,-0.0697999969124794]"
        ],
        "clamping_performed": False,
        "delivery_attempted": False,
        "actions_reached_robot": False,
        "rejection_digest": "",
    }
    rejection["rejection_digest"] = canonical_digest(
        rejection, digest_field="rejection_digest"
    )
    observed = []

    def run_episode(_session, policy, context):
        observed.append((context["candidate_id"], context["cell_id"]))
        if len(observed) == 1:
            evidence = {
                "status": "blocked",
                "episode_failure_stage": "action_delivery_rejected",
                "first_observation_retained": True,
                "candidate_policy_queried": True,
                "candidate_action_returned": True,
                "actions_reached_robot": False,
                "arm_moved": False,
                "policy_outcome_interpretable": False,
                "visual_evidence": {
                    "status": "complete",
                    "episode_terminal_status": "failed_after_first_observation",
                    "human_review_available": True,
                    "terminal_observation_invented": False,
                },
                "lossless_frame_manifest_digest": "sha256:" + "a" * 64,
                "review_video_digest": "sha256:" + "b" * 64,
                "candidate_policy_action_queries": [
                    {"raw_action_chunk": [[0.0, 0.0, 0.0, -0.06750612128169087]]}
                ],
                "returned_action_sequence_digest": canonical_digest(
                    {
                        "value": [
                            {
                                "raw_action_chunk": [
                                    [0.0, 0.0, 0.0, -0.06750612128169087]
                                ]
                            }
                        ]
                    }
                ),
                "action_delivery_readback_digest": rejection["rejection_digest"],
                "action_delivery_rejection": rejection,
            }
            raise PolicyCanaryEpisodeFailure(
                cause=RuntimeError("joint limit rejection"), evidence=evidence
            )
        return {
            "status": "completed",
            "candidate_policy_queried": True,
            "actions_reached_robot": True,
            "arm_moved": True,
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
            "lossless_frame_manifest_digest": "sha256:" + "a" * 64,
            "review_video_digest": "sha256:" + "b" * 64,
            "returned_action_sequence_digest": "sha256:" + "1" * 64,
            "action_delivery_readback_digest": "sha256:" + "2" * 64,
            "state_trace_digest": "sha256:" + "3" * 64,
            "contact_force_digest": "sha256:" + "4" * 64,
            "task_object_trajectory_digest": "sha256:" + "5" * 64,
            "deterministic_score_digest": "sha256:" + "6" * 64,
            "scoring_authority": "deterministic_simulator_state",
        }

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=lambda _inputs: {"provider_allocations_observed": 1},
        load_policy=lambda _session, candidate_id: {
            "candidate_id": candidate_id,
            "checkpoint_digest": "sha256:" + "c" * 64,
            "runtime_identity_digest": "sha256:" + "d" * 64,
        },
        run_episode=run_episode,
        close_policy=lambda _policy: None,
        close_session=lambda _session: {
            "status": "closed",
            "provider_allocations_observed": 1,
            "teardown_completed": True,
            "provider_zero_confirmed": True,
        },
    )

    assert len(observed) == 20
    failed = result["episodes"][0]
    assert failed["candidate_policy_queried"] is True
    assert failed["actions_reached_robot"] is False
    assert failed["episode_failure_stage"] == "action_delivery_rejected"
    assert failed["action_delivery_rejection"]["clamping_performed"] is False
    assert result["candidate_policy_queried"] is True
    assert result["status"] == "blocked"
    assert validate_session_result(result)["status"] == "blocked"


def test_authority_consumption_is_single_use(tmp_path: Path) -> None:
    authority, _inputs = _authority(tmp_path)
    path = tmp_path / "consumed.json"

    first = consume_session_authority_once(authority, consumption_path=path)
    second = consume_session_authority_once(authority, consumption_path=path)

    assert first["status"] == "consumed"
    assert second["status"] == "already_consumed"
    assert second["maximum_provider_allocations"] == 0


def test_closeout_failure_cannot_report_completed_or_provider_zero(tmp_path: Path) -> None:
    authority, inputs = _authority(tmp_path)
    evidence = {
        "status": "completed",
        "candidate_policy_queried": True,
        "actions_reached_robot": True,
        "arm_moved": True,
        "checkpoint_digest": "sha256:" + "c" * 64,
        "runtime_identity_digest": "sha256:" + "e" * 64,
        "lossless_frame_manifest_digest": "sha256:" + "a" * 64,
        "review_video_digest": "sha256:" + "b" * 64,
        "returned_action_sequence_digest": "sha256:" + "1" * 64,
        "action_delivery_readback_digest": "sha256:" + "2" * 64,
        "state_trace_digest": "sha256:" + "3" * 64,
        "contact_force_digest": "sha256:" + "4" * 64,
        "task_object_trajectory_digest": "sha256:" + "5" * 64,
        "deterministic_score_digest": "sha256:" + "6" * 64,
        "scoring_authority": "deterministic_simulator_state",
    }

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=lambda _inputs: {"provider_allocations_observed": 1},
        load_policy=lambda _session, candidate_id: {"candidate_id": candidate_id},
        run_episode=lambda _session, _policy, _context: evidence,
        close_policy=lambda _policy: None,
        close_session=lambda _session: {
            "status": "teardown_failed",
            "provider_allocations_observed": 1,
            "teardown_completed": False,
            "provider_zero_confirmed": False,
        },
    )

    assert result["status"] == "blocked"
    assert result["provider_allocations_observed"] == 1
    with pytest.raises(PolicyCanarySessionError, match="result_closeout_invalid"):
        validate_session_result(result)


def test_result_validator_rejects_forged_second_allocation(tmp_path: Path) -> None:
    authority, inputs = _authority(tmp_path)

    def run_episode(_session, policy, _context):
        return {
            "status": "completed",
            "candidate_policy_queried": True,
            "actions_reached_robot": True,
            "arm_moved": True,
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
            "lossless_frame_manifest_digest": "sha256:" + "a" * 64,
            "review_video_digest": "sha256:" + "b" * 64,
            "returned_action_sequence_digest": "sha256:" + "1" * 64,
            "action_delivery_readback_digest": "sha256:" + "2" * 64,
            "state_trace_digest": "sha256:" + "3" * 64,
            "contact_force_digest": "sha256:" + "4" * 64,
            "task_object_trajectory_digest": "sha256:" + "5" * 64,
            "deterministic_score_digest": "sha256:" + "6" * 64,
            "scoring_authority": "deterministic_simulator_state",
        }

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=lambda _inputs: {"provider_allocations_observed": 1},
        load_policy=lambda _session, candidate_id: {
            "checkpoint_digest": "sha256:" + "c" * 64,
            "runtime_identity_digest": "sha256:" + "d" * 64,
        },
        run_episode=run_episode,
        close_policy=lambda _policy: None,
        close_session=lambda _session: {
            "status": "closed",
            "provider_allocations_observed": 1,
            "teardown_completed": True,
            "provider_zero_confirmed": True,
        },
    )
    result["session_closeout"]["provider_allocations_observed"] = 2
    result["provider_allocations_observed"] = 2
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")

    with pytest.raises(PolicyCanarySessionError, match="result_closeout_invalid"):
        validate_session_result(result)


def _preload_gate_session_kwargs(tmp_path: Path, calls: dict) -> dict:
    authority, inputs = _authority(tmp_path)

    def open_session(_inputs):
        calls["open"] += 1
        return {"session": "warm", "provider_allocations_observed": 1}

    def load_policy(_session, candidate_id):
        calls["loads"].append(candidate_id)
        raise AssertionError("load_policy must not run when the observation gate blocks")

    return dict(
        authority=authority,
        runtime_inputs=inputs,
        open_session=open_session,
        load_policy=load_policy,
        run_episode=lambda *_args: (_ for _ in ()).throw(AssertionError("no episode")),
        close_policy=lambda _policy: None,
        close_session=lambda _session: calls.__setitem__("close", calls["close"] + 1)
        or {
            "status": "closed",
            "provider_allocations_observed": 1,
            "teardown_completed": True,
            "provider_zero_confirmed": True,
        },
    )


def test_blocked_preload_observation_gate_loads_zero_policies_and_still_closes(
    tmp_path: Path,
) -> None:
    """Scene 839873: unusable observations must never cost a policy load."""

    calls = {"open": 0, "close": 0, "loads": []}
    gate_receipt = {
        "policy_observation_integrity_passed": False,
        "blockers": ["native_task_appearance_reference_parity_missing"],
    }
    result = execute_paired_session(
        **_preload_gate_session_kwargs(tmp_path, calls),
        prepolicy_observation_gate=lambda session: gate_receipt,
    )
    assert result["status"] == "blocked"
    assert result["session_failure_type"] == "PolicyCanarySessionError"
    assert result["policy_loads"] == []
    assert result["episodes"] == []
    assert result["candidate_policy_queried"] is False
    assert result["preload_observation_gate"] == gate_receipt
    assert calls == {"open": 1, "close": 1, "loads": []}


def test_preload_observation_gate_must_return_an_explicit_pass(tmp_path: Path) -> None:
    calls = {"open": 0, "close": 0, "loads": []}
    result = execute_paired_session(
        **_preload_gate_session_kwargs(tmp_path, calls),
        prepolicy_observation_gate=lambda session: {"passed": True},
    )
    assert result["status"] == "blocked"
    assert result["policy_loads"] == []
    assert calls["loads"] == []
