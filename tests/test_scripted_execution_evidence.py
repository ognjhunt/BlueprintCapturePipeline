"""Synthetic runtime contracts; no native execution or physical performance claim."""
import json

import pytest

from blueprint_pipeline.native_task_arena_policy_canary_session import (
    build_session_authority, execute_paired_session, validate_session_result,
)
from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker
from blueprint_pipeline.policy_execution_evidence import (
    KINDS_FIELD, LEARNED, SCRIPTED, SCRIPTED_ACTION_SOURCE, rollout_counts,
    rollout_counts_valid, scripted_fields, scripted_episode_outcome,
)
from blueprint_pipeline.policy_canary_session_bundle_identity import _episode_shape_valid, BETA_SHAPE
from tests.test_native_task_arena_policy_canary_session import _activation, _record, _runtime_inputs
from tests.test_beta_policy_screening_coordinator import native_result, plan, seal
from tests.test_beta_policy_screening_closeout import attempt, Provider  # noqa: F401 - shared fixture


def _session(tmp_path, *, mixed=False):
    activation = _activation()
    kinds = {activation["candidate_ids"][0]: SCRIPTED,
             activation["candidate_ids"][1]: LEARNED if mixed else SCRIPTED}
    activation[KINDS_FIELD] = kinds
    seal(activation, "activation_digest")
    inputs = _runtime_inputs(tmp_path, activation)
    inputs[KINDS_FIELD] = kinds
    seal(inputs, "runtime_inputs_digest")
    def record(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return _record(path)
    authority = build_session_authority(activation_manifest=activation,
        activation_record=record("activation.json", activation), runtime_inputs=inputs,
        runtime_input_record=record("inputs.json", inputs), resource_name="blueprint-native-task-policy-canary-scripted-test",
        hard_cap_usd=4, hard_ttl_seconds=3600)
    return authority, inputs, kinds


def _episode(kind):
    row = {"status": "completed", "candidate_policy_queried": True,
           "actions_reached_robot": True, "arm_moved": True,
           "scoring_authority": "deterministic_simulator_state",
           "episode": {"score": {"status": "scored", "task_succeeded": True}}}
    row.update({key: "sha256:" + "a" * 64 for key in (
        "checkpoint_digest", "runtime_identity_digest", "lossless_frame_manifest_digest", "review_video_digest",
        "returned_action_sequence_digest", "action_delivery_readback_digest", "state_trace_digest",
        "contact_force_digest", "task_object_trajectory_digest", "deterministic_score_digest")})
    if kind == SCRIPTED:
        row.update(scripted_fields(invoked=True))
    return row


def _run(authority, inputs, kinds, **kwargs):
    return execute_paired_session(authority=authority, runtime_inputs=inputs,
        open_session=lambda _: {"provider_allocations_observed": 1},
        load_policy=lambda _, candidate: {"candidate_id": candidate},
        run_episode=lambda _, policy, context: _episode(kinds[policy["candidate_id"]]),
        close_policy=lambda _: None,
        close_session=lambda _: {"provider_allocations_observed": 1,
                                "teardown_completed": True, "provider_zero_confirmed": True}, **kwargs)


@pytest.mark.parametrize("mixed", [False, True])
def test_scripted_and_mixed_sessions_have_disjoint_counts_and_invocation(tmp_path, mixed):
    authority, inputs, kinds = _session(tmp_path, mixed=mixed)
    assert authority["learned_policy_rollout_count"] == (10 if mixed else 0)
    assert authority["scripted_rollout_count"] == (10 if mixed else 20)
    result = validate_session_result(_run(authority, inputs, kinds))
    assert result["learned_policy_rollout_count"] == (10 if mixed else 0)
    assert result["scripted_rollout_count"] == (10 if mixed else 20)
    assert result["candidate_policy_queried"] is mixed
    assert result["scripted_controller_invoked"] is True
    for row in result["episodes"]:
        if kinds[row["candidate_id"]] == SCRIPTED:
            assert row["candidate_policy_queried"] is False
            assert row["policy_outcome_interpretable"] is False
            assert row["scripted_outcome_interpretable"] is True
            assert row["action_source"] == SCRIPTED_ACTION_SOURCE
        else:
            assert row["policy_outcome_interpretable"] is True
            assert "scripted_outcome_interpretable" not in row
    assert result["official_ranking_performed"] is False
    assert result["scene_promotion_performed"] is False


@pytest.mark.parametrize("mutation", ["learned_count", "learned_query", "policy_outcome", "source", "map", "invocation"])
def test_sealed_scripted_result_cannot_promote_to_learned_evidence(tmp_path, mutation):
    authority, inputs, kinds = _session(tmp_path)
    result = _run(authority, inputs, kinds)
    if mutation == "learned_count":
        result["learned_policy_rollout_count"] = 20
        result["scripted_rollout_count"] = 0
    elif mutation == "map":
        result.pop(KINDS_FIELD)
    elif mutation == "invocation":
        result["scripted_controller_invoked"] = False
    else:
        field = {"learned_query": "candidate_policy_queried", "policy_outcome": "policy_outcome_interpretable",
                 "source": "action_source"}[mutation]
        result["episodes"][0][field] = "learned_policy" if mutation == "source" else True
    seal(result, "result_digest")
    with pytest.raises(ValueError):
        validate_session_result(result)


def test_tracker_marks_real_script_invocation_without_learned_query_even_on_error():
    class Script:
        execution_evidence_kind = SCRIPTED
        candidate_policy_queried = False
        def infer(self, _):
            raise RuntimeError("scripted_failure")
    tracker = _PolicyQueryTracker(Script())
    assert not tracker.scripted_controller_invoked
    with pytest.raises(RuntimeError, match="scripted_failure"):
        tracker.infer({})
    assert tracker.scripted_controller_invoked and not tracker.candidate_policy_queried


def test_scripted_counts_are_retained_episodes_not_a_fabricated_invocation(tmp_path):
    from blueprint_pipeline.native_task_arena_policy_canary_session import PolicyCanaryEpisodeFailure
    authority, inputs, kinds = _session(tmp_path)
    def failure(*_):
        row = {**_episode(SCRIPTED), **scripted_fields(invoked=False), "status": "blocked",
               "actions_reached_robot": False, "arm_moved": False,
               "visual_evidence": {"media_gap": {"type": "before_first_observation", "reason": "fixture_startup_failure"}}}
        raise PolicyCanaryEpisodeFailure(cause=RuntimeError("fixture_startup_failure"), evidence=row)
    result = execute_paired_session(authority=authority, runtime_inputs=inputs,
        open_session=lambda _: {"provider_allocations_observed": 1}, load_policy=lambda *_: {},
        run_episode=failure, close_policy=lambda _: None,
        close_session=lambda _: {"provider_allocations_observed": 1, "teardown_completed": True, "provider_zero_confirmed": True})
    validate_session_result(result)
    assert result["learned_policy_rollout_count"] == 0 and result["scripted_rollout_count"] == 20
    assert not result["scripted_controller_invoked"] and not result["candidate_policy_queried"]


def _scripted_beta_result():
    value = plan()
    result = native_result(value)
    kinds = {result["candidate_ids"][0]: SCRIPTED}
    result.update(rollout_counts(kinds, 10), scripted_controller_invoked=True, candidate_policy_queried=False)
    for row in result["episodes"]:
        row.update(scripted_fields(invoked=True), scripted_outcome_interpretable=True)
    return value, seal(result, "result_digest")


def test_scripted_summary_retains_real_score_with_no_learned_or_physical_claim():
    from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary
    from blueprint_pipeline.success_claim_contracts import POLICY_ACTION_SOURCES, build_policy_action_execution
    value, result = _scripted_beta_result()
    summary = candidate_result_summary(result, plan=value, candidate=value["candidates"][0])
    assert summary["execution_evidence_kind"] == SCRIPTED
    assert summary["episodes_scored"] == 10 and summary["scripted_rollout_count"] == 10
    assert summary["learned_policy_rollout_count"] == 0 and not summary["candidate_policy_queried"]
    assert SCRIPTED_ACTION_SOURCE not in POLICY_ACTION_SOURCES
    proof = build_policy_action_execution(action_source=SCRIPTED_ACTION_SOURCE, actions_executed_in_simulator=True, action_trace_present=True)
    assert proof["status"] == "FAIL"


def test_scripted_bundle_shape_uses_disjoint_counts():
    manifest = {"candidate_ids": ["scripted-arm"], "episodes_per_policy": 3,
                **rollout_counts({"scripted-arm": SCRIPTED}, 3)}
    assert _episode_shape_valid(manifest, BETA_SHAPE)
    assert not _episode_shape_valid({**manifest, "learned_policy_rollout_count": 3}, BETA_SHAPE)
    assert not rollout_counts_valid({**manifest, "scripted_rollout_count": True}, 3)


def test_scripted_controller_error_and_actual_failure_remain_individual_failures():
    row = {**scripted_fields(invoked=True), "status": "blocked", "episode_failure_cause": "scripted_controller"}
    assert scripted_episode_outcome(row) == ("failure", "scripted_controller_failed")
    row = _episode(SCRIPTED)
    row["episode"]["score"]["task_succeeded"] = False
    assert scripted_episode_outcome(row) == ("failure", "task_not_completed")
    row["candidate_policy_queried"] = True
    with pytest.raises(ValueError):
        scripted_episode_outcome(row)


@pytest.mark.parametrize("missing", ["scripted_outcome_interpretable", "actions_reached_robot", "arm_moved"])
def test_scripted_success_needs_its_own_action_witness_but_stationary_failure_remains_failure(missing):
    row = {**_episode(SCRIPTED), "scripted_outcome_interpretable": True}
    assert scripted_episode_outcome(row) == ("success", "task_succeeded")
    row[missing] = False
    assert scripted_episode_outcome(row) == ("void", "scripted_task_score_without_interpretable_action_evidence")
    assert row["episode"]["score"]["task_succeeded"] is True  # Raw scorer verdict remains unchanged.
    row["episode"]["score"]["task_succeeded"] = False
    assert scripted_episode_outcome(row) == ("failure", "task_not_completed")


@pytest.mark.parametrize("sealed_current", [False, True])
def test_scripted_timeout_retains_bound_children_and_never_invents_interrupted_invocation(tmp_path, sealed_current):
    from blueprint_pipeline.team_evaluation_worker_retention import seal_scripted_interruption
    value, result = _scripted_beta_result()
    candidate = value["candidates"][0]
    inputs = {**{key: result[key] for key in ("run_kind", "candidate_ids", KINDS_FIELD,
        "task_success_contract", "task_success_contract_digest", "beta_screening_admission",
        "scene_revision_digest", "plan_digest", "configuration_digest")},
        "cells": candidate["scenario_cells"], "runtime_inputs_digest": "sha256:" + "a" * 64}
    authority = {"run_id": "scripted-test", "authority_digest": "sha256:" + "b" * 64}
    def child(index):
        return seal({**result, "status": "runtime_selected_cell_completed_pending_aggregation",
            "selected_cell_index": index, "episodes": [result["episodes"][index]], "scripted_rollout_count": 1}, "result_digest")
    root = tmp_path / "cell_runs/01"
    root.mkdir(parents=True)
    if sealed_current:
        from blueprint_pipeline.native_task_arena_policy_canary_session import PROVIDER_RESULT_FILENAME
        (root / PROVIDER_RESULT_FILENAME).write_text(json.dumps(child(1)))
    outputs = []
    assert seal_scripted_interruption(inputs=inputs, authority=authority, children=[child(0)],
        child_root=root, output_root=tmp_path, index=1, reason="fixture_deadline",
        seal_result=lambda **kw: outputs.append(kw["result"]))
    output = outputs[0]
    assert output["scripted_rollout_count"] == len(output["episodes"]) == (2 if sealed_current else 1)
    assert output["learned_policy_rollout_count"] == 0 and output["candidate_policy_queried"] is False
    assert output["interrupted_scripted_invocation_status"] == ("retained_receipt" if sealed_current else "unknown")
    assert output["planned_scripted_rollout_count"] == 10 and output["blockers"] == ["fixture_deadline"]
    assert output["automatic_retry_performed"] is False and output["scene_promotion_performed"] is False


def test_scripted_failure_receipt_keeps_controller_invocation_separate(tmp_path):
    from blueprint_pipeline.policy_canary_worker_evidence import _write_episode_failure_gap
    path = _write_episode_failure_gap(output_root=tmp_path, run_id="fixture",
        context={"candidate_id": "scripted-test", "cell_id": "cell-1", "seed": 1,
                 "execution_evidence_kind": SCRIPTED}, failure=RuntimeError("fixture_script_error"),
        progress={**scripted_fields(invoked=True), "scripted_controller_failed": True})
    receipt = json.loads(path.read_text())
    assert receipt["scripted_controller_invoked"] is True and receipt["candidate_policy_queried"] is False
    assert receipt["candidate_policy_query_attempted"] is False
    assert receipt["episode_failure_cause"] == "scripted_controller"
    assert receipt["policy_outcome_interpretable"] is False
    assert receipt["episode"]["score"]["status"] == "not_scored"
    assert "policy_query_receipt" in receipt["evidence_artifacts"]


def test_scripted_session_closes_against_its_own_typed_authority(request):
    from pathlib import Path
    from blueprint_pipeline import beta_policy_screening_closeout as closeout
    args, inner, authority = request.getfixturevalue("attempt")
    kinds = {inner["candidate_ids"][0]: SCRIPTED}
    fields = rollout_counts(kinds, 10)
    authority.update(fields)
    seal(authority, "authority_digest")
    prepared, adapter = args["prepared"], args["adapter"]
    authority_path = Path(prepared["authority"]["path"])
    authority_path.write_text(json.dumps(authority))
    prepared["authority"] = _record(authority_path)
    bundle_path = Path(prepared["bundle_receipt"]["path"])
    bundle = json.loads(bundle_path.read_text())
    bundle.update(fields, authority_digest=authority["authority_digest"])
    bundle_path.write_text(json.dumps(bundle))
    prepared["bundle_receipt"] = _record(bundle_path)
    seal(prepared, "prepared_digest")
    adapter["authorization_consumption"]["authority_digest"] = authority["authority_digest"]
    seal(adapter["authorization_consumption"], "consumption_digest")
    inner.update(fields, scripted_controller_invoked=True, candidate_policy_queried=False)
    for row in inner["episodes"]:
        row.update(scripted_fields(invoked=True), scripted_outcome_interpretable=True)
    seal(inner, "result_digest")
    adapter["native_control_result_digest"] = inner["result_digest"]
    receipt = closeout.collect_scoped_closeout(**args, provider=Provider())
    result = closeout.join_scoped_session_closeout(inner=inner, receipt=receipt, **args)
    assert result["status"] == "completed_unqualified"
    assert result["learned_policy_rollout_count"] == 0 and result["scripted_rollout_count"] == 10
    assert result["candidate_policy_queried"] is False
    assert result["episodes"] == inner["episodes"]
