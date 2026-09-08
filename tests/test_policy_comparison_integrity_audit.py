"""Offline adversarial producer-boundary regressions for ADP-003/004/005/009D."""
from copy import deepcopy

import pytest

from blueprint_pipeline.adp_task_scoring import (
    TaskNeutralScoringError,
    score_task_episode_from_spec,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    PolicyCanarySessionError,
    execute_paired_session,
    validate_session_result,
)
from tests.test_adp_task_scoring import _rigid_v2_sample, _rigid_v2_spec
from tests.test_native_task_arena_policy_canary_session import _authority


def _result(tmp_path, *, score=None):
    authority, inputs = _authority(tmp_path)

    def load(_session, candidate):
        return {
            "candidate_id": candidate,
            "checkpoint_digest": "sha256:" + ("c" if candidate == CANDIDATE_IDS[0] else "d") * 64,
            "runtime_identity_digest": "sha256:" + "e" * 64,
        }

    def episode(_session, policy, context):
        return {
            **policy,
            **({"episode": {"score": score}} if score is not None else {}),
            "status": "completed",
            "candidate_policy_queried": True,
            "actions_reached_robot": True,
            "arm_moved": True,
            "scoring_authority": "deterministic_simulator_state",
            **{key: "sha256:" + "a" * 64 for key in (
                "lossless_frame_manifest_digest", "review_video_digest",
                "returned_action_sequence_digest", "action_delivery_readback_digest",
                "state_trace_digest", "contact_force_digest",
                "task_object_trajectory_digest", "deterministic_score_digest",
            )},
        }

    return execute_paired_session(
        authority=authority, runtime_inputs=inputs,
        open_session=lambda _: {"provider_allocations_observed": 1},
        load_policy=load, run_episode=episode, close_policy=lambda _: None,
        close_session=lambda _: {
            "provider_allocations_observed": 1,
            "teardown_completed": True, "provider_zero_confirmed": True,
        },
    )


def _seal(value):
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


@pytest.mark.parametrize("score", [
    {"status": "undetermined", "task_succeeded": None},
    {"status": "not_scored"},
])
def test_completed_unscored_episode_is_not_interpretable(tmp_path, score):
    result = _result(tmp_path, score=score)
    assert result["status"] == "completed_unqualified"
    assert all(row["policy_outcome_interpretable"] is False for row in result["episodes"])


@pytest.mark.parametrize("field", ["checkpoint_digest", "runtime_identity_digest"])
def test_same_candidate_cannot_change_frozen_identity_between_cells(tmp_path, field):
    result = _result(tmp_path)
    result["episodes"][1][field] = "sha256:" + "9" * 64
    with pytest.raises(PolicyCanarySessionError, match="candidate_identity"):
        validate_session_result(_seal(result))


def test_matching_cell_labels_cannot_hide_different_resolved_science(tmp_path):
    result = _result(tmp_path)
    row = result["episodes"][10]
    row["resolved_scenario"] = {**row["resolved_scenario"], "index": 999}
    row["resolved_scenario_digest"] = canonical_digest(row["resolved_scenario"])
    with pytest.raises(PolicyCanarySessionError, match="scenario"):
        validate_session_result(_seal(result))


def test_result_reader_rehashes_resolved_scenario(tmp_path):
    result = _result(tmp_path)
    result["episodes"][0]["resolved_scenario"]["index"] = 999
    with pytest.raises(PolicyCanarySessionError, match="scenario"):
        validate_session_result(_seal(result))


def test_episode_reordering_preserves_pair_acceptance(tmp_path):
    result = _result(tmp_path)
    result["episodes"].reverse()
    result["candidate_ids"].reverse()
    assert validate_session_result(_seal(result))["status"] == "completed_unqualified"


@pytest.mark.parametrize("mutation", ["seed", "duplicate", "missing", "extra_candidate"])
def test_existing_pair_boundary_refuses_bad_membership(tmp_path, mutation):
    result = _result(tmp_path)
    if mutation == "seed":
        result["episodes"][0]["seed"] += 1
    elif mutation == "duplicate":
        result["episodes"][1] = deepcopy(result["episodes"][0])
    elif mutation == "missing":
        result["episodes"].pop()
    else:
        result["candidate_ids"].append("third")
    with pytest.raises(PolicyCanarySessionError):
        validate_session_result(_seal(result))


def _trajectory():
    # Hand-computed: 3 cm lift, 15 cm translation, released at 8 cm width,
    # supported and stationary at the destination for steps 3, 4 and 5.
    return [_rigid_v2_sample(i, xyz) for i, xyz in enumerate([
        [1, 2, .8], [1, 2, .83], [1.15, 2, .83],
        [1.15, 2, .8], [1.15, 2, .8], [1.15, 2, .8],
    ])]


@pytest.mark.parametrize("missing_step", [1, 2, 3, 4])
def test_missing_native_sample_cannot_be_scored_as_complete_trajectory(missing_step):
    samples = _trajectory()
    del samples[missing_step]
    with pytest.raises(TaskNeutralScoringError, match="sample_step"):
        score_task_episode_from_spec(task_spec=_rigid_v2_spec(), samples=samples)


def test_complete_analytic_trajectory_is_success_and_ignores_policy_claim():
    samples = _trajectory()
    for sample in samples:
        sample["policy_reported_success"] = False
    assert score_task_episode_from_spec(task_spec=_rigid_v2_spec(), samples=samples)["task_succeeded"] is True


@pytest.mark.parametrize("field,value,expected", [
    ("minimum_lift_m", .03, True),
    ("minimum_lift_m", .0301, False),
    ("minimum_translation_m", .15, True),
    ("minimum_translation_m", .1501, False),
    ("release_gripper_width_min_m", .08, True),
    ("release_gripper_width_min_m", .0801, False),
])
def test_analytic_thresholds_are_inclusive_and_not_softened(field, value, expected):
    spec = _rigid_v2_spec()
    spec[field] = value
    assert score_task_episode_from_spec(task_spec=spec, samples=_trajectory())["task_succeeded"] is expected


@pytest.mark.parametrize("field", ["task_object_pose_world", "gripper_width_m"])
def test_nonfinite_state_never_yields_success(field):
    samples = _trajectory()
    if field == "task_object_pose_world":
        samples[-1][field][0] = float("nan")
        with pytest.raises(TaskNeutralScoringError):
            score_task_episode_from_spec(task_spec=_rigid_v2_spec(), samples=samples)
    else:
        samples[-1][field] = float("inf")
        assert score_task_episode_from_spec(task_spec=_rigid_v2_spec(), samples=samples)["task_succeeded"] is not True


def test_preregistered_decision_preserves_missingness_and_input_order():
    from blueprint_pipeline.adp_prospective_design import compile_decision, ADPProspectiveDesignError
    from tests.test_adp_prospective_design import _schedule, _binary_results
    schedule = _schedule()
    results = _binary_results(schedule, baseline_successes=20, alternative_successes=30)
    # Every trial remains in its frozen denominator, even with no supplied results.
    empty = compile_decision(schedule=schedule, trial_results=[])
    assert all(row["successes"] == 0 and row["status_counts"]["missing"] == row["frozen_denominator"]
        for row in empty["candidate_summaries"].values())
    original = compile_decision(schedule=schedule, trial_results=results[3:])
    assert original == compile_decision(schedule=schedule, trial_results=list(reversed(results[3:])))
    with pytest.raises(ADPProspectiveDesignError, match="duplicate"):
        compile_decision(schedule=schedule, trial_results=[*results, results[0]])
    changed = deepcopy(schedule)
    changed["statistical_design"]["minimum_decision_relevant_difference"] = .01
    with pytest.raises(ADPProspectiveDesignError):
        compile_decision(schedule=changed, trial_results=results)


@pytest.mark.parametrize("bad_reset", [2, 3])
def test_task_reset_drift_blocks_before_first_policy_query(tmp_path, bad_reset):
    from blueprint_pipeline.adp009d_policy_episode import PolicyEpisodeError
    from tests.test_adp009d_policy_episode import _LifecycleEnvironment, _LifecyclePolicy, _run

    class StaleTaskEnvironment(_LifecycleEnvironment):
        def read_object_sample(self):
            sample = super().read_object_sample()
            if self.reset_count == bad_reset:
                sample["can_pose_world"][0] += .10
            return sample

    policy = _LifecyclePolicy()
    progress = {}
    with pytest.raises(PolicyEpisodeError, match="reset.*mismatch"):
        _run(environment=StaleTaskEnvironment(), policy=policy,
            max_policy_queries=1, settle_window_samples=1,
            media_output_dir=tmp_path, episode_id="stale-task",
            require_complete_multicamera_media=True, require_prestart_readiness=True,
            progress=progress)
    assert policy.observations == []
    assert progress["episode_started"] is False


@pytest.mark.parametrize("mutation", ["scenario", "child_digest"])
def test_child_aggregation_binds_resolved_values_to_the_frozen_input(tmp_path, monkeypatch, mutation):
    from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
    result = _result(tmp_path)
    authority, inputs = _authority(tmp_path)
    children = []
    for index in range(10):
        child = {
            "selected_cell_index": index,
            "status": "runtime_selected_cell_completed_pending_aggregation",
            "task_success_contract": inputs["task_success_contract"],
            "task_success_contract_digest": inputs["task_success_contract_digest"],
            "episodes": deepcopy([result["episodes"][index], result["episodes"][index + 10]]),
        }
        children.append(_seal(child))
    if mutation == "scenario":
        # Both candidates agree on the same wrong cell: pair equality alone
        # cannot substitute for equality to the frozen execution inputs.
        for row in children[0]["episodes"]:
            row["resolved_scenario"] = {"factor": "canonical", "index": 999}
            row["resolved_scenario_digest"] = canonical_digest(row["resolved_scenario"])
        _seal(children[0])
    else:
        children[0]["result_digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(worker, "_write_indexed_telemetry", lambda *args: ({}, []))
    with pytest.raises(RuntimeError, match="isolated_cell"):
        worker._aggregate_isolated_cell_results(authority=authority, inputs=inputs,
            child_results=children, output_root=tmp_path, construction_lineage_mode="audit")
