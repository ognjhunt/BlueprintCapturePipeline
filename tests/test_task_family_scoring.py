"""Beta families score native state, sealed task thresholds and ordered work."""
from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_scoring import (
    CONTRACT_SCHEMA,
    FAMILY_ALIASES,
    TELEMETRY_SCHEMA,
    TaskFamilyScoringError,
    canonical_task_family,
    score_task_family_episode,
    seal_task_family_contract,
    validate_task_family_contract,
)

_RUNTIME = "sha256:" + "a" * 64
_SCOPE = {"scene_id": "site-one", "task_id": "job-one", "task_context_digest": "sha256:" + "b" * 64}
_CASES = {
    "pouring": (
        {"minimum_transferred_volume_ml": 40., "maximum_spilled_volume_ml": 3., "maximum_volume_balance_error_ml": .01,
         "maximum_final_in_flight_volume_ml": .1},
        {"source_volume_ml": 100., "receiver_volume_ml": 0., "in_flight_volume_ml": 0., "spilled_volume_ml": 0.},
        {"source_volume_ml": 49., "receiver_volume_ml": 50., "in_flight_volume_ml": 0., "spilled_volume_ml": 1.},
    ),
    "wiping": (
        {"minimum_cleaned_fraction": .95, "maximum_remaining_dirty_area_m2": .005},
        {"dirty_area_m2": .1}, {"dirty_area_m2": .001},
    ),
    "insertion": (
        {"minimum_insertion_depth_m": .04, "maximum_lateral_error_m": .002,
         "maximum_alignment_error_rad": .02, "maximum_contact_force_n": 10.},
        {"insertion_depth_m": 0., "lateral_error_m": .01, "alignment_error_rad": .1, "contact_force_n": 0.},
        {"insertion_depth_m": .05, "lateral_error_m": .001, "alignment_error_rad": .01, "contact_force_n": 5.},
    ),
    "stacking": (
        {"minimum_support_margin_m": .01, "maximum_linear_speed_m_s": .01, "maximum_angular_speed_rad_s": .05},
        {"support_contact": False, "com_support_margin_m": -.1, "linear_speed_m_s": 0., "angular_speed_rad_s": 0., "robot_contact_active": False},
        {"support_contact": True, "com_support_margin_m": .05, "linear_speed_m_s": 0., "angular_speed_rad_s": 0., "robot_contact_active": False},
    ),
    "button_press": (
        {"minimum_button_travel_m": .004, "maximum_contact_force_n": 5.},
        {"button_travel_m": 0., "button_activated": False, "contact_force_n": 0.},
        {"button_travel_m": .005, "button_activated": True, "contact_force_n": 2.},
    ),
    "cloth_folding": (
        {"maximum_fold_keypoint_error_m": .01, "minimum_fold_overlap_fraction": .9, "maximum_linear_speed_m_s": .01, "require_release": True},
        {"fold_keypoint_error_m": .2, "fold_overlap_fraction": .1, "linear_speed_m_s": 0., "robot_contact_active": False},
        {"fold_keypoint_error_m": .005, "fold_overlap_fraction": .95, "linear_speed_m_s": 0., "robot_contact_active": False},
    ),
}


def _stage(kind, key="work"):
    return {"stage_id": key, "task_family": kind, "criteria": copy.deepcopy(_CASES[kind][0]),
            "minimum_hold_seconds": .2}


def _contract(kind):
    return seal_task_family_contract(task_family=kind, scope=_SCOPE, stages=[_stage(kind)],
                                    reset={"work": _CASES[kind][1]}, runtime_identity_digest=_RUNTIME,
                                    maximum_sample_gap_seconds=.11)


def _telemetry(contract, rows):
    trace = {"schema_version": TELEMETRY_SCHEMA, "source": "native_simulator_state",
             "scope": contract["scope"], "contract_digest": contract["contract_digest"],
             "runtime_identity_digest": _RUNTIME,
             "samples": [{"time_seconds": i / 10, "measurements": copy.deepcopy(row), "safety_violation": False}
                         for i, row in enumerate(rows)]}
    trace["telemetry_digest"] = canonical_digest(trace, digest_field="telemetry_digest")
    return trace


def _trace(kind):
    contract = _contract(kind)
    return contract, _telemetry(contract, [{"work": _CASES[kind][1]}] + [{"work": _CASES[kind][2]}] * 4)


def _reseal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


@pytest.mark.parametrize("kind", _CASES)
def test_each_family_scores_native_quantities_and_preserves_evidence_boundary(kind):
    contract, trace = _trace(kind)
    result = score_task_family_episode(contract=contract, telemetry=trace)
    assert result["task_succeeded"] is True
    assert result["claim_ceiling"] == "development_only"
    assert result["physical_validation"] is False
    assert result["scoring_source"] == "deterministic_native_state"
    assert result["stage_completions"][0]["time_seconds"] == .3
    assert result["report_digest"] == canonical_digest(result, digest_field="report_digest")


@pytest.mark.parametrize("kind", _CASES)
def test_zero_action_native_state_fails_even_with_claimed_policy_success(kind):
    contract = _contract(kind)
    trace = _telemetry(contract, [{"work": _CASES[kind][1]}] * 5)
    trace["policy_success"] = True
    result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
    assert result["status"] == "failed"
    assert result["task_succeeded"] is False


@pytest.mark.parametrize("kind", _CASES)
def test_missing_native_metric_is_undetermined_even_with_reward(kind):
    contract, trace = _trace(kind)
    del trace["samples"][2]["measurements"]["work"]
    trace["samples"][2]["success"] = True
    trace["samples"][2]["reward"] = 1
    result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
    assert result["status"] == "undetermined"
    assert result["task_succeeded"] is False
    assert result["readback_gaps"] == ["native_measurements:work:2"]


def test_transient_unsafe_contact_cannot_be_erased_by_safe_terminal_state():
    contract, trace = _trace("insertion")
    trace["samples"][1]["measurements"]["work"]["contact_force_n"] = 50.
    result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
    assert result["task_succeeded"] is False
    assert "contact_force_exceeded:work" in result["failed_criteria"]


def test_pouring_requires_conservation_and_counts_spill_for_whole_episode():
    contract, trace = _trace("pouring")
    trace = copy.deepcopy(trace)
    trace["samples"][1]["measurements"]["work"] = {
        "source_volume_ml": 44., "receiver_volume_ml": 50., "in_flight_volume_ml": 0., "spilled_volume_ml": 6.}
    result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
    assert "spill_limit_exceeded:work" in result["failed_criteria"]
    trace["samples"][1]["measurements"]["work"]["spilled_volume_ml"] = 1.
    result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
    assert "volume_balance_exceeded:work" in result["failed_criteria"]


def test_pouring_stream_in_flight_is_conserved_until_receiver_holds_transfer():
    contract = _contract("pouring")
    flowing = {"source_volume_ml": 49., "receiver_volume_ml": 20.,
               "in_flight_volume_ml": 30., "spilled_volume_ml": 1.}
    trace = _telemetry(contract, [{"work": _CASES["pouring"][1]}, {"work": flowing}]
                       + [{"work": _CASES["pouring"][2]}] * 4)
    result = score_task_family_episode(contract=contract, telemetry=trace)
    assert result["task_succeeded"] is True
    assert result["failed_criteria"] == []


def test_pouring_does_not_complete_while_transfer_still_has_in_flight_material():
    contract = _contract("pouring")
    # Enough material has arrived, and conservation balances, but the stream
    # has not settled into a container or an observed spill yet.
    flowing = {"source_volume_ml": 39., "receiver_volume_ml": 50.,
               "in_flight_volume_ml": 10., "spilled_volume_ml": 1.}
    trace = _telemetry(contract, [{"work": _CASES["pouring"][1]}] + [{"work": flowing}] * 5)
    result = score_task_family_episode(contract=contract, telemetry=trace)
    assert result["task_succeeded"] is False
    assert "stage_incomplete:work" in result["failed_criteria"]
    assert "terminal_state_not_held:work" in result["failed_criteria"]
    assert "volume_balance_exceeded:work" not in result["failed_criteria"]


def test_pouring_escaped_material_fails_even_if_rest_of_stream_reaches_receiver():
    contract = _contract("pouring")
    flowing = {"source_volume_ml": 49., "receiver_volume_ml": 20.,
               "in_flight_volume_ml": 26., "spilled_volume_ml": 5.}
    trace = _telemetry(contract, [{"work": _CASES["pouring"][1]}, {"work": flowing}]
                       + [{"work": _CASES["pouring"][2]}] * 4)
    result = score_task_family_episode(contract=contract, telemetry=trace)
    assert result["task_succeeded"] is False
    assert "spill_limit_exceeded:work" in result["failed_criteria"]
    assert "volume_balance_exceeded:work" not in result["failed_criteria"]


@pytest.mark.parametrize("kind", ("stacking", "cloth_folding"))
def test_settling_must_hold_at_end_not_only_pass_once(kind):
    contract, trace = _trace(kind)
    trace["samples"][-2]["measurements"] = {"work": _CASES[kind][1]}
    result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
    assert result["task_succeeded"] is False
    assert "terminal_state_not_held:work" in result["failed_criteria"]


@pytest.mark.parametrize("kind", ("stacking", "cloth_folding"))
def test_held_in_target_cannot_pass_released_task(kind):
    contract, trace = _trace(kind)
    for sample in trace["samples"][1:]:
        sample["measurements"]["work"]["robot_contact_active"] = True
    result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
    assert result["task_succeeded"] is False
    if kind == "cloth_folding":
        contract["stages"][0]["criteria"]["require_release"] = False
        contract = _reseal(contract, "contract_digest")
        trace["contract_digest"] = contract["contract_digest"]
        result = score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))
        assert result["task_succeeded"] is True


def _multistep():
    stages = [_stage("insertion", "insert"), _stage("button_press", "press")]
    initial = {"insert": _CASES["insertion"][1], "press": _CASES["button_press"][1]}
    contract = seal_task_family_contract(task_family="multi_step", scope=_SCOPE, stages=stages, reset=initial,
                                        runtime_identity_digest=_RUNTIME, maximum_sample_gap_seconds=.11)
    inserted = {**initial, "insert": _CASES["insertion"][2]}
    complete = {**inserted, "press": _CASES["button_press"][2]}
    return contract, initial, inserted, complete


def test_multi_step_requires_order_and_new_completion_after_previous_stage():
    contract, initial, inserted, complete = _multistep()
    correct = _telemetry(contract, [initial] + [inserted] * 3 + [complete] * 3)
    assert score_task_family_episode(contract=contract, telemetry=correct)["task_succeeded"] is True
    # Pressing before insertion cannot satisfy insert-then-press, including
    # when the switch remains down after insertion finishes.
    early_press = {**initial, "press": _CASES["button_press"][2]}
    reversed_trace = _telemetry(contract, [initial] + [early_press] * 3 + [complete] * 4)
    result = score_task_family_episode(contract=contract, telemetry=reversed_trace)
    assert result["task_succeeded"] is False
    assert "stage_incomplete:press" in result["failed_criteria"]


@pytest.mark.parametrize("mutation,code", [
    (lambda c, t: t.update(source="policy_reward"), "source_invalid"),
    (lambda c, t: t.update(runtime_identity_digest="sha256:" + "c" * 64), "binding_mismatch"),
    (lambda c, t: t["scope"].update(task_id="different-job"), "binding_mismatch"),
    (lambda c, t: t["samples"][2].update(time_seconds=.1), "sample_time_invalid"),
    (lambda c, t: t["samples"][-1].update(time_seconds=5), "sample_gap_exceeded"),
    (lambda c, t: t["samples"][0]["measurements"]["work"].update(dirty_area_m2=.2), "reset_mismatch"),
])
def test_identity_source_time_and_reset_are_validated(mutation, code):
    contract, trace = _trace("wiping")
    trace = copy.deepcopy(trace)
    mutation(contract, trace)
    with pytest.raises(TaskFamilyScoringError, match=code):
        score_task_family_episode(contract=contract, telemetry=_reseal(trace, "telemetry_digest"))


def test_tampered_contract_and_nonfinite_native_state_never_score():
    contract, trace = _trace("wiping")
    contract["stages"][0]["criteria"]["minimum_cleaned_fraction"] = .1
    with pytest.raises(TaskFamilyScoringError, match="contract_digest_invalid"):
        score_task_family_episode(contract=contract, telemetry=trace)
    contract, trace = _trace("wiping")
    trace["samples"][1]["measurements"]["work"]["dirty_area_m2"] = float("nan")
    with pytest.raises(TaskFamilyScoringError, match="not_finite_json"):
        score_task_family_episode(contract=contract, telemetry=trace)


def test_aliases_resolve_concrete_profiles_without_generic_deformable_claim():
    for family, aliases in FAMILY_ALIASES.items():
        assert all(canonical_task_family(alias) == family for alias in aliases)
    assert canonical_task_family("deformable") is None
    assert canonical_task_family("humanoid") is None


def test_contract_cannot_reset_to_already_completed_work_or_skip_required_criteria():
    contract = _contract("button_press")
    contract["reset"]["work"] = _CASES["button_press"][2]
    with pytest.raises(TaskFamilyScoringError, match="reset_already_satisfied"):
        validate_task_family_contract(_reseal(contract, "contract_digest"))
    contract = _contract("insertion")
    del contract["stages"][0]["criteria"]["maximum_contact_force_n"]
    with pytest.raises(TaskFamilyScoringError, match="criteria_invalid"):
        validate_task_family_contract(_reseal(contract, "contract_digest"))
    assert contract["schema_version"] == CONTRACT_SCHEMA
