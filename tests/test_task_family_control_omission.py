"""Pure replay of typed control omission; fixtures are no runtime evidence."""
from __future__ import annotations

import copy

import pytest

from blueprint_pipeline import task_evaluation_control_stage_policy as policy
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_control_omission import (
    OMITTED_CONTROLS, make_family_control_omission, validate_family_control_omission,
)
from blueprint_pipeline.task_family_result_admission import (
    family_policy_episode_interpretable, family_site_qualification_valid,
)
from blueprint_pipeline.task_family_scoring import score_task_family_episode, seal_task_family_contract

SHA = "sha256:" + "a" * 64
PROVENANCE = {"native_runtime_contract_digest": SHA, "scenario_cell_id": "cell1",
    "scenario_instance_digest": SHA, "seed": 0}


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _task():
    return seal_task_family_contract(task_family="button_press",
        scope={"scene_id": "scene1", "task_id": "task1", "task_context_digest": SHA},
        stages=[{"stage_id": "button", "task_family": "button_press", "minimum_hold_seconds": 0,
            "criteria": {"minimum_button_travel_m": .01, "maximum_contact_force_n": 20}}],
        reset={"button": {"button_travel_m": 0, "button_activated": False, "contact_force_n": 0}},
        runtime_identity_digest=SHA)


def _episode(task, *, succeeded):
    measurements = [copy.deepcopy(task["reset"]["button"]) for _ in range(3)]
    if succeeded:
        for row in measurements[1:]:
            row.update(button_travel_m=.02, button_activated=True)
    trace = {"schema_version": "task_family_telemetry.v1", "source": "native_simulator_state",
        "scope": task["scope"], "contract_digest": task["contract_digest"], "runtime_identity_digest": SHA,
        "native_provenance": PROVENANCE, "samples": [{"time_seconds": index,
            "measurements": {"button": row}, "safety_violation": False, "native_provenance": PROVENANCE}
            for index, row in enumerate(measurements)]}
    _seal(trace, "telemetry_digest")
    return _seal({"schema_version": "task_family_native_episode.v1", "status": "completed",
        "runtime_closed": True, "failure": None, "scoring_source": "deterministic_native_state",
        "physical_validation": False, "native_provenance": PROVENANCE, "maximum_steps": 2,
        "visual_evidence": {"status": "complete"}, "telemetry": trace,
        "scoring": score_task_family_episode(contract=task, telemetry=trace), "task_succeeded": succeeded}, "receipt_digest")


def _states(*, robot_present, delivered):
    return [{"simulation_time_seconds": index, "native_provenance": PROVENANCE,
        "safety_violation": False, "robot_present": robot_present,
        "robot_action_applied": delivered and index > 0} for index in range(3)]


def _site(task, *, omitted):
    controls = {}
    for name in (["robot_free_probe"] if omitted else ["robot_free_probe", *OMITTED_CONTROLS]):
        positive = name == "deterministic_scripted_positive"
        controls[name] = {"passed": True, "episode": _episode(task, succeeded=positive),
            "native_state_trace": _states(robot_present=name != "robot_free_probe", delivered=positive)}
    value = {"schema_version": "task_family_native_site_qualification.v1", "status": "passed", "failure": None,
        "candidate_policy_queried": False, "scoring_source": "deterministic_native_state", "physical_validation": False,
        "task_success_contract_digest": task["contract_digest"], "native_provenance": PROVENANCE, "controls": controls}
    if omitted:
        value.update(control_omission_authority=make_family_control_omission(task["contract_digest"]),
            omitted_controls=list(OMITTED_CONTROLS), controls_qualified=False, qualified_comparison_permitted=False)
    return _seal(value, "receipt_digest")


def test_standing_pause_mints_only_existing_typed_omission_and_keeps_scoring_unchanged(monkeypatch):
    from blueprint_pipeline.native_task_arena_policy_canary_session import validate_control_omission_authority

    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    authority = make_family_control_omission(SHA)
    assert validate_family_control_omission(authority, SHA)
    assert validate_control_omission_authority(authority, contract_digest=SHA) == authority
    assert authority["authorized_by"] == policy.AUTHORIZED_BY
    assert authority["authorization_reference"] == policy.AUTHORIZATION_REFERENCE
    assert authority["user_request"] == policy.USER_REQUEST
    assert authority["source_task_success_contract_digest"] == authority["result_task_success_contract_digest"] == SHA
    assert authority["task_scoring_criteria_changed"] is False
    assert authority["qualified_comparison_permitted"] is False
    assert authority["omitted_controls"] == list(OMITTED_CONTROLS)
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", False)
    assert make_family_control_omission(SHA) is None
    assert validate_family_control_omission(authority, SHA)  # historical read, no trial execution


def test_existing_typed_authority_without_explanatory_request_remains_readable(monkeypatch):
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    authority = make_family_control_omission(SHA)
    authority.pop("user_request")
    _seal(authority, "authority_digest")
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", False)
    assert validate_family_control_omission(authority, SHA)


@pytest.mark.parametrize("field,value", [
    ("authorized_by", "site-owner"), ("authorization_reference", "unrelated-instruction"),
    ("user_request", "skip all physics"), ("omitted_controls", ["robot_free_probe"]),
    ("source_task_success_contract_digest", "sha256:" + "b" * 64),
    ("result_task_success_contract_digest", "sha256:" + "b" * 64),
    ("task_scoring_criteria_changed", True), ("qualified_comparison_permitted", True),
    ("claim_ceiling", "qualified_evaluation"), ("run_kind", "arbitrary"),
])
def test_outer_resealed_authority_cannot_change_pause_scope_or_task(monkeypatch, field, value):
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    authority = make_family_control_omission(SHA)
    authority[field] = value
    _seal(authority, "authority_digest")
    assert not validate_family_control_omission(authority, SHA)


def test_invalid_or_wrong_digest_never_authorizes_omission(monkeypatch):
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    with pytest.raises(ValueError, match="contract_digest_invalid"):
        make_family_control_omission("not-a-digest")
    authority = make_family_control_omission(SHA)
    assert not validate_family_control_omission(authority, "sha256:" + "b" * 64)
    authority["authority_digest"] = "sha256:" + "0" * 64
    assert not validate_family_control_omission(authority, SHA)
    assert not validate_family_control_omission(None, SHA)


def test_robot_free_only_site_is_readable_under_valid_omission_after_pause_changes(monkeypatch):
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    task = _task()
    site = _site(task, omitted=True)
    assert family_site_qualification_valid(task, site)
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", False)
    assert family_site_qualification_valid(task, site)
    assert set(site["controls"]) == {"robot_free_probe"}
    assert site["controls_qualified"] is False


def test_legacy_full_control_proof_remains_readable_while_future_trials_are_paused(monkeypatch):
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    task = _task()
    assert family_site_qualification_valid(task, _site(task, omitted=False))


@pytest.mark.parametrize("fault", ["missing_probe", "missing_authority", "bad_authority", "controls_claim",
    "comparison_claim", "resumed_negative", "omitted_probe", "robot_present", "robot_actuated", "unsafe", "probe_succeeds"])
def test_omission_never_substitutes_for_robot_free_physics_or_claims_control_success(monkeypatch, fault):
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    task = _task()
    site = _site(task, omitted=True)
    probe = site["controls"]["robot_free_probe"]
    if fault == "missing_probe":
        site["controls"] = {}
    elif fault == "missing_authority":
        site.pop("control_omission_authority")
    elif fault == "bad_authority":
        site["control_omission_authority"]["authorized_by"] = "untrusted"
        _seal(site["control_omission_authority"], "authority_digest")
    elif fault == "controls_claim":
        site["controls_qualified"] = True
    elif fault == "comparison_claim":
        site["qualified_comparison_permitted"] = True
    elif fault == "resumed_negative":
        site["controls"]["zero_action_negative"] = _site(task, omitted=False)["controls"]["zero_action_negative"]
    elif fault == "omitted_probe":
        site["omitted_controls"] = ["robot_free_probe"]
    elif fault in {"robot_present", "robot_actuated", "unsafe"}:
        field = {"robot_present": "robot_present", "robot_actuated": "robot_action_applied", "unsafe": "safety_violation"}[fault]
        probe["native_state_trace"][-1][field] = True
    elif fault == "probe_succeeds":
        probe["episode"] = _episode(task, succeeded=True)
    _seal(site, "receipt_digest")
    assert not family_site_qualification_valid(task, site)


@pytest.mark.parametrize("succeeded", [True, False])
def test_diagnostic_policy_score_remains_interpretable_with_typed_omission_and_no_arm_motion(monkeypatch, succeeded):
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", True)
    task = _task()
    native = _episode(task, succeeded=succeeded)
    score = {**native["scoring"], "status": "scored"}
    row = {"status": "completed", "candidate_policy_queried": True, "actions_reached_robot": True,
        "arm_moved": False, "scoring_authority": "deterministic_simulator_state",
        "deterministic_score_digest": canonical_digest({"value": score}), "episode": {
            "native_family_receipt": native, "native_site_qualification": _site(task, omitted=True),
            "score": score, "state_trace": _states(robot_present=True, delivered=True),
            "motion_evidence": {"actions_reached_robot": True, "native_action_steps": 2}}}
    assert family_policy_episode_interpretable(row, task)
    assert row["episode"]["score"]["task_succeeded"] is succeeded
