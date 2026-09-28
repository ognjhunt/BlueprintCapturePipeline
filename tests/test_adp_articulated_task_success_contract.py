"""The frozen open/close success definition: a joint interval, held, with the assembly still put."""
from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp_articulated_task_success_contract import (
    SCHEMA_VERSION,
    compatibility_articulated_success_criteria,
    confirm_articulated_task_success_contract,
    confirmed_articulated_task_success_contract_matches_published,
    confirmed_task_success_contract_matches_published,
    seal_articulated_task_success_contract,
    seal_task_success_contract,
    task_kind_of_contract,
    task_success_contract_schema_version,
    validate_articulated_task_success_contract,
    validate_task_success_contract,
)
from blueprint_pipeline.adp_task_scoring import (
    RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION,
    TaskNeutralScoringError,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.articulation_graph_contract import validate_articulation_graph

SITE, TASK = "site-capture-drawer", "website-drawer-open"


def _graph(*, interval=(0.18, 0.3), stroke=0.3):
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "carcass", "is_root": True, "semantic_role": "cabinet_carcass"},
            {"link_id": "drawer_1", "is_root": False, "semantic_role": "task_drawer"},
            {"link_id": "drawer_0", "is_root": False, "semantic_role": "fixed_drawer"},
        ],
        "joints": [
            {"joint_id": "task_part_joint", "parent_link_id": "carcass", "child_link_id": "drawer_1",
             "joint_type": "prismatic", "role": "target", "axis": [1, 0, 0], "limits": [0.0, stroke],
             "reset_position": 0.0, "reset_tolerance": 0.005,
             "drive": {"drive_type": "none", "stiffness": 0.0, "damping": 5.0, "maximum_force": 0.0}},
            {"joint_id": "drawer_0_fixed", "parent_link_id": "carcass", "child_link_id": "drawer_0",
             "joint_type": "fixed", "role": "locked", "axis": [0, 0, 0], "limits": [0.0, 0.0],
             "reset_position": 0.0, "reset_tolerance": 0.005,
             "drive": {"drive_type": "none", "stiffness": 0.0, "damping": 0.0, "maximum_force": 0.0}},
        ],
        "collision_pairs": [],
        "success_predicate": {"combination": "all",
                              "joint_intervals": {"task_part_joint": list(interval)}},
    }


def _spec(**overrides):
    return {"schema_version": "adp_task_spec.v2", "task_kind": "articulated_open_close",
            "articulation_graph": validate_articulation_graph(_graph()),
            "settle_window_samples": 15, "maximum_settled_target_speed": 0.02,
            "locked_joint_motion_tolerance": 0.01, "movement_epsilon": 0.003, **overrides}


def _sealed(**overrides):
    return seal_articulated_task_success_contract(
        task_spec=_spec(), site_id=SITE, task_id=TASK,
        author_source="compatibility_default", author_id="blueprint:articulated_defaults.v1",
        confirmation_status="confirmed", confirmed_by_team_id="team-1", **overrides)


def _reseal(contract):
    contract["contract_digest"] = ""
    contract["contract_digest"] = cross_runtime_canonical_digest(contract, digest_field="contract_digest")
    return contract


def test_the_default_criteria_are_a_translation_of_the_frozen_spec():
    criteria = compatibility_articulated_success_criteria(_spec())
    assert criteria["target_joint"]["joint_id"] == "task_part_joint"
    assert criteria["opening"] == {"mode": "required", "success_interval": [0.18, 0.3],
                                   "joint_hard_limits": [0.0, 0.3], "reset_position": 0.0}
    assert criteria["hold"] == {"mode": "required", "window_samples": 15,
                                "maximum_settled_target_speed": 0.02}
    assert criteria["locked_joints"]["joint_ids"] == ["drawer_0_fixed"]
    assert criteria["assembly_root"] == {"mode": "required"}
    assert criteria["safety"] == {"mode": "required"}
    ledger = criteria["temporal_invariants"]
    assert ledger["forbidden_collision_allowed"] is False
    assert ledger["joint_limit_violation_allowed"] is False
    assert ledger["rebound_below_threshold_allowed"] is False
    # No rigid predicate leaks in: there is no destination, lift or placement here.
    assert not {"destination_containment", "orientation", "support", "gripper_state"} & set(criteria)


def test_a_sealed_contract_round_trips_and_binds_its_scope():
    contract = _sealed()
    assert contract["schema_version"] == SCHEMA_VERSION
    assert validate_articulated_task_success_contract(
        contract, expected_site_id=SITE, expected_task_id=TASK) == contract
    with pytest.raises(TaskNeutralScoringError, match="site_binding_mismatch"):
        validate_articulated_task_success_contract(contract, expected_site_id="other-site")
    tampered = _reseal(copy.deepcopy(contract) | {"scope": {"site_id": SITE, "task_id": "other"}})
    with pytest.raises(TaskNeutralScoringError, match="task_binding_mismatch"):
        validate_articulated_task_success_contract(tampered, expected_task_id=TASK)


@pytest.mark.parametrize("mutate,expected", [
    (lambda c: c["criteria"]["opening"].update(success_interval=[0.0, 0.3]), "criteria_opening_invalid"),
    (lambda c: c["criteria"]["opening"].update(success_interval=[0.18, 0.45]), "criteria_opening_invalid"),
    (lambda c: c["criteria"].update(safety={"mode": "ignored"}), "criteria_safety_not_frozen"),
    (lambda c: c["criteria"]["temporal_invariants"].update(forbidden_collision_allowed=True),
     "criteria_temporal_invariants_invalid"),
    (lambda c: c["criteria"]["hold"].update(window_samples=0), "criteria_hold_invalid"),
    (lambda c: c["criteria"].pop("assembly_root"), "criteria_missing_field:assembly_root"),
])
def test_a_relaxed_or_unopenable_predicate_is_refused(mutate, expected):
    contract = copy.deepcopy(_sealed())
    mutate(contract)
    _reseal(contract)
    with pytest.raises(TaskNeutralScoringError, match=expected):
        validate_articulated_task_success_contract(contract)


def test_an_undigested_or_unconfirmed_contract_is_refused():
    contract = copy.deepcopy(_sealed())
    contract["criteria"]["motion"]["movement_epsilon"] = 0.05
    with pytest.raises(TaskNeutralScoringError, match="digest_mismatch"):
        validate_articulated_task_success_contract(contract)
    proposal = seal_articulated_task_success_contract(
        task_spec=_spec(), site_id=SITE, task_id=TASK, author_source="agent_proposal",
        author_id="agent", confirmation_status="proposal_only")
    with pytest.raises(TaskNeutralScoringError, match="not_confirmed"):
        validate_articulated_task_success_contract(proposal)
    with pytest.raises(TaskNeutralScoringError, match="agent_must_originate_proposal"):
        seal_articulated_task_success_contract(
            task_spec=_spec(), site_id=SITE, task_id=TASK, author_source="agent_proposal",
            author_id="agent", confirmation_status="confirmed", confirmed_by_team_id="team-1")


def test_a_team_confirms_exactly_the_published_proposal():
    proposal = seal_articulated_task_success_contract(
        task_spec=_spec(), site_id=SITE, task_id=TASK, author_source="agent_proposal",
        author_id="agent", confirmation_status="proposal_only")
    confirmed = confirm_articulated_task_success_contract(proposal, confirmed_by_team_id="team-1")
    assert confirmed["provenance"]["proposal_digest"] == proposal["contract_digest"]
    assert confirmed_articulated_task_success_contract_matches_published(
        published=proposal, selected=confirmed)
    # A team cannot confirm different criteria and present it as the same contract.
    altered = copy.deepcopy(confirmed)
    altered["criteria"]["opening"]["success_interval"] = [0.25, 0.3]
    _reseal(altered)
    assert not confirmed_articulated_task_success_contract_matches_published(
        published=proposal, selected=altered)


def test_the_facade_dispatches_on_task_kind_without_touching_the_rigid_lane():
    contract = _sealed()
    assert task_kind_of_contract(contract) == "articulated_open_close"
    assert task_success_contract_schema_version("articulated_open_close") == SCHEMA_VERSION
    assert (task_success_contract_schema_version("rigid_pick_place")
            == RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION)
    assert validate_task_success_contract(contract, task_kind="articulated_open_close") == contract
    assert confirmed_task_success_contract_matches_published(
        task_kind="articulated_open_close", published=contract, selected=contract)
    # The rigid branch still reaches the rigid validator, which refuses this one.
    with pytest.raises(TaskNeutralScoringError, match="rigid_task_success_contract"):
        validate_task_success_contract(contract, task_kind="rigid_pick_place")
    sealed_through_facade = seal_task_success_contract(
        task_kind="articulated_open_close", task_spec=_spec(), site_id=SITE, task_id=TASK,
        author_source="task_owner", author_id="owner", confirmation_status="confirmed",
        confirmed_by_team_id="team-1")
    assert sealed_through_facade["schema_version"] == SCHEMA_VERSION
