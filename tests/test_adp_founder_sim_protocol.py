from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp_founder_sim_protocol import (
    APPROVAL_SCHEMA_VERSION,
    PROTOCOL_ID,
    FounderSimProtocolError,
    admit_founder_sim_execution,
    build_founder_approval_receipt,
    build_founder_sim_protocol,
    expected_founder_approval_statement,
)
from blueprint_pipeline.adp_prospective_design import validate_schedule_for_execution


def test_protocol_freezes_one_sim_task_two_real_candidates_and_no_physical_claim() -> None:
    protocol = build_founder_sim_protocol()
    assert protocol["decision"]["baseline_candidate_id"] == ("pi05_droid_jointpos_polaris")
    assert protocol["decision"]["alternative_candidate_id"] == ("nvidia/GR00T-N1.7-DROID")
    assert [row["role"] for row in protocol["candidates"]] == [
        "baseline",
        "alternative",
    ]
    assert protocol["scene"]["simready_asset_generation_required"] is False
    assert protocol["scene"]["interaction_assets"]["local_control_robot_source"]["revision"] == (
        "71f066ad0be9cd271f7ed58c030243ef157af9f4"
    )
    assert protocol["scene"]["simulator_stack"]["physics_backend"] == "PhysX"
    assert protocol["scene"]["simulator_stack"]["environment_type"] == ("ManagerBasedRLEnv")
    assert protocol["scene"]["simulator_stack"]["arena_environment"] == (
        "pick_and_place_maple_table"
    )
    assert protocol["scene"]["scenario_variation_policy"]["scene_cousins_in_this_protocol"] is (
        False
    )
    assert protocol["claim_boundary"]["development_only"] is True
    assert protocol["claim_boundary"]["physical_holdout"] is False
    assert protocol["execution_state"]["production_simulation_started"] is False


def test_protocol_schedule_is_power_consistent_and_exactly_eighty_eight_episodes() -> None:
    protocol = build_founder_sim_protocol()
    schedule = protocol["schedule"]
    admission = validate_schedule_for_execution(schedule)
    assert schedule["power_requirement"]["minimum_trials_per_candidate"] == 44
    assert schedule["repetitions_per_candidate_condition"] == 44
    assert schedule["trials_per_candidate"] == 44
    assert schedule["total_trial_budget"] == 88
    assert admission == protocol["schedule_admission"]
    assert len({row["reset_digest"] for row in protocol["conditions"]}) == 1
    for offset in range(0, len(schedule["rows"]), 2):
        pair = schedule["rows"][offset : offset + 2]
        assert {row["candidate_role"] for row in pair} == {"baseline", "alternative"}
        assert (
            len(
                {
                    (
                        row["condition_id"],
                        row["reset_digest"],
                        row["repetition"],
                        row["seed"],
                    )
                    for row in pair
                }
            )
            == 1
        )


def test_protocol_is_deterministic_and_requires_exact_founder_digest() -> None:
    first = build_founder_sim_protocol()
    second = build_founder_sim_protocol()
    assert first == second
    approval = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approved": True,
        "approver_role": "blueprint_founder_sim_owner",
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": first["protocol_digest"],
    }
    admitted = admit_founder_sim_execution(first, approval)
    assert admitted["status"].startswith("founder_approved")
    assert admitted["physical_execution_authorized"] is False
    assert admitted["paid_compute_authorized"] is False

    changed = copy.deepcopy(approval)
    changed["protocol_digest"] = "sha256:" + "0" * 64
    with pytest.raises(FounderSimProtocolError, match="approval_protocol_digest_mismatch"):
        admit_founder_sim_execution(first, changed)


def test_any_protocol_change_invalidates_canonical_admission() -> None:
    protocol = build_founder_sim_protocol()
    approval = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approved": True,
        "approver_role": "blueprint_founder_sim_owner",
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol["protocol_digest"],
    }
    protocol["task"]["termination"]["maximum_action_steps"] = 599
    with pytest.raises(FounderSimProtocolError, match="protocol_not_canonical"):
        admit_founder_sim_execution(protocol, approval)


def test_observed_exact_digest_statement_compiles_to_durable_approval_receipt() -> None:
    statement = expected_founder_approval_statement()
    receipt = build_founder_approval_receipt(
        statement=statement,
        evidence_ref="codex-task://019fccab-db18-78d2-8f11-1e8a55076c2f/user-message",
    )

    assert receipt["approval_statement"] == statement
    assert receipt["approval_statement_sha256"]
    assert receipt["approved_scope"] == "development_only_simulation"
    assert receipt["physical_execution_authorized"] is False
    assert receipt["uncapped_paid_compute_authorized"] is False
    admitted = admit_founder_sim_execution(build_founder_sim_protocol(), receipt)
    assert admitted["approval"] == receipt


def test_approval_receipt_rejects_paraphrase_or_missing_evidence_reference() -> None:
    with pytest.raises(FounderSimProtocolError, match="approval_statement_not_exact"):
        build_founder_approval_receipt(
            statement="I broadly approve this simulation.",
            evidence_ref="codex-task://observed",
        )
    with pytest.raises(FounderSimProtocolError, match="approval_evidence_ref_missing"):
        build_founder_approval_receipt(
            statement=expected_founder_approval_statement(),
            evidence_ref="",
        )
