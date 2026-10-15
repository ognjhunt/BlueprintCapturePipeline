"""Admission of ONE team candidate to a practice run or a scored evaluation fails closed on every gate.

Beta journey step: L5 team policy evaluation. Control plane only: no network,
provider, GPU or spend. The scene is articulated because only articulated
scenes have a robot-free success probe; the receipt comes from the real scorer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from blueprint_pipeline.adp_articulated_task_success_contract import seal_articulated_task_success_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_quick10_admission import (
    TeamCandidateAdmissionError,
    admit_team_candidate,
    execution_spec,
    runtime_binding,
)
from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
from tests.test_robot_free_probe_gate import REVISION, scored_probe_receipt
from tests.test_task_evaluation_scene_robot_free_readiness import _spec

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
ORIGIN = "https://policy.acme-robotics.example"


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _contract() -> dict[str, Any]:
    return seal_articulated_task_success_contract(
        task_spec=_spec(), site_id="site-dishwasher", task_id="open-dishwasher-door",
        author_source="compatibility_default", author_id="blueprint:articulated_defaults.v1",
        confirmation_status="confirmed", confirmed_by_team_id="team-acme")


def _inputs(**overrides: Any) -> dict[str, Any]:
    candidate = {"schema_version": "blueprint.team_policy_candidate.v1", "team_id": "team-acme",
                 "checkpoint_id": "pi05-ft-7", "candidate_id": "team-acme.pi05-ft-7",
                 "embodiment_preset_id": "droid_franka_panda_robotiq_2f85_v1",
                 "policy_interface_id": "openpi_droid_https.v1",
                 "delivery": {"mode": "https_policy", "endpoint_url": ORIGIN + "/v1/act",
                              "metadata_url": ORIGIN + "/v1/metadata", "action_chunk_rows": 10,
                              "open_loop_horizon": 8, "model_identity_digest": "sha256:" + "7" * 64,
                              "team_policy_id": "acme-pi05-ft-7"}}
    rights = _sealed({"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
                      "team_id": "team-acme", "candidate_id": "team-acme.pi05-ft-7", "checkpoint_id": "pi05-ft-7",
                      "license_to_run_granted": True, "policy_self_grading_permitted": False,
                      "official_ranking_permitted": False, "secret_material_recorded": False,
                      "adp_qualified_evidence_permitted": False}, "rights_digest")
    lease = {"schema_version": "blueprint.team_policy_credential_lease_availability.v1", "kind": "kms_bearer_lease",
             "status": "available", "single_use": True, "lease_id": "lease-1",
             "credential_ref_digest": "sha256:" + "6" * 64, "credential_env": "BLUEPRINT_TEAM_POLICY_BEARER_FILE",
             "origin": ORIGIN, "candidate_id": "team-acme.pi05-ft-7", "secret_material_recorded": False,
             "expires_at_iso": "2026-10-15T18:00:00Z"}
    budget = {"payer": "blueprint", "customer_price_usd": 0, "max_attempts": 1, "episodes": 200,
              "cap_usd": 20, "hard_ttl_seconds": 9000}
    founder = _sealed({"schema_version": "blueprint.team_candidate_founder_admission.v1",
                       "approved_by": "founder", "approval_reference": "owner-decision-team-acme-quick10",
                       "expires_at_iso": "2026-10-16T00:00:00Z",
                       "scope": {"team_id": "team-acme", "candidate_id": "team-acme.pi05-ft-7",
                                 "scene_revision_digest": REVISION, "run_kind": "team_candidate_evaluation"}},
                      "approval_digest")
    scene = {"scene_revision_digest": REVISION, "task_spec": _spec(), "task_success_contract": _contract(),
             "embodiment_preset_id": "droid_franka_panda_robotiq_2f85_v1"}
    value = {"candidate": candidate, "rights": rights, "credential_lease": lease, "budget": budget,
             "founder_admission": founder, "scene": scene, "probe_receipt": scored_probe_receipt(REVISION)}
    value.update(overrides)
    return value


def _blockers(**overrides: Any) -> list[str]:
    with pytest.raises(TeamCandidateAdmissionError) as raised:
        admit_team_candidate(**_inputs(**overrides), now=NOW)
    return raised.value.blockers


def test_an_admitted_team_candidate_seals_exactly_what_the_runtime_may_run(tmp_path, monkeypatch) -> None:
    values = _inputs()
    admission = admit_team_candidate(**values, now=NOW)
    assert admission["status"] == "admitted" and admission["runtime"] == "team_evaluation"
    assert admission["evaluation_purpose"] == "scored" and admission["episodes"] == 200
    assert admission["claim_ceiling"] == "diagnostic_policy_execution"
    assert admission["adp_qualified_evidence_eligible"] is False
    assert admission["robot_free_probe_receipt_digest"] == values["probe_receipt"]["receipt_digest"]
    assert admission["task_success_contract_digest"] == values["scene"]["task_success_contract"]["contract_digest"]
    assert admission["admission_digest"] == canonical_digest(admission, digest_field="admission_digest")
    spec = execution_spec(admission, task_success_contract=values["scene"]["task_success_contract"],
                          prompt="open the dishwasher door", max_policy_queries=40)
    assert spec["candidate_id"] == "team-acme.pi05-ft-7" and spec["policy_interface_id"] == "openpi_droid_https.v1"
    assert spec["execution_spec_digest"] == canonical_digest(spec, digest_field="execution_spec_digest")
    token = tmp_path / "team-policy-bearer"
    token.write_text("team-secret-token\n", encoding="utf-8")
    token.chmod(0o644)
    monkeypatch.setenv("BLUEPRINT_TEAM_POLICY_BEARER_FILE", str(token))
    with pytest.raises(ValueError):
        canary_policy_client(spec)  # a group- or world-readable token file is refused
    token.chmod(0o600)
    client = canary_policy_client(spec)
    assert type(client).__name__ == "DroidHttpsTeamPolicyClient"
    assert "team-secret-token" not in str(spec)
    binding = runtime_binding(admission)
    assert binding["admission_digest"] == admission["admission_digest"]
    assert set(binding) == {"schema_version", "admission_digest", "candidate_id", "evaluation_purpose",
                            "robot_free_probe_receipt_digest", "scene_revision_digest",
                            "task_success_contract_digest"}


def test_every_gate_fails_closed_with_its_own_reason() -> None:
    base = _inputs()
    assert _blockers(rights={**base["rights"], "license_to_run_granted": False}) == [
        "team_candidate_rights_record_missing_or_mismatched"]
    assert _blockers(rights=_sealed({**base["rights"], "license_to_run_granted": False}, "rights_digest")) == [
        "team_candidate_rights_scope_insufficient"]
    assert _blockers(rights=_sealed({**base["rights"], "adp_qualified_evidence_permitted": True},
                                    "rights_digest")) == ["team_candidate_rights_scope_insufficient"]
    assert _blockers(credential_lease={**base["credential_lease"], "status": "claimed"}) == [
        "team_candidate_credential_lease_unavailable"]
    assert _blockers(credential_lease={**base["credential_lease"], "expires_at_iso": "2026-10-15T11:00:00Z"}) == [
        "team_candidate_credential_lease_unavailable"]
    assert _blockers(credential_lease={**base["credential_lease"], "origin": "https://other.example"}) == [
        "team_candidate_credential_lease_unavailable"]
    assert _blockers(budget={**base["budget"], "cap_usd": 21}) == ["team_candidate_budget_cap_invalid"]
    assert _blockers(budget={**base["budget"], "episodes": 1}) == ["team_candidate_budget_cap_invalid"]
    assert _blockers(budget={**base["budget"], "max_attempts": 2}) == ["team_candidate_budget_cap_invalid"]
    assert _blockers(founder_admission={**base["founder_admission"], "approved_by": ""}) == [
        "team_candidate_founder_admission_missing"]
    expired = _sealed({**base["founder_admission"], "expires_at_iso": "2026-10-15T00:00:00Z"}, "approval_digest")
    assert _blockers(founder_admission=expired) == ["team_candidate_founder_admission_missing"]
    other_scene = _sealed({**base["founder_admission"], "scope": {
        **base["founder_admission"]["scope"], "scene_revision_digest": "sha256:" + "8" * 64}}, "approval_digest")
    assert _blockers(founder_admission=other_scene) == ["team_candidate_founder_admission_missing"]


def test_the_robot_free_probe_is_mandatory_for_this_scene_revision_and_task() -> None:
    assert _blockers(probe_receipt=None) == ["team_evaluation_robot_free_probe_receipt_missing"]
    # Practice is feedback: it may run while the probe is pending, never after it failed.
    practice_budget = {**_inputs()["budget"], "episodes": 10}
    practice = admit_team_candidate(**_inputs(probe_receipt=None, budget=practice_budget), purpose="practice", now=NOW)
    assert practice["evaluation_purpose"] == "practice" and practice["robot_free_probe_receipt_digest"] is None
    with pytest.raises(TeamCandidateAdmissionError) as raised:
        admit_team_candidate(**_inputs(probe_receipt=scored_probe_receipt(REVISION, passes=False),
                                       budget=practice_budget), purpose="practice", now=NOW)
    assert raised.value.blockers == ["team_evaluation_robot_free_probe_not_passed"]
    assert _blockers(budget=practice_budget) == ["team_candidate_budget_cap_invalid"]  # a scored run is 200
    assert _blockers(probe_receipt=scored_probe_receipt(REVISION, passes=False)) == [
        "team_evaluation_robot_free_probe_not_passed"]
    assert _blockers(probe_receipt=scored_probe_receipt("sha256:" + "8" * 64)) == [
        "team_evaluation_robot_free_probe_scene_revision_mismatch"]
    rigid = {**_inputs()["scene"], "task_spec": {**_spec(), "task_kind": "rigid_pick_place"}}
    assert "team_evaluation_robot_free_probe_task_kind_mismatch" in _blockers(scene=rigid)


def test_interface_and_delivery_mismatches_are_refused_by_name() -> None:
    base = _inputs()
    candidate = base["candidate"]
    assert _blockers(candidate={**candidate, "embodiment_preset_id": "unitree_g1"}) == [
        "team_candidate_interface_mismatch:embodiment", "team_candidate_interface_mismatch:scene_embodiment"]
    assert _blockers(candidate={**candidate, "policy_interface_id": "groot_n17_droid.v1"}) == [
        "team_candidate_interface_mismatch:delivery"]
    # No runtime serves a team's uploaded weights on the GPU yet; its HTTPS lease cannot apply either.
    assert _blockers(candidate={**candidate, "delivery": {"mode": "private_checkpoint"}}) == [
        "team_candidate_credential_lease_unavailable", "team_candidate_private_checkpoint_runtime_unavailable"]
    assert _blockers(candidate={**candidate, "candidate_id": "pi05_droid"}) == sorted([
        "team_candidate_cannot_impersonate_frozen_candidate",
        "team_candidate_credential_lease_unavailable", "team_candidate_founder_admission_missing",
        "team_candidate_rights_record_missing_or_mismatched"])
    plain = {**candidate["delivery"], "endpoint_url": "http://policy.acme-robotics.example/v1/act"}
    assert "team_candidate_endpoint_invalid" in _blockers(candidate={**candidate, "delivery": plain})
    wide = {**candidate["delivery"], "action_chunk_rows": 50}
    assert _blockers(candidate={**candidate, "delivery": wide}) == ["team_candidate_interface_mismatch:action_chunk"]
    contract = base["scene"]["task_success_contract"]
    tampered = {**base["scene"], "task_success_contract": {**contract, "criteria": {
        **contract["criteria"], "assembly_root": "changed-after-sealing"}}}
    assert _blockers(scene=tampered) == ["team_candidate_scene_success_contract_invalid"]
