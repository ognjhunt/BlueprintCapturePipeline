"""Paused controls: a robot team evaluates a job whose owner contract requires per-cell controls, as a diagnostic.

Owner decision 2026-09-21 (docs/website-controls-pause-2026-09-21.md): control
trials are paused for all runs. Beta screening runs such a contract under the
typed diagnostic omission (``beta_policy_screening_control_omission``); a team
evaluation now does the same. The job is built with the website's own record
builders (``per_cell_controls_required`` true), its thresholds confirmed by the
owner, compiled into the team half, probed, listed, admitted, staged and sealed
through the static preflight. Nothing is allocated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import zipfile

import pytest

from blueprint_pipeline import task_evaluation_control_stage_policy as policy
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_policy_canary_control_gate import controls_required
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    PolicyCanarySessionError,
    validate_runtime_input_manifest,
)
from blueprint_pipeline.team_candidate_quick10_admission import (
    TeamCandidateAdmissionError,
    admit_team_candidate,
    runtime_binding,
    validate_admission,
)
from blueprint_pipeline.team_candidate_quick10_registry import offer_eligibility, scene_facts
from blueprint_pipeline.team_candidate_quick10_result import TeamCandidateResultError, team_candidate_terminal
from tests.test_task_evaluation_owner_success_confirmation import confirm
from tests.test_team_candidate_quick10_rehearsal import TEAM, _fake_closed_team_result
from tests.test_team_scene_packet_owner_confirmation import _compile, _job

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
ORIGIN = "https://policy.acme-robotics.example"
STRICT = {"mode": "required_per_cell", "control_ids": ["zero_action_negative", "deterministic_scripted_positive"]}


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _scene(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """The confirmed strict team half, its retained passing probe, and the operator profile reading both."""

    from blueprint_pipeline.rigid_robot_free_probe_plan import physics_plan, probe_context
    from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
    from blueprint_pipeline.rigid_robot_free_readiness_score import score_rigid_robot_free_readiness
    from blueprint_pipeline.robot_free_readiness_retention import retain_receipt
    from tests.test_native_task_arena_bundle import _runtime_source_packet
    from tests.test_rigid_robot_free_probe import FlatSupport, binding

    job, inputs = _job(tmp_path, monkeypatch)
    inputs["runtime_source_receipt_path"] = _runtime_source_packet(tmp_path / "runtime")
    confirm(job)
    compiled = _compile(job, inputs, tmp_path / "confirmed")
    half = json.loads(Path(compiled["path"]).read_text())
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    profile = {"capture_root": str(tmp_path / "capture"), "scene_id": "site-paused-controls",
               "task_id": compiled["task_id"], "team_scene_inputs_path": compiled["path"],
               "robot_free_probe_root": str(probe_root), "open_loop_horizon": 8,
               "max_hourly_rate_usd": 0.5, "scored_matrix_salt": "5" * 64}
    # The probe lane's retention: scored with the team half's (source) contract.
    spec = scene_facts(profile)["scene"]["task_spec"]
    trace = record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"]),
        plan=physics_plan(probe_context(spec)), binding=binding(spec, half["scene_revision_digest"]))
    receipt = score_rigid_robot_free_readiness(task_spec=spec, physics_probe=trace,
                                               configured_scene_revision_digest=half["scene_revision_digest"])
    assert receipt["status"] == "passed", receipt.get("probe_error")
    retain_receipt(probe_root, receipt)
    return SimpleNamespace(half=half, receipt=receipt, profile=profile)


def _records(revision: str) -> dict[str, Any]:
    """The team's candidate, rights, lease, budget and founder approval (as the admission tests seal them)."""

    candidate = {"schema_version": "blueprint.team_policy_candidate.v1", "team_id": "team-acme",
                 "checkpoint_id": "pi05-ft-7", "candidate_id": TEAM,
                 "embodiment_preset_id": "droid_franka_panda_robotiq_2f85_v1",
                 "policy_interface_id": "openpi_droid_https.v1",
                 "delivery": {"mode": "https_policy", "endpoint_url": ORIGIN + "/v1/act",
                              "metadata_url": ORIGIN + "/v1/metadata", "action_chunk_rows": 10,
                              "open_loop_horizon": 8, "model_identity_digest": "sha256:" + "7" * 64,
                              "team_policy_id": "acme-pi05-ft-7"}}
    rights = _sealed({"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
                      "team_id": "team-acme", "candidate_id": TEAM, "checkpoint_id": "pi05-ft-7",
                      "license_to_run_granted": True, "policy_self_grading_permitted": False,
                      "official_ranking_permitted": False, "secret_material_recorded": False,
                      "adp_qualified_evidence_permitted": False}, "rights_digest")
    lease = {"schema_version": "blueprint.team_policy_credential_lease_availability.v1", "kind": "kms_bearer_lease",
             "status": "available", "single_use": True, "lease_id": "lease-1",
             "credential_ref_digest": "sha256:" + "6" * 64, "credential_env": "BLUEPRINT_TEAM_POLICY_BEARER_FILE",
             "origin": ORIGIN, "candidate_id": TEAM, "secret_material_recorded": False,
             "expires_at_iso": "2026-10-15T18:00:00Z"}
    budget = {"payer": "blueprint", "customer_price_usd": 0, "max_attempts": 1, "episodes": 200,
              "cap_usd": 20, "hard_ttl_seconds": 9000}
    founder = _sealed({"schema_version": "blueprint.team_candidate_founder_admission.v1",
                       "approved_by": "founder", "approval_reference": "owner-decision-team-acme",
                       "expires_at_iso": "2026-10-16T00:00:00Z",
                       "scope": {"team_id": "team-acme", "candidate_id": TEAM, "scene_revision_digest": revision,
                                 "run_kind": "team_candidate_evaluation"}}, "approval_digest")
    return {"candidate": candidate, "rights": rights, "credential_lease": lease, "budget": budget,
            "founder_admission": founder}


def _admit(facts: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    values = {**_records(facts["scene"]["scene_revision_digest"]), "scene": facts["scene"],
              "probe_receipt": facts["probe_receipt"]}
    values.update(overrides)
    return admit_team_candidate(**values, now=NOW)


def _blockers(facts: dict[str, Any], **overrides: Any) -> list[str]:
    with pytest.raises(TeamCandidateAdmissionError) as raised:
        _admit(facts, **overrides)
    return raised.value.blockers


def test_a_per_cell_controls_owner_contract_is_evaluated_by_a_team_under_the_carried_omission(
        tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import team_candidate_quick10_launch as launch
    from blueprint_pipeline.team_candidate_quick10_bundle import build_team_candidate_session_bundle
    from blueprint_pipeline.team_candidate_quick10_staging import stage_team_runtime

    scene = _scene(tmp_path, monkeypatch)
    source = scene.half["task_success_contract"]
    assert source["provenance"]["confirmation_status"] == "confirmed" and controls_required(source)

    facts, blockers = offer_eligibility(scene.profile)
    assert blockers == [] and facts is not None
    contract = facts["scene"]["task_success_contract"]
    authority = facts["scene"]["diagnostic_control_omission_authority"]
    # Only the controls criterion is gone: every scoring criterion and threshold is the owner's.
    assert contract["criteria"] == {key: row for key, row in source["criteria"].items() if key != "controls"}
    assert ({key: row for key, row in contract.items() if key not in ("criteria", "contract_digest")}
            == {key: row for key, row in source.items() if key not in ("criteria", "contract_digest")})
    assert facts["inputs"]["task_success_contract"] == contract
    assert facts["inputs"]["task_success_contract_digest"] == contract["contract_digest"]
    assert facts["scene"]["source_task_success_contract"] == source
    assert authority["claim_ceiling"] == "diagnostic_policy_execution"
    assert authority["task_scoring_criteria_changed"] is False
    assert authority["qualified_comparison_permitted"] is False
    assert authority["authorized_by"] == "blueprint-platform-owner"
    assert authority["authorization_reference"] == "docs/website-controls-pause-2026-09-21.md"
    # Both contract digests are bound.
    assert authority["source_task_success_contract_digest"] == source["contract_digest"]
    assert authority["result_task_success_contract_digest"] == contract["contract_digest"]
    # The probe stays bound to the contract it scored.
    assert scene.receipt["task_success_contract_digest"] == source["contract_digest"]
    assert facts["probe"]["receipt_digest"] == scene.receipt["receipt_digest"]

    admission = _admit(facts)
    assert admission["diagnostic_control_omission_authority"] == authority
    assert admission["task_success_contract_digest"] == contract["contract_digest"]
    assert admission["robot_free_probe_receipt_digest"] == scene.receipt["receipt_digest"]
    assert admission["claim_ceiling"] == "diagnostic_policy_execution"
    assert admission["evidence_scope"] == "development_only"
    assert validate_admission(admission) == admission
    assert runtime_binding(admission)["diagnostic_control_omission_authority"] == authority

    cells = launch._session_cells(purpose="scored", facts=facts, profile=scene.profile)[:10]
    paths = stage_team_runtime(root=tmp_path / "stage", run_id="team-scored-paused-controls",
                               scene_inputs=facts["inputs"], admission=admission, prompt=facts["prompt"],
                               max_policy_queries=facts["max_policy_queries"], hard_cap_usd=20.0,
                               hard_ttl_seconds=9000, cells=cells)
    inputs = validate_runtime_input_manifest(json.loads(paths["runtime_inputs"].read_text()))
    assert inputs["claim_ceiling"] == "diagnostic_policy_execution"
    assert inputs["task_success_contract"] == contract
    assert inputs["team_candidate_admission"]["diagnostic_control_omission_authority"] == authority
    assert all(cell["control_diagnostic"]["mode"] == "nonblocking_diagnostic_pending" for cell in inputs["cells"])
    assert json.loads(paths["execution_spec"].read_text())["task_success_contract"] == contract

    bundle = build_team_candidate_session_bundle(
        job_dir=tmp_path / "bundle", packet_dir=facts["packet_dir"],
        runtime_source_packet_receipt=facts["runtime_source_packet_receipt"],
        runtime_input_manifest_path=paths["runtime_inputs"], session_authority_path=paths["session_authority"],
        execution_spec_path=paths["execution_spec"], implementation_commit="a" * 40)
    assert bundle["static_startup_preflight"]["status"] == "passed"
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        sealed = json.loads(archive.read("provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json"))
    assert sealed == inputs and not controls_required(sealed["task_success_contract"])

    def resealed(binding: dict | None = None, task: dict | None = None) -> dict:
        value = json.loads(json.dumps(inputs))
        if binding is not None:
            value["team_candidate_admission"] = binding
        if task is not None:
            value.update(task_success_contract=task, task_success_contract_digest=task["contract_digest"])
            value["team_candidate_admission"]["task_success_contract_digest"] = task["contract_digest"]
        value["runtime_inputs_digest"] = canonical_digest(value, digest_field="runtime_inputs_digest")
        return value

    validate_runtime_input_manifest(resealed())
    other = {**authority, "result_task_success_contract_digest": "sha256:" + "c" * 64}
    other["authority_digest"] = canonical_digest(other, digest_field="authority_digest")
    with pytest.raises(PolicyCanarySessionError, match="policy_canary_team_admission_binding_invalid"):
        validate_runtime_input_manifest(resealed(binding={**inputs["team_candidate_admission"],
                                                          "diagnostic_control_omission_authority": other}))
    # Controls kept beside an omission: refused.
    with pytest.raises(PolicyCanarySessionError, match="policy_canary_team_admission_binding_invalid"):
        validate_runtime_input_manifest(resealed(task=source))

    # The terminal carries the authority, says controls were not run, and stays diagnostic.
    closed = _fake_closed_team_result(inputs)
    terminal = team_candidate_terminal(session_results=[closed], purpose="scored", episodes=10,
                                       runtime_binding=inputs["team_candidate_admission"])
    assert terminal["diagnostic_control_omission_authority"] == authority
    assert "per-cell controls were not run (paused by the platform owner)" in terminal["note"]
    assert terminal["scoring"]["claim_ceiling"] == "diagnostic_policy_execution"
    assert terminal["scoring"]["success_contract_digest"] == contract["contract_digest"]
    with pytest.raises(TeamCandidateResultError, match="team_candidate_control_omission_diagnostic_only"):
        team_candidate_terminal(session_results=[closed], purpose="scored", episodes=10,
                                runtime_binding=inputs["team_candidate_admission"],
                                site_task_qualification_receipt={"schema_version": "any"})


def test_without_the_pause_a_per_cell_controls_contract_is_refused_exactly_as_before(tmp_path, monkeypatch) -> None:
    scene = _scene(tmp_path, monkeypatch)
    paused, _ = offer_eligibility(scene.profile)
    assert paused is not None
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", False)
    assert offer_eligibility(scene.profile) == (None, ["team_candidate_strict_controls_runtime_unavailable"])
    # The strict scene itself, and a stale omission once the pause is lifted, are refused by the admission.
    strict = scene_facts(scene.profile)
    strict.update(probe_receipt=scene.receipt)
    assert _blockers(strict) == ["team_candidate_strict_controls_runtime_unavailable"]
    assert _blockers(paused) == ["team_candidate_strict_controls_runtime_unavailable"]


def test_the_admission_binds_the_omission_and_weakens_no_other_gate(tmp_path, monkeypatch) -> None:
    scene = _scene(tmp_path, monkeypatch)
    facts, _ = offer_eligibility(scene.profile)
    assert facts is not None
    authority = facts["scene"]["diagnostic_control_omission_authority"]
    contract = facts["scene"]["task_success_contract"]
    source = facts["scene"]["source_task_success_contract"]
    # A strict contract under the pause is admitted only through the omission.
    assert _blockers({**facts, "scene": {key: value for key, value in facts["scene"].items()
                                         if key not in ("diagnostic_control_omission_authority",
                                                        "source_task_success_contract")}
                                        | {"task_success_contract": source}}) == [
        "team_candidate_strict_controls_runtime_unavailable"]
    # An omission that names someone else, or a contract that changed more than its controls, is refused.
    other = {**authority, "authorized_by": "someone-else"}
    other["authority_digest"] = canonical_digest(other, digest_field="authority_digest")
    assert _blockers({**facts, "scene": {**facts["scene"], "diagnostic_control_omission_authority": other}}) == [
        "team_candidate_control_omission_invalid"]
    changed = json.loads(json.dumps(contract))
    changed["criteria"]["temporal_invariants"]["maximum_retries"] = 3
    changed["contract_digest"] = cross_runtime_canonical_digest(changed, digest_field="contract_digest")
    assert "team_candidate_control_omission_invalid" in _blockers(
        {**facts, "scene": {**facts["scene"], "task_success_contract": changed}})
    unrelated = {**facts["scene"], "source_task_success_contract": {**source, "criteria": {
        **source["criteria"], "controls": {**STRICT, "mode": "optional"}}}}
    assert "team_candidate_control_omission_invalid" in _blockers({**facts, "scene": unrelated})
    # The omission's ceiling is diagnostic: no site-task qualification rides on it.
    assert _blockers(facts, site_task_qualification_receipt={"schema_version": "any"}) == [
        "team_candidate_control_omission_diagnostic_only"]
    # Rights, spend, approval, embodiment and the probe still refuse by name.
    records = _records(facts["scene"]["scene_revision_digest"])
    assert _blockers(facts, rights=_sealed({**records["rights"], "license_to_run_granted": False},
                                           "rights_digest")) == ["team_candidate_rights_scope_insufficient"]
    assert _blockers(facts, budget={**records["budget"], "cap_usd": 21}) == ["team_candidate_budget_cap_invalid"]
    assert _blockers(facts, founder_admission={**records["founder_admission"], "approved_by": ""}) == [
        "team_candidate_founder_admission_missing"]
    assert _blockers(facts, candidate={**records["candidate"], "embodiment_preset_id": "other"}) == [
        "team_candidate_interface_mismatch:embodiment", "team_candidate_interface_mismatch:scene_embodiment"]
    assert _blockers(facts, probe_receipt=None)
    # A sealed admission whose omission was edited afterwards is refused.
    admission = _admit(facts)
    tampered = _sealed({**admission, "diagnostic_control_omission_authority": other | {
        "result_task_success_contract_digest": "sha256:" + "d" * 64}}, "admission_digest")
    with pytest.raises(TeamCandidateAdmissionError) as raised:
        validate_admission(tampered)
    assert raised.value.blockers == ["team_candidate_control_omission_invalid"]
