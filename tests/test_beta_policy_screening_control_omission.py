"""Paused controls: a website job whose owner contract requires per-cell controls screens as a diagnostic.

Owner decision 2026-09-21 (docs/website-controls-pause-2026-09-21.md): control
trials are paused for all runs and website runs bind the typed diagnostic
omission. The job here is built with the website's own record builders
(``per_cell_controls_required`` true), its thresholds confirmed by the owner,
then compiled, probed, planned and prepared through the sealed static preflight.
Nothing is allocated.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_screening_control_omission import (
    omit_paused_controls,
    validate_paused_controls_omission,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_policy_canary_control_gate import controls_required
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    PolicyCanarySessionError,
    validate_runtime_input_manifest,
)
from tests.test_task_evaluation_owner_success_confirmation import confirm
from tests.test_team_scene_packet_owner_confirmation import _compile, _job


def _screening(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
    from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
    from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
    from blueprint_pipeline.rigid_robot_free_probe_plan import physics_plan, probe_context
    from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
    from blueprint_pipeline.rigid_robot_free_readiness_score import score_rigid_robot_free_readiness
    from blueprint_pipeline.robot_free_readiness_retention import retain_receipt
    from blueprint_pipeline.team_candidate_quick10_registry import scene_facts
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
               "max_hourly_rate_usd": 0.5, "screening_candidate_cap_usd": 3}
    # The probe lane's retention: scored with the team half's (source) contract.
    spec = scene_facts(profile)["scene"]["task_spec"]
    trace = record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"]),
        plan=physics_plan(probe_context(spec)), binding=binding(spec, half["scene_revision_digest"]))
    receipt = score_rigid_robot_free_readiness(task_spec=spec, physics_probe=trace,
                                               configured_scene_revision_digest=half["scene_revision_digest"])
    assert receipt["status"] == "passed", receipt.get("probe_error")
    retain_receipt(probe_root, receipt)
    source = {"capture_id": "paused-controls-capture", "task_id": profile["task_id"],
              "scene_intent_id": "paused-controls-intent", "scene_intent_digest": "sha256:" + "a" * 64,
              "task_context_digest": "sha256:" + "b" * 64, "scene_revision_digest": half["scene_revision_digest"]}
    catalog = load_default_catalog()
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=Path(profile["capture_root"]),
                                     source=source, inputs_dir=tmp_path / "screening-inputs")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
                                cost_quotes=planning["cost_quotes"], campaign_id="paused-controls-campaign")
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == "pi05_droid")
    assert candidate["status"] == "selected", candidate["blockers"]
    return SimpleNamespace(half=half, receipt=receipt, plan=plan, candidate=candidate,
                           scene=launch._read(Path(plan["scene_inputs"]["path"])))


def test_a_per_cell_controls_owner_contract_screens_under_the_carried_omission(tmp_path, monkeypatch) -> None:
    run = _screening(tmp_path, monkeypatch)
    source = run.half["task_success_contract"]
    assert source["provenance"]["confirmation_status"] == "confirmed" and controls_required(source)
    scene = run.scene
    contract, authority = scene["task_success_contract"], scene["diagnostic_control_omission_authority"]
    # Only the controls criterion is gone: every scoring criterion and threshold is the owner's.
    assert "controls" not in contract["criteria"]
    assert contract["criteria"] == {key: row for key, row in source["criteria"].items() if key != "controls"}
    assert ({key: row for key, row in contract.items() if key not in ("criteria", "contract_digest")}
            == {key: row for key, row in source.items() if key not in ("criteria", "contract_digest")})
    assert scene["source_task_success_contract"] == source
    assert authority["claim_ceiling"] == "diagnostic_policy_execution"
    assert authority["task_scoring_criteria_changed"] is False
    assert authority["qualified_comparison_permitted"] is False
    assert authority["authorized_by"] == "blueprint-platform-owner"
    assert authority["authorization_reference"] == "docs/website-controls-pause-2026-09-21.md"
    assert authority["source_task_success_contract_digest"] == source["contract_digest"]
    assert authority["result_task_success_contract_digest"] == contract["contract_digest"]
    # The probe stays bound to the contract it scored; candidates run the omitted one.
    assert run.receipt["task_success_contract_digest"] == source["contract_digest"]
    assert run.plan["robot_free_probe_digest"] == run.receipt["receipt_digest"]
    assert run.plan["task_success_contract_digest"] == contract["contract_digest"]
    assert run.candidate["execution_spec"]["task_success_contract_digest"] == contract["contract_digest"]

    prepared = launch.prepare_candidate(plan=run.plan, candidate=run.candidate, attempt_dir=tmp_path / "attempt")
    assert launch._read(tmp_path / "attempt" / "static_preflight.json")["status"] == "passed"
    bundle = launch._read(Path(prepared["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        sealed = json.loads(archive.read("provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json"))
    assert sealed["claim_ceiling"] == "diagnostic_policy_execution"
    assert sealed["task_success_contract"] == contract
    assert sealed["beta_screening_admission"]["diagnostic_control_omission_authority"] == authority
    assert sealed["beta_screening_admission"]["robot_free_probe_receipt_digest"] == run.receipt["receipt_digest"]
    authority_record = launch._read(Path(prepared["authority"]["path"]))
    assert authority_record["beta_screening_admission"] == sealed["beta_screening_admission"]

    def resealed(admission: dict | None = None, task: dict | None = None) -> dict:
        value = json.loads(json.dumps(sealed))
        if admission is not None:
            value["beta_screening_admission"] = admission
        if task is not None:
            value.update(task_success_contract=task, task_success_contract_digest=task["contract_digest"])
            value["beta_screening_admission"]["task_success_contract_digest"] = task["contract_digest"]
        value["runtime_inputs_digest"] = canonical_digest(value, digest_field="runtime_inputs_digest")
        return value

    validate_runtime_input_manifest(resealed())
    other = {**authority, "result_task_success_contract_digest": "sha256:" + "c" * 64}
    other["authority_digest"] = canonical_digest(other, digest_field="authority_digest")
    with pytest.raises(PolicyCanarySessionError, match="policy_canary_beta_admission_binding_invalid"):
        validate_runtime_input_manifest(resealed(admission={**sealed["beta_screening_admission"],
                                                            "diagnostic_control_omission_authority": other}))
    # Controls kept beside an omission: refused.
    with pytest.raises(PolicyCanarySessionError, match="policy_canary_beta_admission_binding_invalid"):
        validate_runtime_input_manifest(resealed(task=source))


def test_the_omission_changes_nothing_but_controls_and_needs_the_pause(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import task_evaluation_control_stage_policy as policy
    from blueprint_pipeline.adp_task_scoring import _compatibility_rigid_success_criteria, seal_rigid_task_success_contract
    from tests.test_rigid_robot_free_probe import rigid_case

    spec = rigid_case(tmp_path)[3]

    def sealed(**extra: dict) -> dict:
        criteria = {**_compatibility_rigid_success_criteria(spec), **extra}
        return seal_rigid_task_success_contract(task_spec=spec, site_id="site-1", task_id="task-1",
            author_source="task_owner", author_id="owner-1", confirmation_status="confirmed",
            confirmed_by_team_id="owner-1", criteria=criteria)

    plain = sealed()
    assert omit_paused_controls(plain) == (plain, None)
    strict = sealed(controls={"mode": "required_per_cell",
                              "control_ids": ["zero_action_negative", "deterministic_scripted_positive"]})
    assert controls_required(strict)
    contract, authority = omit_paused_controls(strict)
    assert contract == plain and authority is not None
    assert validate_paused_controls_omission(authority, source=strict, contract=contract) == authority
    changed = json.loads(json.dumps(contract))
    changed["criteria"]["temporal_invariants"]["maximum_retries"] = 3
    other = {**authority, "authorized_by": "someone-else"}
    other["authority_digest"] = canonical_digest(other, digest_field="authority_digest")
    for refused in ({"authority": authority, "source": strict, "contract": changed},
                    {"authority": authority, "source": plain, "contract": contract},
                    {"authority": other, "source": strict, "contract": contract}):
        with pytest.raises(ValueError):
            validate_paused_controls_omission(**refused)
    monkeypatch.setattr(policy, "CONTROLS_PAUSED", False)
    assert omit_paused_controls(strict) == (strict, None)
