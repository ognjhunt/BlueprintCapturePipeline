from __future__ import annotations

import json

import pytest

from blueprint_pipeline import policy_canary_unallocated_resume as resume
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    PROVIDER_BUNDLE_SCHEMA_VERSION, PROVIDER_RESULT_FILENAME, consume_session_authority_once,
)
from tests.test_native_task_arena_policy_canary_session import _authority


def _case(tmp_path, monkeypatch):
    authority, _ = _authority(tmp_path)
    bundle = {"schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION, "status": "ready",
        "execution_mode": "internal_policy_canary_paired_session", "run_kind": authority["run_kind"],
        "claim_ceiling": authority["claim_ceiling"], "maximum_provider_allocations": 1, "retry_cap": 0,
        "candidate_policy_queried": False, "expected_output_filename": PROVIDER_RESULT_FILENAME,
        "bundle_path": str(tmp_path / "bundle.zip"), "bundle_sha256": "sha256:" + "b" * 64,
        "bundle_size_bytes": 1, **{k: authority[k] for k in ("candidate_ids", "episodes_per_policy",
            "learned_policy_rollout_count", "runtime_inputs_digest", "authority_digest", "task_success_contract_digest")}}
    parent, successor = tmp_path / "parent", tmp_path / "successor"
    attempt = parent / "attempts/attempt_001"
    provider = attempt / "vast_provider_run"
    provider.mkdir(parents=True)
    consume_session_authority_once(authority, consumption_path=parent / "policy_canary_session_consumption.json")
    adapter = {"status": "blocked", "reason": "vast_paid_launch_lock_blocked",
        "blockers": ["vast_paid_launch_lock_busy"], "provider_bundle_sha256": bundle["bundle_sha256"],
        "api_call_performed": False, "vast_side_effects_may_have_occurred": False,
        "provider_create_attempted": False, "continuing_spend_from_this_run": False,
        "vast_instance_ids": [], "estimated_cost_usd": 0.0}
    (provider / "vast_provider_adapter_result.json").write_text(json.dumps(adapter))
    (provider / "vast_launch_lock_manifest.json").write_text(json.dumps({"status": "blocked",
        "lock_acquired": False, "blockers": ["vast_paid_launch_lock_busy"]}))
    (provider / "vast_teardown_manifest.json").write_text(json.dumps({"status": "not_required_launch_lock_blocked",
        "vast_instance_ids": [], "continuing_spend_from_this_run": False}))
    proof = attempt / "absence.json"
    proof.write_text('{"verified_by_existing_absence_reader":true}')
    # Absence-reader validation has its own real receipt/digest tests; exercise
    # the resume boundary independently, then replay the retained live records.
    monkeypatch.setattr(resume, "billing_staged_objects_absent", lambda row: (True, proof))
    binding = {"authority_digest": authority["authority_digest"], "bundle_sha256": bundle["bundle_sha256"]}
    paid = {"status": "admitted", "blockers": [], "allocation_binding": binding,
        "allocation_binding_digest": canonical_digest(binding)}
    funding = {"status": "admitted", "allocation_binding_digest": paid["allocation_binding_digest"],
        "maximum_cost_usd": authority["hard_cap_usd"], "provider": "vast", "resource_class": "gpu_render", "request_count": 1}
    return dict(parent_job_dir=parent, successor_job_dir=successor, authority=authority, bundle=bundle,
        funding_admission=funding, paid_admission=paid), adapter


def test_unallocated_resume_preserves_history_and_claims_only_one_successor(tmp_path, monkeypatch):
    args, _ = _case(tmp_path, monkeypatch)
    parent = args["parent_job_dir"]
    before = {p.relative_to(parent): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    receipt = resume.prepare_unallocated_session_resume(**args)
    assert not args["successor_job_dir"].exists()
    assert receipt["maximum_remaining_provider_allocations"] == 1
    assert receipt["new_preparation_requests_authorized"] == 0
    assert resume.prepare_unallocated_session_resume(**args, apply=True) == receipt
    assert all((parent / p).read_bytes() == raw for p, raw in before.items())
    with pytest.raises(FileExistsError):
        resume.prepare_unallocated_session_resume(**{**args, "successor_job_dir": tmp_path / "second"}, apply=True)


@pytest.mark.parametrize("field,value", [("provider_create_attempted", True), ("api_call_performed", True),
    ("vast_side_effects_may_have_occurred", None), ("continuing_spend_from_this_run", True),
    ("vast_instance_ids", [123]), ("estimated_cost_usd", .01), ("reason", "startup_failed"),
    ("provider_bundle_sha256", "sha256:" + "c" * 64)])
def test_unallocated_resume_refuses_any_provider_mutation_or_changed_bundle(tmp_path, monkeypatch, field, value):
    args, adapter = _case(tmp_path, monkeypatch)
    adapter[field] = value
    p = args["parent_job_dir"] / "attempts/attempt_001/vast_provider_run/vast_provider_adapter_result.json"
    p.write_text(json.dumps(adapter))
    with pytest.raises(ValueError, match="provider_mutation_not_excluded"):
        resume.prepare_unallocated_session_resume(**args)


def test_unallocated_resume_refuses_unknown_extra_attempts_or_missing_absence(tmp_path, monkeypatch):
    args, _ = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(resume, "billing_staged_objects_absent", lambda row: (False, None))
    with pytest.raises(ValueError, match="staging_absence_unproven"):
        resume.prepare_unallocated_session_resume(**args)
    (args["parent_job_dir"] / "attempts/attempt_002").mkdir()
    with pytest.raises(ValueError, match="attempt_history_invalid"):
        resume.prepare_unallocated_session_resume(**args)


def test_unallocated_resume_cannot_change_or_replenish_funding(tmp_path, monkeypatch):
    args, _ = _case(tmp_path, monkeypatch)
    args["funding_admission"]["maximum_cost_usd"] += 1
    with pytest.raises(ValueError, match="funding_binding_invalid"):
        resume.prepare_unallocated_session_resume(**args)
