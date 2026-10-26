"""Canonical capacity is observed, and only proven unspent races may retry."""
from __future__ import annotations

import fcntl
import json

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import vast_launch_slots as slots


def test_three_real_canonical_slots_wait_then_resume_without_touching_lock_bytes(tmp_path, monkeypatch):
    base = tmp_path / "vast.lock"
    monkeypatch.setenv("VAST_LAUNCH_LOCK_FILE", str(base))
    monkeypatch.setenv(slots.MAX_CONCURRENT_PAID_LAUNCHES_ENV, "3")
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    gate = slots.vast_launch_gate_path(base)
    gate.write_text("existing deployment metadata\n")
    paths = slots.vast_launch_lock_paths(base)
    handles = []
    for index, path in enumerate(paths):
        path.write_text(json.dumps({"owner": index}) + "\n")
        handle = path.open("r+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    before = {path: path.read_bytes() for path in [gate, *paths]}
    kwargs = {"plan": {}, "candidate": {}, "attempt_dir": tmp_path, "prepared": {}}
    try:
        assert launch.preflight_candidate_capacity(**kwargs) == {
            "capacity_available": False, "blockers": ["vast_paid_launch_lock_busy"]}
        handles.pop().close()
        assert launch.preflight_candidate_capacity(**kwargs) == {"capacity_available": True, "blockers": []}
        with gate.open("r+") as deployment:
            fcntl.flock(deployment, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert launch.preflight_candidate_capacity(**kwargs) == {
                "capacity_available": False, "blockers": ["vast_release_deployment_in_progress"]}
    finally:
        for handle in handles:
            handle.close()
    assert {path: path.read_bytes() for path in before} == before
    assert not (tmp_path / "allocator_launch_intent.json").exists()


@pytest.mark.parametrize("change", [{}, {"blockers": ["vast_paid_launch_lock_busy", "another_failure"]},
    {"provider_create_attempted": True}, {"status": "failed"}, {"blockers": ["vast_paid_launch_lock_unusable"]}])
def test_only_exact_proven_unallocated_lock_refusal_carries_retry_authority(tmp_path, monkeypatch, change):
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    adapter = {"status": "blocked", "provider_mutations_performed": 0, "provider_create_attempted": False,
               "blockers": ["vast_paid_launch_lock_busy"], **change}
    path = launch._write(tmp_path / "allocator_result.json", adapter)
    if change.get("provider_create_attempted"):
        # A create may have happened: never turn unrelated provider-zero into
        # authority to reissue it. Stop the test at the live reconciliation seam.
        from blueprint_pipeline import team_candidate_quick10_launch as native
        monkeypatch.setattr(native, "_default_provider_zero", lambda: {
            "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
            "provider_zero_verified": False, "live_instance_count": None, "blockers": ["unproven"]})
    result = launch.reconcile_candidate(plan={}, candidate={}, attempt_dir=tmp_path, prepared={})
    if not change:
        assert result["retryable_capacity_refusal"] and result["no_policy_api_calls_proven"]
        assert result["capacity_refusal_receipt"] == launch._record(path)
        assert result["no_allocation_proven"] and result["provider_allocation_performed"] is False
    else:
        assert result.get("retryable_capacity_refusal") is not True
        assert result.get("no_policy_api_calls_proven") is not True
