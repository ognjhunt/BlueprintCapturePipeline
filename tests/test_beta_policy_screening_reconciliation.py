"""Closed paid failures settle at their bound; uncertain teardown never does."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import beta_policy_screening_closeout as closeout
from blueprint_pipeline import team_candidate_quick10_launch as native
from tests.test_beta_policy_screening_coordinator import native_result, plan, seal


@pytest.fixture
def closed(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_validate_prepared", lambda *args: None)
    monkeypatch.setattr(native, "_default_provider_zero", lambda: {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "provider_zero_verified": True, "live_instance_count": 0, "blockers": [],
    })
    # This suite isolates coordinator/bridge behavior. The closeout helper's
    # own tests exercise real sealed authority bindings and provider inventory
    # semantics with injected provider reads; no provider I/O belongs here.
    monkeypatch.setattr(closeout, "collect_scoped_closeout", lambda **_: {
        "schema_version": closeout.SCHEMA, "scope": closeout.SCOPE,
        "provider_absence_confirmed": True, "global_provider_zero_claimed": False,
        "receipt_digest": "sha256:" + "a" * 64, "blockers": [],
    })
    monkeypatch.setattr(closeout, "scoped_closeout_verified", lambda receipt, **kwargs:
        receipt.get("provider_absence_confirmed") is True
        and len(kwargs["adapter"].get("vast_instance_ids", [])) == 1
        and closeout._own_teardown(kwargs["adapter"]))
    def forbidden(*args, **kwargs):
        raise AssertionError("reconciliation must never spawn")
    monkeypatch.setattr(native, "_start_allocator", forbidden)
    adapter = {"status": "blocked", "vast_instance_ids": [7001], "provider_mutations_performed": 1,
               "provider_create_attempted": True, "continuing_spend_from_this_run": False,
               "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
                                     "all_staged_objects_absent": True}}
    return tmp_path, adapter


def observe(root, adapter, value=None):
    (root / "allocator_result.json").write_text(json.dumps(adapter))
    value = value or plan()
    return launch.reconcile_candidate(plan=value, candidate=value["candidates"][0], attempt_dir=root, prepared={})


@pytest.mark.parametrize("evidence", ["missing", "nonjson", "unbound_digest", "partial_episodes"])
def test_proven_closed_allocation_with_unusable_result_is_terminal_paid_failure(closed, evidence):
    root, adapter = closed
    value = plan()
    path = root / "native_result.json"
    if evidence != "missing":
        result = native_result(value)
        if evidence == "partial_episodes":
            result["episodes"] = result["episodes"][:1]
            seal(result, "result_digest")
        path.write_text("not-json" if evidence == "nonjson" else json.dumps(result))
        adapter["native_control_result_path"] = str(path)
        adapter["native_control_result_digest"] = ("sha256:" + "f" * 64
                                                   if evidence == "unbound_digest" else result["result_digest"])
    outcome = observe(root, adapter, value)
    assert outcome["status"] == "blocked"
    assert outcome["provider_allocation_performed"] is True
    assert outcome["provider_zero_verified"] is True
    assert outcome["no_allocation_proven"] is False
    assert "result" not in outcome
    assert outcome["blockers"]
    assert not (root / "allocator_launch_intent.json").exists()


@pytest.mark.parametrize("unproven", ["instances", "teardown", "staged_objects", "retained_session", "spend"])
def test_provider_absence_alone_cannot_settle_an_incompletely_closed_attempt(closed, unproven):
    root, adapter = closed
    if unproven == "instances":
        adapter["vast_instance_ids"] = []
    elif unproven == "teardown":
        adapter["provider_closeout"]["provider_zero_confirmed"] = False
    elif unproven == "staged_objects":
        adapter["provider_closeout"]["all_staged_objects_absent"] = False
    elif unproven == "retained_session":
        adapter["provider_closeout"]["warm_session_retained"] = True
    else:
        adapter["continuing_spend_from_this_run"] = True
    outcome = observe(root, adapter)
    assert outcome["status"] == "reconciliation_required"
    assert outcome["no_allocation_proven"] is False
    assert "result" not in outcome


def test_failed_scoped_provider_observation_keeps_closed_local_attempt_reserved(closed, monkeypatch):
    root, adapter = closed
    monkeypatch.setattr(native, "_default_provider_zero", lambda: {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "provider_zero_verified": False, "live_instance_count": None, "blockers": ["provider_unavailable"],
    })
    monkeypatch.setattr(closeout, "collect_scoped_closeout", lambda **_: {
        "schema_version": closeout.SCHEMA, "scope": closeout.SCOPE,
        "provider_absence_confirmed": False, "global_provider_zero_claimed": False,
        "receipt_digest": "sha256:" + "b" * 64, "blockers": ["provider_unavailable"],
    })
    outcome = observe(root, adapter)
    assert outcome["status"] == "reconciliation_required" and outcome["provider_zero_verified"] is False
    assert outcome["no_allocation_proven"] is False
