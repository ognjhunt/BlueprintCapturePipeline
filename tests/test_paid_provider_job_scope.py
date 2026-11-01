"""Job-scoped provider zero lets website jobs run paid GPU work side by side.

The account-wide reading made every website job wait for every other one. The
job-scoped reading proves what that wait protected (this job owns nothing live,
everything live is another job's owned and labeled work, and the fleet budget
has room) while other jobs run.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import paid_provider_job_scope as scope
from blueprint_pipeline import vast_launch_slots as slots
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import (
    TaskEvaluationSceneConfigurationAuthorityError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import (
    _provider_zero as authority_provider_zero,
)

NOW = datetime(2026, 10, 31, 22, 0, 0, tzinfo=timezone.utc)
OTHER_JOB = "blueprint-task-evaluation-scene-config-20261031t215500z-"
RATE = 0.8  # the scene-configuration lane's authorized maximum hourly rate


def _instance(instance_id: str, name: str, *, protected: bool = True, cost: float = 0.6) -> dict:
    return {
        "provider": "vast", "id": instance_id, "name": name, "state": "running",
        "live": True, "booted": True, "age_seconds": 300.0, "cost_per_hr_usd": cost,
        "cost_rate_known": True, "cleanup_authorized": False, "protected": protected,
        "reap_candidate": False, "reap_candidate_reason": None,
    }


def _guard(instances=None, *, generated_at=None, max_live=10, max_burn=10.0,
           daily=12.0, max_daily=100.0, **overrides) -> dict:
    instances = [_instance("101", OTHER_JOB + "a")] if instances is None else instances
    live = [row for row in instances if row["live"]]
    burn = round(sum(row["cost_per_hr_usd"] for row in live), 4)
    value = {
        "schema_version": "gpu_spend_guard.v1",
        "generated_at": (generated_at or NOW - timedelta(seconds=40)).isoformat(),
        "status": "passed",
        "blockers": [],
        "inventory_results": [{"provider": "vast", "required": True, "status": "succeeded",
                               "row_count": len(live)}],
        "live_instance_count": len(live),
        "total_burn_per_hour_usd": burn,
        "provider_zero_verified": not live,
        "provider_zero": {"status": "verified" if not live else "unverified",
                          "required_provider_ids": ["vast"],
                          "global_live_instance_count": len(live),
                          "global_total_burn_per_hour_usd": burn},
        "fleet_budget": {"schema_version": "gpu_fleet_budget_guard.v1", "status": "passed",
                         "live_instance_count": len(live), "total_burn_per_hour_usd": burn,
                         "max_live_instances": max_live, "max_burn_usd_per_hour": max_burn,
                         "daily_spend_usd": daily, "max_daily_spend_usd": max_daily,
                         "blockers": []},
        "reap_mode": True,
        "instances": instances,
        "reap_candidate_ids": [],
        "reap_results": [],
    }
    value.update(overrides)
    return value


def _live(*rows: tuple[str, str]) -> list[dict]:
    return [{"instance_id": instance_id, "name": name} for instance_id, name in rows]


def test_another_jobs_owned_instance_does_not_block_this_job() -> None:
    observed = scope.evaluate_job_scope(guard=_guard(), now=NOW,
                                        live_inventory=_live(("101", OTHER_JOB + "a")))
    assert observed["verified"] is True, observed["blockers"]
    assert observed["job_live_resource_count"] == 0
    assert observed["foreign_live_resource_count"] == 1


def test_empty_account_is_trivially_job_zero() -> None:
    observed = scope.evaluate_job_scope(guard=_guard([]), now=NOW, live_inventory=[])
    assert observed["verified"] is True


@pytest.mark.parametrize(
    ("guard", "live", "kwargs", "blocker"),
    [
        # This job's own resource is still billing.
        (_guard(), _live(("101", OTHER_JOB + "a")),
         {"own_label_prefixes": [OTHER_JOB]}, "job_scope_own_resource_live:101"),
        (_guard(), _live(("101", OTHER_JOB + "a")),
         {"own_instance_ids": ["101"]}, "job_scope_own_resource_live:101"),
        # Unattributable spend might be ours.
        (_guard([_instance("102", "")]), _live(("102", "")), {},
         "job_scope_unattributable_live_resource:102"),
        (_guard([_instance("103", "someone-else")]), _live(("103", "someone-else")), {},
         "job_scope_unattributable_live_resource:103"),
        # Labeled but with no live owner: orphan or unrecorded create.
        (_guard([_instance("104", OTHER_JOB + "b", protected=False)]),
         _live(("104", OTHER_JOB + "b")), {}, "job_scope_unowned_live_resource:104"),
        # Created after the guard's snapshot: never assumed to be someone else's.
        (_guard(), _live(("101", OTHER_JOB + "a"), ("105", OTHER_JOB + "c")), {},
         "job_scope_live_resource_not_in_spend_guard:105"),
        # Orphan reaping must be caught up.
        (_guard(reap_candidate_ids=["106"]), None, {}, "job_scope_orphan_reap_candidates_remaining"),
        (_guard(reap_results=[{"id": "107", "status": "failed"}]), None, {},
         "job_scope_orphan_reap_not_confirmed"),
        # The account-wide money cap, with room for this job.
        (_guard([_instance(str(i), OTHER_JOB + str(i)) for i in range(10)], max_burn=100.0),
         None, {}, "job_scope_fleet_live_instance_headroom_exhausted"),
        (_guard(max_burn=0.6), None, {"added_burn_usd_per_hour": 0.5},
         "job_scope_fleet_burn_headroom_exhausted"),
        (_guard(daily=100.0), None, {}, "job_scope_fleet_daily_spend_headroom_exhausted"),
        (_guard(max_live=None), None, {}, "job_scope_fleet_live_instance_ceiling_missing"),
        (_guard(max_burn=None), None, {}, "job_scope_fleet_burn_ceiling_missing"),
        # The guard itself must be fresh, reaping and unblocked.
        (_guard(generated_at=NOW - timedelta(minutes=6)), None, {}, "job_scope_spend_guard_stale"),
        (_guard(reap_mode=False), None, {}, "job_scope_spend_guard_reap_mode_missing"),
        (_guard(status="blocked", blockers=["spend_admission:paid_work_admission_locked"]),
         None, {}, "job_scope_spend_guard_blocked"),
        (_guard(inventory_results=[{"provider": "vast", "status": "failed"}]), None, {},
         "job_scope_provider_inventory_unverified:vast"),
        (_guard(live_instance_count=0), None, {}, "job_scope_spend_guard_live_inventory_inconsistent"),
    ],
)
def test_job_scope_refuses_what_the_account_wide_zero_protected(guard, live, kwargs, blocker) -> None:
    observed = scope.evaluate_job_scope(guard=guard, now=NOW, live_inventory=live, **kwargs)
    assert observed["verified"] is False
    assert blocker in observed["blockers"]


def test_fleet_budget_status_must_pass() -> None:
    guard = _guard()
    guard["fleet_budget"]["status"] = "blocked"
    assert "job_scope_fleet_budget_not_passed" in scope.evaluate_job_scope(guard=guard, now=NOW)["blockers"]


def test_teardown_reconciliation_needs_no_headroom_but_a_post_teardown_guard() -> None:
    guard = _guard([_instance(str(i), OTHER_JOB + str(i)) for i in range(10)], max_burn=100.0)
    observed = scope.evaluate_job_scope(guard=guard, now=NOW, added_instances=0,
                                        own_instance_ids=["999"])
    assert observed["verified"] is True, observed["blockers"]
    late = scope.evaluate_job_scope(guard=guard, now=NOW, added_instances=0,
                                    not_before=NOW - timedelta(seconds=10),
                                    not_before_subject="teardown")
    assert "job_scope_spend_guard_predates_teardown" in late["blockers"]


def _receipt(**kwargs) -> dict:
    guard = kwargs.pop("guard", _guard())
    kwargs.setdefault("added_burn_usd_per_hour", RATE)
    return scope.build_job_provider_zero_receipt(
        job_key="prep-1", live_inventory=kwargs.pop("live", _live(("101", OTHER_JOB + "a"))),
        guard=guard, guard_record={"path": "/g.json", "sha256": "sha256:" + "a" * 64},
        observed_at=NOW, **kwargs,
    )


def test_receipt_is_sealed_job_scoped_and_never_claims_account_zero() -> None:
    receipt = _receipt()
    assert receipt["schema_version"] == "adp_paid_provider_job_zero.v1"
    assert receipt["provider_zero"] is False
    assert receipt["job_provider_zero"] is True
    assert receipt["global_live_resource_count"] == 1
    assert scope.validate_job_provider_zero_receipt(receipt) == receipt


def test_receipt_refuses_to_seal_an_unproven_scope() -> None:
    with pytest.raises(scope.ProviderJobScopeError, match="job_scope_unowned_live_resource:104"):
        _receipt(guard=_guard([_instance("104", OTHER_JOB + "b", protected=False)]),
                 live=_live(("104", OTHER_JOB + "b")))


def _reseal(value: dict) -> dict:
    value["provider_zero_digest"] = ""
    value["provider_zero_digest"] = canonical_digest(value, digest_field="provider_zero_digest")
    return value


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(job_live_resource_count=1),
        lambda r: r.update(inventory=[{"instance_id": "1", "name": ""}], global_live_resource_count=1),
        lambda r: r["fleet"].update(live_instance_count=10),
        lambda r: r["scope"].update(own_label_prefixes=[OTHER_JOB]),
        lambda r: r.update(blockers=["x"]),
    ],
)
def test_validation_rechecks_a_resealed_receipt(mutate) -> None:
    receipt = _receipt()
    mutate(receipt)
    with pytest.raises(scope.ProviderJobScopeError):
        scope.validate_job_provider_zero_receipt(_reseal(receipt))


def test_validation_refuses_a_tampered_digest() -> None:
    receipt = dict(_receipt(), global_live_resource_count=0, inventory=[])
    with pytest.raises(scope.ProviderJobScopeError, match="job_scope_receipt_invalid"):
        scope.validate_job_provider_zero_receipt(receipt)


def test_scene_configuration_paid_authority_accepts_only_a_valid_job_receipt(tmp_path: Path) -> None:
    path = tmp_path / "zero.json"
    path.write_text(json.dumps(_receipt()))
    assert authority_provider_zero(path)["schema_version"] == "adp_paid_provider_job_zero.v1"
    path.write_text(json.dumps(_reseal(dict(_receipt(), job_live_resource_count=1))))
    with pytest.raises(TaskEvaluationSceneConfigurationAuthorityError,
                       match="scene_configuration_provider_zero_invalid"):
        authority_provider_zero(path)


def _observed(rows: list[dict]) -> dict:
    return {"provider": "vast", "status": "observed", "api_confirmed": True, "blockers": [],
            "resources": rows, "live_resource_count": len(rows),
            "raw_provider_response_recorded": False}


def test_collector_keeps_the_classic_receipt_when_the_account_is_empty(tmp_path: Path) -> None:
    calls = []

    def global_collector(*, inventory_collector):
        calls.append(inventory_collector())
        return {"schema_version": "adp_paid_provider_zero.v1"}

    receipt = scope.collect_job_scoped_vast_provider_zero(
        job_key="prep-1", added_burn_usd_per_hour=RATE, guard_path=tmp_path / "missing.json",
        inventory_collector=lambda: _observed([]), global_receipt_collector=global_collector,
    )
    assert receipt == {"schema_version": "adp_paid_provider_zero.v1"}
    assert calls and calls[0]["resources"] == []


def test_collector_issues_a_job_receipt_beside_another_jobs_instance(tmp_path: Path) -> None:
    guard_path = tmp_path / "latest.json"
    guard_path.write_text(json.dumps(_guard()))
    receipt = scope.collect_job_scoped_vast_provider_zero(
        job_key="prep-1", added_burn_usd_per_hour=RATE, guard_path=guard_path,
        inventory_collector=lambda: _observed(_live(("101", OTHER_JOB + "a"))),
        global_receipt_collector=lambda **_: pytest.fail("account is not empty"),
        now=lambda: NOW,
    )
    assert receipt["schema_version"] == "adp_paid_provider_job_zero.v1"
    assert receipt["spend_guard"]["path"] == str(guard_path)
    scope.validate_job_provider_zero_receipt(receipt)


def test_collector_refuses_an_unconfirmed_inventory_and_a_missing_guard(tmp_path: Path) -> None:
    with pytest.raises(scope.ProviderJobScopeError, match="inventory_query_failed"):
        scope.collect_job_scoped_vast_provider_zero(
            job_key="prep-1", added_burn_usd_per_hour=RATE, inventory_collector=lambda: dict(_observed([]), api_confirmed=False))
    with pytest.raises(scope.ProviderJobScopeError, match="spend_guard_unreadable"):
        scope.collect_job_scoped_vast_provider_zero(
            job_key="prep-1", added_burn_usd_per_hour=RATE, guard_path=tmp_path / "missing.json",
            inventory_collector=lambda: _observed(_live(("101", OTHER_JOB + "a"))))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "global"), ("", "global"), ("jobs", "global"), ("GLOBAL", "global"),
     ("job", "job"), (" Job ", "job")],
)
def test_scope_is_global_unless_the_owner_asks_for_job(raw, expected) -> None:
    environ = {} if raw is None else {scope.SCOPE_ENV: raw}
    assert scope.provider_zero_scope(environ) == expected


@pytest.mark.parametrize(
    ("fleet", "requested", "expected"),
    [
        (None, None, 3),      # historical default
        (None, "6", 3),       # no fleet ceiling in scope: never above the default
        ("10", None, 3),      # raising is explicit, never implied by the fleet budget
        ("10", "6", 6),       # raised within the fleet budget
        ("10", "50", 10),     # never past the fleet live-instance ceiling
        ("50", "50", 10),     # never past the compiled hard ceiling
        ("2", None, 2),       # a tighter fleet budget tightens the default
        ("10", "0", 3),       # invalid request keeps the default
        ("10", "nope", 3),
        ("nope", "6", 3),     # unreadable fleet ceiling keeps the historical one
    ],
)
def test_slot_count_is_configurable_only_within_the_fleet_budget(
    monkeypatch, tmp_path: Path, fleet, requested, expected
) -> None:
    for name, value in ((slots.FLEET_MAX_LIVE_INSTANCES_ENV, fleet),
                        (slots.MAX_CONCURRENT_PAID_LAUNCHES_ENV, requested)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    paths = slots.vast_launch_lock_paths(tmp_path / "vast_paid_launch.lock")
    assert len(paths) == expected
    assert paths[0].name == "vast_paid_launch.lock"


def test_admission_counts_the_incoming_jobs_own_hourly_burn() -> None:
    """Codex P1 on #3209: a fleet already at its hourly ceiling admitted one more.

    $0.60/h live against a $0.60/h ceiling passed because the incoming job's
    burn was never added. It is now, so the same reading refuses.
    """

    at_ceiling = _guard(max_burn=0.6)
    with pytest.raises(scope.ProviderJobScopeError, match="job_scope_fleet_burn_headroom_exhausted"):
        _receipt(guard=at_ceiling)
    assert _receipt(guard=_guard(max_burn=1.4))["fleet"]["added_burn_usd_per_hour"] == RATE


@pytest.mark.parametrize("rate", [None, 0, 0.0, -0.8, True, float("nan"), "0.8"])
def test_admission_without_a_positive_incoming_rate_is_refused(tmp_path: Path, rate) -> None:
    with pytest.raises(scope.ProviderJobScopeError, match="job_scope_incoming_hourly_rate_missing"):
        _receipt(added_burn_usd_per_hour=rate)
    with pytest.raises(scope.ProviderJobScopeError, match="job_scope_incoming_hourly_rate_missing"):
        scope.collect_job_scoped_vast_provider_zero(
            job_key="prep-1", added_burn_usd_per_hour=rate,
            inventory_collector=lambda: pytest.fail("refused before any provider read"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["fleet"].update(added_burn_usd_per_hour=0.0),
        lambda r: r["fleet"].update(total_burn_per_hour_usd=9.5),
        lambda r: r["fleet"].update(max_burn_usd_per_hour=None),
        lambda r: r["fleet"].update(added_instances=0),
    ],
)
def test_validation_rechecks_burn_headroom_from_the_sealed_fleet(mutate) -> None:
    receipt = _receipt()
    mutate(receipt)
    with pytest.raises(scope.ProviderJobScopeError, match="job_scope_receipt_not_job_zero"):
        scope.validate_job_provider_zero_receipt(_reseal(receipt))


def test_paid_authority_refuses_a_job_receipt_that_budgeted_less_than_its_rate() -> None:
    from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import (
        _require_job_zero_burn_covers,
    )

    receipt = _receipt(guard=_guard(max_burn=10.0), added_burn_usd_per_hour=0.5)
    with pytest.raises(TaskEvaluationSceneConfigurationAuthorityError,
                       match="burn_headroom_understated"):
        _require_job_zero_burn_covers(receipt, 0.8)
    _require_job_zero_burn_covers(_receipt(), 0.8)  # budgeted exactly the authority's rate
    _require_job_zero_burn_covers({"schema_version": "adp_paid_provider_zero.v1"}, 0.8)
