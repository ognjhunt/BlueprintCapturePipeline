"""A started launch that never consumed its one-use authority never reached the allocator.

capture-447ae4f4, 2026-11-01 02:58Z: the supervised dispatcher raised after
``launch_started.json`` and before the allocator (#3224). The lease kept the
profile's 7.5 h hard TTL and its scene-configuration row a full $20 hold.

Covers (for impacted-test selection):
  src/blueprint_pipeline/task_evaluation_launch_preallocation.py
  src/blueprint_pipeline/task_evaluation_launch_reconciler.py
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest
from blueprint_pipeline.task_evaluation_launch_dispatcher_ownership import DISPATCHER_LOCK_NAME
from blueprint_pipeline.task_evaluation_launch_preallocation import (
    preallocation_evidence_still_holds,
    preallocation_stop_evidence,
)
from blueprint_pipeline.task_evaluation_launch_reconciler import PRE_RUN_STALE_SECONDS, reconcile_launches
from blueprint_pipeline.task_evaluation_owner_dispatch_scope import SCOPE_ENV
from tests.test_task_evaluation_launch_reconciler_stale_profile import (
    LAUNCH_ID,
    PROFILE_ID,
    _profile,
    _write,
    _zero_guard,
)


def started_run(tmp_path: Path, *, age_seconds: float = 20 * 60, standing: bool = True,
                authorization: bool = True, consumed: bool = False, allocator: bool = False) -> dict:
    """What the dispatcher leaves when it raises after launch_started and before the allocator."""
    state = tmp_path / "state"
    profile = _profile(reconciliation=True)
    if standing:
        profile["standing_launch_authorization"] = {"maximum_launches": 1, "maximum_total_spend_usd": 20.0}
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    _write(tmp_path / "profiles" / f"{PROFILE_ID}.json", profile)
    request = {"launch_id": LAUNCH_ID, "run_id": LAUNCH_ID, "launch_profile_id": PROFILE_ID,
               "launch_profile_digest": profile["profile_digest"]}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    queue = tmp_path / "queue"
    _write(queue / "processing" / f"{LAUNCH_ID}-abc.json", request)
    (queue / DISPATCHER_LOCK_NAME).touch()
    run = state / LAUNCH_ID
    _write(run / "launch_request.json", request)
    _write(run / "launch_profile.json", profile)
    started = {"schema_version": "task_evaluation_launch_started.v1", "launch_id": LAUNCH_ID, "run_id": LAUNCH_ID,
               "request_digest": request["request_digest"], "binding_digest": "sha256:" + "b" * 64,
               "started_at": (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat(),
               "hard_ttl_seconds": 27000, "process_id": 999999, "automatic_retry_authorized": False}
    started["started_digest"] = canonical_digest(started, digest_field="started_digest")
    _write(run / "launch_started.json", started)
    directory = tmp_path / "standing-authorizations"
    if authorization:
        _write(directory / f"{PROFILE_ID}.json", {"profile_id": PROFILE_ID})
    if consumed:
        _write(directory / "consumed" / PROFILE_ID / f"{LAUNCH_ID}.json", {"launch_id": LAUNCH_ID})
    if allocator:
        _write(run / "allocator" / "admission.json", {})
    return {"state": state, "queue": queue, "run": run, "profiles": tmp_path / "profiles",
            "guard": tmp_path / "guard.json", "directory": directory}


def _reconcile(fx: dict) -> dict:
    observed_at = datetime.now(timezone.utc)
    _write(fx["guard"], _zero_guard(observed_at=observed_at))
    return reconcile_launches(queue_root=fx["queue"], state_root=fx["state"], profile_dir=fx["profiles"],
                              guard_report_path=fx["guard"], now=observed_at, publish_progress=False)["launches"][0]


@pytest.fixture(autouse=True)
def _no_owner_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SCOPE_ENV, raising=False)
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False)


def test_unconsumed_one_use_authority_proves_no_allocator_and_is_contained_in_one_pass(tmp_path: Path) -> None:
    fx = started_run(tmp_path)
    evidence = preallocation_stop_evidence(run_root=fx["run"], launch_id=LAUNCH_ID, state_root=fx["state"])
    assert evidence["basis"] == "standing_authorization_unconsumed"
    assert preallocation_evidence_still_holds(evidence, run_root=fx["run"])

    row = _reconcile(fx)

    assert row["status"] == "provider_zero_confirmed"
    assert (fx["queue"] / "blocked" / f"{LAUNCH_ID}-abc.json").is_file()
    recovery = json.loads((fx["run"] / "orphan_recovery_receipt.json").read_text())
    assert recovery["allocator_invoked"] is False and recovery["hard_ttl_seconds"] == PRE_RUN_STALE_SECONDS


@pytest.mark.parametrize("change", [
    {"consumed": True},        # the dispatcher recorded the launch: the allocator may have run
    {"allocator": True},       # allocator output exists
    {"authorization": False},  # not provably the dispatcher's directory
    {"standing": False},       # the profile needs no one-use authority
])
def test_any_sign_of_an_allocator_keeps_the_full_hard_ttl(tmp_path: Path, change: dict) -> None:
    fx = started_run(tmp_path, **change)
    assert preallocation_stop_evidence(run_root=fx["run"], launch_id=LAUNCH_ID, state_root=fx["state"]) is None
    assert _reconcile(fx)["status"] == "processing_within_ttl"


def test_a_fresh_stopped_run_still_waits_out_a_dispatcher_pass(tmp_path: Path) -> None:
    fx = started_run(tmp_path, age_seconds=60)
    assert _reconcile(fx)["status"] == "processing_within_ttl"


def test_a_live_dispatcher_keeps_its_started_run(tmp_path: Path) -> None:
    import fcntl
    import os

    fx = started_run(tmp_path)
    descriptor = os.open(fx["queue"] / DISPATCHER_LOCK_NAME, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        assert _reconcile(fx)["status"] == "processing_within_ttl"
    finally:
        os.close(descriptor)


def test_retained_evidence_fails_once_consumption_or_run_evidence_appears(tmp_path: Path) -> None:
    fx = started_run(tmp_path)
    evidence = preallocation_stop_evidence(run_root=fx["run"], launch_id=LAUNCH_ID, state_root=fx["state"])
    _write(fx["directory"] / "consumed" / PROFILE_ID / f"{LAUNCH_ID}.json", {"launch_id": LAUNCH_ID})
    assert not preallocation_evidence_still_holds(evidence, run_root=fx["run"])
    fresh = started_run(tmp_path / "second")
    evidence = preallocation_stop_evidence(run_root=fresh["run"], launch_id=LAUNCH_ID, state_root=fresh["state"])
    _write(fresh["run"] / "launch_receipt.json", {"launch_id": LAUNCH_ID, "status": "blocked"})
    assert not preallocation_evidence_still_holds(evidence, run_root=fresh["run"])
