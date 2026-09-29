# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_job_lease.py
#   src/blueprint_pipeline/remote_cpu_job_contract.py
#   src/blueprint_pipeline/remote_cpu_job_records.py
"""ADP-009D/day-28, plan 14 PR 1: remote attempts are leased by worker identity, never by a PID."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_job_lease as lease
from blueprint_pipeline import remote_cpu_job_records as records
from tests.test_remote_cpu_job_contract import EXECUTION, QUEUE_NAME, _config, _descriptor

T0 = 2_000_000_000.0


def _transport(config: dict, descriptor: dict) -> str:
    return f"gs://{config['transport_bucket']}/transport/{descriptor['job_id']}/{descriptor['attempt_id']}-{'a' * 32}.json"


def _identity(descriptor: dict, name: str = EXECUTION) -> str:
    return contract.worker_identity_for(descriptor["execution"], name)


def _transport_updates(config: dict, descriptor: dict, dispatched_at: float) -> dict:
    # Every presigned URL expires at the hard deadline: dispatch + 600 + 1800 + 120 s.
    return {"transport_object": _transport(config, descriptor), "transport_generation": 7,
            "write_urls_expire_at_epoch": dispatched_at + 2520, "read_urls_expire_at_epoch": dispatched_at + 2520}


def _dispatch(root: Path, config: dict, descriptor: dict, now: float, *, name: str = EXECUTION) -> None:
    if descriptor["attempt"] == 1:
        lease.claim_handoff(root, descriptor=descriptor, config=config, now=now)
    job, attempt = descriptor["job_id"], descriptor["attempt_id"]
    lease.transition(root, job, attempt_id=attempt, to_state="dispatching", now=now,
                     updates=_transport_updates(config, descriptor, now))
    lease.transition(root, job, attempt_id=attempt, to_state="dispatched", now=now + 2,
                     updates={"worker_identity": _identity(descriptor, name)})


def _heartbeat(descriptor: dict, sequence: int, *, name: str = EXECUTION, phase: str = "stage") -> dict:
    return {
        "schema_version": "remote_cpu_job_heartbeat.v1", "attempt_id": descriptor["attempt_id"],
        "execution_name": name, "sequence": sequence, "phase": phase, "elapsed_seconds": 30.0 * sequence,
        "bytes_fetched": 100 * sequence, "bytes_uploaded": 0,
    }


def _compute(descriptor: dict, config: dict, **changes) -> dict:
    compute = {
        "execution_completed": True, "running_count": 0, "listing_complete": True, "listing_pages": 2,
        "executions_for_attempt": 1, "unfinished_executions_for_attempt": 0,
        "transport_object": _transport(config, descriptor), "transport_generation": 7,
        "transport_deleted": True, "transport_absent_at_generation": True,
    }
    compute.update(changes)
    return compute


def _teardown(descriptor: dict, config: dict, *, dispatched_at: float, now: float, outcome: str = "completed",
              name: str | None = EXECUTION) -> dict:
    compute = _compute(descriptor, config) if name else _compute(
        descriptor, config, execution_completed=False, executions_for_attempt=0)
    return records.teardown_record(
        descriptor=descriptor, worker_identity=None if name is None else _identity(descriptor, name),
        outcome=outcome, compute=compute,
        provider={"staging_versions_deleted": 3, "staging_versions_remaining": 0, "staging_listing_complete": True,
                  "write_urls_expire_at_epoch": dispatched_at + 2520, "read_urls_expire_at_epoch": dispatched_at + 2520},
        observed_at_epoch=now,
    )


def _record(root: Path, job_id: str) -> dict:
    return json.loads((root / "leases" / f"{job_id}.json").read_text(encoding="utf-8"))


def _reasons(call) -> tuple[str, ...]:
    with pytest.raises(contract.RemoteCpuContractError) as caught:
        call()
    return caught.value.reasons


def test_stale_heartbeat_expires_the_attempt_and_fences_a_late_receipt(tmp_path: Path) -> None:
    root, config = tmp_path / "remote-cpu-jobs", _config()
    first = _descriptor(config)
    job, attempt = first["job_id"], first["attempt_id"]
    _dispatch(root, config, first, T0)

    renewed = lease.observe_heartbeat(root, job, _heartbeat(first, 1), execution_running=True, now=T0 + 30)
    assert renewed["renewed"] is True and renewed["state"] == "running"
    assert _record(root, job)["lease_expires_at_epoch"] == T0 + 30 + 180
    assert lease.observe_heartbeat(root, job, _heartbeat(first, 1), execution_running=True, now=T0 + 60)["renewed"] is False
    assert lease.observe_heartbeat(root, job, _heartbeat(first, 2), execution_running=False, now=T0 + 60)["renewed"] is False
    assert _record(root, job)["lease_expires_at_epoch"] == T0 + 210
    assert lease.expire_stale(root, now=T0 + 209) == []
    assert [row["job_id"] for row in lease.live_leases(root, now=T0 + 209)] == [job]

    assert lease.expire_stale(root, now=T0 + 211) == [attempt]
    expired = _record(root, job)
    assert expired["state"] == "expired" and expired["outcome"] == "heartbeat_stale"
    assert lease.live_leases(root, now=T0 + 211) == []
    assert "remote_cpu_lease_heartbeat_fenced:state_expired" in _reasons(
        lambda: lease.observe_heartbeat(root, job, _heartbeat(first, 3), execution_running=True, now=T0 + 212)
    )
    assert "remote_cpu_lease_transition_refused:expired->collecting" in _reasons(
        lambda: lease.transition(root, job, attempt_id=attempt, to_state="collecting", now=T0 + 213)
    )

    second = _descriptor(config, attempt=2, nonce="f" * 32)
    assert "remote_cpu_lease_prior_attempt_not_compute_zero" in _reasons(
        lambda: lease.claim_handoff(root, descriptor=second, config=config, now=T0 + 214)
    )
    assert "remote_cpu_lease_compute_zero_unproven" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state=None, now=T0 + 215,
        updates={"compute_zero": _compute(first, config, unfinished_executions_for_attempt=1)},
    ))
    assert "remote_cpu_lease_compute_zero_transport_mismatch" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state=None, now=T0 + 215,
        updates={"compute_zero": _compute(first, config, transport_generation=8)},
    ))
    lease.transition(root, job, attempt_id=attempt, to_state=None, now=T0 + 216,
                     updates={"compute_zero": _compute(first, config)})
    claimed = lease.claim_handoff(root, descriptor=second, config=config, now=T0 + 217)
    assert (claimed["attempt"], claimed["attempt_id"], claimed["state"]) == (2, second["attempt_id"], "claimed")
    assert [(row["attempt_id"], row["compute_zero_proven"], row["provider_zero_proven"])
            for row in claimed["prior_attempts"]] == [(attempt, True, False)]
    assert "remote_cpu_lease_attempt_fenced" in _reasons(
        lambda: lease.transition(root, job, attempt_id=attempt, to_state="collecting", now=T0 + 218)
    )
    _dispatch(root, config, second, T0 + 220, name="blueprint-remote-cpu-episode-compilation-b2c4d")
    assert "remote_cpu_heartbeat_fenced:attempt_id" in _reasons(lambda: lease.observe_heartbeat(
        root, job, _heartbeat(first, 9, name="blueprint-remote-cpu-episode-compilation-b2c4d"),
        execution_running=True, now=T0 + 230,
    ))
    assert f"remote_cpu_lease_exists:{job}" in _reasons(lambda: lease.claim_handoff(
        root, descriptor=_descriptor(config, attempt=2, nonce="e" * 32), config=config, now=T0 + 231,
    ))


def test_a_second_collector_cannot_transition_a_locked_lease(tmp_path: Path) -> None:
    root, config = tmp_path / "remote-cpu-jobs", _config()
    descriptor = _descriptor(config)
    job, attempt = descriptor["job_id"], descriptor["attempt_id"]
    _dispatch(root, config, descriptor, T0)
    before = (root / "leases" / f"{job}.json").read_bytes()

    holder = os.open(root / "leases" / f"{job}.lock", os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert f"remote_cpu_lease_locked:{job}" in _reasons(
            lambda: lease.transition(root, job, attempt_id=attempt, to_state="collecting", now=T0 + 5)
        )
        assert f"remote_cpu_lease_locked:{job}" in _reasons(
            lambda: lease.observe_heartbeat(root, job, _heartbeat(descriptor, 1), execution_running=True, now=T0 + 5)
        )
        assert lease.expire_stale(root, now=T0 + 10_000) == []
        assert (root / "leases" / f"{job}.json").read_bytes() == before
    finally:
        os.close(holder)

    assert "remote_cpu_lease_attempt_fenced" in _reasons(lambda: lease.transition(
        root, job, attempt_id=f"{job}-a1-{'9' * 32}", to_state="collecting", now=T0 + 6,
    ))
    assert "remote_cpu_lease_transition_refused:dispatched->completed" in _reasons(
        lambda: lease.transition(root, job, attempt_id=attempt, to_state="completed", now=T0 + 6, updates={"outcome": "x"})
    )
    assert "remote_cpu_lease_worker_identity_immutable" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state=None, now=T0 + 6,
        updates={"worker_identity": _identity(descriptor, "blueprint-remote-cpu-episode-compilation-other")},
    ))
    collecting = lease.transition(root, job, attempt_id=attempt, to_state="collecting", now=T0 + 7)
    assert collecting["state"] == "collecting"
    assert [row["to"] for row in collecting["transitions"]] == ["claimed", "dispatching", "dispatched", "collecting"]
    assert f"remote_cpu_lease_exists:{job}" in _reasons(
        lambda: lease.claim_handoff(root, descriptor=_descriptor(config, nonce="1" * 32), config=config, now=T0 + 8)
    )
    assert lease.claim_handoff(root, descriptor=descriptor, config=config, now=T0 + 9)["state"] == "collecting"
    unbucketed = _config(transport_bucket=None)
    assert "remote_cpu_lease_config_invalid:transport_bucket" in _reasons(lambda: lease.claim_handoff(
        tmp_path / "other-root", descriptor=_descriptor(unbucketed), config=unbucketed, now=T0 + 10,
    ))


def test_attempts_without_provider_zero_keep_their_capacity_slot(tmp_path: Path) -> None:
    root, config = tmp_path / "remote-cpu-jobs", _config()
    first = _descriptor(config)
    other_row = {"queue": "task-evaluation-episode-compilations", "name": f"prep-2-{'e' * 64}.json",
                 "envelope_digest": "sha256:" + "e" * 64}
    other = _descriptor(config, queue_row=other_row,
                        output_root="/var/lib/blueprint/task-evaluation-inputs/compiled-episodes/prep-2",
                        inputs=[{**row, "materialize_at": row["materialize_at"].replace(QUEUE_NAME, other_row["name"])}
                                for row in _descriptor(config)["inputs"]])
    job, attempt = first["job_id"], first["attempt_id"]

    lease.claim_handoff(root, descriptor=first, config=config, now=T0)
    assert lease.slots_in_use(root) == 0
    lease.transition(root, job, attempt_id=attempt, to_state="awaiting_capacity", now=T0)
    assert lease.slots_in_use(root) == 0
    assert "remote_cpu_lease_transport_missing" in _reasons(
        lambda: lease.transition(root, job, attempt_id=attempt, to_state="dispatching", now=T0 + 1)
    )
    assert lease.slots_in_use(root) == 0
    lease.transition(root, job, attempt_id=attempt, to_state="dispatching", now=T0 + 1,
                     updates=_transport_updates(config, first, T0 + 1))
    assert lease.slots_in_use(root) == 1
    _dispatch(root, config, other, T0 + 2)
    assert lease.slots_in_use(root) == 2

    assert lease.expire_stale(root, now=T0 + 1 + 600) == [attempt]
    assert _record(root, job)["outcome"] == "start_timeout"
    assert lease.slots_in_use(root) == 2
    lease.transition(root, job, attempt_id=attempt, to_state=None, now=T0 + 700,
                     updates={"compute_zero": _compute(first, config, execution_completed=False,
                                                       executions_for_attempt=0)})
    assert lease.slots_in_use(root) == 2
    assert "remote_cpu_lease_provider_zero_unproven" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state="fallback_host", now=T0 + 701, updates={"outcome": "fallback_host"},
    ))
    second = _descriptor(config, attempt=2, nonce="f" * 32)
    lease.claim_handoff(root, descriptor=second, config=config, now=T0 + 702)
    assert lease.slots_in_use(root) == 2
    _dispatch(root, config, second, T0 + 703, name="blueprint-remote-cpu-episode-compilation-b2c4d")
    assert lease.slots_in_use(root) == 3

    lease.transition(root, job, attempt_id=second["attempt_id"], to_state=None, now=T0 + 5000,
                     updates={"teardown": _teardown(first, config, dispatched_at=T0 + 1, now=T0 + 5000, outcome="expired:start_timeout",
                                                   name=None)})
    assert lease.slots_in_use(root) == 2
    assert _record(root, job)["prior_attempts"][0]["provider_zero_proven"] is True

    other_job, other_attempt = other["job_id"], other["attempt_id"]
    lease.transition(root, other_job, attempt_id=other_attempt, to_state="collecting", now=T0 + 5001)
    assert "remote_cpu_lease_provider_zero_unproven" in _reasons(lambda: lease.transition(
        root, other_job, attempt_id=other_attempt, to_state="completed", now=T0 + 5002, updates={"outcome": "completed"},
    ))
    done = lease.transition(root, other_job, attempt_id=other_attempt, to_state="completed", now=T0 + 5003,
                            updates={"outcome": "completed", "teardown": _teardown(other, config, dispatched_at=T0 + 2, now=T0 + 5003)})
    assert done["provider_zero_proven"] is True and done["teardown_digest"].startswith("sha256:")
    assert lease.slots_in_use(root) == 1
    assert not (root / "live" / other_job).exists() and (root / "live" / job).exists()
    assert "remote_cpu_lease_transition_refused:completed->collecting" in _reasons(lambda: lease.transition(
        root, other_job, attempt_id=other_attempt, to_state="collecting", now=T0 + 5004,
    ))
    (root / "leases" / f"{job}.json").write_text("{}", encoding="utf-8")
    assert lease.slots_in_use(root) == contract.MAX_ATTEMPTS_CAP
    (root / "leases" / f"{job}.json").write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    assert lease.slots_in_use(root) == contract.MAX_ATTEMPTS_CAP


def test_provider_zero_waits_for_the_write_urls_the_lease_recorded(tmp_path: Path) -> None:
    root, config = tmp_path / "remote-cpu-jobs", _config()
    descriptor = _descriptor(config)
    job, attempt = descriptor["job_id"], descriptor["attempt_id"]
    _dispatch(root, config, descriptor, T0)
    recorded = _record(root, job)
    assert (recorded["write_urls_expire_at_epoch"], recorded["read_urls_expire_at_epoch"]) == (T0 + 2520, T0 + 2520)
    assert "remote_cpu_lease_transport_immutable" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state=None, now=T0 + 3,
        updates={**_transport_updates(config, descriptor, T0), "write_urls_expire_at_epoch": T0 + 10}))
    lease.transition(root, job, attempt_id=attempt, to_state="collecting", now=T0 + 300)

    # A fast success deletes staging at once, but its presigned PUTs stay usable until they expire.
    early = _teardown(descriptor, config, dispatched_at=T0, now=T0 + 400)
    assert early["compute_zero_proven"] is True and early["provider_zero_proven"] is False
    lease.transition(root, job, attempt_id=attempt, to_state=None, now=T0 + 400, updates={"teardown": early})
    assert _record(root, job)["compute_zero_proven"] is True and lease.slots_in_use(root) == 1
    assert "remote_cpu_field_unexpected:provider.named_objects_absent" in _reasons(lambda: records.teardown_record(
        descriptor=descriptor, worker_identity=_identity(descriptor), outcome="completed",
        compute=_compute(descriptor, config), provider={**early["provider_zero"], "named_objects_absent": True},
        observed_at_epoch=T0 + 400,
    ))

    # Evidence with any expiry but the one the lease recorded never frees the slot,
    for dispatched_at in (T0 - 5000, T0 + 1):
        bogus = _teardown(descriptor, config, dispatched_at=dispatched_at, now=T0 + 2600)
        assert bogus["provider_zero_proven"] is True
        assert "remote_cpu_lease_teardown_unbound" in _reasons(lambda: lease.transition(
            root, job, attempt_id=attempt, to_state=None, now=T0 + 2600, updates={"teardown": bogus}))
    # and neither does evidence observed after the transition's own clock.
    proven = _teardown(descriptor, config, dispatched_at=T0, now=T0 + 2520)
    assert proven["provider_zero_proven"] is True
    assert "remote_cpu_lease_teardown_observed_in_the_future" in _reasons(lambda: lease.transition(
        root, job, attempt_id=attempt, to_state="completed", now=T0 + 2519,
        updates={"teardown": proven, "outcome": "completed"}))
    assert lease.slots_in_use(root) == 1
    lease.transition(root, job, attempt_id=attempt, to_state="completed", now=T0 + 2520,
                     updates={"teardown": proven, "outcome": "completed"})
    assert lease.slots_in_use(root) == 0

    # A prior attempt keeps its deadlines and URL expiries, so its own teardown still binds.
    again = tmp_path / "retry" / "remote-cpu-jobs"
    _dispatch(again, config, descriptor, T0)
    assert lease.expire_stale(again, now=T0 + 600) == [attempt]
    lease.transition(again, job, attempt_id=attempt, to_state=None, now=T0 + 601,
                     updates={"compute_zero": _compute(descriptor, config)})
    second = _descriptor(config, attempt=2, nonce="f" * 32)
    prior = lease.claim_handoff(again, descriptor=second, config=config, now=T0 + 602)["prior_attempts"][0]
    assert prior["deadlines"] == {
        "dispatch_started_at_epoch": T0, "start_by_epoch": T0 + 600, "hard_deadline_epoch": T0 + 2520}
    assert (prior["write_urls_expire_at_epoch"], prior["read_urls_expire_at_epoch"]) == (T0 + 2520, T0 + 2520)
    assert "remote_cpu_lease_teardown_unbound" in _reasons(lambda: lease.transition(
        again, job, attempt_id=second["attempt_id"], to_state=None, now=T0 + 3000,
        updates={"teardown": _teardown(descriptor, config, dispatched_at=T0 + 1, now=T0 + 3000)}))
    assert lease.slots_in_use(again) == 1
    lease.transition(again, job, attempt_id=second["attempt_id"], to_state=None, now=T0 + 3000,
                     updates={"teardown": _teardown(descriptor, config, dispatched_at=T0, now=T0 + 3000)})
    assert lease.slots_in_use(again) == 0


def test_live_markers_are_durable_and_unreadable_leases_are_named(tmp_path: Path, monkeypatch) -> None:
    root, config = tmp_path / "remote-cpu-jobs", _config()
    descriptor = _descriptor(config)
    job, attempt = descriptor["job_id"], descriptor["attempt_id"]
    synced: list[Path] = []
    real_fsync = lease.fsync_directory
    monkeypatch.setattr(lease, "fsync_directory", lambda path: (synced.append(Path(path)), real_fsync(path))[1])

    lease.claim_handoff(root, descriptor=descriptor, config=config, now=T0)
    assert (root / "live" / job).exists() and synced.count(root / "live") == 1
    lease.transition(root, job, attempt_id=attempt, to_state="awaiting_capacity", now=T0 + 1)
    assert synced.count(root / "live") == 1
    lease.transition(root, job, attempt_id=attempt, to_state="fallback_host", now=T0 + 2, updates={"outcome": "host"})
    assert not (root / "live" / job).exists() and synced.count(root / "live") == 2
    assert lease.slot_census(root) == {"slots_in_use": 0, "unreadable": []}

    other = tmp_path / "other" / "remote-cpu-jobs"
    _dispatch(other, config, descriptor, T0)
    assert lease.slot_census(other) == {"slots_in_use": 1, "unreadable": []}
    (other / "leases" / f"{job}.json").write_text("{}", encoding="utf-8")
    (other / "live" / ".nfs000001").write_text("", encoding="utf-8")
    (other / "live" / "X-Amz-Signature=SECRETSIG").write_text("", encoding="utf-8")
    census = lease.slot_census(other)
    assert census["slots_in_use"] == 3 * contract.MAX_ATTEMPTS_CAP == lease.slots_in_use(other)
    assert [name for name in census["unreadable"] if not name.startswith("<key#")] == [".nfs000001", job]
    assert len(census["unreadable"]) == 3 and "SECRETSIG" not in json.dumps(census)


class _NoPidOs:
    def __getattr__(self, name: str):
        if name in {"getpid", "getppid", "kill"}:
            raise AssertionError(f"lease liveness must not use os.{name}")
        return getattr(os, name)


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def test_liveness_uses_worker_identity_and_expiry_never_a_pid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lease, "os", _NoPidOs())
    root, config = tmp_path / "remote-cpu-jobs", _config()
    descriptor = _descriptor(config)
    job = descriptor["job_id"]
    _dispatch(root, config, descriptor, T0)
    lease.observe_heartbeat(root, job, _heartbeat(descriptor, 1), execution_running=True, now=T0 + 30)

    record = _record(root, job)
    assert not {key for key in _keys(record) if "pid" in key or "host" in key}
    assert record["worker_identity"] == _identity(descriptor)
    assert record["deadlines"]["hard_deadline_epoch"] == T0 + 600 + 1800 + 120
    [live] = lease.live_leases(root, now=T0 + 209)
    assert (live["worker_identity"], live["attempt_id"]) == (_identity(descriptor), descriptor["attempt_id"])

    elsewhere = tmp_path / "another-host" / "remote-cpu-jobs"
    shutil.copytree(root, elsewhere)
    assert [row["worker_identity"] for row in lease.live_leases(elsewhere, now=T0 + 209)] == [_identity(descriptor)]
    assert lease.live_leases(elsewhere, now=T0 + 210) == []

    sequence, now = 2, T0 + 60
    while now < T0 + 2520:
        lease.observe_heartbeat(root, job, _heartbeat(descriptor, sequence), execution_running=True, now=now)
        assert _record(root, job)["lease_expires_at_epoch"] <= T0 + 2520
        sequence, now = sequence + 1, now + 30
    assert _record(root, job)["lease_expires_at_epoch"] == T0 + 2520
    assert lease.live_leases(root, now=T0 + 2519) and lease.live_leases(root, now=T0 + 2520) == []
    assert lease.expire_stale(root, now=T0 + 2520) == [descriptor["attempt_id"]]
    assert _record(root, job)["outcome"] == "hard_deadline_passed"
