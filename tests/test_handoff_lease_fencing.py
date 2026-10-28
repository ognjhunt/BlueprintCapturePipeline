"""AUT-01: actual ledger/output interleavings and failed heartbeat effects."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline.core.handoff_lease import (
    HandoffLeaseError,
    bind_handoff_lease,
    handoff_lease_checkpoint,
)
from blueprint_pipeline.run_e2e import _write_run_e2e_stage_ledger


def claim(root: Path, owner: str, *, now=None):
    state, ledger = listener._claim_job_lease(
        root, scene_id="scene", capture_id="capture", owner=owner,
        lease_seconds=60, now=now,
    )
    assert state == "claimed"
    return ledger


def finish(root: Path, ledger: dict, owner: str, result: str):
    return listener._write_output_commit(
        root, scene_id="scene", capture_id="capture", owner=owner,
        token=ledger["lease_token"], attempt_count=ledger["attempt_count"],
        result={"status": "ok", "result": result},
        ledger_update={"status": "completed", "output_commit_status": "committed"},
    )


def test_expired_a_cannot_overwrite_b_output_or_ledger(tmp_path):
    first = claim(tmp_path, "a", now=datetime.now(timezone.utc) - timedelta(hours=1))
    second = claim(tmp_path, "b")
    committed = finish(tmp_path, second, "b", "newer")
    ledger_before = (tmp_path / listener.JOB_LEDGER_FILENAME).read_bytes()
    output_before = (tmp_path / listener.JOB_OUTPUT_COMMIT_FILENAME).read_bytes()
    with pytest.raises(HandoffLeaseError, match="handoff_job_lease_lost"):
        finish(tmp_path, first, "a", "stale")
    assert (tmp_path / listener.JOB_LEDGER_FILENAME).read_bytes() == ledger_before
    assert (tmp_path / listener.JOB_OUTPUT_COMMIT_FILENAME).read_bytes() == output_before
    assert json.loads(ledger_before)["output_result_sha256"] == committed["result_sha256"]
    assert len(list((tmp_path / "pipeline/job_output_commits").glob("*.json"))) == 2


def test_pointer_failure_repairs_from_decided_attempt_without_rerun(tmp_path, monkeypatch):
    lease = claim(tmp_path, "a")
    write = listener.write_json

    def fail_pointer(path, value):
        if path.name == listener.JOB_OUTPUT_COMMIT_FILENAME:
            raise OSError("injected pointer replacement failure")
        return write(path, value)

    monkeypatch.setattr(listener, "write_json", fail_pointer)
    with pytest.raises(OSError):
        finish(tmp_path, lease, "a", "finished")
    decided = json.loads((tmp_path / listener.JOB_LEDGER_FILENAME).read_text())
    assert decided["status"] == "completed"
    assert not (tmp_path / listener.JOB_OUTPUT_COMMIT_FILENAME).exists()
    monkeypatch.setattr(listener, "write_json", write)
    state, recovered = listener._claim_job_lease(
        tmp_path, scene_id="scene", capture_id="capture", owner="b", lease_seconds=60,
    )
    assert state == "completed"
    assert recovered == decided
    pointer = json.loads((tmp_path / listener.JOB_OUTPUT_COMMIT_FILENAME).read_text())
    assert pointer["result_sha256"] == decided["output_result_sha256"]


def test_staging_failure_does_not_publish_or_finish(tmp_path, monkeypatch):
    lease = claim(tmp_path, "a")
    before = (tmp_path / listener.JOB_LEDGER_FILENAME).read_bytes()

    def crash(*args, **kwargs):
        raise RuntimeError("crashed after attempt fsync")

    monkeypatch.setattr(listener, "_finish_job_lease", crash)
    with pytest.raises(RuntimeError):
        finish(tmp_path, lease, "a", "staged")
    assert (tmp_path / listener.JOB_LEDGER_FILENAME).read_bytes() == before
    assert not (tmp_path / listener.JOB_OUTPUT_COMMIT_FILENAME).exists()


@pytest.mark.parametrize("failure", [False, OSError("injected heartbeat I/O")])
def test_heartbeat_failure_is_observable_before_next_effect(tmp_path, monkeypatch, failure):
    lease = claim(tmp_path, "a")
    heartbeat = listener._JobLeaseHeartbeat(
        capture_root=tmp_path, owner="a", token=lease["lease_token"], lease_seconds=60,
    )
    waits = iter([False, True])
    monkeypatch.setattr(heartbeat._stop, "wait", lambda _interval: next(waits))

    def beat(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(listener, "_heartbeat_job_lease", beat)
    heartbeat._run()
    with bind_handoff_lease(heartbeat.checkpoint):
        with pytest.raises(HandoffLeaseError):
            handoff_lease_checkpoint("next_provider_launch")
        with pytest.raises(HandoffLeaseError):
            _write_run_e2e_stage_ledger(tmp_path, {"status": "completed"})
    assert not (tmp_path / "pipeline/run_e2e_stage_ledger.json").exists()


def test_dead_daemon_is_unproven_and_expired_lease_cannot_be_renewed(tmp_path):
    lease = claim(tmp_path, "a", now=datetime.now(timezone.utc) - timedelta(hours=1))
    assert not listener._heartbeat_job_lease(
        tmp_path, owner="a", token=lease["lease_token"], lease_seconds=60,
    )
    heartbeat = listener._JobLeaseHeartbeat(
        capture_root=tmp_path, owner="a", token=lease["lease_token"], lease_seconds=60,
    )
    with pytest.raises(HandoffLeaseError, match="expired"):
        heartbeat.checkpoint()
    heartbeat._failure = None
    heartbeat._entered = True
    with pytest.raises(HandoffLeaseError, match="unproven"):
        heartbeat.checkpoint()


def test_unbound_standalone_checkpoint_is_compatible():
    handoff_lease_checkpoint("standalone_execution")
