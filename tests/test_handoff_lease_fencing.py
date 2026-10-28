"""AUT-01: actual ledger/output interleavings and failed heartbeat effects."""

import json
from concurrent.futures import ThreadPoolExecutor
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
from blueprint_pipeline.core.stage_graph import StageSpec, run_stage_graph


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


def test_actual_long_stage_publication_and_fresh_admission_are_fenced(tmp_path, monkeypatch):
    from blueprint_pipeline import paid_resource_admission, site_package_orchestrator, website_task_preparation
    from blueprint_pipeline import website_task_context
    from blueprint_pipeline.paid_resource_allocator import main as allocator_main

    def lost(boundary):
        raise HandoffLeaseError("lost", boundary)

    monkeypatch.setattr(website_task_context, "safe_request", lambda *a, **kw: pytest.fail("remote effect"))
    with bind_handoff_lease(lost):
        for writer in (site_package_orchestrator.write_json, website_task_preparation.write_json):
            with pytest.raises(HandoffLeaseError):
                writer(tmp_path / "live.json", {"status": "stale"})
        with pytest.raises(HandoffLeaseError):
            paid_resource_admission.require_paid_resource_admission({}, resource_class="gpu_render",
                                                                    expected_schema_version="paid_lane_admission.v1")
        with pytest.raises(HandoffLeaseError):
            allocator_main(["gpu-canary"])
        with pytest.raises(HandoffLeaseError):
            website_task_context.website_webapp_request(capture_id="capture", operation="prepared-scene", payload={})
    assert not (tmp_path / "live.json").exists()


def test_parallel_stage_graph_keeps_each_handoff_context_without_site_leakage():
    observed = []

    def site_run(site):
        def check(boundary):
            observed.append((site, boundary))
            if site == "revoked":
                raise HandoffLeaseError("lost", boundary)

        def effect():
            handoff_lease_checkpoint("provider_create")
            return {"status": "completed", "site": site}

        with bind_handoff_lease(check):
            result = run_stage_graph([StageSpec("one", effect), StageSpec("two", effect)], max_concurrency=2)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        revoked, healthy = list(pool.map(site_run, ["revoked", "healthy"]))
    assert all(row.status == "failed" for row in revoked.executions)
    assert all(row.status == "completed" for row in healthy.executions)
    assert observed.count(("revoked", "provider_create")) == 2
    assert observed.count(("healthy", "provider_create")) == 2


def test_lost_lease_cannot_reset_or_promote_shared_output_transaction(tmp_path):
    from blueprint_pipeline.core.output_run_transaction import OutputRunTransaction
    output = tmp_path / "shared"
    output.mkdir()
    artifact = output / "current.json"
    artifact.write_text("newer evidence")

    def lost(boundary):
        raise HandoffLeaseError("lost", boundary)

    with bind_handoff_lease(lost):
        with pytest.raises(HandoffLeaseError):
            with OutputRunTransaction(output, lane="construction", request_fingerprint="same-request"):
                pytest.fail("stale transaction entered")
    assert artifact.read_text() == "newer evidence"


def test_fencing_existing_grant_preserves_only_identity_checked_reconciliation():
    from blueprint_pipeline.paid_resource_admission import (
        require_paid_resource_admission, require_paid_resource_admission_grant, PaidResourceAdmissionBlocked,
    )
    grant = require_paid_resource_admission({"schema_version": "paid_lane_admission.v1",
        "status": "admitted", "resource_class": "vast_provider_adapter", "blockers": [],
        "allocation_binding_digest": "sha256:" + "a" * 64},
        resource_class="vast_provider_adapter", expected_schema_version="paid_lane_admission.v1")

    def lost(boundary):
        raise HandoffLeaseError("lost", boundary)

    with bind_handoff_lease(lost):
        with pytest.raises(HandoffLeaseError):
            require_paid_resource_admission_grant(grant, resource_class="vast_provider_adapter")
        require_paid_resource_admission_grant(grant, resource_class="vast_provider_adapter",
            allocation_binding_digest="sha256:" + "a" * 64, require_allocation_binding=True,
            permit_reconciliation=True)
        with pytest.raises(PaidResourceAdmissionBlocked, match="binding_mismatch"):
            require_paid_resource_admission_grant(grant, resource_class="vast_provider_adapter",
                allocation_binding_digest="sha256:" + "b" * 64, require_allocation_binding=True,
                permit_reconciliation=True)


def test_opaque_grant_keeps_origin_lease_in_a_nested_worker_without_context():
    from blueprint_pipeline.paid_resource_admission import (
        require_paid_resource_admission, require_paid_resource_admission_grant,
    )
    lost = False

    def check(boundary):
        if lost:
            raise HandoffLeaseError("lost", boundary)

    with bind_handoff_lease(check):
        grant = require_paid_resource_admission({"schema_version": "example.v1", "status": "admitted"},
            resource_class="evaluator_api", expected_schema_version="example.v1")
    lost = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(require_paid_resource_admission_grant, grant, resource_class="evaluator_api")
        with pytest.raises(HandoffLeaseError):
            future.result()
