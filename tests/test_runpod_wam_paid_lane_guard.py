"""Fast-lane guard tests for the RunPod WAM async runner paid-lane hardening.

Hermetic (no subprocess, no network): the unified pre-spend chokepoint must gate
the create path, a pending_teardown.v1 record must survive a crash between
launch and collect, and teardown can only be proven by an API-confirmed
terminal state.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline import paid_lane_guard
from blueprint_pipeline import runpod_wam_async_runner as runner
from blueprint_pipeline.runpod_provider_adapter import RUNPOD_API_GATE_ENV


def _wam_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "wam_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo hi\n")
    return bundle


def _enable_paid_gates(monkeypatch) -> None:
    monkeypatch.setenv(RUNPOD_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_POD_LAUNCH_GATE_ENV, "true")
    monkeypatch.setenv(runner.RUNPOD_WAM_MAX_SPEND_USD_ENV, "0.75")
    monkeypatch.setenv(runner.RUNPOD_WAM_DISABLE_WARM_CANDIDATE_ENV, "true")
    monkeypatch.setattr(
        runner,
        "_read_runpod_api_key",
        lambda: (
            "runpod-secret-not-persisted",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )


def test_wam_create_refuses_spend_on_failing_unified_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    # Every legacy gate passes, but the unified chokepoint rejects the unpinned
    # image — no RunPod API call may happen.
    _enable_paid_gates(monkeypatch)

    def forbidden_runpod_request(**kwargs):
        raise AssertionError("no provider API call may happen on a failing preflight")

    monkeypatch.setattr(runner, "_runpod_request", forbidden_runpod_request)
    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=_wam_bundle(tmp_path),
        provider_bundle_url="https://spaces.example/bundle.zip?X-Amz-Signature=s",
        provider_output_put_url="https://spaces.example/output.zip?X-Amz-Signature=s",
        allow_paid_runpod_launch=True,
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:latest",
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert "runpod_wam_pre_spend_preflight_not_passed" in manifest["blockers"]
    assert any("image_not_pinned" in b for b in manifest["blockers"])
    preflight = manifest["pre_spend_preflight"]
    assert preflight["schema_version"] == "pre_spend_preflight.v1"
    assert preflight["lane"] == "runpod_wam_async"
    assert preflight["status"] == "FAIL"
    # No pending teardown record: nothing was launched.
    assert paid_lane_guard.load_pending_teardowns(include_closed=True) == []


def test_wam_create_opens_pending_teardown_and_binds_pod_id(
    tmp_path: Path, monkeypatch
) -> None:
    _enable_paid_gates(monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_runpod_request(**kwargs):
        # The teardown obligation must already be on disk before the billable call.
        assert paid_lane_guard.load_pending_teardowns(), (
            "pending_teardown record must exist before the provider create call"
        )
        calls.append(dict(kwargs))
        return 200, {"id": "pod-123"}

    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=_wam_bundle(tmp_path),
        provider_bundle_url="https://spaces.example/bundle.zip?X-Amz-Signature=s",
        provider_output_put_url="https://spaces.example/output.zip?X-Amz-Signature=s",
        allow_paid_runpod_launch=True,
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="now",
    )
    assert manifest["status"] == "pod_created"
    assert manifest["pre_spend_preflight"]["status"] == "PASS"
    records = paid_lane_guard.load_pending_teardowns()
    assert len(records) == 1
    record = records[0]
    assert record["provider"] == "runpod"
    assert record["lane"] == "runpod_wam_async"
    assert record["instance_id"] == "pod-123"
    state = json.loads(
        (tmp_path / "job" / "runpod_wam_async_state.json").read_text(encoding="utf-8")
    )
    assert state["pending_teardown_record"] == record["path"]


def test_wam_create_http_error_cancels_unbound_pending_record(
    tmp_path: Path, monkeypatch
) -> None:
    import urllib.error

    _enable_paid_gates(monkeypatch)

    def failing_runpod_request(**kwargs):
        raise urllib.error.HTTPError(
            "https://rest.runpod.io/v1/pods", 500, "boom", None, None
        )

    monkeypatch.setattr(runner, "_runpod_request", failing_runpod_request)
    manifest = runner.create_runpod_wam_async_run(
        job_dir=tmp_path / "job",
        bundle_path=_wam_bundle(tmp_path),
        provider_bundle_url="https://spaces.example/bundle.zip?X-Amz-Signature=s",
        provider_output_put_url="https://spaces.example/output.zip?X-Amz-Signature=s",
        allow_paid_runpod_launch=True,
        skip_public_staging_verification=True,
        image_name="docker.io/example/wam:20260629",
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert paid_lane_guard.load_pending_teardowns() == []
    cancelled = paid_lane_guard.load_pending_teardowns(include_closed=True)
    assert len(cancelled) == 1
    assert cancelled[0]["status"] == "cancelled_no_allocation"


def test_delete_pod_verifies_terminal_state_via_api(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_runpod_request(**kwargs):
        calls.append((kwargs["method"], kwargs["path"]))
        if kwargs["method"] == "DELETE":
            return 200, {}
        import urllib.error

        raise urllib.error.HTTPError(kwargs["path"], 404, "gone", None, None)

    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    manifest = runner._delete_pod(
        job_dir=tmp_path, pod_id="pod-1", api_key="k", generated_at="now"
    )
    assert manifest["status"] == "completed"
    assert manifest["terminal_state_api_confirmed"] is True
    assert manifest["verified_pod_status"] == "not_found"
    assert ("GET", "/pods/pod-1") in calls


def test_delete_pod_still_present_is_not_api_confirmed(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_runpod_request(**kwargs):
        if kwargs["method"] == "DELETE":
            return 200, {}
        return 200, {"id": "pod-1", "desiredStatus": "RUNNING"}

    monkeypatch.setattr(runner, "_runpod_request", fake_runpod_request)
    manifest = runner._delete_pod(
        job_dir=tmp_path, pod_id="pod-1", api_key="k", generated_at="now"
    )
    assert manifest["terminal_state_api_confirmed"] is False


def test_teardown_proof_from_poll_requires_api_confirmed_delete() -> None:
    # Legacy self-reported delete manifest (no API verification) must fail the proof.
    self_reported = runner._teardown_proof_from_runpod_poll(
        pod_id="pod-1",
        teardown_requested=True,
        teardown_manifest={"status": "completed"},
        teardown_action="delete",
        pod_status="RUNNING",
        keep_running_on_success=False,
        generated_at="now",
    )
    assert self_reported["status"] == "FAIL"
    assert any(
        "terminal_status_not_api_confirmed" in b for b in self_reported["blockers"]
    )

    confirmed = runner._teardown_proof_from_runpod_poll(
        pod_id="pod-1",
        teardown_requested=True,
        teardown_manifest={
            "status": "completed",
            "terminal_state_api_confirmed": True,
            "verified_pod_status": "not_found",
        },
        teardown_action="delete",
        pod_status="RUNNING",
        keep_running_on_success=False,
        generated_at="now",
    )
    assert confirmed["status"] == "PASS"
    assert confirmed["provider_terminal_status_source"] == "provider_api"


def test_teardown_proof_from_poll_accepts_api_observed_terminal_pod_status() -> None:
    proof = runner._teardown_proof_from_runpod_poll(
        pod_id="pod-1",
        teardown_requested=True,
        teardown_manifest=None,
        teardown_action="delete",
        pod_status="not_found",
        keep_running_on_success=False,
        generated_at="now",
    )
    assert proof["status"] == "PASS"
    assert proof["provider_terminal_status"] == "not_found"
    assert proof["provider_terminal_status_source"] == "provider_api"


def test_wam_reliability_manifest_closes_pending_record_on_proven_teardown(
    tmp_path: Path,
) -> None:
    record = paid_lane_guard.open_pending_teardown(
        provider="runpod",
        lane="runpod_wam_async",
        run_id="wam-run-1",
        instance_id="pod-9",
    )
    state = {
        "schema_version": runner.RUNPOD_WAM_STATE_SCHEMA_VERSION,
        "generated_at": "now",
        "pending_teardown_record": record["path"],
    }
    poll_manifest = {
        "pod_id": "pod-9",
        "pod_status": "not_found",
        "teardown_requested": True,
        "teardown_action": "delete",
        "output_zip_present": True,
        "provider_output_terminal": True,
        "provider_output_validation": {"status": "completed"},
    }
    runner._write_wam_provider_reliability_manifest(
        job_dir=tmp_path,
        state=state,
        poll_manifest=poll_manifest,
        teardown_manifest={
            "status": "completed",
            "terminal_state_api_confirmed": True,
            "verified_pod_status": "not_found",
        },
        generated_at="now",
    )
    assert paid_lane_guard.load_pending_teardowns() == []
    closed = paid_lane_guard.load_pending_teardowns(include_closed=True)
    assert closed[0]["status"] == "closed"
    assert closed[0]["teardown_proof"]["status"] == "PASS"


def test_wam_reliability_manifest_keeps_record_open_on_unproven_teardown(
    tmp_path: Path,
) -> None:
    record = paid_lane_guard.open_pending_teardown(
        provider="runpod",
        lane="runpod_wam_async",
        run_id="wam-run-2",
        instance_id="pod-10",
    )
    state = {
        "schema_version": runner.RUNPOD_WAM_STATE_SCHEMA_VERSION,
        "generated_at": "now",
        "pending_teardown_record": record["path"],
    }
    poll_manifest = {
        "pod_id": "pod-10",
        "pod_status": "RUNNING",
        "teardown_requested": True,
        "teardown_action": "delete",
    }
    runner._write_wam_provider_reliability_manifest(
        job_dir=tmp_path,
        state=state,
        poll_manifest=poll_manifest,
        teardown_manifest={"status": "completed"},
        generated_at="now",
    )
    assert len(paid_lane_guard.load_pending_teardowns()) == 1
    reliability = json.loads(
        (tmp_path / runner.RUNPOD_WAM_PROVIDER_RELIABILITY_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert reliability["teardown_proven"] is False
    assert reliability["open_billing_risk"] is True
