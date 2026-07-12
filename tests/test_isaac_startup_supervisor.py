"""Hermetic fake-provider tests for the atomic canary-to-worker supervisor.

Covers the P0-1 acceptance scenarios: no-capacity, no-runtime, stale marker,
bad driver, empty frame, success-and-delete, success-and-promote, exception
during promotion, and cap exhaustion — plus the one-live-resource and
no-second-cold-launch invariants and API-confirmed teardown proof on every
terminal non-promoted path.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time

import pytest

from blueprint_pipeline import isaac_startup_supervisor as SUP
from blueprint_pipeline import machine_quarantine_registry as Q
from blueprint_pipeline.paid_lane_guard import load_pending_teardowns

DIGEST = "sha256:" + "c" * 64


class FakeProvider:
    """In-memory provider with API-consistent allocate/inspect/terminate.

    Each allocation lands on a fresh machine by default (machine-1, machine-2,
    ...); pass ``fixed_machine_id`` to model the provider handing back the same
    host repeatedly.
    """

    name = "runpod"

    def __init__(self, *, allocate_results=None, fixed_machine_id=None):
        self.fixed_machine_id = fixed_machine_id
        self.allocate_results = list(allocate_results or [])
        self.allocate_calls: list[dict] = []
        self.terminated: list[str] = []
        self.live: set[str] = set()
        self.machines: dict[str, str] = {}
        self._counter = 0

    def _register(self, instance_id: str) -> None:
        self.live.add(instance_id)
        self.machines[instance_id] = (
            self.fixed_machine_id or f"machine-{len(self.machines) + 1}"
        )

    def allocate(self, **ctx):
        self.allocate_calls.append(ctx)
        if self.allocate_results:
            result = self.allocate_results.pop(0)
            if result.get("status") == "launched":
                self._register(result["instance_id"])
            return result
        self._counter += 1
        instance_id = f"pod-{self._counter}"
        self._register(instance_id)
        return {"status": "launched", "instance_id": instance_id}

    def inspect(self, instance_id):
        if instance_id not in self.live:
            return {"status": "unavailable", "http": 404, "instance_id": instance_id}
        return {
            "status": "observed",
            "http": 200,
            "instance_id": instance_id,
            "desiredStatus": "RUNNING",
            "runtime_present": False,
            "public_ip_present": False,
            "machineId": self.machines[instance_id],
            "costPerHr": 0.49,
        }

    def terminate(self, instance_id):
        self.live.discard(instance_id)
        self.terminated.append(instance_id)
        return {"status": "terminated", "http": 200}


def _request(tmp_path, **overrides):
    kwargs = dict(
        run_id="run-1",
        launch_nonce="nonce-1",
        provider="runpod",
        gpu_types=("NVIDIA A40", "NVIDIA RTX A6000"),
        image_ref=f"docker.io/x/worker@{DIGEST}",
        image_manifest_checksum="deadbeef",
        max_attempts=3,
        wall_clock_cap_seconds=3600.0,
        total_spend_cap_usd=5.0,
        hourly_rate_usd=0.49,
        per_attempt_reserved_seconds=3600.0,
        terminal_mode=SUP.TERMINAL_MODE_TERMINATE,
    )
    kwargs.update(overrides)
    return SUP.StartupSupervisorRequest(**kwargs)


def _run(tmp_path, request, provider, *, marker=None, canary=None, **kwargs):
    def default_marker(**ctx):
        return {"status": "marker_verified"}

    def default_canary(**ctx):
        return {"status": "passed"}

    def inventory():
        return {"live_resource_count": len(provider.live), "resources": sorted(provider.live)}

    kwargs.setdefault("quarantine_registry_dir", tmp_path / "quarantine")
    kwargs.setdefault("teardown_registry_dir", tmp_path / "teardowns")
    return SUP.run_startup_supervisor(
        request=request,
        provider_client=provider,
        out_dir=tmp_path / "out",
        inventory=kwargs.pop("inventory", inventory),
        wait_for_marker=marker or default_marker,
        run_canary=canary or default_canary,
        **kwargs,
    )


def _artifact(tmp_path, name):
    return json.loads((tmp_path / "out" / name).read_text())


# --------------------------- terminal scenarios ---------------------------


def test_success_and_delete_verifies_teardown_and_zero_inventory(tmp_path):
    provider = FakeProvider()
    manifest = _run(tmp_path, _request(tmp_path), provider)
    assert manifest["status"] == "passed_and_terminated"
    assert manifest["blockers"] == []
    attempt = manifest["attempts"][0]
    assert attempt["outcome"] == "canary_passed_terminated"
    assert attempt["teardown_proof"]["status"] == "PASS"
    assert attempt["teardown_proof"]["provider_terminal_status"] == "not_found"
    assert manifest["final_inventory"]["live_resource_count"] == 0
    assert provider.terminated == ["pod-1"]
    # All required artifacts exist.
    for name in (
        SUP.MANIFEST_FILENAME,
        SUP.ATTEMPTS_FILENAME,
        SUP.QUARANTINE_REFERENCES_FILENAME,
        SUP.FINAL_INVENTORY_FILENAME,
        "provider_phase_trace.json",
    ):
        assert (tmp_path / "out" / name).exists()
    # The pending teardown record is closed.
    open_records = load_pending_teardowns(registry_dir=tmp_path / "teardowns")
    assert open_records == []


def test_success_and_promote_transfers_ownership_without_second_launch(tmp_path):
    provider = FakeProvider()
    full_job_calls: list[dict] = []

    def full_job(allocation):
        full_job_calls.append(dict(allocation))
        return {"status": "started"}

    manifest = _run(
        tmp_path,
        _request(tmp_path, terminal_mode=SUP.TERMINAL_MODE_PROMOTE),
        provider,
        full_job=full_job,
    )
    assert manifest["status"] == "passed_and_promoted"
    # No second cold launch: exactly one allocation ever requested.
    assert len(provider.allocate_calls) == 1
    assert provider.terminated == []
    receipt = manifest["ownership_transfer_receipt"]
    assert receipt["instance_id"] == "pod-1"
    assert receipt["teardown_obligation_transferred"] is True
    assert receipt["revoked"] is False
    assert full_job_calls and full_job_calls[0]["instance_id"] == "pod-1"
    # The pending teardown record stays OPEN, owned by the full job.
    records = load_pending_teardowns(registry_dir=tmp_path / "teardowns")
    assert len(records) == 1
    assert records[0]["status"] == "open"
    assert records[0]["owner_lane"] == "full_job"
    receipt_file = _artifact(tmp_path, SUP.OWNERSHIP_RECEIPT_FILENAME)
    assert receipt_file["pending_teardown_record"] == records[0]["path"]


def test_exception_during_promotion_runs_finalizer_and_revokes_receipt(tmp_path):
    provider = FakeProvider()

    def full_job(allocation):
        raise RuntimeError("bundle upload exploded")

    manifest = _run(
        tmp_path,
        _request(tmp_path, terminal_mode=SUP.TERMINAL_MODE_PROMOTE),
        provider,
        full_job=full_job,
    )
    assert manifest["status"] == "promotion_failed"
    assert "startup_supervisor_promotion_failed_terminated" in manifest["blockers"]
    assert provider.terminated == ["pod-1"]
    receipt = manifest["ownership_transfer_receipt"]
    assert receipt["revoked"] is True
    attempt = manifest["attempts"][0]
    assert attempt["failure_class"] == "exception_during_promotion"
    assert attempt["teardown_proof"]["status"] == "PASS"
    assert manifest["final_inventory"]["live_resource_count"] == 0
    assert load_pending_teardowns(registry_dir=tmp_path / "teardowns") == []


def test_no_capacity_attempts_are_capacity_outcomes_and_retry(tmp_path):
    provider = FakeProvider(
        allocate_results=[
            {
                "status": "blocked",
                "capacity_outcome": True,
                "allocation_created": False,
                "blockers": ["runpod_secure_cloud_create_capacity_unavailable"],
            },
            {"status": "launched", "instance_id": "pod-9"},
        ]
    )
    manifest = _run(tmp_path, _request(tmp_path), provider)
    assert manifest["status"] == "passed_and_terminated"
    first, second = manifest["attempts"]
    assert first["failure_class"] == SUP.FAILURE_CLASS_NO_CAPACITY
    assert first["outcome"] == "allocation_failed"
    # No instance was created, so no teardown proof is required for attempt 1.
    assert "teardown_proof" not in first
    assert second["outcome"] == "canary_passed_terminated"


def test_no_runtime_marker_timeout_terminates_proves_and_quarantines(tmp_path):
    provider = FakeProvider()
    markers = iter(
        [{"status": "marker_timeout"}, {"status": "marker_verified"}]
    )
    manifest = _run(
        tmp_path, _request(tmp_path), provider, marker=lambda **ctx: next(markers)
    )
    assert manifest["status"] == "passed_and_terminated"
    first = manifest["attempts"][0]
    assert first["failure_class"] == SUP.FAILURE_CLASS_NO_RUNTIME
    assert first["teardown_proof"]["status"] == "PASS"
    assert provider.terminated[0] == "pod-1"
    # The dead machine is durably quarantined for this digest+Isaac identity.
    entry = Q.find_active_quarantine(
        provider="runpod",
        machine_id="machine-1",
        image_digest=DIGEST,
        isaac_version="6.0.0",
        registry_dir=tmp_path / "quarantine",
    )
    assert entry is not None
    assert entry["failure_class"] == "container_never_started"
    assert entry["phase"] == Q.PHASE_PRE_RUNTIME


def test_stale_marker_is_terminated_but_not_machine_quarantined(tmp_path):
    provider = FakeProvider()
    markers = iter([{"status": "stale_marker"}, {"status": "marker_verified"}])
    manifest = _run(
        tmp_path, _request(tmp_path), provider, marker=lambda **ctx: next(markers)
    )
    first = manifest["attempts"][0]
    assert first["failure_class"] == SUP.FAILURE_CLASS_STALE_MARKER
    assert first["teardown_proof"]["status"] == "PASS"
    assert (
        Q.load_quarantine_entries(registry_dir=tmp_path / "quarantine") == []
    )
    assert manifest["status"] == "passed_and_terminated"


def test_bad_driver_canary_quarantines_before_teardown_and_retries(tmp_path):
    provider = FakeProvider()
    canaries = iter(
        [
            {
                "status": "blocked",
                "blockers": ["isaac_sim_6_rtx_driver_unsupported"],
                "failure_class": "driver_incompatible",
                "gpu_name": "NVIDIA L40S",
                "driver_version": "570.124.06",
            },
            {"status": "passed"},
        ]
    )
    manifest = _run(
        tmp_path, _request(tmp_path), provider, canary=lambda **ctx: next(canaries)
    )
    assert manifest["status"] == "passed_and_terminated"
    first = manifest["attempts"][0]
    assert first["outcome"] == "canary_failed"
    assert first["failure_class"] == "driver_incompatible"
    assert first["teardown_proof"]["status"] == "PASS"
    entry = Q.find_active_quarantine(
        provider="runpod",
        machine_id="machine-1",
        image_digest=DIGEST,
        isaac_version="6.0.0",
        registry_dir=tmp_path / "quarantine",
    )
    assert entry is not None
    assert entry["failure_class"] == "driver_incompatible"
    assert entry["phase"] == Q.PHASE_RUNTIME_CANARY
    assert entry["driver_version"] == "570.124.06"
    refs = _artifact(tmp_path, SUP.QUARANTINE_REFERENCES_FILENAME)
    assert refs["references"][0]["action"] == "recorded"


def test_empty_frame_canary_failure_terminates_with_proof(tmp_path):
    provider = FakeProvider()
    canaries = iter(
        [
            {
                "status": "blocked",
                "blockers": ["rtx_smoke_frame_render_failed"],
                "failure_class": "empty_frame",
            },
            {"status": "passed"},
        ]
    )
    manifest = _run(
        tmp_path, _request(tmp_path), provider, canary=lambda **ctx: next(canaries)
    )
    first = manifest["attempts"][0]
    assert first["failure_class"] == "empty_frame"
    assert first["teardown_proof"]["status"] == "PASS"
    assert manifest["status"] == "passed_and_terminated"


def test_cap_exhaustion_blocks_before_provider_allocation(tmp_path):
    provider = FakeProvider()
    request = _request(
        tmp_path,
        total_spend_cap_usd=0.6,
        hourly_rate_usd=0.49,
        per_attempt_reserved_seconds=3600.0,
        max_attempts=3,
        wall_clock_cap_seconds=1e9,
    )
    ticks = {"now": 0.0}

    def clock():
        ticks["now"] += 600.0
        return ticks["now"]

    canaries = iter([{"status": "blocked", "failure_class": "empty_frame"}])
    manifest = _run(
        tmp_path, request, provider, canary=lambda **ctx: next(canaries), clock=clock
    )
    # Attempt 1 settles a ~0.16 USD elapsed upper bound; admitting attempt 2's
    # 0.49 reservation would cross the 0.60 cap, so no second allocation.
    assert manifest["status"] == "blocked"
    assert "startup_supervisor_spend_cap_exhausted" in manifest["blockers"]
    assert len(provider.allocate_calls) == 1
    second = manifest["attempts"][1]
    assert second["outcome"] == "blocked_spend_cap"
    ledger = _artifact(tmp_path, "startup_cumulative_spend_ledger.json")
    assert ledger["schema_version"] == "startup_cumulative_spend_ledger.v1"
    assert manifest["final_inventory"]["live_resource_count"] == 0


# --------------------------- invariants ---------------------------


def test_existing_billable_resource_blocks_new_allocation(tmp_path):
    provider = FakeProvider()
    manifest = _run(
        tmp_path,
        _request(tmp_path),
        provider,
        inventory=lambda: {"live_resource_count": 1, "resources": ["stray-pod"]},
    )
    assert manifest["status"] == "blocked"
    assert (
        "startup_supervisor_existing_billable_resource_present"
        in manifest["blockers"]
    )
    assert provider.allocate_calls == []


def test_quarantined_machine_reallocation_is_terminated_immediately(tmp_path):
    Q.record_machine_quarantine(
        provider="runpod",
        machine_id="machine-1",
        image_digest=DIGEST,
        isaac_version="6.0.0",
        failure_class="container_never_started",
        phase=Q.PHASE_PRE_RUNTIME,
        registry_dir=tmp_path / "quarantine",
    )
    provider = FakeProvider(fixed_machine_id="machine-1")
    marker_calls: list[dict] = []

    def marker(**ctx):
        marker_calls.append(ctx)
        return {"status": "marker_verified"}

    manifest = _run(
        tmp_path, _request(tmp_path, max_attempts=1), provider, marker=marker
    )
    attempt = manifest["attempts"][0]
    assert attempt["failure_class"] == SUP.FAILURE_CLASS_QUARANTINED_MACHINE
    assert attempt["teardown_proof"]["status"] == "PASS"
    # Terminated before waiting on a marker: no useful-work wait on a dead host.
    assert marker_calls == []
    assert provider.terminated == ["pod-1"]
    refs = _artifact(tmp_path, SUP.QUARANTINE_REFERENCES_FILENAME)
    assert refs["references"][0]["action"] == "matched"


def test_every_non_promoted_terminal_path_has_api_confirmed_teardown(tmp_path):
    provider = FakeProvider()
    canaries = iter(
        [
            {"status": "blocked", "failure_class": "empty_frame"},
            {"status": "blocked", "failure_class": "driver_incompatible"},
            {"status": "blocked", "failure_class": "rtx_init_failed"},
        ]
    )
    manifest = _run(
        tmp_path,
        _request(tmp_path, max_attempts=3),
        provider,
        canary=lambda **ctx: next(canaries),
    )
    assert manifest["status"] == "blocked"
    assert "startup_supervisor_attempts_exhausted" in manifest["blockers"]
    for attempt in manifest["attempts"]:
        proof = attempt["teardown_proof"]
        assert proof["status"] == "PASS"
        assert proof["provider_terminal_status_source"] == "provider_api"
    assert load_pending_teardowns(registry_dir=tmp_path / "teardowns") == []
    assert manifest["final_inventory"]["live_resource_count"] == 0


def test_wall_clock_cap_stops_new_attempts(tmp_path):
    provider = FakeProvider()
    clock_values = iter([0.0, 0.0, 10.0, 10.0, 4000.0, 4000.0, 4000.0, 4000.0])

    def clock():
        try:
            return next(clock_values)
        except StopIteration:
            return 4000.0

    canaries = iter([{"status": "blocked", "failure_class": "empty_frame"}])
    manifest = _run(
        tmp_path,
        _request(tmp_path, wall_clock_cap_seconds=3600.0),
        provider,
        canary=lambda **ctx: next(canaries),
        clock=clock,
    )
    assert manifest["status"] == "blocked"
    assert "startup_supervisor_wall_clock_cap_exceeded" in manifest["blockers"]


def test_unpinned_image_blocks_before_any_provider_call(tmp_path):
    provider = FakeProvider()
    manifest = _run(
        tmp_path, _request(tmp_path, image_ref="docker.io/x/worker:latest"), provider
    )
    assert manifest["status"] == "blocked"
    assert "startup_supervisor_image_not_digest_pinned" in manifest["blockers"]
    assert provider.allocate_calls == []


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        (
            "wall_clock_cap_seconds",
            float("nan"),
            "startup_supervisor_wall_clock_cap_invalid",
        ),
        (
            "total_spend_cap_usd",
            float("inf"),
            "startup_supervisor_total_spend_cap_invalid",
        ),
        (
            "hourly_rate_usd",
            float("-inf"),
            "startup_supervisor_hourly_rate_invalid",
        ),
        (
            "per_attempt_reserved_seconds",
            float("nan"),
            "startup_supervisor_per_attempt_reserved_seconds_invalid",
        ),
    ],
)
def test_nonfinite_supervisor_caps_block_before_provider_call(
    tmp_path, field, value, blocker
):
    provider = FakeProvider()

    manifest = _run(tmp_path, _request(tmp_path, **{field: value}), provider)

    assert manifest["status"] == "blocked"
    assert blocker in manifest["blockers"]
    assert provider.allocate_calls == []


def test_provider_reported_image_digest_mismatch_terminates_and_blocks(tmp_path):
    class _WrongImageProvider(FakeProvider):
        def inspect(self, instance_id):
            result = super().inspect(instance_id)
            if result.get("http") == 200:
                result["imageName"] = "docker.io/x/worker@sha256:" + "d" * 64
            return result

    provider = _WrongImageProvider()
    manifest = _run(tmp_path, _request(tmp_path, max_attempts=1), provider)

    assert manifest["status"] == "blocked"
    assert "startup_supervisor_provider_image_digest_mismatch" in manifest["blockers"]
    assert provider.terminated == ["pod-1"]
    assert manifest["attempts"][0]["outcome"] == "provider_image_not_verified"


def test_provider_exposed_empty_image_identity_terminates_and_blocks(tmp_path):
    class _MissingImageProvider(FakeProvider):
        def inspect(self, instance_id):
            result = super().inspect(instance_id)
            if result.get("http") == 200:
                result["imageName"] = None
            return result

    provider = _MissingImageProvider()
    manifest = _run(tmp_path, _request(tmp_path, max_attempts=1), provider)

    assert manifest["status"] == "blocked"
    assert (
        "startup_supervisor_provider_image_digest_unverifiable"
        in manifest["blockers"]
    )
    assert provider.terminated == ["pod-1"]


def test_supervisor_exception_mid_attempt_still_finalizes_teardown(tmp_path):
    provider = FakeProvider()

    def exploding_canary(**ctx):
        raise SUP.SupervisorInterrupted("sigterm")

    manifest = _run(tmp_path, _request(tmp_path), provider, canary=exploding_canary)
    assert manifest["status"] == "aborted"
    assert "startup_supervisor_interrupted" in manifest["blockers"]
    # Finalizer terminated the live pod and proved it gone.
    assert provider.terminated == ["pod-1"]
    assert manifest["finalizer_teardown_proof"]["status"] == "PASS"
    assert load_pending_teardowns(registry_dir=tmp_path / "teardowns") == []
    assert manifest["final_inventory"]["live_resource_count"] == 0


def test_main_thread_interrupt_does_not_wait_for_blocked_callback(tmp_path):
    provider = FakeProvider()
    callback_started = threading.Event()
    release_callback = threading.Event()

    def blocked_marker(**ctx):
        callback_started.set()
        release_callback.wait(timeout=10)
        return {"status": "marker_verified"}

    def interrupt_main():
        assert callback_started.wait(timeout=2)
        os.kill(os.getpid(), signal.SIGTERM)

    interrupter = threading.Thread(target=interrupt_main)
    interrupter.start()
    started = time.monotonic()
    try:
        manifest = _run(
            tmp_path,
            _request(tmp_path),
            provider,
            marker=blocked_marker,
        )
    finally:
        release_callback.set()
        interrupter.join(timeout=2)

    assert time.monotonic() - started < 2.0
    assert manifest["status"] == "aborted"
    assert "startup_supervisor_interrupted" in manifest["blockers"]
    assert provider.terminated == ["pod-1"]
    assert manifest["finalizer_teardown_proof"]["status"] == "PASS"


def test_allocate_raising_keeps_ambiguous_pending_teardown_open_and_settles(tmp_path):
    class _RaisingProvider(FakeProvider):
        def allocate(self, **ctx):
            raise RuntimeError("provider api 502")

    provider = _RaisingProvider()
    with pytest.raises(RuntimeError):
        _run(tmp_path, _request(tmp_path), provider)
    records = load_pending_teardowns(registry_dir=tmp_path / "teardowns")
    assert len(records) == 1
    assert records[0]["allocation_outcome_ambiguous"] is True
    assert records[0]["ambiguity_reason"] == (
        "provider_allocate_raised_before_allocation"
    )
    manifest = _artifact(tmp_path, SUP.MANIFEST_FILENAME)
    assert manifest["status"] == "aborted"
    attempt = manifest["attempts"][0]
    assert attempt["outcome"] == "allocation_raised"
    assert "provider api 502" in attempt["allocation_error"]
    ledger = _artifact(tmp_path, "startup_cumulative_spend_ledger.json")
    assert ledger["attempts"][0]["settled"] is True


def test_ambiguous_no_id_allocation_response_aborts_without_retry(tmp_path):
    provider = FakeProvider(
        allocate_results=[
            {
                "status": "blocked",
                "blockers": ["runpod_create_outcome_ambiguous"],
                "allocation_outcome_ambiguous": True,
            },
            {"status": "launched", "instance_id": "must-not-launch"},
        ]
    )

    with pytest.raises(RuntimeError, match="provider_allocation_outcome_ambiguous"):
        _run(tmp_path, _request(tmp_path), provider)

    assert len(provider.allocate_calls) == 1
    records = load_pending_teardowns(registry_dir=tmp_path / "teardowns")
    assert len(records) == 1
    assert records[0]["allocation_outcome_ambiguous"] is True
    manifest = _artifact(tmp_path, SUP.MANIFEST_FILENAME)
    assert "startup_supervisor_allocation_outcome_ambiguous" in manifest["blockers"]


def test_unverified_teardown_blocks_next_supervisor_attempt(tmp_path):
    class _UnkillableProvider(FakeProvider):
        def terminate(self, instance_id):
            self.terminated.append(instance_id)
            return {"status": "terminate_failed", "http": 500}

    provider = _UnkillableProvider()
    manifest = _run(
        tmp_path,
        _request(tmp_path, max_attempts=3),
        provider,
        marker=lambda **_ctx: {"status": "marker_timeout"},
    )

    assert len(provider.allocate_calls) == 1
    assert "startup_supervisor_teardown_unverified" in manifest["blockers"]
    assert "startup_supervisor_teardown_unverified_before_retry" in manifest[
        "blockers"
    ]
    assert len(load_pending_teardowns(registry_dir=tmp_path / "teardowns")) == 1


def test_pending_close_error_retains_state_and_blocks_supervisor_retry(
    tmp_path, monkeypatch
):
    provider = FakeProvider()
    monkeypatch.setattr(
        SUP,
        "close_pending_teardown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    manifest = _run(tmp_path, _request(tmp_path, max_attempts=3), provider)

    assert len(provider.allocate_calls) == 1
    assert "startup_supervisor_teardown_unverified" in manifest["blockers"]
    proof = manifest["finalizer_teardown_proof"]
    assert proof["pending_teardown_status"] == "close_failed"
    assert proof["pending_teardown_close_error_type"] == "OSError"


def test_teardown_trace_write_failure_cannot_skip_provider_delete(
    tmp_path, monkeypatch
):
    provider = FakeProvider()
    original_record = SUP.PhaseTraceRecorder.record

    def flaky_record(self, phase, **kwargs):
        if phase == SUP.PHASE_DELETE:
            raise OSError("trace disk full")
        return original_record(self, phase, **kwargs)

    monkeypatch.setattr(SUP.PhaseTraceRecorder, "record", flaky_record)

    manifest = _run(tmp_path, _request(tmp_path, max_attempts=1), provider)

    assert provider.terminated == ["pod-1"]
    assert manifest["status"] == "blocked"
    assert "startup_supervisor_cleanup_evidence_write_failed" in manifest["blockers"]
    assert manifest["cleanup_evidence_errors"] == [
        {"phase": SUP.PHASE_DELETE, "error_type": "OSError"}
    ]


def test_unexpected_exception_finalizes_and_reraises(tmp_path):
    provider = FakeProvider()

    def exploding_canary(**ctx):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        _run(tmp_path, _request(tmp_path), provider, canary=exploding_canary)
    assert provider.terminated == ["pod-1"]
    manifest = _artifact(tmp_path, SUP.MANIFEST_FILENAME)
    assert manifest["status"] == "aborted"
    assert "startup_supervisor_exception" in manifest["blockers"]
    assert load_pending_teardowns(registry_dir=tmp_path / "teardowns") == []


def test_phase_trace_and_attempts_artifacts_are_bound_to_nonce(tmp_path):
    provider = FakeProvider()
    _run(tmp_path, _request(tmp_path), provider)
    trace = _artifact(tmp_path, "provider_phase_trace.json")
    assert trace["launch_nonce"] == "nonce-1"
    phases = [row["phase"] for row in trace["rows"]]
    for phase in (
        "pre_spend_inventory",
        "allocation_requested",
        "allocation_created",
        "machine_identity_observed",
        "early_marker",
        "delete",
        "teardown_verification",
        "final_inventory",
    ):
        assert phase in phases
    attempts = _artifact(tmp_path, SUP.ATTEMPTS_FILENAME)
    assert attempts["launch_nonce"] == "nonce-1"
    assert attempts["attempts"][0]["launch_nonce"] == "nonce-1-a01"
    assert attempts["spend_ledger"]["includes_failed_attempts"] is True
