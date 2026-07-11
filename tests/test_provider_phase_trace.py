"""Hermetic tests for the automatic provider phase/heartbeat trace."""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline import provider_phase_trace as T


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _recorder(tmp_path, clock=None, **overrides):
    kwargs = dict(
        run_id="run-1",
        attempt_id="attempt-1",
        launch_nonce="nonce-1",
        provider="runpod",
        clock=clock or _Clock(),
    )
    kwargs.update(overrides)
    return T.PhaseTraceRecorder(tmp_path / T.TRACE_FILENAME, **kwargs)


def test_state_changes_persist_rows_with_required_fields(tmp_path):
    clock = _Clock()
    rec = _recorder(tmp_path, clock)
    rec.record(T.PHASE_PRE_SPEND_INVENTORY)
    clock.now = 5.0
    rec.record(T.PHASE_ALLOCATION_REQUESTED)
    clock.now = 12.0
    rec.record(T.PHASE_ALLOCATION_CREATED, allocation_id="pod-1")

    payload = json.loads((tmp_path / T.TRACE_FILENAME).read_text())
    assert payload["schema_version"] == T.SCHEMA_VERSION
    rows = payload["rows"]
    assert [r["sequence"] for r in rows] == [1, 2, 3]
    assert [r["phase"] for r in rows] == [
        T.PHASE_PRE_SPEND_INVENTORY,
        T.PHASE_ALLOCATION_REQUESTED,
        T.PHASE_ALLOCATION_CREATED,
    ]
    for row in rows:
        assert row["run_id"] == "run-1"
        assert row["attempt_id"] == "attempt-1"
        assert row["launch_nonce"] == "nonce-1"
        assert row["utc_timestamp"]
        assert row["elapsed_seconds"] >= 0
    assert rows[-1]["allocation_id"] == "pod-1"
    assert payload["raw_provider_responses_recorded"] is False


def test_unknown_phase_rejected(tmp_path):
    rec = _recorder(tmp_path)
    with pytest.raises(T.PhaseTraceRejected):
        rec.record("made_up_phase")


def test_heartbeat_persists_only_after_interval(tmp_path):
    clock = _Clock()
    rec = _recorder(tmp_path, clock, heartbeat_interval_seconds=60)
    rec.record(T.PHASE_IMAGE_PULL_NO_RUNTIME)
    clock.now = 30.0
    assert rec.heartbeat() is None
    clock.now = 61.0
    row = rec.heartbeat()
    assert row is not None and row["kind"] == "heartbeat"
    assert row["phase"] == T.PHASE_IMAGE_PULL_NO_RUNTIME
    # A second immediate heartbeat is suppressed until the next interval.
    assert rec.heartbeat() is None
    clock.now = 125.0
    assert rec.heartbeat() is not None


def test_heartbeat_before_any_phase_is_noop(tmp_path):
    rec = _recorder(tmp_path)
    assert rec.heartbeat() is None


def test_trace_rebinds_rows_to_each_retry_without_resetting_sequence(tmp_path):
    rec = _recorder(tmp_path)
    rec.bind_attempt(attempt_id="run-1-attempt-01", launch_nonce="nonce-1-a01")
    first = rec.record(T.PHASE_PRE_SPEND_INVENTORY)
    rec.bind_attempt(attempt_id="run-1-attempt-02", launch_nonce="nonce-1-a02")
    second = rec.record(T.PHASE_PRE_SPEND_INVENTORY)
    assert first["attempt_id"] == "run-1-attempt-01"
    assert first["launch_nonce"] == "nonce-1-a01"
    assert second["attempt_id"] == "run-1-attempt-02"
    assert second["launch_nonce"] == "nonce-1-a02"
    assert second["sequence"] == first["sequence"] + 1
    assert T.validate_phase_trace(rec.payload()) == []


def test_signed_url_query_strings_are_never_persisted(tmp_path):
    rec = _recorder(tmp_path)
    with pytest.raises(T.PhaseTraceRejected):
        rec.record(
            T.PHASE_BUNDLE_DOWNLOAD,
            detail={"url": "https://s3/x?X-Amz-Signature=deadbeef"},
        )
    row = rec.record(
        T.PHASE_BUNDLE_DOWNLOAD, detail={"url": "https://s3/bucket/key?version=2"}
    )
    assert row["detail"]["url"] == "https://s3/bucket/key?<query-stripped>"
    text = (tmp_path / T.TRACE_FILENAME).read_text()
    assert "version=2" not in text


def test_raw_response_and_secret_detail_keys_rejected(tmp_path):
    rec = _recorder(tmp_path)
    for key in ("raw_response", "runpod_api_key", "signed_url", "auth_token"):
        with pytest.raises(T.PhaseTraceRejected):
            rec.record(T.PHASE_CAPACITY_PROBE, detail={key: "x"})
    with pytest.raises(T.PhaseTraceRejected):
        rec.record(T.PHASE_CAPACITY_PROBE, detail={"nested": {"a": 1}})


def test_missing_identity_rejected(tmp_path):
    with pytest.raises(T.PhaseTraceRejected):
        T.PhaseTraceRecorder(tmp_path / "t.json", run_id="", attempt_id="a",
                             launch_nonce="n")
    with pytest.raises(T.PhaseTraceRejected):
        T.PhaseTraceRecorder(tmp_path / "t.json", run_id="r", attempt_id="a",
                             launch_nonce="")


def test_callback_ingestion_rejects_stale_duplicate_out_of_order(tmp_path):
    rec = _recorder(tmp_path)
    rec.record(T.PHASE_EARLY_MARKER)
    good = {"launch_nonce": "nonce-1", "phase": T.PHASE_CUDA_DRIVER_CHECK,
            "sequence": 5}
    accepted = rec.ingest_callback_row(good)
    assert accepted["sequence"] == 5 and accepted["kind"] == "callback"
    # Stale nonce.
    with pytest.raises(T.PhaseTraceRejected):
        rec.ingest_callback_row({**good, "launch_nonce": "old-nonce", "sequence": 9})
    # Duplicate / out-of-order sequence.
    with pytest.raises(T.PhaseTraceRejected):
        rec.ingest_callback_row(dict(good))
    with pytest.raises(T.PhaseTraceRejected):
        rec.ingest_callback_row({**good, "sequence": 3})
    with pytest.raises(T.PhaseTraceRejected):
        rec.ingest_callback_row({**good, "sequence": 6, "phase": "bogus"})


def test_validate_phase_trace_accepts_recorder_output(tmp_path):
    clock = _Clock()
    rec = _recorder(tmp_path, clock)
    for phase in (T.PHASE_PRE_SPEND_INVENTORY, T.PHASE_ALLOCATION_REQUESTED,
                  T.PHASE_TEARDOWN_VERIFICATION, T.PHASE_FINAL_INVENTORY):
        clock.now += 10
        rec.record(phase)
    assert T.validate_phase_trace(rec.payload()) == []


def test_validate_phase_trace_flags_tampering(tmp_path):
    rec = _recorder(tmp_path)
    rec.record(T.PHASE_PRE_SPEND_INVENTORY)
    rec.record(T.PHASE_ALLOCATION_REQUESTED)
    payload = rec.payload()
    payload["rows"][1]["sequence"] = 1
    blockers = T.validate_phase_trace(payload)
    assert "phase_trace_duplicate_sequence" in blockers
    assert "phase_trace_out_of_order_sequence" in blockers

    payload = rec.payload()
    payload["rows"][0]["launch_nonce"] = "other"
    assert "phase_trace_row_nonce_mismatch" in T.validate_phase_trace(payload)

    assert "phase_trace_rows_missing" in T.validate_phase_trace(
        {"schema_version": T.SCHEMA_VERSION, "launch_nonce": "n", "rows": []}
    )


def test_phase_durations_from_state_changes(tmp_path):
    clock = _Clock()
    rec = _recorder(tmp_path, clock)
    rec.record(T.PHASE_ALLOCATION_REQUESTED)
    clock.now = 20.0
    rec.record(T.PHASE_IMAGE_PULL_NO_RUNTIME)
    clock.now = 50.0
    rec.heartbeat()  # heartbeats must not distort durations
    clock.now = 320.0
    rec.record(T.PHASE_RUNTIME_PRESENT)
    durations = T.phase_durations(rec.payload())
    assert durations[T.PHASE_ALLOCATION_REQUESTED] == 20.0
    assert durations[T.PHASE_IMAGE_PULL_NO_RUNTIME] == 300.0
    assert durations[T.PHASE_RUNTIME_PRESENT] == 0.0
