"""One stop signal per Vast provider run: a repeat never skips the instance DELETE."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import vast_provider_adapter as vpa
from blueprint_pipeline.vast_signal_guard import RECORD_FILENAME, OneStopSignal


def test_first_signal_is_the_stop_and_every_later_one_is_recorded_and_ignored(tmp_path):
    guard = OneStopSignal(tmp_path)
    assert guard.absorbs(15) is False
    assert guard.absorbs(15) is True and guard.absorbs(2) is True
    record = json.loads((tmp_path / RECORD_FILENAME).read_text())
    assert record["stop_taken_by"] == "interrupt" and record["ignored_signal_counts"] == {"15": 1, "2": 1}


def test_once_teardown_begins_even_a_first_signal_is_ignored(tmp_path):
    """A stop that lands after a normal run began its teardown cannot skip the DELETE."""
    guard = OneStopSignal(tmp_path)
    guard.teardown()
    assert guard.absorbs(15) is True
    assert json.loads((tmp_path / RECORD_FILENAME).read_text())["stop_taken_by"] == "teardown"


def test_a_record_that_cannot_be_written_never_interrupts_teardown(tmp_path, monkeypatch):
    from blueprint_pipeline import vast_signal_guard as guard_module

    def full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(guard_module, "write_json", full)
    guard = OneStopSignal(tmp_path)
    guard.teardown()
    assert guard.absorbs(15) is True


def test_second_sigterm_during_teardown_still_destroys_the_instance_and_returns_a_result(tmp_path, monkeypatch):
    """The allocator's monitor and its timer may both signal; the DELETE and the result must survive."""
    from tests.test_vast_provider_adapter import _configure_live_gates, _created_instance_detail, _paid_grant

    _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv(vpa.VAST_LAUNCH_LOCK_FILE_ENV, str(tmp_path / "own-test-launch.lock"))
    monkeypatch.setenv(vpa.HF_TOKEN_FILE_ENV, str(tmp_path / "missing_hf_token"))
    monkeypatch.setattr(vpa.time, "sleep", lambda *_: None)
    handlers = {}

    def capture(signum, handler):
        handlers[signum] = handler

    monkeypatch.setattr(vpa.signal, "signal", capture)
    calls = []

    def stop():
        handlers[vpa.signal.SIGTERM](vpa.signal.SIGTERM, None)

    def fake_api(*, method, path, api_key, payload=None, timeout_seconds=30):
        calls.append((method, path))
        if method == "POST" and path == "/bundles/":
            return 200, {"offers": [{"id": 506, "ask_contract_id": 506, "gpu_name": "RTX 3090", "dph_total": 0.20,
                                     "driver_version": "580.159.03", "machine_id": 910, "num_gpus": 1,
                                     "rentable": True}]}
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "PUT" and path == "/asks/506/":
            return 200, {"success": True, "new_contract": 991}
        if method == "GET" and path == "/instances/991/":
            return 200, _created_instance_detail(dph_total=0.2)
        if method == "PUT" and path == "/instances/request_logs/991/":
            stop()  # The monitor's stop: unwinds the run into its teardown.
            pytest.fail("the first stop must interrupt the run")
        if method == "DELETE" and path == "/instances/991/":
            stop()  # The timer's stop lands during the DELETE, then a third one.
            stop()
            return 200, {"success": True}
        raise AssertionError((method, path))

    monkeypatch.setattr(vpa, "_api_json", fake_api)
    job = tmp_path / "run"
    try:
        result = vpa.run_vast_provider_adapter(
            job_dir=job, mode="live-startup-probe", paid_resource_admission_grant=_paid_grant(),
            allow_vast_api_call=True, allow_instance_launch=True, poll_interval_seconds=0,
            startup_timeout_seconds=20, machine_avoidlist_path=tmp_path / "vast_machine_avoidlist.json")
    except KeyboardInterrupt:
        pytest.fail("a repeated stop escaped teardown: the DELETE and the result were skipped")
    assert result["reason"] == "vast_probe_interrupted"
    assert result["continuing_spend_from_this_run"] is False
    assert calls.count(("DELETE", "/instances/991/")) == 1
    teardown = json.loads((job / "vast_teardown_manifest.json").read_text())
    assert teardown["status"] == "completed" and teardown["continuing_spend_from_this_run"] is False
    # The stop's own evidence is intact; the ignored repeats are recorded beside it.
    assert json.loads((job / "vast_signal_handling_manifest.json").read_text())["status"] == "probe_interrupted_by_signal"
    assert json.loads((job / RECORD_FILENAME).read_text())["ignored_signal_counts"] == {"15": 2}

