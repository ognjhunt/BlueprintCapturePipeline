"""AUT-10 current-source stop intent, bounded outage and exact-pid isolation."""

import json
import threading
import time

import pytest

from blueprint_pipeline import allocator_invocation as jobs
from blueprint_pipeline import beta_policy_screening_control as control
from blueprint_pipeline import beta_policy_screening_worker as worker


def binding(**extra):
    return {"site_id": "site-current", "source": {"capture_id": "capture"},
            "prepared_digest": "sha256:" + "a" * 64, "initial_observed_at_epoch": 1000,
            "execution_deadline_epoch": None, **extra}


@pytest.mark.parametrize("code", ["beta_screening_current_capture_consent_required",
    "beta_screening_current_task_context_changed", "website_control_task-context_http_409:consent_revoked"])
def test_revocation_persists_exact_historical_stop_and_never_reauthorizes(tmp_path, monkeypatch, code):
    def revoked(*args, **kwargs):
        raise ValueError(code)
    monkeypatch.setattr(worker, "require_current_source", revoked)
    value = binding()
    stopped = control.observe(control=value, root=tmp_path, now=1001)
    assert stopped["reason"] == "beta_screening_current_source_revoked"
    assert stopped["prepared_digest"] == value["prepared_digest"]
    assert stopped["applicability"] == "historical_only" and stopped["new_processing_authorized"] is False
    monkeypatch.setattr(worker, "require_current_source", lambda *a, **k: {})
    assert control.observe(control=value, root=tmp_path, now=1002) == stopped


def test_unreachable_control_has_bounded_grace_then_stops(tmp_path, monkeypatch):
    def unavailable(*a, **kw):
        raise OSError("no service")
    monkeypatch.setattr(worker, "require_current_source", unavailable)
    assert control.observe(control=binding(), root=tmp_path, now=1599)["status"] == "monitor_unavailable"
    assert not (tmp_path / "stop_requested.json").exists()
    stopped = control.observe(control=binding(), root=tmp_path, now=1600)
    assert stopped["reason"] == "beta_screening_control_unavailable_timeout"


def test_deadline_can_only_shorten_and_expiry_needs_no_new_contact(tmp_path, monkeypatch):
    observed = {"execution_deadline_at_ms": 1200 * 1000}
    monkeypatch.setattr(worker, "require_current_source", lambda *a, **k: observed)
    value = binding(execution_deadline_epoch=1300)
    assert control.observe(control=value, root=tmp_path, now=1001)["execution_deadline_epoch"] == 1200
    observed["execution_deadline_at_ms"] = 1500 * 1000
    assert control.observe(control=value, root=tmp_path, now=1100)["execution_deadline_epoch"] == 1200
    monkeypatch.setattr(worker, "require_current_source", lambda *a, **k: pytest.fail("expired authority renewed"))
    assert control.observe(control=value, root=tmp_path, now=1200)["reason"] == "beta_screening_authority_expired"


def test_unrelated_site_control_is_untouched_and_wrong_control_is_refused(tmp_path, monkeypatch):
    current = tmp_path / "current"
    other = tmp_path / "other"
    current.mkdir()
    other.mkdir()
    def check(source, *, site_id):
        if site_id == "site-current":
            raise ValueError("beta_screening_current_task_context_changed")
        return {}
    monkeypatch.setattr(worker, "require_current_source", check)
    control.observe(control=binding(), root=current, now=1001)
    assert control.observe(control=binding(site_id="site-other"), root=other, now=1001)["status"] == "current"
    assert not (other / "stop_requested.json").exists()
    with pytest.raises(ValueError, match="binding_changed"):
        control.observe(control=binding(site_id="site-other"), root=current, now=1002)


def test_exact_invocation_stopper_refuses_reused_pid(tmp_path, monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_launch as native
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding={"name": "exact"}, argv=["gpu-canary"])
    value = jobs.manifest(path)
    (path.parent / "started.json").write_text(json.dumps({"pid": 123, "start_ticks": "old",
                                                        "manifest_digest": value["manifest_digest"]}))
    monkeypatch.setattr(native, "_process_start_ticks", lambda pid: "new")
    monkeypatch.setattr(jobs.os, "kill", lambda *a: pytest.fail("signalled reused pid"))
    with pytest.raises(ValueError, match="stop_identity_unproven"):
        jobs.stop_invocation(path)


def test_independent_monitor_stops_itself_after_bounded_outage(tmp_path, monkeypatch):
    def unavailable(*a, **kw):
        raise OSError("control unavailable")
    monkeypatch.setattr(worker, "require_current_source", unavailable)
    monkeypatch.setattr(control.time, "time", lambda: 1601)
    stopped = []
    monitor = control.CurrentSourceMonitor({"control": binding(), "root": str(tmp_path)},
                                            stop=lambda: stopped.append("exact-owned-process"))
    monkeypatch.setattr(monitor.done, "wait", lambda interval: False)
    monitor._run()
    assert stopped == ["exact-owned-process"]
    assert json.loads((tmp_path / "stop_requested.json").read_text())["new_processing_authorized"] is False


def test_delayed_success_cannot_admit_create_after_other_observer_revoked(tmp_path, monkeypatch):
    authority = {"authority_digest": "sha256:" + "d" * 64, "resource_name": "exact-resource"}
    bundle = {"bundle_sha256": "sha256:" + "e" * 64}
    value = binding(initial_observed_at_epoch=time.time())
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding={**authority, **bundle, "provider": "vast"},
                                  argv=["gpu-canary"], control=value)
    pending, resume = threading.Event(), threading.Event()
    calls, outcomes = [], []

    def current(*args, **kwargs):
        calls.append(threading.current_thread().name)
        if len(calls) == 2:
            pending.set()
            assert resume.wait(5)
            return {}
        if len(calls) == 3:
            raise ValueError("beta_screening_current_capture_consent_required")
        return {}

    def allocate():
        try:
            jobs.run_invocation(path, allocator=lambda args: jobs.before_provider_create(
                authority=authority, bundle=bundle))
        except ValueError as exc:
            outcomes.append(str(exc))

    monkeypatch.setattr(worker, "require_current_source", current)
    thread = threading.Thread(target=allocate, name="delayed-allocator")
    thread.start()
    try:
        assert pending.wait(5)
        stop = control.observe(control=value, root=path.parent)
        assert stop["new_processing_authorized"] is False
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive()
    assert outcomes == ["beta_screening_current_source_unproven_before_create"]
    assert not (path.parent / "provider_create_intent.json").exists()
    assert control.observe(control=value, root=path.parent) == stop
    assert json.loads((path.parent / "execution_control.json").read_text())["committed_generation"] == 1


@pytest.mark.parametrize("latest", ["shortened", "unavailable"])
def test_delayed_success_cannot_regress_newer_deadline_or_monitor_state(tmp_path, monkeypatch, latest):
    pending, resume = threading.Event(), threading.Event()
    outcomes = []

    def current(*args, **kwargs):
        if threading.current_thread().name == "old-success":
            pending.set()
            assert resume.wait(5)
            return {"execution_deadline_at_ms": 1300_000}
        if latest == "unavailable":
            raise OSError("service unavailable")
        return {"execution_deadline_at_ms": 1200_000}

    monkeypatch.setattr(worker, "require_current_source", current)
    value = binding(execution_deadline_epoch=1300)
    thread = threading.Thread(name="old-success", target=lambda: outcomes.append(
        control.observe(control=value, root=tmp_path, now=1100)))
    thread.start()
    try:
        assert pending.wait(5)
        newer = control.observe(control=value, root=tmp_path, now=1101)
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive() and len(outcomes) == 1
    final = json.loads((tmp_path / "execution_control.json").read_text())
    assert final == newer == outcomes[0]
    assert final["committed_generation"] == 2
    if latest == "shortened":
        assert final["execution_deadline_epoch"] == 1200
    else:
        assert final["status"] == "monitor_unavailable" and final["last_success_at_epoch"] == 1000


def test_stop_committed_after_read_before_create_admission_is_absorbing(tmp_path, monkeypatch):
    value = binding(initial_observed_at_epoch=time.time())
    real_observe = control.observe
    monkeypatch.setattr(worker, "require_current_source", lambda *a, **k: {})

    def read_then_revoke(**kwargs):
        observed = real_observe(**kwargs)
        def revoked(*a, **k):
            raise ValueError("beta_screening_current_task_context_changed")
        monkeypatch.setattr(worker, "require_current_source", revoked)
        real_observe(**kwargs)
        return observed

    monkeypatch.setattr(control, "observe", read_then_revoke)
    with pytest.raises(ValueError, match="unproven_before_create"):
        with control.current_control_admission(control=value, root=tmp_path):
            pytest.fail("stopped authority entered create-intent scope")
