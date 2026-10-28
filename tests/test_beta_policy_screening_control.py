"""AUT-10 current-source stop intent, bounded outage and exact-pid isolation."""

import json

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
