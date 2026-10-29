"""AUT-10 current-source stop intent, bounded outage and exact-pid isolation."""

import json
import threading
import time

import pytest

from blueprint_pipeline import allocator_invocation as jobs
from blueprint_pipeline import beta_policy_screening_control as control
from blueprint_pipeline import beta_policy_screening_worker as worker


def binding(**extra):
    scoped = extra.get("source_control_binding")
    if scoped is not None:
        extra = {**{key: scoped[key] for key in ("source", "site_id", "prepared_digest")}, **extra}
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


def test_scoped_monitor_relay_binds_exact_resource_and_preserves_unrelated_site(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control_transport as transport
    from tests.test_beta_policy_screening_control_contracts import binding as scope, response
    now = [1001]
    monkeypatch.setattr(control.time, "time", lambda: now[0])
    revoked = set()
    def read(*, binding, **kwargs):
        return response(scope=binding, now=now[0], stopped=binding["site_id"] in revoked)
    monkeypatch.setattr(transport, "request_control", read)
    a = scope()
    b = {**scope(), "site_id": "site-two", "authority_digest": "sha256:" + "9" * 64, "resource_name": "other"}
    def monitored(scoped, root):
        return control.CurrentSourceMonitor({"root": root, "control": binding(
            source=scoped["source"], site_id=scoped["site_id"], source_control_binding=scoped,
            execution_deadline_epoch=2000)}, stop=lambda: None)
    delivered_a, delivered_b = [], []
    with monitored(a, tmp_path / "a") as first, monitored(b, tmp_path / "b"):
        remove = control.subscribe_runtime_control(authority_digest=a["authority_digest"],
            resource_name=a["resource_name"], sink=lambda binding, value: delivered_a.append((binding, value)))
        remove_b = control.subscribe_runtime_control(authority_digest=b["authority_digest"],
            resource_name=b["resource_name"], sink=lambda binding, value: delivered_b.append((binding, value)))
        with pytest.raises(ValueError, match="monitor_unavailable"):
            control.require_runtime_control_monitor(authority_digest=a["authority_digest"], resource_name="unrelated")
        now[0] = 1002
        revoked.add(a["site_id"])
        stopped = control.observe(control=first.control, root=first.root)
        first._publish(stopped)
        assert delivered_a[-1][1]["status"] == "stopped"
        assert delivered_a[-1][1]["applicability"] == "historical_only"
        assert delivered_b[-1][1]["status"] == "current" and len(delivered_b) == 1
        with pytest.raises(ValueError, match="monitor_unavailable"):
            control.require_runtime_control_monitor(authority_digest=a["authority_digest"], resource_name=a["resource_name"])
        control.require_runtime_control_monitor(authority_digest=b["authority_digest"], resource_name=b["resource_name"])
        remove()
        remove_b()
    with pytest.raises(ValueError, match="monitor_unavailable"):
        control.require_runtime_control_monitor(authority_digest=b["authority_digest"], resource_name=b["resource_name"])


def test_unavailable_scoped_control_never_refreshes_gpu_snapshot(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control_transport as transport
    from tests.test_beta_policy_screening_control_contracts import binding as scope, response
    now = [1001]
    monkeypatch.setattr(control.time, "time", lambda: now[0])
    monkeypatch.setattr(transport, "request_control", lambda **kwargs: response(now=now[0]))
    value = binding(source_control_binding=scope(), execution_deadline_epoch=2000)
    delivered = []
    monitor = control.CurrentSourceMonitor({"root": tmp_path, "control": value}, stop=lambda: None)
    with monitor:
        unsubscribe = control.subscribe_runtime_control(authority_digest=scope()["authority_digest"],
            resource_name=scope()["resource_name"], sink=lambda b, c: delivered.append(c))
        def unavailable(**kwargs):
            raise OSError("control unavailable")
        monkeypatch.setattr(transport, "request_control", unavailable)
        now[0] = 1599
        state = control.observe(control=value, root=tmp_path)
        monitor._publish(state)
        assert state["status"] == "monitor_unavailable" and len(delivered) == 1
        now[0] = 1601
        stopped = control.observe(control=value, root=tmp_path)
        monitor._publish(stopped)
        assert stopped["new_processing_authorized"] is False and len(delivered) == 1
        unsubscribe()


def test_wrapper_rebinds_provider_cap_to_shorter_actual_expiry_and_exact_invocation(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control_transport as transport
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_beta_policy_screening_control_contracts import binding as scope, response
    monkeypatch.setattr(control.time, "time", lambda: 1001)
    scoped = scope()
    scoped["attempt_id"] = tmp_path.name
    invocation_binding = {key: scoped[key] for key in ("prepared_digest", "plan_digest", "candidate_id",
        "configuration_digest", "authority_digest", "resource_name", "attempt_id")}
    invocation_binding.update(bundle_sha256="sha256:" + "f" * 64, provider="vast")
    scoped["invocation_id"] = canonical_digest(invocation_binding)
    value = binding(source=scoped["source"], site_id=scoped["site_id"], source_control_binding=scoped,
                    execution_deadline_epoch=2500)
    path = jobs.retain_invocation(attempt_dir=tmp_path, binding=invocation_binding,
        argv=["gpu-canary", "--execute", "--team-candidate-execution-deadline-epoch", "2500"], control=value)
    monkeypatch.setattr(transport, "request_control", lambda **kwargs: response(scope=scoped, now=1001, deadline=2000))
    captured = []
    assert jobs.run_invocation(path, allocator=lambda argv: captured.append(list(argv)) or 0) == 0
    assert captured[0][-1] == "2000.0"
    assert jobs.manifest(path)["argv"][-1] == "2500"
    actual = json.loads((path.parent / "effective_execution_deadline.json").read_text())
    assert actual["execution_deadline_epoch"] == 2000 and actual["binding"]["invocation_id"] == scoped["invocation_id"]


def test_runtime_control_delivery_failure_stops_exact_monitor_without_new_authority(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control_transport as transport
    from tests.test_beta_policy_screening_control_contracts import binding as scope, response
    monkeypatch.setattr(control.time, "time", lambda: 1001)
    monkeypatch.setattr(transport, "request_control", lambda **kwargs: response(now=1001))
    stopped = []
    monitor = control.CurrentSourceMonitor({"root": tmp_path, "control": binding(
        source_control_binding=scope(), execution_deadline_epoch=2000)}, stop=lambda: stopped.append("own-only"))
    calls = []
    def sink(binding, value):
        calls.append(value)
        if len(calls) > 1:
            raise OSError("existing SSH control push unavailable")
    with monitor:
        unsubscribe = control.subscribe_runtime_control(authority_digest=scope()["authority_digest"],
            resource_name=scope()["resource_name"], sink=sink)
        with monkeypatch.context() as patch:
            patch.setattr(monitor.done, "wait", lambda interval: False)
            monitor._run()
        assert stopped == ["own-only"] and monitor.done.is_set()
        with pytest.raises(ValueError, match="monitor_unavailable"):
            control.require_runtime_control_monitor(authority_digest=scope()["authority_digest"],
                                                   resource_name=scope()["resource_name"])
        unsubscribe()


def test_interrupted_initial_runtime_push_removes_sink_before_provider_teardown(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_control_transport as transport
    from tests.test_beta_policy_screening_control_contracts import binding as scope, response
    monkeypatch.setattr(control.time, "time", lambda: 1001)
    monkeypatch.setattr(transport, "request_control", lambda **kwargs: response(now=1001))
    monitor = control.CurrentSourceMonitor({"root": tmp_path, "control": binding(
        source_control_binding=scope(), execution_deadline_epoch=2000)}, stop=lambda: None)
    calls = []
    def interrupted(binding, value):
        calls.append("initial push")
        raise KeyboardInterrupt("local SIGTERM while initial existing-SSH push runs")
    with monitor:
        with pytest.raises(KeyboardInterrupt):
            control.subscribe_runtime_control(authority_digest=scope()["authority_digest"],
                resource_name=scope()["resource_name"], sink=interrupted)
        assert not monitor._sinks
        monitor._publish(control.observe(control=monitor.control, root=monitor.root))
        assert calls == ["initial push"]
