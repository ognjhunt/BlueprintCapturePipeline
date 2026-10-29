"""Retained GPU-local authority, with no SSH, network, model or GPU calls."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline import beta_screening_runtime_control as runtime_control
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_beta_policy_screening_control_contracts import binding, response


def staged_control(tmp_path, monkeypatch, *, suffix=""):
    monkeypatch.setattr(runtime_control.time, "time", lambda: 1000.0)
    scope = binding()
    if suffix:
        for key in ("campaign_id", "site_id", "attempt_id", "resource_name"):
            scope[key] += suffix
    inputs = {"run_kind": runtime_control.BETA_RUN_KIND, "run_id": "beta-exact-one" + suffix,
              "candidate_ids": [scope["candidate_id"]], "plan_digest": scope["plan_digest"],
              "configuration_digest": scope["configuration_digest"],
              "scene_revision_digest": scope["source"]["scene_revision_digest"]}
    inputs["runtime_inputs_digest"] = canonical_digest(inputs)
    authority = {**inputs, "resource_name": scope["resource_name"], "authority_digest": ""}
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    scope["authority_digest"] = authority["authority_digest"]
    runtime = tmp_path / "bundle" / "provider_runtime"
    (runtime / "runtime_inputs").mkdir(parents=True)
    (runtime / "runtime_inputs" / "policy_canary_session_authority.json").write_text(json.dumps(authority))
    (runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json").write_text(json.dumps(inputs))
    return runtime, inputs, authority, scope


def install(runtime, authority, scope, *, now=1000, deadline=3000, revision=1, stopped=False):
    control = response(scope=scope, now=now, deadline=deadline, revision=revision, stopped=stopped)
    delivery = runtime_control.make_delivery(authority=authority, binding=scope, control=control)
    return runtime_control.install_runtime_control(runtime_root=runtime, delivery=delivery)


def test_actual_fixed_receiver_and_restarted_guard_retain_exact_scope(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    delivered = install(runtime, authority, scope)
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    assert guard.check("before_cell_spawn") == delivered["control"]
    restarted = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    assert restarted.check("before_policy_query") == delivered["control"]
    state = json.loads((runtime.parent / runtime_control.STATE_FILENAME).read_text())
    assert state["binding_digest"] == cross_runtime_canonical_digest(scope)
    assert state["execution_deadline_at_ms"] == 3_000_000
    assert "token" not in json.dumps(delivered).lower()


@pytest.mark.parametrize("field", ["attempt_id", "invocation_id", "prepared_digest", "authority_digest", "resource_name",
                                  "campaign_id", "site_id", "candidate_id", "configuration_digest", "plan_digest"])
def test_rebound_attempt_or_session_never_gets_a_second_processing_admission(tmp_path, monkeypatch, field):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    install(runtime, authority, scope)
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    guard.check("before_policy_load")
    changed = deepcopy(scope)
    changed[field] = "sha256:" + "9" * 64 if field.endswith("digest") or field == "invocation_id" else "different"
    with pytest.raises(ValueError):
        install(runtime, authority, changed, now=1001, revision=2)
    assert guard.check("before_policy_query")["binding"] == scope


@pytest.mark.parametrize("field", ["scene_intent_id", "scene_intent_digest", "capture_id", "task_id", "task_context_digest", "scene_revision_digest"])
def test_each_retained_source_selector_is_immutable(tmp_path, monkeypatch, field):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    install(runtime, authority, scope)
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    guard.check("before_cell_spawn")
    changed = deepcopy(scope)
    changed["source"][field] = "sha256:" + "9" * 64 if field.endswith("digest") else "different"
    with pytest.raises(ValueError):
        install(runtime, authority, changed, now=1001, revision=2)


def test_stop_and_expiry_are_absorbing_while_unrelated_resource_remains_current(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    other, other_inputs, other_authority, other_scope = staged_control(tmp_path / "other", monkeypatch, suffix="-other")
    install(runtime, authority, scope)
    install(other, other_authority, other_scope)
    stopped = install(runtime, authority, scope, now=1001, revision=2, stopped=True)
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    with pytest.raises(runtime_control.RuntimeControlStopped) as raised:
        guard.check("before_policy_query")
    receipt = raised.value.receipt
    assert receipt["source_control"] == stopped["control"]
    assert receipt["applicability"] == "historical_only" and receipt["read_only_closeout_allowed"] is True
    assert runtime_control.RuntimeControlGuard(runtime_root=other, inputs=other_inputs, authority=other_authority).check("before_cell_spawn")["status"] == "current"
    assert install(runtime, authority, scope, now=1002, revision=3) == stopped
    (runtime.parent / runtime_control.CONTROL_FILENAME).unlink()
    with pytest.raises(runtime_control.RuntimeControlStopped) as restarted:
        runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority).check("before_cell_spawn")
    assert restarted.value.receipt == receipt


def test_controller_loss_stops_at_existing_grace_without_resetting_observation_time(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    install(runtime, authority, scope)
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    monkeypatch.setattr(runtime_control.time, "time", lambda: 1599.0)
    guard.check("before_policy_query")
    monkeypatch.setattr(runtime_control.time, "time", lambda: 1600.0)
    with pytest.raises(runtime_control.RuntimeControlStopped, match="controller_stale"):
        guard.check("before_policy_query")
    install(runtime, authority, scope, now=1601, revision=2)
    with pytest.raises(runtime_control.RuntimeControlStopped, match="controller_stale"):
        guard.check("before_cell_spawn")


def test_expiry_shortening_and_stale_delivery_cannot_extend_processing(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    first = install(runtime, authority, scope)
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    guard.check("before_policy_load")
    shortened = install(runtime, authority, scope, now=1001, deadline=1100, revision=2)
    assert shortened["maximum_execution_deadline_at_ms"] == first["maximum_execution_deadline_at_ms"]
    assert install(runtime, authority, scope, now=1000, revision=1) == shortened
    with pytest.raises(ValueError, match="deadline_extended"):
        install(runtime, authority, scope, now=1002, deadline=2000, revision=3)
    monkeypatch.setattr(runtime_control.time, "time", lambda: 1100.0)
    with pytest.raises(runtime_control.RuntimeControlStopped, match="authority_expired"):
        guard.check("before_policy_query")


@pytest.mark.parametrize("failure", ["missing", "bad_digest", "symlink", "group_writable", "future"])
def test_missing_untrusted_or_future_control_blocks_without_processing(tmp_path, monkeypatch, failure):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    path = runtime.parent / runtime_control.CONTROL_FILENAME
    if failure != "missing":
        install(runtime, authority, scope, now=1061 if failure == "future" else 1000)
    if failure == "bad_digest":
        value = json.loads(path.read_text())
        value["control"]["binding"]["attempt_id"] = "forged"
        path.write_text(json.dumps(value))
    elif failure == "symlink":
        target = path.with_name("untrusted.json")
        path.rename(target)
        path.symlink_to(target)
    elif failure == "group_writable":
        path.chmod(0o660)
    with pytest.raises(runtime_control.RuntimeControlStopped):
        runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority).check("before_policy_load")


def test_initial_controller_wait_does_no_processing_and_is_bounded(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    install(runtime, authority, scope)
    assert runtime_control.await_initial_control(runtime_root=runtime)["binding"] == scope
    (runtime.parent / runtime_control.CONTROL_FILENAME).unlink()
    with pytest.raises(runtime_control.RuntimeControlStopped):
        runtime_control.await_initial_control(runtime_root=runtime, timeout_seconds=0)


def test_runtime_input_session_mismatch_is_never_admitted(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    install(runtime, authority, scope)
    changed = {**inputs, "run_id": "other-session"}
    with pytest.raises(ValueError, match="session_mismatch"):
        runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=changed, authority=authority)


def test_existing_provider_watchdog_deadline_can_only_shorten_the_retained_maximum(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    install(runtime, authority, scope)
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "1200.0")
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    guard.check("before_policy_load")
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "3000.0")
    restarted = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    monkeypatch.setattr(runtime_control.time, "time", lambda: 1200.0)
    with pytest.raises(runtime_control.RuntimeControlStopped, match="authority_expired"):
        restarted.check("before_policy_query")


def test_provider_deadline_cannot_extend_a_smaller_sealed_session_deadline(tmp_path, monkeypatch):
    runtime, inputs, authority, scope = staged_control(tmp_path, monkeypatch)
    authority["hard_deadline_epoch"] = 1200.0
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    scope["authority_digest"] = authority["authority_digest"]
    (runtime / "runtime_inputs/policy_canary_session_authority.json").write_text(json.dumps(authority))
    install(runtime, authority, scope)
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "3000")
    guard = runtime_control.RuntimeControlGuard(runtime_root=runtime, inputs=inputs, authority=authority)
    guard.check("before_policy_load")
    monkeypatch.setattr(runtime_control.time, "time", lambda: 1200.0)
    with pytest.raises(runtime_control.RuntimeControlStopped, match="authority_expired"):
        guard.check("before_policy_query")
