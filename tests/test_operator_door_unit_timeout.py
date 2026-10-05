"""A timed-out caller cannot turn an unobserved unit job into a new operation."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/spool_runner.py
#   deploy/operator-door/operator_door/requests.py

import json
from pathlib import Path

import pytest

from tests.test_operator_door_runner import FakeRunner, _result, config as config
from tests.test_operator_door_client import client, door as door
from operator_door.hostinfo import CommandResult, HostInfo
from operator_door.requests import enqueue, observed_unit_outcome, request_state
from operator_door.spool_runner import process_spool


UNIT = "blueprint-pipeline-control-plane.service"
REQUEST = {"kind": "unit", "unit": UNIT, "action": "start"}
OPERATION_KEY = "unit-timeout-retry-1234"
BEFORE = {"ActiveState": "inactive", "SubState": "dead", "Type": "oneshot",
          "InvocationID": "previous", "Result": "success", "ExecMainStatus": "0"}
RUNNING = {"ActiveState": "activating", "SubState": "start", "Type": "oneshot",
           "InvocationID": "submitted", "Result": "success", "ExecMainStatus": "0"}


class TimeoutRunner(FakeRunner):
    def __init__(self, *, before=BEFORE, after=RUNNING, action="start"):
        super().__init__()
        self.before, self.after, self.action = before, after, action
        self.actions = 0

    def run(self, argv, timeout):
        if argv[:2] == ["systemctl", self.action]:
            self.calls.append(list(argv))
            self.actions += 1
            return CommandResult(124, "", "timeout")
        if argv[:2] == ["systemctl", "show"]:
            self.calls.append(list(argv))
            value = self.after if self.actions else self.before
            if value is None:
                return CommandResult(124, "", "timeout")
            return CommandResult(0, "".join(f"{key}={value}\n" for key, value in value.items()), "")
        return super().run(argv, timeout)


def _submit(config, runner, request=REQUEST):
    request_id = enqueue(config, request, requested_by="owner", operation_key=OPERATION_KEY)
    process_spool(config, runner=runner)
    return request_id


def _observe(config, request_id, value):
    state = request_state(config, request_id)
    state["unit_state"] = [] if value is None else [{"Id": UNIT, **value}]
    state["observed_outcome"] = observed_unit_outcome(state)
    return state


@pytest.mark.parametrize("action,unit", [("start", UNIT),
    ("restart", "blueprint-pubsub-handoff-listener.timer")])
def test_timeout_retains_unit_and_operation_identity_without_duplicate_action(config, action, unit):
    request = {"kind": "unit", "unit": unit, "action": action}
    runner = TimeoutRunner(action=action)
    request_id = _submit(config, runner, request)
    result = _result(config, request_id)
    assert result["status"] == "unknown" and result["code"] == "unit_action_timed_out"
    assert result["unit"] == unit and result["unit_action"] == request
    assert result["unit_observation"]["InvocationID"] == "submitted"
    assert result["returncode"] == 124
    assert enqueue(config, request, requested_by="owner", operation_key=OPERATION_KEY) == request_id
    process_spool(config, runner=runner)
    assert runner.actions == 1
    # Even after the ordinary result/spool retention, the operation tombstone
    # cannot publish a new start with the same owner/key.
    for directory in ("results", "completed"):
        (Path(config.spool_root) / directory / f"{request_id}.json").unlink()
    assert enqueue(config, request, requested_by="owner", operation_key=OPERATION_KEY) == request_id
    process_spool(config, runner=runner)
    assert runner.actions == 1


@pytest.mark.parametrize("terminal,result,expected", [("inactive", "success", True),
    ("failed", "exit-code", False)])
def test_timeout_stays_pending_while_running_and_reopens_only_bound_late_terminal(config, terminal, result, expected):
    request_id = _submit(config, TimeoutRunner())
    assert client._terminal(_observe(config, request_id, RUNNING)) is None
    later = {**RUNNING, "ActiveState": terminal, "Result": result,
             "ExecMainStatus": "0" if expected else "1"}
    assert client._terminal(_observe(config, request_id, later)) is expected
    later["InvocationID"] = "replacement"
    changed = _observe(config, request_id, later)
    assert changed["observed_outcome"] == {"status": "unknown", "reason": "unit_invocation_changed"}
    assert client._terminal(changed) is None


@pytest.mark.parametrize("before,after", [(BEFORE, BEFORE), (None, RUNNING), (BEFORE, None)])
def test_missing_readback_or_old_invocation_cannot_finish_timed_out_action(config, before, after):
    request_id = _submit(config, TimeoutRunner(before=before, after=after))
    result = _result(config, request_id)
    assert result["status"] == "unknown" and result["unit"] == UNIT
    assert "InvocationID" not in result["unit_observation"]
    assert client._terminal(_observe(config, request_id, BEFORE)) is None
    assert client._terminal(_observe(config, request_id, {**RUNNING, "ActiveState": "inactive"})) is None


def test_timeout_cannot_resolve_from_missing_or_different_unit_readback(config):
    request_id = _submit(config, TimeoutRunner())
    assert client._terminal(_observe(config, request_id, None)) is None
    state = _observe(config, request_id, {**RUNNING, "ActiveState": "inactive"})
    state["unit_state"][0]["Id"] = "blueprint-unrelated.service"
    state["observed_outcome"] = observed_unit_outcome(state)
    assert state["observed_outcome"] == {"status": "unknown", "reason": "unit_identity_changed"}
    assert client._terminal(state) is None


def test_timeout_readback_preserves_existing_dispatch_hold(config):
    root = Path(config.spool_root) / "holds"
    root.mkdir()
    held = root / "blueprint-agent-run-dispatcher.timer.json"
    held.write_text('{"owner":"founder","require_explicit_release":true}')
    before = held.read_bytes()
    request_id = _submit(config, TimeoutRunner())
    _observe(config, request_id, RUNNING)
    assert held.read_bytes() == before


def test_interrupted_readback_retains_previously_published_unit_identity(config):
    class InterruptedReadback(TimeoutRunner):
        def run(self, argv, timeout):
            if argv[:2] == ["systemctl", "show"] and self.actions:
                raise RuntimeError("fixture readback interrupted")
            if argv[:2] == ["systemctl", "start"]:
                accepted = next((Path(config.spool_root) / "results").glob("*.json"))
                assert json.loads(accepted.read_text())["unit"] == UNIT
            return super().run(argv, timeout)

    runner = InterruptedReadback()
    request_id = _submit(config, runner)
    result = _result(config, request_id)
    assert result["status"] == "unknown" and result["unit"] == UNIT
    assert result["code"] == "runner_error:RuntimeError"
    assert client._terminal(_observe(config, request_id, {**RUNNING, "ActiveState": "inactive"})) is None
    assert enqueue(config, REQUEST, requested_by="owner", operation_key=OPERATION_KEY) == request_id
    process_spool(config, runner=runner)
    assert runner.actions == 1


def test_http_poll_reopens_timed_out_invocation_without_resubmitting(door, monkeypatch):
    runner = TimeoutRunner()
    request_id = _submit(door["config"], runner)
    observed = {"Id": UNIT, **RUNNING}
    monkeypatch.setattr(HostInfo, "unit_properties", lambda self, units: [dict(observed)])
    state = client._json("GET", f"/requests/{request_id}")
    assert state["result"]["status"] == "unknown" and client._terminal(state) is None
    observed["ActiveState"] = "inactive"
    state = client._json("GET", f"/requests/{request_id}")
    assert state["observed_outcome"] == {"status": "observed_completed", "invocation_id": "submitted"}
    assert client._terminal(state) is True
    assert runner.actions == 1
