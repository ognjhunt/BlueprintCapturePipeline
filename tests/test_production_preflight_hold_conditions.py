"""A path appearing in condition arguments is not an execution hold."""

import pytest

from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight


@pytest.mark.parametrize("condition", [
    "{ path=/usr/bin/test ; argv[]=/usr/bin/test -x /bin/false ; ignore_errors=no ; code=exited ; status=0 }",
    "{ path=/usr/bin/false ; argv[]=/usr/bin/false ; ignore_errors=no ; code=exited ; status=0 }",
    "{ path=/usr/bin/false ; argv[]=/usr/bin/false ; ignore_errors=no }",
    "/usr/bin/false",
])
def test_successful_or_unproven_condition_never_hides_failed_required_unit(monkeypatch, condition: str) -> None:
    monkeypatch.setattr(preflight, "unit_properties", lambda _unit: {"ActiveState": ["inactive"]})
    props = {"LoadState": ["loaded"], "ActiveState": ["failed"], "Result": ["exit-code"],
             "ExecMainStatus": ["2"], "TriggeredBy": ["blueprint-beta-policy-screening.timer"],
             "ExecCondition": [condition]}
    rows = preflight.unit_health_checks({"blueprint-beta-policy-screening.service": {"properties": props}}, lane="beta_rigid")
    assert {row["code"] for row in rows if row["severity"] == "blocker"} == {"unit_failed_state", "unit_trigger_inactive"}
    assert not any(row["code"] == "unit_intentionally_held" for row in rows)


def test_actual_failed_false_condition_is_a_hold_but_other_failed_predicate_is_not(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "unit_properties", lambda _unit: {"ActiveState": ["inactive"]})
    base = {"LoadState": ["loaded"], "ActiveState": ["inactive"], "Result": ["exec-condition"],
            "TriggeredBy": ["blueprint-beta-policy-screening.timer"]}
    actual = {**base, "ExecCondition": ["{ path=/bin/false ; argv[]=/bin/false ; ignore_errors=no ; code=exited ; status=1 }"]}
    held = preflight.unit_health_checks({"blueprint-beta-policy-screening.service": {"properties": actual}}, lane="beta_rigid")
    assert held and all(row["code"] == "unit_intentionally_held" and row["severity"] == "info" for row in held)
    other = {**base, "ExecCondition": ["{ path=/usr/bin/test ; argv[]=/usr/bin/test -x /bin/false ; ignore_errors=no ; code=exited ; status=1 }"]}
    rows = preflight.unit_health_checks({"blueprint-beta-policy-screening.service": {"properties": other}}, lane="beta_rigid")
    assert any(row["code"] == "unit_trigger_inactive" and row["severity"] == "blocker" for row in rows)
    assert not any(row["code"] == "unit_intentionally_held" for row in rows)


def test_a_paused_lanes_disarmed_trigger_is_not_a_blocker(monkeypatch) -> None:
    """The release disarms paused lanes' triggers (2026-11-12 frame); the preflight must not count them."""
    from blueprint_pipeline.control_plane_stall_alarm import read_deploy_paused_triggers

    paused = read_deploy_paused_triggers()
    assert "blueprint-task-evaluation-sam31-preparation-execution.timer" in paused
    states = {"blueprint-task-evaluation-sam31-preparation-execution.timer": "inactive",
              "blueprint-task-evaluation-sam31-preparation-execution.path": "inactive"}
    monkeypatch.setattr(preflight, "unit_properties", lambda unit: {"ActiveState": [states[unit]]})
    props = {"LoadState": ["loaded"], "ActiveState": ["inactive"], "Result": ["success"],
             "TriggeredBy": [" ".join(states)]}
    rows = preflight.unit_health_checks(
        {"blueprint-task-evaluation-sam31-preparation-execution.service": {"properties": props}})
    by_trigger = {row["trigger"]: row for row in rows}
    paused_row = by_trigger["blueprint-task-evaluation-sam31-preparation-execution.timer"]
    assert (paused_row["severity"], paused_row["code"]) == ("info", "unit_trigger_not_required")
    # Its path unit is still armed by every release, so an inactive one stays a blocker.
    armed_row = by_trigger["blueprint-task-evaluation-sam31-preparation-execution.path"]
    assert (armed_row["severity"], armed_row["code"]) == ("blocker", "unit_trigger_inactive")
