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
