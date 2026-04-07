from __future__ import annotations

from autoresearch.test_support import require_target


def test_recapture_plan_outputs_parse_for_all_cases() -> None:
    context = require_target("recapture_planner")
    for case in context.manifest["eval_cases"]:
        payload = context.json_output(case["case_id"])
        assert isinstance(payload["required"], bool)
        assert isinstance(payload["steps"], list)


def test_access_constrained_case_surfaces_access_pending() -> None:
    context = require_target("recapture_planner")
    expectations = context.expectations("access_constrained")
    payload = context.json_output("access_constrained")
    assert payload["access_pending"] is expectations["access_pending"]


def test_mixed_access_case_surfaces_access_pending() -> None:
    context = require_target("recapture_planner")
    expectations = context.expectations("mixed_access")
    payload = context.json_output("mixed_access")
    assert payload["access_pending"] is expectations["access_pending"]
