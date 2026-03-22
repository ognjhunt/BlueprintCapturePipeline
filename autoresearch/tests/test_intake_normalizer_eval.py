from __future__ import annotations

from autoresearch.test_support import require_target


def test_intake_outputs_parse_for_all_cases() -> None:
    context = require_target("intake_normalizer")
    for case in context.manifest["eval_cases"]:
        payload = context.json_output(case["case_id"])
        assert payload["schema_version"] == "v1"
        assert payload["scene_id"]
        assert payload["capture_id"]
        assert isinstance(payload["missing_required_fields"], list)


def test_intake_missing_fields_case_stays_fail_closed() -> None:
    context = require_target("intake_normalizer")
    expectations = context.expectations("missing_required_fields")
    payload = context.json_output("missing_required_fields")
    assert payload["status"] == expectations["required_status"]
    assert sorted(payload["missing_required_fields"]) == sorted(
        expectations["expected_missing_required_fields"]
    )
