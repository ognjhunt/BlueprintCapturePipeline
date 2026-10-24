"""An exhausted offer search must preserve the preceding create-refusal proof."""

from copy import deepcopy

import pytest

from blueprint_pipeline.vast_create_failure_diagnosis import (
    definite_create_refusal_without_instance,
)


def _exhausted_search() -> dict:
    return {
        "status": "blocked",
        "reason": "vast_offer_selection_blocked",
        "provider_create_attempted": True,
        "vast_instance_ids": [],
        "vast_side_effects_may_have_occurred": False,
        "continuing_spend_from_this_run": False,
        "create_failure_diagnosis": {
            "http_status_code": 400,
            "error_preview": "no_such_ask",
            "definite_create_refusal": True,
            "create_inventory_verified": True,
            "create_inventory_http_status_code": 200,
            "create_produced_no_instance": True,
            "matching_attempt_instance_ids": [],
            "attempted_labels": ["exact-attempt"],
        },
        "provider_attempt_classification": {
            "classification": "pre_execution_provider_null",
            "scientific_attempt_consumed": False,
        },
    }


def test_definite_refusal_survives_exhausted_offer_search() -> None:
    assert definite_create_refusal_without_instance(_exhausted_search())


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "status", "completed"),
        (None, "reason", "vast_provider_runtime_failed"),
        (None, "vast_instance_ids", [123]),
        (None, "vast_side_effects_may_have_occurred", True),
        (None, "continuing_spend_from_this_run", True),
        ("create_failure_diagnosis", "error_preview", ""),
        ("create_failure_diagnosis", "http_status_code", 500),
        ("create_failure_diagnosis", "definite_create_refusal", False),
        ("create_failure_diagnosis", "create_inventory_verified", False),
        ("create_failure_diagnosis", "create_inventory_http_status_code", 503),
        ("create_failure_diagnosis", "matching_attempt_instance_ids", [123]),
        ("create_failure_diagnosis", "attempted_labels", []),
        ("provider_attempt_classification", "scientific_attempt_consumed", True),
    ],
)
def test_exhausted_search_keeps_ambiguous_create_protected(section, key, value) -> None:
    adapter = deepcopy(_exhausted_search())
    target = adapter if section is None else adapter[section]
    target[key] = value
    assert not definite_create_refusal_without_instance(adapter)
