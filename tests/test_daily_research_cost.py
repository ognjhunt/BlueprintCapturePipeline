"""Recorded-usage regressions; no provider calls and no claim of final billing."""
from decimal import Decimal

import pytest

from tools.daily_research import discovery

ACTUAL = {"input_tokens": 3363036, "output_tokens": 15233, "total_tokens": 3378269,
          "input_tokens_details": {"cached_tokens": 3224626},
          "output_tokens_details": {"reasoning_tokens": 3906}}


def test_preserved_root_usage_is_sub_two_dollars_not_thirty_three():
    value = discovery.estimated_model_cost(ACTUAL)
    assert value["recorded_tokens_usd"] == {"minimum": "0.7516126", "maximum": "1.5654702"}
    assert value["estimate_usd"] == "1.5654702" and value["usage"] == ACTUAL
    assert value["billed_usd"] is None and value["hard_total_cap"] is False
    assert "per_request_context_regime" in value["unknown_components"]
    # Aggregate input >272K never establishes the pricing tier of each request.
    assert discovery.estimated_model_cost(ACTUAL, context_regime="short")["recorded_tokens_usd"] == {
        "minimum": "0.7516126", "maximum": "0.8208176"}
    assert discovery.estimated_model_cost(ACTUAL, context_regime="long")["recorded_tokens_usd"] == {
        "minimum": "1.4270602", "maximum": "1.5654702"}


def test_cache_writes_replace_uncached_rate_and_reasoning_is_not_added():
    usage = {**ACTUAL, "input_tokens_details": {"cached_tokens": 3224626, "cache_write_tokens": 138410}}
    result = discovery.estimated_model_cost(usage, context_regime="short", regional_processing=False)
    assert result["recorded_tokens_usd"] == {"minimum": "0.8208176", "maximum": "0.8208176"}
    assert result["unknown_components"] == []
    result = discovery.estimated_model_cost(usage, context_regime="short", regional_processing=True)
    assert Decimal(result["estimate_usd"]) == Decimal("0.8208176") * Decimal("1.1")


@pytest.mark.parametrize("changes", [
    {"model": "unpriced-model"}, {"service_tier": "priority"}, {"context_regime": "aggregate_long"},
])
def test_unverified_model_or_pricing_context_cannot_claim_a_known_charge(changes):
    result = discovery.estimated_model_cost(ACTUAL, **changes)
    assert result["known"] is False and result["estimate_usd"] is None


@pytest.mark.parametrize("details", [{"cached_tokens": True}, {"cached_tokens": 3363037},
                                     {"cached_tokens": 3224626, "cache_write_tokens": 138411}])
def test_invalid_token_categories_are_unknown(details):
    assert discovery.estimated_model_cost({**ACTUAL, "input_tokens_details": details})["known"] is False


def test_single_turn_details_retained_and_session_aggregate_is_never_added():
    turns = [{"id": "turn-root", "status": "completed", "usage": ACTUAL}]
    result = discovery.model_cost_observation(turns)
    assert result["estimate_usd"] == "1.5654702" and result["reported_turn_count"] == 1
    assert result["observed_turns"][0]["usage"] == ACTUAL
    assert discovery.model_cost_observation(turns * 2)["usage_state"] == "invalid_turn_inventory"
    turns.append({"id": "turn-qa", "status": "in_progress", "usage": None})
    result = discovery.model_cost_observation(turns)
    assert result["estimate_usd"] is None and result["reported_estimate_usd"] == "1.5654702"
    assert result["known"] is False and result["pending_turns"] == [{"turn_id": "turn-qa", "status": "in_progress"}]


def test_recalculation_preserves_original_reported_proof_once():
    original = {"known": True, "estimate_usd": "33.5454009", "hard_total_cap": False}
    row = {"canary_model_estimate": original}
    corrected = discovery.model_cost_observation([{"id": "turn-root", "usage": ACTUAL}])
    discovery.preserve_estimate(row, "canary_model_estimate", corrected)
    discovery.preserve_estimate(row, "canary_model_estimate", corrected)
    assert row["canary_model_estimate_history"] == [original]
    assert row["canary_model_estimate"]["billed_usd"] is None
