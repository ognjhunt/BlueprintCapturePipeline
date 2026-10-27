"""Retained inference costs inform a selected-capture audit without creating authority."""

import copy
import json

import pytest

from blueprint_pipeline import selected_atlas_preview_recovery as recovery
from tests import test_selected_atlas_preview_recovery as shared

# Reuse the same original capture and transport fakes for the third projection.
site = shared.site
recover = shared.recover


def sha(character):
    return "sha256:" + character * 64


@pytest.fixture
def inference(site):
    run = sha("7")
    calls = [{"call_digest": sha(key), "provider": provider, "model": model, "state": state,
              "source_digest": sha("1"), "context_digest": sha("2"), "run_digest": run if known else None,
              "recorded_cost_usd": 0.25 if known else None, "bounded_unknown_cost_usd": None,
              "usage_digest": sha("3") if known else None}
             for key, provider, model, state, known in (
                 ("a", "openai", "gpt-6.1-sol", "recorded", True),
                 ("b", "gemini", "gemini-3.8-flash", "admitted", False),
                 ("c", "openai", "gpt-6.1-sol", "unknown", False))]
    return {"schema_version": "website_capture_inference_accounting.v1", "request_id": "fixture",
            "capture_id": "walkthrough-fixture", "source_video_digest": site.config["source_video_digest"],
            "budget_digest": sha("4"), "ledger_status": "complete", "known_usage_cost_usd": 0.25,
            "bounded_unknown_cost_usd": 0, "unbounded_unknown_call_count": 2, "calls": calls,
            "linked_assessment": {"status": "verified", "job_digest": sha("5"), "run_digest": run,
                                  "known_usage_cost_usd": 0.25, "ledger_overlap": "matched_call_ids"},
            "privacy": {"status": "unknown", "historical_cost_status": "unknown"},
            "preparation_ledger_overlap": "not_established"}


def parse(site, value):
    return recovery._capture_inference_accounting(value, request_id="fixture", capture_id="walkthrough-fixture",
        source_video_digest=site.config["source_video_digest"])


def test_optional_projection_preserves_older_getters_and_null_is_unknown(site):
    before = recover(site)["atlas"]
    assert before["capture_inference_accounting"] is None and before["preflight_blockers"] == []
    site.config["capture_inference_accounting"] = None
    assert parse(site, None) is None
    assert recover(site)["atlas"]["atlas_status_digest"] == before["atlas_status_digest"]


def test_known_zero_remains_reported_but_unknown_calls_and_privacy_do_not_become_zero(site, inference):
    inference["calls"][0]["recorded_cost_usd"] = 0
    inference["known_usage_cost_usd"] = 0
    result = parse(site, inference)
    assert result["calls"][0]["recorded_cost_usd"] == 0
    assert result["calls"][0]["usage_digest"] is not None
    assert result["calls"][1]["recorded_cost_usd"] is None
    assert result["unbounded_unknown_call_count"] == 2
    assert result["privacy"]["historical_cost_status"] == "unknown"


@pytest.mark.parametrize("ledger_status", ["missing", "incomplete", "invalid", "unavailable"])
def test_unavailable_history_is_retained_as_unknown_without_an_extra_retry_gate(site, inference, ledger_status):
    inference.update(ledger_status=ledger_status, calls=[], known_usage_cost_usd=0,
                     bounded_unknown_cost_usd=0, unbounded_unknown_call_count=1)
    inference["linked_assessment"].update(status="not_linked", job_digest=None, run_digest=None,
                                         known_usage_cost_usd=None, ledger_overlap="not_available")
    site.config["capture_inference_accounting"] = inference
    result = recover(site)["atlas"]
    assert result["capture_inference_accounting"]["ledger_status"] == ledger_status
    assert result["capture_inference_accounting"]["unbounded_unknown_call_count"] == 1
    assert result["preflight_blockers"] == []
    assert recover(site, retry_digest=result["atlas_status_digest"])["status"] == "atlas_retry_queued"


def test_linked_run_cost_is_a_separate_overlapping_fact_not_added_to_calls_or_atlas_allowance(site, inference):
    site.config["capture_inference_accounting"] = inference
    result = recover(site)["atlas"]
    costs = result["capture_inference_accounting"]
    assert costs["known_usage_cost_usd"] == 0.25
    assert costs["linked_assessment"]["known_usage_cost_usd"] == 0.25
    assert costs["linked_assessment"]["ledger_overlap"] == "matched_call_ids"
    assert costs["preparation_ledger_overlap"] == "not_established"
    assert result["allowance"]["reserved_or_settled_cost_usd"] == 1.5
    assert result["allowance"]["remaining_cost_usd"] == 23.5


def test_explicit_bounded_unknown_is_kept_separate_from_known_and_unbounded_costs(site, inference):
    inference["calls"][1]["bounded_unknown_cost_usd"] = 1.8
    inference["bounded_unknown_cost_usd"] = 1.8
    inference["unbounded_unknown_call_count"] = 1
    result = parse(site, inference)
    assert result["known_usage_cost_usd"] == 0.25 and result["bounded_unknown_cost_usd"] == 1.8
    assert result["calls"][1]["recorded_cost_usd"] is None
    assert result["unbounded_unknown_call_count"] == 1


@pytest.mark.parametrize("field,value", [
    ("request_id", "other"), ("capture_id", "walkthrough-other"),
    ("source_video_digest", sha("9")), ("budget_digest", "private-bucket/key"),
    ("schema_version", "universal_spending_allowance.v1"),
    ("preparation_ledger_overlap", "add_to_atlas_cap"),
])
def test_only_exact_selected_capture_and_source_accounting_is_accepted(site, inference, field, value):
    inference[field] = value
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid"):
        parse(site, inference)


@pytest.mark.parametrize("field", ["call_digest", "source_digest", "context_digest", "run_digest", "usage_digest"])
def test_call_identifiers_cannot_be_private_paths_or_unvalidated_strings(site, inference, field):
    inference["calls"][0][field] = "https://private.invalid/?token=secret"
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid") as failure:
        parse(site, inference)
    assert "private" not in str(failure.value)


@pytest.mark.parametrize("provider,model", [
    ("openai", "gemini-3.8-flash"), ("gemini", "gpt-6.1-sol"),
    ("anthropic", "gpt-6.1-sol"), ("openai", "private-model-override"),
])
def test_projection_cannot_introduce_a_provider_or_model_pair(site, inference, provider, model):
    inference["calls"][0].update(provider=provider, model=model)
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid"):
        parse(site, inference)


@pytest.mark.parametrize("field", ["known_usage_cost_usd", "bounded_unknown_cost_usd", "unbounded_unknown_call_count"])
@pytest.mark.parametrize("invalid", [float("nan"), True, -1])
def test_invalid_costs_and_counts_never_become_accounting_facts(site, inference, field, invalid):
    inference[field] = invalid
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid"):
        parse(site, inference)


@pytest.mark.parametrize("change", ["known_total", "bounded_total", "too_few_unknown", "too_many_complete_unknown",
                                    "fractional_unknown", "recorded_without_usage", "unknown_with_usage",
                                    "known_with_bound", "known_admitted", "duplicate_call"])
def test_projection_must_preserve_cost_state_and_completeness_arithmetic(site, inference, change):
    if change == "known_total":
        inference["known_usage_cost_usd"] += 0.5
    elif change == "bounded_total":
        inference["bounded_unknown_cost_usd"] += 0.5
    elif change == "too_few_unknown":
        inference["unbounded_unknown_call_count"] = 1
    elif change == "too_many_complete_unknown":
        inference["unbounded_unknown_call_count"] = 3
    elif change == "fractional_unknown":
        inference["unbounded_unknown_call_count"] = 2.5
    elif change == "recorded_without_usage":
        inference["calls"][0]["usage_digest"] = None
    elif change == "unknown_with_usage":
        inference["calls"][1]["usage_digest"] = sha("3")
    elif change == "known_with_bound":
        inference["calls"][0]["bounded_unknown_cost_usd"] = 0.5
    elif change == "known_admitted":
        inference["calls"][0]["state"] = "admitted"
    else:
        inference["calls"][1]["call_digest"] = inference["calls"][0]["call_digest"]
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid"):
        parse(site, inference)


@pytest.mark.parametrize("status", ["missing", "invalid", "unavailable"])
def test_no_ledger_evidence_cannot_claim_no_unknown_calls(site, inference, status):
    inference.update(ledger_status=status, calls=[], known_usage_cost_usd=0,
                     bounded_unknown_cost_usd=0, unbounded_unknown_call_count=0)
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid"):
        parse(site, inference)


def test_incomplete_projection_can_report_additional_hidden_unknown_calls(site, inference):
    inference.update(ledger_status="incomplete", unbounded_unknown_call_count=5)
    assert parse(site, inference)["unbounded_unknown_call_count"] == 5


@pytest.mark.parametrize("section", ["top", "call", "linked", "privacy"])
def test_private_or_unrecognized_fields_are_rejected_in_every_projection_section(site, inference, section):
    target = {"top": inference, "call": inference["calls"][0], "linked": inference["linked_assessment"],
              "privacy": inference["privacy"]}[section]
    target["private_payload"] = {"email": "owner@example.invalid", "token": "private-provider-key"}
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid") as failure:
        parse(site, inference)
    assert "owner@example" not in str(failure.value) and "private-provider" not in str(failure.value)


@pytest.mark.parametrize("change", ["unverified_cost", "missing_job_digest", "unmatched_overlap", "privacy_zero"])
def test_linked_or_privacy_claims_cannot_overstate_retained_evidence(site, inference, change):
    if change == "unverified_cost":
        inference["linked_assessment"]["status"] = "mismatch"
    elif change == "missing_job_digest":
        inference["linked_assessment"]["job_digest"] = None
    elif change == "unmatched_overlap":
        inference["linked_assessment"]["run_digest"] = sha("8")
    else:
        inference["privacy"]["historical_cost_status"] = "zero"
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid"):
        parse(site, inference)


def test_projection_is_bounded_to_64_calls_and_stably_orders_retained_ids(site, inference):
    sample = copy.deepcopy(inference["calls"][1])
    inference["calls"] = [{**sample, "call_digest": f"sha256:{index:064x}"} for index in reversed(range(64))]
    inference.update(known_usage_cost_usd=0, bounded_unknown_cost_usd=0, unbounded_unknown_call_count=64)
    inference["linked_assessment"].update(status="not_linked", job_digest=None, run_digest=None,
                                         known_usage_cost_usd=None, ledger_overlap="not_available")
    result = parse(site, inference)
    assert len(result["calls"]) == 64
    assert result["calls"][0]["call_digest"] == "sha256:" + "0" * 64
    inference["calls"].append({**sample, "call_digest": sha("f")})
    inference["unbounded_unknown_call_count"] += 1
    with pytest.raises(ValueError, match="selected_handoff_atlas_inference_accounting_invalid"):
        parse(site, inference)


@pytest.mark.parametrize("change", ["cost", "linked_cost", "budget_id", "historical_source", "historical_context",
                                    "unknown_state", "privacy_fact", "completeness"])
def test_reviewed_snapshot_detects_changed_inference_facts_and_retry_writes_nothing(site, inference, change):
    site.config["capture_inference_accounting"] = inference
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    if change == "cost":
        inference["calls"][0]["recorded_cost_usd"] = 0.5
        inference["known_usage_cost_usd"] = 0.5
    elif change == "linked_cost":
        inference["linked_assessment"]["known_usage_cost_usd"] = 0.5
    elif change == "budget_id":
        inference["budget_digest"] = sha("9")
    elif change == "historical_source":
        inference["calls"][0]["source_digest"] = sha("9")
    elif change == "historical_context":
        inference["calls"][0]["context_digest"] = sha("9")
    elif change == "unknown_state":
        inference["calls"][1]["state"] = "unknown"
    elif change == "privacy_fact":
        inference["privacy"]["status"] = "source_bound_not_reviewed"
    else:
        inference["ledger_status"] = "incomplete"
    current = recover(site)["atlas"]
    assert current["atlas_status_digest"] != reviewed and current["preflight_blockers"] == []
    before = site.job_path.read_bytes()
    with pytest.raises(ValueError, match="selected_handoff_atlas_status_changed"):
        recover(site, retry_digest=reviewed)
    assert site.job_path.read_bytes() == before


def test_row_order_does_not_invalidate_reviewed_snapshot_or_duplicate_paid_cost(site, inference):
    site.config["capture_inference_accounting"] = inference
    reviewed = recover(site)["atlas"]["atlas_status_digest"]
    inference["calls"].reverse()
    assert recover(site)["atlas"]["atlas_status_digest"] == reviewed
    result = recover(site, retry_digest=reviewed)
    assert result["status"] == "atlas_retry_queued"
    assert result["atlas"]["capture_inference_accounting"]["known_usage_cost_usd"] == 0.25
    assert "owner@example" not in json.dumps(result)
