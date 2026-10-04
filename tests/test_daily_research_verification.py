"""Evidence gate and complete-cohort scoring without network or paid jobs."""
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from tests.daily_research_verification_fixture import assessment
from tests.test_daily_research_runner import NOW, output
from tools.daily_research import verification


def candidate():
    return {**output()["candidates"][0], "candidate_key": "synthetic-1"}


def test_primary_site_task_human_facts_promote_only_with_separate_gates():
    c = candidate()
    value = assessment(c, NOW)
    saved = deepcopy(value)
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "verified" and result["eligible_for_qualified_promotion"]
    assert result["assessment"] == saved and value == saved
    assert result["separate_gates"] == verification.SEPARATE_GATES


@pytest.mark.parametrize("dimension", verification.FACTS)
@pytest.mark.parametrize("state", ["inference", "unresolved", "stale", "unreachable"])
def test_missing_or_inferred_fact_is_unresolved_not_rejected(dimension, state):
    c = candidate()
    value = assessment(c, NOW)
    value["claims"][dimension]["status"] = state
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "unresolved" and not result["eligible_for_qualified_promotion"]
    assert dimension in " ".join(result["reasons"])


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(candidate_digest="0" * 64),
    lambda v: v.update(valid_until=NOW.isoformat()),
    lambda v: v["sources"][0].update(retrieval="snippet"),
    lambda v: v["sources"][0].update(retrieval="unreachable"),
    lambda v: v["sources"][0].update(classification="vendor"),
    lambda v: v["sources"][0].update(freshness="historical"),
    lambda v: v["sources"][0].update(checked_at=(NOW + timedelta(seconds=1)).isoformat()),
    lambda v: v["claims"]["physical_site"].update(source_refs=["invented"]),
    lambda v: v["counterevidence"].update(status="unresolved"),
    lambda v: v["counterevidence"].update(searches=[]),
])
def test_claim_labels_without_bound_current_retrieved_support_cannot_pass(mutation):
    c = candidate()
    value = assessment(c, NOW)
    mutation(value)
    assert verification.evaluate(c, value, NOW)["status"] == "unresolved"


def test_contradiction_requires_support_and_retains_all_sources():
    c = candidate()
    value = assessment(c, NOW)
    value["counterevidence"].update(status="contradicted", source_refs=["synthetic-primary"],
                                  reason="The entire proposed manual task is automated in the synthetic source")
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "rejected" and result["assessment"] == value
    value["sources"][0]["retrieval"] = "unreachable"
    assert verification.evaluate(c, value, NOW)["status"] == "unresolved"


def test_full_cohort_dedupe_coverage_counts_and_unknown_actual_cost():
    first = candidate()
    duplicate = {**first, "candidate_key": "duplicate", "site": first["site"].upper() + "!"}
    other_site = {**first, "candidate_key": "second-site", "location": "Different physical site"}
    other_task = {**first, "candidate_key": "second-task", "task": "Different physical workflow"}
    candidates = [first, duplicate, other_site, other_task]
    values = {c["candidate_key"]: assessment(c, NOW) for c in candidates[:2]}
    result = verification.cohort(candidates, values, NOW)
    assert result["unique_site_task_candidates"] == 3 and result["duplicates"] == 1
    assert result["verified_unique_site_task_candidates"] == 1
    assert result["unresolved_count"] == 2 and result["rejected_count"] == 0
    assert result["verification_coverage"] == .5
    assert result["results"][1]["duplicate_of"] == first["candidate_key"]
    assert not result["results"][1]["eligible_for_qualified_promotion"]
    assert result["actual_cost_usd"] is None and result["verified_unique_per_usd"] is None


def test_duplicate_conflicting_assessments_require_resolution_before_promotion():
    first = candidate()
    second = {**first, "candidate_key": "duplicate"}
    result = verification.cohort([first, second], {first["candidate_key"]: assessment(first, NOW)}, NOW)
    assert result["verified_unique_site_task_candidates"] == 0
    assert result["unresolved_count"] == 2
    assert all(not r["eligible_for_qualified_promotion"] for r in result["results"])


def test_partial_selected_checks_cannot_create_accuracy_or_cost_winner():
    c = candidate()
    runs = {"exa": {"candidates": [c], "assessments": {c["candidate_key"]: assessment(c, NOW)}, "actual_cost_usd": 7.766},
            "gemini": {"candidates": [{**c, "candidate_key": str(i)} for i in range(30)]}}
    result = verification.compare(runs, NOW)
    assert result["accuracy_winner"] is None and not result["cost_comparison_available"]
    assert result["providers"]["gemini"]["verification_coverage"] == 0
    assert result["providers"]["gemini"]["unresolved_count"] == 30


def test_actual_cost_scope_required_for_comparison():
    c = candidate()
    run = {"candidates": [c], "assessments": {c["candidate_key"]: assessment(c, NOW)}, "actual_cost_usd": 2}
    assert not verification.compare({"a": run, "b": run}, NOW)["cost_comparison_available"]
    run.update(comparison_scope="same matched scope", cost_scope="discovery_and_verification")
    assert verification.compare({"a": run, "b": run}, NOW)["cost_comparison_available"]
    assert verification.cohort([c], {}, NOW, 0)["actual_cost_usd"] == 0


def test_inert_float_metadata_preserved_and_portable_numeric_hash():
    c = {**candidate(), "inert_metadata": {"score": .5, "round": 1.0, "small": 1e-7}}
    value = assessment(c, NOW)
    value["extra"] = {"score": .5, "count": 1.0}
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "verified" and result["assessment"]["extra"] == value["extra"]
    assert verification.digest(1) == verification.digest(1.0)
    assert verification.digest(.5) != verification.digest("n:3fe0000000000000")


@pytest.mark.parametrize("value", [2**53, float(2**53), 1e20])
def test_nonportable_whole_numbers_need_string_ids(value):
    c = {**candidate(), "metadata": value}
    result = verification.evaluate(c, None, NOW)
    assert result["status"] == "unresolved" and result["candidate_digest"] is None
    assert "large IDs as strings" in result["reasons"][0]


def test_retained_cross_repo_verification_fixture():
    value = json.loads((Path(__file__).parent / "fixtures/daily_research/lead-verification.json").read_text())
    result = verification.evaluate(value["candidate"], value["assessment"], verification.moment(value["now"]))
    assert {k: result[k] for k in value["expected"]} == value["expected"]


def test_semantic_duplicate_equivalence_reversed_order_and_chained_aliases():
    candidates = [{**candidate(), "candidate_key": k, "organization": f"Alias {k}", "location": f"Address alias {k}"}
                  for k in ("a", "b", "canonical")]
    values = {c["candidate_key"]: assessment(c, NOW) for c in candidates}
    checks = {"a": {"duplicate": True, "duplicate_of": "b", "reason": "Same operator/site/task from sources"},
              "b": {"duplicate": True, "duplicate_of": "canonical", "reason": "Same operator/site/task from sources"}}
    result = verification.cohort(candidates, values, NOW, duplicate_checks=checks)
    assert result["verified_unique_site_task_candidates"] == 1
    assert [r["candidate_key"] for r in result["results"] if r["eligible_for_qualified_promotion"]] == ["canonical"]
    assert result["results"][0]["duplicate_of"] == "canonical"


@pytest.mark.parametrize("checks", [
    {"a": {"duplicate": True, "duplicate_of": "b", "reason": "alias"}, "b": {"duplicate": True, "duplicate_of": "a", "reason": "alias"}},
    {"a": {"duplicate": True, "duplicate_of": "absent", "reason": "alias"}},
    {"a": {"duplicate": True, "duplicate_of": "b", "reason": ""}},
    {"a": {"duplicate": True, "duplicate_of": "b", "reason": "alias"}, "b": {"duplicate": True, "duplicate_of": "a", "reason": ""}},
])
def test_unresolved_semantic_duplicate_cannot_inflate_verified_yield(checks):
    candidates = [{**candidate(), "candidate_key": k, "location": f"Address alias {k}"} for k in ("a", "b")]
    result = verification.cohort(candidates, {c["candidate_key"]: assessment(c, NOW) for c in candidates}, NOW, duplicate_checks=checks)
    assert not result["results"][0]["eligible_for_qualified_promotion"]
    assert result["results"][0]["status"] == "unresolved"


def test_same_city_named_facilities_survive_without_semantic_alias_decision():
    first = {**candidate(), "site": "North plant, 1 Test Street", "location": "Chicago, Illinois, US"}
    second = {**first, "candidate_key": "south", "site": "South plant, 2 Test Street"}
    result = verification.cohort([first, second], {c["candidate_key"]: assessment(c, NOW) for c in [first, second]}, NOW)
    assert result["candidate_count"] == result["unique_site_task_candidates"] == 2
    assert result["duplicates"] == 0 and result["verified_unique_site_task_candidates"] == 2
    assert all(r["eligible_for_qualified_promotion"] for r in result["results"])


@pytest.mark.parametrize("aliases", [{"schema_version": verification.VERSION},
    {"version": verification.VERSION, "schema_version": verification.VERSION}])
def test_lossless_version_alias_is_bound_without_rewriting_raw_assessment(aliases):
    c = candidate()
    value = assessment(c, NOW)
    value.pop("version")
    value.update(aliases)
    original = deepcopy(value)
    result = verification.evaluate(c, value, NOW)
    assert result["eligible_for_qualified_promotion"] and result["assessment"] == original
    assert result["assessment_digest"] == verification.digest(original) and value == original


@pytest.mark.parametrize("canonical", [None, "other.version"])
def test_explicit_null_or_conflicting_version_is_not_aliased(canonical):
    c = candidate()
    value = assessment(c, NOW)
    value.update(version=canonical, schema_version=verification.VERSION)
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "unresolved" and not result["eligible_for_qualified_promotion"]
    assert result["validation_errors"][0]["path"] == "/version"


@pytest.mark.parametrize("contradiction", [False, True])
def test_unknown_expiry_retains_claim_feedback_without_promotion_or_rejection(contradiction):
    c = candidate()
    value = assessment(c, NOW)
    value["valid_until"] = None
    value["claims"]["human_workflow"]["status"] = "contradicted" if contradiction else "unresolved"
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "unresolved" and result["assessment_valid"]
    assert not result["eligible_for_qualified_promotion"] and result["validation_errors"] == []
    assert any("human_workflow" in reason for reason in result["reasons"])
    assert value["valid_until"] is None and result["assessment_digest"] == verification.digest(value)


def test_legacy_retained_cohort_digest_is_unchanged():
    c = candidate()
    value = assessment(c, NOW)
    value["valid_until"] = None
    result = verification.cohort([c], {c["candidate_key"]: value}, NOW, result_version=verification.RESULT_VERSION)
    assert "result_version" not in result and "validation_errors" not in result["results"][0]
    assert result["results"][0]["version"] == verification.RESULT_VERSION
    assert result["assessed_count"] == 0
    assert result["results"][0]["reasons"] == ["assessment: repair candidate binding, dates, source IDs or assessment structure"]


def test_qa_feedback_uses_nested_structural_paths_but_accepts_honest_unknown_expiry():
    from tools.daily_research.consumer import qa_validation_feedback
    c = candidate()
    row = {"packet": {"candidates": [c], "lead_verification_result_version": verification.DIAGNOSTIC_RESULT_VERSION},
           "packet_digest": "synthetic-packet", "qa": {"crm_digest": "synthetic-crm"}}
    value = assessment(c, NOW)
    value["valid_until"] = None
    qa = {"schema_version": "blueprint.research-qa.v1", "packet_digest": "synthetic-packet",
          "crm_digest": "synthetic-crm", "source_support_verified": False, "accepted_keys": [],
          "summary": "Synthetic unresolved assessment", "checks": [{"candidate_key": c["candidate_key"],
          "source_support_verified": False, "duplicate": False, "reason": "Missing evidence", "lead_verification": value}]}
    assert qa_validation_feedback(row, qa) == []
    value["candidate_digest"] = "wrong"
    value["sources"].append(deepcopy(value["sources"][0]))
    paths = {issue["path"] for issue in qa_validation_feedback(row, qa)}
    assert paths == {"/checks/0/lead_verification/candidate_digest", "/checks/0/lead_verification/sources/1/id"}


def test_retained_query_record_is_lossless_and_future_search_cannot_supply_a_check():
    c = candidate()
    value = assessment(c, NOW)
    query = {"id": "synthetic-Q1", "query": "Synthetic task automation", "checked_at": NOW.isoformat(), "scope": "One bounded synthetic search"}
    value["counterevidence"]["searches"] = [query]
    original = deepcopy(value)
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "verified" and result["assessment"] == original and value == original
    query["checked_at"] = (NOW + timedelta(seconds=1)).isoformat()
    assert verification.evaluate(c, value, NOW)["status"] == "unresolved"


def test_diagnostic_fixture_is_exactly_recomputed_without_any_live_source():
    cases = json.loads((Path(__file__).parent / "fixtures/daily_research/lead-verification-diagnostics.json").read_text())
    for case in cases:
        result = verification.evaluate(case["candidate"], case["assessment"], verification.moment(case["now"]))
        assert {key: result[key] for key in case["expected"]} == case["expected"], case["name"]


@pytest.mark.parametrize("bad_key", [[], {}, None])
def test_malformed_check_identity_receives_feedback_without_a_lookup_crash(bad_key):
    from tools.daily_research.consumer import qa_validation_feedback
    c = candidate()
    row = {"packet": {"candidates": [c], "lead_verification_result_version": verification.DIAGNOSTIC_RESULT_VERSION},
           "packet_digest": "packet", "qa": {"crm_digest": "crm"}}
    qa = {"schema_version": "blueprint.research-qa.v1", "packet_digest": "packet", "crm_digest": "crm",
          "source_support_verified": False, "accepted_keys": [], "summary": "Unresolved synthetic record",
          "checks": [{"candidate_key": bad_key, "source_support_verified": False, "duplicate": False,
                      "reason": "Retained malformed identity", "lead_verification": assessment(c, NOW)}]}
    assert any(issue["path"] == "/checks/0/candidate_key" for issue in qa_validation_feedback(row, qa))


def test_nonportable_qa_metadata_returns_feedback_and_preserves_original_assessment():
    from tools.daily_research.consumer import qa_validation_feedback
    c = candidate()
    value = assessment(c, NOW)
    value["metadata"] = float("inf")
    row = {"packet": {"candidates": [c], "lead_verification_result_version": verification.DIAGNOSTIC_RESULT_VERSION},
           "packet_digest": "packet", "qa": {"crm_digest": "crm"}}
    qa = {"schema_version": "blueprint.research-qa.v1", "packet_digest": "packet", "crm_digest": "crm",
          "source_support_verified": False, "accepted_keys": [], "summary": "Unresolved synthetic record",
          "checks": [{"candidate_key": c["candidate_key"], "source_support_verified": False, "duplicate": False,
                      "reason": "Retained nonportable metadata", "lead_verification": value}]}
    issues = qa_validation_feedback(row, qa)
    assert issues and all(len(issue["offending_value_digest"]) == 64 for issue in issues)
    assert value["metadata"] == float("inf")


@pytest.mark.parametrize("version", [verification.RESULT_VERSION, verification.DIAGNOSTIC_RESULT_VERSION])
@pytest.mark.parametrize("url", [123, True, ["https://fixture.example"]])
def test_non_string_source_url_is_unresolved_not_a_stuck_qa_crash(version, url):
    c = candidate()
    value = assessment(c, NOW)
    value["sources"][0]["url"] = url
    result = verification.evaluate(c, value, NOW, result_version=version)
    assert result["status"] == "unresolved" and not result["eligible_for_qualified_promotion"]
