"""Evidence gate and complete-cohort scoring without network or paid jobs."""
from copy import deepcopy
from datetime import timedelta
import json
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
    duplicate = {**first, "candidate_key": "duplicate", "site": "Alternate site label"}
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
