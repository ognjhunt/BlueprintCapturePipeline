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


# --- Located diagnostics (2026-10-04 replay class) ---------------------------------
# The October 4 QA assessments used `schema_version` for the version marker and an
# honest `valid_until: null` (the evidence skill says to leave freshness unresolved).
# The gate then returned one generic reason and evaluated no claim. These synthetic
# shapes pin the repair: same decisions, but every failed check is named and claims
# are still reported, without inventing dates or accepting an unbound assessment.
CATCH_ALL = "assessment: repair candidate binding, dates, source IDs or assessment structure"


def _legacy_evaluate(candidate, assessment, now):
    """Frozen copy of the pre-2026-10-04 decision logic; only reasons may differ now."""
    reasons, rejected = [], False
    try:
        candidate_digest, assessment_digest = verification.digest(candidate), verification.digest(assessment)
    except (ValueError, TypeError, OverflowError):
        candidate_digest, assessment_digest = None, None
        reasons.append("digest")
    result = {"assessment_valid": False}
    if not verification.identity_key(candidate):
        reasons.append("identity")
    if not isinstance(assessment, dict):
        reasons.append("assessment")
    else:
        try:
            assessed_at = verification.moment(assessment.get("assessed_at"))
            if (assessment.get("version") != verification.VERSION
                    or candidate_digest is None or assessment_digest is None
                    or assessment.get("candidate_digest") != candidate_digest):
                raise ValueError("binding")
            if not assessed_at <= now < verification.moment(assessment.get("valid_until")):
                raise ValueError("freshness")
            claims, sources = assessment.get("claims"), assessment.get("sources")
            counter = assessment.get("counterevidence")
            if (not isinstance(claims, dict) or not isinstance(sources, list)
                    or not isinstance(counter, dict)):
                raise TypeError("structure")
            indexed = {s["id"]: s for s in sources if isinstance(s, dict) and verification.text(s.get("id"))}
            if len(indexed) != len(sources):
                raise ValueError("source identity")
            result["assessment_valid"] = True
            for name in verification.CLAIMS:
                claim = claims.get(name)
                if not isinstance(claim, dict) or claim.get("status") not in verification.STATES or not verification.text(claim.get("reason")):
                    reasons.append(name)
                    result["assessment_valid"] = False
                    continue
                refs = claim.get("source_refs")
                linked = [indexed[r] for r in refs if isinstance(r, str) and r in indexed] if isinstance(refs, list) else []
                usable = (bool(linked) and len(linked) == len(refs)
                          and all(verification.source_usable(s, assessed_at, primary=name in verification.FACTS) for s in linked))
                state = claim["status"]
                if state == "contradicted" and usable:
                    rejected = True
                    reasons.append(name)
                elif state not in ({"verified_fact"} if name in verification.FACTS else {"verified_fact", "inference"}) or not usable:
                    reasons.append(name)
            refs = counter.get("source_refs")
            linked = [indexed[r] for r in refs if isinstance(r, str) and r in indexed] if isinstance(refs, list) else []
            usable = (bool(linked) and len(linked) == len(refs)
                      and all(verification.source_usable(s, assessed_at) and s.get("classification") != "vendor" for s in linked))
            searches = counter.get("searches")
            searched = isinstance(searches, list) and bool(searches) and all(verification.text(s) for s in searches)
            if counter.get("status") == "contradicted" and usable and verification.text(counter.get("reason")):
                rejected = True
                reasons.append("counterevidence")
            elif (counter.get("status") != "checked" or not verification.text(counter.get("reason"))
                  or not (searched or usable) or (refs and not usable)):
                reasons.append("counterevidence")
            if counter.get("status") not in {"checked", "unresolved", "contradicted"}:
                result["assessment_valid"] = False
        except (KeyError, TypeError, ValueError):
            reasons.append("catch-all")
    status = "rejected" if rejected else "unresolved" if reasons else "verified"
    return {"status": status, "eligible_for_qualified_promotion": status == "verified",
            "assessment_valid": result["assessment_valid"]}


def oct4_shape(c):
    """Synthetic assessment with the retained October 4 field shape (no real lead data)."""
    value = assessment(c, NOW)
    value["schema_version"] = value.pop("version")
    value["valid_until"] = None
    value["claims"]["human_workflow"].update(status="unresolved", reason="Company-wide staffing wording; exact-site step unconfirmed")
    value["counterevidence"].update(status="unresolved", reason="Bounded automation search left the station method open")
    value.update(assessment_status="unresolved", freshness_boundary_reason="No defensible expiry; missing facts stay unresolved",
                 contact_assessment={"status": "unresolved"}, duplicate_assessment={"duplicate": False},
                 qualification_gates={"buying_intent": "unknown"})
    return value


def test_oct4_shape_names_each_failed_check_and_still_reports_claims():
    c = candidate()
    value = oct4_shape(c)
    saved = deepcopy(value)
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "unresolved" and not result["eligible_for_qualified_promotion"]
    assert result["assessment_valid"] is False
    assert CATCH_ALL not in result["reasons"]
    assert any(r.startswith("version:") and "schema_version" in r for r in result["reasons"])
    assert any(r.startswith("valid_until:") for r in result["reasons"])
    assert any(r.startswith("human_workflow:") for r in result["reasons"])
    assert any(r.startswith("counterevidence:") for r in result["reasons"])
    assert result["assessment"] == saved and value == saved


def test_null_valid_until_blocks_promotion_without_inventing_a_date():
    c = candidate()
    value = assessment(c, NOW)
    value["valid_until"] = None
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "unresolved" and not result["eligible_for_qualified_promotion"]
    assert [r.split(":")[0] for r in result["reasons"]] == ["valid_until"]
    assert result["assessment"]["valid_until"] is None


@pytest.mark.parametrize(("mutation", "prefix"), [
    (lambda v: v.update(candidate_digest="0" * 64), "candidate_digest:"),
    (lambda v: v.update(version="blueprint.lead-verification.v0"), "version:"),
    (lambda v: v.pop("assessed_at"), "assessed_at:"),
    (lambda v: v.update(assessed_at=NOW.replace(tzinfo=None).isoformat()), "assessed_at:"),
    (lambda v: v.update(assessed_at=(NOW + timedelta(seconds=1)).isoformat()), "assessed_at:"),
    (lambda v: v.update(valid_until=NOW.isoformat()), "valid_until:"),
    (lambda v: v.update(valid_until="2026-10-05"), "valid_until:"),
    (lambda v: v.update(claims=[]), "claims:"),
    (lambda v: v.update(sources={}), "sources:"),
    (lambda v: v.update(counterevidence="checked"), "counterevidence:"),
    (lambda v: v["sources"].append(deepcopy(v["sources"][0])), "sources:"),
])
def test_each_unbound_or_malformed_check_has_its_own_located_reason(mutation, prefix):
    c = candidate()
    value = assessment(c, NOW)
    mutation(value)
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "unresolved" and not result["eligible_for_qualified_promotion"]
    assert CATCH_ALL not in result["reasons"]
    assert any(r.startswith(prefix) for r in result["reasons"]), result["reasons"]


def test_unbound_assessment_reports_but_never_applies_a_contradiction():
    c = candidate()
    value = assessment(c, NOW)
    value["candidate_digest"] = "0" * 64
    value["claims"]["site_task"].update(status="contradicted", reason="Operator page says this task moved off site")
    result = verification.evaluate(c, value, NOW)
    assert result["status"] == "unresolved"
    assert any(r.startswith("site_task:") and "diagnostic" in r for r in result["reasons"])


def _equivalence_corpus():
    c = candidate()
    mutations = [
        lambda v: None,
        lambda v: v.update(valid_until=None),
        lambda v: v.update(schema_version=v.pop("version")),
        lambda v: v.update(candidate_digest="0" * 64),
        lambda v: v.update(version="x"),
        lambda v: v.pop("assessed_at"),
        lambda v: v.update(assessed_at=(NOW + timedelta(seconds=1)).isoformat()),
        lambda v: v.update(valid_until=NOW.isoformat()),
        lambda v: v.update(valid_until="2026-10-05"),
        lambda v: v.update(claims=[]),
        lambda v: v.update(sources={}),
        lambda v: v.update(counterevidence="checked"),
        lambda v: v["sources"].append(deepcopy(v["sources"][0])),
        lambda v: v["claims"]["site_task"].update(status="contradicted"),
        lambda v: (v.update(candidate_digest="0" * 64), v["claims"]["site_task"].update(status="contradicted")),
        lambda v: (v.update(valid_until=None), v["claims"]["site_task"].update(status="contradicted")),
        lambda v: v["counterevidence"].update(status="contradicted", source_refs=["synthetic-primary"]),
        lambda v: v["counterevidence"].update(status="bogus"),
        lambda v: v["claims"].pop("operator"),
        lambda v: v["sources"][0].update(freshness="historical"),
    ]
    for mutate in mutations:
        value = assessment(c, NOW)
        mutate(value)
        yield c, value
    yield c, oct4_shape(c)
    yield c, None
    yield {**c, "organization": ""}, assessment(c, NOW)


@pytest.mark.parametrize("case", list(_equivalence_corpus()))
def test_decisions_match_frozen_legacy_gate_only_reasons_change(case):
    c, value = case
    legacy = _legacy_evaluate(c, deepcopy(value), NOW)
    current = verification.evaluate(c, deepcopy(value), NOW)
    assert {k: current[k] for k in legacy} == legacy
