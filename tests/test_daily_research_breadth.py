"""Breadth-first discovery keeps a broad sourced backlog without granting outreach authority.

Ported from ognjhunt/BlueprintCapturePipeline#2561 onto the v3 discovery_inventory.
Supports ADP-010 partner discovery (partner-phase day-7 gate): broad research must
survive collection and QA, and only fully reviewed candidates reach the CRM.
Hermetic: no provider, Notion, search or sink calls.
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tests.test_daily_research_adaptive import result
from tests.test_daily_research_runner import DAY, NOW, decision
from tests.test_daily_research_search import SearchAPI
from tests.test_daily_research_search import fixture as search_fixture
from tools.daily_research import discovery, recovery, search, verification
from tools.daily_research.consumer import qa_text
from tools.daily_research.runner import (
    canonical,
    preflight,
    prompt,
    status_summary,
    validate_output,
)

SCHEMA = Path(__file__).resolve().parents[1] / "tools/daily_research/daily-research.v3.schema.json"
FAMILIES = ("machine tending", "picking", "inspection")
DISPOSITIONS = ("unresolved", "rejected", "duplicate", "learning", "candidate")


@pytest.fixture
def fixture(tmp_path):
    yield from search_fixture.__wrapped__(tmp_path)


def backlog(count=24):
    return [{"operator": f"Sourced operator {n}", "site": f"Facility {n}", "location": "Test city",
             "task_hypothesis": f"Recurring {FAMILIES[n % 3]} at this site; robot fit and buying interest unknown",
             "source_urls": [f"https://operator{n}.example/process"],
             "evidence_gap": "Verify exact task conditions, incumbent automation and a credible routing contact",
             "disposition": DISPOSITIONS[n % 5]} for n in range(count)]


def scoped(count=1):
    out, ctx, policy = result(count)
    out["coverage"].update(defined_run_scope=["Synthetic multi-family site/task scope"],
                           unresolved_promising_branches=[], completion_state="coverage_complete")
    return out, ctx, policy


@pytest.mark.parametrize("count", [24,300])
def test_broad_inventory_survives_collection_and_qa_without_admitting_backlog_leads(fixture,count):
    runner, api, ledger = fixture
    out = scoped(1)[0]
    out["discovery_inventory"] = backlog(count)
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(out)
    api.raw, api.turn_status = canonical(out).encode(), "completed"
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review", row.get("error")
    manifest = row["packet"]["discovery_inventory_manifest"]
    assert manifest["record_count"] ==count and manifest["complete_retention"] is True
    assert manifest["eligibility"] == "discovery_only_no_promotion"
    cursor, retained = 0, []
    while cursor is not None:
        page = discovery.read_inventory_page(manifest, ledger, cursor)
        retained.extend(page["records"])
        cursor = page["next_cursor"]
    assert retained == out["discovery_inventory"]
    # Backlog leads marked "candidate" stay research leads; only the formal candidate is reviewable.
    assert len(verification.packet_candidates(row["packet"])) == 1
    qa = qa_text(row, row["crm_snapshot"], "synthetic-crm-digest")
    assert "research backlog, not accepted CRM rows" in qa and "Team Directory entry never authorizes" in qa
    assert "premature concentration" in qa and "max_results=20" in qa
    reviewed = runner.review(DAY, decision(row))
    assert len(reviewed["delivery"]["sheets"]["payload"]["candidates"]) == 1
    assert "Sourced operator" not in canonical(reviewed["delivery"])
    assert len(api.payloads) ==1
    funnel = status_summary(reviewed)["discovery_funnel"]
    assert funnel["raw_inventory_records"] ==count and funnel["formal_candidates"] ==1
    assert funnel["accepted_for_crm"] ==1 and funnel["counts_are_not_interchangeable"] is True


@pytest.mark.parametrize("field,value,pointer", [
    ("source_urls", ["https://user:password@operator0.example/source"], "/discovery_inventory/0/source_urls/0"),
    ("disposition", "needs_research", "/discovery_inventory/0"),
    ("evidence_gap", " ", "/discovery_inventory/0"),
    ("operator", "", "/discovery_inventory/0/operator"),
])
def test_bad_inventory_entries_get_located_same_session_repair_feedback(field, value, pointer):
    out, ctx, policy = scoped(1)
    out["discovery_inventory"] = backlog(1)
    out["discovery_inventory"][0][field] = value
    row = {"date": DAY, "research_contract_version": 3, "knowledge_context": ctx, "refresh_policy": policy,
           "search_provider": search.PROFILE, "discovery_profile": "adaptive-sites-v1"}
    feedback = recovery.validation_feedback(out, row, set(), NOW)
    assert {(item["path"], item["reason"]) for item in feedback} == {(pointer, "discovery_inventory_invalid")}
    assert feedback[0]["allowed_semantics"] == recovery.RULES["discovery_inventory_invalid"]


def test_breadth_guidance_reaches_only_new_adaptive_sessions_and_matches_the_tool_contract():
    out, ctx, policy = result(1)
    before = deepcopy(out)
    assert len(validate_output(out, DAY, set(), contract_version=3, knowledge_context=ctx,
                               refresh_policy=policy, observed_at=NOW)[0]) == 1
    assert out == before  # Legacy v3 output needs no inventory.
    for legacy in (prompt(DAY), prompt(DAY, ctx, 3)):
        assert discovery.TEAM_DIRECTORY not in legacy and "max_results=20" not in legacy
    assert "Find up to THREE" in prompt(DAY)
    native = prompt(DAY, ctx, 3, adaptive=True)
    assert discovery.TEAM_DIRECTORY in native and "first CRM-ready row is not a reason to end exploration" in native
    assert "max_results=20" not in native  # The native lane has no blueprint_search tool.
    selected = prompt(DAY, ctx, 3, adaptive=True, search_provider=search.PROFILE, target_usd=5)
    assert discovery.TEAM_DIRECTORY in selected and "max_results=20" in selected
    session = preflight(SearchAPI(), search_provider=search.PROFILE)["session_agent_override"]
    assert "max_results=20" in session["instructions"] and session["tools"] == search.tools()
    # The advertised result-set guidance matches the enforced, unchanged tool contract.
    declared = next(tool for tool in search.tools() if tool["name"] == search.SEARCH)
    assert declared["parameters"]["properties"]["max_results"] == {"type": "integer", "minimum": 1, "maximum": 20}
    assert search.search_body({"query": "q"})["max_results"] == 10
    assert search.search_body({"query": "q", "max_results": 20})["max_results"] == 20
    with pytest.raises(search.ToolFailure, match="search_arguments_invalid"):
        search.search_body({"query": "q", "max_results": 21})


def test_enumeration_precedes_deep_qualification_and_keeps_unknown_tasks():
    text=discovery.instructions(5)
    assert "first stage broad enumeration" in text and "on the order of hundreds" in text
    assert "Set unsupported fields to null" in text and "prioritize a bounded subset" in text
    inventory=[{"operator":"Sourced operator","site":"Named warehouse","location":"US location",
        "task_hypothesis":None,"source_urls":["https://operator.example/locations"],
        "evidence_gap":"Exact manual workflow has not been researched","disposition":"unresolved"}]
    assert list(discovery.inventory_issues(inventory)) ==[]
