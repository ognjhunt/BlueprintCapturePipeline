"""A broad research backlog must survive without becoming outreach authority."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tests.test_daily_research_adaptive import result
from tests.test_daily_research_knowledge import enable_v3
from tests.test_daily_research_runner import DAY, NOW, decision
from tests.test_daily_research_runner import fixture as runner_fixture
from tools.daily_research import discovery, search
from tools.daily_research.consumer import qa_text
from tools.daily_research.runner import canonical, output_issues, prompt, validate_output


@pytest.fixture
def fixture(tmp_path):
    yield from runner_fixture.__wrapped__(tmp_path)


def backlog(count=24):
    return [{"organization": f"Sourced operator {n}", "site": f"Facility {n}", "location": "Test city",
             "task": "Sourced recurring task", "capability_family": ["machine tending", "picking", "inspection"][n % 3],
             "site_task_confidence": "high", "robot_fit_confidence": "low",
             "robot_fit": "Plausible fit; integration and support remain unknown", "buying_interest": "unknown; no buyer response",
             "status": "needs_research", "remaining_questions": ["Verify robot support and exact task conditions"],
             "sources": [{"claim": "Operator describes recurring work", "url": f"https://operator{n}.example/process",
                          "publisher": "Operator", "quote": "Exact synthetic process passage", "source_date": None,
                          "checked_date": DAY}]} for n in range(count)]


def test_broad_backlog_survives_packet_restart_and_does_not_admit_24_prospects(fixture, tmp_path):
    runner, api, ledger = fixture
    enable_v3(runner, tmp_path)
    runner.config.update(discovery_profile="adaptive-sites-v1", max_runtime_seconds=1800, qa_reserved_seconds=600)
    out, ctx, policy = result(1)
    out["opportunity_shortlist"] = backlog()
    schema = json.loads((Path(__file__).resolve().parents[1] / "tools/daily_research/daily-research.v3.schema.json").read_text())
    Draft202012Validator(schema).validate(out)
    candidates, _ = validate_output(out, DAY, set(), contract_version=3, knowledge_context=ctx, refresh_policy=policy, observed_at=NOW)
    assert len(candidates) == 1
    api.raw, api.tool_count = canonical(out).encode(), 17
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review"
    assert ledger.get(DAY)["packet"]["opportunity_shortlist"] == out["opportunity_shortlist"]
    qa = qa_text(row, row["crm_snapshot"], "synthetic-crm-digest")
    assert "research backlog, not accepted CRM rows" in qa
    assert "Sourced operator 23" in qa
    reviewed = runner.review(DAY, decision(row))
    assert len(reviewed["delivery"]["sheets"]["payload"]["candidates"]) == 1
    assert reviewed["packet"]["opportunity_shortlist"][0]["buying_interest"].startswith("unknown")
    assert len(api.payloads) == 1


@pytest.mark.parametrize("field,value", [("checked_date", "2026-09-29"), ("source_date", "2026-10-04"),
                                          ("url", "https://user:password@operator.example/source"), ("quote", "")])
def test_bad_shortlist_sources_produce_located_same_session_repair_feedback(field, value):
    out, ctx, policy = result(1)
    out["opportunity_shortlist"] = backlog(1)
    out["opportunity_shortlist"][0]["sources"][0][field] = value
    issues = list(output_issues(out, DAY, contract_version=3, knowledge_context=ctx, refresh_policy=policy, observed_at=NOW, collect=True))
    assert {"pointer": "/opportunity_shortlist/0/sources/0", "code": "discovery_shortlist_invalid"} in issues


def test_legacy_v3_output_does_not_need_backlog_and_legacy_prompt_stays_narrow():
    out, ctx, policy = result(1)
    before = deepcopy(out)
    assert len(validate_output(out, DAY, set(), contract_version=3, knowledge_context=ctx, refresh_policy=policy, observed_at=NOW)[0]) == 1
    assert out == before
    assert "opportunity_shortlist" not in prompt(DAY)
    assert "Find up to THREE" in prompt(DAY)
    prospective = prompt(DAY, ctx, 3, adaptive=True, search_provider=search.PROFILE, target_usd=5)
    assert discovery.TEAM_DIRECTORY in prospective
    assert "opportunity_shortlist" in prospective
    assert "max_results=20" in prospective
    assert "first CRM-ready row is not a reason to end exploration" in prospective
