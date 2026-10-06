"""Frozen qualified evidence input, consumer trust boundary, and free tool tests. Synthetic only."""
import base64
import copy
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests.daily_team_evidence_fixture import TODAY, export, pin, repack
from tests.test_daily_research_search import fixture as search_fixture
from tools.daily_research import team_universe as te
from tools.daily_research.runner import canonical


@pytest.fixture
def fixture(tmp_path):
    yield from search_fixture.__wrapped__(tmp_path)


def snapshot(raw):
    return SimpleNamespace(team_universe_snapshot=lambda: {"pin": pin(raw), "data": base64.b64encode(raw).decode()})


def attached(raw=None, today=TODAY):
    raw = raw or export()
    record, frozen = te.attach(snapshot(raw), today)
    body = {"input": "Synthetic daily research.", "metadata": {}, "environment": {"files": []}}
    te.bind(body, record, frozen)
    return {"date": today.isoformat(), "team_universe": record, "create_payload": body, "metadata": body["metadata"]}


def test_all_206_canonical_rows_reachable_without_count_quota():
    row = attached(export(206))
    result = te.execute(row, {})
    assert result["ok"] and result["total"] == len(result["teams"]) == 206 and result["next_cursor"] is None
    assert sum(r["recommendation_eligible"] for r in result["teams"]) == 69
    assert all(not r["recommendation_eligible"] for r in result["teams"] if r["status"] in {"reference_only", "pending"})
    assert all(r["qualification"]["evaluation_compatibility"] == "not_verified" for r in result["teams"])
    assert "contact_email" not in canonical(result) and "never call it confirmed" in row["create_payload"]["input"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(status="beta_candidate"),
    lambda r: r["qualification"].update(promotion_allowed=True),
    lambda r: r["qualification"].update(evaluation_compatibility="supported"),
    lambda r: r["qualification"].update(willingness_to_work_with_blueprint="yes"),
    lambda r: r["qualification"].update(robot_forms=["software_only"]),
    lambda r: r["qualification"].update(task_families=["invented"]),
    lambda r: r["assessment"].update(identified_hardware="unproven robot product"),
    lambda r: r["assessment"].update(offering_relation="future fixture task"),
    lambda r: r["assessment"]["proofs"][0].update(url="https://thirdparty.example/post"),
    lambda r: r["assessment"]["proofs"][0].update(page_sha256=None),
    lambda r: r["assessment"]["proofs"][0].update(quote_sha256="0" * 64),
    lambda r: r["source_binding"].update(result_sha256=None),
    lambda r: r.update(contact_email="synthetic@synthetic.example"),
])
def test_unqualified_or_tampered_provenance_refused_even_after_rows_digest_updated(mutation):
    value = json.loads(export())
    row = next(r for r in value["teams"] if r["status"] == "capability_prospect")
    mutation(row)
    with pytest.raises(te.TeamEvidenceError):
        te.load(repack(value), today=TODAY)


def test_export_pin_binding_duplicates_and_assessment_currentness():
    raw = export()
    for bad in [{**pin(raw), "audit_sha256": "0" * 64}, {**pin(raw), "generation": "0"}, {**pin(raw), "version": True}]:
        with pytest.raises(te.TeamEvidenceError):
            te.load(raw, expected=bad, today=TODAY)
    with pytest.raises(te.TeamEvidenceError):
        te.load(raw.replace(b'"schema_version":', b'"schema_version":"extra","schema_version":', 1), today=TODAY)
    with pytest.raises(te.TeamEvidenceError, match="not_current"):
        te.load(raw, today=TODAY - timedelta(days=1))


def test_expired_capability_held_for_new_run_without_rewriting_export():
    raw = export()
    row = attached(raw, TODAY + timedelta(days=549))
    frozen = te.frozen(row)
    assert te.encoded({"schema_version": te.EXPORT, "manifest": frozen["manifest"], "teams": frozen["teams"]}) == raw
    result = te.execute(row, {})
    assert not any(r["recommendation_eligible"] for r in result["teams"])
    assert any(r.get("export_status") == "capability_prospect" and r["status"] == "pending" for r in result["teams"])


def test_frozen_tool_filters_exact_ids_and_detects_changed_file_or_pin():
    row = attached()
    result = te.execute(row, {"task_family": "palletizing_depalletizing"})
    assert result["ok"] and result["total"] == 1
    key = result["teams"][0]["team_key"]
    assert te.execute(row, {"team_key": key})["total"] == 1
    changed = copy.deepcopy(row)
    changed["create_payload"]["environment"]["files"][0]["data"] = base64.b64encode(b"{}").decode()
    assert not te.execute(changed, {})["ok"]
    row["team_universe"]["pin"]["audit_sha256"] = "0" * 64
    assert not te.execute(row, {})["ok"]
    for arguments in [{"live": True}, {"cursor": True}, {"cursor": 50}, {"task_family": []}, "bad"]:
        assert not te.execute(attached(), arguments)["ok"]


def test_missing_or_unavailable_is_actionable_ordinary_research():
    record, raw = te.attach(SimpleNamespace(), TODAY)
    assert record["state"] == "unavailable" and raw is None
    body = {"input": "Ordinary research."}
    te.bind(body, record, raw)
    assert "ordinary research" in body["input"] and "without inventing" in body["input"]
    assert not te.execute({"date": TODAY.isoformat()}, {})["ok"]


def test_actual_daily_intent_freezes_team_input_tool_and_recovery_does_not_refetch(fixture):
    runner, api, ledger = fixture
    value = json.loads(export())
    value["manifest"]["assessed_on"] = runner.clock().date().isoformat()
    for team in value["teams"]:
        team["screen_checked_on"] = value["manifest"]["assessed_on"]
        if team["assessment"]["as_of"]:
            team["assessment"]["as_of"] = value["manifest"]["assessed_on"]
    raw = repack(value)
    calls = []
    def read_once():
        calls.append(True)
        return snapshot(raw).team_universe_snapshot()
    ledger.team_universe_snapshot = read_once
    row = runner.start_or_resume()
    assert row["team_universe"]["state"] == "attached" and len(calls) == 1
    assert api.payloads[0] == row["create_payload"] and te.frozen(row)["teams"] == json.loads(raw)["teams"]
    assert row["metadata"]["team_universe_input_digest"] == row["team_universe"]["sha256"]
    assert te.READ in {tool["name"] for tool in api.payloads[0]["agent"]["tools"]}
    api.actions = [{"type": "function_call", "turn_id": row["turn_id"], "call_id": "team-call-1", "name": te.READ, "arguments": "{}"}]
    row = runner.start_or_resume(allow_create=False)
    assert not api.executions and api.result_events and len(calls) == 1
    frozen_payload = copy.deepcopy(row["create_payload"])
    runner.start_or_resume()
    assert len(calls) == 1 and len(api.payloads) == 1 and ledger.get(row["date"])["create_payload"] == frozen_payload


def test_canonical_encoding_required_before_pin_or_freeze():
    pretty = json.dumps(json.loads(export()), indent=2).encode()
    with pytest.raises(te.TeamEvidenceError, match="noncanonical"):
        te.load(pretty, today=TODAY)
    record, raw = te.attach(SimpleNamespace(team_universe_snapshot=lambda: {"pin": {**pin(export()), "sha256": te.sha(pretty), "uri": te.object_uri(te.sha(pretty)), "bytes": len(pretty)}, "data": base64.b64encode(pretty).decode()}), TODAY)
    assert raw is None and record["state"] == "unavailable" and record["code"] == "team_universe_export_noncanonical"


def test_not_established_general_invitation_is_valid_without_relationship():
    value = json.loads(export())
    value["teams"][0]["qualification"]["partner_intent"] = {"state": "not_established", "kind": "general"}
    assert te.load(repack(value), today=TODAY)["teams"][0]["qualification"]["relationship_implied"] is False


def test_explicit_invitation_needs_separate_owned_retained_proof():
    value = json.loads(export())
    row = next(r for r in value["teams"] if r["status"] == "capability_prospect")
    row["qualification"]["partner_intent"] = {"state": "explicit_invitation", "kind": "pilot"}
    with pytest.raises(te.TeamEvidenceError):
        te.load(repack(value), today=TODAY)
    proof = copy.deepcopy(row["assessment"]["proofs"][0])
    proof.update(field="seeking_partners", quote="We welcome pilot partners for our synthetic deployment program.")
    proof["quote_sha256"] = te.sha(proof["quote"].encode())
    row["invitation_evidence"] = [proof]
    assert te.load(repack(value), today=TODAY)["teams"]
    proof["url"] = "https://unrelated.example/partners"
    with pytest.raises(te.TeamEvidenceError):
        te.load(repack(value), today=TODAY)


def test_run_day_and_utc_assessment_are_separate_across_central_midnight():
    raw = export()
    record, frozen = te.attach(snapshot(raw), TODAY, run_date="2026-10-04")
    body = {"input": "Synthetic daily research.", "metadata": {}, "environment": {"files": []}}
    te.bind(body, record, frozen)
    row = {"date": "2026-10-04", "team_universe": record, "create_payload": body, "metadata": body["metadata"]}
    assert te.execute(row, {})["ok"] and te.frozen(row)["as_of"] == TODAY.isoformat()


def test_held_currentness_not_established_is_visible_and_never_recommendable():
    value = json.loads(export())
    row = next(r for r in value["teams"] if r["status"] == "pending")
    row["assessment"]["current_basis"] = "not_established"
    raw = repack(value)
    assert te.load(raw, today=TODAY)["teams"]
    result = te.execute(attached(raw), {})
    assert not next(r for r in result["teams"] if r["team_key"] == row["team_key"])["recommendation_eligible"]
