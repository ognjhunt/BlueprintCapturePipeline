"""Repair evidence, ordinary daily lifecycle, and immutable export contracts."""
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from tests.test_daily_research_consumer import consumer_setup
from tests.test_daily_research_knowledge import policy_bundle, v3
from tests.test_daily_research_runner import AGENT, DAY, NOW
from tools.daily_research import recovery, render, search
from tools.daily_research.runner import Ledger, Refusal, canonical, digest, validate_output


def precision_fixture(tmp_path, precise="2026-10-02T01:48:47.469036+00:00"):
    moment = datetime.fromisoformat(precise)
    # checked_day uses the canonical America/Chicago calendar, including DST.
    day = recovery.checked_day(precise)
    row = {"date": day, "turn_id": "turn_original", "started_at": (moment - timedelta(seconds=60)).isoformat()}
    output = {"candidates": [{"evidence": [
        {"origin": "live", "source_checked_at": day, "checked_date": day, "revalidated_at": precise,
         "url": "https://operator.example/task"},
        {"origin": "snapshot", "source_checked_at": "2026-09-29", "revalidated_at": None}]}]}
    request = {"name": search.READ, "arguments": {"url": "https://operator.example/task"},
               "call_id": "call_read", "turn_id": row["turn_id"]}
    source = {"url": request["arguments"]["url"], "requested_url": request["arguments"]["url"],
              "checked_at": precise, "truncated": False,
              "evidence_scope": "complete_static_extracted_text_not_javascript_rendered"}
    event = {"type": "agent.session.input.tool_result", "call_id": "call_read",
             "turn_id": row["turn_id"], "success": True, "output": canonical(source)}
    raw = (canonical(event) + "\n").encode()
    ledger = Ledger(tmp_path)
    filename = day + "-tool-call_read.json"
    ledger.write_bytes(filename, raw)
    row["application_tool_calls"] = {"call_read": {"phase": "research", "request": request,
        "request_digest": digest(request), "success": True, "result_acknowledged": True,
        "result_file": filename, "result_bytes": len(raw), "result_sha256": hashlib.sha256(raw).hexdigest(),
        "result_digest": digest(event)}}
    return output, row, ledger, moment + timedelta(seconds=1)


@pytest.mark.parametrize("precise", ["2026-10-02T01:48:47.469036+00:00", "2026-11-01T06:30:00+00:00", "2026-11-01T07:30:00+00:00"])
def test_precision_normalization_binds_exact_receipt_across_midnight_and_dst(tmp_path, precise):
    output, row, ledger, observed = precision_fixture(tmp_path, precise)
    original = deepcopy(output)
    derived, changes = recovery.normalize_live_date_precision(output, row, ledger, observed)
    assert output == original and derived["candidates"][0]["evidence"][1] == original["candidates"][0]["evidence"][1]
    assert changes[0]["original"] == row["date"] and changes[0]["derived"] == precise
    assert derived["candidates"][0]["evidence"][0]["source_checked_at"] == precise
    assert changes[0]["receipt"]["result_sha256"] == row["application_tool_calls"]["call_read"]["result_sha256"]


@pytest.mark.parametrize("field,value", [("phase", "qa"), ("result_acknowledged", False),
    ("result_sha256", "0" * 64), ("result_digest", "0" * 64), ("result_bytes", 1)])
def test_precision_cannot_use_unbound_unacknowledged_or_corrupt_receipts(tmp_path, field, value):
    output, row, ledger, observed = precision_fixture(tmp_path)
    row["application_tool_calls"]["call_read"][field] = value
    with pytest.raises(Refusal, match="receipt_missing|binding_invalid"):
        recovery.normalize_live_date_precision(output, row, ledger, observed)


def test_background_is_not_a_robot_grade_and_employer_sources_require_agent_qa():
    _, _, policy, context = policy_bundle()
    output = v3(context)
    candidate = output["candidates"][0]
    candidate["evidence"][0]["url"] = "https://employmenthero.com/employer/job"
    ordinary = deepcopy(candidate["evidence"][0])
    ordinary.update(role="background", evidence_level=None)
    candidate["evidence"].append(ordinary)
    from pathlib import Path

    import jsonschema
    schema = json.loads((Path(__file__).parents[1] / "tools/daily_research/daily-research.v3.schema.json").read_text())
    jsonschema.validate(output, schema)
    accepted, _ = validate_output(output, DAY, set(), contract_version=3, knowledge_context=context,
                                  refresh_policy=policy, observed_at=NOW)
    assert accepted[0]["operator_affiliation_qa_required"] is True
    assert accepted[0]["qualification_status"] == "unqualified"
    candidate["evidence"] = [e for e in candidate["evidence"] if e["role"] != "capability"]
    with pytest.raises(Refusal, match="task_capability_geography_evidence_required"):
        validate_output(output, DAY, set(), contract_version=3, knowledge_context=context,
                        refresh_policy=policy, observed_at=NOW)


def test_feedback_checks_coverage_after_independent_delta_and_date_errors():
    from tests.test_daily_research_knowledge import delta
    _, _, policy, context = policy_bundle()
    output = v3(context)
    proposal = delta()
    proposal["evidence"][0].update(classification="operator", evidence_level=None)
    output["proposed_knowledge_deltas"] = [proposal]
    output["candidates"][0]["evidence"][0]["revalidated_at"] = NOW.isoformat()
    row = {"date": DAY, "research_contract_version": 3, "knowledge_context": context,
           "refresh_policy": policy, "discovery_profile": "adaptive-sites-v1", "search_provider": search.PROFILE}
    feedback = recovery.validation_feedback(output, row, set(), NOW)
    assert {i["path"] for i in feedback} >= {"/proposed_knowledge_deltas/0/evidence/0/evidence_level",
        "/candidates/0/evidence/0/source_checked_at", "/coverage"}
    output["proposed_knowledge_deltas"], output["candidates"] = None, None
    assert recovery.validation_feedback(output, row, set(), NOW)


def test_ordinary_daily_worker_repairs_then_qa_and_publishes_without_new_root(tmp_path, monkeypatch):
    generator = consumer_setup(tmp_path, failed=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        original = ledger.get(DAY)
        raw = ledger.read_bytes(DAY + "-artifact.json")
        repaired = json.loads(raw)
        repaired["proposed_knowledge_deltas"] = []
        repair_turns, calls = [], []
        listing, artifact = api.listing, api.artifact
        def repair_input(sid, event, key, day, request_digest, deadline_ms):
            current = ledger.get(day)["validation_repairs"][-1]
            assert json.loads(ledger.read_bytes(current["input_file"])) == event
            bridge.call("repair_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
            calls.append(key)
            repair_turns.append({"id": "turn_daily_repair", "session_id": sid, "agent_id": AGENT,
                "subagent_id": None, "status": "completed", "completed_at": int((NOW + timedelta(seconds=25)).timestamp())})
        def values(resource, sid=None):
            result = listing(resource, sid)
            if resource == "turns":
                result.extend(repair_turns)
            if resource == "artifacts" and repair_turns:
                result.append({"id": "artifact_daily_repair", "turn_id": "turn_daily_repair", "path": recovery.REPAIR_PATH})
            return result
        api.repair_input, api.listing = repair_input, values
        api.artifact = lambda sid, aid: canonical(repaired).encode() if aid == "artifact_daily_repair" else artifact(sid, aid)
        class FixedDatetime:
            @staticmethod
            def now(_zone):
                return consumer.clock()
        monkeypatch.setattr(render, "datetime", FixedDatetime)
        monkeypatch.setattr(render, "configured", lambda *_: consumer.config)
        monkeypatch.setattr(render, "Consumer", lambda *a, **kw: type(consumer)(*a, **kw, clock=consumer.clock))
        monkeypatch.setattr(render.time, "sleep", lambda _: None)
        assert bridge.call("work_item")["stage"] == "validation_repair_pending"
        result = render.consume_workflow(bridge, tmp_path, api_factory=lambda *_: api)
        assert result["state"] == "completed"
        final = ledger.get(DAY)
        assert len(api.payloads) == len(calls) == len(api.inputs) == 1
        assert final["started_at"] == final["validation_repair_authority"]["started_at"] == original["started_at"]
        assert final["raw_output_digest"] == original["raw_output_digest"] and ledger.read_bytes(DAY + "-artifact.json") == raw
        assert final["original_validation_failure"]["error"] == "knowledge_delta_evidence_invalid"
        assert all(d["receipt"]["readback_verified"] for d in final["delivery"].values())
        with ledger.lock(), pytest.raises(Refusal, match="not_admitted"):
            revision = final["validation_repairs"][-1]
            bridge.call("repair_check", day=DAY, request_digest=revision["request_digest"], deadline_ms=revision["deadline_ms"])
    finally:
        generator.close()
