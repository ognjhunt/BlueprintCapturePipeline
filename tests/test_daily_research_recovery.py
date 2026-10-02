"""Repair evidence, ordinary daily lifecycle, and immutable export contracts."""
import hashlib
import inspect
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

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


@pytest.mark.parametrize("fields", [("organization", "task"), ("claim", "publisher")])
def test_partial_field_repairs_are_located_and_not_false_no_progress(fields):
    _, _, policy, context = policy_bundle()
    output = v3(context)
    row = {"date": DAY, "research_contract_version": 3, "knowledge_context": context, "refresh_policy": policy}
    item = output["candidates"][0]
    prefix = "/candidates/0"
    if fields[0] == "claim":
        item = item["evidence"][0]
        prefix += "/evidence/0"
    original = deepcopy(item)
    for field in fields:
        item[field] = ""
    first = recovery.validation_feedback(output, row, set(), NOW)
    assert {prefix + "/" + field for field in fields} <= {issue["path"] for issue in first}
    item[fields[0]] = original[fields[0]]
    partial = recovery.validation_feedback(output, row, set(), NOW)
    assert recovery.feedback_signature(first) != recovery.feedback_signature(partial)
    assert prefix + "/" + fields[0] not in {issue["path"] for issue in partial}
    assert prefix + "/" + fields[1] in {issue["path"] for issue in partial}
    with pytest.raises(Refusal):
        validate_output(output, DAY, set(), contract_version=3, knowledge_context=context,
                        refresh_policy=policy, observed_at=NOW)
    item[fields[1]] = original[fields[1]]
    assert recovery.validation_feedback(output, row, set(), NOW) == []


def test_failure_signature_tracks_invalid_value_but_ignores_unrelated_metadata():
    _, _, policy, context = policy_bundle()
    output = v3(context)
    row = {"date": DAY, "research_contract_version": 3, "knowledge_context": context, "refresh_policy": policy}
    source = output["candidates"][0]["evidence"][0]
    source["classification"] = "invalid-a"
    first = recovery.validation_feedback(output, row, set(), NOW)
    source["quote"] += " unrelated valid wording"
    assert recovery.feedback_signature(first) == recovery.feedback_signature(recovery.validation_feedback(output, row, set(), NOW))
    source["classification"] = "invalid-b"
    changed = recovery.validation_feedback(output, row, set(), NOW)
    assert recovery.feedback_signature(first) != recovery.feedback_signature(changed)
    assert recovery.feedback_signature(changed) == recovery.feedback_signature(recovery.validation_feedback(output, row, set(), NOW))


@pytest.mark.parametrize("target", ["organization", "evidence"])
def test_malformed_urls_produce_corrective_feedback_instead_of_parser_exception(target):
    _, _, policy, context = policy_bundle()
    output = v3(context)
    row = {"date": DAY, "research_contract_version": 3, "knowledge_context": context, "refresh_policy": policy}
    candidate = output["candidates"][0]
    if target == "organization":
        candidate["organization_url"] = "https://["
        pointer = "/candidates/0/organization_url"
    else:
        candidate["evidence"][0]["url"] = "https://["
        pointer = "/candidates/0/evidence/0/url"
    feedback = recovery.validation_feedback(output, row, set(), NOW)
    assert any(issue["path"] == pointer and issue["reason"] == "source_url_invalid" for issue in feedback)


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


def test_cold_disabled_recovery_discovers_and_cancels_active_correction(tmp_path, monkeypatch):
    generator = consumer_setup(tmp_path, failed=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        turns, calls = [], []
        listing = api.listing
        def repair_input(sid, _event, key, day, request_digest, deadline_ms):
            bridge.call("repair_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
            calls.append(key)
            turns.append({"id": "turn_repair", "agent_id": AGENT, "session_id": sid,
                          "subagent_id": None, "status": "in_progress"})
        def values(resource, sid=None):
            result = listing(resource, sid)
            if resource == "turns":
                result.extend(deepcopy(turns))
            return result
        cancel = api.cancel
        def cancelled(sid, key):
            cancel(sid, key)
            turns[0]["status"] = "cancelled"
        api.repair_input, api.listing, api.cancel = repair_input, values, cancelled
        running = recovery.RepairLoop(ledger, consumer.config, api, clock=consumer.clock).step(DAY)
        assert running["validation_repairs"][-1]["state"] == "running"
        with ledger.lock():
            control = bridge.call("control")
            bridge.call("configure", value={**control, "enabled": False})
        assert bridge.call("active_qa") == DAY
        class FixedDatetime:
            @staticmethod
            def now(_zone):
                return consumer.clock()
        monkeypatch.setattr(render, "datetime", FixedDatetime)
        monkeypatch.setattr(render, "configured", lambda *_: {**consumer.config, "enabled": False})
        monkeypatch.setattr(render.time, "sleep", lambda _: None)
        # New controller follows the active-intent lookup despite disabled QA
        # and terminal original root state; it never creates or publishes.
        result = render.invoke("reconcile", bridge, tmp_path, api_factory=lambda *_: api)
        assert result["state"] == "validation_repair_blocked"
        final = ledger.get(DAY)
        assert final["validation_repairs"][-1]["turn_status"] == "cancelled"
        assert final["validation_repairs"][-1]["cancel_attempted"] is True
        assert len(api.cancellations) == len(api.payloads) == len(calls) == 1
        assert not api.inputs and not bridge.call("active_qa")
    finally:
        generator.close()


@pytest.mark.parametrize("reason", ["disabled", "expired", "stopped"])
def test_status_read_failure_does_not_skip_one_durable_repair_cancellation(tmp_path, reason):
    generator = consumer_setup(tmp_path, failed=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        turns, calls = [], []
        listing = api.listing
        def repair_input(sid, _event, key, day, request_digest, deadline_ms):
            bridge.call("repair_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
            calls.append(key)
            turns.append({"id": "turn_repair", "agent_id": AGENT, "session_id": sid,
                          "subagent_id": None, "status": "in_progress"})
        def values(resource, sid=None):
            result = listing(resource, sid)
            if resource == "turns":
                result.extend(turns)
            return result
        api.repair_input, api.listing = repair_input, values
        recovery.RepairLoop(ledger, consumer.config, api, clock=consumer.clock).step(DAY)
        if reason == "disabled":
            with ledger.lock():
                bridge.call("configure", value={**bridge.call("control"), "enabled": False})
        def unavailable(_resource, _identifier):
            raise TimeoutError("synthetic provider status unavailable")
        api.get = unavailable
        clock = (lambda: NOW + timedelta(seconds=181)) if reason == "expired" else consumer.clock
        for _ in range(2):
            row = recovery.RepairLoop(ledger, consumer.config, api, clock=clock, stopped=lambda: reason == "stopped").step(DAY)
            revision = row["validation_repairs"][-1]
            assert revision["state"] == "cancel_pending" and revision["cancel_attempted"] is True
            assert revision["observation_error"] == "validation_repair_provider_observation_unavailable"
            assert revision.get("turn_status") != "cancelled"  # GET failure proves no native terminal state.
        assert bridge.call("active_qa") == DAY
        assert len(api.cancellations) == len(api.payloads) == len(calls) == 1 and not api.inputs
    finally:
        generator.close()


def sdk_error(code="environment_connection_timeout", request_id="req_0123456789abcdef"):
    import httpx2
    from openai import BadRequestError
    request = httpx2.Request("POST", "https://api.openai.com/v1/agents/sessions/private/events",
                             headers={"Authorization": "Bearer never-retain-this"})
    response = httpx2.Response(400, request=request, headers={"x-request-id": request_id})
    return BadRequestError("private message and secret never-retain-this", response=response,
                           body={"code": code, "message": "private body never-retain-this"})


def verify_sdk_error_receipt():
    assert recovery.repair_error_receipt(sdk_error(), "provider_submission") == {
        "stage": "provider_submission", "class": "BadRequestError",
        "code": "environment_connection_timeout", "http_status": 400,
        "request_id": "req_0123456789abcdef"}
    error = sdk_error(code="secret-code", request_id="Bearer secret")
    error.status_code = True
    receipt = recovery.repair_error_receipt(error, "unknown-private-stage")
    assert receipt == {"stage": "dispatch", "class": "BadRequestError", "code": None,
                       "http_status": None, "request_id": None}
    assert "private" not in canonical(receipt) and "secret" not in canonical(receipt)
    error = RuntimeError("secret")
    error.status_code, error.request_id, error.code = 400, "req_0123456789abcdef", "idle_timeout"
    assert recovery.repair_error_receipt(error, "preconditions") == {
        "stage": "preconditions", "class": "other", "code": None,
        "http_status": None, "request_id": None}
    assert recovery.repair_error_receipt(Refusal("secret"), "preconditions")["code"] is None
    import httpx2
    from openai import InternalServerError
    request = httpx2.Request("POST", "https://api.openai.com/v1/agents/sessions/private/events")
    for hint, expected in [("15", 15), ("86401", 86400), ("not-a-delay", None),
                           ("Fri, 02 Oct 2099 10:00:00 GMT", 86400)]:
        response = httpx2.Response(503, request=request, headers={"retry-after": hint, "x-private": "never-retain-this",
            "date": "Fri, 02 Oct 2026 09:00:00 GMT", "x-request-id": "req_0123456789abcdef"})
        error = InternalServerError("private", response=response, body={"code": "service_unavailable_error"})
        receipt = recovery.repair_error_receipt(error, "provider_submission")
        assert receipt.get("retry_after_seconds") == expected
        assert receipt["http_status"] == 503 and receipt["code"] == "service_unavailable_error"
        assert "private" not in canonical(receipt) and "never-retain-this" not in canonical(receipt)


def test_receipt_retains_only_typed_allowlisted_metadata():
    runtime = os.environ.get("BLUEPRINT_RESEARCH_SDK_PYTHON", sys.executable)
    script = "from tools.daily_research import recovery\nfrom tools.daily_research.runner import Refusal, canonical\n"
    script += inspect.getsource(sdk_error) + "\n" + inspect.getsource(verify_sdk_error_receipt)
    script += "\nverify_sdk_error_receipt()\n"
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([runtime, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", ["rejected", "lost_reply", "local_refusal", "reply_persistence"])
def test_error_receipt_persists_and_restart_never_resubmits(tmp_path, monkeypatch, failure):
    generator = consumer_setup(tmp_path, failed=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        calls = []
        put = ledger.put
        persisted_reply_failed = False
        def fail_one_put(row):
            nonlocal persisted_reply_failed
            current = (row.get("validation_repairs") or [{}])[-1]
            if failure == "reply_persistence" and current.get("state") == "running" and not persisted_reply_failed:
                persisted_reply_failed = True
                raise Refusal("firestore_bridge_unavailable")
            return put(row)
        monkeypatch.setattr(ledger, "put", fail_one_put)
        def submit(sid, event, key, day, request_digest, deadline_ms):
            calls.append((sid, event, key, request_digest, deadline_ms))
            if failure == "local_refusal":
                raise Refusal("validation_repair_input_not_admitted")
            bridge.call("repair_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
            api.repair_input_phase = "provider_submission"
            if failure == "rejected":
                raise RuntimeError("synthetic input rejection")
            if failure == "lost_reply":
                raise TimeoutError("accepted reply may be lost; private data")
        api.repair_input = submit
        loop = recovery.RepairLoop(ledger, consumer.config, api, clock=consumer.clock)
        first = loop.step(DAY)
        revision = deepcopy(first["validation_repairs"][-1])
        assert revision["input_attempted"] is True
        assert revision["input_error_receipt"]["stage"] == (
            "reply_persistence" if failure == "reply_persistence" else
            "preconditions" if failure == "local_refusal" else "provider_submission")
        assert revision["state"] == ("running" if failure == "reply_persistence" else "input_unresolved")
        assert ledger.get(DAY)["validation_repairs"][-1]["input_error_receipt"] == revision["input_error_receipt"]
        recovery.RepairLoop(ledger, consumer.config, api, clock=consumer.clock).step(DAY)
        final = ledger.get(DAY)["validation_repairs"][-1]
        assert len(calls) == 1
        assert (final["request_digest"], final["deadline_ms"], final["input_attempted"]) == (
            revision["request_digest"], revision["deadline_ms"], True)
        assert not api.inputs
        assert "private" not in canonical(final["input_error_receipt"])
    finally:
        generator.close()


def test_qa_lost_reply_has_sanitized_receipt_and_is_observed_without_resend(tmp_path):
    generator = consumer_setup(tmp_path)
    consumer, api, ledger, _bridge, _ = next(generator)
    try:
        qa_input = api.qa_input
        def lost_reply(*args):
            api.qa_input_phase = "provider_submission"
            qa_input(*args)
            raise TimeoutError("private provider request was accepted but reply lost")
        api.qa_input = lost_reply
        assert consumer.step()["state"] == "qa_input_unresolved"
        receipt = ledger.get(DAY)["qa"]["input_error_receipt"]
        assert receipt == {"stage": "provider_submission", "class": "TimeoutError",
                           "code": None, "http_status": None, "request_id": None}
        assert consumer.step()["state"] == "reviewed"
        assert len(api.inputs) == len(api.payloads) == 1
    finally:
        generator.close()
