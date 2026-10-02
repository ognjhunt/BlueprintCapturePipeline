"""Actual saved-session dispatch to a hermetic company host, never inference."""
import json
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests.test_daily_research_consumer import consumer_setup
from tests.test_daily_research_runner import AGENT, DAY, NOW
from tools.daily_research import history, publication, render, search
from tools.daily_research.firestore import FencedProvider
from tools.daily_research.runner import Refusal, Runner


def test_legacy_tools_are_unchanged_and_history_is_explicit():
    assert [t["name"] for t in search.tools()] == [search.SEARCH, search.READ]
    assert len(search.tools(publication.PROFILE)) == 4
    assert [t["name"] for t in search.tools(publication.PROFILE, history.PROFILE)][-2:] == [history.SEARCH, history.FETCH]


def test_rejected_history_and_early_publication_expose_native_error(tmp_path):
    generator = consumer_setup(tmp_path, publication=True, history=True, research_running=True)
    consumer, api, ledger, _, _ = next(generator)
    try:
        events, state = [], {"actions": []}
        original_get = api.get
        def get(resource, rid):
            value = original_get(resource, rid)
            if resource == "session":
                value["required_actions"] = deepcopy(state["actions"])
            return value
        api.get = get
        provider = object.__new__(FencedProvider)
        provider.ledger, provider.get, provider.listing, provider.clock = ledger, get, api.listing, consumer.clock
        api.tool_admit = provider.tool_admit
        api.tool_result = lambda sid, event, key: events.append(deepcopy(event))
        runner = Runner(ledger, consumer.config, api, clock=consumer.clock)
        runner.required_history = True
        for cid, name, arguments, code in [
            ("early_upload", publication.PUBLISH, {"destination": "notion", "strategy": "full"}, "publication_requires_review"),
            ("invalid_history", history.SEARCH, {"query": "", "page_size": "50"}, "company_history_arguments_invalid"),
        ]:
            state["actions"] = [{"type": "function_call", "turn_id": "turn_1", "call_id": cid,
                                 "name": name, "arguments": arguments}]
            assert runner.start_or_resume(allow_create=False)["state"] == "running"
            event = events[-1]
            result = json.loads(event["output"])
            assert event["call_id"] == cid and event["success"] is False
            assert json.loads(event["error"]) == result["error"]
            assert result["error"]["code"] == code
        assert ledger.get(DAY)["delivery"] == {}
    finally:
        generator.close()


def test_agent_selects_history_queries_pages_records_and_corrects_errors_in_all_phases(tmp_path):
    generator = consumer_setup(tmp_path, publication=True, history=True, research_running=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        row = ledger.get(DAY)
        assert row["create_payload"]["agent"]["tools"] == search.tools(publication.PROFILE, history.PROFILE)
        assert "learning_context" not in row and not (tmp_path / "history-requests.json").exists()
        events, posts, state = [], [], {"actions": [], "publication_status": "in_progress"}
        original_get, original_listing = api.get, api.listing
        def get(resource, rid):
            value = original_get(resource, rid)
            if resource == "session":
                value["required_actions"] = deepcopy(state["actions"])
            return value
        def listing(resource, sid=None):
            values = original_listing(resource, sid)
            if resource == "turns" and posts:
                values.append({"id": "turn_publication", "session_id": "sess_1", "agent_id": AGENT,
                    "status": state["publication_status"], "subagent_id": None,
                    "completed_at": int((NOW + timedelta(seconds=40)).timestamp())})
            return values
        api.get, api.listing = get, listing
        provider = object.__new__(FencedProvider)
        provider.ledger, provider.get, provider.listing, provider.clock = ledger, get, listing, consumer.clock
        provider.api = SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(create=lambda sid, **kw: posts.append((sid, kw)))))
        api.tool_admit, api.publication_input = provider.tool_admit, provider.publication_input
        api.tool_result = lambda sid, event, key: events.append((sid, deepcopy(event), key))
        runner = Runner(ledger, consumer.config, api, clock=consumer.clock)
        runner.required_history = True
        def ask(phase, cid, name, arguments):
            tid = {"research": "turn_1", "qa": "turn_qa", "publication": "turn_publication"}[phase]
            state["actions"] = [{"type": "function_call", "turn_id": tid, "call_id": cid, "name": name, "arguments": arguments}]
            if phase == "research":
                assert runner.start_or_resume(allow_create=False)["state"] == "running"
            else:
                assert consumer.step()["state"] == ("qa_running" if phase == "qa" else "publication_running")
            assert events[-1][0] == "sess_1" and events[-1][1]["turn_id"] == tid
            return json.loads(events[-1][1]["output"])
        premature = ask("research", "premature_upload", publication.PUBLISH, {"destination": "notion", "strategy": "full"})
        assert not premature["ok"] and premature["error"]["code"] == "publication_requires_review"
        assert ledger.get(DAY)["delivery"] == {} and not (tmp_path / "history-requests.json").exists()
        bad = ask("research", "invalid_history", history.SEARCH, {"query": "", "page_size": "50"})
        assert not bad["ok"] and bad["error"]["issues"][0]["field"] == "page_size"
        first = ask("research", "chosen_query", history.SEARCH, {"query": "unexpected repetitive work", "filters": {"city": "Seattle"}, "page_size": 3})
        assert first["ok"] and first["semantic"]["status"] == "unavailable"
        ask("research", "chosen_next_page", history.SEARCH, {"query": "unexpected repetitive work", "cursor": first["next_cursor"], "page_size": 3})
        # Resubmitting the pending action uses identical retained output, without another company read.
        before = (tmp_path / "history-requests.json").read_bytes()
        runner.start_or_resume(allow_create=False)
        assert (tmp_path / "history-requests.json").read_bytes() == before
        state["actions"], api.turn_status = [], "completed"
        assert runner.start_or_resume(allow_create=False)["state"] == "awaiting_review"
        api.qa_status = "in_progress"
        assert consumer.step()["state"] == "qa_running"
        missing = ask("qa", "bad_record_id", history.FETCH, {"record_id": "absent_record"})
        assert not missing["ok"] and missing["error"]["code"] == "company_history_record_not_found"
        full = ask("qa", "chosen_record", history.FETCH, '{"record_id":"record_a"}')
        assert len(full["record"]["content"]) > 10000 and full["record"]["created_at"] == "2026-09-29T08:00:00Z"
        state["actions"], api.qa_status = [], "completed"
        assert consumer.step()["state"] == "reviewed"
        assert consumer.step()["state"] == "publication_running"
        inspected = ask("publication", "publication_history", history.SEARCH, {"query": "", "filters": {"kind": "research"}})
        assert inspected["ok"]
        requests = json.loads((tmp_path / "history-requests.json").read_bytes())
        assert len(requests) == 5 and all(v["request"]["op"] in {"history_search", "history_fetch"} for v in requests)
        assert requests[0]["request"] == {"op": "history_search", "day": DAY, "query": "unexpected repetitive work", "filters": {"city": "Seattle"}, "page_size": 3}
        assert all(v["binding"] == ledger.get(DAY)["history_binding"] for v in requests)
        assert len(posts) == 1
        assert render.export_snapshot(bridge, DAY, tmp_path / "export")["missing_files"] == []
        with ledger.lock():
            control = bridge.call("control")
            control["learning"]["binding"]["companyId"] = "replacement-company"
            bridge.call("configure", value=control)
            with pytest.raises(Refusal, match="company_history_authority_changed"):
                provider.tool_admit(ledger.get(DAY), "publication")
        assert len(json.loads((tmp_path / "history-requests.json").read_bytes())) == 5
    finally:
        generator.close()
