"""Agent choices/errors drive delivery through the real bridge; no inference."""
import hashlib
import json
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests.test_daily_research_consumer import consumer_setup
from tests.test_daily_research_runner import AGENT, DAY, NOW
from tools.daily_research import publication, render, search
from tools.daily_research.consumer import Consumer
from tools.daily_research.firestore import FencedProvider
from tools.daily_research.runner import Refusal


def test_explicit_new_profile_preserves_old_tools_and_declares_publication():
    assert [t["name"] for t in search.tools()] == [search.SEARCH, search.READ]
    assert [t["name"] for t in search.tools(publication.PROFILE)] == [search.SEARCH, search.READ, publication.INSPECT, publication.PUBLISH]


@pytest.mark.parametrize("rejected", [False, True])
def test_saved_agent_inspects_repairs_its_format_and_uploads_with_feedback(tmp_path, rejected):
    generator = consumer_setup(tmp_path, publication=True, publication_rejection=rejected)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        assert consumer.step()["state"] == "reviewed"
        row = ledger.get(DAY)
        assert row["create_payload"]["agent"]["tools"] == search.tools(publication.PROFILE)
        original_raw = ledger.read_bytes(DAY + "-qa.json")
        original_payload = deepcopy(row["delivery"]["notion"]["payload"])
        posts, events, state = [], [], {"actions": [], "status": "in_progress"}
        listing, get = api.listing, api.get
        def values(resource, sid=None):
            result = listing(resource, sid)
            if resource == "turns" and posts:
                result.append({"id": "turn_publication", "session_id": "sess_1", "agent_id": AGENT,
                    "status": state["status"], "subagent_id": None, "completed_at": int((NOW + timedelta(seconds=40)).timestamp()),
                    "usage": {"input_tokens": 100, "output_tokens": 20}})
            if resource == "items" and posts:
                result.append({"id": "final_publication_reasoning", "turn_id": "turn_publication", "type": "message",
                    "content": "Both approved destinations verified; original evidence retained."})
            return result
        api.listing = values
        def session(resource, rid):
            result = get(resource, rid)
            if resource == "session":
                result["required_actions"] = deepcopy(state["actions"])
            return result
        api.get = session
        provider = object.__new__(FencedProvider)
        provider.ledger, provider.get, provider.listing, provider.clock = ledger, api.get, api.listing, consumer.clock
        provider.api = SimpleNamespace(sessions=SimpleNamespace(events=SimpleNamespace(create=lambda sid, **kw: posts.append((sid, kw)))))
        api.publication_input = provider.publication_input
        api.tool_admit = provider.tool_admit
        api.tool_result = lambda sid, event, key: events.append((sid, deepcopy(event), key))
        assert consumer.step()["state"] == "publication_running"
        def ask(cid, name, arguments):
            state["actions"] = [{"type": "function_call", "turn_id": "turn_publication", "call_id": cid,
                                 "name": name, "arguments": arguments}]
            assert consumer.step()["state"] == "publication_running"
            return json.loads(events[-1][1]["output"])
        inspected = ask("inspect_delivery", publication.INSPECT, {})
        assert inspected["success"] and "destinations" in inspected["output"]
        failed = ask("bad_format", publication.PUBLISH, {"destination": "notion", "strategy": "concise"})
        assert failed["success"] is False and failed["error"]["code"]
        fixed = ask("chosen_concise", publication.PUBLISH, {"destination": "notion", "strategy": "concise",
            "summary": "Supported task claim: https://plant.example/tasks — orderability unknown."})
        if rejected:
            assert fixed["success"] is False
            assert fixed["error"]["provider_feedback"]["http_status"] == 400
            assert fixed["error"]["recovery_policy"] == "new_agent_presentation_after_verified_absence"
            fixed = ask("agent_corrected_provider_error", publication.PUBLISH, {"destination": "notion", "strategy": "concise",
                "summary": "Supported source https://plant.example/tasks; commercial interest remains unknown."})
        assert fixed["receipt"]["readback_verified"] is True
        sheet = ask("chosen_crm", publication.PUBLISH, {"destination": "sheets", "strategy": "full"})
        assert sheet["receipt"]["readback_verified"] is True
        state.update(actions=[], status="completed")
        assert consumer.step()["state"] == "completed"
        final = ledger.get(DAY)
        assert len(posts) == 1 and posts[0][0] == "sess_1"
        assert final["delivery"]["notion"]["payload"] == original_payload
        assert ledger.read_bytes(DAY + "-qa.json") == original_raw
        assert final["publication"]["usage"]["output_tokens"] == 20
        assert json.loads(ledger.read_bytes(DAY + "-publication-evidence.json"))[0]["content"]
        destination = tmp_path / "agent-publication-export"
        assert render.export_snapshot(bridge, DAY, destination)["missing_files"] == []
        exported_manifest = json.loads((destination / "publication-manifest.json").read_bytes())
        assert json.loads(exported_manifest["manifest_json"])["schema_version"] == "blueprint.research-publication-manifest.v1"
        if rejected:
            envelope = bridge.call("snapshot", day=DAY)["publication_manifest"]
            proof = json.loads(envelope["manifest_json"])
            assert set(proof["attempt_history"]["notion"]) == {"0", "1"}
            assert proof["attempt_history"]["notion"]["0"]["claimed"] != proof["publication_claimed"]["notion"]
            for field in ("source_row_blob", "response_digest", "request_digest"):
                damaged = deepcopy(proof)
                original = damaged["attempt_history"]["notion"]["0"]
                target = original if field == "source_row_blob" else original["rejection"]
                target[field] = "f" * 64
                raw = json.dumps(damaged)
                with pytest.raises(Refusal, match="publication_manifest_binding_invalid"):
                    render.publication_manifest(final, {"manifest_json": raw, "manifest_digest": hashlib.sha256(raw.encode()).hexdigest()})
    finally:
        generator.close()


@pytest.mark.parametrize("interruption", ["stop", "deadline", "authority"])
def test_uncertain_publication_is_cancelled_once_on_bound_interruption(tmp_path, interruption):
    generator = consumer_setup(tmp_path, publication=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        assert consumer.step()["state"] == "reviewed"
        submissions, cancellations = [], []
        def lost_input(*args):
            submissions.append(args)
            raise TimeoutError()
        def lost_cancel(*args):
            cancellations.append(args)
            raise TimeoutError()
        api.publication_input, api.cancel = lost_input, lost_cancel
        assert consumer.step()["state"] == "publication_input_unresolved"
        if interruption == "stop":
            consumer = Consumer(ledger, consumer.config, api, clock=consumer.clock, stopped=lambda: True)
        elif interruption == "deadline":
            consumer.clock = lambda: NOW + timedelta(seconds=1801)
        else:
            with ledger.lock():
                control = bridge.call("control")
                control["workflow"]["publication_authority_reference"] = "replacement-authority"
                bridge.call("configure", value=control)
        assert consumer.step()["state"] == "publication_cancel_pending"
        assert consumer.step()["state"] == "publication_cancel_pending"
        row = ledger.get(DAY)
        assert len(submissions) == len(cancellations) == 1
        assert cancellations[0] == ("sess_1", row["run_key"] + ":publication")
        assert row["publication"]["cancel_reply_unresolved"] is True
        assert row["state"] == "reviewed" and all(d["state"] == "pending" for d in row["delivery"].values())
        # Re-enabling after a lost cancellation cannot resume publication tools.
        consumer.stopped = lambda: False
        listing = api.listing
        def accepted_turn(resource, sid=None):
            values = listing(resource, sid)
            if resource == "turns":
                values.append({"id": "turn_publication", "session_id": "sess_1", "agent_id": AGENT,
                               "status": "in_progress", "subagent_id": None})
            return values
        api.listing = accepted_turn
        assert consumer.step()["state"] == "publication_cancel_pending"
        assert len(cancellations) == 1 and not ledger.get(DAY).get("application_tool_calls")
    finally:
        generator.close()
