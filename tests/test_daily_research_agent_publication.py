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
        publication_input = json.loads(ledger.read_bytes(DAY + "-publication-input.json"))
        assert "Sheets accepts only strategy full with summary omitted" in publication_input["input"][0]["content"][0]["text"]
        assert "Every new candidate claim or batch" in publication_input["input"][0]["content"][0]["text"]
        assert "Do not invent these gates or refresh evidence dates" in publication_input["input"][0]["content"][0]["text"]
        def ask(cid, name, arguments):
            state["actions"] = [{"type": "function_call", "turn_id": "turn_publication", "call_id": cid,
                                 "name": name, "arguments": arguments}]
            assert consumer.step()["state"] == "publication_running"
            event = events[-1][1]
            result = json.loads(event["output"])
            if result["success"] is False:
                assert event["success"] is False
                assert json.loads(event["error"]) == result["error"]
            return result
        inspected = ask("inspect_delivery", publication.INSPECT, {})
        assert inspected["success"] and "destinations" in inspected["output"]
        failed = ask("bad_format", publication.PUBLISH, {"destination": "notion", "strategy": "concise"})
        assert failed["success"] is False and failed["error"]["code"]
        assert failed["error"]["issues"][0]["path"] == "/summary"
        fixed = ask("chosen_concise", publication.PUBLISH, {"destination": "notion", "strategy": "concise",
            "summary": "Supported task claim: https://plant.example/tasks — orderability unknown."})
        if rejected:
            assert fixed["success"] is False
            assert fixed["error"]["provider_feedback"]["http_status"] == 400
            assert fixed["error"]["recovery_policy"] == "new_agent_presentation_after_verified_absence"
            fixed = ask("agent_corrected_provider_error", publication.PUBLISH, {"destination": "notion", "strategy": "concise",
                "summary": "Supported source https://plant.example/tasks; commercial interest remains unknown."})
        assert fixed["receipt"]["readback_verified"] is True
        for cid, choice in [("sheets_concise_with_summary", {"destination": "sheets", "strategy": "concise", "summary": "A display summary"}),
                            ("sheets_concise_without_summary", {"destination": "sheets", "strategy": "concise"})]:
            invalid = ask(cid, publication.PUBLISH, choice)
            assert invalid["error"]["code"] == "publication_agent_tool_arguments_invalid"
            assert invalid["error"]["issues"][0]["path"] == "/strategy"
            assert ledger.get(DAY)["delivery"]["sheets"]["state"] == "pending"
            assert ledger.get(DAY)["application_tool_calls"][cid]["request"]["arguments"] == choice
        sheet = ask("chosen_crm", publication.PUBLISH, {"destination": "sheets", "strategy": "full"})
        assert sheet["receipt"]["readback_verified"] is True
        # A stopped/rebound worker may collect an already-terminal exact turn
        # using GETs; it must not send cancellation, inference or another upload.
        if rejected:
            with ledger.lock():
                control = bridge.call("control")
                control["workflow"]["publication_authority_reference"] = "replacement-authority"
                bridge.call("configure", value=control)
        else:
            consumer = Consumer(ledger, consumer.config, api, clock=consumer.clock, stopped=lambda: True)
        result_count = len(events)
        state.update(actions=[], status="completed")
        assert consumer.step()["state"] == "completed"
        final = ledger.get(DAY)
        assert len(posts) == 1 and posts[0][0] == "sess_1"
        assert len(events) == result_count and not api.cancellations
        assert not final["publication"].get("cancel_attempted")
        assert bridge.call("active_qa") is None
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


@pytest.mark.parametrize("interruption", ["stop", "deadline", "authority", "disabled", "prior_cancel", "revoked", "malformed"])
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
        if interruption == "prior_cancel":
            with ledger.lock():
                publication.cancel(consumer, ledger.get(DAY), "publication_deadline_reached")
            consumer = Consumer(ledger, consumer.config, api, clock=consumer.clock, stopped=lambda: True)
        if interruption == "stop":
            consumer = Consumer(ledger, consumer.config, api, clock=consumer.clock, stopped=lambda: True)
        elif interruption == "deadline":
            consumer.clock = lambda: NOW + timedelta(seconds=1801)
        elif interruption != "prior_cancel":
            with ledger.lock():
                control = bridge.call("control")
                if interruption == "disabled":
                    control["enabled"] = False
                elif interruption == "revoked":
                    control["workflow"].pop("publication_authority_reference")
                elif interruption == "malformed":
                    control["workflow"] = "revoked-workflow"
                else:
                    control["workflow"]["publication_authority_reference"] = "replacement-authority"
                bridge.call("configure", value=control)
            if interruption in {"disabled", "revoked", "malformed"}:
                consumer = Consumer(ledger, consumer.config, api, clock=consumer.clock)
        assert consumer.step()["state"] == "publication_cancel_pending"
        assert consumer.step()["state"] == "publication_cancel_pending"
        row = ledger.get(DAY)
        expected_cancellations = int(interruption in {"deadline", "prior_cancel"})
        assert len(submissions) == 1 and len(cancellations) == expected_cancellations
        if expected_cancellations:
            assert cancellations[0] == ("sess_1", row["run_key"] + ":publication")
            assert row["publication"]["cancel_reply_unresolved"] is True
        else:
            assert row["publication"]["observation_only_reason"]
            assert not row["publication"].get("cancel_attempted")
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
        assert len(cancellations) == expected_cancellations and not ledger.get(DAY).get("application_tool_calls")
        consumer.stopped = lambda: interruption in {"stop", "prior_cancel"}
        def terminal_turn(resource, sid=None):
            values = listing(resource, sid)
            if resource == "turns":
                values.append({"id": "turn_publication", "session_id": "sess_1", "agent_id": AGENT,
                    "status": "cancelled", "subagent_id": None, "completed_at": int((NOW + timedelta(seconds=40)).timestamp()),
                    "usage": {"input_tokens": 15}})
            if resource == "items":
                values.append({"turn_id": "turn_publication", "type": "message", "content": "Stopped; publication remains incomplete."})
            return values
        api.listing = terminal_turn
        assert consumer.step()["state"] == "publication_agent_incomplete"
        final = ledger.get(DAY)
        assert final["publication"]["turn_status"] == "cancelled"
        assert final["publication"]["usage"] == {"input_tokens": 15}
        if expected_cancellations:
            assert final["publication"]["cancel_reply_unresolved"] is True
        assert final["state"] == "reviewed" and len(submissions) == 1 and len(cancellations) == expected_cancellations
        assert bridge.call("active_qa") is None
        assert json.loads(ledger.read_bytes(DAY + "-publication-evidence.json"))[0]["content"]
        if interruption in {"stop", "disabled", "prior_cancel"}:
            assert consumer.step()["state"] == "workflow_disabled"
        elif interruption in {"revoked", "malformed"}:
            with pytest.raises(Refusal, match="workflow_authority_missing"):
                consumer.step()
    finally:
        generator.close()
