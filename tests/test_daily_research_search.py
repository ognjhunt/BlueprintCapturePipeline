"""Hermetic full runner/profile integration; no provider, search or sinks."""
import base64
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from tests.test_daily_research_adaptive import result
from tests.test_daily_research_knowledge import enable_v3
from tests.test_daily_research_runner import DAY, NOW, FakeAPI
from tests.test_daily_research_runner import fixture as runner_fixture
from tools.daily_research import discovery, render, search
from tools.daily_research.consumer import Consumer, qa_text
from tools.daily_research.firestore import FencedProvider
from tools.daily_research.runner import (
    AGENT,
    Refusal,
    canonical,
    check_agent,
    configuration,
    digest,
    preflight,
)


class SearchAPI(FakeAPI):
    def __init__(self):
        super().__init__()
        self.agent["instructions"] = "Reviewed evidence. Native web_search only."
        self.present = True
        self.executions = []
        self.result_events = []
        self.turn_status = "in_progress"
        self.actions = []

    def search_binding_present(self):
        return self.present

    def get(self, resource, rid):
        value = super().get(resource, rid)
        if resource == "session" and self.payloads:
            value["agent"].update(deepcopy(self.payloads[0].get("agent", {})))
            value["required_actions"] = deepcopy(self.actions)
        return value

    def tool_admit(self, row, phase):
        assert row["search_provider"] == search.PROFILE

    def application_tool(self, name, args):
        self.executions.append((name, args))
        return {"results": [{"url": "https://operator.example/source", "snippet": "Complete passage"}]}

    def tool_result(self, sid, event, key):
        self.result_events.append((sid, deepcopy(event), key))


@pytest.fixture
def fixture(tmp_path):
    generator = runner_fixture.__wrapped__(tmp_path)
    runner, _, ledger = next(generator)
    enable_v3(runner, tmp_path)
    runner.config.update(discovery_profile="adaptive-sites-v1", max_runtime_seconds=1800,
                         qa_reserved_seconds=600, search_provider=search.PROFILE,
                         recurring_budget_authority_reference="owner-approved-synthetic-test-target")
    api = SearchAPI()
    output = result()[0]
    output["coverage"].update(defined_run_scope=["Synthetic exact site/task industry/region scope"],
                              unresolved_promising_branches=[], completion_state="coverage_complete")
    api.raw = canonical(output).encode()
    runner.api = api
    yield runner, api, ledger
    try:
        next(generator)
    except StopIteration:
        pass


def test_missing_secret_refuses_before_create_and_does_not_change_old_intent(fixture):
    runner, api, ledger = fixture
    api.present = False
    with pytest.raises(Refusal, match="perplexity_binding_missing"):
        runner.start_or_resume()
    assert not api.payloads and ledger.rows() == []


def test_new_session_replaces_tools_without_saved_agent_or_sandbox_changes(fixture):
    runner, api, ledger = fixture
    original = deepcopy(api.agent)
    row = runner.start_or_resume()
    payload = api.payloads[0]
    assert row["search_provider"] == search.PROFILE
    assert payload["agent"]["tools"] == search.tools()
    assert payload["agent"]["service_tier"] == "default"
    assert payload["environment"]["network"] == {"access": "disabled"}
    assert payload["agent_id"] == original["id"] and api.agent == original
    assert "no prospect-count stopping rule" in payload["input"]
    assert "Native web_search only." not in payload["input"]
    assert "PERPLEXITY_API_KEY" not in canonical(payload)
    assert ledger.get(DAY)["create_payload"] == payload


def test_owner_mcp_is_explicit_additive_and_frozen_before_create(fixture):
    runner, api, ledger = fixture
    connections = [{"type": "mcp", "server_label": label,
        "transport": {"type": "http", "server_url": url, "headers": {}},
        "credential_id": "credential_synthetic_owner_" + label, "allowed_tools": None,
        "connection_origin": "service", "required": False, "request_metadata": {}}
        for label, (url, _) in search.MCP_READ_TOOLS.items() if label in {"googlesheets", "slack"}]
    api.agent["tools"].extend(deepcopy(connections))
    original = deepcopy(api.agent)
    with pytest.raises(Refusal, match="agent_configuration_mismatch"):
        runner.start_or_resume()
    assert not api.payloads and ledger.rows() == []
    runner.config.update(mcp_profile=search.MCP_PROFILE, publication_profile="agent-owned-v1")
    checked = preflight(api, search_provider=search.PROFILE, publication_profile="agent-owned-v1", mcp_profile=search.MCP_PROFILE)
    assert checked["inference_started"] is False and not api.payloads and not api.executions
    with_history = preflight(api, search_provider=search.PROFILE, publication_profile="agent-owned-v1",
                             history_profile="agent-history-v1", mcp_profile=search.MCP_PROFILE)
    assert with_history["session_agent_override"]["tools"] == search.tools("agent-owned-v1", "agent-history-v1") + search.mcp_tools(connections)
    row = runner.start_or_resume()
    payload = api.payloads[0]
    assert payload["agent"]["tools"] == search.tools("agent-owned-v1") + search.mcp_tools(connections)
    assert api.agent == original and payload["environment"]["network"] == {"access": "disabled"}
    assert payload["vault_ids"] == ["vault_synthetic_googlesheets", "vault_synthetic_slack"]
    assert "headers" not in payload["environment"]
    assert row["mcp_binding"] == connections == row["preflight"]["mcp_binding"]
    assert payload["metadata"]["mcp_binding_digest"] == digest(connections)
    assert payload["metadata"]["mcp_vault_binding_digest"] == digest(row["mcp_vault_binding"])
    assert len(row["mcp_vault_binding"]) == 2
    assert ledger.get(DAY)["create_payload"] == payload
    assert all(tool["required"] is False and tool["credential_id"] == source["credential_id"]
        for tool, source in zip(payload["agent"]["tools"][-2:], connections, strict=True))
    assert not any(name.startswith(("update", "slack_send", "slack_schedule"))
        for tool in payload["agent"]["tools"][-2:] for name in tool["allowed_tools"])
    assert [tool["name"] for tool in search.tools("agent-owned-v1", "agent-history-v1")][-2:] == [
        "search_company_history", "fetch_company_history_record"]
    session = api.get("session", row["session_id"])
    Consumer.check_session(row, session)
    for tool in session["agent"]["tools"][-2:]:
        tool["transport"].pop("headers")
    Consumer.check_session(row, session)
    changed = deepcopy(connections)
    changed[0]["credential_id"] += "_changed"
    api.agent["tools"][-2:] = changed
    notion = {"type": "mcp", "server_label": "notion",
        "transport": {"type": "http", "server_url": "https://mcp.notion.com/mcp", "headers": {}},
        "credential_id": "credential_synthetic_owner_notion", "allowed_tools": None,
        "connection_origin": "service", "required": False, "request_metadata": {}}
    api.agent["tools"].append(notion)
    Consumer.check_session(row, api.get("session", row["session_id"]))
    fresh = preflight(api, search_provider=search.PROFILE, mcp_profile=search.MCP_PROFILE)
    assert fresh["mcp_binding"] == changed + [notion] and row["mcp_binding"] == connections
    assert len(api.payloads) == 1 and ledger.get(DAY)["create_payload"] == payload


@pytest.mark.parametrize("change", ["missing", "duplicate", "unrelated", "endpoint", "auth_type", "vault_id"])
def test_owner_vault_metadata_rejects_unbound_scope_before_intent_or_create(fixture, change):
    runner, api, ledger = fixture
    tool = {"type": "mcp", "server_label": "notion", "transport": {"type": "http", "server_url": "https://mcp.notion.com/mcp"},
        "credential_id": "credential_synthetic_notion", "allowed_tools": None, "connection_origin": "service",
        "required": False, "request_metadata": {}}
    api.agent["tools"].append(tool)
    runner.config["mcp_profile"] = search.MCP_PROFILE
    credential = {"id": tool["credential_id"], "vault_id": "vault_synthetic_notion",
        "auth": {"type": "mcp_oauth", "mcp_server_url": tool["transport"]["server_url"]}}
    api.vault_credentials = {"vault_synthetic_notion": [credential]}
    if change == "missing":
        api.vault_credentials = {}
    elif change == "duplicate":
        api.vault_credentials["vault_synthetic_second"] = [{**credential, "vault_id": "vault_synthetic_second"}]
    elif change == "unrelated":
        api.vault_credentials["vault_synthetic_notion"].append({**credential, "id": "credential_unrelated"})
    elif change == "endpoint":
        credential["auth"]["mcp_server_url"] += "/unreviewed"
    elif change == "auth_type":
        credential["auth"]["type"] = "environment_variable"
    else:
        credential["vault_id"] = "vault_different"
    with pytest.raises(Refusal, match="research_mcp_vault_"):
        runner.start_or_resume()
    assert ledger.rows() == [] and not api.payloads and not api.executions
    assert all(call[0] == "GET" for call in api.calls)


@pytest.mark.parametrize("change", ["missing_attachment", "extra_attachment", "duplicate_attachment", "binding", "payload"])
def test_charged_mcp_vault_binding_cannot_change_during_recovery(fixture, change):
    runner, api, ledger = fixture
    tool = {"type": "mcp", "server_label": "slack", "transport": {"type": "http", "server_url": "https://mcp.slack.com/mcp"},
        "credential_id": "credential_synthetic_slack", "allowed_tools": None, "connection_origin": "service",
        "required": False, "request_metadata": {}}
    api.agent["tools"].append(tool)
    runner.config["mcp_profile"] = search.MCP_PROFILE
    row = runner.start_or_resume()
    original = deepcopy(row["create_payload"])
    session = api.get("session", row["session_id"])
    if change == "missing_attachment":
        session.pop("vault_ids")
    elif change == "extra_attachment":
        session["vault_ids"].append("vault_unrelated")
    elif change == "duplicate_attachment":
        session["vault_ids"] *= 2
    elif change == "binding":
        row["mcp_vault_binding"][0]["vault_id"] = "vault_other"
    else:
        row["create_payload"]["vault_ids"] = ["vault_other"]
    with pytest.raises(Refusal, match="research_mcp_vault_binding_changed"):
        Consumer.check_session(row, session)
    assert ledger.get(DAY)["create_payload"] == original and len(api.payloads) == 1 and not api.executions


def test_old_charged_mcp_intent_does_not_gain_vaults_or_read_new_inventory(fixture):
    runner, api, ledger = fixture
    tool = {"type": "mcp", "server_label": "slack", "transport": {"type": "http", "server_url": "https://mcp.slack.com/mcp"},
        "credential_id": "credential_synthetic_slack", "allowed_tools": None, "connection_origin": "service",
        "required": False, "request_metadata": {}}
    api.agent["tools"].append(tool)
    runner.config["mcp_profile"] = search.MCP_PROFILE
    row = runner.start_or_resume()
    # Reproduce the original, already charged pre-vault intent without changing its tools/instructions.
    row.pop("mcp_vault_binding")
    row["metadata"].pop("mcp_vault_binding_digest")
    row["create_payload"].pop("vault_ids")
    legacy_digest_input = deepcopy(row["create_payload"])
    legacy_digest_input["metadata"].pop("payload_digest")
    row["metadata"]["payload_digest"] = digest(legacy_digest_input)
    api.payloads[0] = deepcopy(row["create_payload"])
    api.sessions[0]["metadata"] = deepcopy(row["metadata"])
    api.sessions[0].pop("vault_ids")
    ledger.put(row)
    original = deepcopy(row["create_payload"])
    api.resolve_mcp_vaults = lambda _: pytest.fail("charged recovery must not resolve or attach new vaults")
    api.agent["tools"].append({"type": "mcp", "server_label": "later_owner_connection"})
    resumed = runner.start_or_resume(allow_create=False)
    Consumer.check_session(resumed, api.get("session", resumed["session_id"]))
    assert resumed["create_payload"] == original and "mcp_vault_binding" not in resumed
    assert "vault_ids" not in resumed["create_payload"] and len(api.payloads) == 1


@pytest.mark.parametrize("allowed", [None, ["notion-fetch", "notion-create-pages"]])
def test_owner_notion_is_frozen_and_session_tools_are_only_official_reads(fixture, allowed):
    runner, api, ledger = fixture
    tool = {"type": "mcp", "server_label": "notion",
        "transport": {"type": "http", "server_url": "https://mcp.notion.com/mcp", "headers": {}},
        "credential_id": "credential_synthetic_owner_notion", "allowed_tools": allowed,
        "connection_origin": "service", "required": False, "request_metadata": {}}
    api.agent["tools"].append(deepcopy(tool))
    original = deepcopy(api.agent)
    runner.config["mcp_profile"] = search.MCP_PROFILE
    row = runner.start_or_resume()
    expected = ["notion-get-tool-access", "notion-search", "notion-fetch"] if allowed is None else ["notion-fetch"]
    assert api.agent == original and row["mcp_binding"] == [tool]
    assert row["metadata"]["mcp_binding_digest"] == digest([tool])
    assert api.payloads[0]["agent"]["tools"][-1] == {**tool, "allowed_tools": expected}
    assert "notion-create-pages" not in api.payloads[0]["agent"]["tools"][-1]["allowed_tools"]
    assert "notion-get-tool-access" in api.payloads[0]["agent"]["instructions"]
    session = api.get("session", row["session_id"])
    Consumer.check_session(row, session)
    session["agent"]["tools"][-1]["allowed_tools"].append("notion-update-page")
    with pytest.raises(Refusal, match="agent_search_profile_mismatch"):
        Consumer.check_session(row, session)
    assert len(api.payloads) == 1 and ledger.get(DAY)["create_payload"] == row["create_payload"]
    assert not api.executions


@pytest.mark.parametrize("change", ["headers", "metadata", "endpoint", "origin", "unknown_server", "malformed_label"])
def test_owner_mcp_rejects_unsafe_configuration_before_intent_or_create(fixture, change):
    runner, api, ledger = fixture
    tool = {"type": "mcp", "server_label": "googlesheets",
        "transport": {"type": "http", "server_url": search.MCP_READ_TOOLS["googlesheets"][0], "headers": {}},
        "credential_id": "credential_synthetic_owner", "allowed_tools": None,
        "connection_origin": "service", "required": False, "request_metadata": {}}
    if change == "headers":
        tool["transport"]["headers"] = {"Authorization": "synthetic-secret-must-not-be-retained"}
    elif change == "metadata":
        tool["request_metadata"] = {"token": "synthetic-secret-must-not-be-retained"}
    elif change == "endpoint":
        tool["transport"]["server_url"] += "/unreviewed"
    elif change == "origin":
        tool["connection_origin"] = "environment"
    elif change == "unknown_server":
        tool["server_label"] = "unreviewed"
    else:
        tool["server_label"] = ["googlesheets"]
    api.agent["tools"].append(tool)
    runner.config["mcp_profile"] = search.MCP_PROFILE
    with pytest.raises(Refusal, match="research_mcp_configuration_invalid"):
        runner.start_or_resume()
    assert not api.payloads and ledger.rows() == []


@pytest.mark.parametrize("change", ["credential", "write_tool", "required", "binding", "headers", "payload"])
def test_owner_mcp_session_uses_only_original_read_only_binding(fixture, change):
    runner, api, _ = fixture
    tool = {"type": "mcp", "server_label": "slack",
        "transport": {"type": "http", "server_url": search.MCP_READ_TOOLS["slack"][0]},
        "credential_id": "credential_synthetic_owner", "allowed_tools": ["slack_read_thread", "slack_send_message"],
        "connection_origin": "service", "required": False, "request_metadata": {}}
    api.agent["tools"].append(tool)
    runner.config["mcp_profile"] = search.MCP_PROFILE
    row = runner.start_or_resume()
    session = api.get("session", row["session_id"])
    assert session["agent"]["tools"][-1]["allowed_tools"] == ["slack_read_thread"]
    if change == "credential":
        session["agent"]["tools"][-1]["credential_id"] += "_changed"
    elif change == "write_tool":
        session["agent"]["tools"][-1]["allowed_tools"].append("slack_send_message")
    elif change == "required":
        session["agent"]["tools"][-1]["required"] = True
    elif change == "binding":
        row["mcp_binding"][0]["credential_id"] += "_changed"
    elif change == "headers":
        session["agent"]["tools"][-1]["transport"]["headers"] = {"Authorization": "synthetic-unsafe-header"}
    else:
        row["create_payload"]["agent"]["tools"][-1]["credential_id"] += "_changed"
    with pytest.raises(Refusal, match="agent_search_profile_mismatch|research_mcp_binding_changed"):
        Consumer.check_session(row, session)
    assert not api.executions and len(api.payloads) == 1


def test_mcp_does_not_retrofit_an_existing_legacy_session(fixture):
    runner, api, ledger = fixture
    row = runner.start_or_resume()
    original = deepcopy(row["create_payload"])
    runner.config["mcp_profile"] = search.MCP_PROFILE
    resumed = runner.start_or_resume(allow_create=False)
    assert resumed["create_payload"] == original and "mcp_profile" not in resumed
    assert "mcp_binding_digest" not in resumed["metadata"] and len(api.payloads) == 1
    assert ledger.get(DAY)["create_payload"] == original


def test_fenced_provider_refuses_changed_mcp_profile_with_original_budget(fixture):
    runner, api, _ = fixture
    row = runner.start_or_resume()
    row["mcp_profile"] = search.MCP_PROFILE
    control = {"enabled": True, "config": {"search_provider": search.PROFILE,
        "recurring_budget_authority_reference": row["recurring_budget_authority_reference"],
        "soft_target_usd": row["soft_target_usd"]}}
    calls = []
    class Bridge:
        def call(self, op):
            calls.append(op)
            return control if op == "control" else True
    provider = FencedProvider.__new__(FencedProvider)
    provider.ledger = type("Ledger", (), {"bridge": Bridge()})()
    with pytest.raises(Refusal, match="disabled_or_profile_changed"):
        provider.tool_admit(row, "research")
    assert calls == ["assert_lease", "control"] and not api.executions


def test_exact_pending_root_call_is_served_and_retained_then_ten_collected(fixture):
    runner, api, ledger = fixture
    runner.start_or_resume()
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "call_1",
                    "name": search.SEARCH, "arguments": {"query": "exact operator recurring task"}}]
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "running" and len(api.executions) == 1
    saved = ledger.get(DAY)["application_tool_calls"]["call_1"]
    assert saved["attempted"] and saved["success"] and saved["result_acknowledged"]
    assert json.loads(ledger.read_bytes(saved["result_file"]))["success"] is True
    runner.start_or_resume(allow_create=False)
    assert len(api.executions) == 1
    assert api.result_events[0] == api.result_events[1]
    api.turn_status, api.actions = "completed", []
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and len(row["packet"]["candidates"]) == 10
    assert row["application_tool_usage"]["attempted_search_requests"] == 1
    assert row["application_tool_usage"]["hard_total_cap"] is False


def test_wrong_turn_or_tools_do_not_run_search(fixture):
    runner, api, _ = fixture
    runner.start_or_resume()
    api.actions = [{"type": "function_call", "turn_id": "wrong", "call_id": "call_1",
                    "name": search.SEARCH, "arguments": {"query": "task"}}]
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "cancel_pending" and not api.executions


def test_pinned_deadline_and_cleanup_guard_survive_changed_configuration(fixture):
    runner, api, ledger = fixture
    row = runner.start_or_resume()
    runner.config["max_runtime_seconds"] = 180
    runner.clock = lambda: NOW + timedelta(seconds=1200)
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "cancel_pending" and not api.executions
    assert ledger.get(DAY)["cleanup_required"] is True and len(api.payloads) == 1


def test_sdk_shaped_resolved_functions_include_defer_loading(fixture):
    runner, api, _ = fixture
    row = runner.start_or_resume()
    session = api.get("session", row["session_id"])
    check_agent(session["agent"], search.PROFILE)
    session["agent"]["tools"][0]["defer_loading"] = True
    with pytest.raises(Refusal, match="agent_search_profile_mismatch"):
        check_agent(session["agent"], search.PROFILE)


def test_profile_requires_adaptive_config_and_disabled_example():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "tools/daily_research/perplexity-daily.config.example.json").read_text())
    assert configuration(cfg)["enabled"] is False
    assert cfg["soft_target_usd"] is None
    with pytest.raises(Refusal, match="recurring_research_budget_not_approved"):
        configuration({**cfg, "enabled": True})
    with pytest.raises(Refusal, match="recurring_research_budget_not_approved"):
        configuration({**cfg, "enabled": True, "soft_target_usd": 2,
                       "recurring_budget_authority_reference": " PENDING-owner"})
    with pytest.raises(Refusal, match="search_profile_invalid"):
        configuration({**cfg, "discovery_profile": None})


def test_qa_text_uses_same_selected_tools_and_has_no_contact_transmission(fixture):
    runner, api, _ = fixture
    api.turn_status = "completed"
    row = runner.start_or_resume()
    text = qa_text(row, row["crm_snapshot"], "digest")
    assert "blueprint_search" in text and "blueprint_read_source" in text
    assert "Native web search only." not in text
    session = api.get("session", row["session_id"])
    Consumer.check_session(row, session)


def test_preflight_only_reads_existing_resources_and_checks_binding(fixture):
    _, api, _ = fixture
    assert preflight(api, search_provider=search.PROFILE)["inference_started"] is False
    assert not api.payloads and not api.executions
    assert [x[0] for x in api.calls] == ["GET", "GET"]


def test_fenced_provider_blocks_disable_or_profile_change_without_http(fixture):
    runner, _, ledger = fixture
    row = runner.start_or_resume()
    class Bridge:
        def call(self, op):
            return {"enabled": False, "config": {"search_provider": search.PROFILE}} if op == "control" else True
    provider = FencedProvider.__new__(FencedProvider)
    provider.ledger = type("Ledger", (), {"bridge": Bridge()})()
    with pytest.raises(Refusal, match="disabled_or_profile_changed"):
        provider.tool_admit(row, "research")
    assert ledger.get(DAY)["state"] == "running"


def test_actual_consumer_serves_exact_qa_call_and_exports_full_file_evidence(fixture, tmp_path):
    runner, api, ledger = fixture
    api.turn_status = "completed"
    row = runner.start_or_resume()
    row["qa"] = {"turn_id": "turn_qa", "baseline_turn_ids": ["turn_1"], "state": "qa_running", "cancel_attempted": False}
    api.actions = [{"type": "function_call", "turn_id": "turn_qa", "call_id": "call_qa",
                    "name": search.SEARCH, "arguments": {"query": "verify exact source and contradiction"}}]
    original_listing = api.listing
    def listing(resource, sid=None):
        if resource == "turns":
            return [{"id": "turn_1", "status": "completed"}, {"id": "turn_qa", "status": "in_progress",
                    "session_id": row["session_id"], "agent_id": AGENT, "subagent_id": None}]
        return original_listing(resource, sid)
    api.listing = listing
    class Bridge:
        def call(self, op, **kwargs):
            if op == "control":
                return {"enabled": True, "workflow": {"enabled": True, "qa_authority_reference": "owner-qa",
                        "publication_authority_reference": "owner-publish"}}
            if op == "snapshot":
                files = {}
                for path in ledger.root.glob(DAY + "-*.json"):
                    kind = path.name[len(DAY) + 1:-5]
                    if kind in {"status"}:
                        continue
                    files[kind] = base64.b64encode(path.read_bytes()).decode()
                return {"row": ledger.get(DAY), "files": files, "missing_files": ["qa", "qa-evidence"]}
            raise AssertionError(op)
    bridge = Bridge()
    ledger.bridge = bridge
    consumer = Consumer(ledger, runner.config, api, clock=lambda: NOW)
    assert consumer.observe(row, NOW + timedelta(seconds=1800)) is None
    saved = ledger.get(DAY)["application_tool_calls"]["call_qa"]
    assert saved["phase"] == "qa" and saved["success"] is True
    exported = render.export_snapshot(bridge, DAY, tmp_path / "export")
    assert exported["missing_files"] == ["qa", "qa-evidence"]
    assert (tmp_path / "export" / saved["result_file"]).read_bytes() == ledger.read_bytes(saved["result_file"])
    # A foreign turn, even with an untouched file SHA, is not valid exported evidence.
    changed = ledger.get(DAY)
    changed["qa"]["turn_id"] = "different"
    ledger.put(changed)
    with pytest.raises(Refusal, match="research_tool_result_digest_mismatch"):
        render.export_snapshot(bridge, DAY, tmp_path / "refused")


def test_packet_ceiling_retains_full_raw_artifact_and_never_admits_qa_or_publication(fixture, monkeypatch):
    runner, api, ledger = fixture
    api.turn_status = "completed"
    monkeypatch.setattr(search, "MAX_PACKET", 100)
    row = runner.start_or_resume()
    assert row["state"] == "failed" and row["error"] == "research_profile_packet_resource_ceiling_raw_retained"
    assert ledger.read_bytes(DAY + "-artifact.json") == api.raw
    assert not row.get("qa") and not row.get("delivery") and "packet" not in row


def test_all_future_lifecycle_rows_refuse_oversize_before_durable_state_changes(fixture, monkeypatch):
    runner, _, ledger = fixture
    old = runner.start_or_resume()
    changed = deepcopy(old)
    changed["delivery"] = {"synthetic_large_plan": "x" * 10000}
    monkeypatch.setattr(search, "MAX_RECORD", len(canonical(old).encode()) + 100)
    with pytest.raises(Refusal, match="research_tool_record_resource_ceiling"):
        ledger.put(changed)
    assert ledger.get(DAY) == old


@pytest.mark.parametrize("count", [10, 15, 50])
def test_reaching_ten_neither_finishes_the_turn_nor_discards_later_defensible_rows(fixture, count):
    runner, api, _ = fixture
    output = result(count)[0]
    output["coverage"].update(shortfall_reason=None, defined_run_scope=["Bounded synthetic task/industry/region scope"],
                              unresolved_promising_branches=["Promising operator branch still unexamined"],
                              completion_state="time_interrupted", stop_reason="Admitted run time exhausted; coverage remains incomplete")
    api.raw = canonical(output).encode()
    row = runner.start_or_resume()
    assert row["state"] == "running" and not api.cancellations
    api.turn_status = "completed"
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and len(row["packet"]["candidates"]) == count
    assert row["packet"]["coverage"]["completion_state"] == "time_interrupted"
    assert row["packet"]["coverage"]["unresolved_promising_branches"]
    assert row["packet"]["discovery_counts"]["candidate_count_is_stopping_rule"] is False
    assert row["packet"]["discovery_counts"]["target_new"] is None


def test_defined_coverage_can_return_fewer_than_ten_without_count_shortfall(fixture):
    runner, api, _ = fixture
    output = result(2)[0]
    output["coverage"].update(shortfall_reason=None, defined_run_scope=["Specific narrow industry/location hypothesis"],
                              unresolved_promising_branches=[], completion_state="coverage_complete")
    api.raw, api.turn_status = canonical(output).encode(), "completed"
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review" and len(row["packet"]["candidates"]) == 2


def test_unresolved_in_scope_branch_cannot_claim_coverage_complete():
    coverage = result()[0]["coverage"]
    coverage.update(defined_run_scope=["Bounded task/industry/region hypotheses"], completion_state="coverage_complete",
                    unresolved_promising_branches=["Promising in-scope branch not yet examined"])
    with pytest.raises(ValueError, match="discovery_completion_has_unresolved_branches"):
        discovery.validate_coverage(coverage, 50)


def test_approved_recurring_target_is_pinned_not_forced_to_legacy_one_dollar(fixture):
    runner, api, _ = fixture
    runner.config["soft_target_usd"] = 2
    row = runner.start_or_resume()
    assert row["soft_target_usd"] == 2 and "$2" in api.payloads[0]["input"]
    runner.config["soft_target_usd"] = 4
    assert runner.start_or_resume(allow_create=False)["soft_target_usd"] == 2


def test_missing_profile_budget_pin_is_not_inferred_from_controller(fixture):
    runner, _, _ = fixture
    row = runner.start_or_resume()
    provider = FencedProvider.__new__(FencedProvider)
    row.pop("recurring_budget_authority_reference")
    with pytest.raises(Refusal, match="research_tool_budget_authority_not_pinned"):
        provider.tool_admit(row, "research")
