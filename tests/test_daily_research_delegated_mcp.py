"""Prospective delegated research: no live MCP, provider call or writes."""
from copy import deepcopy

import pytest

from tests import test_daily_research_search as search_helpers
from tools.daily_research import search
from tools.daily_research.consumer import Consumer
from tools.daily_research.runner import Refusal, configuration, digest, preflight


@pytest.fixture
def delegated_fixture(tmp_path):
    yield from search_helpers.fixture.__wrapped__(tmp_path)

def connection(label, catalog):
    return {"type": "mcp", "server_label": label,
            "transport": {"type": "http", "server_url": catalog[label][0], "headers": {}},
            "credential_id": "credential_synthetic_" + label, "allowed_tools": None,
            "connection_origin": "service", "required": False, "request_metadata": {}}


def test_new_profile_adds_paid_mcp_and_existing_reads_without_mutating_saved_agent(delegated_fixture):
    runner, api, ledger = delegated_fixture
    profile = search.MCP_RESEARCH_PROFILE
    catalog = search.MCP_PROFILES[profile]
    connections = [connection(label, catalog) for label in catalog]
    api.agent["tools"].extend(deepcopy(connections))
    saved = deepcopy(api.agent)
    runner.config.update(mcp_profile=profile, publication_profile="agent-owned-v1")
    assert configuration(runner.config)["mcp_profile"] == profile
    checked = preflight(api, search_provider=search.PROFILE, publication_profile="agent-owned-v1", mcp_profile=profile)
    assert checked["inference_started"] is False
    assert checked["native_research_cost_status"] == "unknown_not_metered_by_host"
    assert api.payloads == [] and api.executions == []
    row = runner.start_or_resume()
    assert api.agent == saved
    assert row["create_payload"]["agent"]["tools"] == search.tools("agent-owned-v1") + search.mcp_tools(connections, profile)
    assert row["metadata"]["mcp_binding_digest"] == digest(connections)
    assert row["create_payload"]["vault_ids"] == sorted("vault_synthetic_" + label for label in catalog)
    instructions = row["create_payload"]["agent"]["instructions"]
    assert "effort=ultra" in instructions and "runId" in instructions
    assert "not a hard cap" in instructions and "unknown, never zero" in instructions
    assert row["create_payload"]["environment"]["network"]["access"] == "disabled"
    Consumer.check_session(row, api.get("session", row["session_id"]))
    changed = deepcopy(row)
    changed["mcp_binding"][0]["credential_id"] = "credential_changed"
    with pytest.raises(Refusal, match="research_mcp_binding_changed"):
        Consumer.check_session(changed, api.get("session", row["session_id"]))
    assert ledger.get(row["date"])["create_payload"] == row["create_payload"]


@pytest.mark.parametrize("label", list(search.MCP_RESEARCH_TOOLS))
def test_old_readonly_profile_cannot_admit_new_paid_connections(label):
    tool = connection(label, search.MCP_RESEARCH_TOOLS)
    with pytest.raises(search.ToolFailure, match="research_mcp_configuration_invalid"):
        search.mcp_connections([tool])
    assert search.mcp_tools([tool], search.MCP_RESEARCH_PROFILE)[0]["allowed_tools"] == list(search.MCP_RESEARCH_TOOLS[label][1])


@pytest.mark.parametrize("change", ["endpoint", "credential", "headers", "duplicate"])
def test_new_profile_refuses_untrusted_connection_before_paid_intent(delegated_fixture, change):
    runner, api, ledger = delegated_fixture
    connections = [connection("exa", search.MCP_RESEARCH_TOOLS)]
    if change == "endpoint":
        connections[0]["transport"]["server_url"] += "?exaApiKey=must_not_enter_payload"
    elif change == "credential":
        connections[0]["credential_id"] = "inline-secret"
    elif change == "headers":
        connections[0]["transport"]["headers"] = {"Authorization": "inline-secret"}
    else:
        connections.append(deepcopy(connections[0]))
    api.agent["tools"].extend(connections)
    runner.config["mcp_profile"] = search.MCP_RESEARCH_PROFILE
    with pytest.raises(Refusal, match="research_mcp_configuration_invalid"):
        runner.start_or_resume()
    assert ledger.rows() == [] and api.payloads == []


def test_new_profile_intersects_owner_allowlist_without_exposing_mutation_tools():
    catalog = search.MCP_PROFILES[search.MCP_RESEARCH_PROFILE]
    tool = connection("blueprint", catalog)
    tool["allowed_tools"] = ["get_gemini_deep_research", "send_email", "start_gemini_deep_research"]
    assert search.mcp_tools([tool], search.MCP_RESEARCH_PROFILE)[0]["allowed_tools"] == [
        "start_gemini_deep_research", "get_gemini_deep_research"]
    readonly = [connection(label, search.MCP_READ_TOOLS) for label in search.MCP_READ_TOOLS]
    assert search.mcp_tools(readonly) == search.mcp_tools(readonly, search.MCP_RESEARCH_PROFILE)


@pytest.mark.parametrize("suffix", ["?login", "?login=", "?tools=agent_run", "?tools=web_search_exa,web_fetch_exa,web_search_advanced_exa,agent_run", "?login&tools=agent_run"])
def test_exa_documented_oauth_query_retains_full_url_and_vault_identity(delegated_fixture, suffix):
    runner, api, _ = delegated_fixture
    tool = connection("exa", search.MCP_RESEARCH_TOOLS)
    tool["transport"]["server_url"] += suffix
    original = deepcopy(tool)
    api.agent["tools"].append(tool)
    runner.config["mcp_profile"] = search.MCP_RESEARCH_PROFILE
    row = runner.start_or_resume()
    assert row["mcp_binding"] == [original]
    assert row["mcp_vault_binding"][0]["mcp_server_url"] == original["transport"]["server_url"]
    assert row["mcp_vault_binding"][0]["credential_auth"]["mcp_server_url"] == original["transport"]["server_url"]
    assert row["create_payload"]["agent"]["tools"][-1]["transport"]["server_url"] == original["transport"]["server_url"]
    Consumer.check_session(row, api.get("session", row["session_id"]))
    # A credential for the bare endpoint cannot silently attach to its OAuth URL.
    api.vault_credentials = {"vault_synthetic_exa": [{"id": tool["credential_id"], "vault_id": "vault_synthetic_exa",
        "auth": {"type": "mcp_oauth", "mcp_server_url": search.MCP_RESEARCH_TOOLS["exa"][0]}}]}
    with pytest.raises(Refusal, match="research_mcp_vault_binding_invalid"):
        preflight(api, search_provider=search.PROFILE, mcp_profile=search.MCP_RESEARCH_PROFILE)


@pytest.mark.parametrize("url", [
    "https://mcp.exa.ai/mcp?exaApiKey=inline_secret", "https://mcp.exa.ai/mcp?login&token=secret",
    "https://mcp.exa.ai/mcp?login&login", "https://mcp.exa.ai/mcp?tools=agent_run&tools=web_fetch_exa",
    "https://mcp.exa.ai/mcp?login=yes", "https://mcp.exa.ai/mcp?tools=agent_run%26exaApiKey%3Dsecret",
    "https://mcp.exa.ai/mcp?tools=agent_run,unknown_tool", "https://mcp.exa.ai/mcp?tools=",
    "https://mcp.exa.ai/mcp?login#", "https://user@mcp.exa.ai/mcp?login",
    "https://mcp.exa.ai.evil.example/mcp?login", "http://mcp.exa.ai/mcp?login",
    "https://mcp.exa.ai/other?login", "https://mcp.exa.ai/mcp?login\n",
])
def test_exa_query_rejects_secrets_duplicates_and_endpoint_injection(url):
    tool = connection("exa", search.MCP_RESEARCH_TOOLS)
    tool["transport"]["server_url"] = url
    with pytest.raises(search.ToolFailure, match="research_mcp_configuration_invalid"):
        search.mcp_connections([tool], search.MCP_RESEARCH_PROFILE)
