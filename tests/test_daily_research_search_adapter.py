"""Offline Perplexity/public-source contracts and durable application-tool receipts."""
import gzip
import hashlib
import json
import signal
import socket
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tools.daily_research import search
from tools.daily_research.firestore import FencedProvider
from tools.daily_research.runner import PROJECT, Ledger, Provider, Refusal, canonical, digest

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
DAY = "2026-10-01"


@pytest.fixture(autouse=True)
def no_live_transport(monkeypatch):
    """No test can accidentally use an inherited real key or network connection."""
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("live network is forbidden in adapter tests")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def provider_response():
    return {
        "id": "search_offline_1",
        "results": [{
            "title": "Primary operator page",
            "url": "https://operator.example/workcell",
            "snippet": "Full first passage.\nFull decisive final passage.",
            "date": "2026-09-22",
            "last_updated": "2026-09-30",
            "publisher": "Operator",
            "provider_extra": {"preserved": [1, 2, 3]},
        }],
        "usage": {"search_context_size": "high", "cost": {"total_cost": 0.001}},
        "provider_metadata": {"unrecognized_future_field": "retained"},
    }


def install_http(monkeypatch, responses):
    calls = []
    responses = iter(responses)

    def fake_request(host, path, **kwargs):
        calls.append((host, path, kwargs))
        return next(responses)

    monkeypatch.setattr(search, "request", fake_request)
    return calls


def test_search_uses_official_fast_post_schema_and_retains_every_response_field(monkeypatch):
    response = provider_response()
    raw = canonical(response).encode()
    calls = install_http(monkeypatch, [(200, {"content-type": "application/json"}, raw)])
    monkeypatch.setenv("PERPLEXITY_API_KEY", "synthetic-offline-placeholder")
    arguments = {"query": "operator primary sources", "max_results": 20,
                 "search_domain_filter": ["operator.example", "-vendor.example"],
                 "search_recency_filter": "month"}
    result = search.ApplicationTools()(search.SEARCH, arguments)
    assert len(calls) == 1
    host, path, kwargs = calls[0]
    assert (host, path, kwargs["method"]) == ("api.perplexity.ai", "/search", "POST")
    expected = {**arguments, "search_type": "fast", "search_context_size": "high"}
    assert json.loads(kwargs["body"]) == expected
    assert kwargs["headers"] == {"Authorization": "Bearer synthetic-offline-placeholder",
                                 "Content-Type": "application/json"}
    assert result["request"] == expected and result["response"] == response
    assert result["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["truncated"] is False
    assert result["search_type"] == "fast" and result["provider"] == "perplexity"
    assert result["billing_receipt_verified"] is False
    assert "not_full_primary_page" in result["evidence_scope"]
    assert datetime.fromisoformat(result["checked_at"]).tzinfo is not None
    assert "synthetic-offline-placeholder" not in canonical(result)


def test_search_defaults_are_fast_high_and_have_no_model_substitution():
    assert search.search_body({"query": "primary evidence"}) == {
        "query": "primary evidence", "max_results": 10,
        "search_type": "fast", "search_context_size": "high"}
    declarations = search.tools()
    assert {tool["name"] for tool in declarations} == {search.SEARCH, search.READ}
    assert all(tool["type"] == "function" for tool in declarations)
    assert all(tool["parameters"]["additionalProperties"] is False for tool in declarations)
    assert "No native-search fallback" in search.instructions()


@pytest.mark.parametrize("arguments", [
    None, [], {}, {"query": ""}, {"query": " "}, {"query": "x" * 2001},
    {"query": " " * 2000 + "x"},
    {"query": 4}, {"query": "x", "max_results": True},
    {"query": "x", "max_results": 0}, {"query": "x", "max_results": 21},
    {"query": "x", "max_results": 1.5}, {"query": "x", "model": "sonar"},
    {"query": "x", "search_type": "pro"}, {"query": "x", "search_context_size": "low"},
    {"query": "x", "search_recency_filter": "minute"},
    {"query": "x", "search_recency_filter": []},
    {"query": "x", "search_recency_filter": {}},
    {"query": "x", "search_domain_filter": "example.com"},
    {"query": "x", "search_domain_filter": ["https://example.com"]},
    {"query": "x", "search_domain_filter": ["127.0.0.1"]},
    {"query": "x", "search_domain_filter": ["example.com"] * 21},
])
def test_search_rejects_unadmitted_arguments_before_any_paid_post(arguments, monkeypatch):
    calls = install_http(monkeypatch, [])
    monkeypatch.setenv("PERPLEXITY_API_KEY", "synthetic-offline-placeholder")
    with pytest.raises(search.ToolFailure, match="^search_arguments_invalid$"):
        search.ApplicationTools()(search.SEARCH, arguments)
    assert calls == []


def test_missing_secret_binding_and_unknown_tool_do_not_make_requests(monkeypatch):
    calls = install_http(monkeypatch, [])
    with pytest.raises(search.ToolFailure, match="^perplexity_binding_missing$"):
        search.ApplicationTools()(search.SEARCH, {"query": "evidence"})
    with pytest.raises(search.ToolFailure, match="^research_tool_not_allowed$"):
        search.ApplicationTools()("arbitrary_external_write", {})
    assert calls == []


@pytest.mark.parametrize("raw", [
    b"not JSON", b"[]", b'{"results":[]}', b'{"id":"x","results":{}}',
    b'{"id":"x","results":[{"title":"x","url":"https://example.com"}]}',
    b'{"id":"x","results":[{"title":"x","url":"https://example.com","snippet":3}]}',
])
def test_search_invalid_response_has_stable_secret_free_refusal(monkeypatch, raw):
    calls = install_http(monkeypatch, [(200, {}, raw)])
    monkeypatch.setenv("PERPLEXITY_API_KEY", "synthetic-offline-placeholder")
    with pytest.raises(search.ToolFailure, match="^perplexity_response_invalid$"):
        search.ApplicationTools()(search.SEARCH, {"query": "evidence"})
    assert len(calls) == 1


def test_provider_failure_never_reflects_upstream_body_or_retries(monkeypatch):
    calls = install_http(monkeypatch, [(503, {}, b"secret-bearing upstream prose")])
    monkeypatch.setenv("PERPLEXITY_API_KEY", "synthetic-offline-placeholder")
    with pytest.raises(search.ToolFailure, match="^perplexity_http_failure$"):
        search.ApplicationTools()(search.SEARCH, {"query": "evidence"})
    assert len(calls) == 1


def dns_rows(*addresses):
    return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM,
             6, "", (ip, 443)) for ip in addresses]


@pytest.mark.parametrize("ips", [
    ["127.0.0.1"], ["10.0.0.1"], ["192.168.2.1"], ["169.254.169.254"],
    ["::1"], ["fc00::1"], ["100.64.0.1"], ["93.184.216.34", "10.0.0.1"], [],
])
def test_dns_private_reserved_empty_and_mixed_answers_fail_closed(monkeypatch, ips):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: dns_rows(*ips))
    with pytest.raises(search.ToolFailure, match="^source_destination_not_public$"):
        search.addresses("operator.example")


def test_dns_failure_is_typed_and_public_addresses_are_sorted_deduplicated(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs:
                        dns_rows("93.184.216.35", "93.184.216.34", "93.184.216.34"))
    assert search.addresses("operator.example") == ["93.184.216.34", "93.184.216.35"]

    def unavailable(*args, **kwargs):
        raise OSError("upstream resolver private details")

    monkeypatch.setattr(socket, "getaddrinfo", unavailable)
    with pytest.raises(search.ToolFailure, match="^source_dns_unavailable$"):
        search.addresses("operator.example")


@pytest.mark.parametrize("url", [
    "http://operator.example/", "ftp://operator.example/", "https://127.0.0.1/",
    "https://[::1]/", "https://localhost/", "https://operator.example:8443/",
    "https://user:pass@operator.example/", "https://user@operator.example/",
    "https://operator.example:bad/", "https://operator.example:99999/",
])
def test_source_requires_public_https_domain_without_credentials_or_custom_port(monkeypatch, url):
    calls = install_http(monkeypatch, [])
    with pytest.raises(search.ToolFailure, match="^source_destination_invalid$"):
        search.source({"url": url})
    assert calls == []


def test_source_preserves_complete_visible_evidence_metadata_links_and_byte_digest(monkeypatch):
    raw = (b'<html><head><meta property="article:published_time" content="2026-09-01">'
           b'<style>hidden style</style></head><body><h1>Full source title</h1>'
           b'<p>First decisive passage &amp; context.</p><script>hidden secret script</script>'
           b'<a href="/details">Final decisive passage.</a></body></html>')
    calls = install_http(monkeypatch, [(200, {"Content-Type": "text/html; charset=utf-8",
        "Last-Modified": "Wed, 30 Sep 2026 00:00:00 GMT"}, raw)])
    result = search.ApplicationTools()(search.READ, {"url": "https://operator.example/workcell?q=evidence"})
    assert calls == [("operator.example", "/workcell?q=evidence", {
        "headers": {"User-Agent": "BlueprintResearch/1.0", "Accept-Encoding": "identity"}})]
    assert result["text"] == "Full source title\nFirst decisive passage & context.\nFinal decisive passage."
    assert result["metadata"] == [{"property": "article:published_time", "content": "2026-09-01"}]
    assert result["links"] == ["/details"] and result["truncated"] is False
    assert result["requested_url"] == result["url"] == "https://operator.example/workcell?q=evidence"
    assert result["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["last_modified"] == "Wed, 30 Sep 2026 00:00:00 GMT"
    assert "not_javascript_rendered" in result["evidence_scope"]


@pytest.mark.parametrize("location", [
    "http://operator.example/private", "https://127.0.0.1/", "https://user:pass@operator.example/",
])
def test_redirect_revalidates_destination_before_request(monkeypatch, location):
    calls = install_http(monkeypatch, [(302, {"Location": location}, b"")])
    with pytest.raises(search.ToolFailure, match="^source_destination_invalid$"):
        search.source({"url": "https://operator.example/source"})
    assert len(calls) == 1


def test_redirect_destination_dns_is_checked_and_private_target_never_connects(monkeypatch):
    dns, connections = [], []

    def resolver(host, *args, **kwargs):
        dns.append(host)
        return dns_rows("93.184.216.34" if host == "operator.example" else "10.0.0.2")

    class Connection:
        def __init__(self, host, address):
            connections.append((host, address))

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return SimpleNamespace(status=302, getheaders=lambda: [("Location", "https://internal.example/")],
                                   read=lambda limit: b"")

        def close(self):
            pass

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setattr(search, "PinnedHTTPS", Connection)
    with pytest.raises(search.ToolFailure, match="^source_destination_not_public$"):
        search.source({"url": "https://operator.example/source"})
    assert dns == ["operator.example", "internal.example"]
    assert connections == [("operator.example", "93.184.216.34")]


def test_a_caller_can_block_domains_before_any_request_and_on_every_redirect(monkeypatch):
    # The site screen verifier blocks LinkedIn's domains; the agent's own tool call blocks none.
    blocked = ("linkedin.com", "lnkd.in")
    calls = install_http(monkeypatch, [(302, {"Location": "https://www.linkedin.com/in/synthetic-profile"}, b"")])
    with pytest.raises(search.ToolFailure, match="^source_destination_not_allowed$"):
        search.source({"url": "https://operator.example/team"}, blocked_domains=blocked)
    for url in ("https://linkedin.com/company/synthetic", "https://WWW.LinkedIn.com/in/synthetic", "https://lnkd.in/x"):
        with pytest.raises(search.ToolFailure, match="^source_destination_not_allowed$"):
            search.source({"url": url}, blocked_domains=blocked)
    assert [call[0] for call in calls] == ["operator.example"]
    # A wrapper that holds a blocked URL is refused too, on a redirect as on a first request.
    wrapped = "https://archive.example/web/2025/https://www.linkedin.com/in/synthetic-profile"
    calls = install_http(monkeypatch, [(302, {"Location": wrapped}, b"")])
    with pytest.raises(search.ToolFailure, match="^source_destination_not_allowed$"):
        search.source({"url": "https://operator.example/team"}, blocked_domains=blocked)
    for url in (wrapped, "https://translate.example/?u=https%3A%2F%2Flnkd.in%2Fx", "https://linkedin.com.example/"):
        with pytest.raises(search.ToolFailure, match="^source_destination_not_allowed$"):
            search.source({"url": url}, blocked_domains=blocked)
    assert [call[0] for call in calls] == ["operator.example"]
    # Without blocked domains, the agent's own reads are unchanged.
    install_http(monkeypatch, [(200, {"Content-Type": "text/plain"}, b"Synthetic page.")])
    assert search.source({"url": "https://linkedin.com.example/"})["text"] == "Synthetic page."


def test_https_connect_pins_validated_ip_and_keeps_original_tls_hostname(monkeypatch):
    calls, sock = [], object()
    monkeypatch.setattr(socket, "create_connection", lambda endpoint, timeout:
                        (calls.append((endpoint, timeout)) or sock))
    connection = search.PinnedHTTPS("operator.example", "93.184.216.34")
    wrapped = object()
    connection._context = SimpleNamespace(wrap_socket=lambda value, **kwargs:
        (calls.append((value, kwargs)) or wrapped))
    connection.connect()
    assert calls == [(("93.184.216.34", 443), 15), (sock, {"server_hostname": "operator.example"})]
    assert connection.sock is wrapped and connection.host == "operator.example"


def test_request_byte_ceiling_refuses_without_truncation_and_closes_connection(monkeypatch):
    closed, reads = [], []

    class Connection:
        def __init__(self, host, address):
            assert (host, address) == ("operator.example", "93.184.216.34")

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return SimpleNamespace(status=200, getheaders=list, read=lambda limit:
                (reads.append(limit) or b"x" * (search.MAX_RESPONSE + 1)))

        def close(self):
            closed.append(True)

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: dns_rows("93.184.216.34"))
    monkeypatch.setattr(search, "PinnedHTTPS", Connection)
    with pytest.raises(search.ToolFailure, match="^source_response_too_large_no_truncation$"):
        search.request("operator.example", "/")
    assert reads == [search.MAX_RESPONSE + 1] and closed == [True]


def test_gzip_inflation_ceiling_is_explicit_source_gap(monkeypatch):
    raw = gzip.compress(b"x" * (search.MAX_RESPONSE + 1))
    install_http(monkeypatch, [(200, {"Content-Type": "text/plain", "Content-Encoding": "gzip"}, raw)])
    with pytest.raises(search.ToolFailure, match="^source_response_too_large_no_truncation$"):
        search.source({"url": "https://operator.example/"})


@pytest.mark.parametrize("raw,content_type", [
    (b" \n\t", "text/plain"),
    (b"<html><script>rendered-only-evidence</script><style>hidden</style></html>", "text/html"),
])
def test_empty_static_source_is_explicit_gap_instead_of_claiming_page_read(monkeypatch, raw, content_type):
    install_http(monkeypatch, [(200, {"Content-Type": content_type}, raw)])
    with pytest.raises(search.ToolFailure, match="^source_static_text_missing$"):
        search.source({"url": "https://operator.example/"})


def row():
    return {"date": DAY, "started_at": NOW.isoformat(), "search_provider": search.PROFILE,
            "soft_target_usd": 1, "recurring_budget_authority_reference": "owner-synthetic-approved-budget",
            "run_key": "blueprint-researcher:" + DAY, "session_id": "sess_offline",
            "turn_id": "turn_research", "qa": {"turn_id": "turn_qa"},
            "research_runtime_seconds": 60, "total_runtime_seconds": 90}


def action(*, cid="call_1", tid="turn_research", name=search.SEARCH, arguments=None):
    return {"type": "function_call", "turn_id": tid, "call_id": cid, "name": name,
            "arguments": arguments if arguments is not None else {"query": "operator evidence"}}


class MemoryLedger:
    def __init__(self):
        self.rows = []
        self.files = {}
        self.writes = []

    def put(self, value):
        self.rows.append(deepcopy(value))

    def write_bytes(self, name, value):
        assert name not in self.files, "tool-result files must be immutable"
        self.files[name] = bytes(value)
        self.writes.append(name)

    def read_bytes(self, name):
        if name not in self.files:
            raise FileNotFoundError(name)
        return self.files[name]


def saved_event(value, ledger, cid="call_1"):
    return json.loads(ledger.read_bytes(value["application_tool_calls"][cid]["result_file"]))


class ToolAPI:
    def __init__(self, ledger, result=None):
        self.ledger = ledger
        self.result = result if result is not None else {"response": provider_response(), "truncated": False}
        self.executions, self.replies, self.admissions = [], [], []
        self.lose_ack = False
        self.failure = None

    def tool_admit(self, value, phase):
        self.admissions.append(phase)
        assert value["search_provider"] == search.PROFILE

    def application_tool(self, name, arguments):
        saved = self.ledger.rows[-1] if isinstance(self.ledger, MemoryLedger) else self.ledger.get(DAY)
        durable = saved["application_tool_calls"]
        assert any(call["attempted"] is True for call in durable.values())
        self.executions.append((name, deepcopy(arguments)))
        if self.failure:
            raise self.failure
        return deepcopy(self.result)

    def tool_result(self, sid, event, key):
        saved = self.ledger.rows[-1] if isinstance(self.ledger, MemoryLedger) else self.ledger.get(DAY)
        assert saved_event(saved, self.ledger, event["call_id"]) == event
        self.replies.append((sid, deepcopy(event), key))
        if self.lose_ack:
            self.lose_ack = False
            raise TimeoutError("synthetic lost tool-result acknowledgement")


def respond(value, pending, ledger, api, *, phase="research", clock=lambda: NOW, stopped=lambda: False):
    return search.respond(value, {"required_actions": pending}, ledger, api,
                          phase=phase, clock=clock, stopped=stopped)


def test_pending_call_repoll_has_one_paid_execution_and_exact_durable_result():
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    assert respond(value, [action()], ledger, api) is True
    assert respond(value, [action()], ledger, api) is True
    assert len(api.executions) == 1 and len(api.replies) == 2
    assert api.replies[0] == api.replies[1]
    event = saved_event(value, ledger)
    assert event == {"type": "agent.session.input.tool_result", "turn_id": "turn_research",
                     "call_id": "call_1", "success": True, "output": canonical(api.result)}
    assert value["application_tool_calls"]["call_1"]["result_digest"] == digest(event)
    receipt = value["application_tool_calls"]["call_1"]
    raw = ledger.read_bytes(receipt["result_file"])
    assert receipt["result_file"] == DAY + "-tool-call_1.json"
    assert receipt["result_bytes"] == len(raw)
    assert receipt["result_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "event" not in receipt and "Full decisive final passage" not in canonical(value)
    assert ledger.writes == [receipt["result_file"]]
    assert api.replies[0][2] == "blueprint-researcher:2026-10-01:tool:call_1"
    assert ledger.rows[0]["application_tool_calls"]["call_1"]["attempted"] is False
    assert ledger.rows[1]["application_tool_calls"]["call_1"]["attempted"] is True


def test_application_http_post_is_not_replayed_after_lost_tool_result_ack(monkeypatch):
    calls = install_http(monkeypatch, [(200, {}, canonical(provider_response()).encode())])
    monkeypatch.setenv("PERPLEXITY_API_KEY", "synthetic-offline-placeholder")
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    api.application_tool = search.ApplicationTools()
    api.lose_ack = True
    with pytest.raises(TimeoutError):
        respond(value, [action()], ledger, api)
    respond(deepcopy(ledger.rows[-1]), [action()], ledger, api)
    assert len(calls) == 1 and calls[0][2]["method"] == "POST"
    assert len(api.replies) == 2 and api.replies[0] == api.replies[1]
    assert json.loads(api.replies[0][1]["output"])["response"] == provider_response()


def test_lost_tool_result_ack_resumes_from_disk_with_same_event_and_idempotency(tmp_path):
    value = row()
    ledger = Ledger(tmp_path / "receipt")
    api = ToolAPI(ledger)
    api.lose_ack = True
    with pytest.raises(TimeoutError):
        respond(value, [action()], ledger, api)
    ledger.db.close()
    resumed = Ledger(tmp_path / "receipt")
    api.ledger = resumed
    try:
        value = resumed.get(DAY)
        assert "result_acknowledged" not in value["application_tool_calls"]["call_1"]
        assert respond(value, [action()], resumed, api) is True
        assert len(api.executions) == 1 and api.replies[0] == api.replies[1]
        assert resumed.get(DAY)["application_tool_calls"]["call_1"]["result_acknowledged"] is True
    finally:
        resumed.db.close()


def test_attempt_receipt_without_result_never_replays_paid_post():
    value, ledger = row(), MemoryLedger()
    pending = action()
    binding = {field: pending[field] for field in ("turn_id", "call_id", "name", "arguments")}
    value["application_tool_calls"] = {"call_1": {
        "request_digest": digest(binding), "request": binding, "phase": "research", "attempted": True}}
    api = ToolAPI(ledger)
    assert respond(value, [pending], ledger, api) is True
    assert api.executions == []
    assert api.replies[0][1]["success"] is False
    assert api.replies[0][1]["error"] == "research_tool_reply_unresolved_no_replay"


@pytest.mark.parametrize("outcome", [
    {"success": True, "output": canonical(provider_response())},
    {"success": False, "error": "source_http_failure"},
])
def test_immutable_result_file_recovers_after_crash_before_row_pointer_without_paid_replay(outcome):
    value, ledger = row(), MemoryLedger()
    pending = action()
    binding = {field: pending[field] for field in ("turn_id", "call_id", "name", "arguments")}
    value["application_tool_calls"] = {"call_1": {
        "request_digest": digest(binding), "request": binding, "phase": "research", "attempted": True}}
    ledger.put(value)
    event = {"type": "agent.session.input.tool_result", "turn_id": "turn_research", "call_id": "call_1",
             **outcome}
    raw = (canonical(event) + "\n").encode()
    filename = DAY + "-tool-call_1.json"
    ledger.write_bytes(filename, raw)
    api = ToolAPI(ledger)
    respond(value, [pending], ledger, api)
    assert api.executions == [] and api.replies[0][1] == event
    assert ledger.writes == [filename] and ledger.read_bytes(filename) == raw
    receipt = value["application_tool_calls"]["call_1"]
    assert receipt["result_file"] == filename and receipt["result_bytes"] == len(raw)
    assert receipt["result_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["result_digest"] == digest(event) and receipt["success"] is outcome["success"]


@pytest.mark.parametrize("changes", [
    {"name": "arbitrary_external_write"}, {"turn_id": "turn_foreign"},
    {"turn_id": None}, {"type": "computer_use_approval_request"},
])
def test_required_action_binds_exact_turn_type_and_allowed_name(changes):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    with pytest.raises(Refusal, match="^research_tool_action_binding_invalid$"):
        respond(value, [{**action(), **changes}], ledger, api)
    assert api.executions == api.replies == [] and ledger.rows == []


@pytest.mark.parametrize("phase,tid", [("research", "turn_qa"), ("qa", "turn_research")])
def test_research_and_qa_cannot_execute_each_others_calls(phase, tid):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    with pytest.raises(Refusal, match="^research_tool_action_binding_invalid$"):
        respond(value, [action(tid=tid)], ledger, api, phase=phase)
    assert api.executions == api.replies == []


@pytest.mark.parametrize("changes", [
    {"arguments": {"query": "different evidence"}},
    {"name": search.READ, "arguments": {"url": "https://operator.example/"}},
])
def test_reused_call_id_cannot_change_name_or_arguments(changes):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    with pytest.raises(Refusal, match="^research_tool_call_identity_conflict$"):
        respond(value, [{**action(), **changes}], ledger, api)
    assert len(api.executions) == len(api.replies) == 1


@pytest.mark.parametrize("tamper", ["bytes", "digest", "binding"])
def test_durable_result_digest_tampering_refuses_without_execution_or_submission(tamper):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    receipt = value["application_tool_calls"]["call_1"]
    if tamper == "digest":
        receipt["result_digest"] = "0" * 64
    else:
        event = saved_event(value, ledger)
        if tamper == "bytes":
            event["output"] = "changed bytes"
        else:
            event["turn_id"] = "turn_foreign"
            receipt["result_digest"] = digest(event)
        raw = (canonical(event) + "\n").encode()
        ledger.files[receipt["result_file"]] = raw
        if tamper == "binding":
            receipt["result_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(Refusal, match="^research_tool_result_digest_mismatch$"):
        respond(value, [action()], ledger, api)
    assert len(api.executions) == len(api.replies) == 1


@pytest.mark.parametrize("phase,seconds", [("research", 60), ("qa", 90)])
def test_exact_absolute_deadline_prevents_execution_and_tool_result_submission(phase, seconds):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    pending = action(tid="turn_qa" if phase == "qa" else "turn_research")
    with pytest.raises(Refusal, match="^research_tool_stopped_or_expired$"):
        respond(value, [pending], ledger, api, phase=phase,
                clock=lambda: NOW + timedelta(seconds=seconds))
    assert api.executions == api.replies == api.admissions == []
    assert value["application_tool_calls"]["call_1"]["attempted"] is False


@pytest.mark.parametrize("phase,expired", [("research", 2700), ("qa", 3600)])
def test_sixty_minute_row_tools_run_past_the_old_envelope_until_its_own_deadline(phase, expired):
    value, ledger = {**row(), "research_runtime_seconds": 2700, "total_runtime_seconds": 3600}, MemoryLedger()
    api = ToolAPI(ledger)
    turn = "turn_qa" if phase == "qa" else "turn_research"
    # 1801 seconds is past the old 1800-second total and the old 1200-second research window.
    assert respond(value, [action(tid=turn)], ledger, api, phase=phase,
                   clock=lambda: NOW + timedelta(seconds=1801)) is True
    assert len(api.executions) == len(api.replies) == 1
    with pytest.raises(Refusal, match="^research_tool_stopped_or_expired$"):
        respond(value, [action(cid="call_2", tid=turn)], ledger, api, phase=phase,
                clock=lambda: NOW + timedelta(seconds=expired))
    assert len(api.executions) == len(api.replies) == 1
    assert value["application_tool_calls"]["call_2"]["attempted"] is False


def test_stopped_before_execution_and_stopped_after_result_both_fail_closed():
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    with pytest.raises(Refusal, match="^research_tool_stopped_or_expired$"):
        respond(value, [action()], ledger, api, stopped=lambda: True)
    assert api.executions == api.replies == []
    with pytest.raises(Refusal, match="^research_tool_stopped_or_expired$"):
        respond(value, [action()], ledger, api, stopped=lambda: bool(api.executions))
    assert len(api.executions) == 1 and api.replies == []
    assert "result_file" in value["application_tool_calls"]["call_1"]


@pytest.mark.parametrize("admission,change", [(1, "stop"), (1, "deadline"), (2, "stop"), (2, "deadline"), (3, "stop"), (3, "deadline")])
def test_stop_and_deadline_after_slow_admission_prevent_the_next_mutation(admission, change):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    state = {"stop": False, "now": NOW}
    original = api.tool_admit

    def admit(value, phase):
        original(value, phase)
        if len(api.admissions) == admission:
            state.update(stop=change == "stop", now=NOW + timedelta(seconds=60) if change == "deadline" else NOW)

    api.tool_admit = admit
    with pytest.raises(Refusal, match="^research_tool_stopped_or_expired$"):
        respond(value, [action()], ledger, api, clock=lambda: state["now"], stopped=lambda: state["stop"])
    assert len(api.executions) == (0 if admission in (1, 2) else 1)
    assert api.replies == []
    if admission == 3:
        assert saved_event(value, ledger)["success"] is True


def test_absolute_request_alarm_covers_blocked_dns_and_restores_signal_handler(monkeypatch):
    prior_handler = signal.getsignal(signal.SIGALRM)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    dns = []

    def blocked_dns(host, *args, **kwargs):
        dns.append(host)
        time.sleep(1)
        raise AssertionError("request alarm failed to interrupt DNS")

    monkeypatch.setattr(socket, "getaddrinfo", blocked_dns)
    # This test is about the request alarm; without this, the wrap-up reply would answer the call first.
    monkeypatch.setattr(search, "WRAP_UP_SECONDS", 0)
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    api.application_tool = search.ApplicationTools()
    near_deadline = NOW + timedelta(seconds=59.98)
    pending = action(name=search.READ, arguments={"url": "https://operator.example/source"})
    assert respond(value, [pending], ledger, api, clock=lambda: near_deadline)
    assert dns == ["operator.example"]
    assert json.loads(api.replies[0][1]["error"]) == {"code": "research_tool_absolute_deadline"}
    assert json.loads(api.replies[0][1]["output"]) == {"ok": False, "error": {"code": "research_tool_absolute_deadline"}}
    assert api.replies[0][1]["success"] is False
    assert signal.getsignal(signal.SIGALRM) == prior_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    respond(value, [pending], ledger, api, clock=lambda: near_deadline)
    assert dns == ["operator.example"]  # Even an absolute timeout is never re-executed.


def test_request_alarm_does_not_replace_an_existing_owner_watchdog(monkeypatch):
    mutations = []
    monkeypatch.setattr(signal, "getitimer", lambda kind: (5.0, 0.0))
    monkeypatch.setattr(signal, "signal", lambda *args: mutations.append(args))
    monkeypatch.setattr(signal, "setitimer", lambda *args: mutations.append(args))
    with pytest.raises(search.ToolFailure, match="^research_tool_watchdog_already_active$"), search.bounded_request(10):
        pytest.fail("an existing watchdog must block execution")
    assert mutations == []


def test_request_alarm_rejects_nonpositive_allowance_before_execution():
    with pytest.raises(search.ToolFailure, match="^research_tool_absolute_deadline$"), search.bounded_request(0):
        pytest.fail("an expired allowance must block execution")


@pytest.mark.parametrize("error", [
    "source_response_too_large_no_truncation", "source_format_unsupported", "source_destination_not_public",
    "source_http_failure", "source_request_unavailable",
])
def test_source_refusal_is_a_visible_failed_tool_result_without_replay(error):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    api.failure = search.ToolFailure(error)
    pending = action(name=search.READ, arguments={"url": "https://operator.example/source"})
    respond(value, [pending], ledger, api)
    respond(value, [pending], ledger, api)
    assert len(api.executions) == 1 and api.replies[0] == api.replies[1]
    event = api.replies[0][1]
    assert json.loads(event["error"]) == {"code": error} and event["success"] is False
    assert json.loads(event["output"]) == {"ok": False, "error": {"code": error}}


def test_oversized_tool_result_is_visible_refusal_instead_of_truncated_success(monkeypatch):
    monkeypatch.setattr(search, "MAX_RESPONSE", 50)
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger, result={"text": "full decisive evidence" * 10, "truncated": False})
    respond(value, [action()], ledger, api)
    event = api.replies[0][1]
    assert event["success"] is False
    assert json.loads(event["error"]) == {"code": "research_tool_result_too_large_no_truncation"}
    assert json.loads(event["output"]) == {"ok": False, "error": json.loads(event["error"])}
    assert "full decisive evidence" not in event["output"]


def test_upstream_execution_error_is_secret_free_and_never_replayed():
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    api.failure = RuntimeError("synthetic-private-credential-and-upstream-prose")
    respond(value, [action()], ledger, api)
    respond(value, [action()], ledger, api)
    assert len(api.executions) == 1
    assert api.replies[0][1]["error"] == "research_tool_unavailable_no_replay"
    assert "output" not in api.replies[0][1] and api.replies[0] == api.replies[1]
    assert "synthetic-private" not in canonical(value)


def budget_reply(reply):
    error = json.loads(reply["error"])
    return reply["success"] is False and error["code"] == "research_tool_budget_exhausted" and "final output" in error["guidance"]


def test_call_budget_fails_soft_reserves_qa_and_keeps_a_hard_stop(monkeypatch):
    monkeypatch.setattr(search, "MAX_CALLS", 3)
    monkeypatch.setattr(search, "QA_RESERVED_CALLS", 1)
    monkeypatch.setattr(search, "BUDGET_GRACE_CALLS", 2)
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    respond(value, [action(cid="call_2")], ledger, api)
    # Research may not use QA's reserved share: it is told to finish instead of being cancelled.
    respond(value, [action(cid="call_3")], ledger, api)
    assert len(api.executions) == 2 and budget_reply(api.replies[-1][1])
    # QA still has its reserve, then gets the same fail-soft reply at the shared ceiling.
    respond(value, [action(cid="call_4", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 3 and api.replies[-1][1]["success"] is True
    respond(value, [action(cid="call_5", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 3 and budget_reply(api.replies[-1][1])
    # A replay of a budget reply returns the same retained bytes and executes nothing.
    respond(value, [action(cid="call_5", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 3 and api.replies[-1] == api.replies[-2]
    # Each phase has its own grace replies; after them, the hard ceiling still refuses.
    respond(value, [action(cid="call_6", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 3 and budget_reply(api.replies[-1][1])
    with pytest.raises(Refusal, match="^research_tool_evidence_resource_ceiling$"):
        respond(value, [action(cid="call_7", tid="turn_qa")], ledger, api, phase="qa")
    assert {call["phase"] for call in value["application_tool_calls"].values()} == {"research", "qa"}
    assert value["application_tool_usage"]["attempted_search_requests"] == 3


def time_reply(reply):
    error = json.loads(reply["error"])
    return reply["success"] is False and error["code"] == "research_tool_time_nearly_up" and "final output" in error["guidance"]


def test_research_is_told_to_finish_before_its_time_runs_out(monkeypatch):
    """2026-10-06: research ran into the hard time guard mid-search and lost the day's output. Inside the last
    quarter of a short window (at most WRAP_UP_SECONDS), each new research call gets a recorded finish-now reply
    instead of executing; it never turns into the budget refusal, and QA keeps its own time."""
    monkeypatch.setattr(search, "BUDGET_GRACE_CALLS", 1)
    value, ledger = row(), MemoryLedger()  # A 60-second research window: its last 15 seconds are wrap-up time.
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api, clock=lambda: NOW + timedelta(seconds=44))
    assert len(api.executions) == 1 and api.replies[-1][1]["success"] is True
    def late():
        return NOW + timedelta(seconds=46)

    for number in range(2, 5):  # More than the grace limit: a time reply is never a refusal.
        respond(value, [action(cid=f"call_{number}")], ledger, api, clock=late)
        assert len(api.executions) == 1 and time_reply(api.replies[-1][1])
    respond(value, [action(cid="call_4")], ledger, api, clock=late)
    assert len(api.executions) == 1 and api.replies[-1] == api.replies[-2]
    respond(value, [action(cid="call_5", tid="turn_qa")], ledger, api, phase="qa", clock=late)
    assert len(api.executions) == 2 and api.replies[-1][1]["success"] is True


def test_research_grace_replies_do_not_use_up_qa_grace(monkeypatch):
    monkeypatch.setattr(search, "MAX_CALLS", 3)
    monkeypatch.setattr(search, "QA_RESERVED_CALLS", 1)
    monkeypatch.setattr(search, "BUDGET_GRACE_CALLS", 1)
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    respond(value, [action(cid="call_2")], ledger, api)
    respond(value, [action(cid="call_3")], ledger, api)
    assert budget_reply(api.replies[-1][1])
    with pytest.raises(Refusal, match="^research_tool_evidence_resource_ceiling$"):
        respond(value, [action(cid="call_4")], ledger, api)
    # Research used all of its grace replies. QA keeps its reserve and its own grace reply.
    respond(value, [action(cid="call_5", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 3 and api.replies[-1][1]["success"] is True
    respond(value, [action(cid="call_6", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 3 and budget_reply(api.replies[-1][1])


def repair_row():
    value = row()
    value.update(raw_output_digest="sha256:" + "a" * 64, validation_repairs=[{"turn_id": "turn_repair"}],
                 validation_repair_authority={"kind": "workflow", "duration_seconds": 90, "started_at": NOW.isoformat(),
                     "request": {"scope": "same-session-validation-repair-and-qa-no-outreach",
                                 "session_id": "sess_offline", "root_turn_id": "turn_research",
                                 "raw_output_sha256": "sha256:" + "a" * 64,
                                 "authority_reference": "owner-synthetic-workflow-authority"}})
    return value


def test_validation_repair_keeps_the_qa_call_reserve(monkeypatch):
    monkeypatch.setattr(search, "MAX_CALLS", 2)
    monkeypatch.setattr(search, "QA_RESERVED_CALLS", 1)
    value, ledger = repair_row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    # Repair runs before QA, so like research it may not use QA's reserved share.
    respond(value, [action(cid="call_2", tid="turn_repair")], ledger, api, phase="repair")
    assert len(api.executions) == 1 and budget_reply(api.replies[-1][1])
    respond(value, [action(cid="call_3", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 2 and api.replies[-1][1]["success"] is True


def test_byte_budget_fails_soft_and_includes_prior_research_during_qa(monkeypatch):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    used = sum(call["result_bytes"] for call in value["application_tool_calls"].values())
    monkeypatch.setattr(search, "MAX_EVIDENCE", used + search.MAX_RESPONSE + 20000)
    monkeypatch.setattr(search, "QA_RESERVED_EVIDENCE", 0)
    respond(value, [action(cid="call_2", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 1 and budget_reply(api.replies[-1][1])


def test_research_byte_budget_leaves_the_qa_evidence_reserve(monkeypatch):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    used = sum(call["result_bytes"] for call in value["application_tool_calls"].values())
    monkeypatch.setattr(search, "QA_RESERVED_EVIDENCE", 1000)
    monkeypatch.setattr(search, "MAX_EVIDENCE", used + search.MAX_RESPONSE + 20000 + 999)
    respond(value, [action(cid="call_2")], ledger, api)
    assert len(api.executions) == 1 and budget_reply(api.replies[-1][1])
    respond(value, [action(cid="call_3", tid="turn_qa")], ledger, api, phase="qa")
    assert len(api.executions) == 2 and api.replies[-1][1]["success"] is True


def test_research_last_result_cannot_shrink_the_qa_evidence_reserve(monkeypatch):
    # A recorded event escapes its JSON output again, so one result can reach about
    # twice MAX_RESPONSE. Research must stop early enough that QA keeps its share.
    monkeypatch.setattr(search, "MAX_RESPONSE", 100_000)
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    respond(value, [action()], ledger, api)
    used = sum(call["result_bytes"] for call in value["application_tool_calls"].values())
    monkeypatch.setattr(search, "QA_RESERVED_EVIDENCE", 200_000)
    monkeypatch.setattr(search, "MAX_EVIDENCE", used + 200_000 + 100_000 + 20_000 + 1)
    api.result = {"quoted": '"' * 49_000}
    assert len(canonical(api.result).encode()) <= search.MAX_RESPONSE
    respond(value, [action(cid="call_2")], ledger, api)
    used = sum(call["result_bytes"] for call in value["application_tool_calls"].values())
    assert search.MAX_EVIDENCE - used >= search.QA_RESERVED_EVIDENCE
    assert budget_reply(api.replies[-1][1])


def test_existing_record_ceiling_prevents_every_mutation(monkeypatch):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    monkeypatch.setattr(search, "MAX_RECORD", len(canonical(value).encode()) - 1)
    with pytest.raises(Refusal, match="^research_tool_record_resource_ceiling$"):
        respond(value, [action()], ledger, api)
    assert ledger.rows == api.executions == api.replies == []


def test_prospective_input_record_overflow_refuses_before_intent_or_execution(monkeypatch):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    monkeypatch.setattr(search, "MAX_RECORD", 1000)
    with pytest.raises(Refusal, match="resource_ceiling"):
        respond(value, [action(arguments={"query": "x" * 1800})], ledger, api)
    assert ledger.rows == api.executions == api.replies == []


def test_prospective_call_evidence_overflow_refuses_before_intent_or_execution(monkeypatch):
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    monkeypatch.setattr(search, "MAX_RESPONSE", 100)
    monkeypatch.setattr(search, "MAX_EVIDENCE", 20500)
    with pytest.raises(Refusal, match="resource_ceiling"):
        respond(value, [action(arguments={"query": "x" * 21000})], ledger, api)
    assert ledger.rows == api.executions == api.replies == []


def test_environment_connection_is_ignored_and_unselected_profile_executes_nothing():
    value, ledger = row(), MemoryLedger()
    api = ToolAPI(ledger)
    assert respond(value, [{"type": "environment_connection", "environment_id": "env_offline"}], ledger, api)
    value["search_provider"] = None
    assert respond(value, [action()], ledger, api) is False
    assert api.executions == api.replies == []


@pytest.mark.parametrize("enabled,profile,qa_enabled,phase", [
    (False, search.PROFILE, True, "research"), (True, None, True, "research"),
    (True, search.PROFILE, False, "qa"),
])
def test_fresh_firestore_control_fence_blocks_disabled_or_changed_search(enabled, profile, qa_enabled, phase):
    calls = []

    def bridge_call(op):
        calls.append(op)
        if op == "control":
            return {"enabled": enabled, "config": {"search_provider": profile},
                    "workflow": {"enabled": qa_enabled}}

    provider = FencedProvider.__new__(FencedProvider)
    provider.ledger = SimpleNamespace(bridge=SimpleNamespace(call=bridge_call))
    with pytest.raises(Refusal, match="^research_tool_disabled_or_profile_changed$"):
        provider.tool_admit(row(), phase)
    assert calls == ["assert_lease", "control"]


def _sdk_wire_probe():
    import httpx2 as httpx
    import openai

    assert openai.__version__ == "3.22.1", "wire proof requires the deployed SDK pin"
    calls = []

    def mock_http(request):
        calls.append(request)
        if request.url.path.endswith("/events"):
            return httpx.Response(202)
        return httpx.Response(200, json={"id": "sess_offline", "required_actions": [action()]})

    provider = Provider.__new__(Provider)
    provider.client = openai.OpenAI(api_key="synthetic-offline-placeholder", project=PROJECT, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(mock_http), follow_redirects=False, trust_env=False))
    provider.api = provider.client.beta.agents
    override = {"tools": search.tools(), "service_tier": "default", "instructions": search.instructions()}
    try:
        provider.create({"agent_id": "agent_offline", "agent": override,
                         "environment": {"type": "openai_hosted", "network": {"access": "disabled"}},
                         "input": "Offline wire proof", "stream": False})
        pending = provider.get("session", "sess_offline")["required_actions"]
        assert pending == [action()]
        events = [
            {"type": "agent.session.input.tool_result", "turn_id": "turn_research", "call_id": "call_1",
             "success": True, "output": canonical(provider_response())},
            {"type": "agent.session.input.tool_result", "turn_id": "turn_qa", "call_id": "call_2",
             "success": False, "error": "source_response_too_large_no_truncation"},
        ]
        for event in events:
            provider.tool_result("sess_offline", event, "exact-idempotency:" + event["call_id"])
        assert json.loads(calls[0].content)["agent"] == override
        assert all(request.headers["OpenAI-Project"] == PROJECT for request in calls)
        assert all(request.headers["OpenAI-Beta"] == "agents=v1" for request in calls)
        posted = [request for request in calls if request.url.path.endswith("/events")]
        assert len(posted) == 2
        for request, event in zip(posted, events):
            assert request.method == "POST"
            assert request.url.path == "/v1/agents/sessions/sess_offline/events"
            assert json.loads(request.content) == {"events": [event]}
            assert request.headers["Idempotency-Key"] == "exact-idempotency:" + event["call_id"]
    finally:
        provider.client.close()


def _sdk_transport_probe(monkeypatch, status):
    import httpx2 as httpx
    import openai

    assert openai.__version__ == "3.22.1", "wire proof requires the deployed SDK pin"
    requests = []

    def mock_http(request):
        requests.append(request)
        return httpx.Response(status, headers={"Location": "https://redirect.example/"},
                              json={"error": {"message": "offline failure"}})

    monkeypatch.setattr(openai, "DefaultHttpxClient", lambda **kwargs:
        httpx.Client(transport=httpx.MockTransport(mock_http), trust_env=False, **kwargs))
    provider = Provider("synthetic-offline-placeholder")
    try:
        assert provider.client.max_retries == 0
        assert provider.client._client.follow_redirects is False
        with pytest.raises(openai.APIStatusError):
            provider.tool_result("sess_offline", {"type": "agent.session.input.tool_result", "turn_id": "turn_research",
                "call_id": "call_1", "success": False, "error": "explicit-source-gap"}, "exact-idempotency:call_1")
        assert len(requests) == 1 and requests[0].url.host == "api.openai.com"
    finally:
        provider.client.close()


def _isolated_sdk_probe(name, status=None):
    import os
    import subprocess
    import sys
    from pathlib import Path

    runtime = os.environ.get("BLUEPRINT_RESEARCH_SDK_PYTHON", sys.executable)
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_") and key != "PERPLEXITY_API_KEY"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = ("import runpy,sys,pytest; ns=runpy.run_path(sys.argv[1]); "
            "ns['_sdk_wire_probe']()") if name == "wire" else (
            "import runpy,sys,pytest; ns=runpy.run_path(sys.argv[1]); "
            "m=pytest.MonkeyPatch(); "
            "ns['_sdk_transport_probe'](m,int(sys.argv[2])); m.undo()")
    args = [runtime, "-c", code, str(Path(__file__).resolve())]
    if status is not None:
        args.append(str(status))
    subprocess.run(args, cwd=Path(__file__).resolve().parents[1], env=env,
                   capture_output=True, text=True, timeout=30, check=True)


def test_deployed_sdk_wire_has_function_tool_override_and_exact_success_failure_events():
    # Main Pipeline and the isolated research package intentionally use distinct SDKs.
    _isolated_sdk_probe("wire")


@pytest.mark.parametrize("status", [307, 503])
def test_deployed_sdk_tool_result_does_not_retry_or_follow_redirects(status):
    _isolated_sdk_probe("transport", status)
