"""Hermetic agent-utility checks; no provider calls or spending."""

import copy
import http.client
import json
import socket
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import parallel_findall as findall
from blueprint_pipeline.safe_outbound_http import SafeHttpResponse

RUN_ID = "findall_fixture_123"
FAKE_KEY = "offline-test-key"
SPEC_PATH = Path(__file__).resolve().parents[1] / "docs/examples/parallel_findall_spec.json"


@pytest.fixture(autouse=True)
def no_network_or_real_credentials(monkeypatch):
    monkeypatch.delenv(findall.API_KEY_ENV, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("network forbidden in FindAll setup verification")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(findall.safe_outbound_http, "open_request", refuse)


@pytest.fixture
def spec():
    return json.loads(SPEC_PATH.read_text())


def run_status(active=False):
    return {
        "findall_id": RUN_ID,
        "status": {"status": "running" if active else "completed", "is_active": active},
        "generator": "base",
        "future_provider_field": {"preserved": True},
    }


def fake_response(monkeypatch, payload, status=200):
    requests = []

    def respond(request, **options):
        requests.append((request, options))
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return SafeHttpResponse(status, body, request.full_url, request.full_url)

    monkeypatch.setattr(findall.safe_outbound_http, "open_request", respond)
    return requests


def test_prepare_is_offline_detached_and_never_authorizes_creation(spec):
    envelope = findall.prepare_run(spec)
    assert envelope["url"] == "https://api.parallel.ai/v1beta/findall/runs"
    assert envelope["method"] == "POST"
    assert envelope["required_auth_header"] == "x-api-key"
    assert envelope["network_called"] is False
    assert envelope["execution_authorized"] is False
    assert envelope["body_json"] == spec
    spec["match_conditions"][0]["name"] = "changed"
    assert envelope["body_json"]["match_conditions"][0]["name"] == "fixed_arm_workcell"
    assert not hasattr(findall.FindAllClient, "create")
    assert not hasattr(findall.FindAllClient, "create_run")


@pytest.mark.parametrize("field,value", [
    ("objective", ""), ("entity_type", 1), ("generator", "unknown"),
    ("generator", []), ("match_limit", True), ("match_limit", 4),
    ("match_limit", 1001), ("match_conditions", []),
    ("match_conditions", [{"name": "a", "description": ""}]),
    ("match_conditions", [{"name": "a", "description": "a", "extra": "field"}]),
])
def test_invalid_specs_refused(spec, field, value):
    spec[field] = value
    with pytest.raises(findall.FindAllError):
        findall.prepare_run(spec)


def test_preview_and_duplicate_conditions_refused(spec):
    spec.update(generator="preview", match_limit=11)
    with pytest.raises(findall.FindAllError, match="preview_limit"):
        findall.prepare_run(spec)
    spec["match_limit"] = 5
    spec["match_conditions"].append(copy.deepcopy(spec["match_conditions"][0]))
    with pytest.raises(findall.FindAllError, match="duplicate_condition"):
        findall.prepare_run(spec)


def test_unknown_fields_refused_without_reflecting_values(spec):
    spec["api_key"] = FAKE_KEY
    with pytest.raises(findall.FindAllError) as error:
        findall.prepare_run(spec)
    assert FAKE_KEY not in str(error.value)


def test_check_only_tests_binding_membership(monkeypatch, capsys):
    class PresenceOnly(dict):
        def get(self, key, default=None):
            if key == findall.API_KEY_ENV:
                raise AssertionError("credential value read")
            return super().get(key, default)

        def __getitem__(self, key):
            if key == findall.API_KEY_ENV:
                raise AssertionError("credential value read")
            return super().__getitem__(key)

    monkeypatch.setattr(findall.os, "environ", PresenceOnly({findall.API_KEY_ENV: FAKE_KEY}))
    assert findall.main(["check"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["binding_present"] is True
    assert output["credential_validated"] is False
    assert output["network_called"] is False
    assert FAKE_KEY not in json.dumps(output)


def test_prepare_cli_does_not_need_credentials(capsys):
    assert findall.main(["prepare", str(SPEC_PATH)]) == 0
    assert json.loads(capsys.readouterr().out)["network_called"] is False


@pytest.mark.parametrize("method,suffix", [("status", ""), ("result", "/result")])
def test_reads_use_exact_endpoint_header_and_existing_safe_boundary(monkeypatch, method, suffix):
    run = run_status(active=True)
    candidate = {
        "candidate_id": "candidate_fixture", "match_status": "generated",
        "basis": [{"citations": [{"url": "https://example.org", "title": "fixture"}]}],
    }
    payload = run if method == "status" else {
        "run": run, "candidates": [candidate], "last_event_id": "fixture-event",
    }
    calls = fake_response(monkeypatch, payload)
    client = findall.FindAllClient(FAKE_KEY)
    assert FAKE_KEY not in repr(client)
    assert getattr(client, method)(RUN_ID) == payload
    assert len(calls) == 1
    request, options = calls[0]
    assert request.full_url == f"{findall.RUNS_URL}/{RUN_ID}{suffix}"
    assert request.get_method() == "GET"
    assert request.data is None
    assert request.get_header("X-api-key") == FAKE_KEY
    assert options["policy"].allowed_hosts == frozenset({"api.parallel.ai"})
    assert options["policy"].follow_same_origin_redirects is False
    assert options["max_response_bytes"] == 16 * 1024 * 1024


@pytest.mark.parametrize("run_id", ["https://evil.invalid", "findall_a/result", "findall_a?x=1", "findall_", "bad"])
def test_run_id_cannot_retarget_request(run_id):
    with pytest.raises(findall.FindAllError, match="findall_id_invalid"):
        findall.FindAllClient(FAKE_KEY).status(run_id)


@pytest.mark.parametrize("payload", [[], {}, {"findall_id": "findall_other", "status": {}},
                                      {"findall_id": RUN_ID, "status": {"status": "running", "is_active": "yes"}}])
def test_invalid_status_response_refused(monkeypatch, payload):
    fake_response(monkeypatch, payload)
    with pytest.raises(findall.FindAllError):
        findall.FindAllClient(FAKE_KEY).status(RUN_ID)


@pytest.mark.parametrize("candidates", [None, {}, ["invalid"]])
def test_invalid_result_response_refused(monkeypatch, candidates):
    fake_response(monkeypatch, {"run": run_status(), "candidates": candidates})
    with pytest.raises(findall.FindAllError, match="candidates_invalid"):
        findall.FindAllClient(FAKE_KEY).result(RUN_ID)


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError(findall.RUNS_URL, 401, FAKE_KEY, None, None),
    urllib.error.URLError(FAKE_KEY), TimeoutError(FAKE_KEY),
    http.client.BadStatusLine(FAKE_KEY),
    http.client.IncompleteRead(FAKE_KEY.encode(), 100),
    findall.safe_outbound_http.SafeOutboundHttpError(FAKE_KEY),
])
def test_errors_do_not_expose_credentials_or_retry(monkeypatch, error):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(findall.safe_outbound_http, "open_request", refuse)
    with pytest.raises(findall.FindAllError) as caught:
        findall.FindAllClient(FAKE_KEY).status(RUN_ID)
    assert len(calls) == 1
    assert FAKE_KEY not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_bad_json_and_http_body_not_printed(monkeypatch, capsys):
    fake_response(monkeypatch, FAKE_KEY.encode())
    monkeypatch.setenv(findall.API_KEY_ENV, FAKE_KEY)
    assert findall.main(["status", RUN_ID, "--use-runtime-key"]) == 2
    assert FAKE_KEY not in capsys.readouterr().err
    fake_response(monkeypatch, {"error": FAKE_KEY}, status=429)
    with pytest.raises(findall.FindAllError, match="findall_http_error:429"):
        findall.FindAllClient(FAKE_KEY).status(RUN_ID)


def test_cli_requires_explicit_credential_source():
    with pytest.raises(SystemExit) as caught:
        findall.main(["status", RUN_ID])
    assert caught.value.code == 2


def test_private_entry_refuses_nonterminal(monkeypatch, capsys):
    monkeypatch.setattr(findall.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(findall.getpass, "getpass", lambda *a: pytest.fail("must not prompt"))
    assert findall.main(["status", RUN_ID, "--prompt-key"]) == 2
    assert "private_terminal_required" in capsys.readouterr().err


def test_private_user_entry_only_in_memory(monkeypatch, capsys):
    monkeypatch.setattr(findall.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(findall.getpass, "getpass", lambda *a: FAKE_KEY)
    fake_response(monkeypatch, run_status())
    assert findall.main(["status", RUN_ID, "--prompt-key"]) == 0
    assert FAKE_KEY not in capsys.readouterr().out


def test_private_entry_refuses_real_getpass_echo_fallback(monkeypatch, capsys):
    class TerminalWithoutEchoControl:
        def isatty(self):
            return True

        def fileno(self):
            return 0

        def readline(self, *args):
            pytest.fail("must not read a key through echoed fallback")

    real_open = findall.os.open

    def no_controlling_terminal(path, *args, **kwargs):
        if path == "/dev/tty":
            raise OSError("synthetic terminal unavailable")
        return real_open(path, *args, **kwargs)

    def no_echo_control(fd):
        raise findall.getpass.termios.error("synthetic echo-control failure")

    monkeypatch.setattr(findall.sys, "stdin", TerminalWithoutEchoControl())
    monkeypatch.setattr(findall.getpass.os, "open", no_controlling_terminal)
    monkeypatch.setattr(findall.getpass.termios, "tcgetattr", no_echo_control)
    assert findall.main(["auth-check", "--prompt-key"]) == 2
    assert "private_terminal_echo_control_required" in capsys.readouterr().err


@pytest.mark.parametrize("error", [http.client.BadStatusLine(FAKE_KEY),
                                  http.client.IncompleteRead(FAKE_KEY.encode(), 100)])
def test_protocol_error_cli_is_sanitized(monkeypatch, capsys, error):
    monkeypatch.setenv(findall.API_KEY_ENV, FAKE_KEY)

    def fail_protocol(*args, **kwargs):
        raise error

    monkeypatch.setattr(findall.safe_outbound_http, "open_request", fail_protocol)
    assert findall.main(["auth-check", "--use-runtime-key"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert FAKE_KEY not in output.err
    assert "findall_transport_failed" in output.err


@pytest.mark.parametrize("monitors", [[], [{"monitor_id": "private", "settings": {"query": "private"}}]])
def test_auth_check_needs_no_run_id_and_does_not_expose_monitor_data(monkeypatch, capsys, monitors):
    monkeypatch.setattr(findall.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(findall.getpass, "getpass", lambda *a: FAKE_KEY)
    calls = fake_response(monkeypatch, {"monitors": monitors})
    assert findall.main(["auth-check", "--prompt-key"]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["authenticated"] is True
    assert parsed["findall_scope_verified"] is False
    assert parsed["paid_creation_authorized"] is False
    assert "private" not in output
    assert FAKE_KEY not in output
    request, options = calls[0]
    assert request.full_url == "https://api.parallel.ai/v1/monitors?limit=1"
    assert request.get_method() == "GET"
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [{}, {"monitors": None}, {"monitors": ["bad"]}])
def test_auth_check_invalid_response_refused(monkeypatch, payload):
    fake_response(monkeypatch, payload)
    with pytest.raises(findall.FindAllError, match="auth_check_response_invalid"):
        findall.FindAllClient(FAKE_KEY).auth_check()


@pytest.mark.parametrize("key", ["", "bad\nkey", "bad key", None])
def test_invalid_key_refused_without_reflection(key):
    with pytest.raises(findall.FindAllError, match="findall_runtime_key_invalid"):
        findall.FindAllClient(key)
