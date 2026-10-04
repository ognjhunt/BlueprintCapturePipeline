"""Synthetic execution contracts; all provider networking/real keys are blocked."""

import copy
import http.client
import json
import socket
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import parallel_findall_execution as execution
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission,
)
from blueprint_pipeline.safe_outbound_http import SafeHttpResponse

FAKE_KEY = "synthetic-offline-key"
OPERATION_ID = "blueprint-fixture-discovery-1"
RUN_ID = "findall_fixture_execution"


@pytest.fixture(autouse=True)
def prohibit_network_and_real_credentials(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("network forbidden in synthetic FindAll verification")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(execution.safe_outbound_http, "open_request", refuse)


@pytest.fixture
def spec():
    path = Path(__file__).resolve().parents[1] / "docs/examples/parallel_findall_spec.json"
    return json.loads(path.read_text())


def prepare(spec, **overrides):
    return execution.prepare_submission(spec, **{
        "operation_id": OPERATION_ID, "maximum_cost_usd": "1.00", **overrides,
    })


def synthetic_grant(prepared, *, resource_class=execution.PAID_RESOURCE_CLASS, binding=True):
    admission = {
        "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
        "resource_class": resource_class, "status": "admitted", "blockers": [],
    }
    if binding:
        admission["allocation_binding_digest"] = prepared["allocation_binding_digest"]
    return require_paid_resource_admission(
        admission, resource_class=resource_class,
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


class SyntheticJournal:
    """Shared fixture state models atomic authority+claim, never production I/O."""

    def __init__(self, state=None):
        self.state = state if state is not None else {}
        self.events = []
        self.receipt = None

    def claim_submission(self, prepared):
        operation = prepared["operation_id"]
        self.events.append("claim")
        if operation in self.state:
            return False
        self.state[operation] = {"state": "submission_unresolved", "prepared": prepared}
        return True

    def record_created(self, operation_id, binding, run):
        self.events.append("record")
        assert self.state[operation_id]["prepared"]["allocation_binding_digest"] == binding
        self.state[operation_id]["state"] = "created"
        self.receipt = copy.deepcopy(run)


def submit(client, spec, grant, journal, **overrides):
    return client.create(spec, **{
        "operation_id": OPERATION_ID, "maximum_cost_usd": "1.00",
        "paid_resource_admission_grant": grant, "journal": journal, **overrides,
    })


def run():
    return {"findall_id": RUN_ID, "generator": "base",
            "status": {"status": "queued", "is_active": True},
            "future_field": {"preserved": True}}


def provider_stub(monkeypatch, journal, *, payload=None, status=200, error=None):
    calls = []

    def respond(request, **options):
        assert journal.state[OPERATION_ID]["state"] == "submission_unresolved"
        journal.events.append("post")
        calls.append((request, options))
        if error is not None:
            raise error
        value = payload if payload is not None else run()
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        return SafeHttpResponse(status, body, request.full_url, request.full_url)

    monkeypatch.setattr(execution.safe_outbound_http, "open_request", respond)
    return calls


@pytest.mark.parametrize("generator,limit,expected", [
    ("preview", 10, "0.10"), ("base", 5, "0.40"),
    ("core", 18, "4.70"), ("core", 30, "6.50"), ("pro", 1000, "1010.00"),
])
def test_offline_estimate_is_exact_and_never_authorizes(spec, generator, limit, expected):
    spec.update(generator=generator, match_limit=limit)
    prepared = prepare(spec, maximum_cost_usd=expected)
    assert prepared["estimated_maximum_cost_usd"] == expected
    assert prepared["pricing_version"] == "parallel-findall-2026-10-03"
    assert prepared["provider_enforced_dollar_cap"] is False
    assert prepared["execution_authorized"] is False
    assert prepared["network_called"] is False


@pytest.mark.parametrize("budget", ["0.39", "NaN", "Infinity", "-1", "invalid", 1.0, True])
def test_invalid_or_insufficient_budget_is_offline_refusal(spec, budget):
    with pytest.raises(execution.FindAllError):
        prepare(spec, maximum_cost_usd=budget)


def test_binding_covers_body_operation_and_budget_and_ignores_json_key_order(spec):
    original = prepare(spec)
    assert prepare(dict(reversed(list(spec.items())))) == original
    assert prepare(spec, maximum_cost_usd="1") == original
    for field, value in [("objective", "different question"), ("generator", "preview"),
                         ("match_limit", 6), ("metadata", {"receipt": "fixture"})]:
        changed = {**spec, field: value}
        assert prepare(changed)["allocation_binding_digest"] != original["allocation_binding_digest"]
    assert prepare(spec, operation_id="new") != original
    assert prepare(spec, maximum_cost_usd="2") != original
    original["body_json"]["objective"] = "mutated detached review"
    assert spec["objective"] != original["body_json"]["objective"]


def test_missing_forged_wrong_class_and_unbound_grants_cannot_claim_or_dispatch(spec):
    prepared = prepare(spec)
    forged = PaidResourceAdmissionGrant(
        resource_class=execution.PAID_RESOURCE_CLASS,
        schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION, _issuer=object(),
        allocation_binding_digest=prepared["allocation_binding_digest"],
    )
    for grant in (None, forged, synthetic_grant(prepared, resource_class="evaluator_api"),
                  synthetic_grant(prepared, binding=False), synthetic_grant(prepare(spec, maximum_cost_usd="2"))):
        journal = SyntheticJournal()
        with pytest.raises(PaidResourceAdmissionBlocked):
            submit(execution.AdmittedFindAllClient(FAKE_KEY), spec, grant, journal)
        assert journal.events == []


def test_valid_create_claims_before_post_records_run_and_preserves_all_fields(monkeypatch, spec):
    journal = SyntheticJournal()
    calls = provider_stub(monkeypatch, journal)
    result = submit(execution.AdmittedFindAllClient(FAKE_KEY), spec, synthetic_grant(prepare(spec)), journal)
    assert result == run() == journal.receipt
    assert journal.events == ["claim", "post", "record"]
    assert len(calls) == 1
    request, options = calls[0]
    assert request.get_method() == "POST"
    assert request.full_url == execution.RUNS_URL
    assert json.loads(request.data) == spec
    assert request.get_header("X-api-key") == FAKE_KEY
    assert request.get_header("Content-type") == "application/json"
    assert options["policy"].allowed_hosts == frozenset({"api.parallel.ai"})
    assert options["policy"].follow_same_origin_redirects is False
    assert options["timeout_seconds"] == 30
    assert options["max_response_bytes"] == 16 * 1024 * 1024
    assert FAKE_KEY not in json.dumps(journal.state)


@pytest.mark.parametrize("error", [TimeoutError(FAKE_KEY), urllib.error.URLError(FAKE_KEY),
    http.client.BadStatusLine(FAKE_KEY), http.client.IncompleteRead(FAKE_KEY.encode(), 100),
    execution.safe_outbound_http.SafeOutboundHttpError(FAKE_KEY),
    urllib.error.HTTPError(execution.RUNS_URL, 429, FAKE_KEY, None, None)])
def test_ambiguous_create_never_retries_even_after_controller_restart(monkeypatch, spec, error):
    journal = SyntheticJournal()
    calls = provider_stub(monkeypatch, journal, error=error)
    grant = synthetic_grant(prepare(spec))
    with pytest.raises(execution.FindAllSubmissionUnresolved) as caught:
        submit(execution.AdmittedFindAllClient(FAKE_KEY), spec, grant, journal)
    assert caught.value.findall_id is None
    assert FAKE_KEY not in str(caught.value)
    assert caught.value.__suppress_context__ is True
    restarted = SyntheticJournal(journal.state)
    with pytest.raises(execution.FindAllError, match="already_claimed"):
        submit(execution.AdmittedFindAllClient(FAKE_KEY), spec, grant, restarted)
    assert len(calls) == 1
    assert journal.state[OPERATION_ID]["state"] == "submission_unresolved"


def test_changed_body_cannot_replay_claimed_operation_even_with_new_bound_grant(monkeypatch, spec):
    journal = SyntheticJournal()
    calls = provider_stub(monkeypatch, journal)
    client = execution.AdmittedFindAllClient(FAKE_KEY)
    submit(client, spec, synthetic_grant(prepare(spec)), journal)
    changed = {**spec, "objective": "changed question"}
    with pytest.raises(execution.FindAllError, match="already_claimed"):
        submit(client, changed, synthetic_grant(prepare(changed)), journal)
    assert len(calls) == 1


@pytest.mark.parametrize("payload,status", [(b"invalid-json", 200), ({}, 200),
    ([], 200), ({"findall_id": "evil/path"}, 200), ({"error": FAKE_KEY}, 402)])
def test_invalid_create_receipts_leave_claim_unresolved(monkeypatch, spec, payload, status):
    journal = SyntheticJournal()
    calls = provider_stub(monkeypatch, journal, payload=payload, status=status)
    with pytest.raises(execution.FindAllSubmissionUnresolved):
        submit(execution.AdmittedFindAllClient(FAKE_KEY), spec, synthetic_grant(prepare(spec)), journal)
    assert len(calls) == 1
    assert journal.state[OPERATION_ID]["state"] == "submission_unresolved"


@pytest.mark.parametrize("broken", ["persistence", "status", "generator"])
def test_known_run_id_survives_post_receipt_failures(monkeypatch, spec, broken):
    journal = SyntheticJournal()
    payload = run()
    if broken == "persistence":
        def fail(*args):
            raise OSError(FAKE_KEY)
        journal.record_created = fail
    elif broken == "status":
        payload["status"] = None
    else:
        payload["generator"] = "pro"
    calls = provider_stub(monkeypatch, journal, payload=payload)
    with pytest.raises(execution.FindAllSubmissionUnresolved) as caught:
        submit(execution.AdmittedFindAllClient(FAKE_KEY), spec, synthetic_grant(prepare(spec)), journal)
    assert caught.value.findall_id == RUN_ID
    assert str(caught.value) == f"findall_submission_unresolved:{RUN_ID}"
    assert len(calls) == 1


@pytest.mark.parametrize("claim", [False, None, "true", 1, OSError(FAKE_KEY)])
def test_journal_failure_or_nonliteral_success_cannot_dispatch(spec, claim):
    journal = SyntheticJournal()
    def outcome(*args):
        if isinstance(claim, Exception):
            raise claim
        return claim
    journal.claim_submission = outcome
    with pytest.raises(execution.FindAllError) as caught:
        submit(execution.AdmittedFindAllClient(FAKE_KEY), spec, synthetic_grant(prepare(spec)), journal)
    assert FAKE_KEY not in str(caught.value)


def test_cancel_is_exact_204_no_body_and_subsequent_status_is_explicit(monkeypatch):
    calls = []
    def cancel(request, **options):
        calls.append((request, options))
        return SafeHttpResponse(204, b"", request.full_url, request.full_url)
    monkeypatch.setattr(execution.safe_outbound_http, "open_request", cancel)
    assert execution.AdmittedFindAllClient(FAKE_KEY).cancel(RUN_ID) is None
    assert len(calls) == 1
    request, options = calls[0]
    assert request.full_url == f"{execution.RUNS_URL}/{RUN_ID}/cancel"
    assert request.get_method() == "POST"
    assert request.data is None
    assert request.get_header("X-api-key") == FAKE_KEY
    assert options["policy"].follow_same_origin_redirects is False


@pytest.mark.parametrize("status", [200, 404, 409, 429])
def test_cancel_errors_are_not_success_receipts_and_do_not_retry(monkeypatch, status):
    calls = []
    def fail(request, **options):
        calls.append(True)
        return SafeHttpResponse(status, FAKE_KEY.encode(), request.full_url, request.full_url)
    monkeypatch.setattr(execution.safe_outbound_http, "open_request", fail)
    with pytest.raises(execution.FindAllError) as caught:
        execution.AdmittedFindAllClient(FAKE_KEY).cancel(RUN_ID)
    assert FAKE_KEY not in str(caught.value)
    assert len(calls) == 1


def test_cancel_cannot_retarget_request():
    with pytest.raises(execution.FindAllError, match="findall_id_invalid"):
        execution.AdmittedFindAllClient(FAKE_KEY).cancel("findall_fixture/other")
