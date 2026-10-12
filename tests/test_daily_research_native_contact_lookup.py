"""Native lookup lifecycle with synthetic evidence and an in-memory durable host; no network."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import paid_resource_admission as admission
from tests.daily_research_verification_fixture import assessment
from tests.test_daily_research_contact_lookup import (
    FULLENRICH_KEY,
    FakeFullEnrich,
    profile,
    searched,
)
from tools.daily_research import contact_lookup as lookup
from tools.daily_research import native_contact_lookup as native
from tools.daily_research import site_screen as screen
from tools.daily_research.runner import Refusal, digest


def input_value():
    now = datetime.now(timezone.utc) - timedelta(seconds=10)
    candidate = {
        "candidate_key": "synthetic-native",
        "organization": "Synthetic Operator 1",
        "organization_url": "https://operator-1.example",
        "site": "Packing center",
        "location": "Fixture City, TX",
        "task": "packing",
    }
    value = assessment(candidate, now)
    value["claims"]["plausible_fit"].update(status="unresolved", source_refs=[])
    value["claims"]["human_workflow"].update(status="unresolved", source_refs=[])
    value["sources"][0]["url"] = candidate["organization_url"] + "/facility"
    text = value["sources"][0]["quote"]
    provenance = {
        "run_key": "synthetic-only",
        "row_blob": "a" * 64,
        "candidate_key": candidate["candidate_key"],
    }
    return {
        "schema_version": native.INPUT,
        "input_id": digest(provenance),
        "candidate": candidate,
        "assessment": value,
        "sources": [
            {
                "url": value["sources"][0]["url"],
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "provenance": provenance,
    }


class Host:
    def __init__(self):
        self.calls, self.ops = {}, []
        self.enabled = True
        self.reject_dispatch = False

    def call(self, op, **fields):
        self.ops.append(op)
        if op == "native_contact_scope":
            return {
                "dispatch_enabled": self.enabled,
                "approval_reference": "synthetic-owner",
                "person_search": True,
            }
        if op == "native_contact_call_get":
            return deepcopy(self.calls.get(fields["key"]))
        if op == "native_contact_call_append":
            event = fields["event"]
            assert lookup._event(screen.canonical(event).encode())
            row = self.calls.setdefault(event["call"], {"events": []})
            row["events"].append(deepcopy(event))
            lookup.fold(row["events"])
            return True
        if op == "native_contact_dispatch_admission":
            assert fields["body_digest"] == digest(fields["body"])
            if self.reject_dispatch:
                raise Refusal("native_contact_scope_expired")
            return {
                "schema_version": admission.PAID_LANE_ADMISSION_SCHEMA_VERSION,
                "status": "admitted",
                "resource_class": native.RESOURCE,
                "blockers": [],
                "allocation_binding_digest": digest(fields),
            }
        raise AssertionError(op)


def run(host, value, api):
    return native.execute(host, value, FULLENRICH_KEY, admission=admission, transport=api)


def test_unknown_fit_and_human_workflow_remain_questions_without_faking_provider_screen():
    value = input_value()
    site = native.native_site(value)
    assert site["operator_domain"] == "operator-1.example"
    assert site["native_provenance"] == value["provenance"]
    assert site["person"] == {} and site["published_person_email"] is False
    assert "parallel" not in json.dumps(site, default=str).lower()
    assert site["address"]["city"] == "Fixture City"


@pytest.mark.parametrize(
    "change,code",
    [
        (
            lambda v: v["assessment"]["claims"]["operator"].update(status="unresolved"),
            "site_unverified",
        ),
        (lambda v: v["assessment"].update(candidate_digest="0" * 64), "assessment_invalid"),
        (lambda v: v["sources"][0].update(text="changed"), "source_changed"),
        (lambda v: v["sources"][0].update(url="https://unrelated.example"), "source_changed"),
        (lambda v: v["assessment"]["sources"][0].update(quote="invented"), "claim_source_unproven"),
        (
            lambda v: v["assessment"].update(
                valid_until=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            ),
            "assessment_stale",
        ),
    ],
)
def test_native_binding_or_source_damage_never_dispatches(change, code):
    host, api, value = Host(), FakeFullEnrich(), input_value()
    change(value)
    with pytest.raises(Refusal, match="native_contact_" + code):
        run(host, value, api)
    assert api.calls == [] and host.ops == []


def test_disabled_scope_needs_no_key_and_makes_no_journal_or_provider_call():
    host = Host()
    host.enabled = False
    result = native.execute(
        host,
        input_value(),
        None,
        admission=admission,
        transport=lambda *a, **k: pytest.fail("unexpected provider call"),
    )
    assert result["state"] == "native_contact_lookup_disabled"
    assert host.ops == ["native_contact_scope"]


def test_real_lookup_rules_reuse_company_and_person_results_across_daily_input_ids():
    host, api, value = Host(), FakeFullEnrich(), input_value()
    api.people["operator-1.example"] = [searched("Jordan Fixture", title="Operations Director")]
    api.emails[("Jordan", "Fixture", "operator-1.example")] = (
        "jordan.fixture@operator-1.example",
        "DELIVERABLE",
        profile("Jordan Fixture"),
    )
    first = run(host, value, api)
    assert first["state"] == "native_contact_lookup_complete"
    assert first["lookup"]["person"]["site_responsibility"]["route"] == "corporate_referral"
    assert first["lookup"]["provider"]["credits"] == "1"
    assert first["sends_authorized"] is False
    # The next native day is a distinct immutable input but the same company/person lookup identities.
    second_value = deepcopy(value)
    second_value["input_id"] = "b" * 64
    second = run(host, second_value, api)
    assert second["state"] == "native_contact_lookup_complete" and second["paid_calls"] == 0
    assert len(api.posts(lookup.SEARCH_PATH)) == len(api.posts(lookup.ENRICH_PATH)) == 1
    assert "personal-mail" not in json.dumps(host.calls) and "555-010" not in json.dumps(host.calls)


def test_uncertain_paid_post_is_retained_and_never_repeated_on_next_input():
    host, api, value = Host(), FakeFullEnrich(), input_value()
    api.scripted = [screen.TransportError("synthetic_unknown", sent=True)]
    with pytest.raises(lookup.Unknown):
        run(host, value, api)
    assert len(api.calls) == 1
    value["input_id"] = "c" * 64
    result = run(host, value, api)
    assert result["state"] == "native_contact_lookup_hold"
    assert len(api.calls) == 1
    assert (
        next(iter(lookup.fold(next(iter(host.calls.values()))["events"]).values()))["state"]
        == "unknown"
    )


def test_final_admission_failure_prevents_post_and_conservatively_keeps_intent():
    host, api = Host(), FakeFullEnrich()
    host.reject_dispatch = True
    with pytest.raises(Refusal, match="native_contact_scope_expired"):
        run(host, input_value(), api)
    assert api.calls == []
    assert next(iter(host.calls.values()))["events"][0]["event"] == "intent"


def test_operator_worker_refusal_is_preserved(monkeypatch):
    monkeypatch.setenv(screen.WORKER_FLAG, "true")
    with pytest.raises(lookup.LookupFailure, match="contact_lookup_worker_needs_paid_admission"):
        lookup.refuse_on_worker()


@pytest.mark.parametrize("freshness,allowed", [("unknown", True), ("historical", False)])
def test_undated_operator_statement_is_a_contact_question_not_current_floor_proof(
    freshness, allowed
):
    value = input_value()
    value["assessment"]["valid_until"] = None
    value["assessment"]["sources"][0]["freshness"] = freshness
    original = deepcopy(value)
    if allowed:
        native.native_site(value)
    else:
        with pytest.raises(Refusal, match="native_contact_claim_source_unproven"):
            native.native_site(value)
    assert value == original
