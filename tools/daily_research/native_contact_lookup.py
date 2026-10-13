"""Native research to the existing contact lookup contract. No qualification or send authority.

The normal operator lookup guard stays intact. This worker adapter uses the
existing private bridge, worker lease, canonical paid gate and FullEnrich
client. An absent/disabled scoped control makes no provider call.
"""

import hashlib
from datetime import datetime, timezone
from urllib.parse import urlsplit

from tools.daily_research import contact_lookup as lookup
from tools.daily_research import site_screen as screen
from tools.daily_research import verification
from tools.daily_research.runner import Refusal, canonical, digest

INPUT = "blueprint.native-contact-lookup-input.v1"
CONTROL = "blueprint.native-contact-lookup-control.v1"
RESOURCE = "fullenrich_work_email"


def native_site(value, *, today=None):
    """Retain native provenance; never invent a Parallel run, screen result or person proof."""
    if not isinstance(value, dict) or value.get("schema_version") != INPUT:
        raise Refusal("native_contact_input_invalid")
    if (
        not isinstance(value.get("input_id"), str)
        or not screen.SHA.fullmatch(value["input_id"])
        or not isinstance(value.get("provenance"), dict)
    ):
        raise Refusal("native_contact_input_invalid")
    candidate, assessment, sources = (
        value.get("candidate"),
        value.get("assessment"),
        value.get("sources"),
    )
    if (
        not isinstance(candidate, dict)
        or not isinstance(assessment, dict)
        or not isinstance(sources, list)
    ):
        raise Refusal("native_contact_input_invalid")
    if verification.assessment_issues(candidate, assessment):
        raise Refusal("native_contact_assessment_invalid")
    now = datetime.now(timezone.utc)
    if verification.diagnostic_moment(assessment["assessed_at"]) > now or (
        assessment.get("valid_until") is not None
        and verification.diagnostic_moment(assessment["valid_until"]) <= now
    ):
        raise Refusal("native_contact_assessment_stale")
    if assessment["counterevidence"]["status"] == "contradicted":
        raise Refusal("native_contact_counterevidence_unresolved")
    claims = assessment.get("claims", {})
    if any(
        claims.get(key, {}).get("status") != "verified_fact"
        for key in ("operator", "physical_site")
    ) or claims.get("site_task", {}).get("status") not in {"verified_fact", "inference"}:
        raise Refusal("native_contact_site_unverified")
    # Unknown fit, manual workflow and interest are questions, never made-up exclusions.
    wanted = {key: candidate.get(key) for key in ("organization", "site", "location", "task")}
    if any(not isinstance(text, str) or not text.strip() for text in wanted.values()):
        raise Refusal("native_contact_identity_invalid")
    url = candidate.get("organization_url", "")
    parsed = urlsplit(url)
    domain = screen.site_domain(url)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not domain
        or domain in screen.FREE_MAIL
        or domain in screen.NOT_OPERATOR_DOMAINS
    ):
        raise Refusal("native_contact_operator_domain_unproven")
    ledger = {
        item.get("id"): item for item in assessment.get("sources", []) if isinstance(item, dict)
    }
    pages = []
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("text"), str)
            or hashlib.sha256(source["text"].encode()).hexdigest() != source.get("text_sha256")
            or screen.site_domain(source.get("url", "")) != domain
            or screen.never_fetch(source["url"])
        ):
            raise Refusal("native_contact_source_changed")
        pages.append((source["url"], source["text"], source["text_sha256"]))
    for key in ("operator", "physical_site", "site_task"):
        refs = claims[key].get("source_refs", [])
        # A recently read, undated operator statement can support a contact
        # question without proving present floor work. Keep its unknown
        # freshness; this adapter grants neither qualification nor outreach.
        if not any(
            ref in ledger
            and ledger[ref].get("freshness") in {"current", "unknown"}
            and ledger[ref].get("classification") == "operator"
            and ledger[ref].get("retrieval") == "static"
            and isinstance(ledger[ref].get("quote"), str)
            and ledger[ref]["quote"].strip()
            and any(
                ledger[ref].get("url") == url and ledger[ref]["quote"] in text
                for url, text, _ in pages
            )
            for ref in refs
        ):
            raise Refusal("native_contact_claim_source_unproven")
    identity = [screen.words(wanted[key]) for key in ("organization", "site", "location", "task")]
    site_key = hashlib.sha256(canonical(identity).encode()).hexdigest()
    contact = {"site_key": site_key, "operator_domains": [domain], "person": {}, "email": {}}
    address = screen.parse_location(wanted["location"])
    return {
        "site_key": site_key,
        "contact": contact,
        "address": address,
        "task_input": {
            "site_name": wanted["site"],
            "location": wanted["location"],
            "operator": wanted["organization"],
            "task": wanted["task"],
            **address,
        },
        "person": {},
        "person_quote": "",
        "published_person_email": False,
        "operator": wanted["organization"],
        "operator_domains": [domain],
        "operator_domain": domain,
        "kept_pages": pages,
        "today": today or now.date(),
        "native_provenance": value["provenance"],
    }


class DurableCalls(dict):
    """Lazy read of company/person identities across dates, using the original event fold."""

    def __init__(self, bridge, context):
        super().__init__()
        self.bridge, self.context = bridge, context

    def get(self, key, default=None):
        if key not in self:
            value = self.bridge.call("native_contact_call_get", context=self.context, key=key)
            if value is not None:
                events = value.get("events", [])
                # Validate each stored event rather than trusting a cached result summary.
                if any(lookup._event(canonical(event).encode()) is None for event in events):
                    raise Refusal("native_contact_journal_invalid")
                folded = lookup.fold(events)
                if key not in folded:
                    raise Refusal("native_contact_journal_invalid")
                self[key] = folded[key]
        return super().get(key, default)


class DurableBook:
    def __init__(self, bridge, context):
        self.bridge, self.context = bridge, context
        self.calls = DurableCalls(bridge, context)

    def append(self, **event):
        value = {"schema_version": lookup.JOURNAL, **event, "recorded_at": screen._now()}
        self.bridge.call(
            "native_contact_call_append",
            context=self.context,
            event=value,
            request=self.request if event["event"] == "intent" else None,
        )


class BoundCalls(lookup.Calls):
    def __call__(self, kind, identity, site_key, body, seal):
        key = self.key(kind, identity)
        self.book.request = {
            "kind": kind,
            "identity": identity,
            "site_key": site_key,
            "body": body(key),
        }
        return super().__call__(kind, identity, site_key, body, seal)


class AdmittedClient(lookup.FullEnrichClient):
    """The last action before POST rechecks the host lease/scope and canonical opaque grant."""

    def __init__(self, key, bridge, context, *, admission, **options):
        super().__init__(key, **options)
        self.bridge, self.context, self.admission = bridge, context, admission

    def post(self, path, body):
        bound = self.bridge.call(
            "native_contact_dispatch_admission",
            context=self.context,
            path=path,
            body=body,
            body_digest=digest(body),
        )
        grant = self.admission.require_paid_resource_admission(
            bound,
            resource_class=RESOURCE,
            expected_schema_version=self.admission.PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
        self.admission.require_paid_resource_admission_grant(
            grant,
            resource_class=RESOURCE,
            allocation_binding_digest=bound.get("allocation_binding_digest"),
            require_allocation_binding=True,
        )
        return super().post(path, body)


def execute(bridge, value, api_key, *, admission, transport=lookup.https):
    """Caller holds the existing worker lease. Durable intents precede every possible paid POST."""
    site = native_site(value)
    context = {"input_id": value["input_id"], "input_digest": digest(value)}
    base = {
        "input_id": value["input_id"],
        "site_key": site["site_key"],
        "native_provenance": site["native_provenance"],
        "sends_authorized": False,
        "qualification_authorized": False,
        "assessment": value["assessment"],
    }
    control = bridge.call("native_contact_scope", context=context)
    if not control or control.get("dispatch_enabled") is not True:
        return {**base, "state": "native_contact_lookup_disabled", "paid_calls": 0}
    book = DurableBook(bridge, context)
    client = AdmittedClient(api_key, bridge, context, admission=admission, transport=transport)
    calls = BoundCalls(
        book,
        "apply",
        client=client,
        bounds={
            "owner_reference": control["approval_reference"],
            "max_credits": None,
            "max_calls": None,
        },
        code={"source": "native_research", "input_id": value["input_id"]},
    )
    person, reason = lookup.target(site, calls, search=control.get("person_search") is True)
    if person is None:
        return {
            **base,
            "state": "native_contact_lookup_hold",
            "reason": reason,
            "paid_calls": calls.counts["made"],
        }
    key, call = calls(
        "enrich",
        lookup.enrich_identity(site, person),
        site["site_key"],
        lookup.enrich_body(site, person),
        lookup.seal_start,
    )
    if call and call["state"] == "created":
        try:
            status, raw = client.result(call["enrichment_id"])
        except lookup.LookupFailure as exc:
            if str(exc) != "contact_lookup_result_unavailable":
                raise
            # Preserve the created job and retry only its unbilled GET.
            fields = None
        else:
            fields = lookup.seal_result(status, raw, key, call, person, site)
        if fields is not None:
            book.append(call=key, attempt=call["attempt"], **fields)
            book.calls.pop(key, None)
            call = book.calls.get(key)
    result = lookup.result(site, person, key, call) if call is not None else None
    return {
        **base,
        "state": "native_contact_lookup_complete"
        if result
        and lookup.usable_lookup(
            result, operator_domains=site["operator_domains"], site_key=site["site_key"]
        )
        else "native_contact_lookup_pending",
        "lookup": result,
        "paid_calls": calls.counts["made"],
        "retry_result_read": bool(call and call["state"] == "created"),
    }


def step(bridge, day, *, api_key=None):
    import os

    from tools.daily_research.firestore import FirestoreLedger

    with FirestoreLedger(bridge).lock():
        control = bridge.call("control")
        selected = control.get("native_contact_lookup") or {}
        if selected.get("enabled") is not True:
            return {"state": "native_contact_lookup_disabled", "paid_calls": 0}
        prepared = bridge.call("native_contact_inputs")
        if not prepared:
            return {"state": "native_contact_lookup_idle", "paid_calls": 0}
        # Existing projected canonical import closure, as used by FindAll.
        from tools.daily_research.findall import runtime

        results = []
        for value in prepared:
            try:
                result = execute(
                    bridge,
                    value,
                    api_key or os.environ.get(lookup.KEY_ENV, ""),
                    admission=runtime().admission,
                )
            except (lookup.LookupFailure, lookup.Refused, lookup.Unknown) as exc:
                # The journal remains canonical. No new POST is inferred from
                # a failed result read or an uncertain provider response.
                result = {
                    "input_id": value["input_id"],
                    "state": "native_contact_lookup_blocked",
                    "reason": str(exc),
                    "paid_calls": None,
                    "cost_status": "journal_reconciliation_required",
                    "sends_authorized": False,
                    "qualification_authorized": False,
                }
            results.append(result)
            bridge.call(
                "native_contact_result_put",
                input_id=value["input_id"],
                input_digest=digest(value),
                result=result,
            )
        return {
            "state": "native_contact_lookup_observed",
            "date": day,
            "scope_id": selected["scope_id"],
            "inputs": len(results),
            "input_ids": [value["input_id"] for value in prepared],
            "paid_calls_known": sum(result["paid_calls"] or 0 for result in results),
            "usage_incomplete": any(result["paid_calls"] is None for result in results),
            "retry_result_read": any(result.get("retry_result_read") is True for result in results),
            "sends_authorized": False,
        }
