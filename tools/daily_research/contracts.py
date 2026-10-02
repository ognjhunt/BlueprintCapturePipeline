"""Research v2 provenance and proposed-delta validation, without sink writes."""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from tools.daily_research import freshness
from tools.daily_research.knowledge import (
    LEVELS,
    SnapshotError,
    calendar_date,
    fact_state,
    require,
    shape,
    source_moment,
    text,
    timestamp,
    url,
)

CENTRAL = ZoneInfo("America/Chicago")
EVIDENCE_V2 = {"origin", "evidence_level", "source_checked_at", "snapshot_loaded_at", "revalidated_at", "snapshot_record_id", "snapshot_fact_id"}
DELTA_FIELDS = frozenset({"record_id", "fact_id", "reason", "proposed_statement", "evidence", "unknowns"})
DELTA_EVIDENCE_FIELDS = frozenset({"url", "publisher", "publication_date", "source_checked_at", "classification", "evidence_level", "quote"})
SCOPES = {"as_of_background", "current_operational", "deployment_critical"}
CLASSIFICATIONS = {"operator", "vendor", "independent"}


def probe(check, collect=False):
    """Run one strict check and return its stable code, or None.

    Strict callers keep their exact fail-fast behaviour: malformed values still
    raise their original exception. Collect mode reports them as schema issues.
    """
    try:
        check()
    except SnapshotError as exc:
        return str(exc)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        if not collect:
            raise
        return "output_schema_invalid"
    return None


def checked_day(value):
    return source_moment(value).astimezone(CENTRAL).date().isoformat()


def lookup(context, record_id, fact_id):
    require(isinstance(context, dict), "knowledge_context_missing")
    for record in context["records"]:
        if record["record_id"] == record_id:
            for fact in record["facts"]:
                if fact["fact_id"] == fact_id:
                    return fact
    raise SnapshotError("snapshot_fact_not_in_context")


def evidence(value, day, context, observed_at=None, *, policy=None):
    for _pointer, code in evidence_issues(value, day, context, observed_at, policy=policy):
        raise SnapshotError(code)


def evidence_issues(value, day, context, observed_at=None, *, policy=None, collect=False):
    """Every provenance rule for one evidence entry as (pointer, code), in strict order."""
    def level():
        if value["role"] in {"capability", "background"}:
            require(value["evidence_level"] in LEVELS, "evidence_level_invalid")
            require(value["evidence_level"] != "unknown", "unsupported_evidence_level")
        else:
            require(value["evidence_level"] is None, "site_evidence_level_must_be_null")

    def snapshot_scope():
        if value["origin"] == "snapshot":
            require(value["assertion_scope"] == "as_of_background", "cached_operational_assertion_forbidden")

    checks = [
        ("/evidence_level", level),
        ("/checked_date", lambda: require(value["checked_date"] == checked_day(value["source_checked_at"]), "evidence_date_integrity_invalid")),
        ("/checked_date", lambda: require(calendar_date(value["checked_date"]) <= date.fromisoformat(day), "evidence_date_in_future")),
        ("/source_checked_at", lambda: require(source_moment(value["source_checked_at"]) <= timestamp(context["snapshot_loaded_at"])
                                               or value["origin"] == "live", "evidence_date_in_future")),
    ]
    if policy is not None:
        checks += [("/assertion_scope", lambda: require(value["assertion_scope"] in SCOPES, "evidence_assertion_scope_invalid")),
                   ("/assertion_scope", snapshot_scope)]
    for pointer, check in checks:
        code = probe(check, collect)
        if code:
            yield pointer, code
    if value["origin"] == "live":
        live_binding = next((f"/{name}" for name in ("snapshot_loaded_at", "snapshot_record_id", "snapshot_fact_id")
                             if value[name] is not None), "/checked_date")
        for pointer, check in (
                ("/source_checked_at", lambda: require(source_moment(value["source_checked_at"])
                                                       <= (observed_at or datetime.now(timezone.utc)), "evidence_date_in_future")),
                (live_binding, lambda: require(value["checked_date"] == day and value["snapshot_loaded_at"] is None
                                               and value["snapshot_record_id"] is None and value["snapshot_fact_id"] is None,
                                               "live_evidence_binding_invalid")),
                ("/revalidated_at", lambda: require(value["revalidated_at"] in {None, value["source_checked_at"]},
                                                    "evidence_date_integrity_invalid"))):
            code = probe(check, collect)
            if code:
                yield pointer, code
        return
    code = probe(lambda: require(value["origin"] == "snapshot", "evidence_origin_invalid"), collect)
    if code:
        yield "/origin", code
        return
    code = probe(lambda: require(value["role"] in ({"capability", "background"} if policy is not None else {"capability"}),
                                 "live_task_geography_required"), collect)
    if code:
        yield "/role", code
    found = []
    code = probe(lambda: found.append(lookup(context, value["snapshot_record_id"], value["snapshot_fact_id"])), collect)
    if code:
        yield "/snapshot_fact_id", code
        return
    fact = found[0]

    def usable():
        if policy is not None:
            freshness.cached_citation(fact, policy, value["snapshot_record_id"], value["role"])
        else:
            require(fact["load_state"] == "usable_background"
                    and fact_state(fact, observed_at or datetime.now(timezone.utc)) == "usable_background", "cached_fact_not_usable")

    def vendor_kind():
        if fact["evidence_level"] == "vendor_claim":
            require(value["claim_kind"] == "vendor_claim", "vendor_claim_presented_as_fact")

    source = {"url": value["url"], "publisher": value["publisher"], "publication_date": value["source_date"],
              "source_checked_at": value["source_checked_at"], "revalidated_at": value["revalidated_at"],
              "classification": value["classification"], "quote": value["quote"]}
    binding = next((f"/{name}" for name, expected in (("claim", fact.get("statement")), ("evidence_level", fact.get("evidence_level")),
                                                       ("snapshot_loaded_at", context.get("snapshot_loaded_at")))
                    if value[name] != expected), "/claim")
    for pointer, check in (
            ("/snapshot_fact_id", usable),
            (binding, lambda: require(value["claim"] == fact["statement"] and value["evidence_level"] == fact["evidence_level"]
                                      and value["snapshot_loaded_at"] == context["snapshot_loaded_at"], "cached_fact_binding_invalid")),
            ("/quote", lambda: require(source in fact["sources"], "cached_source_binding_invalid")),
            ("/claim_kind", lambda: require(value["claim_kind"] != "hypothesis", "cached_fact_binding_invalid")),
            ("/claim_kind", vendor_kind)):
        code = probe(check, collect)
        if code:
            yield pointer, code


def deltas(values, day, context, observed_at=None, contract_version=2):
    for _pointer, code in delta_issues(values, day, context, observed_at, contract_version):
        raise SnapshotError(code)


def delta_issues(values, day, context, observed_at=None, contract_version=2, *, collect=False):
    """Every proposal rule as (pointer relative to the list, code), in strict order."""
    code = probe(lambda: require(isinstance(values, list) and len(values) <= 10, "knowledge_deltas_invalid"), collect)
    if code:
        yield "", code
        return
    age_reason = "refresh_due" if contract_version == 3 else "stale"
    for index, delta in enumerate(values):
        base = f"/{index}"
        code = probe(lambda: shape(delta, set(DELTA_FIELDS)), collect)
        if code:
            yield base, code
            continue

        def binding():
            if delta["reason"] == "discovery":
                require(delta["record_id"] is None and delta["fact_id"] is None, "knowledge_delta_binding_invalid")
            else:
                lookup(context, delta["record_id"], delta["fact_id"])

        for pointer, check in (
                ("/reason", lambda: require(delta["reason"] in {"gap", "conflict", age_reason, "unsupported", "discovery", "consequential"},
                                            "knowledge_delta_reason_invalid")),
                ("/record_id", binding),
                ("/proposed_statement", lambda: text(delta["proposed_statement"]))):
            code = probe(check, collect)
            if code:
                yield base + pointer, code
        unknowns = delta["unknowns"]
        code = probe(lambda: require(isinstance(unknowns, list) and 1 <= len(unknowns) <= 20, "knowledge_delta_unknowns_required"), collect)
        if code:
            yield base + "/unknowns", code
        else:
            for position, unknown in enumerate(unknowns):
                code = probe(lambda: text(unknown), collect)
                if code:
                    yield f"{base}/unknowns/{position}", code
        items = delta["evidence"]
        code = probe(lambda: require(isinstance(items, list) and 1 <= len(items) <= 4, "knowledge_delta_evidence_required"), collect)
        if code:
            yield base + "/evidence", code
            continue
        for position, item in enumerate(items):
            prefix = f"{base}/evidence/{position}"
            code = probe(lambda: shape(item, set(DELTA_EVIDENCE_FIELDS), {"assertion_scope"} if contract_version == 3 else ()), collect)
            if code:
                yield prefix, code
                continue
            checks = []
            # v3's scope describes this proposed claim; it does not approve it
            # or rewrite the reviewed snapshot. Older scope-free deltas remain valid.
            if "assertion_scope" in item:
                checks.append(("/assertion_scope", lambda: require(isinstance(item["assertion_scope"], str) and item["assertion_scope"] in SCOPES,
                                                                   "knowledge_delta_assertion_scope_invalid")))
            classified = isinstance(item["classification"], str) and item["classification"] in CLASSIFICATIONS
            checks += [
                ("/url", lambda: url(item["url"])),
                ("/publisher", lambda: text(item["publisher"], 200)),
                ("/quote", lambda: text(item["quote"])),
                ("/source_checked_at", lambda: require(checked_day(item["source_checked_at"]) == day, "delta_live_evidence_required")),
                ("/source_checked_at", lambda: require(source_moment(item["source_checked_at"]) <= (observed_at or datetime.now(timezone.utc)),
                                                       "evidence_date_in_future")),
                ("/evidence_level" if classified else "/classification",
                 lambda: require(item["classification"] in CLASSIFICATIONS and item["evidence_level"] in LEVELS, "knowledge_delta_evidence_invalid")),
                ("/publication_date", lambda: item["publication_date"] is None
                 or require(calendar_date(item["publication_date"]) <= date.fromisoformat(day), "source_date_in_future")),
            ]
            for pointer, check in checks:
                code = probe(check, collect)
                if code:
                    yield prefix + pointer, code
