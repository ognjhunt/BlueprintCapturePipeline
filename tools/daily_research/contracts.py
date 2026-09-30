"""Research v2 provenance and proposed-delta validation, without sink writes."""
from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from tools.daily_research.knowledge import (
    LEVELS,
    SnapshotError,
    require,
    shape,
    source_moment,
    text,
    timestamp,
    url,
)

CENTRAL = ZoneInfo("America/Chicago")
EVIDENCE_V2 = {"origin", "evidence_level", "source_checked_at", "snapshot_loaded_at", "revalidated_at", "snapshot_record_id", "snapshot_fact_id"}


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


def evidence(value, day, context, observed_at=None):
    require(value["evidence_level"] in LEVELS, "evidence_level_invalid")
    require(value["evidence_level"] != "unknown", "unsupported_evidence_level")
    require(value["checked_date"] == checked_day(value["source_checked_at"]), "evidence_date_integrity_invalid")
    require(date.fromisoformat(value["checked_date"]) <= date.fromisoformat(day), "evidence_date_in_future")
    require(source_moment(value["source_checked_at"]) <= timestamp(context["snapshot_loaded_at"])
            or value["origin"] == "live", "evidence_date_in_future")
    if value["origin"] == "live":
        require(source_moment(value["source_checked_at"]) <= (observed_at or datetime.now(timezone.utc)), "evidence_date_in_future")
        require(value["checked_date"] == day and value["snapshot_loaded_at"] is None
                and value["snapshot_record_id"] is None and value["snapshot_fact_id"] is None,
                "live_evidence_binding_invalid")
        require(value["revalidated_at"] in {None, value["source_checked_at"]}, "evidence_date_integrity_invalid")
        return
    require(value["origin"] == "snapshot", "evidence_origin_invalid")
    require(value["role"] == "capability", "live_task_geography_required")
    fact = lookup(context, value["snapshot_record_id"], value["snapshot_fact_id"])
    require(fact["load_state"] == "usable_background", "cached_fact_not_usable")
    require(value["claim"] == fact["statement"] and value["evidence_level"] == fact["evidence_level"]
            and value["snapshot_loaded_at"] == context["snapshot_loaded_at"], "cached_fact_binding_invalid")
    source = {"url": value["url"], "publisher": value["publisher"], "publication_date": value["source_date"],
              "source_checked_at": value["source_checked_at"], "revalidated_at": value["revalidated_at"],
              "classification": value["classification"], "quote": value["quote"]}
    require(source in fact["sources"], "cached_source_binding_invalid")
    require(value["claim_kind"] != "hypothesis", "cached_fact_binding_invalid")
    if fact["evidence_level"] == "vendor_claim":
        require(value["claim_kind"] == "vendor_claim", "vendor_claim_presented_as_fact")


def deltas(values, day, context, observed_at=None):
    require(isinstance(values, list) and len(values) <= 10, "knowledge_deltas_invalid")
    for delta in values:
        shape(delta, {"record_id", "fact_id", "reason", "proposed_statement", "evidence", "unknowns"})
        require(delta["reason"] in {"gap", "conflict", "stale", "unsupported", "discovery", "consequential"}, "knowledge_delta_reason_invalid")
        if delta["reason"] == "discovery":
            require(delta["record_id"] is None and delta["fact_id"] is None, "knowledge_delta_binding_invalid")
        else:
            lookup(context, delta["record_id"], delta["fact_id"])
        text(delta["proposed_statement"])
        require(isinstance(delta["unknowns"], list) and 1 <= len(delta["unknowns"]) <= 20, "knowledge_delta_unknowns_required")
        for unknown in delta["unknowns"]:
            text(unknown)
        require(isinstance(delta["evidence"], list) and 1 <= len(delta["evidence"]) <= 4, "knowledge_delta_evidence_required")
        for item in delta["evidence"]:
            shape(item, {"url", "publisher", "publication_date", "source_checked_at", "classification", "evidence_level", "quote"})
            url(item["url"])
            text(item["publisher"], 200)
            text(item["quote"])
            require(checked_day(item["source_checked_at"]) == day, "delta_live_evidence_required")
            require(source_moment(item["source_checked_at"]) <= (observed_at or datetime.now(timezone.utc)), "evidence_date_in_future")
            require(item["classification"] in {"operator", "vendor", "independent"} and item["evidence_level"] in LEVELS,
                    "knowledge_delta_evidence_invalid")
            if item["publication_date"] is not None:
                require(date.fromisoformat(item["publication_date"]) <= date.fromisoformat(day), "source_date_in_future")
