"""Bounded, read-only reviewed Notion mirror; no connector or model calls.

Content hashes provide integrity and binding, not reviewer authenticity. Only an
owner-reviewed export belongs at the configured path. Fixture exports are synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import re
from datetime import date, datetime, time, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "blueprint.knowledge-snapshot.v1"
MAX_BYTES = 262_144
MAX_CONTEXT_BYTES = 32_768
LIVE_FIELDS = {"availability", "geography", "deployment", "integration", "safety", "support", "price", "supervision"}
LEVELS = {"vendor_claim", "demonstrated_capability", "named_deployment", "current_availability", "unknown"}
FIELDS = {"task_claim", "specification", "embodiment", "supported_hardware", "limit", "unknown", "reliability", "cycle_time", *LIVE_FIELDS}


class SnapshotError(ValueError):
    """Stable refusal codes; do not reflect source text."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def content_hash(value):
    payload = {k: v for k, v in value.items() if k != "content_hash"}
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def require(predicate, code="knowledge_schema_invalid"):
    if not predicate:
        raise SnapshotError(code)


def text(value, maximum=2000):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value) is not None)


def timestamp(value):
    require(isinstance(value, str))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SnapshotError("knowledge_date_invalid") from None
    require(parsed.tzinfo is not None, "knowledge_date_invalid")
    return parsed


def source_moment(value):
    # Date-only reviews keep their honest granularity. Midnight is used only
    # for conservative freshness arithmetic, never emitted as provenance.
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return datetime.combine(date.fromisoformat(value), time.min, ZoneInfo("America/Chicago"))
    return timestamp(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "knowledge_duplicate_json_key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def url(value):
    text(value)
    parsed = urlsplit(value)
    try:
        ipaddress.ip_address(parsed.hostname or "")
    except ValueError:
        pass
    else:
        raise SnapshotError("knowledge_url_invalid")
    require(parsed.scheme == "https" and bool(parsed.hostname) and "." in parsed.hostname
            and not parsed.username and not parsed.password, "knowledge_url_invalid")


def shape(value, required, optional=()):
    require(isinstance(value, dict) and required <= set(value) <= required | set(optional))


def strings(value, maximum=20):
    require(isinstance(value, list) and len(value) <= maximum)
    for entry in value:
        text(entry, 120)
    require(len(value) == len(set(value)))


def _validate(value, now):
    shape(value, {"schema_version", "exported_at", "source_pages", "companies", "records", "content_hash"})
    require(value["schema_version"] == SCHEMA_VERSION, "knowledge_version_unsupported")
    exported = timestamp(value["exported_at"])
    require(exported <= now, "knowledge_date_in_future")
    pages = value["source_pages"]
    require(isinstance(pages, list) and 1 <= len(pages) <= 30)
    page_ids = set()
    for page in pages:
        shape(page, {"page_id", "url", "revision"})
        identifier(page["page_id"])
        require(page["page_id"] not in page_ids)
        page_ids.add(page["page_id"])
        url(page["url"])
        require(urlsplit(page["url"]).hostname in {"notion.so", "www.notion.so", "app.notion.com"})
        shape(page["revision"], {"kind", "value"})
        require(page["revision"]["kind"] in {"native_revision_id", "page_last_edited_at"})
        if page["revision"]["kind"] == "page_last_edited_at":
            require(timestamp(page["revision"]["value"]) <= exported, "knowledge_date_in_future")
        else:
            text(page["revision"]["value"], 200)
    companies = value["companies"]
    require(isinstance(companies, list) and len(companies) <= 100)
    company_ids = set()
    for company in companies:
        shape(company, {"company_id", "name", "roles", "source_page_ids"})
        identifier(company["company_id"])
        require(company["company_id"] not in company_ids)
        company_ids.add(company["company_id"])
        text(company["name"], 200)
        strings(company["roles"])
        strings(company["source_page_ids"])
        require(bool(company["source_page_ids"]) and set(company["source_page_ids"]) <= page_ids)
    records = value["records"]
    require(isinstance(records, list) and len(records) <= 100)
    record_ids = set()
    for record in records:
        shape(record, {"record_id", "company_id", "record_type", "product", "task_tags", "geography_tags", "facts"})
        identifier(record["record_id"])
        require(record["record_id"] not in record_ids and record["company_id"] in company_ids)
        record_ids.add(record["record_id"])
        identifier(record["record_type"])  # Extensible robot/policy/software/world_model/vertical roles.
        shape(record["product"], {"name", "version"})
        text(record["product"]["name"], 200)
        if record["product"]["version"] is not None:
            text(record["product"]["version"], 120)
        strings(record["task_tags"])
        strings(record["geography_tags"])
        require(isinstance(record["facts"], list) and len(record["facts"]) <= 20)
        fact_ids = set()
        for fact in record["facts"]:
            shape(fact, {"fact_id", "field", "statement", "status", "evidence_level", "confidence",
                         "freshness_days", "task_tags", "geography_tags", "source_page_ids", "sources", "limits", "conflicts"}, {"specification"})
            identifier(fact["fact_id"])
            require(fact["fact_id"] not in fact_ids)
            fact_ids.add(fact["fact_id"])
            require(fact["field"] in FIELDS and fact["status"] in {"reviewed", "conflicted", "unsupported", "unknown"}
                    and fact["evidence_level"] in LEVELS and fact["confidence"] in {"low", "medium", "high", "unknown"})
            text(fact["statement"])
            require(type(fact["freshness_days"]) is int and 1 <= fact["freshness_days"] <= 90)
            for key in ("task_tags", "geography_tags", "source_page_ids", "limits", "conflicts"):
                strings(fact[key])
            require(bool(fact["source_page_ids"]) and set(fact["source_page_ids"]) <= page_ids)
            require(fact["status"] != "conflicted" or bool(fact["conflicts"]))
            require(fact["status"] != "reviewed" or not fact["conflicts"])
            require(isinstance(fact["sources"], list) and len(fact["sources"]) <= 4)
            if fact["status"] == "reviewed":
                require(bool(fact["sources"]) and fact["evidence_level"] != "unknown")
            for source in fact["sources"]:
                shape(source, {"url", "publisher", "publication_date", "source_checked_at", "revalidated_at", "classification", "quote"})
                url(source["url"])
                text(source["publisher"], 200)
                if source["quote"] is not None:
                    text(source["quote"])
                require(source["classification"] in {"operator", "vendor", "independent"})
                checked = source_moment(source["source_checked_at"])
                require(checked <= exported, "knowledge_date_in_future")
                if source["publication_date"] is not None:
                    require(isinstance(source["publication_date"], str))
                    require(date.fromisoformat(source["publication_date"]) <= checked.date(), "knowledge_date_in_future")
                if source["revalidated_at"] is not None:
                    require(checked <= source_moment(source["revalidated_at"]) <= exported, "knowledge_date_invalid")
            if "specification" in fact:
                spec = fact["specification"]
                shape(spec, {"name", "value", "unit", "conditions"})
                require(fact["field"] == "specification")
                text(spec["name"], 120)
                require(type(spec["value"]) in {float, int} and math.isfinite(spec["value"]))
                text(spec["unit"], 80)
                strings(spec["conditions"])
                require(bool(spec["conditions"]))
    require(isinstance(value["content_hash"], str) and value["content_hash"] == content_hash(value), "knowledge_hash_mismatch")
    require(len(canonical(value).encode()) <= MAX_BYTES, "knowledge_snapshot_too_large")
    return value


def validate(value, now):
    require(now.tzinfo is not None, "knowledge_date_invalid")
    try:
        return _validate(value, now)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError("knowledge_schema_invalid") from None


def load(path, now):
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "knowledge_snapshot_too_large")
        return validate(parse(raw), now)
    except OSError:
        raise SnapshotError("knowledge_snapshot_missing") from None
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError("knowledge_json_invalid") from None


def fact_state(fact, now):
    if fact["status"] != "reviewed":
        return fact["status"]
    if fact["field"] in LIVE_FIELDS or fact["evidence_level"] == "current_availability":
        return "live_required"
    # All cited support must meet the declared field freshness envelope.
    if any((now - source_moment(s["revalidated_at"] or s["source_checked_at"])).total_seconds()
           > fact["freshness_days"] * 86400 for s in fact["sources"]):
        return "stale"
    return "usable_background"


def select(snapshot, now, filters=None):
    filters = filters or {}
    shape(filters, set(), {"company_ids", "record_ids", "task_tags", "geography_tags"})
    for values in filters.values():
        strings(values)
    records = []
    for record in snapshot["records"]:
        if any(filters.get(key) and record[field] not in filters[key]
               for key, field in (("company_ids", "company_id"), ("record_ids", "record_id"))):
            continue
        facts = []
        for fact in record["facts"]:
            if filters.get("task_tags") and not set(filters["task_tags"]) & set(fact["task_tags"] or record["task_tags"]):
                continue
            geography = fact["geography_tags"] or record["geography_tags"]
            # Unknown geography is a research gap, never service eligibility.
            if (filters.get("geography_tags") and geography and not set(filters["geography_tags"]) & set(geography)
                    and not (fact["field"] == "geography" and fact["status"] == "unknown")):
                continue
            facts.append({**fact, "load_state": fact_state(fact, now)})
        if facts:
            records.append({**record, "facts": facts})
    require(len(records) <= 12 and sum(len(r["facts"]) for r in records) <= 60, "knowledge_filter_required")
    companies = [c for c in snapshot["companies"] if c["company_id"] in {r["company_id"] for r in records}]
    selected_page_ids = {page_id for company in companies for page_id in company["source_page_ids"]}
    selected_page_ids.update(page_id for record in records for fact in record["facts"] for page_id in fact["source_page_ids"])
    context = {"schema_version": snapshot["schema_version"], "content_hash": snapshot["content_hash"],
               "exported_at": snapshot["exported_at"], "snapshot_loaded_at": now.isoformat(),
               "source_pages": [page for page in snapshot["source_pages"] if page["page_id"] in selected_page_ids],
               "companies": companies, "records": records}
    require(len(canonical(context).encode()) <= MAX_CONTEXT_BYTES, "knowledge_context_too_large")
    return context


def export(reviewed, path, now):
    """Local writer only: explicit reviewed JSON input; never fetch or infer claims."""
    value = {**reviewed, "schema_version": SCHEMA_VERSION, "exported_at": now.isoformat()}
    value["content_hash"] = content_hash(value)
    validate(value, now)
    Path(path).write_text(canonical(value) + "\n", encoding="utf-8")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_input")
    parser.add_argument("output")
    args = parser.parse_args()
    # Export input is bounded, even before validation. No private CRM/contact fields.
    with open(args.reviewed_input, "rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, "knowledge_snapshot_too_large")
    export(parse(raw), args.output, datetime.now(timezone.utc))


if __name__ == "__main__":
    main()
