"""Explicit approved refresh overlay for research v3; no source revalidation.

Thresholds prioritize review, never invalidate dated background solely by age.
Approval references record authority; hashes bind bytes, not reviewer identity.
"""
from __future__ import annotations

import hashlib

from tools.daily_research import knowledge as k

VERSION = "blueprint.knowledge-refresh-policy.v1"
MAX_BYTES = 65_536
CLASSES = {
    "stable_versioned_embodiment_or_specification": 90,
    "vendor_capability_or_limit": 30,
    "dated_historical_report": 90,
    "operational_status_or_requirements": 7,
    "unresolved_conflict": None,
    "explicit_unknown": None,
}
ANNOTATIONS = {"load_state", "refresh_class", "refresh_due", "reuse_mode"}


def policy_hash(value):
    return hashlib.sha256(k.canonical({key: item for key, item in value.items() if key != "policy_hash"}).encode()).hexdigest()


def fact_hash(fact):
    return hashlib.sha256(k.canonical({key: item for key, item in fact.items() if key not in ANNOTATIONS}).encode()).hexdigest()


def hash_string(value):
    k.require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "refresh_policy_hash_invalid")


def check_class(fact, policy_class):
    k.require(policy_class in CLASSES, "refresh_policy_class_unsupported")
    # Original source fields/status override labels, even in an approved overlay.
    if fact["status"] == "conflicted" or fact["conflicts"]:
        allowed = {"unresolved_conflict"}
    elif fact["status"] in {"unknown", "unsupported"} or fact["field"] == "unknown" or fact["evidence_level"] == "unknown":
        allowed = {"explicit_unknown"}
    elif fact["evidence_level"] == "current_availability":
        allowed = {"operational_status_or_requirements"}
    elif fact["field"] == "deployment":
        allowed = {"dated_historical_report", "operational_status_or_requirements"}
    elif fact["field"] in k.LIVE_FIELDS:
        allowed = {"operational_status_or_requirements"}
    elif fact["field"] in {"specification", "embodiment", "supported_hardware"}:
        allowed = {"stable_versioned_embodiment_or_specification"}
    else:
        allowed = {"vendor_capability_or_limit"}
    k.require(policy_class in allowed, "refresh_policy_unsafe_classification")


def validate(policy, snapshot, file_sha256, now):
    try:
        k.shape(policy, {"schema_version", "approved_at", "approval_reference", "snapshot_content_hash",
                         "snapshot_file_sha256", "classes", "assignments", "policy_hash"})
        k.require(policy["schema_version"] == VERSION, "refresh_policy_version_unsupported")
        k.require(k.timestamp(policy["approved_at"]) <= now, "refresh_policy_date_in_future")
        k.text(policy["approval_reference"], 300)
        k.require(not policy["approval_reference"].startswith("PENDING"), "refresh_policy_approval_missing")
        for key in ("snapshot_content_hash", "snapshot_file_sha256", "policy_hash"):
            hash_string(policy[key])
        k.require(policy["snapshot_content_hash"] == snapshot["content_hash"]
                  and policy["snapshot_file_sha256"] == file_sha256, "refresh_policy_snapshot_binding_invalid")
        k.require(policy["classes"] == CLASSES and all(type(v) is type(CLASSES[key]) for key, v in policy["classes"].items()),
                  "refresh_policy_thresholds_unsupported")
        facts = {(r["record_id"], f["fact_id"]): f for r in snapshot["records"] for f in r["facts"]}
        k.require(isinstance(policy["assignments"], list) and len(policy["assignments"]) <= 200, "refresh_policy_assignments_invalid")
        seen = set()
        for assignment in policy["assignments"]:
            k.shape(assignment, {"record_id", "fact_id", "policy_class", "fact_hash"})
            k.identifier(assignment["record_id"])
            k.identifier(assignment["fact_id"])
            key = (assignment["record_id"], assignment["fact_id"])
            k.require(key in facts and key not in seen, "refresh_policy_assignment_binding_invalid")
            seen.add(key)
            k.require(assignment["fact_hash"] == fact_hash(facts[key]), "refresh_policy_fact_binding_invalid")
            check_class(facts[key], assignment["policy_class"])
        k.require(seen == set(facts), "refresh_policy_incomplete")
        k.require(policy["policy_hash"] == policy_hash(policy), "refresh_policy_hash_mismatch")
        k.require(len(k.canonical(policy).encode()) <= MAX_BYTES, "refresh_policy_too_large")
        return policy
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, k.SnapshotError):
            raise
        raise k.SnapshotError("refresh_policy_invalid") from None


def build(snapshot, file_sha256, assignments, approval_reference, approved_at):
    """Local artifact preparation from explicit approved classifications only."""
    facts = {(r["record_id"], f["fact_id"]): f for r in snapshot["records"] for f in r["facts"]}
    result = {"schema_version": VERSION, "approved_at": approved_at.isoformat(), "approval_reference": approval_reference,
              "snapshot_content_hash": snapshot["content_hash"], "snapshot_file_sha256": file_sha256,
              "classes": CLASSES.copy(), "assignments": []}
    for assignment in assignments:
        key = (assignment["record_id"], assignment["fact_id"])
        k.require(key in facts, "refresh_policy_assignment_binding_invalid")
        result["assignments"].append({**assignment, "fact_hash": fact_hash(facts[key])})
    result["policy_hash"] = policy_hash(result)
    return validate(result, snapshot, file_sha256, approved_at)


def read_bounded(path, maximum, missing_code):
    try:
        with open(path, "rb") as handle:
            raw = handle.read(maximum + 1)
    except OSError:
        raise k.SnapshotError(missing_code) from None
    k.require(len(raw) <= maximum, "refresh_input_too_large")
    return raw


def load(snapshot_path, policy_path, now):
    # One read binds parsed snapshot data and exact file bytes together.
    try:
        raw = read_bounded(snapshot_path, k.MAX_BYTES, "knowledge_snapshot_missing")
        snapshot = k.validate(k.parse(raw), now)
        policy = k.parse(read_bounded(policy_path, MAX_BYTES, "refresh_policy_missing"))
        return snapshot, validate(policy, snapshot, hashlib.sha256(raw).hexdigest(), now)
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, k.SnapshotError):
            raise
        raise k.SnapshotError("refresh_input_json_invalid") from None


def assignment_for(policy, record_id, fact):
    for assignment in policy["assignments"]:
        if (assignment["record_id"], assignment["fact_id"]) == (record_id, fact["fact_id"]):
            k.require(assignment["fact_hash"] == fact_hash(fact), "refresh_policy_fact_binding_invalid")
            check_class(fact, assignment["policy_class"])
            return assignment
    raise k.SnapshotError("refresh_policy_assignment_binding_invalid")


def refresh_due(fact, policy_class, now):
    threshold = CLASSES[policy_class]
    if threshold is None:
        return None  # Conflict/gap relevance, not an age-based review clock.
    return any((now - k.source_moment(s["revalidated_at"] or s["source_checked_at"])).total_seconds()
               >= threshold * 86400 for s in fact["sources"])


def reuse_mode(fact, policy_class):
    if fact["status"] == "conflicted" or fact["conflicts"]:
        return "conflict_guardrail"
    if fact["status"] != "reviewed" or fact["field"] == "unknown" or fact["evidence_level"] == "unknown":
        return "gap_only"
    if policy_class == "dated_historical_report":
        return "historical_background"
    if fact["field"] in k.LIVE_FIELDS or fact["evidence_level"] == "current_availability":
        return "live_required_context"
    return "dated_background"


def select(snapshot, policy, now, filters=None):
    context = k.select(snapshot, now, filters)
    context["refresh_policy"] = {key: policy[key] for key in ("schema_version", "policy_hash", "approval_reference", "classes")}
    for record in context["records"]:
        for fact in record["facts"]:
            policy_class = assignment_for(policy, record["record_id"], fact)["policy_class"]
            fact.update(refresh_class=policy_class, refresh_due=refresh_due(fact, policy_class, now), reuse_mode=reuse_mode(fact, policy_class))
    k.require(len(k.canonical(context).encode()) <= k.MAX_CONTEXT_BYTES, "knowledge_context_too_large")
    return context


def validate_context(context, policy):
    k.require(policy["schema_version"] == VERSION and policy["policy_hash"] == policy_hash(policy), "refresh_policy_hash_mismatch")
    k.require(policy["classes"] == CLASSES and policy["snapshot_content_hash"] == context["content_hash"], "refresh_policy_snapshot_binding_invalid")
    k.require(context["refresh_policy"] == {key: policy[key] for key in ("schema_version", "policy_hash", "approval_reference", "classes")},
              "refresh_policy_context_binding_invalid")
    for record in context["records"]:
        for fact in record["facts"]:
            assignment = assignment_for(policy, record["record_id"], fact)
            policy_class = assignment["policy_class"]
            k.require(fact["refresh_class"] == policy_class and fact["reuse_mode"] == reuse_mode(fact, policy_class)
                      and fact["refresh_due"] is refresh_due(fact, policy_class, k.timestamp(context["snapshot_loaded_at"])),
                      "refresh_policy_context_binding_invalid")


def cached_citation(fact, policy, record_id, role):
    policy_class = assignment_for(policy, record_id, fact)["policy_class"]
    k.require(reuse_mode(fact, policy_class) not in {"gap_only", "conflict_guardrail"}, "cached_fact_not_usable")
    # Historical/operational/specification/negative-limit facts are supplemental
    # background, never coverage for a positive task-capability match.
    if role == "capability":
        k.require(fact["field"] == "task_claim" and policy_class == "vendor_capability_or_limit"
                  and fact["field"] not in k.LIVE_FIELDS and fact["evidence_level"] != "current_availability",
                  "cached_positive_capability_not_supported")
    else:
        k.require(role == "background", "live_task_geography_required")


def assessment(context, policy, now):
    validate_context(context, policy)
    return {"as_of": now.isoformat(), "policy_hash": policy["policy_hash"], "facts": [
        {"record_id": r["record_id"], "fact_id": f["fact_id"], "refresh_class": (a := assignment_for(policy, r["record_id"], f))["policy_class"],
         "refresh_due": refresh_due(f, a["policy_class"], now), "reuse_mode": reuse_mode(f, a["policy_class"])}
        for r in context["records"] for f in r["facts"]]}
