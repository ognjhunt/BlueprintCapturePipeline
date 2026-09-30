"""Offline, digest-bound code exception for the accepted first pilot search."""

import re

from experiments.provider_eval_recovery.harness import Ledger, digest, read_json, write_once
from .protocol import CURRENT_DATE, PROTOCOL, SEEDS, code_hash, evidence, search_request
from .citations import audit_urls

BASE_CODE = "5cc4c0c4b1da09a0698b3661e658e3bb81ef42235e422b2fc92beea411186118"
CELL = "01_parallel_fast"
RECEIPT_NAME = "citation_continuation.json"


class ContinuationError(RuntimeError):
    pass


def expected_receipt(root, paths, plan, public, attempt, retained_sha256):
    if (plan.get("code_sha256") != BASE_CODE
            or not all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
                       for value in (attempt, retained_sha256))):
        raise ContinuationError("exact_reviewed_completed_search_continuation_required")
    envelope = search_request(1, "parallel_fast", public["cases"][0], SEEDS[0])
    expected_key = digest({"plan": digest(plan), "cell": CELL, "step": "search1", "request": envelope})
    ledger = Ledger(root / "live_journal.jsonl", "10.00")
    reservation = next((event for event in ledger.events
                        if event["kind"] == "reserved" and event["attempt_id"] == attempt), {})
    completion = next((event for event in ledger.events
                       if event["kind"] == "completed" and event["attempt_id"] == attempt), {})
    if (attempt != expected_key or ledger.states.get(attempt) != "completed"
            or reservation.get("protocol") != PROTOCOL or reservation.get("plan_sha256") != digest(plan)
            or reservation.get("cell") != CELL or reservation.get("provider") != "parallel"
            or reservation.get("step") != "search1" or reservation.get("role") != PROTOCOL + ":search1"
            or reservation.get("request_sha256") != digest(envelope)
            or completion.get("raw_sha256") != retained_sha256):
        raise ContinuationError("exact_completed_first_pilot_search_required")
    retained = read_json(paths / "raw" / (attempt + ".json"))
    if digest(retained) != retained_sha256 or retained.get("request") != envelope:
        raise ContinuationError("accepted_search_envelope_or_request_digest_failure")
    failure = read_json(paths / "receipts" / (CELL + ".json"))
    expected_failure = {"protocol": PROTOCOL, "case_id": "BP-EVAL-01", "cell": CELL, "mode": "parallel_fast",
        "status": "unknown_contract_or_evidence_stop_not_provider_quality",
        "answer": "Unknown: no supported answer within this protocol.", "sources": [], "operational_coverage": None,
        "grade": "pending isolated reviewer; controller triage is not ground truth",
        "retained_attempts": {attempt: retained_sha256}, "current_date": CURRENT_DATE}
    if failure != expected_failure:
        raise ContinuationError("exact_preserved_local_normalizer_failure_receipt_required")
    # Provider warnings and failed normalization must be resolved before any new
    # paid stage. This remains an offline verification, never an API retry.
    normalized = evidence("parallel_fast", [retained["raw"]], 5500)
    if not normalized:
        raise ContinuationError("accepted_search_has_no_normalized_public_evidence")
    return {"schema": "accepted_citation_continuation.v1", "base_plan_sha256": digest(plan),
            "base_code_sha256": BASE_CODE, "patched_code_sha256": code_hash(), "cell": CELL,
            "attempt_id": attempt, "retained_envelope_sha256": retained_sha256,
            "preserved_failure_sha256": digest(failure), "normalized_evidence_sha256": digest(normalized),
            "citation_url_audit": audit_urls("parallel_fast", retained["raw"]),
            "journal_mutation": False, "search_redispatch": False}


def authorize(root, paths, plan, public, attempt, retained_sha256):
    """Caller holds the existing exclusive journal lock and verifies sole owner."""
    receipt = expected_receipt(root, paths, plan, public, attempt, retained_sha256)
    ledger = Ledger(root / "live_journal.jsonl", "10.00")
    adaptive = [event for event in ledger.events if event["kind"] == "reserved" and
                (event.get("protocol") == PROTOCOL or event.get("plan_sha256") == digest(plan)
                 or event.get("role", "").startswith(PROTOCOL + ":"))]
    if len(adaptive) != 1 or adaptive[0]["attempt_id"] != attempt:
        raise ContinuationError("continuation_creation_requires_only_accepted_search_attempt")
    if (paths / "continued_receipts").exists() or (paths / "reviews").exists():
        raise ContinuationError("continuation_creation_refused_existing_later_artifacts")
    write_once(paths / RECEIPT_NAME, receipt)


def verify(root, paths, plan, public):
    saved = read_json(paths / RECEIPT_NAME)
    expected = expected_receipt(root, paths, plan, public, saved["attempt_id"], saved["retained_envelope_sha256"])
    if saved != expected:
        raise ContinuationError("accepted_citation_continuation_integrity_failure")


def receipt_path(paths, cell):
    if cell == CELL and (paths / RECEIPT_NAME).exists():
        return paths / "continued_receipts" / (cell + ".json")
    return paths / "receipts" / (cell + ".json")
