"""Offline, digest-bound code exception for the accepted first pilot search."""

import re

from experiments.provider_eval_recovery.harness import Ledger, digest, read_json, write_once
from .protocol import (CURRENT_DATE, LIMITS, PROTOCOL, SEEDS, code_hash, decision, evidence,
                       model_envelopes, model_input, search_request)
from .citations import audit_urls

BASE_CODE = "5cc4c0c4b1da09a0698b3661e658e3bb81ef42235e422b2fc92beea411186118"
CELL = "01_parallel_fast"
RECEIPT_NAME = "citation_continuation.json"
FOLLOWUP_RECEIPT_NAME = "followup_citation_continuation.json"
FIRST_PATCH_CODE = "02c42b29999a6403588a023acae8abcfba94488a12dbd36cc224682609cb74c0"


class ContinuationError(RuntimeError):
    pass


def expected_receipt(root, paths, plan, public, attempt, retained_sha256, *, patched_code=None):
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
            "base_code_sha256": BASE_CODE, "patched_code_sha256": patched_code or code_hash(), "cell": CELL,
            "attempt_id": attempt, "retained_envelope_sha256": retained_sha256,
            "preserved_failure_sha256": digest(failure), "normalized_evidence_sha256": digest(normalized),
            "citation_url_audit": audit_urls("parallel_fast", retained["raw"]),
            "journal_mutation": False, "search_redispatch": False}


def authorize(root, paths, plan, public, attempt, retained_sha256):
    """Caller holds the existing exclusive journal lock and verifies sole owner."""
    if (paths / RECEIPT_NAME).exists() and attempt != read_json(paths / RECEIPT_NAME)["attempt_id"]:
        receipt = expected_followup(root, paths, plan, public, attempt, retained_sha256)
        ledger = Ledger(root / "live_journal.jsonl", "10.00")
        adaptive = adaptive_reservations(ledger, plan)
        if {event["attempt_id"] for event in adaptive} != set(receipt["completed_attempts"]):
            raise ContinuationError("followup_creation_requires_only_four_completed_steps")
        if (paths / "followup_receipts").exists() or (paths / "reviews").exists():
            raise ContinuationError("followup_creation_refused_existing_later_artifacts")
        write_once(paths / FOLLOWUP_RECEIPT_NAME, receipt)
        return
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
    if (paths / FOLLOWUP_RECEIPT_NAME).exists():
        saved = read_json(paths / FOLLOWUP_RECEIPT_NAME)
        expected = expected_followup(root, paths, plan, public, saved["attempt_id"], saved["retained_envelope_sha256"])
        if saved != expected:
            raise ContinuationError("followup_citation_continuation_integrity_failure")
        return
    saved = read_json(paths / RECEIPT_NAME)
    expected = expected_receipt(root, paths, plan, public, saved["attempt_id"], saved["retained_envelope_sha256"])
    if saved != expected:
        raise ContinuationError("accepted_citation_continuation_integrity_failure")


def receipt_path(paths, cell):
    if cell == CELL and (paths / FOLLOWUP_RECEIPT_NAME).exists():
        return paths / "followup_receipts" / (cell + ".json")
    if cell == CELL and (paths / RECEIPT_NAME).exists():
        return paths / "continued_receipts" / (cell + ".json")
    return paths / "receipts" / (cell + ".json")


def adaptive_reservations(ledger, plan):
    return [event for event in ledger.events if event["kind"] == "reserved" and
            (event.get("protocol") == PROTOCOL or event.get("plan_sha256") == digest(plan)
             or event.get("role", "").startswith(PROTOCOL + ":"))]


def expected_followup(root, paths, plan, public, attempt, retained_sha256):
    """Verify the frozen first-search proof and exact four-step dependency chain."""
    from .runner import model_text
    first_proof = read_json(paths / RECEIPT_NAME)
    first_expected = expected_receipt(root, paths, plan, public, first_proof["attempt_id"],
                                     first_proof["retained_envelope_sha256"], patched_code=FIRST_PATCH_CODE)
    if first_proof != first_expected:
        raise ContinuationError("original_first_search_continuation_integrity_failure")
    ledger = Ledger(root / "live_journal.jsonl", "10.00")
    completed, envelopes = {}, {}

    def retained(step, request, provider):
        key = digest({"plan": digest(plan), "cell": CELL, "step": step, "request": request})
        rows = [row for row in ledger.events if row["kind"] == "reserved" and row["attempt_id"] == key]
        done = [row for row in ledger.events if row["kind"] == "completed" and row["attempt_id"] == key]
        if (len(rows) != 1 or len(done) != 1 or ledger.states.get(key) != "completed"
                or rows[0].get("plan_sha256") != digest(plan) or rows[0].get("protocol") != PROTOCOL
                or rows[0].get("cell") != CELL or rows[0].get("step") != step
                or rows[0].get("role") != PROTOCOL + ":" + step or rows[0].get("provider") != provider
                or rows[0].get("request_sha256") != digest(request)):
            raise ContinuationError("exact_completed_four_step_chain_required")
        envelope = read_json(paths / "raw" / (key + ".json"))
        if envelope.get("request") != request or digest(envelope) != done[0]["raw_sha256"]:
            raise ContinuationError("four_step_retained_request_or_envelope_digest_failure")
        completed[key] = digest(envelope)
        envelopes[step] = envelope
        return key, envelope["raw"]

    case = public["cases"][0]
    _, first = retained("search1", search_request(1, "parallel_fast", case, SEEDS[0]), "parallel")
    sources = evidence("parallel_fast", [first], 2000)
    items = model_input("assess", case, public["common_prompt"], sources, query=SEEDS[0])
    count_request, assessment_request = model_envelopes("assess", items)
    _, count = retained("assess_count", count_request, "openai")
    _, assessment = retained("assess", assessment_request, "openai")
    tokens = count.get("input_tokens")
    if (type(tokens) is not int or not 0 < tokens <= LIMITS["assess"][0]
            or assessment.get("usage", {}).get("input_tokens") != tokens):
        raise ContinuationError("retained_count_and_assessment_usage_must_agree")
    coverage = decision(model_text(assessment, "assess"), 1, SEEDS[0])
    if not coverage["needs_more"]:
        raise ContinuationError("retained_assessment_must_authorize_followup")
    key, followup = retained("search2", search_request(1, "parallel_fast", case, coverage["query"]), "parallel")
    if attempt != key or retained_sha256 != completed[key]:
        raise ContinuationError("exact_completed_followup_attempt_and_envelope_required")
    failure = read_json(paths / "continued_receipts" / (CELL + ".json"))
    expected_failure = {"protocol": PROTOCOL, "case_id": "BP-EVAL-01", "cell": CELL, "mode": "parallel_fast",
        "status": "unknown_local_citation_normalization_error_not_provider_warning",
        "answer": "Unknown: no supported answer within this protocol.", "sources": sources, "operational_coverage": coverage,
        "grade": "pending isolated reviewer; controller triage is not ground truth",
        "retained_attempts": completed, "current_date": CURRENT_DATE}
    if failure != expected_failure:
        raise ContinuationError("exact_preserved_followup_failure_receipt_required")
    normalized = evidence("parallel_fast", [first, followup], 5500)
    if not normalized:
        raise ContinuationError("followup_has_no_safe_public_evidence")
    return {"schema": "accepted_followup_citation_continuation.v1", "base_plan_sha256": digest(plan),
        "base_code_sha256": BASE_CODE, "prior_patched_code_sha256": FIRST_PATCH_CODE,
        "patched_code_sha256": code_hash(), "first_continuation_sha256": digest(first_proof),
        "preserved_first_failure_sha256": first_proof["preserved_failure_sha256"],
        "preserved_followup_failure_sha256": digest(failure), "cell": CELL,
        "attempt_id": attempt, "retained_envelope_sha256": retained_sha256,
        "completed_attempts": completed, "retained_input_tokens": tokens,
        "operational_coverage": coverage, "normalized_evidence_sha256": digest(normalized),
        "citation_url_audit": audit_urls("parallel_fast", followup),
        "journal_mutation": False, "search_redispatch": False, "assessment_redispatch": False}
