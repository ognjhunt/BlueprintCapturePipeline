"""One-shot private live comparison CLI; preflight is network-free by default."""

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION, require_paid_resource_admission,
)

from .adapters import MODEL, MODES, normalize, request
from .harness import ROOT, Ledger, digest, exclusive, read_json, write_once
from .live_http import (COUNT_ALLOWANCE, COUNT_ENDPOINT, COUNT_METHOD, HTTPTransport,
                        LiveBlocked, PROJECT, credential_presence,
                        openai_envelope)
from .public_inputs import DECLARED_ORIGINAL_SHA256, load_public

OWNER = "01a0f3b8-6abe-775b-bfea-5102185b80ce"
CATALOG = "6abd5bd3b9f081a18f7946316d12ad95"
APPROVAL = "Sentinel_741e7736ff8c8191a085c8e2ece40576: yes i approve"


def code_hash():
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "live_rubric.md"]
    return digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def preflight(inputs, output, access, phase):
    """Validate all deterministic execute gates without contacting any provider."""
    output = Path(output).resolve()
    _, cases, provenance = load_public(inputs)
    if not provenance["byte_parity_with_original_public"]:
        raise LiveBlocked("exact_reviewed_public_bytes_required")
    if (access.get("execution_owner_task_id") != OWNER or access.get("catalog_version") != CATALOG
            or access.get("budget_approval") != APPROVAL
            or access.get("key_presence") != dict.fromkeys(("parallel", "perplexity", "openai"), True)
            or access.get("openai_model_http_status") != 200
            or access.get("verified_model") != MODEL
            or access.get("tokenizer_id") != COUNT_METHOD
            or access.get("prior_spend_usd") != "0.00" or access.get("prior_paid_calls") != 0
            or access.get("journal_root") != str(output)
            or not all(credential_presence().values())):
        raise LiveBlocked("fresh_single_owner_access_budget_or_zero_prior_spend_receipt_required")
    plan = {"execution": "live_authorized", "budget_usd": "10.00", "model": MODEL,
            "openai_project": PROJECT, "case_count": 20, "phase": phase,
            "journal_root": str(output), "public_inputs_sha256": DECLARED_ORIGINAL_SHA256,
            "reviewer_spec_sha256": hashlib.sha256((ROOT / "live_rubric.md").read_bytes()).hexdigest(),
            "reviewer_spec_provenance": "recovery_protocol; Library oracle/spec parent-side",
            "tokenizer_id": COUNT_METHOD, "code_sha256": code_hash(),
            "access_receipt_sha256": digest(access), "execution_owner_task_id": OWNER,
            "case_hashes": {f"{i:02d}": digest(case) for i, case in enumerate(cases, 1)},
            "inference_roles": ["synthesis"], "count_allowance_usd_per_request": str(COUNT_ALLOWANCE)}
    HTTPTransport(output, plan, access, token_counter=lambda _: None)
    with exclusive(output):
        write_once(output / "live_access.json", access)
        write_once(output / "public_provenance.json", provenance)
        write_once(output / "live_scope.json", {k: v for k, v in plan.items() if k != "phase"})
        ledger = Ledger(output / "live_journal.jsonl", "10.00")
        if phase == "remaining":
            for index in range(1, 3):
                for mode in MODES:
                    receipt_path = output / "live_receipts" / f"{index:02d}_{mode}.json"
                    if not receipt_path.exists():
                        raise LiveBlocked("complete_pilot_receipts_required_before_remaining")
                    verify_receipt(output, read_json(receipt_path), ledger,
                                   f"{index:02d}_{mode}", cases[index - 1])
    return plan, cases, provenance


def grant(plan, provider):
    resource = "openai_api_candidate" if provider == "openai" else "evaluator_api"
    return require_paid_resource_admission(
        {"schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "status": "admitted",
         "resource_class": resource, "blockers": [], "allocation_binding_digest": digest(plan)},
        resource_class=resource, expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)


def canonical_receipt(output, ledger, cell, case):
    mode = cell.split("_", 1)[1]
    expected_roles = {"search": mode.split("_", 1)[0], "count_synthesis": "openai",
                      "synthesis": "openai"}
    events = [e for e in ledger.events if e["kind"] == "reserved" and e.get("cell") == cell]
    if (len(events) != 3 or {e.get("role") for e in events} != set(expected_roles)
            or any(e.get("provider") != expected_roles[e["role"]]
                   or ledger.states[e["attempt_id"]] != "completed" for e in events)):
        raise LiveBlocked("exact_completed_pilot_roles_required")
    retained, responses = {}, {}
    for event in events:
        key = event["attempt_id"]
        raw = read_json(Path(output) / "live_raw" / (key + ".json"))
        completed = next(e for e in ledger.events if e["attempt_id"] == key and e["kind"] == "completed")
        if digest(raw) != completed["raw_sha256"] or raw["request_sha256"] != event["request_sha256"]:
            raise LiveBlocked("receipt_response_integrity_failure")
        retained[key], responses[event["role"]] = digest(raw), raw
    raw, counted, answer = (responses[role] for role in ("search", "count_synthesis", "synthesis"))
    answer_text = output_text(answer["raw"])
    if answer["raw"]["usage"]["input_tokens"] != counted["raw"].get("input_tokens"):
        raise LiveBlocked("returned_input_usage_disagrees_with_provider_count")
    return {"case_id": case["case_id"], "cell": cell, "mode": mode,
            "public_case_sha256": digest(case), "answer": answer_text,
            "sources": normalize(mode, raw["raw"]), "raw_response": raw,
            "count_response": counted, "sol_response": answer, "retained_attempts": retained,
            "citation_urls_observed": sorted(set(re.findall(r"https?://[^\s<>\])]+", answer_text))),
            "semantic_grade": "pending isolated parent-side oracle review",
            "count_pricing": "unpublished; allowance retained, not claimed free",
            "cell_reserved_usd": str(sum((ledger.reservations[k] for k in retained), Decimal(0)))}


def verify_receipt(output, receipt, ledger, cell, case):
    if receipt != canonical_receipt(output, ledger, cell, case):
        raise LiveBlocked("pilot_receipt_content_integrity_failure")


def output_text(raw):
    usage = raw.get("usage", {})
    if (raw.get("model") != MODEL or raw.get("status") != "completed"
            or raw.get("service_tier", "default") != "default"
            or type(usage.get("input_tokens")) is not int or not 0 < usage["input_tokens"] <= 6000
            or type(usage.get("output_tokens")) is not int or not 0 < usage["output_tokens"] <= 2048):
        raise LiveBlocked("returned_response_status_tier_or_usage_outside_frozen_budget")
    texts = [part["text"] for item in raw.get("output", []) if item.get("type") == "message"
             for part in item.get("content", []) if part.get("type") == "output_text"]
    text = "\n".join(texts)
    if not text.strip():
        raise LiveBlocked("no_completed_visible_answer_no_automatic_retry")
    return text


def run(inputs, output, access, phase, *, execute=False, owner_task_id=None, opener=None):
    plan, cases, provenance = preflight(inputs, output, access, phase)
    output = Path(plan["journal_root"])
    if not execute:
        return {"status": "preflight_passed_no_network", "phase": phase,
                "credential_presence": credential_presence(), "planned_cells": 8 if phase == "pilot" else 72,
                "public_input_sha256": provenance["local_message_bytes_sha256"],
                "reservation_usd": str(Ledger(output / "live_journal.jsonl", "10.00").exposure),
                "count_pricing": "unpublished; $0.02 per request retained as allowance"}
    if owner_task_id != OWNER:
        raise LiveBlocked("sole_fresh_execution_owner_required")
    measured = {}
    transport = HTTPTransport(output, plan, access, opener=opener,
                              token_counter=lambda text: measured.get(digest(text)))
    selected = list(enumerate(cases, 1))[:2] if phase == "pilot" else list(enumerate(cases, 1))[2:]
    for index, case in selected:
        for mode in MODES:
            cell = f"{index:02d}_{mode}"
            provider = mode.split("_", 1)[0]
            transport.send(provider, request(mode, case, 0), cell=cell, role="search",
                           attempt=1, grant=grant(plan, provider), public_input=case)
            text = transport.model_input(cell, "synthesis", case)
            count_envelope = {"method": "POST", "url": COUNT_ENDPOINT, "timeout_seconds": 30,
                              "body": {"model": MODEL, "input": text, "reasoning": {"effort": "low"}}}
            counted = transport.send("openai", count_envelope, cell=cell, role="count_synthesis",
                                     attempt=1, grant=grant(plan, "openai"), public_input=case)
            measured[digest(text)] = counted["raw"]["input_tokens"]
            answer = transport.send("openai", openai_envelope(text, measured[digest(text)]), cell=cell,
                                    role="synthesis", attempt=1, grant=grant(plan, "openai"),
                                    public_input=case, input_token_count=measured[digest(text)])
            output_text(answer["raw"])
            with exclusive(output):
                ledger = Ledger(output / "live_journal.jsonl", "10.00")
                receipt = canonical_receipt(output, ledger, cell, case)
                write_once(output / "live_receipts" / (cell + ".json"), receipt)
    ledger = Ledger(output / "live_journal.jsonl", "10.00")
    return {"status": "phase_completed_semantic_review_pending", "phase": phase,
            "cells": len(selected) * 4, "cumulative_reserved_usd": str(ledger.exposure),
            "provider_dispatches_in_journal": len(ledger.reservations), "journal_root": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "real_public/inputs.parent-message.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--access-receipt", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "remaining"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-owner-task-id")
    args = parser.parse_args()
    try:
        result = run(args.input, args.output, read_json(args.access_receipt), args.phase,
                     execute=args.execute, owner_task_id=args.execution_owner_task_id)
    except LiveBlocked as exc:
        # LiveBlocked contains only the transport's fixed, sanitized identifiers.
        print(json.dumps({"status": "blocked", "reason": str(exc),
                          "next": "inspect private journal; never rerun an uncertain attempt"}))
        raise SystemExit(2) from None
    except (ValueError, OSError, KeyError, TypeError):
        # Do not print arbitrary exception details: provider/proxy errors may contain secrets.
        print(json.dumps({"status": "blocked", "reason": "preflight_or_dispatch_failed_closed",
                          "next": "inspect private journal; never rerun an uncertain attempt"}))
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
