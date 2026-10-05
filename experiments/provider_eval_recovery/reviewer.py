"""Deterministic scaffold grading. Reviewer truth never enters provider requests.

Real-case entailment still requires an isolated GPT6.1Sol/human adjudicator using
the supplied reviewer spec. A URL match is not an entailment proof.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def grade(answer, sources, oracle):
    expected = oracle["claims"]
    claims = answer.get("claims", [])
    ids = [claim["claim_id"] for claim in claims]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("answer must cover each oracle claim once, including unknowns")
    available = {source["url"] for source in sources}
    rows = []
    for claim in claims:
        truth = expected[claim["claim_id"]]
        state = claim["state"]
        if state not in ("supported", "contradicted", "unknown"):
            raise ValueError("invalid answer claim state")
        citations = claim.get("citations", [])
        if not isinstance(citations, list) or len(citations) != len(set(citations)):
            raise ValueError("invalid citations")
        valid = all(url in available for url in citations)
        unknown = state == "unknown"
        entailment = False
        if not unknown and valid:
            for source in sources:
                if source["url"] in citations:
                    # Only synthetic truth has a pinned literal support passage.
                    for evidence in truth.get("evidence", []):
                        if (source["url"] == evidence["url"]
                                and evidence["text"] in source["text"]):
                            entailment = True
        correct = state == truth["state"] and (unknown or entailment)
        primary = any(urlsplit(url).hostname in truth.get("primary_domains", [])
                      for url in citations if url in available)
        rows.append({"claim_id": claim["claim_id"], "state_correct": correct,
                     "citation_resolves": valid and bool(citations),
                     "source_primary": primary, "synthetic_entailment": entailment,
                     "unknown_correct": unknown and truth["state"] == "unknown",
                     "unsupported_assertion": not unknown and not (correct and citations),
                     "unknown_without_citation": unknown and not citations})
    return {"method": "deterministic_synthetic_scaffold_not_real_case_adjudication",
            "rows": rows, "correct": sum(r["state_correct"] for r in rows),
            "total": len(rows),
            "unsupported_assertions": sum(r["unsupported_assertion"] for r in rows)}


def review_all(bundle, output):
    # This entrypoint runs separately after the provider replay has exited.
    # It has no provider transport, no tool invocation and no inference call.
    from .adapters import MODEL, MODES, Limits, normalize, request
    from .harness import ROOT, Ledger, digest, exclusive, load_frozen, read_json, write_once
    from decimal import Decimal

    bundle, output = Path(bundle), Path(output)
    manifest, cases, fixtures = load_frozen(bundle)
    oracle_bytes = (bundle / "reviewer/oracle.json").read_bytes()
    if hashlib.sha256(oracle_bytes).hexdigest() != manifest["sha256"]["reviewer/oracle.json"]:
        raise ValueError("reviewer oracle hash mismatch")
    oracle = json.loads(oracle_bytes)
    if set(oracle) != {case["case_id"] for case in cases}:
        raise ValueError("reviewer/public case identity mismatch")
    plan = read_json(output / "plan.json")
    if plan["fixture_manifest_sha256"] != digest(manifest):
        raise ValueError("reviewer run/fixture identity mismatch")
    if plan["model"] != MODEL or plan["modes"] != list(MODES):
        raise ValueError("reviewer matrix/model mismatch")
    case_by_id = {case["case_id"]: case for case in cases}
    limit_data = {**plan["limits"], "cell_cost_cap": Decimal(plan["limits"]["cell_cost_cap"])}
    limits = Limits(**limit_data)
    if plan["code_sha256"]["reviewer.py"] != hashlib.sha256(
            (ROOT / "reviewer.py").read_bytes()).hexdigest():
        raise ValueError("reviewer code changed after freeze")
    reviewed, correct, total, unsupported = 0, 0, 0, 0
    with exclusive(output):
        ledger = Ledger(output / "journal.jsonl", plan["run_cost_cap_usd"])
        expected_cells = {case["case_id"] + "_" + mode for case in cases for mode in MODES}
        seen = set()
        validated = []
        for path in sorted((output / "receipts").glob("*.json")):
            receipt = read_json(path)
            cell = receipt["cell"]
            if cell not in expected_cells or path.name != cell + ".json" or cell in seen:
                raise ValueError("reviewer duplicate or unexpected matrix identity")
            seen.add(cell)
            case, mode = case_by_id[receipt["case_id"]], receipt["mode"]
            identity = digest({"plan": plan, "case": case, "mode": mode})
            fixture = fixtures[cell]
            envelopes = [request(mode, case, i, limits) for i in range(limits.max_calls)]
            if (receipt["cell_identity"] != identity or receipt["requests"] != envelopes
                    or receipt["model"] != MODEL or receipt["case_id"] + "_" + mode != cell
                    or receipt["answer"] != fixture["answer"]
                    or receipt["usage"] != fixture["controller_usage"]
                    or len(receipt["responses"]) != limits.max_calls):
                raise ValueError("reviewer frozen receipt identity/content failure")
            for envelope, raw in zip(envelopes, receipt["responses"]):
                key = raw["attempt_id"]
                retained = read_json(output / "raw" / (key + ".json"))
                completed = [event for event in ledger.events
                             if event["attempt_id"] == key and event["kind"] == "completed"]
                if (raw != retained or raw["request_sha256"] != digest(envelope)
                        or not completed or completed[-1]["raw_sha256"] != digest(raw)):
                    raise ValueError("reviewer retained response digest failure")
            sources = normalize(mode, {"results": [result for response in receipt["responses"]
                                                    for result in response["raw"]["results"]]},
                                limits)
            if sources != receipt["sources"]:
                raise ValueError("reviewer source normalization identity failure")
            validated.append((path, receipt))
        # Validate every input before writing any grades. A partial matrix remains
        # partial; copied/renamed cells can never fill a missing expected identity.
        for path, receipt in validated:
            result = grade(receipt["answer"], receipt["sources"], oracle[receipt["case_id"]])
            retained = {"cell": receipt["cell"], "receipt_sha256": digest(receipt),
                        "oracle_sha256": manifest["sha256"]["reviewer/oracle.json"],
                        "rubric_sha256": manifest["sha256"]["reviewer/spec.md"],
                        "reviewer": "isolated_deterministic_synthetic_process",
                        "model_inference": "none", "grade": result}
            write_once(output / "reviews" / path.name, retained)
            reviewed += 1
            correct += result["correct"]
            total += result["total"]
            unsupported += result["unsupported_assertions"]
        summary = {"reviewed_cells": reviewed, "expected_cells": len(expected_cells),
                   "status": "complete" if seen == expected_cells else "partial",
                   "synthetic_claims_correct": correct, "synthetic_claims_total": total,
                   "synthetic_unsupported_assertions": unsupported,
                   "real_case_semantic_grading": "pending_source_bundle_and_isolated_Sol_review",
                   "live_provider_calls": 0, "actual_external_usd": "0.00"}
        write_once(output / "review_summaries" / (digest(summary) + ".json"), summary)
    return summary


def main():
    from .harness import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "synthetic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review_all(args.bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
