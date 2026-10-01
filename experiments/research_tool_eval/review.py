"""Blinded human grading and deterministic quote checks; no model is a grader."""
import argparse
import json
import random
import re
from pathlib import Path

from harness import ROOT, GateError, digest, run_plan, verify_freeze


def words(text):
    return " ".join(re.findall(r"\w+", str(text).lower()))


def quote_audit(answer, sources):
    """Exact passage support is a screen, not semantic entailment or independence."""
    by_url = {s["url"]: words(s.get("text", "")) for s in sources}
    rows = []
    for claim in answer.get("claims", []):
        quote = str(claim.get("quote", ""))
        normalized = words(quote)
        cited = claim.get("source_url") in by_url
        match = bool(normalized) and cited and normalized in by_url[claim["source_url"]]
        rows.append({"claim_id": claim.get("id"), "citation_in_retrieved_evidence": cited,
                     "quote_in_retained_passage": match, "quote_within_word_limit": len(quote.split()) <= 20,
                     "semantic_support": "pending_human_review",
                     "screen": "candidate_support" if match and len(quote.split()) <= 20 else "needs_source_check"})
    return rows


def make_review(report):
    if report["freeze"] != verify_freeze():
        raise GateError("result/answer-rubric freeze mismatch")
    cases = {c["id"]: c for c in json.loads((ROOT / "cases.json").read_text())}
    rubric = {r["case_id"]: r for r in json.loads((ROOT / "rubric.json").read_text())}
    # Do not show provider mode, costs, token usage, latency or internal IDs to graders.
    cells = list(report["cells"])
    random.Random(20260930).shuffle(cells)
    review, mapping = [], {}
    for i, cell in enumerate(cells):
        bid = "blind-" + str(i + 1).zfill(3)
        content_hash = cell_hash(cell)
        mapping[bid] = {"case_id": cell["case_id"], "mode": cell["mode"],
                        "controller_request_id": cell.get("controller_request_id"), "content_sha256": content_hash}
        case = cases[cell["case_id"]]
        answer = {k: v for k, v in (cell.get("answer") or {}).items() if k != "usage"}
        review.append({"blind_id": bid, "case_id": case["id"], "question": case["question"],
                       "content_sha256": content_hash, "status": cell["status"],
                       "answer": answer, "sources": cell["sources"],
                       "quote_screen": quote_audit(answer, cell["sources"]),
                       "expected_fact_grades": [{"fact_id": f["id"], "statement": f["statement"],
                                                 "expected_source_ids": f["source_ids"], "grade": None,
                                                 "support_note": None} for f in rubric[case["id"]]["expected_facts"]],
                       "useful_field_grades": dict.fromkeys(case["requested_fields"]),
                       "forbidden_inferences": rubric[case["id"]]["forbidden_inferences"],
                       "required_unknowns": rubric[case["id"]]["required_unknowns"],
                       "claim_grades": [{"claim_id": x.get("id"), "supported": None, "citation_correct": None,
                                         "quote_supported": None, "source_freshness": None} for x in answer.get("claims", [])],
                       "unsupported_claims": None, "critical_error": None,
                       "unknown_and_conflict_handling": None, "freshness": None,
                       "reviewer": None, "reviewed_at": None, "review_note": None})
    return {"freeze": report["freeze"], "mock_only": report.get("offline", True),
            "run_plan_sha256": digest(report["run_plan"]),
            "quality_results": "not_evaluated", "review": review}, mapping


def cell_hash(cell):
    return digest({k: v for k, v in cell.items() if k != "grading"})


def validate_matrix(report):
    stage = report.get("run_plan", {}).get("stage")
    if stage not in {"raw20", "raw20+task6"}:
        raise GateError("selected preregistered stage required")
    expected = run_plan(stage == "raw20+task6")
    if report["run_plan"] != expected:
        raise GateError("declared matrix differs from frozen selected protocol")
    want = {(x["case_id"], x["mode"]) for x in expected["cells"]}
    got = [(x["case_id"], x["mode"]) for x in report["cells"]]
    if len(got) != len(want) or set(got) != want:
        raise GateError("all frozen selected cells, including explicit failures/not-run cells, are required")
    if any(x.get("status") not in {"completed", "failed", "uncertain", "not_run"} for x in report["cells"]):
        raise GateError("every cell needs a terminal/uncertain/not-run state")


def apply_review(report, review, mapping):
    if review["mock_only"] or report.get("offline", True):
        raise GateError("mock rehearsals cannot produce provider quality rankings")
    if report["freeze"] != review["freeze"] or report["freeze"] != verify_freeze():
        raise GateError("grade/answer freeze mismatch")
    validate_matrix(report)
    if review["run_plan_sha256"] != digest(report["run_plan"]):
        raise GateError("reviewed selection/denominators changed")
    original = {(c["case_id"], c["mode"]): c for c in report["cells"]}
    if len(review["review"]) != len(original) or len(mapping) != len(original):
        raise GateError("all preregistered cells including failures must be reviewed")
    seen = set()
    for entry in review["review"]:
        identity = mapping[entry["blind_id"]]
        key = (identity["case_id"], identity["mode"])
        if key in seen or key not in original:
            raise GateError("duplicate or foreign review cell")
        seen.add(key)
        cell = original[key]
        if cell.get("controller_request_id") != identity["controller_request_id"]:
            raise GateError("review mapping changed")
        if cell_hash(cell) != identity["content_sha256"] or cell_hash(cell) != entry["content_sha256"]:
            raise GateError("graded answer/evidence/accounting bytes changed")
        expected, _ = make_review({**report, "cells": [cell]})
        skeleton = expected["review"][0]
        for k in ["answer", "sources", "case_id", "question", "status", "forbidden_inferences", "required_unknowns"]:
            if entry[k] != skeleton[k]:
                raise GateError("blinded evidence or rubric content changed")
        facts = entry["expected_fact_grades"]
        if [x["fact_id"] for x in facts] != [x["fact_id"] for x in skeleton["expected_fact_grades"]]:
            raise GateError("expected fact denominator changed")
        for actual_fact, expected_fact in zip(facts, skeleton["expected_fact_grades"]):
            if {k: v for k, v in actual_fact.items() if k not in {"grade", "support_note"}} != {
                k: v for k, v in expected_fact.items() if k not in {"grade", "support_note"}
            }:
                raise GateError("frozen expected fact/source content changed")
        if any(x["grade"] not in {"supported", "partial", "omitted", "wrong"} or not x["support_note"] for x in facts):
            raise GateError("every expected fact needs a supported/partial/omitted/wrong grade and note")
        fields = entry["useful_field_grades"]
        if fields.keys() != skeleton["useful_field_grades"].keys() or any(type(x) is not bool for x in fields.values()):
            raise GateError("requested-field denominator or accepted-field grades incomplete")
        claims = entry["claim_grades"]
        if [x["claim_id"] for x in claims] != [x["claim_id"] for x in skeleton["claim_grades"]]:
            raise GateError("all claims including unsupported claims must be graded")
        for c in claims:
            if any(type(c.get(k)) is not bool for k in ("supported", "citation_correct", "quote_supported")):
                raise GateError("claim support/citation/quote labels incomplete")
            if c.get("source_freshness") not in {"current", "historical_bounded", "stale", "unknown"}:
                raise GateError("claim freshness label missing")
        if type(entry["critical_error"]) is not bool or type(entry["unknown_and_conflict_handling"]) is not bool:
            raise GateError("critical-error and unknown/conflict adjudication incomplete")
        if entry["freshness"] not in {"current", "historical_bounded", "stale", "unknown"}:
            raise GateError("case freshness incomplete")
        if not isinstance(entry["unsupported_claims"], list) or not entry["reviewer"] or not entry["reviewed_at"]:
            raise GateError("unsupported-claim list and independent reviewer evidence required")
        unsupported = {x["claim_id"] for x in claims if not x["supported"]}
        if set(entry["unsupported_claims"]) != unsupported:
            raise GateError("unsupported claim IDs disagree with claim-level grades")
        if cell["status"] != "completed" and (any(fields.values()) or any(x["grade"] in {"supported", "partial"} for x in facts)):
            raise GateError("failed/uncertain/not-run cells cannot earn answer credit")
        credit = {"supported": 1, "partial": .5, "omitted": 0, "wrong": 0}
        cell["grading"] = {"state": "human_reviewed", "coverage": sum(credit[x["grade"]] for x in facts) / len(facts),
                           "unsupported_claims": entry["unsupported_claims"], "claim_grades": claims,
                           "citation_support": sum(x["citation_correct"] for x in claims) / len(claims) if claims else None,
                           "quote_support": sum(x["quote_supported"] for x in claims) / len(claims) if claims else None,
                           "useful_accepted_fields": [k for k, v in fields.items() if v],
                           "freshness": entry["freshness"], "critical_error": entry["critical_error"],
                           "unknown_and_conflict_handling": entry["unknown_and_conflict_handling"],
                           "reviewer": entry["reviewer"], "reviewed_at": entry["reviewed_at"]}
    report["quality_results"] = "human_reviewed_20_case_pilot_not_general_benchmark"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "apply"])
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--review", type=Path)
    p.add_argument("--mapping", type=Path)
    args = p.parse_args()
    report = json.loads(args.results.read_text())
    if args.command == "prepare":
        r, mapping = make_review(report)
        args.output.mkdir(exist_ok=False, parents=True)
        (args.output / "blinded_review.json").write_text(json.dumps(r, indent=2) + "\n")
        (args.output / "parent_only_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    else:
        if not args.review or not args.mapping:
            p.error("apply needs --review and --mapping")
        result = apply_review(report, json.loads(args.review.read_text()), json.loads(args.mapping.read_text()))
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
