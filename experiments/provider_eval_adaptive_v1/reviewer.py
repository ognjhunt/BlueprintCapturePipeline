"""Separate reviewer process. Oracle never enters the adaptive controller."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.provider_eval_recovery.harness import digest, exclusive, read_json, write_once
from experiments.provider_eval_recovery.live_runner import OWNER

from .protocol import CURRENT_DATE, MODES, PROTOCOL, ROOT, evidence
from .runner import Blocked, CellStop, Transport, admission


class ReviewerTransport(Transport):
    def _send(self, provider, cell, step, envelope, reserve):
        if step not in {"review", "review_count"}:
            saved = self.retained(cell, step)
            if saved is None:
                raise Blocked("isolated_reviewer_cannot_initiate_research")
        return super()._send(provider, cell, step, envelope, reserve)


def bind_reviewer(transport, bundle):
    if (set(bundle) != {"spec", "cases"} or not isinstance(bundle["spec"], str)
            or set(bundle["cases"]) != {case["id"] for case in transport.public["cases"]}):
        raise Blocked("exact_parent_local_reviewer_bundle_required")
    with exclusive(transport.root):
        write_once(transport.paths / "reviewer_scope.json", {"spec_sha256": digest(bundle["spec"]),
                   "oracle_hashes": {key: digest(value) for key, value in bundle["cases"].items()}})


def review_case(transport, index, oracle, spec):
    if not isinstance(transport, ReviewerTransport):
        raise Blocked("isolated_readonly_research_reviewer_transport_required")
    case, outputs = transport.public["cases"][index - 1], []
    pinned = read_json(transport.paths / "reviewer_scope.json")
    if pinned["spec_sha256"] != digest(spec) or pinned["oracle_hashes"][case["id"]] != digest(oracle):
        raise Blocked("frozen_reviewer_oracle_spec_changed")
    for mode in MODES:
        cell = f"{index:02d}_{mode}"
        receipt = read_json(transport.paths / "receipts" / (cell + ".json"))
        if receipt.get("protocol") != PROTOCOL or receipt.get("cell") != cell or receipt.get("case_id") != case["id"]:
            raise Blocked("exact_adaptive_four_output_case_required")
        from .runner import research
        # Rebuild and compare from admitted retained evidence. All steps must
        # already exist; this process is prohibited from initiating research.
        for step in ("search1",):
            if transport.retained(cell, step) is None:
                raise Blocked("reviewer_cannot_initiate_research")
        canonical = research(transport, index, mode)
        if canonical != receipt:
            raise Blocked("reviewer_receipt_changed")
        raws = [transport.retained(cell, "search1")["raw"]]
        followup = transport.retained(cell, "search2")
        if followup is not None:
            raws.append(followup["raw"])
        try:
            sources = evidence(mode, raws, 1000)
        except ValueError:
            sources = []
        outputs.append({"arm": str(len(outputs) + 1), "answer": receipt["answer"], "status": receipt["status"],
                        "sources": sources, "evidence_scope": "bounded retained excerpts; unsupported claims remain unknown"})
    items = [{"role": "developer", "content": "Trusted current date: " + CURRENT_DATE + ". "
              "You are the isolated reviewer. Controller relevance decisions are not ground truth. "
              "Use the frozen oracle/spec and supplied retained evidence. Grade all four arms separately for "
              "source entailment, primary provenance, dated citation support, contradictions, unsupported claims "
              "and appropriate unknowns. Unknowns remain in the denominator. Do not fabricate missing evidence "
              "Contract/input-warning stops are harness diagnostics, not provider-quality scores; mark those arms unscorable. "
              "or infer site suitability. Flag limitations of partial excerpts. Return compact JSON with four arm "
              "assessments and review unknowns; no provider tools, browsing or overall winner."},
             {"role": "user", "content": json.dumps({"public_case": case, "oracle": oracle, "spec": spec,
                                                       "outputs": outputs}, sort_keys=True, ensure_ascii=False)}]
    status, assessment = "independent_model_review_requires_parent_adjudication", None
    try:
        assessment = json.loads(transport._infer_items(f"{index:02d}_review", "review", items))
    except (CellStop, ValueError):
        status = "independent_review_unknown_budget_or_format_stop"
    result = {"protocol": PROTOCOL, "case_id": case["id"], "status": status, "review": assessment,
              "oracle_sha256": digest(oracle), "spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
              "ground_truth": "frozen parent oracle; model judgment is not ground truth"}
    with exclusive(transport.root):
        write_once(transport.paths / "reviews" / (case["id"] + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--access-receipt", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "provider_eval_recovery/real_public/inputs.parent-message.json")
    parser.add_argument("--reviewer-bundle", type=Path, required=True,
                        help="parent-local mapping: spec text and cases keyed by BP-EVAL-ID; never read by controller")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-owner-task-id")
    args = parser.parse_args()
    try:
        plan, public, paths = admission(args.output, read_json(args.access_receipt), args.input)
        if not args.execute:
            print(json.dumps({"status": "review_preflight_no_network_no_oracle_read", "protocol": PROTOCOL}))
            return
        if args.execution_owner_task_id != OWNER:
            raise Blocked("sole_fresh_execution_owner_required")
        bundle = read_json(args.reviewer_bundle)
        transport = ReviewerTransport(args.output, read_json(args.access_receipt), public, plan, paths)
        bind_reviewer(transport, bundle)
        for index, case in enumerate(public["cases"], 1):
            review_case(transport, index, bundle["cases"][case["id"]], bundle["spec"])
        print(json.dumps({"status": "independent_reviews_complete_parent_adjudication_required", "protocol": PROTOCOL}))
    except Blocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        raise SystemExit(2) from None
    except (OSError, KeyError, TypeError, ValueError):
        print(json.dumps({"status": "blocked", "reason": "isolated_reviewer_failed_closed"}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
