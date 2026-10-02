"""Offline, single-writer replay with durable pre-dispatch spend reservations."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from .adapters import MODEL, MODES, RATES, Limits, normalize, public_case, request

ROOT = Path(__file__).resolve().parent


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    """Atomic exclusive installation plus fsync; never replace retained evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    blob = encoded(value) + b"\n"
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError("immutable artifact conflict: " + path.name)
        return
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".pending")
    with temporary.open("xb") as handle:
        os.chmod(temporary, 0o600)
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive(output):
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".lock").open("a") as handle:
        os.chmod(output / ".lock", 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Ledger:
    """Hash-chained journal. A torn final line blocks; it is never silently lost."""

    def __init__(self, path, cost_cap):
        self.path = Path(path)
        self.cap = Decimal(cost_cap)
        if not self.cap.is_finite() or self.cap <= 0:
            raise ValueError("invalid run cap")
        self.events = []
        self.states = {}
        self.reservations = {}
        self.reconciliations = {}
        if self.path.exists():
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):
                raise ValueError("torn ledger: manual offline reconciliation required")
            for line in data.splitlines():
                event = json.loads(line)
                claimed = event.pop("sha256")
                if claimed != digest(event) or event["previous"] != self.previous:
                    raise ValueError("ledger integrity failure")
                if event["sequence"] != len(self.events):
                    raise ValueError("ledger sequence failure")
                event["sha256"] = claimed
                self._accept(event)
        if self.exposure > self.cap:
            raise ValueError("retained exposure exceeds frozen cap")

    @property
    def previous(self):
        return self.events[-1]["sha256"] if self.events else "0" * 64

    @property
    def exposure(self):
        return sum((cost for key, cost in self.reservations.items()
                    if self.states[key] != "not_accepted"), Decimal(0))

    def _accept(self, event):
        key, kind = event["attempt_id"], event["kind"]
        previous = self.states.get(key)
        if kind == "reserved":
            if previous is not None:
                raise ValueError("attempt already reserved")
            amount = Decimal(event["amount_usd"])
            if not amount.is_finite() or amount <= 0:
                raise ValueError("invalid reservation")
            self.reservations[key] = amount
        elif kind == "uncertain":
            if previous != "reserved":
                raise ValueError("invalid uncertain transition")
        elif kind == "reconcile_pending":
            if previous not in ("reserved", "uncertain", "reconcile_pending"):
                raise ValueError("invalid pending reconciliation")
            self.reconciliations[key] = self.reconciliations.get(key, 0) + 1
        elif kind in ("completed", "not_accepted"):
            if previous not in ("reserved", "uncertain", "reconcile_pending"):
                raise ValueError("invalid terminal transition")
            if kind == "not_accepted" and not event.get("nonacceptance_proof"):
                raise ValueError("nonacceptance proof required")
            if kind == "completed" and not event.get("raw_sha256"):
                raise ValueError("retained response digest required")
        else:
            raise ValueError("unknown ledger event")
        self.states[key] = kind
        self.events.append(event)

    def append(self, kind, key, **fields):
        event = {"sequence": len(self.events), "previous": self.previous,
                 "kind": kind, "attempt_id": key, **fields}
        event["sha256"] = digest(event)
        # Validate before durable mutation. Roll back memory if fsync fails by
        # abandoning this Ledger instance and reopening from disk on next run.
        if kind == "reserved" and self.exposure + Decimal(fields["amount_usd"]) > self.cap:
            raise ValueError("run spend cap exhausted before dispatch")
        self._accept(event)
        with self.path.open("ab") as handle:
            os.chmod(self.path, 0o600)
            handle.write(encoded(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return event


def load_frozen(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    if manifest["kind"] != "synthetic_development_only":
        raise ValueError("real source bundle requires schema/spec integration and review first")
    checked_bytes = {}
    for name, expected in manifest["sha256"].items():
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("unsafe manifest path")
        content = (bundle / member).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("frozen artifact hash mismatch: " + name)
        checked_bytes[name] = content
    required = {"public/inputs.json", "reviewer/oracle.json", "reviewer/spec.md",
                "fixtures/responses.json"}
    if set(manifest["sha256"]) != required:
        raise ValueError("manifest must bind every consumed partition")
    cases = json.loads(checked_bytes["public/inputs.json"])["cases"]
    if len(cases) != 20 or len({c["case_id"] for c in cases}) != 20:
        raise ValueError("exactly twenty unique frozen cases required")
    cases = [public_case(c) for c in cases]
    # Do not parse reviewer truth in the provider/controller replay process.
    return manifest, cases, json.loads(checked_bytes["fixtures/responses.json"])


def sol_cost(usage):
    """Standard, short-context rates. Reasoning is already part of output_tokens."""
    if usage["model"] != MODEL or usage["service_tier"] != "standard":
        raise ValueError("controller/reviewer model or tier substitution rejected")
    fields = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens",
              "reasoning_tokens")
    if any(type(usage[k]) is not int or usage[k] < 0 for k in fields):
        raise ValueError("invalid token usage")
    total, cached, writes, output, reasoning = [usage[k] for k in fields]
    if total > 6000 or output > 2048 or cached + writes > total or reasoning > output:
        raise ValueError("frozen Sol token budget exceeded or inconsistent usage")
    return ((total - cached - writes) * Decimal("2") + cached * Decimal("0.10")
            + writes * Decimal("2.50") + output * Decimal("10")) / Decimal(1000000)


class Replay:
    """Only caller supplied frozen outcomes; this class cannot access a provider."""

    def __init__(self, fixture):
        self.fixture = fixture
        self.calls = 0

    def send(self, query, attempt):
        self.calls += 1
        # No sleep: metadata is the synthetic provider latency, not wall latency.
        return self.fixture["queries"][query][attempt - 1]


def query_replay(ledger, output, envelope, fixture, query, limits, cell_key, mode):
    rate = RATES[mode]
    replay = Replay(fixture)
    for attempt in range(1, limits.max_attempts + 1):
        key = digest({"cell": cell_key, "query": query, "attempt": attempt,
                      "request": envelope})
        raw_path = output / "raw" / (key + ".json")
        state = ledger.states.get(key)
        if state == "completed":
            raw = read_json(raw_path)
            completed = next(e for e in ledger.events
                             if e["attempt_id"] == key and e["kind"] == "completed")
            if digest(raw) != completed["raw_sha256"]:
                raise ValueError("retained response integrity failure")
            return raw, replay.calls
        if state == "not_accepted":
            continue
        if state is not None:
            # Crash between reserve/send/result is ambiguous. Offline reconciliation
            # can adopt a completely retained, identity-bound successful response.
            if raw_path.exists():
                raw = read_json(raw_path)
                if raw["request_sha256"] != digest(envelope) or raw["attempt_id"] != key:
                    raise ValueError("retained response request identity failure")
                if raw["outcome"] == "success":
                    normalize(mode, raw["raw"], limits)
                    ledger.append("completed", key, raw_sha256=digest(raw),
                                  reason="adopted_retained_response_no_resend",
                                  synthetic_latency_ms=raw["latency_ms"])
                    return raw, replay.calls
            count = ledger.reconciliations.get(key, 0)
            if count < limits.max_reconciliations:
                ledger.append("reconcile_pending", key,
                              reason="no documented synchronous-search recovery lookup")
            return None, replay.calls
        cell_exposure = sum((ledger.reservations[e["attempt_id"]] for e in ledger.events
                             if e["kind"] == "reserved" and e.get("cell") == cell_key
                             and ledger.states[e["attempt_id"]] != "not_accepted"), Decimal(0))
        if cell_exposure + rate > limits.cell_cost_cap:
            raise ValueError("cell spend cap exhausted before dispatch")
        ledger.append("reserved", key, amount_usd=str(rate), request_sha256=digest(envelope),
                      cell=cell_key, query=query, attempt=attempt,
                      endpoint=envelope["url"], model=MODEL)
        dispatch_started = time.monotonic()
        outcome = replay.send(query, attempt)
        dispatch_wall_ms = round((time.monotonic() - dispatch_started) * 1000, 3)
        if type(outcome["latency_ms"]) is not int or outcome["latency_ms"] < 0:
            raise ValueError("invalid latency")
        timing = {"synthetic_latency_ms": outcome["latency_ms"],
                  "offline_dispatch_wall_ms": dispatch_wall_ms}
        if outcome["outcome"] == "not_accepted":
            ledger.append("not_accepted", key,
                          nonacceptance_proof="frozen_fixture_proof_not_a_live_provider_claim",
                          **timing)
            continue
        if outcome["outcome"] != "success":
            ledger.append("uncertain", key, reason="timeout_or_unknown_acceptance",
                          provider_id=outcome.get("provider_id"), **timing)
            return None, replay.calls
        # A malformed successful response remains reserved for reconciliation;
        # never discard its possible charge just because normalization fails.
        raw = {"attempt_id": key, "request_sha256": digest(envelope), **outcome}
        write_once(raw_path, raw)
        normalize(mode, raw["raw"], limits)
        ledger.append("completed", key, raw_sha256=digest(raw), **timing)
        return raw, replay.calls
    return None, replay.calls


def cost_bounds():
    """One-time planning ceiling, not spend approval or an actual invoice."""
    cells = 20 * 4
    base = 20 * sum(RATES.values())
    retry_bound = base * 2
    # Three Sol calls/cell: planner/controller, synthesis, isolated semantic reviewer.
    # Caps include prompt, source context, oracle (reviewer only), and
    # reasoning. A future executor must measure tokenizer counts before requests.
    outer = cells * 3 * (Decimal(6000) * Decimal(2) + Decimal(2048) * Decimal(10))
    outer /= Decimal(1000000)
    # Extra allowance for cache writes on every input token (no cache discount).
    write_premium = cells * 3 * Decimal(6000) * Decimal("0.50") / Decimal(1000000)
    return {"approval": "one-time cumulative USD10 approved; credentials/access pending",
            "approval_evidence": "Sentinel_741e7736ff8c8191a085c8e2ece40576: yes i approve",
            "openai_project": "proj_F2tFJuxLaovJru8RrtXRaqNj",
            "model": MODEL, "cells": cells, "search_requests_base": cells,
            "search_request_attempt_cap": cells * 2,
            "sol_attempt_cap": cells * 3, "sol_input_tokens_per_call_cap": 6000,
            "sol_total_output_tokens_per_call_cap": 2048,
            "raw_search_base_usd": str(base), "raw_search_retry_ceiling_usd": str(retry_bound),
            "outer_sol_uncached_ceiling_usd": str(outer),
            "outer_sol_cache_write_premium_ceiling_usd": str(write_premium),
            "subtotal_all_in_usd": str(retry_bound + outer + write_premium),
            "approved_total_cap_usd": "10.00",
            "provider_extras_allowance_usd": "0.50",
            "subtotal_with_extras_allowance_usd": str(retry_bound + outer + write_premium
                                                     + Decimal("0.50")),
            "remainder_for_tax_fx_reconciliation_usd": str(Decimal(10) - retry_bound
                                                           - outer - write_premium
                                                           - Decimal("0.50")),
            "pilot_provider_attempt_cap": 8, "pilot_sol_attempt_cap": 24,
            "pilot_uncached_sol_usd": "0.77952", "pilot_cache_write_allowance_usd": "0.072",
            "pilot_provider_usd": "0.024", "pilot_subtotal_usd": "0.87552",
            "pilot_inclusive_cap_usd": "1.00",
            "cumulative_accounting": "pilot cells adopted, remaining18 share same journal/cap",
            "optional_separate_task_core_20_base_usd": "0.50",
            "optional_separate_task_pro_20_base_usd": "2.00",
            "optional_task_both_two_attempt_provider_ceiling_usd": "5.00",
            "optional_deeper_approval": "excluded from approved USD10; separate approval required",
            "excluded_until_rebudgeted": ["Sol fast mode, nonstandard or regional tier",
                                         "Sol >6000 input or >2048 total output per call",
                                         "more than three Sol calls per cell",
                                         "paid URL extract/fetch, hosted sandbox or extra tools",
                                         "unsafe retry after uncertain acceptance"],
            "tax_fx_condition": "execute only if total inclusive quote fits approved cap"}


def run(bundle, output, limits=Limits(), cap=Decimal("10"), phase="all"):
    start = time.monotonic()
    output = Path(output)
    manifest, cases, fixtures = load_frozen(bundle)
    selected = cases[:2] if phase == "pilot" else cases[2:] if phase == "remaining" else cases
    if phase not in ("all", "pilot", "remaining"):
        raise ValueError("invalid phase")
    limit_data = asdict(limits)
    limit_data["cell_cost_cap"] = str(limits.cell_cost_cap)
    plan = {"schema": "provider_eval_offline.v1", "model": MODEL, "modes": list(MODES),
            "fixture_manifest_sha256": digest(manifest), "limits": limit_data,
            "run_cost_cap_usd": str(cap), "execution": "offline_replay",
            "openai_project": "proj_F2tFJuxLaovJru8RrtXRaqNj",
            "approved_budget_usd": "10.00", "case_ids": [case["case_id"] for case in cases],
            "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in ("adapters.py", "harness.py", "reviewer.py")}}
    with exclusive(output):
        write_once(output / "plan.json", plan)
        ledger = Ledger(output / "journal.jsonl", cap)
        receipts, dispatched = [], 0
        for case in selected:
            for mode in MODES:
                cell = case["case_id"] + "_" + mode
                receipt_path = output / "receipts" / (cell + ".json")
                cell_identity = digest({"plan": plan, "case": case, "mode": mode})
                fixture = fixtures[cell]
                envelopes = [request(mode, case, i, limits) for i in range(limits.max_calls)]
                responses = []
                for index, envelope in enumerate(envelopes):
                    response, calls = query_replay(ledger, output, envelope, fixture, index,
                                                   limits, cell_identity, mode)
                    dispatched += calls
                    if response is None:
                        break
                    responses.append(response)
                if len(responses) != limits.max_calls:
                    # No immutable terminal receipt while acceptance is unresolved.
                    # Bounded journal state is the durable per-case pending receipt.
                    pending = {"cell": cell, "cell_identity": cell_identity,
                               "requests": envelopes, "status": "pending_reconciliation",
                               "journal_head": ledger.previous, "live_provider_calls": 0}
                    write_once(output / "pending" / (cell + "_" + digest(pending) + ".json"),
                               pending)
                    continue
                # Apply one shared evidence cap across all rounds, not per round.
                combined_raw = {"results": [result for resp in responses
                                            for result in resp["raw"]["results"]]}
                sources = normalize(mode, combined_raw, limits)
                usage = fixture["controller_usage"]
                outer_cost = sol_cost(usage)
                if usage["model"] != MODEL:
                    raise ValueError("model substitution")
                total_cell_cost = limits.max_calls * RATES[mode] + outer_cost
                if total_cell_cost > limits.cell_cost_cap:
                    raise ValueError("cell inference plus search cost cap exceeded")
                inference_key = digest({"cell": cell_identity, "role": "controller_fixture"})
                inference_digest = digest({"usage": usage, "answer": fixture["answer"]})
                if inference_key not in ledger.states:
                    ledger.append("reserved", inference_key, amount_usd=str(outer_cost),
                                  cell=cell_identity, model=MODEL)
                    ledger.append("completed", inference_key,
                                  raw_sha256=inference_digest)
                else:
                    if ledger.states[inference_key] == "reserved":
                        # Only an offline frozen transcript can be adopted this
                        # way; a future live Sol attempt stays uncertain until its
                        # identity-bound output or nonacceptance proof is retained.
                        ledger.append("completed", inference_key,
                                      raw_sha256=inference_digest,
                                      reason="adopted_frozen_controller_transcript_no_inference")
                    completed = [e for e in ledger.events if e["attempt_id"] == inference_key
                                 and e["kind"] == "completed"]
                    if not completed or completed[-1]["raw_sha256"] != inference_digest:
                        raise ValueError("controller fixture integrity or completion failure")
                attempt_ids = {e["attempt_id"] for e in ledger.events
                               if e["kind"] == "reserved" and e.get("cell") == cell_identity}
                timings = [e for e in ledger.events if e["attempt_id"] in attempt_ids
                           and "synthetic_latency_ms" in e]
                receipt = {"cell": cell, "cell_identity": cell_identity,
                           "case_id": case["case_id"], "mode": mode, "model": MODEL,
                           "prompt_sha256": digest(public_case(case)),
                           "requests": envelopes, "responses": responses,
                           "sources": sources, "answer": fixture["answer"],
                           "grade_status": "awaiting_separate_reviewer_process",
                           "usage": usage, "attempt_ids": sorted(attempt_ids),
                           "synthetic_provider_latency_ms": sum(
                               event["synthetic_latency_ms"] for event in timings),
                           "outer_inference_latency": "not_measured; frozen answer replay",
                           "reviewer_latency": "deterministic_local; no inference call",
                           "cost": {"actual_external_usd": "0.00",
                                    "simulated_search_usd": str(limits.max_calls * RATES[mode]),
                                    "simulated_outer_sol_usd": str(outer_cost),
                                    "invoice_reconciled": False},
                           "claim_ceiling": "synthetic_replay_not_provider_quality_evidence"}
                write_once(receipt_path, receipt)
                receipts.append(receipt)
        latencies = [r["synthetic_provider_latency_ms"] for r in receipts]
        expected = len(selected) * len(MODES)
        report = {"status": "complete" if len(receipts) == expected else "pending_reconciliation",
                  "real_cases_integrated": 0, "synthetic_case_count": 20,
                  "completed_cells": len(receipts), "expected_cells": expected,
                  "phase": phase, "cumulative_expected_cells": 80,
                  "cumulative_retained_cells": len(list((output / "receipts").glob("*.json"))),
                  "fixture_sends_this_invocation": dispatched, "live_provider_calls": 0,
                  "actual_external_cost_usd": "0.00", "model": MODEL,
                  "simulated_cost_exposure_usd": str(ledger.exposure),
                  "pending_attempts": [key for key, state in ledger.states.items()
                                       if state not in ("completed", "not_accepted")],
                  "synthetic_latency_p50_ms": statistics.median(latencies) if latencies else None,
                  "synthetic_latency_p95_ms": sorted(latencies)[int(.95 * (len(latencies) - 1))]
                      if latencies else None,
                  "provider_ranking": None,
                  "limitations": ["real Library bundle transfer blocked, hash unverified",
                                  "no live provider quality, latency or invoice observations",
                                  "deterministic synthetic entailment; semantic review pending",
                                  "native excerpt token and character caps differ"],
                  "cost_plan": cost_bounds()}
        # Invocation summaries are immutable too; resume appends a new report.
        write_once(output / "invocations" / (digest(report) + ".json"), report)
    return {**report, "offline_wall_seconds": round(time.monotonic() - start, 6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "synthetic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "remaining", "all"), default="all")
    args = parser.parse_args()
    print(json.dumps(run(args.bundle, args.output, phase=args.phase), indent=2))


if __name__ == "__main__":
    main()
