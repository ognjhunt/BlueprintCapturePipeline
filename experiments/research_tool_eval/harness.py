"""Offline-first Blueprint provider comparison. Standard library only; no HTTP client.

Transport/controller are injected boundaries. CLI is hermetic. This does not
launch, edit or impersonate the existing Agents API researcher.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-6.1-sol"
VERSION = "blueprint-public-research-eval-v1"
RAW_MODES = ("parallel-fast", "perplexity-fast", "parallel-advanced", "perplexity-standard")
TASK_MODES = ("parallel-core", "parallel-pro")
HARD_CASES = ("BP20-04", "BP20-07", "BP20-08", "BP20-13", "BP20-16", "BP20-20")
RATES = {"parallel-fast": .001, "parallel-advanced": .005,
         "perplexity-fast": .001, "perplexity-standard": .005,
         "parallel-core": .025, "parallel-pro": .1}
RESPONSE_BYTES = 512_000
CONTEXT_CHARS = 16_000
OUTPUT_TOKENS = 1600  # includes reasoning, verified by the parent-owned controller
INPUT_TOKENS = 8000
CONTROLLER_RESERVE = .032  # standard short-context: 8000*$2/M + 1600*$10/M
MAX_POLLS = 24  # 30-second cadence covers the published Core/Pro 5/10-minute windows
POLL_INTERVAL_S = 30
TASK_DEADLINE_S = 720


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GateError(RuntimeError):
    pass


class UncertainWork(RuntimeError):
    """An accepted/possibly accepted paid dispatch must never be blindly repeated."""


class ConfirmedRejected(RuntimeError):
    """Parent transport may use only after proving the request was not accepted/billed."""


class Transport(Protocol):
    offline: bool

    async def call(self, request: dict, *, timeout_s: float, max_bytes: int) -> dict: ...


class Controller(Protocol):
    offline: bool
    model: str
    snapshot: str

    async def answer(self, request: dict, *, timeout_s: float, max_bytes: int) -> dict: ...


@dataclass(frozen=True)
class Adapter:
    mode: str

    def __post_init__(self):
        if self.mode not in RATES:
            raise ValueError("unrecognized provider mode")

    @property
    def delegated(self):
        return self.mode in TASK_MODES

    def request(self, case: dict, *, memory_scope_key=None) -> dict:
        # Do not accept caller-created cases; only frozen public input is eligible.
        validate_public_case(case)
        if self.mode.startswith("parallel-"):
            origin = "https://api.parallel.ai"
            auth = "x-api-key"
            if not self.delegated:
                path = "/v1/search"
                body = {"objective": case["question"], "search_queries": case["queries"],
                        "mode": self.mode.split("-")[1], "max_chars_total": CONTEXT_CHARS,
                        "client_model": MODEL, "advanced_settings": {"max_results": 10}}
            else:
                path = "/v1/tasks/runs"
                body = {"input": case["question"] + "\nAs of: " + case["as_of"],
                        "processor": self.mode.split("-")[1],
                        "memory_scope_key": memory_scope_key or "offline_bp20_" + digest([case["id"], self.mode])[:24],
                        "metadata": {"case_id": case["id"], "eval_version": VERSION},
                        "task_spec": {"output_schema": {"type": "json", "json_schema": {
                            "type": "object", "properties": {
                                "facts": {"type": "string", "description":
                                          "Bounded facts with versions, citations, dates and unknowns. Public web only."},
                                "useful_fields": {"type": "string", "description":
                                                  "Populate only these fields: " + ", ".join(case["requested_fields"])},
                                "unknowns": {"type": "string"}},
                            "required": ["facts", "useful_fields", "unknowns"], "additionalProperties": False}}}}
                # Documented isolated application scope avoids omitted/personal
                # memory fallback. This does NOT disable memory. Parent must
                # verify it is empty and resolve the provider documentation conflict.
        else:
            origin, path, auth = "https://api.perplexity.ai", "/search", "Authorization: Bearer"
            body = {"query": case["queries"], "search_type": "fast" if self.mode.endswith("fast") else "web",
                    "max_results": 10, "max_tokens": 4000, "max_tokens_per_page": 800}
        return {"method": "POST", "origin": origin, "path": path, "body": body,
                "auth_header_name": auth, "api_version": "v1" if origin.endswith("parallel.ai") else "Search /search",
                "mode": self.mode, "provider_model_version": "provider-managed; not snapshot-pinnable",
                "case_id": case["id"], "timeout_s": 30,
                "max_response_bytes": RESPONSE_BYTES}

    def poll(self, run_id: str) -> dict:
        if not self.delegated or not re.fullmatch(r"[A-Za-z0-9_-]{1,150}", run_id):
            raise ValueError("invalid durable provider run ID")
        return {"method": "GET", "origin": "https://api.parallel.ai",
                "path": f"/v1/tasks/runs/{run_id}/result?timeout=10", "body": None,
                "auth_header_name": "x-api-key", "mode": self.mode,
                "timeout_s": 15, "max_response_bytes": RESPONSE_BYTES}

    def normalize(self, response: dict, *, retrieved_at: str) -> dict:
        if self.delegated:
            if response.get("run", {}).get("status") != "completed":
                raise ValueError("non-terminal Task result cannot be scored")
            # Keep raw response durably, but bound the Sol-facing delegated context.
            serialized = canonical(response.get("output"))
            return {"research_output_excerpt": serialized[:CONTEXT_CHARS], "sources": [],
                    "context_truncated": len(serialized) > CONTEXT_CHARS,
                    "kind": "delegated_research_system"}
        result = []
        for item in response.get("results", [])[:10]:
            url = item.get("url", "")
            if not is_public_url(url):
                continue
            passages = item.get("excerpts", []) if self.mode.startswith("parallel-") else [item.get("snippet", "")]
            text = "\n".join(str(x) for x in passages)
            if len(url) > 2048:
                continue
            result.append({"url": url, "title": str(item.get("title", ""))[:200], "text": text,
                           "published_at": item.get("publish_date", item.get("date")),
                           "last_updated": item.get("last_updated"),
                           "source_checked_at": None, "retrieved_at": retrieved_at})
        # Identical deterministic context cap and result order for both raw providers.
        remaining = CONTEXT_CHARS
        for item in result:
            item["text"] = item["text"][:remaining]
            remaining -= len(item["text"])
        # Whole evidence envelope (including URLs/metadata), not just passage lengths.
        while result and len(canonical(result)) > CONTEXT_CHARS:
            last = result[-1]
            excess = len(canonical(result)) - CONTEXT_CHARS
            if len(last["text"]) > excess:
                last["text"] = last["text"][:-excess]
            else:
                result.pop()
        return {"sources": result, "kind": "raw_search"}


def is_public_url(url: str) -> bool:
    import ipaddress
    from urllib.parse import urlsplit
    try:
        p = urlsplit(url)
        if p.scheme != "https" or p.username or p.password or not p.hostname:
            return False
        host = p.hostname.lower()
        if host in {"localhost", "app.notion.com", "docs.google.com"} or host.endswith((".local", ".internal")):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return "." in host
    except ValueError:
        return False


def validate_public_case(case: dict):
    cases = json.loads((ROOT / "cases.json").read_text())
    if case not in cases or case.get("disclosure") != "public_only":
        raise GateError("only exact frozen public cases may be disclosed")


class Journal:
    """Durable single-writer journal. Reservations precede external side effects."""

    def __init__(self, path: Path, *, cap_usd: float, max_calls: int = 184):
        if not math.isfinite(cap_usd) or cap_usd < 0 or max_calls <= 0:
            raise GateError("invalid budget")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls(
              id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL,
              reserve REAL NOT NULL, actual REAL, response TEXT, provider_id TEXT,
              dispatches INTEGER NOT NULL DEFAULT 0, polls INTEGER NOT NULL DEFAULT 0,
              elapsed_s REAL NOT NULL DEFAULT 0, error TEXT, created_at TEXT NOT NULL,
              next_poll_at REAL NOT NULL DEFAULT 0);
        """)
        self.bind("budget", {"cap_usd": cap_usd, "max_calls": max_calls})
        self.cap_usd, self.max_calls = cap_usd, max_calls

    def bind(self, key: str, value: Any):
        expected = canonical(value)
        with self.db:
            old = self.db.execute("SELECT value FROM meta WHERE k=?", (key,)).fetchone()
            if old and old[0] != expected:
                raise GateError("run identity/budget changed; create a new reviewed run")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, expected))

    def row(self, ident: str):
        r = self.db.execute("SELECT * FROM calls WHERE id=?", (ident,)).fetchone()
        return dict(r) if r else None

    def scope(self):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('scope',?)", (canonical(uuid.uuid4().hex),))
        return json.loads(self.db.execute("SELECT value FROM meta WHERE k='scope'").fetchone()[0])

    def reserve(self, ident: str, request: dict, amount: float):
        if not math.isfinite(amount) or amount <= 0:
            raise GateError("missing/non-positive cost bound")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.row(ident):
                return
            if self.db.execute("SELECT 1 FROM meta WHERE k='cancelled'").fetchone():
                raise GateError("run cancelled; new dispatches disabled")
            spent = self.db.execute("SELECT COALESCE(SUM(MAX(reserve,COALESCE(actual,0))),0),COUNT(*) FROM calls").fetchone()
            if spent[0] + amount > self.cap_usd + 1e-10 or spent[1] >= self.max_calls:
                raise GateError("estimated dispatch budget or call count exceeded")
            self.db.execute("INSERT INTO calls(id,request,state,reserve,created_at) VALUES (?,?, 'reserved',?,?)",
                            (ident, canonical(request), amount, now()))

    def update(self, ident: str, **fields):
        allowed = {"state", "actual", "response", "provider_id", "dispatches", "polls", "elapsed_s", "error", "next_poll_at"}
        if not fields or not set(fields) <= allowed:
            raise ValueError("invalid journal field")
        if fields.get("actual") is not None and (not math.isfinite(fields["actual"]) or fields["actual"] < 0):
            raise GateError("invalid actual cost")
        with self.db:
            self.db.execute("UPDATE calls SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                            (*fields.values(), ident))

    def cancel(self):
        self.bind("cancelled", True)

    def claim(self, ident: str):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM meta WHERE k='cancelled'").fetchone():
                raise GateError("run cancelled; new dispatches disabled")
            spent = self.db.execute("SELECT COALESCE(SUM(MAX(reserve,COALESCE(actual,0))),0) FROM calls").fetchone()[0]
            if spent > self.cap_usd + 1e-10:
                raise GateError("known spending/reservations exceed budget; dispatch disabled")
            changed = self.db.execute("UPDATE calls SET state='sent',dispatches=dispatches+1 "
                                      "WHERE id=? AND state IN ('reserved','rejected') AND dispatches<2", (ident,))
            if changed.rowcount != 1:
                raise UncertainWork("dispatch already claimed or retry limit reached")

    def totals(self):
        rows = [dict(x) for x in self.db.execute("SELECT * FROM calls")]
        unknown = [x["id"] for x in rows if x["actual"] is None]
        return {"reserved_upper_estimate_usd": round(sum(max(x["reserve"], x["actual"] or 0) for x in rows), 6),
                "known_actual_usd": round(sum(x["actual"] or 0 for x in rows), 6),
                "actual_total_usd": None if unknown else round(sum(x["actual"] for x in rows), 6),
                "unknown_actual_call_ids": unknown, "logical_calls": len(rows),
                "dispatches": sum(x["dispatches"] for x in rows), "polls": sum(x["polls"] for x in rows),
                "hard_all_in_cap": False}


class Harness:
    def __init__(self, journal: Journal, transport: Transport, controller: Controller, *, clock=time.time):
        # User-authorized offline boundary. Do not turn an example approval JSON into paid authority.
        if not transport.offline or not controller.offline:
            raise GateError("live execution disabled until parent wires secure access and canonical paid admission")
        if controller.model != MODEL or not controller.snapshot:
            raise GateError("controller model and snapshot must be pinned")
        self.journal, self.transport, self.controller = journal, transport, controller
        self.clock, self.scope = clock, journal.scope()
        self.freeze = verify_freeze()
        verify_caches()
        self.prompt = (ROOT / "prompt.txt").read_text()
        journal.bind("identity", {"version": VERSION, "freeze": self.freeze, "prompt_sha256": digest(self.prompt),
                                  "controller": controller.model, "controller_snapshot": controller.snapshot,
                                  "offline": True, "rates": RATES,
                                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def call_id(self, request: dict, cell=None):
        value = {"freeze": self.freeze, "request": request}
        if cell is not None:
            value["cell"] = cell
        return digest(value)

    def request_for(self, case: dict, mode: str):
        adapter = Adapter(mode)
        scope = "bp20_" + self.scope + "_" + case["id"].replace("-", "_") + "_" + mode.split("-")[1]
        return adapter.request(case, memory_scope_key=scope if adapter.delegated else None)

    async def dispatch(self, request: dict, reserve: float, *, controller=False, cell=None) -> dict:
        ident = self.call_id(request, cell)
        self.journal.reserve(ident, request, reserve)
        row = self.journal.row(ident)
        if row["state"] == "completed":
            return json.loads(row["response"])
        if row["state"] not in {"reserved", "rejected"}:
            raise UncertainWork(f"reconcile {ident}; do not resubmit")
        # At most two proved-unaccepted attempts; crashes after intent leave 'sent'.
        if row["dispatches"] >= 2:
            raise GateError("proved-rejection retry limit reached")
        self.journal.claim(ident)
        started = time.monotonic()
        try:
            fn = self.controller.answer if controller else self.transport.call
            response = await asyncio.wait_for(fn(request, timeout_s=request["timeout_s"], max_bytes=RESPONSE_BYTES),
                                              timeout=request["timeout_s"])
            raw = canonical(response)
            if len(raw.encode()) > RESPONSE_BYTES:
                raise ValueError("response exceeds retained-byte limit")
            pid = response.get("run_id")
            if pid:
                self.journal.update(ident, state="pending", response=raw, provider_id=pid,
                                    elapsed_s=time.monotonic() - started)
            else:
                usage = response.get("usage", {})
                actual = usage.get("actual_usd")  # transport must distinguish invoice from estimate
                self.journal.update(ident, state="completed", response=raw, actual=actual,
                                    elapsed_s=time.monotonic() - started)
            return response
        except ConfirmedRejected:
            self.journal.update(ident, state="rejected", error="proved_not_accepted",
                                elapsed_s=time.monotonic() - started)
            raise
        except BaseException as exc:
            self.journal.update(ident, state="uncertain", error=type(exc).__name__,
                                elapsed_s=time.monotonic() - started)
            raise

    async def reconcile(self, request: dict) -> dict:
        ident = self.call_id(request)
        row = self.journal.row(ident)
        if not row or not row["provider_id"]:
            raise UncertainWork("provider ID unavailable; parent must reconcile dashboard/invoice")
        if row["state"] == "completed":
            return json.loads(row["response"])
        if row["polls"] >= MAX_POLLS:
            raise GateError("durable poll limit reached; parent reconciliation required")
        if self.clock() < row["next_poll_at"]:
            raise UncertainWork("poll cadence not reached; no HTTP call made")
        age = self.clock() - datetime.fromisoformat(row["created_at"]).timestamp()
        if age > TASK_DEADLINE_S:
            raise GateError("Task observation deadline exceeded; remote task may remain billable; parent reconciliation required")
        self.journal.update(ident, polls=row["polls"] + 1, next_poll_at=self.clock() + POLL_INTERVAL_S)
        poll = Adapter(request["mode"]).poll(row["provider_id"])
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(self.transport.call(poll, timeout_s=poll["timeout_s"], max_bytes=RESPONSE_BYTES),
                                              timeout=poll["timeout_s"])
            raw = canonical(response)
            if len(raw.encode()) > RESPONSE_BYTES:
                raise ValueError("response exceeds retained-byte limit")
            status = response.get("run", {}).get("status")
            self.journal.update(ident, elapsed_s=row["elapsed_s"] + time.monotonic() - started)
            if status == "completed":
                self.journal.update(ident, state="completed", response=raw,
                                    actual=response.get("usage", {}).get("actual_usd"))
            elif status in {"failed", "cancelled"}:
                self.journal.update(ident, state=status, response=raw)
                raise UncertainWork("terminal unsuccessful work; preserve reservation and verify billing")
            else:
                self.journal.update(ident, state="pending", response=raw)
                raise UncertainWork("provider work remains pending; resume GET only")
            return response
        except (TimeoutError, OSError):
            self.journal.update(ident, error="poll_failed")
            raise

    async def run_case(self, case: dict, mode: str) -> dict:
        adapter = Adapter(mode)
        req = self.request_for(case, mode)
        ident = self.call_id(req)
        row = self.journal.row(ident)
        if row and row["provider_id"]:
            raw = await self.reconcile(req)
        else:
            raw = await self.dispatch(req, RATES[mode])
            if raw.get("run_id"):
                raw = await self.reconcile(req)
        evidence = adapter.normalize(raw, retrieved_at=self.journal.row(ident)["created_at"])
        # Evidence is delimited JSON data; provider branding/mode is absent from Sol inputs.
        answer_request = {"case": case, "instructions": self.prompt, "evidence": evidence,
                          "model": MODEL, "snapshot": self.controller.snapshot,
                          "max_input_tokens": INPUT_TOKENS, "max_output_tokens": OUTPUT_TOKENS,
                          "external_tools": [], "timeout_s": 60, "service_tier": "standard"}
        # Unique controller cell key keeps independent answers per arm, even for identical evidence.
        cell = case["id"] + ":" + mode
        answer = await self.dispatch(answer_request, CONTROLLER_RESERVE, controller=True, cell=cell)
        answer_id = self.call_id(answer_request, cell)
        return {"case_id": case["id"], "mode": mode, "comparison_class": evidence["kind"],
                "status": "completed",
                "controller": MODEL, "controller_snapshot": self.controller.snapshot,
                "snapshot_version": self.freeze, "prompt_version": digest(self.prompt),
                "provider_request_id": ident, "controller_request_id": answer_id,
                "provider_usage": raw.get("usage"), "controller_usage": answer.get("usage"),
                "answer": answer, "sources": evidence.get("sources", []),
                "grading": {"state": "pending_blinded_human_review", "unsupported_claims": None,
                            "citation_support": None, "quote_support": None, "coverage": None,
                            "freshness": None, "useful_accepted_fields": None},
                "latency_s": {"provider_submit": self.journal.row(ident)["elapsed_s"],
                              "controller": self.journal.row(answer_id)["elapsed_s"]}, "offline": True}


def verify_freeze() -> str:
    freeze = json.loads((ROOT / "freeze.json").read_text())
    for name, expected in freeze["files"].items():
        if Path(name).name != name or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise GateError("frozen benchmark bytes changed")
    return digest(freeze)


def verify_caches(*, require=False) -> dict:
    """Check full-read snapshots when present; portable frozen facts remain in JSON.

    Parent must require these checks before evidence signoff for paid evaluation.
    Third-party full pages are deliberately absent from the portable PR.
    """
    verified, missing = [], []
    sources = json.loads((ROOT / "source_evidence.json").read_text())
    docs = json.loads((ROOT / "provider_docs_manifest.json").read_text())
    paths = [(ROOT / "source_cache" / (x["id"] + ".txt"), x["text_sha256"]) for x in sources]
    paths += [(ROOT / "docs_cache" / (x["id"] + ".md"), x["snapshot_sha256"]) for x in docs]
    for path, expected in paths:
        if not path.is_file():
            missing.append(path.name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise GateError("cached source/document bytes changed: " + path.name)
        else:
            verified.append(path.name)
    if require and missing:
        raise GateError("full-read snapshots missing; independent recheck/refreeze required")
    return {"verified_count": len(verified), "missing": missing, "full_source_review_ready": not missing}


class MockTransport:
    offline = True

    async def call(self, request, *, timeout_s, max_bytes):
        mode = request["mode"]
        if request["method"] == "GET":
            return {"run": {"status": "completed"}, "output": {"content": {"facts": "MOCK ONLY", "useful_fields": "null", "unknowns": "all"}, "basis": []},
                    "usage": {"actual_usd": 0, "simulated_tariff_usd": RATES[mode]}}
        if mode in TASK_MODES:
            return {"run_id": "trun_mock_" + request["case_id"].replace("-", "_") + mode.split("-")[1]}
        return {"results": [{"url": "https://example.org/public-mock", "title": "Hermetic fixture",
                             "snippet": "MOCK ONLY. No provider or research quality evidence.",
                             "excerpts": ["MOCK ONLY. No provider or research quality evidence."]}],
                "usage": {"actual_usd": 0, "simulated_tariff_usd": RATES[mode]}}


class MockController:
    offline, model, snapshot = True, MODEL, "hermetic-mock-v1"

    async def answer(self, request, *, timeout_s, max_bytes):
        return {"summary": "MOCK ONLY; no answer evaluation performed.",
                "useful_fields": dict.fromkeys(request["case"]["requested_fields"]),
                "claims": [], "sources": [], "conflicts": [], "unknowns": ["all fields require evaluation"],
                "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "actual_usd": 0}}


def estimate(deepen=False) -> dict:
    retrieval = 20 * sum(RATES[x] for x in RAW_MODES)
    controllers = 80
    if deepen:
        retrieval += len(HARD_CASES) * sum(RATES[x] for x in TASK_MODES)
        controllers += 12
    return {"raw_cases": 20, "raw_cells": 80, "task_cells": 12 if deepen else 0,
            "provider_estimate_usd": round(retrieval, 3), "controller_calls": controllers,
            "typical_tokens_estimate_usd": round(retrieval + controllers * .016, 3),
            "bounded_token_envelope_estimate_usd": round(retrieval + controllers * CONTROLLER_RESERVE, 3),
            "unknowns": ["incremental hosted Agents API compute", "tax/contract/region or service-tier uplift",
                         "unreconciled paid outcomes; provider-side cancellation/billing"],
            "proposed_dispatch_budget_usd": 10, "approved": False, "hard_all_in_cap": False}


def run_plan(deepen=False):
    cases = json.loads((ROOT / "cases.json").read_text())
    cells = []
    for i, c in enumerate(cases):
        order = RAW_MODES[i % 4:] + RAW_MODES[:i % 4]
        cells.extend([{"case_id": c["id"], "mode": m} for m in order])
        if deepen and c["id"] in HARD_CASES:
            cells.extend([{"case_id": c["id"], "mode": m} for m in TASK_MODES])
    return {"stage": "raw20+task6" if deepen else "raw20", "cells": cells}


async def mock_run(path: Path, deepen=False):
    if path.exists():
        raise GateError("use a new output directory; call-level restart is tested via API")
    path.mkdir(parents=True)
    journal = Journal(path / "journal.sqlite3", cap_usd=10)
    harness = Harness(journal, MockTransport(), MockController())
    cells = []
    cases = json.loads((ROOT / "cases.json").read_text())
    for i, case in enumerate(cases):
        # Balanced deterministic arm-order rotation; freeze this before paid evaluation.
        order = RAW_MODES[i % 4:] + RAW_MODES[:i % 4]
        for mode in order:
            cells.append(await harness.run_case(case, mode))
        if deepen and case["id"] in HARD_CASES:
            for mode in TASK_MODES:
                cells.append(await harness.run_case(case, mode))
    report = {"offline": True, "quality_results": "not_evaluated", "freeze": harness.freeze,
              "run_plan": run_plan(deepen),
              "estimate": estimate(deepen), "accounting": journal.totals(), "cells": cells}
    (path / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(canonical({"offline": True, "cells": len(cells), "actual_usd": report["accounting"]["actual_total_usd"],
                     "result": str(path / "results.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "estimate", "mock"])
    parser.add_argument("--deepen", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "mock")
    parser.add_argument("--require-caches", action="store_true")
    args = parser.parse_args()
    if args.command == "mock":
        asyncio.run(mock_run(args.out, args.deepen))
    elif args.command == "estimate":
        print(json.dumps(estimate(args.deepen), indent=2))
    else:
        print(canonical({"freeze_sha256": verify_freeze(), "cases": 20, "live": "disabled",
                         "cache_check": verify_caches(require=args.require_caches)}))


if __name__ == "__main__":
    main()
