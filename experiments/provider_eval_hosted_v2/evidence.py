"""Complete, provider-isolated evidence files and durable research functions."""

from pathlib import Path
import re
from urllib.parse import urlsplit

from blueprint_pipeline.agent_execution.contracts import AgentTool, ToolReconciliation
from blueprint_pipeline.agent_execution.operations import OperationPending, ToolRefused
from experiments.provider_eval_adaptive_v1.citations import CitationNormalizationError, validated_url
from experiments.provider_eval_recovery.harness import Ledger, digest, exclusive, read_json, write_once

PROTOCOL = "hosted_agent_research_v2"
MODES = ("parallel_fast", "parallel_advanced", "perplexity_fast", "perplexity_standard")
MAX_SEARCHES = 3  # includes valid reused searches, not just new paid searches
MAX_FETCHES = 3


def search_request(mode, case, query):
    # Typed JSON, never shell syntax. Ordinary names such as Scan&Sand are valid.
    if mode not in MODES or not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
        raise ToolRefused("bounded_search_query_required")
    if any(ord(char) < 32 or ord(char) == 127 for char in query):
        raise ToolRefused("query_control_character_refused")
    objective = "Public research as of 2026-09-30. " + case["question"]
    if mode.startswith("parallel"):
        body = {"mode": mode.split("_")[1], "search_queries": [query], "objective": objective,
                "max_results": 10, "max_chars_total": 20000, "client_model": "gpt-6.1-sol"}
        url = "https://api.parallel.ai/v1/search"
    else:
        body = {"query": query + "\n" + objective, "max_results": 10, "max_tokens": 20000,
                "search_type": "fast" if mode.endswith("fast") else "web"}
        url = "https://api.perplexity.ai/search"
    return {"method": "POST", "url": url, "body": body}


class Evidence:
    def __init__(self, root, case, mode):
        if mode not in MODES or not re.fullmatch(r"BP-EVAL-[0-9]{2}", case.get("id", "")):
            raise ValueError("frozen_case_and_provider_arm_required")
        self.root = Path(root).resolve() / PROTOCOL / case["id"] / mode
        self.case, self.mode = case, mode
        self.root.mkdir(parents=True, exist_ok=True)
        write_once(self.root / "scope.json", {"protocol": PROTOCOL, "case": case, "mode": mode})

    def sources(self):
        return [self.source(path.stem) for path in sorted((self.root / "sources").glob("*.json"))]

    def ingest(self, raw, provenance):
        if not isinstance(raw, dict) or raw.get("warnings") or not isinstance(raw.get("results"), list):
            raise ToolRefused("provider_warning_or_invalid_source_list")
        accepted, quarantined = [], []
        for rank, item in enumerate(raw["results"], 1):
            if not isinstance(item, dict):
                raise ToolRefused("invalid_source_shape")
            try:
                url, rule = validated_url(self.mode, item.get("url"))
            except CitationNormalizationError:
                quarantined.append({"rank": rank, "raw_url_sha256": digest(item.get("url"))})
                continue
            text = item.get("excerpts", []) if self.mode.startswith("parallel") else [item.get("snippet", "")]
            if not isinstance(text, list) or any(not isinstance(s, str) for s in text):
                raise ToolRefused("invalid_full_source_text")
            source = {"url": url, "title": item.get("title", ""), "text": "\n".join(text),
                      "published_at": item.get("publish_date", item.get("date")), "provenance": provenance,
                      "rank": rank, "citation_rule": rule, "raw_url_sha256": digest(item.get("url")),
                      "authority": "unverified; source verification required"}
            source["id"] = digest(source)
            write_once(self.root / "sources" / (source["id"] + ".json"), source)
            accepted.append(source["id"])
        write_once(self.root / "imports" / (digest({"raw": raw, "provenance": provenance}) + ".json"),
                   {"provenance": provenance, "raw_sha256": digest(raw), "accepted": accepted, "quarantined": quarantined})
        return accepted

    def manifest(self):
        return [{"source_id": s["id"], "url": s["url"], "title": s["title"],
                 "characters": len(s["text"]), "path": "/workspace/evidence/" + s["id"] + ".json",
                 "published_at": s["published_at"], "provenance": s["provenance"]} for s in self.sources()]

    def read(self, arguments):
        source = self.source(arguments["source_id"])
        start, length = arguments["start"], arguments["length"]
        end = min(start + length, len(source["text"]))
        return {"source_id": source["id"], "url": source["url"], "start": start, "end": end,
                "total_characters": len(source["text"]), "text": source["text"][start:end],
                "next_start": end if end < len(source["text"]) else None}

    def source(self, source_id):
        if not isinstance(source_id, str) or not re.fullmatch(r"[a-f0-9]{64}", source_id):
            raise ToolRefused("source_from_this_arm_required")
        source = read_json(self.root / "sources" / (source_id + ".json"))
        if source.get("id") != source_id or digest({k: v for k, v in source.items() if k != "id"}) != source_id:
            raise ToolRefused("source_integrity_failure")
        return source

    def find(self, arguments):
        text = self.source(arguments["source_id"])["text"]
        term, start = arguments["term"], arguments["start"]
        hits = []
        for match in re.finditer(re.escape(term), text[start:], re.IGNORECASE):
            if len(hits) == 20:
                return {"offsets": hits, "next_start": hits[-1] + max(1, len(term)), "total_characters": len(text)}
            hits.append(start + match.start())
        return {"offsets": hits, "next_start": None, "total_characters": len(text)}

    def reused_count(self):
        return len(list((self.root / "reused_searches").glob("*.json")))

    def reuse(self, aggregate_root):
        """Only completed, warning-free, corrected native requests from this arm."""
        aggregate_root = Path(aggregate_root)
        ledger = Ledger(aggregate_root / "live_journal.jsonl", "10.00")
        for row in ledger.events:
            if (row["kind"] != "reserved" or row.get("protocol") != "bounded_adaptive_v1"
                    or row.get("cell") != self.case["id"][-2:] + "_" + self.mode
                    or row.get("step") not in {"search1", "search2"}
                    or ledger.states.get(row["attempt_id"]) != "completed"):
                continue
            attempt = row["attempt_id"]
            envelope = read_json(aggregate_root / "protocols/bounded_adaptive_v1/raw" / (attempt + ".json"))
            done = next(e for e in ledger.events if e["kind"] == "completed" and e["attempt_id"] == attempt)
            if digest(envelope) != done["raw_sha256"] or digest(envelope["request"]) != row["request_sha256"]:
                raise ToolRefused("retained_search_integrity_failure")
            body = envelope["request"]["body"]
            parallel = self.mode.startswith("parallel")
            expected_url = "https://api.parallel.ai/v1/search" if parallel else "https://api.perplexity.ai/search"
            expected_mode = self.mode.split("_", 1)[1] if parallel else ("fast" if self.mode.endswith("fast") else "web")
            if (envelope["request"].get("url") != expected_url
                    or body.get("mode" if parallel else "search_type") != expected_mode):
                raise ToolRefused("retained_search_arm_contract_mismatch")
            query = body.get("search_queries", [None])[0] if parallel else body.get("query", "").split("\n", 1)[0]
            objective = body.get("objective", "") if parallel else body.get("query", "")
            if (self.case["question"] not in objective or self.case["entity"] not in query
                    or len(query) > 200 or envelope["raw"].get("warnings")):
                continue
            ids = self.ingest(envelope["raw"], {"kind": "reused_corrected_search", "attempt_id": attempt,
                "retained_envelope_sha256": digest(envelope), "original_protocol": "bounded_adaptive_v1"})
            write_once(self.root / "reused_searches" / (attempt + ".json"), {"attempt": attempt, "source_ids": ids})
        if self.reused_count() > MAX_SEARCHES:
            raise ToolRefused("equal_search_opportunity_exceeded")
        return self.manifest()


class ResearchTools:
    """No model loop. Hosted agent selects functions; paid routes are caller-owned."""
    def __init__(self, evidence, *, search, fetch):
        self.evidence, self.search_route, self.fetch_route = evidence, search, fetch

    def _import_raw(self, name, raw, request, context):
        if name == "search":
            return self.evidence.ingest(raw, {"kind": "hosted_agent_search", "operation_id": context.operation_id})
        if raw.get("url") != request["url"] or not isinstance(raw.get("text"), str):
            raise ToolRefused("source_fetch_contract_failed")
        provider_raw = {"results": [{"url": raw["url"], "title": raw.get("title", ""),
                                    "excerpts": [raw["text"]], "snippet": raw["text"]}]}
        return self.evidence.ingest(provider_raw, {"kind": "public_page_fetch", "operation_id": context.operation_id,
            "retrieved_at": raw.get("retrieved_at"), "body_sha256": raw.get("body_sha256"),
            "redirects_followed": False})

    def _replay(self, name, arguments, context, folder):
        saved, intent, raw = (read_json(folder / filename) for filename in ("result.json", "intent.json", "raw.json"))
        request = (search_request(self.evidence.mode, self.evidence.case, arguments["query"]) if name == "search"
                   else {"method": "GET", "url": self.evidence.source(arguments["source_id"])["url"]})
        if (saved["arguments"] != arguments or intent != {"arguments": arguments, "request": request}
                or saved.get("operation_id") != context.operation_id or saved.get("name") != name
                or saved.get("request_sha256") != digest(request) or saved.get("raw_sha256") != digest(raw)):
            raise ToolRefused("operation_receipt_integrity_failure")
        anchor = getattr(self.search_route, "verify_retained", None)
        if name == "search" and anchor is not None and anchor(self.evidence.mode, request, context) != raw:
            raise ToolRefused("operation_paid_response_anchor_integrity_failure")
        ids = self._import_raw(name, raw, request, context)
        output = saved["output"]
        canonical = {s["source_id"]: s for s in self.evidence.manifest()}
        entries = output.get("sources", [])
        if (output.get("new_source_ids") != ids or not isinstance(entries, list)
                or len({s.get("source_id") for s in entries}) != len(entries)
                or any(s != canonical.get(s.get("source_id")) for s in entries)
                or not set(ids).issubset({s.get("source_id") for s in entries})):
            raise ToolRefused("operation_source_manifest_integrity_failure")
        return output

    def operation(self, name, arguments, context):
        if name == "list_evidence":
            return {"sources": self.evidence.manifest(), "searches_used": self.evidence.reused_count() +
                    len(list((self.evidence.root / "operations/search").glob("*/intent.json"))),
                    "max_searches": MAX_SEARCHES, "complete_evidence_available": True}
        if name == "read_evidence":
            return self.evidence.read(arguments)
        if name == "find_evidence":
            return self.evidence.find(arguments)
        root = self.evidence.root
        folder = root / "operations" / name / context.operation_id
        with exclusive(root):
            if (folder / "result.json").exists():
                return self._replay(name, arguments, context, folder)
            if (folder / "intent.json").exists():
                raise OperationPending("uncertain_tool_no_redispatch")
            used = len(list((root / "operations" / name).glob("*/intent.json")))
            used += self.evidence.reused_count() if name == "search" else 0
            if used >= (MAX_SEARCHES if name == "search" else MAX_FETCHES):
                raise ToolRefused("equal_arm_budget_stop")
            if name == "search":
                request = search_request(self.evidence.mode, self.evidence.case, arguments["query"])
            else:
                source = self.evidence.source(arguments["source_id"])
                url = source["url"]
                parsed = urlsplit(url)
                if parsed.port is not None or parsed.hostname in {"localhost"}:
                    raise ToolRefused("public_source_only")
                request = {"url": url, "method": "GET"}
            write_once(folder / "intent.json", {"arguments": arguments, "request": request})
            raw = (self.search_route(self.evidence.mode, request, context) if name == "search"
                   else self.fetch_route(request, context))
            write_once(folder / "raw.json", raw)
            try:
                ids = self._import_raw(name, raw, request, context)
            except ToolRefused:
                # Dispatch already completed: retain the raw response and stop,
                # never settle this as a refusal before external execution.
                raise OperationPending("completed_response_validation_failed_no_redispatch") from None
            output = {"new_source_ids": ids, "sources": self.evidence.manifest()}
            write_once(folder / "result.json", {"arguments": arguments, "output": output,
                "operation_id": context.operation_id, "name": name, "request_sha256": digest(request), "raw_sha256": digest(raw)})
            return output

    def reconcile(self, name, arguments, context):
        path = self.evidence.root / "operations" / name / context.operation_id / "result.json"
        if path.exists():
            try:
                output = self._replay(name, arguments, context, path.parent)
            except (ToolRefused, OperationPending, KeyError, TypeError, ValueError, FileNotFoundError):
                return ToolReconciliation("pending")
            return ToolReconciliation("completed", output)
        # A started provider call is never declared not_started from file absence.
        return ToolReconciliation("pending")

    def tools(self):
        string = {"type": "string", "minLength": 1, "maxLength": 200}
        source = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
        offset = {"type": "integer", "minimum": 0}
        fields = {"list_evidence": {}, "read_evidence": {"source_id": source, "start": offset,
                  "length": {"type": "integer", "minimum": 1, "maximum": 16000}},
                  "find_evidence": {"source_id": source, "term": string, "start": offset},
                  "search": {"query": string}, "fetch_source": {"source_id": source}}
        tools = []
        for name, properties in fields.items():
            external = name in {"search", "fetch_source"}
            tools.append(AgentTool(name, "hosted-eval-v2", "Provider-isolated " + name + "; unknowns remain explicit.",
                {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
                "external_side_effect" if external else "read_only",
                lambda args, ctx, name=name: self.operation("fetch" if name == "fetch_source" else name, args, ctx),
                (lambda args, ctx, name=name: self.reconcile("fetch" if name == "fetch_source" else name, args, ctx)) if external else None))
        return tuple(tools)
