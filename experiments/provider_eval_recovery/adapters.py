"""Current documented wire envelopes only. No networking or credential handling."""

from dataclasses import dataclass
from decimal import Decimal
import re
from urllib.parse import parse_qsl, urlsplit

MODEL = "gpt-6.1-sol"
MODES = ("parallel_fast", "parallel_advanced", "perplexity_fast", "perplexity_standard")
RATES = dict(zip(MODES, map(Decimal, ("0.001", "0.005", "0.001", "0.005"))))


@dataclass(frozen=True)
class Limits:
    max_calls: int = 1
    max_attempts: int = 2
    max_reconciliations: int = 2
    max_sources: int = 10
    evidence_chars: int = 12000
    timeout_seconds: int = 30
    cell_cost_cap: Decimal = Decimal("0.125")

    def __post_init__(self):
        for field in ("max_calls", "max_reconciliations", "max_sources", "evidence_chars",
                      "timeout_seconds"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError("limits must be positive integers")
        if self.max_attempts not in (1, 2) or self.max_calls > 3 or self.max_sources > 10:
            raise ValueError("frozen experiment bounds exceeded")
        if not self.cell_cost_cap.is_finite() or self.cell_cost_cap <= 0:
            raise ValueError("invalid cost cap")


PUBLIC_KEYS = {"case_id", "prompt", "search_queries", "hypothetical_scenario"}


def public_case(case):
    # Reject, rather than silently strip, fields from the wrong input partition.
    if set(case) != PUBLIC_KEYS:
        raise ValueError("public case schema: extra or missing keys")
    if not isinstance(case["case_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case["case_id"]):
        raise ValueError("unsafe case id")
    for field in ("prompt", "hypothetical_scenario"):
        if not isinstance(case[field], str) or not case[field].strip():
            raise ValueError("invalid public text")
    queries = case["search_queries"]
    if not isinstance(queries, list) or len(queries) != 3:
        raise ValueError("exactly three frozen queries required")
    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("invalid query")
    return {key: case[key] for key in sorted(PUBLIC_KEYS)}


def request(mode, case, query_index, limits=Limits()):
    case = public_case(case)
    if mode not in MODES or not 0 <= query_index < limits.max_calls:
        raise ValueError("mode or query index outside frozen matrix")
    # Identical text and query sequence. Native excerpt controls are not equivalent
    # token/character units; apply the identical downstream character cap too.
    query = case["search_queries"][query_index]
    objective = case["prompt"] + "\nHypothetical scenario: " + case["hypothetical_scenario"]
    shared_query = objective + "\nResearch query: " + query
    if mode.startswith("parallel_"):
        return {"method": "POST", "url": "https://api.parallel.ai/v1/search",
                "body": {"mode": mode.split("_")[1], "objective": shared_query,
                         "search_queries": [shared_query], "max_chars_total": limits.evidence_chars,
                         "client_model": MODEL}, "timeout_seconds": limits.timeout_seconds}
    return {"method": "POST", "url": "https://api.perplexity.ai/search",
            "body": {"query": shared_query,
                     "search_type": "fast" if mode == "perplexity_fast" else "web",
                     "max_results": limits.max_sources, "max_tokens": limits.evidence_chars},
            "timeout_seconds": limits.timeout_seconds}


def normalize(mode, raw, limits=Limits()):
    if mode not in MODES or not isinstance(raw.get("results"), list):
        raise ValueError("invalid provider response")
    normalized, remaining = [], limits.evidence_chars
    for result in raw["results"][:limits.max_sources]:
        url = result.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
            raise ValueError("invalid citation URL")
        if parsed.password or parsed.fragment:
            raise ValueError("credential-bearing or noncanonical URL")
        if parsed.query:
            # Confirmed public BMW article locale selector. Do not admit arbitrary
            # query keys/values, signed links, redirect targets or personal IDs.
            parameters = parse_qsl(parsed.query, keep_blank_values=True,
                                   strict_parsing=True, max_num_fields=1)
            if (parsed.scheme != "https" or parsed.hostname != "www.press.bmwgroup.com"
                    or not re.fullmatch(r"/global/article/detail/T[0-9]+[A-Z]{2}/[A-Za-z0-9-]+", parsed.path)
                    or len(parameters) != 1 or parameters[0][0] != "language"
                    or not re.fullmatch(r"[a-z]{2}(?:-[A-Z]{2})?", parameters[0][1])):
                raise ValueError("credential-bearing or unsupported citation query")
        text = ("\n".join(result.get("excerpts", [])) if mode.startswith("parallel_")
                else result.get("snippet", ""))
        if not isinstance(text, str):
            raise ValueError("invalid snippet")
        title = result.get("title", "")
        if not isinstance(title, str):
            raise ValueError("invalid title")
        # Include title and URL in the common context cap, not just snippets.
        overhead = len(title) + len(url)
        if overhead > remaining:
            break
        text = text[:remaining - overhead]
        remaining -= overhead + len(text)
        normalized.append({"url": url, "title": title, "text": text,
                           "published_at": result.get("publish_date", result.get("date"))})
    return normalized


def live_send(*args, **kwargs):
    raise RuntimeError("live transport is deliberately absent; offline experiment only")
