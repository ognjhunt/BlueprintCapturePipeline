"""Existing binding HTTP routes; default gate refuses live research execution."""

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.client
import ipaddress
import json
from pathlib import Path
import socket
import ssl
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from blueprint_pipeline.agent_execution.operations import OperationPending, ToolRefused
from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission_grant
from experiments.provider_eval_adaptive_v1.citations import validated_url
from experiments.provider_eval_recovery.harness import Ledger, digest, exclusive, read_json, write_once
from experiments.provider_eval_recovery.live_http import (EXTRAS_PER_SEARCH_ATTEMPT, MAX_RESPONSE_BYTES,
    NoRedirect, existing_key)
from .evidence import PROTOCOL
from .hosted import live_admission

RATES = {"parallel_fast": "0.001", "parallel_advanced": "0.005", "perplexity_fast": "0.001", "perplexity_standard": "0.005"}


class ExistingSearchRoute:
    def __init__(self, aggregate_root, *, authorize=live_admission, opener=None):
        self.root = Path(aggregate_root).resolve()
        self.authorize = authorize
        self.opener = opener

    def _identity(self, request, context):
        key = digest({"protocol": PROTOCOL, "operation_id": context.operation_id, "request": request})
        return key, self.root / "protocols" / PROTOCOL / "raw" / (key + ".json")

    @staticmethod
    def _completed(ledger, key, path, request):
        if ledger.states.get(key) != "completed":
            raise OperationPending("completed_search_anchor_unavailable_no_retry")
        saved = read_json(path)
        done = next(e for e in ledger.events if e["kind"] == "completed" and e["attempt_id"] == key)
        reserved = next(e for e in ledger.events if e["kind"] == "reserved" and e["attempt_id"] == key)
        if (saved["request"] != request or digest(saved) != done["raw_sha256"]
                or reserved["request_sha256"] != digest(request)):
            raise ToolRefused("retained_search_changed")
        return saved["raw"]

    def verify_retained(self, mode, request, context):
        """Read-only independent paid-response anchor; no keys or dispatch."""
        key, path = self._identity(request, context)
        with exclusive(self.root):
            ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
            reserved = next((e for e in ledger.events if e["kind"] == "reserved" and e["attempt_id"] == key), {})
            if reserved.get("mode") != mode or reserved.get("protocol") != PROTOCOL:
                raise ToolRefused("retained_search_arm_anchor_mismatch")
            return self._completed(ledger, key, path, request)

    def __call__(self, mode, request, context):
        from decimal import Decimal
        expected = "https://api.parallel.ai/v1/search" if mode.startswith("parallel") else "https://api.perplexity.ai/search"
        if request["url"] != expected or request["method"] != "POST":
            raise ToolRefused("provider_isolation_refused")
        grant = self.authorize(context)
        require_paid_resource_admission_grant(grant, resource_class="evaluator_api",
            allocation_binding_digest=context.authority_digest, require_allocation_binding=True)
        key, path = self._identity(request, context)
        with exclusive(self.root):
            ledger = Ledger(self.root / "live_journal.jsonl", "10.00")
            if ledger.states.get(key) == "completed":
                return self._completed(ledger, key, path, request)
            if key in ledger.states:
                raise OperationPending("uncertain_paid_search_no_retry")
            provider = mode.split("_")[0]
            headers = {"Content-Type": "application/json"}
            value = existing_key(provider)
            headers["x-api-key" if provider == "parallel" else "Authorization"] = value if provider == "parallel" else "Bearer " + value
            amount = Decimal(RATES[mode]) + EXTRAS_PER_SEARCH_ATTEMPT
            ledger.append("reserved", key, amount_usd=str(amount), protocol=PROTOCOL, provider=provider,
                          role=PROTOCOL + ":search", request_sha256=digest(request), mode=mode)
            try:
                opener = self.opener or build_opener(NoRedirect())
                started = perf_counter()
                with opener.open(Request(expected, data=json.dumps(request["body"]).encode(), headers=headers, method="POST"), timeout=30) as response:
                    if response.status != 200:
                        raise ValueError("search_http_status")
                    raw_bytes = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw_bytes) > MAX_RESPONSE_BYTES:
                    raise ValueError("search_response_bound")
                raw = json.loads(raw_bytes)
                if not isinstance(raw, dict):
                    raise ValueError("search_response_shape")
                retained = {"request": request, "raw": raw, "http_latency_seconds": perf_counter() - started,
                            "billing": "unreconciled; full reservation retained"}
                write_once(path, retained)
                ledger.append("completed", key, raw_sha256=digest(retained))
                return raw
            except Exception:
                ledger.append("uncertain", key, reason="hosted_search_outcome_uncertain_no_retry")
                raise OperationPending("uncertain_paid_search_no_retry") from None


class TextHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def public_fetch(request, context, *, authorize=live_admission, resolve=socket.getaddrinfo, connect=None):
    """Bounded full public text, no cookies/auth or redirects; DNS is pinned."""
    authorize(context)
    if validated_url("parallel_fast", request["url"])[0] != request["url"]:
        raise ToolRefused("fetch_requires_validated_display_citation")
    parsed = urlsplit(request["url"])
    if (request.get("method") != "GET" or parsed.scheme not in {"https", "http"}
            or parsed.username is not None or parsed.password is not None or parsed.port is not None
            or not parsed.hostname):
        raise ToolRefused("public_fetch_url_refused")
    rows = resolve(parsed.hostname, 443 if parsed.scheme == "https" else 80, type=socket.SOCK_STREAM)
    addresses = [row[4][0] for row in rows]
    if not addresses or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast for ip in addresses):
        raise ToolRefused("public_fetch_nonpublic_destination_refused")
    ip = addresses[0]
    if connect is None:
        connection = (http.client.HTTPSConnection(parsed.hostname, timeout=20) if parsed.scheme == "https"
                      else http.client.HTTPConnection(parsed.hostname, timeout=20))
        pinned = socket.create_connection((ip, 443 if parsed.scheme == "https" else 80), timeout=20)
        connection.sock = ssl.create_default_context().wrap_socket(pinned, server_hostname=parsed.hostname) if parsed.scheme == "https" else pinned
    else:
        connection = connect(parsed, ip)
    try:
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        connection.request("GET", target, headers={"User-Agent": "BlueprintPrivateEval/2", "Accept": "text/html,text/plain,application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ToolRefused("source_fetch_http_or_redirect_refused")
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ToolRefused("source_fetch_size_gap_no_partial_page")
        kind = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        if kind not in {"text/html", "text/plain", "application/json", "application/xhtml+xml"}:
            raise ToolRefused("source_content_not_supported_pdf_or_binary_unknown")
        text = body.decode("utf-8")
        if kind in {"text/html", "application/xhtml+xml"}:
            parser = TextHTML()
            parser.feed(text)
            text = "\n".join(parser.parts)
        return {"url": request["url"], "text": text, "body_sha256": hashlib.sha256(body).hexdigest(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "redirects_followed": False,
                "complete_body": True, "content_type": kind}
    finally:
        connection.close()
