"""Private, immutable attempted-request evidence; no dispatch or replay command."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


class MeteredCallEvidenceError(RuntimeError):
    """Evidence cannot be persisted or the endpoint is forbidden."""


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _private_directory(path: Path) -> None:
    if not path.exists():
        _private_directory(path.parent)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _sync_directory(path.parent)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise MeteredCallEvidenceError("metered_evidence_directory_invalid")


def _immutable_write(path: Path, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _endpoint(request: urllib.request.Request) -> tuple[str, str, str, str | None]:
    parsed = urllib.parse.urlsplit(request.full_url)
    if (parsed.hostname or "").endswith("amazonaws.com"):
        return (
            "aws",
            "cost_explorer" if parsed.hostname == "ce.us-east-1.amazonaws.com" else "unknown",
            "denied",
            None,
        )
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
        or request.get_method() != "GET"
    ):
        return "unknown", "unknown", "denied", None
    pairs = {
        ("rest.runpod.io", "/v1/billing/pods"): ("runpod", "pods"),
        ("rest.runpod.io", "/v1/billing/endpoints"): ("runpod", "endpoints"),
        ("rest.runpod.io", "/v1/billing/networkvolumes"): ("runpod", "networkvolumes"),
        ("console.vast.ai", "/api/v0/charges/"): ("vast", "charges"),
        ("api.digitalocean.com", "/v2/customers/my/balance"): ("digitalocean", "balance"),
        ("api.digitalocean.com", "/v2/customers/my/invoices"): ("digitalocean", "invoices"),
    }
    match = pairs.get((parsed.hostname, parsed.path))
    if match is None:
        return "unknown", "unknown", "denied", None
    return *match, "billing_read", f"https://{parsed.hostname}{parsed.path}"


class MeteredTransport:
    """Wrap the existing Transport(Request, timeout), without retries or extra calls.

    An attempted event means dispatch *may* have happened. A process death between
    its durable write and the outcome remains unresolved, never a billed charge.
    """

    def __init__(self, transport: Callable, root: Path, *, run_id: str | None = None):
        self.transport = transport
        if root.is_symlink():
            raise MeteredCallEvidenceError("metered_evidence_directory_invalid")
        self.root = root.resolve()
        self.run_id = run_id or uuid.uuid4().hex
        if not self.run_id.isalnum():
            raise MeteredCallEvidenceError("metered_run_id_invalid")
        self.count = 0
        self.repeats: dict[str, int] = {}

    def __call__(self, request: urllib.request.Request, timeout: float) -> bytes:
        try:
            provider, service, operation, endpoint = _endpoint(request)
        except ValueError:
            provider, service, operation, endpoint = "unknown", "unknown", "denied", None
        request_digest = digest((request.get_method() + " " + request.full_url).encode())
        query_digest = digest(urllib.parse.urlsplit(request.full_url).query.encode())
        self.count += 1
        retry = self.repeats.get(request_digest, 0)
        self.repeats[request_digest] = retry + 1
        attempt_id = uuid.uuid4().hex
        directory = self.root / self.run_id
        _private_directory(directory)
        for private in (self.root, directory):
            info = private.lstat()
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise MeteredCallEvidenceError("metered_evidence_directory_not_private")
        base = {
            "schema_version": "blueprint.metered_call_event.v1",
            "run_id": self.run_id,
            "attempt_id": attempt_id,
            "attempt_sequence": self.count,
            "retry_index": retry,
            "page_identity": query_digest,
            "request_digest": request_digest,
            "provider": provider,
            "service": service,
            "operation": operation,
            "endpoint": endpoint,
            "billed_cost": None,
            "currency": None,
        }

        def persist(event: str, **details) -> None:
            _immutable_write(
                directory / f"{attempt_id}.{event}.json",
                base
                | {
                    "event": event,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    **details,
                },
            )

        persist("attempted")  # MUST succeed and fsync before calling the underlying transport.
        if endpoint is None:
            persist(
                "error",
                outcome="denied",
                exception="endpoint_forbidden",
                alert="metered_endpoint_attempt_denied",
            )
            raise MeteredCallEvidenceError("metered_endpoint_forbidden")
        try:
            payload = self.transport(request, timeout)
        except Exception as exc:
            # Never persist exception messages: providers can echo credentials or bodies.
            persist("error", outcome="error", exception_digest=digest(type(exc).__name__.encode()))
            raise
        persist(
            "received",
            outcome="received",
            response_digest=digest(payload),
            response_size_bytes=len(payload),
        )
        return payload


def read_attempts(root: Path) -> list[dict]:
    """Read-only projection. Missing outcomes remain unresolved; never replay."""
    attempts = []
    for path in sorted(root.glob("*/*.attempted.json")):
        attempted = json.loads(path.read_bytes())
        prefix = path.name.removesuffix(".attempted.json")
        received = path.with_name(prefix + ".received.json")
        error = path.with_name(prefix + ".error.json")
        outcome = "unresolved"
        if received.is_file():
            outcome = "received"
        elif error.is_file():
            outcome = json.loads(error.read_bytes()).get("outcome", "error")
        attempts.append(attempted | {"outcome": outcome, "billed_cost": None})
    return attempts


def summarize_attempts(root: Path) -> dict:
    """Local counts by request day, separate from usage and billed money."""
    counters: dict[tuple, dict] = {}
    try:
        for attempt in read_attempts(root):
            provider, service = attempt.get("provider"), attempt.get("service")
            if provider not in {
                "aws",
                "runpod",
                "vast",
                "digitalocean",
                "unknown",
            } or service not in {
                "cost_explorer",
                "pods",
                "endpoints",
                "networkvolumes",
                "charges",
                "balance",
                "invoices",
                "unknown",
            }:
                continue
            observed = datetime.fromisoformat(attempt["recorded_at"])
            day = observed.astimezone(ZoneInfo("America/Chicago")).date().isoformat()
            key = (provider, service, day)
            row = counters.setdefault(
                key,
                {
                    "provider": provider,
                    "service": service,
                    "request_day": day,
                    "timezone": "America/Chicago",
                    "attempted": 0,
                    "received": 0,
                    "error": 0,
                    "denied": 0,
                    "unresolved": 0,
                    "billed_cost": None,
                },
            )
            row["attempted"] += 1
            outcome = attempt.get("outcome")
            row[
                outcome
                if outcome in {"received", "error", "denied", "unresolved"}
                else "unresolved"
            ] += 1
        return {
            "coverage": "local_instrumented_transport_only",
            "rows": list(counters.values()),
            "billed_cost": None,
            "warnings": ["metered_attempt_outcome_requires_attention"]
            if any(r["denied"] or r["error"] or r["unresolved"] for r in counters.values())
            else [],
            "all_workflow_call_coverage": "unknown",
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "coverage": "unavailable",
            "rows": [],
            "billed_cost": None,
            "warnings": ["local_attempt_evidence_unreadable"],
        }
