"""Durable admission publication to the existing signed WebApp control plane."""

from __future__ import annotations

import json
import os
import time
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit

from ..common import write_json
from ..webapp_sync import _pipeline_sync_headers, validated_https_sync_url
from .contracts import AgentExecutionError, canonical_json, digest
from .journal import AgentJournal


ADMISSION_PATH = "/api/internal/pipeline/agent-execution/admissions"


def admission_payload(record):
    if "blueprint-webapp" not in record.owner_client_ids:
        return None
    task = record.task.snapshot()
    # Supervision owns execution; the Website still starts its result collector.
    return {"schema_version": "blueprint_webapp_agent_admission.v1", "task_id": task.task_id,
            "task_digest": task.task_digest, "run_id": task.run_id, "source_commit": task.source_commit,
            "runtime": task.admission.runtime, "model": task.model,
            "title": "Task Evaluation: " + task.capability.replace("_", " "),
            "owner_client_id": "blueprint-webapp", "expires_at": min(task.deadline, task.admission.expires_at),
            "enabled": record.enabled, "autostart": record.autostart or record.supervision is not None, "proof_effect": "none"}


def _endpoint(endpoint=None, *, expected_path=ADMISSION_PATH):
    explicit = endpoint if endpoint is not None else os.environ.get("BLUEPRINT_AGENT_WEBAPP_ADMISSION_URL", "").strip()
    configured = explicit or os.environ.get("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        return None
    parts = urlsplit(validated_https_sync_url(configured))
    if expected_path not in {ADMISSION_PATH, "/api/internal/pipeline/agent-execution/engineering",
                             "/api/internal/pipeline/agent-execution/paperclip-bindings"}:
        raise AgentExecutionError("agent_webapp_admission_endpoint_invalid")
    if explicit and parts.path != expected_path:
        raise AgentExecutionError("agent_webapp_admission_endpoint_invalid")
    return urlunsplit((parts.scheme, parts.netloc, expected_path, "", ""))


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def post_admission(payload, *, endpoint=None, token=None, expected_path=ADMISSION_PATH):
    endpoint = _endpoint(endpoint, expected_path=expected_path)
    token = os.environ.get("PIPELINE_SYNC_TOKEN", "").strip() if token is None else token
    if not endpoint or not token:
        raise AgentExecutionError("agent_webapp_admission_not_configured")
    body = canonical_json(payload).encode()
    headers = _pipeline_sync_headers(token, body)
    headers["Content-Type"] = "application/json"
    outgoing = request.Request(endpoint, method="POST", data=body, headers=headers)
    try:
        with request.build_opener(_NoRedirect(), request.ProxyHandler({})).open(outgoing, timeout=3) as response:
            raw = response.read(32_001)
    except (error.HTTPError, error.URLError, OSError, TimeoutError):
        raise AgentExecutionError("agent_webapp_admission_delivery_unresolved") from None
    if len(raw) > 32_000:
        raise AgentExecutionError("agent_webapp_admission_response_too_large")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        raise AgentExecutionError("agent_webapp_admission_response_invalid") from None


class WebappAdmissionOutbox:
    def __init__(self, journal: AgentJournal, *, post=post_admission):
        self.journal, self.post = journal, post
        self.root = journal.root / "webapp-admissions"
        self.pending = self.root / "pending"
        self.receipts = self.root / "receipts"
        self.pending.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.receipts.mkdir(mode=0o700, parents=True, exist_ok=True)

    def queue(self, record):
        payload = admission_payload(record)
        if payload is None:
            return None
        identity = digest(payload)[7:]
        with self.journal.own_task("webapp-admission:" + identity):
            if not (self.pending / (identity + ".json")).exists():
                write_json(self.pending / (identity + ".json"), payload)
        return "sha256:" + identity

    def flush(self, *, limit=1):
        results = []
        for path in sorted(self.pending.glob("*.json"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
            if len(results) >= limit:
                break
            try:
                with self.journal.own_task("webapp-admission:" + path.stem):
                    receipt_path = self.receipts / path.name
                    if receipt_path.exists():
                        continue
                    if path.is_symlink():
                        raise AgentExecutionError("agent_webapp_admission_link_refused")
                    payload = json.loads(path.read_text())
                    if digest(payload) != "sha256:" + path.stem:
                        raise AgentExecutionError("agent_webapp_admission_outbox_changed")
                    response = self.post(payload)
                    if (not isinstance(response, dict)
                            or response.get("schema_version") != "blueprint_webapp_agent_admission_receipt.v1"
                            or response.get("admission") != payload or response.get("proof_effect") != "none"):
                        raise AgentExecutionError("agent_webapp_admission_readback_mismatch")
                    receipt = {"schema_version": "blueprint_agent_webapp_admission_delivery.v1",
                               "task_id": payload["task_id"], "task_digest": payload["task_digest"],
                               "admission_digest": digest(payload), "response_digest": digest(response),
                               "observed_at": time.time(), "status": "stored_in_webapp", "proof_effect": "none"}
                    write_json(receipt_path, receipt)
                    results.append(receipt)
            except (AgentExecutionError, OSError, ValueError):
                # Keep the original outbox bytes after loss or refusal. This
                # does not alter execution status or assert recipient delivery.
                results.append({"admission_id": path.stem, "status": "delivery_pending", "proof_effect": "none"})
                # An independently refused task cannot monopolize every tick.
                # Only scheduling metadata changes; queued payload bytes stay
                # immutable and the receiver deduplicates their task identity.
                try:
                    newest = max(item.stat().st_mtime_ns for item in self.pending.glob("*.json"))
                    retry_order = max(time.time_ns(), newest + 1)
                    os.utime(path, ns=(path.stat().st_atime_ns, retry_order), follow_symlinks=False)
                except OSError:
                    pass
        return results
