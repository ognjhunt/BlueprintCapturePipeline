"""Dependency-light native Claude transport shared by admitted local callers.

Fixed endpoints, scoped existing credentials, no redirect/retry, bounded bodies,
and serialized durable reservation admission. No Agents SDK import is required.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

MODEL = "claude-opus-5-5"  # Retained metadata endpoint contract.
KEY_FILE_ENV = "ANTHROPIC_API_KEY_FILE"
_API_URL = "https://api.anthropic.com/v1/messages"
_MAX_REQUEST_BYTES = 30_000_000
_MAX_MESSAGE_RESPONSE_BYTES = 8 * 1024 * 1024

class ClaudeAuthoringBlocked(RuntimeError):
    """The provider, rights, spend, or output boundary failed closed."""


class _NoProviderRedirects(urllib_request.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        # Refuse without draining an unbounded redirect body or retaining its
        # socket; opener.open raises before the response context is entered.
        try:
            fp.close()
        finally:
            raise ClaudeAuthoringBlocked("claude_provider_redirect_refused")

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _request_json(req: urllib_request.Request, *, timeout: int,
                  maximum_bytes: int) -> Mapping[str, Any]:
    # These two adapters have fixed contracts. Never forward their credentials
    # to a redirect or a caller-selected host, path, query, or HTTP method.
    endpoints = {
        "POST": "https://api.anthropic.com/v1/messages",
        "GET": f"https://api.anthropic.com/v1/models/{MODEL}",
    }
    endpoint = endpoints.get(req.get_method())
    if endpoint is None or req.full_url != endpoint:
        raise ClaudeAuthoringBlocked("claude_provider_endpoint_invalid")
    opener = urllib_request.build_opener(_NoProviderRedirects())
    with opener.open(req, timeout=timeout) as response:
        if response.geturl() != endpoint or not 200 <= response.getcode() < 300:
            raise ClaudeAuthoringBlocked("claude_provider_response_invalid")
        length = response.headers.get("Content-Length")
        if length is not None:
            if not re.fullmatch(r"[0-9]{1,10}", length):
                raise ClaudeAuthoringBlocked("claude_provider_response_invalid")
            expected_bytes = int(length)
            if expected_bytes > maximum_bytes:
                raise ClaudeAuthoringBlocked("claude_provider_response_bytes_exceeded")
        else:
            expected_bytes = None
        body = bytearray()
        while True:
            chunk = response.read(min(64 * 1024, maximum_bytes + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > maximum_bytes:
                raise ClaudeAuthoringBlocked("claude_provider_response_bytes_exceeded")
        if expected_bytes is not None and len(body) != expected_bytes:
            raise ClaudeAuthoringBlocked("claude_provider_response_truncated")
        try:
            value = json.loads(body)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ClaudeAuthoringBlocked("claude_provider_response_invalid") from None
        if not isinstance(value, dict):
            raise ClaudeAuthoringBlocked("claude_provider_response_invalid")
        return value


def _scoped_key(key_file: str | Path | None = None) -> str:
    named = str(key_file) if key_file is not None else os.environ.get(KEY_FILE_ENV, "")
    path = Path(named)
    if not named or not path.is_absolute():
        raise ClaudeAuthoringBlocked("claude_key_file_missing")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ClaudeAuthoringBlocked("claude_key_file_missing") from exc
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        mode = os.fstat(stream.fileno()).st_mode
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise ClaudeAuthoringBlocked("claude_key_file_permissions_invalid")
        value = stream.read(16_384).strip()
        too_long = bool(stream.read(1))
    if too_long:
        raise ClaudeAuthoringBlocked("claude_key_file_invalid")
    if not value:
        raise ClaudeAuthoringBlocked("claude_key_file_empty")
    return value


def _post_message(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    if len(body) > _MAX_REQUEST_BYTES:
        raise ClaudeAuthoringBlocked("claude_request_bytes_exceeded")
    req = urllib_request.Request(_API_URL, data=body, method="POST", headers={
        "content-type": "application/json", "anthropic-version": "2023-06-01",
    })
    req.add_unredirected_header("x-api-key", key)
    # urllib has no automatic model retry. Unknown outcomes retain the full
    # reservation and must be reconciled before any same-identity replay.
    return _request_json(req, timeout=600, maximum_bytes=_MAX_MESSAGE_RESPONSE_BYTES)


@contextmanager
def _admission_lock(audit: Any):
    """Serialize Claude reservations sharing one attempt ledger."""
    audit.run_root.mkdir(parents=True, exist_ok=True)
    with (audit.run_root / ".claude_authoring_admission.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


