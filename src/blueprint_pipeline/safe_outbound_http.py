"""Centralized fail-closed outbound HTTP boundary (FABLE-007).

Every urllib-based outbound HTTP call in the pipeline routes through this
module so scheme, host, redirect, timeout, and response-size policy live at
exactly ONE audited transport site instead of ad hoc ``urlopen`` calls
scattered across modules.

Policy summary:

* Allowed schemes are ``https`` everywhere, plus ``http`` only as an explicit
  loopback exception (``allow_loopback_http=True`` AND a loopback host) for
  services this pipeline intentionally runs on 127.0.0.1 (for example the
  persistent Isaac task executor).
* Callers that talk to one fixed API base pin ``allowed_hosts`` so a crafted
  path or URL cannot retarget the transport to another host.
* Redirects fail closed. A response whose final URL differs from the request
  URL is rejected: cross-scheme or cross-host movement is always
  ``outbound_http_redirect_escape_blocked``; same-origin redirects are only
  accepted when the policy explicitly opts in.
* Response bodies are size-capped fail-closed
  (``outbound_http_response_too_large``).
* Presigned object-store GET/PUT URLs stay supported:
  :func:`presigned_transfer_policy` pins the presigned URL's own https host
  for that single transfer while still rejecting unsupported schemes and
  redirect escape.

The single ``urllib.request.urlopen`` call below is the one deliberate,
triaged B310 site in the repository; it only ever receives a URL that already
passed :func:`validate_outbound_url`.
"""

from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Mapping

SCHEMA_VERSION = "safe_outbound_http.v1"

DEFAULT_TIMEOUT_SECONDS = 90.0
MAX_TIMEOUT_SECONDS = 3600.0
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
PRESIGNED_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
FILE_TRANSFER_CHUNK_BYTES = 1024 * 1024
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")


class SafeOutboundHttpError(ValueError):
    """Raised (with a precise snake_case blocker message) on policy violation."""


@dataclass(frozen=True)
class OutboundHttpPolicy:
    """Outbound transport policy for one caller.

    ``allowed_hosts=None`` means any host is acceptable (still https-only
    unless the loopback exception applies); a non-``None`` frozenset pins the
    transport to exactly those lowercase hostnames.
    """

    allowed_hosts: frozenset[str] | None = None
    allow_loopback_http: bool = False
    follow_same_origin_redirects: bool = False
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES


@dataclass(frozen=True)
class SafeHttpResponse:
    """Status, size-capped body, and final URL of one policy-validated call."""

    status: int
    body: bytes
    url: str
    final_url: str | None


@dataclass(frozen=True)
class SafeHttpFileTransfer:
    """Secret-free receipt facts for one digest-bound streaming transfer."""

    status: int
    transferred_bytes: int
    sha256: str
    host: str


def validate_outbound_url(
    url: str, *, policy: OutboundHttpPolicy
) -> urllib.parse.ParseResult:
    """Validate scheme/host/credentials for ``url`` under ``policy`` (fail closed)."""
    parsed = urllib.parse.urlparse(str(url or ""))
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    loopback_http = (
        scheme == "http" and policy.allow_loopback_http and host in LOOPBACK_HOSTS
    )
    if scheme != "https" and not loopback_http:
        raise SafeOutboundHttpError(f"outbound_http_scheme_not_allowed:{scheme or 'missing'}:{host}")
    if not host:
        raise SafeOutboundHttpError(f"outbound_http_host_missing:{str(url)[:120]}")
    if "@" in parsed.netloc:
        raise SafeOutboundHttpError(f"outbound_http_credentials_in_url_blocked:{host}")
    if policy.allowed_hosts is not None and host not in policy.allowed_hosts:
        raise SafeOutboundHttpError(f"outbound_http_host_not_allowed:{host}")
    return parsed


def pinned_api_policy(
    base_url: str,
    *,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> OutboundHttpPolicy:
    """Policy for a fixed https API base: the base URL's host is the only host."""
    parsed = urllib.parse.urlparse(str(base_url or ""))
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme != "https" or not host:
        raise SafeOutboundHttpError(
            f"outbound_http_scheme_not_allowed:{scheme or 'missing'}:{host or 'missing'}"
        )
    return OutboundHttpPolicy(
        allowed_hosts=frozenset({host}),
        max_response_bytes=max_response_bytes,
    )


def loopback_service_policy(
    *, max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
) -> OutboundHttpPolicy:
    """Policy for local sidecar services: loopback hosts only."""
    return OutboundHttpPolicy(
        allowed_hosts=LOOPBACK_HOSTS,
        allow_loopback_http=True,
        max_response_bytes=max_response_bytes,
    )


def service_endpoint_policy(
    endpoint_url: str,
    *,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> OutboundHttpPolicy:
    """Pin one configured HTTPS service origin or explicit loopback HTTP origin."""
    parsed = urllib.parse.urlparse(str(endpoint_url or ""))
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not host or (
        scheme != "https" and not (scheme == "http" and host in LOOPBACK_HOSTS)
    ):
        raise SafeOutboundHttpError(
            f"outbound_http_scheme_not_allowed:{scheme or 'missing'}:{host or 'missing'}"
        )
    return OutboundHttpPolicy(
        allowed_hosts=frozenset({host}),
        allow_loopback_http=scheme == "http" and host in LOOPBACK_HOSTS,
        max_response_bytes=max_response_bytes,
    )


def presigned_transfer_policy(
    presigned_url: str,
    *,
    max_response_bytes: int = PRESIGNED_MAX_RESPONSE_BYTES,
) -> OutboundHttpPolicy:
    """Policy for one presigned object-store GET/PUT: pin that URL's https host."""
    parsed = urllib.parse.urlparse(str(presigned_url or ""))
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme != "https" or not host:
        raise SafeOutboundHttpError(
            f"outbound_http_scheme_not_allowed:{scheme or 'missing'}:{host or 'missing'}"
        )
    return OutboundHttpPolicy(
        allowed_hosts=frozenset({host}),
        max_response_bytes=max_response_bytes,
    )


def _validated_timeout(timeout_seconds: object) -> float:
    try:
        timeout = float(timeout_seconds)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SafeOutboundHttpError(
            f"outbound_http_timeout_invalid:{timeout_seconds!r}"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise SafeOutboundHttpError(f"outbound_http_timeout_invalid:{timeout}")
    return min(timeout, MAX_TIMEOUT_SECONDS)


def _origin(parsed: urllib.parse.ParseResult) -> tuple[str, str, int | None]:
    return (parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port)


def _enforce_redirect_policy(
    requested: urllib.parse.ParseResult,
    final_url: str | None,
    *,
    policy: OutboundHttpPolicy,
) -> None:
    if not final_url or final_url == requested.geturl():
        return
    final = urllib.parse.urlparse(final_url)
    if _origin(final) != _origin(requested):
        raise SafeOutboundHttpError(
            f"outbound_http_redirect_escape_blocked:{(final.hostname or '').lower() or 'missing'}"
        )
    if not policy.follow_same_origin_redirects:
        raise SafeOutboundHttpError(
            f"outbound_http_redirect_blocked:{(final.hostname or '').lower() or 'missing'}"
        )
    # Same-origin redirect explicitly allowed: the final URL must still pass policy.
    validate_outbound_url(final_url, policy=policy)


class _PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate a redirect before urllib sends the redirected request."""

    def __init__(self, policy: OutboundHttpPolicy) -> None:
        super().__init__()
        self._policy = policy

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        requested = validate_outbound_url(str(req.full_url), policy=self._policy)
        resolved = urllib.parse.urljoin(str(req.full_url), str(newurl))
        final = urllib.parse.urlparse(resolved)
        if _origin(final) != _origin(requested):
            raise SafeOutboundHttpError(
                "outbound_http_redirect_escape_blocked:"
                f"{(final.hostname or '').lower() or 'missing'}"
            )
        if not self._policy.follow_same_origin_redirects:
            raise SafeOutboundHttpError(
                "outbound_http_redirect_blocked:"
                f"{(final.hostname or '').lower() or 'missing'}"
            )
        validate_outbound_url(resolved, policy=self._policy)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _open_with_policy(
    request_obj: urllib.request.Request,
    timeout: float,
    policy: OutboundHttpPolicy,
):
    """Open with a redirect handler that blocks escape before network I/O."""
    opener = urllib.request.build_opener(_PolicyRedirectHandler(policy))
    return opener.open(request_obj, timeout=timeout)


def _read_capped(response: object, max_bytes: int) -> bytes:
    reader = getattr(response, "read", None)
    if not callable(reader):
        raise SafeOutboundHttpError("outbound_http_response_unreadable")
    try:
        body = reader(max_bytes + 1)
    except TypeError:
        # Minimal transports (test doubles) expose read() without a size arg;
        # the cap below still applies to whatever they return.
        body = reader()
    if body is None:
        body = b""
    if len(body) > max_bytes:
        raise SafeOutboundHttpError(f"outbound_http_response_too_large:{len(body)}>{max_bytes}")
    return bytes(body)


def open_request(
    request_obj: urllib.request.Request,
    *,
    policy: OutboundHttpPolicy,
    timeout_seconds: object = DEFAULT_TIMEOUT_SECONDS,
    max_response_bytes: int | None = None,
) -> SafeHttpResponse:
    """Open one prebuilt ``urllib.request.Request`` through the audited boundary.

    ``urllib.error.HTTPError``/``URLError`` propagate so callers keep their
    existing provider-status handling; policy violations raise
    :class:`SafeOutboundHttpError` before (scheme/host) or instead of
    (redirect/size) accepting the response.
    """
    url = str(request_obj.full_url)
    requested = validate_outbound_url(url, policy=policy)
    timeout = _validated_timeout(timeout_seconds)
    cap = policy.max_response_bytes if max_response_bytes is None else int(max_response_bytes)
    if cap <= 0:
        raise SafeOutboundHttpError(f"outbound_http_response_cap_invalid:{cap}")
    # Redirect targets are validated by _PolicyRedirectHandler before urllib
    # issues the redirected request, so credentials and request bodies cannot
    # escape to another origin and a blocked redirect has no side effect.
    with _open_with_policy(request_obj, timeout, policy) as response:
        final_url: str | None = None
        geturl = getattr(response, "geturl", None)
        if callable(geturl):
            final_url = str(geturl() or "") or None
        _enforce_redirect_policy(requested, final_url, policy=policy)
        body = _read_capped(response, cap)
        status = int(getattr(response, "status", 200) or 200)
    return SafeHttpResponse(status=status, body=body, url=url, final_url=final_url)


def request(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: object = DEFAULT_TIMEOUT_SECONDS,
    policy: OutboundHttpPolicy,
    max_response_bytes: int | None = None,
) -> SafeHttpResponse:
    """Build and open one policy-validated outbound HTTP request."""
    validate_outbound_url(url, policy=policy)
    request_obj = urllib.request.Request(
        url, data=data, method=str(method).upper(), headers=dict(headers or {})
    )
    return open_request(
        request_obj,
        policy=policy,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
    )


def _require_sha256(value: object) -> str:
    digest = str(value or "")
    if _SHA256_RE.fullmatch(digest) is None:
        raise SafeOutboundHttpError("outbound_http_file_digest_invalid")
    return digest


def _hash_file_capped(path: Path, *, max_bytes: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(FILE_TRANSFER_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise SafeOutboundHttpError(
                    f"outbound_http_file_too_large:{size}>{max_bytes}"
                )
            digest.update(chunk)
    return size, "sha256:" + digest.hexdigest()


def download_file(
    url: str,
    *,
    output_path: str | Path,
    expected_sha256: str,
    max_bytes: int,
    timeout_seconds: object,
    policy: OutboundHttpPolicy,
    headers: Mapping[str, str] | None = None,
) -> SafeHttpFileTransfer:
    """Stream one exact HTTPS object to an atomically published local file."""

    requested = validate_outbound_url(url, policy=policy)
    expected = _require_sha256(expected_sha256)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise SafeOutboundHttpError(f"outbound_http_file_cap_invalid:{max_bytes!r}")
    timeout = _validated_timeout(timeout_seconds)
    destination = Path(output_path)
    if destination.is_symlink():
        raise SafeOutboundHttpError("outbound_http_output_symlink_forbidden")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "wb") as output_stream:
            request_obj = urllib.request.Request(
                url, method="GET", headers=dict(headers or {})
            )
            with _open_with_policy(request_obj, timeout, policy) as response:
                final_url: str | None = None
                geturl = getattr(response, "geturl", None)
                if callable(geturl):
                    final_url = str(geturl() or "") or None
                _enforce_redirect_policy(requested, final_url, policy=policy)
                status = int(getattr(response, "status", 200) or 200)
                if not 200 <= status < 300:
                    raise SafeOutboundHttpError(
                        f"outbound_http_transfer_status_invalid:{status}"
                    )
                reader = getattr(response, "read", None)
                if not callable(reader):
                    raise SafeOutboundHttpError("outbound_http_response_unreadable")
                while True:
                    chunk = reader(FILE_TRANSFER_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise SafeOutboundHttpError(
                            f"outbound_http_file_too_large:{size}>{max_bytes}"
                        )
                    digest.update(chunk)
                    output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        observed = "sha256:" + digest.hexdigest()
        if observed != expected:
            raise SafeOutboundHttpError("outbound_http_file_digest_mismatch")
        os.replace(temporary_path, destination)
        temporary_path = None
        return SafeHttpFileTransfer(
            status=status,
            transferred_bytes=size,
            sha256=observed,
            host=(requested.hostname or "").lower(),
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def upload_file(
    url: str,
    *,
    input_path: str | Path,
    expected_sha256: str,
    max_bytes: int,
    timeout_seconds: object,
    policy: OutboundHttpPolicy,
    content_type: str = "application/octet-stream",
    headers: Mapping[str, str] | None = None,
) -> SafeHttpFileTransfer:
    """Stream one exact local file to a pinned presigned HTTPS PUT URL."""

    requested = validate_outbound_url(url, policy=policy)
    expected = _require_sha256(expected_sha256)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise SafeOutboundHttpError(f"outbound_http_file_cap_invalid:{max_bytes!r}")
    source = Path(input_path)
    if source.is_symlink():
        raise SafeOutboundHttpError("outbound_http_input_symlink_forbidden")
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SafeOutboundHttpError("outbound_http_input_file_missing") from exc
    if not source.is_file():
        raise SafeOutboundHttpError("outbound_http_input_file_invalid")
    size, observed = _hash_file_capped(source, max_bytes=max_bytes)
    if observed != expected:
        raise SafeOutboundHttpError("outbound_http_file_digest_mismatch")
    timeout = _validated_timeout(timeout_seconds)
    request_headers = {
        "Content-Type": content_type,
        "Content-Length": str(size),
        **dict(headers or {}),
    }
    with source.open("rb") as input_stream:
        request_obj = urllib.request.Request(
            url,
            data=input_stream,  # type: ignore[arg-type]
            method="PUT",
            headers=request_headers,
        )
        with _open_with_policy(request_obj, timeout, policy) as response:
            final_url: str | None = None
            geturl = getattr(response, "geturl", None)
            if callable(geturl):
                final_url = str(geturl() or "") or None
            _enforce_redirect_policy(requested, final_url, policy=policy)
            status = int(getattr(response, "status", 200) or 200)
            if not 200 <= status < 300:
                raise SafeOutboundHttpError(
                    f"outbound_http_transfer_status_invalid:{status}"
                )
            _read_capped(response, policy.max_response_bytes)
    return SafeHttpFileTransfer(
        status=status,
        transferred_bytes=size,
        sha256=observed,
        host=(requested.hostname or "").lower(),
    )
