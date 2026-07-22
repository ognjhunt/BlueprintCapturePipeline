"""Shared fail-closed validation for untrusted identifiers, paths, and URLs."""

from __future__ import annotations

import http.client
import ipaddress
import os
import queue
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GCS_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_DNS_LOOKUP_SLOTS = threading.BoundedSemaphore(value=16)
_METADATA_HOSTS = {
    "metadata",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
}


class SecurityValidationError(ValueError):
    """Raised when untrusted input violates a security boundary."""


def strict_identifier(value: object, *, field: str, max_length: int = 128) -> str:
    """Return a single safe path component with a stable public grammar."""

    text = value.strip() if isinstance(value, str) else ""
    if not text or len(text) > max_length or not _IDENTIFIER_RE.fullmatch(text):
        raise SecurityValidationError(
            f"{field} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,{max_length - 1}}}"
        )
    if text in {".", ".."}:
        raise SecurityValidationError(f"{field} may not be a dot path component")
    return text


def strict_gcs_bucket(value: object, *, field: str = "bucket") -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not _GCS_BUCKET_RE.fullmatch(text) or ".." in text:
        raise SecurityValidationError(f"{field} is not a valid GCS bucket name")
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return text
    raise SecurityValidationError(f"{field} may not be formatted as an IP address")


def contained_path(root: Path, *components: str, field: str = "path") -> Path:
    """Join components and prove the resolved result remains below ``root``."""

    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*components).resolve(strict=False)
    if candidate == resolved_root or not candidate.is_relative_to(resolved_root):
        raise SecurityValidationError(f"{field} escapes its designated root")
    return candidate


def prove_path_contained(root: Path, candidate: Path, *, field: str = "path") -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=False)
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise SecurityValidationError(f"{field} escapes its designated root")
    return resolved


def comma_separated_origins(value: str | None) -> tuple[str, ...]:
    origins: list[str] = []
    for item in str(value or "").split(","):
        text = item.strip()
        if text and text not in origins:
            origins.append(_canonical_origin(text))
    return tuple(origins)


def origins_from_env(name: str) -> tuple[str, ...]:
    return comma_separated_origins(os.getenv(name))


def _canonical_origin(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise SecurityValidationError("allowed origins must be absolute HTTPS origins")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SecurityValidationError("allowed origins may not contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise SecurityValidationError("allowed origins may not contain a path")
    host = parsed.hostname.lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SecurityValidationError("allowed origin port is invalid") from exc
    return f"https://{host}" + (f":{port}" if port and port != 443 else "")


def _url_origin(parsed: urllib.parse.SplitResult) -> str:
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SecurityValidationError("remote URL port is invalid") from exc
    return f"https://{host}" + (f":{port}" if port and port != 443 else "")


def _ip_is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _bounded_getaddrinfo(
    host: str,
    port: int,
    *,
    timeout_seconds: float | None,
) -> list[tuple[Any, ...]]:
    """Run the system resolver behind a bounded, fail-closed worker slot."""

    if timeout_seconds is None:
        return list(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise SecurityValidationError("DNS resolution exceeded total time limit")
    started = time.monotonic()
    if not _DNS_LOOKUP_SLOTS.acquire(timeout=timeout):
        raise SecurityValidationError("DNS resolver capacity is exhausted")
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        _DNS_LOOKUP_SLOTS.release()
        raise SecurityValidationError("DNS resolution exceeded total time limit")
    outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            value: object = list(
                socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            )
            outcome.put_nowait((True, value))
        except Exception as exc:  # noqa: BLE001 - re-raised on the request thread
            outcome.put_nowait((False, exc))
        finally:
            _DNS_LOOKUP_SLOTS.release()

    thread = threading.Thread(
        target=resolve,
        name="blueprint-bounded-dns",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        _DNS_LOOKUP_SLOTS.release()
        raise
    try:
        succeeded, value = outcome.get(timeout=remaining)
    except queue.Empty as exc:
        raise SecurityValidationError("DNS resolution exceeded total time limit") from exc
    if not succeeded:
        if isinstance(value, Exception):
            raise value
        raise SecurityValidationError("DNS resolution failed")
    if not isinstance(value, list):
        raise SecurityValidationError("DNS resolver returned an invalid result")
    return cast(list[tuple[Any, ...]], value)


def resolve_public_ips(
    host: str,
    port: int = 443,
    *,
    timeout_seconds: float | None = None,
) -> tuple[str, ...]:
    """Resolve every address and reject mixed/public-private DNS answers."""

    try:
        answers = _bounded_getaddrinfo(
            host,
            port,
            timeout_seconds=timeout_seconds,
        )
    except OSError as exc:
        raise SecurityValidationError("remote URL host DNS resolution failed") from exc
    addresses = tuple(sorted({str(answer[4][0]) for answer in answers if answer[4]}))
    if not addresses or any(not _ip_is_public(address) for address in addresses):
        raise SecurityValidationError("remote URL host resolves to a non-public address")
    return addresses


def resolve_loopback_ips(
    host: str,
    port: int,
    *,
    timeout_seconds: float | None = None,
) -> tuple[str, ...]:
    """Resolve every address and require an all-loopback answer set."""

    try:
        answers = _bounded_getaddrinfo(
            host,
            port,
            timeout_seconds=timeout_seconds,
        )
    except OSError as exc:
        raise SecurityValidationError(
            "loopback service host DNS resolution failed"
        ) from exc
    addresses = tuple(sorted({str(answer[4][0]) for answer in answers if answer[4]}))
    if not addresses or any(
        not ipaddress.ip_address(address).is_loopback for address in addresses
    ):
        raise SecurityValidationError(
            "plain HTTP service host does not resolve exclusively to loopback"
        )
    return addresses


@dataclass(frozen=True)
class ValidatedRemoteUrl:
    url: str
    origin: str
    host: str
    port: int
    resolved_ips: tuple[str, ...]


def exact_https_origin(url: str) -> str:
    """Return the canonical origin after validating HTTPS URL structure."""

    return validate_remote_https_url(
        url,
        require_allowed_origin=False,
        resolve_dns=False,
    ).origin


def validate_remote_https_url(
    url: str,
    *,
    allowed_origins: Sequence[str] = (),
    require_allowed_origin: bool = True,
    resolve_dns: bool = True,
    dns_timeout_seconds: float | None = None,
) -> ValidatedRemoteUrl:
    """Validate a buyer/provider URL before any network request is made."""

    normalized_url = str(url or "").strip()
    if len(normalized_url) > 8192:
        raise SecurityValidationError("remote URL exceeds length limit")
    parsed = urllib.parse.urlsplit(normalized_url)
    if parsed.scheme.lower() != "https":
        raise SecurityValidationError("remote URL must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password:
        raise SecurityValidationError("remote URL must have a host and no embedded credentials")
    if parsed.fragment:
        raise SecurityValidationError("remote URL may not contain a fragment")
    host = parsed.hostname.lower().rstrip(".")
    if (
        host in _METADATA_HOSTS
        or host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
    ):
        raise SecurityValidationError("remote URL host is not public")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not _ip_is_public(str(literal)):
        raise SecurityValidationError("remote URL host is not public")
    origin = _url_origin(parsed)
    canonical_allowed = tuple(_canonical_origin(item) for item in allowed_origins)
    if require_allowed_origin and not canonical_allowed:
        raise SecurityValidationError("remote URL origin allowlist is not configured")
    if canonical_allowed and origin not in canonical_allowed:
        raise SecurityValidationError("remote URL origin is not approved")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise SecurityValidationError("remote URL port is invalid") from exc
    resolved = (
        resolve_public_ips(host, port, timeout_seconds=dns_timeout_seconds)
        if resolve_dns
        else ()
    )
    return ValidatedRemoteUrl(
        url=urllib.parse.urlunsplit(parsed),
        origin=origin,
        host=host,
        port=port,
        resolved_ips=resolved,
    )


def _canonical_ip(value: object) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Normalize a numeric peer address, including IPv4-mapped IPv6 peers."""

    address = ipaddress.ip_address(str(value).split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _numeric_socket(
    connect_ip: str,
    port: int,
    timeout: float,
) -> socket.socket:
    """Connect to a numeric address without invoking DNS or proxy discovery."""

    address = _canonical_ip(connect_ip)
    family = socket.AF_INET6 if isinstance(address, ipaddress.IPv6Address) else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        endpoint: tuple[object, ...]
        if family == socket.AF_INET6:
            endpoint = (str(address), port, 0, 0)
        else:
            endpoint = (str(address), port)
        sock.connect(endpoint)
        peer = _canonical_ip(sock.getpeername()[0])
        if peer != address:
            raise SecurityValidationError(
                "numeric connection peer did not match the requested IP"
            )
    except Exception:
        sock.close()
        raise
    return sock


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose transport target is an already validated IP."""

    def __init__(
        self,
        host: str,
        *,
        port: int,
        connect_ip: str,
        timeout: float,
    ) -> None:
        super().__init__(host=host, port=port, timeout=timeout)
        self._connect_ip = connect_ip
        self._connect_timeout = timeout

    def connect(self) -> None:
        if getattr(self, "_tunnel_host", None) is not None:
            raise SecurityValidationError("HTTP proxy tunneling is prohibited")
        self.sock = _numeric_socket(self._connect_ip, self.port, self._connect_timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to an IP while authenticating the URL hostname."""

    def __init__(
        self,
        host: str,
        *,
        port: int,
        connect_ip: str,
        timeout: float,
        context: ssl.SSLContext | None = None,
    ) -> None:
        tls_context = context or ssl.create_default_context()
        try:
            tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        except (AttributeError, ValueError) as exc:
            raise SecurityValidationError("TLS 1.2 minimum could not be enforced") from exc
        super().__init__(
            host=host,
            port=port,
            timeout=timeout,
            context=tls_context,
        )
        self._connect_ip = connect_ip
        self._connect_timeout = timeout
        self._tls_context = tls_context

    def connect(self) -> None:
        if getattr(self, "_tunnel_host", None) is not None:
            raise SecurityValidationError("HTTPS proxy tunneling is prohibited")
        raw_socket = _numeric_socket(self._connect_ip, self.port, self._connect_timeout)
        try:
            # ``self.host`` remains the validated URL hostname, so certificate
            # verification and SNI do not degrade to the numeric transport IP.
            self.sock = self._tls_context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except Exception:
            raw_socket.close()
            raise


def _connection_for_ip(
    *,
    validated: ValidatedRemoteUrl,
    scheme: str,
    connect_ip: str,
    timeout: float,
) -> http.client.HTTPConnection:
    if scheme == "https":
        return _PinnedHTTPSConnection(
            validated.host,
            port=validated.port,
            connect_ip=connect_ip,
            timeout=timeout,
        )
    if scheme == "http":
        return _PinnedHTTPConnection(
            validated.host,
            port=validated.port,
            connect_ip=connect_ip,
            timeout=timeout,
        )
    raise SecurityValidationError("remote URL scheme is unsupported")


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SecurityValidationError("remote request exceeded total time limit")
    return remaining


def _open_pinned_connection(
    *,
    validated: ValidatedRemoteUrl,
    scheme: str,
    deadline: float,
) -> http.client.HTTPConnection:
    """Connect only to the addresses captured by the validation resolution."""

    last_error: OSError | None = None
    for connect_ip in validated.resolved_ips:
        address = _canonical_ip(connect_ip)
        if scheme == "https" and not _ip_is_public(str(address)):
            raise SecurityValidationError("validated HTTPS endpoint is not public")
        if scheme == "http" and not address.is_loopback:
            raise SecurityValidationError("validated HTTP service endpoint is not loopback")
        connection = _connection_for_ip(
            validated=validated,
            scheme=scheme,
            connect_ip=str(address),
            timeout=_remaining_seconds(deadline),
        )
        try:
            connection.connect()
        except OSError as exc:
            connection.close()
            last_error = exc
            continue
        sock = connection.sock
        if sock is None:
            connection.close()
            raise SecurityValidationError("remote connection did not expose a peer socket")
        try:
            peer = _canonical_ip(sock.getpeername()[0])
        except (OSError, ValueError) as exc:
            connection.close()
            raise SecurityValidationError("remote connection peer could not be verified") from exc
        if peer != address:
            connection.close()
            raise SecurityValidationError("remote connection peer did not match the pinned IP")
        if scheme == "https" and not _ip_is_public(str(peer)):
            connection.close()
            raise SecurityValidationError("remote connection peer is not public")
        if scheme == "http" and not peer.is_loopback:
            connection.close()
            raise SecurityValidationError("service connection peer is not loopback")
        return connection
    raise SecurityValidationError("all validated remote IP connections failed") from last_error


def _request_target(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    return target


def _host_header(validated: ValidatedRemoteUrl, *, scheme: str) -> str:
    host = f"[{validated.host}]" if ":" in validated.host else validated.host
    default_port = 443 if scheme == "https" else 80
    return host if validated.port == default_port else f"{host}:{validated.port}"


def _request_headers(
    headers: Mapping[str, str] | None,
    *,
    validated: ValidatedRemoteUrl,
    scheme: str,
) -> dict[str, str]:
    prohibited = {
        "connection",
        "content-length",
        "host",
        "proxy-authorization",
        "proxy-connection",
        "transfer-encoding",
    }
    result: dict[str, str] = {}
    for raw_name, raw_value in (headers or {}).items():
        name = str(raw_name).strip()
        value = str(raw_value)
        if not name or name.lower() in prohibited:
            raise SecurityValidationError(f"remote request header is prohibited: {name or 'empty'}")
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise SecurityValidationError("remote request headers may not contain line breaks")
        result[name] = value
    result["Host"] = _host_header(validated, scheme=scheme)
    return result


def _refresh_connection_timeout(
    connection: http.client.HTTPConnection,
    *,
    deadline: float,
) -> None:
    sock = connection.sock
    if sock is None:
        raise SecurityValidationError("remote connection closed before completion")
    sock.settimeout(_remaining_seconds(deadline))


@dataclass(frozen=True)
class BoundedHttpResponse:
    body: bytes
    status: int
    content_type: str
    final_url: str


def _header_value(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value)
    return ""


def _validate_loopback_http_url(
    url: str,
    *,
    dns_timeout_seconds: float | None = None,
) -> ValidatedRemoteUrl:
    normalized_url = str(url or "").strip()
    if len(normalized_url) > 8192:
        raise SecurityValidationError("service URL exceeds length limit")
    parsed = urllib.parse.urlsplit(normalized_url)
    if parsed.scheme.lower() != "http":
        raise SecurityValidationError("loopback service URL must use HTTP")
    if not parsed.hostname or parsed.username or parsed.password:
        raise SecurityValidationError(
            "loopback service URL must have a host and no embedded credentials"
        )
    if parsed.query or parsed.fragment:
        raise SecurityValidationError(
            "loopback service URL may not contain a query or fragment"
        )
    host = parsed.hostname.lower().rstrip(".")
    if host != "localhost":
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise SecurityValidationError(
                "plain HTTP is allowed only for a literal loopback service"
            ) from exc
        if not address.is_loopback:
            raise SecurityValidationError(
                "plain HTTP is allowed only for a loopback service"
            )
    try:
        port = parsed.port or 80
    except ValueError as exc:
        raise SecurityValidationError("loopback service URL port is invalid") from exc
    origin_host = f"[{host}]" if ":" in host else host
    origin = f"http://{origin_host}" + (f":{port}" if port != 80 else "")
    resolved_ips = resolve_loopback_ips(
        host,
        port,
        timeout_seconds=dns_timeout_seconds,
    )
    return ValidatedRemoteUrl(
        url=urllib.parse.urlunsplit(parsed),
        origin=origin,
        host=host,
        port=port,
        resolved_ips=resolved_ips,
    )


def _fetch_bounded_url(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = 30,
    max_bytes: int,
    allowed_origins: Sequence[str],
    allowed_content_types: Iterable[str] = (),
    max_redirects: int = 2,
    output_path: Path | None = None,
    allow_loopback_http: bool = False,
) -> BoundedHttpResponse:
    """Fetch through a DNS-pinned direct connection with bounded I/O."""

    if max_bytes <= 0:
        raise SecurityValidationError("response byte limit must be positive")
    if max_redirects < 0:
        raise SecurityValidationError("redirect limit may not be negative")
    allowed_types = tuple(item.lower() for item in allowed_content_types)
    current_url = url
    current_method = method.upper()
    current_data = data
    timeout_limit = max(1, min(int(timeout_seconds), 600))
    deadline = time.monotonic() + timeout_limit
    for redirect_count in range(max_redirects + 1):
        parsed = urllib.parse.urlsplit(current_url)
        if parsed.scheme.lower() == "http" and allow_loopback_http:
            validated = _validate_loopback_http_url(
                current_url,
                dns_timeout_seconds=_remaining_seconds(deadline),
            )
        else:
            validated = validate_remote_https_url(
                current_url,
                allowed_origins=allowed_origins,
                require_allowed_origin=True,
                resolve_dns=True,
                dns_timeout_seconds=_remaining_seconds(deadline),
            )
        scheme = urllib.parse.urlsplit(validated.url).scheme.lower()
        connection = _open_pinned_connection(
            validated=validated,
            scheme=scheme,
            deadline=deadline,
        )
        response: http.client.HTTPResponse | None = None
        try:
            request_headers = _request_headers(
                headers,
                validated=validated,
                scheme=scheme,
            )
            _refresh_connection_timeout(connection, deadline=deadline)
            connection.request(
                current_method,
                _request_target(validated.url),
                body=current_data,
                headers=request_headers,
                encode_chunked=False,
            )
            _refresh_connection_timeout(connection, deadline=deadline)
            response = connection.getresponse()
            status = int(getattr(response, "status", 200))
            response_headers = dict(response.headers.items())
            if status in _REDIRECT_STATUSES:
                if redirect_count >= max_redirects:
                    raise SecurityValidationError("remote URL exceeded redirect limit")
                location = str(response.headers.get("Location") or "").strip()
                if not location:
                    raise SecurityValidationError("remote URL redirect omitted Location")
                current_url = urllib.parse.urljoin(validated.url, location)
                if status == 303:
                    current_method = "GET"
                    current_data = None
                continue
            if status >= 400:
                raise urllib.error.HTTPError(
                    validated.url,
                    status,
                    str(getattr(response, "reason", "HTTP request failed")),
                    response.headers,
                    None,
                )
            content_type = _header_value(response_headers, "Content-Type").split(";", 1)[0].strip().lower()
            content_length_text = _header_value(response_headers, "Content-Length").strip()
            if content_length_text:
                try:
                    content_length = int(content_length_text)
                except ValueError as exc:
                    raise SecurityValidationError("remote response Content-Length is invalid") from exc
                if content_length < 0 or content_length > max_bytes:
                    raise SecurityValidationError("remote response exceeds byte limit")
            if status != 204 and allowed_types and not any(
                content_type == item or (item.endswith("/*") and content_type.startswith(item[:-1]))
                for item in allowed_types
            ):
                raise SecurityValidationError("remote response Content-Type is not allowed")
            chunks: list[bytes] = []
            total = 0
            output_handle = output_path.open("wb") if output_path is not None else None
            try:
                while True:
                    _refresh_connection_timeout(connection, deadline=deadline)
                    chunk = response.read(min(64 * 1024, max_bytes - total + 1))
                    if time.monotonic() > deadline:
                        raise SecurityValidationError("remote response exceeded total time limit")
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise SecurityValidationError("remote response exceeds byte limit")
                    if output_handle is not None:
                        output_handle.write(chunk)
                    else:
                        chunks.append(chunk)
            except Exception:
                if output_handle is not None:
                    output_handle.close()
                    if output_path is not None:
                        output_path.unlink(missing_ok=True)
                raise
            finally:
                if output_handle is not None and not output_handle.closed:
                    output_handle.close()
        finally:
            if response is not None:
                response.close()
            connection.close()
        if status in _REDIRECT_STATUSES:
            continue
        return BoundedHttpResponse(
            body=b"" if output_path is not None else b"".join(chunks),
            status=status,
            content_type=content_type,
            final_url=validated.url,
        )
    raise SecurityValidationError("remote URL redirect processing failed")


def fetch_bounded_https(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = 30,
    max_bytes: int,
    allowed_origins: Sequence[str],
    allowed_content_types: Iterable[str] = (),
    max_redirects: int = 2,
    output_path: Path | None = None,
) -> BoundedHttpResponse:
    """Fetch an allowlisted public HTTPS URL without automatic redirects."""

    return _fetch_bounded_url(
        url,
        method=method,
        data=data,
        headers=headers,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
        allowed_origins=allowed_origins,
        allowed_content_types=allowed_content_types,
        max_redirects=max_redirects,
        output_path=output_path,
        allow_loopback_http=False,
    )


def fetch_bounded_service_url(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = 30,
    max_bytes: int,
    allowed_origins: Sequence[str],
    allowed_content_types: Iterable[str] = (),
    max_redirects: int = 0,
) -> BoundedHttpResponse:
    """Fetch remote HTTPS or an exact loopback HTTP service endpoint."""

    return _fetch_bounded_url(
        url,
        method=method,
        data=data,
        headers=headers,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
        allowed_origins=allowed_origins,
        allowed_content_types=allowed_content_types,
        max_redirects=max_redirects,
        output_path=None,
        allow_loopback_http=True,
    )


def json_shape_within_limits(
    value: object,
    *,
    max_depth: int = 32,
    max_items: int = 100_000,
) -> bool:
    remaining = max_items

    def visit(item: object, depth: int) -> bool:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > max_depth:
            return False
        if isinstance(item, Mapping):
            return all(visit(key, depth + 1) and visit(child, depth + 1) for key, child in item.items())
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return all(visit(child, depth + 1) for child in item)
        return True

    return visit(value, 0)
