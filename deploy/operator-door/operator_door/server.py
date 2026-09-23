"""HTTP surface of the operator door.

Caddy terminates TLS and forwards ``/api/live-pipeline/operator/*`` here on
loopback. Every route except ``healthz`` needs a bearer token with the right
scope; every request, allowed or not, is appended to the audit log. Refusals
carry the rule that fired and nothing else.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import API_PREFIX, VERSION
from .auth import TokenIdentity, TokenStore
from .config import DoorConfig
from .fsview import FileView, FsRefused
from .hostinfo import HostInfo, HostRefused, validate_unit
from .requests import RequestRefused, enqueue, list_requests, request_state, required_scope, validate_request
from .status import build_status

_REQUEST_ROUTE = re.compile(r"^/requests/([^/]+)$")
_ARCHIVE_SLOTS = threading.BoundedSemaphore(2)


class DoorApp:
    def __init__(self, config: DoorConfig, *, host: HostInfo) -> None:
        self.config = config
        self.host = host
        self.tokens = TokenStore(config.token_file)
        self.files = FileView(config)
        self._audit_lock = threading.Lock()

    def audit(self, entry: dict[str, Any], *, denied: bool = False) -> None:
        # Unauthenticated denials get their own file, so a flood from the internet
        # cannot rotate the authenticated history away.
        path = Path(self.config.audit_path)
        if denied:
            path = path.with_name("denied.jsonl")
        line = json.dumps(entry, sort_keys=True) + "\n"
        sys.stderr.write("operator-door " + line)  # one journald line per request
        with self._audit_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size > self.config.audit_rotate_bytes:
                    path.replace(path.with_name(path.name + ".1"))  # keep one previous file
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
            except OSError:
                pass  # auditing must never take the door down; journald has the line


def _caller(identity: TokenIdentity) -> dict[str, Any]:
    return {"name": identity.name, "scopes": sorted(identity.scopes)}


def make_handler(app: DoorApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "blueprint-operator-door/" + VERSION
        sys_version = ""
        # A silent or slow client must not hold one of the door's few threads.
        timeout = app.config.request_timeout_seconds

        # -- plumbing ---------------------------------------------------

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return None  # the audit log is the access log

        def _headers(self, status: int, content_type: str, length: int | None,
                     extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if length is not None:
                self.send_header("Content-Length", str(length))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self._headers(status, "application/json", len(body))
            self.wfile.write(body)
            self._status = status

        def _audit(self, identity: TokenIdentity | None, status: int, outcome: str) -> None:
            route, query, _ = self._route()
            app.audit({
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "remote": self.headers.get("X-Forwarded-For") or self.client_address[0],
                "token": identity.name if identity else None,
                "method": self.command,
                "route": route or urllib.parse.urlsplit(self.path).path[:200],
                # The same parsed value the handler used (duplicate keys are refused).
                "path": query.get("path") or query.get("unit"),
                "status": status,
                "outcome": outcome,
            }, denied=identity is None)

        def _route(self) -> tuple[str, dict[str, str], bool]:
            parsed = urllib.parse.urlsplit(self.path)
            if not parsed.path.startswith(API_PREFIX):
                return "", {}, False
            pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            query = dict(pairs)
            return parsed.path[len(API_PREFIX):] or "/", query, len(query) != len(pairs)

        def _identity(self) -> TokenIdentity | None:
            try:
                return app.tokens.verify(self.headers.get("Authorization"))
            except Exception:  # noqa: BLE001 - a broken token file denies, never crashes
                return None

        def _dispatch(self, method: str) -> None:
            route, query, duplicated = self._route()
            self._streaming = False
            if method == "GET" and route == "/healthz":
                self._send_json(200, {"ok": True, "version": VERSION})
                return
            identity = self._identity()
            if identity is None:
                self._audit(None, 401, "denied")  # before replying, so a denial is never unrecorded
                self._send_json(401, {"error": "unauthorized"})
                return
            self._status = 500
            outcome = "allowed"
            try:
                if duplicated:
                    raise _BadRequest(400, "query_duplicate_key")
                if method == "POST" and route == "/requests":
                    self._post_request(identity)
                elif method == "GET":
                    if "read" not in identity.scopes:
                        raise _Denied("scope_missing:read")
                    self._get(route, query, identity)
                else:
                    self._send_json(404, {"error": "not_found"})
            except (BrokenPipeError, ConnectionResetError):
                outcome = "client_gone"
            except Exception as error:  # noqa: BLE001 - never leak internals to the caller
                outcome = self._fail(error)
            self._audit(identity, self._status, outcome)

        def _fail(self, error: Exception) -> str:
            if self._streaming:
                # Headers and part of a body are already out; a JSON error written now
                # would corrupt the stream. Close instead; the client sees a truncation.
                self.close_connection = True
                return "stream_failed"
            try:
                if isinstance(error, _Denied):
                    self._send_json(403, {"error": error.code})
                    return "denied"
                if isinstance(error, (FsRefused, HostRefused, RequestRefused)):
                    self._send_json(404 if error.code == "not_found" else 403, {"error": error.code})
                    return "refused"
                if isinstance(error, _BadRequest):
                    self._send_json(error.status, {"error": error.code})
                    return "refused"
                self._send_json(500, {"error": "internal_error"})
            except OSError:
                pass
            return "error"

        # -- routes -----------------------------------------------------

        def _get(self, route: str, query: dict[str, str], identity: TokenIdentity) -> None:
            if route == "/whoami":
                self._send_json(200, _caller(identity))
            elif route == "/status":
                self._send_json(200, build_status(app.config, app.host, caller=_caller(identity)))
            elif route == "/fs/list":
                self._send_json(200, app.files.list_dir(
                    _required(query, "path"), sort=query.get("sort", "name"), match=query.get("match")))
            elif route == "/fs/read":
                self._read_file(query)
            elif route == "/fs/archive":
                self._archive(query)
            elif route == "/journal":
                text = app.host.journal(_required(query, "unit"), lines=_int(query, "lines", 200),
                                        since=query.get("since"))
                body = text.encode("utf-8")
                self._headers(200, "text/plain; charset=utf-8", len(body))
                self.wfile.write(body)
                self._status = 200
            elif route == "/units":
                states = tuple(filter(None, query.get("state", "").split(",")))
                units = app.host.list_units(query.get("pattern", "blueprint-*"), states=states)
                self._send_json(200, {"units": units})
            elif route == "/units/show":
                names = [validate_unit(name) for name in _required(query, "unit").split(",")]
                self._send_json(200, {"units": app.host.unit_properties(names)})
            elif route == "/requests":
                self._send_json(200, {"requests": list_requests(app.config)})
            elif (match := _REQUEST_ROUTE.match(route)) is not None:
                state = request_state(app.config, match.group(1))
                unit = (state.get("result") or {}).get("unit")
                if isinstance(unit, str):
                    try:
                        state["unit_state"] = app.host.unit_properties([unit])
                    except HostRefused:
                        state["unit_state"] = None
                self._send_json(200, state)
            else:
                self._send_json(404, {"error": "not_found"})

        def _read_file(self, query: dict[str, str]) -> None:
            length = query.get("length")
            data, meta = app.files.read_range(
                _required(query, "path"), offset=_int(query, "offset", 0),
                length=None if length is None else _int(query, "length", 0),
            )
            self._headers(200, "application/octet-stream", len(data), {
                "X-Door-Size": str(meta["size"]), "X-Door-Offset": str(meta["offset"]),
                "X-Door-Length": str(meta["length"]), "X-Door-Eof": "true" if meta["eof"] else "false",
            })
            self.wfile.write(data)
            self._status = 200

        def _archive(self, query: dict[str, str]) -> None:
            path = _required(query, "path")
            app.files.prepare_archive(path)  # every refusal happens before any response byte
            if not _ARCHIVE_SLOTS.acquire(blocking=False):
                raise _BadRequest(429, "archive_busy")
            try:
                self._headers(200, "application/gzip", None, {"Connection": "close"})
                self._status = 200
                self._streaming = True
                app.files.stream_archive(path, self.wfile.write)
            finally:
                _ARCHIVE_SLOTS.release()

        def _post_request(self, identity: TokenIdentity) -> None:
            length_header = self.headers.get("Content-Length")
            if length_header is None:
                raise _BadRequest(411, "length_required")
            try:
                length = int(length_header)
            except ValueError as error:
                raise _BadRequest(400, "length_invalid") from error
            if length < 0:
                raise _BadRequest(400, "length_invalid")
            if length > app.config.max_request_body:
                raise _BadRequest(413, "body_too_large")
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise _BadRequest(400, "body_not_json") from error
            try:
                normalized = validate_request(body)
            except RequestRefused as refusal:
                raise _BadRequest(400, refusal.code) from refusal
            scope = required_scope(normalized["kind"])
            if scope not in identity.scopes:
                raise _Denied(f"scope_missing:{scope}")
            request_id = enqueue(app.config, normalized, requested_by=identity.name)
            self._send_json(202, {"id": request_id, "state": "pending"})

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            self._dispatch("POST")

        def do_PUT(self) -> None:  # noqa: N802 - stdlib naming
            self._dispatch("PUT")

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib naming
            self._dispatch("DELETE")

    return Handler


class _Denied(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _BadRequest(Exception):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


def _required(query: dict[str, str], key: str) -> str:
    value = query.get(key)
    if not value:
        raise _BadRequest(400, f"query_missing:{key}")
    return value


def _int(query: dict[str, str], key: str, default: int) -> int:
    raw = query.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise _BadRequest(400, f"query_invalid:{key}") from error


class DoorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(config: DoorConfig, *, host: HostInfo) -> DoorServer:
    app = DoorApp(config, host=host)
    return DoorServer((config.listen_host, config.listen_port), make_handler(app))

