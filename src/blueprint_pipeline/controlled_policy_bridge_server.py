"""One job-bound HTTPS bridge into a qualified, isolated policy container.

This process runs on the dedicated sandbox worker. It never loads a scene or
scores an outcome. The sandbox executor grants the action transport only after
measured qualification, and removes the image and containers after finish.
"""
from __future__ import annotations

import hmac
import json
import os
import ssl
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

from .controlled_policy_configuration import canonical_request_digest
from .controlled_policy_remote_sandbox import validate_remote_sandbox_bridge

_MAX_REQUEST = 8 * 1024 * 1024 + 4096


class QualifiedPolicyBridge:
    def __init__(self, *, contract: Mapping[str, Any], job_request: Mapping[str, Any],
                 endpoint_url: str, bearer_token: str, manifest_path: Path,
                 bind_host: str, bind_port: int, tls_certificate: Path, tls_private_key: Path,
                 maximum_seconds: int = 1800, max_policy_calls: int = 128,
                 tls_certificate_pem: str | None = None) -> None:
        from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
        normalized = validate_company_policy_container_contract_v2(contract)
        self.binding = {
            "job_id": job_request["job_id"],
            "canonical_request_digest": canonical_request_digest(job_request),
            "contract_digest": normalized["contract_digest"],
            "image_ref": normalized["container"]["image"],
        }
        artifact = ((job_request.get("policy_package") or {}).get("docker_container") or {}).get("model_artifact")
        self.model_artifact_sha256 = artifact.get("sha256") if isinstance(artifact, Mapping) else None
        self.endpoint_url = endpoint_url
        self.bearer_token = bearer_token
        self.tls_certificate_pem = tls_certificate_pem
        self.manifest_path = manifest_path
        if type(maximum_seconds) is not int or not 1 <= maximum_seconds <= 3600:
            raise ValueError("controlled_policy_bridge_duration_invalid")
        if type(max_policy_calls) is not int or not 1 <= max_policy_calls <= 128:
            raise ValueError("controlled_policy_bridge_query_cap_invalid")
        self.maximum_seconds = maximum_seconds
        self.max_policy_calls = max_policy_calls
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.terminal_read = threading.Event()
        self.transport: Callable[[bytes, float], bytes] | None = None
        self.qualification: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None
        self.calls = 0
        self.server = HTTPServer((bind_host, bind_port), self._handler())
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(tls_certificate), str(tls_private_key))
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(10)

            def log_message(self, *_args: Any) -> None:
                # Observations, actions, tokens and endpoints are private.
                return

            def _reply(self, code: int, value: Mapping[str, Any]) -> None:
                data = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                if not hmac.compare_digest(self.headers.get("Authorization", ""),
                                           "Bearer " + owner.bearer_token):
                    self._reply(401, {"error": "unauthorized"})
                    return
                if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                    self._reply(415, {"error": "json_required"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", ""))
                    if not 0 < length <= _MAX_REQUEST:
                        raise ValueError("body_size")
                    body = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                    if not isinstance(body, dict):
                        raise ValueError("body_type")
                except (ValueError, json.JSONDecodeError):
                    self._reply(400, {"error": "request_invalid"})
                    return
                route = self.path.removeprefix("/v1/controlled-policy")
                if route not in {"/ready", "/actions", "/finish", "/terminal"}:
                    self._reply(404, {"error": "route_invalid"})
                    return
                if any(body.get(key) != value for key, value in owner.binding.items()):
                    self._reply(409, {"error": "binding_mismatch"})
                    return
                with owner.lock:
                    if route == "/ready":
                        if set(body) != set(owner.binding) or owner.qualification is None:
                            self._reply(409, {"error": "not_qualified"})
                            return
                        self._reply(200, {**owner.binding, "status": "qualified_before_first_observation",
                            "qualification_receipt_digest": owner.qualification["qualification_digest"]})
                        return
                    if route == "/terminal":
                        if set(body) != set(owner.binding):
                            self._reply(400, {"error": "terminal_request_invalid"})
                            return
                        if owner.result is None:
                            self._reply(200, {"status": "terminal_pending"})
                            return
                        terminal = owner.result.get("terminal_receipt") or {}
                        complete = (owner.result.get("status") == "controlled_session_completed"
                                    and terminal.get("cleanup_complete") is True)
                        self._reply(200, {**owner.binding,
                            "status": "controlled_session_completed" if complete else "blocked",
                            "cleanup_complete": terminal.get("cleanup_complete") is True,
                            "terminal_receipt_digest": terminal.get("receipt_digest"),
                            "policy_calls": owner.calls})
                        owner.terminal_read.set()
                        return
                    if owner.transport is None or owner.qualification is None or owner.finished.is_set():
                        self._reply(409, {"error": "session_unavailable"})
                        return
                    if route == "/finish":
                        if set(body) != set(owner.binding) | {"policy_calls"} or body["policy_calls"] != owner.calls:
                            self._reply(409, {"error": "call_count_mismatch"})
                            return
                        owner.finished.set()
                        self._reply(200, {"status": "cleanup_pending"})
                        return
                    if set(body) != set(owner.binding) | {"observation"}:
                        self._reply(400, {"error": "observation_invalid"})
                        return
                    if owner.calls >= owner.max_policy_calls:
                        self._reply(429, {"error": "query_limit"})
                        return
                    try:
                        encoded = json.dumps(body["observation"], sort_keys=True,
                                             separators=(",", ":"), allow_nan=False).encode()
                        response = owner.transport(encoded, 30.0)
                        actions = json.loads(response)
                    except (ValueError, TypeError, RuntimeError, OSError, KeyError):
                        self._reply(502, {"error": "policy_call_failed"})
                        return
                    owner.calls += 1
                    self._reply(200, actions)

        return Handler

    def qualified_session(self, transport: Callable[[bytes, float], bytes],
                          qualification: Mapping[str, Any]) -> dict[str, Any]:
        if (qualification.get("status") != "qualified_before_first_observation"
                or qualification.get("first_observation_permitted") is not True):
            raise ValueError("controlled_policy_bridge_qualification_required")
        manifest = validate_remote_sandbox_bridge({
            "schema_version": "blueprint.qualified_policy_bridge.v1", **self.binding,
            "endpoint_url": self.endpoint_url, "bearer_token": self.bearer_token,
            "qualification_receipt_digest": qualification["qualification_digest"],
            "expires_at_iso": (datetime.now(timezone.utc) + timedelta(seconds=self.maximum_seconds)).isoformat(),
            "model_artifact_sha256": self.model_artifact_sha256,
            "tls_certificate_pem": self.tls_certificate_pem,
        })
        if self.manifest_path.exists() or self.manifest_path.is_symlink():
            raise ValueError("controlled_policy_bridge_manifest_exists")
        with self.lock:
            self.transport = transport
            self.qualification = dict(qualification)
        fd = os.open(self.manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump(manifest, output, sort_keys=True, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        if not self.finished.wait(self.maximum_seconds):
            raise ValueError("controlled_policy_bridge_session_deadline")
        return {"schema_version": "blueprint.qualified_policy_bridge_session.v1", "policy_calls": self.calls}

    def run(self, execute_sandbox: Callable[[Callable[..., Mapping[str, Any]]], Mapping[str, Any]]) -> dict[str, Any]:
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        try:
            try:
                result = dict(execute_sandbox(self.qualified_session))
            except Exception as exc:
                result = {"status": "blocked", "error_class": type(exc).__name__,
                          "terminal_receipt": {"cleanup_complete": False}}
            with self.lock:
                self.result = result
            self.terminal_read.wait(120 if self.qualification is not None else 5)
            return result
        finally:
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=10)
