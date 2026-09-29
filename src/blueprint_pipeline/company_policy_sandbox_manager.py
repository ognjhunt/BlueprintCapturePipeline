"""Trusted worker manager for job-bound policy sandboxes and network leases.

Only the production control plane may call this API.  It prepares one isolated
container before native GPU spend, then temporarily admits the allocated GPU's
measured outbound IPv4.  A terminal child closes that network lease.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from .company_policy_session_preparation import prepare_company_policy_session
from .controlled_policy_configuration import canonical_request_digest
from .controlled_policy_remote_sandbox import validate_remote_sandbox_bridge


_JOB = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,191}$")
_MAX_BODY = 192 * 1024


def _metadata_token() -> str:
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        value = json.load(response)
    token = value.get("access_token")
    if value.get("token_type") != "Bearer" or not isinstance(token, str) or len(token) < 32:
        raise ValueError("policy_sandbox_manager_vm_identity_unavailable")
    return token


class FirewallLease:
    """Change only the preapproved bridge rule, with exact-state checks."""

    def __init__(self, *, project: str, rule: str, base_source: str, target_tag: str) -> None:
        self.url = f"https://compute.googleapis.com/compute/v1/projects/{project}/global/firewalls/{rule}"
        self.operations = f"https://compute.googleapis.com/compute/v1/projects/{project}/global/operations"
        self.base_source = str(ipaddress.ip_network(base_source, strict=True))
        self.target_tag = target_tag
        self.lock = threading.Lock()
        if not self.base_source.endswith("/32") or not target_tag:
            raise ValueError("policy_sandbox_manager_firewall_scope_invalid")

    @staticmethod
    def _request(method: str, url: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        request = urllib.request.Request(url, method=method, data=data, headers={
            "Authorization": "Bearer " + _metadata_token(),
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError("policy_sandbox_manager_compute_response_invalid")
        return value

    def _current(self) -> dict[str, Any]:
        row = self._request("GET", self.url)
        if (row.get("direction") != "INGRESS"
                or row.get("targetTags") != [self.target_tag]
                or row.get("allowed") != [{"IPProtocol": "tcp", "ports": ["8443"]}]
                or self.base_source not in row.get("sourceRanges", [])):
            raise ValueError("policy_sandbox_manager_firewall_rule_drift")
        return row

    def _set(self, sources: list[str]) -> None:
        with self.lock:
            before = self._current()
            if set(before["sourceRanges"]) == set(sources):
                return
            if not set(before["sourceRanges"]).issubset(set(sources)) and sources != [self.base_source]:
                raise ValueError("policy_sandbox_manager_firewall_lease_conflict")
            operation = self._request("PATCH", self.url, {"sourceRanges": sources,
                "fingerprint": before.get("fingerprint")})
            name = operation.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("policy_sandbox_manager_firewall_operation_missing")
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                result = self._request("GET", f"{self.operations}/{name}")
                if result.get("status") == "DONE":
                    if result.get("error"):
                        raise ValueError("policy_sandbox_manager_firewall_operation_failed")
                    break
                time.sleep(2)
            else:
                raise TimeoutError("policy_sandbox_manager_firewall_operation_timeout")
            if set(self._current()["sourceRanges"]) != set(sources):
                raise ValueError("policy_sandbox_manager_firewall_verify_failed")

    def allow(self, outbound_ipv4: str) -> None:
        address = ipaddress.ip_address(outbound_ipv4)
        if not isinstance(address, ipaddress.IPv4Address) or not address.is_global:
            raise ValueError("policy_sandbox_manager_outbound_address_invalid")
        self._set([self.base_source, f"{address}/32"])

    def close(self) -> None:
        self._set([self.base_source])

    def closed(self) -> bool:
        return set(self._current()["sourceRanges"]) == {self.base_source}


class SandboxManager:
    def __init__(self, settings: Mapping[str, Any], *, firewall: FirewallLease | None = None) -> None:
        self.settings = dict(settings)
        self.root = Path(str(settings["sessions_root"])).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.firewall = firewall or FirewallLease(
            project=str(settings["firewall_project"]), rule=str(settings["firewall_rule"]),
            base_source=str(settings["firewall_base_source"]),
            target_tag=str(settings["firewall_target_tag"]),
        )
        self.lock = threading.Lock()
        self.active_job: str | None = None
        self.child: subprocess.Popen[bytes] | None = None
        # The service unit kills its child cgroup on restart. Reconcile any
        # address left by an interrupted process before admitting another job.
        self.firewall.close()

    def _argv(self, session: Path, key_id: str) -> list[str]:
        settings = self.settings
        release = Path(str(settings["release_root"])).resolve()
        script = release / "scripts/serve_company_policy_sandbox_bridge.py"
        if not script.is_file() or not script.resolve().is_relative_to(release):
            raise ValueError("policy_sandbox_manager_release_missing")
        argv = [sys.executable, str(script)]
        for name, path in (
            ("plan", session / "plan.json"),
            ("contract", session / "contract.json"),
            ("job-request", session / "job-request.json"),
            ("authority", session / "authority.json"),
            ("worker-boot-receipt", session / "worker-boot-receipt.json"),
            ("attestation-key-file", session / "attestation-key"),
            ("bearer-token-file", session / "bearer-token"),
            ("tls-certificate", Path(str(settings["bridge_tls_certificate"]))),
            ("tls-private-key", Path(str(settings["bridge_tls_private_key"]))),
            ("manifest-out", session / "bridge-manifest.json"),
            ("output", session / "sandbox-output.json"),
        ):
            argv.extend(["--" + name, str(path)])
        argv.extend([
            "--attestation-key-id", key_id,
            "--endpoint-url", str(settings["bridge_endpoint_url"]),
            "--bind-port", str(settings["bridge_bind_port"]),
            "--maximum-seconds", str(settings.get("session_lifetime_seconds", 2400)),
            "--ack", "authorized-controlled-policy-session",
        ])
        contract = json.loads((session / "contract.json").read_text())
        request = json.loads((session / "job-request.json").read_text())
        artifact = ((request.get("policy_package") or {}).get("docker_container") or {}).get("model_artifact")
        if artifact is not None:
            argv.extend(["--private-model-bucket", str(settings["private_model_bucket"]),
                "--approved-model-runner-image", str(settings["approved_model_runner_image"])])
        if contract["container"]["visibility"] == "private":
            image = contract["container"]["image"]
            if image.startswith("us-central1-docker.pkg.dev/blueprint-8c1ca/pipeline-jobs/"):
                argv.append("--blueprint-owned-vm-identity")
            else:
                argv.extend(["--broker-base-url", str(settings["broker_base_url"]),
                    "--broker-token-file", str(settings["broker_token_file"])])
        return argv

    def _watch(self, job_id: str, child: subprocess.Popen[bytes]) -> None:
        child.wait()
        try:
            self.firewall.close()
        finally:
            with self.lock:
                if self.active_job == job_id:
                    self.active_job = None
                    self.child = None

    def prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if set(payload) != {"job_request", "contract", "admission_receipt"}:
            raise ValueError("policy_sandbox_manager_request_schema_invalid")
        request = payload["job_request"]
        if not isinstance(request, Mapping):
            raise ValueError("policy_sandbox_manager_job_request_invalid")
        job_id = request.get("job_id")
        if not isinstance(job_id, str) or not _JOB.fullmatch(job_id):
            raise ValueError("policy_sandbox_manager_job_id_invalid")
        with self.lock:
            if self.active_job == job_id and self.child is not None and self.child.poll() is None:
                session = self.root / job_id
                stored_request = json.loads((session / "job-request.json").read_text())
                stored_contract = json.loads((session / "contract.json").read_text())
                stored_admission = json.loads((session / "admission-receipt.json").read_text())
                if (canonical_request_digest(stored_request) != canonical_request_digest(request)
                        or stored_contract != payload["contract"]
                        or stored_admission != payload["admission_receipt"]):
                    raise ValueError("policy_sandbox_manager_retry_binding_conflict")
                prepared = {"plan_digest": json.loads((session / "plan.json").read_text())["plan_digest"]}
                child = self.child
            else:
                if self.active_job is not None or not self.firewall.closed():
                    raise ValueError("policy_sandbox_manager_busy_or_network_open")
                settings = self.settings
                prepared = prepare_company_policy_session(
                    job_request=request, contract=payload["contract"],
                    admission_receipt=payload["admission_receipt"], root=self.root,
                    pipeline_release_sha=str(settings["release_sha"]),
                    worker_identity=str(settings["worker_identity"]),
                    proxy_image=str(settings["proxy_image"]),
                    proxy_contract_digest=str(settings["proxy_contract_digest"]),
                    seccomp_profile_path=str(settings["seccomp_profile_path"]),
                    seccomp_profile_digest=str(settings["seccomp_profile_digest"]),
                    apparmor_profile_source_path=str(settings["apparmor_profile_source_path"]),
                    apparmor_profile_digest=str(settings["apparmor_profile_digest"]),
                    registry_addresses=list(settings["registry_addresses"]),
                    allowed_registry_hosts=list(settings["allowed_registry_hosts"]),
                    approved_by=str(settings["approved_by"]),
                    lifetime_seconds=int(settings.get("session_lifetime_seconds", 2400)),
                )
                session = Path(prepared["session_dir"])
                argv = self._argv(session, prepared["attestation_key_id"])
                environment = {**os.environ, "PYTHONPATH": str(settings["release_root"]) + "/src"}
                with open(session / "bridge.log", "xb", buffering=0) as log:
                    os.chmod(log.fileno(), 0o600)
                    child = subprocess.Popen(argv, cwd=str(settings["release_root"]),
                        env=environment, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                self.active_job = job_id
                self.child = child
                threading.Thread(target=self._watch, args=(job_id, child), daemon=True).start()
        manifest_path = session / "bridge-manifest.json"
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if manifest_path.is_file():
                manifest = validate_remote_sandbox_bridge(json.loads(manifest_path.read_text()))
                if (manifest["job_id"] != job_id
                        or manifest["canonical_request_digest"] != canonical_request_digest(request)):
                    raise ValueError("policy_sandbox_manager_manifest_binding_invalid")
                return {"status": "qualified", "manifest": manifest,
                    "plan_digest": prepared["plan_digest"]}
            if child.poll() is not None:
                raise ValueError("policy_sandbox_manager_child_exited_before_qualification")
            time.sleep(1)
        raise TimeoutError("policy_sandbox_manager_qualification_timeout")

    def allow_network(self, job_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if set(payload) != {"instance_id", "outbound_ipv4"}:
            raise ValueError("policy_sandbox_manager_network_request_invalid")
        instance_id = payload["instance_id"]
        if type(instance_id) is not int or instance_id <= 0:
            raise ValueError("policy_sandbox_manager_instance_id_invalid")
        with self.lock:
            if self.active_job != job_id or self.child is None or self.child.poll() is not None:
                raise ValueError("policy_sandbox_manager_session_not_active")
        self.firewall.allow(str(payload["outbound_ipv4"]))
        return {"status": "network_allowed", "job_id": job_id,
            "instance_id": instance_id, "outbound_ipv4": str(payload["outbound_ipv4"])}

    def close_network(self, job_id: str) -> dict[str, Any]:
        if not (self.root / job_id).is_dir():
            raise ValueError("policy_sandbox_manager_session_unknown")
        self.firewall.close()
        return {"status": "network_closed", "job_id": job_id}


def serve_manager(*, settings: Mapping[str, Any], token: str,
                  certificate: Path, private_key: Path, firewall: FirewallLease | None = None) -> None:
    manager = SandboxManager(settings, firewall=firewall)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self._reply(401, {"error": "unauthorized"})
                return
            try:
                size = int(self.headers.get("Content-Length", ""))
                if not 0 < size <= _MAX_BODY:
                    raise ValueError("policy_sandbox_manager_request_size_invalid")
                body = json.loads(self.rfile.read(size), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                if not isinstance(body, dict):
                    raise ValueError("policy_sandbox_manager_request_object_required")
                if self.path == "/v1/sessions":
                    result = manager.prepare(body)
                else:
                    match = re.fullmatch(r"/v1/sessions/([A-Za-z0-9._:-]{8,192})/(network|close)", self.path)
                    if not match:
                        self._reply(404, {"error": "route_unknown"})
                        return
                    result = (manager.allow_network(match[1], body) if match[2] == "network"
                        else manager.close_network(match[1]))
                self._reply(200, result)
            except (ValueError, KeyError, TypeError, TimeoutError, OSError, urllib.error.URLError) as exc:
                self._reply(409, {"error": type(exc).__name__, "code": str(exc)[:120]})

        def _reply(self, status: int, value: Mapping[str, Any]) -> None:
            data = json.dumps(dict(value), sort_keys=True, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((str(settings.get("bind_host", "0.0.0.0")),
        int(settings.get("bind_port", 8444))), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(certificate), str(private_key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
