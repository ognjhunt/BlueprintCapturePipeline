"""Trusted executor for one isolated company-policy sandbox session.

This executor runs only inside a dedicated Blueprint worker.  It does not
allocate a provider, grant launch authority, publish a profile, or submit an
episode.  It turns an immutable sandbox plan into measured pre-observation
evidence, and always removes customer containers, credentials, and image bytes
when the session closes or fails.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import secrets
import shutil
import socket
import stat
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .company_policy_container_contract_v2 import (
    validate_company_policy_container_contract_v2,
)
from .company_policy_proxy import validate_action_response
from .company_policy_sandbox_v2 import (
    DENIAL_PROBES,
    CompanyPolicySandboxV2Error,
    evaluate_preobservation_sandbox_qualification,
)
from .decision_evidence_contracts import cross_runtime_canonical_digest


EXECUTOR_RECEIPT_SCHEMA_VERSION = "company_policy_sandbox_executor_receipt.v1"
TERMINAL_RECEIPT_SCHEMA_VERSION = "company_policy_sandbox_terminal_receipt.v1"
WORKER_BOOT_RECEIPT_SCHEMA_VERSION = "company_policy_worker_boot_receipt.v1"


class CompanyPolicySandboxExecutorError(RuntimeError):
    pass


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]: ...


class CredentialBroker(Protocol):
    def claim(self, *, lease_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def acknowledge(self, *, lease_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


class SubprocessCommandRunner:
    def run(self, argv: Sequence[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(item) for item in argv],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class HttpCredentialBroker:
    def __init__(
        self,
        *,
        base_url: str,
        token_file: Path,
        client_id: str,
        timeout_seconds: int = 15,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        loopback = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
        if (
            (parsed.scheme != "https" and not loopback)
            or parsed.username
            or parsed.password
            or parsed.fragment
            or not parsed.netloc
        ):
            raise CompanyPolicySandboxExecutorError("company_policy_broker_url_invalid")
        if not client_id or len(client_id) > 80:
            raise CompanyPolicySandboxExecutorError("company_policy_broker_client_id_invalid")
        self.base_url = base_url.rstrip("/")
        self.token_file = token_file.expanduser().resolve()
        self.client_id = client_id
        self.timeout_seconds = timeout_seconds

    def _post(self, *, lease_id: str, action: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        token_mode = stat.S_IMODE(self.token_file.stat().st_mode)
        if token_mode & 0o077:
            raise CompanyPolicySandboxExecutorError("company_policy_broker_token_mode_invalid")
        token = self.token_file.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise CompanyPolicySandboxExecutorError("company_policy_broker_token_invalid")
        encoded = json.dumps(dict(body), sort_keys=True, separators=(",", ":")).encode("utf-8")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        nonce = f"company-policy-broker-{secrets.token_hex(24)}"
        signature = hmac.new(
            token.encode("utf-8"),
            b".".join(
                [timestamp.encode(), self.client_id.encode(), nonce.encode(), encoded]
            ),
            hashlib.sha256,
        ).hexdigest()
        url = (
            f"{self.base_url}/company-policy-registry-credential-leases/"
            f"{urllib.parse.quote(lease_id, safe='')}/{action}"
        )
        request = urllib.request.Request(
            url,
            data=encoded,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Blueprint-Pipeline-Timestamp": timestamp,
                "X-Blueprint-Pipeline-Client-Id": self.client_id,
                "X-Blueprint-Pipeline-Nonce": nonce,
                "X-Blueprint-Pipeline-Signature": f"sha256={signature}",
            },
        )
        try:
            with urllib.request.build_opener(_NoRedirect).open(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read(256 * 1024 + 1)
                if len(raw) > 256 * 1024:
                    raise CompanyPolicySandboxExecutorError(
                        "company_policy_broker_response_too_large"
                    )
                payload = json.loads(raw)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise CompanyPolicySandboxExecutorError(
                "company_policy_broker_request_failed"
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            raise CompanyPolicySandboxExecutorError("company_policy_broker_response_invalid")
        return dict(payload)

    def claim(self, *, lease_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error: CompanyPolicySandboxExecutorError | None = None
        for _attempt in range(3):
            try:
                return self._post(lease_id=lease_id, action="claim", body=body)
            except CompanyPolicySandboxExecutorError as exc:
                last_error = exc
        raise last_error or CompanyPolicySandboxExecutorError(
            "company_policy_broker_claim_failed"
        )

    def acknowledge(self, *, lease_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error: CompanyPolicySandboxExecutorError | None = None
        for _attempt in range(3):
            try:
                return self._post(lease_id=lease_id, action="acknowledge", body=body)
            except CompanyPolicySandboxExecutorError as exc:
                last_error = exc
        raise last_error or CompanyPolicySandboxExecutorError(
            "company_policy_broker_acknowledgement_failed"
        )


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(dict(value), sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if path.read_bytes() != encoded:
            raise CompanyPolicySandboxExecutorError("company_policy_executor_readback_failed")
    finally:
        temporary.unlink(missing_ok=True)


def _signed_receipt(
    value: Mapping[str, Any], *, key: bytes, key_id: str
) -> dict[str, Any]:
    if len(key) < 32 or not key_id:
        raise CompanyPolicySandboxExecutorError("company_policy_attestation_key_invalid")
    receipt = {**dict(value), "attestation_key_id": key_id, "receipt_digest": ""}
    receipt["receipt_digest"] = cross_runtime_canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt["attestation_hmac_sha256"] = hmac.new(
        key, str(receipt["receipt_digest"]).encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return receipt


def verify_executor_receipt(
    receipt: Mapping[str, Any], *, key: bytes, expected_plan_digest: str
) -> None:
    supplied = str(receipt.get("attestation_hmac_sha256") or "")
    unsigned = dict(receipt)
    unsigned.pop("attestation_hmac_sha256", None)
    expected_digest = cross_runtime_canonical_digest(unsigned, digest_field="receipt_digest")
    expected_signature = hmac.new(
        key, expected_digest.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if (
        receipt.get("schema_version") != EXECUTOR_RECEIPT_SCHEMA_VERSION
        or receipt.get("source") != "trusted_company_policy_sandbox_executor"
        or receipt.get("plan_digest") != expected_plan_digest
        or receipt.get("receipt_digest") != expected_digest
        or not hmac.compare_digest(supplied, expected_signature)
    ):
        raise CompanyPolicySandboxExecutorError(
            "company_policy_executor_receipt_attestation_invalid"
        )


def _verify_worker_boot_receipt(
    receipt: Mapping[str, Any], *, key: bytes, plan: Mapping[str, Any]
) -> str:
    supplied = str(receipt.get("attestation_hmac_sha256") or "")
    unsigned = dict(receipt)
    unsigned.pop("attestation_hmac_sha256", None)
    expected_digest = cross_runtime_canonical_digest(
        unsigned, digest_field="receipt_digest"
    )
    expected_signature = hmac.new(
        key, expected_digest.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if (
        len(key) < 32
        or receipt.get("schema_version") != WORKER_BOOT_RECEIPT_SCHEMA_VERSION
        or receipt.get("source") != "trusted_company_policy_worker_bootstrap"
        or receipt.get("status") != "dedicated_ephemeral_worker_ready"
        or receipt.get("worker_identity") != plan.get("worker_identity")
        or receipt.get("pipeline_release_sha") != plan.get("pipeline_release_sha")
        or receipt.get("sandbox_attempt_id") != plan.get("sandbox_attempt_id")
        or receipt.get("dedicated_ephemeral_worker") is not True
        or receipt.get("scene_bytes_present") is not False
        or receipt.get("observation_bytes_present") is not False
        or receipt.get("mounted_customer_input_paths") != []
        or receipt.get("receipt_digest") != expected_digest
        or not hmac.compare_digest(supplied, expected_signature)
    ):
        raise CompanyPolicySandboxExecutorError(
            "company_policy_worker_boot_receipt_attestation_invalid"
        )
    return expected_digest


def _run_checked(
    runner: CommandRunner, argv: Sequence[str], *, timeout_seconds: int, blocker: str
) -> subprocess.CompletedProcess[str]:
    result = runner.run(argv, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        raise CompanyPolicySandboxExecutorError(blocker)
    return result


def _runtime_preflight_and_smoke(
    runner: CommandRunner, plan: Mapping[str, Any]
) -> str:
    runtime_info = _run_checked(
        runner,
        plan["commands"]["runtime_preflight"][0],
        timeout_seconds=15,
        blocker="company_policy_sandbox_docker_info_failed",
    )
    try:
        runtimes = json.loads(runtime_info.stdout)
    except json.JSONDecodeError as exc:
        raise CompanyPolicySandboxExecutorError(
            "company_policy_sandbox_runtime_inventory_invalid"
        ) from exc
    if not isinstance(runtimes, Mapping) or "runsc" not in runtimes:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_runsc_missing")
    docker_version = _run_checked(
        runner,
        plan["commands"]["runtime_preflight"][1],
        timeout_seconds=15,
        blocker="company_policy_sandbox_docker_version_failed",
    ).stdout.strip()
    if not docker_version:
        raise CompanyPolicySandboxExecutorError(
            "company_policy_sandbox_docker_version_invalid"
        )
    proxy_image = str(plan["images"]["blueprint_proxy"])
    _run_checked(
        runner,
        plan["commands"]["proxy_image_pull"],
        timeout_seconds=300,
        blocker="company_policy_sandbox_proxy_image_pull_failed",
    )
    proxy_repo_digests_result = _run_checked(
        runner,
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", proxy_image],
        timeout_seconds=30,
        blocker="company_policy_sandbox_proxy_image_inspect_failed",
    )
    try:
        proxy_repo_digests = json.loads(proxy_repo_digests_result.stdout)
    except json.JSONDecodeError as exc:
        raise CompanyPolicySandboxExecutorError(
            "company_policy_sandbox_proxy_image_digest_readback_invalid"
        ) from exc
    if not isinstance(proxy_repo_digests, list) or proxy_image not in proxy_repo_digests:
        raise CompanyPolicySandboxExecutorError(
            "company_policy_sandbox_proxy_image_digest_readback_mismatch"
        )
    runtime_smoke = _run_checked(
        runner,
        plan["commands"]["runtime_smoke"],
        timeout_seconds=120,
        blocker="company_policy_sandbox_runsc_smoke_failed",
    )
    if runtime_smoke.stdout.strip() != "runsc-smoke-ok":
        raise CompanyPolicySandboxExecutorError(
            "company_policy_sandbox_runsc_smoke_receipt_invalid"
        )
    return docker_version


def _validate_command_plan(
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    allowed_runtime_root: Path,
) -> None:
    commands = plan.get("commands")
    targets = plan.get("cleanup_targets")
    images = plan.get("images")
    if not isinstance(commands, Mapping) or not isinstance(targets, Mapping) or not isinstance(images, Mapping):
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_command_plan_invalid")
    runtime_root = allowed_runtime_root.expanduser().resolve()
    config_dir = Path(str(targets.get("registry_config_directory") or "")).resolve()
    ipc_dir = Path(str(targets.get("ipc_directory") or "")).resolve()
    for path in (config_dir, ipc_dir):
        if path == runtime_root or runtime_root not in path.parents:
            raise CompanyPolicySandboxExecutorError(
                "company_policy_sandbox_runtime_path_outside_worker_root"
            )
    policy_name = str(targets.get("policy_container") or "")
    proxy_name = str(targets.get("proxy_container") or "")
    policy_image = str(images.get("policy") or "")
    proxy_image = str(images.get("blueprint_proxy") or "")
    expected_cleanup = [
        ["docker", "rm", "--force", policy_name],
        ["docker", "rm", "--force", proxy_name],
        ["docker", "image", "rm", policy_image],
    ]
    if commands.get("cleanup") != expected_cleanup:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_cleanup_plan_invalid")
    if commands.get("image_pull") != ["docker", "--config", str(config_dir), "pull", policy_image]:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_image_pull_plan_invalid")
    if commands.get("proxy_image_pull") != ["docker", "pull", proxy_image]:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_proxy_pull_plan_invalid")
    if commands.get("runtime_preflight") != [
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        ["docker", "version", "--format", "{{.Server.Version}}"],
    ]:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_runtime_preflight_plan_invalid")
    container = contract["container"]
    resources = container["resources"]
    security = plan["security"]
    seccomp = str(security["seccomp_profile_path"])
    apparmor = str(security["apparmor_profile_id"])
    expected_policy_run = [
        "docker", "run", "--detach", "--name", policy_name,
        "--runtime", "runsc", "--network", f"container:{proxy_name}",
        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
        f"--security-opt=seccomp={seccomp}", f"--security-opt=apparmor={apparmor}",
        "--pids-limit", str(resources["pids_limit"]), "--cpus", str(resources["cpus"]),
        "--memory", f'{resources["memory_mib"]}m', "--tmpfs",
        f'/tmp:rw,noexec,nosuid,nodev,size={resources["tmpfs_mib"]}m',
        "--user", f'{container["run_as_uid"]}:{container["run_as_gid"]}',
        "--log-driver=none",
    ]
    if container["gpu_required"]:
        expected_policy_run.extend(["--gpus", "all"])
    expected_policy_run.extend(
        [policy_image, *[str(item) for item in container["serve_command"]]]
    )
    action_schema_b64 = base64.b64encode(
        json.dumps(
            contract["action_schema"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).decode("ascii")
    expected_proxy_run = [
        "docker", "run", "--detach", "--name", proxy_name,
        "--runtime", "runsc", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true", f"--security-opt=seccomp={seccomp}",
        f"--security-opt=apparmor={apparmor}", "--pids-limit=64", "--memory=256m",
        "--cpus=1", "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m", "--mount",
        f"type=bind,src={ipc_dir},dst=/run/blueprint-ipc", "--log-driver=none",
        proxy_image, "serve", "--unix-socket", "/run/blueprint-ipc/policy.sock",
        "--upstream", f'http://127.0.0.1:{container["port"]}', "--route", "/v1/actions",
        "--action-schema-b64", action_schema_b64,
    ]
    smoke_program = (
        "import os;required=" + ("True" if container["gpu_required"] else "False")
        + ";present=any(os.path.exists(p) for p in "
        "('/dev/nvidiactl','/dev/dxg'));assert (not required) or present;"
        "print('runsc-smoke-ok')"
    )
    expected_runtime_smoke = [
        "docker", "run", "--rm", "--runtime", "runsc", "--network=none",
        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
        f"--security-opt=seccomp={seccomp}", f"--security-opt=apparmor={apparmor}",
    ]
    if container["gpu_required"]:
        expected_runtime_smoke.extend(["--gpus", "all"])
    expected_runtime_smoke.extend([proxy_image, "python", "-c", smoke_program])
    if commands.get("policy_run") != expected_policy_run:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_policy_command_invalid")
    if commands.get("proxy_run") != expected_proxy_run:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_proxy_command_invalid")
    if commands.get("runtime_smoke") != expected_runtime_smoke:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_runtime_smoke_plan_invalid")
    probes = commands.get("network_probes")
    if not isinstance(probes, Mapping) or set(probes) != set(DENIAL_PROBES):
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_network_probe_plan_invalid")
    probe_targets = {
        "dns_resolution": ("dns", "example.com", "443"),
        "public_ipv4": ("tcp", "1.1.1.1", "443"),
        "public_ipv6": ("tcp", "2606:4700:4700::1111", "443"),
        "rfc1918": ("tcp", "10.255.255.1", "443"),
        "host_gateway": ("tcp", "172.17.0.1", "2375"),
        "link_local": ("tcp", "169.254.1.1", "80"),
        "cloud_metadata": ("tcp", "169.254.169.254", "80"),
        "registry_after_pull": (
            "tcp", str(plan["registry"]["resolved_global_addresses"][0]), "443"
        ),
        "redirect_target": ("tcp", "93.184.216.34", "80"),
    }
    for name, (kind, host, port) in probe_targets.items():
        expected = [
            "docker", "exec", proxy_name, "python", "-m",
            "blueprint_pipeline.company_policy_proxy", "probe", "--kind", kind,
            "--host", host, "--port", port,
        ]
        if probes.get(name) != expected:
            raise CompanyPolicySandboxExecutorError(
                "company_policy_sandbox_network_probe_plan_invalid"
            )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _zero_rgb_png(width: int, height: int) -> bytes:
    rows = b"".join(b"\x00" + b"\x00" * (width * 3) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _zero_shape(shape: Sequence[int]) -> Any:
    if not shape:
        return 0.0
    return [_zero_shape(shape[1:]) for _ in range(shape[0])]


def build_synthetic_observation(contract: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_company_policy_container_contract_v2(contract)
    cameras: dict[str, Any] = {}
    for camera in normalized["observation_schema"]["cameras"]:
        image = _zero_rgb_png(int(camera["width"]), int(camera["height"]))
        cameras[str(camera["name"])] = {
            "encoding": "lossless_png",
            "width": camera["width"],
            "height": camera["height"],
            "color_space": camera["color_space"],
            "sha256": _sha256_bytes(image),
            "data_base64": base64.b64encode(image).decode("ascii"),
        }
    state = {
        str(field["name"]): _zero_shape([int(item) for item in field["shape"]])
        for field in normalized["observation_schema"]["state_fields"]
    }
    return {
        "schema_version": "blueprint_company_policy_observation.v1",
        "request_id": "synthetic-conformance-no-scene",
        "synthetic": True,
        "prompt": "synthetic interface conformance only; no task or scene is present",
        "cameras": cameras,
        "state": state,
    }


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def unix_proxy_request(socket_path: str, payload: Mapping[str, Any], timeout_ms: int) -> Mapping[str, Any]:
    body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    connection = _UnixHTTPConnection(socket_path, timeout_ms / 1000.0)
    try:
        connection.request(
            "POST",
            "/v1/actions",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read(64 * 1024 + 1)
    finally:
        connection.close()
    if response.status != 200 or len(raw) > 64 * 1024:
        raise CompanyPolicySandboxExecutorError("company_policy_proxy_conformance_failed")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CompanyPolicySandboxExecutorError(
            "company_policy_proxy_conformance_response_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise CompanyPolicySandboxExecutorError(
            "company_policy_proxy_conformance_response_invalid"
        )
    return dict(value)


def wait_for_unix_socket(socket_path: str, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if Path(socket_path).is_socket():
            return True
        time.sleep(0.1)
    return False


def _docker_config(path: Path, *, server: str, username: str, secret: str) -> None:
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    auth = base64.b64encode(f"{username}:{secret}".encode("utf-8")).decode("ascii")
    _atomic_json(path / "config.json", {"auths": {server: {"auth": auth}}})


def execute_company_policy_sandbox_preobservation(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    broker: CredentialBroker | None,
    runner: CommandRunner,
    attestation_key: bytes,
    attestation_key_id: str,
    worker_boot_receipt: Mapping[str, Any],
    output_path: Path,
    proxy_request: Callable[[str, Mapping[str, Any], int], Mapping[str, Any]] = unix_proxy_request,
    socket_ready: Callable[[str, int], bool] = wait_for_unix_socket,
    apparmor_profiles_path: Path = Path("/sys/kernel/security/apparmor/profiles"),
    allowed_runtime_root: Path = Path("/run/blueprint"),
) -> dict[str, Any]:
    """Execute and measure the sandbox through synthetic conformance only."""

    normalized_contract = validate_company_policy_container_contract_v2(contract)
    expected_plan_digest = cross_runtime_canonical_digest(plan, digest_field="plan_digest")
    if plan.get("plan_digest") != expected_plan_digest:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_plan_digest_mismatch")
    if plan.get("contract_digest") != normalized_contract["contract_digest"]:
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_contract_digest_mismatch")
    if plan.get("runtime_class") != "runsc":
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_runtime_unqualified")
    worker_boot_receipt_digest = _verify_worker_boot_receipt(
        worker_boot_receipt,
        key=attestation_key,
        plan=plan,
    )
    _validate_command_plan(
        plan,
        normalized_contract,
        allowed_runtime_root=allowed_runtime_root,
    )

    security = dict(plan.get("security") or {})
    for path_field, digest_field in (
        ("seccomp_profile_path", "seccomp_profile_digest"),
        ("apparmor_profile_source_path", "apparmor_profile_digest"),
    ):
        profile_path = Path(str(security.get(path_field) or "")).resolve()
        if _sha256_bytes(profile_path.read_bytes()) != security.get(digest_field):
            raise CompanyPolicySandboxExecutorError(
                f"company_policy_sandbox_{path_field}_digest_mismatch"
            )
    apparmor_id = str(security.get("apparmor_profile_id") or "")
    loaded_profiles = apparmor_profiles_path.read_text(encoding="utf-8")
    if not any(line.startswith(f"{apparmor_id} ") for line in loaded_profiles.splitlines()):
        raise CompanyPolicySandboxExecutorError("company_policy_sandbox_apparmor_not_loaded")

    cleanup = list(plan["commands"]["cleanup"])
    cleanup_targets = dict(plan["cleanup_targets"])
    docker_config_dir = Path(str(cleanup_targets["registry_config_directory"]))
    ipc_dir = Path(str(cleanup_targets["ipc_directory"]))
    policy_image = str(plan["images"]["policy"])
    credential_acknowledged = False
    phase_order: list[str] = []
    network_probes: dict[str, Any] = {}
    executor_receipt: dict[str, Any] | None = None
    terminal_blockers: list[str] = []
    try:
        docker_version = _runtime_preflight_and_smoke(runner, plan)
        visibility = normalized_contract["container"]["visibility"]
        if visibility == "private":
            if broker is None:
                raise CompanyPolicySandboxExecutorError("company_policy_broker_missing")
            binding = dict(plan["credential_broker_request_binding"])
            lease_id = str(binding.pop("registry_credential_lease_id") or "")
            claim_body = {
                **binding,
                "sandbox_plan_digest": expected_plan_digest,
            }
            claim = broker.claim(lease_id=lease_id, body=claim_body)
            credential = claim.get("credential")
            delivery = claim.get("delivery_receipt")
            if not isinstance(credential, Mapping) or not isinstance(delivery, Mapping):
                raise CompanyPolicySandboxExecutorError("company_policy_broker_claim_invalid")
            registry_server = credential.get("registry_server")
            username = credential.get("username")
            secret = credential.get("secret")
            if registry_server != plan["registry"]["host"]:
                raise CompanyPolicySandboxExecutorError("company_policy_broker_registry_mismatch")
            if (
                not isinstance(username, str)
                or not username
                or not isinstance(secret, str)
                or not secret
            ):
                raise CompanyPolicySandboxExecutorError(
                    "company_policy_broker_credential_invalid"
                )
            _docker_config(
                docker_config_dir,
                server=str(registry_server),
                username=username,
                secret=secret,
            )
            delivery_id = str(delivery.get("delivery_id") or "")
            if not delivery_id:
                raise CompanyPolicySandboxExecutorError("company_policy_broker_delivery_invalid")
        else:
            docker_config_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            _atomic_json(docker_config_dir / "config.json", {"auths": {}})
            lease_id = ""
            claim_body = {}
            delivery_id = ""
        phase_order.append("credential_redeemed_for_exact_pull")

        _run_checked(
            runner,
            plan["commands"]["image_pull"],
            timeout_seconds=900,
            blocker="company_policy_sandbox_image_pull_failed",
        )
        repo_digests_result = _run_checked(
            runner,
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", policy_image],
            timeout_seconds=30,
            blocker="company_policy_sandbox_image_inspect_failed",
        )
        try:
            repo_digests = json.loads(repo_digests_result.stdout)
        except json.JSONDecodeError as exc:
            raise CompanyPolicySandboxExecutorError(
                "company_policy_sandbox_image_digest_readback_invalid"
            ) from exc
        if not isinstance(repo_digests, list) or policy_image not in repo_digests:
            raise CompanyPolicySandboxExecutorError(
                "company_policy_sandbox_image_digest_readback_mismatch"
            )
        pull_receipt = {
            "schema_version": "company_policy_image_pull_receipt.v1",
            "plan_digest": expected_plan_digest,
            "image": policy_image,
            "pulled_image_digest": "sha256:" + policy_image.rsplit("@sha256:", 1)[1],
            "repo_digests": sorted(str(item) for item in repo_digests),
            "registry_redirect_behavior": "not_measured_by_docker_engine",
            "worker_boot_receipt_digest": worker_boot_receipt_digest,
            "scene_bytes_present_during_pull": False,
            "observation_bytes_present_during_pull": False,
        }
        pull_receipt["image_pull_receipt_digest"] = cross_runtime_canonical_digest(
            pull_receipt, digest_field="image_pull_receipt_digest"
        )
        phase_order.append("policy_image_pulled_and_digest_verified")

        shutil.rmtree(docker_config_dir)
        if docker_config_dir.exists():
            raise CompanyPolicySandboxExecutorError("company_policy_docker_config_delete_failed")
        if visibility == "private":
            acknowledgement = broker.acknowledge(
                lease_id=lease_id,
                body={
                    **claim_body,
                    "schema_version": "company_policy_registry_credential_acknowledgement.v1",
                    "delivery_id": delivery_id,
                    "image_pull_receipt_digest": pull_receipt["image_pull_receipt_digest"],
                    "pulled_image_digest": pull_receipt["pulled_image_digest"],
                },
            )
            lease_receipt = acknowledgement.get("lease_receipt")
            if (
                not isinstance(lease_receipt, Mapping)
                or lease_receipt.get("status") != "consumed"
                or lease_receipt.get("ciphertext_deleted") is not True
                or lease_receipt.get("delivery_id") != delivery_id
            ):
                raise CompanyPolicySandboxExecutorError(
                    "company_policy_broker_acknowledgement_invalid"
                )
            credential_acknowledged = True
        else:
            credential_acknowledged = True
        phase_order.append("registry_credential_and_docker_config_deleted")

        ipc_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        phase_order.append("network_namespace_created")
        _run_checked(
            runner,
            plan["commands"]["proxy_run"],
            timeout_seconds=60,
            blocker="company_policy_sandbox_proxy_start_failed",
        )
        phase_order.append("blueprint_proxy_started")
        _run_checked(
            runner,
            plan["commands"]["policy_run"],
            timeout_seconds=60,
            blocker="company_policy_sandbox_policy_start_failed",
        )
        phase_order.append("policy_container_started")

        probe_commands = dict(plan["commands"]["network_probes"])
        for probe in DENIAL_PROBES:
            probe_result = _run_checked(
                runner,
                probe_commands[probe],
                timeout_seconds=15,
                blocker=f"company_policy_sandbox_probe_failed:{probe}",
            )
            try:
                row = json.loads(probe_result.stdout)
            except json.JSONDecodeError as exc:
                raise CompanyPolicySandboxExecutorError(
                    f"company_policy_sandbox_probe_receipt_invalid:{probe}"
                ) from exc
            if not isinstance(row, Mapping) or row.get("status") != "denied":
                raise CompanyPolicySandboxExecutorError(
                    f"company_policy_sandbox_egress_reachable:{probe}"
                )
            network_probes[probe] = dict(row)
        phase_order.append("no_egress_measured")

        synthetic_request = build_synthetic_observation(normalized_contract)
        socket_path = str(ipc_dir / "policy.sock")
        if not socket_ready(
            socket_path,
            int(normalized_contract["container"]["resources"]["startup_timeout_seconds"]),
        ):
            raise CompanyPolicySandboxExecutorError("company_policy_proxy_socket_not_ready")
        response = proxy_request(
            socket_path,
            synthetic_request,
            int(normalized_contract["container"]["resources"]["request_timeout_ms"]),
        )
        validate_action_response(
            response, action_schema=normalized_contract["action_schema"]
        )
        phase_order.append("synthetic_conformance_completed")
        network_probes["blueprint_proxy_unix_socket"] = {
            "status": "reachable",
            "redirect_followed": False,
        }
        network_probes["policy_action_route_through_proxy"] = {
            "status": "reachable",
            "redirect_followed": False,
        }

        evidence = {
            "plan_digest": expected_plan_digest,
            "policy_image_digest_verified": True,
            "credential_deleted_before_policy_start": credential_acknowledged,
            "docker_config_deleted_before_policy_start": not docker_config_dir.exists(),
            "dedicated_worker_verified": True,
            "worker_boot_receipt_digest": worker_boot_receipt_digest,
            "scene_bytes_present_during_pull": False,
            "observation_bytes_present_during_pull": False,
            "registry_redirect_behavior": "not_measured_by_docker_engine",
            "image_pull_receipt_digest": pull_receipt["image_pull_receipt_digest"],
            "isolated_runtime_verified": True,
            "raw_container_logs_retained": False,
            "teardown_plan_armed": True,
            "network_probes": network_probes,
            "synthetic_conformance": {
                "status": "conformant",
                "contract_digest": plan["contract_digest"],
                "real_observation_used": False,
                "request_digest": cross_runtime_canonical_digest(synthetic_request),
            },
            "completed_phase_order": phase_order,
        }
        executor_receipt = _signed_receipt(
            {
                "schema_version": EXECUTOR_RECEIPT_SCHEMA_VERSION,
                "source": "trusted_company_policy_sandbox_executor",
                "status": "qualified_before_first_observation",
                "plan_digest": expected_plan_digest,
                "contract_digest": plan["contract_digest"],
                "sandbox_attempt_id": plan["sandbox_attempt_id"],
                "worker_identity": plan["worker_identity"],
                "runtime_class": "runsc",
                "docker_server_version": docker_version,
                "proxy_image_digest_verified": True,
                "runsc_smoke_verified": True,
                "evidence": evidence,
                "launch_authority_granted": False,
                "provider_mutation_authorized": False,
                "provider_mutation_performed": False,
            },
            key=attestation_key,
            key_id=attestation_key_id,
        )
        qualification = evaluate_preobservation_sandbox_qualification(
            plan=plan,
            executor_receipt=executor_receipt,
            attestation_key=attestation_key,
        )
        result = {
            "status": "qualified_dry_run_no_real_observation",
            "executor_receipt": executor_receipt,
            "qualification_receipt": qualification,
            "real_observation_sent": False,
            "launch_authority_granted": False,
            "provider_mutation_authorized": False,
            "provider_mutation_performed": False,
        }
    except (
        CompanyPolicySandboxExecutorError,
        CompanyPolicySandboxV2Error,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        terminal_blockers.append(str(exc))
        result = {
            "status": "blocked_before_first_observation",
            "blockers": sorted(set(terminal_blockers)),
            "real_observation_sent": False,
            "launch_authority_granted": False,
            "provider_mutation_authorized": False,
            "provider_mutation_performed": False,
        }
    finally:
        cleanup_failures: list[str] = []
        for command in cleanup:
            try:
                cleanup_result = runner.run(command, timeout_seconds=60)
            except (OSError, subprocess.SubprocessError):
                cleanup_failures.append("company_policy_sandbox_cleanup_command_failed")
                continue
            if cleanup_result.returncode != 0 and "No such" not in (
                cleanup_result.stderr + cleanup_result.stdout
            ):
                cleanup_failures.append("company_policy_sandbox_cleanup_command_failed")
        for container_name in (
            str(cleanup_targets["policy_container"]),
            str(cleanup_targets["proxy_container"]),
        ):
            try:
                readback = runner.run(
                    ["docker", "container", "inspect", container_name],
                    timeout_seconds=15,
                )
            except (OSError, subprocess.SubprocessError):
                cleanup_failures.append("company_policy_sandbox_container_readback_failed")
                continue
            if readback.returncode == 0:
                cleanup_failures.append("company_policy_sandbox_container_retained")
        try:
            image_readback = runner.run(
                ["docker", "image", "inspect", policy_image], timeout_seconds=15
            )
        except (OSError, subprocess.SubprocessError):
            cleanup_failures.append("company_policy_sandbox_image_readback_failed")
        else:
            if image_readback.returncode == 0:
                cleanup_failures.append("company_policy_sandbox_customer_image_retained")
        shutil.rmtree(docker_config_dir, ignore_errors=True)
        shutil.rmtree(ipc_dir, ignore_errors=True)
        for path, blocker in (
            (docker_config_dir, "company_policy_sandbox_registry_config_retained"),
            (ipc_dir, "company_policy_sandbox_ipc_directory_retained"),
        ):
            if path.exists():
                cleanup_failures.append(blocker)
        if cleanup_failures:
            result = {
                "status": "blocked_teardown_incomplete",
                "blockers": sorted(set(cleanup_failures)),
                "real_observation_sent": False,
                "launch_authority_granted": False,
                "provider_mutation_authorized": False,
                "provider_mutation_performed": False,
            }
        terminal = _signed_receipt(
            {
                "schema_version": TERMINAL_RECEIPT_SCHEMA_VERSION,
                "source": "trusted_company_policy_sandbox_executor",
                "status": result["status"],
                "plan_digest": expected_plan_digest,
                "executor_receipt_digest": (
                    executor_receipt.get("receipt_digest") if executor_receipt else None
                ),
                "cleanup_complete": not cleanup_failures,
                "credential_ciphertext_deleted": credential_acknowledged,
                "customer_policy_image_removed": not cleanup_failures,
                "real_observation_sent": False,
                "blockers": result.get("blockers", []),
                "launch_authority_granted": False,
                "provider_mutation_authorized": False,
                "provider_mutation_performed": False,
            },
            key=attestation_key,
            key_id=attestation_key_id,
        )
        result["terminal_receipt"] = terminal
        _atomic_json(output_path, result)
    return result


__all__ = [
    "CompanyPolicySandboxExecutorError",
    "HttpCredentialBroker",
    "SubprocessCommandRunner",
    "build_synthetic_observation",
    "execute_company_policy_sandbox_preobservation",
    "unix_proxy_request",
    "wait_for_unix_socket",
    "verify_executor_receipt",
]
