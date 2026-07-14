"""Run the GR00T + OSCAR thin-image build on a verified DO CPU builder.

The launcher is fail-closed and dry by default. A paid mutation requires an
admitted build-plane record, ``--allow-paid``, a live catalog match for the
known 320 GB profile, zero builder-tagged droplets, a launch-bound host key, a
positive spend cap, and a two-hour-or-less TTL. A detached watchdog
independently deletes the droplet at the deadline.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .groot_oscar_infrastructure_admission import (
    BUILD_SCHEMA_VERSION,
    DIGITALOCEAN_CPU_BUILDER_PROFILE,
    build_build_plane_admission,
    build_cpu_build_execution_admission,
    build_digitalocean_cpu_builder_profile_evidence,
    build_live_machine_capability_evidence,
)
from .paid_resource_admission import require_paid_resource_admission

SCHEMA_VERSION = "groot_oscar_digitalocean_builder_run.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_digitalocean_builder_watchdog.v1"
DO_API = "https://api.digitalocean.com/v2"
BUILDER_TAG = "blueprint-groot-oscar-builder"
TEARDOWN_TAG = "auto-teardown-required"
READINESS_TIMEOUT_SECONDS = 15 * 60


def verify_packet_tarball(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the exact transfer archive before any paid builder allocation."""

    path = Path(str(packet.get("tarball_path") or "")).expanduser().resolve()
    declared = str(packet.get("tarball_sha256") or "").strip()
    blockers: list[str] = []
    if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        blockers.append("digitalocean_builder_packet_tarball_digest_invalid")
    observed = ""
    if not path.is_file():
        blockers.append("digitalocean_builder_packet_tarball_missing")
    else:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        observed = digest.hexdigest()
        if declared and observed != declared:
            blockers.append("digitalocean_builder_packet_tarball_digest_mismatch")
    return {
        "schema_version": "groot_oscar_builder_packet_tarball_verification.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "tarball_path": str(path),
        "declared_sha256": declared or None,
        "observed_sha256": observed or None,
        "raw_secret_values_recorded": False,
    }


def live_machine_probe_command(*, mount_path: str = "/") -> str:
    """Return a dependency-free probe whose JSON comes from the live host."""

    encoded_mount = json.dumps(mount_path)
    return f"""python3 - <<'PY'
import json, os, platform, shutil, subprocess
mount_path = {encoded_mount}
stats = os.statvfs(mount_path)
def ok(argv):
    try:
        return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
print(json.dumps({{
    "observation_source": "live_machine_probe",
    "system": platform.system(),
    "architecture": platform.machine(),
    "mount_path": mount_path,
    "free_bytes": stats.f_bavail * stats.f_frsize,
    "docker_cli_present": shutil.which("docker") is not None,
    "docker_daemon_responding": ok(["docker", "info"]),
    "docker_buildx_available": ok(["docker", "buildx", "version"]),
    "builder_ready_marker": os.path.isfile("/root/blueprint-builder-ready"),
}}, sort_keys=True))
PY"""


def parse_live_machine_probe(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("live_machine_probe_output_missing")
    try:
        observation = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError("live_machine_probe_output_invalid") from exc
    if not isinstance(observation, Mapping):
        raise ValueError("live_machine_probe_output_not_object")
    return build_live_machine_capability_evidence(observation)


def observe_local_machine(*, mount_path: str | Path) -> dict[str, Any]:
    """Measure the machine running the allocator; do not accept caller claims."""

    mount = Path(mount_path).expanduser().resolve()
    stats = os.statvfs(mount)

    def succeeds(command: Sequence[str]) -> bool:
        try:
            return (
                subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                ).returncode
                == 0
            )
        except (OSError, subprocess.SubprocessError):
            return False

    return build_live_machine_capability_evidence(
        {
            "observation_source": "live_machine_probe",
            "system": platform.system(),
            "architecture": platform.machine(),
            "mount_path": str(mount),
            "free_bytes": stats.f_bavail * stats.f_frsize,
            "docker_cli_present": shutil.which("docker") is not None,
            "docker_daemon_responding": succeeds(["docker", "info"]),
            "docker_buildx_available": succeeds(["docker", "buildx", "version"]),
            "builder_ready_marker": True,
        }
    )


DETACHED_LAUNCH_SCHEMA_VERSION = "groot_oscar_digitalocean_builder_launch.v1"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_secret(path: Path) -> str:
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"secret file empty: {path}")
    return value


def _request(
    *, token: str, method: str, path: str, payload: Mapping[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    parsed_path = urllib.parse.urlsplit(path)
    if not path.startswith("/") or parsed_path.scheme or parsed_path.netloc or parsed_path.fragment:
        raise ValueError("digitalocean_api_path_must_be_relative")
    if method not in {"DELETE", "GET", "POST"}:
        raise ValueError("digitalocean_api_method_not_allowed")
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        DO_API + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        # URL is bound above to the constant DigitalOcean API origin.
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            raw = response.read()
            parsed = json.loads(raw) if raw else {}
            return int(response.status), parsed if isinstance(parsed, dict) else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        return int(exc.code), parsed if isinstance(parsed, dict) else {}


def _host_key_material(private_path: Path) -> tuple[str, str, str]:
    private_path = private_path.expanduser().resolve()
    public_path = Path(str(private_path) + ".pub")
    private = private_path.read_bytes()
    public = public_path.read_text(encoding="utf-8").strip()
    completed = subprocess.run(
        ["ssh-keygen", "-lf", str(public_path), "-E", "sha256"],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.split()
    if len(fields) < 2 or not fields[1].startswith("SHA256:"):
        raise ValueError("launch_bound_host_key_fingerprint_unavailable")
    return (
        base64.b64encode(private).decode(),
        base64.b64encode((public + "\n").encode()).decode(),
        fields[1],
    )


def build_cloud_init(*, host_private_b64: str, host_public_b64: str, shutdown_minutes: int) -> str:
    """Return cloud-init with an exact client-generated SSH host identity."""

    if not host_private_b64 or not host_public_b64:
        raise ValueError("launch_bound_host_key_material_missing")
    if shutdown_minutes <= 0 or shutdown_minutes > 120:
        raise ValueError("shutdown_minutes_must_be_between_1_and_120")
    return f"""#cloud-config
ssh_deletekeys: false
bootcmd:
  - [bash, -c, "printf '%s' '{host_private_b64}' | base64 -d > /etc/ssh/ssh_host_ed25519_key && chmod 600 /etc/ssh/ssh_host_ed25519_key && printf '%s' '{host_public_b64}' | base64 -d > /etc/ssh/ssh_host_ed25519_key.pub && chmod 644 /etc/ssh/ssh_host_ed25519_key.pub && rm -f /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_rsa_key.pub /etc/ssh/ssh_host_ecdsa_key /etc/ssh/ssh_host_ecdsa_key.pub"]
package_update: true
packages:
  - ca-certificates
  - curl
  - git
  - jq
  - python3
  - docker.io
  - docker-buildx
write_files:
  - path: /etc/ssh/ssh_host_ed25519_key
    permissions: '0600'
    encoding: b64
    content: {host_private_b64}
  - path: /etc/ssh/ssh_host_ed25519_key.pub
    permissions: '0644'
    encoding: b64
    content: {host_public_b64}
runcmd:
  - rm -f /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_rsa_key.pub /etc/ssh/ssh_host_ecdsa_key /etc/ssh/ssh_host_ecdsa_key.pub
  - systemctl restart ssh
  - systemctl enable --now docker
  - mkdir -p /root/blueprint-build /root/.blueprint-secrets
  - chmod 700 /root/.blueprint-secrets
  - docker info
  - docker buildx version
  - touch /root/blueprint-builder-ready
  - shutdown -h +{shutdown_minutes}
"""


def build_droplet_payload(
    *, name: str, region: str, ssh_key_id: int, user_data: str
) -> dict[str, Any]:
    profile = DIGITALOCEAN_CPU_BUILDER_PROFILE
    return {
        "name": name,
        "region": region,
        "size": profile["size_slug"],
        "image": profile["image_slug"],
        "ssh_keys": [ssh_key_id],
        "backups": False,
        "ipv6": False,
        "monitoring": True,
        "tags": [BUILDER_TAG, TEARDOWN_TAG],
        "user_data": user_data,
    }


def known_hosts_line(*, ip: str, public_key_text: str) -> str:
    fields = public_key_text.strip().split()
    if len(fields) < 2 or fields[0] != "ssh-ed25519":
        raise ValueError("launch_bound_public_host_key_invalid")
    return f"{ip} {fields[0]} {fields[1]}\n"


def _ssh_options(*, private_key: Path, known_hosts: Path) -> list[str]:
    return [
        "-i",
        str(private_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=15",
    ]


def _delete_and_verify(*, token: str, droplet_id: str) -> dict[str, Any]:
    delete_http, _ = _request(token=token, method="DELETE", path=f"/droplets/{droplet_id}")
    verify_http: int | None = None
    for _ in range(30):
        verify_http, _ = _request(token=token, method="GET", path=f"/droplets/{droplet_id}")
        if verify_http == 404:
            break
        time.sleep(5)
    return {
        "delete_http_status": delete_http,
        "verify_http_status": verify_http,
        "provider_absence_confirmed": verify_http == 404,
    }


def _delete_with_fail_closed_evidence(*, token: str, droplet_id: str) -> dict[str, Any]:
    try:
        return _delete_and_verify(token=token, droplet_id=droplet_id)
    except Exception as exc:  # noqa: BLE001 - teardown uncertainty must be persisted
        return {
            "delete_http_status": None,
            "verify_http_status": None,
            "provider_absence_confirmed": False,
            "teardown_error_type": type(exc).__name__,
        }


def _list_droplets_by_tag(
    *, token: str, tag: str, per_page: int = 200, max_pages: int = 100
) -> tuple[int, list[dict[str, Any]]]:
    """Read every matching inventory page or fail closed without partial data."""

    rows: list[dict[str, Any]] = []
    encoded_tag = urllib.parse.quote(tag, safe="")
    for page in range(1, max_pages + 1):
        http_status, payload = _request(
            token=token,
            method="GET",
            path=(
                f"/droplets?tag_name={encoded_tag}&per_page={per_page}&page={page}"
            ),
        )
        if http_status != 200:
            return http_status, []
        page_rows = payload.get("droplets", []) if isinstance(payload, Mapping) else []
        if not isinstance(page_rows, list):
            return 502, []
        rows.extend(row for row in page_rows if isinstance(row, dict))
        if len(page_rows) < per_page:
            return 200, rows
    return 508, []


def _reconcile_ambiguous_create(
    *,
    token: str,
    name: str,
    region: str,
    attempts: int = 7,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    """Find and delete an accepted create whose response may have been lost."""

    observations: list[dict[str, Any]] = []
    deleted_ids: set[str] = set()
    final_exact_match_count: int | None = None
    inventory_verified = False
    for attempt in range(max(1, attempts)):
        if attempt:
            sleeper(5)
        try:
            http_status, tagged_rows = _list_droplets_by_tag(
                token=token, tag=BUILDER_TAG
            )
        except Exception as exc:  # noqa: BLE001 - mutation outcome must be reconciled
            observations.append(
                {
                    "attempt": attempt + 1,
                    "inventory_http_status": None,
                    "transport_error_type": type(exc).__name__,
                }
            )
            inventory_verified = False
            continue
        exact_matches = [
            row
            for row in tagged_rows
            if isinstance(row, Mapping)
            and row.get("name") == name
            and isinstance(row.get("region"), Mapping)
            and row["region"].get("slug") == region
            and BUILDER_TAG in (row.get("tags") or [])
            and TEARDOWN_TAG in (row.get("tags") or [])
        ]
        observations.append(
            {
                "attempt": attempt + 1,
                "inventory_http_status": http_status,
                "exact_match_count": len(exact_matches),
            }
        )
        if http_status != 200:
            inventory_verified = False
            continue
        inventory_verified = True
        final_exact_match_count = len(exact_matches)
        for row in exact_matches:
            droplet_id = str(row.get("id") or "").strip()
            if not droplet_id or droplet_id in deleted_ids:
                continue
            deleted_ids.add(droplet_id)
            try:
                deletion = _delete_and_verify(token=token, droplet_id=droplet_id)
            except Exception as exc:  # noqa: BLE001 - preserve teardown uncertainty
                deletion = {
                    "provider_absence_confirmed": False,
                    "error_type": type(exc).__name__,
                }
            observations.append(
                {
                    "attempt": attempt + 1,
                    "reconciled_droplet_id": droplet_id,
                    "deletion": deletion,
                }
            )
    absence_confirmed = bool(inventory_verified and final_exact_match_count == 0)
    return {
        "schema_version": "groot_oscar_digitalocean_create_reconciliation.v1",
        "status": "provider_terminal" if absence_confirmed else "teardown_unverified",
        "name": name,
        "region": region,
        "tag": BUILDER_TAG,
        "attempts": observations,
        "reconciled_droplet_ids": sorted(deleted_ids),
        "provider_absence_confirmed": absence_confirmed,
        "raw_secret_values_recorded": False,
    }


def watchdog(*, state_path: Path, token_file: Path) -> int:
    state = _load_object(state_path)
    output = state_path.parent
    droplet_id = str(state["droplet_id"])
    deadline = float(state["deadline_epoch"])
    cancelled = output / "watchdog_cancelled"
    while time.time() < deadline:
        if cancelled.is_file():
            write_json(
                output / "watchdog_result.json",
                {
                    "schema_version": WATCHDOG_SCHEMA_VERSION,
                    "status": "cancelled_after_supervisor_teardown",
                    "droplet_id": droplet_id,
                },
            )
            return 0
        time.sleep(15)
    result = _delete_and_verify(token=_read_secret(token_file), droplet_id=droplet_id)
    payload = {
        "schema_version": WATCHDOG_SCHEMA_VERSION,
        "status": (
            "provider_terminal" if result["provider_absence_confirmed"] else "teardown_unverified"
        ),
        "droplet_id": droplet_id,
        **result,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "watchdog_result.json", payload)
    return 0 if result["provider_absence_confirmed"] else 2


def _live_profile(*, token: str, region: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sizes_http, sizes_payload = _request(token=token, method="GET", path="/sizes?per_page=200")
    droplets_http, tagged_droplets = _list_droplets_by_tag(
        token=token, tag=BUILDER_TAG
    )
    if sizes_http != 200 or droplets_http != 200:
        raise RuntimeError("digitalocean_builder_inventory_query_failed")
    size = next(
        (
            row
            for row in sizes_payload.get("sizes", [])
            if isinstance(row, dict)
            and row.get("slug") == DIGITALOCEAN_CPU_BUILDER_PROFILE["size_slug"]
        ),
        {},
    )
    builders = [
        row
        for row in tagged_droplets
        if isinstance(row, dict) and BUILDER_TAG in (row.get("tags") or [])
    ]
    profile = build_digitalocean_cpu_builder_profile_evidence(
        size=size, region=region, observed_live_builders=len(builders)
    )
    return profile, builders


def _blocked_result(blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_pre_allocation",
        "blockers": sorted(set(blockers)),
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }


def run_builder(
    *,
    output_dir: Path,
    packet_manifest_path: Path,
    builder_evidence_path: Path,
    spend_path: Path,
    token_file: Path,
    docker_username_file: Path,
    docker_password_file: Path,
    login_private_key: Path,
    host_private_key: Path,
    ssh_key_id: int,
    region: str,
    allow_paid: bool,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    packet = _load_object(packet_manifest_path)
    builder = _load_object(builder_evidence_path)
    spend = _load_object(spend_path)
    admission = build_build_plane_admission(packet=packet, builder=builder, spend=spend)
    write_json(output / "build_plane_admission.json", admission)
    blockers = list(admission["blockers"])
    packet_tarball_verification = verify_packet_tarball(packet)
    write_json(
        output / "packet_tarball_verification.json", packet_tarball_verification
    )
    blockers.extend(packet_tarball_verification["blockers"])
    if not allow_paid:
        blockers.append("digitalocean_builder_allow_paid_flag_missing")
    if blockers:
        result = _blocked_result(blockers)
        write_json(output / "builder_run_result.json", result)
        return result

    require_paid_resource_admission(
        admission,
        resource_class="cpu_build",
        expected_schema_version=BUILD_SCHEMA_VERSION,
    )

    token = _read_secret(token_file)
    profile, builders = _live_profile(token=token, region=region)
    write_json(output / "live_builder_profile_evidence.json", profile)
    if profile["status"] != "verified" or builders:
        result = _blocked_result(profile["blockers"] or ["digitalocean_builder_overlap_detected"])
        write_json(output / "builder_run_result.json", result)
        return result

    host_private_b64, host_public_b64, fingerprint = _host_key_material(host_private_key)
    if fingerprint != builder.get("ssh_host_key_sha256"):
        raise RuntimeError("builder_launch_bound_host_key_fingerprint_mismatch")
    ttl = int(spend["hard_ttl_seconds"])
    hourly = float(profile["observed"]["price_hourly_usd"])
    maximum_cost = hourly * ttl / 3600
    if maximum_cost > float(spend["max_spend_usd"]):
        raise RuntimeError("digitalocean_builder_cost_exceeds_authorized_cap")
    name = f"blueprint-groot-oscar-thin-{str(packet['source_commit'])[:8]}"
    user_data = build_cloud_init(
        host_private_b64=host_private_b64,
        host_public_b64=host_public_b64,
        shutdown_minutes=max(1, min(120, (ttl + 59) // 60)),
    )
    create_payload = build_droplet_payload(
        name=name, region=region, ssh_key_id=ssh_key_id, user_data=user_data
    )
    started = time.time()
    create_error_type: str | None = None
    try:
        create_http, create_response = _request(
            token=token, method="POST", path="/droplets", payload=create_payload
        )
    except Exception as exc:  # noqa: BLE001 - a lost create response is ambiguous
        create_http, create_response = 0, {}
        create_error_type = type(exc).__name__
    droplet = create_response.get("droplet") if isinstance(create_response, dict) else None
    droplet_id = str((droplet or {}).get("id") or "")
    write_json(
        output / "create_result.json",
        {
            "http_status": create_http,
            "transport_error_type": create_error_type,
            "droplet_id": droplet_id or None,
            "name": name,
            "region": region,
            "size_slug": DIGITALOCEAN_CPU_BUILDER_PROFILE["size_slug"],
            "user_data_raw_recorded": False,
            "raw_secret_values_recorded": False,
        },
    )
    create_succeeded = create_http in {200, 201, 202} and bool(droplet_id)
    definitive_rejection = (
        400 <= create_http < 500 and create_http not in {408, 409, 425, 429}
    )
    if not create_succeeded and not definitive_rejection:
        reconciliation = _reconcile_ambiguous_create(
            token=token,
            name=name,
            region=region,
        )
        write_json(output / "ambiguous_create_reconciliation.json", reconciliation)
        absence_confirmed = reconciliation["provider_absence_confirmed"] is True
        result = {
            **_blocked_result(
                [
                    (
                        "digitalocean_builder_ambiguous_create_reconciled"
                        if absence_confirmed
                        else "digitalocean_builder_ambiguous_create_teardown_unverified"
                    )
                ]
            ),
            "status": (
                "ambiguous_create_reconciled_no_allocation"
                if absence_confirmed
                else "ambiguous_create_teardown_unverified"
            ),
            "provider_mutation_performed": True,
            "provider_absence_confirmed": absence_confirmed,
        }
        write_json(output / "builder_run_result.json", result)
        return result
    if not create_succeeded:
        result = {
            **_blocked_result(["digitalocean_builder_create_rejected"]),
            "status": "create_rejected_no_allocation",
            "provider_mutation_performed": True,
            "provider_absence_confirmed": True,
        }
        write_json(output / "builder_run_result.json", result)
        return result

    state_path = output / "allocation_state.json"
    write_json(
        state_path,
        {
            "droplet_id": droplet_id,
            "deadline_epoch": started + ttl,
            "source_commit": packet["source_commit"],
            "maximum_spend_usd": maximum_cost,
        },
    )
    watchdog_log = (output / "watchdog.log").open("ab")
    watchdog_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_digitalocean_builder",
            "watchdog",
            "--state",
            str(state_path),
            "--token-file",
            str(token_file),
        ],
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (output / "watchdog.pid").write_text(f"{watchdog_process.pid}\n", encoding="utf-8")
    build_exit: int | None = None
    public_ip = ""
    teardown: dict[str, Any] = {"provider_absence_confirmed": False}
    try:
        deadline = started + ttl
        readiness_deadline = min(deadline, started + READINESS_TIMEOUT_SECONDS)
        while time.time() < readiness_deadline:
            inspect_http, inspect = _request(
                token=token, method="GET", path=f"/droplets/{droplet_id}"
            )
            if inspect_http != 200:
                time.sleep(5)
                continue
            row = inspect.get("droplet") if isinstance(inspect, dict) else {}
            networks = ((row or {}).get("networks") or {}).get("v4") or []
            public = [
                item.get("ip_address")
                for item in networks
                if isinstance(item, dict) and item.get("type") == "public"
            ]
            if public:
                public_ip = str(public[0])
                break
            time.sleep(5)
        if not public_ip:
            raise RuntimeError("digitalocean_builder_public_ip_timeout")

        public_key = Path(str(host_private_key.expanduser().resolve()) + ".pub").read_text(
            encoding="utf-8"
        )
        known_hosts = output / "launch_bound_known_hosts"
        known_hosts.write_text(
            known_hosts_line(ip=public_ip, public_key_text=public_key),
            encoding="utf-8",
        )
        options = _ssh_options(private_key=login_private_key.expanduser(), known_hosts=known_hosts)
        live_capability: dict[str, Any] | None = None
        remote_preflight = live_machine_probe_command(mount_path="/")
        while time.time() < readiness_deadline:
            completed = subprocess.run(
                ["ssh", *options, f"root@{public_ip}", remote_preflight],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                try:
                    candidate = parse_live_machine_probe(completed.stdout)
                except ValueError:
                    candidate = None
                if candidate is not None:
                    live_capability = candidate
                    write_json(output / "live_machine_capability.json", candidate)
                    if (
                        candidate["status"] == "verified"
                        and candidate.get("builder_ready_marker") is True
                    ):
                        break
            time.sleep(10)
        if (
            live_capability is None
            or live_capability["status"] != "verified"
            or live_capability.get("builder_ready_marker") is not True
        ):
            raise RuntimeError("digitalocean_builder_runtime_preflight_failed")

        execution_admission = build_cpu_build_execution_admission(
            allocation_admission=admission,
            live_machine=live_capability,
        )
        write_json(output / "cpu_build_execution_admission.json", execution_admission)
        if execution_admission["status"] != "admitted":
            raise RuntimeError("digitalocean_builder_execution_admission_blocked")

        packet_tarball = Path(packet_tarball_verification["tarball_path"])
        packet_tarball_sha256 = str(
            packet_tarball_verification["observed_sha256"] or ""
        )
        subprocess.run(
            [
                "scp",
                *options,
                str(packet_tarball),
                str(docker_username_file.expanduser()),
                str(docker_password_file.expanduser()),
                f"root@{public_ip}:/root/blueprint-build/",
            ],
            check=True,
        )
        remote_tarball = "/root/blueprint-build/" + packet_tarball.name
        remote_command = " && ".join(
            [
                "set -euo pipefail",
                "install -d -m 700 /root/.blueprint-secrets",
                "install -m 600 /root/blueprint-build/docker_username /root/.blueprint-secrets/docker_username",
                "install -m 600 /root/blueprint-build/docker_pat /root/.blueprint-secrets/docker_pat",
                "rm -f /root/blueprint-build/docker_username /root/blueprint-build/docker_pat",
                "mkdir -p /root/blueprint-build/run",
                "printf '%s  %s\\n' "
                f"{shlex.quote(packet_tarball_sha256)} "
                f"{shlex.quote(remote_tarball)} | sha256sum -c -",
                f"tar -xzf {shlex.quote(remote_tarball)} -C /root/blueprint-build/run",
                "cd /root/blueprint-build/run/groot_oscar_thin_remote_build",
                "BLUEPRINT_REMOTE_IMAGE_BUILD_DOCKER_LOGIN=true ./remote_build_groot_oscar_thin_images.sh",
            ]
        )
        with (output / "remote_build.log").open("wb") as log:
            completed = subprocess.run(
                ["ssh", *options, f"root@{public_ip}", "bash", "-s"],
                input=(remote_command + "\n").encode(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        build_exit = completed.returncode
        remote_result_dir = "/root/blueprint-build/run/groot_oscar_thin_remote_build"
        results_dir = output / "remote_results"
        ensure_dir(results_dir)
        subprocess.run(
            [
                "scp",
                *options,
                f"root@{public_ip}:{remote_result_dir}/*.json",
                str(results_dir) + "/",
            ],
            check=False,
        )
        build_status = "completed" if build_exit == 0 else "failed"
    except Exception as exc:
        write_json(
            output / "builder_error.json",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        build_status = "failed"
    finally:
        teardown = _delete_with_fail_closed_evidence(
            token=token, droplet_id=droplet_id
        )
        elapsed = max(0.0, time.time() - started)
        teardown.update(
            {
                "schema_version": "groot_oscar_digitalocean_builder_teardown.v1",
                "droplet_id": droplet_id,
                "elapsed_seconds": elapsed,
                "maximum_compute_spend_usd": hourly * elapsed / 3600,
                "raw_secret_values_recorded": False,
            }
        )
        write_json(output / "teardown.json", teardown)
        if teardown["provider_absence_confirmed"]:
            (output / "watchdog_cancelled").touch()

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "completed"
            if build_status == "completed" and teardown["provider_absence_confirmed"]
            else "failed"
        ),
        "blockers": [
            *([] if build_status == "completed" else ["remote_thin_image_build_failed"]),
            *(
                []
                if teardown["provider_absence_confirmed"]
                else ["digitalocean_builder_teardown_unverified"]
            ),
        ],
        "droplet_id": droplet_id,
        "build_exit_code": build_exit,
        "source_commit": packet["source_commit"],
        "provider_absence_confirmed": teardown["provider_absence_confirmed"],
        "maximum_compute_spend_usd": teardown["maximum_compute_spend_usd"],
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "image_build_is_not_model_cache_verification": True,
            "image_build_is_not_runpod_startup": True,
            "image_build_is_not_task_success": True,
        },
    }
    write_json(output / "builder_run_result.json", result)
    return result


def launch_detached_builder(*, output_dir: Path, run_arguments: Sequence[str]) -> dict[str, Any]:
    """Start the paid-gated supervisor outside the invoking terminal session."""

    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    result_path = output / "builder_run_result.json"
    if result_path.exists():
        raise ValueError("builder_output_already_has_terminal_result")
    lock_path = output / "supervisor.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("builder_output_already_has_supervisor_lock") from exc
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
        lock.write(f"created_by_pid={os.getpid()}\n")
    log_path = output / "supervisor.log"
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.paid_resource_allocator",
        "cpu-build-run",
        *run_arguments,
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    payload = {
        "schema_version": DETACHED_LAUNCH_SCHEMA_VERSION,
        "status": "supervisor_started",
        "pid": process.pid,
        "output_dir": str(output),
        "log_path": str(log_path),
        "start_new_session": True,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "supervisor_launch.json", payload)
    (output / "supervisor.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw and raw[0] in {"run", "launch"}:
        print("legacy_cpu_builder_launcher_disabled:use_blueprint-allocate-cpu-build")
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    watch = subparsers.add_parser("watchdog")
    watch.add_argument("--state", required=True)
    watch.add_argument("--token-file", required=True)
    args = parser.parse_args(raw)
    return watchdog(state_path=Path(args.state), token_file=Path(args.token_file))


if __name__ == "__main__":
    raise SystemExit(main())
