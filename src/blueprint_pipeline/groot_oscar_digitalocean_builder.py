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
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .groot_oscar_infrastructure_admission import (
    DIGITALOCEAN_CPU_BUILDER_PROFILE,
    build_build_plane_admission,
    build_digitalocean_cpu_builder_profile_evidence,
)

SCHEMA_VERSION = "groot_oscar_digitalocean_builder_run.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_digitalocean_builder_watchdog.v1"
DO_API = "https://api.digitalocean.com/v2"
BUILDER_TAG = "blueprint-groot-oscar-builder"
TEARDOWN_TAG = "auto-teardown-required"
READINESS_TIMEOUT_SECONDS = 15 * 60


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
        with urllib.request.urlopen(request, timeout=60) as response:
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


def build_cloud_init(
    *, host_private_b64: str, host_public_b64: str, shutdown_minutes: int
) -> str:
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
    delete_http, _ = _request(
        token=token, method="DELETE", path=f"/droplets/{droplet_id}"
    )
    verify_http: int | None = None
    for _ in range(30):
        verify_http, _ = _request(
            token=token, method="GET", path=f"/droplets/{droplet_id}"
        )
        if verify_http == 404:
            break
        time.sleep(5)
    return {
        "delete_http_status": delete_http,
        "verify_http_status": verify_http,
        "provider_absence_confirmed": verify_http == 404,
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
    result = _delete_and_verify(
        token=_read_secret(token_file), droplet_id=droplet_id
    )
    payload = {
        "schema_version": WATCHDOG_SCHEMA_VERSION,
        "status": (
            "provider_terminal"
            if result["provider_absence_confirmed"]
            else "teardown_unverified"
        ),
        "droplet_id": droplet_id,
        **result,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "watchdog_result.json", payload)
    return 0 if result["provider_absence_confirmed"] else 2


def _live_profile(
    *, token: str, region: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sizes_http, sizes_payload = _request(
        token=token, method="GET", path="/sizes?per_page=200"
    )
    droplets_http, droplets_payload = _request(
        token=token, method="GET", path="/droplets?per_page=200"
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
        for row in droplets_payload.get("droplets", [])
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
    admission = build_build_plane_admission(
        packet=packet, builder=builder, spend=spend
    )
    write_json(output / "build_plane_admission.json", admission)
    blockers = list(admission["blockers"])
    if not allow_paid:
        blockers.append("digitalocean_builder_allow_paid_flag_missing")
    if blockers:
        result = _blocked_result(blockers)
        write_json(output / "builder_run_result.json", result)
        return result

    token = _read_secret(token_file)
    profile, builders = _live_profile(token=token, region=region)
    write_json(output / "live_builder_profile_evidence.json", profile)
    if profile["status"] != "verified" or builders:
        result = _blocked_result(
            profile["blockers"] or ["digitalocean_builder_overlap_detected"]
        )
        write_json(output / "builder_run_result.json", result)
        return result

    host_private_b64, host_public_b64, fingerprint = _host_key_material(
        host_private_key
    )
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
    create_http, create_response = _request(
        token=token, method="POST", path="/droplets", payload=create_payload
    )
    droplet = (
        create_response.get("droplet") if isinstance(create_response, dict) else None
    )
    droplet_id = str((droplet or {}).get("id") or "")
    write_json(
        output / "create_result.json",
        {
            "http_status": create_http,
            "droplet_id": droplet_id or None,
            "name": name,
            "region": region,
            "size_slug": DIGITALOCEAN_CPU_BUILDER_PROFILE["size_slug"],
            "user_data_raw_recorded": False,
            "raw_secret_values_recorded": False,
        },
    )
    if create_http not in {200, 201, 202} or not droplet_id:
        result = {
            **_blocked_result(["digitalocean_builder_create_failed"]),
            "status": "create_failed_no_allocation_id",
            "provider_mutation_performed": True,
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
    (output / "watchdog.pid").write_text(
        f"{watchdog_process.pid}\n", encoding="utf-8"
    )
    build_status = "blocked"
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

        public_key = Path(
            str(host_private_key.expanduser().resolve()) + ".pub"
        ).read_text(encoding="utf-8")
        known_hosts = output / "launch_bound_known_hosts"
        known_hosts.write_text(
            known_hosts_line(ip=public_ip, public_key_text=public_key),
            encoding="utf-8",
        )
        options = _ssh_options(
            private_key=login_private_key.expanduser(), known_hosts=known_hosts
        )
        ready = False
        remote_preflight = (
            "test -f /root/blueprint-builder-ready && "
            "docker info >/dev/null && docker buildx version >/dev/null && "
            "test $(uname -m) = x86_64 && "
            "test $(df -Pk / | awk 'NR==2 {print $4}') -ge 125829120"
        )
        while time.time() < readiness_deadline:
            completed = subprocess.run(
                ["ssh", *options, f"root@{public_ip}", remote_preflight],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                ready = True
                break
            time.sleep(10)
        if not ready:
            raise RuntimeError("digitalocean_builder_runtime_preflight_failed")

        packet_tarball = Path(str(packet["tarball_path"])).expanduser().resolve()
        if not packet_tarball.is_file():
            raise RuntimeError("digitalocean_builder_packet_tarball_missing")
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
                "install -m 600 /root/blueprint-build/docker_username /root/.blueprint-secrets/docker_username",
                "install -m 600 /root/blueprint-build/docker_pat /root/.blueprint-secrets/docker_pat",
                "rm -f /root/blueprint-build/docker_username /root/blueprint-build/docker_pat",
                "mkdir -p /root/blueprint-build/run",
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
        remote_result_dir = (
            "/root/blueprint-build/run/groot_oscar_thin_remote_build"
        )
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
    except BaseException as exc:
        write_json(
            output / "builder_error.json",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        build_status = "failed"
    finally:
        teardown = _delete_and_verify(token=token, droplet_id=droplet_id)
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
        "blockers": (
            [] if build_status == "completed" else ["remote_thin_image_build_failed"]
        ),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--packet-manifest", required=True)
    run.add_argument("--builder-evidence", required=True)
    run.add_argument("--spend", required=True)
    run.add_argument(
        "--token-file", default="~/.blueprint-secrets/digitalocean_api_token"
    )
    run.add_argument(
        "--docker-username-file", default="~/.blueprint-secrets/docker_username"
    )
    run.add_argument(
        "--docker-password-file", default="~/.blueprint-secrets/docker_pat"
    )
    run.add_argument("--login-private-key", required=True)
    run.add_argument("--host-private-key", required=True)
    run.add_argument("--ssh-key-id", required=True, type=int)
    run.add_argument("--region", default="sfo3")
    run.add_argument("--allow-paid", action="store_true")
    watch = subparsers.add_parser("watchdog")
    watch.add_argument("--state", required=True)
    watch.add_argument("--token-file", required=True)
    args = parser.parse_args(argv)
    if args.command == "watchdog":
        return watchdog(state_path=Path(args.state), token_file=Path(args.token_file))
    result = run_builder(
        output_dir=Path(args.output_dir),
        packet_manifest_path=Path(args.packet_manifest),
        builder_evidence_path=Path(args.builder_evidence),
        spend_path=Path(args.spend),
        token_file=Path(args.token_file),
        docker_username_file=Path(args.docker_username_file),
        docker_password_file=Path(args.docker_password_file),
        login_private_key=Path(args.login_private_key),
        host_private_key=Path(args.host_private_key),
        ssh_key_id=args.ssh_key_id,
        region=args.region,
        allow_paid=args.allow_paid,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
