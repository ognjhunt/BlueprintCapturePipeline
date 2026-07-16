"""Governed DigitalOcean pre-baked host preparation for GR00T + OSCAR.

This module is an adapter behind ``paid_resource_allocator gpu-canary``.  It is
not a public allocation entrypoint.  One bounded H100 allocation pulls the
exact release digest while the verified RunPod model cache is copied to an
independent DigitalOcean volume.  The powered-off boot disk is snapshotted,
the compute allocation is destroyed, and both retained artifacts are verified
before the source RunPod cache may be considered replaceable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .gpu_render_providers import DigitalOceanRenderProvider
from .groot_oscar_digitalocean_builder import (
    _host_key_material,
    _read_private_secret,
    _read_secret,
    _request,
    _ssh_options,
    known_hosts_line,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import PaidProviderLaneLeaseSet
from .production_gpu_campaign_budget import (
    CampaignBudgetExceeded,
    ProductionGpuCampaignBudget,
)


SCHEMA_VERSION = "groot_oscar_digitalocean_prebaked_host.v1"
ADMISSION_SCHEMA_VERSION = "groot_oscar_digitalocean_prebaked_host_admission.v1"
PREFLIGHT_SCHEMA_VERSION = "groot_oscar_digitalocean_prebaked_host_preflight.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_digitalocean_prebaked_host_watchdog.v1"
RESULT_FILENAME = "digitalocean_prebaked_host_result.json"
DEFAULT_REGION = "ams3"
DEFAULT_SIZE = "gpu-h100x1-80gb"
DEFAULT_SOURCE_IMAGE = "gpu-h100x1-base"
DEFAULT_VOLUME_GIB = 50
MAX_PREBAKE_GPU_SECONDS = 1_396
REQUIRED_FUTURE_GPU_SECONDS = 3_980
MAX_HOURLY_RATE_USD = 3.50
RETENTION_TTL_SECONDS = 3 * 60 * 60
MAX_SNAPSHOT_GIB = 720
DIGITALOCEAN_VOLUME_USD_PER_GIB_MONTH = 0.10
DIGITALOCEAN_SNAPSHOT_USD_PER_GIB_MONTH = 0.06
HOURS_PER_BILLING_MONTH = 730
RESOURCE_PREFIX = "blueprint-groot-oscar-prebake-"
VOLUME_PREFIX = "blueprint-groot-oscar-models-do-"
IMAGE_PREFIX = "blueprint-groot-oscar-host-do-"
LANE = "groot_oscar_digitalocean_prebaked_host"
RUNPOD_S3_ENDPOINT = "https://s3api-eur-is-1.runpod.io/"
RUNPOD_S3_BUCKET = "hw3pjbt675"
RUNPOD_CACHE_PREFIX = ".blueprint-model-cache/blueprint-groot-oscar-v1"


def _object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _digest_ref(value: object) -> bool:
    text = str(value or "")
    marker = "@sha256:"
    if marker not in text:
        return False
    return len(text.rsplit(marker, 1)[-1]) == 64 and all(
        char in "0123456789abcdef" for char in text.rsplit(marker, 1)[-1]
    )


def build_prebake_admission(
    *,
    release: Mapping[str, Any],
    model_cache: Mapping[str, Any],
    preflight: Mapping[str, Any],
    volume_size_gib: int,
    reservation_seconds: int,
    future_gpu_seconds: int,
    initial_spent_usd: float,
    initial_gpu_seconds: int,
    total_spend_cap_usd: float,
    gpu_wall_cap_seconds: int,
    max_hourly_rate_usd: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    image_ref = str(release.get("release_image_ref") or release.get("resolved_digest_ref") or "")
    thin = release.get("thin_release_contract")
    if release.get("status") != "completed":
        blockers.append("prebake_release_evidence_not_completed")
    if not _digest_ref(image_ref):
        blockers.append("prebake_release_image_not_digest_pinned")
    if not isinstance(thin, Mapping) or thin.get("status") != "passed":
        blockers.append("prebake_thin_release_contract_not_passed")
    elif thin.get("models_externalized") is not True:
        blockers.append("prebake_release_models_not_externalized")
    if model_cache.get("status") != "passed":
        blockers.append("prebake_model_cache_not_verified")
    if model_cache.get("provider_volume_id") != RUNPOD_S3_BUCKET:
        blockers.append("prebake_source_model_volume_mismatch")
    if model_cache.get("remote_prefix") != RUNPOD_CACHE_PREFIX:
        blockers.append("prebake_source_model_prefix_mismatch")
    if model_cache.get("runtime_path_mapping_verified") is not True:
        blockers.append("prebake_model_cache_runtime_mapping_unverified")
    if preflight.get("status") != "ready":
        blockers.append("prebake_digitalocean_preflight_not_ready")
    if volume_size_gib != DEFAULT_VOLUME_GIB:
        blockers.append("prebake_model_volume_size_must_equal_50_gib")
    if type(reservation_seconds) is not int or not 60 < reservation_seconds <= MAX_PREBAKE_GPU_SECONDS:
        blockers.append("prebake_reservation_outside_authorized_window")
    if future_gpu_seconds < REQUIRED_FUTURE_GPU_SECONDS:
        blockers.append("prebake_future_campaign_reservation_below_3980_seconds")
    if initial_gpu_seconds + reservation_seconds + future_gpu_seconds > gpu_wall_cap_seconds:
        blockers.append("prebake_combined_plan_exceeds_gpu_wall_cap")
    if gpu_wall_cap_seconds > 21_000:
        blockers.append("prebake_gpu_wall_cap_exceeds_21000")
    if total_spend_cap_usd != 20.0:
        blockers.append("prebake_total_spend_cap_must_equal_20")
    if not 0 < max_hourly_rate_usd <= MAX_HOURLY_RATE_USD:
        blockers.append("prebake_hourly_rate_exceeds_authorized_ceiling")
    maximum_gpu_spend = max_hourly_rate_usd * reservation_seconds / 3600.0
    if initial_spent_usd + maximum_gpu_spend > total_spend_cap_usd:
        blockers.append("prebake_reservation_exceeds_total_spend_cap")
    maximum_storage_spend = (
        (
            volume_size_gib * DIGITALOCEAN_VOLUME_USD_PER_GIB_MONTH
            + MAX_SNAPSHOT_GIB * DIGITALOCEAN_SNAPSHOT_USD_PER_GIB_MONTH
        )
        / HOURS_PER_BILLING_MONTH
        * (RETENTION_TTL_SECONDS / 3600.0)
    )
    future_gpu_spend = max_hourly_rate_usd * future_gpu_seconds / 3600.0
    if (
        initial_spent_usd
        + maximum_gpu_spend
        + future_gpu_spend
        + maximum_storage_spend
        > total_spend_cap_usd
    ):
        blockers.append("prebake_combined_compute_and_storage_plan_exceeds_spend_cap")
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "release_image_ref": image_ref or None,
        "model_manifest_digest": model_cache.get("model_manifest_digest"),
        "reservation_gpu_seconds": reservation_seconds,
        "future_campaign_allowance_gpu_seconds": future_gpu_seconds,
        "maximum_gpu_spend_usd": round(maximum_gpu_spend, 6),
        "maximum_retained_storage_spend_usd": round(maximum_storage_spend, 6),
        "retention_ttl_seconds": RETENTION_TTL_SECONDS,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "admission_is_not_provider_allocation": True,
            "host_bake_is_not_runtime_startup": True,
            "model_copy_is_not_semantic_task_success": True,
        },
    }


def read_only_preflight(
    *, token: str, region: str, size: str, source_image: str, name: str
) -> dict[str, Any]:
    blockers: list[str] = []
    account_http, account_body = _request(token=token, method="GET", path="/account")
    sizes_http, sizes_body = _request(token=token, method="GET", path="/sizes?per_page=200")
    images_http, images_body = _request(
        token=token, method="GET", path=f"/images/{source_image}"
    )
    droplets_http, droplets_body = _request(
        token=token,
        method="GET",
        path=f"/droplets?tag_name={LANE}&per_page=200",
    )
    volumes_http, volumes_body = _request(
        token=token, method="GET", path="/volumes?per_page=200"
    )
    private_images_http, private_images_body = _request(
        token=token, method="GET", path="/images?private=true&per_page=200"
    )
    account = account_body.get("account") if isinstance(account_body, Mapping) else {}
    if account_http != 200 or not isinstance(account, Mapping) or account.get("status") != "active":
        blockers.append("digitalocean_account_not_active")
    sizes = sizes_body.get("sizes") if isinstance(sizes_body, Mapping) else []
    size_row = next(
        (dict(row) for row in sizes or [] if isinstance(row, Mapping) and row.get("slug") == size),
        {},
    )
    if sizes_http != 200 or size_row.get("available") is not True:
        blockers.append("digitalocean_prebake_size_unavailable")
    if region not in (size_row.get("regions") or []):
        blockers.append("digitalocean_prebake_size_region_unavailable")
    source_value = images_body.get("image") if isinstance(images_body, Mapping) else {}
    source_row = dict(source_value) if isinstance(source_value, Mapping) else {}
    if images_http != 200 or source_row.get("status") != "available":
        blockers.append("digitalocean_prebake_source_image_unavailable")
    if region not in (source_row.get("regions") or []):
        blockers.append("digitalocean_prebake_source_image_region_unavailable")
    droplets = droplets_body.get("droplets") if isinstance(droplets_body, Mapping) else []
    matching_droplets = [
        str(row.get("id"))
        for row in droplets or []
        if isinstance(row, Mapping) and str(row.get("name") or "").startswith(RESOURCE_PREFIX)
    ]
    volumes = volumes_body.get("volumes") if isinstance(volumes_body, Mapping) else []
    matching_volumes = [
        str(row.get("id"))
        for row in volumes or []
        if isinstance(row, Mapping) and str(row.get("name") or "").startswith(VOLUME_PREFIX)
    ]
    private_images = (
        private_images_body.get("images") if isinstance(private_images_body, Mapping) else []
    )
    matching_images = [
        str(row.get("id"))
        for row in private_images or []
        if isinstance(row, Mapping) and str(row.get("name") or "").startswith(IMAGE_PREFIX)
    ]
    if droplets_http != 200:
        blockers.append("digitalocean_prebake_droplet_inventory_failed")
    elif matching_droplets:
        blockers.append("digitalocean_prebake_compute_overlap")
    if volumes_http != 200:
        blockers.append("digitalocean_prebake_volume_inventory_failed")
    elif matching_volumes:
        blockers.append("digitalocean_prebake_volume_overlap")
    if private_images_http != 200:
        blockers.append("digitalocean_prebake_image_inventory_failed")
    elif matching_images:
        blockers.append("digitalocean_prebake_image_overlap")
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "provider": "digitalocean",
        "name": name,
        "region": region,
        "size": size,
        "source_image": source_image,
        "price_hourly_usd": size_row.get("price_hourly"),
        "gpu_memory_mb": size_row.get("memory"),
        "source_image_id": source_row.get("id"),
        "campaign_resource_inventory": {
            "droplet_ids": matching_droplets,
            "volume_ids": matching_volumes,
            "image_ids": matching_images,
        },
        "api_confirmed": not any(
            value != 200
            for value in (
                account_http,
                sizes_http,
                images_http,
                droplets_http,
                volumes_http,
                private_images_http,
            )
        ),
        "raw_provider_response_recorded": False,
        "raw_secret_values_recorded": False,
    }


def _cloud_init(*, host_private_b64: str, host_public_b64: str, ttl_minutes: int) -> str:
    return f"""#cloud-config
ssh_deletekeys: false
bootcmd:
  - [bash, -c, "printf '%s' '{host_private_b64}' | base64 -d > /etc/ssh/ssh_host_ed25519_key && chmod 600 /etc/ssh/ssh_host_ed25519_key && printf '%s' '{host_public_b64}' | base64 -d > /etc/ssh/ssh_host_ed25519_key.pub && chmod 644 /etc/ssh/ssh_host_ed25519_key.pub"]
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
  - systemctl restart ssh
  - systemctl enable --now docker
  - install -d -m 700 /root/.blueprint-secrets /root/blueprint-prebake
  - touch /root/blueprint-prebake/host-ready
  - shutdown -h +{ttl_minutes}
"""


def _delete_and_verify(token: str, kind: str, resource_id: str) -> dict[str, Any]:
    delete_http, _ = _request(token=token, method="DELETE", path=f"/{kind}/{resource_id}")
    verify_http: int | None = None
    for _ in range(30):
        verify_http, _ = _request(token=token, method="GET", path=f"/{kind}/{resource_id}")
        if verify_http == 404:
            break
        time.sleep(2)
    return {
        "delete_http_status": delete_http,
        "verify_http_status": verify_http,
        "provider_absence_confirmed": verify_http == 404,
    }


def _exact_resource_ids(token: str, kind: str, name: str) -> tuple[bool, list[str]]:
    path = {
        "droplets": f"/droplets?tag_name={LANE}&per_page=200",
        "volumes": "/volumes?per_page=200",
        "images": "/images?private=true&per_page=200",
    }[kind]
    http_status, payload = _request(token=token, method="GET", path=path)
    rows = payload.get(kind) if http_status == 200 and isinstance(payload, Mapping) else []
    if http_status != 200 or not isinstance(rows, list):
        return False, []
    return True, [
        str(row.get("id"))
        for row in rows
        if isinstance(row, Mapping) and row.get("name") == name and row.get("id")
    ]


def _delete_exact_named(token: str, kind: str, name: str) -> dict[str, Any]:
    confirmed, resource_ids = _exact_resource_ids(token, kind, name)
    deletes = [
        _delete_and_verify(token, kind, resource_id) for resource_id in resource_ids
    ]
    final_confirmed, final_ids = _exact_resource_ids(token, kind, name)
    return {
        "initial_inventory_confirmed": confirmed,
        "matching_resource_ids": resource_ids,
        "delete_results": deletes,
        "final_inventory_confirmed": final_confirmed,
        "final_matching_resource_ids": final_ids,
        "provider_absence_confirmed": final_confirmed and not final_ids,
    }


def watchdog(*, state_path: Path, token_file: Path) -> int:
    state = _object(state_path)
    out = state_path.parent
    write_json(
        out / "watchdog_armed.json",
        {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": "armed",
            "pid": os.getpid(),
            "deadline_epoch": state.get("deadline_epoch"),
            "watchdog_nonce": state.get("watchdog_nonce"),
            "raw_secret_values_recorded": False,
        },
    )
    while time.time() < float(state.get("deadline_epoch") or 0):
        if (out / "watchdog_cancelled").is_file():
            return 0
        time.sleep(2)
        state = _object(state_path)
    token = _read_secret(token_file)
    cleanup: dict[str, Any] = {}
    for field, name_field, kind in (
        ("droplet_id", "droplet_name", "droplets"),
        ("snapshot_id", "snapshot_name", "images"),
        ("volume_id", "volume_name", "volumes"),
    ):
        resource_id = str(state.get(field) or "")
        if resource_id:
            rows = [
                _delete_and_verify(token, kind, resource_id)
            ]
            cleanup[field] = {
                "delete_results": rows,
                "provider_absence_confirmed": all(
                    row.get("provider_absence_confirmed") is True for row in rows
                ),
            }
        elif state.get(name_field):
            cleanup[field] = _delete_exact_named(
                token, kind, str(state.get(name_field) or "")
            )
    terminal = bool(cleanup) and all(
        row.get("provider_absence_confirmed") is True for row in cleanup.values()
    )
    write_json(
        out / "watchdog_result.json",
        {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": (
                "deadline_cleanup_complete"
                if terminal
                else "deadline_cleanup_unverified"
            ),
            "cleanup": cleanup,
            "raw_secret_values_recorded": False,
        },
    )
    return 0


def _presigned_download_manifest(
    *, access_key_file: Path, secret_key_file: Path, expected_manifest_digest: str
) -> tuple[Path, dict[str, Any]]:
    import boto3  # type: ignore[import-untyped]
    from botocore.config import Config  # type: ignore[import-untyped]

    access_key = _read_private_secret(access_key_file)
    secret_key = _read_private_secret(secret_key_file)
    client = boto3.client(
        "s3",
        endpoint_url=RUNPOD_S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="EUR-IS-1",
        config=Config(signature_version="s3v4"),
    )
    manifest_key = f"{RUNPOD_CACHE_PREFIX}/groot_oscar_model_cache_manifest.json"
    manifest_bytes = client.get_object(Bucket=RUNPOD_S3_BUCKET, Key=manifest_key)["Body"].read()
    manifest = json.loads(manifest_bytes)
    if manifest.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("prebake_source_manifest_digest_mismatch")
    rows: list[dict[str, Any]] = []
    for item in manifest.get("files") or []:
        relative = str(item.get("path") or "")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("prebake_source_manifest_path_invalid")
        key = f"{RUNPOD_CACHE_PREFIX}/{relative}"
        rows.append(
            {
                "path": relative,
                "size_bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
                "url": client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": RUNPOD_S3_BUCKET, "Key": key},
                    ExpiresIn=1800,
                ),
            }
        )
    rows.append(
        {
            "path": "groot_oscar_model_cache_manifest.json",
            "size_bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "url": client.generate_presigned_url(
                "get_object",
                Params={"Bucket": RUNPOD_S3_BUCKET, "Key": manifest_key},
                ExpiresIn=1800,
            ),
        }
    )
    fd, name = tempfile.mkstemp(prefix="blueprint-do-cache-download-", suffix=".json")
    path = Path(name)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle)
    os.chmod(path, 0o600)
    return path, {
        "file_count": int(manifest.get("file_count") or 0),
        "total_size_bytes": int(manifest.get("total_size_bytes") or 0),
        "model_manifest_digest": expected_manifest_digest,
    }


def _remote_prebake_script(
    *, image_ref: str, volume_name: str, expected: Mapping[str, Any]
) -> str:
    downloader = r'''import concurrent.futures, hashlib, json, os, pathlib, urllib.request
root=pathlib.Path('/models/blueprint-groot-oscar-v1')
root.mkdir(parents=True, exist_ok=True)
rows=json.load(open('/root/blueprint-prebake/downloads.json'))['rows']
def one(row):
    target=root / row['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    temp=target.with_name(target.name+'.partial')
    h=hashlib.sha256(); size=0
    with urllib.request.urlopen(row['url'], timeout=900) as response, open(temp,'wb') as out:
        while True:
            chunk=response.read(8*1024*1024)
            if not chunk: break
            out.write(chunk); h.update(chunk); size += len(chunk)
    if size != row['size_bytes'] or h.hexdigest() != row['sha256']:
        raise RuntimeError('download_integrity_mismatch:'+row['path'])
    os.replace(temp,target)
    return row['path']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(one, rows))
'''
    evidence = json.dumps(
        {
            "schema_version": "groot_oscar_digitalocean_prebake_remote.v1",
            "status": "passed",
            "release_image_ref": image_ref,
            "model_manifest_digest": expected["model_manifest_digest"],
            "verified_file_count": expected["file_count"],
            "verified_size_bytes": expected["total_size_bytes"],
            "cache_root": "/models/blueprint-groot-oscar-v1",
            "raw_secret_values_recorded": False,
        },
        sort_keys=True,
    )
    return "\n".join(
        [
            "set -euo pipefail",
            "trap 'rm -f /root/blueprint-prebake/downloads.json /root/.blueprint-secrets/docker_username /root/.blueprint-secrets/docker_pat; docker logout docker.io >/dev/null 2>&1 || true' EXIT",
            f"device=/dev/disk/by-id/scsi-0DO_Volume_{shlex.quote(volume_name)}",
            "test -e \"$device\"",
            "install -d -m 755 /models",
            "mountpoint -q /models || mount \"$device\" /models",
            "install -d -m 755 /models/blueprint-groot-oscar-v1 /etc/blueprint",
            "docker login docker.io --username \"$(cat /root/.blueprint-secrets/docker_username)\" --password-stdin < /root/.blueprint-secrets/docker_pat >/dev/null",
            f"docker pull {shlex.quote(image_ref)} >/root/blueprint-prebake/docker-pull.log 2>&1 & docker_pid=$!",
            "python3 - <<'PY' &\n" + downloader + "PY\ncache_pid=$!",
            "wait \"$docker_pid\"",
            "wait \"$cache_pid\"",
            f"docker image inspect {shlex.quote(image_ref)} >/dev/null",
            f"printf '%s\\n' {shlex.quote(image_ref)} > /etc/blueprint/worker-image-ref",
            f"printf '%s\\n' {shlex.quote(evidence)} > /root/blueprint-prebake/remote_evidence.json",
            "cp /root/blueprint-prebake/remote_evidence.json /models/blueprint-groot-oscar-v1/digitalocean_model_cache_verification.json",
            "sync",
        ]
    )


def _provider_inventory(provider: DigitalOceanRenderProvider, prefix: str) -> dict[str, Any]:
    result = provider.billable_inventory(name_prefix=prefix)
    return {
        "api_confirmed": result.get("api_confirmed") is True,
        "live_resource_count": result.get("live_resource_count"),
        "resources": result.get("resources") or [],
        "blockers": result.get("blockers") or [],
    }


def run_prebake(
    *,
    output_dir: Path,
    release_evidence_path: Path,
    model_cache_evidence_path: Path,
    token_file: Path,
    docker_username_file: Path,
    docker_password_file: Path,
    runpod_s3_access_key_file: Path,
    runpod_s3_secret_key_file: Path,
    login_private_key: Path,
    host_private_key: Path,
    ssh_key_id: int,
    region: str,
    size: str,
    source_image: str,
    volume_size_gib: int,
    reservation_seconds: int,
    future_gpu_seconds: int,
    campaign_budget_ledger: Path,
    initial_spent_usd: float,
    initial_gpu_seconds: int,
    total_spend_cap_usd: float,
    gpu_wall_cap_seconds: int,
    max_hourly_rate_usd: float,
    execute: bool,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    release = _object(release_evidence_path)
    cache = _object(model_cache_evidence_path)
    nonce = hashlib.sha256(
        f"{release.get('release_image_ref')}:{cache.get('model_manifest_digest')}:{time.time_ns()}".encode()
    ).hexdigest()[:12]
    name = RESOURCE_PREFIX + nonce
    volume_name = VOLUME_PREFIX + nonce
    image_name = IMAGE_PREFIX + nonce
    try:
        token = _read_secret(token_file)
        preflight = read_only_preflight(
            token=token, region=region, size=size, source_image=source_image, name=name
        )
    except Exception as exc:
        preflight = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["digitalocean_prebake_preflight_failed"],
            "error_type": type(exc).__name__,
        }
    observed_rate = preflight.get("price_hourly_usd")
    if isinstance(observed_rate, (int, float)) and observed_rate > max_hourly_rate_usd:
        preflight = dict(preflight)
        preflight["status"] = "blocked"
        preflight["blockers"] = [*(preflight.get("blockers") or []), "digitalocean_prebake_observed_rate_over_cap"]
    write_json(output / "digitalocean_prebake_preflight.json", preflight)
    admission = build_prebake_admission(
        release=release,
        model_cache=cache,
        preflight=preflight,
        volume_size_gib=volume_size_gib,
        reservation_seconds=reservation_seconds,
        future_gpu_seconds=future_gpu_seconds,
        initial_spent_usd=initial_spent_usd,
        initial_gpu_seconds=initial_gpu_seconds,
        total_spend_cap_usd=total_spend_cap_usd,
        gpu_wall_cap_seconds=gpu_wall_cap_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    write_json(output / "digitalocean_prebake_admission.json", admission)
    bound = {
        "schema_version": "groot_oscar_digitalocean_prebake_bound_request.v1",
        "provider": "digitalocean",
        "name": name,
        "volume_name": volume_name,
        "snapshot_name": image_name,
        "region": region,
        "size": size,
        "source_image": source_image,
        "volume_size_gib": volume_size_gib,
        "release_image_ref": admission.get("release_image_ref"),
        "model_manifest_digest": admission.get("model_manifest_digest"),
        "reservation_seconds": reservation_seconds,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "bound_provider_request.json", bound)
    if admission["status"] != "admitted" or not execute:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_ready" if admission["status"] == "admitted" else "blocked",
            "blockers": admission["blockers"],
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
        write_json(output / RESULT_FILENAME, result)
        return result

    reservation_id = name
    budget = ProductionGpuCampaignBudget(
        campaign_budget_ledger,
        initial_spent_usd=initial_spent_usd,
        initial_used_gpu_seconds=initial_gpu_seconds,
        total_spend_cap_usd=total_spend_cap_usd,
        combined_gpu_wall_cap_seconds=gpu_wall_cap_seconds,
    )
    try:
        reservation = budget.reserve(
            reservation_id=reservation_id,
            gpu_seconds=reservation_seconds,
            max_hourly_rate_usd=max_hourly_rate_usd,
        )
    except (CampaignBudgetExceeded, ValueError) as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [str(getattr(exc, "admission", {}).get("blocker") or exc)],
            "provider_mutations_performed": 0,
        }
        write_json(output / RESULT_FILENAME, result)
        return result
    write_json(output / "campaign_budget_reservation.json", reservation)

    provider = DigitalOceanRenderProvider()
    leases = PaidProviderLaneLeaseSet(
        providers={"digitalocean": provider},
        lane=LANE,
        job_dir=str(output),
        resource_name_prefix=RESOURCE_PREFIX,
    )
    lease = leases.acquire()
    write_json(output / "provider_lane_lease.json", lease)
    if lease.get("status") != "acquired":
        budget.settle(
            reservation_id=reservation_id,
            charged_gpu_seconds=0,
            charged_usd=0,
            outcome="provider_lane_lease_blocked_before_mutation",
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": lease.get("blockers") or ["provider_lane_lease_blocked"],
            "provider_mutations_performed": 0,
        }
        write_json(output / RESULT_FILENAME, result)
        return result

    pending = open_pending_teardown(
        provider="digitalocean",
        lane=LANE,
        run_id=nonce,
        resource_name=name,
        provider_location=region,
        job_dir=output,
        max_age_seconds=reservation_seconds,
    )
    write_json(output / "pending_teardown_opened.json", pending)
    started = time.time()
    state_path = output / "watchdog_state.json"
    state = {
        "deadline_epoch": started + reservation_seconds,
        "watchdog_nonce": os.urandom(16).hex(),
        "droplet_id": None,
        "volume_id": None,
        "snapshot_id": None,
        "droplet_name": name,
        "volume_name": volume_name,
        "snapshot_name": image_name,
        "replacement_cache_verified": False,
    }
    write_json(state_path, state)
    log = (output / "watchdog.log").open("ab")
    watch = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_digitalocean_prebaked_host",
            "watchdog",
            "--state",
            str(state_path),
            "--token-file",
            str(token_file),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    log.close()
    (output / "watchdog.pid").write_text(f"{watch.pid}\n", encoding="utf-8")
    armed_deadline = time.time() + 10
    while time.time() < armed_deadline and not (output / "watchdog_armed.json").is_file():
        if watch.poll() is not None:
            break
        time.sleep(0.05)
    if not (output / "watchdog_armed.json").is_file() or watch.poll() is not None:
        cancel_pending_teardown(pending["path"], reason="watchdog_not_armed_no_allocation")
        leases.release("watchdog_not_armed", provider_mutation_started=False)
        budget.settle(
            reservation_id=reservation_id,
            charged_gpu_seconds=0,
            charged_usd=0,
            outcome="watchdog_not_armed_before_mutation",
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["digitalocean_prebake_watchdog_not_armed"],
            "provider_mutations_performed": 0,
        }
        write_json(output / RESULT_FILENAME, result)
        return result

    droplet_id = ""
    volume_id = ""
    snapshot_id = ""
    snapshot_size_gib = 0.0
    provider_mutations = 0
    volume_create_attempted = False
    droplet_create_attempted = False
    snapshot_create_attempted = False
    allocation_started_at: float | None = None
    allocation_ended_at: float | None = None
    remote_verified = False
    teardown = {"provider_absence_confirmed": False}
    temp_download_manifest: Path | None = None
    error_type: str | None = None
    try:
        for secret_path in (
            docker_username_file,
            docker_password_file,
            login_private_key,
            runpod_s3_access_key_file,
            runpod_s3_secret_key_file,
        ):
            _read_private_secret(secret_path)
        host_private_b64, host_public_b64, _ = _host_key_material(host_private_key)
        volume_create_attempted = True
        volume_http, volume_body = _request(
            token=token,
            method="POST",
            path="/volumes",
            payload={
                "name": volume_name,
                "region": region,
                "size_gigabytes": volume_size_gib,
                "filesystem_type": "ext4",
                "description": "Bounded verified GR00T OSCAR model cache",
                "tags": [LANE],
            },
        )
        provider_mutations += 1
        volume = volume_body.get("volume") if isinstance(volume_body, Mapping) else {}
        volume_id = str((volume or {}).get("id") or "")
        if volume_http == 0 or volume_http >= 500:
            for _ in range(6):
                confirmed, matches = _exact_resource_ids(token, "volumes", volume_name)
                if confirmed and len(matches) == 1:
                    volume_id = matches[0]
                    break
                time.sleep(2)
        if not volume_id:
            raise RuntimeError("digitalocean_prebake_volume_create_failed")
        state["volume_id"] = volume_id
        write_json(state_path, state)
        user_data = _cloud_init(
            host_private_b64=host_private_b64,
            host_public_b64=host_public_b64,
            ttl_minutes=max(2, math.ceil(reservation_seconds / 60)),
        )
        droplet_create_attempted = True
        allocation_started_at = time.time()
        droplet_http, droplet_body = _request(
            token=token,
            method="POST",
            path="/droplets",
            payload={
                "name": name,
                "region": region,
                "size": size,
                "image": source_image,
                "ssh_keys": [ssh_key_id],
                "volume_ids": [volume_id],
                "backups": False,
                "ipv6": False,
                "monitoring": True,
                "tags": [LANE, "blueprint-isaac-render"],
                "user_data": user_data,
            },
        )
        provider_mutations += 1
        droplet = droplet_body.get("droplet") if isinstance(droplet_body, Mapping) else {}
        droplet_id = str((droplet or {}).get("id") or "")
        if droplet_http == 0 or droplet_http >= 500:
            for _ in range(6):
                confirmed, matches = _exact_resource_ids(token, "droplets", name)
                if confirmed and len(matches) == 1:
                    droplet_id = matches[0]
                    break
                time.sleep(2)
        if not droplet_id:
            raise RuntimeError("digitalocean_prebake_droplet_create_failed")
        bind_pending_teardown_instance(pending["path"], droplet_id)
        state["droplet_id"] = droplet_id
        write_json(state_path, state)
        public_ip = ""
        deadline = started + reservation_seconds
        while time.time() < deadline:
            status, body = _request(token=token, method="GET", path=f"/droplets/{droplet_id}")
            row = body.get("droplet") if status == 200 and isinstance(body, Mapping) else {}
            networks = ((row or {}).get("networks") or {}).get("v4") or []
            public = [
                item.get("ip_address")
                for item in networks
                if isinstance(item, Mapping) and item.get("type") == "public"
            ]
            if public:
                public_ip = str(public[0])
                break
            time.sleep(3)
        if not public_ip:
            raise RuntimeError("digitalocean_prebake_public_ip_timeout")
        public_key = Path(str(host_private_key.expanduser().resolve()) + ".pub").read_text(encoding="utf-8")
        known_hosts = output / "launch_bound_known_hosts"
        known_hosts.write_text(known_hosts_line(ip=public_ip, public_key_text=public_key), encoding="utf-8")
        options = _ssh_options(private_key=login_private_key.expanduser(), known_hosts=known_hosts)
        while time.time() < deadline:
            ready = subprocess.run(
                ["ssh", *options, f"root@{public_ip}", "test -f /root/blueprint-prebake/host-ready && nvidia-smi >/dev/null && docker info >/dev/null"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ready.returncode == 0:
                break
            time.sleep(5)
        else:
            raise RuntimeError("digitalocean_prebake_host_runtime_timeout")
        temp_download_manifest, expected = _presigned_download_manifest(
            access_key_file=runpod_s3_access_key_file,
            secret_key_file=runpod_s3_secret_key_file,
            expected_manifest_digest=str(cache.get("model_manifest_digest") or ""),
        )
        transfers = (
            (temp_download_manifest, "/root/blueprint-prebake/downloads.json"),
            (docker_username_file, "/root/.blueprint-secrets/docker_username"),
            (docker_password_file, "/root/.blueprint-secrets/docker_pat"),
        )
        for local, remote in transfers:
            subprocess.run(
                ["scp", *options, str(local.expanduser()), f"root@{public_ip}:{remote}"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        remote_script = _remote_prebake_script(
            image_ref=str(admission["release_image_ref"]),
            volume_name=volume_name,
            expected=expected,
        )
        with (output / "remote_prebake.log").open("wb") as remote_log:
            completed = subprocess.run(
                ["ssh", *options, f"root@{public_ip}", "bash", "-s"],
                input=(remote_script + "\n").encode(),
                stdout=remote_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError("digitalocean_prebake_remote_execution_failed")
        subprocess.run(
            [
                "scp",
                *options,
                f"root@{public_ip}:/root/blueprint-prebake/remote_evidence.json",
                str(output / "remote_prebake_evidence.json"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        remote_evidence = _object(output / "remote_prebake_evidence.json")
        remote_verified = bool(
            remote_evidence.get("status") == "passed"
            and remote_evidence.get("release_image_ref") == admission["release_image_ref"]
            and remote_evidence.get("model_manifest_digest") == admission["model_manifest_digest"]
            and remote_evidence.get("verified_file_count") == cache.get("verified_file_count")
            and remote_evidence.get("verified_size_bytes") == cache.get("verified_size_bytes")
        )
        if not remote_verified:
            raise RuntimeError("digitalocean_prebake_remote_evidence_invalid")
        subprocess.run(
            ["ssh", *options, f"root@{public_ip}", "sync; shutdown -h now"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        while time.time() < deadline:
            status, body = _request(token=token, method="GET", path=f"/droplets/{droplet_id}")
            row = body.get("droplet") if status == 200 and isinstance(body, Mapping) else {}
            if (row or {}).get("status") == "off":
                break
            time.sleep(3)
        else:
            raise RuntimeError("digitalocean_prebake_poweroff_timeout")
        snapshot_create_attempted = True
        action_http, action_body = _request(
            token=token,
            method="POST",
            path=f"/droplets/{droplet_id}/actions",
            payload={"type": "snapshot", "name": image_name},
        )
        provider_mutations += 1
        action = action_body.get("action") if isinstance(action_body, Mapping) else {}
        action_id = str((action or {}).get("id") or "")
        if action_http not in {200, 201, 202} or not action_id:
            raise RuntimeError("digitalocean_prebake_snapshot_start_failed")
        while time.time() < deadline:
            status, body = _request(token=token, method="GET", path=f"/actions/{action_id}")
            row = body.get("action") if status == 200 and isinstance(body, Mapping) else {}
            if (row or {}).get("status") == "completed":
                break
            if (row or {}).get("status") == "errored":
                raise RuntimeError("digitalocean_prebake_snapshot_failed")
            time.sleep(5)
        else:
            raise RuntimeError("digitalocean_prebake_snapshot_timeout")
        images_http, images_body = _request(
            token=token, method="GET", path="/images?private=true&per_page=200"
        )
        images = images_body.get("images") if images_http == 200 and isinstance(images_body, Mapping) else []
        snapshot = next(
            (row for row in images or [] if isinstance(row, Mapping) and row.get("name") == image_name),
            {},
        )
        snapshot_id = str(snapshot.get("id") or "")
        if not snapshot_id or snapshot.get("status") != "available" or region not in (snapshot.get("regions") or []):
            raise RuntimeError("digitalocean_prebake_snapshot_verification_failed")
        snapshot_size_gib = float(snapshot.get("size_gigabytes") or 0)
        if not 0 < snapshot_size_gib <= MAX_SNAPSHOT_GIB:
            raise RuntimeError("digitalocean_prebake_snapshot_size_outside_bound")
        state["snapshot_id"] = snapshot_id
        write_json(state_path, state)
    except Exception as exc:
        error_type = type(exc).__name__
        write_json(
            output / "digitalocean_prebake_error.json",
            {"error_type": error_type, "error": str(exc)[:500], "raw_secret_values_recorded": False},
        )
    finally:
        if temp_download_manifest is not None:
            temp_download_manifest.unlink(missing_ok=True)
        if droplet_id:
            teardown = _delete_and_verify(token, "droplets", droplet_id)
            if teardown.get("provider_absence_confirmed") is True:
                allocation_ended_at = time.time()
            provider_mutations += 1
        elif droplet_create_attempted:
            teardown = _delete_exact_named(token, "droplets", name)
            recovered_value = teardown.get("matching_resource_ids")
            recovered_ids: list[Any] = (
                recovered_value if isinstance(recovered_value, list) else []
            )
            if len(recovered_ids) == 1:
                droplet_id = str(recovered_ids[0])
                bind_pending_teardown_instance(pending["path"], droplet_id)
            if recovered_ids and teardown.get("provider_absence_confirmed") is True:
                allocation_ended_at = time.time()
        else:
            teardown = {"provider_absence_confirmed": True, "no_droplet_id": True}
        write_json(output / "teardown.json", teardown)

    success = bool(remote_verified and snapshot_id and teardown.get("provider_absence_confirmed"))
    if success:
        volume_http, volume_body = _request(token=token, method="GET", path=f"/volumes/{volume_id}")
        volume = volume_body.get("volume") if volume_http == 200 and isinstance(volume_body, Mapping) else {}
        success = bool(
            (volume or {}).get("id") == volume_id
            and (volume or {}).get("size_gigabytes") == volume_size_gib
            and not ((volume or {}).get("droplet_ids") or [])
        )
    retention_deadline_epoch: float | None = None
    maximum_storage_spend = (
        (
            volume_size_gib * DIGITALOCEAN_VOLUME_USD_PER_GIB_MONTH
            + snapshot_size_gib * DIGITALOCEAN_SNAPSHOT_USD_PER_GIB_MONTH
        )
        / HOURS_PER_BILLING_MONTH
        * (RETENTION_TTL_SECONDS / 3600.0)
    )
    if success:
        retention_deadline_epoch = time.time() + RETENTION_TTL_SECONDS
        state["replacement_cache_verified"] = True
        state["retention_mode"] = "bounded_prebaked_host_and_model_cache"
        state["deadline_epoch"] = retention_deadline_epoch
        state["maximum_retained_storage_spend_usd"] = round(
            maximum_storage_spend, 6
        )
        write_json(state_path, state)
        storage_terminal = watch.poll() is None
        write_json(
            output / "bounded_digitalocean_retention.json",
            {
                "schema_version": "groot_oscar_digitalocean_bounded_retention.v1",
                "status": "retained" if storage_terminal else "blocked",
                "snapshot_id": snapshot_id,
                "snapshot_size_gib": snapshot_size_gib,
                "model_volume_id": volume_id,
                "model_volume_size_gib": volume_size_gib,
                "retention_deadline_epoch": retention_deadline_epoch,
                "retention_ttl_seconds": RETENTION_TTL_SECONDS,
                "maximum_retained_storage_spend_usd": round(
                    maximum_storage_spend, 6
                ),
                "automatic_delete_at_deadline": True,
                "watchdog_pid": watch.pid,
                "watchdog_alive": storage_terminal,
                "source_runpod_cache_preserved": True,
                "raw_secret_values_recorded": False,
            },
        )
        if not storage_terminal:
            success = False
            error_type = "digitalocean_prebake_retention_watchdog_not_alive"
    if not success:
        storage_cleanup: dict[str, Any] = {}
        if snapshot_id:
            storage_cleanup["snapshot"] = _delete_and_verify(token, "images", snapshot_id)
        elif snapshot_create_attempted:
            storage_cleanup["snapshot"] = _delete_exact_named(
                token, "images", image_name
            )
        if volume_id:
            storage_cleanup["volume"] = _delete_and_verify(token, "volumes", volume_id)
        elif volume_create_attempted:
            storage_cleanup["volume"] = _delete_exact_named(
                token, "volumes", volume_name
            )
        write_json(output / "failed_retained_resource_cleanup.json", storage_cleanup)
        storage_terminal = all(
            row.get("provider_absence_confirmed") is True
            for row in storage_cleanup.values()
        )
    if (
        not success
        and teardown.get("provider_absence_confirmed") is True
        and storage_terminal
    ):
        (output / "watchdog_cancelled").touch()
        try:
            watch.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    compute_observed = bool(droplet_id)
    teardown_confirmed = teardown.get("provider_absence_confirmed") is True
    if compute_observed and not teardown_confirmed:
        # The exact billing stop is unknown. Consume the full bounded
        # reservation so a later stage cannot spend against uncertain time.
        elapsed_seconds = reservation_seconds
        charged_rate = max_hourly_rate_usd
    else:
        elapsed_seconds = min(
            reservation_seconds,
            max(
                0,
                math.ceil(
                    (allocation_ended_at or time.time())
                    - (allocation_started_at or (allocation_ended_at or time.time()))
                ),
            ),
        )
        charged_rate = float(observed_rate or max_hourly_rate_usd)
    charged_usd = round(charged_rate * elapsed_seconds / 3600.0, 6)
    settlement = budget.settle(
        reservation_id=reservation_id,
        charged_gpu_seconds=elapsed_seconds if compute_observed else 0,
        charged_usd=charged_usd if compute_observed else 0,
        outcome=(
            "prebaked_host_verified_compute_terminal"
            if success
            else (
                "prebake_teardown_unverified_full_reservation"
                if compute_observed and not teardown_confirmed
                else "prebake_failed_compute_terminal"
            )
        ),
    )
    write_json(output / "campaign_budget_settlement.json", settlement)
    proof = {
        "status": "PASS" if teardown.get("provider_absence_confirmed") else "FAIL",
        "provider_absence_confirmed": teardown.get("provider_absence_confirmed") is True,
        "retained_resource_state_terminal": storage_terminal,
        "watchdog_process_terminal": watch.poll() is not None,
        "provider": "digitalocean",
        "instance_id": droplet_id or None,
    }
    if droplet_id:
        close_pending_teardown(pending["path"], proof)
    elif teardown_confirmed:
        cancel_pending_teardown(
            pending["path"],
            reason="provider_inventory_verified_no_matching_compute",
            evidence=teardown,
        )
    else:
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason="digitalocean_create_outcome_and_teardown_unverified",
            evidence=teardown,
        )
    release_summary = leases.release(
        "prebake_terminal",
        provider_mutation_started=bool(
            volume_create_attempted
            or droplet_create_attempted
            or snapshot_create_attempted
        ),
    )
    write_json(output / "provider_lane_release.json", release_summary)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if success else "failed",
        "blockers": [] if success else [str(error_type or "digitalocean_prebake_verification_failed")],
        "provider": "digitalocean",
        "droplet_id": droplet_id or None,
        "provider_absence_confirmed": teardown.get("provider_absence_confirmed") is True,
        "snapshot_id": snapshot_id or None,
        "snapshot_name": image_name if snapshot_id else None,
        "model_volume_id": volume_id if success else None,
        "model_volume_name": volume_name if success else None,
        "release_image_ref": admission.get("release_image_ref"),
        "model_manifest_digest": admission.get("model_manifest_digest"),
        "replacement_cache_verified": success,
        "source_runpod_cache_deleted": False,
        "retention_watchdog_pid": watch.pid if success else None,
        "retention_deadline_epoch": retention_deadline_epoch if success else None,
        "maximum_retained_storage_spend_usd": (
            round(maximum_storage_spend, 6) if success else 0
        ),
        "elapsed_gpu_seconds": elapsed_seconds if droplet_id else 0,
        "measured_gpu_spend_usd": charged_usd if droplet_id else 0,
        "provider_mutations_performed": provider_mutations,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "prebaked_host_is_not_runtime_startup": True,
            "verified_model_cache_is_not_policy_execution": True,
            "provider_teardown_is_not_semantic_task_success": True,
        },
    }
    write_json(output / RESULT_FILENAME, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watchdog")
    watch.add_argument("--state", required=True)
    watch.add_argument("--token-file", required=True)
    args = parser.parse_args(argv)
    if args.command == "watchdog":
        return watchdog(state_path=Path(args.state), token_file=Path(args.token_file))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
