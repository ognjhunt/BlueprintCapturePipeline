"""Allocator-only control plane for one retained Cosmos3 Vast session."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .gpu_render_providers import (
    _latest_redacted_vast_ssh_output_fields,
    _validated_vast_known_hosts_pin,
    _vast_ssh_endpoint,
    enroll_vast_ssh_host_key,
    get_render_provider,
)
from .provider_bundle_staging_common import (
    BUNDLE_ROUTE,
    OUTPUT_ROUTE,
    staging_url_with_token,
)
from .retained_gpu_session_lifecycle import record_retained_gpu_state


SCHEMA_VERSION = "policy_ranking_successor_retained_session.v1"
REFRESH_SCHEMA_VERSION = "policy_ranking_successor_refresh_request.v1"
SESSION_NAME = "policy_ranking_successor_retained_session.json"
REMOTE_CONTROL = "/workspace/blueprint_vast_probe/cosmos3_retained/successor_retained_control.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_private_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_mode & 0o077:
        raise ValueError("successor_retained_session_manifest_missing_or_insecure")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("successor_retained_session_manifest_invalid")
    return resolved, value


def create_retained_session_manifest(
    *,
    job_dir: str | Path,
    adapter_result: Mapping[str, Any],
    watchdog_handoff: Mapping[str, Any],
    source_commit: str,
    dirty_state_declaration: str,
    bundle_sha256: str,
    authorization_receipt_sha256: str,
    image_digest: str,
    checkpoint: str,
    checkpoint_revision: str,
) -> dict[str, Any]:
    root = Path(job_dir).expanduser().resolve()
    ids = list(adapter_result.get("vast_instance_ids") or [])
    if adapter_result.get("retained_owned") is not True or len(ids) != 1:
        raise ValueError("successor_retained_adapter_ownership_not_proven")
    instance_id = str(ids[0])
    provider = get_render_provider("vast")
    inspection = provider.inspect(instance_id)
    if inspection.get("status") != "observed" or inspection.get("direct_port_ready") is not True:
        raise ValueError("successor_retained_provider_binding_not_observed")
    connection = dict(inspection.get("ssh_connection") or {})
    host_key = enroll_vast_ssh_host_key(connection, attempt_dir=root / "retained_ssh")
    if host_key.get("status") != "enrolled" or host_key.get("tofu_pinned") is not True:
        raise ValueError("successor_retained_ssh_host_key_not_enrolled")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "retained_owned",
        "created_at": utc_now_iso(),
        "provider": "vast",
        "provider_instance_id": instance_id,
        "source_commit": source_commit,
        "dirty_state_declaration": dirty_state_declaration,
        "current_runtime_bundle_sha256": bundle_sha256,
        "authorization_receipt_sha256": authorization_receipt_sha256,
        "image_digest": image_digest,
        "checkpoint": checkpoint,
        "checkpoint_revision": checkpoint_revision,
        "watchdog_pid": watchdog_handoff.get("watchdog_pid"),
        "watchdog_deadline_epoch": watchdog_handoff.get("watchdog_deadline_epoch"),
        "ssh_connection": connection,
        "known_hosts_file": host_key["known_hosts_file"],
        "known_hosts_sha256": host_key["known_hosts_sha256"],
        "refresh_count": 0,
        "last_refresh": None,
        "continuing_spend": True,
        "signed_urls_stored": False,
    }
    path = root / SESSION_NAME
    write_json(path, manifest)
    os.chmod(path, 0o600)
    return {**manifest, "session_manifest": str(path)}


def _run_remote_refresh(
    *,
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
    identity_file: str | Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    connection = dict(manifest.get("ssh_connection") or {})
    endpoint = _vast_ssh_endpoint(connection)
    if endpoint is None:
        return {"status": "blocked", "blockers": ["successor_retained_ssh_endpoint_invalid"]}
    host, port = endpoint
    known_hosts = str(manifest.get("known_hosts_file") or "")
    pin = _validated_vast_known_hosts_pin(known_hosts, host=host, port=port)
    identity = Path(identity_file).expanduser()
    if (
        pin is None
        or identity.is_symlink()
        or not identity.is_file()
        or identity.stat().st_mode & 0o077
    ):
        return {
            "status": "blocked",
            "blockers": ["successor_retained_ssh_identity_or_host_pin_invalid"],
        }
    command = [
        "ssh",
        "-i",
        str(identity.resolve(strict=True)),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={pin[0]}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=8",
        "--",
        f"root@{host}",
        "python",
        REMOTE_CONTROL,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            input=(json.dumps(request, sort_keys=True) + "\n").encode("utf-8"),
            capture_output=True,
            timeout=max(60.0, min(float(timeout_seconds), 14_400.0)),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "blocked",
            "blockers": ["successor_retained_refresh_ssh_timeout"],
            **_latest_redacted_vast_ssh_output_fields(stdout=exc.stdout, stderr=exc.stderr),
        }
    output = _latest_redacted_vast_ssh_output_fields(
        stdout=completed.stdout, stderr=completed.stderr
    )
    parsed: dict[str, Any] = {}
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines()[::-1]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    return {
        "status": "completed"
        if completed.returncode == 0 and parsed.get("status") == "completed"
        else "blocked",
        "blockers": []
        if completed.returncode == 0 and parsed.get("status") == "completed"
        else ["successor_retained_refresh_remote_failed"],
        "returncode": completed.returncode,
        "remote_result": parsed,
        **output,
    }


def refresh_retained_session(
    *,
    session_manifest: str | Path,
    bundle_path: str | Path,
    public_base_url: str,
    token_file: str | Path,
    source_commit: str,
    dirty_state_declaration: str,
    authorization_receipt_sha256: str,
    identity_file: str | Path = "~/.ssh/id_ed25519",
    timeout_seconds: float = 14_400.0,
) -> dict[str, Any]:
    manifest_path, manifest = _load_private_manifest(session_manifest)
    if manifest.get("status") != "retained_owned" or manifest.get("continuing_spend") is not True:
        raise ValueError("successor_retained_session_not_owned")
    if time.time() >= float(manifest.get("watchdog_deadline_epoch") or 0):
        raise ValueError("successor_retained_session_watchdog_expired")
    if (
        source_commit != manifest.get("source_commit")
        and dirty_state_declaration != "declared_dirty_overlay"
    ):
        raise ValueError("successor_retained_refresh_source_binding_invalid")
    bundle = Path(bundle_path).expanduser().resolve(strict=True)
    new_bundle_sha256 = _sha256_file(bundle)
    if not re.fullmatch(r"[0-9a-f]{64}", authorization_receipt_sha256):
        raise ValueError("successor_retained_refresh_authorization_receipt_invalid")
    token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    request = {
        "schema_version": REFRESH_SCHEMA_VERSION,
        "action": "refresh",
        "bundle_url": staging_url_with_token(public_base_url, BUNDLE_ROUTE, token),
        "output_put_url": staging_url_with_token(public_base_url, OUTPUT_ROUTE, token),
        "source_commit": source_commit,
        "dirty_state_declaration": dirty_state_declaration,
        "runtime_bundle_sha256": new_bundle_sha256,
        "authorization_receipt_sha256": authorization_receipt_sha256,
        "image_digest": manifest["image_digest"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_revision": manifest["checkpoint_revision"],
        "provider_instance_id": manifest["provider_instance_id"],
        "previous_bundle_sha256": manifest["current_runtime_bundle_sha256"],
    }
    record_retained_gpu_state(
        manifest_path.parent,
        "refresh_in_progress",
        evidence={key: value for key, value in request.items() if not key.endswith("_url")},
    )
    remote = _run_remote_refresh(
        manifest=manifest,
        request=request,
        identity_file=identity_file,
        timeout_seconds=timeout_seconds,
    )
    remote_result = dict(remote.get("remote_result") or {})
    completed = remote.get("status") == "completed"
    retained_server = remote_result.get("server_remained_loaded") is True
    next_state = "experiment_running" if completed else "retained_owned"
    record_retained_gpu_state(
        manifest_path.parent,
        next_state,
        evidence={
            "previous_bundle_sha256": request["previous_bundle_sha256"],
            "new_bundle_sha256": new_bundle_sha256,
            "server_remained_loaded": retained_server,
            "remote_status": remote.get("status"),
        },
    )
    provider_teardown: dict[str, Any] = {"status": "not_requested"}
    provider_zero: dict[str, Any] = {"status": "not_verified"}
    if completed:
        terminal_evidence = {
            "new_bundle_sha256": new_bundle_sha256,
            "remote_audit_sha256": remote_result.get("audit_sha256"),
            "server_remained_loaded": retained_server,
        }
        record_retained_gpu_state(
            manifest_path.parent, "terminal_success", evidence=terminal_evidence
        )
        record_retained_gpu_state(
            manifest_path.parent, "teardown_requested", evidence=terminal_evidence
        )
        provider = get_render_provider("vast")
        provider_teardown = provider.terminate(str(manifest["provider_instance_id"]))
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            provider_zero = provider.inspect(str(manifest["provider_instance_id"]))
            if (
                provider_zero.get("status") == "absent"
                and provider_zero.get("provider_absence_confirmed") is True
            ):
                break
            time.sleep(1.0)
        if provider_zero.get("provider_absence_confirmed") is not True:
            raise RuntimeError("successor_retained_terminal_teardown_not_proven")
        record_retained_gpu_state(
            manifest_path.parent,
            "provider_absent",
            evidence={
                "provider_instance_id": manifest["provider_instance_id"],
                "provider_absence_confirmed": True,
            },
        )
    manifest["status"] = "provider_absent" if completed else "retained_owned"
    manifest["current_runtime_bundle_sha256"] = (
        new_bundle_sha256 if completed else manifest["current_runtime_bundle_sha256"]
    )
    manifest["refresh_count"] = int(manifest.get("refresh_count") or 0) + 1
    manifest["last_refresh"] = {
        "refresh_time": utc_now_iso(),
        "previous_bundle_sha256": request["previous_bundle_sha256"],
        "new_bundle_sha256": new_bundle_sha256,
        "authorization_receipt_sha256": authorization_receipt_sha256,
        "source_commit": source_commit,
        "dirty_state_declaration": dirty_state_declaration,
        "remote_status": remote.get("status"),
        "audit_sha256": remote_result.get("audit_sha256"),
        "server_remained_loaded": retained_server,
        "signed_urls_stored": False,
    }
    manifest["continuing_spend"] = not completed
    write_json(manifest_path, manifest)
    os.chmod(manifest_path, 0o600)
    result = {
        "schema_version": "policy_ranking_successor_retained_refresh_result.v1",
        "status": manifest["status"],
        "provider_instance_id": manifest["provider_instance_id"],
        "continuing_spend": not completed,
        "refresh": manifest["last_refresh"],
        "remote": remote,
        "provider_teardown": provider_teardown,
        "provider_zero": provider_zero,
        "signed_urls_stored": False,
        "blockers": remote.get("blockers") or [],
    }
    return result


__all__ = [
    "SCHEMA_VERSION",
    "SESSION_NAME",
    "create_retained_session_manifest",
    "refresh_retained_session",
]
