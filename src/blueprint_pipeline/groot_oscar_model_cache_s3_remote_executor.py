"""Closed remote handler for an admitted CPU model-cache S3 packet.

There is deliberately no CLI. The canonical DigitalOcean adapter invokes the
fixed packet entrypoint after local admission and binds it to a one-time secret,
the verified archive digest, droplet identity, launch-bound host key, retained
volume watchdog, and exact result paths.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import re
import stat
import sys
import time
import urllib.request
import base64
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, write_json
from .groot_oscar_model_cache import (
    COSMOS_MODEL_RELATIVE_ROOT,
    VERIFICATION_SCHEMA_VERSION,
    prepare_model_cache,
    verify_model_cache,
)
from .groot_oscar_runpod_s3_model_cache import (
    DEFAULT_REMOTE_PREFIX,
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
    _issue_transport_execution_capability,
    _upload_and_verify_model_cache_impl,
)


PACKET_SCHEMA_VERSION = "groot_oscar_model_cache_s3_remote_packet.v1"
EXECUTION_SCHEMA_VERSION = "groot_oscar_model_cache_s3_remote_execution.v1"
PARENT_BINDING_SCHEMA_VERSION = "groot_oscar_model_cache_s3_parent_binding.v1"
RUNTIME_CACHE_ROOT = Path("/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1")
RUNTIME_COSMOS_MODEL_ROOT = RUNTIME_CACHE_ROOT / COSMOS_MODEL_RELATIVE_ROOT
VERIFICATION_ROOT = Path("/workspace/.blueprint-model-cache-verification")
REMOTE_PACKET_ROOT = Path("/root/blueprint-build/run/groot_oscar_model_cache_s3_remote")
PACKET_PATH = REMOTE_PACKET_ROOT / "packet.json"
CONTEXT_ROOT = REMOTE_PACKET_ROOT / "context"
CONTEXT_MANIFEST_PATH = REMOTE_PACKET_ROOT / "context_manifest.json"
DEPENDENCY_MANIFEST_PATH = REMOTE_PACKET_ROOT / "dependency_manifest.json"
DEPENDENCY_LOCK_PATH = REMOTE_PACKET_ROOT / "uv.lock"
REQUIREMENTS_CLOSURE_PATH = REMOTE_PACKET_ROOT / "requirements_closure.json"
OUTPUT_DIR = REMOTE_PACKET_ROOT / "results"
PARENT_BINDING_PATH = Path("/root/blueprint-build/model_cache_parent_binding.json")
PARENT_CAPABILITY_PATH = Path("/root/.blueprint-secrets/model_cache_parent_capability")
CONSUMED_CAPABILITY_PATH = Path(
    "/root/.blueprint-secrets/model_cache_parent_capability.consumed"
)
EXECUTION_LOCK_PATH = Path("/root/blueprint-build/model_cache_execution.lock")
HF_TOKEN_PATH = Path("/root/.blueprint-secrets/hf_token")
S3_ACCESS_KEY_PATH = Path("/root/.blueprint-secrets/runpod_s3_access_key")
S3_SECRET_KEY_PATH = Path("/root/.blueprint-secrets/runpod_s3_secret_key")
VENV_SITE_PACKAGES = Path(
    "/root/blueprint-build/model-cache-venv/lib/python3.12/site-packages"
)
DEPENDENCY_RESULT_PATH = Path(
    "/root/blueprint-build/model_cache_dependency_verification.json"
)
TRANSPORT_RESULT_NAME = "runpod_s3_model_cache_transport_result.json"
CANARY_VERIFICATION_NAME = "external_model_cache_verification.json"
EXECUTION_RESULT_NAME = "model_cache_s3_remote_execution_result.json"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAFE_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}")
_SAFE_DROPLET_ID = re.compile(r"[0-9]{1,32}")
_PACKET_KEYS = frozenset(
    {
        "schema_version",
        "packet_kind",
        "source_commit",
        "source_patch_sha256",
        "runtime_cache_root",
        "verification_root",
        "remote_prefix",
        "data_center_id",
        "allocation_nonce",
        "volume_evidence",
        "volume_watchdog_handoff",
        "context_manifest_sha256",
        "dependency_manifest_sha256",
        "dependency_lock_sha256",
        "requirements_closure_sha256",
        "result_files",
        "raw_secret_values_recorded",
    }
)
_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "packet_kind",
        "tarball_sha256",
        "capability_sha256",
        "droplet_id",
        "name",
        "region",
        "ssh_host_key_sha256",
        "provider_volume_id",
        "allocation_nonce",
        "builder_deadline_epoch",
        "volume_watchdog_deadline_epoch",
        "archive_members",
        "binding_hmac_sha256",
        "raw_secret_values_recorded",
    }
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("typed_model_cache_json_not_object")
    return payload


def _normalize_distribution_name(value: object) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip().lower())


def _verified_installed_dependencies(
    dependency_manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Prove the fixed venv contains exactly the locked application closure."""

    rows = dependency_manifest.get("requirements")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("typed_model_cache_dependency_requirements_invalid")
    expected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"name", "version"}:
            raise RuntimeError("typed_model_cache_dependency_requirements_invalid")
        name = _normalize_distribution_name(row.get("name"))
        version = str(row.get("version") or "")
        if not name or not version or name in expected:
            raise RuntimeError("typed_model_cache_dependency_requirements_invalid")
        expected[name] = version
    installed = {
        _normalize_distribution_name(distribution.metadata.get("Name")): distribution.version
        for distribution in importlib.metadata.distributions(
            path=[str(VENV_SITE_PACKAGES)]
        )
        if distribution.metadata.get("Name")
    }
    bootstrap = {"pip", "setuptools", "wheel"}
    application_installed = {
        name: version for name, version in installed.items() if name not in bootstrap
    }
    if application_installed != expected:
        raise RuntimeError("typed_model_cache_dependency_inventory_mismatch")
    return dict(sorted(application_installed.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _secret(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_mode & 0o077:
        raise ValueError("typed_model_cache_secret_file_missing_or_not_private")
    value = resolved.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("typed_model_cache_secret_file_empty")
    return value


def _context_blockers(packet: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if _sha256(CONTEXT_MANIFEST_PATH) != packet.get("context_manifest_sha256"):
        blockers.append("typed_model_cache_context_manifest_digest_mismatch")
        return blockers
    if _sha256(DEPENDENCY_MANIFEST_PATH) != packet.get("dependency_manifest_sha256"):
        blockers.append("typed_model_cache_dependency_manifest_digest_mismatch")
    if _sha256(DEPENDENCY_LOCK_PATH) != packet.get("dependency_lock_sha256"):
        blockers.append("typed_model_cache_dependency_lock_digest_mismatch")
    if _sha256(REQUIREMENTS_CLOSURE_PATH) != packet.get(
        "requirements_closure_sha256"
    ):
        blockers.append("typed_model_cache_dependency_closure_digest_mismatch")
    manifest = _load_object(CONTEXT_MANIFEST_PATH)
    rows = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    expected = {
        str(row.get("path")): str(row.get("sha256"))
        for row in rows
        if isinstance(row, Mapping)
    }
    actual: dict[str, str] = {}
    for path in CONTEXT_ROOT.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            blockers.append("typed_model_cache_context_nonregular_file")
            continue
        actual[path.relative_to(REMOTE_PACKET_ROOT).as_posix()] = _sha256(path)
    if actual != expected:
        blockers.append("typed_model_cache_context_inventory_mismatch")
    return blockers


def _live_host_identity() -> dict[str, str]:
    def metadata(name: str) -> str:
        url = f"http://169.254.169.254/metadata/v1/{name}"
        with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
            return response.read(4096).decode("utf-8").strip()

    public = Path("/etc/ssh/ssh_host_ed25519_key.pub").read_text(
        encoding="utf-8"
    ).split()
    if len(public) < 2 or public[0] != "ssh-ed25519":
        raise ValueError("typed_model_cache_live_host_key_invalid")
    key_blob = base64.b64decode(public[1], validate=True)
    fingerprint = base64.b64encode(hashlib.sha256(key_blob).digest()).decode().rstrip("=")
    return {
        "droplet_id": metadata("id"),
        "name": metadata("hostname"),
        "region": metadata("region"),
        "ssh_host_key_sha256": "SHA256:" + fingerprint,
    }


def _contract_blockers(
    packet: Mapping[str, Any], binding: Mapping[str, Any], capability: bytes
) -> list[str]:
    blockers: list[str] = []
    volume = packet.get("volume_evidence")
    volume = volume if isinstance(volume, Mapping) else {}
    handoff = packet.get("volume_watchdog_handoff")
    handoff = handoff if isinstance(handoff, Mapping) else {}
    if set(packet) != _PACKET_KEYS:
        blockers.append("typed_model_cache_packet_fields_invalid")
    if set(binding) != _BINDING_KEYS:
        blockers.append("typed_model_cache_parent_binding_fields_invalid")
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        blockers.append("typed_model_cache_packet_schema_invalid")
    if packet.get("packet_kind") != "model_cache_s3":
        blockers.append("typed_model_cache_packet_kind_invalid")
    if binding.get("schema_version") != PARENT_BINDING_SCHEMA_VERSION:
        blockers.append("typed_model_cache_parent_binding_schema_invalid")
    if binding.get("packet_kind") != "model_cache_s3":
        blockers.append("typed_model_cache_parent_binding_kind_invalid")
    if packet.get("raw_secret_values_recorded") is not False:
        blockers.append("typed_model_cache_packet_secret_contract_invalid")
    if binding.get("raw_secret_values_recorded") is not False:
        blockers.append("typed_model_cache_parent_binding_secret_contract_invalid")
    if len(capability) < 32:
        blockers.append("typed_model_cache_parent_capability_too_short")
    if _HEX40.fullmatch(str(packet.get("source_commit") or "")) is None:
        blockers.append("typed_model_cache_source_commit_invalid")
    if packet.get("source_patch_sha256") != hashlib.sha256(b"").hexdigest():
        blockers.append("typed_model_cache_source_patch_not_clean")
    if _HEX64.fullmatch(str(binding.get("tarball_sha256") or "")) is None:
        blockers.append("typed_model_cache_tarball_digest_invalid")
    if hashlib.sha256(capability).hexdigest() != binding.get("capability_sha256"):
        blockers.append("typed_model_cache_parent_capability_invalid")
    signed_binding = {
        key: value for key, value in binding.items() if key != "binding_hmac_sha256"
    }
    expected_hmac = hmac.new(
        capability,
        json.dumps(signed_binding, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(
        expected_hmac, str(binding.get("binding_hmac_sha256") or "")
    ):
        blockers.append("typed_model_cache_parent_binding_hmac_invalid")
    if _SAFE_DROPLET_ID.fullmatch(str(binding.get("droplet_id") or "")) is None:
        blockers.append("typed_model_cache_parent_droplet_id_invalid")
    if not str(binding.get("ssh_host_key_sha256") or "").startswith("SHA256:"):
        blockers.append("typed_model_cache_parent_host_key_invalid")
    try:
        live_host = _live_host_identity()
    except Exception:  # noqa: BLE001 - live identity must fail closed
        blockers.append("typed_model_cache_live_host_identity_unavailable")
    else:
        for field in ("droplet_id", "name", "region", "ssh_host_key_sha256"):
            if live_host[field] != str(binding.get(field) or ""):
                blockers.append(f"typed_model_cache_live_host_{field}_mismatch")
    nonce = str(packet.get("allocation_nonce") or "")
    if _SAFE_NONCE.fullmatch(nonce) is None:
        blockers.append("typed_model_cache_allocation_nonce_invalid")
    data_center = str(packet.get("data_center_id") or "")
    if data_center not in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
        blockers.append("typed_model_cache_data_center_invalid")
    if packet.get("runtime_cache_root") != str(RUNTIME_CACHE_ROOT):
        blockers.append("typed_model_cache_runtime_root_invalid")
    if packet.get("verification_root") != str(VERIFICATION_ROOT):
        blockers.append("typed_model_cache_verification_root_invalid")
    prefix = str(packet.get("remote_prefix") or "")
    if prefix != DEFAULT_REMOTE_PREFIX or f"/workspace/{prefix}" != str(
        RUNTIME_CACHE_ROOT
    ):
        blockers.append("typed_model_cache_runtime_prefix_mapping_invalid")
    if packet.get("result_files") != [
        TRANSPORT_RESULT_NAME,
        CANARY_VERIFICATION_NAME,
        EXECUTION_RESULT_NAME,
    ]:
        blockers.append("typed_model_cache_result_contract_invalid")
    if volume.get("schema_version") != "groot_oscar_runpod_network_volume_evidence.v1":
        blockers.append("typed_model_cache_volume_schema_invalid")
    if volume.get("status") != "verified" or volume.get("provider_api_verified") is not True:
        blockers.append("typed_model_cache_volume_not_verified")
    if (
        volume.get("id") != binding.get("provider_volume_id")
        or volume.get("data_center_id") != data_center
        or volume.get("allocation_nonce") != nonce
        or volume.get("allocation_name_verified") is not True
        or nonce not in str(volume.get("name") or "")
    ):
        blockers.append("typed_model_cache_volume_identity_mismatch")
    if binding.get("allocation_nonce") != nonce:
        blockers.append("typed_model_cache_parent_nonce_mismatch")
    deadline_raw = binding.get("builder_deadline_epoch")
    deadline = (
        float(deadline_raw)
        if isinstance(deadline_raw, (int, float)) and not isinstance(deadline_raw, bool)
        else None
    )
    if deadline is None or deadline <= time.time() + 600:
        blockers.append("typed_model_cache_builder_deadline_too_near")
    volume_deadline_raw = handoff.get("watchdog_deadline_epoch")
    volume_deadline = (
        float(volume_deadline_raw)
        if isinstance(volume_deadline_raw, (int, float))
        and not isinstance(volume_deadline_raw, bool)
        else None
    )
    if (
        handoff.get("schema_version") != "groot_oscar_model_volume_watchdog_handoff.v1"
        or handoff.get("status") != "storage_preparation_watchdog_armed"
        or handoff.get("volume_id") != volume.get("id")
        or handoff.get("teardown_owner") != "independent_model_volume_watchdog"
        or volume_deadline is None
        or deadline is None
        or volume_deadline <= (deadline or 0) + 2100
        or binding.get("volume_watchdog_deadline_epoch") != volume_deadline_raw
    ):
        blockers.append("typed_model_cache_volume_watchdog_handoff_invalid")
    blockers.extend(_context_blockers(packet))
    return blockers


def _canary_verification(
    *,
    packet: Mapping[str, Any],
    local: Mapping[str, Any],
    transport: Mapping[str, Any],
    transport_result_sha256: str,
) -> dict[str, Any]:
    volume = packet["volume_evidence"]
    manifest_digest = local.get("model_manifest_digest")
    blockers: list[str] = []
    if transport.get("status") != "completed":
        blockers.append("typed_model_cache_transport_not_completed")
    if not manifest_digest or transport.get("model_manifest_digest") != manifest_digest:
        blockers.append("typed_model_cache_transport_manifest_digest_mismatch")
    if transport.get("remote_model_manifest_digest") != manifest_digest:
        blockers.append("typed_model_cache_remote_manifest_digest_mismatch")
    if transport.get("provider_volume_id") != volume.get("id"):
        blockers.append("typed_model_cache_transport_volume_mismatch")
    if transport.get("remote_provider_volume_id") != volume.get("id"):
        blockers.append("typed_model_cache_remote_volume_mismatch")
    if transport.get("remote_prefix") != packet.get("remote_prefix"):
        blockers.append("typed_model_cache_transport_prefix_mismatch")
    if transport.get("verification_method") != (
        "full_s3_redownload_and_sha256_manifest_verification"
    ):
        blockers.append("typed_model_cache_transport_verification_method_invalid")
    if not isinstance(transport.get("verified_size_bytes"), int) or int(
        transport.get("verified_size_bytes") or 0
    ) <= 0:
        blockers.append("typed_model_cache_transport_verified_bytes_invalid")
    if not isinstance(transport.get("remote_verified_file_count"), int) or int(
        transport.get("remote_verified_file_count") or 0
    ) <= 0:
        blockers.append("typed_model_cache_remote_verified_file_count_invalid")
    if _HEX64.fullmatch(str(transport.get("remote_verification_sha256") or "")) is None:
        blockers.append("typed_model_cache_remote_verification_digest_invalid")
    if _HEX64.fullmatch(transport_result_sha256) is None:
        blockers.append("typed_model_cache_transport_result_digest_invalid")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "model_manifest_digest": manifest_digest,
        "expected_model_manifest_digest": manifest_digest,
        "cache_root": str(RUNTIME_CACHE_ROOT),
        "provider_volume_id": volume.get("id"),
        "verified_file_count": transport.get("remote_verified_file_count") if not blockers else 0,
        "verified_size_bytes": transport.get("verified_size_bytes") if not blockers else 0,
        "checks": {"models_cached_offline": not blockers},
        "remote_prefix": packet.get("remote_prefix"),
        "runtime_path_mapping_verified": not blockers,
        "transport_result_sha256": transport_result_sha256,
        "remote_verification_sha256": transport.get("remote_verification_sha256"),
        "raw_secret_values_recorded": False,
    }


def execute_remote_packet() -> dict[str, Any]:
    """Execute only at the canonical adapter's fixed paths and capability."""

    result: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "failed",
        "blockers": ["typed_model_cache_remote_execution_failed"],
        "outer_volume_deletion_required": True,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    transport: dict[str, Any] = {}
    terminal_exists = False
    try:
        ensure_dir(OUTPUT_DIR)
        terminal_exists = (OUTPUT_DIR / EXECUTION_RESULT_NAME).exists()
        if terminal_exists:
            result["blockers"] = ["typed_model_cache_terminal_result_already_exists"]
            raise RuntimeError("typed_model_cache_terminal_result_already_exists")
        lock_fd = os.open(
            EXECUTION_LOCK_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(lock_fd)
        if CONSUMED_CAPABILITY_PATH.exists():
            raise RuntimeError("typed_model_cache_capability_already_consumed")
        PARENT_CAPABILITY_PATH.replace(CONSUMED_CAPABILITY_PATH)
        capability = CONSUMED_CAPABILITY_PATH.read_bytes()
        packet = _load_object(PACKET_PATH)
        binding = _load_object(PARENT_BINDING_PATH)
        blockers = _contract_blockers(packet, binding, capability)
        if blockers:
            result = {
                **result,
                "status": "blocked",
                "blockers": sorted(set(blockers)),
            }
            raise RuntimeError("typed_model_cache_contract_blocked")
        if sys.version_info[:2] != (3, 12) or not VENV_SITE_PACKAGES.is_dir():
            raise RuntimeError("typed_model_cache_python_runtime_mismatch")
        dependency_manifest = _load_object(DEPENDENCY_MANIFEST_PATH)
        if dependency_manifest.get("schema_version") != "blueprint_python_wheelhouse.v1":
            raise RuntimeError("typed_model_cache_dependency_manifest_invalid")
        installed = _verified_installed_dependencies(dependency_manifest)
        sys.path.append(str(VENV_SITE_PACKAGES))
        import boto3
        import botocore
        import huggingface_hub
        write_json(
            DEPENDENCY_RESULT_PATH,
            {
                "schema_version": "groot_oscar_model_cache_dependency_verification.v1",
                "status": "verified",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "dependency_manifest_sha256": packet["dependency_manifest_sha256"],
                "installed_distributions": dict(sorted(installed.items())),
                "required_import_versions": {
                    "boto3": boto3.__version__,
                    "botocore": botocore.__version__,
                    "huggingface-hub": huggingface_hub.__version__,
                },
                "dependency_imports_after_parent_contract_validation": True,
                "raw_secret_values_recorded": False,
            },
        )
        hf_token = _secret(HF_TOKEN_PATH)
        _secret(S3_ACCESS_KEY_PATH)
        _secret(S3_SECRET_KEY_PATH)
        manifest = prepare_model_cache(RUNTIME_CACHE_ROOT, token=hf_token)
        sonic = _load_object(RUNTIME_CACHE_ROOT / "sonic/config.json")
        if sonic.get("model_name") != str(RUNTIME_COSMOS_MODEL_ROOT):
            raise RuntimeError("typed_model_cache_sonic_runtime_path_mismatch")
        local = verify_model_cache(RUNTIME_CACHE_ROOT)
        if (
            local.get("status") != "passed"
            or local.get("model_manifest_digest") != manifest.get("manifest_digest")
        ):
            raise RuntimeError("typed_model_cache_local_verification_failed")
        volume = packet["volume_evidence"]
        transport = _upload_and_verify_model_cache_impl(
            cache_root=RUNTIME_CACHE_ROOT,
            verification_root=VERIFICATION_ROOT,
            volume_id=str(volume["id"]),
            data_center_id=str(packet["data_center_id"]),
            access_key_file=S3_ACCESS_KEY_PATH,
            secret_key_file=S3_SECRET_KEY_PATH,
            volume_evidence=volume,
            allocation_nonce=str(packet["allocation_nonce"]),
            remote_prefix=str(packet["remote_prefix"]),
            live_probe_attempts=12,
            live_probe_interval_seconds=5.0,
            execution_capability=_issue_transport_execution_capability(
                remote_parent_binding=binding,
                remote_parent_capability=capability,
                remote_packet=packet,
            ),
        )
        write_json(OUTPUT_DIR / TRANSPORT_RESULT_NAME, transport)
        transport_result_sha256 = _sha256(OUTPUT_DIR / TRANSPORT_RESULT_NAME)
        canary = _canary_verification(
            packet=packet,
            local=local,
            transport=transport,
            transport_result_sha256=transport_result_sha256,
        )
        write_json(OUTPUT_DIR / CANARY_VERIFICATION_NAME, canary)
        if canary["status"] != "passed":
            raise RuntimeError("typed_model_cache_canary_verification_blocked")
        result = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "status": "completed",
            "blockers": [],
            "packet_kind": "model_cache_s3",
            "source_commit": packet["source_commit"],
            "source_patch_sha256": packet["source_patch_sha256"],
            "tarball_sha256": binding["tarball_sha256"],
            "droplet_id": binding["droplet_id"],
            "provider_volume_id": volume["id"],
            "model_manifest_digest": local["model_manifest_digest"],
            "provider_mutations_performed": transport["provider_mutations_performed"],
            "outer_volume_deletion_required": False,
            "gpu_compute_allocated": False,
            "raw_secret_values_recorded": False,
        }
    except Exception as exc:  # noqa: BLE001 - terminal secret-free evidence
        transport_digest = (
            _sha256(OUTPUT_DIR / TRANSPORT_RESULT_NAME)
            if (OUTPUT_DIR / TRANSPORT_RESULT_NAME).is_file()
            else None
        )
        if result.get("status") != "blocked":
            result.update(
                {
                    "status": "failed",
                    "blockers": ["typed_model_cache_remote_execution_failed"],
                    "error_type": type(exc).__name__,
                }
            )
        result.update(
            {
                "provider_mutations_performed": int(
                    transport.get("provider_mutations_performed") or 0
                ),
                "transport_result_sha256": transport_digest,
                "partial_upload_cleanup_verified": transport.get(
                    "partial_upload_cleanup_verified"
                ),
                "upload_attempt_count": int(transport.get("upload_attempt_count") or 0),
                "upload_success_count": int(transport.get("upload_success_count") or 0),
                "cleanup_delete_attempt_count": int(
                    transport.get("cleanup_delete_attempt_count") or 0
                ),
                "cleanup_delete_success_count": int(
                    transport.get("cleanup_delete_success_count") or 0
                ),
                "final_provider_observed_prefix_empty": transport.get(
                    "final_provider_observed_prefix_empty"
                ),
                "outer_volume_deletion_required": True,
                "gpu_compute_allocated": False,
            }
        )
    finally:
        cleanup = {}
        for path in (
            HF_TOKEN_PATH,
            S3_ACCESS_KEY_PATH,
            S3_SECRET_KEY_PATH,
            PARENT_CAPABILITY_PATH,
            CONSUMED_CAPABILITY_PATH,
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                cleanup[str(path)] = False
            else:
                cleanup[str(path)] = not path.exists()
        result["secret_cleanup_verified"] = bool(cleanup and all(cleanup.values()))
        if not result["secret_cleanup_verified"]:
            result["status"] = "failed"
            result["outer_volume_deletion_required"] = True
            result["blockers"] = sorted(
                set([*result.get("blockers", []), "typed_model_cache_secret_cleanup_unverified"])
            )
    if not terminal_exists:
        write_json(OUTPUT_DIR / EXECUTION_RESULT_NAME, result)
    return result
