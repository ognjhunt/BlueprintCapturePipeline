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
import hmac
import io
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .groot_oscar_infrastructure_admission import (
    BUILD_SCHEMA_VERSION,
    DIGITALOCEAN_CPU_BUILDER_PROFILE,
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
    build_build_plane_admission,
    build_cpu_build_execution_admission,
    build_digitalocean_cpu_builder_profile_evidence,
    build_live_machine_capability_evidence,
    validate_carrier_image_archive,
)
from .groot_oscar_model_cache_wheelhouse import (
    _wheel_compatible,
    plan_model_cache_wheelhouse,
)
from .groot_oscar_remote_build_results import (
    REMOTE_BUILD_REQUIRED_RESULTS as REMOTE_BUILD_REQUIRED_RESULTS,  # noqa: F401
    validate_remote_build_results,
)
from .paid_resource_admission import require_paid_resource_admission

SCHEMA_VERSION = "groot_oscar_digitalocean_builder_run.v1"
WATCHDOG_SCHEMA_VERSION = "groot_oscar_digitalocean_builder_watchdog.v1"
DO_API = "https://api.digitalocean.com/v2"
BUILDER_TAG = "blueprint-groot-oscar-builder"
TEARDOWN_TAG = "auto-teardown-required"
READINESS_TIMEOUT_SECONDS = 15 * 60
MODEL_CACHE_PACKET_DIRECTORY = "groot_oscar_model_cache_s3_remote"
MODEL_CACHE_TARBALL_NAME = "groot_oscar_model_cache_s3_remote_packet.tar.gz"
MODEL_CACHE_RESULT_FILES = (
    "runpod_s3_model_cache_transport_result.json",
    "external_model_cache_verification.json",
    "model_cache_s3_remote_execution_result.json",
)
CARRIER_PACKET_DIRECTORY = "groot_oscar_carrier_remote_build"
CARRIER_BUILD_SCRIPT = "remote_build_groot_oscar_carrier.sh"
CARRIER_RESULT_NAME = "groot_oscar_carrier_remote_build_result.json"


def _safe_archive_member(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _validate_model_cache_archive(path: Path, packet: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if path.name != MODEL_CACHE_TARBALL_NAME:
        blockers.append("digitalocean_model_cache_tarball_name_invalid")
    declared = packet.get("archive_members")
    declared = declared if isinstance(declared, list) else []
    if (
        not declared
        or not all(isinstance(name, str) and _safe_archive_member(name) for name in declared)
        or len(declared) != len(set(declared))
    ):
        return ["digitalocean_model_cache_archive_member_contract_invalid"]
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                blockers.append("digitalocean_model_cache_archive_duplicate_member")
            if any(not _safe_archive_member(name) for name in names):
                blockers.append("digitalocean_model_cache_archive_unsafe_path")
            if any(not member.isfile() for member in members):
                blockers.append("digitalocean_model_cache_archive_nonregular_member")
            if names != declared:
                blockers.append("digitalocean_model_cache_archive_inventory_mismatch")
            if blockers:
                return sorted(set(blockers))
            payloads: dict[str, bytes] = {}
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    blockers.append("digitalocean_model_cache_archive_member_unreadable")
                    continue
                payloads[member.name] = extracted.read()
    except (OSError, tarfile.TarError):
        return ["digitalocean_model_cache_archive_unreadable"]
    prefix = MODEL_CACHE_PACKET_DIRECTORY + "/"
    try:
        inner_packet = json.loads(payloads[prefix + "packet.json"])
        context = json.loads(payloads[prefix + "context_manifest.json"])
        dependencies = json.loads(payloads[prefix + "dependency_manifest.json"])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return ["digitalocean_model_cache_archive_contract_json_invalid"]
    if not all(isinstance(item, dict) for item in (inner_packet, context, dependencies)):
        blockers.append("digitalocean_model_cache_archive_contract_json_invalid")
        return blockers
    if (
        inner_packet.get("schema_version") != "groot_oscar_model_cache_s3_remote_packet.v1"
        or inner_packet.get("packet_kind") != "model_cache_s3"
        or inner_packet.get("result_files") != list(MODEL_CACHE_RESULT_FILES)
        or inner_packet.get("raw_secret_values_recorded") is not False
    ):
        blockers.append("digitalocean_model_cache_inner_packet_invalid")
    inner_volume = inner_packet.get("volume_evidence")
    inner_volume = inner_volume if isinstance(inner_volume, dict) else {}
    if (
        inner_packet.get("source_commit") != packet.get("source_commit")
        or inner_packet.get("source_patch_sha256") != packet.get("source_patch_sha256")
        or inner_packet.get("allocation_nonce") != packet.get("allocation_nonce")
        or inner_packet.get("data_center_id") != packet.get("data_center_id")
        or inner_volume.get("id") != packet.get("provider_volume_id")
    ):
        blockers.append("digitalocean_model_cache_inner_outer_binding_mismatch")
    context_bytes = payloads[prefix + "context_manifest.json"]
    if hashlib.sha256(context_bytes).hexdigest() != inner_packet.get("context_manifest_sha256"):
        blockers.append("digitalocean_model_cache_context_manifest_digest_mismatch")
    context_rows = context.get("files") if isinstance(context.get("files"), list) else []
    dependency_rows = (
        dependencies.get("wheels") if isinstance(dependencies.get("wheels"), list) else []
    )
    expected_members = {
        prefix + "packet.json",
        prefix + "context_manifest.json",
        prefix + "dependency_manifest.json",
        prefix + "remote_entrypoint.py",
        prefix + "uv.lock",
        prefix + "requirements_closure.json",
    }
    context_paths = [str(row.get("path") or "") for row in context_rows if isinstance(row, dict)]
    if len(context_paths) != len(set(context_paths)):
        blockers.append("digitalocean_model_cache_context_path_duplicate")
    for row in context_rows:
        if not isinstance(row, dict):
            blockers.append("digitalocean_model_cache_context_row_invalid")
            continue
        relative = str(row.get("path") or "")
        member = prefix + relative
        expected_members.add(member)
        body = payloads.get(member)
        if (
            not _safe_archive_member(relative)
            or body is None
            or hashlib.sha256(body).hexdigest() != row.get("sha256")
            or len(body) != row.get("bytes")
        ):
            blockers.append("digitalocean_model_cache_context_digest_mismatch")
    for row in dependency_rows:
        if not isinstance(row, dict) or set(row) != {
            "bytes",
            "distribution",
            "filename",
            "sha256",
            "version",
        }:
            blockers.append("digitalocean_model_cache_dependency_row_invalid")
            continue
        filename = str(row.get("filename") or "")
        member = prefix + "wheelhouse/" + filename
        expected_members.add(member)
        body = payloads.get(member)
        if (
            not filename.endswith(".whl")
            or "/" in filename
            or not str(row.get("distribution") or "")
            or not str(row.get("version") or "")
            or body is None
            or hashlib.sha256(body).hexdigest() != row.get("sha256")
            or len(body) != row.get("bytes")
        ):
            blockers.append("digitalocean_model_cache_dependency_digest_mismatch")
        elif body is not None:
            if not _wheel_compatible(filename):
                blockers.append("digitalocean_model_cache_dependency_wheel_tag_invalid")
            try:
                with zipfile.ZipFile(io.BytesIO(body)) as wheel:
                    wheel_names_in_archive = wheel.namelist()
                    if len(wheel_names_in_archive) != len(set(wheel_names_in_archive)):
                        blockers.append("digitalocean_model_cache_wheel_duplicate_member")
                    for wheel_member in wheel.infolist():
                        member_path = Path(wheel_member.filename)
                        parts = member_path.parts
                        mode = (wheel_member.external_attr >> 16) & 0o170000
                        if (
                            len(parts) >= 3
                            and parts[0].endswith(".data")
                            and parts[1]
                            in {
                                "purelib",
                                "platlib",
                            }
                        ):
                            top_level = parts[2]
                        else:
                            top_level = parts[0] if parts else ""
                        module_name = top_level.removesuffix(".py")
                        if (
                            not _safe_archive_member(wheel_member.filename)
                            or wheel_member.filename.endswith(".pth")
                            or member_path.name in {"sitecustomize.py", "usercustomize.py"}
                            or mode == stat.S_IFLNK
                            or module_name in sys.stdlib_module_names
                            or module_name == "blueprint_pipeline"
                        ):
                            blockers.append("digitalocean_model_cache_wheel_startup_hook_forbidden")
            except zipfile.BadZipFile:
                blockers.append("digitalocean_model_cache_dependency_wheel_invalid")
    wheel_names = [
        str(row.get("filename") or "") for row in dependency_rows if isinstance(row, dict)
    ]
    if len(wheel_names) != len(set(wheel_names)):
        blockers.append("digitalocean_model_cache_dependency_filename_duplicate")
    if dependencies.get("schema_version") != "blueprint_python_wheelhouse.v1":
        blockers.append("digitalocean_model_cache_dependency_manifest_invalid")
    if (
        dependencies.get("python_version") != "3.12"
        or dependencies.get("implementation") != "cpython"
        or "manylinux_2_17_x86_64"
        not in (
            dependencies.get("platform_tags")
            if isinstance(dependencies.get("platform_tags"), list)
            else []
        )
        or not re.fullmatch(r"[0-9a-f]{64}", str(dependencies.get("lockfile_sha256") or ""))
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(dependencies.get("requirements_closure_sha256") or ""),
        )
    ):
        blockers.append("digitalocean_model_cache_dependency_runtime_binding_invalid")
    if not dependency_rows:
        blockers.append("digitalocean_model_cache_dependency_inventory_empty")
    lock_body = payloads.get(prefix + "uv.lock")
    closure_body = payloads.get(prefix + "requirements_closure.json")
    if (
        lock_body is None
        or hashlib.sha256(lock_body).hexdigest() != dependencies.get("lockfile_sha256")
        or hashlib.sha256(lock_body).hexdigest() != inner_packet.get("dependency_lock_sha256")
    ):
        blockers.append("digitalocean_model_cache_dependency_lock_digest_mismatch")
    if lock_body is not None:
        try:
            locked_plan = plan_model_cache_wheelhouse(lock_body)
        except (ValueError, UnicodeDecodeError):
            blockers.append("digitalocean_model_cache_dependency_lock_plan_invalid")
        else:
            locked_wheels = [
                {key: value for key, value in row.items() if key != "url"}
                for row in locked_plan["wheels"]
            ]
            if dependencies.get("requirements") != locked_plan["requirements"]:
                blockers.append("digitalocean_model_cache_dependency_closure_not_locked")
            if dependency_rows != locked_wheels:
                blockers.append("digitalocean_model_cache_dependency_wheels_not_locked")
    if (
        closure_body is None
        or hashlib.sha256(closure_body).hexdigest()
        != dependencies.get("requirements_closure_sha256")
        or hashlib.sha256(closure_body).hexdigest()
        != inner_packet.get("requirements_closure_sha256")
    ):
        blockers.append("digitalocean_model_cache_dependency_closure_digest_mismatch")
    if set(declared) != expected_members:
        blockers.append("digitalocean_model_cache_archive_allowlist_mismatch")
    if any("sitecustomize.py" in name or "__pycache__" in name for name in declared):
        blockers.append("digitalocean_model_cache_archive_startup_hook_forbidden")
    observed_member_map = {
        name: {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
        for name, body in payloads.items()
    }
    observed_member_manifest_sha256 = hashlib.sha256(
        json.dumps(observed_member_map, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if packet.get("archive_member_manifest_sha256") != observed_member_manifest_sha256:
        blockers.append("digitalocean_model_cache_archive_member_manifest_mismatch")
    if (
        packet.get("fixed_remote_directory") != MODEL_CACHE_PACKET_DIRECTORY
        or packet.get("fixed_result_files") != list(MODEL_CACHE_RESULT_FILES)
        or packet.get("arbitrary_entrypoint_supported") is not False
    ):
        blockers.append("digitalocean_model_cache_outer_packet_dispatch_invalid")
    return sorted(set(blockers))


def model_cache_archive_verifier_script(*, packet: Mapping[str, Any], tarball_path: Path) -> str:
    """Return a stdlib-only verifier/extractor that runs before packet imports."""

    blockers = _validate_model_cache_archive(tarball_path, packet)
    if blockers:
        raise ValueError("model_cache_archive_not_verified:" + ",".join(blockers))
    expected: dict[str, dict[str, Any]] = {}
    with tarfile.open(tarball_path, "r:gz") as archive:
        for member in archive.getmembers():
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("model_cache_archive_member_unreadable")
            body = stream.read()
            expected[member.name] = {
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
    expected_tarball = hashlib.sha256(tarball_path.read_bytes()).hexdigest()
    expected_map_sha256 = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    remote_tarball = "/root/blueprint-build/" + MODEL_CACHE_TARBALL_NAME
    return f'''import hashlib, json, os, pathlib, stat, tarfile
TARBALL = pathlib.Path({remote_tarball!r})
DESTINATION = pathlib.Path("/root/blueprint-build/run/{MODEL_CACHE_PACKET_DIRECTORY}")
RESULT = pathlib.Path("/root/blueprint-build/model_cache_archive_verification.json")
EXPECTED_TARBALL = {expected_tarball!r}
EXPECTED_MAP_SHA256 = {expected_map_sha256!r}
EXPECTED = json.loads({json.dumps(json.dumps(expected, sort_keys=True))})
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
blockers = []
if sha256_file(TARBALL) != EXPECTED_TARBALL:
    blockers.append("remote_model_cache_tarball_digest_mismatch")
try:
    with tarfile.open(TARBALL, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            blockers.append("remote_model_cache_archive_duplicate_member")
        if names != sorted(EXPECTED):
            blockers.append("remote_model_cache_archive_inventory_mismatch")
        for member in members:
            parts = pathlib.PurePosixPath(member.name).parts
            if not member.isfile() or member.issym() or member.islnk():
                blockers.append("remote_model_cache_archive_nonregular_member")
                continue
            if not parts or member.name.startswith("/") or ".." in parts:
                blockers.append("remote_model_cache_archive_unsafe_path")
                continue
            stream = archive.extractfile(member)
            body = stream.read() if stream is not None else b""
            row = EXPECTED.get(member.name, {{}})
            if len(body) != row.get("bytes") or hashlib.sha256(body).hexdigest() != row.get("sha256"):
                blockers.append("remote_model_cache_archive_member_digest_mismatch")
        if not blockers:
            if DESTINATION.exists():
                blockers.append("remote_model_cache_destination_already_exists")
            else:
                for member in members:
                    relative = pathlib.PurePosixPath(member.name).relative_to("{MODEL_CACHE_PACKET_DIRECTORY}")
                    destination = DESTINATION.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    body = stream.read() if stream is not None else b""
                    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(body)
except (OSError, tarfile.TarError, ValueError) as exc:
    blockers.append("remote_model_cache_archive_verification_exception:" + type(exc).__name__)
payload = {{
    "schema_version": "groot_oscar_model_cache_archive_verification.v1",
    "status": "verified" if not blockers else "blocked",
    "blockers": sorted(set(blockers)),
    "tarball_sha256": EXPECTED_TARBALL,
    "expected_member_map_sha256": EXPECTED_MAP_SHA256,
    "archive_members": sorted(EXPECTED),
    "stdlib_only_preimport_verification": True,
    "raw_secret_values_recorded": False,
}}
RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
raise SystemExit(0 if not blockers else 2)
'''


def _load_model_cache_inner_packet(tarball_path: Path) -> dict[str, Any]:
    member_name = MODEL_CACHE_PACKET_DIRECTORY + "/packet.json"
    with tarfile.open(tarball_path, "r:gz") as archive:
        member = archive.getmember(member_name)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("model_cache_inner_packet_unreadable")
        payload = json.loads(stream.read())
    if not isinstance(payload, dict):
        raise ValueError("model_cache_inner_packet_not_object")
    return payload


def build_model_cache_parent_binding(
    *,
    packet: Mapping[str, Any],
    inner_packet: Mapping[str, Any],
    capability: bytes,
    droplet_id: str,
    name: str,
    region: str,
    ssh_host_key_sha256: str,
    builder_deadline_epoch: float,
) -> dict[str, Any]:
    handoff = inner_packet.get("volume_watchdog_handoff")
    handoff = handoff if isinstance(handoff, Mapping) else {}
    unsigned = {
        "schema_version": "groot_oscar_model_cache_s3_parent_binding.v1",
        "packet_kind": "model_cache_s3",
        "tarball_sha256": packet.get("tarball_sha256"),
        "capability_sha256": hashlib.sha256(capability).hexdigest(),
        "droplet_id": droplet_id,
        "name": name,
        "region": region,
        "ssh_host_key_sha256": ssh_host_key_sha256,
        "provider_volume_id": packet.get("provider_volume_id"),
        "allocation_nonce": packet.get("allocation_nonce"),
        "builder_deadline_epoch": builder_deadline_epoch,
        "volume_watchdog_deadline_epoch": handoff.get("watchdog_deadline_epoch"),
        "archive_members": packet.get("archive_members"),
        "raw_secret_values_recorded": False,
    }
    signature = hmac.new(
        capability,
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()
    return {**unsigned, "binding_hmac_sha256": signature}


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
        if packet.get("packet_kind") == "model_cache_s3":
            blockers.extend(_validate_model_cache_archive(path, packet))
        elif packet.get("packet_kind") == "carrier_image":
            blockers.extend(validate_carrier_image_archive(packet))
    return {
        "schema_version": "groot_oscar_builder_packet_tarball_verification.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "tarball_path": str(path),
        "declared_sha256": declared or None,
        "observed_sha256": observed or None,
        "raw_secret_values_recorded": False,
    }


def validate_remote_carrier_result(
    results_dir: Path, *, packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the carrier build receipt to the exact packet and registry digest."""

    blockers: list[str] = []
    path = results_dir / CARRIER_RESULT_NAME
    payload = _load_object(path) if path.is_file() else {}
    resolved = str(payload.get("resolved_digest_ref") or "")
    expected_tag = str(packet.get("carrier_image_ref") or "")
    expected_base = str(packet.get("carrier_base_image_ref") or "")
    expected_dockerfile = str(packet.get("carrier_dockerfile_sha256") or "")
    if payload.get("schema_version") != "groot_oscar_carrier_remote_build_result.v1":
        blockers.append("carrier_remote_build_result_schema_invalid")
    if payload.get("status") != "completed" or payload.get("blockers") not in ([], ()):
        blockers.append("carrier_remote_build_not_completed")
    if payload.get("image_ref") != expected_tag:
        blockers.append("carrier_remote_build_image_ref_mismatch")
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", resolved):
        blockers.append("carrier_remote_build_digest_ref_invalid")
    if payload.get("base_image_ref") != expected_base:
        blockers.append("carrier_remote_build_base_ref_mismatch")
    if payload.get("dockerfile_sha256") != expected_dockerfile:
        blockers.append("carrier_remote_build_dockerfile_sha256_mismatch")
    if payload.get("source_commit") != packet.get("source_commit"):
        blockers.append("carrier_remote_build_source_commit_mismatch")
    if payload.get("platform") != "linux/amd64":
        blockers.append("carrier_remote_build_platform_invalid")
    if payload.get("raw_secret_values_recorded") is not False:
        blockers.append("carrier_remote_build_secret_boundary_invalid")
    return {
        "schema_version": "groot_oscar_carrier_remote_build_verification.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "resolved_digest_ref": resolved or None,
        "raw_secret_values_recorded": False,
    }


def live_machine_probe_command(
    *,
    mount_path: str = "/",
    packet_kind: str = "thin_release",
    s3_endpoint_host: str | None = None,
) -> str:
    """Return a dependency-free probe whose JSON comes from the live host."""

    encoded_mount = json.dumps(mount_path)
    encoded_kind = json.dumps(packet_kind)
    encoded_s3_host = repr(s3_endpoint_host)
    return f"""python3 - <<'PY'
import json, os, platform, shutil, socket, subprocess, sys, tempfile, urllib.error, urllib.request
mount_path = {encoded_mount}
packet_kind = {encoded_kind}
s3_endpoint_host = {encoded_s3_host}
stats = os.statvfs(mount_path)
def ok(argv):
    try:
        return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
def venv_ok():
    root = tempfile.mkdtemp(prefix="blueprint-venv-probe-")
    try:
        return ok([sys.executable, "-m", "venv", root]) and os.path.isfile(os.path.join(root, "bin", "pip"))
    finally:
        shutil.rmtree(root, ignore_errors=True)
def dns_ok():
    try:
        socket.getaddrinfo(s3_endpoint_host, 443)
        return True
    except OSError:
        return False
def https_ok():
    try:
        urllib.request.urlopen("https://" + s3_endpoint_host + "/", timeout=15).close()
        return True
    except urllib.error.HTTPError:
        return True
    except (OSError, urllib.error.URLError):
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
    "python3_available": shutil.which("python3") is not None,
    "python_version": f"{{sys.version_info.major}}.{{sys.version_info.minor}}",
    "python_venv_available": venv_ok(),
    "dns_resolution_verified": dns_ok() if packet_kind == "model_cache_s3" else None,
    "outbound_https_verified": https_ok() if packet_kind == "model_cache_s3" else None,
    "s3_endpoint_host": s3_endpoint_host,
    "builder_ready_marker": os.path.isfile("/root/blueprint-builder-ready"),
}}, sort_keys=True))
PY"""


def parse_live_machine_probe(
    stdout: str,
    *,
    packet_kind: str = "thin_release",
    expected_s3_endpoint_host: str | None = None,
) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("live_machine_probe_output_missing")
    try:
        observation = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError("live_machine_probe_output_invalid") from exc
    if not isinstance(observation, Mapping):
        raise ValueError("live_machine_probe_output_not_object")
    return build_live_machine_capability_evidence(
        observation,
        packet_kind=packet_kind,
        expected_s3_endpoint_host=expected_s3_endpoint_host,
    )


def observe_local_machine(
    *,
    mount_path: str | Path,
    packet_kind: str = "thin_release",
    s3_endpoint_host: str | None = None,
) -> dict[str, Any]:
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

    dns_verified = False
    https_verified = False
    if packet_kind == "model_cache_s3" and s3_endpoint_host:
        try:
            import socket

            socket.getaddrinfo(s3_endpoint_host, 443)
            dns_verified = True
        except OSError:
            # Fail closed: the default remains unverified when DNS probing fails.
            dns_verified = False
        try:
            with urllib.request.urlopen(  # nosec B310 - fixed RunPod endpoint
                "https://" + str(s3_endpoint_host) + "/", timeout=15
            ):
                https_verified = True
        except urllib.error.HTTPError:
            https_verified = True
        except (OSError, urllib.error.URLError):
            # Fail closed: the default remains unverified on transport failure.
            https_verified = False
    with tempfile.TemporaryDirectory(prefix="blueprint-venv-probe-") as probe_dir:
        venv_verified = (
            succeeds([sys.executable, "-m", "venv", probe_dir])
            and (Path(probe_dir) / "bin/pip").is_file()
        )
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
            "python3_available": shutil.which("python3") is not None,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python_venv_available": venv_verified,
            "dns_resolution_verified": dns_verified,
            "outbound_https_verified": https_verified,
            "s3_endpoint_host": s3_endpoint_host,
            "builder_ready_marker": Path("/root/blueprint-builder-ready").is_file(),
        },
        packet_kind=packet_kind,
        expected_s3_endpoint_host=s3_endpoint_host,
    )


DETACHED_LAUNCH_SCHEMA_VERSION = "groot_oscar_digitalocean_builder_launch.v1"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_remote_model_cache_results(
    results_dir: Path,
    *,
    packet: Mapping[str, Any],
    parent_binding: Mapping[str, Any] | None = None,
    droplet_id: str | None = None,
) -> dict[str, Any]:
    """Accept only the three fixed cache results bound to the outer packet."""

    blockers: list[str] = []
    entries = list(results_dir.iterdir())
    actual = {path.name for path in entries}
    if actual != set(MODEL_CACHE_RESULT_FILES):
        blockers.append("remote_model_cache_result_inventory_invalid")
    if any(path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) for path in entries):
        blockers.append("remote_model_cache_result_nonregular_entry")
    payloads: dict[str, dict[str, Any]] = {}
    for name in MODEL_CACHE_RESULT_FILES:
        try:
            payloads[name] = _load_object(results_dir / name)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append(f"remote_model_cache_result_invalid:{name}")
    transport = payloads.get(MODEL_CACHE_RESULT_FILES[0], {})
    canary = payloads.get(MODEL_CACHE_RESULT_FILES[1], {})
    execution = payloads.get(MODEL_CACHE_RESULT_FILES[2], {})
    volume_id = packet.get("provider_volume_id")
    transport_path = results_dir / MODEL_CACHE_RESULT_FILES[0]
    transport_sha256 = (
        hashlib.sha256(transport_path.read_bytes()).hexdigest()
        if transport_path.is_file() and not transport_path.is_symlink()
        else None
    )
    if (
        transport.get("status") != "completed"
        or transport.get("provider_volume_id") != volume_id
        or transport.get("verification_method")
        != "full_s3_redownload_and_sha256_manifest_verification"
        or type(transport.get("remote_verified_file_count")) is not int
        or int(transport.get("remote_verified_file_count") or 0) <= 0
        or type(transport.get("verified_size_bytes")) is not int
        or int(transport.get("verified_size_bytes") or 0) <= 0
        or transport.get("gpu_compute_allocated") is not False
        or transport.get("raw_secret_values_recorded") is not False
    ):
        blockers.append("remote_model_cache_transport_not_verified")
    if (
        canary.get("schema_version") != "groot_oscar_external_model_cache_verification.v2"
        or canary.get("status") != "passed"
        or canary.get("provider_volume_id") != volume_id
        or canary.get("cache_root") != "/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1"
        or canary.get("checks", {}).get("models_cached_offline") is not True
        or canary.get("runtime_path_mapping_verified") is not True
        or canary.get("transport_result_sha256") != transport_sha256
        or canary.get("remote_verification_sha256") != transport.get("remote_verification_sha256")
        or canary.get("verified_file_count") != transport.get("remote_verified_file_count")
        or canary.get("verified_size_bytes") != transport.get("verified_size_bytes")
        or canary.get("remote_prefix") != transport.get("remote_prefix")
        or canary.get("raw_secret_values_recorded") is not False
    ):
        blockers.append("remote_model_cache_canary_verification_invalid")
    if (
        execution.get("schema_version") != "groot_oscar_model_cache_s3_remote_execution.v1"
        or execution.get("status") != "completed"
        or execution.get("source_commit") != packet.get("source_commit")
        or execution.get("source_patch_sha256") != packet.get("source_patch_sha256")
        or execution.get("tarball_sha256") != packet.get("tarball_sha256")
        or (
            parent_binding is not None
            and execution.get("tarball_sha256") != parent_binding.get("tarball_sha256")
        )
        or (droplet_id is not None and execution.get("droplet_id") != droplet_id)
        or execution.get("provider_volume_id") != volume_id
        or execution.get("provider_mutations_performed")
        != transport.get("provider_mutations_performed")
        or execution.get("secret_cleanup_verified") is not True
        or execution.get("outer_volume_deletion_required") is not False
        or execution.get("gpu_compute_allocated") is not False
        or execution.get("raw_secret_values_recorded") is not False
    ):
        blockers.append("remote_model_cache_execution_result_invalid")
    digests = {
        transport.get("model_manifest_digest"),
        canary.get("model_manifest_digest"),
        execution.get("model_manifest_digest"),
    }
    if None in digests or len(digests) != 1:
        blockers.append("remote_model_cache_manifest_digest_mismatch")
    return {
        "schema_version": "groot_oscar_remote_model_cache_results_verification.v1",
        "status": "verified" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "required_results": list(MODEL_CACHE_RESULT_FILES),
        "provider_volume_id": volume_id,
        "model_manifest_digest": next(iter(digests)) if len(digests) == 1 else None,
        "raw_secret_values_recorded": False,
    }


def _read_secret(path: Path) -> str:
    value = path.expanduser().read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"secret file empty: {path}")
    return value


def _read_private_secret(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_mode & 0o077:
        raise ValueError(f"secret file missing or not private: {resolved}")
    return _read_secret(resolved)


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


def build_cloud_init(
    *,
    host_private_b64: str,
    host_public_b64: str,
    shutdown_minutes: int,
    packet_kind: str = "thin_release",
    runtime_bundle_requested: bool = False,
) -> str:
    """Return cloud-init with an exact client-generated SSH host identity."""

    if not host_private_b64 or not host_public_b64:
        raise ValueError("launch_bound_host_key_material_missing")
    if shutdown_minutes <= 0 or shutdown_minutes > 120:
        raise ValueError("shutdown_minutes_must_be_between_1_and_120")
    if packet_kind not in {"thin_release", "carrier_image", "model_cache_s3"}:
        raise ValueError("builder_packet_kind_unsupported")
    if runtime_bundle_requested and packet_kind != "model_cache_s3":
        raise ValueError("runtime_bundle_requires_model_cache_packet")
    if packet_kind in {"thin_release", "carrier_image"}:
        package_lines = (
            "  - ca-certificates\n  - curl\n  - git\n  - jq\n  - python3\n"
            "  - docker.io\n  - docker-buildx"
        )
        runtime_commands = (
            "  - systemctl enable --now docker\n  - docker info\n  - docker buildx version"
        )
        ready_command = (
            "  - bash -c 'docker info >/dev/null && docker buildx version "
            ">/dev/null && touch /root/blueprint-builder-ready'"
        )
    else:
        package_lines = "  - ca-certificates\n  - python3\n  - python3-venv"
        runtime_commands = (
            "  - python3 -m venv /root/blueprint-venv-probe\n"
            "  - test -x /root/blueprint-venv-probe/bin/pip\n"
            "  - rm -rf /root/blueprint-venv-probe"
        )
        ready_checks = "python3 -m venv /root/blueprint-venv-ready-probe"
        if runtime_bundle_requested:
            package_lines += "\n  - docker.io"
            runtime_commands += "\n  - systemctl enable --now docker\n  - docker info"
            ready_checks = f"docker info >/dev/null && {ready_checks}"
        ready_command = (
            "  - bash -c '"
            f"{ready_checks} && "
            "test -x /root/blueprint-venv-ready-probe/bin/pip && "
            "rm -rf /root/blueprint-venv-ready-probe && "
            "touch /root/blueprint-builder-ready'"
        )
    return f"""#cloud-config
ssh_deletekeys: false
bootcmd:
  - [bash, -c, "printf '%s' '{host_private_b64}' | base64 -d > /etc/ssh/ssh_host_ed25519_key && chmod 600 /etc/ssh/ssh_host_ed25519_key && printf '%s' '{host_public_b64}' | base64 -d > /etc/ssh/ssh_host_ed25519_key.pub && chmod 644 /etc/ssh/ssh_host_ed25519_key.pub && rm -f /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_rsa_key.pub /etc/ssh/ssh_host_ecdsa_key /etc/ssh/ssh_host_ecdsa_key.pub"]
package_update: true
packages:
{package_lines}
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
  - mkdir -p /root/blueprint-build /root/.blueprint-secrets
  - chmod 700 /root/.blueprint-secrets
{runtime_commands}
{ready_command}
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
            path=(f"/droplets?tag_name={encoded_tag}&per_page={per_page}&page={page}"),
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
            http_status, tagged_rows = _list_droplets_by_tag(token=token, tag=BUILDER_TAG)
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
    droplet_id = str(state.get("droplet_id") or "")
    name = str(state.get("name") or "")
    region = str(state.get("region") or "")
    if not droplet_id and (not name or not region):
        raise ValueError("watchdog_allocation_identity_missing")
    deadline = float(state["deadline_epoch"])
    write_json(
        output / "watchdog_armed.json",
        {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": "armed",
            "pid": os.getpid(),
            "deadline_epoch": deadline,
            "droplet_id": droplet_id or None,
            "name": name or None,
            "region": region or None,
            "watchdog_nonce": state.get("watchdog_nonce"),
            "raw_secret_values_recorded": False,
        },
    )
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
    token = _read_secret(token_file)
    if droplet_id:
        result = _delete_with_fail_closed_evidence(token=token, droplet_id=droplet_id)
    else:
        result = _reconcile_ambiguous_create(token=token, name=name, region=region)
    payload = {
        "schema_version": WATCHDOG_SCHEMA_VERSION,
        "status": (
            "provider_terminal" if result["provider_absence_confirmed"] else "teardown_unverified"
        ),
        "droplet_id": droplet_id or None,
        "name": name or None,
        "region": region or None,
        **result,
        "raw_secret_values_recorded": False,
    }
    write_json(output / "watchdog_result.json", payload)
    return 0 if result["provider_absence_confirmed"] else 2


def _live_profile(*, token: str, region: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sizes_http, sizes_payload = _request(token=token, method="GET", path="/sizes?per_page=200")
    droplets_http, tagged_droplets = _list_droplets_by_tag(token=token, tag=BUILDER_TAG)
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
    hf_token_file: Path | None = None,
    runpod_s3_access_key_file: Path | None = None,
    runpod_s3_secret_key_file: Path | None = None,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    ensure_dir(output)
    packet = _load_object(packet_manifest_path)
    packet_kind = str(packet.get("packet_kind") or "thin_release")
    builder = _load_object(builder_evidence_path)
    spend = _load_object(spend_path)
    admission = build_build_plane_admission(packet=packet, builder=builder, spend=spend)
    write_json(output / "build_plane_admission.json", admission)
    blockers = list(admission["blockers"])
    packet_tarball_verification = verify_packet_tarball(packet)
    write_json(output / "packet_tarball_verification.json", packet_tarball_verification)
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

    try:
        token = _read_secret(token_file)
        profile, builders = _live_profile(token=token, region=region)
    except Exception as exc:  # noqa: BLE001 - persist a secret-free terminal result
        result = {
            **_blocked_result(["digitalocean_builder_live_profile_unverified"]),
            "error_type": type(exc).__name__,
        }
        write_json(output / "builder_run_result.json", result)
        return result
    write_json(output / "live_builder_profile_evidence.json", profile)
    if profile["status"] != "verified" or builders:
        result = _blocked_result(profile["blockers"] or ["digitalocean_builder_overlap_detected"])
        write_json(output / "builder_run_result.json", result)
        return result

    try:
        host_private_b64, host_public_b64, fingerprint = _host_key_material(host_private_key)
    except Exception as exc:  # noqa: BLE001 - persist a secret-free terminal result
        result = {
            **_blocked_result(["builder_launch_bound_host_key_unavailable"]),
            "error_type": type(exc).__name__,
        }
        write_json(output / "builder_run_result.json", result)
        return result
    if fingerprint != builder.get("ssh_host_key_sha256"):
        result = _blocked_result(["builder_launch_bound_host_key_fingerprint_mismatch"])
        write_json(output / "builder_run_result.json", result)
        return result
    ttl = int(spend["hard_ttl_seconds"])
    hourly = float(profile["observed"]["price_hourly_usd"])
    maximum_cost = hourly * ttl / 3600
    if maximum_cost > float(spend["max_spend_usd"]):
        result = {
            **_blocked_result(["digitalocean_builder_cost_exceeds_authorized_cap"]),
            "required_maximum_compute_spend_usd": maximum_cost,
            "authorized_maximum_spend_usd": float(spend["max_spend_usd"]),
        }
        write_json(output / "builder_run_result.json", result)
        return result
    try:
        _read_secret(login_private_key)
        if packet_kind in {"thin_release", "carrier_image"}:
            _read_private_secret(docker_username_file)
            _read_private_secret(docker_password_file)
        else:
            if not all((hf_token_file, runpod_s3_access_key_file, runpod_s3_secret_key_file)):
                raise ValueError("model_cache_secret_paths_missing")
            for prerequisite_file in (
                hf_token_file,
                runpod_s3_access_key_file,
                runpod_s3_secret_key_file,
            ):
                assert prerequisite_file is not None
                _read_private_secret(prerequisite_file)
    except Exception as exc:  # noqa: BLE001 - persist a secret-free terminal result
        result = {
            **_blocked_result(["digitalocean_builder_local_credentials_unavailable"]),
            "error_type": type(exc).__name__,
        }
        write_json(output / "builder_run_result.json", result)
        return result
    if packet_kind == "model_cache_s3":
        name = f"blueprint-groot-oscar-cache-{str(packet['allocation_nonce'])[:16]}"
    elif packet_kind == "carrier_image":
        name = f"blueprint-groot-oscar-carrier-{str(packet['source_commit'])[:8]}"
    else:
        name = f"blueprint-groot-oscar-thin-{str(packet['source_commit'])[:8]}"
    user_data = build_cloud_init(
        host_private_b64=host_private_b64,
        host_public_b64=host_public_b64,
        shutdown_minutes=max(1, min(120, (ttl + 59) // 60)),
        packet_kind=packet_kind,
        runtime_bundle_requested=bool(packet.get("runtime_bundle_requested")),
    )
    create_payload = build_droplet_payload(
        name=name, region=region, ssh_key_id=ssh_key_id, user_data=user_data
    )
    started = time.time()
    watchdog_nonce = os.urandom(16).hex()
    state_path = output / "allocation_state.json"
    write_json(
        state_path,
        {
            "droplet_id": None,
            "name": name,
            "region": region,
            "deadline_epoch": started + ttl,
            "source_commit": packet["source_commit"],
            "maximum_spend_usd": maximum_cost,
            "watchdog_nonce": watchdog_nonce,
        },
    )
    watchdog_error_type: str | None = None
    try:
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
        watchdog_armed_path = output / "watchdog_armed.json"
        watchdog_start_deadline = time.time() + 10
        watchdog_armed = False
        while time.time() < watchdog_start_deadline:
            if watchdog_armed_path.is_file() and watchdog_process.poll() is None:
                try:
                    armed = _load_object(watchdog_armed_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    armed = {}
                watchdog_armed = bool(
                    armed.get("schema_version") == WATCHDOG_SCHEMA_VERSION
                    and armed.get("status") == "armed"
                    and armed.get("pid") == watchdog_process.pid
                    and armed.get("watchdog_nonce") == watchdog_nonce
                    and armed.get("name") == name
                    and armed.get("region") == region
                )
                if watchdog_armed:
                    break
            if watchdog_process.poll() is not None:
                break
            time.sleep(0.05)
        if watchdog_process.poll() is not None:
            watchdog_error_type = "WatchdogProcessExited"
    except Exception as exc:  # noqa: BLE001 - no provider mutation has occurred
        watchdog_armed = False
        watchdog_error_type = type(exc).__name__
    if not watchdog_armed:
        (output / "watchdog_cancelled").touch()
        result = {
            **_blocked_result(["digitalocean_builder_watchdog_not_armed_before_create"]),
            "error_type": watchdog_error_type,
        }
        write_json(output / "builder_run_result.json", result)
        return result
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
    definitive_rejection = 400 <= create_http < 500 and create_http not in {408, 409, 425, 429}
    if not create_succeeded and not definitive_rejection:
        reconciliation = _reconcile_ambiguous_create(
            token=token,
            name=name,
            region=region,
        )
        write_json(output / "ambiguous_create_reconciliation.json", reconciliation)
        absence_confirmed = reconciliation["provider_absence_confirmed"] is True
        if absence_confirmed:
            (output / "watchdog_cancelled").touch()
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
        (output / "watchdog_cancelled").touch()
        result = {
            **_blocked_result(["digitalocean_builder_create_rejected"]),
            "status": "create_rejected_no_allocation",
            "provider_mutation_performed": True,
            "provider_absence_confirmed": True,
        }
        write_json(output / "builder_run_result.json", result)
        return result

    write_json(
        state_path,
        {
            "droplet_id": droplet_id,
            "name": name,
            "region": region,
            "deadline_epoch": started + ttl,
            "source_commit": packet["source_commit"],
            "maximum_spend_usd": maximum_cost,
            "watchdog_nonce": watchdog_nonce,
        },
    )
    build_exit: int | None = None
    public_ip = ""
    teardown: dict[str, Any] = {"provider_absence_confirmed": False}
    capability_path: Path | None = None
    local_capability_cleanup_verified = True
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
        data_center_id = str(packet.get("data_center_id") or "")
        if packet_kind == "model_cache_s3":
            if data_center_id not in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
                raise RuntimeError("digitalocean_model_cache_data_center_unsupported")
            s3_endpoint_host = f"s3api-{data_center_id.lower()}.runpod.io"
        else:
            s3_endpoint_host = None
        remote_preflight = live_machine_probe_command(
            mount_path="/",
            packet_kind=packet_kind,
            s3_endpoint_host=s3_endpoint_host,
        )
        while time.time() < readiness_deadline:
            completed = subprocess.run(
                ["ssh", *options, f"root@{public_ip}", remote_preflight],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                try:
                    candidate = parse_live_machine_probe(
                        completed.stdout,
                        packet_kind=packet_kind,
                        expected_s3_endpoint_host=s3_endpoint_host,
                    )
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
            runtime_bundle_requested=bool(packet.get("runtime_bundle_requested")),
        )
        write_json(output / "cpu_build_execution_admission.json", execution_admission)
        if execution_admission["status"] != "admitted":
            raise RuntimeError("digitalocean_builder_execution_admission_blocked")

        packet_tarball = Path(packet_tarball_verification["tarball_path"])
        packet_tarball_sha256 = str(packet_tarball_verification["observed_sha256"] or "")
        transfers: list[tuple[Path, str]] = [
            (packet_tarball, f"/root/blueprint-build/{packet_tarball.name}")
        ]
        if packet_kind in {"thin_release", "carrier_image"}:
            transfers.extend(
                [
                    (docker_username_file.expanduser(), "/root/blueprint-build/docker_username"),
                    (docker_password_file.expanduser(), "/root/blueprint-build/docker_pat"),
                ]
            )
        else:
            inner_packet = _load_model_cache_inner_packet(packet_tarball)
            capability = os.urandom(32)
            binding = build_model_cache_parent_binding(
                packet=packet,
                inner_packet=inner_packet,
                capability=capability,
                droplet_id=droplet_id,
                name=name,
                region=region,
                ssh_host_key_sha256=fingerprint,
                builder_deadline_epoch=deadline,
            )
            write_json(output / "model_cache_parent_binding.json", binding)
            verifier_script = model_cache_archive_verifier_script(
                packet=packet, tarball_path=packet_tarball
            )
            verifier_path = output / "model_cache_archive_verifier.py"
            verifier_path.write_text(verifier_script, encoding="utf-8")
            verifier_sha256 = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
            capability_fd, capability_name = tempfile.mkstemp(
                prefix="blueprint-model-cache-capability-"
            )
            capability_path = Path(capability_name)
            with os.fdopen(capability_fd, "wb") as capability_file:
                capability_file.write(capability)
            os.chmod(capability_path, 0o600)
            assert hf_token_file is not None
            assert runpod_s3_access_key_file is not None
            assert runpod_s3_secret_key_file is not None
            transfers.extend(
                [
                    (verifier_path, "/root/blueprint-build/model_cache_archive_verifier.py"),
                    (
                        output / "model_cache_parent_binding.json",
                        "/root/blueprint-build/model_cache_parent_binding.json",
                    ),
                    (
                        capability_path,
                        "/root/blueprint-build/model_cache_parent_capability.incoming",
                    ),
                    (hf_token_file.expanduser(), "/root/blueprint-build/hf_token.incoming"),
                    (
                        runpod_s3_access_key_file.expanduser(),
                        "/root/blueprint-build/runpod_s3_access_key.incoming",
                    ),
                    (
                        runpod_s3_secret_key_file.expanduser(),
                        "/root/blueprint-build/runpod_s3_secret_key.incoming",
                    ),
                ]
            )
        for local_path, remote_path in transfers:
            subprocess.run(
                [
                    "scp",
                    *options,
                    str(local_path),
                    f"root@{public_ip}:{remote_path}",
                ],
                check=True,
            )
        if capability_path is not None:
            try:
                capability_path.unlink(missing_ok=True)
            except OSError as exc:
                local_capability_cleanup_verified = False
                raise RuntimeError("digitalocean_builder_local_capability_cleanup_failed") from exc
            else:
                local_capability_cleanup_verified = not capability_path.exists()
        remote_tarball = "/root/blueprint-build/" + packet_tarball.name
        if packet_kind in {"thin_release", "carrier_image"}:
            packet_directory = (
                CARRIER_PACKET_DIRECTORY
                if packet_kind == "carrier_image"
                else "groot_oscar_thin_remote_build"
            )
            build_script = (
                CARRIER_BUILD_SCRIPT
                if packet_kind == "carrier_image"
                else "remote_build_groot_oscar_thin_images.sh"
            )
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
                    f"cd /root/blueprint-build/run/{packet_directory}",
                    f"BLUEPRINT_REMOTE_IMAGE_BUILD_DOCKER_LOGIN=true ./{build_script}",
                ]
            )
        else:
            remote_command = " && ".join(
                [
                    "set -euo pipefail",
                    "trap 'rm -f /root/.blueprint-secrets/hf_token /root/.blueprint-secrets/runpod_s3_access_key /root/.blueprint-secrets/runpod_s3_secret_key /root/.blueprint-secrets/model_cache_parent_capability /root/blueprint-build/*.incoming' EXIT",
                    "install -d -m 700 /root/.blueprint-secrets /root/blueprint-build/run",
                    "install -m 600 /root/blueprint-build/hf_token.incoming /root/.blueprint-secrets/hf_token",
                    "install -m 600 /root/blueprint-build/runpod_s3_access_key.incoming /root/.blueprint-secrets/runpod_s3_access_key",
                    "install -m 600 /root/blueprint-build/runpod_s3_secret_key.incoming /root/.blueprint-secrets/runpod_s3_secret_key",
                    "install -m 600 /root/blueprint-build/model_cache_parent_capability.incoming /root/.blueprint-secrets/model_cache_parent_capability",
                    "rm -f /root/blueprint-build/*.incoming",
                    "printf '%s  %s\\n' "
                    f"{shlex.quote(packet_tarball_sha256)} "
                    f"{shlex.quote(remote_tarball)} | sha256sum -c -",
                    "printf '%s  %s\\n' "
                    f"{shlex.quote(verifier_sha256)} "
                    "/root/blueprint-build/model_cache_archive_verifier.py | sha256sum -c -",
                    "PYTHONDONTWRITEBYTECODE=1 python3 /root/blueprint-build/model_cache_archive_verifier.py",
                    "python3 -m venv /root/blueprint-build/model-cache-venv",
                    "/root/blueprint-build/model-cache-venv/bin/pip install --no-index --no-deps /root/blueprint-build/run/groot_oscar_model_cache_s3_remote/wheelhouse/*.whl",
                    "PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/root/blueprint-build/run/groot_oscar_model_cache_s3_remote/context/src /root/blueprint-build/model-cache-venv/bin/python -S /root/blueprint-build/run/groot_oscar_model_cache_s3_remote/remote_entrypoint.py",
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
        results_dir = output / "remote_results"
        ensure_dir(results_dir)
        if packet_kind in {"thin_release", "carrier_image"}:
            packet_directory = (
                CARRIER_PACKET_DIRECTORY
                if packet_kind == "carrier_image"
                else "groot_oscar_thin_remote_build"
            )
            subprocess.run(
                [
                    "scp",
                    *options,
                    "root@" + public_ip + f":/root/blueprint-build/run/{packet_directory}/*.json",
                    str(results_dir) + "/",
                ],
                check=True,
            )
            if packet_kind == "carrier_image":
                result_verification = validate_remote_carrier_result(results_dir, packet=packet)
                verification_name = "remote_carrier_build_result_verification.json"
            else:
                result_verification = validate_remote_build_results(results_dir)
                verification_name = "remote_build_results_verification.json"
        else:
            remote_results = "/root/blueprint-build/run/groot_oscar_model_cache_s3_remote/results"
            retrievals: list[dict[str, Any]] = []

            def retrieve(remote_path: str, local_path: Path) -> None:
                try:
                    copy = subprocess.run(
                        [
                            "scp",
                            *options,
                            f"root@{public_ip}:{remote_path}",
                            str(local_path),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    exit_code = copy.returncode
                except Exception as exc:  # noqa: BLE001 - preserve other artifacts
                    exit_code = None
                    error_type = type(exc).__name__
                else:
                    error_type = None
                retrievals.append(
                    {
                        "remote_basename": Path(remote_path).name,
                        "local_basename": local_path.name,
                        "exit_code": exit_code,
                        "retrieved_regular_file": local_path.is_file()
                        and not local_path.is_symlink(),
                        "error_type": error_type,
                    }
                )

            for result_name in MODEL_CACHE_RESULT_FILES:
                retrieve(f"{remote_results}/{result_name}", results_dir / result_name)
            for evidence_name in (
                "model_cache_archive_verification.json",
                "model_cache_dependency_verification.json",
            ):
                retrieve(f"/root/blueprint-build/{evidence_name}", output / evidence_name)
            write_json(
                output / "model_cache_artifact_retrieval.json",
                {
                    "schema_version": "groot_oscar_model_cache_artifact_retrieval.v1",
                    "status": (
                        "complete"
                        if all(row["retrieved_regular_file"] for row in retrievals)
                        else "partial"
                    ),
                    "artifacts": retrievals,
                    "raw_secret_values_recorded": False,
                },
            )
            archive_path = output / "model_cache_archive_verification.json"
            dependency_path = output / "model_cache_dependency_verification.json"
            archive_verification = _load_object(archive_path) if archive_path.is_file() else {}
            dependency_verification = (
                _load_object(dependency_path) if dependency_path.is_file() else {}
            )
            if (
                archive_verification.get("status") != "verified"
                or archive_verification.get("tarball_sha256") != packet_tarball_sha256
                or archive_verification.get("expected_member_map_sha256")
                != packet.get("archive_member_manifest_sha256")
                or dependency_verification.get("status") != "verified"
            ):
                raise RuntimeError("digitalocean_model_cache_remote_preimport_unverified")
            result_verification = validate_remote_model_cache_results(
                results_dir,
                packet=packet,
                parent_binding=binding,
                droplet_id=droplet_id,
            )
            verification_name = "remote_model_cache_results_verification.json"
        write_json(output / verification_name, result_verification)
        if result_verification["status"] != "verified":
            raise RuntimeError(
                "digitalocean_remote_build_results_unverified:"
                + ",".join(result_verification["blockers"])
            )
        build_status = "completed" if build_exit == 0 else "failed"
    except Exception as exc:
        write_json(
            output / "builder_error.json",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        build_status = "failed"
    finally:
        if capability_path is not None:
            try:
                capability_path.unlink(missing_ok=True)
            except OSError:
                local_capability_cleanup_verified = False
            else:
                local_capability_cleanup_verified = (
                    local_capability_cleanup_verified and not capability_path.exists()
                )
        teardown = _delete_with_fail_closed_evidence(token=token, droplet_id=droplet_id)
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
            if build_status == "completed"
            and teardown["provider_absence_confirmed"]
            and local_capability_cleanup_verified
            else "failed"
        ),
        "blockers": [
            *(
                []
                if build_status == "completed"
                else [
                    (
                        "remote_model_cache_preparation_failed"
                        if packet_kind == "model_cache_s3"
                        else (
                            "remote_carrier_image_build_failed"
                            if packet_kind == "carrier_image"
                            else "remote_thin_image_build_failed"
                        )
                    )
                ]
            ),
            *(
                []
                if teardown["provider_absence_confirmed"]
                else ["digitalocean_builder_teardown_unverified"]
            ),
            *(
                []
                if local_capability_cleanup_verified
                else ["digitalocean_builder_local_capability_cleanup_unverified"]
            ),
        ],
        "droplet_id": droplet_id,
        "build_exit_code": build_exit,
        "source_commit": packet["source_commit"],
        "packet_kind": packet_kind,
        "provider_volume_id": packet.get("provider_volume_id"),
        "model_manifest_digest": (
            result_verification.get("model_manifest_digest")
            if build_status == "completed" and packet_kind == "model_cache_s3"
            else None
        ),
        "outer_volume_deletion_required": packet_kind == "model_cache_s3"
        and not (
            build_status == "completed"
            and teardown["provider_absence_confirmed"]
            and local_capability_cleanup_verified
        ),
        "local_capability_cleanup_verified": local_capability_cleanup_verified,
        "remote_secret_cleanup_proven_by_droplet_absence": packet_kind == "model_cache_s3"
        and teardown["provider_absence_confirmed"],
        "provider_absence_confirmed": teardown["provider_absence_confirmed"],
        "maximum_compute_spend_usd": teardown["maximum_compute_spend_usd"],
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "image_build_is_not_model_cache_verification": packet_kind
            in {"thin_release", "carrier_image"},
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
