"""Canonical guarded GPU job for the owned G1 microwave GR00T fine-tune."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request
import uuid
import zipfile

from .common import ensure_dir, write_json
from .g1_microwave_finetune_provider_bundle import (
    BUNDLE_URL_ENV,
    CHECKPOINT_PART_PUT_URLS_ENV,
    CHECKPOINT_PUT_URL_ENV,
    IMAGE_REF,
    OUTPUT_PUT_URL_ENV,
    render_provider_bootstrap,
)
from .g1_microwave_groot_finetune_component import REMOTE_FINAL_CHECKPOINT
from .gpu_render_providers import RenderLaunchSpec, get_render_provider
from .groot_oscar_runpod_watchdog import arm_watchdog, terminate_canary_resources
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    require_paid_resource_admission,
)
from .qualification_control_admission import admit_qualification_control_mutation
from .single_g1_kitchen_qualification_admission import qualification_pre_spend_preflight
from .wam_provider_object_store import signed_output_object_binding_sha256


SCHEMA_VERSION = "g1_microwave_finetune_provider_job.v1"
PROBE_KIND = "single-kitchen-finetune"
SUPPORTED_PROVIDERS = ("runpod", "vast")
# Startup allocation has been flaky across both A40 and L40S pools.  A40 is the
# only pool that has nevertheless exposed a real CUDA runtime, loaded both
# sealed checkpoint shards, and returned a training traceback for this lane.
# Pin to that proven 48 GB runtime while staying below the paid-lane rate cap.
GPU_TYPES = ("NVIDIA A40",)
MAX_HOURLY_RATE_USD = 1.10
HARD_WALL_SECONDS = 7_500
WATCH_SECONDS = 7_200
POLL_SECONDS = 30
# The sealed GR00T+SONIC image is 47.1 GB compressed (35 layers; largest
# layer 14.1 GB).  An uncached secure-cloud host can legitimately remain in
# provider image-pull state for longer than 20 minutes.  This stays bounded by
# the independent 7,500-second hard-wall watchdog and does not relax any
# training, output, or teardown gate.
STARTUP_TIMEOUT_SECONDS = 2_400
MIN_GPU_RAM_MB = 40_000
TERMINAL_PROVIDER_STATUSES = {"EXITED", "STOPPED", "TERMINATED", "FAILED", "DEAD"}
RUNNING_PROVIDER_STATUSES = {"ACTIVE", "RUNNING"}
MAX_CHECKPOINT_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
MAX_CHECKPOINT_ARCHIVE_MEMBERS = 20_000
MAX_LOCAL_CHECKPOINT_COLLECTION_BYTES = 4 * 1024 * 1024 * 1024
MAX_OUTPUT_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_OUTPUT_ARCHIVE_MEMBERS = 100_000
MAX_OUTPUT_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
VAST_CHECKPOINT_PYTHON = "/opt/gr00t-venv/bin/python"
DEFAULT_QUALIFICATION_IDENTITY_FILE = "~/.ssh/id_ed25519"


def _inventory_scope_excluding_bound_instance(
    inventory: Mapping[str, Any],
    *,
    bound_instance_id: str = "",
) -> dict[str, Any]:
    """Derive a fail-closed launch/teardown scope from global provider inventory."""

    blockers: list[str] = []
    raw_resources = inventory.get("resources")
    resources: list[dict[str, Any]] = []
    if inventory.get("api_confirmed") is not True:
        blockers.append("provider_inventory_api_unconfirmed")
    if not isinstance(raw_resources, list):
        blockers.append("provider_inventory_resources_invalid")
    else:
        resources = [dict(row) for row in raw_resources if isinstance(row, Mapping)]
        if len(resources) != len(raw_resources):
            blockers.append("provider_inventory_resource_row_invalid")
    source_live_count = inventory.get("live_resource_count")
    if (
        isinstance(source_live_count, bool)
        or not isinstance(source_live_count, int)
        or source_live_count != len(resources)
    ):
        blockers.append("provider_inventory_live_count_inconsistent")

    normalized_bound_id = str(bound_instance_id or "").strip()
    bound_rows = (
        [
            row
            for row in resources
            if str(row.get("instance_id") or "") == normalized_bound_id
        ]
        if normalized_bound_id
        else []
    )
    if normalized_bound_id and len(bound_rows) != 1:
        blockers.append("bound_retained_instance_inventory_binding_invalid")
    other_resources = [row for row in resources if row not in bound_rows]
    return {
        "schema_version": "g1_microwave_finetune_inventory_scope.v1",
        "status": "passed" if not blockers else "blocked",
        "api_confirmed": inventory.get("api_confirmed") is True,
        "source_live_resource_count": source_live_count,
        "bound_retained_instance_id": normalized_bound_id or None,
        "bound_retained_instance_present": len(bound_rows) == 1,
        "other_live_resource_count": len(other_resources) if not blockers else None,
        "other_live_resources": other_resources,
        "blockers": sorted(set(blockers)),
    }


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}_missing_or_invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name}_not_object")
    return payload


def _read_secret_url(path: Path, *, name: str) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError(f"{name}_file_missing_or_unsafe")
    value = path.read_text(encoding="utf-8").strip()
    if not value.startswith("https://"):
        raise ValueError(f"{name}_invalid")
    return value


def _bundle_evidence(bundle_path: Path) -> dict[str, Any]:
    report_path = bundle_path.with_suffix(bundle_path.suffix + ".json")
    report = _load_mapping(report_path, name="finetune_provider_bundle_report")
    expected = str((report.get("bundle") or {}).get("sha256") or "")
    if (
        report.get("status") != "qualified_provider_bundle"
        or report.get("image_ref") != IMAGE_REF
        or not bundle_path.is_file()
    ):
        raise ValueError("finetune_provider_bundle_not_qualified")
    import hashlib

    actual = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("finetune_provider_bundle_sha256_mismatch")
    return report


def _staging_evidence(
    stage_dir: Path,
    bundle_path: Path,
    *,
    output_put_url: str | None = None,
    output_get_url: str | None = None,
) -> dict[str, Any]:
    manifest = _load_mapping(
        stage_dir / "wam_provider_object_store_staging_manifest.json",
        name="finetune_object_store_staging_manifest",
    )
    round_trip = manifest.get("signed_output_round_trip") or {}
    output_binding = str(manifest.get("output_url_object_binding_sha256") or "").lower()
    put_url = output_put_url or _read_secret_url(
        stage_dir / "provider_output_put_url.txt", name="finetune_staging_output_put_url"
    )
    get_url = output_get_url or _read_secret_url(
        stage_dir / "provider_output_get_url.txt", name="finetune_staging_output_get_url"
    )
    try:
        current_output_binding = signed_output_object_binding_sha256(put_url, get_url)
    except ValueError as exc:
        raise ValueError("finetune_object_store_staging_not_qualified") from exc
    if (
        manifest.get("status") != "completed"
        or Path(str(manifest.get("bundle_path") or "")).resolve() != bundle_path.resolve()
        or int(manifest.get("bundle_size_bytes") or -1) != bundle_path.stat().st_size
        or round_trip.get("status") != "passed"
        or len(output_binding) != 64
        or any(character not in "0123456789abcdef" for character in output_binding)
        or output_binding != current_output_binding
        or manifest.get("raw_secret_values_recorded") is not False
    ):
        raise ValueError("finetune_object_store_staging_not_qualified")
    return manifest


def _safe_extract_output(archive_path: Path, destination: Path) -> Path:
    snapshot = Path(tempfile.mkdtemp(prefix=".finetune-output-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_OUTPUT_ARCHIVE_MEMBERS:
                raise ValueError("finetune_output_archive_member_count_invalid")
            total_size = 0
            for member in members:
                relative = Path(member.filename)
                target = (snapshot / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(
                    snapshot.resolve()
                ):
                    raise ValueError("finetune_output_archive_member_unsafe")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError("finetune_output_archive_link_forbidden")
                total_size += int(member.file_size)
                if total_size > MAX_OUTPUT_UNCOMPRESSED_BYTES:
                    raise ValueError("finetune_output_archive_uncompressed_size_invalid")
            archive.extractall(snapshot)
        _load_mapping(
            snapshot / "g1_microwave_finetune_worker_report.json",
            name="finetune_worker_report",
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(snapshot), str(destination))
        return destination
    except BaseException:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_checkpoint(archive_path: Path, destination: Path) -> dict[str, Any]:
    """Extract one hash-bound model tree without trusting archive paths or links."""

    snapshot = Path(tempfile.mkdtemp(prefix=".finetune-checkpoint-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_CHECKPOINT_ARCHIVE_MEMBERS:
                raise ValueError("finetune_checkpoint_archive_member_count_invalid")
            total_size = 0
            for member in members:
                relative = Path(member.filename)
                target = (snapshot / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(
                    snapshot.resolve()
                ):
                    raise ValueError("finetune_checkpoint_archive_member_unsafe")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError("finetune_checkpoint_archive_link_forbidden")
                total_size += int(member.file_size)
                if total_size > MAX_CHECKPOINT_ARCHIVE_BYTES:
                    raise ValueError("finetune_checkpoint_archive_uncompressed_size_invalid")
            archive.extractall(snapshot)
        candidates = []
        for config in snapshot.rglob("config.json"):
            model = config.parent
            if config.is_file() and not config.is_symlink() and any(
                path.is_file() and not path.is_symlink()
                for path in model.glob("*.safetensors")
            ):
                candidates.append(model)
        expected_numbered = snapshot / "checkpoint-500"
        numbered = [model for model in candidates if model.name.startswith("checkpoint-")]
        if expected_numbered in candidates and numbered == [expected_numbered]:
            # Legacy workers archived both the output-root mirror and the
            # canonical numbered checkpoint. Bind to the exact bounded final
            # step and ignore only that known root mirror.
            model = expected_numbered
        elif candidates == [snapshot]:
            # Current workers archive checkpoint-500's contents directly.
            model = snapshot
        else:
            raise ValueError("finetune_checkpoint_model_tree_not_single_final_step")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(snapshot), str(destination))
        installed_model = destination / model.relative_to(snapshot)
        inventory = []
        for path in sorted(item for item in installed_model.rglob("*") if item.is_file()):
            if path.is_symlink():
                raise ValueError("finetune_checkpoint_extracted_link_forbidden")
            inventory.append(
                {
                    "relative_path": path.relative_to(installed_model).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        return {
            "checkpoint_path": str(installed_model),
            "checkpoint_files": inventory,
            "checkpoint_tree_sha256": hashlib.sha256(
                json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    except BaseException:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def _collect_checkpoint(
    *,
    get_urls: Sequence[str],
    output_dir: Path,
    worker_report: Mapping[str, Any],
    vast_target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = worker_report.get("checkpoint_archive")
    checkpoint = dict(checkpoint) if isinstance(checkpoint, Mapping) else {}
    upload = checkpoint.get("upload")
    upload = dict(upload) if isinstance(upload, Mapping) else {}
    expected_sha = str(checkpoint.get("sha256") or "").lower()
    try:
        expected_size = int(checkpoint.get("size_bytes"))
    except (TypeError, ValueError):
        expected_size = -1
    transport = str(upload.get("transport") or "single_object")
    part_rows = upload.get("parts")
    part_rows = list(part_rows) if isinstance(part_rows, list) else []
    if (
        upload.get("status") != "passed"
        or len(expected_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha)
        or expected_size <= 0
        or expected_size > MAX_CHECKPOINT_ARCHIVE_BYTES
    ):
        return {
            "status": "blocked",
            "blockers": ["g1_microwave_finetune_checkpoint_binding_invalid"],
        }
    download_plan: list[tuple[str, int, str]] = []
    if transport == "ordered_parts":
        for index, row in enumerate(part_rows):
            part = dict(row) if isinstance(row, Mapping) else {}
            try:
                size = int(part.get("size_bytes"))
            except (TypeError, ValueError):
                size = -1
            digest = str(part.get("sha256") or "").lower()
            if (
                part.get("part_number") != index + 1
                or size <= 0
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or index >= len(get_urls)
            ):
                return {
                    "status": "blocked",
                    "blockers": ["g1_microwave_finetune_checkpoint_parts_invalid"],
                }
            download_plan.append((get_urls[index], size, digest))
        if not download_plan or sum(row[1] for row in download_plan) != expected_size:
            return {
                "status": "blocked",
                "blockers": ["g1_microwave_finetune_checkpoint_parts_invalid"],
            }
    elif transport == "single_object" and get_urls:
        download_plan = [(get_urls[0], expected_size, expected_sha)]
    else:
        return {
            "status": "blocked",
            "blockers": ["g1_microwave_finetune_checkpoint_transport_invalid"],
        }
    if vast_target:
        return _stream_checkpoint_to_vast(
            download_plan=download_plan,
            expected_sha=expected_sha,
            expected_size=expected_size,
            target=vast_target,
        )
    if (
        transport == "ordered_parts"
        and expected_size > MAX_LOCAL_CHECKPOINT_COLLECTION_BYTES
    ):
        probes = []
        try:
            for index, (get_url, part_size, part_sha) in enumerate(download_plan):
                request = urllib.request.Request(
                    get_url,
                    headers={"Range": "bytes=0-0"},
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    status = int(response.status)
                    first_byte = response.read(1)
                    content_range = str(response.headers.get("Content-Range") or "")
                    content_length = int(response.headers.get("Content-Length") or -1)
                if status == 206 and "/" in content_range:
                    observed_size = int(content_range.rsplit("/", 1)[-1])
                elif status == 200:
                    observed_size = content_length
                else:
                    raise ValueError("finetune_checkpoint_part_probe_status_invalid")
                if len(first_byte) != 1 or observed_size != part_size:
                    raise ValueError("finetune_checkpoint_part_probe_size_invalid")
                probes.append(
                    {
                        "part_number": index + 1,
                        "size_bytes": part_size,
                        "sha256": part_sha,
                        "http_status": status,
                        "object_size_verified": True,
                    }
                )
            return {
                "status": "completed",
                "collection_mode": "object_store_bound_ordered_parts",
                "checkpoint_object_store_bound": True,
                "checkpoint_host_collected": False,
                "archive_size_bytes": expected_size,
                "archive_sha256": expected_sha,
                "parts": probes,
                "raw_signed_urls_recorded": False,
                "blockers": [],
            }
        except (OSError, ValueError, urllib.error.URLError) as exc:
            return {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "blockers": ["g1_microwave_finetune_checkpoint_part_probe_failed"],
            }
    archive_path = output_dir.parent / "g1_microwave_finetune_checkpoint.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    try:
        copied = 0
        aggregate_digest = hashlib.sha256()
        with temporary.open("wb") as handle:
            for get_url, part_size, part_sha in download_plan:
                part_copied = 0
                part_digest = hashlib.sha256()
                with urllib.request.urlopen(get_url, timeout=3_600) as response:
                    if int(response.status) != 200:
                        raise ValueError("finetune_checkpoint_download_status_invalid")
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        part_copied += len(chunk)
                        if copied > expected_size or copied > MAX_CHECKPOINT_ARCHIVE_BYTES:
                            raise ValueError("finetune_checkpoint_download_size_invalid")
                        handle.write(chunk)
                        aggregate_digest.update(chunk)
                        part_digest.update(chunk)
                if part_copied != part_size or part_digest.hexdigest() != part_sha:
                    raise ValueError("finetune_checkpoint_part_binding_mismatch")
        os.replace(temporary, archive_path)
        if (
            archive_path.stat().st_size != expected_size
            or aggregate_digest.hexdigest() != expected_sha
        ):
            raise ValueError("finetune_checkpoint_download_binding_mismatch")
        extracted = _safe_extract_checkpoint(archive_path, output_dir)
        return {
            "status": "completed",
            "archive_path": str(archive_path),
            "archive_size_bytes": expected_size,
            "archive_sha256": expected_sha,
            **extracted,
            "blockers": [],
        }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "blockers": ["g1_microwave_finetune_checkpoint_collection_failed"],
        }
    finally:
        temporary.unlink(missing_ok=True)


def _collect_checkpoint_with_vast_admission(
    *,
    get_urls: Sequence[str],
    output_dir: Path,
    worker_report: Mapping[str, Any],
    vast_target: Mapping[str, Any] | None = None,
    admission_out: str | Path | None = None,
) -> dict[str, Any]:
    """Admit a retained-session mutation immediately before checkpoint streaming."""

    admission_path = (
        str(Path(admission_out).expanduser().resolve())
        if admission_out not in {None, ""}
        else None
    )
    if vast_target:
        try:
            admit_qualification_control_mutation(
                admission_out,
                dict(vast_target.get("_admission_manifest") or {}),
                dict(vast_target.get("_admission_inspection") or {}),
                str(vast_target.get("instance_id") or ""),
                "install-checkpoint",
                "groot_microwave_finetune",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "status": "blocked",
                "qualification_control_admission_passed": False,
                "qualification_control_admission_path": admission_path,
                "blockers": [str(exc)],
            }
    collection = _collect_checkpoint(
        get_urls=get_urls,
        output_dir=output_dir,
        worker_report=worker_report,
        vast_target=vast_target,
    )
    if vast_target:
        return {
            **collection,
            "qualification_control_admission_passed": True,
            "qualification_control_admission_path": admission_path,
        }
    return collection


def _checkpoint_receiver_script() -> str:
    """Return the fixed remote receiver used for a disk-free checkpoint handoff."""

    return r'''import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import zipfile

expected_sha = sys.argv[1]
expected_size = int(sys.argv[2])
destination = Path(sys.argv[3]).resolve()
allowed_root = Path("/workspace/microwave_finetune").resolve()
if destination.parent != allowed_root or destination.name != "checkpoint-500":
    raise ValueError("checkpoint_destination_not_fixed")
allowed_root.mkdir(parents=True, exist_ok=True)
archive_path = allowed_root / ".checkpoint-transfer.zip"
snapshot = allowed_root / ".checkpoint-500-receiving"
if snapshot.exists():
    shutil.rmtree(snapshot)
snapshot.mkdir()
digest = hashlib.sha256()
copied = 0
try:
    with archive_path.open("wb") as handle:
        while True:
            chunk = sys.stdin.buffer.read(8 * 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > expected_size or copied > 64 * 1024 * 1024 * 1024:
                raise ValueError("checkpoint_stream_size_invalid")
            digest.update(chunk)
            handle.write(chunk)
    if copied != expected_size or digest.hexdigest() != expected_sha:
        raise ValueError("checkpoint_stream_binding_mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if not members or len(members) > 20000:
            raise ValueError("checkpoint_archive_member_count_invalid")
        total_size = 0
        snapshot_root = snapshot.resolve()
        for member in members:
            relative = Path(member.filename)
            target = (snapshot / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(snapshot_root):
                raise ValueError("checkpoint_archive_member_unsafe")
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError("checkpoint_archive_link_forbidden")
            total_size += int(member.file_size)
            if total_size > 64 * 1024 * 1024 * 1024:
                raise ValueError("checkpoint_archive_uncompressed_size_invalid")
        archive.extractall(snapshot)
    candidates = []
    for config in snapshot.rglob("config.json"):
        model = config.parent
        if config.is_file() and not config.is_symlink() and any(
            path.is_file() and not path.is_symlink() for path in model.glob("*.safetensors")
        ):
            candidates.append(model)
    expected_numbered = snapshot / "checkpoint-500"
    numbered = [model for model in candidates if model.name.startswith("checkpoint-")]
    if expected_numbered in candidates and numbered == [expected_numbered]:
        model = expected_numbered
    elif candidates == [snapshot]:
        model = snapshot
    else:
        raise ValueError("checkpoint_model_tree_not_single_final_step")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(model), str(destination))
    inventory = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError("checkpoint_extracted_link_forbidden")
        item_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                item_digest.update(chunk)
        inventory.append({
            "relative_path": path.relative_to(destination).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": item_digest.hexdigest(),
        })
    tree_sha = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    print(json.dumps({
        "status": "completed",
        "checkpoint_path": str(destination),
        "checkpoint_files": inventory,
        "checkpoint_tree_sha256": tree_sha,
        "archive_sha256": expected_sha,
        "archive_size_bytes": expected_size,
    }, sort_keys=True))
finally:
    archive_path.unlink(missing_ok=True)
    if snapshot.exists():
        shutil.rmtree(snapshot)
'''


def _load_vast_checkpoint_target(
    session_manifest: str | Path,
    *,
    identity_file: str | Path = DEFAULT_QUALIFICATION_IDENTITY_FILE,
) -> dict[str, Any]:
    manifest_path = Path(session_manifest).expanduser().resolve()
    manifest = _load_mapping(manifest_path, name="finetune_checkpoint_vast_session")
    if manifest.get("provider") != "vast" or manifest.get("continuing_spend") is not True:
        raise ValueError("finetune_checkpoint_vast_session_not_live")
    instance_id = str(manifest.get("instance_id") or "")
    resource_name = str(manifest.get("resource_name") or "")
    connection = dict(manifest.get("ssh_connection") or {})
    host_key = dict(manifest.get("ssh_host_key") or {})
    known_hosts = Path(str(host_key.get("known_hosts_file") or "")).expanduser().resolve()
    host = str(connection.get("ssh_host") or "")
    try:
        port = int(connection.get("ssh_port"))
    except (TypeError, ValueError) as exc:
        raise ValueError("finetune_checkpoint_vast_ssh_port_invalid") from exc
    inspected = get_render_provider("vast").inspect(instance_id)
    if (
        inspected.get("status") != "observed"
        or inspected.get("actual_status") != "running"
        or inspected.get("name") != resource_name
        or inspected.get("ssh_host") != host
        or int(inspected.get("ssh_port") or -1) != port
        or not known_hosts.is_file()
    ):
        raise ValueError("finetune_checkpoint_vast_target_not_observed")
    return {
        "instance_id": instance_id,
        "resource_name": resource_name,
        "ssh_host": host,
        "ssh_port": port,
        "known_hosts_file": str(known_hosts),
        "identity_file": str(Path(identity_file).expanduser().resolve()),
        "checkpoint_path": REMOTE_FINAL_CHECKPOINT,
        "_admission_manifest": manifest,
        "_admission_inspection": inspected,
    }


def _stream_checkpoint_to_vast(
    *,
    download_plan: Sequence[tuple[str, int, str]],
    expected_sha: str,
    expected_size: int,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    script_b64 = base64.b64encode(_checkpoint_receiver_script().encode()).decode("ascii")
    remote_command = shlex.join(
        [
            VAST_CHECKPOINT_PYTHON,
            "-c",
            f"import base64;exec(base64.b64decode({script_b64!r}))",
            expected_sha,
            str(expected_size),
            str(target["checkpoint_path"]),
        ]
    )
    argv = [
        "ssh",
        "-i",
        str(target["identity_file"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={target['known_hosts_file']}",
        "-p",
        str(target["ssh_port"]),
        f"root@{target['ssh_host']}",
        remote_command,
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise OSError("checkpoint_vast_stream_pipes_missing")
        digest = hashlib.sha256()
        copied = 0
        for get_url, part_size, part_sha in download_plan:
            part_copied = 0
            part_digest = hashlib.sha256()
            with urllib.request.urlopen(get_url, timeout=3_600) as response:
                if int(response.status) != 200:
                    raise ValueError("finetune_checkpoint_download_status_invalid")
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    part_copied += len(chunk)
                    if copied > expected_size or copied > MAX_CHECKPOINT_ARCHIVE_BYTES:
                        raise ValueError("finetune_checkpoint_download_size_invalid")
                    digest.update(chunk)
                    part_digest.update(chunk)
                    process.stdin.write(chunk)
            if part_copied != part_size or part_digest.hexdigest() != part_sha:
                raise ValueError("finetune_checkpoint_part_binding_mismatch")
        process.stdin.close()
        stdout = process.stdout.read().decode("utf-8", errors="replace")
        process.stderr.read()
        returncode = process.wait(timeout=3_600)
        if copied != expected_size or digest.hexdigest() != expected_sha:
            raise ValueError("finetune_checkpoint_download_binding_mismatch")
        if returncode != 0:
            raise ValueError("finetune_checkpoint_vast_receiver_failed")
        payload = json.loads(stdout.strip().splitlines()[-1])
        if (
            payload.get("status") != "completed"
            or payload.get("archive_sha256") != expected_sha
            or int(payload.get("archive_size_bytes") or -1) != expected_size
            or payload.get("checkpoint_path") != REMOTE_FINAL_CHECKPOINT
        ):
            raise ValueError("finetune_checkpoint_vast_receipt_invalid")
        return {
            **payload,
            "status": "completed",
            "archive_streamed_direct_to_vast": True,
            "vast_instance_id": target["instance_id"],
            "vast_resource_name": target["resource_name"],
            "blockers": [],
        }
    except (BrokenPipeError, OSError, ValueError, json.JSONDecodeError) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=30)
        return {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "blockers": ["g1_microwave_finetune_checkpoint_vast_stream_failed"],
        }


def _collect_output(
    *,
    get_url: str,
    output_dir: Path,
    max_seconds: int,
    provider: Any | None = None,
    instance_id: str = "",
) -> dict[str, Any]:
    started = time.monotonic()
    archive_path = output_dir.parent / "g1_microwave_finetune_output.zip"
    last_http = 0
    runtime_seen = False
    last_provider_inspection: dict[str, Any] = {}
    while time.monotonic() - started < max_seconds:
        temporary = archive_path.with_suffix(".zip.tmp")
        try:
            with urllib.request.urlopen(get_url, timeout=120) as response, temporary.open(
                "wb"
            ) as handle:
                last_http = int(response.status)
                copied = 0
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > MAX_OUTPUT_ARCHIVE_BYTES:
                        raise ValueError("finetune_output_archive_download_size_invalid")
                    handle.write(chunk)
            os.replace(temporary, archive_path)
            extracted = _safe_extract_output(archive_path, output_dir)
            report = _load_mapping(
                extracted / "g1_microwave_finetune_worker_report.json",
                name="finetune_worker_report",
            )
            return {
                "status": "completed" if report.get("status") == "completed" else "blocked",
                "worker_report": report,
                "archive_path": str(archive_path),
                "archive_size_bytes": archive_path.stat().st_size,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "blockers": list(report.get("blockers") or []),
            }
        except urllib.error.HTTPError as exc:
            last_http = int(exc.code)
            if exc.code not in {403, 404}:
                break
        except (OSError, ValueError, zipfile.BadZipFile):
            break
        finally:
            temporary.unlink(missing_ok=True)
        elapsed = time.monotonic() - started
        inspect = getattr(provider, "inspect", None)
        if callable(inspect) and instance_id:
            observed = inspect(instance_id)
            if isinstance(observed, Mapping):
                last_provider_inspection = dict(observed)
                runtime_ready = observed.get("runtime_ready")
                if runtime_ready is None:
                    runtime_ready = observed.get("runtime_present")
                provider_states = {
                    str(observed.get(key) or "").upper()
                    for key in ("desiredStatus", "actual_status", "cur_state")
                }
                runtime_ready = runtime_ready is True or bool(
                    provider_states & RUNNING_PROVIDER_STATUSES
                )
                runtime_seen = bool(runtime_seen or runtime_ready is True)
                provider_status = str(observed.get("desiredStatus") or "").upper()
                if observed.get("error") or provider_status in TERMINAL_PROVIDER_STATUSES:
                    return {
                        "status": "blocked",
                        "elapsed_seconds": round(elapsed, 3),
                        "last_http_status": last_http,
                        "runtime_seen": runtime_seen,
                        "last_provider_inspection": last_provider_inspection,
                        "blockers": [
                            "g1_microwave_finetune_provider_runtime_terminated_before_output"
                        ],
                    }
        if elapsed >= STARTUP_TIMEOUT_SECONDS and not runtime_seen:
            return {
                "status": "blocked",
                "elapsed_seconds": round(elapsed, 3),
                "last_http_status": last_http,
                "runtime_seen": False,
                "last_provider_inspection": last_provider_inspection,
                "blockers": [
                    "g1_microwave_finetune_provider_runtime_not_ready_before_startup_deadline"
                ],
            }
        time.sleep(POLL_SECONDS)
    return {
        "status": "blocked",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "last_http_status": last_http,
        "runtime_seen": runtime_seen,
        "last_provider_inspection": last_provider_inspection,
        "blockers": ["g1_microwave_finetune_output_not_collected_before_deadline"],
    }


def resume_checkpoint_transfer_to_vast(
    *,
    provider_bundle: str | Path,
    checkpoint_object_store_stage_dirs: Sequence[str | Path],
    worker_report_path: str | Path,
    checkpoint_vast_session_manifest: str | Path,
    admission_out: str | Path | None,
    adapter_output: str | Path,
    execute: bool = False,
    qualification_identity_file: str | Path = DEFAULT_QUALIFICATION_IDENTITY_FILE,
) -> dict[str, Any]:
    """Install an already-qualified object-store checkpoint on live Vast."""

    result_path = Path(adapter_output).expanduser().resolve()
    ensure_dir(result_path.parent)
    blockers: list[str] = []
    bundle_path = Path(provider_bundle).expanduser().resolve()
    report_path = Path(worker_report_path).expanduser().resolve()
    get_urls: list[str] = []
    bundle: dict[str, Any] = {}
    worker_report: dict[str, Any] = {}
    target: dict[str, Any] = {}
    try:
        bundle = _bundle_evidence(bundle_path)
        worker_report = _load_mapping(
            report_path,
            name="finetune_worker_report",
        )
        open_loop = dict(worker_report.get("open_loop_qualification") or {})
        if (
            worker_report.get("status") != "completed"
            or open_loop.get("status") != "passed"
            or worker_report.get("blockers")
        ):
            raise ValueError("finetune_worker_report_not_qualified")
        for index, value in enumerate(checkpoint_object_store_stage_dirs, start=1):
            stage = Path(value).expanduser().resolve()
            put_url = _read_secret_url(
                stage / "provider_output_put_url.txt",
                name=f"finetune_checkpoint_part_{index}_put_url",
            )
            get_url = _read_secret_url(
                stage / "provider_output_get_url.txt",
                name=f"finetune_checkpoint_part_{index}_get_url",
            )
            _staging_evidence(
                stage,
                bundle_path,
                output_put_url=put_url,
                output_get_url=get_url,
            )
            get_urls.append(get_url)
        if not get_urls:
            raise ValueError("finetune_checkpoint_part_stage_dirs_missing")
        target = _load_vast_checkpoint_target(
            checkpoint_vast_session_manifest,
            identity_file=qualification_identity_file,
        )
    except (OSError, ValueError) as exc:
        blockers.append(str(exc))

    collection: dict[str, Any] = {
        "status": "not_attempted",
        "blockers": ["g1_microwave_finetune_checkpoint_resume_not_ready"],
    }
    if not execute:
        blockers.append("g1_microwave_finetune_checkpoint_resume_execute_required")
    if not blockers:
        collection = _collect_checkpoint_with_vast_admission(
            get_urls=get_urls,
            output_dir=result_path.parent / "resumed_checkpoint",
            worker_report=worker_report,
            vast_target=target,
            admission_out=admission_out,
        )
        blockers.extend(collection.get("blockers") or [])
    result = {
        "schema_version": "g1_microwave_finetune_checkpoint_resume.v1",
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": (bundle.get("bundle") or {}).get("sha256"),
        "worker_report_path": str(report_path),
        "checkpoint_part_count": len(get_urls),
        "vast_instance_id": target.get("instance_id"),
        "vast_resource_name": target.get("resource_name"),
        "checkpoint_collection": collection,
        "runpod_allocation_performed": False,
        "checkpoint_retraining_performed": False,
        "raw_signed_urls_recorded": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "checkpoint_open_loop_qualified": worker_report.get("status")
            == "completed",
            "checkpoint_installed_on_vast": collection.get("status") == "completed",
            "isaac_semantic_episode_success_not_proven": True,
        },
    }
    write_json(result_path, result)
    return result


def run_finetune_job(
    *,
    provider_name: str,
    provider_bundle: str | Path,
    object_store_stage_dir: str | Path,
    checkpoint_object_store_stage_dir: str | Path,
    checkpoint_object_store_part_stage_dirs: Sequence[str | Path] = (),
    release_evidence: str | Path,
    provider_launch_request: str | Path | None = None,
    preflight_bundle: str | Path | None = None,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    pod_name: str,
    execute: bool,
    checkpoint_vast_session_manifest: str | Path | None = None,
    qualification_identity_file: str | Path = DEFAULT_QUALIFICATION_IDENTITY_FILE,
) -> dict[str, Any]:
    result_path = Path(adapter_output).expanduser().resolve()
    root = result_path.parent
    ensure_dir(root)
    bundle_path = Path(provider_bundle).expanduser().resolve()
    stage_dir = Path(object_store_stage_dir).expanduser().resolve()
    checkpoint_stage_dir = (
        Path(checkpoint_object_store_stage_dir).expanduser().resolve()
    )
    checkpoint_part_stage_dirs = [
        Path(value).expanduser().resolve()
        for value in checkpoint_object_store_part_stage_dirs
    ]
    blockers: list[str] = []
    checkpoint_vast_target: dict[str, Any] = {}
    checkpoint_part_staging: list[dict[str, Any]] = []
    if checkpoint_vast_session_manifest not in {None, ""}:
        try:
            checkpoint_vast_target = _load_vast_checkpoint_target(
                checkpoint_vast_session_manifest,
                identity_file=qualification_identity_file,
            )
        except (OSError, ValueError) as exc:
            blockers.append(str(exc))
    try:
        bundle = _bundle_evidence(bundle_path)
        bundle_url = _read_secret_url(
            stage_dir / "provider_bundle_url.txt", name="finetune_provider_bundle_url"
        )
        put_url = _read_secret_url(
            stage_dir / "provider_output_put_url.txt", name="finetune_provider_output_put_url"
        )
        get_url = _read_secret_url(
            stage_dir / "provider_output_get_url.txt", name="finetune_provider_output_get_url"
        )
        staging = _staging_evidence(
            stage_dir,
            bundle_path,
            output_put_url=put_url,
            output_get_url=get_url,
        )
        checkpoint_put_url = _read_secret_url(
            checkpoint_stage_dir / "provider_output_put_url.txt",
            name="finetune_checkpoint_output_put_url",
        )
        checkpoint_get_url = _read_secret_url(
            checkpoint_stage_dir / "provider_output_get_url.txt",
            name="finetune_checkpoint_output_get_url",
        )
        checkpoint_staging = _staging_evidence(
            checkpoint_stage_dir,
            bundle_path,
            output_put_url=checkpoint_put_url,
            output_get_url=checkpoint_get_url,
        )
        checkpoint_part_staging = [checkpoint_staging]
        checkpoint_part_put_urls = [checkpoint_put_url]
        checkpoint_part_get_urls = [checkpoint_get_url]
        for index, part_stage_dir in enumerate(checkpoint_part_stage_dirs, start=2):
            part_put_url = _read_secret_url(
                part_stage_dir / "provider_output_put_url.txt",
                name=f"finetune_checkpoint_part_{index}_put_url",
            )
            part_get_url = _read_secret_url(
                part_stage_dir / "provider_output_get_url.txt",
                name=f"finetune_checkpoint_part_{index}_get_url",
            )
            checkpoint_part_staging.append(
                _staging_evidence(
                    part_stage_dir,
                    bundle_path,
                    output_put_url=part_put_url,
                    output_get_url=part_get_url,
                )
            )
            checkpoint_part_put_urls.append(part_put_url)
            checkpoint_part_get_urls.append(part_get_url)
    except (OSError, ValueError) as exc:
        bundle = {}
        staging = {}
        checkpoint_staging = {}
        bundle_url = put_url = get_url = checkpoint_put_url = checkpoint_get_url = ""
        checkpoint_part_put_urls = []
        checkpoint_part_get_urls = []
        checkpoint_part_staging = []
        blockers.append(str(exc))
    all_stage_dirs = [stage_dir, checkpoint_stage_dir, *checkpoint_part_stage_dirs]
    if len(set(all_stage_dirs)) != len(all_stage_dirs):
        blockers.append("finetune_proof_and_checkpoint_output_channels_must_be_distinct")
    output_bindings = [
        str(manifest.get("output_url_object_binding_sha256") or "").lower()
        for manifest in [staging, *checkpoint_part_staging]
    ]
    if output_bindings and len(set(output_bindings)) != len(output_bindings):
        blockers.append("finetune_proof_and_checkpoint_output_channels_must_be_distinct")
    try:
        release = _load_mapping(
            Path(release_evidence).expanduser().resolve(), name="finetune_release_evidence"
        )
    except ValueError as exc:
        release = {}
        blockers.append(str(exc))
    if release.get("status") != "completed" or release.get("resolved_digest_ref") != IMAGE_REF:
        blockers.append("g1_microwave_finetune_release_evidence_mismatch")

    resolved_provider = str(provider_name).strip().lower()
    if resolved_provider not in SUPPORTED_PROVIDERS:
        blockers.append(f"g1_microwave_finetune_provider_unsupported:{resolved_provider}")
    provider = get_render_provider(resolved_provider)
    bundle_sha = str((bundle.get("bundle") or {}).get("sha256") or "")
    prefix = f"blueprint-groot-oscar-canary-g1-microwave-finetune-{bundle_sha[:10]}"
    resolved_name = pod_name.strip() if pod_name.strip().startswith(prefix) else f"{prefix}-pod"
    request: dict[str, Any] = {}
    bootstrap_sha256 = ""
    capacity: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    inventory_scope: dict[str, Any] = {}
    retained_target_instance_id = (
        str(checkpoint_vast_target.get("instance_id") or "")
        if resolved_provider == "vast" and checkpoint_vast_target
        else ""
    )
    if bundle and staging and not blockers:
        script = render_provider_bootstrap(expected_bundle_sha256=bundle_sha)
        bootstrap_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
        spec = RenderLaunchSpec(
            name=resolved_name,
            image=IMAGE_REF,
            env={
                BUNDLE_URL_ENV: bundle_url,
                OUTPUT_PUT_URL_ENV: put_url,
                CHECKPOINT_PUT_URL_ENV: checkpoint_put_url,
                CHECKPOINT_PART_PUT_URLS_ENV: json.dumps(
                    checkpoint_part_put_urls, separators=(",", ":")
                ),
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "HF_HOME": "/opt/blueprint/hf_home",
                "HF_HUB_CACHE": "/opt/blueprint/hf_home/hub",
                "HUGGINGFACE_HUB_CACHE": "/opt/blueprint/hf_home/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            bootstrap_argv=["-lc", script],
            entrypoint=["bash"],
            container_disk_gb=220,
            volume_gb=80 if resolved_provider == "runpod" else 0,
            gpu_types=GPU_TYPES,
            gpu_count=1,
            min_vcpu=8,
            min_ram_gb=32,
            max_hourly_rate_usd=MAX_HOURLY_RATE_USD,
            min_gpu_ram_mb=MIN_GPU_RAM_MB,
            requires_rtx=False,
        )
        request = provider.build_request(spec, root)
        request["min_gpu_ram_mb"] = MIN_GPU_RAM_MB
        request["requires_rtx"] = False
        inventory = provider.billable_inventory(name_prefix="")
        inventory_scope = _inventory_scope_excluding_bound_instance(
            inventory,
            bound_instance_id=retained_target_instance_id,
        )
        capacity = provider.capacity_preflight(request)
        viable = [
            row
            for row in capacity.get("viable_gpu_types", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("on_demand_price_usd_per_hour"), (int, float))
            and float(row["on_demand_price_usd_per_hour"]) <= MAX_HOURLY_RATE_USD
        ]
        if (
            inventory_scope.get("status") != "passed"
            or inventory_scope.get("other_live_resource_count") != 0
        ):
            blockers.append("g1_microwave_finetune_prelaunch_inventory_not_zero")
        if capacity.get("status") != "available" or not viable:
            blockers.append("g1_microwave_finetune_40gb_capacity_unavailable")
        pre_spend_preflight, pre_spend_blockers = qualification_pre_spend_preflight(
            root=root,
            capacity=capacity,
            pre_inventory={
                "api_confirmed": inventory_scope.get("status") == "passed",
                "live_resource_count": inventory_scope.get("other_live_resource_count"),
            },
            image_ref=IMAGE_REF,
            execute=execute,
            provider=resolved_provider,
        )
        blockers.extend(pre_spend_blockers)
        request["pre_spend_preflight"] = pre_spend_preflight
        request["prelaunch_spend_guard"] = {
            "schema_version": "g1_microwave_finetune_prelaunch_spend_guard.v1",
            "required_before_provider_launch": True,
            "can_launch": not blockers,
            "blockers": sorted(set(blockers)),
            "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
            "maximum_live_seconds": HARD_WALL_SECONDS,
            "maximum_estimated_spend_usd": round(
                MAX_HOURLY_RATE_USD * HARD_WALL_SECONDS / 3600.0, 2
            ),
            "inventory_scope": inventory_scope,
        }

    admission = {
        "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "resource_class": "gpu_render",
        "scope": "one_g1_microwave_groot_sonic_finetune",
        "provider_mutations_performed": 0,
        "maximum_live_seconds": HARD_WALL_SECONDS,
        "maximum_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "maximum_estimated_spend_usd": round(
            MAX_HOURLY_RATE_USD * HARD_WALL_SECONDS / 3600.0, 2
        ),
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    bound = {
        "schema_version": "g1_microwave_finetune_bound_request.v1",
        "status": "bound" if not blockers else "blocked",
        "provider": resolved_provider,
        "pod_name": resolved_name,
        "pod_name_prefix": prefix,
        "image_ref": IMAGE_REF,
        "bundle_sha256": bundle_sha or None,
        "dataset_sha256": (bundle.get("dataset") or {}).get("sha256"),
        "worker_sha256": (bundle.get("worker") or {}).get("sha256"),
        "provider_bootstrap_sha256": bootstrap_sha256 or None,
        "gpu_count": 1,
        "minimum_gpu_ram_mb": MIN_GPU_RAM_MB,
        "requires_rtx": False,
        "signed_bundle_url_present": bool(bundle_url),
        "signed_output_urls_present": bool(put_url and get_url),
        "signed_checkpoint_output_urls_present": bool(
            checkpoint_put_url and checkpoint_get_url
        ),
        "signed_checkpoint_part_count": len(checkpoint_part_put_urls),
        "checkpoint_output_object_binding_sha256": checkpoint_staging.get(
            "output_url_object_binding_sha256"
        ),
        "checkpoint_collection_mode": (
            "stream_verified_direct_to_retained_vast"
            if checkpoint_vast_target
            else (
                "object_store_bound_ordered_parts"
                if len(checkpoint_part_get_urls) > 1
                else "host_download_and_extract"
            )
        ),
        "checkpoint_vast_target": {
            "instance_id": checkpoint_vast_target.get("instance_id"),
            "resource_name": checkpoint_vast_target.get("resource_name"),
            "checkpoint_path": checkpoint_vast_target.get("checkpoint_path"),
        }
        if checkpoint_vast_target
        else None,
        "capacity": capacity,
        "prelaunch_inventory": inventory,
        "prelaunch_spend_guard": request.get("prelaunch_spend_guard"),
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    preflight = {
        "schema_version": "g1_microwave_finetune_preflight.v1",
        "status": "ready" if not blockers else "blocked",
        "provider": resolved_provider,
        "image_ref": IMAGE_REF,
        "bundle_sha256": bundle_sha or None,
        "provider_bootstrap_sha256": bootstrap_sha256 or None,
        "object_store_staging_status": staging.get("status"),
        "checkpoint_object_store_staging_status": checkpoint_staging.get("status"),
        "capacity": capacity,
        "prelaunch_inventory": inventory,
        "maximum_live_seconds": HARD_WALL_SECONDS,
        "maximum_estimated_spend_usd": admission["maximum_estimated_spend_usd"],
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    write_json(Path(admission_out), admission)
    write_json(Path(bound_request_out), bound)
    if provider_launch_request:
        write_json(Path(provider_launch_request), bound)
    if preflight_bundle:
        write_json(Path(preflight_bundle), preflight)
    if not execute or blockers:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run_ready" if not blockers else "blocked",
            "execute": bool(execute),
            "admission": admission,
            "bound_request": bound,
            "provider_mutations_performed": 0,
            "blockers": sorted(set(blockers)),
        }
        write_json(result_path, result)
        return result

    grant = require_paid_resource_admission(
        admission,
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    deadline = time.time() + HARD_WALL_SECONDS
    armed = arm_watchdog(
        out_dir=root,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_name=resolved_provider,
    )
    watchdog = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--out-dir",
            str(root),
            "--pod-name-prefix",
            prefix,
            "--deadline-epoch",
            str(deadline),
            "--provider",
            resolved_provider,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    run_id = f"g1-microwave-finetune-{uuid.uuid4().hex[:10]}"
    pending = open_pending_teardown(
        provider=resolved_provider,
        lane=PROBE_KIND,
        run_id=run_id,
        resource_name=resolved_name,
        job_dir=root,
        max_age_seconds=HARD_WALL_SECONDS,
    )
    launch: dict[str, Any] = {}
    collection: dict[str, Any] = {}
    checkpoint_collection: dict[str, Any] = {
        "status": "not_attempted",
        "blockers": ["g1_microwave_finetune_checkpoint_output_not_ready"],
    }
    teardown: dict[str, Any] = {}
    try:
        launch = provider.launch(
            root,
            request,
            cold=True,
            paid_resource_admission_grant=grant,
        )
        instance_id = str(launch.get("instance_id") or "")
        if instance_id:
            bind_pending_teardown_instance(pending["path"], instance_id)
            collection = _collect_output(
                get_url=get_url,
                output_dir=root / "collected_output",
                max_seconds=WATCH_SECONDS,
                provider=provider,
                instance_id=instance_id,
            )
            if collection.get("status") == "completed":
                checkpoint_collection = _collect_checkpoint_with_vast_admission(
                    get_urls=checkpoint_part_get_urls,
                    output_dir=root / "collected_checkpoint",
                    worker_report=collection.get("worker_report") or {},
                    vast_target=checkpoint_vast_target or None,
                    admission_out=(
                        root
                        / "g1_microwave_finetune_checkpoint_install_admission.json"
                    ),
                )
        else:
            mark_pending_teardown_ambiguous(
                pending["path"],
                reason="finetune_provider_launch_returned_without_instance_id",
                evidence=launch,
            )
    except BaseException as exc:  # Provider outcome is ambiguous even on caller cancellation.
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason=f"finetune_launch_or_collection_exception:{type(exc).__name__}",
            evidence=launch,
        )
        collection = {
            "status": "blocked",
            "blockers": [f"finetune_launch_or_collection_exception:{type(exc).__name__}"],
        }
    finally:
        teardown = terminate_canary_resources(
            provider=provider,
            pod_name_prefix=prefix,
            armed=armed,
            provider_name=resolved_provider,
        )
        write_json(root / "g1_microwave_finetune_teardown.json", teardown)

    final_inventory = provider.billable_inventory(name_prefix="")
    final_inventory_scope = _inventory_scope_excluding_bound_instance(
        final_inventory,
        bound_instance_id=retained_target_instance_id,
    )
    trainer_provider_global_zero = bool(
        final_inventory.get("api_confirmed") is True
        and final_inventory.get("live_resource_count") == 0
    )
    absence = bool(
        teardown.get("provider_absence_confirmed") is True
        and final_inventory_scope.get("status") == "passed"
        and final_inventory_scope.get("other_live_resource_count") == 0
    )
    if launch.get("instance_id"):
        close_pending_teardown(
            pending["path"],
            {
                "status": "PASS" if absence else "FAIL",
                "provider_absence_confirmed": absence,
                "teardown": teardown,
                "final_inventory": final_inventory,
                "final_inventory_scope": final_inventory_scope,
            },
        )
    elif not launch.get("allocation_outcome_ambiguous"):
        cancel_pending_teardown(
            pending["path"], reason="provider_returned_no_allocation", evidence=launch
        )
    run_blockers = list(collection.get("blockers") or [])
    if launch.get("status") != "launched":
        run_blockers.append("g1_microwave_finetune_provider_not_launched")
    if collection.get("status") != "completed":
        run_blockers.append("g1_microwave_finetune_worker_not_completed")
    if checkpoint_collection.get("status") != "completed":
        run_blockers.extend(checkpoint_collection.get("blockers") or [])
        run_blockers.append("g1_microwave_finetune_checkpoint_not_collected")
    if not absence:
        run_blockers.append("g1_microwave_finetune_provider_zero_not_proven")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not run_blockers else "blocked",
        "provider": resolved_provider,
        "image_ref": IMAGE_REF,
        "bundle_sha256": bundle_sha,
        "provider_bootstrap_sha256": bootstrap_sha256,
        "checkpoint_output_get_url_file_present": bool(checkpoint_get_url),
        "checkpoint_output_part_count": len(checkpoint_part_get_urls),
        "checkpoint_output_object_binding_sha256": checkpoint_staging.get(
            "output_url_object_binding_sha256"
        ),
        "checkpoint_collection_mode": (
            "stream_verified_direct_to_retained_vast"
            if checkpoint_vast_target
            else (
                "object_store_bound_ordered_parts"
                if len(checkpoint_part_get_urls) > 1
                else "host_download_and_extract"
            )
        ),
        "launch": launch,
        "collection": collection,
        "checkpoint_collection": checkpoint_collection,
        "teardown": teardown,
        "final_global_inventory": final_inventory,
        "final_inventory_scope": final_inventory_scope,
        "pending_teardown_record": pending["path"],
        "watchdog_pid": watchdog.pid,
        "provider_mutations_performed": 1 if launch.get("instance_id") else 0,
        "continuing_spend": bool(checkpoint_vast_target) or not absence,
        "blockers": sorted(set(run_blockers)),
        "claim_boundary": {
            "fine_tune_completed": collection.get("status") == "completed",
            "checkpoint_host_collected": bool(
                checkpoint_collection.get("checkpoint_path")
            ),
            "checkpoint_object_store_bound": checkpoint_collection.get(
                "checkpoint_object_store_bound"
            )
            is True,
            "trainer_provider_absence_confirmed": absence,
            "trainer_provider_global_zero_proven": trainer_provider_global_zero,
            "bound_retained_session_continuing_spend": bool(
                checkpoint_vast_target
            ),
            "cross_provider_global_zero_not_proven": bool(
                checkpoint_vast_target
            ),
            "checkpoint_task_qualification_not_proven": True,
            "isaac_semantic_episode_success_not_proven": True,
        },
    }
    write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, default="runpod")
    parser.add_argument("--provider-bundle", required=True)
    parser.add_argument("--object-store-stage-dir")
    parser.add_argument("--checkpoint-object-store-stage-dir")
    parser.add_argument(
        "--checkpoint-object-store-part-stage-dir",
        action="append",
        default=[],
    )
    parser.add_argument("--release-evidence")
    parser.add_argument("--provider-launch-request")
    parser.add_argument("--preflight-bundle")
    parser.add_argument("--admission-out")
    parser.add_argument("--bound-request-out")
    parser.add_argument("--adapter-output", required=True)
    parser.add_argument("--pod-name", default="")
    parser.add_argument("--checkpoint-vast-session-manifest")
    parser.add_argument(
        "--qualification-identity-file",
        default=DEFAULT_QUALIFICATION_IDENTITY_FILE,
    )
    parser.add_argument("--worker-report")
    parser.add_argument("--resume-checkpoint-to-vast", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.resume_checkpoint_to_vast:
            if not args.worker_report:
                parser.error("--worker-report is required with --resume-checkpoint-to-vast")
            if not args.checkpoint_vast_session_manifest:
                parser.error(
                    "--checkpoint-vast-session-manifest is required with "
                    "--resume-checkpoint-to-vast"
                )
            if not args.checkpoint_object_store_part_stage_dir:
                parser.error(
                    "at least one --checkpoint-object-store-part-stage-dir is "
                    "required with --resume-checkpoint-to-vast"
                )
            if not args.execute or not args.admission_out:
                parser.error(
                    "--execute and --admission-out are required with "
                    "--resume-checkpoint-to-vast"
                )
            result = resume_checkpoint_transfer_to_vast(
                provider_bundle=args.provider_bundle,
                checkpoint_object_store_stage_dirs=(
                    args.checkpoint_object_store_part_stage_dir
                ),
                worker_report_path=args.worker_report,
                checkpoint_vast_session_manifest=(
                    args.checkpoint_vast_session_manifest
                ),
                admission_out=args.admission_out,
                adapter_output=args.adapter_output,
                execute=args.execute,
                qualification_identity_file=args.qualification_identity_file,
            )
        else:
            required = {
                "--object-store-stage-dir": args.object_store_stage_dir,
                "--checkpoint-object-store-stage-dir": (
                    args.checkpoint_object_store_stage_dir
                ),
                "--release-evidence": args.release_evidence,
                "--admission-out": args.admission_out,
                "--bound-request-out": args.bound_request_out,
            }
            missing = [flag for flag, value in required.items() if not value]
            if missing:
                parser.error(f"required arguments missing: {', '.join(missing)}")
            result = run_finetune_job(
                provider_name=args.provider,
                provider_bundle=args.provider_bundle,
                object_store_stage_dir=args.object_store_stage_dir,
                checkpoint_object_store_stage_dir=(
                    args.checkpoint_object_store_stage_dir
                ),
                checkpoint_object_store_part_stage_dirs=(
                    args.checkpoint_object_store_part_stage_dir
                ),
                release_evidence=args.release_evidence,
                provider_launch_request=args.provider_launch_request,
                preflight_bundle=args.preflight_bundle,
                admission_out=args.admission_out,
                bound_request_out=args.bound_request_out,
                adapter_output=args.adapter_output,
                pod_name=args.pod_name,
                execute=args.execute,
                checkpoint_vast_session_manifest=(
                    args.checkpoint_vast_session_manifest
                ),
                qualification_identity_file=args.qualification_identity_file,
            )
    except (OSError, ValueError, json.JSONDecodeError):
        return 1
    return 0 if result.get("status") in {"dry_run_ready", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
