"""Fail-closed paid admission for the native NVIDIA Warehouse camera canary."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from .gpu_render_providers import RenderLaunchSpec, get_render_provider
from .groot_oscar_runpod_watchdog import (
    terminate_canary_resources,
    write_owner_teardown_cancel_request,
)
from .isaac_worker_image_manifest import (
    SCHEMA_VERSION as ISAAC_IMAGE_MANIFEST_SCHEMA_VERSION,
)
from .nvidia_warehouse_native_camera_canary import (
    RESULT_SCHEMA_VERSION as CAMERA_RESULT_SCHEMA_VERSION,
)
from .nvidia_warehouse_native_camera_gpu_bundle import (
    BUNDLE_SCHEMA_VERSION,
    INPUT_SECRET_URL_ENV,
    INPUT_SHA256_ENV,
    MAX_WORKER_OUTPUT_BYTES,
    OUTPUT_SECRET_PUT_URL_ENV,
    RECEIPT_SCHEMA_VERSION,
)
from .openpi_policy_ranking_gpu_admission import (
    collect_openpi_policy_ranking_vast_preflight,
)
from .openpi_policy_ranking_runpod import (
    _read_private_https_url,
    _stop_watchdog_after_provider_zero,
    _wait_for_watchdog,
    _wait_for_watchdog_terminal,
    _write_private_json,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    load_pending_teardowns,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_provider_lane_lease import (
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    release_paid_provider_lane_lease,
    transfer_paid_provider_compute_lane_lease_to_watchdog,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .policy_ranking_thesis import canonical_sha256
from .production_gpu_campaign_budget import (
    CampaignBudgetExceeded,
    ProductionGpuCampaignBudget,
)
from .safe_outbound_http import presigned_transfer_policy
from .safe_outbound_http import request as safe_http_request


SCHEMA_VERSION = "nvidia_warehouse_native_camera_gpu_admission.v1"
RELEASE_SCHEMA_VERSION = "nvidia_warehouse_native_camera_gpu_release.v1"
PROBE_KIND = "new-site-native-camera"
MAX_PREFLIGHT_AGE_SECONDS = 300
MIN_CONTAINER_DISK_BYTES = 80 * 1024**3
MIN_GPU_MEMORY_BYTES = 16 * 1024**3
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_REF = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
CANARY_NAME_PREFIX = "blueprint-native-warehouse-camera-"
PAID_LANE = "nvidia_warehouse_native_camera_gpu_canary"
OUTPUT_ARCHIVE_NAME = "nvidia_warehouse_native_camera_provider_output.zip"
OUTPUT_VALIDATION_NAME = "nvidia_warehouse_native_camera_output_validation.json"
MONITOR_NAME = "nvidia_warehouse_native_camera_monitor.json"
MAX_OUTPUT_ARCHIVE_MEMBERS = 32
MAX_OUTPUT_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_CONSECUTIVE_TRANSIENT_OUTPUT_ERRORS = 3
MAXIMUM_SUPPORTED_GLOBAL_PAID_GPUS = 2
MIN_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS = 60
MAX_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS = 3600
GLOBAL_PAID_GPU_LAUNCH_LOCK = Path.home() / ".blueprint-secrets" / "paid_gpu_global_launch.lock"


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"nvidia_warehouse_camera_gpu_json_not_object:{path}")
    return dict(value)


def _provider_machine_ids(values: Any) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    machine_ids = {
        int(value)
        for value in values[:64]
        if (isinstance(value, int) and not isinstance(value, bool) and value > 0)
        or (isinstance(value, str) and value.isdigit() and int(value) > 0)
    }
    return sorted(machine_ids)


def _global_paid_gpu_inventory(
    *, vast_provider: Any, runpod_provider: Any | None = None
) -> dict[str, Any]:
    """Read both configured paid-GPU providers without mutating either."""

    providers = {
        "vast": vast_provider,
        "runpod": runpod_provider or get_render_provider("runpod"),
    }
    observations: dict[str, Any] = {}
    total = 0
    blockers: list[str] = []
    for name, provider in providers.items():
        try:
            inventory = provider.billable_inventory(name_prefix="")
        except Exception as exc:  # noqa: BLE001 - fail closed before allocation
            inventory = {
                "api_confirmed": False,
                "live_resource_count": None,
                "error_type": type(exc).__name__,
            }
        count = inventory.get("live_resource_count")
        confirmed = inventory.get("api_confirmed") is True
        valid_count = type(count) is int and count >= 0
        if not confirmed or not valid_count:
            blockers.append(f"native_camera_global_{name}_inventory_unverified")
        else:
            total += count
        observations[name] = {
            "api_confirmed": confirmed,
            "live_resource_count": count if valid_count else None,
            "resources": [
                {
                    "instance_id": row.get("instance_id"),
                    "name": row.get("name"),
                    "provider_status": row.get("provider_status"),
                    "gpu_name": row.get("gpu_name"),
                }
                for row in inventory.get("resources", [])
                if isinstance(row, Mapping)
            ],
        }
    result = {
        "schema_version": "blueprint_global_paid_gpu_inventory.v1",
        "status": "verified" if not blockers else "blocked",
        "observed_at_epoch": time.time(),
        "providers": observations,
        "total_live_paid_gpus_observed": total if not blockers else None,
        "blockers": blockers,
        "provider_mutations_performed": 0,
        "raw_provider_response_recorded": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _concurrency_aware_native_preflight(
    *,
    preflight: Mapping[str, Any],
    global_inventory: Mapping[str, Any],
    maximum_concurrent_paid_gpus_global: int,
) -> dict[str, Any]:
    """Bind a normal Vast capacity snapshot to the prospective global ceiling."""

    result = dict(preflight)
    blockers = [
        str(value)
        for value in result.get("blockers", [])
        if value != "openpi_gpu_preflight_billable_inventory_not_zero"
    ]
    total = global_inventory.get("total_live_paid_gpus_observed")
    ceiling_valid = (
        type(maximum_concurrent_paid_gpus_global) is int
        and 1 <= maximum_concurrent_paid_gpus_global <= MAXIMUM_SUPPORTED_GLOBAL_PAID_GPUS
    )
    below_ceiling = bool(
        ceiling_valid
        and global_inventory.get("status") == "verified"
        and type(total) is int
        and total < maximum_concurrent_paid_gpus_global
    )
    if not ceiling_valid:
        blockers.append("native_camera_global_paid_gpu_ceiling_invalid")
    blockers.extend(str(value) for value in global_inventory.get("blockers", []))
    if not below_ceiling:
        blockers.append("native_camera_global_paid_gpu_ceiling_reached_or_unverified")
    attempt_inventory = result.get("attempt_billable_inventory")
    attempt_inventory = attempt_inventory if isinstance(attempt_inventory, Mapping) else {}
    if not (
        attempt_inventory.get("api_confirmed") is True
        and attempt_inventory.get("live_resource_count") == 0
    ):
        blockers.append("native_camera_attempt_inventory_not_zero")
    result.update(
        {
            "status": "verified" if not blockers else "blocked",
            "blockers": sorted(set(blockers)),
            "provider_inventory_verified_zero": total == 0,
            "provider_inventory_below_global_ceiling": below_ceiling,
            "maximum_concurrent_paid_gpus_global": (maximum_concurrent_paid_gpus_global),
            "global_paid_gpu_inventory": dict(global_inventory),
        }
    )
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _acquire_global_paid_gpu_launch_lock() -> Any:
    """Serialize the final cross-provider inventory check and camera create."""

    GLOBAL_PAID_GPU_LAUNCH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = GLOBAL_PAID_GPU_LAUNCH_LOCK.open("a+", encoding="utf-8")
    GLOBAL_PAID_GPU_LAUNCH_LOCK.chmod(0o600)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_global_paid_gpu_launch_lock(handle: Any) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def build_native_camera_gpu_release_evidence(
    *, image_manifest: Mapping[str, Any], expected_source_commit: str
) -> dict[str, Any]:
    """Promote registry metadata to the exact-source camera release contract."""

    expected = str(expected_source_commit or "").strip().lower()
    identity_value = image_manifest.get("worker_build_identity")
    identity = identity_value if isinstance(identity_value, Mapping) else {}
    blockers: list[str] = []
    if image_manifest.get("schema_version") != ISAAC_IMAGE_MANIFEST_SCHEMA_VERSION:
        blockers.append("native_camera_gpu_release_image_manifest_schema_invalid")
    if image_manifest.get("status") != "completed":
        blockers.append("native_camera_gpu_release_image_manifest_not_completed")
    digest_ref = str(image_manifest.get("resolved_digest_ref") or "").strip()
    if not _DIGEST_REF.fullmatch(digest_ref):
        blockers.append("native_camera_gpu_release_image_not_digest_pinned")
    if image_manifest.get("runnable_platform") != "linux/amd64":
        blockers.append("native_camera_gpu_release_platform_invalid")
    if image_manifest.get("raw_secret_values_recorded") is not False:
        blockers.append("native_camera_gpu_release_manifest_secret_boundary_invalid")
    if not _COMMIT.fullmatch(expected):
        blockers.append("native_camera_gpu_release_expected_commit_invalid")
    if identity.get("status") != "verified":
        blockers.append("native_camera_gpu_release_build_identity_unverified")
    if identity.get("source_commit") != expected:
        blockers.append("native_camera_gpu_release_source_commit_mismatch")
    if identity.get("source_dirty_patch_sha256") != CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256:
        blockers.append("native_camera_gpu_release_dirty_overlay_forbidden")
    if identity.get("worker_image_family") != "isaac-eval-worker":
        blockers.append("native_camera_gpu_release_image_family_invalid")
    if identity.get("isaac_sim_major_version") != 6:
        blockers.append("native_camera_gpu_release_isaac_major_invalid")
    startup_timeout = image_manifest.get("recommended_startup_no_runtime_timeout_seconds")
    if not (
        type(startup_timeout) is int
        and MIN_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS
        <= startup_timeout
        <= MAX_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS
    ):
        blockers.append("native_camera_gpu_release_startup_timeout_invalid")
    result: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "source_commit": identity.get("source_commit"),
        "source_dirty_patch_sha256": identity.get("source_dirty_patch_sha256"),
        "resolved_digest_ref": digest_ref or None,
        "runnable_platform": image_manifest.get("runnable_platform"),
        "isaac_sim_major_version": identity.get("isaac_sim_major_version"),
        "worker_image_family": identity.get("worker_image_family"),
        "startup_no_runtime_timeout_seconds": startup_timeout,
        "image_manifest_sha256": canonical_sha256(dict(image_manifest)),
        "source_identity_from_immutable_registry_config": True,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "release_engineering_only": True,
            "provider_startup_proven": False,
            "native_camera_canary_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
        },
    }
    result["release_sha256"] = canonical_sha256(result)
    return result


def build_native_camera_gpu_admission(
    *,
    release: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    preflight: Mapping[str, Any],
    spend: Mapping[str, Any],
    expected_source_commit: str,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    expected_commit = str(expected_source_commit or "").strip().lower()
    source_commit = str(release.get("source_commit") or "").strip().lower()
    if not _COMMIT.fullmatch(expected_commit):
        blockers.append("native_camera_gpu_expected_source_commit_invalid")
    if release.get("schema_version") != RELEASE_SCHEMA_VERSION:
        blockers.append("native_camera_gpu_release_schema_invalid")
    if release.get("status") != "passed":
        blockers.append("native_camera_gpu_release_not_passed")
    if source_commit != expected_commit or not _COMMIT.fullmatch(source_commit):
        blockers.append("native_camera_gpu_release_source_commit_mismatch")
    image_ref = str(release.get("resolved_digest_ref") or "").strip()
    if not _DIGEST_REF.fullmatch(image_ref):
        blockers.append("native_camera_gpu_release_image_not_digest_pinned")
    if release.get("runnable_platform") != "linux/amd64":
        blockers.append("native_camera_gpu_release_platform_invalid")
    if release.get("isaac_sim_major_version") != 6:
        blockers.append("native_camera_gpu_release_isaac_major_invalid")
    if release.get("source_dirty_patch_sha256") != CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256:
        blockers.append("native_camera_gpu_release_dirty_overlay_forbidden")
    startup_timeout = release.get("startup_no_runtime_timeout_seconds")
    if not (
        type(startup_timeout) is int
        and MIN_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS
        <= startup_timeout
        <= MAX_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS
    ):
        blockers.append("native_camera_gpu_release_startup_timeout_invalid")

    if input_bundle.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        blockers.append("native_camera_gpu_input_receipt_schema_invalid")
    if input_bundle.get("status") != "completed":
        blockers.append("native_camera_gpu_input_receipt_not_completed")
    bundle_sha = str(input_bundle.get("bundle_sha256") or "")
    if not _SHA256.fullmatch(bundle_sha):
        blockers.append("native_camera_gpu_input_bundle_sha_invalid")
    receipt_declared = str(input_bundle.get("receipt_sha256") or "")
    receipt_payload = dict(input_bundle)
    receipt_payload.pop("receipt_sha256", None)
    if receipt_declared != canonical_sha256(receipt_payload):
        blockers.append("native_camera_gpu_input_receipt_identity_invalid")
    manifest_value = input_bundle.get("manifest")
    manifest = manifest_value if isinstance(manifest_value, Mapping) else {}
    manifest_declared = str(manifest.get("manifest_sha256") or "")
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("source_commit") != expected_commit
        or manifest.get("label_free") is not True
        or manifest.get("rankings_or_policy_outcomes_accessed") is not False
        or manifest.get("purpose") != "private_internal_nvidia_warehouse_native_camera_canary"
    ):
        blockers.append("native_camera_gpu_input_freeze_invalid")
    if manifest_declared != canonical_sha256(manifest_payload):
        blockers.append("native_camera_gpu_input_manifest_identity_invalid")

    if preflight.get("schema_version") not in {
        "openpi_policy_ranking_runpod_preflight.v1",
        "openpi_policy_ranking_provider_preflight.v2",
    }:
        blockers.append("native_camera_gpu_preflight_schema_invalid")
    if preflight.get("status") != "verified":
        blockers.append("native_camera_gpu_preflight_not_verified")
    provider = str(preflight.get("provider") or "")
    if provider not in {"runpod", "vast"} or preflight.get("provider_api_verified") is not True:
        blockers.append("native_camera_gpu_provider_not_verified")
    observed = preflight.get("observed_at_epoch")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    if type(observed) not in {int, float} or not math.isfinite(float(observed)):
        blockers.append("native_camera_gpu_preflight_observed_at_invalid")
    elif not 0 <= now - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("native_camera_gpu_preflight_stale_or_future")
    global_ceiling = preflight.get("maximum_concurrent_paid_gpus_global", 1)
    if not (
        type(global_ceiling) is int and 1 <= global_ceiling <= MAXIMUM_SUPPORTED_GLOBAL_PAID_GPUS
    ):
        blockers.append("native_camera_gpu_global_paid_gpu_ceiling_invalid")
    below_ceiling_proven = preflight.get("provider_inventory_below_global_ceiling") is True or (
        "maximum_concurrent_paid_gpus_global" not in preflight
        and preflight.get("provider_inventory_verified_zero") is True
    )
    if not below_ceiling_proven:
        blockers.append("native_camera_gpu_global_paid_gpu_ceiling_not_proven")
    if preflight.get("single_gpu_available") is not True:
        blockers.append("native_camera_gpu_single_gpu_unavailable")
    gpu_memory = preflight.get("gpu_memory_bytes")
    if type(gpu_memory) is not int or gpu_memory < MIN_GPU_MEMORY_BYTES:
        blockers.append("native_camera_gpu_memory_below_floor")
    container_disk = preflight.get("container_disk_bytes")
    if type(container_disk) is not int or container_disk < MIN_CONTAINER_DISK_BYTES:
        blockers.append("native_camera_gpu_container_disk_below_floor")
    hourly = preflight.get("on_demand_price_usd_per_hour")
    if type(hourly) not in {int, float} or not math.isfinite(float(hourly)) or float(hourly) <= 0:
        blockers.append("native_camera_gpu_hourly_price_invalid")

    ttl = spend.get("hard_ttl_seconds")
    max_spend = spend.get("max_spend_usd")
    if spend.get("paid_mutation_authorized") is not True:
        blockers.append("native_camera_gpu_paid_mutation_not_authorized")
    if spend.get("one_resource_limit") is not True:
        blockers.append("native_camera_gpu_one_resource_limit_missing")
    if spend.get("independent_teardown_watchdog") is not True:
        blockers.append("native_camera_gpu_watchdog_missing")
    if spend.get("watchdog_armed_before_allocation") is not True:
        blockers.append("native_camera_gpu_watchdog_not_armed_before_allocation")
    if type(ttl) is not int or not 60 <= ttl <= 3600:
        blockers.append("native_camera_gpu_ttl_invalid")
    elif type(startup_timeout) is int and startup_timeout > ttl - 60:
        blockers.append("native_camera_gpu_startup_timeout_exceeds_runtime_budget")
    if (
        type(max_spend) not in {int, float}
        or not math.isfinite(float(max_spend))
        or float(max_spend) <= 0
    ):
        blockers.append("native_camera_gpu_max_spend_invalid")
    if (
        type(ttl) is int
        and type(max_spend) in {int, float}
        and type(hourly) in {int, float}
        and float(hourly) * ttl / 3600 > float(max_spend)
    ):
        blockers.append("native_camera_gpu_ttl_cost_exceeds_max_spend")
    if spend.get("physical_robot_endpoint_access_allowed") is not False:
        blockers.append("native_camera_gpu_physical_robot_endpoint_not_forbidden")

    provider_resource_class = "gpu_render" if provider == "vast" else "runpod_provider_adapter"
    shared = build_paid_lane_admission(
        resource_class=provider_resource_class,
        blockers=blockers,
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted" if not blockers and shared["status"] == "admitted" else "blocked",
        "probe_kind": PROBE_KIND,
        "blockers": sorted(set(blockers)),
        "source_commit": source_commit or None,
        "release_image_ref": image_ref or None,
        "input_bundle_sha256": bundle_sha or None,
        "provider": provider or None,
        "provider_resource_class": provider_resource_class,
        "gpu_type_id": preflight.get("gpu_type_id"),
        "limits": {
            "hard_ttl_seconds": ttl,
            "max_spend_usd": max_spend,
            "one_resource": True,
            "maximum_concurrent_paid_gpus_global": global_ceiling,
        },
        "shared_paid_lane_admission": shared,
        "claim_boundary": {
            "camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def validate_native_camera_gpu_output_archive(
    archive_bytes: bytes,
) -> dict[str, Any]:
    """Validate one terminal camera result and all four hash-bound frames."""

    blockers: list[str] = []
    if not archive_bytes or len(archive_bytes) > MAX_WORKER_OUTPUT_BYTES:
        return {
            "schema_version": "nvidia_warehouse_native_camera_output_validation.v1",
            "status": "blocked",
            "blockers": ["native_camera_output_archive_size_invalid"],
            "terminal_output_present": bool(archive_bytes),
        }
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile:
        return {
            "schema_version": "nvidia_warehouse_native_camera_output_validation.v1",
            "status": "blocked",
            "blockers": ["native_camera_output_archive_invalid_zip"],
            "terminal_output_present": True,
        }
    with archive:
        members = archive.infolist()
        names: set[str] = set()
        total_uncompressed = 0
        if not members or len(members) > MAX_OUTPUT_ARCHIVE_MEMBERS:
            blockers.append("native_camera_output_archive_member_count_invalid")
        for member in members:
            parts = Path(member.filename).parts
            if (
                member.filename in names
                or Path(member.filename).is_absolute()
                or ".." in parts
                or member.flag_bits & 0x1
                or member.is_dir()
            ):
                blockers.append("native_camera_output_archive_member_unsafe")
            names.add(member.filename)
            total_uncompressed += int(member.file_size)
        if total_uncompressed > MAX_OUTPUT_UNCOMPRESSED_BYTES:
            blockers.append("native_camera_output_archive_uncompressed_size_exceeded")

        manifest_name = "native_camera_canary_result.json"
        manifest: dict[str, Any] = {}
        if manifest_name not in names:
            blockers.append("native_camera_output_result_missing")
        else:
            try:
                value = json.loads(archive.read(manifest_name).decode("utf-8"))
            except (UnicodeError, ValueError, json.JSONDecodeError):
                value = {}
                blockers.append("native_camera_output_result_unreadable")
            manifest = dict(value) if isinstance(value, Mapping) else {}

        if manifest:
            if manifest.get("schema_version") != CAMERA_RESULT_SCHEMA_VERSION:
                blockers.append("native_camera_output_result_schema_invalid")
            declared = str(manifest.get("result_sha256") or "")
            identity_payload = dict(manifest)
            identity_payload.pop("result_sha256", None)
            if declared != canonical_sha256(identity_payload):
                blockers.append("native_camera_output_result_identity_invalid")
            if manifest.get("status") not in {"passed", "failed"}:
                blockers.append("native_camera_output_result_not_terminal")
            if (
                manifest.get("label_free") is not True
                or manifest.get("rankings_or_policy_outcomes_accessed") is not False
                or manifest.get("paid_policy_or_wam_model_invoked") is not False
            ):
                blockers.append("native_camera_output_label_free_boundary_invalid")
            claim = manifest.get("claim_boundary")
            claim = claim if isinstance(claim, Mapping) else {}
            if (
                claim.get("native_scene_and_camera_technical_canary_only") is not True
                or claim.get("policy_wam_loop_proven") is not False
                or claim.get("ranking_accuracy") is not False
                or claim.get("physical_success") is not False
                or claim.get("captured_site_transfer_validation") is not False
                or claim.get("phase_b_confirmation") is not False
            ):
                blockers.append("native_camera_output_claim_boundary_invalid")

            failure = manifest.get("failure_evidence")
            failure = failure if isinstance(failure, Mapping) else {}
            failure_before_frames = bool(
                manifest.get("status") == "failed" and failure.get("failure_before_frames") is True
            )
            if failure_before_frames:
                if not str(failure.get("phase") or "") or not str(failure.get("error_type") or ""):
                    blockers.append("native_camera_output_failure_evidence_invalid")
                media = failure.get("media")
                media = media if isinstance(media, list) else []
                if not media:
                    blockers.append("native_camera_output_failure_media_missing")
                for item in media:
                    item = item if isinstance(item, Mapping) else {}
                    relative = str(item.get("relative_path") or "")
                    safe_relative = bool(
                        relative
                        and not Path(relative).is_absolute()
                        and ".." not in Path(relative).parts
                    )
                    if not safe_relative or relative not in names:
                        blockers.append("native_camera_output_failure_media_missing")
                        continue
                    if (
                        str(item.get("sha256") or "")
                        != hashlib.sha256(archive.read(relative)).hexdigest()
                    ):
                        blockers.append("native_camera_output_failure_media_sha256_mismatch")

            assessment = manifest.get("assessment")
            assessment = assessment if isinstance(assessment, Mapping) else {}
            views = assessment.get("views")
            views = views if isinstance(views, Mapping) else {}
            for view_id in () if failure_before_frames else ("external", "wrist"):
                view = views.get(view_id)
                view = view if isinstance(view, Mapping) else {}
                frames = view.get("frames")
                frames = frames if isinstance(frames, Mapping) else {}
                for phase in ("initial", "commanded"):
                    frame = frames.get(phase)
                    frame = frame if isinstance(frame, Mapping) else {}
                    relative = str(frame.get("relative_path") or "")
                    safe_relative = bool(
                        relative
                        and not Path(relative).is_absolute()
                        and ".." not in Path(relative).parts
                    )
                    if not safe_relative or relative not in names:
                        blockers.append(f"native_camera_output_frame_missing:{view_id}:{phase}")
                        continue
                    if (
                        str(frame.get("sha256") or "")
                        != hashlib.sha256(archive.read(relative)).hexdigest()
                    ):
                        blockers.append(
                            f"native_camera_output_frame_sha256_mismatch:{view_id}:{phase}"
                        )
                    if frame.get("resolution") != [640, 480]:
                        blockers.append(
                            f"native_camera_output_frame_resolution_invalid:{view_id}:{phase}"
                        )

    return {
        "schema_version": "nvidia_warehouse_native_camera_output_validation.v1",
        "status": "completed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "terminal_output_present": True,
        "canary_status": manifest.get("status"),
        "canary_result": manifest,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size_bytes": len(archive_bytes),
        "archive_member_count": len(members),
        "raw_secret_values_recorded": False,
    }


def _build_native_camera_vast_launch_request(
    *,
    provider: Any,
    root: Path,
    pod_name: str,
    release: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    preflight: Mapping[str, Any],
    input_secret_url: str,
    output_secret_put_url: str,
    excluded_machine_ids: Sequence[int] = (),
) -> dict[str, Any]:
    capacity = preflight.get("capacity_request")
    capacity = capacity if isinstance(capacity, Mapping) else {}
    command = (
        "exec /isaac-sim/python.sh -m "
        "blueprint_pipeline.nvidia_warehouse_native_camera_gpu_bundle "
        "worker --workspace /workspace/native-camera-canary"
    )
    spec = RenderLaunchSpec(
        name=pod_name,
        image=str(release["resolved_digest_ref"]),
        env={
            INPUT_SECRET_URL_ENV: input_secret_url,
            INPUT_SHA256_ENV: str(input_bundle["bundle_sha256"]),
            OUTPUT_SECRET_PUT_URL_ENV: output_secret_put_url,
        },
        bootstrap_argv=["-lc", command],
        entrypoint=["bash"],
        container_disk_gb=int(preflight["container_disk_bytes"]) // 1024**3,
        volume_gb=0,
        max_hourly_rate_usd=float(preflight["on_demand_price_usd_per_hour"]),
        min_gpu_ram_mb=int(
            capacity.get("min_gpu_ram_mb") or int(preflight["gpu_memory_bytes"]) // 1_000_000
        ),
        requires_rtx=True,
        vast_launch_mode="args",
    )
    request = provider.build_request(spec, root)
    request.update(
        {
            "prelaunch_spend_guard": {
                "required_before_provider_launch": True,
                "can_launch": True,
                "blockers": [],
            },
            "min_reliability": capacity.get("min_reliability"),
            "require_avx": True,
            "require_known_supported_isaac_driver": True,
            "preferred_gpu_keywords": capacity.get("preferred_gpu_keywords"),
            "excluded_machine_ids": sorted(set(excluded_machine_ids)),
        }
    )
    return request


def build_native_camera_gpu_provider_request(
    *,
    release: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    preflight: Mapping[str, Any],
    spend: Mapping[str, Any],
    expected_source_commit: str,
    job_id: str,
    launcher_source_commit: str | None = None,
) -> dict[str, Any]:
    """Bind the admitted canary to its exact worker without persisting secrets."""

    admission = build_native_camera_gpu_admission(
        release=release,
        input_bundle=input_bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=expected_source_commit,
    )
    if admission["status"] != "admitted":
        return {
            "schema_version": "nvidia_warehouse_native_camera_gpu_launch.v1",
            "status": "blocked",
            "blockers": admission["blockers"],
            "admission": admission,
            "provider_mutations_performed": 0,
        }
    provider = str(preflight["provider"])
    global_ceiling = int(admission["limits"]["maximum_concurrent_paid_gpus_global"])
    excluded_machine_ids = _provider_machine_ids(preflight.get("excluded_machine_ids"))
    ttl = int(spend["hard_ttl_seconds"])
    bundle_sha = str(input_bundle["bundle_sha256"])
    worker_command = (
        "exec /isaac-sim/python.sh -m "
        "blueprint_pipeline.nvidia_warehouse_native_camera_gpu_bundle "
        "worker --workspace /workspace/native-camera-canary"
    )
    request: dict[str, Any] = {
        "schema_version": "nvidia_warehouse_native_camera_gpu_request.v2",
        "status": "request_manifest_ready",
        "operation": "execute_label_free_native_warehouse_camera_canary",
        "job_id": job_id,
        "provider": provider,
        "image": str(release["resolved_digest_ref"]),
        "input_bundle_sha256": bundle_sha,
        "launcher_source_commit": launcher_source_commit,
        "provider_request_shape": {
            "api_payload_is_provider_adapter_template": True,
            "api_payload_values_are_redacted": True,
            "docker_entrypoint": ["bash"],
            "docker_start_cmd": ["-lc", worker_command],
            "environment": {
                "secret_env_var_names": [
                    INPUT_SECRET_URL_ENV,
                    OUTPUT_SECRET_PUT_URL_ENV,
                ],
                "plaintext_env_var_names": [INPUT_SHA256_ENV],
                "plaintext_env_values": {INPUT_SHA256_ENV: bundle_sha},
                "secret_values_in_artifact": False,
            },
            "gpu": {
                "gpu_count": 1,
                "preferred_gpu_type_id": str(preflight["gpu_type_id"]),
                "container_disk_in_gb": int(preflight["container_disk_bytes"]) // 1024**3,
                "volume_in_gb": 0,
                "min_gpu_memory_bytes": int(preflight["gpu_memory_bytes"]),
            },
            "limits": {
                "max_active_workers": 1,
                "requested_budget_usd": float(spend["max_spend_usd"]),
                "hard_timeout_seconds": ttl,
                "startup_no_runtime_timeout_seconds": int(
                    release["startup_no_runtime_timeout_seconds"]
                ),
                "independent_watchdog_required": True,
                "watchdog_armed_before_allocation": True,
                "attempt_inventory_zero_required_before_launch": True,
                "global_inventory_below_ceiling_required_before_launch": True,
                "owned_resource_absence_required_after_launch": True,
                "maximum_concurrent_paid_gpus_global": int(global_ceiling),
                "terminal_delete_required": True,
                "excluded_machine_ids": excluded_machine_ids,
            },
            "output_contract": {
                "individual_external_camera_frame_required": True,
                "individual_wrist_camera_frame_required": True,
                "camera_canary_result_required": True,
                "upload_before_shutdown_required": True,
            },
        },
        "physical_robot_endpoint_access_allowed": False,
        "provider_mutations_performed": 0,
        "claim_boundary": {
            "camera_technical_canary_only": True,
            "policy_wam_loop_proven": False,
            "ranking_accuracy": False,
            "physical_success": False,
        },
    }
    request["manifest_sha256"] = canonical_sha256(request)
    return {
        "schema_version": "nvidia_warehouse_native_camera_gpu_launch.v1",
        "status": "admitted",
        "blockers": [],
        "admission": admission,
        "bound_request": request,
        "provider_mutations_performed": 0,
    }


def _camera_cleanup_handoff(
    *,
    root: Path,
    provider: Any,
    cleanup: Mapping[str, Any],
    instance_id: str,
    provider_name: str,
) -> dict[str, Any]:
    global_inventory = provider.billable_inventory(name_prefix="")
    absence_proven = bool(
        cleanup.get("provider_absence_confirmed") is True
        and global_inventory.get("api_confirmed") is True
    )
    cancel_request = (
        write_owner_teardown_cancel_request(
            root=root,
            pod_name_prefix=CANARY_NAME_PREFIX,
            provider_name=provider_name,
            instance_id=instance_id,
        )
        if absence_proven
        else {}
    )
    watchdog_terminal = _wait_for_watchdog_terminal(root) if cancel_request else {}
    settlement = watchdog_terminal.get("campaign_budget_settlement")
    settlement = settlement if isinstance(settlement, Mapping) else {}
    control_terminal = bool(
        watchdog_terminal.get("control_plane_terminal") is True
        and settlement.get("status") == "settled"
    )
    return {
        "provider_absence_confirmed": absence_proven,
        "global_inventory": global_inventory,
        "watchdog_cancel_requested": bool(cancel_request),
        "watchdog_terminal": watchdog_terminal,
        "control_plane_terminal": control_terminal,
        "continuing_spend": not control_terminal,
    }


def _monitor_native_camera_output_and_teardown(
    *,
    root: Path,
    output_secret_get_url: str,
    provider: Any,
    armed: Mapping[str, Any],
    instance_id: str,
    provider_name: str,
    deadline_epoch: float,
    startup_begin_epoch: float,
    startup_no_runtime_timeout_seconds: int,
    poll_interval_seconds: float = 15.0,
) -> dict[str, Any]:
    response_bytes: bytes | None = None
    http_status: int | None = None
    consecutive_transient_errors = 0
    startup_deadline_epoch = min(
        deadline_epoch - 60,
        startup_begin_epoch + startup_no_runtime_timeout_seconds,
    )
    provider_startup_observed = False
    while time.time() < deadline_epoch - 60:
        try:
            response = safe_http_request(
                output_secret_get_url,
                method="GET",
                timeout_seconds=60,
                policy=presigned_transfer_policy(
                    output_secret_get_url,
                    max_response_bytes=MAX_WORKER_OUTPUT_BYTES,
                ),
                max_response_bytes=MAX_WORKER_OUTPUT_BYTES,
            )
            http_status = response.status
            consecutive_transient_errors = 0
            if response.status == 200 and response.body:
                response_bytes = response.body
                break
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            consecutive_transient_errors = 0
            if exc.code not in {403, 404}:
                break
        except urllib.error.URLError as exc:
            consecutive_transient_errors += 1
            if consecutive_transient_errors >= MAX_CONSECUTIVE_TRANSIENT_OUTPUT_ERRORS:
                return {
                    "status": "monitor_failed_watchdog_retained",
                    "blockers": [f"native_camera_output_monitor_failed:{type(exc).__name__}"],
                    "transient_error_attempts": consecutive_transient_errors,
                    "continuing_spend": True,
                    "watchdog_deadline_epoch": deadline_epoch,
                    "raw_secret_values_recorded": False,
                }
        except Exception as exc:  # noqa: BLE001 - watchdog still owns the deadline
            return {
                "status": "monitor_failed_watchdog_retained",
                "blockers": [f"native_camera_output_monitor_failed:{type(exc).__name__}"],
                "continuing_spend": True,
                "watchdog_deadline_epoch": deadline_epoch,
                "raw_secret_values_recorded": False,
            }
        inspection = provider.inspect(instance_id)
        provider_statuses = {
            str(inspection.get("actual_status") or "").lower(),
            str(inspection.get("cur_state") or "").lower(),
        }
        if provider_statuses & {"running", "active"}:
            provider_startup_observed = True
        provider_terminal = bool(
            inspection.get("provider_absence_confirmed") is True
            or str(inspection.get("actual_status") or "").lower()
            in {"exited", "stopped", "dead", "offline"}
        )
        if provider_terminal:
            teardown = terminate_canary_resources(
                provider=provider,
                pod_name_prefix=CANARY_NAME_PREFIX,
                armed=armed,
                provider_name=provider_name,
            )
            handoff = _camera_cleanup_handoff(
                root=root,
                provider=provider,
                cleanup=teardown,
                instance_id=instance_id,
                provider_name=provider_name,
            )
            return {
                "status": (
                    "failed" if handoff.get("control_plane_terminal") is True else "blocked"
                ),
                "blockers": ["native_camera_worker_terminal_without_output"],
                "advance_to_policy_wam": False,
                "provider_inspection": inspection,
                "teardown": teardown,
                **handoff,
                "raw_secret_values_recorded": False,
            }
        if not provider_startup_observed and time.time() >= startup_deadline_epoch:
            teardown = terminate_canary_resources(
                provider=provider,
                pod_name_prefix=CANARY_NAME_PREFIX,
                armed=armed,
                provider_name=provider_name,
            )
            handoff = _camera_cleanup_handoff(
                root=root,
                provider=provider,
                cleanup=teardown,
                instance_id=instance_id,
                provider_name=provider_name,
            )
            return {
                "status": (
                    "failed" if handoff.get("control_plane_terminal") is True else "blocked"
                ),
                "blockers": ["native_camera_provider_startup_timeout_without_runtime"],
                "advance_to_policy_wam": False,
                "provider_startup_observed": False,
                "startup_begin_epoch": startup_begin_epoch,
                "startup_deadline_epoch": startup_deadline_epoch,
                "startup_no_runtime_timeout_seconds": startup_no_runtime_timeout_seconds,
                "provider_inspection": inspection,
                "teardown": teardown,
                **handoff,
                "raw_secret_values_recorded": False,
            }
        time.sleep(max(0.1, poll_interval_seconds))
    if response_bytes is None:
        return {
            "status": "output_not_observed_watchdog_retained",
            "blockers": [f"native_camera_output_not_observed:http_{http_status}"],
            "continuing_spend": True,
            "watchdog_deadline_epoch": deadline_epoch,
            "raw_secret_values_recorded": False,
        }

    (root / OUTPUT_ARCHIVE_NAME).write_bytes(response_bytes)
    validation = validate_native_camera_gpu_output_archive(response_bytes)
    write_json(root / OUTPUT_VALIDATION_NAME, validation)
    teardown = terminate_canary_resources(
        provider=provider,
        pod_name_prefix=CANARY_NAME_PREFIX,
        armed=armed,
        provider_name=provider_name,
    )
    global_inventory = provider.billable_inventory(name_prefix="")
    absence_proven = bool(
        teardown.get("provider_absence_confirmed") is True
        and global_inventory.get("api_confirmed") is True
    )
    if absence_proven:
        write_owner_teardown_cancel_request(
            root=root,
            pod_name_prefix=CANARY_NAME_PREFIX,
            provider_name=provider_name,
            instance_id=instance_id,
        )
    watchdog_terminal = _wait_for_watchdog_terminal(root) if absence_proven else {}
    settlement = watchdog_terminal.get("campaign_budget_settlement")
    settlement = settlement if isinstance(settlement, Mapping) else {}
    control_terminal = bool(
        absence_proven
        and watchdog_terminal.get("control_plane_terminal") is True
        and settlement.get("status") == "settled"
    )
    canary_status = validation.get("canary_status")
    result = {
        "schema_version": "nvidia_warehouse_native_camera_monitor.v1",
        "status": (
            "completed"
            if validation.get("status") == "completed" and control_terminal
            else "blocked"
        ),
        "blockers": sorted(
            set(
                [
                    *(validation.get("blockers") or []),
                    *([] if absence_proven else ["native_camera_provider_absence_unverified"]),
                    *([] if control_terminal else ["native_camera_control_plane_not_terminal"]),
                ]
            )
        ),
        "canary_status": canary_status,
        "advance_to_policy_wam": bool(
            validation.get("status") == "completed"
            and canary_status == "passed"
            and control_terminal
        ),
        "output_validation": validation,
        "teardown": teardown,
        "final_global_inventory": global_inventory,
        "provider_absence_confirmed": absence_proven,
        "control_plane_terminal": control_terminal,
        "campaign_budget_settlement": dict(settlement),
        "continuing_spend": not control_terminal,
        "raw_secret_values_recorded": False,
    }
    write_json(root / MONITOR_NAME, result)
    return result


def run_native_camera_gpu_lane(
    *,
    release_evidence: str | Path,
    input_bundle_receipt: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    pod_name: str,
    expected_source_commit: str,
    launcher_source_commit: str | None = None,
    execute: bool,
    hard_ttl_seconds: int,
    max_spend_usd: float,
    input_secret_url_file: str | Path | None = None,
    output_secret_put_url_file: str | Path | None = None,
    output_secret_get_url_file: str | Path | None = None,
    campaign_budget_ledger: str | Path | None = None,
    campaign_initial_spent_usd: float | None = None,
    campaign_initial_used_gpu_seconds: int | None = None,
    campaign_total_spend_cap_usd: float = 20.0,
    campaign_wall_cap_seconds: int = 36_000,
    provider_name: str = "vast",
    maximum_concurrent_paid_gpus_global: int = 1,
) -> dict[str, Any]:
    """Validate or launch one guarded, label-free native camera canary."""

    root = Path(adapter_output).expanduser().resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    if not _COMMIT.fullmatch(str(launcher_source_commit or "")):
        result = {
            "status": "blocked",
            "blockers": ["native_camera_gpu_launcher_source_commit_invalid"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result
    if not pod_name.startswith(CANARY_NAME_PREFIX):
        result = {
            "status": "blocked",
            "blockers": ["native_camera_gpu_pod_name_outside_watchdog_scope"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result
    release = _read_object(release_evidence)
    bundle = _read_object(input_bundle_receipt)
    preflight = _read_object(preflight_bundle)
    excluded_machine_ids = list(preflight.get("excluded_machine_ids") or [])
    resolved_provider = str(provider_name or "vast").strip().lower()
    if resolved_provider != "vast":
        result = {
            "status": "blocked",
            "blockers": ["native_camera_gpu_live_provider_must_be_vast"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result
    provider = get_render_provider(resolved_provider) if execute else None
    if execute and provider is not None:
        base_preflight = collect_openpi_policy_ranking_vast_preflight(
            name_prefix=CANARY_NAME_PREFIX,
            container_disk_bytes=int(preflight.get("container_disk_bytes") or 0),
            capacity_probe=provider.capacity_preflight,
            inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
        )
        global_inventory = _global_paid_gpu_inventory(vast_provider=provider)
        preflight = _concurrency_aware_native_preflight(
            preflight=base_preflight,
            global_inventory=global_inventory,
            maximum_concurrent_paid_gpus_global=(maximum_concurrent_paid_gpus_global),
        )
        if excluded_machine_ids:
            preflight["excluded_machine_ids"] = excluded_machine_ids
            preflight.pop("manifest_sha256", None)
            preflight["manifest_sha256"] = canonical_sha256(preflight)
        write_json(root / "native_camera_provider_preflight_launch_refresh.json", preflight)
    if str(preflight.get("provider") or "") != resolved_provider:
        result = {
            "status": "blocked",
            "blockers": ["native_camera_gpu_preflight_provider_mismatch"],
            "expected_provider": resolved_provider,
            "observed_provider": preflight.get("provider"),
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        return result
    spend = {
        "paid_mutation_authorized": True,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "hard_ttl_seconds": int(hard_ttl_seconds),
        "max_spend_usd": float(max_spend_usd),
        "physical_robot_endpoint_access_allowed": False,
    }
    prepared = build_native_camera_gpu_provider_request(
        release=release,
        input_bundle=bundle,
        preflight=preflight,
        spend=spend,
        expected_source_commit=expected_source_commit,
        job_id=pod_name,
        launcher_source_commit=launcher_source_commit,
    )
    write_json(Path(admission_out), prepared)
    request = prepared.get("bound_request")
    if isinstance(request, Mapping):
        write_json(Path(bound_request_out), dict(request))
    if prepared["status"] != "admitted":
        result = {
            "status": "blocked",
            "blockers": prepared["blockers"],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    if not execute:
        result = {
            **prepared,
            "status": "dry_run_ready",
            "provider_mutations_performed": 0,
            "watchdog_process_started": False,
            "budget_reservation_created": False,
        }
        write_json(Path(adapter_output), result)
        return result

    missing_execute = [
        name
        for name, value in (
            ("input_secret_url_file", input_secret_url_file),
            ("output_secret_put_url_file", output_secret_put_url_file),
            ("output_secret_get_url_file", output_secret_get_url_file),
            ("campaign_budget_ledger", campaign_budget_ledger),
            ("campaign_initial_spent_usd", campaign_initial_spent_usd),
            ("campaign_initial_used_gpu_seconds", campaign_initial_used_gpu_seconds),
        )
        if value is None
    ]
    if missing_execute:
        result = {
            **prepared,
            "status": "blocked",
            "blockers": [
                "native_camera_gpu_execute_arguments_missing:" + ",".join(missing_execute)
            ],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result

    input_secret_url = _read_private_https_url(
        input_secret_url_file, field="native_camera_input_secret_url_file"
    )
    output_secret_put_url = _read_private_https_url(
        output_secret_put_url_file, field="native_camera_output_secret_put_url_file"
    )
    output_secret_get_url = _read_private_https_url(
        output_secret_get_url_file, field="native_camera_output_secret_get_url_file"
    )
    price = float(preflight.get("on_demand_price_usd_per_hour") or 0.0)
    budget = ProductionGpuCampaignBudget(
        campaign_budget_ledger,
        initial_spent_usd=campaign_initial_spent_usd,
        initial_used_gpu_seconds=campaign_initial_used_gpu_seconds,
        total_spend_cap_usd=campaign_total_spend_cap_usd,
        combined_gpu_wall_cap_seconds=campaign_wall_cap_seconds,
    )
    try:
        reservation = budget.reserve(
            reservation_id=pod_name,
            gpu_seconds=hard_ttl_seconds,
            max_hourly_rate_usd=price,
        )
    except CampaignBudgetExceeded as exc:
        result = {
            **prepared,
            "status": "blocked",
            "blockers": [str(exc)],
            "provider_mutations_performed": 0,
            "campaign_budget_admission": exc.admission,
        }
        write_json(Path(adapter_output), result)
        return result
    reserved_at_epoch = time.time()
    budget_context = {
        "status": "reserved",
        "ledger_path": str(Path(campaign_budget_ledger).expanduser().resolve()),
        "reservation_id": pod_name,
        "reserved_at_epoch": reserved_at_epoch,
        "reservation": reservation,
        "identity": {
            "initial_spent_usd": campaign_initial_spent_usd,
            "initial_used_gpu_seconds": campaign_initial_used_gpu_seconds,
            "total_spend_cap_usd": campaign_total_spend_cap_usd,
            "combined_gpu_wall_cap_seconds": campaign_wall_cap_seconds,
        },
    }
    write_json(root / "native_camera_campaign_budget_reservation.json", budget_context)

    deadline = time.time() + hard_ttl_seconds
    watchdog = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--out-dir",
            str(root),
            "--pod-name-prefix",
            CANARY_NAME_PREFIX,
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
    try:
        armed = _wait_for_watchdog(
            root=root,
            process=watchdog,
            prefix=CANARY_NAME_PREFIX,
            deadline=deadline,
        )
    except Exception:
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="watchdog_not_armed_no_mutation",
        )
        raise
    write_json(root / "native_camera_watchdog_armed_receipt.json", armed)

    inventory = provider.billable_inventory(name_prefix=CANARY_NAME_PREFIX)
    reconciliation = build_paid_provider_lane_reconciliation(
        provider=resolved_provider,
        lane=PAID_LANE,
        provider_inventory=inventory,
        open_pending_teardowns=load_pending_teardowns(),
    )
    lease = acquire_paid_provider_lane_lease(
        provider=resolved_provider,
        lane=PAID_LANE,
        job_dir=str(root),
        ttl_seconds=hard_ttl_seconds + 600,
        reconciliation=reconciliation,
    )
    write_json(root / "native_camera_paid_provider_lane_lease.json", lease)
    if lease.get("status") != "acquired":
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="paid_provider_lane_not_acquired_no_mutation",
        )
        result = {
            **prepared,
            "status": "blocked",
            "blockers": list(lease.get("blockers") or []),
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result

    try:
        pending = open_pending_teardown(
            provider=resolved_provider,
            lane=PAID_LANE,
            run_id=pod_name,
            resource_kind="compute_instance",
            resource_name=pod_name,
            job_dir=root,
            max_age_seconds=hard_ttl_seconds + 600,
        )
        require_paid_resource_admission(
            prepared["admission"]["shared_paid_lane_admission"],
            resource_class=str(prepared["admission"]["provider_resource_class"]),
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except Exception:  # noqa: BLE001 - provider boundary has not been crossed
        release_paid_provider_lane_lease(
            lease,
            reason="local_pre_provider_failure_no_mutation",
            provider_mutation_started=False,
        )
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="local_pre_provider_failure_no_mutation",
        )
        raise

    receipt_path = root / "provider_lane_handoff_receipt.json"
    receipt = {
        "status": "pending_watchdog_transfer",
        "lease_path": lease["path"],
        "owner_pid": watchdog.pid,
        "provider_lane_release_mode": "watchdog_direct_compute",
        "pod_pending_teardown_record": pending["path"],
        "pod_id": None,
        "pod_name_prefix": CANARY_NAME_PREFIX,
        "campaign_kind": "nvidia_warehouse_native_camera",
        "paid_lane": PAID_LANE,
        "maximum_concurrent_paid_gpus_global": (maximum_concurrent_paid_gpus_global),
        "campaign_budget": budget_context,
    }
    _write_private_json(receipt_path, receipt)
    handoff = transfer_paid_provider_compute_lane_lease_to_watchdog(
        lease,
        watchdog_pid=watchdog.pid,
        pending_teardown_record=pending["path"],
        watchdog_deadline_epoch=deadline,
        resource_name_prefix=CANARY_NAME_PREFIX,
    )
    write_json(root / "native_camera_paid_provider_lane_handoff.json", handoff)
    if handoff.get("status") != "accepted":
        cancel_pending_teardown(
            pending["path"],
            reason="compute_lane_handoff_failed_no_mutation",
            evidence={"provider_mutations_performed": 0},
        )
        release_paid_provider_lane_lease(
            lease,
            reason="compute_lane_handoff_failed_no_mutation",
            provider_mutation_started=False,
        )
        _stop_watchdog_after_provider_zero(watchdog)
        budget.settle(
            reservation_id=pod_name,
            charged_gpu_seconds=0,
            charged_usd=0.0,
            outcome="compute_lane_handoff_failed_no_mutation",
        )
        receipt_path.unlink(missing_ok=True)
        result = {
            **prepared,
            "status": "blocked",
            "blockers": list(handoff.get("blockers") or []),
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    receipt = {
        **handoff,
        "provider_lane_release_mode": "watchdog_direct_compute",
        "pod_pending_teardown_record": pending["path"],
        "pod_id": None,
        "pod_name_prefix": CANARY_NAME_PREFIX,
        "campaign_kind": "nvidia_warehouse_native_camera",
        "paid_lane": PAID_LANE,
        "maximum_concurrent_paid_gpus_global": (maximum_concurrent_paid_gpus_global),
        "campaign_budget": budget_context,
    }
    _write_private_json(receipt_path, receipt)

    try:
        launch_lock = _acquire_global_paid_gpu_launch_lock()
        try:
            final_base_preflight = collect_openpi_policy_ranking_vast_preflight(
                name_prefix=CANARY_NAME_PREFIX,
                container_disk_bytes=int(preflight.get("container_disk_bytes") or 0),
                capacity_probe=provider.capacity_preflight,
                inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
            )
            final_global_inventory = _global_paid_gpu_inventory(vast_provider=provider)
            write_json(
                root / "native_camera_global_gpu_inventory_final_prelaunch.json",
                final_global_inventory,
            )
            final_preflight = _concurrency_aware_native_preflight(
                preflight=final_base_preflight,
                global_inventory=final_global_inventory,
                maximum_concurrent_paid_gpus_global=(maximum_concurrent_paid_gpus_global),
            )
            if excluded_machine_ids:
                final_preflight["excluded_machine_ids"] = excluded_machine_ids
                final_preflight.pop("manifest_sha256", None)
                final_preflight["manifest_sha256"] = canonical_sha256(final_preflight)
            write_json(
                root / "native_camera_provider_preflight_final_prelaunch.json",
                final_preflight,
            )
            final_prepared = build_native_camera_gpu_provider_request(
                release=release,
                input_bundle=bundle,
                preflight=final_preflight,
                spend=spend,
                expected_source_commit=expected_source_commit,
                job_id=pod_name,
                launcher_source_commit=launcher_source_commit,
            )
            write_json(
                root / "native_camera_gpu_admission_final_prelaunch.json",
                final_prepared,
            )
            if final_prepared.get("status") != "admitted":
                launch = {
                    "status": "blocked",
                    "blockers": list(final_prepared.get("blockers") or []),
                    "allocation_created": False,
                    "provider_mutations_performed": 0,
                }
            else:
                launch_grant = require_paid_resource_admission(
                    final_prepared["admission"]["shared_paid_lane_admission"],
                    resource_class=str(final_prepared["admission"]["provider_resource_class"]),
                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                )
                prepared = final_prepared
                final_request = final_prepared.get("bound_request")
                if isinstance(final_request, Mapping):
                    write_json(Path(bound_request_out), dict(final_request))
                launch_request = _build_native_camera_vast_launch_request(
                    provider=provider,
                    root=root,
                    pod_name=pod_name,
                    release=release,
                    input_bundle=bundle,
                    preflight=final_preflight,
                    input_secret_url=input_secret_url,
                    output_secret_put_url=output_secret_put_url,
                    excluded_machine_ids=_provider_machine_ids(
                        final_preflight.get("excluded_machine_ids")
                    ),
                )
                launch = provider.launch(
                    root,
                    launch_request,
                    cold=True,
                    paid_resource_admission_grant=launch_grant,
                )
        finally:
            _release_global_paid_gpu_launch_lock(launch_lock)
        adapter = {
            **dict(launch),
            "status": "submitted" if launch.get("status") == "launched" else "blocked",
            "provider": "vast",
            "raw_secret_values_recorded": False,
        }
        adapter.pop("error", None)
        write_json(Path(adapter_output), adapter)
    except Exception as exc:
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason="native_camera_adapter_raised_after_create_boundary",
            evidence={"error_type": type(exc).__name__},
        )
        cleanup = terminate_canary_resources(
            provider=provider,
            pod_name_prefix=CANARY_NAME_PREFIX,
            armed=armed,
            provider_name=resolved_provider,
        )
        cleanup_handoff = _camera_cleanup_handoff(
            root=root,
            provider=provider,
            cleanup=cleanup,
            instance_id=pod_name,
            provider_name=resolved_provider,
        )
        result = {
            "status": "failed",
            "blockers": ["native_camera_gpu_adapter_failed_or_ambiguous"],
            "error_type": type(exc).__name__,
            "immediate_cleanup": cleanup,
            "cleanup_handoff": cleanup_handoff,
            "continuing_spend": cleanup_handoff["continuing_spend"],
            "raw_secret_values_recorded": False,
        }
        write_json(Path(adapter_output), result)
        return result

    instance_id = str(adapter.get("instance_id") or "").strip()
    if adapter.get("status") == "submitted" and instance_id:
        bind_pending_teardown_instance(pending["path"], instance_id)
        receipt["pod_id"] = instance_id
        _write_private_json(receipt_path, receipt)
        monitor = _monitor_native_camera_output_and_teardown(
            root=root,
            output_secret_get_url=output_secret_get_url,
            provider=provider,
            armed=armed,
            instance_id=instance_id,
            provider_name=resolved_provider,
            deadline_epoch=deadline,
            startup_begin_epoch=reserved_at_epoch,
            startup_no_runtime_timeout_seconds=int(
                release["startup_no_runtime_timeout_seconds"]
            ),
        )
        result = {
            **adapter,
            "status": monitor["status"],
            "watchdog_pid": watchdog.pid,
            "watchdog_deadline_epoch": deadline,
            "campaign_budget_reservation": reservation,
            "pending_teardown_record": pending["path"],
            "monitor": monitor,
            "continuing_spend": monitor.get("continuing_spend") is True,
            "raw_secret_values_recorded": False,
        }
        write_json(Path(adapter_output), result)
        return result

    if adapter.get("allocation_outcome_ambiguous") is True:
        mark_pending_teardown_ambiguous(
            pending["path"],
            reason="native_camera_create_result_missing_instance_id",
            evidence={"adapter_status": adapter.get("status")},
        )
    cleanup = terminate_canary_resources(
        provider=provider,
        pod_name_prefix=CANARY_NAME_PREFIX,
        armed=armed,
        provider_name=resolved_provider,
    )
    cleanup_handoff = _camera_cleanup_handoff(
        root=root,
        provider=provider,
        cleanup=cleanup,
        instance_id=pod_name,
        provider_name=resolved_provider,
    )
    result = {
        **adapter,
        "status": "failed",
        "blockers": sorted(
            set(
                [
                    *(adapter.get("blockers") or []),
                    "native_camera_gpu_instance_id_missing",
                ]
            )
        ),
        "immediate_cleanup": cleanup,
        "cleanup_handoff": cleanup_handoff,
        "continuing_spend": cleanup_handoff["continuing_spend"],
    }
    write_json(Path(adapter_output), result)
    return result


__all__ = [
    "PROBE_KIND",
    "RELEASE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_native_camera_gpu_admission",
    "build_native_camera_gpu_provider_request",
    "build_native_camera_gpu_release_evidence",
    "run_native_camera_gpu_lane",
    "validate_native_camera_gpu_output_archive",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("build-release")
    release.add_argument("--image-manifest", required=True)
    release.add_argument("--expected-source-commit", required=True)
    release.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    image_manifest = _read_object(args.image_manifest)
    result = build_native_camera_gpu_release_evidence(
        image_manifest=image_manifest,
        expected_source_commit=args.expected_source_commit,
    )
    write_json(Path(args.output), result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
