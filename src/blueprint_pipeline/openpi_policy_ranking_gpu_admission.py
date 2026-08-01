"""Fail-closed admission contract for the OpenPI policy-ranking GPU lane."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .gpu_render_providers import get_render_provider
from .new_site_diagnostic_canary_gpu import (
    INPUT_RECEIPT_SCHEMA_VERSION as CANARY_INPUT_RECEIPT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION as CANARY_INPUT_SCHEMA_VERSION,
)
from .openpi_current_reference_gpu_bundle import (
    INPUT_RECEIPT_SCHEMA_VERSION as CURRENT_REFERENCE_INPUT_RECEIPT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION as CURRENT_REFERENCE_INPUT_SCHEMA_VERSION,
)
from .paid_resource_admission import build_paid_lane_admission
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "openpi_policy_ranking_gpu_admission.v1"
PROBE_KIND = "openpi-policy-ranking"
NEW_SITE_CANARY_PROBE_KIND = "new-site-diagnostic-canary"
CURRENT_REFERENCE_POLICY_CANARY_PROBE_KIND = "openpi-current-reference-policy-canary"
OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
MENAGERIE_REVISION = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
CHECKPOINT_BYTES = 47_286_181_297
CURRENT_REFERENCE_CHECKPOINT_BYTES = 35_280_084_249
MAX_TTL_SECONDS = 14_400
MAX_PREFLIGHT_AGE_SECONDS = 300
MIN_GPU_MEMORY_BYTES = 24 * 1024**3
MIN_CONTAINER_DISK_BYTES = 80 * 1024**3
VAST_DEFAULT_MIN_GPU_RAM_MB = 45_000
VAST_DEFAULT_MAX_HOURLY_RATE_USD = 0.75
VAST_DEFAULT_MIN_RELIABILITY = 0.98
VAST_DEFAULT_GPU_KEYWORDS = ("A40", "RTX A6000", "RTX 6000Ada", "L40")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST_REF = re.compile(r"[^\s:@]+(?:/[^\s:@]+)*@sha256:[0-9a-f]{64}")


def collect_openpi_policy_ranking_runpod_preflight(
    *,
    name_prefix: str,
    gpu_type_ids: Sequence[str],
    container_disk_bytes: int,
    capacity_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    inventory_probe: Callable[[str], Mapping[str, Any]],
    max_existing_live_resources: int = 0,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Collect a mutation-free capacity and zero-inventory snapshot.

    This lane downloads the four frozen checkpoints into disposable container
    storage and loads them one at a time. It therefore needs no retained model
    volume, but it does require 24 GiB GPU RAM and 80 GiB container disk.
    """

    requested = [str(value).strip() for value in gpu_type_ids if str(value).strip()]
    capacity = dict(
        capacity_probe(
            {
                "cloudType": "SECURE",
                "gpuTypeIds": requested,
                "min_gpu_ram_mb": MIN_GPU_MEMORY_BYTES // 1_000_000,
                "requires_rtx": True,
            }
        )
    )
    attempt_inventory = dict(inventory_probe(name_prefix))
    inventory = dict(inventory_probe(""))
    viable = [
        dict(row)
        for row in capacity.get("viable_gpu_types", [])
        if isinstance(row, Mapping)
        and row.get("single_gpu_offer_available") is True
        and int(row.get("memory_in_gb") or 0) * 1024**3 >= MIN_GPU_MEMORY_BYTES
        and float(row.get("on_demand_price_usd_per_hour") or 0.0) > 0
    ]
    selected = viable[0] if viable else {}
    live_resource_count = inventory.get("live_resource_count")
    inventory_zero = bool(inventory.get("api_confirmed") is True and live_resource_count == 0)
    inventory_within_concurrency = bool(
        inventory.get("api_confirmed") is True
        and type(live_resource_count) is int
        and 0 <= live_resource_count <= max_existing_live_resources
    )
    provider_api_verified = bool(
        capacity.get("status") == "available"
        and inventory.get("api_confirmed") is True
        and attempt_inventory.get("api_confirmed") is True
    )
    blockers: list[str] = []
    if not requested:
        blockers.append("openpi_gpu_preflight_gpu_types_missing")
    if not provider_api_verified:
        blockers.append("openpi_gpu_preflight_provider_api_not_verified")
    if not selected:
        blockers.append("openpi_gpu_preflight_single_gpu_unavailable")
    if not inventory_within_concurrency:
        blockers.append("openpi_gpu_preflight_billable_inventory_exceeds_concurrency")
    if container_disk_bytes < MIN_CONTAINER_DISK_BYTES:
        blockers.append("openpi_gpu_preflight_container_disk_below_80_gib")
    result: dict[str, Any] = {
        "schema_version": "openpi_policy_ranking_runpod_preflight.v1",
        "status": "verified" if not blockers else "blocked",
        "provider": "runpod",
        "observed_at_epoch": clock(),
        "blockers": sorted(set(blockers)),
        "provider_api_verified": provider_api_verified,
        "provider_inventory_verified_zero": inventory_zero,
        "provider_inventory_within_concurrency_limit": inventory_within_concurrency,
        "maximum_existing_live_resources": max_existing_live_resources,
        "single_gpu_available": bool(selected),
        "gpu_type_id": selected.get("gpu_type_id"),
        "gpu_memory_bytes": int(selected.get("memory_in_gb") or 0) * 1024**3,
        "on_demand_price_usd_per_hour": selected.get("on_demand_price_usd_per_hour"),
        "container_disk_bytes": int(container_disk_bytes),
        "requested_gpu_types": requested,
        "capacity_snapshot": capacity,
        "billable_inventory": inventory,
        "attempt_billable_inventory": attempt_inventory,
        "provider_mutations_performed": 0,
        "reservation_proven": False,
        "raw_secret_values_recorded": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def collect_openpi_policy_ranking_vast_preflight(
    *,
    name_prefix: str,
    container_disk_bytes: int,
    capacity_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    inventory_probe: Callable[[str], Mapping[str, Any]],
    max_hourly_rate_usd: float = VAST_DEFAULT_MAX_HOURLY_RATE_USD,
    min_gpu_ram_mb: int = VAST_DEFAULT_MIN_GPU_RAM_MB,
    min_reliability: float = VAST_DEFAULT_MIN_RELIABILITY,
    preferred_gpu_keywords: Sequence[str] = VAST_DEFAULT_GPU_KEYWORDS,
    max_existing_live_resources: int = 0,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Collect the frozen lane's mutation-free Vast offer snapshot."""

    preferred = [str(value).strip() for value in preferred_gpu_keywords if str(value).strip()]
    request = {
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "min_gpu_ram_mb": int(min_gpu_ram_mb),
        "min_reliability": float(min_reliability),
        "require_avx": True,
        "require_known_supported_isaac_driver": False,
        "require_direct_port": False,
        "preferred_gpu_keywords": preferred,
    }
    capacity = dict(capacity_probe(request))
    attempt_inventory = dict(inventory_probe(name_prefix))
    inventory = dict(inventory_probe(""))
    selected_value = capacity.get("selected_offer")
    selected = dict(selected_value) if isinstance(selected_value, Mapping) else {}
    gpu_ram_mb = int(selected.get("gpu_ram_mb") or 0)
    price = float(selected.get("on_demand_price_usd_per_hour") or 0.0)
    single_gpu = bool(
        selected
        and int(selected.get("num_gpus") or 0) == 1
        and gpu_ram_mb >= int(min_gpu_ram_mb)
        and 0 < price <= float(max_hourly_rate_usd)
    )
    live_resource_count = inventory.get("live_resource_count")
    inventory_zero = bool(inventory.get("api_confirmed") is True and live_resource_count == 0)
    inventory_within_concurrency = bool(
        inventory.get("api_confirmed") is True
        and type(live_resource_count) is int
        and 0 <= live_resource_count <= max_existing_live_resources
    )
    provider_api_verified = bool(
        capacity.get("status") == "available"
        and inventory.get("api_confirmed") is True
        and attempt_inventory.get("api_confirmed") is True
    )
    blockers: list[str] = []
    if not provider_api_verified:
        blockers.append("openpi_gpu_preflight_provider_api_not_verified")
    if not single_gpu:
        blockers.append("openpi_gpu_preflight_single_gpu_unavailable")
    if not inventory_within_concurrency:
        blockers.append("openpi_gpu_preflight_billable_inventory_exceeds_concurrency")
    if container_disk_bytes < MIN_CONTAINER_DISK_BYTES:
        blockers.append("openpi_gpu_preflight_container_disk_below_80_gib")
    result: dict[str, Any] = {
        "schema_version": "openpi_policy_ranking_provider_preflight.v2",
        "status": "verified" if not blockers else "blocked",
        "provider": "vast",
        "observed_at_epoch": clock(),
        "blockers": sorted(set(blockers)),
        "provider_api_verified": provider_api_verified,
        "provider_inventory_verified_zero": inventory_zero,
        "provider_inventory_within_concurrency_limit": inventory_within_concurrency,
        "maximum_existing_live_resources": max_existing_live_resources,
        "single_gpu_available": single_gpu,
        "gpu_type_id": selected.get("gpu_type_id"),
        "gpu_memory_bytes": gpu_ram_mb * 1_000_000,
        # Reserve and launch against the frozen ceiling, not the transient
        # selected offer price, so stale-offer fallback cannot exceed budget.
        "on_demand_price_usd_per_hour": float(max_hourly_rate_usd),
        "selected_offer_price_usd_per_hour": price or None,
        "container_disk_bytes": int(container_disk_bytes),
        "requested_gpu_types": preferred,
        "capacity_request": request,
        "capacity_snapshot": capacity,
        "billable_inventory": inventory,
        "attempt_billable_inventory": attempt_inventory,
        "selected_offer": selected or None,
        "provider_mutations_performed": 0,
        "reservation_proven": False,
        "raw_secret_values_recorded": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def build_openpi_policy_ranking_gpu_admission(
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
    receipt_schema = input_bundle.get("schema_version")
    is_canary = receipt_schema == CANARY_INPUT_RECEIPT_SCHEMA_VERSION
    is_current_reference = receipt_schema == CURRENT_REFERENCE_INPUT_RECEIPT_SCHEMA_VERSION
    manifest = input_bundle.get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    runtime_source = manifest.get("runtime_source")
    runtime_source = runtime_source if isinstance(runtime_source, Mapping) else {}
    if not _COMMIT.fullmatch(expected_commit):
        blockers.append("openpi_gpu_expected_source_commit_invalid")
    if not _COMMIT.fullmatch(source_commit):
        blockers.append("openpi_gpu_release_source_commit_mismatch")
    if is_current_reference:
        if (
            runtime_source.get("commit") != expected_commit
            or runtime_source.get("overlay_required") is not True
            or manifest.get("image_source_commit") != source_commit
        ):
            blockers.append("openpi_gpu_runtime_source_overlay_identity_mismatch")
        archive_url = str(runtime_source.get("archive_url") or "")
        if archive_url != (
            "https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/"
            f"tar.gz/{expected_commit}"
        ) or not _SHA256.fullmatch(str(runtime_source.get("archive_sha256") or "")):
            blockers.append("openpi_gpu_runtime_source_archive_invalid")
    elif source_commit != expected_commit:
        blockers.append("openpi_gpu_release_source_commit_mismatch")
    image_ref = str(release.get("resolved_digest_ref") or "").strip()
    if not _DIGEST_REF.fullmatch(image_ref):
        blockers.append("openpi_gpu_release_image_not_digest_pinned")
    if release.get("schema_version") != "openpi_policy_ranking_gpu_release.v1":
        blockers.append("openpi_gpu_release_schema_invalid")
    if release.get("status") != "passed":
        blockers.append("openpi_gpu_release_not_passed")
    if release.get("runnable_platform") != "linux/amd64":
        blockers.append("openpi_gpu_release_platform_invalid")
    if release.get("openpi_revision") != OPENPI_REVISION:
        blockers.append("openpi_gpu_release_openpi_revision_mismatch")
    if release.get("menagerie_revision") != MENAGERIE_REVISION:
        blockers.append("openpi_gpu_release_menagerie_revision_mismatch")
    if release.get("checkpoint_bytes_embedded") not in {0, False}:
        blockers.append("openpi_gpu_release_embeds_checkpoints")
    if release.get("interiorgs_assets_embedded") is not False:
        blockers.append("openpi_gpu_release_embeds_interiorgs")

    if receipt_schema not in {
        "openpi_policy_ranking_gpu_input_bundle_receipt.v1",
        CANARY_INPUT_RECEIPT_SCHEMA_VERSION,
        CURRENT_REFERENCE_INPUT_RECEIPT_SCHEMA_VERSION,
    }:
        blockers.append("openpi_gpu_input_bundle_receipt_schema_invalid")
    bundle_sha = str(input_bundle.get("bundle_sha256") or "")
    if not _SHA256.fullmatch(bundle_sha):
        blockers.append("openpi_gpu_input_bundle_sha256_invalid")
    expected_manifest_schema = (
        CANARY_INPUT_SCHEMA_VERSION
        if is_canary
        else (
            CURRENT_REFERENCE_INPUT_SCHEMA_VERSION
            if is_current_reference
            else "openpi_policy_ranking_gpu_input_bundle.v2"
        )
    )
    if manifest.get("schema_version") != expected_manifest_schema:
        blockers.append("openpi_gpu_input_bundle_manifest_schema_invalid")
    if manifest.get("raw_3dgs_included") is not False:
        blockers.append("openpi_gpu_input_bundle_contains_raw_3dgs")
    if manifest.get("redistribution_authorized") is not False:
        blockers.append("openpi_gpu_input_bundle_rights_boundary_invalid")
    current_reference_requery = bool(
        is_current_reference
        and manifest.get("purpose") == "label_free_current_reference_same_policy_requery"
    )
    expected_purpose = (
        "private_internal_noncommercial_new_site_diagnostic_canary"
        if is_canary
        else (
            (
                "label_free_current_reference_same_policy_requery"
                if current_reference_requery
                else "label_free_current_reference_real_policy_identity_canary"
            )
            if is_current_reference
            else "private_internal_noncommercial_research_gpu_execution"
        )
    )
    if manifest.get("purpose") != expected_purpose:
        blockers.append("openpi_gpu_input_bundle_purpose_invalid")
    if not is_current_reference and not _SHA256.fullmatch(
        str(manifest.get("background_sha256") or "")
    ):
        blockers.append("openpi_gpu_input_background_sha256_invalid")
    if is_canary:
        if (
            manifest.get("arm_id") != "skeleton_only"
            or manifest.get("label_free") is not True
            or manifest.get("variant") != "center"
            or not str(manifest.get("scene_id") or "")
            or not str(manifest.get("task_instruction") or "")
            or not str(manifest.get("policy_id") or "")
            or not _SHA256.fullmatch(str(manifest.get("protocol_sha256") or ""))
        ):
            blockers.append("openpi_gpu_input_canary_freeze_invalid")
        declared_manifest_sha = str(manifest.get("manifest_sha256") or "")
        manifest_payload = dict(manifest)
        manifest_payload.pop("manifest_sha256", None)
        if declared_manifest_sha != canonical_sha256(manifest_payload):
            blockers.append("openpi_gpu_input_canary_manifest_sha256_invalid")
    elif is_current_reference:
        policy_ids = manifest.get("policy_ids")
        policy_ids_valid = isinstance(policy_ids, list) and (
            (
                current_reference_requery
                and len(policy_ids) == 1
                and policy_ids[0] in {"pi05_droid", "pi0_droid", "pi0_fast_droid"}
                and manifest.get("same_candidate_policy_id") == policy_ids[0]
                and manifest.get("observation_schema")
                == "openpi_current_reference_generated_observation.v1"
            )
            or (
                not current_reference_requery
                and policy_ids == ["pi05_droid", "pi0_droid", "pi0_fast_droid"]
                and manifest.get("same_candidate_policy_id") is None
                and manifest.get("observation_schema")
                in {None, "ctrl_world_public_initial_observation.v1"}
            )
        )
        if (
            not policy_ids_valid
            or manifest.get("requests_per_policy") != 1
            or manifest.get("label_free") is not True
            or manifest.get("confirmation_eligible") is not False
            or manifest.get("physical_outcome_included") is not False
            or manifest.get("checkpoint_weights_included") is not False
        ):
            blockers.append("openpi_gpu_current_reference_canary_freeze_invalid")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 11:
            blockers.append("openpi_gpu_current_reference_file_inventory_invalid")
        declared_manifest_sha = str(manifest.get("manifest_sha256") or "")
        manifest_payload = dict(manifest)
        manifest_payload.pop("manifest_sha256", None)
        if declared_manifest_sha != canonical_sha256(manifest_payload):
            blockers.append("openpi_gpu_current_reference_manifest_sha256_invalid")
    else:
        scenes = manifest.get("scenes")
        scenes = scenes if isinstance(scenes, list) else []
        scene_ids = {
            str(row.get("source_scene_id") or "") for row in scenes if isinstance(row, Mapping)
        }
        scene_kinds = {
            str(row.get("source_scene_kind") or "") for row in scenes if isinstance(row, Mapping)
        }
        if (
            manifest.get("scene_count") != 2
            or len(scenes) != 2
            or len(scene_ids) != 2
            or scene_kinds != {"captured_3dgs", "controlled_nvidia_usd"}
        ):
            blockers.append("openpi_gpu_input_scene_cohort_invalid")
        for index, row in enumerate(scenes):
            if not isinstance(row, Mapping) or not _SHA256.fullmatch(
                str(row.get("background_sha256") or "")
            ):
                blockers.append(f"openpi_gpu_input_scene_background_sha256_invalid:{index}")

    if preflight.get("schema_version") not in {
        "openpi_policy_ranking_runpod_preflight.v1",
        "openpi_policy_ranking_provider_preflight.v2",
    }:
        blockers.append("openpi_gpu_preflight_schema_invalid")
    if preflight.get("status") != "verified":
        blockers.append("openpi_gpu_preflight_not_verified")
    provider_name = str(preflight.get("provider") or "")
    if (
        provider_name not in {"runpod", "vast"}
        or preflight.get("provider_api_verified") is not True
    ):
        blockers.append("openpi_gpu_provider_not_verified")
    observed = preflight.get("observed_at_epoch")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    if type(observed) not in {int, float} or not math.isfinite(float(observed)):
        blockers.append("openpi_gpu_preflight_observed_at_invalid")
    elif not 0 <= now - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("openpi_gpu_preflight_stale_or_future")
    if (
        preflight.get("provider_inventory_verified_zero") is not True
        and preflight.get("provider_inventory_within_concurrency_limit") is not True
    ):
        blockers.append("openpi_gpu_provider_inventory_not_zero")
    if preflight.get("single_gpu_available") is not True:
        blockers.append("openpi_gpu_single_gpu_not_available")
    if int(preflight.get("gpu_memory_bytes") or 0) < MIN_GPU_MEMORY_BYTES:
        blockers.append("openpi_gpu_memory_below_24_gib")
    if not str(preflight.get("gpu_type_id") or "").strip():
        blockers.append("openpi_gpu_type_missing")
    if float(preflight.get("on_demand_price_usd_per_hour") or 0.0) <= 0:
        blockers.append("openpi_gpu_hourly_price_missing")
    if int(preflight.get("container_disk_bytes") or 0) < MIN_CONTAINER_DISK_BYTES:
        blockers.append("openpi_gpu_container_disk_below_80_gib")

    if spend.get("paid_mutation_authorized") is not True:
        blockers.append("openpi_gpu_paid_mutation_not_authorized")
    if spend.get("one_resource_limit") is not True:
        blockers.append("openpi_gpu_one_resource_limit_missing")
    if spend.get("independent_teardown_watchdog") is not True:
        blockers.append("openpi_gpu_independent_watchdog_missing")
    if spend.get("watchdog_armed_before_allocation") is not True:
        blockers.append("openpi_gpu_watchdog_not_armed_before_allocation")
    ttl = spend.get("hard_ttl_seconds")
    max_spend = spend.get("max_spend_usd")
    price = preflight.get("on_demand_price_usd_per_hour")
    if type(ttl) is not int or not 0 < ttl <= MAX_TTL_SECONDS:
        blockers.append("openpi_gpu_ttl_invalid")
    if type(max_spend) not in {int, float} or float(max_spend) <= 0:
        blockers.append("openpi_gpu_max_spend_invalid")
    elif (
        type(ttl) is int
        and type(price) in {int, float}
        and (float(price) * ttl / 3600 > float(max_spend))
    ):
        blockers.append("openpi_gpu_ttl_cost_exceeds_max_spend")
    if spend.get("physical_robot_endpoint_access_allowed") is not False:
        blockers.append("openpi_gpu_physical_robot_endpoint_not_forbidden")

    provider_resource_class = "gpu_render" if provider_name == "vast" else "runpod_provider_adapter"
    shared = build_paid_lane_admission(
        resource_class=provider_resource_class,
        blockers=blockers,
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted" if not blockers and shared["status"] == "admitted" else "blocked",
        "probe_kind": (
            NEW_SITE_CANARY_PROBE_KIND
            if is_canary
            else (
                CURRENT_REFERENCE_POLICY_CANARY_PROBE_KIND if is_current_reference else PROBE_KIND
            )
        ),
        "execution_mode": (
            "new_site_diagnostic_canary"
            if is_canary
            else (
                "current_reference_policy_identity_canary"
                if is_current_reference
                else "full_campaign"
            )
        ),
        "blockers": sorted(set(blockers)),
        "source_commit": source_commit or None,
        "release_image_ref": image_ref or None,
        "input_bundle_sha256": bundle_sha or None,
        "runtime_source_commit": (
            runtime_source.get("commit") if is_current_reference else source_commit
        ),
        "runtime_source_overlay_required": is_current_reference,
        "checkpoint_size_bytes": (
            CURRENT_REFERENCE_CHECKPOINT_BYTES if is_current_reference else CHECKPOINT_BYTES
        ),
        "gpu_type_id": preflight.get("gpu_type_id"),
        "provider": provider_name or None,
        "provider_resource_class": provider_resource_class,
        "limits": {
            "hard_ttl_seconds": ttl,
            "max_spend_usd": max_spend,
            "one_resource": True,
        },
        "shared_paid_lane_admission": shared,
        "claim_boundary": {
            "admission_is_not_gpu_execution": True,
            "admission_is_not_policy_result": True,
            "physical_robot_access_forbidden": True,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name-prefix", default="blueprint-openpi-ranking-")
    parser.add_argument(
        "--gpu-type-id",
        action="append",
        default=["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S"],
    )
    parser.add_argument("--container-disk-gib", type=int, default=100)
    args = parser.parse_args(argv)
    provider = get_render_provider("runpod")
    result = collect_openpi_policy_ranking_runpod_preflight(
        name_prefix=args.name_prefix,
        gpu_type_ids=args.gpu_type_id,
        container_disk_bytes=args.container_disk_gib * 1024**3,
        capacity_probe=provider.capacity_preflight,
        inventory_probe=lambda prefix: provider.billable_inventory(name_prefix=prefix),
    )
    write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROBE_KIND",
    "build_openpi_policy_ranking_gpu_admission",
    "collect_openpi_policy_ranking_vast_preflight",
    "collect_openpi_policy_ranking_runpod_preflight",
]
