"""Fail-closed admission for one SAM 3.1 source-track GPU canary."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_artifact_manifest import seal_lane_terminal_artifacts


REQUEST_SCHEMA_VERSION = "semantic_sam31_gpu_canary_request.v1"
PREFLIGHT_SCHEMA_VERSION = "semantic_sam31_gpu_provider_preflight.v1"
ADMISSION_SCHEMA_VERSION = "semantic_sam31_gpu_canary_admission.v1"
PROBE_KIND = "semantic-sam31-source-tracks"
OPERATION = "source_track_canary"
#: The allocator lane this admission belongs to, used to bind the terminal
#: artifact manifest to the same lane the paid path seals under.
ALLOCATOR_LANE = "semantic_sam31_source_tracks"
CHECKPOINT_FAMILY = "facebook/sam3.1"
OFFICIAL_CODE_REVISION = "96914d2425f90a64f45ca977c2b5165418099543"
CHECKPOINT_REPOSITORY_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
CHECKPOINT_DIGEST = "sha256:0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
LICENSE_TERMS_DIGEST = "sha256:4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be"
MIN_GPU_MEMORY_BYTES = 24 * 1024**3
MIN_CONTAINER_DISK_BYTES = 40 * 1024**3
MAX_PREFLIGHT_AGE_SECONDS = 300
MAX_TTL_SECONDS = 3_600
MAX_RETRY_CAP = 0
MAX_CANARY_FRAMES = 128
MAX_CANARY_INPUT_BUNDLE_BYTES = 512 * 1024**2
SOURCE_PROFILES = {
    "iphone_arkit_lidar",
    "iphone_arkit_non_lidar",
    "camera_360_equirectangular",
    "camera_360_native",
    "monocular_video",
}

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value)))


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def collect_sam31_vast_preflight(
    *,
    name_prefix: str,
    container_disk_bytes: int,
    watchdog: Mapping[str, Any],
    conflicting_owner_present: bool,
    capacity_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    inventory_probe: Callable[[str], Mapping[str, Any]],
    max_hourly_rate_usd: float,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Collect mutation-free capacity, watchdog, and provider-zero evidence."""

    capacity_request = {
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "min_gpu_ram_mb": 24_000,
        "min_reliability": 0.98,
        "require_avx": True,
        "require_known_supported_isaac_driver": False,
        "require_direct_port": False,
        "preferred_gpu_keywords": ["L40S", "L40", "A40", "RTX 6000Ada", "RTX A6000"],
    }
    capacity = dict(capacity_probe(capacity_request))
    scoped_inventory = dict(inventory_probe(name_prefix))
    global_inventory = dict(inventory_probe(""))
    raw_offer = capacity.get("selected_offer")
    offer = dict(raw_offer) if isinstance(raw_offer, Mapping) else {}
    memory_bytes = int(offer.get("gpu_ram_mb") or 0) * 1_000_000
    hourly_rate = float(
        offer.get("on_demand_price_usd_per_hour") or offer.get("hourly_rate_usd") or 0
    )
    provider_api_verified = bool(
        capacity.get("status") == "available"
        and scoped_inventory.get("api_confirmed") is True
        and global_inventory.get("api_confirmed") is True
    )
    provider_zero = bool(
        scoped_inventory.get("live_resource_count") == 0
        and global_inventory.get("live_resource_count") == 0
    )
    watchdog_value = dict(watchdog)
    blockers: list[str] = []
    if not provider_api_verified:
        blockers.append("sam31_gpu_provider_api_not_verified")
    if not provider_zero:
        blockers.append("sam31_gpu_provider_inventory_not_zero")
    if conflicting_owner_present:
        blockers.append("sam31_gpu_conflicting_owner_present")
    if (
        watchdog_value.get("status") != "armed"
        or watchdog_value.get("independent_process") is not True
    ):
        blockers.append("sam31_gpu_independent_watchdog_not_armed")
    if not (
        offer
        and memory_bytes >= MIN_GPU_MEMORY_BYTES
        and 0 < hourly_rate <= float(max_hourly_rate_usd)
    ):
        blockers.append("sam31_gpu_single_gpu_unavailable")
    if container_disk_bytes < MIN_CONTAINER_DISK_BYTES:
        blockers.append("sam31_gpu_container_disk_below_floor")
    result = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "verified" if not blockers else "blocked",
        "provider": "vast",
        "observed_at_epoch": float(clock()),
        "provider_api_verified": provider_api_verified,
        "provider_inventory_verified_zero": provider_zero,
        "conflicting_owner_present": bool(conflicting_owner_present),
        "watchdog": watchdog_value,
        "single_gpu_available": bool(offer and memory_bytes >= MIN_GPU_MEMORY_BYTES),
        "gpu_type_id": offer.get("gpu_name") or offer.get("gpu_type_id"),
        "gpu_memory_bytes": memory_bytes,
        "container_disk_bytes": int(container_disk_bytes),
        "on_demand_price_usd_per_hour": hourly_rate or None,
        "selected_offer": offer or None,
        "capacity_request": capacity_request,
        "capacity_snapshot": capacity,
        "scoped_billable_inventory": scoped_inventory,
        "global_billable_inventory": global_inventory,
        "blockers": sorted(set(blockers)),
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "capacity_reserved": False,
        "proof_effect": "none",
        "claim_ceiling": "provider_capacity_and_zero_inventory_snapshot_only",
    }
    result["preflight_digest"] = canonical_digest(result, digest_field="preflight_digest")
    return result


def build_sam31_gpu_canary_admission(
    *,
    request: Mapping[str, Any],
    preflight: Mapping[str, Any],
    provider: str,
    expected_source_commit: str,
    checkout_source_commit: str,
    checkout_clean: bool,
    max_spend_usd: float | None,
    hard_ttl_seconds: int | None,
    retry_cap: int | None,
    authority_id: str | None,
    execute: bool,
    execution_adapter_qualified: bool = False,
    observed_now_epoch: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind exact model, input, legal, budget, and provider facts before mutation."""

    source = _copy(request)
    provider_snapshot = _copy(preflight)
    blockers: list[str] = []
    supplied_digest = source.pop("request_digest", None)
    expected_request_digest = canonical_digest(source, digest_field="request_digest")
    source["request_digest"] = supplied_digest
    if supplied_digest != expected_request_digest:
        blockers.append("sam31_gpu_request_digest_mismatch")
    exact_values = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": OPERATION,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "checkpoint_repository_revision": CHECKPOINT_REPOSITORY_REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "license_terms_digest": LICENSE_TERMS_DIGEST,
        "proof_effect": "none",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    for field, expected in exact_values.items():
        if source.get(field) != expected:
            blockers.append(f"sam31_gpu_{field}_mismatch")
    if source.get("source_profile") not in SOURCE_PROFILES:
        blockers.append("sam31_gpu_source_profile_unsupported")
    if source.get("source_commit_sha") != expected_source_commit:
        blockers.append("sam31_gpu_request_source_commit_mismatch")
    if _COMMIT.fullmatch(expected_source_commit) is None:
        blockers.append("sam31_gpu_expected_source_commit_invalid")
    if checkout_source_commit != expected_source_commit:
        blockers.append("sam31_gpu_checkout_source_commit_mismatch")
    if not checkout_clean:
        blockers.append("sam31_gpu_checkout_not_clean")
    if _IMAGE.fullmatch(str(source.get("worker_image_digest") or "")) is None:
        blockers.append("sam31_gpu_worker_image_digest_invalid")
    for field in (
        "worker_stack_manifest_digest",
        "input_bundle_digest",
        "source_track_run_request_digest",
        "capture_digest",
        "retained_video_digest",
        "camera_solution_digest",
        "frame_registry_digest",
        "license_use_authorization_digest",
        "privacy_use_authorization_digest",
        "trade_controls_review_digest",
        "execution_authorization_digest",
    ):
        if _DIGEST.fullmatch(str(source.get(field) or "")) is None:
            blockers.append(f"sam31_gpu_{field}_invalid")
    frame_count = source.get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or not 1 <= frame_count <= MAX_CANARY_FRAMES
    ):
        blockers.append("sam31_gpu_frame_count_invalid")
    bundle_size = source.get("input_bundle_size_bytes")
    if (
        isinstance(bundle_size, bool)
        or not isinstance(bundle_size, int)
        or not 1 <= bundle_size <= MAX_CANARY_INPUT_BUNDLE_BYTES
    ):
        blockers.append("sam31_gpu_input_bundle_size_invalid")
    required_true = (
        "checkpoint_access_authorized",
        "commercial_evidence_use_authorized",
        "rights_cleared_for_external_processing",
        "privacy_safe_for_external_processing",
        "trade_controls_reviewed",
        "model_self_grading_forbidden",
        "metric_claim_upgrade_forbidden",
        "physics_claim_upgrade_forbidden",
        "physical_claim_upgrade_forbidden",
        "network_access_during_inference_forbidden",
    )
    for field in required_true:
        if source.get(field) is not True:
            blockers.append(f"sam31_gpu_{field}_required")
    if source.get("customer_data_training_allowed") is not False:
        blockers.append("sam31_gpu_customer_training_must_be_false")
    allowed_uses = source.get("allowed_evidence_uses")
    if allowed_uses != ["semantic_analysis"]:
        blockers.append("sam31_gpu_allowed_evidence_uses_invalid")

    if provider != "vast" or provider_snapshot.get("provider") != "vast":
        blockers.append("sam31_gpu_vast_first_required")
    if provider_snapshot.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        blockers.append("sam31_gpu_preflight_schema_invalid")
    if provider_snapshot.get("status") != "verified":
        blockers.append("sam31_gpu_preflight_not_verified")
    if provider_snapshot.get("provider_api_verified") is not True:
        blockers.append("sam31_gpu_provider_api_not_verified")
    if provider_snapshot.get("provider_inventory_verified_zero") is not True:
        blockers.append("sam31_gpu_provider_inventory_not_zero")
    if provider_snapshot.get("conflicting_owner_present") is not False:
        blockers.append("sam31_gpu_conflicting_owner_present")
    watchdog = provider_snapshot.get("watchdog")
    watchdog = watchdog if isinstance(watchdog, Mapping) else {}
    if watchdog.get("status") != "armed" or watchdog.get("independent_process") is not True:
        blockers.append("sam31_gpu_independent_watchdog_not_armed")
    if provider_snapshot.get("single_gpu_available") is not True:
        blockers.append("sam31_gpu_single_gpu_unavailable")
    memory = provider_snapshot.get("gpu_memory_bytes")
    if not isinstance(memory, int) or isinstance(memory, bool) or memory < MIN_GPU_MEMORY_BYTES:
        blockers.append("sam31_gpu_memory_below_floor")
    disk = provider_snapshot.get("container_disk_bytes")
    if not isinstance(disk, int) or isinstance(disk, bool) or disk < MIN_CONTAINER_DISK_BYTES:
        blockers.append("sam31_gpu_container_disk_below_floor")
    hourly = provider_snapshot.get("on_demand_price_usd_per_hour")
    if not _finite(hourly, minimum=0.000001):
        blockers.append("sam31_gpu_hourly_price_invalid")
    observed = provider_snapshot.get("observed_at_epoch")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    if not _finite(observed):
        blockers.append("sam31_gpu_preflight_observed_at_invalid")
    elif not 0 <= now - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("sam31_gpu_preflight_stale_or_future")

    if not _finite(max_spend_usd, minimum=0.000001):
        blockers.append("sam31_gpu_explicit_budget_missing")
    if (
        not isinstance(hard_ttl_seconds, int)
        or isinstance(hard_ttl_seconds, bool)
        or not 1 <= hard_ttl_seconds <= MAX_TTL_SECONDS
    ):
        blockers.append("sam31_gpu_explicit_ttl_invalid")
    if (
        not isinstance(retry_cap, int)
        or isinstance(retry_cap, bool)
        or not 0 <= retry_cap <= MAX_RETRY_CAP
    ):
        blockers.append("sam31_gpu_explicit_retry_cap_invalid")
    if not isinstance(authority_id, str) or not authority_id.strip():
        blockers.append("sam31_gpu_paid_authority_missing")
    for supplied, field, blocker in (
        (max_spend_usd, "max_spend_usd", "sam31_gpu_budget_binding_mismatch"),
        (hard_ttl_seconds, "hard_ttl_seconds", "sam31_gpu_ttl_binding_mismatch"),
        (retry_cap, "retry_cap", "sam31_gpu_retry_binding_mismatch"),
        (authority_id, "authority_id", "sam31_gpu_authority_binding_mismatch"),
    ):
        if supplied != source.get(field):
            blockers.append(blocker)
    worst_case = (
        float(hourly) * float(hard_ttl_seconds) / 3600.0
        if _finite(hourly, minimum=0.000001)
        and isinstance(hard_ttl_seconds, int)
        and not isinstance(hard_ttl_seconds, bool)
        else math.inf
    )
    if _finite(max_spend_usd, minimum=0.000001) and worst_case > float(max_spend_usd):
        blockers.append("sam31_gpu_budget_below_worst_case_cost")
    if execute and not execution_adapter_qualified and not blockers:
        blockers.append("sam31_vast_execution_adapter_not_qualified")

    bound_request = {
        **source,
        "request_digest": expected_request_digest,
        "bound_provider": provider,
        "bound_preflight_digest": canonical_digest(provider_snapshot),
        "bound_checkout_source_commit": checkout_source_commit,
        "bound_checkout_clean": checkout_clean,
        "provider_mutation_authorized": bool(
            execute and execution_adapter_qualified and not blockers
        ),
    }
    bound_request["bound_request_digest"] = canonical_digest(
        bound_request, digest_field="bound_request_digest"
    )
    status = (
        "execute_ready"
        if execute and execution_adapter_qualified and not blockers
        else ("dry_run_ready" if not blockers else "blocked")
    )
    admission = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": status,
        "probe_kind": PROBE_KIND,
        "provider": provider,
        "blockers": sorted(set(blockers)),
        "request_digest": expected_request_digest,
        "bound_request_digest": bound_request["bound_request_digest"],
        "source_commit_sha": checkout_source_commit,
        "worker_image_digest": source.get("worker_image_digest"),
        "checkpoint_digest": source.get("checkpoint_digest"),
        "input_bundle_digest": source.get("input_bundle_digest"),
        "input_bundle_size_bytes": source.get("input_bundle_size_bytes"),
        "frame_count": source.get("frame_count"),
        "max_spend_usd": max_spend_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": retry_cap,
        "authority_id": authority_id,
        "watchdog_armed": watchdog.get("status") == "armed",
        "provider_zero_verified": provider_snapshot.get("provider_inventory_verified_zero") is True,
        "provider_mutations_performed": 0,
        "paid_execution_started": False,
        "execution_adapter_qualified": bool(execution_adapter_qualified),
        "raw_secret_values_recorded": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "paid_gpu_admission_only",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    admission["admission_digest"] = canonical_digest(admission, digest_field="admission_digest")
    return admission, bound_request


def seal_sam31_terminal_admission(
    admission: Mapping[str, Any], *, adapter_output: str | Path
) -> dict[str, Any]:
    """Give the terminal adapter result the two paths its launch contract reads.

    ``adapter_output`` is not an incidental copy of the admission: it is the
    ``terminal_contract.result_path`` of the live profile that makes
    ``semantic-sam31-source-tracks`` website-reachable, and that contract's
    ``required_path_fields`` are read straight off it. A result written there
    without ``artifact_manifest_path`` and ``teardown_manifest_path`` ends
    ``allocator_terminal_artifact_missing:`` for both no matter what happened on
    the provider -- the failure that cost a paid run on 2026-08-13 -- and any
    evidence the attempt retained sits on disk with nothing naming it.

    The attempt root is the adapter result's own directory, because that is the
    expression the paid lane builds ``<root>/vast_provider_run`` from. Sealing a
    root the evidence is not under is the #501 defect: it reports ``completed``
    with ``blockers: []`` while the launch blocks on artifacts the sealer never
    looked for.

    Both fields are named unconditionally, ``None`` when there is nothing to
    name, so an attempt that produced no manifest is distinguishable from one
    whose manifest went missing. The shared seal only ever adds blockers or
    downgrades a status, so this cannot turn a blocked attempt into a passing
    one.
    """

    terminal = dict(admission)
    terminal.setdefault("artifact_manifest_path", None)
    terminal.setdefault("teardown_manifest_path", None)
    return seal_lane_terminal_artifacts(
        terminal,
        attempt_root=Path(adapter_output).expanduser().resolve().parent,
        lane=ALLOCATOR_LANE,
    )


def prepare_sam31_gpu_canary(
    *,
    request_path: str | Path,
    preflight_path: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    provider: str,
    expected_source_commit: str,
    checkout_source_commit: str,
    checkout_clean: bool,
    max_spend_usd: float | None,
    hard_ttl_seconds: int | None,
    retry_cap: int | None,
    authority_id: str | None,
    execute: bool,
    execution_adapter_qualified: bool = False,
) -> dict[str, Any]:
    admission, bound = build_sam31_gpu_canary_admission(
        request=_read(request_path),
        preflight=_read(preflight_path),
        provider=provider,
        expected_source_commit=expected_source_commit,
        checkout_source_commit=checkout_source_commit,
        checkout_clean=checkout_clean,
        max_spend_usd=max_spend_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        retry_cap=retry_cap,
        authority_id=authority_id,
        execute=execute,
        execution_adapter_qualified=execution_adapter_qualified,
    )
    write_json(Path(admission_out), admission)
    write_json(Path(bound_request_out), bound)
    if admission["status"] != "execute_ready":
        # The admission receipt keeps its own digest-bound shape; only the
        # terminal allocator result gains the paths the launch contract reads.
        write_json(
            Path(adapter_output),
            seal_sam31_terminal_admission(admission, adapter_output=adapter_output),
        )
    return admission


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "ALLOCATOR_LANE",
    "CHECKPOINT_DIGEST",
    "CHECKPOINT_REPOSITORY_REVISION",
    "LICENSE_TERMS_DIGEST",
    "MAX_CANARY_INPUT_BUNDLE_BYTES",
    "OFFICIAL_CODE_REVISION",
    "PREFLIGHT_SCHEMA_VERSION",
    "PROBE_KIND",
    "REQUEST_SCHEMA_VERSION",
    "build_sam31_gpu_canary_admission",
    "collect_sam31_vast_preflight",
    "prepare_sam31_gpu_canary",
    "seal_sam31_terminal_admission",
]
