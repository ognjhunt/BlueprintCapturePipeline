"""Single-use FLORA evaluation behind the canonical GPU paid-resource gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .adp_isaac_lab_arena_vast import (
    _bounded_spend_gate_open, _remaining_session_live_minutes,
    _vast_credential_file_present, run_arena_native_control_vast,
)
from .common import utc_now_iso, write_json
from .flora_provider_bundle import (
    CLAIM_CLASS, MODEL_ID, MODEL_REVISION, digest, read_object, validate_flora_provider_bundle,
)
from .gpu_render_providers import get_render_provider
from .paid_lane_guard import (
    SPEND_ADMISSION_LOCK_PATH_ENV, PreSpendPreflightBlocked, image_contract_from_ref,
    require_pre_spend_preflight,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION, build_paid_lane_admission, require_paid_resource_admission,
)
from .spend_authority_consumption_root import consumption_root, prepare_consumption_root
from .vast_provider_transfer_upload import EXPECTED_PROVIDER_UPLOAD_BYTES_ENV
from .vast_independent_watchdog_control import (
    _caller_exit_survival_contract, _caller_exit_survival_proven,
    validate_independent_vast_watchdog_names,
)

PROBE_KIND = "flora-episode-benchmark"
GPU_POLICY = {"policy_id": "flora_dual_hopper", "allowed_gpu_keywords": ["H100", "H200"],
              "denied_gpu_keywords": [], "required_gpu_count": 2, "minimum_cuda_max_good": 13.0}
MINIMUM_DRIVER_VERSION = "580.95.05"
PREFIX = "blueprint-flora-"
PROVIDER_OUTPUT_ARCHIVE_MAX_BYTES = 1_000_000_000
# The shared transport permits one initial PUT and two transient retries.
PROVIDER_OUTPUT_UPLOAD_ATTEMPTS = 3


def _resource_name(authority: Mapping[str, Any]) -> str:
    return PREFIX + digest(str(authority["authorization_id"]))[7:39]


def add_flora_allocator_arguments(parser: Any) -> None:
    for option in ("bundle", "bundle-receipt", "attempt-authority", "job-dir"):
        parser.add_argument("--flora-" + option)


def validate_flora_attempt_authority(value: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    authority = dict(value)
    fixed = {"schema_version": "flora_paid_attempt_authority.v1", "status": "authorized",
        "provider": "vast", "probe_kind": PROBE_KIND, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "claim_class": CLAIM_CLASS,
        "maximum_provider_allocations": 1, "automatic_paid_retry_allowed": False,
        "provider_zero_required_after_return": True, "tensor_parallel_size": 2,
        "model_terms_accepted": True, "external_video_processing_authorized": True,
        "research_only": True, "allowed_geolocation_country_codes": ["US"]}
    if any(authority.get(key) != expected for key, expected in fixed.items()):
        raise ValueError("flora_attempt_authority_scope_invalid")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", str(authority.get("authorization_id") or ""))
        or not authority.get("authorized_by") or not authority.get("user_authorization_evidence")
        or not authority.get("model_terms_evidence") or not authority.get("video_rights_evidence")
        or authority.get("authority_digest") != digest({k:v for k,v in authority.items() if k != "authority_digest"})):
        raise ValueError("flora_attempt_authority_identity_invalid")
    for key in ("bundle_sha256", "source_commit", "container_image"):
        if authority.get(key) != bundle.get(key):
            raise ValueError("flora_attempt_authority_bundle_mismatch:" + key)
    for key in ("hard_cap_usd", "max_hourly_rate_usd"):
        number = authority.get(key)
        if type(number) not in {int, float} or not math.isfinite(number) or number <= 0:
            raise ValueError("flora_attempt_authority_budget_invalid")
    ttl = authority.get("hard_ttl_seconds")
    if type(ttl) is not int or not 1800 <= ttl <= 14400:
        raise ValueError("flora_attempt_authority_ttl_invalid")
    if authority["max_hourly_rate_usd"] * ttl / 3600 > authority["hard_cap_usd"]:
        raise ValueError("flora_attempt_authority_compute_cap_insufficient")
    archive_limit = authority.get("provider_output_archive_max_bytes")
    if type(archive_limit) is not int or archive_limit != PROVIDER_OUTPUT_ARCHIVE_MAX_BYTES:
        raise ValueError("flora_attempt_authority_output_archive_limit_invalid")
    for key, minimum in (("expected_provider_download_bytes", 60_000_000_000 + int(bundle.get("bundle_size_bytes", 0))),
                          ("expected_provider_upload_bytes", archive_limit * PROVIDER_OUTPUT_UPLOAD_ATTEMPTS)):
        if type(authority.get(key)) is not int or authority[key] < minimum:
            raise ValueError("flora_attempt_authority_transfer_budget_invalid")
    try:
        expires = datetime.fromisoformat(str(authority["expires_at"]).replace("Z", "+00:00"))
        if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
            raise ValueError("expired")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("flora_attempt_authority_expired_or_invalid") from exc
    return authority


def _consume_once(authority: Mapping[str, Any]) -> dict[str, Any]:
    root = prepare_consumption_root()
    path = root / ("flora-" + str(authority["authorization_id"]) + ".json")
    record = {"schema_version": "flora_paid_attempt_consumption.v1", "consumed_at": utc_now_iso(),
              "authority_digest": authority["authority_digest"], "maximum_provider_allocations": 1}
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise ValueError("flora_attempt_authority_already_consumed") from exc
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {**record, "status": "consumed"}


def _environment_preflight(authority: Mapping[str, Any], bundle: Mapping[str, Any], job: Path) -> list[str]:
    blockers: list[str] = []
    label = _resource_name(authority)
    validate_independent_vast_watchdog_names(pod_name_prefix=PREFIX, resource_name_exact=label)
    if not _caller_exit_survival_proven(_caller_exit_survival_contract()):
        blockers.append("independent_vast_watchdog_caller_exit_survival_unproven")
    if (consumption_root() / ("flora-" + str(authority["authorization_id"]) + ".json")).exists():
        blockers.append("flora_attempt_authority_already_consumed")
    minutes = _remaining_session_live_minutes(job=job, hard_cap_usd=authority["hard_cap_usd"],
        hard_ttl_seconds=authority["hard_ttl_seconds"], max_hourly_rate_usd=authority["max_hourly_rate_usd"])
    if minutes < 30:
        blockers.append("flora_remaining_live_window_insufficient")
    provider = get_render_provider("vast")
    inventory = provider.billable_inventory(name_prefix="")
    write_json(job / "preflight/provider_inventory.json", inventory)
    if (inventory.get("status") != "observed" or inventory.get("api_confirmed") is not True
            or inventory.get("live_resource_count") != 0):
        blockers.append("flora_provider_zero_preflight_unproven")
    capacity = provider.capacity_preflight({"max_hourly_rate_usd": authority["max_hourly_rate_usd"],
        "hard_cap_usd": authority["hard_cap_usd"], "hard_ttl_seconds": authority["hard_ttl_seconds"],
        "max_live_minutes": minutes, "container_disk_gb": 200, "min_gpu_ram_mb": 80_000,
        "min_compute_cap": 900, "max_compute_cap": 900, "gpu_selection_policy": GPU_POLICY,
        "allowed_geolocation_country_codes": authority["allowed_geolocation_country_codes"],
        "preferred_gpu_keywords": ["H200", "H100"], "minimum_driver_version": MINIMUM_DRIVER_VERSION,
        "require_known_supported_isaac_driver": False, "require_direct_port": True,
        "expected_provider_download_bytes": authority["expected_provider_download_bytes"],
        "expected_provider_upload_bytes": authority["expected_provider_upload_bytes"]})
    write_json(job / "preflight/provider_capacity.json", capacity)
    if capacity.get("status") != "available":
        blockers.extend(capacity.get("blockers") or ["flora_provider_capacity_unavailable"])
    try:
        require_pre_spend_preflight(lane=PROBE_KIND, provider="vast", credential_present=_vast_credential_file_present(),
            capacity_evidence={"available": capacity.get("status") == "available", "detail": "live exact-two-device offer selection"},
            image_contract=image_contract_from_ref(str(bundle["container_image"])),
            runtime_contract={"startup_marker": "vast_instance_started_or_blocked", "progress_marker": "vast_provider_bundle_progress",
                "startup_timeout_seconds": minutes * 60, "no_progress_timeout_seconds": 1800},
            spend_gate_open=_bounded_spend_gate_open(max_hourly_rate_usd=authority["max_hourly_rate_usd"],
                hard_cap_usd=authority["hard_cap_usd"], remaining_live_minutes=minutes), record_dir=job / "preflight",
            spend_admission_lock=None if str(os.getenv(SPEND_ADMISSION_LOCK_PATH_ENV) or "").strip() else {})
    except PreSpendPreflightBlocked as exc:
        blockers.extend(exc.preflight.get("blockers") or [])
    return sorted(set(blockers))


def run_flora_allocator_lane(args: Any, source_context: tuple[list[str], str]) -> int:
    blockers, source_commit = source_context
    blockers = list(blockers)
    job = Path(args.flora_job_dir or "output/flora_allocator_blocked").resolve()
    authority = bundle = None
    if args.provider != "vast":
        blockers.append("flora_provider_must_be_vast")
    for field in ("flora_bundle", "flora_bundle_receipt", "flora_attempt_authority", "flora_job_dir", "admission_out", "adapter_output"):
        if not getattr(args, field, None):
            blockers.append(field + "_missing")
    if not blockers:
        try:
            bundle = validate_flora_provider_bundle(read_object(args.flora_bundle_receipt), bundle_path=args.flora_bundle)
            if bundle["source_commit"] != source_commit:
                raise ValueError("flora_source_commit_mismatch")
            authority = validate_flora_attempt_authority(read_object(args.flora_attempt_authority), bundle)
            blockers.extend(_environment_preflight(authority, bundle, job))
        except (ValueError, OSError, KeyError) as exc:
            blockers.append(str(exc))
    binding = {"probe_kind": PROBE_KIND, "source_commit": source_commit,
               "bundle_sha256": bundle.get("bundle_sha256") if bundle else None,
               "authority_digest": authority.get("authority_digest") if authority else None}
    admission = build_paid_lane_admission(resource_class="vast_provider_adapter", blockers=blockers)
    admission.update(allocation_binding=binding, allocation_binding_digest=digest(binding), claim_class=CLAIM_CLASS)
    write_json(Path(args.admission_out) if args.admission_out else job / "admission.json", admission)
    result: dict[str, Any] = {"status": "blocked" if blockers else "dry_run_ready", "blockers": sorted(set(blockers)),
                              "provider_mutations_performed": 0, "claim_class": CLAIM_CLASS}
    if args.execute and not blockers and authority is not None and bundle is not None:
        grant = require_paid_resource_admission(admission, resource_class="vast_provider_adapter",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
        try:
            # Consume before staging; even failed staging cannot authorize a retry.
            consumption = _consume_once(authority)
            result = run_arena_native_control_vast(approval_path=args.flora_attempt_authority, job_dir=job,
                prepared_bundle=bundle, paid_resource_admission_grant=grant, execute=True,
                max_hourly_rate_usd=authority["max_hourly_rate_usd"], hard_cap_usd=authority["hard_cap_usd"],
                hard_ttl_seconds=authority["hard_ttl_seconds"], expected_output_filename="evaluator_runtime_result.json",
                container_image=bundle["container_image"], ngc_image_login_mode="never", provider_bundle_kind="evaluator",
                result_schema_version="flora_vast_result.v1", object_store_key_prefix="blueprint/flora-episode-benchmark",
                instance_label_prefix=PREFIX, instance_label_exact=_resource_name(authority), blocker_prefix="flora",
                min_gpu_ram_mb=80_000, min_compute_cap=900, max_compute_cap=900, minimum_driver_version=MINIMUM_DRIVER_VERSION,
                require_known_supported_isaac_driver=False, enable_isaac_smoke=False, forward_hf_token=False,
                stale_offer_create_retry_limit=0, candidate_policy_query_expected=False,
                preferred_gpu_keywords=("H200", "H100"), gpu_selection_policy=GPU_POLICY,
                require_independent_watchdog=True, authorization_consumption=consumption,
                require_direct_port=True,
                expected_provider_download_bytes=authority["expected_provider_download_bytes"],
                expected_provider_upload_bytes=authority["expected_provider_upload_bytes"],
                provider_runtime_environment={
                    EXPECTED_PROVIDER_UPLOAD_BYTES_ENV: str(authority["provider_output_archive_max_bytes"]),
                },
                allowed_geolocation_country_codes=authority["allowed_geolocation_country_codes"])
        except (ValueError, OSError) as exc:
            result = {"status": "blocked", "blockers": [str(exc)], "provider_mutations_performed": None,
                      "provider_zero_verified": False}
    write_json(Path(args.adapter_output) if args.adapter_output else job / "result.json", result)
    success = result.get("status") in {"dry_run_ready", "completed"}
    print(json.dumps({"success": success, "status": result.get("status")}))
    return 0 if success else 2
