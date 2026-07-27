"""Fail-closed paid-GPU admission for the Cosmos3 successor smoke canary.

This module is an adapter behind ``paid_resource_allocator gpu-canary``.  It
cannot be used as a standalone launcher.  The first paid phase is restricted to
one Vast RTX PRO 6000 Blackwell instance: it must prove the exact pinned stack
before the already-frozen ten-rollout smoke matrix is allowed to continue.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .policy_ranking_successor_cosmos import (
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    COSMOS_FRAMEWORK_REVISION,
    COSMOS_REVISION,
    EXPERIMENT_ID,
    VLLM_IMAGE,
    VLLM_IMAGE_DIGEST,
    canonical_sha256,
    validate_smoke_inventory_manifest,
)
from .vast_wam_authorized_runner import run_vast_wam_authorized_runner
from .vast_provider_adapter import (
    VAST_API_GATE_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    _env_truthy as _vast_env_truthy,
)


PROBE_KIND = "policy-ranking-successor-cosmos"
SCHEMA_VERSION = "policy_ranking_successor_gpu_admission.v1"
AUTHORIZATION_SCHEMA = "policy_ranking_successor_compute_authorization.v1"
PREFLIGHT_SCHEMA = "policy_ranking_successor_vast_preflight.v1"
BUNDLE_SCHEMA = "policy_ranking_successor_cosmos_bundle.v1"
PUBLIC_IMAGE = f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}"

MAX_COMPUTE_CAP_USD = 6.0
MAX_HOURLY_RATE_USD = 1.05
TARGET_SPEND_USD = 3.25
HARD_TTL_SECONDS = 10_800
DISK_GB = 250
MIN_GPU_RAM_MB = 95_000
MIN_RELIABILITY = 0.98
MAX_PREFLIGHT_AGE_SECONDS = 900
AUTHORIZATION_ID = "policy-ranking-successor-cosmos-smoke-20260727-cap-6p00-v2"
PREDECESSOR_AUTHORIZATION_ID = (
    "policy-ranking-successor-cosmos-smoke-20260727-cap-3p25-v1"
)
AUTHORIZATION_CONSUMPTION_ROOT = (
    Path.home() / ".blueprint-spend-authority" / "consumed"
)
EXPECTED_BUNDLE_SHA256 = (
    "f733fdddf9ef3780d303d5672c2c30c767807c43e87b4d9f5fc6dd860256c873"
)
EXPECTED_BUNDLE_SIZE_BYTES = 294_148
EXPECTED_EMBEDDED_INPUT_HASHES = {
    "initial_observation_sha256": (
        "8843f0fc9c68914dfb62222c961db19b37a5f155e602ff4a545eea1dcf42636d"
    ),
    "smoke_inventory_sha256": (
        "c925d168c166ec7bf53ed9252a33a937416a3f6fa9ecad8a01d0e201920a07aa"
    ),
    "action_streams_sha256": (
        "9ece91bb2a2e50165bf21c6fa62afd25831281ade80607d7a6987e29e4f80c58"
    ),
}
RTX_ALLOWED_KEYWORDS = ("RTX PRO 6000",)
RTX_SELECTION_POLICY: Mapping[str, Any] = {
    "policy_id": "policy_ranking_successor_rtx_pro_6000_blackwell_preflight",
    "denied_gpu_keywords": (),
    "allowed_gpu_keywords": RTX_ALLOWED_KEYWORDS,
    "reason": (
        "provisional exact-stack compatibility preflight; scientific rollouts "
        "continue only after the Blackwell stack passes"
    ),
}

REQUIRED_BUNDLE_ENTRIES = frozenset(
    {
        "provider_runtime/wam_provider_runtime_runner.py",
        "provider_runtime/run_wam_provider_runtime.sh",
        "provider_runtime/wam_provider_runtime_manifest.json",
        "provider_runtime/wam_rollout_input_manifest.json",
        "provider_runtime/cosmos3_input/initial_observation.png",
        "provider_runtime/cosmos3_input/smoke_request_inventory.json",
        "provider_runtime/cosmos3_input/action_streams.json",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_successor_bundle(
    path: str | Path,
    *,
    receipt: Mapping[str, Any] | None = None,
    smoke_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    blockers: list[str] = []
    names: set[str] = set()
    manifest: Mapping[str, Any] = {}
    embedded_hashes: dict[str, str] = {}
    if not resolved.is_file():
        blockers.append("successor_cosmos_provider_bundle_missing")
    else:
        try:
            with zipfile.ZipFile(resolved) as archive:
                names = set(archive.namelist())
                bad_member = archive.testzip()
                if bad_member is not None:
                    blockers.append("successor_cosmos_provider_bundle_zip_invalid")
                if "provider_runtime/wam_provider_runtime_manifest.json" in names:
                    manifest_value = json.loads(
                        archive.read(
                            "provider_runtime/wam_provider_runtime_manifest.json"
                        ).decode("utf-8")
                    )
                    manifest = _mapping(manifest_value)
                embedded_hashes = {
                    "initial_observation_sha256": hashlib.sha256(
                        archive.read(
                            "provider_runtime/cosmos3_input/initial_observation.png"
                        )
                    ).hexdigest(),
                    "smoke_inventory_sha256": canonical_sha256(
                        json.loads(
                            archive.read(
                                "provider_runtime/cosmos3_input/smoke_request_inventory.json"
                            ).decode("utf-8")
                        )
                    ),
                    "action_streams_sha256": canonical_sha256(
                        json.loads(
                            archive.read(
                                "provider_runtime/cosmos3_input/action_streams.json"
                            ).decode("utf-8")
                        )
                    ),
                }
        except (
            OSError,
            KeyError,
            ValueError,
            zipfile.BadZipFile,
            json.JSONDecodeError,
        ):
            blockers.append("successor_cosmos_provider_bundle_unreadable")
    missing = sorted(REQUIRED_BUNDLE_ENTRIES - names)
    if missing:
        blockers.append("successor_cosmos_provider_bundle_entries_missing")
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        blockers.append("successor_cosmos_provider_bundle_manifest_invalid")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        blockers.append("successor_cosmos_provider_bundle_experiment_mismatch")
    if manifest.get("checkpoint_revision") != CHECKPOINT_REVISION:
        blockers.append("successor_cosmos_provider_bundle_checkpoint_mismatch")
    if manifest.get("public_image") != PUBLIC_IMAGE:
        blockers.append("successor_cosmos_provider_bundle_image_mismatch")
    receipt_value = _mapping(receipt)
    bundle_sha256 = _sha256_file(resolved) if resolved.is_file() else None
    bundle_size_bytes = resolved.stat().st_size if resolved.is_file() else 0
    if receipt_value.get("schema_version") != (
        "policy_ranking_successor_cosmos_bundle_receipt.v1"
    ):
        blockers.append("successor_cosmos_provider_bundle_receipt_invalid")
    if receipt_value.get("experiment_id") != EXPERIMENT_ID:
        blockers.append("successor_cosmos_provider_bundle_receipt_experiment_mismatch")
    if receipt_value.get("bundle_sha256") != bundle_sha256:
        blockers.append("successor_cosmos_provider_bundle_receipt_hash_mismatch")
    if receipt_value.get("bundle_size_bytes") != bundle_size_bytes:
        blockers.append("successor_cosmos_provider_bundle_receipt_size_mismatch")
    if bundle_sha256 != EXPECTED_BUNDLE_SHA256:
        blockers.append("successor_cosmos_provider_bundle_frozen_hash_mismatch")
    if bundle_size_bytes != EXPECTED_BUNDLE_SIZE_BYTES:
        blockers.append("successor_cosmos_provider_bundle_frozen_size_mismatch")
    for key, expected in EXPECTED_EMBEDDED_INPUT_HASHES.items():
        if embedded_hashes.get(key) != expected or receipt_value.get(key) != expected:
            blockers.append(f"successor_cosmos_provider_bundle_receipt_{key}_mismatch")
    if smoke_inventory is not None and receipt_value.get(
        "smoke_inventory_sha256"
    ) != canonical_sha256(smoke_inventory):
        blockers.append("successor_cosmos_external_smoke_inventory_hash_mismatch")
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "bundle_path": str(resolved),
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle_size_bytes,
        "required_entry_count": len(REQUIRED_BUNDLE_ENTRIES),
        "manifest": dict(manifest),
        "embedded_input_hashes": embedded_hashes,
    }


def build_successor_gpu_admission(
    *,
    authorization: Mapping[str, Any],
    environment: Mapping[str, Any],
    smoke_inventory: Mapping[str, Any],
    provider_preflight: Mapping[str, Any],
    bundle_inspection: Mapping[str, Any],
    expected_source_commit: str,
    execute: bool,
    observed_now_epoch: float | None = None,
    initial_blockers: Sequence[str] = (),
) -> dict[str, Any]:
    blockers = list(initial_blockers)
    checkpoint = _mapping(environment.get("checkpoint"))
    upstream = _mapping(environment.get("upstream_source"))
    cosmos = _mapping(upstream.get("cosmos"))
    framework = _mapping(upstream.get("cosmos_framework"))
    vllm = _mapping(upstream.get("vllm_omni"))
    if environment.get("experiment_id") != EXPERIMENT_ID:
        blockers.append("successor_environment_experiment_mismatch")
    if cosmos.get("revision") != COSMOS_REVISION:
        blockers.append("successor_cosmos_revision_mismatch")
    if framework.get("revision") != COSMOS_FRAMEWORK_REVISION:
        blockers.append("successor_framework_revision_mismatch")
    if checkpoint.get("repository") != CHECKPOINT_REPOSITORY:
        blockers.append("successor_checkpoint_repository_mismatch")
    if checkpoint.get("revision") != CHECKPOINT_REVISION:
        blockers.append("successor_checkpoint_revision_mismatch")
    if vllm.get("runtime_image") != PUBLIC_IMAGE:
        blockers.append("successor_runtime_image_mismatch")
    if checkpoint.get("remote_code_policy") != (
        "no_unpinned_remote_code_and_trust_remote_code_false"
    ):
        blockers.append("successor_remote_code_policy_invalid")

    try:
        inventory_validation = validate_smoke_inventory_manifest(smoke_inventory)
    except ValueError as exc:
        inventory_validation = {"status": "blocked", "reason": str(exc)}
        blockers.append("successor_smoke_inventory_invalid")
    if bundle_inspection.get("status") != "passed":
        blockers.extend(str(item) for item in bundle_inspection.get("blockers") or [])

    if provider_preflight.get("schema_version") != PREFLIGHT_SCHEMA:
        blockers.append("successor_vast_preflight_schema_invalid")
    if provider_preflight.get("status") != "verified":
        blockers.append("successor_vast_preflight_not_verified")
    if provider_preflight.get("provider") != "vast":
        blockers.append("successor_vast_preflight_provider_invalid")
    if provider_preflight.get("provider_inventory_verified_zero") is not True:
        blockers.append("successor_vast_inventory_not_zero")
    if provider_preflight.get("provider_mutations_performed") != 0:
        blockers.append("successor_vast_preflight_mutation_boundary_invalid")
    offer = _mapping(provider_preflight.get("selected_offer"))
    gpu_name = str(offer.get("gpu_name") or offer.get("gpu_type_id") or "")
    if "RTX PRO 6000" not in gpu_name.upper():
        blockers.append("successor_vast_preflight_gpu_not_rtx_pro_6000")
    try:
        gpu_ram_mb = int(offer.get("gpu_ram_mb") or 0)
        hourly_rate = float(
            offer.get("hourly_rate_usd")
            or offer.get("on_demand_price_usd_per_hour")
            or 0.0
        )
        reliability = float(offer.get("reliability") or 0.0)
    except (TypeError, ValueError):
        gpu_ram_mb, hourly_rate, reliability = 0, 0.0, 0.0
    if gpu_ram_mb < MIN_GPU_RAM_MB:
        blockers.append("successor_vast_preflight_gpu_ram_below_95gb")
    if not 0.0 < hourly_rate <= MAX_HOURLY_RATE_USD:
        blockers.append("successor_vast_preflight_hourly_rate_above_frozen_ceiling")
    if reliability < MIN_RELIABILITY:
        blockers.append("successor_vast_preflight_reliability_below_frozen_floor")
    observed = provider_preflight.get("observed_at_epoch")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    if type(observed) not in {int, float} or not math.isfinite(float(observed)):
        blockers.append("successor_vast_preflight_timestamp_invalid")
    elif execute and not 0.0 <= now - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("successor_vast_preflight_stale_or_future")

    authorization_blockers: list[str] = []
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        authorization_blockers.append("successor_compute_authorization_schema_invalid")
    if authorization.get("experiment_id") != EXPERIMENT_ID:
        authorization_blockers.append("successor_compute_authorization_experiment_mismatch")
    if authorization.get("authorization_id") != AUTHORIZATION_ID:
        authorization_blockers.append("successor_compute_authorization_id_invalid")
    if authorization.get("maximum_provider_allocations") != 1:
        authorization_blockers.append(
            "successor_compute_authorization_allocation_limit_invalid"
        )
    if authorization.get("single_use_consumption_required") is not True:
        authorization_blockers.append("successor_compute_authorization_single_use_invalid")
    if authorization.get("infrastructure_retry_of") != PREDECESSOR_AUTHORIZATION_ID:
        authorization_blockers.append(
            "successor_compute_authorization_retry_predecessor_invalid"
        )
    if authorization.get("predecessor_provider_allocations") != 0:
        authorization_blockers.append(
            "successor_compute_authorization_retry_allocation_count_invalid"
        )
    if authorization.get("predecessor_compute_spend_usd") != 0.0:
        authorization_blockers.append(
            "successor_compute_authorization_retry_prior_spend_invalid"
        )
    if authorization.get("additional_compute_authority_usd") != 2.75:
        authorization_blockers.append(
            "successor_compute_authorization_retry_additional_authority_invalid"
        )
    if authorization.get("paid_mutation_authorized") is not True:
        authorization_blockers.append("successor_compute_not_explicitly_authorized")
    try:
        authorized_cap = float(authorization.get("authorized_compute_cap_usd"))
    except (TypeError, ValueError):
        authorized_cap = math.nan
    if not math.isfinite(authorized_cap) or authorized_cap != MAX_COMPUTE_CAP_USD:
        authorization_blockers.append("successor_compute_cap_must_equal_6_usd")
    required_controls = {
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "automatic_spend_cutoff": True,
        "teardown_required": True,
        "provider_zero_verification_required": True,
        "physical_robot_endpoint_access_allowed": False,
        "hard_ttl_seconds": HARD_TTL_SECONDS,
    }
    for key, expected in required_controls.items():
        if authorization.get(key) != expected:
            authorization_blockers.append(f"successor_compute_authorization_{key}_invalid")
    blockers.extend(authorization_blockers)

    source_commit = str(expected_source_commit or "").strip().lower()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        blockers.append("successor_expected_source_commit_invalid")

    shared = build_paid_lane_admission(
        resource_class="vast_provider_adapter", blockers=blockers
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "admitted" if not blockers else "blocked",
        "probe_kind": PROBE_KIND,
        "experiment_id": EXPERIMENT_ID,
        "execute_requested": bool(execute),
        "blockers": sorted(set(blockers)),
        "source_commit": source_commit or None,
        "public_image": PUBLIC_IMAGE,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "smoke_inventory_validation": inventory_validation,
        "smoke_inventory_sha256": canonical_sha256(smoke_inventory),
        "provider_preflight_sha256": canonical_sha256(provider_preflight),
        "provider_bundle_sha256": bundle_inspection.get("bundle_sha256"),
        "selected_offer": dict(offer),
        "authorization": {
            "status": "accepted" if not authorization_blockers else "blocked",
            "authorized_compute_cap_usd": (
                authorized_cap if math.isfinite(authorized_cap) else None
            ),
            "hard_ttl_seconds": authorization.get("hard_ttl_seconds"),
        },
        "limits": {
            "hard_cap_usd": MAX_COMPUTE_CAP_USD,
            "target_spend_usd": TARGET_SPEND_USD,
            "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
            "hard_ttl_seconds": HARD_TTL_SECONDS,
            "one_resource": True,
            "disk_gb": DISK_GB,
            "min_gpu_ram_mb": MIN_GPU_RAM_MB,
            "min_reliability": MIN_RELIABILITY,
            "allowed_gpu_keywords": list(RTX_ALLOWED_KEYWORDS),
        },
        "shared_paid_lane_admission": shared,
        "provider_mutations_performed": 0,
        "claim_boundary": {
            "admission_is_not_gpu_execution": True,
            "admission_is_not_generated_media": True,
            "admission_is_not_wam_causal_validity": True,
            "physical_robot_access_forbidden": True,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _consume_authorization_once(
    authorization: Mapping[str, Any], *, expected_source_commit: str
) -> dict[str, Any]:
    authorization_id = str(authorization.get("authorization_id") or "")
    if authorization_id != AUTHORIZATION_ID or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{15,127}", authorization_id
    ):
        return {
            "status": "blocked",
            "blockers": ["successor_compute_authorization_id_invalid"],
        }
    root = AUTHORIZATION_CONSUMPTION_ROOT
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_mode = root.stat().st_mode
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_root_unavailable"],
        }
    if root_mode & 0o077:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_root_insecure"],
        }
    record_path = root / f"{authorization_id}.json"
    record = {
        "schema_version": "policy_ranking_successor_authorization_consumption.v1",
        "authorization_id": authorization_id,
        "authorization_sha256": canonical_sha256(authorization),
        "experiment_id": EXPERIMENT_ID,
        "source_commit": expected_source_commit,
        "consumed_at_epoch": time.time(),
        "maximum_provider_allocations": 1,
    }
    try:
        descriptor = os.open(
            record_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["successor_compute_authorization_already_consumed"],
            "authorization_id": authorization_id,
        }
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_write_failed"],
            "authorization_id": authorization_id,
        }
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return {
            "status": "blocked",
            "blockers": ["successor_authorization_consumption_write_failed"],
            "authorization_id": authorization_id,
        }
    return {
        "status": "consumed",
        "authorization_id": authorization_id,
        "consumption_record_sha256": _sha256_file(record_path),
        "record_location_disclosed": False,
    }


def run_successor_gpu_lane(
    *,
    authorization_path: str | Path,
    environment_path: str | Path,
    smoke_inventory_path: str | Path,
    provider_preflight_path: str | Path,
    provider_bundle_path: str | Path,
    provider_bundle_receipt_path: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    job_dir: str | Path,
    public_base_url: str | None,
    token_file: str | Path | None,
    secret_env_file: str | Path | None,
    output_path: str | Path | None,
    session_budget_ledger: str | Path | None,
    expected_source_commit: str,
    execute: bool,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    input_blockers: list[str] = []

    def load_input(label: str, path: str | Path) -> dict[str, Any]:
        try:
            return _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            input_blockers.append(f"successor_{label}_unreadable")
            return {}

    authorization = load_input("authorization", authorization_path)
    environment = load_input("environment", environment_path)
    smoke_inventory = load_input("smoke_inventory", smoke_inventory_path)
    provider_preflight = load_input("provider_preflight", provider_preflight_path)
    bundle_receipt = load_input("bundle_receipt", provider_bundle_receipt_path)
    bundle = inspect_successor_bundle(
        provider_bundle_path,
        receipt=bundle_receipt,
        smoke_inventory=smoke_inventory,
    )
    admission = build_successor_gpu_admission(
        authorization=authorization,
        environment=environment,
        smoke_inventory=smoke_inventory,
        provider_preflight=provider_preflight,
        bundle_inspection=bundle,
        expected_source_commit=expected_source_commit,
        execute=execute,
        observed_now_epoch=observed_now_epoch,
        initial_blockers=input_blockers,
    )
    write_json(Path(admission_out), admission)
    bound = {
        "schema_version": "policy_ranking_successor_bound_gpu_request.v1",
        "status": "bound" if admission["status"] == "admitted" else "blocked",
        "experiment_id": EXPERIMENT_ID,
        "source_commit": expected_source_commit,
        "provider": "vast",
        "probe_kind": PROBE_KIND,
        "public_image": PUBLIC_IMAGE,
        "provider_bundle_sha256": bundle.get("bundle_sha256"),
        "smoke_inventory_sha256": canonical_sha256(smoke_inventory),
        "selected_offer_id": _mapping(admission.get("selected_offer")).get(
            "ask_contract_id"
        ),
        "limits": admission["limits"],
        "blockers": admission["blockers"],
        "provider_mutations_performed": 0,
    }
    bound["manifest_sha256"] = canonical_sha256(bound)
    write_json(Path(bound_request_out), bound)
    if not execute:
        dry_ready = admission["status"] == "admitted"
        result = {
            "status": "dry_run_ready" if dry_ready else "blocked",
            "reason": (
                "admission_validated_without_provider_mutation"
                if dry_ready
                else "paid_execution_not_requested_and_admission_blocked"
            ),
            "blockers": admission["blockers"],
            "provider_mutations_performed": 0,
            "admission_status": admission["status"],
        }
        write_json(Path(adapter_output), result)
        return result
    try:
        grant = require_paid_resource_admission(
            admission["shared_paid_lane_admission"],
            resource_class="vast_provider_adapter",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked as exc:
        result = {
            "status": "blocked",
            "reason": "shared_paid_resource_admission_blocked",
            "blockers": [*admission["blockers"], *exc.blockers],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    launch_gate_blockers = [
        f"missing_env_{name}"
        for name in (VAST_API_GATE_ENV, VAST_INSTANCE_LAUNCH_GATE_ENV)
        if not _vast_env_truthy(name)
    ]
    if launch_gate_blockers:
        result = {
            "status": "blocked",
            "reason": "provider_launch_env_gate_blocked_before_authorization_consumption",
            "blockers": launch_gate_blockers,
            "authorization_consumed": False,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    consumption = _consume_authorization_once(
        authorization,
        expected_source_commit=expected_source_commit,
    )
    if consumption["status"] != "consumed":
        result = {
            "status": "blocked",
            "reason": "single_use_compute_authorization_blocked",
            "blockers": consumption.get("blockers") or [],
            "authorization_consumption": consumption,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    admission["authorization_consumption"] = consumption
    admission["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in admission.items() if key != "manifest_sha256"}
    )
    write_json(Path(admission_out), admission)
    bound["authorization_consumption"] = consumption
    bound["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in bound.items() if key != "manifest_sha256"}
    )
    write_json(Path(bound_request_out), bound)
    result = run_vast_wam_authorized_runner(
        job_dir=job_dir,
        bundle_path=provider_bundle_path,
        public_base_url=public_base_url,
        token_file=token_file,
        secret_env_file=secret_env_file,
        output_path=output_path,
        session_budget_ledger=session_budget_ledger,
        allow_paid_vast_launch=True,
        max_hourly_rate=MAX_HOURLY_RATE_USD,
        target_spend_usd=TARGET_SPEND_USD,
        hard_cap_usd=MAX_COMPUTE_CAP_USD,
        max_live_minutes=HARD_TTL_SECONDS // 60,
        session_max_live_minutes=HARD_TTL_SECONDS // 60,
        startup_timeout_seconds=3600,
        public_image=PUBLIC_IMAGE,
        disk_gb=DISK_GB,
        min_gpu_ram_mb=MIN_GPU_RAM_MB,
        min_compute_cap=1200,
        max_compute_cap=0,
        min_reliability=MIN_RELIABILITY,
        preferred_gpu_keywords=RTX_ALLOWED_KEYWORDS,
        prefer_isaac_rt=False,
        gpu_selection_policy=RTX_SELECTION_POLICY,
        paid_resource_admission_grant=grant,
    )
    result["authorization_consumption"] = consumption
    write_json(Path(adapter_output), result)
    return result


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "HARD_TTL_SECONDS",
    "MAX_COMPUTE_CAP_USD",
    "PREFLIGHT_SCHEMA",
    "PROBE_KIND",
    "PUBLIC_IMAGE",
    "build_successor_gpu_admission",
    "inspect_successor_bundle",
    "run_successor_gpu_lane",
]
