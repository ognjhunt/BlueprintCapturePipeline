"""Fail-closed paid Vast admission for a Cosmos3 Reasoner schema canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .gpu_render_providers import get_render_provider
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .policy_ranking_evaluator_diagnostic import (
    COSMOS_MODEL,
    COSMOS_REVISION,
    PAIR_OUTPUT_SCHEMA,
)
from .policy_ranking_evaluator_diagnostic_cosmos_bundle import (
    MODEL_CONFIG_SHA256,
    NATIVE_REASONER_ARCHITECTURE,
    PROVIDER_RUNTIME_RUNNER_SHA256,
    PUBLIC_IMAGE,
    RECEIPT_SCHEMA_VERSION,
    RUN_SCRIPT_SHA256,
)
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .vast_provider_adapter import (
    VAST_API_GATE_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    _env_truthy as _vast_env_truthy,
)
from .vast_wam_authorized_runner import run_vast_wam_authorized_runner


PROBE_KIND = "policy-ranking-cosmos-reasoner"
EXPERIMENT_ID = "policy_ranking_roboarena_disjoint_reasoner_successor_20260728"
AUTHORIZATION_SCHEMA = "policy_ranking_cosmos_reasoner_compute_authorization.v2"
PREFLIGHT_SCHEMA = "policy_ranking_cosmos_reasoner_vast_preflight.v2"
ADMISSION_SCHEMA = "policy_ranking_cosmos_reasoner_gpu_admission.v2"
AUTHORIZATION_ID_PREFIX = "policy-ranking-cosmos-reasoner-successor-20260728-allocation-"
AUTHORIZATION_ID = f"{AUTHORIZATION_ID_PREFIX}1"
AUTHORIZATION_CONSUMPTION_ROOT = Path.home() / ".blueprint-spend-authority" / "consumed"
EXTERNAL_AUTHORIZATION_ROOT = Path.home() / ".blueprint-spend-authority" / "authorizations"
MAX_HOURLY_RATE_USD = 2.50
TARGET_SPEND_USD = 1.50
HARD_CAP_USD = 2.00
ARM_CAP_USD = 2.00
HARD_TTL_SECONDS = 2_400
MAX_AUTHORIZED_ALLOCATIONS_FOR_ARM = int(ARM_CAP_USD // HARD_CAP_USD)
TARGET_MAX_LIVE_MINUTES = math.floor(TARGET_SPEND_USD / MAX_HOURLY_RATE_USD * 60)
MAX_PREFLIGHT_AGE_SECONDS = 900
DISK_GB = 200
MIN_GPU_RAM_MB = 80_000
MIN_RELIABILITY = 0.98
MIN_CUDA_MAX_GOOD = 12.8
GPU_KEYWORDS = ("H100",)
EXPECTED_PAIR_COUNT = 1
RUN_PURPOSE = "structured_output_canary_only"
EXPECTED_ENTRIES = {
    "provider_runtime/evaluator_provider_runtime_runner.py",
    "provider_runtime/run_evaluator_provider_runtime.sh",
    "provider_runtime/evaluator_provider_runtime_manifest.json",
    "provider_runtime/evaluator_input_manifest.json",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path}")
    return dict(value)


def _digest_matches(value: Mapping[str, Any], field: str) -> bool:
    recorded = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    return isinstance(recorded, str) and recorded == canonical_sha256(payload)


def _valid_authorization_id(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(AUTHORIZATION_ID_PREFIX):
        return False
    suffix = value.removeprefix(AUTHORIZATION_ID_PREFIX)
    return (
        suffix.isascii()
        and suffix.isdigit()
        and str(int(suffix)) == suffix
        and 0 < int(suffix) <= MAX_AUTHORIZED_ALLOCATIONS_FOR_ARM
    )


def load_external_authorization(path: str | Path) -> dict[str, Any]:
    """Load, but never mint, a user-provisioned task authorization artifact."""

    resolved = Path(path).expanduser().resolve()
    authority_root = EXTERNAL_AUTHORIZATION_ROOT.expanduser().resolve()
    if not resolved.is_relative_to(authority_root):
        raise ValueError("cosmos_reasoner_authorization_not_in_external_authority_root")
    stat_result = resolved.stat()
    if stat_result.st_mode & 0o777 != 0o600:
        raise ValueError("cosmos_reasoner_authorization_file_mode_invalid")
    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        raise ValueError("cosmos_reasoner_authorization_file_owner_invalid")
    record = _read_json(resolved)
    recorded_digest = record.get("authorization_sha256")
    digest_payload = {key: value for key, value in record.items() if key != "authorization_sha256"}
    if recorded_digest != canonical_sha256(digest_payload):
        raise ValueError("cosmos_reasoner_authorization_digest_invalid")
    record["external_authorization_file_verified_runtime"] = True
    record["external_authorization_file_path_runtime"] = str(resolved)
    return record


def collect_vast_preflight(*, name_prefix: str) -> dict[str, Any]:
    provider = get_render_provider("vast")
    request = {
        "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "min_gpu_ram_mb": MIN_GPU_RAM_MB,
        "min_reliability": MIN_RELIABILITY,
        "min_cuda_max_good": MIN_CUDA_MAX_GOOD,
        "require_avx": True,
        "require_known_supported_isaac_driver": False,
        "require_direct_port": False,
        "preferred_gpu_keywords": list(GPU_KEYWORDS),
    }
    capacity = provider.capacity_preflight(request)
    attempt_inventory = provider.billable_inventory(name_prefix=name_prefix)
    inventory = provider.billable_inventory(name_prefix="")
    viable = [
        dict(row)
        for row in capacity.get("viable_gpu_types", [])
        if isinstance(row, Mapping)
        and "H100" in str(row.get("gpu_name") or "").upper()
        and int(row.get("num_gpus") or 0) == 1
        and int(row.get("gpu_ram_mb") or 0) >= MIN_GPU_RAM_MB
        and 0 < float(row.get("hourly_rate_usd") or 0) <= MAX_HOURLY_RATE_USD
        and float(row.get("reliability") or 0) >= MIN_RELIABILITY
        and float(row.get("cuda_max_good") or 0) >= MIN_CUDA_MAX_GOOD
    ]
    viable.sort(key=lambda row: (float(row["hourly_rate_usd"]), -float(row["reliability"])))
    selected = viable[0] if viable else {}
    inventory_zero = bool(
        inventory.get("api_confirmed") is True and inventory.get("live_resource_count") == 0
    )
    api_verified = bool(
        capacity.get("status") == "available"
        and inventory.get("api_confirmed") is True
        and attempt_inventory.get("api_confirmed") is True
    )
    blockers: list[str] = []
    if not api_verified:
        blockers.append("cosmos_reasoner_vast_api_not_verified")
    if not inventory_zero:
        blockers.append("cosmos_reasoner_vast_inventory_not_zero")
    if not selected:
        blockers.append("cosmos_reasoner_compatible_single_h100_offer_unavailable")
    result: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "verified" if not blockers else "blocked",
        "experiment_id": EXPERIMENT_ID,
        "provider": "vast",
        "observed_at_epoch": time.time(),
        "blockers": blockers,
        "provider_api_verified": api_verified,
        "provider_inventory_verified_zero": inventory_zero,
        "selected_offer": selected or None,
        "capacity_request": request,
        "minimum_cuda_max_good": MIN_CUDA_MAX_GOOD,
        "capacity_snapshot": capacity,
        "attempt_billable_inventory": attempt_inventory,
        "billable_inventory": inventory,
        "provider_mutations_performed": 0,
        "reservation_proven": False,
        "task_security_exception": {
            "rotation_metadata_present": False,
            "existing_key_use_authorized_by_user": True,
            "live_key_validated": api_verified,
            "key_exposure_evidence_found": False,
            "rotation_event_claimed": False,
        },
        "raw_secret_values_recorded": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def inspect_bundle(
    bundle_path: str | Path,
    receipt: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    path = Path(bundle_path).expanduser().resolve()
    blockers: list[str] = []
    entries: list[str] = []
    runtime_manifest: dict[str, Any] = {}
    input_manifest: dict[str, Any] = {}
    entrypoint_text = ""
    runner_text = ""
    entrypoint_sha256 = ""
    runner_sha256 = ""
    if not path.is_file():
        blockers.append("cosmos_reasoner_bundle_missing")
    else:
        try:
            with zipfile.ZipFile(path) as archive:
                entries = sorted(archive.namelist())
                if archive.testzip() is not None:
                    blockers.append("cosmos_reasoner_bundle_zip_integrity_failed")
                missing = sorted(EXPECTED_ENTRIES - set(entries))
                if missing:
                    blockers.append("cosmos_reasoner_bundle_entries_missing")
                if not missing:
                    entrypoint_bytes = archive.read(
                        "provider_runtime/run_evaluator_provider_runtime.sh"
                    )
                    runner_bytes = archive.read(
                        "provider_runtime/evaluator_provider_runtime_runner.py"
                    )
                    entrypoint_sha256 = hashlib.sha256(entrypoint_bytes).hexdigest()
                    runner_sha256 = hashlib.sha256(runner_bytes).hexdigest()
                    entrypoint_text = entrypoint_bytes.decode("utf-8")
                    runner_text = runner_bytes.decode("utf-8")
                    runtime_manifest = json.loads(
                        archive.read(
                            "provider_runtime/evaluator_provider_runtime_manifest.json"
                        ).decode("utf-8")
                    )
                    input_manifest = json.loads(
                        archive.read("provider_runtime/evaluator_input_manifest.json").decode(
                            "utf-8"
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"cosmos_reasoner_bundle_invalid:{type(exc).__name__}")
    if entrypoint_text or runner_text:
        blockers.extend(
            provider_runtime_contract_blockers(
                provider_bundle_kind="evaluator",
                entrypoint_text=entrypoint_text,
                runner_text=runner_text,
            )
        )
    bundle_sha = file_sha256(path) if path.is_file() else None
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        blockers.append("cosmos_reasoner_bundle_receipt_schema_invalid")
    if not _digest_matches(receipt, "receipt_sha256"):
        blockers.append("cosmos_reasoner_bundle_receipt_digest_invalid")
    if receipt.get("bundle_sha256") != bundle_sha:
        blockers.append("cosmos_reasoner_bundle_receipt_hash_mismatch")
    if receipt.get("pair_count") != EXPECTED_PAIR_COUNT:
        blockers.append("cosmos_reasoner_canary_pair_count_not_one")
    if receipt.get("run_purpose") != RUN_PURPOSE:
        blockers.append("cosmos_reasoner_canary_purpose_invalid")
    if (
        runtime_manifest.get("model") != COSMOS_MODEL
        or runtime_manifest.get("model_revision") != COSMOS_REVISION
    ):
        blockers.append("cosmos_reasoner_model_identity_mismatch")
    if not _digest_matches(runtime_manifest, "manifest_sha256"):
        blockers.append("cosmos_reasoner_runtime_manifest_digest_invalid")
    if not _digest_matches(input_manifest, "manifest_sha256"):
        blockers.append("cosmos_reasoner_input_manifest_digest_invalid")
    if not (
        runtime_manifest.get("public_image") == PUBLIC_IMAGE
        and receipt.get("public_image") == PUBLIC_IMAGE
    ):
        blockers.append("cosmos_reasoner_runtime_image_identity_mismatch")
    if not (
        runtime_manifest.get("reasoner_architecture_mode") == "native_frozen_model_config"
        and runtime_manifest.get("native_reasoner_architecture") == NATIVE_REASONER_ARCHITECTURE
        and runtime_manifest.get("model_config_sha256") == MODEL_CONFIG_SHA256
        and runtime_manifest.get("reasoner_architecture_override") is None
    ):
        blockers.append("cosmos_reasoner_native_architecture_contract_invalid")
    if "--hf-overrides" in runner_text:
        blockers.append("cosmos_reasoner_unsupported_architecture_override_present")
    if not (
        runner_sha256 == PROVIDER_RUNTIME_RUNNER_SHA256
        and entrypoint_sha256 == RUN_SCRIPT_SHA256
        and runtime_manifest.get("provider_runtime_runner_sha256") == PROVIDER_RUNTIME_RUNNER_SHA256
        and runtime_manifest.get("provider_runtime_entrypoint_sha256") == RUN_SCRIPT_SHA256
        and receipt.get("provider_runtime_runner_sha256") == PROVIDER_RUNTIME_RUNNER_SHA256
        and receipt.get("provider_runtime_entrypoint_sha256") == RUN_SCRIPT_SHA256
    ):
        blockers.append("cosmos_reasoner_provider_runtime_digest_invalid")
    if not (
        receipt.get("native_reasoner_architecture") == NATIVE_REASONER_ARCHITECTURE
        and receipt.get("model_config_sha256") == MODEL_CONFIG_SHA256
    ):
        blockers.append("cosmos_reasoner_receipt_native_architecture_invalid")
    if (
        input_manifest.get("pair_count") != EXPECTED_PAIR_COUNT
        or input_manifest.get("claim_class") != "post_unseal_diagnostic_only"
        or input_manifest.get("run_purpose") != RUN_PURPOSE
    ):
        blockers.append("cosmos_reasoner_input_manifest_invalid")
    expected_schema_sha256 = canonical_sha256(PAIR_OUTPUT_SCHEMA)
    if not (
        input_manifest.get("output_schema") == PAIR_OUTPUT_SCHEMA
        and input_manifest.get("output_schema_sha256") == expected_schema_sha256
    ):
        blockers.append("cosmos_reasoner_output_schema_contract_invalid")
    runtime_source = str(runtime_manifest.get("source_commit") or "").strip().lower()
    receipt_source = str(receipt.get("source_commit") or "").strip().lower()
    if runtime_source != receipt_source:
        blockers.append("cosmos_reasoner_bundle_source_commit_internally_inconsistent")
    expected_source = str(expected_source_commit or "").strip().lower()
    if expected_source and (runtime_source != expected_source or receipt_source != expected_source):
        blockers.append("cosmos_reasoner_bundle_source_commit_mismatch")
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "bundle_sha256": bundle_sha,
        "bundle_size_bytes": path.stat().st_size if path.is_file() else 0,
        "zip_member_count": len(entries),
        "runtime_manifest": runtime_manifest,
        "source_commit": runtime_source or None,
        "receipt_source_commit": receipt_source or None,
        "input_manifest_sha256": input_manifest.get("manifest_sha256"),
        "pair_count": input_manifest.get("pair_count"),
        "run_purpose": input_manifest.get("run_purpose"),
        "provider_runtime_runner_sha256": runner_sha256 or None,
        "provider_runtime_entrypoint_sha256": entrypoint_sha256 or None,
    }


def build_admission(
    *,
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    bundle: Mapping[str, Any],
    expected_source_commit: str,
    execute: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA or not _valid_authorization_id(
        authorization.get("authorization_id")
    ):
        blockers.append("cosmos_reasoner_authorization_identity_invalid")
    if authorization.get("experiment_id") != EXPERIMENT_ID:
        blockers.append("cosmos_reasoner_authorization_experiment_mismatch")
    if (
        authorization.get("paid_mutation_authorized") is not True
        or authorization.get("maximum_provider_allocations") != 1
        or authorization.get("authorized_pair_count") != EXPECTED_PAIR_COUNT
        or authorization.get("run_purpose") != RUN_PURPOSE
    ):
        blockers.append("cosmos_reasoner_paid_authorization_invalid")
    if not (
        authorization.get("authorization_origin") == "external_workspace_user_task_authority"
        and authorization.get("external_authorization_file_verified_runtime") is True
    ):
        blockers.append("cosmos_reasoner_external_authorization_not_verified")
    if (
        authorization.get("authorized_compute_cap_usd") != HARD_CAP_USD
        or authorization.get("reasoner_arm_total_cap_usd") != ARM_CAP_USD
        or authorization.get("hard_ttl_seconds") != HARD_TTL_SECONDS
    ):
        blockers.append("cosmos_reasoner_authorization_limits_invalid")
    security = _mapping(authorization.get("task_security_exception"))
    if not (
        security.get("existing_vast_key_use_explicitly_authorized") is True
        and security.get("rotation_metadata_missing_acknowledged") is True
        and security.get("provider_side_rotation_event_claimed") is False
        and security.get("key_exposure_evidence_found") is False
        and security.get("live_authenticated_validation_required") is True
    ):
        blockers.append("cosmos_reasoner_vast_security_exception_invalid")
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA or preflight.get("status") != "verified":
        blockers.append("cosmos_reasoner_vast_preflight_invalid")
    if (
        preflight.get("provider_inventory_verified_zero") is not True
        or preflight.get("provider_mutations_performed") != 0
    ):
        blockers.append("cosmos_reasoner_vast_preflight_boundary_invalid")
    offer = _mapping(preflight.get("selected_offer"))
    if "H100" not in str(offer.get("gpu_name") or "").upper():
        blockers.append("cosmos_reasoner_selected_offer_not_h100")
    try:
        price = float(offer.get("hourly_rate_usd") or 0)
        ram = int(offer.get("gpu_ram_mb") or 0)
        reliability = float(offer.get("reliability") or 0)
        cuda_max_good = float(offer.get("cuda_max_good") or 0)
    except (TypeError, ValueError):
        price, ram, reliability, cuda_max_good = 0.0, 0, 0.0, 0.0
    if (
        not 0 < price <= MAX_HOURLY_RATE_USD
        or ram < MIN_GPU_RAM_MB
        or reliability < MIN_RELIABILITY
    ):
        blockers.append("cosmos_reasoner_selected_offer_limits_invalid")
    if cuda_max_good < MIN_CUDA_MAX_GOOD:
        blockers.append("cosmos_reasoner_selected_offer_cuda_incompatible")
    observed = preflight.get("observed_at_epoch")
    if type(observed) not in {int, float} or not math.isfinite(float(observed)):
        blockers.append("cosmos_reasoner_preflight_timestamp_invalid")
    elif execute and not 0 <= time.time() - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("cosmos_reasoner_preflight_stale_or_future")
    if bundle.get("status") != "passed":
        blockers.extend(str(item) for item in bundle.get("blockers") or [])
    source = str(expected_source_commit).strip().lower()
    if len(source) != 40 or any(char not in "0123456789abcdef" for char in source):
        blockers.append("cosmos_reasoner_expected_source_commit_invalid")
    if authorization.get("source_commit") != source:
        blockers.append("cosmos_reasoner_authorization_source_commit_mismatch")
    if bundle.get("source_commit") != source or bundle.get("receipt_source_commit") != source:
        blockers.append("cosmos_reasoner_bundle_source_commit_mismatch")
    shared = build_paid_lane_admission(resource_class="vast_provider_adapter", blockers=blockers)
    result: dict[str, Any] = {
        "schema_version": ADMISSION_SCHEMA,
        "status": "admitted" if not blockers else "blocked",
        "probe_kind": PROBE_KIND,
        "experiment_id": EXPERIMENT_ID,
        "execute_requested": execute,
        "source_commit": source,
        "authorization_id": authorization.get("authorization_id"),
        "blockers": sorted(set(blockers)),
        "public_image": PUBLIC_IMAGE,
        "model": COSMOS_MODEL,
        "model_revision": COSMOS_REVISION,
        "bundle_sha256": bundle.get("bundle_sha256"),
        "pair_count": bundle.get("pair_count"),
        "run_purpose": bundle.get("run_purpose"),
        "selected_offer": offer,
        "limits": {
            "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
            "target_spend_usd": TARGET_SPEND_USD,
            "hard_cap_usd": HARD_CAP_USD,
            "reasoner_arm_cap_usd": ARM_CAP_USD,
            "maximum_authorized_allocations_for_arm": MAX_AUTHORIZED_ALLOCATIONS_FOR_ARM,
            "hard_ttl_seconds": HARD_TTL_SECONDS,
            "target_max_live_minutes": TARGET_MAX_LIVE_MINUTES,
            "maximum_concurrent_gpus": 1,
        },
        "task_security_exception": security,
        "shared_paid_lane_admission": shared,
        "provider_mutations_performed": 0,
        "claim_boundary": "single_schema_canary_only_not_ranking_or_confirmation",
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _consume_once(authorization: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    authorization_id = authorization.get("authorization_id")
    if not _valid_authorization_id(authorization_id):
        return {
            "status": "blocked",
            "blockers": ["cosmos_reasoner_authorization_identity_invalid"],
        }
    assert isinstance(authorization_id, str)
    root = AUTHORIZATION_CONSUMPTION_ROOT
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        return {"status": "blocked", "blockers": ["authorization_consumption_root_insecure"]}
    path = root / f"{authorization_id}.json"
    record = {
        "schema_version": "policy_ranking_cosmos_reasoner_authorization_consumption.v1",
        "authorization_id": authorization_id,
        "authorization_sha256": canonical_sha256(authorization),
        "source_commit": source_commit,
        "consumed_at_epoch": time.time(),
        "maximum_provider_allocations": 1,
    }
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = root / f".{authorization_id}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        return {"status": "blocked", "blockers": ["cosmos_reasoner_authorization_already_consumed"]}
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "consumed",
        "authorization_id": authorization_id,
        "consumption_record_sha256": hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def run_gpu_lane(
    *,
    authorization_path: str | Path,
    preflight_path: str | Path,
    bundle_path: str | Path,
    bundle_receipt_path: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    job_dir: str | Path,
    expected_source_commit: str,
    execute: bool,
    public_base_url: str | None = None,
    token_file: str | Path | None = None,
    secret_env_file: str | Path | None = None,
    provider_bundle_url_file: str | Path | None = None,
    provider_output_put_url_file: str | Path | None = None,
    provider_output_get_url_file: str | Path | None = None,
    output_path: str | Path | None = None,
    session_budget_ledger: str | Path | None = None,
) -> dict[str, Any]:
    try:
        authorization = load_external_authorization(authorization_path)
    except (OSError, ValueError) as exc:
        result = {
            "status": "blocked",
            "blockers": [str(exc)],
            "provider_mutations_performed": 0,
        }
        write_json(Path(admission_out), result)
        write_json(Path(adapter_output), result)
        return result
    preflight = _read_json(preflight_path)
    receipt = _read_json(bundle_receipt_path)
    bundle = inspect_bundle(
        bundle_path,
        receipt,
        expected_source_commit=expected_source_commit,
    )
    admission = build_admission(
        authorization=authorization,
        preflight=preflight,
        bundle=bundle,
        expected_source_commit=expected_source_commit,
        execute=execute,
    )
    write_json(Path(admission_out), admission)
    bound: dict[str, Any] = {
        "schema_version": "policy_ranking_cosmos_reasoner_bound_gpu_request.v1",
        "status": "bound" if admission["status"] == "admitted" else "blocked",
        "probe_kind": PROBE_KIND,
        "experiment_id": EXPERIMENT_ID,
        "source_commit": expected_source_commit,
        "provider": "vast",
        "provider_bundle_kind": "evaluator",
        "public_image": PUBLIC_IMAGE,
        "bundle_sha256": bundle.get("bundle_sha256"),
        "pair_count": bundle.get("pair_count"),
        "run_purpose": bundle.get("run_purpose"),
        "selected_offer": admission.get("selected_offer"),
        "limits": admission.get("limits"),
        "blockers": admission.get("blockers"),
        "provider_mutations_performed": 0,
    }
    bound["manifest_sha256"] = canonical_sha256(bound)
    write_json(Path(bound_request_out), bound)
    if not execute:
        result = {
            "status": "dry_run_ready" if admission["status"] == "admitted" else "blocked",
            "blockers": admission["blockers"],
            "provider_mutations_performed": 0,
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
            "blockers": [*admission["blockers"], *exc.blockers],
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    gate_blockers = [
        f"missing_env_{name}"
        for name in (VAST_API_GATE_ENV, VAST_INSTANCE_LAUNCH_GATE_ENV)
        if not _vast_env_truthy(name)
    ]
    if gate_blockers:
        result = {
            "status": "blocked",
            "blockers": gate_blockers,
            "authorization_consumed": False,
            "provider_mutations_performed": 0,
        }
        write_json(Path(adapter_output), result)
        return result
    consumption: dict[str, Any] = {"status": "not_consumed"}

    def consume_before_mutation() -> Mapping[str, Any]:
        nonlocal consumption
        observed = preflight.get("observed_at_epoch")
        if (
            type(observed) not in {int, float}
            or not 0 <= time.time() - float(observed) <= MAX_PREFLIGHT_AGE_SECONDS
        ):
            consumption = {
                "status": "blocked",
                "blockers": ["cosmos_reasoner_preflight_stale_at_mutation"],
            }
        else:
            consumption = _consume_once(authorization, expected_source_commit)
        return consumption

    result = run_vast_wam_authorized_runner(
        job_dir=job_dir,
        bundle_path=bundle_path,
        public_base_url=public_base_url,
        token_file=token_file,
        secret_env_file=secret_env_file,
        provider_bundle_url_file=provider_bundle_url_file,
        provider_output_put_url_file=provider_output_put_url_file,
        provider_output_get_url_file=provider_output_get_url_file,
        output_path=output_path,
        session_budget_ledger=session_budget_ledger,
        allow_paid_vast_launch=True,
        max_hourly_rate=MAX_HOURLY_RATE_USD,
        target_spend_usd=TARGET_SPEND_USD,
        hard_cap_usd=HARD_CAP_USD,
        max_live_minutes=TARGET_MAX_LIVE_MINUTES,
        session_max_live_minutes=TARGET_MAX_LIVE_MINUTES,
        startup_timeout_seconds=3600,
        public_image=PUBLIC_IMAGE,
        disk_gb=DISK_GB,
        min_gpu_ram_mb=MIN_GPU_RAM_MB,
        min_compute_cap=900,
        max_compute_cap=0,
        min_reliability=MIN_RELIABILITY,
        preferred_gpu_keywords=GPU_KEYWORDS,
        gpu_selection_policy={
            "policy_id": "cosmos_reasoner_h100_cuda128_canary_v2",
            "allowed_gpu_keywords": list(GPU_KEYWORDS),
            "denied_gpu_keywords": [],
            "minimum_cuda_max_good": MIN_CUDA_MAX_GOOD,
        },
        require_independent_watchdog=True,
        retain_instance_on_runtime_failure=True,
        retention_binding={
            "source_commit": expected_source_commit,
            "dirty_state_declaration": "clean_exact_commit",
            "bundle_sha256": str(bundle["bundle_sha256"]),
            "authorization_receipt_sha256": file_sha256(Path(authorization_path).expanduser()),
            "image_digest": PUBLIC_IMAGE.split("@", 1)[1],
            "checkpoint": COSMOS_MODEL,
            "checkpoint_revision": COSMOS_REVISION,
        },
        paid_resource_admission_grant=grant,
        pre_provider_mutation_hook=consume_before_mutation,
        provider_bundle_kind="evaluator",
    )
    result["authorization_consumption"] = consumption
    write_json(Path(adapter_output), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--name-prefix", default="blueprint-roboarena-cosmos-reasoner-")
    preflight.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = collect_vast_preflight(name_prefix=args.name_prefix)
    write_json(Path(args.output), result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in {"capacity_snapshot", "billable_inventory", "attempt_billable_inventory"}
            }
        )
    )
    return 0 if result.get("status", "verified") in {"verified", "admitted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
