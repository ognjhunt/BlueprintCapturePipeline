"""One-shot internal G1 beta session identity and bounded transfer contract.

This authority describes one selected candidate from an already admitted beta
plan. It is neither a model license nor a replacement for current capture rights.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import json
from pathlib import Path
import re
import time
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_assets import selected_asset_inventory, selected_candidates
from .native_g1_beta_authority import validate_beta_authority, validate_frozen_cell

SCHEMA = "native_g1_beta_session_authority.v1"
BUNDLE_SCHEMA = "native_g1_beta_provider_bundle.v1"
PROBE_KIND = "native-g1-beta-screening"
PROVIDER_KIND = "native_g1_beta_screening"
RUN_KIND = "blueprint_beta_screening"
RESULT_SCHEMA = "native_g1_beta_session_result.v1"
RESULT_FILENAME = RESULT_SCHEMA + ".json"
MANIFEST = "provider_runtime/native_g1_beta_provider_manifest.json"
ENTRYPOINT = "provider_runtime/run_adp_arena_provider_runtime.sh"
REQUIRED_ENTRIES = (
    MANIFEST, ENTRYPOINT,
    "provider_runtime/inputs/session_authority.json",
    "provider_runtime/inputs/policy_execution_spec.json",
    "provider_runtime/inputs/scene_packet/native_task_arena_packet_receipt.v1.json",
    "provider_runtime/publisher-source/native_g1_publisher_source_stage.v1.json",
    "provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json",
    *("provider_runtime/blueprint_pipeline/" + name + ".py" for name in (
        "native_g1_beta_runtime", "native_g1_beta_session", "native_g1_beta_bundle",
        "native_g1_beta_packet", "native_g1_beta_assets", "native_g1_beta_download", "native_g1_beta_authority",
        "native_g1_beta_authority_composer",
        "native_g1_beta_dependencies", "native_g1_beta_dependency_profile", "native_g1_beta_navigation",
        "native_g1_development_worker", "native_g1_provider_runtime")),
)
MAX_OUTPUT_BYTES = 4 * 1024**3
SETUP_DOWNLOAD_ALLOWANCE_BYTES = 64 * 1024**3
RESOURCE_PREFIX = "blueprint-native-task-policy-canary-"
CONSUMPTION_SCHEMA = "native_task_arena_policy_canary_session_consumption.v1"


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def build_session_authority(*, plan: Mapping[str, Any], candidate: Mapping[str, Any],
                            spec: Mapping[str, Any]) -> dict[str, Any]:
    from .beta_policy_screening_g1 import validate_execution_spec
    from .beta_policy_screening_launch import _check_quote, _identity
    identity = _identity(plan, candidate)
    _check_quote(plan, candidate)
    native = validate_execution_spec(spec, candidate_id=candidate["candidate_id"])
    execution = candidate["execution_spec"]
    if (execution.get("backend") != "native_g1.v1" or native["source"] != plan["source"]
            or native["site_id"] != plan["site_id"]
            or native["runtime_identity_digest"] != execution["runtime_digest"]
            or candidate["admission_digest"] != canonical_digest(execution)):
        raise ValueError("g1_beta_session_execution_mismatch")
    admission = {**identity, "admission_digest": candidate["admission_digest"],
        "scene_revision_digest": plan["scene_revision_digest"], "source": plan["source"],
        "site_id": plan["site_id"], "runtime_identity_digest": execution["runtime_digest"],
        "scenario_plan_digest": plan["scenario_plan_digest"], "maximum_cost_usd": candidate["max_cost_usd"],
        "cost_quote": candidate["cost_quote"]}
    run_id = canonical_digest({**identity, "admission_digest": candidate["admission_digest"]})[7:39]
    value = {"schema_version": SCHEMA, "run_kind": RUN_KIND, "run_id": run_id,
        "candidate_ids": [candidate["candidate_id"]], "episodes_per_policy": candidate["episode_count"],
        "cells": candidate["scenario_cells"], "beta_screening_admission": admission,
        "runtime_inputs_digest": native["execution_spec_digest"],
        "task_success_contract_digest": native["task_success_contract_digest"],
        "rights_authority": native["rights_authority"], "implementation_commit": execution["implementation_commit"],
        "resource_name": RESOURCE_PREFIX + run_id,
        "hard_cap_usd": float(execution.get("provider_hard_cap_usd", candidate["max_cost_usd"])),
        "max_hourly_rate_usd": str(execution["max_hourly_rate_usd"]),
        "hard_ttl_seconds": execution["hard_ttl_seconds"],
        "maximum_provider_allocations": 1, "retry_cap": 0, "automatic_retry_authorized": False,
        "caller_surviving_watchdog_required": True, "provider_wide_launch_lock_required": True,
        "canonical_allocator": "blueprint_pipeline.paid_resource_allocator",
        "claim_ceiling": "development_only", "public_media_publication_authorized": False,
        "model_redistribution_authorized": False, "official_ranking_authorized": False,
        "scene_promotion_authorized": False, "physical_outcome_claimed": False}
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return validate_session_authority(value)


def validate_session_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy(value)
    candidates = selected_candidates(result.get("candidate_ids"))
    cells = result.get("cells")
    admission = result.get("beta_screening_admission") or {}
    if (result.get("schema_version") != SCHEMA or result.get("run_kind") != RUN_KIND
            or len(candidates) != 1 or not isinstance(cells, list) or not cells
            or type(result.get("episodes_per_policy")) is not int
            or result["episodes_per_policy"] != len(cells)
            or len({row.get("cell_id") for row in cells}) != len(cells)
            or result.get("authority_digest") != canonical_digest(result, digest_field="authority_digest")
            or re.fullmatch(r"[0-9a-f]{40}", str(result.get("implementation_commit", ""))) is None
            or re.fullmatch(r"[0-9a-f]{32}", str(result.get("run_id", ""))) is None
            or result.get("resource_name") != RESOURCE_PREFIX + result["run_id"]
            or admission.get("candidate_id") != candidates[0]
            or any(not _digest(admission.get(key)) for key in ("plan_digest", "configuration_digest",
                "admission_digest", "scene_revision_digest", "runtime_identity_digest", "scenario_plan_digest"))
            or any(not _digest(result.get(key)) for key in ("runtime_inputs_digest", "task_success_contract_digest"))):
        raise ValueError("g1_beta_session_authority_identity_invalid")
    for cell in cells:
        validate_frozen_cell(cell)
    expected = {"maximum_provider_allocations": 1, "retry_cap": 0,
        "automatic_retry_authorized": False, "caller_surviving_watchdog_required": True,
        "provider_wide_launch_lock_required": True,
        "canonical_allocator": "blueprint_pipeline.paid_resource_allocator", "claim_ceiling": "development_only",
        "public_media_publication_authorized": False, "model_redistribution_authorized": False,
        "official_ranking_authorized": False, "scene_promotion_authorized": False, "physical_outcome_claimed": False}
    if any(type(result.get(key)) is not type(expected_value) or result[key] != expected_value
           for key, expected_value in expected.items()):
        raise ValueError("g1_beta_session_authority_scope_invalid")
    try:
        cap, rate, total = (Decimal(str(result["hard_cap_usd"])), Decimal(str(result["max_hourly_rate_usd"])),
                            Decimal(str(admission["maximum_cost_usd"])))
        ttl = result["hard_ttl_seconds"]
        valid = (cap.is_finite() and rate.is_finite() and total.is_finite() and 0 < cap <= total <= 15
                 and rate > 0 and type(ttl) is int and 1800 <= ttl <= 4 * 3600 and rate * ttl / 3600 <= cap)
    except (ArithmeticError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("g1_beta_session_authority_budget_invalid")
    rights = result.get("rights_authority") or {}
    validate_beta_authority(rights, candidate_id=candidates[0], scene_plan_digest=rights.get("scene_plan_digest"),
        current_source=admission.get("source"), task_success_contract_digest=result["task_success_contract_digest"])
    return result


def consume_session_authority_once(value, *, consumption_path: Path, now: float | None = None):
    from .beta_policy_screening_launch import _write
    authority = validate_session_authority(value)
    start = time.time() if now is None else now
    if not isinstance(start, (int, float)) or isinstance(start, bool) or not 0 < start < float("inf"):
        raise ValueError("g1_beta_session_start_invalid")
    receipt = {"schema_version": CONSUMPTION_SCHEMA, "status": "consumed",
        "authority_digest": authority["authority_digest"], "maximum_provider_allocations": 1,
        "retry_cap": 0, "started_at_epoch": start, "hard_deadline_epoch": start + authority["hard_ttl_seconds"]}
    receipt["consumption_digest"] = canonical_digest(receipt, digest_field="consumption_digest")
    try:
        _write(Path(consumption_path), receipt)
    except FileExistsError:
        return {"schema_version": CONSUMPTION_SCHEMA, "status": "already_consumed",
            "authority_digest": authority["authority_digest"], "maximum_provider_allocations": 0,
            "retry_cap": 0, "blockers": ["g1_beta_session_authority_already_consumed"]}
    return receipt


def transfer_quote(*, candidate_ids, runtime_packet_bytes: int, bundle_bytes: int,
                   runtime_expanded_bytes: int = 0, scene_packet_expanded_bytes: int = 0, cell_count: int = 0,
                   bundle_expanded_bound_bytes: int = 0, native_wheel_expanded_bytes: int = 0):
    from .native_g1_beta_dependency_profile import TOTAL_DOWNLOAD_BYTES, MAXIMUM_EXPANDED_BYTES, DEPENDENCY_MANIFEST_DIGEST
    assets = selected_asset_inventory(candidate_ids)
    if (any(type(x) is not int or x <= 0 for x in (runtime_packet_bytes, bundle_bytes))
            or any(type(x) is not int or x < 0 for x in (runtime_expanded_bytes, scene_packet_expanded_bytes, cell_count, bundle_expanded_bound_bytes, native_wheel_expanded_bytes))):
        raise ValueError("g1_beta_transfer_input_size_invalid")
    model_bytes = (assets["checkpoint_download_bytes"] + assets["sonic_download_bytes"]
                   + assets["auxiliary_download_bytes"])
    expanded_bytes = (runtime_expanded_bytes + native_wheel_expanded_bytes + runtime_packet_bytes
                      + 2 * max(bundle_bytes, bundle_expanded_bound_bytes) + model_bytes
                      + scene_packet_expanded_bytes * cell_count + MAXIMUM_EXPANDED_BYTES
                      + TOTAL_DOWNLOAD_BYTES + 2 * MAX_OUTPUT_BYTES + SETUP_DOWNLOAD_ALLOWANCE_BYTES)
    return {"schema_version": "native_g1_beta_transfer_quote.v1", "asset_inventory_digest": assets["inventory_digest"],
        "selected_model_download_bytes": model_bytes, "runtime_packet_bytes": runtime_packet_bytes,
        "bundle_bytes": bundle_bytes, "setup_download_allowance_bytes": SETUP_DOWNLOAD_ALLOWANCE_BYTES,
        "setup_allowance_basis": "pinned_container_cold_pull_allowance_no_online_pip_or_apt",
        "dependency_manifest_digest": DEPENDENCY_MANIFEST_DIGEST, "external_dependency_download_bytes": TOTAL_DOWNLOAD_BYTES,
        "maximum_policy_environment_expanded_bytes": MAXIMUM_EXPANDED_BYTES,
        "runtime_expanded_bytes": runtime_expanded_bytes, "scene_packet_expanded_bytes": scene_packet_expanded_bytes,
        "cell_count": cell_count, "bundle_expanded_bound_bytes": bundle_expanded_bound_bytes,
        "native_wheel_expanded_bytes": native_wheel_expanded_bytes, "expected_provider_download_bytes":
            bundle_bytes + runtime_packet_bytes + model_bytes + TOTAL_DOWNLOAD_BYTES + SETUP_DOWNLOAD_ALLOWANCE_BYTES,
        "expected_provider_upload_bytes": MAX_OUTPUT_BYTES,
        "disk_gb": max(200, (expanded_bytes + 10**9 - 1) // 10**9)}


def validate_provider_bundle(value, *, authority):
    bound = validate_session_authority(authority)
    manifest = {key: item for key, item in value.items()
                if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256", "transfer_quote"}}
    if (manifest.get("schema_version") != BUNDLE_SCHEMA or manifest.get("status") != "ready"
            or manifest.get("provider_bundle_kind") != PROVIDER_KIND
            or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")
            or manifest.get("authority_digest") != bound["authority_digest"]
            or manifest.get("candidate_ids") != bound["candidate_ids"]
            or manifest.get("cell_ids") != [cell["cell_id"] for cell in bound["cells"]]
            or manifest.get("implementation_commit") != bound["implementation_commit"]
            or manifest.get("runtime_inputs_digest") != bound["runtime_inputs_digest"]
            or manifest.get("task_success_contract_digest") != bound["task_success_contract_digest"]
            or not _digest(value.get("bundle_sha256")) or type(value.get("bundle_size_bytes")) is not int
            or value["bundle_size_bytes"] <= 0):
        raise ValueError("g1_beta_provider_bundle_binding_invalid")
    return dict(value)
