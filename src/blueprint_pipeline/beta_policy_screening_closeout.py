"""Close a beta attempt without requiring unrelated Vast jobs to stop.

The canonical provider's complete authenticated inventory and exact-instance
readback are observed afresh. Only this sealed authority's allocation and exact
resource name must be absent. Own teardown and staged-object absence remain
mandatory. Legacy ADP/team global-zero requirements are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, validate_provider_bundle, validate_session_authority, validate_session_result,
)
from .policy_canary_retained_billing import adapter_instance_ids
from .policy_canary_staged_object_absence import closeout_staged_objects

SCHEMA = "blueprint.beta_policy_screening_provider_absence.v1"
SCOPE = "exact_instance_and_authority_resource_name"
MAX_OBSERVATION_AGE_SECONDS = 120


def _read_bound(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("beta_screening_closeout_record_invalid")
    raw = path.read_bytes()
    if len(raw) != record.get("size_bytes") or "sha256:" + hashlib.sha256(raw).hexdigest() != record.get("sha256"):
        raise ValueError("beta_screening_closeout_record_changed")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("beta_screening_closeout_record_invalid")
    return value


def _bindings(*, adapter: Mapping[str, Any], prepared: Mapping[str, Any], adapter_path: Path) -> dict[str, Any]:
    authority = validate_session_authority(_read_bound(prepared["authority"]))
    bundle = validate_provider_bundle(_read_bound(prepared["bundle_receipt"]), authority=authority)
    if (authority["run_kind"] != BETA_RUN_KIND
            or prepared.get("prepared_digest") != canonical_digest(prepared, digest_field="prepared_digest")
            or adapter.get("bundle_sha256") != bundle["bundle_sha256"]
            or prepared.get("candidate_id") != authority["candidate_ids"][0]
            or prepared.get("run_id") != authority["run_id"]
            or any(prepared.get(key) != authority["beta_screening_admission"][key]
                   for key in ("plan_digest", "configuration_digest"))):
        raise ValueError("beta_screening_closeout_attempt_binding_mismatch")
    consumption = adapter.get("authorization_consumption") or {}
    if (consumption.get("schema_version") != "native_task_arena_policy_canary_session_consumption.v1"
            or consumption.get("status") != "consumed"
            or consumption.get("authority_digest") != authority["authority_digest"]
            or consumption.get("maximum_provider_allocations") != 1 or consumption.get("retry_cap") != 0
            or consumption.get("consumption_digest") != canonical_digest(consumption, digest_field="consumption_digest")):
        raise ValueError("beta_screening_closeout_consumed_authority_mismatch")
    identities = adapter_instance_ids(adapter, result_path=adapter_path,
        read_json=lambda path, code: json.loads(path.read_text()), error_factory=ValueError)
    if len(identities) != 1:
        raise ValueError("beta_screening_closeout_allocation_identity_invalid")
    closeout = adapter.get("provider_closeout") or {}
    for key in ("adapter_result", "teardown_manifest"):
        record = closeout.get(key) or {}
        path = Path(str(record.get("path") or ""))
        if not path.resolve().is_relative_to(adapter_path.resolve().parent):
            raise ValueError("beta_screening_closeout_foreign_attempt_record")
        evidence = _read_bound(record)
        if evidence.get("vast_instance_ids") != identities:
            raise ValueError("beta_screening_closeout_allocation_record_mismatch")
    return {"authority_digest": authority["authority_digest"], "bundle_sha256": bundle["bundle_sha256"],
            "prepared_digest": prepared["prepared_digest"], "adapter_digest": canonical_digest(adapter),
            "instance_id": identities[0], "resource_name": authority["resource_name"],
            "beta_screening_admission": authority["beta_screening_admission"],
            "candidate_ids": authority["candidate_ids"], "runtime_inputs_digest": authority["runtime_inputs_digest"]}


def _own_teardown(adapter: Mapping[str, Any]) -> bool:
    closeout = adapter.get("provider_closeout") or {}
    staged = closeout_staged_objects(adapter)
    return (closeout.get("provider_zero_confirmed") is True
            and closeout.get("warm_session_retained") is False
            and adapter.get("continuing_spend_from_this_run") is False
            and staged["absent"] is True)


def _observation_blockers(receipt: Mapping[str, Any], *, now: float) -> list[str]:
    inventory, exact = receipt.get("inventory") or {}, receipt.get("exact_instance") or {}
    if not isinstance(inventory, Mapping) or not isinstance(exact, Mapping):
        return ["beta_screening_provider_observation_invalid"]
    start, finish = receipt.get("collection_started_at_epoch"), receipt.get("observed_at_epoch")
    stamp = inventory.get("observed_at_epoch")
    valid_times = all(type(value) in (int, float) and math.isfinite(value) for value in (start, finish, stamp))
    blockers = []
    if (not valid_times or not start <= stamp <= finish <= now + 1
            or now - start > MAX_OBSERVATION_AGE_SECONDS):
        blockers.append("beta_screening_provider_observation_stale")
    resources = inventory.get("resources")
    if (inventory.get("status") != "observed" or inventory.get("provider") != "vast"
            or inventory.get("api_confirmed") is not True or inventory.get("name_prefix") != ""
            or inventory.get("blockers") not in (None, []) or not isinstance(resources, list)
            or type(inventory.get("live_resource_count")) is not int
            or inventory.get("live_resource_count") != len(resources)
            or any(not isinstance(row, Mapping) or not str(row.get("instance_id") or "").isdigit()
                   or int(row["instance_id"]) <= 0 or not isinstance(row.get("name"), str) for row in resources)):
        blockers.append("beta_screening_complete_provider_inventory_unproven")
        resources = []
    if (exact.get("provider") != "vast" or exact.get("status") != "absent"
            or str(exact.get("instance_id")) != str(receipt.get("instance_id"))
            or exact.get("api_confirmed") is not True or exact.get("provider_absence_confirmed") is not True):
        blockers.append("beta_screening_exact_instance_absence_unproven")
    if any(str(row["instance_id"]) == str(receipt.get("instance_id"))
           or row["name"] == receipt.get("resource_name") for row in resources):
        blockers.append("beta_screening_owned_resource_still_live")
    expected_unrelated = [row for row in resources if str(row["instance_id"]) != str(receipt.get("instance_id"))
                          and row["name"] != receipt.get("resource_name")]
    if receipt.get("unrelated_live_resources") != expected_unrelated:
        blockers.append("beta_screening_provider_inventory_projection_mismatch")
    return blockers


def collect_scoped_closeout(*, adapter: Mapping[str, Any], prepared: Mapping[str, Any],
                            adapter_path: Path, provider: Any = None) -> dict[str, Any]:
    """Make fresh read-only provider observations, retaining unrelated resources."""
    binding = _bindings(adapter=adapter, prepared=prepared, adapter_path=adapter_path)
    if provider is None:
        from .gpu_render_providers import VastRenderProvider
        provider = VastRenderProvider()
    started = time.time()
    exact = provider.inspect(str(binding["instance_id"]))
    inventory = provider.billable_inventory(name_prefix="")
    exact = dict(exact) if isinstance(exact, Mapping) else {}
    inventory = dict(inventory) if isinstance(inventory, Mapping) else {}
    resources = inventory.get("resources") if isinstance(inventory, Mapping) else None
    unrelated = [dict(row) for row in resources or [] if isinstance(row, Mapping)
                 and str(row.get("instance_id")) != str(binding["instance_id"])
                 and row.get("name") != binding["resource_name"]]
    receipt = {"schema_version": SCHEMA, "scope": SCOPE, "provider": "vast", **binding,
               "collection_started_at_epoch": started, "observed_at_epoch": time.time(),
               "exact_instance": exact, "inventory": inventory, "unrelated_live_resources": unrelated,
               "raw_provider_response_recorded": False, "global_provider_zero_claimed": False}
    blockers = _observation_blockers(receipt, now=time.time())
    if not _own_teardown(adapter):
        blockers.append("beta_screening_own_teardown_unproven")
    receipt.update(status="blocked" if blockers else "verified", blockers=sorted(set(blockers)),
                   provider_absence_confirmed=not blockers)
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def scoped_closeout_verified(receipt: Mapping[str, Any], *, adapter: Mapping[str, Any],
                             prepared: Mapping[str, Any], adapter_path: Path) -> bool:
    try:
        binding = _bindings(adapter=adapter, prepared=prepared, adapter_path=adapter_path)
        return (receipt.get("schema_version") == SCHEMA and receipt.get("scope") == SCOPE
                and receipt.get("provider") == "vast" and receipt.get("status") == "verified"
                and receipt.get("provider_absence_confirmed") is True and receipt.get("blockers") == []
                and receipt.get("global_provider_zero_claimed") is False
                and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
                and all(receipt.get(key) == value for key, value in binding.items())
                and not _observation_blockers(receipt, now=time.time()) and _own_teardown(adapter))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def join_scoped_session_closeout(*, inner: Mapping[str, Any], adapter: Mapping[str, Any],
                                 prepared: Mapping[str, Any], receipt: Mapping[str, Any],
                                 adapter_path: Path) -> dict[str, Any]:
    """Seal a native beta result with independently verified attempt-scoped absence."""
    if not scoped_closeout_verified(receipt, adapter=adapter, prepared=prepared, adapter_path=adapter_path):
        raise ValueError("beta_screening_scoped_closeout_unproven")
    if closeout_staged_objects(adapter)["blockers"]:
        # Billing can close after staged objects disappear even when scientific
        # outputs were lost. Never publish a score from that failed promotion.
        raise ValueError("beta_screening_scientific_output_not_durable")
    if (inner.get("run_kind") != BETA_RUN_KIND
            or inner.get("beta_screening_admission") != receipt["beta_screening_admission"]
            or inner.get("candidate_ids") != receipt["candidate_ids"]
            or inner.get("result_digest") != adapter.get("native_control_result_digest")
            or inner.get("result_digest") != canonical_digest(inner, digest_field="result_digest")):
        raise ValueError("beta_screening_closeout_native_result_binding_mismatch")
    value = json.loads(json.dumps(inner, allow_nan=False))
    episodes = value.get("episodes")
    returned = (isinstance(episodes, list) and len(episodes) == value.get("learned_policy_rollout_count")
                and all(isinstance(row, Mapping) for row in episodes))
    blockers = [] if returned else ["beta_screening_episodes_incomplete"]
    value["provider_allocations_observed"] = 1
    value["session_closeout"] = {"status": "closed", "provider_allocations_observed": 1,
                                 "teardown_completed": True, "provider_zero_confirmed": True,
                                 "provider_zero_scope": SCOPE, "provider_absence_receipt_digest": receipt["receipt_digest"],
                                 "global_provider_zero_claimed": False, "blockers": blockers}
    value["status"] = ("completed_unqualified" if not blockers
                       and all(row.get("status") == "completed" for row in episodes) else "blocked")
    value["blockers"] = sorted(set(list(value.get("blockers") or []) + blockers))
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return validate_session_result(value)
