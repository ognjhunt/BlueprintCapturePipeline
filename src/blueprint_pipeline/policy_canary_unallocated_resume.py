"""Resume one funded dispatch only when its provider launch never began.

No quota, charge, authority or failed receipt is reset. The successor uses the
same sealed bundle and authority and the same funding binding. An exclusive
parent claim prevents two successors from spending that one allocation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_policy_canary_session import validate_provider_bundle, validate_session_authority
from .policy_canary_staged_object_absence import billing_staged_objects_absent


def prepare_unallocated_session_resume(*, parent_job_dir: str | Path, successor_job_dir: str | Path,
    authority: Mapping[str, Any], bundle: Mapping[str, Any], funding_admission: Mapping[str, Any],
    paid_admission: Mapping[str, Any], apply: bool = False) -> dict[str, Any]:
    """Preview or claim a successor for a proven pre-provider lock refusal only."""
    authority = validate_session_authority(authority)
    bundle = validate_provider_bundle(bundle, authority=authority)
    binding = paid_admission.get("allocation_binding") or {}
    binding_digest = canonical_digest(binding)
    if paid_admission.get("status") != "admitted" or paid_admission.get("blockers") != [] \
        or paid_admission.get("allocation_binding_digest") != binding_digest \
        or binding.get("authority_digest") != authority["authority_digest"] \
        or binding.get("bundle_sha256") != bundle["bundle_sha256"] \
        or funding_admission.get("status") not in {"admitted", "already_reserved"} \
        or funding_admission.get("allocation_binding_digest") != binding_digest \
        or funding_admission.get("maximum_cost_usd") != authority["hard_cap_usd"] \
        or funding_admission.get("provider") != "vast" or funding_admission.get("resource_class") != "gpu_render" \
        or funding_admission.get("request_count") != 1:
        raise ValueError("unallocated_resume_funding_binding_invalid")
    parent, successor = Path(parent_job_dir).resolve(), Path(successor_job_dir).resolve()
    if parent == successor or parent in successor.parents or successor in parent.parents or successor.exists():
        raise ValueError("unallocated_resume_successor_invalid")
    claim = parent / "unallocated_successor_claim.json"
    if claim.exists() or claim.is_symlink():
        raise ValueError("unallocated_resume_successor_already_claimed")
    records = {}

    def read(relative: str) -> dict[str, Any]:
        path = parent / relative
        if path.is_symlink() or not path.resolve().is_relative_to(parent):
            raise ValueError("unallocated_resume_evidence_path_invalid")
        raw = path.read_bytes()
        records[relative] = "sha256:" + hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("unallocated_resume_evidence_invalid")
        return value

    consumption = read("policy_canary_session_consumption.json")
    if consumption.get("consumption_digest") != canonical_digest(consumption, digest_field="consumption_digest") \
        or consumption.get("authority_digest") != authority["authority_digest"] \
        or consumption.get("status") != "consumed" or consumption.get("maximum_provider_allocations") != 1:
        raise ValueError("unallocated_resume_authority_invalid")
    attempts = parent / "attempts"
    if sorted(p.name for p in attempts.iterdir()) != ["attempt_001"]:
        raise ValueError("unallocated_resume_attempt_history_invalid")
    prefix = "attempts/attempt_001/"
    adapter = read(prefix + "vast_provider_run/vast_provider_adapter_result.json")
    lock = read(prefix + "vast_provider_run/vast_launch_lock_manifest.json")
    teardown = read(prefix + "vast_provider_run/vast_teardown_manifest.json")
    if adapter.get("status") != "blocked" or adapter.get("reason") != "vast_paid_launch_lock_blocked" \
        or adapter.get("blockers") != ["vast_paid_launch_lock_busy"] \
        or adapter.get("provider_bundle_sha256") != bundle["bundle_sha256"] \
        or any(adapter.get(k) is not False for k in ("api_call_performed", "vast_side_effects_may_have_occurred",
             "provider_create_attempted", "continuing_spend_from_this_run")) \
        or adapter.get("vast_instance_ids") != [] or type(adapter.get("estimated_cost_usd")) not in (int, float) \
        or adapter["estimated_cost_usd"] != 0:
        raise ValueError("unallocated_resume_provider_mutation_not_excluded")
    if lock.get("status") != "blocked" or lock.get("lock_acquired") is not False \
        or lock.get("blockers") != ["vast_paid_launch_lock_busy"] \
        or teardown.get("status") != "not_required_launch_lock_blocked" \
        or teardown.get("vast_instance_ids") != [] or teardown.get("continuing_spend_from_this_run") is not False:
        raise ValueError("unallocated_resume_lock_refusal_unproven")
    attempt = attempts / "attempt_001"
    if (attempt / "immutable_execution").exists():
        raise ValueError("unallocated_resume_runtime_output_present")
    absent, proof = billing_staged_objects_absent({"attempt_root": str(attempt)})
    if not absent or proof is None:
        raise ValueError("unallocated_resume_staging_absence_unproven")
    records[str(proof.relative_to(parent))] = "sha256:" + hashlib.sha256(proof.read_bytes()).hexdigest()
    receipt = {"schema_version": "policy_canary_unallocated_dispatch_continuation.v1", "status": "ready",
        "parent_job_dir": str(parent), "successor_job_dir": str(successor),
        "authority_digest": authority["authority_digest"], "bundle_sha256": bundle["bundle_sha256"],
        "allocation_binding_digest": binding_digest,
        "prior_consumption_digest": consumption["consumption_digest"], "evidence_sha256": records,
        "provider_allocations_observed": 0, "maximum_remaining_provider_allocations": 1,
        "maximum_cost_usd": authority["hard_cap_usd"], "hard_ttl_seconds": authority["hard_ttl_seconds"],
        "new_preparation_requests_authorized": 0, "existing_funding_binding_required": True,
        "continuation_digest": ""}
    receipt["continuation_digest"] = canonical_digest(receipt, digest_field="continuation_digest")
    if apply:
        # Never delete the old consumption or re-use its output directory.
        with claim.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
        successor.mkdir(parents=True, exist_ok=False)
        (successor / "unallocated_dispatch_continuation.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def validate_unallocated_funding_reuse(path: str | Path, *, funding_admission: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate a claimed, unused successor before reusing its original grant."""
    path = Path(path).resolve()
    receipt = json.loads(path.read_text())
    parent = Path(receipt["parent_job_dir"]).resolve()
    successor = Path(receipt["successor_job_dir"]).resolve()
    if path != successor / "unallocated_dispatch_continuation.json" \
        or receipt.get("schema_version") != "policy_canary_unallocated_dispatch_continuation.v1" \
        or receipt.get("status") != "ready" \
        or receipt.get("continuation_digest") != canonical_digest(receipt, digest_field="continuation_digest") \
        or receipt.get("provider_allocations_observed") != 0 \
        or receipt.get("maximum_remaining_provider_allocations") != 1 \
        or receipt.get("new_preparation_requests_authorized") != 0 \
        or receipt.get("allocation_binding_digest") != funding_admission.get("allocation_binding_digest") \
        or receipt.get("maximum_cost_usd") != funding_admission.get("maximum_cost_usd") \
        or funding_admission.get("provider") != "vast" or funding_admission.get("resource_class") != "gpu_render" \
        or funding_admission.get("request_count") != 1:
        raise ValueError("unallocated_resume_funding_binding_invalid")
    if json.loads((parent / "unallocated_successor_claim.json").read_text()) != receipt \
        or (successor / "policy_canary_session_consumption.json").exists():
        raise ValueError("unallocated_resume_successor_not_unused")
    required = {"policy_canary_session_consumption.json",
        "attempts/attempt_001/vast_provider_run/vast_provider_adapter_result.json",
        "attempts/attempt_001/vast_provider_run/vast_launch_lock_manifest.json",
        "attempts/attempt_001/vast_provider_run/vast_teardown_manifest.json"}
    if not required.issubset(receipt.get("evidence_sha256", {})):
        raise ValueError("unallocated_resume_evidence_invalid")
    for relative, sha in receipt["evidence_sha256"].items():
        evidence = parent / relative
        if evidence.is_symlink() or not evidence.resolve().is_relative_to(parent) \
            or "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest() != sha:
            raise ValueError("unallocated_resume_evidence_changed")
    if sorted(p.name for p in (parent / "attempts").iterdir()) != ["attempt_001"] \
        or not billing_staged_objects_absent({"attempt_root": str(parent / "attempts/attempt_001")})[0]:
        raise ValueError("unallocated_resume_staging_absence_unproven")
    return receipt
