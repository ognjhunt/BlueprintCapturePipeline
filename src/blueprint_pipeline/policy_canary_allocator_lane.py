"""Isolated canonical-allocator branch for one paired policy canary session."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .native_task_arena_policy_canary_session import (
    PROBE_KIND,
    validate_provider_bundle,
    validate_session_authority,
)
from .native_task_arena_vast import (
    POLICY_PROVIDER_RUNTIME_ENVIRONMENT_NAMES,
    run_native_task_arena_policy_canary_session_vast,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


def add_policy_canary_allocator_arguments(parser: Any) -> None:
    parser.add_argument("--native-task-arena-policy-canary-session-authority")
    parser.add_argument("--native-task-arena-policy-canary-session-bundle-receipt")


def _load(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("policy_canary_allocator_input_invalid")
    return dict(value)


def run_policy_canary_allocator_lane(
    args: Any,
    control_context: tuple[list[str], Mapping[str, Any]],
) -> int:
    control_blockers, control_identity = control_context
    blockers = list(control_blockers)
    if args.provider != "vast":
        blockers.append("policy_canary_session_provider_must_be_vast")
    if not args.adp_job_dir:
        blockers.append("policy_canary_session_job_dir_missing")
    if not args.native_task_arena_policy_canary_session_authority:
        blockers.append("policy_canary_session_authority_missing")
    if not args.native_task_arena_policy_canary_session_bundle_receipt:
        blockers.append("policy_canary_session_bundle_receipt_missing")
    authority = None
    prepared_bundle = None
    if not blockers:
        try:
            authority = validate_session_authority(
                _load(args.native_task_arena_policy_canary_session_authority)
            )
            prepared_bundle = validate_provider_bundle(
                _load(args.native_task_arena_policy_canary_session_bundle_receipt),
                authority=authority,
            )
            if (
                float(authority["hard_cap_usd"]) != float(args.adp_max_spend_usd)
                or int(authority["hard_ttl_seconds"])
                != int(args.adp_hard_ttl_seconds)
            ):
                blockers.append("policy_canary_session_resource_bounds_mismatch")
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("policy_canary_session_contract_invalid")
    binding = {
        "program_id": "arm-decision-proof-v1",
        "probe_kind": PROBE_KIND,
        "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "bundle_sha256": prepared_bundle.get("bundle_sha256") if prepared_bundle else None,
        "runtime_inputs_digest": (
            prepared_bundle.get("runtime_inputs_digest") if prepared_bundle else None
        ),
        "authority_digest": authority.get("authority_digest") if authority else None,
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "episodes_per_policy": 10,
        "learned_policy_rollout_count": 20,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "hard_cap_usd": args.adp_max_spend_usd,
        "hard_ttl_seconds": args.adp_hard_ttl_seconds,
    }
    binding_digest = "sha256:" + hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter", blockers=blockers
    )
    admission.update(
        {
            "program_id": "arm-decision-proof-v1",
            "probe_kind": PROBE_KIND,
            "control_plane_identity": dict(control_identity),
            "max_hourly_rate_usd": args.adp_max_hourly_rate_usd,
            "hard_cap_usd": args.adp_max_spend_usd,
            "hard_ttl_seconds": args.adp_hard_ttl_seconds,
            "retry_cap": 0,
            "maximum_provider_allocations": 1,
            "candidate_policy_queried": True,
            "physical_outcome_values_uploaded": False,
            "allocation_binding": binding,
            "allocation_binding_digest": binding_digest,
        }
    )
    write_json(Path(args.admission_out), admission)
    grant = None
    if args.execute:
        try:
            grant = require_paid_resource_admission(
                admission,
                resource_class="vast_provider_adapter",
                expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
            )
        except PaidResourceAdmissionBlocked as exc:
            result = {
                "status": "blocked",
                "blockers": exc.blockers,
                "provider_mutations_performed": 0,
            }
            write_json(Path(args.adapter_output), result)
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
    if prepared_bundle is None or authority is None:
        result = {
            "status": "blocked",
            "blockers": sorted(set(blockers)),
            "provider_mutations_performed": 0,
        }
    else:
        result = run_native_task_arena_policy_canary_session_vast(
            job_dir=args.adp_job_dir,
            prepared_bundle=prepared_bundle,
            session_authority=authority,
            paid_resource_admission_grant=grant,
            execute=args.execute,
            machine_avoidlist_path=args.adp_machine_avoidlist,
            max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
            hard_cap_usd=args.adp_max_spend_usd,
            hard_ttl_seconds=args.adp_hard_ttl_seconds,
            allowed_active_instance_ids=args.adp_allowed_active_vast_instance_id,
            provider_runtime_environment={
                name: os.environ[name]
                for name in POLICY_PROVIDER_RUNTIME_ENVIRONMENT_NAMES
                if name in os.environ
            },
        )
    write_json(Path(args.adapter_output), result)
    success = result.get("status") in {"dry_run_ready", "completed"}
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


__all__ = [
    "PROBE_KIND",
    "add_policy_canary_allocator_arguments",
    "run_policy_canary_allocator_lane",
]
