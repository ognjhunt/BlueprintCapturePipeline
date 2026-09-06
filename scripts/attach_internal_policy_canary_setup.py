#!/usr/bin/env python3
"""Attach one verified internal canary setup before immutable profile publication."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_dispatcher import validate_launch_profile
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    public_launch_profile_descriptor,
    validate_launch_profile_structure,
)
from blueprint_pipeline.task_evaluation_policy_canary_preparation_dispatch import (
    validate_policy_canary_execution_plan,
)
from blueprint_pipeline.task_evaluation_policy_canary_setup import (
    validate_policy_canary_setup,
)
from blueprint_pipeline import task_evaluation_scene_policy_binding as scene_policy


def attach_internal_policy_canary_setup(
    *,
    profile: Mapping[str, Any],
    setup: Mapping[str, Any],
    profile_validator: Callable[[Mapping[str, Any]], list[str]] = validate_launch_profile,
) -> dict[str, Any]:
    """Return a new digest-bound profile; never mutate published bytes in place."""

    source = deepcopy(dict(profile))
    blockers = profile_validator(source)
    if blockers:
        raise ValueError("policy_canary_source_profile_invalid:" + ",".join(blockers))
    canary = validate_policy_canary_setup(setup)
    configured_source_launch_id = source.get(
        "configured_source_launch_id", source.get("profile_id")
    )
    if canary["source_launch_id"] != configured_source_launch_id:
        raise ValueError("policy_canary_setup_profile_binding_mismatch")
    if "internal_policy_canary_setup" in source:
        if source["internal_policy_canary_setup"] != canary:
            raise ValueError("policy_canary_setup_immutable_conflict")
        return source
    output = deepcopy(source)
    output["internal_policy_canary_setup"] = canary
    output["profile_digest"] = canonical_digest(output, digest_field="profile_digest")
    blockers = profile_validator(output)
    if blockers:
        raise ValueError("policy_canary_profile_invalid:" + ",".join(blockers))
    return output


def materialize_policy_canary_launch_profile(
    *,
    base_configured_profile: Mapping[str, Any],
    profile_materialization_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a new current-main canary profile without editing configured bytes."""

    base = deepcopy(dict(base_configured_profile))
    # The base is the completed scene-configuration profile used only as a $0
    # materialization template; this step admits nothing for paid execution. It is
    # validated structurally so the paid-admission owner-record reopen (provider,
    # reservation, consent) is NOT run against the configuration-time attempt the
    # template still points at. That reopen is preserved where it belongs: on the
    # materialized canary `output` below (full `validate_launch_profile`) and again
    # at dispatch time, both of which carry the reserved policy attempt.
    blockers = validate_launch_profile_structure(base)
    if blockers:
        raise ValueError("policy_canary_source_profile_invalid:" + ",".join(blockers))
    wrapper = deepcopy(dict(profile_materialization_input))
    expected_wrapper_fields = {
        "schema_version",
        "profile_id",
        "configured_base_profile_id",
        "configured_base_profile_digest",
        "configured_source_launch_id",
        "source_commit",
        "internal_policy_canary_setup",
        "internal_policy_canary_execution_plan",
        "task_success_contract",
        "task_success_contract_digest",
        "materialization_digest",
    }
    if (
        set(wrapper) - {"scene_id", "scene_attempt_binding"} != expected_wrapper_fields
        or wrapper.get("schema_version")
        != "task_evaluation_policy_canary_profile_materialization_input.v1"
        or wrapper.get("materialization_digest")
        != canonical_digest(wrapper, digest_field="materialization_digest")
        or wrapper.get("configured_base_profile_id") != base.get("profile_id")
        or wrapper.get("configured_base_profile_digest") != base.get("profile_digest")
    ):
        raise ValueError("policy_canary_profile_materialization_input_invalid")
    setup = validate_policy_canary_setup(wrapper["internal_policy_canary_setup"])
    plan = validate_policy_canary_execution_plan(
        wrapper["internal_policy_canary_execution_plan"], public_setup=setup
    )
    if (
        wrapper["configured_source_launch_id"] != setup["source_launch_id"]
        or wrapper["configured_source_launch_id"]
        != plan["configured_source_launch_id"]
        or wrapper.get("scene_id") != plan.get("scene_id")
        or wrapper["task_success_contract"] != setup["task_success_contract"]
        or wrapper["task_success_contract"] != plan["task_success_contract"]
        or wrapper["task_success_contract_digest"]
        != setup["task_success_contract_digest"]
        or wrapper["task_success_contract_digest"]
        != plan["task_success_contract_digest"]
    ):
        raise ValueError("policy_canary_offering_lineage_invalid")
    output = deepcopy(base)
    output.update(
        {
            "profile_id": wrapper["profile_id"],
            "source_commit": wrapper["source_commit"],
            "configured_source_launch_id": wrapper["configured_source_launch_id"],
            "claim_ceiling": "diagnostic_policy_execution",
            "internal_policy_canary_setup": setup,
            "internal_policy_canary_execution_plan": plan,
            "policy_run_setup": plan["legacy_policy_run_setup"],
        }
    )
    binding = plan.get("scene_policy_binding")
    if base.get("scene_intent_digest") is not None or binding is not None:
        if not isinstance(binding, Mapping) or not isinstance(wrapper.get("scene_attempt_binding"), Mapping):
            raise ValueError("scene_policy_profile_binding_missing")
        scene_policy.validate_owner_binding(base, binding, source_commit=wrapper["source_commit"])
        output.update(scene_intent_digest=binding["scene_intent_digest"],
            scene_attempt_id=binding["attempt_id"], scene_policy_candidates=binding["policy_candidates"],
            scene_attempt_binding=wrapper["scene_attempt_binding"])
        blockers = scene_policy.profile_binding_blockers(output)
        if blockers:
            raise ValueError(",".join(blockers))
    resource = plan["resource_authority"]
    output["allocator"] = {
        "entrypoint": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
        "subcommand": "gpu-canary",
        "argv": [
            "--provider",
            "vast",
            "--probe-kind",
            "native-task-arena-policy-canary-session",
            "--policy-canary-preparation-authority-required",
        ],
        "max_spend_usd": resource["hard_cap_usd"],
        "hard_ttl_seconds": resource["hard_ttl_seconds"],
        "retry_cap": 0,
    }
    output["profile_digest"] = canonical_digest(output, digest_field="profile_digest")
    blockers = validate_launch_profile(output)
    if blockers:
        raise ValueError("policy_canary_profile_invalid:" + ",".join(blockers))
    public_launch_profile_descriptor(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--canary-setup")
    parser.add_argument("--profile-materialization-input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    if bool(args.canary_setup) == bool(args.profile_materialization_input):
        raise ValueError("policy_canary_profile_materialization_mode_invalid")
    if args.canary_setup:
        setup = json.loads(Path(args.canary_setup).read_text(encoding="utf-8"))
        output = attach_internal_policy_canary_setup(profile=profile, setup=setup)
    else:
        wrapper = json.loads(
            Path(args.profile_materialization_input).read_text(encoding="utf-8")
        )
        output = materialize_policy_canary_launch_profile(
            base_configured_profile=profile,
            profile_materialization_input=wrapper,
        )
    destination = Path(args.output).expanduser().resolve()
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8", errors="strict")
    except OSError as exc:
        raise ValueError("policy_canary_profile_output_invalid") from exc
    print(json.dumps({"profile_id": output["profile_id"], "profile_digest": output["profile_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
