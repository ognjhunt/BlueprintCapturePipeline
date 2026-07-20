"""Provider-neutral contracts for decision-grade simulator policy comparison."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


POLICY_ADAPTER_SCHEMA_VERSION = "policy_adapter_manifest.v1"
POLICY_EVALUATION_DESIGN_SCHEMA_VERSION = "policy_evaluation_design.v1"
MINIMUM_DECISION_POLICY_COUNT = 7
MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION = 20
SC3_DIRECT_COMPARISON_REPLICATE_TARGET = (36, 37)
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _digest(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "").strip().lower()))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_policy_adapter_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if manifest.get("schema_version") != POLICY_ADAPTER_SCHEMA_VERSION:
        blockers.append("policy_adapter_schema_missing_or_unsupported")
    for field in ("policy_id", "checkpoint_id", "policy_family", "embodiment_id", "version"):
        if not str(manifest.get(field) or "").strip():
            blockers.append(f"policy_adapter_identity_missing:{field}")
    for field in (
        "policy_sha256",
        "checkpoint_sha256",
        "adapter_code_sha256",
        "embodiment_manifest_sha256",
    ):
        if not _digest(manifest.get(field)):
            blockers.append(f"policy_adapter_digest_missing_or_invalid:{field}")

    action = _mapping(manifest.get("action_contract"))
    dimension = action.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        blockers.append("policy_action_dimension_missing_or_invalid")
        dimension = 0
    units = action.get("units")
    if (
        not isinstance(units, Sequence)
        or isinstance(units, (str, bytes, bytearray))
        or len(units) != dimension
        or any(not str(item).strip() for item in units)
    ):
        blockers.append("policy_action_units_missing_or_dimension_mismatch")
    bounds = _rows(action.get("bounds"))
    if len(bounds) != dimension:
        blockers.append("policy_action_bounds_missing_or_dimension_mismatch")
    else:
        for index, bound in enumerate(bounds):
            low = _finite(bound.get("minimum"))
            high = _finite(bound.get("maximum"))
            if low is None or high is None or low >= high:
                blockers.append(f"policy_action_bound_invalid:{index}")
    control_rate = _finite(action.get("control_rate_hz"))
    if control_rate is None or control_rate <= 0:
        blockers.append("policy_action_control_rate_missing_or_invalid")
    if action.get("timestamp_semantics") not in {
        "monotonic_sample_time",
        "monotonic_chunk_start_and_per_sample_offsets",
    }:
        blockers.append("policy_action_timestamp_semantics_missing_or_invalid")
    if not _digest(action.get("normalization_manifest_sha256")):
        blockers.append("policy_action_normalization_digest_missing_or_invalid")
    if action.get("missing_action_behavior") != "block":
        blockers.append("policy_missing_action_behavior_must_block")
    if action.get("out_of_bounds_behavior") != "block":
        blockers.append("policy_out_of_bounds_behavior_must_block")

    blockers = sorted(set(blockers))
    return {
        "schema_version": "policy_adapter_validation.v1",
        "status": "validated" if not blockers else "blocked",
        "policy_id": manifest.get("policy_id"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "blockers": blockers,
    }


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, str, int] | None:
    site_id = str(row.get("site_id") or "").strip()
    task_id = str(row.get("task_id") or "").strip()
    condition_id = str(row.get("condition_id") or "").strip()
    seed = row.get("seed")
    if (
        not site_id
        or not task_id
        or not condition_id
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        return None
    return site_id, task_id, condition_id, seed


def validate_policy_evaluation_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Derive decision-grade eligibility from registry and raw matched rows."""

    blockers: list[str] = []
    if design.get("schema_version") != POLICY_EVALUATION_DESIGN_SCHEMA_VERSION:
        blockers.append("policy_evaluation_design_schema_missing_or_unsupported")
    policies = _rows(design.get("policies"))
    validations = [validate_policy_adapter_manifest(row) for row in policies]
    for validation in validations:
        blockers.extend(
            f"policy_adapter:{validation.get('policy_id') or 'unknown'}:{blocker}"
            for blocker in validation["blockers"]
        )
    policy_ids = [str(row.get("policy_id") or "") for row in policies]
    checkpoint_digests = [str(row.get("checkpoint_sha256") or "") for row in policies]
    if len(set(policy_ids)) < MINIMUM_DECISION_POLICY_COUNT:
        blockers.append(f"independent_policy_count_lt_{MINIMUM_DECISION_POLICY_COUNT}")
    if len(set(checkpoint_digests)) < MINIMUM_DECISION_POLICY_COUNT:
        blockers.append(f"independent_checkpoint_count_lt_{MINIMUM_DECISION_POLICY_COUNT}")
    if len(policy_ids) != len(set(policy_ids)):
        blockers.append("duplicate_policy_id")
    if design.get("hidden_shared_state_prohibited") is not True:
        blockers.append("hidden_shared_state_not_prohibited")
    if design.get("policy_specific_scenario_changes_prohibited") is not True:
        blockers.append("policy_specific_scenario_changes_not_prohibited")

    rows = _rows(design.get("rows"))
    cells_by_policy: dict[str, set[tuple[str, str, str, int]]] = defaultdict(set)
    replicate_seeds: dict[tuple[str, str, str], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    forbidden_flags = (
        "missing_action",
        "zero_action_substitute_used",
        "scripted_target_motion_used",
        "fallback_policy_used",
        "fixture_or_proxy_model_output_used",
        "policy_specific_scenario_change_used",
        "hidden_shared_state_used",
    )
    for index, row in enumerate(rows):
        policy_id = str(row.get("policy_id") or "").strip()
        if policy_id not in set(policy_ids):
            blockers.append(f"evaluation_row_unknown_policy:{index}")
        key = _cell_key(row)
        if key is None:
            blockers.append(f"evaluation_row_cell_identity_invalid:{index}")
            continue
        for flag in forbidden_flags:
            if row.get(flag) is not False:
                blockers.append(f"decision_grade_row_forbidden_or_unproven:{index}:{flag}")
        for field in (
            "observation_sha256",
            "commanded_action_chunk_sha256",
            "policy_runtime_output_sha256",
            "initial_condition_sha256",
            "skeleton_conditioning_sha256",
            "oscar_checkpoint_sha256",
            "model_output_sha256",
            "provider_execution_sha256",
            "next_policy_query_sha256",
            "action_control_suite_sha256",
        ):
            if not _digest(row.get(field)):
                blockers.append(f"evaluation_row_digest_missing_or_invalid:{index}:{field}")
        if row.get("evaluator_profile_id") != "oscar_official_v2":
            blockers.append(f"evaluation_row_oscar_profile_not_proven:{index}")
        if row.get("fresh_official_oscar_model_execution_proven") is not True:
            blockers.append(f"evaluation_row_fresh_oscar_execution_not_proven:{index}")
        run_steps = row.get("fresh_oscar_provider_model_run_steps")
        if isinstance(run_steps, bool) or not isinstance(run_steps, int) or run_steps <= 0:
            blockers.append(f"evaluation_row_fresh_oscar_model_steps_invalid:{index}")
        if row.get("action_control_suite_status") != "passed":
            blockers.append(f"evaluation_row_action_control_suite_not_passed:{index}")
        cells_by_policy[policy_id].add(key)
        site_id, task_id, condition_id, seed = key
        replicate_seeds[(site_id, task_id, condition_id)][policy_id].add(seed)

    registered_policy_ids = set(policy_ids)
    if set(cells_by_policy) != registered_policy_ids:
        blockers.append("not_every_registered_policy_has_evaluation_rows")
    reference_cells = next(iter(cells_by_policy.values()), set())
    for policy_id in sorted(registered_policy_ids):
        if cells_by_policy.get(policy_id, set()) != reference_cells:
            blockers.append(f"asymmetric_matched_cell_coverage:{policy_id}")
    minimum_replicates = design.get("minimum_matched_replicates_per_policy_condition")
    if minimum_replicates != MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION:
        blockers.append(
            "minimum_matched_replicates_must_equal_"
            f"{MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION}"
        )
    for cell, seeds_by_policy in replicate_seeds.items():
        seed_sets = [seeds_by_policy.get(policy_id, set()) for policy_id in policy_ids]
        if any(len(seeds) < MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION for seeds in seed_sets):
            blockers.append("matched_replicate_count_below_minimum:" + ":".join(cell))
        if seed_sets and any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            blockers.append("matched_replicate_seed_sets_differ:" + ":".join(cell))

    direct_sc3_claim_requested = design.get("direct_sc3_comparison_requested") is True
    direct_sc3_target_met = bool(
        direct_sc3_claim_requested
        and replicate_seeds
        and all(
            len(seeds_by_policy.get(policy_id, set()))
            in SC3_DIRECT_COMPARISON_REPLICATE_TARGET
            for seeds_by_policy in replicate_seeds.values()
            for policy_id in policy_ids
        )
    )
    if direct_sc3_claim_requested and not direct_sc3_target_met:
        blockers.append("direct_sc3_comparison_requires_36_or_37_matched_replicates")

    blockers = sorted(set(blockers))
    return {
        "schema_version": "policy_evaluation_design_validation.v1",
        "status": "decision_grade" if not blockers else "blocked",
        "decision_grade_eligible": not blockers,
        "policy_count": len(set(policy_ids)),
        "independent_checkpoint_count": len(set(checkpoint_digests)),
        "matched_cell_count_per_policy": len(reference_cells),
        "minimum_matched_replicates_per_policy_condition": (
            MINIMUM_MATCHED_REPLICATES_PER_POLICY_CONDITION
        ),
        "direct_sc3_comparison_requested": direct_sc3_claim_requested,
        "direct_sc3_replicate_target_met": direct_sc3_target_met,
        "g1_kitchen_fixture_present": any(
            row.get("qualification_fixture") == "g1_kitchen" for row in policies
        ),
        "g1_kitchen_is_product_architecture": False,
        "blockers": blockers,
        "claim_boundary": {
            "policy_family_provider_and_embodiment_neutral": True,
            "fallback_fixture_and_proxy_rows_are_not_decision_grade": True,
            "matched_cells_do_not_by_themselves_prove_real_world_ordering": True,
        },
    }
