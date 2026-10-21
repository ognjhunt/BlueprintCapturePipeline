"""Prepare exact native cell packets for the existing construction worker.

Preparation preserves the canonical packet request and runtime contract as
source predecessors. It creates no construction result or execution authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .exact_workcell_native_runtime import prepare_native_runtime_cell
from .exact_workcell_variation_matrix import ExactWorkcellVariationError, _is_digest


SCHEMA_VERSION = "exact_workcell_native_cell_preparation.v1"
SOURCE_PLAN_FILENAME = "exact_workcell_source_arena_scene_plan.v1.json"
PREPARATION_FILENAME = "exact_workcell_native_cell_preparation.v1.json"


def prepare_native_packet_cell(
    *,
    source_scene_plan: Mapping[str, Any],
    cell_preparation: Mapping[str, Any],
    source_packet_request_digest: str,
    source_runtime_contract_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive one immutable cell without replaying source native qualification.

    ``cell_preparation`` contains exactly request, matrix, cell_id and
    source_plan_digest. Full frozen inputs are retained in the preparation
    artifact so a worker/operator can deterministically reconstruct the plan.
    """
    if not isinstance(cell_preparation, Mapping) or set(cell_preparation) != {
        "request",
        "matrix",
        "cell_id",
        "source_plan_digest",
    }:
        raise ExactWorkcellVariationError(["native_cell_packet_preparation_invalid"])
    try:
        value = json.loads(json.dumps(dict(cell_preparation), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExactWorkcellVariationError(["native_cell_packet_preparation_not_json"]) from exc
    if not all(
        _is_digest(digest)
        for digest in (source_packet_request_digest, source_runtime_contract_digest)
    ):
        raise ExactWorkcellVariationError(["native_cell_packet_source_predecessor_invalid"])
    if (
        source_scene_plan.get("task_kind") != "rigid_pick_place"
        or source_scene_plan.get("robot", {}).get("robot_id") != "franka_panda"
        or source_scene_plan.get("task_spec", {}).get("astra_asset_adoption") is not None
    ):
        raise ExactWorkcellVariationError(["native_cell_packet_profile_unavailable"])
    if source_scene_plan.get("runtime_contract_digest") != source_runtime_contract_digest:
        raise ExactWorkcellVariationError(["native_cell_packet_runtime_predecessor_mismatch"])
    plan = prepare_native_runtime_cell(
        matrix=value["matrix"],
        request=value["request"],
        cell_id=value["cell_id"],
        source_scene_plan=source_scene_plan,
        source_plan_digest=value["source_plan_digest"],
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_not_native_qualified",
        "source_packet_request_digest": source_packet_request_digest,
        "source_runtime_contract_digest": source_runtime_contract_digest,
        "source_plan_digest": value["source_plan_digest"],
        "cell_preparation": value,
        "prepared_plan_digest": plan["plan_digest"],
        "exact_workcell_binding": plan["scenario"]["exact_workcell_binding"],
        "construction_result_created": False,
        "native_application_claimed": False,
        "policy_episode_claimed": False,
        "provider_mutation_performed": False,
        "execution_authority_granted": False,
    }
    receipt["preparation_digest"] = canonical_digest(receipt, digest_field="preparation_digest")
    return plan, receipt


def replay_native_packet_cell(
    *,
    source_scene_plan: Mapping[str, Any],
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact prepared plan from retained source/input bytes."""
    if preparation.get("schema_version") != SCHEMA_VERSION or preparation.get(
        "preparation_digest"
    ) != canonical_digest(preparation, digest_field="preparation_digest"):
        raise ExactWorkcellVariationError(["native_cell_packet_preparation_digest_mismatch"])
    plan, receipt = prepare_native_packet_cell(
        source_scene_plan=source_scene_plan,
        cell_preparation=preparation["cell_preparation"],
        source_packet_request_digest=preparation["source_packet_request_digest"],
        source_runtime_contract_digest=preparation["source_runtime_contract_digest"],
    )
    if receipt != dict(preparation):
        raise ExactWorkcellVariationError(["native_cell_packet_preparation_replay_mismatch"])
    return plan
