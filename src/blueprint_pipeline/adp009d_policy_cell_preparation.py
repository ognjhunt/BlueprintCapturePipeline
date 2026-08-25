"""Prepare the frozen ADP-009D task-A policy cells without a provider.

This module freezes the exact cell/seed inputs shared by both learned-policy
candidates.  It deliberately does not build a native scene packet or claim a
cell executable: every cell still needs its own qualified scripted-positive
controls receipt, and the object-cousin cell additionally remains rights- and
bytes-blocked in the committed suite.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adp009d_scene_policy_readiness import validate_scene_policy_readiness
from .decision_evidence_contracts import canonical_digest
from .dual_task_scenario_suite import COUSIN_BLOCKER, validate_dual_task_scenario_suite
from .native_task_runtime_contract import FROZEN_CANDIDATES


SCHEMA_VERSION = "adp009d_dual_candidate_cell_preparation.v1"
SCENARIO_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"
CONTROLS_BLOCKER = "qualified_cell_specific_controls_receipt_missing"


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if not isinstance(cloned, dict):
        raise ValueError(error)
    return cloned


def prepare_policy_cell_matrix(
    *, scenario_suite: Mapping[str, Any], policy_readiness: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze candidate-identical inputs and honest predecessor blockers."""

    suite = validate_dual_task_scenario_suite(scenario_suite)
    readiness = validate_scene_policy_readiness(
        policy_readiness, scenario_suite=suite
    )
    if (
        readiness.get("scene_id") != suite.get("scene_id")
        or readiness.get("task_id") != suite.get("task_id")
        or readiness.get("candidate_ids") != list(FROZEN_CANDIDATES)
    ):
        raise ValueError("policy_cell_preparation_readiness_binding_mismatch")

    cells: list[dict[str, Any]] = []
    for cell in suite["cells"]:
        instance: dict[str, Any] = {
            "schema_version": SCENARIO_INSTANCE_SCHEMA_VERSION,
            "program_id": "arm-decision-proof-v1",
            "scene_id": suite["scene_id"],
            "task_id": suite["task_id"],
            "cell_id": cell["cell_id"],
            "family": cell["family"],
            "seed": cell["seed"],
            "scenario_suite_digest": suite["suite_digest"],
            "task_freeze_digest": suite["task_freeze_digest"],
            "resolved_parameters": _clone(
                cell["resolved_parameters"],
                error="policy_cell_resolved_parameters_invalid",
            ),
            "factor_records": json.loads(
                json.dumps(cell["factor_records"], allow_nan=False)
            ),
            "policy_neutral": True,
            "caller_asserted_success": False,
            "learned_policy_outcomes_consulted": False,
            "instance_digest": "",
        }
        instance["instance_digest"] = canonical_digest(
            instance, digest_field="instance_digest"
        )
        admission = cell.get("execution_admission")
        cousin_blocked = (
            isinstance(admission, Mapping)
            and admission.get("status") == "blocked"
        )
        blockers = (
            [str(admission.get("blocker_code") or COUSIN_BLOCKER)]
            if cousin_blocked
            else [CONTROLS_BLOCKER]
        )
        candidates = [
            {
                "candidate_id": candidate_id,
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "scenario_instance_digest": instance["instance_digest"],
                "status": (
                    "blocked_before_packet_materialization"
                    if cousin_blocked
                    else "waiting_for_cell_specific_controls"
                ),
                "blockers": blockers,
                "policy_execution_spec_materialized": False,
                "provider_execution_performed": False,
            }
            for candidate_id in FROZEN_CANDIDATES
        ]
        cells.append(
            {
                "cell_id": cell["cell_id"],
                "family": cell["family"],
                "seed": cell["seed"],
                "scheduled_initially": cell["scheduled_initially"],
                "scenario_instance": instance,
                "candidate_runs": candidates,
                "cell_specific_native_packet_materialized": False,
                "cell_specific_controls_qualified": False,
                "blockers": blockers,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": suite["scene_id"],
        "task_id": suite["task_id"],
        "scenario_suite_digest": suite["suite_digest"],
        "scene_policy_readiness_digest": readiness["readiness_digest"],
        "candidate_ids": list(FROZEN_CANDIDATES),
        "cell_count": len(cells),
        "candidate_cell_count": len(cells) * len(FROZEN_CANDIDATES),
        "provider_free_scenario_instances_materialized": len(cells),
        "native_packets_materialized": 0,
        "policy_execution_specs_materialized": 0,
        "executable_candidate_cells_before_controls": 0,
        "cells": cells,
        "learned_policy_outcomes_consulted": False,
        "provider_mutation_performed": False,
        "paid_resource_allocation_performed": False,
        "claim_ceiling": "development_only_predecessor_preparation",
        "materialization_digest": "",
    }
    result["materialization_digest"] = canonical_digest(
        result, digest_field="materialization_digest"
    )
    return result


def materialize_policy_cell_matrix(
    *,
    scenario_suite_path: str | Path,
    policy_readiness_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Read, validate, and write one deterministic provider-free receipt."""

    try:
        suite = json.loads(Path(scenario_suite_path).read_text(encoding="utf-8"))
        readiness = json.loads(
            Path(policy_readiness_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("policy_cell_preparation_input_invalid") from exc
    if not isinstance(suite, Mapping) or not isinstance(readiness, Mapping):
        raise ValueError("policy_cell_preparation_input_invalid")
    result = prepare_policy_cell_matrix(
        scenario_suite=suite,
        policy_readiness=readiness,
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("policy_cell_preparation_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "CONTROLS_BLOCKER",
    "SCHEMA_VERSION",
    "SCENARIO_INSTANCE_SCHEMA_VERSION",
    "materialize_policy_cell_matrix",
    "prepare_policy_cell_matrix",
]
