"""Validate the provider-free launch-readiness seal for ADP-009D Scene 840920.

This contract is intentionally upstream of a policy bundle.  A real policy
bundle must consume a completed controls-positive receipt, whose digest does
not exist yet.  The readiness seal proves everything that can be proved before
that predecessor arrives and fails closed if either frozen candidate is not
rights-admitted and runnable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from .decision_evidence_contracts import canonical_digest
from .dual_task_scenario_suite import validate_dual_task_scenario_suite
from .native_task_runtime_contract import (
    FROZEN_CANDIDATES,
    SUPPORTED_SCENARIO_RUNTIME_TARGETS,
)


SCHEMA_VERSION = "adp009d_scene_policy_readiness.v1"
SCENE_ID = "840920"
TASK_ID = "task_a_washer_door_open"
CONTROLS_PREDECESSOR = "authoritative_controls_positive_receipt_missing"
READY_VERDICT = "READY_WAITING_ONLY_FOR_CONTROLS"


class ScenePolicyReadinessError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def validate_scene_policy_readiness(
    value: Mapping[str, Any], *, scenario_suite: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one outcome-blind, provider-free readiness seal."""

    payload = json.loads(json.dumps(value, allow_nan=False))
    suite = validate_dual_task_scenario_suite(scenario_suite)
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("scene_policy_readiness_schema_invalid")
    if payload.get("scene_id") != SCENE_ID or payload.get("task_id") != TASK_ID:
        errors.append("scene_policy_readiness_task_binding_invalid")
    if payload.get("candidate_ids") != list(FROZEN_CANDIDATES):
        errors.append("scene_policy_readiness_candidates_invalid")
    if payload.get("scenario_suite_digest") != suite.get("suite_digest"):
        errors.append("scene_policy_readiness_scenario_digest_mismatch")
    if payload.get("task_freeze_digest") != suite.get("task_freeze_digest"):
        errors.append("scene_policy_readiness_task_freeze_mismatch")

    scenario = _mapping(payload.get("scenario_matrix"))
    cells = list(suite.get("cells") or [])
    if (
        scenario.get("cell_count") != len(cells)
        or scenario.get("shared_candidate_cell_count") != len(cells)
        or scenario.get("seeds") != sorted({row.get("seed") for row in cells})
        or scenario.get("initial_execution_order")
        != suite.get("initial_execution_order")
    ):
        errors.append("scene_policy_readiness_scenario_matrix_invalid")
    for cell in cells:
        if cell.get("candidate_ids") != list(FROZEN_CANDIDATES):
            errors.append("scene_policy_readiness_scenario_pairing_invalid")
        for factor in cell.get("factor_records") or []:
            target = factor.get("runtime_target")
            if target not in SUPPORTED_SCENARIO_RUNTIME_TARGETS:
                errors.append(f"scene_policy_readiness_runtime_target_unready:{target}")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    by_id = {
        str(row.get("candidate_id") or ""): row
        for row in candidates
        if isinstance(row, Mapping)
    }
    if set(by_id) != set(FROZEN_CANDIDATES) or len(candidates) != 2:
        errors.append("scene_policy_readiness_candidate_inventory_invalid")
    for candidate_id in FROZEN_CANDIDATES:
        row = _mapping(by_id.get(candidate_id))
        expected = EXPECTED_CANDIDATES[candidate_id]
        source = _mapping(row.get("source"))
        checkpoint = _mapping(row.get("checkpoint"))
        for field in ("repository", "revision", "tree"):
            if source.get(field) != expected[f"source_{field}"]:
                errors.append(f"scene_policy_readiness_{candidate_id}_source_invalid")
        if (
            checkpoint.get("repository") != expected["checkpoint_repository"]
            or checkpoint.get("revision") != expected["checkpoint_revision"]
            or checkpoint.get("inventory_digest")
            != expected["checkpoint_inventory_digest"]
            or checkpoint.get("total_bytes") != expected["checkpoint_total_bytes"]
        ):
            errors.append(
                f"scene_policy_readiness_{candidate_id}_checkpoint_invalid"
            )
        for field in (
            "rights_ready",
            "observation_adapter_ready",
            "action_adapter_ready",
            "malformed_output_fails_closed",
            "candidate_can_grade_itself",
        ):
            expected_value = field != "candidate_can_grade_itself"
            if row.get(field) is not expected_value:
                errors.append(
                    f"scene_policy_readiness_{candidate_id}_{field}_invalid"
                )
        if row.get("claim_ceiling") != "development_only":
            errors.append(f"scene_policy_readiness_{candidate_id}_claim_invalid")

    controls = _mapping(payload.get("controls_predecessor"))
    if (
        controls.get("status") != "waiting"
        or controls.get("blocker_code") != CONTROLS_PREDECESSOR
        or controls.get("required_schema")
        != "native_task_arena_control_result.v1"
        or controls.get("receipt_digest") is not None
        or controls.get("bypass_permitted") is not False
    ):
        errors.append("scene_policy_readiness_controls_predecessor_invalid")
    if payload.get("external_blockers") != []:
        errors.append("scene_policy_readiness_external_blocker_invalid")
    if payload.get("verdict") != READY_VERDICT:
        errors.append("scene_policy_readiness_verdict_invalid")

    terminal = _mapping(payload.get("terminal_contract"))
    required_terminal = {
        "retry_cap": 0,
        "teardown_required": True,
        "provider_zero_required": True,
        "webapp_sync_required": True,
        "controls_predecessor_required": True,
        "candidate_identity_required": True,
    }
    if any(terminal.get(key) != expected for key, expected in required_terminal.items()):
        errors.append("scene_policy_readiness_terminal_contract_invalid")
    media = _mapping(payload.get("episode_evidence_contract"))
    if any(
        media.get(key) is not True
        for key in (
            "lossless_policy_frames_required",
            "digest_bound_frame_manifest_required",
            "derived_review_video_required",
            "typed_pre_observation_media_gap_required",
            "deterministic_simulator_success_source_required",
        )
    ) or media.get("policy_self_grading_forbidden") is not True:
        errors.append("scene_policy_readiness_media_contract_invalid")
    if payload.get("readiness_digest") != canonical_digest(
        payload, digest_field="readiness_digest"
    ):
        errors.append("scene_policy_readiness_digest_invalid")
    if errors:
        raise ScenePolicyReadinessError(errors)
    return payload


def load_scene_policy_readiness(
    path: str | Path, *, scenario_suite_path: str | Path
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    scenario = json.loads(Path(scenario_suite_path).read_text(encoding="utf-8"))
    return validate_scene_policy_readiness(report, scenario_suite=scenario)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--scenario-suite", required=True)
    args = parser.parse_args(argv)
    try:
        result = load_scene_policy_readiness(
            args.report, scenario_suite_path=args.scenario_suite
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "validated",
                "verdict": result["verdict"],
                "external_blockers": result["external_blockers"],
                "controls_predecessor": result["controls_predecessor"],
                "readiness_digest": result["readiness_digest"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONTROLS_PREDECESSOR",
    "READY_VERDICT",
    "SCHEMA_VERSION",
    "ScenePolicyReadinessError",
    "load_scene_policy_readiness",
    "validate_scene_policy_readiness",
]
