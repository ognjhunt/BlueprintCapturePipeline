"""Plan an embodiment/policy pair from a verified development task packet.

The normal canary setup binds a published offering and a runnable profile. A
retained development packet may have a newer task contract than that offering.
This contract preserves the exact packet identity for planning without
inventing a published offering, runnable pair, or execution authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_task_scoring import validate_rigid_task_success_contract
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .native_task_arena_bundle import verify_native_task_arena_packet
from .native_task_arena_packet import REQUEST_SCHEMA_VERSION
from .task_evaluation_g1_catalog import G1_PRESET_ID, unavailable_g1_preset


SETUP_SCHEMA = "task_evaluation_packet_planning_setup.v1"
CHOICE_SCHEMA = "task_evaluation_packet_policy_pair_choice.v1"
SETUP_FIELDS = frozenset(
    {
        "schema_version",
        "claim_ceiling",
        "scene_id",
        "task_id",
        "source_packet_receipt_digest",
        "source_packet_request_digest",
        "source_scene_plan_digest",
        "source_declared_task_success_contract_digest",
        "task_success_contract",
        "task_success_contract_digest",
        "robot_presets",
        "setup_digest",
    }
)
CHOICE_FIELDS = frozenset(
    {
        "schema_version",
        "claim_ceiling",
        "setup_digest",
        "source_packet_receipt_digest",
        "robot_preset_id",
        "policy_candidate_ids",
        "objective_id",
        "choice_digest",
    }
)


def validate_packet_planning_setup(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a planning catalog with no implied launch/offering identity."""

    setup = dict(value)
    if (
        set(setup) != SETUP_FIELDS
        or setup.get("schema_version") != SETUP_SCHEMA
        or setup.get("claim_ceiling") != "planning_only"
        or not isinstance(setup.get("scene_id"), str)
        or not isinstance(setup.get("task_id"), str)
        or not setup["scene_id"]
        or not setup["task_id"]
        or any(
            not isinstance(setup.get(field), str)
            or not setup[field].startswith("sha256:")
            or len(setup[field]) != 71
            for field in (
                "source_packet_receipt_digest",
                "source_packet_request_digest",
                "source_scene_plan_digest",
                "source_declared_task_success_contract_digest",
                "task_success_contract_digest",
            )
        )
        or setup.get("robot_presets") != [unavailable_g1_preset()]
        or setup.get("setup_digest") != cross_runtime_canonical_digest(setup, digest_field="setup_digest")
    ):
        raise ValueError("packet_planning_setup_invalid")
    contract = validate_rigid_task_success_contract(
        setup["task_success_contract"],
        require_confirmed=True,
        expected_site_id=setup["scene_id"],
        expected_task_id=setup["task_id"],
    )
    if contract["contract_digest"] != setup["task_success_contract_digest"]:
        raise ValueError("packet_planning_task_contract_mismatch")
    return setup


def make_packet_planning_setup(*, source_packet_dir: Path) -> dict[str, Any]:
    """Project the exact task plus pinned unavailable G1 catalog."""

    source_root, receipt, _ = verify_native_task_arena_packet(source_packet_dir)
    request = json.loads((source_root / f"{REQUEST_SCHEMA_VERSION}.json").read_text())
    task = request.get("task_spec") or {}
    if (
        task.get("task_kind") != "rigid_pick_place"
        or request.get("request_digest") != receipt.get("request_digest")
        or not isinstance(task.get("task_success_contract_digest"), str)
    ):
        raise ValueError("packet_planning_source_task_invalid")
    contract = validate_rigid_task_success_contract(
        task.get("task_success_contract") or {},
        expected_site_id=request["scene_id"],
        expected_task_id=request["task_id"],
    )
    destination = contract["criteria"]["destination_containment"]
    bounds = destination.get("position_bounds_world_m") or {}
    target = task.get("target_position_world_m")
    if (
        destination.get("mode") != "required"
        or not isinstance(target, list)
        or len(target) != 3
        or not all(
            lower <= coordinate <= upper
            for coordinate, lower, upper in zip(
                target,
                bounds.get("minimum") or [],
                bounds.get("maximum") or [],
                strict=True,
            )
        )
    ):
        raise ValueError("packet_planning_source_destination_mismatch")
    setup = {
        "schema_version": SETUP_SCHEMA,
        "claim_ceiling": "planning_only",
        "scene_id": request["scene_id"],
        "task_id": request["task_id"],
        "source_packet_receipt_digest": receipt["receipt_digest"],
        "source_packet_request_digest": receipt["request_digest"],
        "source_scene_plan_digest": receipt["arena_scene_plan_digest"],
        "source_declared_task_success_contract_digest": task["task_success_contract_digest"],
        "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
        "robot_presets": [unavailable_g1_preset()],
    }
    setup["setup_digest"] = cross_runtime_canonical_digest(setup, digest_field="setup_digest")
    return validate_packet_planning_setup(setup)


def validate_packet_policy_pair_choice(
    value: Mapping[str, Any],
    *,
    setup: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind two compatible unavailable candidates to one packet setup."""

    catalog = validate_packet_planning_setup(setup)
    choice = dict(value)
    if (
        set(choice) != CHOICE_FIELDS
        or choice.get("schema_version") != CHOICE_SCHEMA
        or choice.get("claim_ceiling") != "planning_only"
        or choice.get("setup_digest") != catalog["setup_digest"]
        or choice.get("source_packet_receipt_digest") != catalog["source_packet_receipt_digest"]
        or choice.get("robot_preset_id") != G1_PRESET_ID
        or choice.get("choice_digest") != cross_runtime_canonical_digest(choice, digest_field="choice_digest")
    ):
        raise ValueError("packet_policy_pair_choice_binding_invalid")
    selected = choice.get("policy_candidate_ids")
    objective = choice.get("objective_id")
    candidates = catalog["robot_presets"][0]["policy_candidates"]
    by_id = {row["candidate_id"]: row for row in candidates}
    if (
        objective not in {"task_success", "g1_navigation_goal"}
        or not isinstance(selected, list)
        or len(selected) != 2
        or any(not isinstance(item, str) or item not in by_id for item in selected)
        or len(set(selected)) != 2
        or any(by_id[item]["evaluation_objective_id"] != objective for item in selected)
        or selected
        != [row["candidate_id"] for row in candidates if row["candidate_id"] in selected]
    ):
        raise ValueError("packet_policy_pair_choice_candidates_invalid")
    return choice


def make_packet_policy_pair_choice(
    *,
    setup: Mapping[str, Any],
    objective_id: str,
) -> dict[str, Any]:
    catalog = validate_packet_planning_setup(setup)
    candidates = [
        row["candidate_id"]
        for row in catalog["robot_presets"][0]["policy_candidates"]
        if row["evaluation_objective_id"] == objective_id
    ]
    choice = {
        "schema_version": CHOICE_SCHEMA,
        "claim_ceiling": "planning_only",
        "setup_digest": catalog["setup_digest"],
        "source_packet_receipt_digest": catalog["source_packet_receipt_digest"],
        "robot_preset_id": G1_PRESET_ID,
        "policy_candidate_ids": candidates,
        "objective_id": objective_id,
    }
    choice["choice_digest"] = cross_runtime_canonical_digest(choice, digest_field="choice_digest")
    return validate_packet_policy_pair_choice(choice, setup=catalog)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-packet", type=Path, required=True)
    parser.add_argument(
        "--objective", choices=("task_success", "g1_navigation_goal"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("packet_planning_output_exists")
    setup = make_packet_planning_setup(source_packet_dir=args.source_packet)
    choice = make_packet_policy_pair_choice(setup=setup, objective_id=args.objective)
    args.output_dir.mkdir(parents=True)
    for name, value in ((SETUP_SCHEMA, setup), (CHOICE_SCHEMA, choice)):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {"setup_digest": setup["setup_digest"], "choice_digest": choice["choice_digest"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
