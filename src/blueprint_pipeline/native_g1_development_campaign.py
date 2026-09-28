"""Bind G1 manipulation and movement choices to one development site/task.

This is the controller's no-spend admission input. It verifies the two sealed
scene packets and owner decisions without copying scene bytes or model weights.
Paid allocation and episode execution remain separate controller steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .native_g1_development_pair import PAIR_ORDER
from .native_g1_development_selection import verify_g1_packet_choice_bundle
from .native_g1_development_worker import _rights_review
from .native_g1_navigation_goal import validate_g1_navigation_goal_authority
from .task_evaluation_packet_planning_setup import (
    HANDOFF_SCHEMA,
    make_packet_policy_pair_choice,
    validate_packet_policy_handoff,
)


SCHEMA = "native_g1_development_campaign.v1"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_campaign_input_path_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_campaign_input_invalid")
    return value


def _sha256(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_campaign_inventory_missing")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _movement_handoff(book_handoff: Mapping[str, Any]) -> dict[str, Any]:
    setup = book_handoff["setup"]
    choice = make_packet_policy_pair_choice(setup=setup, objective_id="g1_navigation_goal")
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "claim_ceiling": "planning_only",
        "setup": setup,
        "choice": choice,
    }
    handoff["handoff_digest"] = cross_runtime_canonical_digest(
        handoff, digest_field="handoff_digest"
    )
    return validate_packet_policy_handoff(handoff)


def _chosen_movement_handoff(
    book_handoff: Mapping[str, Any], movement_handoff_path: Path | None
) -> dict[str, Any]:
    movement_handoff = (
        validate_packet_policy_handoff(_read(movement_handoff_path))
        if movement_handoff_path is not None
        else _movement_handoff(book_handoff)
    )
    if (
        movement_handoff["choice"]["objective_id"] != "g1_navigation_goal"
        or movement_handoff["setup"] != book_handoff["setup"]
    ):
        raise ValueError("g1_campaign_movement_choice_or_setup_mismatch")
    return movement_handoff


def plan_g1_development_campaign(
    *,
    book_handoff_path: Path,
    movement_handoff_path: Path | None = None,
    manipulation_packet: Path,
    movement_packet: Path,
    inventory_path: Path,
    rights_review_paths: Mapping[str, Path],
    navigation_authority_path: Path,
) -> dict[str, Any]:
    """Verify one four-candidate campaign without allocating or copying data."""

    book_handoff = validate_packet_policy_handoff(_read(book_handoff_path))
    if book_handoff["choice"]["objective_id"] != "task_success":
        raise ValueError("g1_campaign_book_objective_invalid")
    movement_handoff = _chosen_movement_handoff(book_handoff, movement_handoff_path)
    book = verify_g1_packet_choice_bundle(
        handoff_path=book_handoff_path, bundle=manipulation_packet
    )
    # The movement handoff is derived from the same source setup in memory;
    # neither a second handoff file nor a second scene-packet copy is needed.
    movement = verify_g1_packet_choice_bundle(handoff=movement_handoff, bundle=movement_packet)
    movement_scene = _read(movement_packet / "native_task_arena_scene_plan.v1.json")
    if (
        book["scene_id"] != movement["scene_id"]
        or book["task_id"] != movement["task_id"]
        or book["scene_plan_digest"] == movement["scene_plan_digest"]
    ):
        raise ValueError("g1_campaign_site_task_or_objective_mismatch")
    authority = validate_g1_navigation_goal_authority(
        _read(navigation_authority_path), plan=movement_scene
    )
    candidate_ids = [*book["candidate_ids"], *movement_handoff["choice"]["policy_candidate_ids"]]
    if candidate_ids != list(PAIR_ORDER) or set(rights_review_paths) != set(candidate_ids):
        raise ValueError("g1_campaign_candidate_set_invalid")
    inventory_sha256 = _sha256(inventory_path)
    rights = {}
    for candidate in candidate_ids:
        scene_digest = (
            book["scene_plan_digest"]
            if candidate in book["candidate_ids"]
            else movement["scene_plan_digest"]
        )
        rights[candidate] = _rights_review(
            _read(rights_review_paths[candidate]),
            preflight={
                "candidate_id": candidate,
                "scene_plan_digest": scene_digest,
                "inventory_file_sha256": inventory_sha256,
            },
        )
    result = {
        "schema_version": SCHEMA,
        "status": "verified_for_remote_staging_not_executed",
        "claim_ceiling": "development_only",
        "scene_id": book["scene_id"],
        "task_id": book["task_id"],
        "source_packet_receipt_digest": book_handoff["setup"]["source_packet_receipt_digest"],
        "movement_handoff_digest": movement_handoff["handoff_digest"],
        "inventory_file_sha256": inventory_sha256,
        "objectives": [
            {
                "objective_id": "task_success",
                "choice_digest": book_handoff["choice"]["choice_digest"],
                "scene_plan_digest": book["scene_plan_digest"],
                "packet_receipt_digest": book["packet_receipt_digest"],
                "candidate_ids": book["candidate_ids"],
            },
            {
                "objective_id": "g1_navigation_goal",
                "choice_digest": movement_handoff["choice"]["choice_digest"],
                "scene_plan_digest": movement["scene_plan_digest"],
                "packet_receipt_digest": movement["packet_receipt_digest"],
                "candidate_ids": movement["candidate_ids"],
                "navigation_goal_authority_digest": authority["authority_digest"],
            },
        ],
        "rights_review_digests": {
            candidate: rights[candidate]["rights_review_digest"] for candidate in candidate_ids
        },
        "episode_executed": False,
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-handoff", type=Path, required=True)
    parser.add_argument("--movement-handoff", type=Path)
    parser.add_argument("--manipulation-packet", type=Path, required=True)
    parser.add_argument("--movement-packet", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--rights-review", action="append", required=True, metavar="CANDIDATE=PATH")
    parser.add_argument("--navigation-authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rights: dict[str, Path] = {}
    for raw in args.rights_review:
        candidate, separator, path = raw.partition("=")
        if not separator or not candidate or not path or candidate in rights:
            parser.error("--rights-review requires a distinct CANDIDATE=PATH")
        rights[candidate] = Path(path)
    result = plan_g1_development_campaign(
        book_handoff_path=args.book_handoff,
        movement_handoff_path=args.movement_handoff,
        manipulation_packet=args.manipulation_packet,
        movement_packet=args.movement_packet,
        inventory_path=args.inventory,
        rights_review_paths=rights,
        navigation_authority_path=args.navigation_authority,
    )
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        raise ValueError("g1_campaign_output_invalid")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "plan_digest": result["plan_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
