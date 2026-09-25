"""Stage a G1 development pair from the shared Task Evaluation Run catalog.

This is a no-spend bridge from a team-selected robot and policy pair to the
existing local/container G1 worker. It does not make an unavailable catalog
entry runnable or publish a production policy-canary execution profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_task_scoring import TaskNeutralScoringError, validate_rigid_task_success_contract
from .decision_evidence_contracts import canonical_digest
from .native_g1_development_pair import PAIR_ORDER, validate_g1_development_pair
from .native_g1_development_worker import (
    PATH_FIELDS,
    REQUEST_SCHEMA,
    _request,
    _rights_review,
    _verify_packet,
)
from .native_g1_navigation_goal import validate_g1_navigation_goal_authority
from .native_g1_shared_scene_episode import G1_BOX_CANDIDATES, G1_NAVIGATION_CANDIDATES
from .native_task_arena_packet import REQUEST_SCHEMA_VERSION
from .task_evaluation_g1_catalog import G1_PRESET_ID, unavailable_g1_preset
from .task_evaluation_packet_planning_setup import (
    CHOICE_SCHEMA as PACKET_CHOICE_SCHEMA,
    SETUP_SCHEMA as PACKET_SETUP_SCHEMA,
    validate_packet_planning_setup,
    validate_packet_policy_pair_choice,
)
from .task_evaluation_policy_pair_choice import validate_policy_pair_choice
from .task_evaluation_policy_canary_setup import validate_policy_canary_setup


SCHEMA = "native_g1_development_selection.v1"
PACKET_SELECTION_SCHEMA = "native_g1_packet_development_selection.v1"
PLAN_SCHEMA = "native_g1_development_selection_plan.v1"
SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "setup_digest",
        "source_launch_id",
        "robot_preset_id",
        "policy_candidate_ids",
        "objective_id",
        "scene_plan_digest",
        "selection_digest",
    }
)
TEMPLATE_FIELDS = frozenset(
    {
        "schema_version",
        *PATH_FIELDS,
        "sonic_encoder_sha256",
        "sonic_decoder_sha256",
        "port",
        "max_steps",
        "device",
    }
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_selection_input_path_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_selection_input_invalid")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _selected_pair(setup: Mapping[str, Any], selection: Mapping[str, Any]) -> tuple[str, str]:
    if (
        set(selection) != SELECTION_FIELDS
        or selection.get("schema_version") != SCHEMA
        or selection.get("setup_digest") != setup["setup_digest"]
        or selection.get("source_launch_id") != setup["source_launch_id"]
        or selection.get("robot_preset_id") != G1_PRESET_ID
        or selection.get("selection_digest")
        != canonical_digest(selection, digest_field="selection_digest")
    ):
        raise ValueError("g1_selection_contract_invalid")
    chosen = selection.get("policy_candidate_ids")
    if (
        not isinstance(chosen, list)
        or len(chosen) != 2
        or any(not isinstance(candidate, str) for candidate in chosen)
        or len(set(chosen)) != 2
    ):
        raise ValueError("g1_selection_pair_invalid")
    expected = (
        G1_NAVIGATION_CANDIDATES
        if selection.get("objective_id") == "g1_navigation_goal"
        else G1_BOX_CANDIDATES
        if selection.get("objective_id") == "task_success"
        else frozenset()
    )
    ordered = tuple(candidate for candidate in PAIR_ORDER if candidate in chosen)
    if len(ordered) != 2 or set(ordered) != expected or chosen != list(ordered):
        raise ValueError("g1_selection_pair_invalid")
    robots = [row for row in setup["robot_presets"] if row["robot_preset_id"] == G1_PRESET_ID]
    if len(robots) != 1:
        raise ValueError("g1_selection_catalog_missing")
    catalog = unavailable_g1_preset()
    robot = robots[0]
    if any(
        robot.get(key) != catalog[key]
        for key in catalog
        if key not in {"readiness", "policy_candidates"}
    ):
        raise ValueError("g1_selection_catalog_identity_mismatch")
    by_id = {row["candidate_id"]: row for row in robot["policy_candidates"]}
    published = {row["candidate_id"]: row for row in catalog["policy_candidates"]}
    if set(by_id) != set(published) or any(
        {key: value for key, value in by_id[candidate].items() if key != "readiness"}
        != {key: value for key, value in row.items() if key != "readiness"}
        for candidate, row in published.items()
    ):
        raise ValueError("g1_selection_catalog_identity_mismatch")
    return ordered  # type: ignore[return-value]


def seal_g1_development_selection(
    *, setup: Mapping[str, Any], objective_id: str, scene_plan_digest: str
) -> dict[str, Any]:
    """Choose the pinned pair for one objective from a published setup."""

    validated = validate_policy_canary_setup(setup)
    selected = (
        G1_NAVIGATION_CANDIDATES
        if objective_id == "g1_navigation_goal"
        else G1_BOX_CANDIDATES
        if objective_id == "task_success"
        else frozenset()
    )
    if not selected:
        raise ValueError("g1_selection_objective_invalid")
    if (
        not isinstance(scene_plan_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", scene_plan_digest) is None
    ):
        raise ValueError("g1_selection_scene_digest_invalid")
    selection = {
        "schema_version": SCHEMA,
        "setup_digest": validated["setup_digest"],
        "source_launch_id": validated["source_launch_id"],
        "robot_preset_id": G1_PRESET_ID,
        "policy_candidate_ids": [candidate for candidate in PAIR_ORDER if candidate in selected],
        "objective_id": objective_id,
        "scene_plan_digest": scene_plan_digest,
    }
    selection["selection_digest"] = canonical_digest(selection, digest_field="selection_digest")
    _selected_pair(validated, selection)
    return selection


def stage_g1_development_selection(
    *,
    setup_path: Path,
    selection_path: Path | None,
    runtime_template_path: Path,
    rights_review_paths: Mapping[str, Path],
    output_dir: Path,
    navigation_authority_path: Path | None = None,
    objective_id: str | None = None,
    choice_path: Path | None = None,
) -> dict[str, Any]:
    """Write two sealed worker requests from one shared catalog selection."""

    raw_setup = _read(setup_path)
    packet_planning = raw_setup.get("schema_version") == PACKET_SETUP_SCHEMA
    setup = (
        validate_packet_planning_setup(raw_setup)
        if packet_planning
        else validate_policy_canary_setup(raw_setup)
    )
    template = _read(runtime_template_path)
    if (
        set(template) != TEMPLATE_FIELDS
        or template.get("schema_version") != REQUEST_SCHEMA
        or any(
            not isinstance(template.get(field), str) or not template[field].strip()
            for field in PATH_FIELDS
        )
    ):
        raise ValueError("g1_selection_runtime_template_invalid")
    bundle = Path(template["bundle_root"])
    scene_path = bundle / "native_task_arena_scene_plan.v1.json"
    if (
        not bundle.is_absolute()
        or bundle.is_symlink()
        or not scene_path.is_file()
        or scene_path.is_symlink()
    ):
        raise ValueError("g1_selection_scene_packet_missing")
    scene = _read(scene_path)
    if sum(source is not None for source in (selection_path, objective_id, choice_path)) != 1:
        raise ValueError("g1_selection_source_invalid")
    if packet_planning and choice_path is None:
        raise ValueError("g1_selection_packet_choice_required")
    choice = (
        (
            validate_packet_policy_pair_choice(_read(choice_path), setup=setup)
            if packet_planning
            else validate_policy_pair_choice(_read(choice_path), setup=setup)
        )
        if choice_path is not None
        else None
    )
    if packet_planning:
        candidates = tuple(choice["policy_candidate_ids"])
        selection = {
            "schema_version": PACKET_SELECTION_SCHEMA,
            "setup_digest": setup["setup_digest"],
            "source_packet_receipt_digest": setup["source_packet_receipt_digest"],
            "pair_choice_digest": choice["choice_digest"],
            "robot_preset_id": G1_PRESET_ID,
            "policy_candidate_ids": list(candidates),
            "objective_id": choice["objective_id"],
            "scene_plan_digest": scene.get("plan_digest"),
        }
        selection["selection_digest"] = canonical_digest(selection, digest_field="selection_digest")
    else:
        selection = (
            _read(selection_path)
            if selection_path is not None
            else seal_g1_development_selection(
                setup=setup,
                objective_id=choice["objective_id"] if choice is not None else objective_id,
                scene_plan_digest=scene.get("plan_digest"),
            )
        )
        candidates = _selected_pair(setup, selection)
    if choice is not None and (
        choice["robot_preset_id"] != G1_PRESET_ID
        or choice["policy_candidate_ids"] != list(candidates)
    ):
        raise ValueError("g1_selection_pair_choice_invalid")
    if set(rights_review_paths) != set(candidates):
        raise ValueError("g1_selection_rights_review_pair_invalid")
    packet = _verify_packet(bundle)
    if packet_planning:
        request = _read(bundle / f"{REQUEST_SCHEMA_VERSION}.json")
        derivation = request.get("g1_scene_derivation") or {}
        if (
            request.get("request_digest") != packet.get("request_digest")
            or request.get("request_digest")
            != canonical_digest(request, digest_field="request_digest")
            or derivation.get("schema_version") != "native_g1_scene_packet_derivation.v1"
            or derivation.get("source_packet_receipt_digest")
            != setup["source_packet_receipt_digest"]
            or derivation.get("source_scene_plan_digest") != setup["source_scene_plan_digest"]
            or derivation.get("source_declared_task_success_contract_digest")
            != setup["source_declared_task_success_contract_digest"]
            or derivation.get("setup_digest") != setup["setup_digest"]
            or derivation.get("pair_choice_digest") != choice["choice_digest"]
            or request.get("scene_id") != setup["scene_id"]
            or request.get("task_id") != setup["task_id"]
            or (request.get("task_spec") or {}).get("task_success_contract_digest")
            != setup["task_success_contract_digest"]
        ):
            raise ValueError("g1_selection_packet_derivation_mismatch")
    task_spec = scene.get("task_spec") or {}
    task_contract = task_spec.get("task_success_contract") or {}
    scope = task_contract.get("scope") or {}
    public_scope = setup["task_success_contract"]["scope"]
    try:
        validate_rigid_task_success_contract(
            task_contract,
            expected_site_id=public_scope["site_id"],
            expected_task_id=public_scope["task_id"],
        )
    except TaskNeutralScoringError as exc:
        raise ValueError("g1_selection_scene_or_site_mismatch") from exc
    if (
        scene.get("plan_digest") != canonical_digest(scene, digest_field="plan_digest")
        or scene.get("plan_digest") != packet.get("arena_scene_plan_digest")
        or scene.get("plan_digest") != selection.get("scene_plan_digest")
        or scene.get("task_kind") != "rigid_pick_place"
        or (scene.get("robot") or {}).get("robot_id") != "unitree_g1"
        or scene.get("task_id") != public_scope["task_id"]
        or scope.get("site_id") != public_scope["site_id"]
        or scope.get("task_id") != public_scope["task_id"]
        or task_contract.get("criteria") != setup["task_success_contract"]["criteria"]
        or task_spec.get("task_success_contract_digest") != task_contract.get("contract_digest")
    ):
        raise ValueError("g1_selection_scene_or_site_mismatch")
    authority = None
    if selection["objective_id"] == "g1_navigation_goal":
        if navigation_authority_path is None:
            raise ValueError("g1_selection_navigation_authority_missing")
        authority = validate_g1_navigation_goal_authority(
            _read(navigation_authority_path), plan=scene
        )
    elif navigation_authority_path is not None:
        raise ValueError("g1_selection_unexpected_navigation_authority")
    inventory = Path(template["inventory_path"])
    if not inventory.is_absolute() or inventory.is_symlink() or not inventory.is_file():
        raise ValueError("g1_selection_inventory_missing")
    inventory_sha = _sha256(inventory)
    rights = {
        candidate: _rights_review(
            _read(rights_review_paths[candidate]),
            preflight={
                "candidate_id": candidate,
                "scene_plan_digest": scene["plan_digest"],
                "inventory_file_sha256": inventory_sha,
            },
        )
        for candidate in candidates
    }
    requests = []
    for candidate in candidates:
        value = {
            **template,
            "candidate_id": candidate,
            "rights_review": rights[candidate],
            **({"navigation_goal_authority": authority} if authority else {}),
        }
        value["request_digest"] = canonical_digest(value, digest_field="request_digest")
        requests.append(_request(value))
    output = Path(output_dir)
    protected = [
        setup_path,
        runtime_template_path,
        bundle,
        inventory,
        *(Path(template[field]) for field in PATH_FIELDS),
        *rights_review_paths.values(),
        *([selection_path] if selection_path else []),
        *([choice_path] if choice_path else []),
        *([navigation_authority_path] if navigation_authority_path else []),
    ]
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or output.resolve() != output
        or any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected)
    ):
        raise ValueError("g1_selection_output_directory_invalid")
    output.mkdir(parents=True)
    sealed_selection_path = output / (selection["schema_version"] + ".json")
    sealed_selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if choice is not None:
        (
            output / f"{PACKET_CHOICE_SCHEMA if packet_planning else choice['schema_version']}.json"
        ).write_text(json.dumps(choice, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    request_paths = []
    for request in requests:
        path = output / (request["candidate_id"] + ".json")
        path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        request_paths.append(path)
    pair = validate_g1_development_pair(request_paths)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "staged_not_executed",
        "setup_digest": setup["setup_digest"],
        "selection_digest": selection["selection_digest"],
        "selection_path": str(sealed_selection_path),
        "pair_choice_digest": choice["choice_digest"] if choice is not None else None,
        **(
            {"source_packet_receipt_digest": setup["source_packet_receipt_digest"]}
            if packet_planning
            else {"source_launch_id": setup["source_launch_id"]}
        ),
        "robot_preset_id": G1_PRESET_ID,
        "scene_plan_digest": scene["plan_digest"],
        "packet_receipt_digest": packet["receipt_digest"],
        "objective_id": pair["objective_id"],
        "candidate_ids": pair["candidate_ids"],
        "request_paths": [str(path) for path in request_paths],
        "request_digests": pair["request_digests"],
        "rights_review_digests": [
            rights[candidate]["rights_review_digest"] for candidate in candidates
        ],
        "navigation_goal_authority_digest": pair["navigation_goal_authority_digest"],
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    (output / (PLAN_SCHEMA + ".json")).write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--selection", type=Path)
    choice.add_argument("--objective", choices=("task_success", "g1_navigation_goal"))
    choice.add_argument("--choice", type=Path)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--rights-review", action="append", required=True, metavar="CANDIDATE=PATH")
    parser.add_argument("--navigation-authority", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rights: dict[str, Path] = {}
    for entry in args.rights_review:
        candidate, separator, path = entry.partition("=")
        if not separator or not candidate or not path or candidate in rights:
            parser.error("--rights-review requires unique CANDIDATE=PATH values")
        rights[candidate] = Path(path)
    plan = stage_g1_development_selection(
        setup_path=args.setup,
        selection_path=args.selection,
        objective_id=args.objective,
        choice_path=args.choice,
        runtime_template_path=args.runtime_template,
        rights_review_paths=rights,
        output_dir=args.output_dir,
        navigation_authority_path=args.navigation_authority,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
