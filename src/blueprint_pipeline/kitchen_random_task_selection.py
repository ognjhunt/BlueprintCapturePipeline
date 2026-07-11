"""Immutable scene inventory and reproducible random kitchen-task selection.

This module performs no provider calls.  It turns a checked task registry and a
fresh no-spend preflight manifest into the one task identity that every later
provider attempt must preserve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso


REGISTRY_SCHEMA_VERSION = "kitchen_unitree_g1_task_registry.v1"
SELECTION_SCHEMA_VERSION = "kitchen_random_task_selection.v1"
INVENTORY_SCHEMA_VERSION = "kitchen_asset_inventory.v1"
ISAAC_SCENARIO_SCHEMA_VERSION = "kitchen_selected_isaac_scenario.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_mapping(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    payload = _load_mapping(path)
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("kitchen task registry schema mismatch")
    tasks = [dict(row) for row in payload.get("tasks", []) if isinstance(row, Mapping)]
    ids = [str(row.get("task_id") or "") for row in tasks]
    if not tasks or any(not task_id for task_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("kitchen task registry requires unique non-empty task ids")
    return tasks


def build_preflight_task_specs(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    allowed = {
        "task_id",
        "scenario_id",
        "description",
        "required_target_terms",
        "zone",
        "preferred_stance_distance_m",
        "stance_distance_candidates_m",
    }
    return {
        "tasks": [{key: value for key, value in task.items() if key in allowed} for task in tasks]
    }


def inventory_kitchen_scene(kitchen_usd: str | Path) -> dict[str, Any]:
    main = Path(kitchen_usd).expanduser().resolve()
    if not main.is_file():
        raise FileNotFoundError(main)
    root = main.parent
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=str):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    inventory_digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "complete",
        "asset_root": str(root),
        "main_usd": str(main),
        "main_usd_relative_path": main.relative_to(root).as_posix(),
        "main_usd_sha256": _sha256(main),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "inventory_sha256": inventory_digest,
        "files": rows,
        "claim_boundary": (
            "This is a local byte inventory. Provider-side presence and simulator scene-load "
            "proof require separate attempt-bound artifacts."
        ),
    }


def _gate_passed(report: Mapping[str, Any], text: str) -> bool:
    return any(
        str(row.get("status") or "").upper() == "PASS"
        and text.lower() in str(row.get("name") or "").lower()
        for row in report.get("local_gates", [])
        if isinstance(row, Mapping)
    )


def _artifact_mapping(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    path_text = str(_mapping(report.get("artifacts")).get(name) or "")
    path = Path(path_text).expanduser() if path_text else None
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def evaluate_candidates(
    *, tasks: Sequence[Mapping[str, Any]], preflight_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    reports = {
        str(row.get("task_id") or ""): dict(row)
        for row in preflight_manifest.get("tasks", [])
        if isinstance(row, Mapping)
    }
    candidates: list[dict[str, Any]] = []
    for raw in sorted(tasks, key=lambda row: str(row.get("task_id") or "")):
        task = dict(raw)
        task_id = str(task.get("task_id") or "")
        report = reports.get(task_id, {})
        stance = _artifact_mapping(report, "task_stance_plan")
        placement = _artifact_mapping(report, "placement_validation")
        geometry = _artifact_mapping(report, "manipulation_pov_geometry")
        selected = _mapping(_mapping(stance.get("target_resolution")).get("selected"))
        selected_affordance = _mapping(
            _mapping(stance.get("affordance_resolution")).get("selected")
        )
        policy = _mapping(task.get("policy_contract"))
        completion = _mapping(task.get("completion_contract"))
        criteria = [
            dict(row)
            for row in completion.get("registered_criteria", [])
            if isinstance(row, Mapping)
        ]
        blockers: list[str] = []
        if report.get("status") != "passed":
            blockers.append("local_preflight_not_passed")
        if stance.get("status") != "accepted":
            blockers.append("task_stance_plan_not_accepted")
        if str(placement.get("status") or "").upper() != "PASS":
            blockers.append("placement_validation_not_passed")
        if geometry.get("status") != "PASS":
            blockers.append("manipulation_pov_geometry_not_passed")
        if not _gate_passed(report, "full kitchen scene loaded"):
            blockers.append("full_kitchen_scene_gate_not_passed")
        if not _gate_passed(report, "target resolves semantically"):
            blockers.append("target_resolution_gate_not_passed")
        if not _gate_passed(report, "target visible"):
            blockers.append("camera_target_visibility_gate_not_passed")
        first_stance_candidate = next(
            (
                dict(row)
                for row in stance.get("candidates", [])[:1]
                if isinstance(row, Mapping)
            ),
            {},
        )
        if not _gate_passed(report, "collision") and not (
            placement.get("scene_collision_contact_count") == 0
            or first_stance_candidate.get("scene_collision_contact_count") == 0
        ):
            blockers.append("collision_clearance_not_proven")
        if not selected or not str(selected.get("target_object_id") or ""):
            blockers.append("resolved_scene_target_missing")
        if not selected_affordance or not str(
            selected_affordance.get("target_object_id") or ""
        ):
            blockers.append("resolved_scene_affordance_missing")
        if not task.get("affordance_object_ids"):
            blockers.append("registered_affordance_ids_missing")
        if not criteria:
            blockers.append("registered_completion_criterion_missing")
        if not (
            policy.get("locomotion_provider") == "unitree_g1_policy"
            and policy.get("manipulation_provider") == "unitree_groot_n17_sonic_policy"
            and policy.get("action_command") == "UNITREE_G1_SONIC"
            and policy.get("controller_fk_bridge_required") is True
        ):
            blockers.append("unitree_policy_controller_path_not_registered")
        candidates.append(
            {
                "task_id": task_id,
                "eligible": not blockers,
                "rejection_blockers": sorted(set(blockers)),
                "task_prompt": task.get("description"),
                "target_ids": list(task.get("target_object_ids") or []),
                "affordance_ids": list(task.get("affordance_object_ids") or []),
                "resolved_target": selected or None,
                "resolved_affordance": selected_affordance or None,
                "completion_contract": completion,
                "policy_contract": policy,
                "preflight_report": report,
                "stance_plan": stance or None,
                "placement_validation": placement or None,
                "camera_reach_geometry": geometry or None,
            }
        )
    return candidates


def select_random_task(
    *,
    registry_path: str | Path,
    preflight_manifest_path: str | Path,
    kitchen_usd: str | Path,
    out_dir: str | Path,
    seed: int | None = None,
    invalidation_paths: Sequence[str | Path] = (),
    selection_artifact_name: str = "random_task_selection.json",
    specification_artifact_name: str = "selected_task_specification.json",
    inventory_artifact_name: str = "kitchen_asset_inventory.json",
) -> dict[str, Any]:
    out = Path(out_dir).expanduser().resolve()
    tasks = load_registry(registry_path)
    preflight = _load_mapping(preflight_manifest_path)
    inventory = inventory_kitchen_scene(kitchen_usd)
    candidates = evaluate_candidates(tasks=tasks, preflight_manifest=preflight)
    invalidations: list[dict[str, Any]] = []
    invalidation_by_task: dict[str, dict[str, Any]] = {}
    for raw_path in invalidation_paths:
        path = Path(raw_path).expanduser().resolve()
        record = _load_mapping(path)
        task_id = str(record.get("selected_task_id") or "")
        if not (
            record.get("status") == "invalidated"
            and record.get("task_selection_invalidated") is True
            and task_id
        ):
            raise ValueError(f"invalid task-invalidation contract: {path}")
        if task_id in invalidation_by_task:
            raise ValueError(f"duplicate task invalidation: {task_id}")
        bound = {
            **record,
            "source_path": str(path),
            "source_sha256": _sha256(path),
        }
        invalidations.append(bound)
        invalidation_by_task[task_id] = bound
    known_ids = {str(row.get("task_id") or "") for row in candidates}
    unknown_invalidations = sorted(set(invalidation_by_task) - known_ids)
    if unknown_invalidations:
        raise ValueError(
            f"task invalidation not present in candidate registry: {','.join(unknown_invalidations)}"
        )
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        invalidation = invalidation_by_task.get(task_id)
        if invalidation is None:
            continue
        candidate["eligible_before_live_invalidation"] = bool(candidate.get("eligible"))
        candidate["eligible"] = False
        candidate["rejection_blockers"] = sorted(
            set(candidate.get("rejection_blockers") or [])
            | {"invalidated_by_fresh_live_provider_scene_evidence"}
        )
        candidate["live_provider_invalidation"] = invalidation
    eligible_ids = sorted(row["task_id"] for row in candidates if row["eligible"])
    if not eligible_ids:
        raise RuntimeError("no eligible kitchen tasks after fresh local preflight")
    selected_seed = secrets.randbits(64) if seed is None else int(seed)
    if not 0 <= selected_seed < 2**64:
        raise ValueError("seed must be a uint64")
    selected_index = random.Random(selected_seed).randrange(len(eligible_ids))
    selected_id = eligible_ids[selected_index]
    selected = next(row for row in candidates if row["task_id"] == selected_id)
    for name in (
        selection_artifact_name,
        specification_artifact_name,
        inventory_artifact_name,
    ):
        if not name.endswith(".json") or Path(name).name != name:
            raise ValueError("artifact names must be simple JSON filenames")
    inventory_path = out / "scene" / inventory_artifact_name
    _write_json_exclusive(inventory_path, inventory)
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "selected",
        "seed_uint64": selected_seed,
        "selection_algorithm": (
            "sort eligible task_id strings ascending; initialize Python random.Random(seed_uint64); "
            "select randrange(len(eligible_ids))"
        ),
        "eligible_task_ids_sorted": eligible_ids,
        "selected_index": selected_index,
        "selected_task_id": selected_id,
        "selected_task": selected,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_ids),
        "candidates": candidates,
        "live_provider_invalidations": invalidations,
        "registry_path": str(Path(registry_path).expanduser().resolve()),
        "registry_sha256": _sha256(Path(registry_path).expanduser().resolve()),
        "preflight_manifest_path": str(Path(preflight_manifest_path).expanduser().resolve()),
        "preflight_manifest_sha256": _sha256(
            Path(preflight_manifest_path).expanduser().resolve()
        ),
        "kitchen_asset_inventory_path": str(
            inventory_path.resolve()
        ),
        "kitchen_asset_inventory_sha256": _sha256(inventory_path),
        "reroll_policy": (
            "No reroll is permitted for difficulty. Invalidation requires fresh scene evidence "
            "that the selected candidate was ineligible and a recorded invalidation artifact."
        ),
    }
    _write_json_exclusive(out / selection_artifact_name, payload)
    _write_json_exclusive(out / specification_artifact_name, selected)
    return payload


def materialize_selected_task_inputs(
    *, selection_path: str | Path, out_dir: str | Path
) -> dict[str, Any]:
    """Write launch inputs derived only from the immutable selected-task artifact."""

    selection = _load_mapping(selection_path)
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("random task selection schema mismatch")
    selected = _mapping(selection.get("selected_task"))
    stance = _mapping(selected.get("stance_plan"))
    accepted_pose = stance.get("accepted_pose")
    if not (
        isinstance(accepted_pose, Sequence)
        and not isinstance(accepted_pose, (str, bytes, bytearray))
        and len(accepted_pose) == 3
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in accepted_pose)
    ):
        raise ValueError("selected task has no accepted three-dimensional stance")
    completion_contract = _mapping(selected.get("completion_contract"))
    if not completion_contract.get("registered_criteria"):
        raise ValueError("selected task has no registered completion criterion")
    out = Path(out_dir).expanduser().resolve()
    route_path = out / "selected_task_route.json"
    completion_path = out / "task_success_contract.json"
    route = {
        "schema_version": "kitchen_selected_task_route.v1",
        "task_id": selection.get("selected_task_id"),
        "selection_seed_uint64": selection.get("seed_uint64"),
        "route_points": [list(accepted_pose), list(accepted_pose)],
        "accepted_stance_yaw_rad": stance.get("accepted_yaw"),
        "stance_focus_xyz": stance.get("stance_focus_xyz"),
        "route_semantics": (
            "The G1 begins at the already accepted manipulation stance. No fabricated "
            "navigation waypoint or unrelated target coordinate is introduced."
        ),
        "source_selection_path": str(Path(selection_path).expanduser().resolve()),
        "source_selection_sha256": _sha256(Path(selection_path).expanduser().resolve()),
    }
    _write_json_exclusive(route_path, route)
    _write_json_exclusive(
        completion_path,
        {
            **completion_contract,
            "task_id": selection.get("selected_task_id"),
            "task_prompt": selected.get("task_prompt"),
            "target_ids": selected.get("target_ids"),
            "affordance_ids": selected.get("affordance_ids"),
            "resolved_target": selected.get("resolved_target"),
            "source_selection_path": str(Path(selection_path).expanduser().resolve()),
            "source_selection_sha256": _sha256(Path(selection_path).expanduser().resolve()),
        },
    )
    return {
        "status": "materialized",
        "route_path": str(route_path),
        "task_success_contract_path": str(completion_path),
    }


def materialize_selected_isaac_scenario(
    *,
    selection_path: str | Path,
    out_dir: str | Path,
    scenario_eval_run_id: str,
    artifact_name: str = "selected_isaac_scenario.json",
) -> dict[str, Any]:
    """Bind the selected task and accepted stance to one immutable Isaac request.

    The provider still resolves the real USD target and repeats collision/placement
    validation against the loaded G1 model. The local accepted pose is reference
    evidence, while all collision-free scene-grounded standoffs remain available so
    model-specific root/visual offsets cannot turn an approximate local pass into a
    false provider rejection.
    """

    selection_path = Path(selection_path).expanduser().resolve()
    selection = _load_mapping(selection_path)
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("random task selection schema mismatch")
    selected = _mapping(selection.get("selected_task"))
    stance = _mapping(selected.get("stance_plan"))
    accepted_pose = stance.get("accepted_pose")
    accepted_yaw = stance.get("accepted_yaw")
    if not (
        stance.get("status") == "accepted"
        and isinstance(accepted_pose, Sequence)
        and not isinstance(accepted_pose, (str, bytes, bytearray))
        and len(accepted_pose) == 3
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in accepted_pose
        )
        and isinstance(accepted_yaw, (int, float))
        and not isinstance(accepted_yaw, bool)
    ):
        raise ValueError("selected task has no accepted stance contract")
    candidates = [
        dict(row) for row in stance.get("candidates", []) if isinstance(row, Mapping)
    ]
    accepted_candidate = next(
        (
            row
            for row in candidates
            if row.get("placement_validation", {}).get("status") == "accepted"
            and list(row.get("pose") or []) == list(accepted_pose)
            and float(row.get("yaw")) == float(accepted_yaw)
        ),
        {},
    )
    standoff = accepted_candidate.get("standoff_from_target_surface_m")
    if not isinstance(standoff, (int, float)) or isinstance(standoff, bool) or standoff <= 0:
        raise ValueError("accepted stance has no positive target-surface standoff")
    stance_ladder = sorted(
        {
            float(row["standoff_from_target_surface_m"])
            for row in candidates
            if row.get("scene_collision_contact_count") == 0
            and isinstance(row.get("standoff_from_target_surface_m"), (int, float))
            and not isinstance(row.get("standoff_from_target_surface_m"), bool)
            and float(row["standoff_from_target_surface_m"]) > 0
        }
    )
    if not stance_ladder:
        raise ValueError("selected task has no collision-free stance ladder")
    completion = _mapping(selected.get("completion_contract"))
    if not completion.get("registered_criteria"):
        raise ValueError("selected task has no registered completion criterion")
    task_prompt = str(selected.get("task_prompt") or "").strip()
    target_ids = [str(value) for value in selected.get("target_ids", []) if str(value)]
    affordance_ids = [
        str(value) for value in selected.get("affordance_ids", []) if str(value)
    ]
    if not task_prompt or not target_ids or not affordance_ids:
        raise ValueError("selected task lacks prompt, target ids, or affordance ids")
    report = _mapping(selected.get("preflight_report"))
    scenario_id = str(report.get("scenario_id") or selection.get("selected_task_id") or "")
    if not scenario_id or not str(scenario_eval_run_id).strip():
        raise ValueError("scenario id and scenario eval run id are required")
    prompts = list(dict.fromkeys([task_prompt, *target_ids, *affordance_ids]))
    scenario = {
        "scenario_id": scenario_id,
        "scenario_eval_run_id": str(scenario_eval_run_id),
        "task_id": selection.get("selected_task_id"),
        "description": task_prompt,
        "task": task_prompt,
        "task_description": task_prompt,
        "task_instruction": task_prompt,
        "task_target_deferred": True,
        "floor_z_hint": 0.05,
        "perception_target_prompts": prompts,
        "target_object_ids": target_ids,
        "affordance_object_ids": affordance_ids,
        "preferred_stance_distance_m": float(standoff),
        "stance_distance_candidates_m": stance_ladder,
        "task_success_contract": completion,
        "accepted_stance_contract": {
            "status": "accepted",
            "pose_xyz": [float(value) for value in accepted_pose],
            "yaw_rad": float(accepted_yaw),
            "stance_focus_xyz": stance.get("stance_focus_xyz"),
            "resolved_target": selected.get("resolved_target"),
            "resolved_affordance": selected.get("resolved_affordance"),
            "collision_contact_count": accepted_candidate.get(
                "scene_collision_contact_count"
            ),
            "source_selection_sha256": _sha256(selection_path),
            "authority": "local_preflight_reference_only",
            "provider_revalidation_required": True,
            "provider_accepted_stance_may_differ": True,
        },
    }
    payload = {
        "schema_version": ISAAC_SCENARIO_SCHEMA_VERSION,
        "selection_seed_uint64": selection.get("seed_uint64"),
        "source_selection_path": str(selection_path),
        "source_selection_sha256": _sha256(selection_path),
        "scenarios": [scenario],
    }
    out = Path(out_dir).expanduser().resolve()
    if (
        not artifact_name.endswith(".json")
        or Path(artifact_name).name != artifact_name
        or artifact_name in {".", ".."}
    ):
        raise ValueError("artifact_name must be a simple JSON filename")
    path = out / artifact_name
    _write_json_exclusive(path, payload)
    return {"status": "materialized", "scenario_path": str(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--materialize-selected-inputs", action="store_true")
    parser.add_argument("--materialize-isaac-scenario", action="store_true")
    parser.add_argument("--preflight-manifest")
    parser.add_argument("--kitchen-usd")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenario-eval-run-id")
    parser.add_argument("--scenario-artifact-name", default="selected_isaac_scenario.json")
    parser.add_argument("--invalidation", action="append", default=[])
    parser.add_argument("--selection-id")
    parser.add_argument("--active-from-attempt-id")
    parser.add_argument("--selection-artifact-name", default="random_task_selection.json")
    parser.add_argument(
        "--specification-artifact-name", default="selected_task_specification.json"
    )
    parser.add_argument("--inventory-artifact-name", default="kitchen_asset_inventory.json")
    args = parser.parse_args(argv)
    modes = [
        args.prepare,
        args.select,
        args.materialize_selected_inputs,
        args.materialize_isaac_scenario,
    ]
    if sum(bool(value) for value in modes) != 1:
        parser.error(
            "choose exactly one of --prepare, --select, --materialize-selected-inputs, "
            "or --materialize-isaac-scenario"
        )
    out = Path(args.out_dir).expanduser().resolve()
    tasks = load_registry(args.registry)
    if args.prepare:
        path = out / "preflight_task_specs.json"
        _write_json_exclusive(path, build_preflight_task_specs(tasks))
        print(json.dumps({"status": "prepared", "path": str(path)}, sort_keys=True))
        return 0
    if args.materialize_selected_inputs:
        if not args.preflight_manifest:
            parser.error("--materialize-selected-inputs requires selection path in --preflight-manifest")
        result = materialize_selected_task_inputs(
            selection_path=args.preflight_manifest,
            out_dir=out,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.materialize_isaac_scenario:
        if not args.preflight_manifest or not args.scenario_eval_run_id:
            parser.error(
                "--materialize-isaac-scenario requires selection path in "
                "--preflight-manifest and --scenario-eval-run-id"
            )
        result = materialize_selected_isaac_scenario(
            selection_path=args.preflight_manifest,
            out_dir=out,
            scenario_eval_run_id=args.scenario_eval_run_id,
            artifact_name=args.scenario_artifact_name,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if not args.preflight_manifest or not args.kitchen_usd:
        parser.error("--select requires --preflight-manifest and --kitchen-usd")
    if args.seed is None or not args.active_from_attempt_id:
        parser.error(
            "--select requires --seed and --active-from-attempt-id so the immutable "
            "selection generation can be attempt-bound"
        )
    if len(args.invalidation) > 1:
        parser.error("one immutable selection generation accepts at most one supersession")
    # Production automation writes only immutable selection-ID directories.  The
    # direct Python ``select_random_task`` helper remains for local/support callers.
    from .kitchen_attempt_lineage import (
        activate_selection_generation,
        create_selection_generation,
    )

    generation = create_selection_generation(
        generations_dir=out / "selections",
        registry_path=args.registry,
        preflight_manifest_path=args.preflight_manifest,
        kitchen_usd=args.kitchen_usd,
        seed=args.seed,
        invalidation_paths=args.invalidation,
        selection_id=args.selection_id,
    )
    prior_pointer = out / "active_selection_pointer.json"
    pointer = activate_selection_generation(
        run_dir=out,
        generation=generation,
        active_from_attempt_id=args.active_from_attempt_id,
        prior_pointer_path=prior_pointer if prior_pointer.is_file() else None,
        invalidation_path=args.invalidation[0] if args.invalidation else None,
    )
    print(
        json.dumps(
            {
                "status": generation["status"],
                "selection_id": generation["selection_id"],
                "selection_sha256": generation["selection_sha256"],
                "selected_task_id": generation["selected_task_id"],
                "active_from_attempt_id": pointer["active_from_attempt_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
