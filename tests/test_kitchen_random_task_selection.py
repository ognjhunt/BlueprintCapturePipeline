from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.kitchen_random_task_selection import (
    build_preflight_task_specs,
    evaluate_candidates,
    inventory_kitchen_scene,
    materialize_selected_isaac_scenario,
    materialize_selected_task_inputs,
    select_random_task,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _task(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "scenario_id": f"scenario_{task_id}",
        "description": f"Do {task_id}",
        "required_target_terms": [task_id],
        "zone": "manipulation",
        "target_object_ids": [task_id],
        "affordance_object_ids": [f"{task_id}_handle"],
        "policy_contract": {
            "locomotion_provider": "unitree_g1_policy",
            "manipulation_provider": "unitree_groot_n17_sonic_policy",
            "action_command": "UNITREE_G1_SONIC",
            "controller_fk_bridge_required": True,
        },
        "completion_contract": {
            "task_kind": "manipulation",
            "registered_criteria": [
                {
                    "criterion_id": f"{task_id}_angle",
                    "observable_transition": "articulation_angle_rad",
                    "comparison": "increase_at_least",
                    "tolerance": 0.2,
                    "unit": "rad",
                }
            ],
        },
    }


def _passed_report(tmp_path: Path, task_id: str) -> dict:
    task_dir = tmp_path / task_id
    stance = _write_json(
        task_dir / "task_stance_plan.json",
        {
            "status": "accepted",
            "accepted_pose": [1.0, 2.0, 0.84],
            "accepted_yaw": 0.0,
            "stance_focus_xyz": [1.4, 2.0, 1.0],
            "target_resolution": {
                "status": "resolved",
                "selected": {"target_object_id": task_id},
            },
            "affordance_resolution": {
                "status": "resolved",
                "selected": {"target_object_id": f"{task_id}_handle"},
            },
            "candidates": [
                {
                    "pose": [1.0, 2.0, 0.84],
                    "yaw": 0.0,
                    "standoff_from_target_surface_m": 0.24,
                    "scene_collision_contact_count": 0,
                    "placement_validation": {"status": "accepted"},
                },
                {
                    "pose": [1.08, 2.0, 0.84],
                    "yaw": 0.0,
                    "standoff_from_target_surface_m": 0.16,
                    "scene_collision_contact_count": 0,
                    "placement_validation": {"status": "blocked"},
                },
            ],
        },
    )
    placement = _write_json(
        task_dir / "placement_validation.json",
        {"status": "PASS", "scene_collision_contact_count": 0},
    )
    geometry = _write_json(task_dir / "manipulation_pov_geometry.json", {"status": "PASS"})
    return {
        "task_id": task_id,
        "status": "passed",
        "local_gates": [
            {"name": "full kitchen scene loaded", "status": "PASS"},
            {"name": "target resolves semantically", "status": "PASS"},
            {"name": "target visible in manipulation POV", "status": "PASS"},
        ],
        "artifacts": {
            "task_stance_plan": str(stance),
            "placement_validation": str(placement),
            "manipulation_pov_geometry": str(geometry),
        },
    }


def test_preflight_projection_excludes_completion_and_policy_contracts() -> None:
    projected = build_preflight_task_specs([_task("faucet")])
    assert projected["tasks"][0]["task_id"] == "faucet"
    assert "completion_contract" not in projected["tasks"][0]
    assert "policy_contract" not in projected["tasks"][0]


def test_inventory_hashes_complete_scene_tree(tmp_path: Path) -> None:
    root = tmp_path / "Collected_KitchenRoom"
    main = root / "KitchenRoom.usd"
    main.parent.mkdir()
    main.write_bytes(b"#usda kitchen")
    (root / "texture.png").write_bytes(b"texture")
    result = inventory_kitchen_scene(main)
    assert result["file_count"] == 2
    assert result["main_usd_sha256"] == hashlib.sha256(b"#usda kitchen").hexdigest()
    assert len(result["inventory_sha256"]) == 64


def test_candidate_requires_registered_completion_and_unitree_path(tmp_path: Path) -> None:
    task = _task("faucet")
    report = _passed_report(tmp_path, "faucet")
    accepted = evaluate_candidates(tasks=[task], preflight_manifest={"tasks": [report]})
    assert accepted[0]["eligible"] is True

    task["completion_contract"] = {"task_kind": "manipulation", "registered_criteria": []}
    rejected = evaluate_candidates(tasks=[task], preflight_manifest={"tasks": [report]})
    assert rejected[0]["eligible"] is False
    assert "registered_completion_criterion_missing" in rejected[0]["rejection_blockers"]

    task["completion_contract"] = _task("faucet")["completion_contract"]
    stance_path = Path(report["artifacts"]["task_stance_plan"])
    stance = json.loads(stance_path.read_text())
    stance.pop("affordance_resolution")
    stance_path.write_text(json.dumps(stance), encoding="utf-8")
    rejected = evaluate_candidates(tasks=[task], preflight_manifest={"tasks": [report]})
    assert rejected[0]["eligible"] is False
    assert "resolved_scene_affordance_missing" in rejected[0]["rejection_blockers"]


def test_selection_is_seed_reproducible_and_immutable(tmp_path: Path) -> None:
    tasks = [_task("beta"), _task("alpha")]
    registry = _write_json(
        tmp_path / "registry.json",
        {"schema_version": "kitchen_unitree_g1_task_registry.v1", "tasks": tasks},
    )
    preflight = _write_json(
        tmp_path / "preflight.json",
        {"tasks": [_passed_report(tmp_path, "alpha"), _passed_report(tmp_path, "beta")]},
    )
    kitchen = tmp_path / "scene" / "KitchenRoom.usd"
    kitchen.parent.mkdir()
    kitchen.write_text("#usda kitchen", encoding="utf-8")
    out = tmp_path / "run"

    result = select_random_task(
        registry_path=registry,
        preflight_manifest_path=preflight,
        kitchen_usd=kitchen,
        out_dir=out,
        seed=7,
    )
    assert result["eligible_task_ids_sorted"] == ["alpha", "beta"]
    assert result["selected_task_id"] in {"alpha", "beta"}
    assert json.loads((out / "random_task_selection.json").read_text())["seed_uint64"] == 7
    with pytest.raises(FileExistsError):
        select_random_task(
            registry_path=registry,
            preflight_manifest_path=preflight,
            kitchen_usd=kitchen,
            out_dir=out,
            seed=7,
        )

    invalidation = _write_json(
        tmp_path / "invalidate_alpha.json",
        {
            "schema_version": "kitchen_selected_task_invalidation.v1",
            "status": "invalidated",
            "selected_task_id": "alpha",
            "task_selection_invalidated": True,
            "reroll_permitted": True,
        },
    )
    reroll = select_random_task(
        registry_path=registry,
        preflight_manifest_path=preflight,
        kitchen_usd=kitchen,
        out_dir=out,
        seed=9,
        invalidation_paths=[invalidation],
        selection_artifact_name="random_task_selection_reroll_001.json",
        specification_artifact_name="selected_task_specification_reroll_001.json",
        inventory_artifact_name="kitchen_asset_inventory_reroll_001.json",
    )
    assert reroll["selected_task_id"] == "beta"
    alpha = next(row for row in reroll["candidates"] if row["task_id"] == "alpha")
    assert alpha["eligible_before_live_invalidation"] is True
    assert alpha["eligible"] is False
    assert "invalidated_by_fresh_live_provider_scene_evidence" in alpha[
        "rejection_blockers"
    ]
    assert reroll["live_provider_invalidations"][0]["source_sha256"]

    launch_inputs = materialize_selected_task_inputs(
        selection_path=out / "random_task_selection.json",
        out_dir=out,
    )
    route = json.loads(Path(launch_inputs["route_path"]).read_text())
    assert route["route_points"][0] == route["route_points"][1]
    completion = json.loads(Path(launch_inputs["task_success_contract_path"]).read_text())
    assert completion["registered_criteria"][0]["criterion_id"].endswith("_angle")

    isaac = materialize_selected_isaac_scenario(
        selection_path=out / "random_task_selection.json",
        out_dir=out,
        scenario_eval_run_id="fresh-run-7",
    )
    request = json.loads(Path(isaac["scenario_path"]).read_text())
    scenario = request["scenarios"][0]
    assert scenario["scenario_eval_run_id"] == "fresh-run-7"
    assert scenario["task_target_deferred"] is True
    report_standoff = scenario["preferred_stance_distance_m"]
    assert scenario["stance_distance_candidates_m"] == [0.16, 0.24]
    assert report_standoff > 0
    assert scenario["accepted_stance_contract"]["pose_xyz"] == [1.0, 2.0, 0.84]
    assert scenario["accepted_stance_contract"]["resolved_affordance"][
        "target_object_id"
    ].endswith("_handle")
    assert scenario["accepted_stance_contract"]["provider_revalidation_required"] is True
    assert scenario["task_success_contract"]["registered_criteria"][0][
        "criterion_id"
    ].endswith("_angle")
    with pytest.raises(FileExistsError):
        materialize_selected_isaac_scenario(
            selection_path=out / "random_task_selection.json",
            out_dir=out,
            scenario_eval_run_id="fresh-run-7",
        )


def test_candidate_rejects_missing_collision_evidence(tmp_path: Path) -> None:
    task = _task("faucet")
    report = _passed_report(tmp_path, "faucet")
    Path(report["artifacts"]["placement_validation"]).write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    stance_path = Path(report["artifacts"]["task_stance_plan"])
    stance = json.loads(stance_path.read_text())
    stance["candidates"] = []
    stance_path.write_text(json.dumps(stance), encoding="utf-8")
    candidate = evaluate_candidates(tasks=[task], preflight_manifest={"tasks": [report]})[0]
    assert candidate["eligible"] is False
    assert "collision_clearance_not_proven" in candidate["rejection_blockers"]
