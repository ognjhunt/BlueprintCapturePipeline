from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from blueprint_pipeline.composed_paired_entity_placement import (
    plan_composed_paired_entity_placement,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "docs/arm_decision_proof_v1/deformable_scene"
MANIFEST_PATH = EVIDENCE / "840873_composed_paired_entity_placement_manifest.v1.json"
CANONICAL_RECEIPT_PATH = (
    EVIDENCE / "840873_canonical_composed_paired_entity_placement_receipt.v1.json"
)
HELD_RECEIPT_PATH = (
    EVIDENCE
    / "840873_held_out_composed_relocation_paired_entity_placement_receipt.v1.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _planner_inputs_from_manifest(
    manifest: dict[str, Any],
    *,
    cell_id: str,
    canonical_centers: list[list[float]],
) -> dict[str, Any]:
    inventory = manifest["collision_inventory"]
    supports = [
        row["planner_region"]
        for row in inventory["supports"]
        if cell_id in row["cells"]
    ]
    obstacles = [
        row["planner_obstacle"]
        for row in inventory["obstacles"]
        if cell_id in row["cells"]
    ]
    entities = [
        {
            "entity_id": row["entity_id"],
            "footprint_xy_m": row["planner_geometry"]["footprint_xy_m"],
            "height_m": row["planner_geometry"]["height_m"],
        }
        for row in manifest["task_entity_asset_bindings"]
    ]
    parameters = manifest["planner_parameters"]
    separations = dict(parameters["minimum_separations_m"])
    cell = next(row for row in manifest["cells"] if row["cell_id"] == cell_id)
    separations["canonical_region"] = (
        parameters["held_out_composed_relocation_minimum_separation_m"]
        if cell["scenario_family"] == "held_out_composed_relocation"
        else parameters["canonical_minimum_separation_m"]
    )
    robot = manifest["robot_geometry_contract"]
    return {
        "support_regions": supports,
        "obstacle_aabbs": obstacles,
        "entity_specs": entities,
        "canonical_task_centers_m": canonical_centers,
        "robot_spec": {
            "base_footprint_xy_m": robot["base_footprint_xy_m"],
            "base_clearance_height_m": robot["base_clearance_height_m"],
            "reach_annulus_m": robot["reach_annulus_m"],
        },
        "minimum_separations_m": separations,
        "grid_spacing_m": parameters["grid_spacing_m"],
        "frozen_seed": cell["frozen_seed"],
    }


def _aabb_distance(
    first: dict[str, Any], second: dict[str, Any], *, dimensions: int = 3
) -> float:
    return math.sqrt(
        sum(
            max(
                second["aabb_min_m"][axis] - first["aabb_max_m"][axis],
                first["aabb_min_m"][axis] - second["aabb_max_m"][axis],
                0.0,
            )
            ** 2
            for axis in range(dimensions)
        )
    )


def _point_aabb_distance_xy(point: list[float], box: dict[str, Any]) -> float:
    return math.sqrt(
        sum(
            max(
                box["aabb_min_m"][axis] - point[axis],
                point[axis] - box["aabb_max_m"][axis],
                0.0,
            )
            ** 2
            for axis in range(2)
        )
    )


def _cell(manifest: dict[str, Any], family: str) -> dict[str, Any]:
    return next(row for row in manifest["cells"] if row["scenario_family"] == family)


def test_checked_receipts_recompute_from_manifest_without_outcomes() -> None:
    manifest = _read(MANIFEST_PATH)
    canonical = _read(CANONICAL_RECEIPT_PATH)
    held = _read(HELD_RECEIPT_PATH)

    assert manifest["manifest_digest"] == canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    assert canonical["receipt_digest"] == canonical_digest(
        canonical, digest_field="receipt_digest"
    )
    assert held["receipt_digest"] == canonical_digest(
        held, digest_field="receipt_digest"
    )
    assert manifest["outcome_blind_resolution"] == {
        "frozen_seed": 2026081001,
        "learned_policy_outcomes_inspected": False,
        "manual_post_outcome_repositioning_allowed": False,
        "policy_outcomes_used_for_selection": False,
        "protocol_amendment_path": (
            "docs/arm_decision_proof_v1/deformable_scene/"
            "scene_composed_relocation_protocol_amendment.v1.json"
        ),
        "protocol_amendment_sha256": (
            "sha256:ad20a16879729b0666b6b991118626a7285691d1220aef5e48770dbd8d5639bd"
        ),
        "recorded_before_learned_policy_outcome_inspection": True,
        "runtime_resampling_allowed": False,
    }

    source_centers = [
        row["center_world_m"] for row in manifest["source_originals_retained"]
    ]
    canonical_cell = _cell(manifest, "canonical")
    held_cell = _cell(manifest, "held_out_composed_relocation")
    recomputed_canonical = plan_composed_paired_entity_placement(
        **_planner_inputs_from_manifest(
            manifest,
            cell_id=canonical_cell["cell_id"],
            canonical_centers=source_centers,
        )
    )
    canonical_selected_centers = [
        row["center_world_m"]
        for row in recomputed_canonical["selection"]["entity_placements"]
    ]
    recomputed_held = plan_composed_paired_entity_placement(
        **_planner_inputs_from_manifest(
            manifest,
            cell_id=held_cell["cell_id"],
            canonical_centers=canonical_selected_centers + source_centers,
        )
    )

    assert recomputed_canonical == canonical
    assert recomputed_held == held
    assert canonical_cell["planner_receipt_digest"] == canonical["receipt_digest"]
    assert held_cell["planner_receipt_digest"] == held["receipt_digest"]
    assert canonical["selection"]["admissible_candidate_count"] == 145
    assert held["selection"]["admissible_candidate_count"] == 18


def test_cells_reuse_exact_entity_and_asset_bindings_and_only_change_positions() -> None:
    manifest = _read(MANIFEST_PATH)
    assert len(manifest["cells"]) == 2
    assert sorted(row["scenario_family"] for row in manifest["cells"]) == [
        "canonical",
        "held_out_composed_relocation",
    ]

    assets = manifest["task_entity_asset_bindings"]
    assert {row["semantic_role"] for row in assets} == {
        "destination_receptacle",
        "movable_deformable",
    }
    assert all(row["shared_across_cells"] is True for row in assets)
    assert all(
        row["asset_binding_status"]
        == "design_basis_bound_runtime_asset_pending"
        for row in assets
    )
    assert all(row["runtime_asset"]["relative_path"] is None for row in assets)
    assert all(row["runtime_asset"]["digest"] is None for row in assets)
    assert all(
        row["runtime_asset"]["native_simulator_qualified"] is False
        for row in assets
    )

    canonical = {
        row["entity_id"]: row
        for row in _cell(manifest, "canonical")["task_entity_pose_bindings"]
    }
    held = {
        row["entity_id"]: row
        for row in _cell(manifest, "held_out_composed_relocation")[
            "task_entity_pose_bindings"
        ]
    }
    assert canonical.keys() == held.keys()
    for entity_id in canonical:
        assert canonical[entity_id]["position_world_m"] != held[entity_id][
            "position_world_m"
        ]
        assert {
            key: value
            for key, value in canonical[entity_id].items()
            if key != "position_world_m"
        } == {
            key: value
            for key, value in held[entity_id].items()
            if key != "position_world_m"
        }

    assert all(
        row["appearance_gaussians_removed"] is False
        and row["source_collider_removed"] is False
        and row["inpainting_applied"] is False
        and row["task_scoring_role"] == "unscored_background"
        for row in manifest["source_originals_retained"]
    )
    assert manifest["excision_and_inpainting"] == {
        "collider_removal_required": False,
        "gaussian_excision_required": False,
        "inpainting_required": False,
        "reason": (
            "source_originals_remain_unscored_background_and_both_task_entities_"
            "are_separate_inserted_twins"
        ),
    }


def test_resolved_cells_clear_registered_obstacles_support_edges_and_each_other() -> None:
    manifest = _read(MANIFEST_PATH)
    canonical = _read(CANONICAL_RECEIPT_PATH)
    held = _read(HELD_RECEIPT_PATH)
    required = manifest["planner_parameters"]["minimum_separations_m"]

    for receipt in (canonical, held):
        selection = receipt["selection"]
        assert all(selection["geometry_checks"].values())
        entities = selection["entity_placements"]
        robot = selection["robot_base_placement"]
        obstacles = receipt["request"]["obstacle_aabbs"]
        supports = {
            row["support_region_id"]: row
            for row in receipt["request"]["support_regions"]
        }
        assert _aabb_distance(entities[0], entities[1]) >= required[
            "entity_entity"
        ] - 1.0e-9
        assert all(
            _aabb_distance(entity, obstacle) >= required["entity_obstacle"] - 1.0e-9
            for entity in entities
            for obstacle in obstacles
        )
        assert all(
            _aabb_distance(robot, entity) >= required["robot_entity"] - 1.0e-9
            for entity in entities
        )
        assert all(
            _aabb_distance(robot, obstacle) >= required["robot_obstacle"] - 1.0e-9
            for obstacle in obstacles
        )
        for placement in [*entities, robot]:
            support = supports[placement["support_region_id"]]
            assert placement["aabb_min_m"][0] >= (
                support["aabb_min_m"][0] + required["support_edge"] - 1.0e-9
            )
            assert placement["aabb_max_m"][0] <= (
                support["aabb_max_m"][0] - required["support_edge"] + 1.0e-9
            )
            assert placement["aabb_min_m"][1] >= (
                support["aabb_min_m"][1] + required["support_edge"] - 1.0e-9
            )
            assert placement["aabb_max_m"][1] <= (
                support["aabb_max_m"][1] - required["support_edge"] + 1.0e-9
            )

    inter_cell_distance = min(
        _aabb_distance(first, second, dimensions=2)
        for first in canonical["selection"]["entity_placements"]
        for second in held["selection"]["entity_placements"]
    )
    assert round(inter_cell_distance, 9) == 4.776490474
    assert inter_cell_distance >= 1.5
    assert manifest["cross_cell_checks"][
        "minimum_resolved_task_aabb_separation_xy_m"
    ] == round(inter_cell_distance, 9)
    assert all(
        _point_aabb_distance_xy(point, placement) >= 1.5 - 1.0e-9
        for point in held["request"]["canonical_task_centers_m"]
        for placement in held["selection"]["entity_placements"]
    )

    support_exception = manifest["collision_inventory"]["support_contact_exception"]
    assert support_exception["passed"] is True
    assert support_exception[
        "canonical_robot_base_to_table_support_component_clearance_m"
    ] == 0.064922031
    assert support_exception[
        "canonical_robot_base_to_table_support_component_clearance_m"
    ] >= support_exception["required_robot_obstacle_clearance_m"]


def test_every_planner_geometry_row_has_digest_bound_source_derivation() -> None:
    manifest = _read(MANIFEST_PATH)
    inventory = manifest["collision_inventory"]
    assert inventory["stage_active_nonempty_mesh_count"] == 95
    assert inventory["stage_connected_component_count"] == 11594
    assert inventory["floorplan_connected_component_count"] == 3766
    assert inventory["non_floorplan_exact_query_hit_count"] == 25
    assert inventory["query_hit_classification"] == {
        "admitted_table_support_component_count": 1,
        "planner_obstacle_component_count": 24,
    }

    for receipt_path, family in (
        (CANONICAL_RECEIPT_PATH, "canonical"),
        (HELD_RECEIPT_PATH, "held_out_composed_relocation"),
    ):
        receipt = _read(receipt_path)
        cell_id = _cell(manifest, family)["cell_id"]
        support_evidence = [
            row for row in inventory["supports"] if cell_id in row["cells"]
        ]
        obstacle_evidence = [
            row for row in inventory["obstacles"] if cell_id in row["cells"]
        ]
        assert sorted(
            (row["planner_region"] for row in support_evidence), key=str
        ) == sorted(receipt["request"]["support_regions"], key=str)
        assert sorted(
            (row["planner_obstacle"] for row in obstacle_evidence), key=str
        ) == sorted(receipt["request"]["obstacle_aabbs"], key=str)

        for row in support_evidence:
            derivation = row["derivation"]
            if derivation["source_kind"] == "sage_connected_component_top_plane":
                assert derivation["component_geometry_digest"].startswith("sha256:")
                assert derivation["collision_api_applied"] is True
            else:
                assert derivation["source_structure_sha256"].startswith("sha256:")
                assert derivation["sage_floor_mesh_geometry_digest"].startswith(
                    "sha256:"
                )
                assert derivation["all_samples_inside_exact_room_polygon"] is True
                assert derivation["all_samples_hit_exact_sage_floor_at_z0"] is True
                assert len(derivation["corner_and_center_samples"]) == 5

        for row in obstacle_evidence:
            derivation = row["derivation"]
            if derivation["source_kind"] == "sage_connected_component":
                assert derivation["geometry_digest"].startswith("sha256:")
                assert derivation["collision_api_applied"] is True
            else:
                assert derivation["structure_sha256"].startswith("sha256:")
                assert derivation["claim"] == (
                    "conservative_registered_wall_box_not_exact_mesh_surface"
                )

    assert manifest["claim_boundary"]["geometry_plausibility_only"] is True
    assert manifest["claim_boundary"]["runtime_assets_authored"] is False
    assert manifest["claim_boundary"]["native_ik_qualified"] is False
    assert manifest["claim_boundary"]["native_contact_qualified"] is False
    assert manifest["claim_boundary"]["native_camera_visibility_qualified"] is False
    assert "native_full_phase_ik" in manifest["pending_native_gates"]
