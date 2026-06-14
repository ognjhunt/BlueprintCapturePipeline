from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.mujoco_scene_scenario_packet import (
    DEFAULT_SCENARIO_COUNT,
    build_mujoco_scene_asset_research,
    build_mujoco_scene_scenario_packet,
    _materialize_aws_scene_asset,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scene_asset_research_ranks_fail_closed_candidates(tmp_path: Path) -> None:
    manifest = build_mujoco_scene_asset_research(
        output_dir=tmp_path,
        generated_at="2026-06-14T00:00:00+00:00",
    )

    assert manifest["schema_version"] == "mujoco_scene_asset_research.v1"
    assert manifest["candidate_count"] >= 10
    assert manifest["recommended_first_scene"] == "aws_robomaker_small_warehouse_world"
    assert manifest["top_3_immediate_mujoco"] == [
        "aws_robomaker_small_warehouse_world",
        "manchester_nuclear_gazebo_assets",
        "aws_robomaker_bookstore_world",
    ]
    options = {item["asset_id"]: item for item in manifest["options"]}
    assert options["aws_robomaker_small_warehouse_world"]["license"]["id"] == "MIT-0"
    assert options["hssd"]["avoid_or_research_only"] is True
    assert "hssd" in manifest["avoid_or_research_only"]
    assert (tmp_path / "mujoco_scene_asset_research.json").is_file()


def test_build_mujoco_scene_packet_writes_warehouse_tasks_matrix_and_recording_plan(
    tmp_path: Path,
) -> None:
    local_asset_root = tmp_path / "aws-warehouse"
    (local_asset_root / "worlds").mkdir(parents=True)
    (local_asset_root / "models" / "shelf" / "meshes").mkdir(parents=True)
    (local_asset_root / "models" / "shelf" / "materials" / "textures").mkdir(parents=True)
    (local_asset_root / "worlds" / "small_warehouse.world").write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<sdf version="1.6">
  <world name="default">
    <model name="shelf_1">
      <include><uri>model://shelf</uri></include>
      <pose>1 2 0 0 0 0</pose>
    </model>
  </world>
</sdf>
""",
        encoding="utf-8",
    )
    (local_asset_root / "models" / "shelf" / "model.sdf").write_text("<sdf />", encoding="utf-8")
    (local_asset_root / "models" / "shelf" / "meshes" / "shelf_visual.DAE").write_text(
        "<COLLADA />",
        encoding="utf-8",
    )
    (local_asset_root / "models" / "shelf" / "meshes" / "shelf_collision.DAE").write_text(
        "<COLLADA />",
        encoding="utf-8",
    )
    (local_asset_root / "models" / "shelf" / "materials" / "textures" / "shelf.png").write_bytes(
        b"png"
    )

    result = build_mujoco_scene_scenario_packet(
        output_dir=tmp_path / "packet",
        local_asset_root=local_asset_root,
        generated_at="2026-06-14T00:00:00+00:00",
    )

    capture_root = Path(str(result["capture_root"]))
    automation_dir = capture_root / "pipeline" / "simulation_automation"
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    packet = _read_json(automation_dir / "mujoco_scene_scenario_packet.json")
    task_cards = _read_json(robot_eval_dir / "task_cards.json")
    scenario_cards = _read_json(robot_eval_dir / "scenario_cards.json")
    family_library = _read_json(robot_eval_dir / "scenario_family_library.json")
    recording_plan = _read_json(automation_dir / "recording_plan.json")
    inventory = _read_json(automation_dir / "external_scene_asset_inventory.json")
    materialization = _read_json(automation_dir / "external_scene_materialization_manifest.json")
    variation_instances = _read_json(automation_dir / "scenario_variation_instances.json")
    matrix = _read_json(automation_dir / "scenario_eval_matrix.aws_small_warehouse_500.json")

    assert result["status"] == "ready_for_mujoco_conversion_and_recorded_eval"
    assert result["scene_materialization_status"] == "blocked_no_visual_meshes_materialized"
    assert result["scenario_eval_run_count"] == DEFAULT_SCENARIO_COUNT
    assert packet["scene_asset"]["asset_id"] == "aws_robomaker_small_warehouse_world"
    assert packet["task_count"] == 10
    assert packet["scenario_family_count"] == 10
    assert packet["external_scene_asset_not_raw_capture"] is True
    assert packet["claim_boundary"]["simulator_execution_proven"] is False
    assert task_cards["count"] == 10
    assert scenario_cards["count"] == 10
    assert family_library["family_count"] == 10
    assert recording_plan["required_recording_views"] == [
        "sim_robot_follow_pov",
        "overview",
        "side",
    ]
    assert inventory["status"] == "inspected_local_asset_root"
    assert inventory["visual_dae_mesh_count"] == 1
    assert inventory["collision_dae_mesh_count"] == 1
    assert inventory["texture_file_count"] == 1
    assert inventory["included_model_count"] == 1
    assert materialization["status"] == "blocked_no_visual_meshes_materialized"
    assert materialization["conversion_performed"] is False
    assert packet["scene_glb_available_for_mujoco_command"] is False
    assert variation_instances["family_count"] == 10
    assert variation_instances["instance_count"] == 10 * len(
        variation_instances["required_variation_names"]
    )
    assert matrix["status"] == "completed"
    assert matrix["scenario_eval_run_count"] == DEFAULT_SCENARIO_COUNT
    assert matrix["base_scenario_family_count"] == 10
    assert matrix["episode_authoring_contract"]["ai_or_api_proposal_allowed_upstream"] is True
    assert matrix["episode_authoring_contract"]["runtime_ai_route_selection_allowed"] is False
    assert matrix["episode_authoring_contract"]["runtime_ai_route_selection_used"] is False
    assert matrix["required_recording_views"] == [
        "sim_robot_follow_pov",
        "overview",
        "side",
    ]
    assert len(matrix["runs"]) == DEFAULT_SCENARIO_COUNT
    assert all("side" in run["recording_views_required"] for run in matrix["runs"])
    assert all(
        run["episode_authoring"]["runtime_ai_route_selection_allowed"] is False
        for run in matrix["runs"]
    )
    assert {
        run["task_id"] for run in matrix["runs"] if "pallet" in run["task_id"]
    } == {"verify_pallet_jack_clearance"}
    assert (automation_dir / "mujoco_scene_packet_runbook.md").is_file()


def test_materialize_aws_scene_asset_writes_mujoco_consumable_glb(tmp_path: Path) -> None:
    import trimesh

    local_asset_root = tmp_path / "aws-warehouse"
    world_dir = local_asset_root / "worlds"
    model_dir = local_asset_root / "models" / "box_model"
    mesh_dir = model_dir / "meshes"
    world_dir.mkdir(parents=True)
    mesh_dir.mkdir(parents=True)
    (world_dir / "no_roof_small_warehouse.world").write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<sdf version="1.6">
  <world name="default">
    <model name="box_model_001">
      <include><uri>model://box_model</uri></include>
      <pose>1 2 0 0 0 0.25</pose>
    </model>
  </world>
</sdf>
""",
        encoding="utf-8",
    )
    (model_dir / "model.sdf").write_text(
        """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='box_model'>
    <link name='body'>
      <visual name='visual'>
        <geometry>
          <mesh>
            <uri>model://box_model/meshes/box_model_visual.DAE</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
""",
        encoding="utf-8",
    )
    mesh = trimesh.creation.box(extents=(100, 100, 100))
    mesh.visual.vertex_colors = [[180, 90, 40, 255] for _ in range(len(mesh.vertices))]
    mesh.export(mesh_dir / "box_model_visual.DAE")

    capture_root = tmp_path / "capture"
    packet_dir = capture_root / "pipeline" / "simulation_automation"
    packet_dir.mkdir(parents=True)

    manifest = _materialize_aws_scene_asset(
        local_asset_root=local_asset_root,
        capture_root=capture_root,
        packet_dir=packet_dir,
        generated_at="2026-06-14T00:00:00+00:00",
    )

    glb_path = capture_root / "pipeline" / "worldlabs_assets" / "scene.glb"
    materialization_path = packet_dir / "external_scene_materialization_manifest.json"

    assert manifest["status"] == "completed"
    assert manifest["conversion_performed"] is True
    assert manifest["world_file"] == "worlds/no_roof_small_warehouse.world"
    assert manifest["materialized_geometry_count"] > 0
    assert manifest["vertex_count"] > 0
    assert glb_path.is_file()
    assert materialization_path.is_file()
