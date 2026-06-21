from __future__ import annotations

import json
from pathlib import Path

import pytest

import blueprint_pipeline.mujoco_scene_scenario_packet as packet
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
    assert len({run["scenario_eval_run_id"] for run in matrix["runs"]}) == (
        DEFAULT_SCENARIO_COUNT
    )
    assert len({run["episode_seed"] for run in matrix["runs"]}) == DEFAULT_SCENARIO_COUNT
    assert all(run["spawn_pose"]["xyz"] for run in matrix["runs"])
    assert all(run["target_pose"]["xyz"] for run in matrix["runs"])
    assert all(run["scenario_variation_instance_id"] for run in matrix["runs"])
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


def test_packet_helper_edges(tmp_path: Path) -> None:
    assert packet._repo_root().name == "BlueprintCapturePipeline"
    scene = {
        "asset_id": "asset",
        "scene_id": "scene",
        "site_type": "warehouse",
        "source_url": "https://example.com",
        "license_id": "MIT",
        "tasks": ["bad", {"task_id": "task-1", "task_statement": "Inspect shelf"}],
        "scenario_families": ["bad", {"scenario_id": "scenario-1", "task_id": "task-1", "family_label": "A"}],
    }
    assert packet._task_cards(scene)["count"] == 1
    assert packet._scenario_cards(scene)["count"] == 1
    assert packet._scenario_family_library(scene, generated_at="now")["family_count"] == 1
    assert packet._asset_inventory(None)["status"] == "not_inspected_no_local_asset_root"
    assert packet._asset_inventory(tmp_path / "missing")["status"] == "blocked_missing_local_asset_root"

    bad_world = tmp_path / "bad.world"
    bad_world.write_text("<sdf>", encoding="utf-8")
    assert packet._world_included_models(bad_world) == []
    mixed_world = tmp_path / "mixed.world"
    mixed_world.write_text(
        """<sdf><world>
        <model name="without_include" />
        <model name="bad_pose"><include><uri>model://box</uri></include><pose>1 bad 3</pose></model>
        </world></sdf>""",
        encoding="utf-8",
    )
    assert packet._world_included_models(mixed_world)[0]["pose"] == [1.0, 0.0, 3.0]

    worlds = tmp_path / "worlds"
    worlds.mkdir()
    fallback_world = worlds / "z.world"
    fallback_world.write_text("<sdf />", encoding="utf-8")
    assert packet._preferred_aws_world_file(tmp_path) == fallback_world

    assert packet._parse_pose_values([1, "bad"]) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert packet._parse_pose_values("1 bad") == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert packet._parse_scale_values("2 bad") == [2.0, 1.0, 1.0]
    assert packet._model_name_from_uri("model://box/meshes/mesh.dae") == "box"
    assert packet._resolve_model_uri(tmp_path, "model://") is None
    assert packet._resolve_model_uri(tmp_path, "meshes/a.dae") == tmp_path / "meshes" / "a.dae"
    assert packet._sdf_visual_meshes(tmp_path, "missing") == []

    malformed_model = tmp_path / "models" / "bad"
    malformed_model.mkdir(parents=True)
    (malformed_model / "model.sdf").write_text("<sdf>", encoding="utf-8")
    assert packet._sdf_visual_meshes(tmp_path, "bad") == []
    no_mesh_model = tmp_path / "models" / "no_mesh"
    no_mesh_model.mkdir(parents=True)
    (no_mesh_model / "model.sdf").write_text(
        "<sdf><model><link><visual><geometry><box /></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    assert packet._sdf_visual_meshes(tmp_path, "no_mesh") == []
    empty_uri_model = tmp_path / "models" / "empty_uri"
    empty_uri_model.mkdir(parents=True)
    (empty_uri_model / "model.sdf").write_text(
        "<sdf><model><link><visual><geometry><mesh><uri>model://</uri></mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    assert packet._sdf_visual_meshes(tmp_path, "empty_uri") == []

    assert packet._collada_unit_scale(type("Loaded", (), {"units": "0.01 meter"})()) == 0.01
    assert packet._collada_unit_scale(type("Loaded", (), {"units": "centimeter"})()) == 0.01
    assert packet._collada_unit_scale(type("Loaded", (), {"units": "meters"})()) == 1.0
    assert packet._collada_unit_scale(type("Loaded", (), {"units": "unknown"})()) == 1.0
    assert packet._scale_matrix([2.0]).shape == (4, 4)

    class BytesLikeScene:
        def export(self, *, file_type: str) -> bytearray:
            assert file_type == "glb"
            return bytearray(b"glb")

    output = tmp_path / "scene.glb"
    packet._export_scene_glb(BytesLikeScene(), output)
    assert output.read_bytes() == b"glb"


def test_materialization_guard_edges_and_blocker_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    assert packet._materialize_aws_scene_asset(
        local_asset_root=None,
        capture_root=tmp_path / "capture-none",
        packet_dir=packet_dir,
        generated_at="now",
    )["status"] == "skipped_no_local_asset_root"
    assert packet._materialize_aws_scene_asset(
        local_asset_root=tmp_path / "missing-root",
        capture_root=tmp_path / "capture-missing",
        packet_dir=packet_dir,
        generated_at="now",
    )["status"] == "blocked_missing_local_asset_root"

    no_world_root = tmp_path / "no-world"
    no_world_root.mkdir()
    assert packet._materialize_aws_scene_asset(
        local_asset_root=no_world_root,
        capture_root=tmp_path / "capture-no-world",
        packet_dir=packet_dir,
        generated_at="now",
    )["status"] == "blocked_missing_world_file"

    import trimesh

    asset_root = tmp_path / "asset-root"
    (asset_root / "worlds").mkdir(parents=True)
    (asset_root / "worlds" / "small_warehouse.world").write_text(
        """<sdf><world>
        <model name="missing_uri"><include><uri></uri></include></model>
        <model name="missing_dir"><include><uri>model://missing_dir</uri></include></model>
        <model name="no_visual"><include><uri>model://no_visual</uri></include></model>
        <model name="missing_mesh"><include><uri>model://missing_mesh</uri></include></model>
        <model name="bad_load"><include><uri>model://bad_load</uri></include></model>
        <model name="empty_scene"><include><uri>model://empty_scene</uri></include></model>
        <model name="box_model"><include><uri>model://box_model</uri></include></model>
        </world></sdf>""",
        encoding="utf-8",
    )
    for model_name, mesh_name in {
        "no_visual": None,
        "missing_mesh": "missing.dae",
        "bad_load": "bad_load.dae",
        "empty_scene": "empty_scene.dae",
        "box_model": "box_model.dae",
    }.items():
        model_root = asset_root / "models" / model_name
        (model_root / "meshes").mkdir(parents=True)
        if mesh_name is None:
            (model_root / "model.sdf").write_text("<sdf><model /></sdf>", encoding="utf-8")
            continue
        (model_root / "model.sdf").write_text(
            f"""<sdf><model><link><visual name="{model_name}_visual">
            <geometry><mesh><uri>model://{model_name}/meshes/{mesh_name}</uri><scale>1 1 1</scale></mesh></geometry>
            </visual></link></model></sdf>""",
            encoding="utf-8",
        )
        if model_name != "missing_mesh":
            (model_root / "meshes" / mesh_name).write_text("mesh", encoding="utf-8")

    def fake_load(path: Path, *, force: str) -> object:
        text = str(path)
        if "bad_load" in text:
            raise RuntimeError("bad mesh")
        if "empty_scene" in text:
            return trimesh.Scene()
        return trimesh.creation.box(extents=(1, 1, 1))

    monkeypatch.setattr(trimesh, "load", fake_load)
    manifest = packet._materialize_aws_scene_asset(
        local_asset_root=asset_root,
        capture_root=tmp_path / "capture-blockers",
        packet_dir=packet_dir,
        generated_at="now",
    )

    assert manifest["status"] == "completed_with_warnings"
    blockers = "\n".join(manifest["blockers"])
    assert "missing_model_uri" in blockers
    assert "missing_model_directory" in blockers
    assert "no_visual_meshes_in_model_sdf" in blockers
    assert "missing_mesh" in blockers
    assert "load_failed:RuntimeError" in blockers
    assert "no_geometry_nodes" in blockers
    assert manifest["materialized_geometry_count"] > 0


def test_run_generation_edges_and_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert packet._pose_with_jitter({"zone_id": "z", "xyz": [1.0, 2.0]}, seed=1)["xyz"][2] == 0.793
    assert packet._runs_for_scene(scene={"asset_id": "a", "scenario_families": []}, scenario_count=3, variation_manifest={}) == []
    scene = {
        "asset_id": "asset",
        "spawn_zones": [],
        "target_zones": [],
        "scenario_families": [{"scenario_id": "scenario", "task_id": "task", "spawn_zones": ["missing"], "target_zones": ["missing"]}],
    }
    assert packet._runs_for_scene(
        scene=scene,
        scenario_count=3,
        variation_manifest={"instances": ["bad", {"scenario_id": "scenario", "variation_name": "v"}]},
    ) == []

    with pytest.raises(ValueError, match="unsupported scene_asset_id"):
        build_mujoco_scene_scenario_packet(output_dir=tmp_path, scene_asset_id="other")

    capture_root = tmp_path / "bucket" / "scenes" / "scene-cli" / "captures" / "capture-cli"
    result = build_mujoco_scene_scenario_packet(
        capture_root=capture_root,
        scenario_count=1,
        generated_at="now",
    )
    assert result["scenario_eval_run_count"] == 1

    monkeypatch.setattr(
        packet,
        "build_mujoco_scene_scenario_packet",
        lambda **_kwargs: {
            "status": "ok",
            "capture_root": "capture",
            "packet_path": "packet.json",
            "scenario_eval_matrix_path": "matrix.json",
            "scenario_eval_run_count": 7,
            "local_asset_inventory_status": "not_inspected",
            "scene_materialization_status": "skipped",
            "scene_glb_path": None,
        },
    )
    assert packet.main(["--scenario-count", "7"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["scenario_eval_run_count"] == 7
