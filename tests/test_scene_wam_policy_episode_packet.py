from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline import scene_wam_policy_episode_packet as packet


def _capture_root(tmp_path: Path) -> Path:
    root = (
        tmp_path
        / "storage"
        / "local-blueprint"
        / "scenes"
        / "kitchen"
        / "captures"
        / "capture-001"
    )
    (root / "pipeline" / "evaluation_prep").mkdir(parents=True)
    (root / "pipeline" / "evaluation_prep" / "task_anchor_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "task_anchor_manifest.v1",
                "tasks": [
                    {
                        "task_id": "turn_on_sink_handle",
                        "target_object_ids": ["sink_handle"],
                        "anchor_accepted": True,
                        "goal_zone": {"xyz": [1.0, 2.0, 1.0]},
                        "start_zone": {"xyz": [1.0, 1.0, 0.0]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _scene_asset(tmp_path: Path) -> Path:
    scene = tmp_path / "KitchenRoom.usda"
    scene.write_text(
        """
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World" {
    def Cube "SinkHandle" {
        double size = 0.2
        double3 xformOp:translate = (1, 2, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
""".strip(),
        encoding="utf-8",
    )
    return scene


def _write_lightwheel_scenario_specs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "lightwheel_kitchen_g1_scenario_specs.v1",
                "scenario_execution_status": "not_executed",
                "proof_boundary": {
                    "scenario_specs_authored": True,
                    "unitree_g1_spawned_in_lightwheel_kitchen": False,
                    "isaac_sim_execution_proven": False,
                },
                "scenarios": [
                    {
                        "scenario_id": "lightwheel_kitchen_g1_05_narrow_passage_to_sink",
                        "description": "Thread a narrow passage toward the sink.",
                        "scenario_status": "specified_not_executed",
                        "execution_proven": False,
                        "robot_profile_id": "unitree_g1",
                        "spawn_position_xyz": [0.0, -3.75, 0.05],
                        "target_position_xyz": [2.35, 1.25, 0.05],
                    },
                    {
                        "scenario_id": "lightwheel_kitchen_g1_01_entry_to_sink",
                        "description": "Navigate from the open entry side to the sink work area.",
                        "scenario_status": "specified_not_executed",
                        "execution_proven": False,
                        "robot_profile_id": "unitree_g1",
                        "spawn_position_xyz": [-4.25, -3.35, 0.05],
                        "target_position_xyz": [2.2, 0.9, 0.05],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _scene_with_counter_obstacle(tmp_path: Path) -> Path:
    scene = tmp_path / "KitchenRoom.usda"
    scene.write_text(
        """
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World" {
    def Cube "CounterObstacle" {
        double size = 1.0
        double3 xformOp:scale = (1.0, 1.0, 0.8)
        double3 xformOp:translate = (0, 0, 0.8)
        uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:translate"]
    }
    def Cube "SinkHandle" {
        double size = 0.1
        double3 xformOp:translate = (0, 0, 1.2)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
""".strip(),
        encoding="utf-8",
    )
    return scene


def _scene_with_sink_front_panel(tmp_path: Path) -> Path:
    scene = tmp_path / "KitchenRoomWithSinkFront.usda"
    scene.write_text(
        """
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World" {
    def Cube "Kitchen_Cabinet002_Aggregate" {
        double size = 1.0
        double3 xformOp:scale = (2.0, 2.0, 0.4)
        double3 xformOp:translate = (1.0, 1.0, 0.4)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Cube "Sink054_Body001" {
        double size = 1.0
        double3 xformOp:scale = (0.25, 0.35, 0.2)
        double3 xformOp:translate = (2.3, 1.2, 0.7)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Cube "Kitchen_Cabinet002_Door007" {
        double size = 1.0
        double3 xformOp:scale = (0.02, 0.55, 0.35)
        double3 xformOp:translate = (1.98, 1.2, 0.4)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
}
""".strip(),
        encoding="utf-8",
    )
    return scene


def test_scene_wam_policy_episode_packet_blocks_without_real_renderer(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    scene = _scene_asset(tmp_path)
    monkeypatch.setattr(packet.shutil, "which", lambda _name: None)

    result = packet.build_scene_wam_policy_episode_packet(
        capture_root=capture_root,
        scene_asset=scene,
        task_id="turn_on_sink_handle",
        target_object_id="sink_handle",
        output_dir=capture_root / "pipeline" / "scene_wam_policy_episode_packet",
    )

    output_dir = Path(result["initial_policy_observation_path"]).parent
    assert result["status"] == "blocked"
    assert result["scene_physics_required_for_wam_loop"] is False
    assert result["physics_contact_validated"] is False
    assert result["physical_robot_readiness_proven"] is False
    assert result["deployment_readiness_proven"] is False
    assert result["safety_validation_proven"] is False
    assert result["real_world_manipulation_success_proven"] is False
    assert (output_dir / "scene_wam_policy_episode_packet.json").is_file()
    assert (output_dir / "initial_policy_observation.json").is_file()
    assert (output_dir / "scene_episode_task_manifest.json").is_file()
    assert (output_dir / "scene_policy_wam_claim_boundary.json").is_file()

    observation = json.loads(
        (output_dir / "initial_policy_observation.json").read_text(encoding="utf-8")
    )
    assert observation["visual_observation"]["available"] is False
    assert observation["visual_observation"]["blank_or_placeholder_image_used"] is False
    assert observation["claim_boundary"]["physical_robot_readiness_proven"] is False
    assert "initial_policy_observation_render_not_available" in result["blockers"]


def test_scene_wam_policy_episode_packet_ready_with_rendered_frame(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    scene = _scene_asset(tmp_path)

    def fake_render(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        frame = output_dir / "rendered_observations" / "initial_policy_observation.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"fake-jpeg")
        return {
            "schema_version": packet.RENDER_SCHEMA_VERSION,
            "generated_at": kwargs["generated_at"],
            "status": "completed",
            "frame_path": str(frame),
            "real_scene_observation_rendered": True,
            "blockers": [],
        }

    monkeypatch.setattr(packet, "_render_initial_observation", fake_render)
    result = packet.build_scene_wam_policy_episode_packet(
        capture_root=capture_root,
        scene_asset=scene,
        task_id="turn_on_sink_handle",
        target_object_id="sink_handle",
        target_anchor_pose="1,2,1",
        robot_start_pose="1,1,0",
        output_dir=capture_root / "pipeline" / "scene_wam_policy_episode_packet",
    )

    assert result["status"] == "ready_for_policy_wam_loop"
    assert result["physical_robot_readiness_proven"] is False
    assert result["deployment_readiness_proven"] is False
    assert result["safety_validation_proven"] is False
    assert result["real_world_manipulation_success_proven"] is False
    output_dir = Path(result["initial_policy_observation_path"]).parent
    observation = json.loads(
        (output_dir / "initial_policy_observation.json").read_text(encoding="utf-8")
    )
    assert observation["visual_observation"]["available"] is True
    assert observation["robot_profile_id"] == "unitree_g1_sonic"
    assert observation["selected_candidate_id"] == "unitree_groot_n17_sonic_policy"
    claim_boundary = json.loads(
        (output_dir / "scene_policy_wam_claim_boundary.json").read_text(encoding="utf-8")
    )
    assert claim_boundary["physical_robot_readiness_proven"] is False
    assert claim_boundary["deployment_readiness_proven"] is False
    assert claim_boundary["safety_validation_proven"] is False


def test_robot_start_pose_resolution_rejects_obstacle_overlap_and_keeps_truth_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_lightwheel_scenario_specs(
        capture_root / "pipeline" / "lightwheel_kitchen_scenarios.json"
    )
    scene = _scene_with_counter_obstacle(tmp_path)

    def fake_render(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        frame = output_dir / "rendered_observations" / "initial_policy_observation.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"fake-jpeg")
        return {
            "schema_version": packet.RENDER_SCHEMA_VERSION,
            "generated_at": kwargs["generated_at"],
            "status": "completed",
            "frame_path": str(frame),
            "real_scene_observation_rendered": True,
            "blockers": [],
        }

    monkeypatch.setattr(packet, "_render_initial_observation", fake_render)

    result = packet.build_scene_wam_policy_episode_packet(
        capture_root=capture_root,
        scene_asset=scene,
        task_id="turn_on_sink_handle",
        target_object_id="sink_handle",
        target_anchor_pose={"xyz": [0.0, 0.0, 1.2]},
        robot_start_pose={"xyz": [0.0, 0.0, 0.0]},
        output_dir=capture_root / "pipeline" / "scene_wam_policy_episode_packet",
    )

    output_dir = Path(result["initial_policy_observation_path"]).parent
    placement = json.loads(
        (output_dir / "robot_start_pose_resolution.json").read_text(encoding="utf-8")
    )
    observation = json.loads(
        (output_dir / "initial_policy_observation.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "ready_for_policy_wam_loop"
    assert result["input_robot_start_pose_rejected"] is True
    assert placement["input_robot_start_pose_rejected"] is True
    assert placement["evaluated_candidates"][0]["candidate_id"] == "provided_robot_start_pose"
    assert placement["evaluated_candidates"][0]["clearance_check"]["status"] == "failed"
    assert placement["selected_source"] == "usd_target_ring_clearance_candidate"
    assert placement["selected_clearance_check"]["status"] == "passed"
    assert placement["claim_boundary"]["static_usd_aabb_clearance_proxy_used"] is True
    assert placement["claim_boundary"]["clearance_proxy_is_not_physics_contact_proof"] is True
    assert placement["real_collision_geometry_validated"] is False
    assert placement["physics_contact_validated"] is False
    assert observation["robot_start_pose"]["xyz"] != [0.0, 0.0, 0.0]
    assert observation["robot_start_pose_resolution"]["scenario_specs"]["scenario_count"] == 2


def test_scenario_pose_candidates_prefer_open_sink_entry_over_narrow_sink(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _write_lightwheel_scenario_specs(
        capture_root / "pipeline" / "lightwheel_kitchen_scenarios.json"
    )
    scene = _scene_asset(tmp_path)

    candidates, specs = packet._scenario_pose_candidates(
        capture_root=capture_root,
        scene_asset=scene,
        task_id="turn_on_sink_handle",
        target_object_id="Sink054_handle",
        target_pose={"xyz": [2.489866261, 1.069795365, 0.886473915]},
    )

    assert specs["scenario_count"] == 2
    assert candidates[0]["scenario_id"] == "lightwheel_kitchen_g1_01_entry_to_sink"
    assert candidates[0]["pose"]["xyz"] == [-4.25, -3.35, 0.05]
    assert candidates[0]["scenario_metadata_is_execution_proof"] is False


def test_robot_start_pose_resolution_prefers_sink_front_panel_over_side_ring(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    scene = _scene_with_sink_front_panel(tmp_path)

    def fake_render(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        frame = output_dir / "rendered_observations" / "initial_policy_observation.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"fake-jpeg")
        return {
            "schema_version": packet.RENDER_SCHEMA_VERSION,
            "generated_at": kwargs["generated_at"],
            "status": "completed",
            "frame_path": str(frame),
            "real_scene_observation_rendered": True,
            "blockers": [],
        }

    monkeypatch.setattr(packet, "_render_initial_observation", fake_render)

    result = packet.build_scene_wam_policy_episode_packet(
        capture_root=capture_root,
        scene_asset=scene,
        task_id="turn_on_sink_handle",
        target_object_id="Sink054_handle",
        target_anchor_pose={"xyz": [2.49, 1.2, 0.9]},
        robot_start_pose={"xyz": [2.3, 1.2, 0.0]},
        output_dir=capture_root / "pipeline" / "scene_wam_policy_episode_packet",
    )

    output_dir = Path(result["initial_policy_observation_path"]).parent
    placement = json.loads(
        (output_dir / "robot_start_pose_resolution.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "ready_for_policy_wam_loop"
    assert placement["selected_source"] == "usd_sink_front_panel_clearance_candidate"
    assert placement["selected_pose"]["xyz"][0] < 2.0
    assert placement["selected_clearance_check"]["status"] == "passed"
    assert placement["placement_obstacle_manifest"]["broad_aggregate_aabbs_skipped"] is True
    assert (
        placement["claim_boundary"]["broad_aggregate_aabb_skip_is_not_collision_proof"]
        is True
    )
    assert placement["physics_contact_validated"] is False


def test_rendered_image_content_summary_rejects_blank_frame(tmp_path: Path) -> None:
    blank = tmp_path / "blank.jpg"
    Image.new("RGB", (64, 48), "white").save(blank)

    summary = packet._rendered_image_content_summary(blank)

    assert summary["contentful"] is False
    assert "rendered_image_blank_or_uniform" in summary["blockers"]


def test_usd_to_mujoco_visual_mjcf_exports_mesh_and_texture(tmp_path: Path) -> None:
    scene = _scene_asset(tmp_path)
    texture_dir = tmp_path / "texture"
    texture_dir.mkdir()
    Image.new("RGB", (8, 8), (200, 50, 40)).save(texture_dir / "albedo.jpg")
    scene.write_text(
        """
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World" {
    def Scope "Looks" {
        def Material "red_tile" {
            token outputs:surface.connect = </World/Looks/red_tile/PBR.outputs:surface>
            def Shader "PBR" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 1, 1)
                token outputs:surface
            }
            def Shader "Albedo" {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./texture/albedo.jpg@
                float3 outputs:rgb
            }
        }
    }
    def Mesh "TexturedQuad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    ) {
        rel material:binding = </World/Looks/red_tile>
        point3f[] points = [(-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "faceVarying"
        )
    }
}
""".strip(),
        encoding="utf-8",
    )

    result = packet._build_visual_mjcf_from_usd(
        scene_asset=scene,
        output_dir=tmp_path / "out",
        generated_at="2026-06-23T00:00:00+00:00",
        target_pose={"xyz": [0.0, 0.5, 0.0]},
        robot_pose={"xyz": [0.0, -1.0, 0.0]},
    )

    assert result["status"] == "completed"
    assert result["mesh_count"] == 1
    assert result["texture_asset_count"] == 1
    assert result["textured_mesh_count"] == 1
    assert result["texture_rows_sample"][0]["destination_name"] == "texture_0000.png"
    assert result["texture_rows_sample"][0]["converted_to_png_for_mujoco"] is True
    xml_text = Path(result["visual_mjcf_path"]).read_text(encoding="utf-8")
    assert "<texture" in xml_text
    assert "texture_0000.png" in xml_text
    assert 'offwidth="960"' in xml_text
    assert "<mesh" in xml_text


def test_usd_mesh_export_uses_bbox_proxy_when_triangle_limited(tmp_path: Path) -> None:
    scene = tmp_path / "large_mesh.usda"
    scene.write_text(
        """
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World" {
    def Mesh "ManyFaces" {
        point3f[] points = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
        ]
        int[] faceVertexCounts = [4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]
    }
}
""".strip(),
        encoding="utf-8",
    )
    from pxr import Usd, UsdGeom  # type: ignore[import-untyped]

    stage = Usd.Stage.Open(str(scene))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ManyFaces"))
    obj_path = tmp_path / "proxy.obj"

    result = packet._write_obj_from_usd_mesh(
        mesh=mesh,
        output_path=obj_path,
        max_triangles=1,
        world_bounds={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
    )

    assert result is not None
    assert result["bbox_proxy_used"] is True
    assert result["estimated_source_triangle_count"] == 4
    assert result["triangle_count"] == 12
    assert obj_path.read_text(encoding="utf-8").count("\nf ") == 12


def test_mjcf_scene_summary_uses_mujoco_target_geom(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """
<mujoco model="target_box">
  <worldbody>
    <geom name="sink_handle" type="box" size="0.1 0.2 0.3" pos="1 2 0.5"/>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )

    result = packet._scene_bounds_and_target(scene, target_object_id="sink_handle")

    assert result["status"] == "complete"
    assert result["matched_target_count"] == 1
    assert result["selected_target_prim"]["geom_name"] == "sink_handle"
    assert result["mujoco_model_geom_count"] == 1


def test_compose_scene_with_unitree_g1_mjcf_places_robot_and_keeps_scene_noncontact(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """
<mujoco model="visual_scene">
  <worldbody>
    <geom name="Sink054_handle" type="box" size="0.05 0.05 0.05" pos="1 2 0.9"/>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    g1 = tmp_path / "g1.xml"
    g1.write_text(
        """
<mujoco model="minimal_g1">
  <worldbody>
    <body name="pelvis" pos="0 0 0.793">
      <freejoint name="floating_base_joint"/>
      <geom name="g1_torso" type="box" size="0.15 0.08 0.35" rgba="0.2 0.2 0.25 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )

    result = packet._compose_scene_with_unitree_g1_mjcf(
        scene_mjcf_path=scene,
        output_dir=tmp_path / "out",
        generated_at="2026-06-23T00:00:00+00:00",
        robot_pose={"xyz": [0.25, -0.5, 0.0], "rpy": [0.0, 0.0, 0.4]},
        target_pose={"xyz": [1.0, 2.0, 0.9], "rpy": [0.0, 0.0, 0.0]},
        g1_mjcf_path=g1,
    )

    assert result["status"] == "completed"
    assert result["unitree_g1_asset_spawned"] is True
    assert result["unitree_g1_floating_base_joint_found"] is True
    assert result["unitree_g1_root_body_pos"] == [0.25, -0.5, 0.793]
    assert result["explicit_lights_authored"] is True
    assert result["explicit_cameras_authored"] is True
    assert result["scene_visual_collision_enabled"] is False
    assert result["claim_boundary"]["scene_collision_geometry_validated"] is False
    assert result["claim_boundary"]["physics_contact_validated"] is False
    combined_xml = Path(result["combined_mjcf_path"]).read_text(encoding="utf-8")
    assert "blueprint_key_light" in combined_xml
    assert "blueprint_head_pov" in combined_xml
    assert "blueprint_torso_pov" in combined_xml
    assert "scene_Sink054_handle" in combined_xml
    assert 'contype="0"' in combined_xml
