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
