from __future__ import annotations

import json
from pathlib import Path

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
