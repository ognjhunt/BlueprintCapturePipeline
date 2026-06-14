from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.robot_eval_execution import build_simulator_command_artifacts
from blueprint_pipeline.mujoco_g1_simulator_command import (
    _collision_summary,
    _episode_navigation_spec,
    _matrix_runs,
    _obj_vertex_color_summary,
    _render_capture_steps,
    _scene_collision_contact_count,
    _visual_artifact_summary,
    _write_mjcf_wrapper,
    run_mujoco_g1_simulator_command,
)


def test_render_capture_steps_samples_continuous_motion() -> None:
    steps = _render_capture_steps(240)

    assert len(steps) == 24
    assert min(steps) == 0
    assert max(steps) == 239
    assert len({b - a for a, b in zip(sorted(steps), sorted(steps)[1:])}) <= 2


def test_matrix_runs_loads_rows_and_reports_missing_required_ids(tmp_path: Path) -> None:
    matrix_path = tmp_path / "scenario_eval_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "scenario_eval_run_count": 2,
                "runs": [
                    {"scenario_eval_run_id": "run-explicit", "task_id": "walk_to_target"},
                    {"task_id": "walk_to_target", "scenario_id": "scenario-b"},
                ],
            }
        ),
        encoding="utf-8",
    )

    runs, summary = _matrix_runs(matrix_path)

    assert len(runs) == 2
    assert summary["scenario_eval_run_count"] == 2
    assert runs[0]["scenario_eval_run_id"] == "run-explicit"
    assert "scenario_eval_run_id" not in runs[1]
    assert summary["missing_scenario_eval_run_id_indexes"] == [2]
    assert summary["scenario_eval_run_ids_unique"] is True


def test_mujoco_g1_command_rejects_malformed_supplied_matrix_before_scene_work(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "scenario_eval_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "scenario_eval_run_count": 2,
                "runs": [
                    {"scenario_eval_run_id": "duplicate-run", "task_id": "walk_to_target"},
                    {"scenario_eval_run_id": "duplicate-run", "task_id": "walk_to_target"},
                    {"task_id": "walk_to_target"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_mujoco_g1_simulator_command(
            capture_root=tmp_path / "capture-without-scene",
            scenario_eval_matrix_path=matrix_path,
            render_frames=False,
        )

    message = str(exc_info.value)
    assert "scenario_eval_matrix_missing_scenario_eval_run_id" in message
    assert "scenario_eval_matrix_duplicate_scenario_eval_run_id" in message
    assert "scenario_eval_matrix_declared_count_mismatch" in message


def test_episode_navigation_spec_uses_explicit_route_and_stable_seed() -> None:
    run = {
        "scenario_eval_run_id": "run-a",
        "task_id": "walk_to_target",
        "scenario_id": "scenario-a",
        "concrete_mutation": {
            "spawn_pose": [1.0, 2.0, 0.81],
            "target_pose": {"x": 3.0, "y": 4.0, "z": 0.82},
        },
    }
    mesh_info = {"bounds": [[-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]]}

    first = _episode_navigation_spec(run=run, mesh_info=mesh_info, index=1)
    second = _episode_navigation_spec(run=run, mesh_info=mesh_info, index=1)

    assert first == second
    assert first["route_source"] == "matrix_explicit_spawn_and_target"
    assert first["start"] == (1.0, 2.0, 0.81)
    assert first["target"] == (3.0, 4.0, 0.82)
    assert first["route_distance_m"] > 0


def test_obj_vertex_color_summary_counts_rgb_vertices(tmp_path: Path) -> None:
    obj_path = tmp_path / "scene.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0 0.1 0.2 0.3",
                "v 1 0 0 0.4 0.5 0.6",
                "v 0 1 0",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _obj_vertex_color_summary(obj_path)

    assert summary["status"] == "inspected"
    assert summary["vertex_count"] == 3
    assert summary["face_count"] == 1
    assert summary["vertex_rgb_count"] == 2
    assert summary["has_vertex_rgb"] is True
    assert summary["vertex_rgb_fraction"] == 0.666667


def test_mjcf_wrapper_has_separate_scene_collision_geom(tmp_path: Path) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")

    _write_mjcf_wrapper(scene_obj, g1_xml, wrapper)
    xml = wrapper.read_text(encoding="utf-8")

    assert 'name="blueprint_scene_visual"' in xml
    assert 'name="blueprint_scene_collision"' in xml
    assert 'name="blueprint_reference_floor"' in xml
    assert 'material="blueprint_scene_mat" contype="0" conaffinity="0"' in xml
    assert (
        'material="blueprint_scene_collision_mat" contype="1" conaffinity="1"'
        in xml
    )


def test_mjcf_wrapper_prefers_collision_proxy_boxes(tmp_path: Path) -> None:
    scene_obj = tmp_path / "scene.obj"
    g1_xml = tmp_path / "g1.xml"
    wrapper = tmp_path / "wrapper.xml"
    scene_obj.write_text("v 0 0 0\n", encoding="utf-8")
    g1_xml.write_text("<mujoco/>", encoding="utf-8")

    _write_mjcf_wrapper(
        scene_obj,
        g1_xml,
        wrapper,
        collision_proxies=[
            {
                "name": "rack_a",
                "pos": [1.0, 2.0, 0.5],
                "size": [0.4, 0.2, 0.5],
            }
        ],
    )
    xml = wrapper.read_text(encoding="utf-8")

    assert 'name="blueprint_scene_collision"' not in xml
    assert 'name="blueprint_collision_proxy_000_rack_a"' in xml
    assert 'type="box"' in xml
    assert 'contype="1" conaffinity="1"' in xml


def test_scene_collision_contact_count_includes_proxy_geoms() -> None:
    records = [
        {"geom_names": ["blueprint_reference_floor", "geom_1"]},
        {"geom_names": ["blueprint_collision_proxy_001_box", "geom_2"]},
        {"scene_collision_contact": True},
    ]

    assert _scene_collision_contact_count(records) == 2


def test_visual_artifact_summary_classifies_frames_and_records_material_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("blueprint_pipeline.mujoco_g1_simulator_command.shutil.which", lambda _: None)
    overview = tmp_path / "overview_0000.png"
    pov = tmp_path / "sim_robot_follow_pov_0000.png"
    side = tmp_path / "side_0000.png"
    for path, base in ((overview, 20), (pov, 80), (side, 140)):
        image = Image.new("RGB", (24, 16))
        for x in range(image.width):
            for y in range(image.height):
                image.putpixel(
                    (x, y),
                    (
                        (base + x * 5) % 255,
                        (base + y * 9) % 255,
                        (base + x * 3 + y * 4) % 255,
                    ),
                )
        image.save(path)
    frames = [
        {"camera": "overview", "path": str(overview), "step": 0},
        {"camera": "sim_robot_follow_pov", "path": str(pov), "step": 0},
        {"camera": "side", "path": str(side), "step": 0},
    ]
    mesh_info = {
        "source_glb": str(tmp_path / "scene.glb"),
        "converted_obj": str(tmp_path / "scene.obj"),
        "visual_asset_summary": {
            "materials_count": 1,
            "textures_count": 0,
            "images_count": 0,
            "has_vertex_colors": False,
        },
        "obj_vertex_color_summary": {
            "has_vertex_rgb": True,
            "vertex_rgb_fraction": 1.0,
        },
        "mujoco_visual_fidelity_boundary": "test boundary",
    }

    summary = _visual_artifact_summary(
        frames=frames,
        output_root=tmp_path,
        mesh_info=mesh_info,
        model_timestep_s=0.01,
    )

    assert summary["status"] == "complete"
    assert summary["overview_frames"] == [str(overview)]
    assert summary["robot_pov_frames"] == [str(pov)]
    assert summary["side_frames"] == [str(side)]
    assert summary["overview_video"]["status"] == "not_generated"
    assert summary["robot_pov_video"]["status"] == "not_generated"
    assert summary["side_video"]["status"] == "not_generated"
    assert summary["texture_material_evidence"]["status"] == (
        "materialized_vertex_color_scene_evidence_present"
    )
    assert summary["blank_scene_checks"]["status"] == "checked"
    assert summary["blank_scene_checks"]["all_frames_nonblank"] is True


def _install_fake_mujoco_backend(monkeypatch) -> None:
    class FakeOpt:
        timestep = 0.01

    class FakeModel:
        opt = FakeOpt()
        jnt_qposadr = np.array([0])
        key_qpos = np.array([[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]])
        qpos0 = np.array([0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0])

        @classmethod
        def from_xml_path(cls, _path: str) -> "FakeModel":
            return cls()

    class FakeData:
        def __init__(self, _model: FakeModel) -> None:
            self.qpos = np.zeros(7)
            self.qvel = np.zeros(7)
            self.time = 0.0

    class FakeCamera:
        def __init__(self) -> None:
            self.type = None
            self.lookat = np.zeros(3)
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    def fake_name_to_id(_model: FakeModel, _object_type: object, name: str) -> int:
        return 0 if name in {"floating_base_joint", "stand"} else -1

    def fake_step(model: FakeModel, data: FakeData) -> None:
        data.time += model.opt.timestep

    fake_mujoco = types.SimpleNamespace(
        __version__="fake-3.0",
        MjModel=FakeModel,
        MjData=FakeData,
        MjvCamera=FakeCamera,
        mjtObj=types.SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_KEY=2),
        mjtCamera=types.SimpleNamespace(mjCAMERA_FREE=1),
        mj_name2id=fake_name_to_id,
        mj_forward=lambda _model, _data: None,
        mj_step=fake_step,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)


def _seed_fake_capture_and_g1(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    capture_root = tmp_path / "capture"
    scene_glb = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    scene_glb.parent.mkdir(parents=True)
    scene_glb.write_bytes(b"fake glb")
    g1_root = tmp_path / "unitree_g1"
    (g1_root / "assets").mkdir(parents=True)
    (g1_root / "g1.xml").write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")

    def fake_convert(_glb_path: Path, obj_path: Path) -> dict[str, object]:
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        obj_path.write_text("v 0 0 0 0.1 0.2 0.3\n", encoding="utf-8")
        return {
            "source_glb": str(scene_glb),
            "converted_obj": str(obj_path),
            "vertices": 1,
            "faces": 0,
            "bounds": [[-2.0, -1.0, 0.0], [2.0, 1.0, 1.0]],
            "extents": [4.0, 2.0, 1.0],
            "centroid": [0.0, 0.0, 0.5],
            "visual_asset_summary": {
                "materials_count": 1,
                "textures_count": 0,
                "images_count": 0,
                "has_vertex_colors": True,
            },
            "obj_vertex_color_summary": {
                "has_vertex_rgb": True,
                "vertex_rgb_fraction": 1.0,
            },
            "mujoco_visual_fidelity_boundary": "test boundary",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._convert_glb_to_obj",
        fake_convert,
    )
    return capture_root, g1_root


def test_mujoco_g1_command_runs_every_matrix_row_with_fake_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeOpt:
        timestep = 0.01

    class FakeModel:
        opt = FakeOpt()
        jnt_qposadr = np.array([0])
        key_qpos = np.array([[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]])
        qpos0 = np.array([0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0])

        @classmethod
        def from_xml_path(cls, _path: str) -> "FakeModel":
            return cls()

    class FakeData:
        def __init__(self, _model: FakeModel) -> None:
            self.qpos = np.zeros(7)
            self.qvel = np.zeros(7)
            self.time = 0.0

    class FakeCamera:
        def __init__(self) -> None:
            self.type = None
            self.lookat = np.zeros(3)
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    def fake_name_to_id(_model: FakeModel, _object_type: object, name: str) -> int:
        return 0 if name in {"floating_base_joint", "stand"} else -1

    def fake_step(model: FakeModel, data: FakeData) -> None:
        data.time += model.opt.timestep

    fake_mujoco = types.SimpleNamespace(
        __version__="fake-3.0",
        MjModel=FakeModel,
        MjData=FakeData,
        MjvCamera=FakeCamera,
        mjtObj=types.SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_KEY=2),
        mjtCamera=types.SimpleNamespace(mjCAMERA_FREE=1),
        mj_name2id=fake_name_to_id,
        mj_forward=lambda _model, _data: None,
        mj_step=fake_step,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    capture_root = tmp_path / "capture"
    scene_glb = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    scene_glb.parent.mkdir(parents=True)
    scene_glb.write_bytes(b"fake glb")
    g1_root = tmp_path / "unitree_g1"
    (g1_root / "assets").mkdir(parents=True)
    (g1_root / "g1.xml").write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    matrix_path = tmp_path / "scenario_eval_matrix.json"
    matrix_rows = [
        {
            "scenario_eval_run_id": "run-a",
            "episode_id": "episode-a",
            "task_id": "walk_to_target",
            "scenario_id": "scenario-a",
            "scenario_variation_instance_id": "variation-a",
            "variation_name": "lighting_variation",
            "concrete_mutation": {
                "spawn_pose": [-1.0, 0.0, 0.793],
                "target_pose": [1.0, 0.0, 0.793],
            },
        },
        {
            "scenario_eval_run_id": "run-b",
            "task_id": "walk_to_target",
            "scenario_id": "scenario-b",
            "scenario_variation_instance_id": "variation-b",
            "variation_name": "blocked_path",
        },
        {
            "scenario_eval_run_id": "run-c",
            "task_id": "walk_to_target",
            "scenario_id": "scenario-c",
            "scenario_variation_instance_id": "variation-c",
            "variation_name": "narrow_approach_angle",
        },
    ]
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scenario_eval_matrix.v1",
                "status": "completed",
                "scenario_eval_run_count": len(matrix_rows),
                "runs": matrix_rows,
            }
        ),
        encoding="utf-8",
    )

    def fake_convert(_glb_path: Path, obj_path: Path) -> dict[str, object]:
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        obj_path.write_text("v 0 0 0 0.1 0.2 0.3\n", encoding="utf-8")
        return {
            "source_glb": str(scene_glb),
            "converted_obj": str(obj_path),
            "vertices": 1,
            "faces": 0,
            "bounds": [[-2.0, -1.0, 0.0], [2.0, 1.0, 1.0]],
            "extents": [4.0, 2.0, 1.0],
            "centroid": [0.0, 0.0, 0.5],
            "visual_asset_summary": {
                "materials_count": 1,
                "textures_count": 0,
                "images_count": 0,
                "has_vertex_colors": True,
            },
            "obj_vertex_color_summary": {
                "has_vertex_rgb": True,
                "vertex_rgb_fraction": 1.0,
            },
            "mujoco_visual_fidelity_boundary": "test boundary",
        }

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._convert_glb_to_obj",
        fake_convert,
    )

    simulator_output = tmp_path / "mujoco_g1_simulator_output.json"
    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "mujoco-output",
        simulator_output_path=simulator_output,
        scenario_eval_matrix_path=matrix_path,
        steps=4,
        render_frames=False,
        max_rendered_episodes=1,
    )

    assert payload["status"] == "completed"
    assert payload["attempt_count"] == 3
    assert payload["scenario_eval_run_count"] == 3
    assert payload["covered_scenario_eval_run_ids"] == ["run-a", "run-b", "run-c"]
    assert payload["missing_scenario_eval_run_ids"] == []
    assert payload["scenario_eval_run_coverage_complete"] is True
    assert payload["rendered_episode_count"] == 0
    assert payload["deterministic_per_episode_spawn_target_seed_handling"] is True
    assert payload["ai_route_selection_used_at_runtime"] is False
    assert payload["collision_geometry_loaded"] is True
    assert payload["scene_collision_mesh_geom_enabled"] is True
    assert payload["scene_visual_mesh_collision_twin_enabled"] is True
    assert payload["scene_visual_mesh_collisions_enabled"] is False
    assert payload["collision_dynamics_validated"] is True
    assert payload["collision_avoidance_validated"] is True
    assert payload["physics_controlled_preview_proven"] is True
    assert payload["robot_team_handoff_ready"] is False
    assert "balanced_walking_controller_not_integrated_in_default_mujoco_preview" in payload[
        "robot_team_handoff_blockers"
    ]
    assert payload["official_policy_handoff"]["entrypoint"] == (
        "python -m blueprint_pipeline.official_g1_policy_handoff"
    )
    assert [attempt["scenario_eval_run_id"] for attempt in payload["attempts"]] == [
        "run-a",
        "run-b",
        "run-c",
    ]
    assert payload["attempts"][0]["spawn_pose"] == [-1.0, 0.0, 0.793]
    assert payload["attempts"][0]["target_pose"] == [1.0, 0.0, 0.793]
    assert simulator_output.is_file()

    artifacts = build_simulator_command_artifacts(
        job_dir=tmp_path / "job",
        simulator="mujoco",
        simulator_output=payload,
        generated_at="2026-06-14T00:00:00Z",
    )
    trace = artifacts["normalized_attempt_trace"]
    assert trace["attempt_count"] == 3
    assert trace["required_scenario_eval_run_count"] == 3
    assert trace["covered_scenario_eval_run_count"] == 3
    assert trace["missing_scenario_eval_run_count"] == 0
    assert trace["scenario_eval_run_coverage_complete"] is True
    assert [attempt["scenario_eval_run_id"] for attempt in trace["attempts"]] == [
        "run-a",
        "run-b",
        "run-c",
    ]


def test_mujoco_g1_command_stops_before_fake_scene_collision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_mujoco_backend(monkeypatch)
    capture_root, g1_root = _seed_fake_capture_and_g1(tmp_path, monkeypatch)
    matrix_path = tmp_path / "scenario_eval_matrix_collision.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scenario_eval_matrix.v1",
                "status": "completed",
                "scenario_eval_run_count": 1,
                "runs": [
                    {
                        "scenario_eval_run_id": "run-collision-wall",
                        "task_id": "walk_to_target",
                        "scenario_id": "scenario-collision-wall",
                        "concrete_mutation": {
                            "spawn_pose": [-1.0, 0.0, 0.793],
                            "target_pose": [1.0, 0.0, 0.793],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_contact_records(_model, data, _mujoco):
        x_position = float(data.qpos[0])
        if x_position > 0.05:
            return [
                {
                    "contact_index": 0,
                    "geom_ids": [1, 2],
                    "geom_names": ["blueprint_collision_proxy_000_wall", "pelvis"],
                    "body_names": ["world", "pelvis"],
                    "distance": -0.1,
                    "position_xyz": [x_position, 0.0, 0.7],
                    "contact_force_6d": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "scene_collision_contact": True,
                    "reference_floor_contact": False,
                }
            ]
        return []

    monkeypatch.setattr(
        "blueprint_pipeline.mujoco_g1_simulator_command._contact_records",
        fake_contact_records,
    )

    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "mujoco-output-collision",
        simulator_output_path=tmp_path / "mujoco_g1_simulator_output_collision.json",
        scenario_eval_matrix_path=matrix_path,
        steps=8,
        render_frames=False,
        max_rendered_episodes=0,
    )
    attempt = payload["attempts"][0]
    actions = attempt["actions"]

    assert payload["status"] == "completed"
    assert payload["physics_controlled_preview_proven"] is True
    assert payload["collision_dynamics_validated"] is True
    assert payload["collision_avoidance_validated"] is True
    assert payload["robot_scene_contact_event_count"] == 0
    assert payload["collision_response_event_count"] > 0
    assert payload["collision_summary"]["rejected_scene_collision_probe_count"] > 0
    assert attempt["status"] == "completed_collision_governed"
    assert attempt["success"] is True
    assert attempt["metrics"]["robot_scene_contact_event_count"] == 0
    assert attempt["metrics"]["collision_response_event_count"] > 0
    assert any(
        action["policy_action"] in {"stopped_by_collision_probe", "redirected_by_collision_probe"}
        for action in actions
    )
    assert max(action["root_position"][0] for action in actions) <= 0.05


def test_proxy_only_collision_summary_does_not_validate_visible_scene_collision() -> None:
    summary = _collision_summary([], collision_proxy_count=3)

    assert summary["scene_collision_proxy_geoms_enabled"] is True
    assert summary["proxy_collision_model_used"] is True
    assert summary["collision_avoidance_validated"] is True
    assert summary["proxy_collision_governed_preview_proven"] is True
    assert summary["visible_scene_collision_alignment_validated"] is False
    assert summary["collision_dynamics_validated"] is False
    assert summary["physics_controlled_preview_proven"] is False


def test_mujoco_g1_command_covers_500_matrix_rows_with_fake_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_mujoco_backend(monkeypatch)
    capture_root, g1_root = _seed_fake_capture_and_g1(tmp_path, monkeypatch)
    matrix_rows = [
        {
            "scenario_eval_run_id": f"run-{index:04d}",
            "episode_id": f"episode-{index:04d}",
            "task_id": "walk_to_target",
            "scenario_id": f"scenario-{index % 10:02d}",
            "scenario_variation_instance_id": f"variation-{index % 11:02d}",
            "variation_name": f"variation_{index % 11:02d}",
        }
        for index in range(1, 501)
    ]
    matrix_path = tmp_path / "scenario_eval_matrix_500.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "robot_eval_scenario_eval_matrix.v1",
                "status": "completed",
                "scenario_eval_run_count": len(matrix_rows),
                "runs": matrix_rows,
            }
        ),
        encoding="utf-8",
    )

    payload = run_mujoco_g1_simulator_command(
        capture_root=capture_root,
        g1_model_root=g1_root,
        output_dir=tmp_path / "mujoco-output-500",
        simulator_output_path=tmp_path / "mujoco_g1_simulator_output_500.json",
        scenario_eval_matrix_path=matrix_path,
        steps=1,
        render_frames=False,
        max_rendered_episodes=0,
    )

    assert payload["status"] == "completed"
    assert payload["scenario_eval_run_count"] == 500
    assert payload["attempt_count"] == 500
    assert payload["required_scenario_eval_run_count"] == 500
    assert payload["covered_scenario_eval_run_count"] == 500
    assert payload["missing_scenario_eval_run_count"] == 0
    assert payload["scenario_eval_run_coverage_complete"] is True
    assert payload["covered_scenario_eval_run_ids"][0] == "run-0001"
    assert payload["covered_scenario_eval_run_ids"][-1] == "run-0500"
    assert payload["rendered_episode_count"] == 0
    assert payload["ai_route_selection_used_at_runtime"] is False
    assert payload["deterministic_per_episode_spawn_target_seed_handling"] is True


def test_simulator_command_artifacts_block_incomplete_required_run_coverage(
    tmp_path: Path,
) -> None:
    payload = {
        "required_scenario_eval_run_ids": ["run-a", "run-b"],
        "attempts": [
            {
                "attempt_id": "attempt-run-a",
                "scenario_eval_run_id": "run-a",
                "scenario_id": "scenario-a",
                "task_id": "walk_to_target",
                "status": "completed",
                "success": True,
            }
        ],
    }

    artifacts = build_simulator_command_artifacts(
        job_dir=tmp_path / "job",
        simulator="mujoco",
        simulator_output=payload,
        generated_at="2026-06-14T00:00:00Z",
    )

    trace = artifacts["normalized_attempt_trace"]
    manifest = artifacts["manifest"]
    assert trace["status"] == "blocked_incomplete_scenario_eval_run_coverage"
    assert trace["attempt_count_matches_matrix_count"] is False
    assert trace["scenario_eval_run_id_coverage_exact"] is False
    assert trace["missing_scenario_eval_run_ids"] == ["run-b"]
    assert trace["scenario_eval_run_coverage_complete"] is False
    assert manifest["status"] == "blocked_incomplete_scenario_eval_run_coverage"
