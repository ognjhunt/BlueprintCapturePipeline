from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import unitree_groot_n17_sonic_sim2sim_command as sim2sim


def _write_minimal_g1_upper_body_scene(path: Path) -> None:
    bodies_xml = "\n".join(
        (
            f'    <body name="body_{index}" pos="{index * 0.01:.3f} 0 0.8">\n'
            f'      <joint name="{name}" type="hinge" range="-2 2" damping="0.1"/>\n'
            '      <geom type="capsule" size="0.01 0.03" mass="0.02"/>\n'
            "    </body>"
        )
        for index, name in enumerate(sim2sim.UPPER_BODY_JOINT_NAMES)
    )
    actuators_xml = "\n".join(
        f'    <position name="{name}" joint="{name}" kp="20" ctrlrange="-2 2"/>'
        for name in sim2sim.UPPER_BODY_JOINT_NAMES
    )
    path.write_text(
        f"""<mujoco model="minimal_g1_sonic_sim2sim">
  <option timestep="0.01"/>
  <worldbody>
{bodies_xml}
  </worldbody>
  <actuator>
{actuators_xml}
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )


def _write_minimal_g1_upper_body_scene_with_object(path: Path) -> None:
    bodies: list[str] = []
    for index, name in enumerate(sim2sim.UPPER_BODY_JOINT_NAMES):
        body_name = "left_hand_probe_link" if index == 0 else f"body_{index}"
        pos = "0 0 0.8" if index == 0 else f"{index * 0.01:.3f} 0 0.8"
        bodies.append(
            f'    <body name="{body_name}" pos="{pos}">\n'
            f'      <joint name="{name}" type="hinge" range="-2 2" damping="0.1"/>\n'
            '      <geom type="sphere" size="0.025" mass="0.02"/>\n'
            "    </body>"
        )
    bodies.append(
        f'    <body name="{sim2sim.OBJECT_BODY_NAME}" pos="0 0 0">\n'
        f'      <joint name="{sim2sim.OBJECT_FREEJOINT_NAME}" type="free"/>\n'
        f'      <geom name="{sim2sim.OBJECT_GEOM_NAME}" type="sphere" size="0.03" mass="0.05"/>\n'
        "    </body>"
    )
    actuators_xml = "\n".join(
        f'    <position name="{name}" joint="{name}" kp="20" ctrlrange="-2 2"/>'
        for name in sim2sim.UPPER_BODY_JOINT_NAMES
    )
    path.write_text(
        f"""<mujoco model="minimal_g1_sonic_sim2sim_object">
  <option timestep="0.01"/>
  <worldbody>
{chr(10).join(bodies)}
  </worldbody>
  <actuator>
{actuators_xml}
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )


def _write_minimal_g1_projectable_upper_body_scene(path: Path) -> None:
    body_name_by_joint = {
        "left_shoulder_pitch_joint": "left_shoulder_pitch_link",
        "left_elbow_joint": "left_elbow_link",
        "left_wrist_yaw_joint": "left_wrist_yaw_link",
        "left_hand_thumb_0_joint": "left_hand_palm_link",
        "right_shoulder_pitch_joint": "right_shoulder_pitch_link",
        "right_elbow_joint": "right_elbow_link",
        "right_wrist_yaw_joint": "right_wrist_yaw_link",
        "right_hand_thumb_0_joint": "right_hand_palm_link",
    }
    bodies: list[str] = [
        '    <camera name="blueprint_g1_head_pov" pos="0 -1.2 1.1" '
        'xyaxes="1 0 0 0 0 1" fovy="70"/>'
    ]
    for index, name in enumerate(sim2sim.UPPER_BODY_JOINT_NAMES):
        body_name = body_name_by_joint.get(name, f"projectable_body_{index}")
        side_offset = -0.18 if name.startswith("left_") else 0.18
        forward = 0.08 + 0.012 * index
        height = 0.95 - 0.008 * (index % 7)
        bodies.append(
            f'    <body name="{body_name}" pos="{side_offset:.3f} {forward:.3f} {height:.3f}">\n'
            f'      <joint name="{name}" type="hinge" range="-2 2" damping="0.1"/>\n'
            '      <geom type="sphere" size="0.015" mass="0.02"/>\n'
            "    </body>"
        )
    actuators_xml = "\n".join(
        f'    <position name="{name}" joint="{name}" kp="20" ctrlrange="-2 2"/>'
        for name in sim2sim.UPPER_BODY_JOINT_NAMES
    )
    path.write_text(
        f"""<mujoco model="minimal_g1_projectable_sonic_sim2sim">
  <option timestep="0.01"/>
  <worldbody>
{chr(10).join(bodies)}
  </worldbody>
  <actuator>
{actuators_xml}
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )


def test_sim2sim_consumes_real_action_chunk_into_mujoco_joint_trace(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    scene_xml = tmp_path / "minimal_g1.xml"
    _write_minimal_g1_upper_body_scene(scene_xml)
    values = [((index % 19) - 9) / 10.0 for index in range(sim2sim.SONIC_ACTION_DIM * 2)]
    policy_output = job_dir / "policy_action_model_command_output.json"
    policy_output.write_text(
        json.dumps(
            {
                "status": "completed",
                "policy_id": "unitree_groot_n17_sonic_policy",
                "unitree_policy_action_command_ran": True,
                "unitree_groot_n17_sonic_policy_action_command_ran": True,
                "action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": values,
                    "unitree_groot_n17_sonic_action_chunk_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = sim2sim.run_unitree_groot_n17_sonic_sim2sim(
        job_dir=job_dir,
        policy_action_output=policy_output,
        scene_xml=scene_xml,
        steps=2,
        render_video=False,
        generated_at="now",
    )

    assert result["status"] == "completed"
    assert result["unitree_groot_n17_sonic_sim2sim_command_ran"] is True
    assert result["unitree_groot_n17_sonic_action_chunk_consumed"] is True
    assert result["source_action_dim"] == 78
    assert result["source_action_frame_count"] == 2
    assert result["moved_upper_body_or_hand_joint_count"] > 0
    trace_path = Path(result["action_trace_jsonl"])
    assert trace_path.is_file()
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["applied_target_count"] == len(sim2sim.UPPER_BODY_JOINT_NAMES)
    truth = json.loads(
        (job_dir / "unitree_groot_n17_sonic_sim2sim_controller_truth.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["generated_world_rank_fidelity_result_proven"] is False
    assert truth["generated_world_policy_evaluation_scope_proven"] is False
    assert truth["non_ranking_operational_claim_proven"] is False


def test_sim2sim_exports_policy_action_projected_skeleton_trace(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    scene_xml = tmp_path / "minimal_g1_projectable.xml"
    _write_minimal_g1_projectable_upper_body_scene(scene_xml)
    values = [0.2 if index % 2 else -0.2 for index in range(sim2sim.SONIC_ACTION_DIM * 2)]
    policy_output = job_dir / "policy_action_model_command_output.json"
    policy_output.write_text(
        json.dumps(
            {
                "status": "completed",
                "policy_id": "unitree_groot_n17_sonic_policy",
                "unitree_policy_action_command_ran": True,
                "unitree_groot_n17_sonic_policy_action_command_ran": True,
                "action": {
                    "action_type": "unitree_g1_sonic_latent_action_chunk",
                    "action_chunk": values,
                    "unitree_groot_n17_sonic_action_chunk_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = sim2sim.run_unitree_groot_n17_sonic_sim2sim(
        job_dir=job_dir,
        policy_action_output=policy_output,
        scene_xml=scene_xml,
        steps=2,
        render_video=False,
        generated_at="now",
    )

    assert result["status"] == "completed"
    assert result["policy_action_projected_skeleton_status"] == "completed"
    assert result["policy_action_projected_skeleton_projectable_row_count"] == 2
    trace_path = Path(result["policy_action_projected_skeleton_trace_jsonl"])
    assert trace_path.is_file()
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["schema_version"] == sim2sim.PROJECTED_SKELETON_SCHEMA_VERSION
    assert rows[0]["projected_landmark_count"] > 0
    assert rows[0]["claim_boundary"]["policy_derived_action_conditioning"] is True
    assert (
        rows[0]["claim_boundary"]["nominal_kinematic_projection_without_scene_or_wbc_bridge"]
        is False
    )
    assert rows[0]["claim_boundary"]["official_wbc_or_sim_bridge_used"] is True
    manifest = json.loads(
        (job_dir / "policy_action_projected_skeleton_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "completed"
    assert manifest["projectable_row_count"] == 2
    assert manifest["claim_boundary"]["derived_from_policy_action_sim2sim_bridge"] is True


def test_sim2sim_initializes_scene_object_freejoint_and_records_contact_metrics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    scene_xml = tmp_path / "minimal_g1_object.xml"
    _write_minimal_g1_upper_body_scene_with_object(scene_xml)
    (job_dir / "manipulation_scene_object_manifest.json").write_text(
        json.dumps(
            {
                "object_id": "blueprint_light_object",
                "initial_pose": [10.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "policy_action_model_command_output.json").write_text(
        json.dumps(
            {
                "unitree_policy_action_command_ran": True,
                "unitree_groot_n17_sonic_policy_action_command_ran": True,
                "action": {"action_chunk": [0.2] * sim2sim.SONIC_ACTION_DIM},
            }
        ),
        encoding="utf-8",
    )

    result = sim2sim.run_unitree_groot_n17_sonic_sim2sim(
        job_dir=job_dir,
        scene_xml=scene_xml,
        steps=1,
        action_hold_steps=2,
        render_video=False,
        generated_at="now",
    )

    assert result["status"] == "completed"
    assert result["sim_step_count"] == 2
    assert result["object_freejoint_initialized"] is True
    assert result["object_initial_pose"]["available"] is True
    assert result["object_initial_pose"]["position"] == [10.0, 0.0, 0.8]
    assert isinstance(result["object_any_contact_count"], int)
    assert isinstance(result["object_robot_contact_count"], int)
    assert isinstance(result["object_displacement_without_robot_contact"], bool)
    assert isinstance(result["object_horizontal_displacement_m"], float)
    assert isinstance(result["object_horizontal_displacement_without_robot_contact"], bool)
    assert result["object_displacement_success_axis"] == "xy"
    assert "minimum_nearest_hand_object_distance" in result
    assert "policy_action_chunk_integrated_into_contact_rollout" in result
    assert result["contact_rollout_blockers"]
    assert set(result["contact_rollout_blockers"]).issubset(set(result["blockers"]))
    truth = json.loads(
        (job_dir / "unitree_groot_n17_sonic_sim2sim_controller_truth.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["object_freejoint_initialized"] is True
    assert "minimum_nearest_hand_object_distance" in truth
    assert "object_horizontal_displacement_m" in truth
    assert truth["generated_world_rank_fidelity_result_proven"] is False


def test_sim2sim_blocks_without_policy_output(tmp_path: Path) -> None:
    result = sim2sim.run_unitree_groot_n17_sonic_sim2sim(
        job_dir=tmp_path / "job",
        policy_action_output=tmp_path / "missing.json",
        scene_xml=tmp_path / "missing.xml",
        render_video=False,
        generated_at="now",
    )

    assert result["status"] == "blocked"
    assert result["unitree_groot_n17_sonic_sim2sim_command_ran"] is False
    assert "blocked_missing_policy_action_model_command_output" in result["blockers"]


def test_sim2sim_cli_entrypoint_prints_completed_summary(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    scene_xml = tmp_path / "minimal_g1.xml"
    _write_minimal_g1_upper_body_scene(scene_xml)
    (job_dir / "policy_action_model_command_output.json").write_text(
        json.dumps({"action": {"action_chunk": [0.1] * sim2sim.SONIC_ACTION_DIM}}),
        encoding="utf-8",
    )

    exit_code = sim2sim.main(
        [
            "--job-dir",
            str(job_dir),
            "--scene-xml",
            str(scene_xml),
            "--steps",
            "1",
            "--action-hold-steps",
            "3",
            "--no-render-video",
        ]
    )

    assert exit_code == 0
    execution = json.loads(
        (job_dir / "unitree_groot_n17_sonic_sim2sim_execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["sim_step_count"] == 3
    assert execution["action_hold_steps"] == 3


def test_runtime_audit_accepts_python_module_sim2sim_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import unitree_groot_n17_sonic_policy_runtime as runtime

    for name in runtime.ENV_VAR_NAMES:
        monkeypatch.delenv(name, raising=False)
    command = tmp_path / "policy_command.py"
    command.write_text("print('{}')\n", encoding="utf-8")
    n17_checkpoint = tmp_path / "n17"
    n17_checkpoint.mkdir()
    sonic_checkpoint = tmp_path / "sonic"
    sonic_checkpoint.mkdir()
    groot_root = tmp_path / "Isaac-GR00T"
    for relative in runtime.EXPECTED_GROOT_FILES:
        path = groot_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n", encoding="utf-8")
    wbc_root = tmp_path / "GR00T-WholeBodyControl"
    for relative in runtime.EXPECTED_WBC_FILES:
        path = wbc_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n", encoding="utf-8")
    monkeypatch.setenv(runtime.GROOT_ROOT_ENV, str(groot_root))
    monkeypatch.setenv(runtime.WBC_ROOT_ENV, str(wbc_root))
    monkeypatch.setenv(runtime.N17_CHECKPOINT_ENV, str(n17_checkpoint))
    monkeypatch.setenv(runtime.SONIC_CHECKPOINT_ENV, str(sonic_checkpoint))
    monkeypatch.setenv(runtime.POLICY_COMMAND_ENV, f"{sys.executable} {command}")
    monkeypatch.setenv(
        runtime.SIM2SIM_COMMAND_ENV,
        f"{sys.executable} -m blueprint_pipeline.unitree_groot_n17_sonic_sim2sim_command",
    )

    audit = runtime.probe_unitree_groot_n17_sonic_runtime(generated_at="now")

    assert audit["ready_for_sim2sim"] is True
    assert audit["sim2sim_command_available"] is True
    assert f"blocked_missing_{runtime.SIM2SIM_COMMAND_ENV}" not in audit["blockers"]
