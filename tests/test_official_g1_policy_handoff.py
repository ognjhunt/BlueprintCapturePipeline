from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.official_g1_policy_handoff import (
    _camera_set,
    _frame_durations_for_realtime_video,
    _redact_runtime_value,
    _robot_pov_manifest,
    _stream_gate,
    _video_encoding_settings,
    _write_camera_scene_xml,
)


def test_handoff_stream_gate_requires_qpos_qvel_and_policy_observations() -> None:
    incomplete_rows = [
        {
            "qpos": [0.0],
            "joint_positions": {},
            "joint_velocities": {},
            "actuator_controls": [],
            "actuator_forces": [],
            "foot_contact_states": {},
            "command_xyz": [0.5, 0.0, 0.0],
        }
    ]

    gate = _stream_gate(incomplete_rows, control_update_count=0)

    assert gate["passed"] is False
    assert "missing_qvel_stream" in gate["blockers"]
    assert "missing_policy_observation_stream" in gate["blockers"]

    complete_rows = [{**incomplete_rows[0], "qvel": [0.0]}]
    gate = _stream_gate(complete_rows, control_update_count=1)

    assert gate["passed"] is True
    assert gate["blockers"] == []


def test_robot_pov_manifest_is_simulated_body_mounted(tmp_path: Path) -> None:
    manifest = _robot_pov_manifest(
        camera_records=[
            {
                "camera": "robot_pov_head",
                "camera_body_name": "pelvis",
                "simulated_robot_pov": True,
            }
        ],
        robot_pov_frames={"head": [str(tmp_path / "head_0000.png")], "torso": []},
        robot_pov_videos={"head": {"status": "complete"}, "torso": {"status": "not_generated"}},
        render_width=1280,
        render_height=720,
        render_fps=24,
        nonblank_checks={"all_frames_nonblank": True},
        calibration_path=tmp_path / "robot_pov_camera_calibration.json",
    )

    assert manifest["simulated_robot_pov"] is True
    assert manifest["real_robot_pov"] is False
    assert manifest["physical_sensor_data"] is False
    assert manifest["camera_body_name"] == "pelvis"
    assert manifest["render_resolution"] == [1280, 720]


def test_video_settings_respect_resolution_fps_crf_config() -> None:
    settings = _video_encoding_settings(render_fps=30, video_crf=20)

    assert settings["fps"] == 30
    assert settings["video_crf"] == 20
    assert settings["codec"] == "libx264"


def test_frame_durations_preserve_source_sim_time_for_realtime_video() -> None:
    durations, timing = _frame_durations_for_realtime_video(
        frame_count=4,
        render_fps=24,
        frame_times_s=[0.0, 0.5, 1.0, 1.5],
        video_duration_s=2.0,
    )

    assert durations == [0.5, 0.5, 0.5, 0.5]
    assert timing["mode"] == "source_sim_time_realtime"
    assert timing["expected_video_duration_s"] == 2.0


def test_secret_signature_redaction_for_json_artifacts() -> None:
    signature_query = "x-goog-" + "signature=abc123"
    payload = {
        "url": f"https://storage.googleapis.com/bucket/object?{signature_query}&x=1",
        "nested": ["no-secret"],
    }

    redacted = _redact_runtime_value(payload)

    assert "abc123" not in redacted["url"]
    assert "x-goog-redacted-signature-param=<redacted:signed-url-signature>" in redacted["url"]
    assert signature_query.split("=", 1)[0] + "=" not in redacted["url"]


def test_camera_set_expands_robot_pov() -> None:
    assert _camera_set("overview,robot_pov") == [
        "overview",
        "robot_pov_head",
        "robot_pov_torso",
    ]


def test_camera_scene_xml_can_embed_matching_external_scene_collision_mesh(
    tmp_path: Path,
) -> None:
    robot_xml = tmp_path / "robot.xml"
    source_scene_xml = tmp_path / "source_scene.xml"
    external_scene_obj = tmp_path / "warehouse.obj"
    output_scene_xml = tmp_path / "camera_scene.xml"
    robot_xml.write_text("<mujoco/>", encoding="utf-8")
    source_scene_xml.write_text("<mujoco/>", encoding="utf-8")
    external_scene_obj.write_text("v 0 0 0\n", encoding="utf-8")

    _write_camera_scene_xml(
        source_scene_xml,
        robot_xml,
        output_scene_xml,
        render_width=640,
        render_height=360,
        external_scene_obj=external_scene_obj,
    )

    xml = output_scene_xml.read_text(encoding="utf-8")
    assert 'mesh name="blueprint_external_scene_mesh"' in xml
    assert 'name="blueprint_external_scene_visual"' in xml
    assert 'name="blueprint_external_scene_collision"' in xml
    assert f'file="{external_scene_obj}"' in xml
