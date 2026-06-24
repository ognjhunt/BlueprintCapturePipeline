from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline import vast_provider_adapter
from blueprint_pipeline import unitree_groot_n17_sonic_vast_persistent_session as session


def _policy_observation(path: Path, frame: Path) -> Path:
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "turn_on_sink_handle",
        "visual_observation": {"camera_frame_path": str(frame)},
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
        "unitree_g1_sonic_state_source": "test_contract_probe",
    }
    path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    return path


def test_persistent_session_bundle_uses_proven_policy_server_rewrite(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["loop_step_count"] == 12
    assert manifest["allow_structural_wam_fallback"] is True
    bundle_path = Path(str(manifest["bundle_path"]))
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        runner = archive.read(
            "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py"
        ).decode()
        session_input = json.loads(archive.read("provider_runtime/persistent_session_input.json"))

    assert "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh" in names
    assert "provider_runtime/unitree_groot_n17_sonic_provider_runner.py" in names
    assert "provider_runtime/policy_input.json" in names
    assert "provider_runtime/input_frame.png" in names
    assert "provider_instance_reused_for_policy_and_wam_loop" in runner
    assert "bootstrap_venv_policy_server_client_for_persistent_session" in runner
    assert "persistent_policy_worker_command_uses_policy_server_client" in runner
    assert "not self.use_live_wam" in runner
    assert "shlex.quote(str(venv_python))" in runner
    assert "_http_post_json_with_retries" in runner
    assert session_input["loop_step_count"] == 12
    assert session_input["use_live_wam"] is False
    assert session_input["allow_structural_wam_fallback"] is True


def test_run_persistent_session_imports_reused_worker_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_vast(**kwargs):
        captured.update(kwargs)
        captured["policy_command_env"] = session.os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
        )
        captured["persistent_inner_policy_command_env"] = session.os.environ.get(
            session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV
        )
        captured["vast_inner_policy_command_env"] = session.os.environ.get(
            session.INNER_POLICY_COMMAND_ENV
        )
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr(
                "unitree_groot_n17_sonic_wam_persistent_session_output.json",
                json.dumps(
                    {
                        "schema_version": session.OUTPUT_SCHEMA_VERSION,
                        "status": "completed",
                        "persistent_provider_session_used": True,
                        "provider_instance_reused_for_policy_and_wam_loop": True,
                        "repeated_policy_calls_count": 12,
                        "generated_next_observation_count": 11,
                        "live_wam_generation_success_count": 0,
                        "learned_wam_model_success_count": 0,
                        "unitree_groot_n17_sonic_model_executed": True,
                        "unitree_groot_n17_sonic_policy_action_command_ran": True,
                        "unitree_policy_action_command_ran": True,
                        "policy_action_model_command_ran": True,
                        "provider_output_replay_used": False,
                        "blockers": [],
                    }
                ),
            )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.01}

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "run_vast_provider_adapter", fake_vast)

    output, exit_code = session.run_persistent_session(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=12,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
    )

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["persistent_provider_session_used"] is True
    assert output["provider_instance_reused_for_policy_and_wam_loop"] is True
    assert output["repeated_policy_calls_count"] == 12
    assert output["generated_next_observation_count"] == 11
    assert output["provider_output_replay_used"] is False
    assert captured["provider_bundle_kind"] == "unitree_groot_n17_sonic"
    assert captured["enable_blueprint_bundle"] is True
    assert captured["allowed_machine_ids"] == []
    assert captured["policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND
    assert captured["persistent_inner_policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND
    assert captured["vast_inner_policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND


def test_vast_probe_env_forwards_persistent_inner_policy_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV,
        session.DEFAULT_INNER_POLICY_COMMAND,
    )
    monkeypatch.setenv(session.INNER_POLICY_COMMAND_ENV, session.DEFAULT_INNER_POLICY_COMMAND)

    env = vast_provider_adapter._probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
    )

    assert (
        env[session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV]
        == session.DEFAULT_INNER_POLICY_COMMAND
    )
    assert env[session.INNER_POLICY_COMMAND_ENV] == session.DEFAULT_INNER_POLICY_COMMAND
