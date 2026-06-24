from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import unitree_groot_n17_sonic_vast_policy_command as command


def test_vast_policy_command_runs_provider_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    bundle_path = tmp_path / "provider_bundle.zip"
    bundle_path.write_bytes(b"bundle")
    monkeypatch.setenv(command.JOB_ROOT_ENV, str(tmp_path / "jobs"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT", "LucaFrat/groot-bs16")
    monkeypatch.setenv("BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT", "LucaFrat/groot-bs16")
    monkeypatch.setenv(
        command.STANDARD_POLICY_COMMAND_ENV,
        "python -m blueprint_pipeline.unitree_groot_n17_sonic_vast_policy_command",
    )
    monkeypatch.setenv("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_EXCLUDED_MACHINE_ID", "140330")
    monkeypatch.setenv("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_ALLOWED_MACHINE_ID", "51579")
    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured["build_policy_observation_path"] = kwargs["policy_observation_path"]
        captured["policy_command"] = kwargs["policy_command"]
        return {
            "status": "bundle_ready",
            "bundle_path": str(bundle_path),
            "manifest_path": str(tmp_path / "manifest.json"),
            "blockers": [],
        }

    def fake_stage(**kwargs):
        staging_dir = Path(kwargs["job_dir"])
        staging_dir.mkdir(parents=True)
        (staging_dir / "provider_bundle_url.txt").write_text("https://storage.example/b.zip")
        (staging_dir / "provider_output_put_url.txt").write_text(
            "https://storage.example/out.zip?put"
        )
        (staging_dir / "provider_output_get_url.txt").write_text(
            "https://storage.example/out.zip?get"
        )
        return {"status": "completed", "blockers": []}

    def fake_vast(**kwargs):
        captured["machine_avoidlist_path"] = kwargs["machine_avoidlist_path"]
        captured["min_gpu_ram_mb"] = kwargs["min_gpu_ram_mb"]
        captured["allowed_machine_ids"] = kwargs["allowed_machine_ids"]
        captured["forwarded_policy_command"] = command.os.environ[
            command.STANDARD_POLICY_COMMAND_ENV
        ]
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        output_zip.write_bytes(b"zip")
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.01}

    def fake_import(**_kwargs):
        return {
            "status": "completed",
            "blockers": [],
            "unitree_groot_n17_sonic_model_executed": True,
            "unitree_groot_n17_sonic_policy_action_command_ran": True,
            "action": {
                "action_type": "unitree_g1_sonic_latent_action_chunk",
                "action_chunk": [0.1, 0.2],
            },
        }

    monkeypatch.setattr(command, "build_unitree_groot_n17_sonic_policy_provider_bundle", fake_build)
    monkeypatch.setattr(command, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(command, "run_vast_provider_adapter", fake_vast)
    monkeypatch.setattr(command, "import_unitree_groot_n17_sonic_provider_output", fake_import)

    payload = {
        "task_id": "turn_on_sink_handle",
        "observation": {"visual_observation": {"camera_frame_path": str(frame)}},
    }
    output, exit_code = command.run_vast_policy_command(payload)

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert output["provider_output_replay_used"] is False
    assert output["action"]["action_chunk"] == [0.1, 0.2]
    assert Path(str(captured["build_policy_observation_path"])).is_file()
    assert "unitree_groot_n17_sonic_policy_server_command" in str(captured["policy_command"])
    assert "unitree_groot_n17_sonic_policy_server_command" in str(
        captured["forwarded_policy_command"]
    )
    assert "vast_policy_command" not in str(captured["forwarded_policy_command"])
    avoidlist = json.loads(Path(str(captured["machine_avoidlist_path"])).read_text())
    assert 140330 in avoidlist["machine_ids"]
    assert captured["min_gpu_ram_mb"] == 48000
    assert captured["allowed_machine_ids"] == [51579]
    assert "vast_policy_command" in command.os.environ[command.STANDARD_POLICY_COMMAND_ENV]


def test_vast_policy_command_falls_back_when_allowed_machine_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    bundle_path = tmp_path / "provider_bundle.zip"
    bundle_path.write_bytes(b"bundle")
    monkeypatch.setenv(command.JOB_ROOT_ENV, str(tmp_path / "jobs"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_ALLOWED_MACHINE_ID", "55264")
    monkeypatch.setenv(command.ALLOW_UNPINNED_FALLBACK_ENV, "true")
    calls: list[dict[str, object]] = []

    def fake_build(**_kwargs):
        return {
            "status": "bundle_ready",
            "bundle_path": str(bundle_path),
            "manifest_path": str(tmp_path / "manifest.json"),
            "blockers": [],
        }

    def fake_stage(**kwargs):
        staging_dir = Path(kwargs["job_dir"])
        staging_dir.mkdir(parents=True)
        (staging_dir / "provider_bundle_url.txt").write_text("https://storage.example/b.zip")
        (staging_dir / "provider_output_put_url.txt").write_text(
            "https://storage.example/out.zip?put"
        )
        (staging_dir / "provider_output_get_url.txt").write_text(
            "https://storage.example/out.zip?get"
        )
        return {"status": "completed", "blockers": []}

    def fake_vast(**kwargs):
        calls.append(
            {
                "job_dir": kwargs["job_dir"],
                "allowed_machine_ids": kwargs["allowed_machine_ids"],
            }
        )
        if len(calls) == 1:
            return {
                "status": "blocked",
                "blockers": [
                    "no_vast_offer_matching_allowed_machine_ids",
                    "no_vast_offer_at_or_below_max_hourly_rate",
                ],
            }
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        output_zip.write_bytes(b"zip")
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.02}

    def fake_import(**_kwargs):
        return {
            "status": "completed",
            "blockers": [],
            "unitree_groot_n17_sonic_model_executed": True,
            "unitree_groot_n17_sonic_policy_action_command_ran": True,
            "action": {
                "action_type": "unitree_g1_sonic_latent_action_chunk",
                "action_chunk": [0.3],
            },
        }

    monkeypatch.setattr(command, "build_unitree_groot_n17_sonic_policy_provider_bundle", fake_build)
    monkeypatch.setattr(command, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(command, "run_vast_provider_adapter", fake_vast)
    monkeypatch.setattr(command, "import_unitree_groot_n17_sonic_provider_output", fake_import)

    output, exit_code = command.run_vast_policy_command(
        {
            "task_id": "turn_on_sink_handle",
            "observation": {"visual_observation": {"camera_frame_path": str(frame)}},
        }
    )

    assert exit_code == 0
    assert output["status"] == "completed"
    assert [call["allowed_machine_ids"] for call in calls] == [[55264], []]
    assert "vast_provider_run_unpinned_fallback" in output["vast_provider_adapter_result_path"]
    assert output["action"]["action_chunk"] == [0.3]


def test_vast_policy_command_blocks_without_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "out.json"
    monkeypatch.setenv(command.JOB_ROOT_ENV, str(tmp_path / "jobs"))
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(output_path))
    payload = {"observation": {"visual_observation": {"camera_frame_path": str(tmp_path / "x.jpg")}}}

    output, exit_code = command.run_vast_policy_command(payload)

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["blockers"] == ["blocked_missing_policy_visual_observation_frame"]


def test_vast_policy_command_main_writes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"observation": {}}), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_INPUT", str(input_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(output_path))

    def fake_run(payload):
        return {
            "schema_version": command.SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["expected"],
            "payload_seen": payload,
        }, 2

    monkeypatch.setattr(command, "run_vast_policy_command", fake_run)

    assert command.main() == 2
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["blockers"] == ["expected"]
    assert written["payload_seen"]["observation"] == {}
