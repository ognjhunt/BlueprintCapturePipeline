from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline import unitree_lerobot_policy_command_adapter as adapter

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_unitree_lerobot_adapter_blocks_without_command_policy_or_frame(tmp_path: Path) -> None:
    response, exit_code = adapter.run_unitree_lerobot_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "task_prompt": "Make contact with the light object.",
            }
        },
        command=None,
        policy_path=str(tmp_path / "missing-policy"),
        source_root=None,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["unitree_lerobot_policy_action_command_ran"] is False
    assert response["model_ran"] is False
    assert (
        "set_BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_COMMAND_to_runnable_unitree_lerobot_policy_command"
        in response["blockers"]
    )
    assert (
        "set_BLUEPRINT_UNITREE_LEROBOT_MANIPULATION_CHECKPOINT_to_trained_unitree_lerobot_policy_path_or_repo_id"
        in response["blockers"]
    )
    assert "blocked_missing_policy_visual_observation_frame" in response["blockers"]
    assert response["claim_boundary"]["unitree_hand_manipulation_policy_used"] is False
    assert response["claim_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False


def test_unitree_lerobot_adapter_runs_command_and_redacts_runner_response(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"not-an-image-but-existing-policy-frame")
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    runner = tmp_path / "fake_unitree_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "payload = json.loads(sys.stdin.read() or '{}')",
                "assert payload['observation']['task_id'] == 'contact_or_push_light_object'",
                "response = {",
                "    'unitree_lerobot_policy_action_command_ran': True,",
                "    'action_chunk': [0.1, 0.2, 0.3],",
                "    'secret_token': 'do-not-persist',",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_lerobot_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
                "object_state": {"position": [0.36, -0.65, 0.27]},
            }
        },
        command=f"{sys.executable} {runner}",
        policy_path=str(policy_dir),
        source_root=None,
        timeout_seconds=5,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["policy_id"] == adapter.POLICY_ID
    assert response["unitree_lerobot_policy_action_command_ran"] is True
    assert response["model_ran"] is True
    assert response["action"]["action_type"] == "manipulation_contact"
    assert response["action"]["unitree_lerobot_action_chunk_present"] is True
    assert response["runner_response_redacted"]["secret_token"] == "<redacted>"
    assert response["claim_boundary"]["unitree_hand_manipulation_policy_used"] is True
    assert response["claim_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False


def test_unitree_lerobot_adapter_requires_runner_action_for_fresh_model_claim(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"existing-policy-frame")
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    runner = tmp_path / "fake_empty_unitree_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps({'status': 'ok'}))",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_lerobot_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
                "object_state": {"position": [0.36, -0.65, 0.27]},
            }
        },
        command=f"{sys.executable} {runner}",
        policy_path=str(policy_dir),
        source_root=None,
        timeout_seconds=5,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["unitree_lerobot_policy_action_command_ran"] is False
    assert response["model_ran"] is False
    assert any(
        blocker.startswith("blocked_unitree_lerobot_policy_command_failed")
        for blocker in response["blockers"]
    )


def test_unitree_lerobot_adapter_replays_provider_output_without_fresh_claim(
    tmp_path: Path,
) -> None:
    provider_output = tmp_path / "unitree_lerobot_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "schema_version": adapter.SCHEMA_VERSION,
                "status": "completed",
                "unitree_lerobot_policy_action_command_ran": True,
                "action": {
                    "action_type": "manipulation_contact",
                    "target_object_id": "blueprint_light_object",
                    "waypoint": [0.54, -0.65, 0.79],
                },
            }
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_lerobot_policy(
        payload={"observation": {"task_id": "contact_or_push_light_object"}},
        command=None,
        policy_path=None,
        source_root=None,
        provider_output=provider_output,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["policy_id"] == "unitree_lerobot_g1_policy_provider_replay"
    assert response["fresh_unitree_lerobot_model_executed_this_invocation"] is False
    assert response["provider_output_replay_used"] is True
    assert response["action"]["action_type"] == "manipulation_contact"
    assert response["claim_boundary"]["provider_output_replay_used"] is True
    assert (
        response["claim_boundary"][
            "provider_output_replay_is_not_fresh_per_request_model_inference"
        ]
        is True
    )


def test_unitree_lerobot_adapter_replay_rejects_untrusted_schema(
    tmp_path: Path,
) -> None:
    provider_output = tmp_path / "unitree_lerobot_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "schema_version": "unitree_lerobot_provider_output.v1",
                "status": "completed",
                "unitree_lerobot_policy_action_command_ran": True,
                "action": {
                    "action_type": "manipulation_contact",
                    "target_object_id": "blueprint_light_object",
                },
            }
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_lerobot_policy(
        payload={"observation": {"task_id": "contact_or_push_light_object"}},
        command=None,
        policy_path=None,
        source_root=None,
        provider_output=provider_output,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["provider_output_replay_used"] is True
    assert "blocked_unitree_lerobot_provider_output_schema_not_trusted" in response["blockers"]
    assert response["unitree_lerobot_policy_action_command_ran"] is False
    assert response["model_ran"] is False


def test_unitree_lerobot_adapter_replay_requires_provider_action_and_specific_proof(
    tmp_path: Path,
) -> None:
    provider_output = tmp_path / "unitree_lerobot_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "schema_version": adapter.SCHEMA_VERSION,
                "status": "completed",
                "model_ran": True,
                "task_id": "contact_or_push_light_object",
            }
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_lerobot_policy(
        payload={"observation": {"task_id": "contact_or_push_light_object"}},
        command=None,
        policy_path=None,
        source_root=None,
        provider_output=provider_output,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["unitree_lerobot_policy_action_command_ran"] is False
    assert response["model_ran"] is False
    assert "blocked_unitree_lerobot_provider_output_missing_model_execution_proof" in response[
        "blockers"
    ]
    assert "blocked_unitree_lerobot_provider_output_missing_action" in response["blockers"]


def test_unitree_lerobot_adapter_main_writes_blocked_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps({"observation": {"task_id": "approach_target"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_INPUT", str(input_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(output_path))

    exit_code = adapter.main(["--policy-path", str(tmp_path / "missing")])

    assert exit_code == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["policy_id"] == adapter.POLICY_ID
    captured = json.loads(capsys.readouterr().out)
    assert captured["raw_credentials_written_to_artifacts"] is False
