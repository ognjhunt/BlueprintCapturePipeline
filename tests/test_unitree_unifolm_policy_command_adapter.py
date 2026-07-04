from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import unitree_unifolm_policy_command_adapter as adapter


pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_unitree_unifolm_adapter_blocks_without_command_checkpoint_or_frame(
    tmp_path: Path,
) -> None:
    response, exit_code = adapter.run_unitree_unifolm_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "task_prompt": "Make contact with the light object.",
            }
        },
        mode="vla",
        command=None,
        checkpoint=str(tmp_path / "missing-unifolm.pt"),
        source_root=None,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["policy_id"] == "unitree_unifolm_vla_policy"
    assert response["unitree_unifolm_policy_action_command_ran"] is False
    assert response["model_ran"] is False
    assert (
        "set_BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND_to_runnable_unitree_unifolm_policy_command"
        in response["blockers"]
    )
    assert (
        "set_BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT_to_unitree_unifolm_checkpoint_path_or_repo_id"
        in response["blockers"]
    )
    assert (
        "set_BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT_to_unitree_unifolm_vlm_checkpoint_path_or_repo_id"
        in response["blockers"]
    )
    assert "blocked_missing_policy_visual_observation_frame" in response["blockers"]
    assert response["claim_boundary"]["unitree_hand_manipulation_policy_used"] is False
    assert response["claim_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False


def test_unitree_unifolm_adapter_runs_command_and_redacts_runner_response(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"not-an-image-but-existing-policy-frame")
    checkpoint = tmp_path / "pytorch_model.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    runner = tmp_path / "fake_unifolm_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "payload = json.loads(sys.stdin.read() or '{}')",
                "assert payload['observation']['task_id'] == 'contact_or_push_light_object'",
                "response = {",
                "    'unitree_unifolm_policy_action_command_ran': True,",
                "    'action_chunk': [0.1, 0.2, 0.3],",
                "    'secret_token': 'do-not-persist',",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_unifolm_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
                "object_state": {"position": [0.36, -0.65, 0.27]},
            }
        },
        mode="wma",
        command=f"{sys.executable} {runner}",
        checkpoint=str(checkpoint),
        source_root=None,
        timeout_seconds=5,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["policy_id"] == "unitree_unifolm_wma_policy"
    assert response["unitree_unifolm_policy_action_command_ran"] is True
    assert response["model_ran"] is True
    assert response["action"]["action_type"] == "manipulation_contact"
    assert response["action"]["unitree_unifolm_action_chunk_present"] is True
    assert response["runner_response_redacted"]["secret_token"] == "<redacted>"
    assert response["claim_boundary"]["unitree_hand_manipulation_policy_used"] is True
    assert response["claim_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False


def test_unitree_unifolm_adapter_forwards_source_root_aliases(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"existing-policy-frame")
    checkpoint = tmp_path / "pytorch_model.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    source_root = tmp_path / "unifolm-vla"
    source_root.mkdir()
    runner = tmp_path / "source_root_probe_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "assert os.environ['BLUEPRINT_UNITREE_UNIFOLM_POLICY_SOURCE_ROOT'].endswith('unifolm-vla')",
                "assert os.environ['BLUEPRINT_UNITREE_UNIFOLM_SOURCE_ROOT'].endswith('unifolm-vla')",
                "response = {'action': {'action_type': 'stop'}}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_unifolm_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
            }
        },
        mode="wma",
        command=f"{sys.executable} {runner}",
        checkpoint=str(checkpoint),
        source_root=str(source_root),
        timeout_seconds=5,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["action"]["action_type"] == "stop"


def test_unitree_unifolm_adapter_preserves_runner_blockers_on_nonzero_exit(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"existing-policy-frame")
    checkpoint = tmp_path / "pytorch_model.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    runner = tmp_path / "blocked_unifolm_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "response = {",
                "    'status': 'blocked',",
                "    'model_ran': False,",
                "    'blockers': ['blocked_unitree_unifolm_vla_server_call_failed:URLError'],",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
                "sys.exit(2)",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_unifolm_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
            }
        },
        mode="vla",
        command=f"{sys.executable} {runner}",
        checkpoint=str(checkpoint),
        vlm_checkpoint=str(checkpoint),
        source_root=None,
        timeout_seconds=5,
    )

    assert exit_code == 1
    assert response["status"] == "failed"
    assert response["model_ran"] is False
    assert response["runner_response_redacted"]["status"] == "blocked"
    assert response["blockers"] == [
        "blocked_unitree_unifolm_vla_server_call_failed:URLError"
    ]


def test_unitree_unifolm_vla_main_accepts_policy_checkpoint_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"existing-policy-frame")
    checkpoint = tmp_path / "pytorch_model.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    runner = tmp_path / "fake_unifolm_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "payload = json.loads(sys.stdin.read() or '{}')",
                "assert payload['observation']['task_id'] == 'contact_or_push_light_object'",
                "assert os.environ['BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT'].endswith('pytorch_model.pt')",
                "response = {'action_chunk': [0.1, 0.2, 0.3]}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "adapter_output.json"
    input_path.write_text(
        json.dumps(
            {
                "observation": {
                    "task_id": "contact_or_push_light_object",
                    "visual_observation": {"camera_frame_path": str(frame)},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_INPUT", str(input_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(output_path))
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_COMMAND", f"{sys.executable} {runner}")
    monkeypatch.delenv("BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT", raising=False)
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT", str(vlm))

    assert adapter.main(["--mode", "vla"]) == 0

    response = json.loads(output_path.read_text(encoding="utf-8"))
    assert response["status"] == "completed"
    assert response["policy_id"] == "unitree_unifolm_vla_policy"
    assert response["checkpoint_path"] == str(checkpoint)
    assert response["unitree_unifolm_policy_action_command_ran"] is True


def test_unitree_unifolm_adapter_replays_provider_output_without_fresh_claim(
    tmp_path: Path,
) -> None:
    provider_output = tmp_path / "unitree_unifolm_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "status": "completed",
                "unitree_unifolm_policy_action_command_ran": True,
                "unitree_unifolm_model_executed": True,
                "action": {
                    "action_type": "manipulation_contact",
                    "target_object_id": "blueprint_light_object",
                    "waypoint": [0.54, -0.65, 0.79],
                },
            }
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_unifolm_policy(
        payload={"observation": {"task_id": "contact_or_push_light_object"}},
        mode="vla",
        command=None,
        checkpoint=None,
        source_root=None,
        provider_output=provider_output,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["policy_id"] == "unitree_unifolm_vla_policy_provider_replay"
    assert response["unitree_unifolm_policy_action_command_ran"] is False
    assert response["fresh_unitree_unifolm_model_executed_this_invocation"] is False
    assert response["fresh_unitree_unifolm_policy_action_command_ran_this_invocation"] is False
    assert response["provider_output_replay_used"] is True
    assert response["provider_unitree_unifolm_policy_action_command_ran"] is True
    assert response["action"]["action_type"] == "manipulation_contact"
    assert response["claim_boundary"]["unitree_hand_manipulation_policy_used"] is False
    assert response["claim_boundary"]["provider_output_replay_used"] is True
    assert (
        response["claim_boundary"][
            "provider_output_replay_is_not_fresh_per_request_model_inference"
        ]
        is True
    )
    assert (
        response["claim_boundary"][
            "provider_output_replay_is_not_live_unitree_hand_policy_command"
        ]
        is True
    )


def test_unitree_unifolm_adapter_main_writes_blocked_output(
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

    exit_code = adapter.main(["--mode", "vla", "--checkpoint", str(tmp_path / "missing")])

    assert exit_code == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["policy_id"] == "unitree_unifolm_vla_policy"
    captured = json.loads(capsys.readouterr().out)
    assert captured["raw_credentials_written_to_artifacts"] is False
