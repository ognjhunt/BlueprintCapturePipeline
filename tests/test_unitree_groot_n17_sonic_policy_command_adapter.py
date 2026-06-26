from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline import unitree_groot_n17_sonic_policy_command_adapter as adapter


def test_groot_n17_sonic_adapter_blocks_without_command_checkpoint_or_frame(
    tmp_path: Path,
) -> None:
    response, exit_code = adapter.run_unitree_groot_n17_sonic_policy(
        payload={"observation": {"task_id": "contact_or_push_light_object"}},
        command=None,
        n17_checkpoint=str(tmp_path / "missing-n17"),
        sonic_checkpoint=str(tmp_path / "missing-sonic"),
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["policy_id"] == adapter.POLICY_ID
    assert response["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert response["unitree_policy_action_command_ran"] is False
    assert response["unitree_specific_manipulation_candidate_ran"] is False
    assert response["openvla_policy_action_command_ran"] is False
    assert (
        "set_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND_to_runnable_unitree_groot_n17_sonic_policy_command"
        in response["blockers"]
    )
    assert (
        "set_BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT_to_nvidia_groot_n17_checkpoint_or_repo_id"
        in response["blockers"]
    )
    assert (
        "set_BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT_to_finetuned_unitree_g1_sonic_checkpoint_or_repo_id"
        in response["blockers"]
    )
    assert "blocked_missing_policy_visual_observation_frame" in response["blockers"]
    assert response["claim_boundary"]["generated_world_rank_fidelity_result_proven"] is False
    assert response["claim_boundary"]["generated_world_policy_evaluation_scope_proven"] is False
    assert response["claim_boundary"]["non_ranking_operational_claim_proven"] is False


def test_groot_n17_sonic_adapter_runs_command_and_normalizes_action_chunk(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"existing-policy-frame")
    n17_checkpoint = tmp_path / "n17"
    n17_checkpoint.mkdir()
    sonic_checkpoint = tmp_path / "sonic"
    sonic_checkpoint.mkdir()
    runner = tmp_path / "fake_groot_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "payload = json.loads(sys.stdin.read() or '{}')",
                "assert payload['observation']['task_id'] == 'contact_or_push_light_object'",
                "response = {",
                "    'unitree_groot_n17_sonic_policy_action_command_ran': True,",
                "    'sonic_latent_action': [0.1, 0.2, 0.3],",
                "    'secret_token': 'do-not-persist',",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_groot_n17_sonic_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
            }
        },
        command=f"{sys.executable} {runner}",
        n17_checkpoint=str(n17_checkpoint),
        sonic_checkpoint=str(sonic_checkpoint),
        timeout_seconds=5,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["policy_id"] == adapter.POLICY_ID
    assert response["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert response["unitree_policy_action_command_ran"] is True
    assert response["unitree_specific_manipulation_candidate_ran"] is True
    assert response["openvla_policy_action_command_ran"] is False
    assert response["action"]["action_type"] == "unitree_g1_sonic_action_chunk"
    assert response["action"]["unitree_groot_n17_sonic_action_chunk_present"] is True
    assert response["runner_response_redacted"]["secret_token"] == "<redacted>"
    assert response["claim_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False


def test_groot_n17_sonic_adapter_preserves_child_command_blockers(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"existing-policy-frame")
    n17_checkpoint = tmp_path / "n17"
    n17_checkpoint.mkdir()
    sonic_checkpoint = tmp_path / "sonic"
    sonic_checkpoint.mkdir()
    runner = tmp_path / "blocked_groot_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "payload = {",
                "    'schema_version': 'unitree_groot_n17_sonic_policy_server_command.v1',",
                "    'status': 'blocked',",
                "    'blockers': ['set_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL_to_running_gr00t_policy_server'],",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(payload))",
                "raise SystemExit(2)",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_groot_n17_sonic_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
            }
        },
        command=f"{sys.executable} {runner}",
        n17_checkpoint=str(n17_checkpoint),
        sonic_checkpoint=str(sonic_checkpoint),
        timeout_seconds=5,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["child_command_blocked"] is True
    assert (
        "set_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL_to_running_gr00t_policy_server"
        in response["blockers"]
    )
    assert response["unitree_groot_n17_sonic_policy_action_command_ran"] is False


def test_groot_n17_sonic_adapter_allows_policy_server_client_without_sonic_deploy_assets(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"existing-policy-frame")
    runner = tmp_path / "unitree_groot_n17_sonic_policy_server_command_fake.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "payload = {",
                "    'schema_version': 'unitree_groot_n17_sonic_policy_server_command.v1',",
                "    'status': 'blocked',",
                "    'blockers': ['set_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL_to_running_gr00t_policy_server'],",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(payload))",
                "raise SystemExit(2)",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_groot_n17_sonic_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
            }
        },
        command=f"{sys.executable} {runner}",
        n17_checkpoint="nvidia/GR00T-N1.7-3B",
        sonic_checkpoint=str(tmp_path / "mac-local-sonic-assets-not-on-provider"),
        timeout_seconds=5,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert (
        "set_BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT_to_finetuned_unitree_g1_sonic_checkpoint_or_repo_id"
        not in response["blockers"]
    )
    assert response["g1_sonic_checkpoint_required_for_selected_command"] is False
    assert response["default_experimental_checkpoint_applied"] is True
    assert (
        "set_BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL_to_running_gr00t_policy_server"
        in response["blockers"]
    )


def test_groot_n17_sonic_adapter_provider_replay_does_not_count_as_fresh_command(
    tmp_path: Path,
) -> None:
    provider_output = tmp_path / "provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "schema_version": adapter.SCHEMA_VERSION,
                "status": "completed",
                "unitree_groot_n17_sonic_policy_action_command_ran": True,
                "action": {
                    "action_type": "unitree_g1_sonic_action_chunk",
                    "action_chunk": [0.1],
                },
            }
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_unitree_groot_n17_sonic_policy(
        payload={"observation": {"task_id": "contact_or_push_light_object"}},
        command=None,
        n17_checkpoint=None,
        sonic_checkpoint=None,
        provider_output=provider_output,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["provider_output_replay_used"] is True
    assert response["fresh_unitree_groot_n17_sonic_model_executed_this_invocation"] is False
    assert response["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert response["unitree_policy_action_command_ran"] is False
    assert (
        response["claim_boundary"][
            "provider_output_replay_is_not_fresh_per_request_model_inference"
        ]
        is True
    )
