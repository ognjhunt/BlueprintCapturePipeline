from __future__ import annotations

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import unitree_groot_n17_sonic_policy_server_command as command


def _state_fields() -> dict[str, list[float]]:
    return {
        "left_leg": [0.0] * 6,
        "right_leg": [0.0] * 6,
        "waist": [0.0] * 3,
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "left_hand": [0.0] * 7,
        "right_hand": [0.0] * 7,
        "projected_gravity": [0.0, 0.0, -1.0],
    }


class _FakePolicyClient:
    last_observation = None

    def __init__(self, *, host, port, timeout_ms, api_token, strict):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.strict = strict

    def get_action(self, observation):
        _FakePolicyClient.last_observation = observation
        return (
            {
                "action.motion_token": np.array([[0.1, 0.2]], dtype=np.float32),
                "left_hand_joints": np.arange(7, dtype=np.float32),
                "right_hand_joints": np.arange(7, dtype=np.float32) + 10,
            },
            {"server_secret_token": "do-not-write"},
        )

    def close(self):
        pass


def test_policy_server_command_builds_sonic_observation_and_normalizes_action(
    tmp_path,
) -> None:
    frame = tmp_path / "ego.png"
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(frame)

    response, exit_code = command.run_policy_server_command(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "visual_observation": {"camera_frame_path": str(frame)},
                "unitree_g1_sonic_state": _state_fields(),
                "task_description": "push the light object",
            }
        },
        policy_server_url="tcp://policy.local:5559",
        policy_client_factory=_FakePolicyClient,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["selected_candidate_id"] == command.POLICY_ID
    assert response["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert response["unitree_policy_action_command_ran"] is True
    assert response["openvla_policy_action_command_ran"] is False
    assert response["policy_server_host"] == "policy.local"
    assert response["policy_server_port"] == 5559
    assert response["observation_metadata"]["state_keys"] == list(command.REQUIRED_STATE_KEYS)
    assert response["action"]["action_type"] == "unitree_g1_sonic_latent_action_chunk"
    assert response["action"]["unitree_groot_n17_sonic_action_chunk_present"] is True
    assert response["action"]["unitree_g1_sonic_control_fields"] == [
        "left_hand_joints",
        "motion_token",
        "right_hand_joints",
    ]
    assert len(response["action"]["action_chunk"]) == 16
    assert response["policy_server_info_redacted"]["server_secret_token"] == "<redacted>"
    assert response["claim_boundary"]["generated_world_rank_fidelity_result_proven"] is False
    assert response["claim_boundary"]["generated_world_policy_evaluation_scope_proven"] is False

    groot_observation = _FakePolicyClient.last_observation
    assert groot_observation["video"]["ego_view"].shape == (1, 1, 3, 4, 3)
    assert groot_observation["video"]["ego_view"].dtype == np.uint8
    assert groot_observation["state"]["left_arm"].shape == (1, 1, 7)
    assert groot_observation["state"]["left_arm"].dtype == np.float32
    assert groot_observation["language"]["annotation.human.task_description"] == [
        ["push the light object"]
    ]


def test_policy_server_command_blocks_on_missing_sonic_state(tmp_path) -> None:
    frame = tmp_path / "ego.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(frame)

    response, exit_code = command.run_policy_server_command(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "camera_frame_path": str(frame),
            }
        },
        policy_server_host="127.0.0.1",
        policy_server_port=5550,
        policy_client_factory=_FakePolicyClient,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert "blocked_missing_unitree_g1_sonic_state_fields" in response["blockers"]
    assert response["claim_boundary"]["policy_server_command_is_not_model_proof_when_blocked"]


def test_policy_server_command_allows_contract_probe_state_for_sim_attempt(tmp_path) -> None:
    frame = tmp_path / "ego.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(frame)

    response, exit_code = command.run_policy_server_command(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "camera_frame_path": str(frame),
                "unitree_g1_sonic_state": _state_fields(),
                "unitree_g1_sonic_state_source": "simulated_mujoco_contract_probe_zero_state",
            }
        },
        policy_server_url="tcp://127.0.0.1:5550",
        groot_root=str(tmp_path),
        policy_client_factory=_FakePolicyClient,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert response["observation_metadata"][
        "unitree_g1_sonic_state_source_is_contract_probe"
    ] is True
    assert response["claim_boundary"][
        "simulated_or_contract_probe_state_does_not_prove_real_robot_state"
    ] is True


def test_policy_server_command_blocks_when_server_dependency_or_server_fails(
    tmp_path,
) -> None:
    frame = tmp_path / "ego.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(frame)

    def failing_factory(**_kwargs):
        raise ModuleNotFoundError("No module named 'zmq'")

    response, exit_code = command.run_policy_server_command(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "camera_frame_path": str(frame),
                "unitree_g1_sonic_state": _state_fields(),
            }
        },
        policy_server_url="tcp://127.0.0.1:5550",
        policy_client_factory=failing_factory,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["error_type"] == "ModuleNotFoundError"
    assert (
        "blocked_unitree_groot_n17_sonic_policy_server_command_failed:ModuleNotFoundError"
        in response["blockers"]
    )
    assert response["raw_credentials_written_to_artifacts"] is False
