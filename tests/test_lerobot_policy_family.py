from __future__ import annotations

import json
from pathlib import Path


from blueprint_pipeline.lerobot_policy_family import (
    ADAPTER_SCHEMA_VERSION,
    SCRIPTED_PICK_PLACE_FAMILY_ID,
    SCRIPTED_PICK_PLACE_TYPE,
    create_scripted_baseline_checkpoint,
    load_lerobot_policy_checkpoint,
    main as adapter_main,
)


def _observation(*, grasped: bool = False, opening: float = 0.06) -> dict:
    return {
        "task_id": "place_return_in_bin",
        "end_effector": {
            "position_xyz": [0.0, 0.0, 0.66],
            "yaw_rad": 0.0,
            "gripper_opening_m": opening,
        },
        "objects": [
            {
                "object_id": "target_item",
                "role": "commanded_target",
                "position_xyz": [0.10, -0.12, 0.45],
                "grasped": grasped,
            },
            {
                "object_id": "distractor_item",
                "role": "distractor",
                "position_xyz": [-0.14, -0.10, 0.45],
                "grasped": False,
            },
        ],
        "goal_zone": {
            "zone_id": "goal",
            "center_xyz": [0.02, 0.16, 0.45],
            "radius_m": 0.06,
        },
    }


def test_scripted_baseline_checkpoint_is_lerobot_format_and_cpu_loadable(
    tmp_path: Path,
) -> None:
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["type"] == SCRIPTED_PICK_PLACE_TYPE
    assert config["output_features"]["action"]["shape"] == [7]

    loaded = load_lerobot_policy_checkpoint(checkpoint)
    assert loaded.blockers == []
    assert loaded.cpu_loadable is True
    assert loaded.requires_torch_runtime is False
    assert loaded.family_id == SCRIPTED_PICK_PLACE_FAMILY_ID
    assert loaded.policy is not None
    assert len(loaded.checkpoint_sha256) == 64

    manifest = loaded.manifest()
    assert manifest["claim_boundary"]["checkpoint_load_is_not_task_success"] is True
    assert (
        manifest["claim_boundary"]["scripted_baseline_is_not_a_learned_policy"] is True
    )


def test_checkpoint_sha_changes_with_weights(tmp_path: Path) -> None:
    first = load_lerobot_policy_checkpoint(
        create_scripted_baseline_checkpoint(tmp_path / "a")
    )
    second = load_lerobot_policy_checkpoint(
        create_scripted_baseline_checkpoint(tmp_path / "b", overrides={"gain": 0.5})
    )
    assert first.checkpoint_sha256 != second.checkpoint_sha256


def test_loader_fails_closed_on_malformed_checkpoints(tmp_path: Path) -> None:
    missing = load_lerobot_policy_checkpoint(tmp_path / "nope")
    assert "checkpoint_dir_missing" in missing.blockers
    assert missing.cpu_loadable is False

    no_type = tmp_path / "no_type"
    no_type.mkdir()
    (no_type / "config.json").write_text("{}", encoding="utf-8")
    assert "lerobot_config_type_missing" in load_lerobot_policy_checkpoint(no_type).blockers

    no_weights = tmp_path / "no_weights"
    no_weights.mkdir()
    (no_weights / "config.json").write_text(
        json.dumps({"type": SCRIPTED_PICK_PLACE_TYPE}), encoding="utf-8"
    )
    assert (
        "scripted_policy_weights_missing"
        in load_lerobot_policy_checkpoint(no_weights).blockers
    )


def test_learned_lerobot_type_requires_torch_runtime_never_emulated(
    tmp_path: Path,
) -> None:
    learned = tmp_path / "act_policy"
    learned.mkdir()
    (learned / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
    loaded = load_lerobot_policy_checkpoint(learned)
    assert loaded.cpu_loadable is False
    assert loaded.requires_torch_runtime is True
    assert "policy_type_requires_torch_inference_runtime" in loaded.blockers
    assert loaded.policy is None


def test_action_chunk_is_deterministic_7d_and_truncates_on_grip_transition(
    tmp_path: Path,
) -> None:
    loaded = load_lerobot_policy_checkpoint(
        create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    )
    assert loaded.policy is not None
    observation = _observation()
    first = loaded.policy.select_action_chunk(observation, chunk_size=25)
    second = loaded.policy.select_action_chunk(observation, chunk_size=25)
    assert first == second
    assert len(first) == 25
    for action in first:
        assert len(action["action_7d"]) == 7

    # At the grasp pose with the gripper open, the first action closes the
    # gripper; the projection cannot verify the grasp, so the remainder of the
    # chunk must settle in place instead of optimistically transporting.
    at_grasp = _observation()
    at_grasp["end_effector"]["position_xyz"] = [0.10, -0.12, 0.462]
    chunk = loaded.policy.select_action_chunk(at_grasp, chunk_size=25)
    closing = [action for action in chunk if action["gripper_command"] == 0.0]
    assert closing, "expected a close command at the grasp pose"
    settle = [action for action in chunk if action.get("settle_after_grip_transition")]
    assert settle, "grip transition must settle the remainder of the chunk"
    for action in settle:
        assert action["delta_xyz_m"] == [0.0, 0.0, 0.0]


def test_adapter_cli_contract_roundtrip(tmp_path: Path, monkeypatch, capsys) -> None:
    checkpoint = create_scripted_baseline_checkpoint(tmp_path / "ckpt")
    obs_path = tmp_path / "obs.json"
    out_path = tmp_path / "action.json"
    obs_path.write_text(json.dumps({"observation": _observation()}), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_INPUT", str(obs_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(out_path))

    exit_code = adapter_main(["--checkpoint", str(checkpoint)])
    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ADAPTER_SCHEMA_VERSION
    assert payload["status"] == "completed"
    assert payload["policy_id"] == SCRIPTED_PICK_PLACE_FAMILY_ID
    assert len(payload["action_chunk"]) >= 1
    assert len(payload["action"]["action_7d"]) == 7
    boundary = payload["claim_boundary"]
    assert boundary["single_action_is_not_episode_success"] is True
    assert boundary["task_success_proven"] is False
    capsys.readouterr()


def test_adapter_cli_fails_closed_for_non_loadable_checkpoint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    learned = tmp_path / "act_policy"
    learned.mkdir()
    (learned / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
    obs_path = tmp_path / "obs.json"
    out_path = tmp_path / "action.json"
    obs_path.write_text(json.dumps({"observation": _observation()}), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_INPUT", str(obs_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(out_path))

    exit_code = adapter_main(["--checkpoint", str(learned)])
    assert exit_code == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert "policy_type_requires_torch_inference_runtime" in payload["blockers"]
    capsys.readouterr()


def test_scripted_policy_never_registered_in_production_candidates() -> None:
    from blueprint_pipeline.unitree_lerobot_policy_runtime import (
        UNITREE_ACTION_COMMAND_CANDIDATES,
    )

    candidate_ids = {
        candidate.get("candidate_id") for candidate in UNITREE_ACTION_COMMAND_CANDIDATES
    }
    assert SCRIPTED_PICK_PLACE_FAMILY_ID not in candidate_ids
    assert SCRIPTED_PICK_PLACE_TYPE not in candidate_ids
