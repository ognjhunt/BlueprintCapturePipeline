from __future__ import annotations

import json
import numpy as np
import pytest

from blueprint_pipeline import g1_sonic_motion_token_conversion as conversion


def _actions(frame_count: int = 12) -> np.ndarray:
    base = np.arange(len(conversion.SOURCE_ACTION_JOINT_NAMES), dtype=np.float32)
    return np.stack([base + frame * 0.01 for frame in range(frame_count)])


def _source_metadata() -> tuple[dict, dict, list[dict], list[dict]]:
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1",
        "total_episodes": 200,
        "total_frames": 97_766,
        "fps": 50,
        "splits": {"train": "0:100"},
        "features": {
            "action": {
                "shape": [43],
                "names": list(conversion.SOURCE_ACTION_JOINT_NAMES),
            },
            "observation.state": {
                "shape": [43],
                "names": list(conversion.SOURCE_ACTION_JOINT_NAMES),
            },
            "observation.images.ego_view": {
                "shape": [480, 640, 3],
                "video_info": {
                    "video.width": 640,
                    "video.height": 480,
                    "video.fps": 50.0,
                    "video.codec": "h264",
                },
            },
        },
    }
    modality = {
        "state": {
            key: {} for key in (
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            )
        },
        "action": {
            key: {} for key in (
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            )
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"}
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    tasks = [{"task_index": 3, "task": conversion.SOURCE_TASK_DESCRIPTION}]
    lengths = [489] * 166 + [488] * 34
    episodes = [
        {
            "episode_index": index,
            "tasks": [conversion.SOURCE_TASK_DESCRIPTION],
            "length": length,
        }
        for index, length in enumerate(lengths)
    ]
    assert sum(lengths) == 97_766
    return info, modality, tasks, episodes


def test_training_source_audit_blocks_unlicensed_unattested_candidate() -> None:
    info, modality, tasks, episodes = _source_metadata()

    report = conversion.audit_training_source_metadata(
        info=info,
        modality=modality,
        tasks=tasks,
        episodes=episodes,
        declared_license=None,
        fixed_base_upright_attestation=None,
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        "source_dataset_license_undeclared_or_unapproved",
        "source_dataset_fixed_base_upright_attestation_unproven",
    ]
    assert all(report["metadata_checks"].values())


def test_training_source_audit_admits_only_exact_rights_and_pose_evidence() -> None:
    info, modality, tasks, episodes = _source_metadata()
    attestation = {
        "status": "passed",
        "dataset_repo": conversion.SOURCE_DATASET_REPO,
        "dataset_revision": conversion.SOURCE_DATASET_REVISION,
        "fixed_base": True,
        "upright": True,
        "artifact_sha256": "a" * 64,
    }

    report = conversion.audit_training_source_metadata(
        info=info,
        modality=modality,
        tasks=tasks,
        episodes=episodes,
        declared_license="CC-BY-4.0",
        fixed_base_upright_attestation=attestation,
    )

    assert report["status"] == "admitted_for_sonic_training_materialization"
    assert report["blockers"] == []
    assert report["license_approved"] is True
    assert report["fixed_base_upright_attestation_valid"] is True


def test_training_source_audit_rejects_task_or_inventory_drift() -> None:
    info, modality, tasks, episodes = _source_metadata()
    tasks[0]["task"] = "Open a generic door."
    episodes.pop()

    report = conversion.audit_training_source_metadata(
        info=info,
        modality=modality,
        tasks=tasks,
        episodes=episodes,
        declared_license="CC-BY-4.0",
        fixed_base_upright_attestation={
            "status": "passed",
            "dataset_repo": conversion.SOURCE_DATASET_REPO,
            "dataset_revision": conversion.SOURCE_DATASET_REVISION,
            "fixed_base": True,
            "upright": True,
            "artifact_sha256": "b" * 64,
        },
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == ["source_dataset_metadata_contract_invalid"]
    assert report["metadata_checks"]["episode_inventory_exact"] is False
    assert report["metadata_checks"]["single_exact_task"] is False


def test_training_source_cli_writes_hash_bound_blocked_audit(tmp_path) -> None:
    info, modality, tasks, episodes = _source_metadata()
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (metadata / "modality.json").write_text(
        json.dumps(modality), encoding="utf-8"
    )
    (metadata / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8"
    )
    (metadata / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    output = tmp_path / "audit.json"

    exit_code = conversion.main(
        ["--metadata-dir", str(metadata), "--out", str(output)]
    )

    assert exit_code == 1
    report = json.loads(output.read_text())
    assert report["status"] == "blocked"
    assert set(report["source_metadata_artifacts"]) == {
        "info",
        "modality",
        "tasks",
        "episodes",
    }
    assert all(
        len(value["sha256"]) == 64
        for value in report["source_metadata_artifacts"].values()
    )


def _with_non_finite_action(actions, names):
    actions = actions.copy()
    actions[0, 0] = np.nan
    return actions, names


def test_builds_official_g1_mode_windows_in_isaaclab_order() -> None:
    actions = _actions()
    anchors = conversion.fixed_upright_root_anchor_rotations(
        len(actions),
        fixed_base_upright_attested=True,
    )
    inputs, left_hand, right_hand = conversion.build_sonic_encoder_inputs(
        actions,
        action_joint_names=conversion.SOURCE_ACTION_JOINT_NAMES,
        root_anchor_rotations_6d=anchors,
    )

    source_index = {
        name: index
        for index, name in enumerate(conversion.SOURCE_ACTION_JOINT_NAMES)
    }
    expected_body_frame_zero = [
        actions[0, source_index[name]]
        for name in conversion.ISAACLAB_BODY_JOINT_NAMES
    ]
    assert inputs.shape == (len(actions), conversion.ENCODER_INPUT_DIM)
    assert inputs[0, conversion.BODY_POSITION_SLICE][:29].tolist() == pytest.approx(
        expected_body_frame_zero
    )
    assert inputs[0, conversion.BODY_POSITION_SLICE][29:58].tolist() == pytest.approx(
        [value + 0.05 for value in expected_body_frame_zero]
    )
    assert inputs[0, conversion.ROOT_ANCHOR_ROTATION_SLICE][:6].tolist() == list(
        conversion.IDENTITY_ROTATION_6D
    )
    assert left_hand[0].tolist() == pytest.approx(actions[0, 22:29].tolist())
    assert right_hand[0].tolist() == pytest.approx(actions[0, 36:43].tolist())


def test_fixed_root_identity_requires_explicit_attestation() -> None:
    with pytest.raises(
        ValueError,
        match="g1_sonic_conversion_root_pose_missing_without_attestation",
    ):
        conversion.fixed_upright_root_anchor_rotations(
            10,
            fixed_base_upright_attested=False,
        )


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (
            lambda actions, names: (actions, (*names[:-1], "unknown_joint")),
            "g1_sonic_conversion_action_joint_inventory_mismatch",
        ),
        (
            _with_non_finite_action,
            "g1_sonic_conversion_action_trajectory_non_finite",
        ),
    ],
)
def test_rejects_invalid_source_trajectory(mutator, error: str) -> None:
    actions, names = mutator(_actions(), conversion.SOURCE_ACTION_JOINT_NAMES)
    anchors = conversion.fixed_upright_root_anchor_rotations(
        len(actions),
        fixed_base_upright_attested=True,
    )
    with pytest.raises(ValueError, match=error):
        conversion.build_sonic_encoder_inputs(
            actions,
            action_joint_names=names,
            root_anchor_rotations_6d=anchors,
        )


def test_conversion_emits_78d_training_actions_without_upgrading_claims() -> None:
    actions = _actions()
    anchors = conversion.fixed_upright_root_anchor_rotations(
        len(actions),
        fixed_base_upright_attested=True,
    )

    def fake_encoder(inputs: np.ndarray) -> np.ndarray:
        tokens = np.zeros((len(inputs), conversion.MOTION_TOKEN_DIM), dtype=np.float32)
        tokens[:, 0] = inputs[:, conversion.BODY_POSITION_SLICE.start]
        return tokens

    sonic_actions, report = conversion.convert_to_sonic_actions(
        actions,
        action_joint_names=conversion.SOURCE_ACTION_JOINT_NAMES,
        root_anchor_rotations_6d=anchors,
        encoder=fake_encoder,
    )

    assert sonic_actions.shape == (len(actions), 78)
    np.testing.assert_allclose(sonic_actions[:, 64:71], actions[:, 22:29])
    np.testing.assert_allclose(sonic_actions[:, 71:78], actions[:, 36:43])
    assert report["status"] == "converted_training_actions"
    assert report["motion_token_unique_rows_rounded_6"] == len(actions)
    assert report["claim_boundary"] == {
        "converted_actions_are_training_inputs_only": True,
        "converted_actions_are_not_a_trained_checkpoint": True,
        "converted_actions_are_not_task_qualification": True,
        "converted_actions_are_not_episode_success": True,
        "root_pose_source_must_be_attested_separately": True,
    }
    assert report["source_provenance"] == {
        "source_type": "public_dataset_candidate",
        "repo": conversion.SOURCE_DATASET_REPO,
        "revision": conversion.SOURCE_DATASET_REVISION,
    }


def test_conversion_preserves_owned_source_provenance() -> None:
    actions = _actions()
    anchors = conversion.fixed_upright_root_anchor_rotations(
        len(actions), fixed_base_upright_attested=True
    )
    provenance = {
        "source_type": "blueprint_generated_owned_microwave_reach_seed",
        "trajectory_sha256": "a" * 64,
    }

    _, report = conversion.convert_to_sonic_actions(
        actions,
        action_joint_names=conversion.SOURCE_ACTION_JOINT_NAMES,
        root_anchor_rotations_6d=anchors,
        encoder=lambda inputs: np.zeros(
            (len(inputs), conversion.MOTION_TOKEN_DIM), dtype=np.float32
        ),
        source_provenance=provenance,
    )

    assert report["source_provenance"] == provenance


def test_conversion_rejects_source_provenance_without_type() -> None:
    actions = _actions()
    anchors = conversion.fixed_upright_root_anchor_rotations(
        len(actions), fixed_base_upright_attested=True
    )
    with pytest.raises(
        ValueError, match="g1_sonic_conversion_source_provenance_type_missing"
    ):
        conversion.convert_to_sonic_actions(
            actions,
            action_joint_names=conversion.SOURCE_ACTION_JOINT_NAMES,
            root_anchor_rotations_6d=anchors,
            encoder=lambda inputs: np.zeros(
                (len(inputs), conversion.MOTION_TOKEN_DIM), dtype=np.float32
            ),
            source_provenance={"trajectory_sha256": "a" * 64},
        )


def test_builds_only_current_unitree_g1_sonic_training_columns() -> None:
    actions = _actions()

    def fake_encoder(inputs: np.ndarray) -> np.ndarray:
        return np.zeros((len(inputs), conversion.MOTION_TOKEN_DIM), dtype=np.float32)

    columns, report = conversion.build_training_columns(
        actions,
        action_joint_names=conversion.SOURCE_ACTION_JOINT_NAMES,
        encoder=fake_encoder,
        fixed_base_upright_attested=True,
    )

    assert {key: value.shape for key, value in columns.items()} == {
        "observation.projected_gravity": (len(actions), 3),
        "action.motion_token": (len(actions), 64),
        "teleop.left_hand_joints": (len(actions), 7),
        "teleop.right_hand_joints": (len(actions), 7),
    }
    np.testing.assert_allclose(
        columns["observation.projected_gravity"],
        [[0.0, 0.0, -1.0]] * len(actions),
    )
    assert report["training_schema"]["embodiment_tag"] == "UNITREE_G1_SONIC"
    assert report["claim_boundary"]["recorded_token_replay_equivalence_proven"] is False


def test_training_modality_matches_current_registered_sonic_keys() -> None:
    modality = conversion.unitree_g1_sonic_training_modality()
    assert list(modality["state"]) == [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
        "projected_gravity",
    ]
    assert list(modality["action"]) == [
        "motion_token",
        "left_hand_joints",
        "right_hand_joints",
    ]
    assert modality["video"] == {
        "ego_view": {"original_key": "observation.images.ego_view"}
    }


def test_encoder_loader_checks_exact_sha_before_optional_runtime_import(
    tmp_path,
) -> None:
    model = tmp_path / "model_encoder.onnx"
    model.write_bytes(b"not-the-reviewed-encoder")
    with pytest.raises(
        ValueError,
        match="g1_sonic_conversion_encoder_sha256_mismatch",
    ):
        conversion.load_onnx_encoder(model)
