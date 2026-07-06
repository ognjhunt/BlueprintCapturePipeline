from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.lerobot_torch_policy_adapter import (
    GPU_RUNTIME_CONTRACT_SCHEMA_VERSION,
    GROOT_LIBERO_CHECKPOINT_REPO_ID,
    GROOT_LIBERO_INTEGRATION_LABEL,
    _claim_boundary,
    action_semantics_contract,
    build_gpu_runtime_contract,
    main as adapter_main,
    project_action_to_7d,
    resolve_visual_feature_bindings,
    visual_feature_layout,
)


def test_visual_feature_layout_accepts_hwc_groot_libero_shape() -> None:
    layout = visual_feature_layout((256, 256, 3))
    assert layout["input_layout"] == "HWC"
    assert layout["channels"] == 3
    assert layout["height"] == 256
    assert layout["width"] == 256
    assert layout["tensor_layout"] == "CHW"


def test_visual_bindings_resolve_image_and_wrist_image_keys(tmp_path: Path) -> None:
    front = tmp_path / "front.png"
    wrist = tmp_path / "wrist.png"
    front.write_bytes(b"front")
    wrist.write_bytes(b"wrist")
    observation = {
        "visual_observation": {
            "camera_frame_paths": {
                "observation.images.image": str(front),
                "observation.images.wrist_image": str(wrist),
            }
        }
    }

    bindings = resolve_visual_feature_bindings(
        observation,
        ["observation.images.image", "observation.images.wrist_image"],
    )

    by_key = {row["feature_key"]: row for row in bindings}
    assert by_key["observation.images.image"]["source_path"] == str(front)
    assert by_key["observation.images.wrist_image"]["source_path"] == str(wrist)
    assert by_key["observation.images.image"]["available"] is True
    assert by_key["observation.images.wrist_image"]["available"] is True
    assert by_key["observation.images.image"][
        "shared_source_path_with_other_visual_feature"
    ] is False


def test_visual_bindings_mark_single_frame_fallback_as_shared(tmp_path: Path) -> None:
    frame = tmp_path / "scene.png"
    frame.write_bytes(b"scene")
    observation = {"visual_observation": {"camera_frame_path": str(frame)}}

    bindings = resolve_visual_feature_bindings(
        observation,
        ["observation.images.image", "observation.images.wrist_image"],
    )

    assert {row["source_path"] for row in bindings} == {str(frame)}
    assert all(row["available"] is True for row in bindings)
    assert all(row["used_single_frame_fallback"] is True for row in bindings)
    assert all(
        row["shared_source_path_with_other_visual_feature"] is True
        for row in bindings
    )


def test_libero_panda_action_projection_is_labeled_not_semantic_claim() -> None:
    semantics = action_semantics_contract(
        policy_type="groot",
        action_decode_transform="libero",
        embodiment_tag="libero_sim",
        use_relative_actions=False,
    )
    vector = project_action_to_7d(
        [0.25, -0.25, 0.0, 0.0, 0.0, 0.5, -2.0],
        previous_raw=[0.25, -0.25, 0.0, 0.0, 0.0, 0.5, -2.0],
        action_semantics=semantics,
    )

    assert semantics["integration_label"] == GROOT_LIBERO_INTEGRATION_LABEL
    assert semantics["source_action_semantics"] == "libero_panda_7d_action"
    assert semantics["projection_mode"] == (
        "libero_panda_direct_7d_to_blueprint_delta_ee"
    )
    assert semantics["meaningful_manipulator_scoring_requires"] == (
        "libero_panda_simulator_bridge_or_panda_task_evaluator"
    )
    assert vector[0] > 0.0
    assert vector[1] < 0.0

    boundary = _claim_boundary(semantics)
    assert boundary["libero_panda_groot_integration_proof_only"] is True
    assert boundary["panda_or_libero_task_success_proven"] is False
    assert boundary["meaningful_manipulator_scoring_proven"] is False
    assert boundary["blueprint_site_task_success_proven"] is False
    assert boundary["physical_robot_readiness_proven"] is False
    assert boundary["buyer_facing_deployment_claim_allowed"] is False


def test_groot_libero_gpu_runtime_contract_is_not_cpu_baseline() -> None:
    contract = build_gpu_runtime_contract(
        checkpoint=GROOT_LIBERO_CHECKPOINT_REPO_ID,
        device="cuda",
        policy_type="groot",
    )

    assert contract["requires_gpu_runtime"] is True
    assert contract["recommended_device"] == "cuda"
    assert contract["python_package_extra"] == "lerobot[groot]"
    assert contract["checkpoint_size_class"] == "large_12gb_plus"
    assert contract["model_card_size_gb_approx"] == 12.6
    assert contract["exact_groot_libero_checkpoint"] is True
    assert contract["not_cpu_baseline_lane"] is True


def test_runtime_contract_cli_does_not_load_model(capsys) -> None:
    exit_code = adapter_main(
        [
            "--checkpoint",
            GROOT_LIBERO_CHECKPOINT_REPO_ID,
            "--device",
            "cuda",
            "--print-runtime-contract",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == GPU_RUNTIME_CONTRACT_SCHEMA_VERSION
    assert payload["status"] == "configured"
    assert payload["integration_label"] == GROOT_LIBERO_INTEGRATION_LABEL
    assert payload["gpu_runtime_contract"]["requires_gpu_runtime"] is True
    assert payload["gpu_runtime_contract"]["python_package_extra"] == "lerobot[groot]"
    assert payload["claim_boundary"]["libero_panda_groot_integration_proof_only"] is True
    assert payload["claim_boundary"]["policy_command_ran"] is False
    assert payload["claim_boundary"]["real_torch_model_inference"] is False
    assert payload["claim_boundary"]["blueprint_site_task_success_proven"] is False
