from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_successor_cosmos import (
    GPUOffer,
    SuccessorContractError,
    assert_evaluator_eligible,
    build_action_controls,
    build_forward_dynamics_request,
    convert_droid_states_to_action_stream,
    droid_action_stream,
    select_gpu_offer,
    validate_artifact_path,
    validate_compute_admission,
    validate_droid_camera_metadata,
    validate_droid_action_stream,
    validate_droid_timestamps,
    validate_smoke_inventory_manifest,
)


def _actions(offset: float = 0.0) -> list[list[float]]:
    rows = []
    for index in range(16):
        rows.append(
            [
                offset + index * 0.0001,
                offset - index * 0.0002,
                offset + index * 0.0003,
                1.0,
                index * 0.0004,
                index * 0.0005,
                index * 0.0006,
                1.0,
                index * 0.0007,
                float(index % 2),
            ]
        )
    return rows


def test_action_controls_remain_pairwise_distinct() -> None:
    controls = build_action_controls(
        droid_action_stream(_actions()),
        droid_action_stream(_actions(0.002)),
        shuffle_seed=20260727,
    )
    assert set(controls) == {"recorded", "zero", "shuffled", "reversed", "policy_swapped"}
    assert len({item["action_sha256"] for item in controls.values()}) == 5


def test_raw_droid_state_conversion_preserves_exact_shape_and_gripper_flip() -> None:
    states = [[0.001 * index, 0.0, 0.4, 0.0, 0.0, 0.0] for index in range(17)]
    action = convert_droid_states_to_action_stream(
        states,
        [0.0] * 16,
        source_gripper_action_flipped=True,
    )
    assert action["shape"] == [16, 10]
    assert all(row[-1] == 1.0 for row in action["actions"])
    assert action["actions"][0][3:9] == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_raw_droid_state_conversion_rejects_wrong_alignment() -> None:
    with pytest.raises(SuccessorContractError, match="17x6"):
        convert_droid_states_to_action_stream(
            [[0.0] * 6 for _ in range(16)],
            [0.0] * 16,
            source_gripper_action_flipped=False,
        )
    with pytest.raises(SuccessorContractError, match="shape_16"):
        convert_droid_states_to_action_stream(
            [[0.0] * 6 for _ in range(17)],
            [0.0] * 15,
            source_gripper_action_flipped=False,
        )


def test_timestamps_and_camera_order_are_frozen() -> None:
    timestamps = [index / 15.0 for index in range(17)]
    assert validate_droid_timestamps(timestamps)["status"] == "passed"
    with pytest.raises(SuccessorContractError, match="timestamp_alignment"):
        validate_droid_timestamps([float(index) for index in range(17)])
    cameras = {
        "wrist_image_left": {"shape": [360, 640, 3], "fps": 15},
        "exterior_image_1_left": {"shape": [360, 640, 3], "fps": 15},
        "exterior_image_2_left": {"shape": [360, 640, 3], "fps": 15},
    }
    assert validate_droid_camera_metadata(cameras)["output_shape"] == [540, 640, 3]
    with pytest.raises(SuccessorContractError, match="camera_order"):
        validate_droid_camera_metadata(dict(reversed(list(cameras.items()))))


def test_missing_action_stream_fails_closed() -> None:
    with pytest.raises(SuccessorContractError, match="action_stream_missing"):
        validate_droid_action_stream(None)


def test_malformed_frequency_fails_closed() -> None:
    stream = droid_action_stream(_actions())
    stream["frequency_hz"] = 10.0
    with pytest.raises(SuccessorContractError, match="15_hz"):
        validate_droid_action_stream(stream)


def test_incompatible_embodiment_fails_closed() -> None:
    stream = droid_action_stream(_actions())
    stream["embodiment"] = "humanoid"
    with pytest.raises(SuccessorContractError, match="embodiment"):
        validate_droid_action_stream(stream)


def test_bad_shape_and_gripper_fail_closed() -> None:
    stream = droid_action_stream(_actions())
    stream["actions"] = stream["actions"][:-1]
    stream.pop("action_sha256")
    with pytest.raises(SuccessorContractError, match="horizon"):
        validate_droid_action_stream(stream)
    stream = droid_action_stream(_actions())
    stream["actions"][0][-1] = 2.0
    stream.pop("action_sha256")
    with pytest.raises(SuccessorContractError, match="gripper"):
        validate_droid_action_stream(stream)


def test_request_is_policy_blind_and_hash_named() -> None:
    request = build_forward_dynamics_request(
        initial_observation_sha256="a" * 64,
        task_instruction="Pick up the red cup.",
        action_stream=droid_action_stream(_actions()),
        condition="recorded",
        seed=0,
    )
    assert request["name"] == request["request_id"]
    assert request["prompt"] == "Pick up the red cup."
    assert "policy" not in request
    assert request["runtime"]["model_mode"] == "forward_dynamics"
    assert request["runtime"]["num_frames"] == 17
    assert request["source_lock"]["trust_remote_code"] is False


def test_request_rejects_policy_metadata() -> None:
    stream = droid_action_stream(_actions())
    stream["policy_name"] = "best-policy"
    with pytest.raises(SuccessorContractError, match="policy_identity"):
        build_forward_dynamics_request(
            initial_observation_sha256="b" * 64,
            task_instruction="Move the object.",
            action_stream=stream,
            condition="recorded",
            seed=1,
        )


def test_evaluator_cannot_score_causally_invalid_rollout() -> None:
    with pytest.raises(SuccessorContractError, match="causally_invalid"):
        assert_evaluator_eligible(
            {"causal_validity": {"status": "invalid"}, "generated_media_valid": True}
        )
    assert_evaluator_eligible(
        {"causal_validity": {"status": "valid"}, "generated_media_valid": True}
    )


def test_blackwell_requires_exact_stack_preflight() -> None:
    selection = select_gpu_offer(
        [
            GPUOffer("h100", "H100 SXM 80GB", 3.00, 60.0, 0.5, True),
            GPUOffer("bw", "RTX PRO 6000 Blackwell", 0.94, 108.0, 0.5, False),
        ],
        rollout_count=10,
    )
    assert selection["selected_gpu_model"] == "H100 SXM 80GB"
    assert selection["selection_reason"] == "h100_default_blackwell_exact_stack_not_admitted"


def test_blackwell_wins_only_when_admitted_and_materially_cheaper() -> None:
    selection = select_gpu_offer(
        [
            GPUOffer("h100", "H100 SXM 80GB", 3.00, 60.0, 0.5, True),
            GPUOffer("bw", "RTX PRO 6000 Blackwell", 0.94, 108.0, 0.5, True),
        ],
        rollout_count=10,
    )
    assert selection["selected_gpu_model"] == "RTX PRO 6000 Blackwell"


def test_h100_wins_when_all_in_cost_is_reasonably_close() -> None:
    selection = select_gpu_offer(
        [
            GPUOffer("h100", "H100 SXM 80GB", 1.00, 60.0, 0.0, True),
            GPUOffer("bw", "RTX PRO 6000 Blackwell", 1.00, 55.0, 0.0, True),
        ],
        rollout_count=10,
    )
    assert selection["selected_gpu_model"] == "H100 SXM 80GB"
    assert selection["selection_reason"] == "h100_preferred_prices_reasonably_close"


def test_compute_admission_requires_cap_watchdog_ttl_cutoff_and_teardown() -> None:
    blocked = validate_compute_admission(
        {
            "allocator_entrypoint": "python -m blueprint_pipeline.paid_resource_allocator",
            "authorized_compute_cap_usd": None,
            "projected_compute_spend_usd": 2.0,
            "hard_ttl_seconds": 3600,
            "watchdog_enabled": True,
            "automatic_spend_cutoff": True,
            "teardown_required": True,
            "provider_zero_verification_required": True,
        }
    )
    assert blocked["status"] == "blocked"
    assert "explicit_compute_cap_not_authorized" in blocked["blockers"]
    passed = validate_compute_admission(
        {
            "allocator_entrypoint": "python -m blueprint_pipeline.paid_resource_allocator",
            "authorized_compute_cap_usd": 3.0,
            "projected_compute_spend_usd": 2.0,
            "hard_ttl_seconds": 3600,
            "watchdog_enabled": True,
            "automatic_spend_cutoff": True,
            "teardown_required": True,
            "provider_zero_verification_required": True,
        }
    )
    assert passed["allocation_authorized"] is True


def test_artifacts_cannot_escape_successor_namespace(tmp_path: Path) -> None:
    root = tmp_path / "successor"
    assert validate_artifact_path(root / "report.json", root) == (root / "report.json").resolve()
    with pytest.raises(SuccessorContractError, match="outside"):
        validate_artifact_path(tmp_path / "historical" / "report.json", root)


def test_frozen_smoke_inventory_is_complete_and_hash_bound() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/experiments/policy_ranking_successor_experiment_20260727/smoke_request_inventory.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert validate_smoke_inventory_manifest(manifest)["request_count"] == 10
    manifest["requests"][0]["seed"] = 1
    with pytest.raises(SuccessorContractError, match="invalid"):
        validate_smoke_inventory_manifest(manifest)
