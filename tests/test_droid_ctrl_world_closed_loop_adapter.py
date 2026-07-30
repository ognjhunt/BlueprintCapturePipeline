from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.ctrl_world_droid_action_adapter import (
    CTRL_WORLD_CAUSAL_CONDITIONS,
    CTRL_WORLD_CAUSAL_SHUFFLE_ORDER,
    CTRL_WORLD_CAUSAL_SHUFFLE_SEED,
    CTRL_WORLD_FUTURE_FRAME_INDICES,
    CtrlWorldReleasedJointVelocityAdapter,
    build_ctrl_world_current_reference_action_controls,
    cartesian_pose_rows_to_reliability_actions_10d,
    load_ctrl_world_released_joint_velocity_adapter,
)
from blueprint_pipeline.droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_STATE_HISTORY,
    CTRL_WORLD_VIEW_HISTORY_PATHS,
    DroidCtrlWorldCurrentReferenceTransitionAdapter,
)
from blueprint_pipeline.droid_policy_bridge import (
    DROID_EXTERIOR_VIEW_1,
    DROID_EXTERIOR_VIEW_2,
    DROID_WRIST_VIEW,
)
from blueprint_pipeline.policy_wam_closed_loop import (
    ClosedLoopConfig,
    run_policy_wam_closed_loop,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


def _dynamics(current_joint: np.ndarray, joint_velocity: np.ndarray) -> np.ndarray:
    assert current_joint.shape == (7,)
    assert joint_velocity.shape == (15, 7)
    return current_joint[None, :] + np.cumsum(joint_velocity, axis=0) * 0.01


def _fk(joint_position: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = joint_position[:3]
    return transform


def _released_adapter() -> CtrlWorldReleasedJointVelocityAdapter:
    return CtrlWorldReleasedJointVelocityAdapter(
        dynamics_adapter=_dynamics,
        forward_kinematics=_fk,
        gripper_max=0.75,
    )


def _policy_preprocessor(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.uint8)


def _transition_adapter() -> DroidCtrlWorldCurrentReferenceTransitionAdapter:
    return DroidCtrlWorldCurrentReferenceTransitionAdapter(
        action_adapter=_released_adapter(),
        openpi_config_name="pi05_droid",
        action_chunk_rows=15,
        policy_image_preprocessor=_policy_preprocessor,
    )


def _observation() -> dict[str, Any]:
    return {
        DROID_EXTERIOR_VIEW_1: np.zeros((224, 224, 3), dtype=np.uint8),
        DROID_EXTERIOR_VIEW_2: np.ones((224, 224, 3), dtype=np.uint8),
        DROID_WRIST_VIEW: np.full((224, 224, 3), 2, dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float64),
        "observation/gripper_position": np.zeros(1, dtype=np.float64),
        "blueprint/ctrl_world_cartesian_pose_7d": np.zeros(7, dtype=np.float64),
        "prompt": "Pick up the blue block and place it in the white plate.",
    }


def test_released_velocity_adapter_preserves_native_chunk_and_public_state_rule() -> None:
    action = np.zeros((15, 8), dtype=np.float64)
    action[:, 0] = 1.0
    action[:, 7] = np.linspace(0.0, 1.0, 15)
    result = _released_adapter().adapt(
        policy_action=action,
        current_joint_position=np.zeros(7),
        current_gripper_position=np.zeros(1),
        history_cartesian_pose_7d=np.zeros((6, 7)),
        source_action_space="joint_velocity_plus_gripper_position",
    )

    assert result["native_policy_action"].shape == (15, 8)
    assert np.array_equal(result["native_policy_action"], action)
    assert result["action_conditioning_7d"].shape == (11, 7)
    assert result["reliability_actions_10d"].shape == (5, 10)
    assert result["future_frame_indices"] == list(CTRL_WORLD_FUTURE_FRAME_INDICES)
    assert result["next_joint_position"][0] == pytest.approx(0.08)
    assert result["next_gripper_position"][0] <= 0.75
    assert result["official_ctrl_world_learned_action_adapter_used"] is True
    assert result["physical_future_observation_used"] is False


def test_reliability_adapter_encodes_absolute_no_motion_as_null_command() -> None:
    pose = np.asarray([0.42, -0.11, 0.27, 0.2, -0.1, 0.3, 0.5])
    actions = cartesian_pose_rows_to_reliability_actions_10d(np.repeat(pose[None, :], 5, axis=0))

    assert np.allclose(actions[:, :3], 0.0)
    assert np.allclose(actions[:, 3:9], np.asarray([1, 0, 0, 0, 1, 0]))
    assert np.allclose(actions[:, 9], 0.5)


def test_reliability_adapter_uses_incremental_translation_not_absolute_position() -> None:
    poses = np.zeros((5, 7), dtype=np.float64)
    poses[:, 0] = 4.0 + np.arange(5) * 0.02
    actions = cartesian_pose_rows_to_reliability_actions_10d(poses)

    assert actions[0, 0] == 0.0
    assert np.allclose(actions[1:, 0], 0.02)
    assert np.allclose(actions[:, 1:3], 0.0)


def test_causal_controls_transform_only_real_executed_prefixes() -> None:
    own = np.zeros((15, 8), dtype=np.float64)
    own[:, :7] = np.arange(15)[:, None] / 100.0
    own[:, 7] = np.linspace(0.1, 0.7, 15)
    swapped = np.zeros((10, 8), dtype=np.float64)
    swapped[:, :7] = 1.0 + np.arange(10)[:, None] / 100.0
    swapped[:, 7] = np.linspace(0.7, 0.2, 10)

    result = build_ctrl_world_current_reference_action_controls(
        own_policy_action=own,
        own_source_trace_id="pi05-request-hash",
        policy_swapped_action=swapped,
        policy_swapped_source_trace_id="pi0-fast-request-hash",
        current_gripper_hold=0.25,
        shuffle_seed=CTRL_WORLD_CAUSAL_SHUFFLE_SEED,
        temporal_shift_steps=1,
    )

    assert tuple(result["condition_order"]) == CTRL_WORLD_CAUSAL_CONDITIONS
    assert len(set(result["executed_prefix_sha256_by_condition"].values())) == 6
    assert result["native_action_shape_by_condition"]["policy_swapped"] == [10, 8]
    assert result["shuffle_order_first_eight"] == list(CTRL_WORLD_CAUSAL_SHUFFLE_ORDER)
    assert np.array_equal(result["conditions"]["shuffled"][8:], own[8:])
    assert np.array_equal(result["conditions"]["reversed"][8:], own[8:])
    assert np.array_equal(result["conditions"]["shifted"][8:], own[8:])
    assert np.allclose(result["conditions"]["no_motion"][:, :7], 0.0)
    assert np.allclose(result["conditions"]["no_motion"][:, 7], 0.25)
    assert result["policy_swap_is_distinct_real_trace"] is True
    assert len(result["controls_sha256"]) == 64


def test_causal_controls_fail_on_colliding_or_synthetic_policy_swap() -> None:
    own = np.zeros((10, 8), dtype=np.float64)
    own[:, 7] = 0.3
    swapped = np.ones((10, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="trace_not_distinct"):
        build_ctrl_world_current_reference_action_controls(
            own_policy_action=own,
            own_source_trace_id="same",
            policy_swapped_action=swapped,
            policy_swapped_source_trace_id="same",
            current_gripper_hold=0.3,
            shuffle_seed=CTRL_WORLD_CAUSAL_SHUFFLE_SEED,
        )
    with pytest.raises(ValueError, match="prefixes_not_pairwise_distinct"):
        build_ctrl_world_current_reference_action_controls(
            own_policy_action=own,
            own_source_trace_id="own",
            policy_swapped_action=swapped,
            policy_swapped_source_trace_id="other",
            current_gripper_hold=0.3,
            shuffle_seed=CTRL_WORLD_CAUSAL_SHUFFLE_SEED,
        )


def test_released_velocity_adapter_repeats_last_row_for_ten_row_policies() -> None:
    action = np.zeros((10, 8), dtype=np.float64)
    action[-1, 1] = 0.5
    result = _released_adapter().adapt(
        policy_action=action,
        current_joint_position=np.zeros(7),
        current_gripper_position=np.zeros(1),
        history_cartesian_pose_7d=np.zeros((6, 7)),
        source_action_space="joint_velocity_plus_gripper_position",
    )

    assert result["ten_row_padding_rule"] == "repeat_final_row_to_15"
    assert result["native_policy_action_shape"] == [10, 8]


def test_released_velocity_adapter_rejects_policy_droid_absolute_positions() -> None:
    with pytest.raises(ValueError, match="requires_joint_velocity_policy"):
        _released_adapter().adapt(
            policy_action=np.zeros((15, 8)),
            current_joint_position=np.zeros(7),
            current_gripper_position=np.zeros(1),
            history_cartesian_pose_7d=np.zeros((6, 7)),
            source_action_space="absolute_joint_position_plus_gripper_position",
        )


def test_transition_adapter_freezes_three_view_history_and_native_output(
    tmp_path: Path,
) -> None:
    adapter = _transition_adapter()
    prepared = adapter.prepare_transition(
        observation=_observation(),
        policy_action=np.zeros((15, 8)),
        task_prompt=_observation()["prompt"],
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )

    assert prepared["wam_request"]["view_order"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert prepared["wam_request"]["action_conditioning_shape"] == [11, 7]
    assert set(prepared["wam_request"]["current_views"]) == set(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert prepared["wam_request"]["executed_prefix_seconds"] == pytest.approx(8 / 15)
    assert Path(prepared["native_policy_action_path"]).is_file()
    assert len(prepared["native_policy_action_sha256"]) == 64
    assert prepared["openpi_config_name_internal_only"] == "pi05_droid"
    assert all(
        len(rows) == 6 for rows in prepared["wam_request"]["selected_history_views"].values()
    )


def test_current_reference_runs_same_policy_on_wam_generated_three_view_observations(
    tmp_path: Path,
) -> None:
    seen_external_means: list[int] = []

    class Policy:
        policy_id = "frozen_pi05_droid_fixture"

        def infer(self, observation: dict[str, Any]) -> np.ndarray:
            seen_external_means.append(int(np.mean(observation[DROID_EXTERIOR_VIEW_1])))
            return np.zeros((15, 8), dtype=np.float64)

    class Wam:
        arm_id = "ctrl_world_current_reference_fixture"

        def predict(self, request: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
            assert request["view_order"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)
            assert not any("policy" in str(key).lower() for key in request)
            query_index = int(request["query_index"])
            if query_index == 1:
                current_path = Path(request["current_views"][DROID_EXTERIOR_VIEW_1]["path"])
                history_path = Path(
                    request["selected_history_views"][DROID_EXTERIOR_VIEW_1][-1]["path"]
                )
                with Image.open(current_path) as image:
                    assert int(np.asarray(image).mean()) == 11
                with Image.open(history_path) as image:
                    assert int(np.asarray(image).mean()) == 0
            sequences: dict[str, list[str]] = {}
            for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
                view_dir = output_dir / f"generated_{view_index}"
                view_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for frame_index in range(5):
                    path = view_dir / f"frame_{frame_index}.png"
                    Image.new(
                        "RGB",
                        (32, 32),
                        color=((query_index + 1) * 10 + view_index,) * 3,
                    ).save(path)
                    paths.append(str(path))
                sequences[view_id] = paths
            return {"generated_view_frame_sequences": sequences}

    class Gate:
        gate_id = "fixture_reliability_pass"

        def assess(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["prepared_transition"]["reliability_actions_10d"].shape == (
                5,
                10,
            )
            return {"abstain": False, "reasons": []}

    class Terminal:
        criterion_id = "fixture_two_query_terminal"

        def assess(self, *, observation: dict[str, Any], query_index: int) -> dict[str, Any]:
            assert CTRL_WORLD_VIEW_HISTORY_PATHS in observation
            assert CTRL_WORLD_STATE_HISTORY in observation
            return {"terminal": query_index == 1, "reason": "fixture_terminal"}

    result = run_policy_wam_closed_loop(
        initial_observation=_observation(),
        policy_client=Policy(),
        wam_arm=Wam(),
        transition_adapter=_transition_adapter(),
        reliability_gate=Gate(),
        terminal_criterion=Terminal(),
        config=ClosedLoopConfig(
            task_prompt=_observation()["prompt"],
            executed_prefix_steps=8,
            max_policy_queries=3,
            execution_mode="engineering_smoke",
        ),
        output_dir=tmp_path / "loop",
    )

    assert result["status"] == "completed"
    assert result["policy_call_count"] == 2
    assert result["wam_call_count"] == 2
    assert seen_external_means == [0, 11]
    assert (tmp_path / "loop/transition_0000/native_policy_action.npy").is_file()
    assert (tmp_path / "loop/transition_0001/native_policy_action.npy").is_file()


def test_openpi_config_binds_native_action_rows() -> None:
    with pytest.raises(ValueError, match="action_rows_mismatch"):
        DroidCtrlWorldCurrentReferenceTransitionAdapter(
            action_adapter=_released_adapter(),
            openpi_config_name="pi05_droid",
            action_chunk_rows=10,
        )


def test_released_action_runtime_binds_exact_source_checkpoint_and_fk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import ctrl_world_droid_action_adapter as action_module

    root = tmp_path / "ctrl_world"
    source = root / "models/action_adapter/train2.py"
    checkpoint = root / "models/action_adapter/model2_15_9.pth"
    fk_source = root / "models/utils.py"
    for path, content in (
        (source, b"source"),
        (checkpoint, b"checkpoint"),
        (fk_source, b"fk"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(
        action_module, "CTRL_WORLD_RELEASED_ACTION_ADAPTER_SOURCE_SHA256", file_sha256(source)
    )
    monkeypatch.setattr(
        action_module, "OFFICIAL_LOCAL_ACTION_ADAPTER_SHA256", file_sha256(checkpoint)
    )
    monkeypatch.setattr(
        action_module, "CTRL_WORLD_RELEASED_FK_SOURCE_SHA256", file_sha256(fk_source)
    )
    loaded = load_ctrl_world_released_joint_velocity_adapter(
        ctrl_world_source_dir=root,
        gripper_max=0.75,
        device="fixture",
        dynamics_loader=lambda source_path, checkpoint_path, device: (
            _dynamics
            if (source_path, checkpoint_path, device) == (source, checkpoint, "fixture")
            else None
        ),
        forward_kinematics_loader=lambda path: _fk if path == fk_source else None,
    )

    result = loaded.adapter.adapt(
        policy_action=np.zeros((15, 8)),
        current_joint_position=np.zeros(7),
        current_gripper_position=np.zeros(1),
        history_cartesian_pose_7d=np.zeros((6, 7)),
        source_action_space="joint_velocity_plus_gripper_position",
    )
    assert result["action_conditioning_shape"] == [11, 7]
    assert loaded.evidence["official_released_dynamics_and_fk_loaded"] is True
    assert loaded.evidence["absolute_joint_position_conversion_supported"] is False


def test_released_action_runtime_rejects_asset_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import ctrl_world_droid_action_adapter as action_module

    root = tmp_path / "ctrl_world"
    for relative in (
        "models/action_adapter/train2.py",
        "models/action_adapter/model2_15_9.pth",
        "models/utils.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(action_module, "CTRL_WORLD_RELEASED_ACTION_ADAPTER_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        load_ctrl_world_released_joint_velocity_adapter(
            ctrl_world_source_dir=root,
            gripper_max=0.75,
            dynamics_loader=lambda *_: _dynamics,
            forward_kinematics_loader=lambda _: _fk,
        )


def test_released_dynamics_loader_stubs_only_unused_decord_training_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import ctrl_world_droid_action_adapter as action_module

    source = tmp_path / "train2.py"
    source.write_text(
        """
from decord import VideoReader, cpu
class Dynamics:
    def __init__(self, action_dim, action_num, hidden_size): pass
    def to(self, device): return self
    def load_state_dict(self, state): pass
    def eval(self): pass
    def __call__(self, current, velocity, delta, training=False): return velocity
""",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
    fake_torch.no_grad = NoGrad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    original_find_spec = action_module.importlib.util.find_spec
    monkeypatch.setattr(
        action_module.importlib.util,
        "find_spec",
        lambda name: None if name == "decord" else original_find_spec(name),
    )
    execute = action_module._load_released_dynamics(source, checkpoint, "cpu")

    result = execute(np.zeros(7), np.zeros((15, 7)))
    assert result.shape == (15, 7)
    assert execute.blueprint_training_only_import_stubs == ("decord",)  # type: ignore[attr-defined]
    assert "decord" not in sys.modules
