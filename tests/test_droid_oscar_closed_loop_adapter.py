from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from blueprint_pipeline.droid_oscar_closed_loop_adapter import (
    EXTERIOR_VIEW,
    RIGHT_EXTERIOR_VIEW,
    ROBOARENA_CONCAT_POLICY_VIEWS,
    WRIST_VIEW,
    CallableMultiViewOscarWamArm,
    DroidOscarSkeletonTransitionAdapter,
)
from blueprint_pipeline.policy_wam_closed_loop import (
    ClosedLoopConfig,
    run_policy_wam_closed_loop,
)


def _observation() -> dict[str, Any]:
    return {
        EXTERIOR_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
        WRIST_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.zeros(1),
        "prompt": "Pick up the bottle.",
    }


def test_adapter_builds_two_view_oscar_request_and_never_uses_real_future(tmp_path: Path) -> None:
    def builder(**kwargs: Any) -> dict[str, Any]:
        views = {}
        for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
            prefix = view_id.split("/")[-1]
            first = kwargs["output_dir"] / f"{prefix}.png"
            skeleton = kwargs["output_dir"] / f"{prefix}.mp4"
            Image.new("RGB", (32, 32)).save(first)
            skeleton.write_bytes(b"skeleton")
            views[view_id] = {
                "first_frame_path": first,
                "skeleton_video_path": skeleton,
                "camera_calibration_sha256": "a" * 64,
            }
        actions = np.zeros((16, 10))
        actions[:, 3] = 1.0
        actions[:, 7] = 1.0
        return {
            "views": views,
            "reliability_actions_10d": actions,
            "next_joint_position": np.ones(7) * 0.1,
            "next_gripper_position": np.asarray([1.0]),
            "evidence": {"calibrated_projection": True},
        }

    adapter = DroidOscarSkeletonTransitionAdapter(
        conditioning_builder=builder, action_chunk_rows=10
    )
    prepared = adapter.prepare_transition(
        observation=_observation(),
        policy_action=np.zeros((10, 8)),
        task_prompt="Pick up the bottle.",
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )

    assert set(prepared["wam_request"]["views"]) == {EXTERIOR_VIEW, WRIST_VIEW}
    assert prepared["physical_future_observation_used"] is False
    assert prepared["wam_request"]["source_rgb_context"].startswith("one_")

    generated = {}
    for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
        path = tmp_path / f"generated-{view_id.split('/')[-1]}.png"
        Image.new("RGB", (64, 64), color=(20, 30, 40)).save(path)
        generated[view_id] = path
    advanced = adapter.advance_policy_observation(
        previous_observation=_observation(),
        prepared_transition=prepared,
        wam_prediction={"generated_view_frames": generated},
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )
    assert advanced["observation"][EXTERIOR_VIEW].shape == (224, 224, 3)
    assert advanced["observation"][WRIST_VIEW].shape == (224, 224, 3)
    assert advanced["provenance"]["visual_source"] == "wam_prediction"
    assert advanced["provenance"]["physical_future_observation_used"] is False


def test_multiview_wam_uses_same_generator_for_each_view(tmp_path: Path) -> None:
    calls: list[str] = []

    def generator(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["view_id"])
        video = kwargs["output_dir"] / "generated.mp4"
        video.write_bytes(b"generated")
        return {"generated_video_path": video, "provider": "fixture-oscar"}

    def extractor(**kwargs: Any) -> Path:
        Image.new("RGB", (16, 16)).save(kwargs["output_path"])
        return kwargs["output_path"]

    arm = CallableMultiViewOscarWamArm(generator=generator, frame_extractor=extractor)
    request = {
        "task_prompt": "Pick up the bottle.",
        "negative_prompt": "bad quality",
        "executed_prefix_steps": 8,
        "views": {
            EXTERIOR_VIEW: {"first_frame_path": "external.png"},
            WRIST_VIEW: {"first_frame_path": "wrist.png"},
        },
    }
    result = arm.predict(request, output_dir=tmp_path)

    assert calls == [EXTERIOR_VIEW, WRIST_VIEW]
    assert set(result["generated_view_frames"]) == {EXTERIOR_VIEW, WRIST_VIEW}
    assert result["same_frozen_wam_generated_all_views"] is True
    assert result["wam_to_wam_chaining"] is False


def test_roboarena_policy_observation_advances_all_three_generated_views(
    tmp_path: Path,
) -> None:
    observation = _observation()
    observation[RIGHT_EXTERIOR_VIEW] = np.zeros((224, 224, 3), dtype=np.uint8)

    def builder(**kwargs: Any) -> dict[str, Any]:
        views = {}
        for view_id in ROBOARENA_CONCAT_POLICY_VIEWS:
            safe = view_id.split("/")[-1]
            first = kwargs["output_dir"] / f"{safe}.png"
            skeleton = kwargs["output_dir"] / f"{safe}.mp4"
            Image.fromarray(kwargs["observation"][view_id]).save(first)
            skeleton.write_bytes(b"skeleton")
            views[view_id] = {
                "first_frame_path": first,
                "skeleton_video_path": skeleton,
                "camera_calibration_sha256": "c" * 64,
            }
        actions = np.zeros((16, 10))
        actions[:, 3] = 1.0
        actions[:, 7] = 1.0
        return {
            "views": views,
            "reliability_actions_10d": actions,
            "next_joint_position": np.ones(7) * 0.1,
            "next_gripper_position": np.asarray([1.0]),
        }

    adapter = DroidOscarSkeletonTransitionAdapter(
        conditioning_builder=builder,
        action_chunk_rows=16,
        required_policy_views=ROBOARENA_CONCAT_POLICY_VIEWS,
    )
    prepared = adapter.prepare_transition(
        observation=observation,
        policy_action=np.zeros((16, 8)),
        task_prompt="Pick up the bottle.",
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )
    generated = {}
    for index, view_id in enumerate(ROBOARENA_CONCAT_POLICY_VIEWS, start=1):
        path = tmp_path / f"generated-{index}.png"
        Image.new("RGB", (64, 64), color=(index,) * 3).save(path)
        generated[view_id] = path

    advanced = adapter.advance_policy_observation(
        previous_observation=observation,
        prepared_transition=prepared,
        wam_prediction={"generated_view_frames": generated},
        executed_prefix_steps=8,
        query_index=0,
        output_dir=tmp_path,
    )

    assert set(advanced["provenance"]["generated_view_frame_sha256"]) == set(
        ROBOARENA_CONCAT_POLICY_VIEWS
    )
    assert all(
        advanced["observation"][view_id].shape == (224, 224, 3)
        for view_id in ROBOARENA_CONCAT_POLICY_VIEWS
    )


def test_full_policy_oscar_wam_policy_loop_requeries_from_generated_views(
    tmp_path: Path,
) -> None:
    policy_observation_means: list[int] = []
    generation_calls: list[tuple[int, str]] = []

    class Policy:
        policy_id = "hidden_fixture_policy"

        def infer(self, observation: dict[str, Any]) -> np.ndarray:
            policy_observation_means.append(int(np.mean(observation[EXTERIOR_VIEW])))
            return np.zeros((10, 8))

    def builder(**kwargs: Any) -> dict[str, Any]:
        views = {}
        for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
            safe = view_id.split("/")[-1]
            first = kwargs["output_dir"] / f"{safe}.png"
            skeleton = kwargs["output_dir"] / f"{safe}.mp4"
            Image.fromarray(kwargs["observation"][view_id]).save(first)
            skeleton.write_bytes(b"skeleton")
            views[view_id] = {
                "first_frame_path": first,
                "skeleton_video_path": skeleton,
                "camera_calibration_sha256": "b" * 64,
            }
        reliability = np.zeros((16, 10))
        reliability[:, 3] = 1.0
        reliability[:, 7] = 1.0
        return {
            "views": views,
            "reliability_actions_10d": reliability,
            "next_joint_position": np.full(7, kwargs["query_index"] + 1.0),
            "next_gripper_position": np.asarray([0.0]),
        }

    def generator(**kwargs: Any) -> dict[str, Any]:
        query_index = int(kwargs["output_dir"].parent.name.split("_")[-1])
        generation_calls.append((query_index, kwargs["view_id"]))
        video = kwargs["output_dir"] / "generated.mp4"
        video.write_bytes(b"video")
        return {"generated_video_path": video}

    def extractor(**kwargs: Any) -> Path:
        query_index = int(kwargs["output_path"].parent.parent.name.split("_")[-1])
        Image.new("RGB", (32, 32), color=(query_index + 1,) * 3).save(kwargs["output_path"])
        return kwargs["output_path"]

    class Gate:
        gate_id = "fixture_pass_gate"

        def assess(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {"abstain": False, "reasons": []}

    class Terminal:
        criterion_id = "fixture_two_transition_terminal"

        def assess(self, *, observation: dict[str, Any], query_index: int) -> dict[str, Any]:
            del observation
            return {"terminal": query_index == 1, "reason": "fixture_terminal"}

    result = run_policy_wam_closed_loop(
        initial_observation=_observation(),
        policy_client=Policy(),
        wam_arm=CallableMultiViewOscarWamArm(
            generator=generator, frame_extractor=extractor
        ),
        transition_adapter=DroidOscarSkeletonTransitionAdapter(
            conditioning_builder=builder, action_chunk_rows=10
        ),
        reliability_gate=Gate(),
        terminal_criterion=Terminal(),
        config=ClosedLoopConfig(
            task_prompt="Pick up the bottle.",
            executed_prefix_steps=8,
            max_policy_queries=3,
            execution_mode="engineering_smoke",
        ),
        output_dir=tmp_path / "loop",
    )

    assert result["status"] == "completed"
    assert result["policy_call_count"] == 2
    assert result["wam_call_count"] == 2
    assert policy_observation_means == [0, 1]
    assert generation_calls == [
        (0, EXTERIOR_VIEW),
        (0, WRIST_VIEW),
        (1, EXTERIOR_VIEW),
        (1, WRIST_VIEW),
    ]
