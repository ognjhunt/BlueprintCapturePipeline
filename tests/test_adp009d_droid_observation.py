from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_observation import (
    BLOCKER_THIRD_VIEW_OUTSIDE_CONTRACT,
    DROID_EXTERIOR_VIEW_1,
    DROID_EXTERIOR_VIEW_2,
    DROID_WRIST_VIEW,
    DroidObservationError,
    build_droid_observation,
    describe_observation_conversion,
    resize_with_pad,
)

_JOINTS = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]


def _isaac_frame() -> np.ndarray:
    """A 1280x720 frame with a distinctive gradient, as Isaac renders."""

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[..., 0] = np.linspace(0, 255, 1280, dtype=np.uint8)[None, :]
    frame[..., 1] = np.linspace(0, 255, 720, dtype=np.uint8)[:, None]
    return frame


def test_pad_preserves_aspect_and_never_stretches() -> None:
    """16:9 into a square must letterbox, not distort."""

    out = resize_with_pad(_isaac_frame(), height=224, width=224)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    # 1280x720 scaled to fit 224 wide gives 126 rows of content, 98 padded.
    content_rows = [r for r in range(224) if out[r].any()]
    assert len(content_rows) == 126
    # Padding is centred and black.
    assert content_rows[0] == 49
    assert not out[:49].any()
    assert not out[175:].any()


def test_pad_matches_the_vendor_runtime_implementation() -> None:
    """Must agree byte-for-byte with the GR00T runtime's own preprocessing."""

    from blueprint_pipeline.groot_n17_droid_policy_runtime import _resize_with_pad

    frame = _isaac_frame()
    mine = resize_with_pad(frame, height=180, width=320)
    theirs = _resize_with_pad(frame, height=180, width=320)

    assert np.array_equal(mine, theirs)


def test_each_candidate_gets_its_own_measured_shape() -> None:
    """The candidates do not share an observation format; never assume they do."""

    cameras = {DROID_EXTERIOR_VIEW_1: _isaac_frame(), DROID_WRIST_VIEW: _isaac_frame()}

    pi05 = build_droid_observation(
        candidate_id="pi05_droid",
        camera_rgb=cameras,
        joint_position=_JOINTS,
        gripper_position=0.04,
        prompt="pick up the can",
    )
    assert pi05[DROID_EXTERIOR_VIEW_1].shape == (224, 224, 3)

    groot = build_droid_observation(
        candidate_id="groot_n17_droid",
        camera_rgb=cameras,
        joint_position=_JOINTS,
        gripper_position=0.04,
        prompt="pick up the can",
        eef_9d=np.arange(9, dtype=float),
        historical_camera_rgb=cameras,
    )
    assert groot[DROID_EXTERIOR_VIEW_1].shape == (180, 320, 3)
    assert np.array_equal(groot["observation/eef_9d"], np.arange(9, dtype=float))


def test_groot_requires_the_live_nine_dimensional_end_effector_pose() -> None:
    base = dict(
        candidate_id="groot_n17_droid",
        camera_rgb={
            DROID_EXTERIOR_VIEW_1: _isaac_frame(),
            DROID_WRIST_VIEW: _isaac_frame(),
        },
        joint_position=_JOINTS,
        gripper_position=0.04,
        prompt="pick up the can",
        historical_camera_rgb={
            DROID_EXTERIOR_VIEW_1: _isaac_frame(),
            DROID_WRIST_VIEW: _isaac_frame(),
        },
    )

    for eef in (None, [0.0] * 8, [float("nan")] * 9):
        with pytest.raises(DroidObservationError, match="eef_9d_invalid"):
            build_droid_observation(**base, eef_9d=eef)


def test_groot_requires_both_exact_t_minus_15_camera_frames() -> None:
    base = dict(
        candidate_id="groot_n17_droid",
        camera_rgb={
            DROID_EXTERIOR_VIEW_1: _isaac_frame(),
            DROID_WRIST_VIEW: _isaac_frame(),
        },
        joint_position=_JOINTS,
        gripper_position=0.04,
        prompt="pick up the can",
        eef_9d=np.arange(9, dtype=float),
    )

    with pytest.raises(DroidObservationError, match="history_view_unavailable"):
        build_droid_observation(**base)


def test_pi05_observation_satisfies_the_existing_droid_validator() -> None:
    """The output must pass the repository's own DROID contract check."""

    from blueprint_pipeline.droid_policy_bridge import validate_droid_observation

    observation = build_droid_observation(
        candidate_id="pi05_droid",
        camera_rgb={
            DROID_EXTERIOR_VIEW_1: _isaac_frame(),
            DROID_WRIST_VIEW: _isaac_frame(),
        },
        joint_position=_JOINTS,
        gripper_position=0.04,
        prompt="pick up the can",
    )

    assert validate_droid_observation(observation) == []


def test_a_three_view_candidate_names_the_frozen_contract_it_violates() -> None:
    """Cosmos wants a camera the ADP-009D scene deliberately removed."""

    with pytest.raises(DroidObservationError) as excinfo:
        build_droid_observation(
            candidate_id="cosmos3_edge_policy_droid",
            camera_rgb={
                DROID_EXTERIOR_VIEW_1: _isaac_frame(),
                DROID_WRIST_VIEW: _isaac_frame(),
            },
            joint_position=_JOINTS,
            gripper_position=0.04,
            prompt="pick up the can",
        )
    assert BLOCKER_THIRD_VIEW_OUTSIDE_CONTRACT in excinfo.value.errors

    # With the camera supplied it builds, so the blocker is the contract, not a bug.
    observation = build_droid_observation(
        candidate_id="cosmos3_edge_policy_droid",
        camera_rgb={
            DROID_EXTERIOR_VIEW_1: _isaac_frame(),
            DROID_EXTERIOR_VIEW_2: _isaac_frame(),
            DROID_WRIST_VIEW: _isaac_frame(),
        },
        joint_position=_JOINTS,
        gripper_position=0.04,
        prompt="pick up the can",
    )
    assert observation[DROID_EXTERIOR_VIEW_2].shape == (224, 224, 3)


def test_a_missing_view_is_never_substituted_or_duplicated() -> None:
    """A policy fed a duplicated view is silently misinformed about the scene."""

    with pytest.raises(DroidObservationError) as excinfo:
        build_droid_observation(
            candidate_id="pi05_droid",
            camera_rgb={DROID_EXTERIOR_VIEW_1: _isaac_frame()},
            joint_position=_JOINTS,
            gripper_position=0.04,
            prompt="pick up the can",
        )
    assert any(DROID_WRIST_VIEW in e for e in excinfo.value.errors)


def test_malformed_proprioception_and_prompt_fail_closed() -> None:
    cameras = {DROID_EXTERIOR_VIEW_1: _isaac_frame(), DROID_WRIST_VIEW: _isaac_frame()}
    base = dict(
        candidate_id="pi05_droid",
        camera_rgb=cameras,
        joint_position=_JOINTS,
        gripper_position=0.04,
        prompt="pick up the can",
    )

    for override in (
        {"joint_position": [0.0] * 6},
        {"joint_position": [float("nan")] * 7},
        {"gripper_position": float("inf")},
        {"prompt": "   "},
        {"prompt": ""},
    ):
        with pytest.raises(DroidObservationError):
            build_droid_observation(**{**base, **override})


def test_unknown_candidate_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(DroidObservationError):
        build_droid_observation(
            candidate_id="some_other_policy",
            camera_rgb={DROID_EXTERIOR_VIEW_1: _isaac_frame()},
            joint_position=_JOINTS,
            gripper_position=0.04,
            prompt="pick up the can",
        )
    with pytest.raises(DroidObservationError):
        describe_observation_conversion("some_other_policy")


def test_conversion_receipt_records_what_was_padded_and_asserts_no_crop() -> None:
    """The receipt must let a reviewer see exactly what the policy received."""

    report = describe_observation_conversion("pi05_droid")

    assert report["source_resolution_hw"] == [720, 1280]
    assert report["target_resolution_hw"] == [224, 224]
    assert report["content_resolution_hw"] == [126, 224]
    assert report["padded_rows"] == 98
    assert report["padded_columns"] == 0
    assert report["scene_content_cropped"] is False

    wide = describe_observation_conversion("groot_n17_droid")
    assert wide["target_resolution_hw"] == [180, 320]
    # 16:9 into 320x180 is exact, so nothing is padded at all.
    assert wide["padded_rows"] == 0
    assert wide["padded_columns"] == 0
    assert wide["video_delta_indices"] == [-15, 0]
    assert wide["history_sampling"] == "exact_simulator_control_steps"


def test_every_candidate_declares_an_observation_frame_cadence() -> None:
    """Render scheduling is derived from what each policy consumes, not from a
    per-run environment variable someone must remember."""

    from blueprint_pipeline.adp009d_droid_observation import (
        CANDIDATE_OBSERVATION_FRAME_CADENCE,
        CANDIDATE_REQUIRED_VIEWS,
        FRAME_CADENCE_PER_QUERY,
        FRAME_CADENCE_PER_STEP,
    )

    assert set(CANDIDATE_OBSERVATION_FRAME_CADENCE) == set(CANDIDATE_REQUIRED_VIEWS)
    for cadence in CANDIDATE_OBSERVATION_FRAME_CADENCE.values():
        assert cadence in {FRAME_CADENCE_PER_QUERY, FRAME_CADENCE_PER_STEP}
    # pi05 consumes only the current frame; GR00T consumes a t-minus-15-step
    # history, so per-query rendering would silently feed it stale history.
    assert CANDIDATE_OBSERVATION_FRAME_CADENCE["pi05_droid"] == FRAME_CADENCE_PER_QUERY
    assert (
        CANDIDATE_OBSERVATION_FRAME_CADENCE["groot_n17_droid"]
        == FRAME_CADENCE_PER_STEP
    )
