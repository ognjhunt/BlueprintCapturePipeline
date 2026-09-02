"""The camera gate must read radiance, not just scene-graph membership.

r10..r23 reported all three cameras `passed` against frames 88 to 92 percent
pure black, because v1 decided `passed` from a semantic segmentation mask and
recorded, honestly and uselessly, `rgb_or_model_label_used: False`.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from blueprint_pipeline.native_task_camera_observability import (
    BLOCKER_FRAME_UNIFORM,
    BLOCKER_FRAME_VOID,
    BLOCKER_SITE_DOMINANT_COLOR,
    BLOCKER_SITE_VOID,
    BLOCKER_TARGET_VOID,
    CLAIM_WITHOUT_SITE,
    CLAIM_WITH_SITE,
    MAXIMUM_SITE_VOID_PIXEL_FRACTION,
    MAXIMUM_SITE_DOMINANT_RGB_PIXEL_FRACTION,
    MINIMUM_DISTINCT_LUMINANCE_LEVELS,
    NOTICE_SITE_RENDERED_WHILE_UNCLAIMED,
    NativeTaskCameraObservabilityError,
    measure_native_task_camera_observability,
    measure_native_task_frame_render_evidence,
    measure_native_task_semantic_label_pixels,
    validate_native_task_policy_start_camera_observability,
)


def _textured(shape: tuple[int, int], *, low: int = 0, high: int = 255) -> np.ndarray:
    """A frame with real tonal structure, deterministic and CPU-only."""

    rows, cols = shape
    generator = np.random.default_rng(20260819)
    return generator.integers(low, high + 1, size=(rows, cols, 3), dtype=np.uint8)


def _semantic(shape: tuple[int, int]) -> np.ndarray:
    semantic = np.zeros(shape, dtype=np.int32)
    semantic[30:70, 70:130] = 7
    return semantic


def test_semantic_label_pixels_separates_task_and_robot_occlusion() -> None:
    semantic = np.zeros((20, 30), dtype=np.int32)
    semantic[2:6, 3:8] = 7
    semantic[8:18, 10:25] = 9
    labels = {"7": {"class": "task_object"}, "9": {"class": "robot"}}

    task = measure_native_task_semantic_label_pixels(
        semantic_ids=semantic,
        id_to_labels=labels,
        target_label="task_object",
    )
    robot = measure_native_task_semantic_label_pixels(
        semantic_ids=semantic,
        id_to_labels=labels,
        target_label="robot",
    )

    assert task["pixel_count"] == 20
    assert task["bbox_xyxy"] == [3, 2, 7, 5]
    assert robot["pixel_count"] == 150
    assert robot["pixel_fraction"] == pytest.approx(0.25)


def _passing_policy_start_camera(role: str, *, snapshot_id: str = "reset") -> dict:
    return {
        "snapshot_id": snapshot_id,
        "role": role,
        "scene_name": f"{role}_camera",
        "rgb_png": {"sha256": "sha256:" + "a" * 64},
        "observability": {
            "schema_version": "native_task_camera_observability.v2",
            "passed": True,
            "semantic_passed": True,
            "render_passed": True,
            "centroid_within_margin": True,
            "site_appearance_claimed": True,
            "claim": "camera_observes_task_object_in_rendered_site",
            "blockers": [],
            "pixel_count": 1000,
            "pixel_fraction": 0.02,
            "bbox_xyxy": [100, 30, 180, 120],
            "thresholds": {
                "minimum_pixels": 120,
                "minimum_pixel_fraction": 0.002,
            },
            "render_evidence": {
                "passed": True,
                "frame_rendered": True,
                "target_rendered": True,
                "site_rendered": True,
                "blockers": [],
            },
        },
    }


def _policy_start_construction() -> dict:
    return {
        "camera_snapshots": [
            {
                "snapshot_id": "reset",
                "cameras": [
                    _passing_policy_start_camera("external"),
                    _passing_policy_start_camera("wrist"),
                    _passing_policy_start_camera("overview"),
                ],
            }
        ]
    }


def test_policy_start_gate_binds_exact_reset_policy_inputs() -> None:
    result = validate_native_task_policy_start_camera_observability(_policy_start_construction())

    assert result["snapshot_id"] == "reset"
    assert result["required_policy_input_roles"] == ["external", "wrist"]
    assert result["target_visible_roles"] == ["external"]
    assert [row["role"] for row in result["cameras"]] == ["external", "wrist"]
    assert result["cameras"][0]["target_visibility_required"] is True
    assert result["cameras"][1]["target_visibility_required"] is False
    assert result["passed"] is True


def test_policy_start_gate_allows_target_absent_rendered_wrist_at_reset() -> None:
    """pi0.5 approached successfully from this exact wrist-camera condition."""

    construction = _policy_start_construction()
    reset_wrist = construction["camera_snapshots"][0]["cameras"][1]
    reset_wrist["observability"].update(
        {
            "passed": False,
            "semantic_passed": False,
            "pixel_count": 0,
            "pixel_fraction": 0.0,
            "bbox_xyxy": None,
            "centroid_within_margin": False,
            "claim": "camera_observes_task_object_without_site_appearance",
            "blockers": ["native_task_camera_semantic_framing_below_threshold"],
        }
    )
    later = copy.deepcopy(construction["camera_snapshots"][0])
    later["snapshot_id"] = "contact_sweep_clearance_00"
    for camera in later["cameras"]:
        camera["snapshot_id"] = later["snapshot_id"]
    later["cameras"][1] = _passing_policy_start_camera("wrist", snapshot_id=later["snapshot_id"])
    construction["camera_snapshots"].append(later)
    construction["camera_gates"] = {
        "wrist": {
            "passed": True,
            "best_snapshot_id": "contact_sweep_clearance_00",
        }
    }

    result = validate_native_task_policy_start_camera_observability(construction)

    wrist = next(row for row in result["cameras"] if row["role"] == "wrist")
    assert wrist["target_visibility_required"] is False
    assert wrist["target_visible"] is False


def test_policy_start_gate_refuses_target_absent_external_view() -> None:
    construction = _policy_start_construction()
    external = construction["camera_snapshots"][0]["cameras"][0]
    external["observability"].update(
        {
            "passed": False,
            "semantic_passed": False,
            "pixel_count": 0,
            "pixel_fraction": 0.0,
            "bbox_xyxy": None,
            "centroid_within_margin": False,
            "claim": "camera_observes_task_object_without_site_appearance",
            "blockers": ["native_task_camera_semantic_framing_below_threshold"],
        }
    )

    with pytest.raises(
        NativeTaskCameraObservabilityError,
        match="native_task_policy_start_camera_role_not_observable:external",
    ):
        validate_native_task_policy_start_camera_observability(construction)


def test_policy_start_gate_refuses_unrendered_wrist_view() -> None:
    construction = _policy_start_construction()
    wrist = construction["camera_snapshots"][0]["cameras"][1]
    wrist["observability"]["render_passed"] = False
    wrist["observability"]["render_evidence"].update(
        {
            "passed": False,
            "frame_rendered": False,
            "blockers": ["native_task_camera_frame_void"],
        }
    )

    with pytest.raises(
        NativeTaskCameraObservabilityError,
        match="native_task_policy_start_camera_role_not_rendered:wrist",
    ):
        validate_native_task_policy_start_camera_observability(construction)


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    (
        (
            lambda value: value.update({"camera_snapshots": []}),
            "native_task_policy_start_camera_snapshot_missing:reset",
        ),
        (
            lambda value: value["camera_snapshots"][0].update(
                {"cameras": [value["camera_snapshots"][0]["cameras"][0]]}
            ),
            "native_task_policy_start_camera_role_missing:wrist",
        ),
    ),
)
def test_policy_start_gate_refuses_missing_exact_evidence(mutation, blocker) -> None:
    construction = _policy_start_construction()
    mutation(construction)

    with pytest.raises(NativeTaskCameraObservabilityError, match=blocker):
        validate_native_task_policy_start_camera_observability(construction)


def test_policy_start_gate_refuses_duplicate_target_role_contract() -> None:
    with pytest.raises(
        NativeTaskCameraObservabilityError,
        match="native_task_policy_start_camera_snapshots_invalid",
    ):
        validate_native_task_policy_start_camera_observability(
            _policy_start_construction(),
            target_visible_roles=("external", "external"),
        )


# --- the r13..r23 condition ------------------------------------------------


def test_an_all_black_frame_fails_however_populated_the_mask_is() -> None:
    """The case that cost eleven paid runs.

    The mask is not merely populated, it is maximal: every pixel is the task
    object, perfectly centred, far past every semantic threshold.  v1 returned
    `passed: True` for exactly this input.
    """

    shape = (100, 200)
    semantic = np.full(shape, 7, dtype=np.int32)
    black = np.zeros((*shape, 3), dtype=np.uint8)

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"7": {"class": "task_object"}},
        rgb=black,
        site_appearance_render_expected=False,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    # The semantic half still says yes -- that is the whole point.
    assert result["semantic_passed"] is True
    assert result["pixel_count"] == 20000
    # The pixel half says no, and the conjunction fails closed.
    assert result["render_passed"] is False
    assert result["passed"] is False
    assert BLOCKER_FRAME_VOID in result["blockers"]
    assert BLOCKER_TARGET_VOID in result["blockers"]
    assert result["rgb_or_model_label_used"] is True


def test_a_black_frame_fails_even_when_the_site_is_not_expected_to_render() -> None:
    """The permission to skip the site check is not a permission to pass dead.

    While the image ships no NuRec renderer the gate cannot demand the captured
    site; it can still demand that *something* was drawn.
    """

    shape = (64, 64)
    result = measure_native_task_camera_observability(
        semantic_ids=np.full(shape, 3, dtype=np.int32),
        id_to_labels={"3": {"class": "task_object"}},
        rgb=np.zeros((*shape, 3), dtype=np.uint8),
        site_appearance_render_expected=False,
        minimum_pixels=10,
        minimum_pixel_fraction=0.001,
    )

    assert result["passed"] is False
    assert result["render_evidence"]["frame_rendered"] is False


def test_a_uniform_non_zero_frame_is_not_a_render() -> None:
    """`interiorgs_bootstrap_0787_841244/views/star_00.png` is 768x1024 of
    exactly RGB(11, 11, 16): void fraction 0.0, mean luminance 12.7, so neither
    a void check nor the existing blank-black check at 2.0 can see it."""

    shape = (64, 64)
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    frame[..., 0] = 11
    frame[..., 1] = 11
    frame[..., 2] = 16

    evidence = measure_native_task_frame_render_evidence(
        rgb=frame, site_appearance_render_expected=False
    )

    assert evidence["frame"]["void_pixel_fraction"] == 0.0
    assert evidence["frame"]["distinct_luminance_levels"] == 1
    assert evidence["passed"] is False
    assert evidence["blockers"] == [BLOCKER_FRAME_UNIFORM]


def test_a_rendered_target_over_a_flat_bright_clear_color_is_not_a_site() -> None:
    """The exact bb16b12e false-positive shape.

    The target is textured and framed, so frame and target radiance are real.
    Everything outside it is the renderer clear colour.  Tonal-level and
    variance checks see the target edges and used to call this a rendered site.
    """

    shape = (100, 200)
    semantic = _semantic(shape)
    frame = np.full((*shape, 3), 231, dtype=np.uint8)
    frame[30:70, 70:130] = _textured((40, 60), low=40, high=180)

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"7": {"class": "task_object"}},
        rgb=frame,
        site_appearance_render_expected=True,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    evidence = result["render_evidence"]
    assert evidence["frame_rendered"] is True
    assert evidence["target_rendered"] is True
    assert evidence["site_rendered"] is False
    assert evidence["site_region"]["dominant_rgb_pixel_fraction"] == 1.0
    assert BLOCKER_SITE_DOMINANT_COLOR in result["blockers"]
    assert result["passed"] is False


def test_a_framed_object_whose_own_pixels_rendered_nothing_fails() -> None:
    """The inverse of r13..r23: the site renders, the subject does not.

    The mask asserts the object is framed; the pixels under it are void.  The
    frame-level check passes -- something was drawn -- so only the target-region
    check can catch this.
    """

    shape = (100, 200)
    semantic = _semantic(shape)
    frame = _textured(shape, low=40, high=255)
    frame[30:70, 70:130] = 0

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"7": {"class": "task_object"}},
        rgb=frame,
        site_appearance_render_expected=True,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    assert result["semantic_passed"] is True
    assert result["render_evidence"]["frame_rendered"] is True
    assert result["render_evidence"]["target_rendered"] is False
    assert result["passed"] is False
    assert BLOCKER_TARGET_VOID in result["blockers"]


# --- legitimate renders must not be false positives ------------------------


def test_a_dark_but_genuinely_rendered_frame_passes() -> None:
    """A night scene is dark, not empty.

    Mean luminance ~10 of 255 -- below the blank-black threshold a mean-only
    gate would use -- but it carries tonal structure, so it is a render.
    """

    shape = (100, 200)
    semantic = _semantic(shape)
    frame = _textured(shape, low=1, high=20)

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"7": {"class": "task_object"}},
        rgb=frame,
        site_appearance_render_expected=True,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    assert result["render_evidence"]["frame"]["luminance_mean"] < 32.0
    assert (
        result["render_evidence"]["frame"]["distinct_luminance_levels"]
        >= MINIMUM_DISTINCT_LUMINANCE_LEVELS
    )
    assert result["render_passed"] is True
    assert result["passed"] is True
    assert result["claim"] == CLAIM_WITH_SITE


def test_the_measured_real_render_band_clears_the_thresholds() -> None:
    """The passing side of the evidence set, as numbers rather than adjectives.

    Seventeen real renders from this Isaac RTX stack, retained locally under
    `output/` (gitignored run artifacts, so quoted here as numbers rather than
    read from disk -- this test stays hermetic), measure void 0.00000..0.01281
    over 196..256 distinct luminance levels at luminance std 19.59..69.24.
    The worst of each must clear its threshold with room.
    """

    worst_real_void_fraction = 0.01281
    worst_real_distinct_levels = 196

    assert worst_real_void_fraction < MAXIMUM_SITE_VOID_PIXEL_FRACTION
    assert MAXIMUM_SITE_VOID_PIXEL_FRACTION / worst_real_void_fraction > 30.0
    assert worst_real_distinct_levels > MINIMUM_DISTINCT_LUMINANCE_LEVELS
    # ... and the shallowest observed failure is nowhere near it.
    shallowest_observed_failure_void_fraction = 0.88
    assert shallowest_observed_failure_void_fraction > MAXIMUM_SITE_VOID_PIXEL_FRACTION

    worst_genuine_room_dominant_rgb_fraction = 0.02122
    shallowest_flat_clear_false_pass_fraction = 0.72014
    assert (
        worst_genuine_room_dominant_rgb_fraction
        < MAXIMUM_SITE_DOMINANT_RGB_PIXEL_FRACTION
        < shallowest_flat_clear_false_pass_fraction
    )


# --- absent evidence is not evidence ---------------------------------------


def test_a_missing_rgb_frame_is_refused_not_passed() -> None:
    shape = (100, 200)

    with pytest.raises(NativeTaskCameraObservabilityError) as raised:
        measure_native_task_camera_observability(
            semantic_ids=_semantic(shape),
            id_to_labels={"7": {"class": "task_object"}},
            rgb=None,
            site_appearance_render_expected=False,
            minimum_pixels=200,
            minimum_pixel_fraction=0.005,
        )

    assert raised.value.errors == ("native_task_camera_rgb_frame_missing",)


def test_the_rgb_argument_has_no_default_so_it_cannot_be_forgotten() -> None:
    """A caller that omits the frame must fail at the call, not get a verdict."""

    with pytest.raises(TypeError):
        measure_native_task_camera_observability(  # type: ignore[call-arg]
            semantic_ids=_semantic((100, 200)),
            id_to_labels={"7": {"class": "task_object"}},
            site_appearance_render_expected=False,
            minimum_pixels=200,
            minimum_pixel_fraction=0.005,
        )


def test_a_site_expectation_must_be_stated() -> None:
    with pytest.raises(TypeError):
        measure_native_task_camera_observability(  # type: ignore[call-arg]
            semantic_ids=_semantic((100, 200)),
            id_to_labels={"7": {"class": "task_object"}},
            rgb=_textured((100, 200)),
            minimum_pixels=200,
            minimum_pixel_fraction=0.005,
        )


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (np.zeros((4, 4), dtype=np.uint8), "native_task_camera_rgb_shape_invalid"),
        (np.zeros((4, 4, 2), dtype=np.uint8), "native_task_camera_rgb_shape_invalid"),
        (np.zeros((0, 4, 3), dtype=np.uint8), "native_task_camera_rgb_shape_invalid"),
        (
            np.full((4, 4, 3), np.nan, dtype=np.float32),
            "native_task_camera_rgb_non_finite",
        ),
    ],
)
def test_an_unreadable_frame_is_refused(frame: np.ndarray, expected: str) -> None:
    with pytest.raises(NativeTaskCameraObservabilityError) as raised:
        measure_native_task_frame_render_evidence(rgb=frame, site_appearance_render_expected=False)

    assert raised.value.errors == (expected,)


def test_a_frame_that_does_not_match_the_semantic_buffer_is_refused() -> None:
    """Measuring one camera's pixels against another's mask is not a verdict."""

    with pytest.raises(NativeTaskCameraObservabilityError) as raised:
        measure_native_task_camera_observability(
            semantic_ids=_semantic((100, 200)),
            id_to_labels={"7": {"class": "task_object"}},
            rgb=_textured((100, 100)),
            site_appearance_render_expected=False,
            minimum_pixels=200,
            minimum_pixel_fraction=0.005,
        )

    assert raised.value.errors == ("native_task_camera_rgb_semantic_shape_mismatch",)


# --- nothing rendered vs THIS content did not render -----------------------


def test_the_r13_signature_fails_only_when_the_site_is_claimed() -> None:
    """A rendered subject on a void site.

    This is what r10..r23 actually looked like, and what every run looks like
    today: the SimReady asset and the robot render, the captured site does not,
    because the pinned image enables no NuRec renderer.  The honest gate fails
    it when the run claims the site, and records the void without failing when
    the run does not.
    """

    shape = (100, 200)
    semantic = _semantic(shape)
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    frame[25:75, 65:135] = _textured((50, 70), low=60, high=200)

    claimed = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"7": {"class": "task_object"}},
        rgb=frame,
        site_appearance_render_expected=True,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )
    unclaimed = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"7": {"class": "task_object"}},
        rgb=frame,
        site_appearance_render_expected=False,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    site_void = claimed["render_evidence"]["site_region"]["void_pixel_fraction"]
    assert site_void > 0.85  # the r10..r23 band, reproduced

    assert claimed["passed"] is False
    assert BLOCKER_SITE_VOID in claimed["blockers"]

    # Not a pass for the site: a pass for the object, with the site's absence
    # measured, recorded, and excluded from the claim.
    assert unclaimed["passed"] is True
    assert unclaimed["site_appearance_claimed"] is False
    assert unclaimed["claim"] == CLAIM_WITHOUT_SITE
    assert unclaimed["render_evidence"]["site_region"]["void_pixel_fraction"] == site_void


def test_a_site_that_renders_while_unclaimed_is_reported() -> None:
    """The stale-declaration alarm.

    Nobody has to remember to re-check the splat after the image is swapped:
    the run that first renders it says so.
    """

    shape = (100, 200)
    result = measure_native_task_camera_observability(
        semantic_ids=_semantic(shape),
        id_to_labels={"7": {"class": "task_object"}},
        rgb=_textured(shape, low=30, high=220),
        site_appearance_render_expected=False,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    assert result["passed"] is True
    assert result["notices"] == [NOTICE_SITE_RENDERED_WHILE_UNCLAIMED]
    # Rendering is still not the same as claiming it.
    assert result["site_appearance_claimed"] is False
    assert result["claim"] == CLAIM_WITHOUT_SITE


# --- the v1 semantic contract is preserved, not replaced -------------------


def test_exact_semantic_pixels_gate_visibility_and_framing() -> None:
    shape = (100, 200)
    result = measure_native_task_camera_observability(
        semantic_ids=_semantic(shape),
        id_to_labels={"7": {"class": "task_object"}},
        rgb=_textured(shape, low=20, high=240),
        site_appearance_render_expected=True,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    assert result["pixel_count"] == 2400
    assert result["bbox_xyxy"] == [70, 30, 129, 69]
    assert result["centroid_within_margin"] is True
    assert result["semantic_passed"] is True
    assert result["passed"] is True


@pytest.mark.parametrize("use_signed", [False, True])
def test_replicator_rgba_tuple_keys_decode_to_exact_scalar_ids(
    use_signed: bool,
) -> None:
    rgba = (240, 4, 111, 255)
    unsigned = sum(value << (8 * index) for index, value in enumerate(rgba))
    scalar = unsigned - 2**32 if use_signed and unsigned >= 2**31 else unsigned
    semantic = np.zeros((32, 48), dtype=np.int64)
    semantic[8:24, 12:36] = scalar

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={str(rgba): {"class": "task_object"}},
        rgb=_textured((32, 48), low=20, high=240),
        site_appearance_render_expected=True,
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    signed = unsigned - 2**32 if unsigned >= 2**31 else unsigned
    assert result["target_semantic_ids"] == sorted({unsigned, signed})
    assert result["pixel_count"] == 384
    assert result["semantic_passed"] is True


def test_wrong_scene_label_cannot_be_counted_as_the_task() -> None:
    shape = (20, 20)
    result = measure_native_task_camera_observability(
        semantic_ids=np.full(shape, 3, dtype=np.int32),
        id_to_labels={"3": {"class": "approved_can"}},
        rgb=_textured(shape, low=20, high=240),
        site_appearance_render_expected=True,
        minimum_pixels=1,
        minimum_pixel_fraction=0.001,
    )

    assert result["target_semantic_ids"] == []
    assert result["pixel_count"] == 0
    assert result["semantic_passed"] is False
    assert result["passed"] is False
    # A frame with nothing to measure inside it is not a target-region verdict.
    assert result["render_evidence"]["target_rendered"] is None


def test_a_large_but_off_frame_centroid_fails_framing() -> None:
    shape = (100, 100)
    semantic = np.zeros(shape, dtype=np.int32)
    semantic[10:90, :20] = 4

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"4": {"class": "task_object"}},
        rgb=_textured(shape, low=20, high=240),
        site_appearance_render_expected=True,
        minimum_pixels=100,
        minimum_pixel_fraction=0.01,
        centroid_margin_fraction=0.15,
    )

    assert result["pixel_count"] == 1600
    assert result["centroid_within_margin"] is False
    assert result["render_passed"] is True  # the frame is fine; the framing is not
    assert result["passed"] is False


def test_a_float_frame_in_zero_to_one_is_read_at_full_scale() -> None:
    """Isaac hands back uint8 here, but a normalised float frame must not be
    mistaken for a black one."""

    shape = (40, 40)
    frame = _textured(shape, low=20, high=240).astype(np.float32) / 255.0

    evidence = measure_native_task_frame_render_evidence(
        rgb=frame, site_appearance_render_expected=False
    )

    assert evidence["frame"]["luminance_max"] > 200.0
    assert evidence["passed"] is True
    assert evidence["appearance_quality_claimed"] is False
    assert evidence["appearance_fidelity_qualified"] is False
    assert evidence["quality_boundary"] == ("render_presence_only_not_appearance_quality")


def test_an_alpha_channel_and_a_batch_axis_are_accepted() -> None:
    shape = (40, 40)
    rgba = np.concatenate(
        [
            _textured(shape, low=20, high=240),
            np.full((*shape, 1), 255, dtype=np.uint8),
        ],
        axis=-1,
    )

    evidence = measure_native_task_frame_render_evidence(
        rgb=rgba[None, ...], site_appearance_render_expected=False
    )

    assert evidence["frame_resolution_hw"] == [40, 40]
    assert evidence["passed"] is True


# --- policy-input saturation ----------------------------------------------


def _clipped(shape: tuple[int, int], *, fraction: float) -> np.ndarray:
    """A textured frame whose leading ``fraction`` of pixels clip in one channel."""

    rows, cols = shape
    frame = _textured(shape, low=0, high=200)
    clipped = int(round(rows * cols * fraction))
    flat = frame.reshape(-1, 3)
    flat[:clipped, 0] = 255
    return flat.reshape(rows, cols, 3)


def test_a_clamped_splat_observation_is_refused_before_any_policy_query() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        REFUSAL_POLICY_INPUT_FRAME_SATURATED,
        validate_native_task_policy_input_frames,
    )

    with pytest.raises(NativeTaskCameraObservabilityError) as failure:
        validate_native_task_policy_input_frames(
            {
                "exterior_image_1_left": _clipped((180, 320), fraction=0.24),
                "wrist_image_left": _clipped((180, 320), fraction=0.027),
            }
        )

    assert failure.value.errors == (
        f"{REFUSAL_POLICY_INPUT_FRAME_SATURATED}:exterior_image_1_left",
    )


def test_the_r13_saturation_band_separates_the_defect_from_a_robot_frame() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        MAXIMUM_POLICY_INPUT_SATURATED_PIXEL_FRACTION,
        measure_native_task_frame_saturation,
    )

    # scene-839873 r13 construction reset frames, linear HDR pixels with a
    # channel above 1.0: external 0.225, overview 0.240, wrist 0.027.
    defect = measure_native_task_frame_saturation(rgb=_clipped((180, 320), fraction=0.225))
    robot = measure_native_task_frame_saturation(rgb=_clipped((180, 320), fraction=0.027))

    assert defect["saturated_channel_pixel_fraction"] == pytest.approx(0.225, abs=1e-4)
    assert defect["chromatic_clip_pixel_fraction"] == pytest.approx(0.225, abs=1e-4)
    assert defect["saturated_white_pixel_fraction"] == 0.0
    assert defect["passed"] is False
    assert robot["passed"] is True
    assert 0.027 < MAXIMUM_POLICY_INPUT_SATURATED_PIXEL_FRACTION < 0.225


def test_a_clean_observation_passes_and_carries_its_evidence() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        validate_native_task_policy_input_frames,
    )

    receipt = validate_native_task_policy_input_frames(
        {"exterior_image_1_left": _textured((24, 32), low=0, high=254)}
    )

    assert receipt["passed"] is True
    assert receipt["views"]["exterior_image_1_left"]["saturated_channel_pixel_fraction"] == 0.0
    assert receipt["views"]["exterior_image_1_left"]["pixel_count"] == 24 * 32


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        ({}, "native_task_policy_input_frames_invalid"),
        ({"": _textured((4, 4))}, "native_task_policy_input_frames_invalid"),
        ({"wrist_image_left": None}, "native_task_camera_rgb_frame_missing"),
    ],
)
def test_the_saturation_gate_refuses_what_it_cannot_read(frames, expected) -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        validate_native_task_policy_input_frames,
    )

    with pytest.raises(NativeTaskCameraObservabilityError) as failure:
        validate_native_task_policy_input_frames(frames)

    assert expected in failure.value.errors[0]


def test_render_evidence_reports_the_clipped_fraction_for_construction_receipts() -> None:
    evidence = measure_native_task_frame_render_evidence(
        rgb=_clipped((32, 32), fraction=0.25),
        site_appearance_render_expected=False,
    )

    assert evidence["frame"]["saturated_channel_pixel_fraction"] == pytest.approx(0.25)


def test_prepolicy_visual_gate_refuses_scene839873_dark_splat_signature() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        REFUSAL_PREPOLICY_VISUAL_FRAME_NEAR_BLACK,
        measure_native_task_prepolicy_visual_frames,
    )

    dark = np.zeros((180, 320, 3), dtype=np.uint8)
    dark[:, :80] = _textured((180, 80), low=8, high=70)
    receipt = measure_native_task_prepolicy_visual_frames(
        {"external": dark, "wrist": np.roll(dark, 1, axis=1), "overview": np.roll(dark, 2, axis=1)},
        candidate_policy_loaded=False,
    )

    assert receipt["passed"] is False
    assert receipt["frame_structure_passed"] is False
    assert receipt["policy_observation_integrity_passed"] is False
    assert all(
        any(REFUSAL_PREPOLICY_VISUAL_FRAME_NEAR_BLACK in blocker for blocker in row["blockers"])
        for row in receipt["views"].values()
    )
    assert receipt["candidate_policy_loaded"] is False
    assert receipt["candidate_policy_queried"] is False


def test_prepolicy_visual_gate_accepts_three_distinct_structured_views() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        measure_native_task_prepolicy_visual_frames,
    )

    receipt = measure_native_task_prepolicy_visual_frames(
        {
            "external": _textured((24, 32), low=30, high=180),
            "wrist": np.roll(_textured((24, 32), low=35, high=185), 1, axis=1),
            "overview": np.roll(_textured((24, 32), low=40, high=190), 2, axis=0),
        },
        candidate_policy_loaded=False,
    )

    assert receipt["passed"] is True
    assert receipt["frame_structure_passed"] is True
    assert receipt["blockers"] == []
    assert len({row["frame_digest"] for row in receipt["views"].values()}) == 3
    # Structural only: nothing here may unlock a policy query.
    assert receipt["policy_observation_integrity_passed"] is False
    assert receipt["appearance_reference_parity_passed"] is False
    assert receipt["human_visual_review_status"] == "pending"
    assert receipt["policy_observation_integrity_blockers"] == [
        "native_task_appearance_reference_parity_missing",
        "native_task_human_visual_review_not_approved",
    ]


def _three_views() -> dict[str, np.ndarray]:
    return {
        "external": _textured((24, 32), low=30, high=180),
        "wrist": np.roll(_textured((24, 32), low=35, high=185), 1, axis=1),
        "overview": np.roll(_textured((24, 32), low=40, high=190), 2, axis=0),
    }


def _authority(*, backend: str, status: str = "approved", parity: bool = True) -> dict:
    from blueprint_pipeline.native_task_camera_observability import (
        build_policy_observation_integrity_authority,
    )

    return build_policy_observation_integrity_authority(
        appearance_render_backend_receipt_digest=backend,
        reference_renderer_identity="nvcr.io/nvidia/nre/nre@sha256:pinned",
        reference_source_sha256="sha256:" + "9" * 64,
        views={
            view: {
                "reference_png_sha256": "sha256:" + "1" * 64,
                "candidate_png_sha256": "sha256:" + "2" * 64,
            }
            for view in ("external", "wrist", "overview")
        },
        parity_passed=parity,
        human_review_status=status,
        reviewer="reviewer",
        contact_sheet_sha256="sha256:" + "3" * 64,
    )


def test_prepolicy_gate_records_the_caller_supplied_policy_load_state() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        NativeTaskCameraObservabilityError,
        measure_native_task_prepolicy_visual_frames,
    )

    loaded = measure_native_task_prepolicy_visual_frames(
        _three_views(), candidate_policy_loaded=True
    )
    assert loaded["candidate_policy_loaded"] is True
    assert loaded["candidate_policy_queried"] is False
    with pytest.raises(TypeError):
        measure_native_task_prepolicy_visual_frames(_three_views())  # type: ignore[call-arg]
    with pytest.raises(NativeTaskCameraObservabilityError):
        measure_native_task_prepolicy_visual_frames(
            _three_views(), candidate_policy_loaded="no"  # type: ignore[arg-type]
        )


def test_prepolicy_gate_unlocks_only_with_bound_parity_and_approved_review() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        measure_native_task_prepolicy_visual_frames,
    )

    backend = "sha256:" + "b" * 64
    passing = measure_native_task_prepolicy_visual_frames(
        _three_views(),
        candidate_policy_loaded=True,
        observation_integrity_authority=_authority(backend=backend),
        appearance_render_backend_receipt_digest=backend,
    )
    assert passing["policy_observation_integrity_passed"] is True
    assert passing["policy_observation_integrity_blockers"] == []
    assert passing["appearance_reference_parity_binding"]["backend_bound"] is True

    unbound = measure_native_task_prepolicy_visual_frames(
        _three_views(),
        candidate_policy_loaded=True,
        observation_integrity_authority=_authority(backend=backend),
        appearance_render_backend_receipt_digest="sha256:" + "c" * 64,
    )
    assert unbound["policy_observation_integrity_passed"] is False
    assert unbound["policy_observation_integrity_blockers"] == [
        "native_task_appearance_reference_parity_backend_mismatch"
    ]

    failed_parity = measure_native_task_prepolicy_visual_frames(
        _three_views(),
        candidate_policy_loaded=True,
        observation_integrity_authority=_authority(backend=backend, parity=False),
        appearance_render_backend_receipt_digest=backend,
    )
    assert failed_parity["policy_observation_integrity_blockers"] == [
        "native_task_appearance_reference_parity_failed"
    ]

    unreviewed = measure_native_task_prepolicy_visual_frames(
        _three_views(),
        candidate_policy_loaded=True,
        observation_integrity_authority=_authority(backend=backend, status="pending"),
        appearance_render_backend_receipt_digest=backend,
    )
    assert unreviewed["human_visual_review_status"] == "pending"
    assert unreviewed["policy_observation_integrity_blockers"] == [
        "native_task_human_visual_review_not_approved"
    ]


def test_structural_failure_keeps_integrity_false_even_with_a_perfect_authority() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        measure_native_task_prepolicy_visual_frames,
    )

    backend = "sha256:" + "b" * 64
    dark = np.zeros((180, 320, 3), dtype=np.uint8)
    dark[:, :80] = _textured((180, 80), low=8, high=70)
    receipt = measure_native_task_prepolicy_visual_frames(
        {"external": dark, "wrist": np.roll(dark, 1, axis=1), "overview": np.roll(dark, 2, axis=1)},
        candidate_policy_loaded=True,
        observation_integrity_authority=_authority(backend=backend),
        appearance_render_backend_receipt_digest=backend,
    )
    assert receipt["frame_structure_passed"] is False
    assert receipt["appearance_reference_parity_passed"] is True
    assert receipt["policy_observation_integrity_passed"] is False
    assert "native_task_prepolicy_frame_structure_failed" in receipt[
        "policy_observation_integrity_blockers"
    ]


def test_invalid_authority_is_a_typed_refusal() -> None:
    from blueprint_pipeline.native_task_camera_observability import (
        NativeTaskCameraObservabilityError,
        measure_native_task_prepolicy_visual_frames,
        validate_policy_observation_integrity_authority,
    )

    with pytest.raises(NativeTaskCameraObservabilityError) as excinfo:
        validate_policy_observation_integrity_authority({"schema_version": "wrong"})
    assert all(
        error.startswith("native_task_policy_observation_integrity_authority_invalid:")
        for error in excinfo.value.errors
    )
    with pytest.raises(NativeTaskCameraObservabilityError):
        measure_native_task_prepolicy_visual_frames(
            _three_views(),
            candidate_policy_loaded=True,
            observation_integrity_authority={"schema_version": "wrong"},
            appearance_render_backend_receipt_digest="sha256:" + "b" * 64,
        )


def test_chromatic_diagnostics_describe_scattered_saturated_splats_without_gating() -> None:
    """Diagnostic descriptors that separate a coherent surface from splat breakup.

    The three Scene 839873 failing frames measured 5.16 / 5.10 / 2.14 percent
    of pixels with RGB spread above 64; a grey textured surface measures zero.
    Thresholds stay uncalibrated until a known-good/known-bad set exists.
    """

    from blueprint_pipeline.native_task_camera_observability import (
        measure_native_task_frame_chromatic_diagnostics,
    )

    grey = np.repeat(_textured((64, 64), low=60, high=200)[..., :1], 3, axis=-1)
    coherent = measure_native_task_frame_chromatic_diagnostics(grey)
    assert coherent["rgb_spread_pixel_fraction"] == 0.0
    assert coherent["local_chroma_outlier_fraction"] == 0.0
    assert coherent["gating"] == "diagnostic_only_until_calibration_set_preregistered"

    rainbow = grey.copy()
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 64, 120)
    cols = rng.integers(0, 64, 120)
    rainbow[rows, cols] = rng.choice(
        np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]], dtype=np.uint8), 120
    )
    broken = measure_native_task_frame_chromatic_diagnostics(rainbow)
    assert broken["rgb_spread_pixel_fraction"] > 0.02
    assert broken["local_chroma_outlier_fraction"] > 0.02
    assert broken["local_chroma_outlier_fraction"] > coherent["local_chroma_outlier_fraction"]
