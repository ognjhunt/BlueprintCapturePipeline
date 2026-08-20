"""The camera gate must read radiance, not just scene-graph membership.

r10..r23 reported all three cameras `passed` against frames 88 to 92 percent
pure black, because v1 decided `passed` from a semantic segmentation mask and
recorded, honestly and uselessly, `rgb_or_model_label_used: False`.
"""

from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.native_task_camera_observability import (
    BLOCKER_FRAME_UNIFORM,
    BLOCKER_FRAME_VOID,
    BLOCKER_SITE_VOID,
    BLOCKER_TARGET_VOID,
    CLAIM_WITHOUT_SITE,
    CLAIM_WITH_SITE,
    MAXIMUM_SITE_VOID_PIXEL_FRACTION,
    MINIMUM_DISTINCT_LUMINANCE_LEVELS,
    NOTICE_SITE_RENDERED_WHILE_UNCLAIMED,
    NativeTaskCameraObservabilityError,
    measure_native_task_camera_observability,
    measure_native_task_frame_render_evidence,
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
        measure_native_task_frame_render_evidence(
            rgb=frame, site_appearance_render_expected=False
        )

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

    assert raised.value.errors == (
        "native_task_camera_rgb_semantic_shape_mismatch",
    )


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
    assert (
        unclaimed["render_evidence"]["site_region"]["void_pixel_fraction"] == site_void
    )


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
