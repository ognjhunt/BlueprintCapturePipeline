"""Measure task-object visibility and framing from native semantic pixels.

Two independent questions, both required.

``semantic``  -- is the *right object* framed?  Answered from the semantic
segmentation AOV, which is built from scene-graph membership.

``render``    -- did anything actually get drawn?  Answered from the RGB
frame, which is built from radiance.

The v1 contract asked only the first and reported ``passed`` as if it had
answered the second.  A segmentation AOV comes back fully populated whether or
not a single photon was traced, so the two came apart: arena construction runs
r10 through r23 reported all three cameras ``passed`` against frames that were
88 to 92 percent pure black (PR #800, PR #801).  The v1 field
``rgb_or_model_label_used: False`` said plainly that no pixel was consulted;
nothing acted on it.

So the RGB frame is now a required input, an absent one is a refusal rather
than a pass, and ``passed`` is a conjunction that includes radiance.

What this module can and cannot assert
--------------------------------------
Radiance is measured in three separately named places, because "nothing
rendered" and "this particular content did not render" are different claims
and only the first is unconditionally checkable here:

``frame_rendered``   the frame is not void and not one repeated value.  This
                     is the weak claim: something was drawn.
``target_rendered``  the pixels the semantic mask calls the target carry a
                     rendered image rather than nothing.  This is the direct
                     repair of the v1 defect -- the mask can no longer vouch
                     for pixels that show nothing.
``site_rendered``    the frame outside the target is not mostly void, i.e. the
                     captured site appeared.  Measured always; gated only when
                     the caller declares the runtime can render it.

That last conditional is not a convenience.  As of 2026-08-19 the pinned Isaac
image ships no NuRec renderer at all -- ``omni.nurec``, ``omni.rtx.nre`` and
``omni.usd.schema.omni_nurec`` all fail to enable and the RTX path reports
"Failed to create nrend renderer with error code 2" -- so the captured site is
*legitimately* absent from every frame that image produces while the robot and
the SimReady asset still render.  Gating ``site_rendered`` unconditionally
would fail every run for a cause this module cannot see or name.  Leaving it
unmeasured is how r10..r23 happened.  So it is measured, reported, and
attached to an explicit declaration the caller cannot omit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "native_task_camera_observability.v2"
RENDER_EVIDENCE_SCHEMA_VERSION = "native_task_camera_render_evidence.v1"

# --- render-evidence thresholds -------------------------------------------
#
# Failing side -- the adjudicated r10..r23 arena frames: 88 to 92 percent of
# the frame pure black with the task object genuinely in frame.  Rebuilt
# synthetically (a real render confined to a 12/10/8 percent island on void)
# that signature measures void 0.880/0.900/0.921 at luminance std 47..58 over
# 222..229 distinct levels -- so variance and tonal range CANNOT see it, and a
# void fraction is the only statistic that can.
#
# Passing side -- every real render from this Isaac RTX stack retained locally
# under `output/` (gitignored run artifacts, not committed fixtures): five
# `lightwheel_sink_isaac_canary_20260803_attempt10` frames, eight
# `interiorgs_bootstrap_0787_841244/views` room renders, and two review seed
# frames.  Void fraction 0.00000..0.01281, luminance std 19.59..69.24,
# 196..256 distinct luminance levels.  Re-run through the finished gate, 16 of
# those 17 pass; the one that does not is the degenerate frame below.
#
# A ceiling of 0.50 therefore sits 39x above the darkest real render measured
# and 38 points below the shallowest observed failure.  It is also what
# `isaac_review_renderer_canary.SEVERE_CLIPPING_MIN_FRACTION` already uses for
# "too much of this frame is saturated to be a render", so the lane keeps one
# story about how much void is disqualifying.  Half a frame of pure void is
# not reachable by a camera pointed into a room, however dark the room is.
MAXIMUM_SITE_VOID_PIXEL_FRACTION = 0.50

# A frame of one repeated value carries no render at all, whatever that value
# is, and a void fraction cannot see it when the value is not zero.  This is
# not hypothetical: a retained render at
# `output/interiorgs_bootstrap_0787_841244/views/star_00.png` is 768x1024 of
# exactly RGB(11, 11, 16) -- void fraction 0.0, and a mean luminance of 12.7
# that clears the existing blank-black check at 2.0.
# Real renders here measure 196..256 distinct luminance levels, so a floor of 8
# is ~24x below the worst of them: it rejects degenerate frames and leaves even
# a badly underexposed genuine render untouched.
MINIMUM_DISTINCT_LUMINANCE_LEVELS = 8

# Lifted from `isaac_review_renderer_canary.FLAT_FRAME_MAX_LUMINANCE_STD`.
# The lowest-variance real render measured here is 19.59.
MINIMUM_LUMINANCE_STD = 1.0

# Reported, never gated: the band `isaac_review_renderer_canary` calls blank
# black.  Kept as evidence so a frame drifting toward void is visible in the
# receipt before it crosses anything.
NEAR_BLACK_LUMINANCE_MAX = 2.0

BLOCKER_FRAME_VOID = "native_task_camera_rgb_frame_void"
BLOCKER_FRAME_UNIFORM = "native_task_camera_rgb_frame_uniform"
BLOCKER_FRAME_TONAL_RANGE = "native_task_camera_rgb_frame_tonal_range_below_floor"
BLOCKER_TARGET_VOID = "native_task_camera_rgb_target_region_void"
BLOCKER_TARGET_UNIFORM = "native_task_camera_rgb_target_region_uniform"
BLOCKER_TARGET_TONAL_RANGE = (
    "native_task_camera_rgb_target_region_tonal_range_below_floor"
)
BLOCKER_SITE_VOID = "native_task_camera_rgb_site_void_fraction_above_ceiling"
BLOCKER_SEMANTIC_FRAMING = "native_task_camera_semantic_framing_below_threshold"

# Not a blocker: the caller declared the runtime cannot render the captured
# site, and it rendered anyway.  That is not a defect -- it means the image
# changed and the declaration is stale, which is exactly the thing that would
# otherwise slip past unnoticed.
NOTICE_SITE_RENDERED_WHILE_UNCLAIMED = (
    "native_task_camera_site_rendered_while_unclaimed"
)

REFUSAL_RGB_MISSING = "native_task_camera_rgb_frame_missing"
REFUSAL_RGB_SHAPE = "native_task_camera_rgb_shape_invalid"
REFUSAL_RGB_NON_FINITE = "native_task_camera_rgb_non_finite"
REFUSAL_RGB_SEMANTIC_MISMATCH = "native_task_camera_rgb_semantic_shape_mismatch"

CLAIM_WITH_SITE = "camera_observes_task_object_in_rendered_site"
CLAIM_WITHOUT_SITE = "camera_observes_task_object_without_site_appearance"


class NativeTaskCameraObservabilityError(ValueError):
    """Stable semantic/framing/render-evidence failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _as_uint8_rgb(rgb: Any) -> Any:
    """Normalise a native RGB(A) frame to HxWx3 uint8, or refuse.

    Refusal, not a best-effort zero frame: a frame that cannot be read is
    absent evidence, and absent evidence must never reach a verdict.
    """

    import numpy as np

    if rgb is None:
        raise NativeTaskCameraObservabilityError([REFUSAL_RGB_MISSING])
    array = np.asarray(rgb)
    if array.dtype == np.dtype(object) or not array.size:
        raise NativeTaskCameraObservabilityError([REFUSAL_RGB_SHAPE])
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise NativeTaskCameraObservabilityError([REFUSAL_RGB_SHAPE])
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not bool(np.isfinite(array).all()):
            raise NativeTaskCameraObservabilityError([REFUSAL_RGB_NON_FINITE])
        values = array.astype(np.float64)
        if float(values.max()) <= 1.0:
            values = values * 255.0
    elif array.dtype == np.dtype(bool) or np.issubdtype(array.dtype, np.integer):
        values = array.astype(np.float64)
    else:
        raise NativeTaskCameraObservabilityError([REFUSAL_RGB_SHAPE])
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _region_statistics(luminance: Any, void: Any, selector: Any) -> dict[str, Any]:
    """Void fraction and tonal structure over one region of a frame."""

    import numpy as np

    values = luminance[selector]
    if not values.size:
        return {
            "pixel_count": 0,
            "void_pixel_fraction": None,
            "distinct_luminance_levels": 0,
            "luminance_mean": None,
            "luminance_std": None,
        }
    return {
        "pixel_count": int(values.size),
        "void_pixel_fraction": float(void[selector].mean()),
        "distinct_luminance_levels": int(
            np.unique(np.rint(values).astype(np.uint8)).size
        ),
        "luminance_mean": float(values.mean()),
        "luminance_std": float(values.std()),
    }


def _structure_blockers(
    statistics: Mapping[str, Any],
    *,
    void_blocker: str,
    uniform_blocker: str,
    tonal_range_blocker: str,
) -> list[str]:
    """Blockers for "this region carries a rendered image", nothing more.

    Deliberately not an exposure check.  Whether the region is *well lit* is
    not measurable from a single frame without a reference, and a genuinely
    under-exposed render is a real state; what is checkable is whether
    anything was drawn there at all.
    """

    levels = int(statistics["distinct_luminance_levels"])
    void_fraction = statistics["void_pixel_fraction"]
    std = statistics["luminance_std"]
    if void_fraction is not None and float(void_fraction) >= 1.0:
        return [void_blocker]
    if levels <= 1 or std is None or float(std) < MINIMUM_LUMINANCE_STD:
        return [uniform_blocker]
    if levels < MINIMUM_DISTINCT_LUMINANCE_LEVELS:
        return [tonal_range_blocker]
    return []


def measure_native_task_frame_render_evidence(
    *,
    rgb: Any,
    site_appearance_render_expected: bool,
    expected_resolution_hw: Sequence[int] | None = None,
    target_mask: Any = None,
) -> dict[str, Any]:
    """Decide from radiance alone what this frame shows.

    Independent of every semantic threshold, so it cannot be satisfied by a
    populated segmentation mask.  ``target_mask`` selects which pixels the
    semantic layer claims are the target; it is used to ask whether *those*
    pixels rendered, and to separate the site from the subject.

    ``site_appearance_render_expected`` has no default.  A caller that will
    not state whether the runtime can render the captured site cannot get a
    verdict about it.
    """

    import numpy as np

    if not isinstance(site_appearance_render_expected, bool):
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_site_expectation_invalid"]
        )
    frame = _as_uint8_rgb(rgb)
    height, width = (int(value) for value in frame.shape[:2])
    if expected_resolution_hw is not None:
        expected = [int(value) for value in expected_resolution_hw]
        if expected != [height, width]:
            raise NativeTaskCameraObservabilityError([REFUSAL_RGB_SEMANTIC_MISMATCH])

    void = frame.max(axis=-1) == 0
    luminance = frame.astype(np.float64).mean(axis=-1)
    everywhere = np.ones_like(void, dtype=bool)
    frame_statistics = _region_statistics(luminance, void, everywhere)
    frame_statistics["near_black_pixel_fraction"] = float(
        (luminance <= NEAR_BLACK_LUMINANCE_MAX).mean()
    )
    frame_statistics["luminance_min"] = float(luminance.min())
    frame_statistics["luminance_max"] = float(luminance.max())

    mask = None
    if target_mask is not None:
        candidate = np.asarray(target_mask).astype(bool)
        if candidate.shape != void.shape:
            raise NativeTaskCameraObservabilityError([REFUSAL_RGB_SEMANTIC_MISMATCH])
        mask = candidate
    target_statistics = (
        _region_statistics(luminance, void, mask)
        if mask is not None and bool(mask.any())
        else None
    )
    site_statistics = (
        _region_statistics(luminance, void, ~mask)
        if mask is not None and bool((~mask).any())
        else None
    )

    blockers = _structure_blockers(
        frame_statistics,
        void_blocker=BLOCKER_FRAME_VOID,
        uniform_blocker=BLOCKER_FRAME_UNIFORM,
        tonal_range_blocker=BLOCKER_FRAME_TONAL_RANGE,
    )
    frame_rendered = not blockers

    target_rendered: bool | None = None
    if target_statistics is not None:
        target_blockers = _structure_blockers(
            target_statistics,
            void_blocker=BLOCKER_TARGET_VOID,
            uniform_blocker=BLOCKER_TARGET_UNIFORM,
            tonal_range_blocker=BLOCKER_TARGET_TONAL_RANGE,
        )
        target_rendered = not target_blockers
        blockers.extend(target_blockers)

    site_rendered: bool | None = None
    notices: list[str] = []
    if site_statistics is not None:
        site_rendered = bool(
            float(site_statistics["void_pixel_fraction"])
            <= MAXIMUM_SITE_VOID_PIXEL_FRACTION
        )
        if site_appearance_render_expected and not site_rendered:
            blockers.append(BLOCKER_SITE_VOID)
        if not site_appearance_render_expected and site_rendered:
            notices.append(NOTICE_SITE_RENDERED_WHILE_UNCLAIMED)

    return {
        "schema_version": RENDER_EVIDENCE_SCHEMA_VERSION,
        "frame_resolution_hw": [height, width],
        "frame": frame_statistics,
        "target_region": target_statistics,
        "site_region": site_statistics,
        "frame_rendered": frame_rendered,
        "target_rendered": target_rendered,
        "site_rendered": site_rendered,
        "site_appearance_render_expected": site_appearance_render_expected,
        "site_appearance_claimed": bool(
            site_appearance_render_expected and site_rendered
        ),
        "thresholds": {
            "maximum_site_void_pixel_fraction": MAXIMUM_SITE_VOID_PIXEL_FRACTION,
            "minimum_distinct_luminance_levels": MINIMUM_DISTINCT_LUMINANCE_LEVELS,
            "minimum_luminance_std": MINIMUM_LUMINANCE_STD,
            "near_black_luminance_max": NEAR_BLACK_LUMINANCE_MAX,
        },
        "blockers": sorted(set(blockers)),
        "notices": sorted(set(notices)),
        "passed": not blockers,
        "measurement_authority": "native_rgb_frame",
    }


def measure_native_task_camera_observability(
    *,
    semantic_ids: Any,
    id_to_labels: Mapping[str, Any],
    rgb: Any,
    site_appearance_render_expected: bool,
    target_label: str = "task_object",
    minimum_pixels: int,
    minimum_pixel_fraction: float,
    centroid_margin_fraction: float = 0.05,
) -> dict[str, Any]:
    """Gate exact target-class pixels AND the radiance of the pixels they name.

    ``rgb`` has no default on purpose.  A caller that cannot supply the frame
    must fail at the call rather than receive a verdict drawn from the mask.
    """

    import numpy as np

    semantic = np.asarray(semantic_ids)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if semantic.ndim != 2 or not semantic.size:
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_semantic_shape_invalid"]
        )
    if (
        isinstance(minimum_pixels, bool)
        or int(minimum_pixels) < 1
        or not math.isfinite(float(minimum_pixel_fraction))
        or float(minimum_pixel_fraction) <= 0.0
        or not math.isfinite(float(centroid_margin_fraction))
        or float(centroid_margin_fraction) < 0.0
        or float(centroid_margin_fraction) >= 0.5
    ):
        raise NativeTaskCameraObservabilityError(
            ["native_task_camera_threshold_invalid"]
        )
    target_ids: list[int] = []
    for identifier, entry in id_to_labels.items():
        label = entry.get("class") if isinstance(entry, Mapping) else entry
        if label != target_label:
            continue
        try:
            target_ids.append(int(identifier))
        except (TypeError, ValueError) as exc:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_invalid"]
            ) from exc
    mask = np.isin(semantic.astype(np.int64), target_ids)
    count = int(mask.sum())
    height, width = (int(value) for value in mask.shape)
    fraction = count / float(height * width)
    bbox: list[int] | None = None
    centroid: list[float] | None = None
    centroid_framed = False
    if count:
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [
            float(xs.mean() / max(1, width - 1)),
            float(ys.mean() / max(1, height - 1)),
        ]
        margin = float(centroid_margin_fraction)
        centroid_framed = all(margin <= value <= 1.0 - margin for value in centroid)
    semantic_passed = (
        count >= int(minimum_pixels)
        and fraction >= float(minimum_pixel_fraction)
        and centroid_framed
    )
    render = measure_native_task_frame_render_evidence(
        rgb=rgb,
        site_appearance_render_expected=site_appearance_render_expected,
        expected_resolution_hw=[height, width],
        target_mask=mask,
    )
    blockers = list(render["blockers"])
    if not semantic_passed:
        blockers.append(BLOCKER_SEMANTIC_FRAMING)
    passed = semantic_passed and bool(render["passed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "target_label": target_label,
        "target_semantic_ids": target_ids,
        "pixel_count": count,
        "pixel_fraction": fraction,
        "bbox_xyxy": bbox,
        "centroid_xy_fraction": centroid,
        "centroid_within_margin": centroid_framed,
        "frame_resolution_hw": [height, width],
        "thresholds": {
            "minimum_pixels": int(minimum_pixels),
            "minimum_pixel_fraction": float(minimum_pixel_fraction),
            "centroid_margin_fraction": float(centroid_margin_fraction),
        },
        "semantic_passed": semantic_passed,
        "render_passed": bool(render["passed"]),
        "render_evidence": render,
        "site_appearance_claimed": bool(render["site_appearance_claimed"]),
        "claim": (
            CLAIM_WITH_SITE
            if passed and render["site_appearance_claimed"]
            else CLAIM_WITHOUT_SITE
        ),
        "blockers": sorted(set(blockers)),
        "notices": list(render["notices"]),
        "passed": passed,
        "measurement_authority": "native_semantic_segmentation_aov+native_rgb_frame",
        "rgb_or_model_label_used": True,
        "model_label_used": False,
    }


__all__ = [
    "CLAIM_WITHOUT_SITE",
    "CLAIM_WITH_SITE",
    "MAXIMUM_SITE_VOID_PIXEL_FRACTION",
    "MINIMUM_DISTINCT_LUMINANCE_LEVELS",
    "MINIMUM_LUMINANCE_STD",
    "NativeTaskCameraObservabilityError",
    "RENDER_EVIDENCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "measure_native_task_camera_observability",
    "measure_native_task_frame_render_evidence",
]
