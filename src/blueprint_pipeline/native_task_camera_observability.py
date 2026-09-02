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

That last conditional is not a convenience.  The original dev2 Arena image did
not ship ``omni.rtx.spg``, so the captured site was absent while the robot and
SimReady asset still rendered.  The Arena lane now pins a NuRec-capable image
and claims the site; other callers still have to state their own capability.
Measuring the site in both cases preserves the evidence and makes a stale
declaration visible instead of silently changing the claim.
"""

from __future__ import annotations

import ast
import hashlib
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

# Passing side -- eleven genuine InteriorGS room renders (the twelve retained
# views minus the one exactly-uniform rejected frame) have a most-common exact
# RGB fraction of 0.00049..0.02122.  The three sealed bb16b12e ParticleField
# false positives instead show the robot/target over one clear colour covering
# 0.72014..0.87681 of the frame.  A 0.50 ceiling is therefore over 23x above
# the worst genuine site render and still 22 points below the shallowest
# observed false pass.  Object-only canaries are intentionally excluded from
# this site's passing side: a rendered object over a clear colour does not
# prove that the captured room appeared.
MAXIMUM_SITE_DOMINANT_RGB_PIXEL_FRACTION = 0.50

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

# --- policy-input saturation ----------------------------------------------
#
# A ParticleField splat is display-referred sRGB and Omniverse RTX composites
# it as-is.  When the lane instead forced the splat through the HDR pipeline
# (``/rtx/rtpt/gaussian/skipTonemapping/enabled=false``), the ``rgb``
# annotator became a per-channel clamp of radiance up to 60x display white.
# The scene-839873 r13 construction reset frames carried 22.5 percent
# (external) and 24.0 percent (overview) of pixels with a channel above 1.0,
# against 2.7 percent on the wrist camera that mostly framed the robot and
# the table.  Under the clamp those pixels are white blobs with chromatic
# fringes, and that exact frame was handed to both candidates as their
# observation while every retained review PNG had been display-encoded from
# the HDR buffer, so no upstream gate could see it.
#
# This gate reads the exact policy-input arrays, never a review encode, and
# refuses the episode before any candidate query.  A ceiling of 0.10 sits
# well above the 2.7 percent a robot-and-table frame measured and well below
# the 22-24 percent the defect produced.
MAXIMUM_POLICY_INPUT_SATURATED_PIXEL_FRACTION = 0.10
SATURATED_CHANNEL_LEVEL = 255

POLICY_INPUT_SATURATION_SCHEMA_VERSION = "native_task_policy_input_frame_saturation.v1"
REFUSAL_POLICY_INPUT_FRAME_SATURATED = "native_task_policy_input_frame_saturated"
REFUSAL_POLICY_INPUT_FRAMES_INVALID = "native_task_policy_input_frames_invalid"
PREPOLICY_VISUAL_REQUIRED_VIEWS = frozenset({"external", "wrist", "overview"})
MAXIMUM_PREPOLICY_NEAR_BLACK_PIXEL_FRACTION = 0.50
REFUSAL_PREPOLICY_VISUAL_FRAME_NEAR_BLACK = (
    "native_task_prepolicy_visual_frame_near_black_fraction_above_ceiling"
)
REFUSAL_PREPOLICY_VISUAL_FRAME_INVALID = "native_task_prepolicy_visual_frame_invalid"
REFUSAL_PREPOLICY_VISUAL_FRAME_DUPLICATE = "native_task_prepolicy_visual_frame_duplicate"

BLOCKER_FRAME_VOID = "native_task_camera_rgb_frame_void"
BLOCKER_FRAME_UNIFORM = "native_task_camera_rgb_frame_uniform"
BLOCKER_FRAME_TONAL_RANGE = "native_task_camera_rgb_frame_tonal_range_below_floor"
BLOCKER_TARGET_VOID = "native_task_camera_rgb_target_region_void"
BLOCKER_TARGET_UNIFORM = "native_task_camera_rgb_target_region_uniform"
BLOCKER_TARGET_TONAL_RANGE = "native_task_camera_rgb_target_region_tonal_range_below_floor"
BLOCKER_SITE_VOID = "native_task_camera_rgb_site_void_fraction_above_ceiling"
BLOCKER_SITE_DOMINANT_COLOR = "native_task_camera_rgb_site_dominant_color_fraction_above_ceiling"
BLOCKER_SEMANTIC_FRAMING = "native_task_camera_semantic_framing_below_threshold"

# Not a blocker: the caller declared the runtime cannot render the captured
# site, and it rendered anyway.  That is not a defect -- it means the image
# changed and the declaration is stale, which is exactly the thing that would
# otherwise slip past unnoticed.
NOTICE_SITE_RENDERED_WHILE_UNCLAIMED = "native_task_camera_site_rendered_while_unclaimed"

REFUSAL_RGB_MISSING = "native_task_camera_rgb_frame_missing"
REFUSAL_RGB_SHAPE = "native_task_camera_rgb_shape_invalid"
REFUSAL_RGB_NON_FINITE = "native_task_camera_rgb_non_finite"
REFUSAL_RGB_SEMANTIC_MISMATCH = "native_task_camera_rgb_semantic_shape_mismatch"

CLAIM_WITH_SITE = "camera_observes_task_object_in_rendered_site"
CLAIM_WITHOUT_SITE = "camera_observes_task_object_without_site_appearance"

POLICY_START_OBSERVABILITY_SCHEMA_VERSION = "native_task_policy_start_camera_observability.v1"
POLICY_START_SNAPSHOT_ID = "reset"
POLICY_INPUT_CAMERA_ROLES = ("external", "wrist")
POLICY_START_TARGET_VISIBLE_ROLES = ("external",)

REFUSAL_POLICY_START_SNAPSHOTS_INVALID = "native_task_policy_start_camera_snapshots_invalid"
REFUSAL_POLICY_START_SNAPSHOT_MISSING = "native_task_policy_start_camera_snapshot_missing"
REFUSAL_POLICY_START_SNAPSHOT_DUPLICATE = "native_task_policy_start_camera_snapshot_duplicate"
REFUSAL_POLICY_START_CAMERAS_INVALID = "native_task_policy_start_cameras_invalid"
REFUSAL_POLICY_START_ROLE_MISSING = "native_task_policy_start_camera_role_missing"
REFUSAL_POLICY_START_ROLE_DUPLICATE = "native_task_policy_start_camera_role_duplicate"
REFUSAL_POLICY_START_ROLE_NOT_OBSERVABLE = "native_task_policy_start_camera_role_not_observable"
REFUSAL_POLICY_START_ROLE_NOT_RENDERED = "native_task_policy_start_camera_role_not_rendered"


class NativeTaskCameraObservabilityError(ValueError):
    """Stable semantic/framing/render-evidence failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def validate_native_task_policy_start_camera_observability(
    construction_result: Mapping[str, Any],
    *,
    snapshot_id: str = POLICY_START_SNAPSHOT_ID,
    required_roles: Sequence[str] = POLICY_INPUT_CAMERA_ROLES,
    target_visible_roles: Sequence[str] = POLICY_START_TARGET_VISIBLE_ROLES,
) -> dict[str, Any]:
    """Prove the actual policy-start views, never a later scripted best view.

    Construction records several camera snapshots while its scripted controller
    approaches and contacts the task object.  A camera becoming useful *after*
    that controller has moved the robot cannot prove what a learned policy saw
    at reset.  The external view must frame the task object at policy start.  A
    wrist camera may legitimately point at the floor until the arm approaches,
    so it must be a valid rendered site frame but need not contain the target.
    This distinction is observed evidence: pi0.5 approached the washer from the
    exact same target-absent wrist frame on which GR00T later failed.

    The returned summary is deliberately small and serialisable; the complete
    radiance and semantic evidence remains transitively bound by the immutable
    construction-result digest.
    """

    requested_snapshot = str(snapshot_id or "").strip()
    roles = tuple(str(role or "").strip() for role in required_roles)
    semantic_roles = tuple(str(role or "").strip() for role in target_visible_roles)
    errors: list[str] = []
    if (
        not requested_snapshot
        or not roles
        or any(not role for role in roles)
        or any(not role for role in semantic_roles)
        or not set(semantic_roles).issubset(roles)
        or len(set(semantic_roles)) != len(semantic_roles)
    ):
        raise NativeTaskCameraObservabilityError([REFUSAL_POLICY_START_SNAPSHOTS_INVALID])
    if len(set(roles)) != len(roles):
        raise NativeTaskCameraObservabilityError([REFUSAL_POLICY_START_ROLE_DUPLICATE])

    raw_snapshots = construction_result.get("camera_snapshots")
    if not isinstance(raw_snapshots, list):
        raise NativeTaskCameraObservabilityError([REFUSAL_POLICY_START_SNAPSHOTS_INVALID])
    matches = [
        row
        for row in raw_snapshots
        if isinstance(row, Mapping) and str(row.get("snapshot_id") or "") == requested_snapshot
    ]
    if not matches:
        raise NativeTaskCameraObservabilityError(
            [f"{REFUSAL_POLICY_START_SNAPSHOT_MISSING}:{requested_snapshot}"]
        )
    if len(matches) != 1:
        raise NativeTaskCameraObservabilityError(
            [f"{REFUSAL_POLICY_START_SNAPSHOT_DUPLICATE}:{requested_snapshot}"]
        )

    cameras = matches[0].get("cameras")
    if not isinstance(cameras, list):
        raise NativeTaskCameraObservabilityError([REFUSAL_POLICY_START_CAMERAS_INVALID])

    summaries: list[dict[str, Any]] = []
    for role in roles:
        role_rows = [
            row
            for row in cameras
            if isinstance(row, Mapping) and str(row.get("role") or "") == role
        ]
        if not role_rows:
            errors.append(f"{REFUSAL_POLICY_START_ROLE_MISSING}:{role}")
            continue
        if len(role_rows) != 1:
            errors.append(f"{REFUSAL_POLICY_START_ROLE_DUPLICATE}:{role}")
            continue
        row = role_rows[0]
        observability = row.get("observability")
        thresholds = observability.get("thresholds") if isinstance(observability, Mapping) else None
        render = (
            observability.get("render_evidence") if isinstance(observability, Mapping) else None
        )
        try:
            pixel_count = int(observability["pixel_count"])
            pixel_fraction = float(observability["pixel_fraction"])
            minimum_pixels = int(thresholds["minimum_pixels"])
            minimum_fraction = float(thresholds["minimum_pixel_fraction"])
        except (KeyError, TypeError, ValueError):
            pixel_count = -1
            pixel_fraction = -1.0
            minimum_pixels = 0
            minimum_fraction = 0.0
        try:
            bbox_values = [int(value) for value in observability["bbox_xyxy"]]
        except (KeyError, TypeError, ValueError):
            bbox_values = []
        blockers = observability.get("blockers") if isinstance(observability, Mapping) else None
        bbox = observability.get("bbox_xyxy") if isinstance(observability, Mapping) else None
        rgb_png = row.get("rgb_png")
        rgb_png_sha256 = str(rgb_png.get("sha256") or "") if isinstance(rgb_png, Mapping) else ""
        rendered = (
            isinstance(observability, Mapping)
            and observability.get("schema_version") == SCHEMA_VERSION
            and observability.get("render_passed") is True
            and observability.get("site_appearance_claimed") is True
            and isinstance(render, Mapping)
            and render.get("passed") is True
            and render.get("frame_rendered") is True
            and render.get("site_rendered") is True
            and render.get("blockers") == []
            and str(row.get("scene_name") or "")
            and str(row.get("snapshot_id") or "") == requested_snapshot
            and rgb_png_sha256.startswith("sha256:")
            and len(rgb_png_sha256) == 71
            and all(character in "0123456789abcdef" for character in rgb_png_sha256[7:])
        )
        if not rendered:
            errors.append(f"{REFUSAL_POLICY_START_ROLE_NOT_RENDERED}:{role}")
            continue
        target_observable = (
            observability.get("passed") is True
            and observability.get("semantic_passed") is True
            and observability.get("centroid_within_margin") is True
            and observability.get("claim") == CLAIM_WITH_SITE
            and blockers == []
            and isinstance(bbox, list)
            and len(bbox) == 4
            and len(bbox_values) == 4
            and pixel_count >= max(1, minimum_pixels)
            and math.isfinite(pixel_fraction)
            and pixel_fraction >= max(0.0, minimum_fraction)
            and render.get("target_rendered") is True
        )
        if role in semantic_roles and not target_observable:
            errors.append(f"{REFUSAL_POLICY_START_ROLE_NOT_OBSERVABLE}:{role}")
            continue
        summaries.append(
            {
                "role": role,
                "scene_name": str(row.get("scene_name") or ""),
                "pixel_count": pixel_count,
                "pixel_fraction": pixel_fraction,
                "bbox_xyxy": bbox_values,
                "rgb_png_sha256": rgb_png_sha256,
                "target_visibility_required": role in semantic_roles,
                "target_visible": target_observable,
            }
        )
    if errors:
        raise NativeTaskCameraObservabilityError(errors)
    return {
        "schema_version": POLICY_START_OBSERVABILITY_SCHEMA_VERSION,
        "snapshot_id": requested_snapshot,
        "required_policy_input_roles": list(roles),
        "target_visible_roles": list(semantic_roles),
        "cameras": summaries,
        "passed": True,
        "blockers": [],
        "authority": "construction_result_exact_policy_initial_state_snapshot",
    }


def _semantic_identifier_candidates(identifier: Any) -> list[int]:
    """Decode Replicator numeric IDs or its RGBA tuple-key representation."""

    try:
        return [int(identifier)]
    except (TypeError, ValueError):
        pass
    try:
        rgba = ast.literal_eval(str(identifier))
    except (SyntaxError, ValueError):
        return []
    if (
        not isinstance(rgba, tuple)
        or len(rgba) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in rgba)
        or any(value < 0 or value > 255 for value in rgba)
    ):
        return []
    packed = sum(int(value) << (8 * index) for index, value in enumerate(rgba))
    signed = packed - 2**32 if packed >= 2**31 else packed
    return sorted({packed, signed})


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


def _region_statistics(luminance: Any, void: Any, selector: Any, *, rgb: Any) -> dict[str, Any]:
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
            "dominant_rgb_pixel_fraction": None,
        }
    colors = rgb[selector].reshape(-1, 3)
    _, color_counts = np.unique(colors, axis=0, return_counts=True)
    return {
        "pixel_count": int(values.size),
        "void_pixel_fraction": float(void[selector].mean()),
        "distinct_luminance_levels": int(np.unique(np.rint(values).astype(np.uint8)).size),
        "luminance_mean": float(values.mean()),
        "luminance_std": float(values.std()),
        "dominant_rgb_pixel_fraction": float(color_counts.max() / values.size),
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
        raise NativeTaskCameraObservabilityError(["native_task_camera_site_expectation_invalid"])
    frame = _as_uint8_rgb(rgb)
    height, width = (int(value) for value in frame.shape[:2])
    if expected_resolution_hw is not None:
        expected = [int(value) for value in expected_resolution_hw]
        if expected != [height, width]:
            raise NativeTaskCameraObservabilityError([REFUSAL_RGB_SEMANTIC_MISMATCH])

    void = frame.max(axis=-1) == 0
    luminance = frame.astype(np.float64).mean(axis=-1)
    everywhere = np.ones_like(void, dtype=bool)
    frame_statistics = _region_statistics(luminance, void, everywhere, rgb=frame)
    frame_statistics["near_black_pixel_fraction"] = float(
        (luminance <= NEAR_BLACK_LUMINANCE_MAX).mean()
    )
    frame_statistics["luminance_min"] = float(luminance.min())
    frame_statistics["luminance_max"] = float(luminance.max())
    # Reported here, gated on the exact policy-input frames: the fraction of
    # pixels the LDR encode clipped in at least one channel.
    frame_statistics["saturated_channel_pixel_fraction"] = float(
        (frame >= SATURATED_CHANNEL_LEVEL).any(axis=-1).mean()
    )

    mask = None
    if target_mask is not None:
        candidate = np.asarray(target_mask).astype(bool)
        if candidate.shape != void.shape:
            raise NativeTaskCameraObservabilityError([REFUSAL_RGB_SEMANTIC_MISMATCH])
        mask = candidate
    target_statistics = (
        _region_statistics(luminance, void, mask, rgb=frame)
        if mask is not None and bool(mask.any())
        else None
    )
    site_statistics = (
        _region_statistics(luminance, void, ~mask, rgb=frame)
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
        site_blockers: list[str] = []
        if float(site_statistics["void_pixel_fraction"]) > MAXIMUM_SITE_VOID_PIXEL_FRACTION:
            site_blockers.append(BLOCKER_SITE_VOID)
        if (
            float(site_statistics["dominant_rgb_pixel_fraction"])
            > MAXIMUM_SITE_DOMINANT_RGB_PIXEL_FRACTION
        ):
            site_blockers.append(BLOCKER_SITE_DOMINANT_COLOR)
        site_rendered = not site_blockers
        if site_appearance_render_expected:
            blockers.extend(site_blockers)
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
        "site_appearance_claimed": bool(site_appearance_render_expected and site_rendered),
        "site_appearance_presence_claimed": bool(site_appearance_render_expected and site_rendered),
        # RGB statistics establish that non-void site radiance is present. They
        # cannot establish reconstruction fidelity, Gaussian geometry quality,
        # or sharpness without an independently bound reference. Those gates
        # live at producer, representation, and packet boundaries.
        "appearance_quality_claimed": False,
        "appearance_fidelity_qualified": False,
        "quality_boundary": "render_presence_only_not_appearance_quality",
        "thresholds": {
            "maximum_site_void_pixel_fraction": MAXIMUM_SITE_VOID_PIXEL_FRACTION,
            "maximum_site_dominant_rgb_pixel_fraction": (MAXIMUM_SITE_DOMINANT_RGB_PIXEL_FRACTION),
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
    framing_expectation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate exact target-class pixels AND the radiance of the pixels they name.

    ``rgb`` has no default on purpose.  A caller that cannot supply the frame
    must fail at the call rather than receive a verdict drawn from the mask.

    ``framing_expectation`` optionally carries the sealed geometric projection
    of the task object through this camera
    (:mod:`blueprint_pipeline.native_task_camera_framing_expectation`).  When
    present, the configured pixel minimums are scaled down -- never up -- to a
    fraction of what the geometry can produce, so a small object at distance
    is gated against its own physics instead of a constant calibrated on a
    larger scene.  When absent, the configured constants apply unchanged.
    """

    import numpy as np

    semantic = np.asarray(semantic_ids)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if semantic.ndim != 2 or not semantic.size:
        raise NativeTaskCameraObservabilityError(["native_task_camera_semantic_shape_invalid"])
    if (
        isinstance(minimum_pixels, bool)
        or int(minimum_pixels) < 1
        or not math.isfinite(float(minimum_pixel_fraction))
        or float(minimum_pixel_fraction) <= 0.0
        or not math.isfinite(float(centroid_margin_fraction))
        or float(centroid_margin_fraction) < 0.0
        or float(centroid_margin_fraction) >= 0.5
    ):
        raise NativeTaskCameraObservabilityError(["native_task_camera_threshold_invalid"])
    target_ids: list[int] = []
    for identifier, entry in id_to_labels.items():
        label = entry.get("class") if isinstance(entry, Mapping) else entry
        if label != target_label:
            continue
        candidates = _semantic_identifier_candidates(identifier)
        if not candidates:
            raise NativeTaskCameraObservabilityError(
                ["native_task_camera_semantic_identifier_invalid"]
            )
        target_ids.extend(candidates)
    target_ids = sorted(set(target_ids))
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
    framing_thresholds: dict[str, Any] | None = None
    effective_minimum_pixels = int(minimum_pixels)
    effective_minimum_fraction = float(minimum_pixel_fraction)
    if framing_expectation is not None:
        from blueprint_pipeline.native_task_camera_framing_expectation import (
            effective_framing_minimums,
        )

        framing_thresholds = effective_framing_minimums(
            minimum_pixels=int(minimum_pixels),
            minimum_pixel_fraction=float(minimum_pixel_fraction),
            frame_width=width,
            frame_height=height,
            expected_bbox_area_px=framing_expectation["expected_bbox_area_px"],
        )
        effective_minimum_pixels = framing_thresholds["effective_minimum_pixels"]
        effective_minimum_fraction = framing_thresholds["effective_minimum_pixel_fraction"]
    semantic_passed = (
        count >= effective_minimum_pixels
        and fraction >= effective_minimum_fraction
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
            "effective_minimum_pixels": effective_minimum_pixels,
            "effective_minimum_pixel_fraction": effective_minimum_fraction,
        },
        "framing_expectation": (
            dict(framing_expectation) if framing_expectation is not None else None
        ),
        "framing_thresholds": framing_thresholds,
        "semantic_passed": semantic_passed,
        "render_passed": bool(render["passed"]),
        "render_evidence": render,
        "site_appearance_claimed": bool(render["site_appearance_claimed"]),
        "claim": (
            CLAIM_WITH_SITE if passed and render["site_appearance_claimed"] else CLAIM_WITHOUT_SITE
        ),
        "blockers": sorted(set(blockers)),
        "notices": list(render["notices"]),
        "passed": passed,
        "measurement_authority": "native_semantic_segmentation_aov+native_rgb_frame",
        "rgb_or_model_label_used": True,
        "model_label_used": False,
    }


def measure_native_task_frame_saturation(*, rgb: Any) -> dict[str, Any]:
    """Fraction of pixels the LDR encode clipped, in any and in every channel."""

    import numpy as np

    frame = _as_uint8_rgb(rgb)
    saturated = frame >= SATURATED_CHANNEL_LEVEL
    any_channel = saturated.any(axis=-1)
    all_channels = saturated.all(axis=-1)
    fraction = float(any_channel.mean())
    return {
        "schema_version": POLICY_INPUT_SATURATION_SCHEMA_VERSION,
        "pixel_count": int(any_channel.size),
        "saturated_channel_pixel_fraction": fraction,
        "saturated_white_pixel_fraction": float(all_channels.mean()),
        "chromatic_clip_pixel_fraction": float(
            np.logical_and(any_channel, np.logical_not(all_channels)).mean()
        ),
        "maximum_saturated_channel_pixel_fraction": (
            MAXIMUM_POLICY_INPUT_SATURATED_PIXEL_FRACTION
        ),
        "passed": fraction <= MAXIMUM_POLICY_INPUT_SATURATED_PIXEL_FRACTION,
    }


def validate_native_task_policy_input_frames(
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse policy-input frames the LDR encode has clipped into saturation.

    ``frames`` maps a policy view name to the exact array the observation is
    built from.  An unreadable frame is a refusal, not a pass.
    """

    if not isinstance(frames, Mapping) or not frames:
        raise NativeTaskCameraObservabilityError([REFUSAL_POLICY_INPUT_FRAMES_INVALID])
    errors: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    for view, frame in frames.items():
        name = str(view or "").strip()
        if not name or name in views:
            raise NativeTaskCameraObservabilityError([REFUSAL_POLICY_INPUT_FRAMES_INVALID])
        evidence = measure_native_task_frame_saturation(rgb=frame)
        views[name] = evidence
        if not evidence["passed"]:
            errors.append(f"{REFUSAL_POLICY_INPUT_FRAME_SATURATED}:{name}")
    if errors:
        raise NativeTaskCameraObservabilityError(errors)
    return {
        "schema_version": POLICY_INPUT_SATURATION_SCHEMA_VERSION,
        "maximum_saturated_channel_pixel_fraction": (
            MAXIMUM_POLICY_INPUT_SATURATED_PIXEL_FRACTION
        ),
        "views": views,
        "passed": True,
    }


def measure_native_task_prepolicy_visual_frames(
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate all reset cameras before either learned-policy client is loaded.

    This is deliberately stricter than the historical saturation-only check.
    Scene 839873 produced frames with zero clipped pixels that were nevertheless
    58-75% near black and visibly contained only scattered chromatic splats.
    A policy must not be queried against that observation domain.
    """

    import numpy as np

    if not isinstance(frames, Mapping) or set(frames) != PREPOLICY_VISUAL_REQUIRED_VIEWS:
        raise NativeTaskCameraObservabilityError([REFUSAL_POLICY_INPUT_FRAMES_INVALID])
    blockers: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for view in sorted(PREPOLICY_VISUAL_REQUIRED_VIEWS):
        frame = _as_uint8_rgb(frames[view])
        saturation = measure_native_task_frame_saturation(rgb=frame)
        render = measure_native_task_frame_render_evidence(
            rgb=frame,
            site_appearance_render_expected=True,
        )
        near_black_fraction = float(
            (frame.astype(np.float64).mean(axis=-1) <= NEAR_BLACK_LUMINANCE_MAX).mean()
        )
        digest = "sha256:" + hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
        digests[view] = digest
        view_blockers: list[str] = []
        if not saturation["passed"]:
            view_blockers.append(REFUSAL_POLICY_INPUT_FRAME_SATURATED)
        view_blockers.extend(
            f"{REFUSAL_PREPOLICY_VISUAL_FRAME_INVALID}:{blocker}"
            for blocker in render["blockers"]
        )
        if near_black_fraction > MAXIMUM_PREPOLICY_NEAR_BLACK_PIXEL_FRACTION:
            view_blockers.append(REFUSAL_PREPOLICY_VISUAL_FRAME_NEAR_BLACK)
        blockers.extend(f"{blocker}:{view}" for blocker in view_blockers)
        views[view] = {
            "frame_digest": digest,
            "saturation": saturation,
            "render_presence": render,
            "near_black_pixel_fraction": near_black_fraction,
            "maximum_near_black_pixel_fraction": (
                MAXIMUM_PREPOLICY_NEAR_BLACK_PIXEL_FRACTION
            ),
            "blockers": view_blockers,
            "passed": not view_blockers,
        }
    by_digest: dict[str, list[str]] = {}
    for view, digest in digests.items():
        by_digest.setdefault(digest, []).append(view)
    for duplicate_views in by_digest.values():
        if len(duplicate_views) > 1:
            blockers.append(
                f"{REFUSAL_PREPOLICY_VISUAL_FRAME_DUPLICATE}:"
                + ",".join(sorted(duplicate_views))
            )
    return {
        "schema_version": "native_task_prepolicy_visual_gate.v1",
        "required_views": sorted(PREPOLICY_VISUAL_REQUIRED_VIEWS),
        "views": views,
        "candidate_policy_loaded": False,
        "candidate_policy_queried": False,
        "blockers": sorted(set(blockers)),
        "passed": not blockers,
        "measurement_authority": "exact_reset_policy_and_review_rgb_frames",
        "quality_boundary": (
            "structural_and_exposure_gate_only;official_same_pose_nre_parity_pending"
        ),
    }


__all__ = [
    "CLAIM_WITHOUT_SITE",
    "CLAIM_WITH_SITE",
    "MAXIMUM_SITE_DOMINANT_RGB_PIXEL_FRACTION",
    "MAXIMUM_SITE_VOID_PIXEL_FRACTION",
    "MINIMUM_DISTINCT_LUMINANCE_LEVELS",
    "MAXIMUM_POLICY_INPUT_SATURATED_PIXEL_FRACTION",
    "MINIMUM_LUMINANCE_STD",
    "POLICY_INPUT_SATURATION_SCHEMA_VERSION",
    "REFUSAL_POLICY_INPUT_FRAMES_INVALID",
    "REFUSAL_POLICY_INPUT_FRAME_SATURATED",
    "SATURATED_CHANNEL_LEVEL",
    "measure_native_task_frame_saturation",
    "measure_native_task_prepolicy_visual_frames",
    "validate_native_task_policy_input_frames",
    "NativeTaskCameraObservabilityError",
    "POLICY_INPUT_CAMERA_ROLES",
    "POLICY_START_OBSERVABILITY_SCHEMA_VERSION",
    "POLICY_START_SNAPSHOT_ID",
    "POLICY_START_TARGET_VISIBLE_ROLES",
    "RENDER_EVIDENCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "measure_native_task_camera_observability",
    "measure_native_task_frame_render_evidence",
    "validate_native_task_policy_start_camera_observability",
]
