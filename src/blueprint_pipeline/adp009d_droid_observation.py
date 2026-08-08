"""Build DROID policy observations from ADP-009D Isaac camera frames.

Isaac renders the sealed scene at 1280x720; no DROID policy accepts that.  The
frozen candidates do not even agree with each other -- ``pi05_droid`` wants two
224x224 views, ``groot_n17_droid`` wants 180x320, and ``cosmos3_edge_policy_droid``
wants three views -- so the observation shape is a per-candidate fact, not a
harness constant.  This module owns the conversion and refuses to invent one.

Aspect ratio is the substantive decision.  1280x720 is 16:9 and every target is
squarer, so something has to give.  Padding preserves the geometry the policy
was trained to see and leaves the unmapped border explicitly black; a stretch
would silently distort every object in frame, and a centre crop would discard
the scene edges where the approved can sits relative to the arm.  Padding is
also what the vendor runtimes already do, so this matches training-time
preprocessing rather than inventing a third convention.

Nothing here queries a policy or admits a candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DROID_OBSERVATION_SCHEMA_VERSION = "adp009d_droid_observation.v1"

# Isaac renders the ADP-009D policy cameras at this exact size.
ISAAC_RENDER_HEIGHT = 720
ISAAC_RENDER_WIDTH = 1280

# Per-candidate view geometry, taken from each vendor runtime rather than
# assumed to be shared.  Adding a candidate means adding its measured shape,
# never reusing another candidate's.
CANDIDATE_VIEW_SHAPES: dict[str, tuple[int, int]] = {
    "pi05_droid": (224, 224),
    "groot_n17_droid": (180, 320),
    "cosmos3_edge_policy_droid": (224, 224),
}

DROID_WRIST_VIEW = "observation/wrist_image_left"
DROID_EXTERIOR_VIEW_1 = "observation/exterior_image_1_left"
DROID_EXTERIOR_VIEW_2 = "observation/exterior_image_2_left"

# The ADP-009D scene deliberately drops the second exterior camera: the runtime
# sets ``embodiment.camera_config.external_camera_2 = None`` because it is
# outside the frozen two-camera policy contract.  A three-view candidate
# therefore cannot be served without changing that contract, which is a
# programme decision rather than something this module may quietly work around.
CANDIDATE_REQUIRED_VIEWS: dict[str, tuple[str, ...]] = {
    "pi05_droid": (DROID_EXTERIOR_VIEW_1, DROID_WRIST_VIEW),
    "groot_n17_droid": (DROID_EXTERIOR_VIEW_1, DROID_WRIST_VIEW),
    "cosmos3_edge_policy_droid": (
        DROID_WRIST_VIEW,
        DROID_EXTERIOR_VIEW_1,
        DROID_EXTERIOR_VIEW_2,
    ),
}

BLOCKER_UNKNOWN_CANDIDATE = "droid_observation_unknown_candidate"
BLOCKER_VIEW_UNAVAILABLE = "droid_observation_required_view_unavailable"
BLOCKER_THIRD_VIEW_OUTSIDE_CONTRACT = (
    "droid_observation_third_view_outside_frozen_two_camera_contract"
)


class DroidObservationError(ValueError):
    """Fail-closed DROID observation contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def resize_with_pad(image: Any, *, height: int, width: int) -> Any:
    """Fit an image into ``height x width`` preserving aspect, padding the rest.

    Matches the vendor runtimes' preprocessing so the policy sees the geometry
    it was trained on.  The padded border is black and deterministic.
    """

    import numpy as np
    from PIL import Image

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise DroidObservationError(["droid_observation_image_must_be_uint8_rgb"])
    if height < 1 or width < 1:
        raise DroidObservationError(["droid_observation_target_shape_invalid"])
    scale = min(width / source.shape[1], height / source.shape[0])
    resized_width = max(1, round(source.shape[1] * scale))
    resized_height = max(1, round(source.shape[0] * scale))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = np.asarray(
        Image.fromarray(source).resize((resized_width, resized_height), resampling),
        dtype=np.uint8,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def build_droid_observation(
    *,
    candidate_id: str,
    camera_rgb: Mapping[str, Any],
    joint_position: Sequence[float],
    gripper_position: float,
    prompt: str,
) -> dict[str, Any]:
    """Assemble one candidate's DROID observation from Isaac camera frames.

    ``camera_rgb`` maps DROID view names to full-resolution Isaac RGB arrays.
    Every view the candidate requires must be present: a missing view fails
    closed rather than being substituted, padded, or duplicated from another
    camera, because a policy fed a duplicated view would be silently
    misinformed about the scene.
    """

    import numpy as np

    errors: list[str] = []
    if candidate_id not in CANDIDATE_VIEW_SHAPES:
        raise DroidObservationError([f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"])

    required = CANDIDATE_REQUIRED_VIEWS[candidate_id]
    height, width = CANDIDATE_VIEW_SHAPES[candidate_id]

    if DROID_EXTERIOR_VIEW_2 in required and DROID_EXTERIOR_VIEW_2 not in camera_rgb:
        # Name the contract explicitly rather than reporting a generic absence.
        raise DroidObservationError([BLOCKER_THIRD_VIEW_OUTSIDE_CONTRACT])

    views: dict[str, Any] = {}
    for name in required:
        if name not in camera_rgb:
            errors.append(f"{BLOCKER_VIEW_UNAVAILABLE}:{name}")
            continue
        views[name] = resize_with_pad(camera_rgb[name], height=height, width=width)

    joints = np.asarray(joint_position, dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        errors.append("droid_observation_joint_position_invalid")
    gripper = np.asarray([float(gripper_position)], dtype=float)
    if not np.isfinite(gripper).all():
        errors.append("droid_observation_gripper_position_invalid")
    if not str(prompt or "").strip():
        errors.append("droid_observation_prompt_missing")
    if errors:
        raise DroidObservationError(errors)

    observation: dict[str, Any] = dict(views)
    observation["observation/joint_position"] = joints
    observation["observation/gripper_position"] = gripper
    observation["prompt"] = str(prompt)
    return observation


def describe_observation_conversion(
    candidate_id: str,
    *,
    source_hw: tuple[int, int] = (ISAAC_RENDER_HEIGHT, ISAAC_RENDER_WIDTH),
) -> dict[str, Any]:
    """Report the exact conversion applied, for the run receipt.

    ``source_hw`` is the size the cameras actually rendered, which is a run
    decision rather than a constant: rendering at the size the candidates
    consume draws a sixteenth of the pixels for byte-identical content, and a
    receipt that reported a fixed 1280x720 would be describing a conversion
    that did not happen.
    """

    if candidate_id not in CANDIDATE_VIEW_SHAPES:
        raise DroidObservationError([f"{BLOCKER_UNKNOWN_CANDIDATE}:{candidate_id}"])
    source_height, source_width = (int(v) for v in source_hw)
    height, width = CANDIDATE_VIEW_SHAPES[candidate_id]
    scale = min(width / source_width, height / source_height)
    content_width = max(1, round(source_width * scale))
    content_height = max(1, round(source_height * scale))
    return {
        "schema_version": DROID_OBSERVATION_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "source_resolution_hw": [source_height, source_width],
        "target_resolution_hw": [height, width],
        "required_views": list(CANDIDATE_REQUIRED_VIEWS[candidate_id]),
        "method": "aspect_preserving_resize_with_centred_black_pad",
        "content_resolution_hw": [content_height, content_width],
        "padded_rows": height - content_height,
        "padded_columns": width - content_width,
        "scene_content_cropped": False,
    }


__all__ = [
    "CANDIDATE_REQUIRED_VIEWS",
    "CANDIDATE_VIEW_SHAPES",
    "DROID_EXTERIOR_VIEW_1",
    "DROID_EXTERIOR_VIEW_2",
    "DROID_OBSERVATION_SCHEMA_VERSION",
    "DROID_WRIST_VIEW",
    "DroidObservationError",
    "build_droid_observation",
    "describe_observation_conversion",
    "resize_with_pad",
]
