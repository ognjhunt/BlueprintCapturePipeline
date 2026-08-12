"""Derive the unique non-overlapped target collider and its horizontal support.

Step 2 of the shortlist screening sequence frozen in
`adp009a_scene_shortlist_extension_preregistration.v1.json`. Steps 1 and 3 were
implemented; this one was still performed by hand, which is how a scene that
passes on retained bytes but has no workable target gets mistaken for the next
scene to run.

Two details are easy to get wrong and both invert the result:

* An object resting on a support must not count as colliding with that support.
  Inflating a candidate by the contact envelope in every direction makes every
  supported object overlap the thing holding it up, which rejects exactly the
  objects a pick task needs and leaves only wall-mounted ones.
* A support has to be rigid. A candidate resting on a quilt or a pillow passes
  the geometric test and is still not a pick-and-place target.

Reads retained labels only; performs no provider mutation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_contact_envelope import apply_contact_envelope_to_clearance

SCHEMA_VERSION = "adp009a_scene_target_derivation.v1"
_DIGEST_PREFIX = "sha256:"

# Robotiq 2F-85 maximum aperture. The approved contact envelope is subtracted
# from it so the planner works against the clearance that actually remains.
ROBOTIQ_2F85_OPEN_JAW_M = 0.085

# A target must be liftable and reachable, not a wall fixture or a floor item.
MIN_TARGET_WIDTH_M = 0.02
MIN_TARGET_HEIGHT_M = 0.02
MAX_TARGET_HEIGHT_M = 0.40
MIN_TARGET_ELEVATION_M = 0.30

# Vertical tolerance for "resting on", and how far a support may extend beyond
# the target footprint before it stops being the thing holding it up.
SUPPORT_CONTACT_TOLERANCE_M = 0.06
SUPPORT_FOOTPRINT_MARGIN_M = 0.15

# Deformable surfaces. A candidate resting on one of these is excluded: the
# task is a rigid-object pick, and a soft support has no stable pose.
SOFT_SUPPORT_LABELS = frozenset({
    "quilt", "pillow", "body pillow", "blanket", "bed", "duvet", "cushion",
    "mattress", "towel", "carpet", "rug", "curtain", "sofa", "Multi person sofa",
})


class SceneTargetDerivationError(ValueError):
    """A label set failed a fail-closed contract."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return _DIGEST_PREFIX + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _aabb(entry: Mapping[str, Any]) -> tuple[float, ...] | None:
    box = entry.get("bounding_box")
    if not isinstance(box, Sequence) or not box:
        return None
    try:
        xs = [float(p["x"]) for p in box]
        ys = [float(p["y"]) for p in box]
        zs = [float(p["z"]) for p in box]
    except (KeyError, TypeError, ValueError):
        return None
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return not (
        a[1] < b[0] or b[1] < a[0]
        or a[3] < b[2] or b[3] < a[2]
        or a[5] < b[4] or b[5] < a[4]
    )


def derive_scene_targets(labels: Any) -> dict[str, Any]:
    """Return every graspable target resting free on a rigid horizontal support."""
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise SceneTargetDerivationError("scene_target_derivation_labels_invalid")

    entries: list[tuple[Mapping[str, Any], tuple[float, ...]]] = []
    for entry in labels:
        if not isinstance(entry, Mapping):
            raise SceneTargetDerivationError("scene_target_derivation_labels_invalid")
        box = _aabb(entry)
        if box is not None:
            entries.append((entry, box))

    clearance = apply_contact_envelope_to_clearance(ROBOTIQ_2F85_OPEN_JAW_M)
    usable = clearance["resolved_clearance_m"]
    envelope = clearance["effective_contact_envelope_m"]

    targets: list[dict[str, Any]] = []
    for entry, box in entries:
        x0, x1, y0, y1, z0, z1 = box
        width = min(x1 - x0, y1 - y0)
        height = z1 - z0
        if not (MIN_TARGET_WIDTH_M < width < usable):
            continue
        if not (MIN_TARGET_HEIGHT_M < height < MAX_TARGET_HEIGHT_M):
            continue
        if z0 <= MIN_TARGET_ELEVATION_M:
            continue

        # Inflate sideways and upward only. The support is underneath, and it
        # is contact rather than a collision the jaws must clear.
        inflated = (
            x0 - envelope, x1 + envelope,
            y0 - envelope, y1 + envelope,
            z0 + SUPPORT_CONTACT_TOLERANCE_M / 30.0, z1 + envelope,
        )
        if any(
            _overlaps(inflated, other_box)
            for other, other_box in entries
            if other is not entry
        ):
            continue

        supports = [
            other
            for other, other_box in entries
            if other is not entry
            and abs(other_box[5] - z0) < SUPPORT_CONTACT_TOLERANCE_M
            and other_box[0] - SUPPORT_FOOTPRINT_MARGIN_M <= x0
            and other_box[1] + SUPPORT_FOOTPRINT_MARGIN_M >= x1
        ]
        rigid = [s for s in supports if str(s.get("label") or "") not in SOFT_SUPPORT_LABELS]
        if not rigid:
            continue

        targets.append({
            "ins_id": str(entry.get("ins_id") or ""),
            "semantic_label": str(entry.get("label") or ""),
            "width_m": width,
            "height_m": height,
            "elevation_m": z0,
            # Where the object is, not only that one exists. A pick task needs
            # the pose, and recording it also binds the derivation digest to the
            # geometry: without it, moving the target leaves the digest
            # unchanged and two different scenes can agree.
            "center_xy_m": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
            "aabb_m": [x0, x1, y0, y1, z0, z1],
            "support_ins_id": str(rigid[0].get("ins_id") or ""),
            "support_label": str(rigid[0].get("label") or ""),
            "support_top_m": next(
                other_box[5]
                for other, other_box in entries
                if other is rigid[0]
            ),
        })

    targets.sort(key=lambda t: (t["width_m"], t["ins_id"]))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "open_jaw_clearance_m": clearance["open_jaw_clearance_m"],
        "effective_contact_envelope_m": envelope,
        "effective_contact_envelope_calculation": clearance[
            "effective_contact_envelope_calculation"
        ],
        "usable_grasp_width_m": usable,
        "targets": targets,
        "provider_mutation_performed": False,
        "derivation_digest": "",
    }
    result["derivation_digest"] = _digest(result, digest_field="derivation_digest")
    return result


__all__ = [
    "ROBOTIQ_2F85_OPEN_JAW_M",
    "SCHEMA_VERSION",
    "SOFT_SUPPORT_LABELS",
    "SceneTargetDerivationError",
    "derive_scene_targets",
]
