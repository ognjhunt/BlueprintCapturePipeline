"""Required task/evidence fields in Gemini's structured video response."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

_evidence = {"type": "array", "items": {
    "type": "object", "properties": {
        "timestamp_seconds": {"type": "number", "minimum": 0},
        "box_xywh_normalized": {"type": "array", "minItems": 4, "maxItems": 4,
                                "items": {"type": "number", "minimum": 0, "maximum": 1}},
    }, "required": ["timestamp_seconds", "box_xywh_normalized"],
}}
_properties = {
    **{name: {"type": "string"} for name in (
        "target_id", "semantic_label", "segmentation_prompt", "decision_reason",
        "task_basis_quote", "clarification_question", "articulated_part")},
    # An articulated assembly (cabinet with a drawer, refrigerator with a door)
    # is one manipulated target; these name the moving part and its mechanism.
    "articulation_kind": {"type": "string", "enum": ["", "prismatic", "revolute"]},
    "target_class": {"type": "string", "enum": ["person", "movable_object", "fixed_clutter"]},
    "target_role": {"type": "string", "enum": ["task_object", "support", "destination", "obstacle", "background", "person"]},
    "task_effect": {"type": "string", "enum": ["manipulated", "static_contact", "static_obstacle", "unrelated", "uncertain", "privacy"]},
    "placement_relation": {"type": "string", "enum": ["on", "inside", ""]},
    "disposition": {"type": "string", "enum": ["remove", "keep"]},
    "rebuild_intent": {"type": "string", "enum": ["rebuild_and_compose", "none"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "spatial_evidence": _evidence,
}
#: A task that adds a new object absent from the video (a synthetic ball) names
#: its design and the observed support it starts on; nothing visible stands in.
SYNTHETIC_SHAPES = ("sphere", "box", "cylinder")
_synthetic = {
    **{name: {"type": "string"} for name in (
        "semantic_label", "task_basis_quote", "start_support_target_id", "decision_reason")},
    "shape": {"type": "string", "enum": list(SYNTHETIC_SHAPES)},
    "dimensions_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number", "minimum": 0}},
    "mass_kg": {"type": "number", "minimum": 0},
}
#: Optional: the place on the destination surface the task names ("the other end",
#: "beside the start"), boxed in frames where the destination itself is boxed.
_region = {
    **{name: {"type": "string"} for name in ("destination_target_id", "region_basis_quote")},
    "spatial_evidence": {**_evidence, "minItems": 1},
}
RESPONSE_SCHEMA = {
    "type": "object", "required": ["targets"], "properties": {
        "targets": {"type": "array", "items": {
            "type": "object", "properties": _properties, "required": list(_properties),
        }},
        "synthetic_subject": {"type": "object", "properties": _synthetic, "required": list(_synthetic)},
        "destination_region": {"type": "object", "properties": _region, "required": list(_region)},
    },
}
#: The response schema before ``destination_region`` existed. A completed analysis
#: bought under it is still the capture's answer (``analyze_removal_targets``).
PRIOR_RESPONSE_SCHEMA = {**RESPONSE_SCHEMA, "properties": {
    key: value for key, value in RESPONSE_SCHEMA["properties"].items() if key != "destination_region"}}


def _box(value: Any, *, inside_frame: bool = True) -> list[float] | None:
    """A normalized ``[x, y, w, h]`` box, or None; never clamped.

    ``inside_frame`` also requires it to lie in the frame, within the same
    rounding allowance the task-mask anchors use.
    """
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        return None
    if inside_frame and (min(value) < 0 or min(value[2:]) <= 0 or max(value[0] + value[2], value[1] + value[3]) > 1.00001):
        return None
    return [float(v) for v in value]


def region_evidence(value: Any) -> list[dict[str, Any]] | None:
    """A region's timestamped boxes as numbers, or None when any row is malformed."""
    if not isinstance(value, list) or not value:
        return None
    rows = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"timestamp_seconds", "box_xywh_normalized"}:
            return None
        timestamp, box = row["timestamp_seconds"], _box(row["box_xywh_normalized"])
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0 or box is None):
            return None
        rows.append({"timestamp_seconds": float(timestamp), "box_xywh_normalized": box})
    return rows


def validate_destination_region(value: Any, *, targets: Sequence[Mapping[str, Any]],
                                task_description: str | None) -> list[str]:
    """Form and binding errors of a destination region; empty means well-formed.

    The region refines one reported ``on`` destination: its quote is the task's
    own words for the place, and every box is drawn at a timestamp where that
    destination is boxed, with its centre inside the destination's box there.
    Nothing is clamped or repaired. ``task_description`` None skips only the
    verbatim check (a retained plan does not carry the task text).
    """
    if not isinstance(value, Mapping) or set(value) != set(_region):
        return ["destination_region_invalid"]
    quote = value["region_basis_quote"]
    if not isinstance(quote, str) or not quote.strip() or (
            task_description is not None and quote.strip() not in task_description):
        return ["destination_region_basis_missing"]
    destination = [row for row in targets if row.get("target_id") == value["destination_target_id"]]
    if (len(destination) != 1 or destination[0].get("target_role") != "destination"
            or destination[0].get("task_effect") != "static_contact"
            or destination[0].get("placement_relation") != "on"):
        return ["destination_region_target_invalid"]
    evidence = region_evidence(value["spatial_evidence"])
    if evidence is None:
        return ["destination_region_evidence_invalid"]
    anchors = [(row.get("timestamp_seconds"), _box(row.get("box_xywh_normalized"), inside_frame=False))
               for row in destination[0].get("spatial_evidence") or [] if isinstance(row, Mapping)]
    for row in evidence:
        (x, y, width, height), timestamp = row["box_xywh_normalized"], row["timestamp_seconds"]
        x, y = x + width / 2, y + height / 2
        if not any(isinstance(when, (int, float)) and abs(when - timestamp) <= 1e-6 and anchor is not None
                   and anchor[0] <= x <= anchor[0] + anchor[2] and anchor[1] <= y <= anchor[1] + anchor[3]
                   for when, anchor in anchors):
            return ["destination_region_outside_destination_box"]
    return []
