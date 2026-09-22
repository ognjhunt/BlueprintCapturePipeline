"""Required task/evidence fields in Gemini's structured video response."""

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
    "spatial_evidence": {"type": "array", "items": {
        "type": "object", "properties": {
            "timestamp_seconds": {"type": "number", "minimum": 0},
            "box_xywh_normalized": {"type": "array", "minItems": 4, "maxItems": 4,
                                    "items": {"type": "number", "minimum": 0, "maximum": 1}},
        }, "required": ["timestamp_seconds", "box_xywh_normalized"],
    }},
}
RESPONSE_SCHEMA = {
    "type": "object", "required": ["targets"], "properties": {
        "targets": {"type": "array", "items": {
            "type": "object", "properties": _properties, "required": list(_properties),
        }},
    },
}
