"""The structured Gemini response names an articulated assembly's moving part."""

from blueprint_pipeline.clean_plate_removal_response_schema import RESPONSE_SCHEMA


def test_every_target_declares_its_articulation_hint_fields():
    target = RESPONSE_SCHEMA["properties"]["targets"]["items"]
    assert set(target["required"]) == set(target["properties"])
    assert target["properties"]["articulated_part"] == {"type": "string"}
    assert target["properties"]["articulation_kind"]["enum"] == ["", "prismatic", "revolute"]
    # The assembly itself stays a movable object; the hint never adds a class.
    assert target["properties"]["target_class"]["enum"] == ["person", "movable_object", "fixed_clutter"]
