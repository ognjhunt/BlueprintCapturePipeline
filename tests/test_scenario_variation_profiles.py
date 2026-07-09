"""Tests for location-type-scoped scenario variation profiles (audit R013)."""

from __future__ import annotations

from blueprint_pipeline import robot_eval_dataset as red

_FACTORY = {
    "conveyor_motion",
    "machine_guarding_state",
    "agv_cross_traffic",
    "thermal_surface",
    "moving_part_on_line",
}


def test_warehouse_and_default_preserve_base_set() -> None:
    base = {d["variation_id"] for d in red.SCENARIO_VARIATION_DEFINITIONS}
    assert set(red.required_variation_names_for_site_category("warehouse")) == base
    assert set(red.required_variation_names_for_site_category("default")) == base
    assert set(red.required_variation_names_for_site_category(None)) == base
    assert set(red.required_variation_names_for_site_category("totally unknown")) == base


def test_manufacturing_adds_factory_hazards_and_drops_forklift() -> None:
    names = set(red.required_variation_names_for_site_category("manufacturing"))
    assert _FACTORY <= names  # all factory axes present
    # Warehouse-only logistics axes are not forced onto a factory site.
    assert "forklift_nearby" not in names
    assert "cart_shifted" not in names


def test_home_profiles_exclude_industrial_hazards() -> None:
    for category in ("kitchen", "residential"):
        names = set(red.required_variation_names_for_site_category(category))
        assert names.isdisjoint(_FACTORY)
        assert "forklift_nearby" not in names
        assert "lighting_variation" in names


def test_variation_definition_lookup() -> None:
    conveyor = red.variation_definition_for("conveyor_motion")
    assert conveyor is not None and conveyor["label"]
    assert red.variation_definition_for("forklift_nearby") is not None  # base still resolvable
    assert red.variation_definition_for("does_not_exist") is None


def test_factory_definitions_shape() -> None:
    for definition in red.FACTORY_HAZARD_VARIATION_DEFINITIONS:
        assert {"variation_id", "label", "default_status"} <= set(definition)
