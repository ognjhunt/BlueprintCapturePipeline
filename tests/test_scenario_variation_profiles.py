"""Tests for location-type-scoped scenario variation profiles (audit R013/R014)."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import robot_eval_dataset as red
from blueprint_pipeline import site_taxonomy
from blueprint_pipeline import success_claim_contracts as scc

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FACTORY_FIXTURE = (
    _REPO_ROOT
    / "tests"
    / "fixtures"
    / "factory_task_min"
    / "factory_task_scaling_request.json"
)

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


# ---------------------------------------------------------------------------
# R014 — the committed factory/manufacturing fixture, and the R013 tie-in:
# a manufacturing-site task requires the factory hazards; a kitchen task does not.
# ---------------------------------------------------------------------------


def test_factory_fixture_is_committed_and_shaped_like_warehouse() -> None:
    assert _FACTORY_FIXTURE.is_file(), "factory truth fixture request is missing"
    manifest = _FACTORY_FIXTURE.with_name(
        "factory_task_scaling_preflight_manifest.json"
    )
    assert manifest.is_file(), "factory preflight manifest is missing"
    request = json.loads(_FACTORY_FIXTURE.read_text())
    # Schema is NOT forked from the kitchen preflight contract.
    assert request["schema_version"] == "kitchen_task_scaling_preflight_request.v1"
    assert (
        json.loads(manifest.read_text())["schema_version"]
        == "kitchen_task_scaling_preflight.v1"
    )
    assert json.loads(manifest.read_text())["local_preflight_status"] == "passed"


def test_factory_fixture_exercises_both_reach_branches() -> None:
    scenarios = json.loads(_FACTORY_FIXTURE.read_text())["scenarios"]
    assert len(scenarios) == 2
    # Scenario 1 declares target/affordance ids -> reach evidence is required, so
    # visible arm presence alone can never pass it.
    first = scc.derive_task_proof_requirements(scenarios[0])
    assert first["requires_reach_to_affordance"] is True
    # Scenario 2 declares neither -> no reach requirement is derived.
    second = scc.derive_task_proof_requirements(scenarios[1])
    assert second["requires_reach_to_affordance"] is False


def test_factory_fixture_text_resolves_to_manufacturing_profile() -> None:
    task_text = json.loads(_FACTORY_FIXTURE.read_text())["scenarios"][0]["task"]
    resolution = site_taxonomy.resolve_site_type(task_text)
    assert resolution.category == "manufacturing"
    # And that manufacturing profile is exactly the one that carries factory hazards.
    required = set(red.required_variation_names_for_site_category(resolution.category))
    assert _FACTORY <= required


def test_kitchen_task_is_not_forced_to_cover_factory_hazards() -> None:
    kitchen_required = set(red.required_variation_names_for_site_category("kitchen"))
    assert kitchen_required.isdisjoint(_FACTORY)
