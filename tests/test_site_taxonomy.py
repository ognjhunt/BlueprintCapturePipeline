"""Tests for the shared canonical site-type taxonomy (audit R018)."""

from __future__ import annotations

from blueprint_pipeline import episode_spec as ep
from blueprint_pipeline import site_taxonomy as st


def test_recognizes_expanded_industrial_synonyms() -> None:
    assert st.resolve_site_type("regional distribution center").category == "warehouse"
    assert st.resolve_site_type("e-commerce fulfillment center").category == "warehouse"
    assert st.resolve_site_type("automotive assembly plant").category == "manufacturing"
    assert st.resolve_site_type("line-side station").category == "manufacturing"
    assert st.resolve_site_type("cold storage freezer aisle").category == "cold_storage"
    assert st.resolve_site_type("grocery backroom").category == "stockroom"


def test_longest_synonym_wins() -> None:
    # "cold storage" (12) must beat the shorter "aisle" (5) -> warehouse.
    res = st.resolve_site_type("cold storage aisle")
    assert res.category == "cold_storage"
    assert res.matched_token == "cold storage"
    # "kitchen" beats "home"
    assert st.resolve_site_type("home kitchen").category == "kitchen"


def test_industrial_flag_and_set() -> None:
    assert st.resolve_site_type("warehouse aisle").is_industrial is True
    assert st.resolve_site_type("factory floor").is_industrial is True
    assert st.resolve_site_type("home kitchen").is_industrial is False
    assert set(st.industrial_site_categories()) == {
        "warehouse",
        "manufacturing",
        "cold_storage",
        "stockroom",
    }


def test_unrecognized_is_explicit_not_guessed() -> None:
    for text in ("aquarium touch tank", "", None, "   "):
        res = st.resolve_site_type(text)
        assert res.recognized is False
        assert res.category == st.UNKNOWN_SITE_CATEGORY
        assert res.is_industrial is False
        assert res.matched_token is None


def test_versioned_and_enumerated() -> None:
    assert st.SITE_TAXONOMY_VERSION == "v1"
    cats = st.canonical_site_categories()
    for expected in ("warehouse", "manufacturing", "cold_storage", "stockroom", "kitchen"):
        assert expected in cats
    assert st.resolve_site_type("warehouse").taxonomy_version == "v1"


def test_default_task_hints() -> None:
    assert st.default_task_hint_for_category("warehouse")["task_id"] == "warehouse_tote_transfer"
    assert st.default_task_hint_for_category("manufacturing")["task_id"] == "factory_line_side_delivery"
    assert st.default_task_hint_for_category("residential") is None


def test_episode_spec_hints_preserve_legacy_and_add_synonyms() -> None:
    # Legacy literal-token behavior preserved (pinned by existing tests too).
    assert ep._scene_class_task_hints("warehouse")[0]["task_id"] == "warehouse_tote_transfer"
    # Expanded synonyms now recognized instead of falling through to review-only.
    fulfillment = ep._scene_class_task_hints("regional fulfillment center")
    assert any(h["task_id"] == "warehouse_tote_transfer" for h in fulfillment)
    manufacturing = ep._scene_class_task_hints("automotive assembly plant")
    assert any(h["task_id"] == "factory_line_side_delivery" for h in manufacturing)
    # Genuinely unknown site types still produce no hint (so the unrecognized
    # note continues to surface).
    assert ep._scene_class_task_hints("aquarium touch tank") == []
