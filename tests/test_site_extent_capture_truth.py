"""R017: site scale/dimensional capture truth on the manifest and Site card."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from blueprint_pipeline import ios_manifest
from blueprint_pipeline import robot_eval_dataset


def test_ios_manifest_site_extent_fields_round_trip() -> None:
    payload = {
        "scene_id": "warehouse-1",
        "video_uri": "gs://bucket/scenes/warehouse-1/raw/walkthrough.mov",
        "approx_floor_area_m2": 18500.5,
        "ceiling_height_m": 12.4,
        "floor_count": 3,
        "dominant_aisle_width_m": 3.2,
    }
    manifest = ios_manifest.IOSManifest.from_dict(payload)

    assert manifest.approx_floor_area_m2 == 18500.5
    assert manifest.ceiling_height_m == 12.4
    assert manifest.floor_count == 3
    assert manifest.dominant_aisle_width_m == 3.2

    # Round-trip through dataclass serialization -> from_dict preserves values.
    round_tripped = ios_manifest.IOSManifest.from_dict(dataclasses.asdict(manifest))
    assert round_tripped.approx_floor_area_m2 == 18500.5
    assert round_tripped.ceiling_height_m == 12.4
    assert round_tripped.floor_count == 3
    assert round_tripped.dominant_aisle_width_m == 3.2


def test_ios_manifest_without_site_extent_still_parses_as_none() -> None:
    # Backward compatibility: existing captures with no site-extent keys parse,
    # with all new fields defaulting to None (never fabricated).
    manifest = ios_manifest.IOSManifest.from_dict(
        {"scene_id": "legacy", "video_uri": "gs://bucket/legacy/raw.mov"}
    )
    assert manifest.approx_floor_area_m2 is None
    assert manifest.ceiling_height_m is None
    assert manifest.floor_count is None
    assert manifest.dominant_aisle_width_m is None

    # Blank strings are tolerated as absent, not coerced to 0.
    blank = ios_manifest.IOSManifest.from_dict(
        {"scene_id": "blank", "approx_floor_area_m2": "", "floor_count": ""}
    )
    assert blank.approx_floor_area_m2 is None
    assert blank.floor_count is None


def test_site_extent_block_sources_and_marks_missing() -> None:
    # Manifest-declared values win and are labelled capture_manifest.
    from_manifest = robot_eval_dataset._site_extent(
        raw_manifest={
            "approx_floor_area_m2": 20000,
            "ceiling_height_m": 11.0,
            "floor_count": 2,
        },
        metadata={"dominant_aisle_width_m": 2.8},
    )
    assert from_manifest["approx_floor_area_m2"] == 20000.0
    assert from_manifest["floor_count"] == 2
    assert from_manifest["sources"]["approx_floor_area_m2"] == "capture_manifest"
    # Operator metadata is used only when the manifest lacks the field.
    assert from_manifest["dominant_aisle_width_m"] == 2.8
    assert from_manifest["sources"]["dominant_aisle_width_m"] == "site_operator_metadata"
    assert from_manifest["status"] == "declared_present"

    # Fully absent -> nulls plus an explicit needs_capture_or_operator_input marker.
    missing = robot_eval_dataset._site_extent(raw_manifest={}, metadata={})
    assert missing["approx_floor_area_m2"] is None
    assert missing["ceiling_height_m"] is None
    assert missing["floor_count"] is None
    assert missing["dominant_aisle_width_m"] is None
    assert missing["status"] == "needs_capture_or_operator_input"
    for source in missing["sources"].values():
        assert source == "needs_capture_or_operator_input"


def test_site_card_carries_site_extent() -> None:
    context = SimpleNamespace(scene_id="warehouse-1", capture_id="cap-1")
    site_card = robot_eval_dataset._site_card(
        context=context,
        descriptor={},
        raw_manifest={
            "site_type": "warehouse aisle",
            "approx_floor_area_m2": 18500.5,
            "ceiling_height_m": 12.4,
            "floor_count": 3,
            "dominant_aisle_width_m": 3.2,
        },
        site_world_spec={},
        object_geometry_manifest={},
        task_library={},
        source_artifacts={},
        simready_scene_manifest={},
        marble_validation={},
        marble_bridge={},
        worldlabs_world_manifest={},
        cpu_preflight_scorecard={},
        protected_regions_manifest={},
        rights_privacy={},
        generated_at="2026-07-09T00:00:00Z",
    )

    # Existing site_card fields remain intact.
    assert site_card["site_type"] == "warehouse aisle"
    assert "geometry" in site_card

    extent = site_card["site_extent"]
    assert extent["approx_floor_area_m2"] == 18500.5
    assert extent["ceiling_height_m"] == 12.4
    assert extent["floor_count"] == 3
    assert extent["dominant_aisle_width_m"] == 3.2
    assert extent["status"] == "declared_present"
    assert extent["sources"]["approx_floor_area_m2"] == "capture_manifest"
    assert extent["units"]["ceiling_height_m"] == "meters"
    assert "not_derived_or_verified_claims" in extent["claim_boundary"]
