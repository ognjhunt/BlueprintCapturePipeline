"""R069: multi-floor / vertical-structure capture truth on the Site card.

The ``site_levels`` block is additive and carries only declared or measured
capture inputs (per-level descriptors plus mezzanine / multi-level-racking
indicators). Absent values are marked ``needs_capture_or_operator_input`` and
are never fabricated, mirroring the R017 ``site_extent`` capture-truth contract.
"""

from __future__ import annotations

from blueprint_pipeline import robot_eval_dataset


def test_site_levels_block_sources_declared_levels_from_manifest() -> None:
    block = robot_eval_dataset._vertical_structure(
        raw_manifest={
            "site_levels": [
                {"name": "Ground", "elevation_m": 0.0},
                {"level_name": "Mezzanine A", "elevation_m": 4.5, "is_mezzanine": True},
            ],
            "floor_count": 2,
        },
        metadata={},
    )

    assert block["level_count"] == 2
    assert block["levels"][0]["name"] == "Ground"
    assert block["levels"][1]["name"] == "Mezzanine A"
    assert block["levels"][1]["is_mezzanine"] is True
    assert block["floor_count"] == 2
    assert block["sources"]["levels"] == "capture_manifest"
    assert block["sources"]["floor_count"] == "capture_manifest"
    # A declared mezzanine level strengthens the mezzanine indicator without an
    # explicit top-level flag being present.
    assert block["mezzanine_present"] is True
    assert block["is_multi_level"] is True
    assert block["status"] == "declared_present"


def test_site_levels_uses_operator_metadata_only_when_manifest_lacks_levels() -> None:
    block = robot_eval_dataset._vertical_structure(
        raw_manifest={},
        metadata={
            "levels": [{"name": "L1"}, {"name": "L2"}],
            "multi_level_racking": True,
        },
    )

    assert block["level_count"] == 2
    assert block["sources"]["levels"] == "site_operator_metadata"
    assert block["multi_level_racking_present"] is True
    assert block["sources"]["multi_level_racking_present"] == "site_operator_metadata"
    assert block["is_multi_level"] is True


def test_site_levels_absent_marks_needs_capture_and_never_fabricates() -> None:
    block = robot_eval_dataset._vertical_structure(raw_manifest={}, metadata={})

    assert block["levels"] == []
    assert block["level_count"] == 0
    assert block["floor_count"] is None
    assert block["mezzanine_present"] is False
    assert block["multi_level_racking_present"] is False
    assert block["is_multi_level"] is False
    assert block["status"] == "needs_capture_or_operator_input"
    assert block["sources"]["levels"] == "needs_capture_or_operator_input"
    assert block["sources"]["floor_count"] == "needs_capture_or_operator_input"
    assert block["sources"]["mezzanine_present"] == "needs_capture_or_operator_input"


def test_site_levels_floor_count_alone_marks_multi_level() -> None:
    # A flat declared floor_count > 1 is enough to mark the site multi-level even
    # without per-level descriptors; nothing about mezzanines is fabricated.
    block = robot_eval_dataset._vertical_structure(
        raw_manifest={"floor_count": 3},
        metadata={},
    )

    assert block["floor_count"] == 3
    assert block["level_count"] == 0
    assert block["is_multi_level"] is True
    assert block["mezzanine_present"] is False
    assert block["status"] == "declared_present"


def test_site_levels_claim_boundary_marks_declared_not_verified() -> None:
    block = robot_eval_dataset._vertical_structure(
        raw_manifest={"mezzanine": True},
        metadata={},
    )
    assert block["mezzanine_present"] is True
    assert block["sources"]["mezzanine_present"] == "capture_manifest"
    assert "declared_or_measured" in block["claim_boundary"]
    assert "not_derived_or_verified" in block["claim_boundary"]
