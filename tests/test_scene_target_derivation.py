"""Step 2 of the frozen screening sequence, as code.

The preregistration freezes "derive_unique_nonoverlapped_target_collider_and_
horizontal_support" as the step after hashing inputs. Only steps 1 and 3 were
implemented, so geometry eligibility was still decided by hand -- and deciding
it by hand is how a scene that passes on bytes but has no workable target gets
mistaken for the next scene to run.

Two details cost real time when this was derived manually and are pinned here:

* An object resting on a support must not count as colliding with that support.
  Inflating a candidate's box by the contact envelope in every direction makes
  every supported object overlap the thing holding it up, which rejects exactly
  the objects a pick task needs and leaves only wall-mounted ones.
* A support has to be rigid. A candidate resting on a quilt or a pillow is not
  a pick-and-place target, however well it passes the geometric test.
"""

import pytest

from blueprint_pipeline.scene_target_derivation import (
    SceneTargetDerivationError,
    derive_scene_targets,
)

# Robotiq 2F-85 aperture minus the 10 mm approved contact envelope.
USABLE_GRASP_M = 0.075


def _box(x0, y0, z0, x1, y1, z1):
    return [
        {"x": x, "y": y, "z": z}
        for x in (x0, x1)
        for y in (y0, y1)
        for z in (z0, z1)
    ]


def _label(ins_id, label, box):
    return {"ins_id": ins_id, "label": label, "bounding_box": box}


def _table(ins_id="900"):
    # 1.0 x 0.6 m top at 0.75 m.
    return _label(ins_id, "table", _box(0.0, 0.0, 0.0, 1.0, 0.6, 0.75))


def _can_on_table(ins_id="177"):
    # 57 mm wide, 120 mm tall, sitting exactly on the table top.
    return _label(ins_id, "drinks", _box(0.40, 0.25, 0.75, 0.457, 0.307, 0.87))


def test_finds_a_graspable_object_resting_on_a_rigid_support() -> None:
    result = derive_scene_targets([_table(), _can_on_table()])
    assert [t["ins_id"] for t in result["targets"]] == ["177"]
    target = result["targets"][0]
    assert target["semantic_label"] == "drinks"
    assert target["support_label"] == "table"
    assert target["width_m"] == pytest.approx(0.057, abs=1e-3)


def test_the_support_does_not_count_as_an_overlapping_neighbour() -> None:
    """Inflating downward would reject every object that rests on something."""
    result = derive_scene_targets([_table(), _can_on_table()])
    assert result["targets"], (
        "a can on a table must survive; counting the table as a collision "
        "leaves only wall-mounted objects"
    )


def test_rejects_a_target_wider_than_the_usable_grasp_clearance() -> None:
    wide = _label("300", "box", _box(0.2, 0.2, 0.75, 0.2 + 0.090, 0.29, 0.85))
    result = derive_scene_targets([_table(), wide])
    assert result["targets"] == []
    assert result["usable_grasp_width_m"] == pytest.approx(USABLE_GRASP_M, abs=1e-6)


def test_rejects_a_target_resting_on_a_soft_support() -> None:
    """A candidate on a quilt is not a pick-and-place target."""
    quilt = _label("901", "quilt", _box(0.0, 0.0, 0.0, 1.0, 0.6, 0.53))
    plant = _label("241", "Green plants", _box(0.4, 0.25, 0.53, 0.453, 0.303, 0.616))
    result = derive_scene_targets([quilt, plant])
    assert result["targets"] == []


def test_rejects_a_target_crowded_by_a_neighbour() -> None:
    """The jaws need envelope clearance on both sides of the object."""
    neighbour = _label("178", "jar", _box(0.458, 0.25, 0.75, 0.51, 0.30, 0.85))
    result = derive_scene_targets([_table(), _can_on_table(), neighbour])
    assert result["targets"] == []


def test_rejects_a_floor_level_object() -> None:
    floor_item = _label("400", "trash can", _box(0.4, 0.25, 0.0, 0.45, 0.30, 0.1))
    assert derive_scene_targets([_table(), floor_item])["targets"] == []


def test_is_deterministic_and_digest_bound() -> None:
    labels = [_table(), _can_on_table()]
    first = derive_scene_targets(labels)
    second = derive_scene_targets(labels)
    assert first["derivation_digest"] == second["derivation_digest"]
    assert first["derivation_digest"].startswith("sha256:")


def test_digest_changes_when_the_geometry_changes() -> None:
    before = derive_scene_targets([_table(), _can_on_table()])["derivation_digest"]
    moved = _label("177", "drinks", _box(0.10, 0.25, 0.75, 0.157, 0.307, 0.87))
    after = derive_scene_targets([_table(), moved])["derivation_digest"]
    assert before != after


def test_refuses_labels_that_are_not_a_sequence_of_mappings() -> None:
    with pytest.raises(SceneTargetDerivationError):
        derive_scene_targets("not-labels")


def test_ignores_entries_without_a_bounding_box() -> None:
    """Real InteriorGS exports contain at least one box-less entry."""
    result = derive_scene_targets([_table(), _can_on_table(), {"ins_id": "z", "label": "room"}])
    assert [t["ins_id"] for t in result["targets"]] == ["177"]


def test_reports_no_provider_mutation() -> None:
    assert derive_scene_targets([_table(), _can_on_table()])["provider_mutation_performed"] is False
