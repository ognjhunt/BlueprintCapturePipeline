from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
    _subject_bounds_in_scoring_frame,
)
from blueprint_pipeline.task_evaluation_rigid_destination_geometry import (
    RigidDestinationGeometryError,
    derive_rigid_destination_geometry,
)


SUBJECT_IDENTITY = {"id": "scene-841757-book-replacement", "version": "v1"}
DESTINATION_IDENTITY = {"id": "document-tray", "version": "v1"}
BOOK_HALF = [0.14765, 0.19885, 0.01057]
POSE = {
    "position_world_m": [3.25, -6.76, 0.275],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
}
LIMITS = {
    "maximum_penetration_m": 0.001,
    "minimum_support_contact_force_n": 0.01,
    "maximum_forbidden_contact_force_n": 0.1,
    "settle_translation_tolerance_m": 0.002,
    "settle_rotation_tolerance_rad": 0.01,
    "reset_translation_tolerance_m": 0.002,
    "reset_rotation_tolerance_rad": 0.01,
    "minimum_camera_pixels": {"external": 100, "wrist": 100, "overview": 100},
}


def _static(identity: dict, bounds: dict, rigid_paths: list[str], collision_paths: list[str]) -> dict:
    return {
        "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified",
        "replacement_identity": identity,
        "observed_structure": {
            "collision_bounds_body_frame_m": bounds,
            "rigid_body_paths": rigid_paths,
            "collision_prim_paths": collision_paths,
        },
    }


def _inputs(*, wall_height: float = 0.04, transform_offset: float = 0.0) -> dict:
    subject_static = _static(
        SUBJECT_IDENTITY,
        {
            "minimum": [-BOOK_HALF[0] + transform_offset, -BOOK_HALF[1], -BOOK_HALF[2]],
            "maximum": [BOOK_HALF[0] + transform_offset, BOOK_HALF[1], BOOK_HALF[2]],
        },
        ["/Asset"],
        ["/Asset/Collider"],
    )
    destination_static = _static(
        DESTINATION_IDENTITY,
        {"minimum": [-0.165, -0.24, 0.0], "maximum": [0.165, 0.24, 0.005 + wall_height]},
        ["/Asset"],
        [
            "/Asset/Colliders/Bottom",
            "/Asset/Colliders/Left",
            "/Asset/Colliders/Right",
            "/Asset/Colliders/Front",
            "/Asset/Colliders/Back",
        ],
    )
    simready = {
        "schema_version": "task_evaluation_passive_destination_simready.v1",
        "destination_identity": DESTINATION_IDENTITY,
        "intended_support_prim_paths": ["/Asset"],
        "intended_support_collision_prim_paths": ["/Asset/Colliders/Bottom"],
        "interior_bounds_body_frame_m": {
            "minimum": [-0.16, -0.235, 0.005],
            "maximum": [0.16, 0.235, 0.005 + wall_height],
        },
    }
    return {
        "subject_identity": SUBJECT_IDENTITY,
        "destination_identity": DESTINATION_IDENTITY,
        "relation": "inside",
        "pose_world": POSE,
        "subject_static_qualification": subject_static,
        "subject_static_qualification_digest": "sha256:" + "1" * 64,
        "subject_scoring_transform": {
            "position_m": [transform_offset, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "destination_static_qualification": destination_static,
        "destination_static_qualification_digest": "sha256:" + "2" * 64,
        "destination_simready_result": simready,
        "qualification_limits": LIMITS,
    }


def test_geometry_shrinks_the_tolerant_interior_by_the_whole_oriented_subject() -> None:
    geometry = derive_rigid_destination_geometry(**_inputs())
    schema = json.loads(
        Path("docs/schemas/task_evaluation_rigid_destination_geometry.v1.schema.json").read_text()
    )
    import jsonschema

    jsonschema.Draft202012Validator(schema).validate(geometry)
    assert geometry["geometry_digest"] == canonical_digest(geometry, digest_field="geometry_digest")
    assert geometry["status"] == "qualified"
    assert geometry["whole_subject_containment_encoded_by_shrunk_bounds"] is True
    # The subject scoring-frame bounds are exactly what the episode compiler recomputes.
    lower, upper = _subject_bounds_in_scoring_frame(
        bounds=_inputs()["subject_static_qualification"]["observed_structure"][
            "collision_bounds_body_frame_m"
        ],
        transform=_inputs()["subject_scoring_transform"],
    )
    assert geometry["subject_collision_bounds_scoring_frame_m"] == {
        "minimum": lower,
        "maximum": upper,
    }
    interior = geometry["destination_interior_bounds_body_frame_m"]
    assert interior["minimum"] == pytest.approx([-0.16, -0.235, 0.005 - 0.001])
    assert interior["maximum"] == pytest.approx([0.16, 0.235, 0.045])
    assert geometry["containment_floor_tolerance_m"] == 0.001
    bounds = geometry["destination_position_bounds_destination_frame_m"]
    assert bounds["minimum"] == pytest.approx(
        [-0.16 + BOOK_HALF[0], -0.235 + BOOK_HALF[1], 0.004 + BOOK_HALF[2]]
    )
    assert bounds["maximum"] == pytest.approx(
        [0.16 - BOOK_HALF[0], 0.235 - BOOK_HALF[1], 0.045 - BOOK_HALF[2]]
    )
    assert geometry["subject_orientation_destination_frame_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    # Support height is the world height of the subject scoring frame when the
    # subject rests on the tray floor, with penetration and settle tolerance.
    rest_z = POSE["position_world_m"][2] + 0.005 + BOOK_HALF[2]
    assert geometry["support_height_interval_m"] == pytest.approx(
        [rest_z - 0.003, rest_z + 0.003]
    )
    assert geometry["support_height_tolerance_m"] == pytest.approx(0.003)
    # Support contact routes to the destination's exact rigid body, and the exact
    # bottom collider is retained separately as evidence.
    assert geometry["intended_support_prim_paths"] == ["/Asset"]
    assert geometry["intended_support_collision_prim_paths"] == ["/Asset/Colliders/Bottom"]
    assert geometry["insertion_withdrawal_unit_destination_frame"] == [0.0, 0.0, 1.0]
    assert geometry["subject_static_qualification_digest"] == "sha256:" + "1" * 64
    assert geometry["destination_static_qualification_digest"] == "sha256:" + "2" * 64


def test_geometry_refuses_a_destination_too_shallow_for_the_whole_subject() -> None:
    with pytest.raises(RigidDestinationGeometryError, match="rigid_destination_geometry_subject_does_not_fit:z"):
        derive_rigid_destination_geometry(**_inputs(wall_height=0.02))


def test_geometry_uses_the_task_scoring_transform_not_the_asset_root() -> None:
    offset = derive_rigid_destination_geometry(**_inputs(transform_offset=0.03))
    centered = derive_rigid_destination_geometry(**_inputs())
    assert offset["subject_collision_bounds_scoring_frame_m"] == centered[
        "subject_collision_bounds_scoring_frame_m"
    ]


@pytest.mark.parametrize(
    ("mutate", "blocker"),
    [
        (
            lambda inputs: inputs["destination_simready_result"].update(
                intended_support_collision_prim_paths=["/Asset/Colliders/Lid"]
            ),
            "rigid_destination_geometry_support_prim_unknown",
        ),
        (
            lambda inputs: inputs["destination_simready_result"].update(
                intended_support_prim_paths=["/Asset/Colliders/Bottom"]
            ),
            "rigid_destination_geometry_support_body_unknown",
        ),
        (
            lambda inputs: inputs.update(relation="beside"),
            "rigid_destination_geometry_relation_invalid",
        ),
        (
            lambda inputs: inputs["destination_static_qualification"].update(
                replacement_identity=SUBJECT_IDENTITY
            ),
            "rigid_destination_geometry_identity_mismatch",
        ),
        (
            lambda inputs: inputs.update(
                pose_world={"position_world_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 2]}
            ),
            "rigid_destination_geometry_pose_invalid",
        ),
        (
            lambda inputs: inputs["qualification_limits"].pop("maximum_penetration_m"),
            "rigid_destination_geometry_limits_invalid",
        ),
    ],
)
def test_geometry_fails_closed(mutate, blocker) -> None:
    inputs = copy.deepcopy(_inputs())
    mutate(inputs)
    with pytest.raises(RigidDestinationGeometryError, match=blocker):
        derive_rigid_destination_geometry(**inputs)
