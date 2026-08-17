"""A target joint's commanded direction must open, not close, against the parent.

Scene 840920's washer door hinges vertically at its left edge with limits
``[0, 1.2] rad`` -- positive rotation only, so positive is the asset's own
declaration of "open".  Its axis was ``[0, 0, +1]``, and positive rotation about
``+Z`` sweeps the far edge from ``y = -0.302`` toward ``y = -0.240``, which is
the cabinet's front face.  The door opened *into* the machine.

On 2026-08-17 the probe commanded 15/30/45/55 degrees and read back
6.01 degrees every time -- the door wedged against the cabinet after roughly six
degrees and stopped.  400 N.m of drive authority could not move it, because the
obstruction was the parent link.

Nothing caught this before because the graph validator only checks that the
authored joint frame reproduces the declared axis; it never asks which way the
declared axis actually swings the child.  This pins the missing question, by
geometry, for any target revolute joint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


MANIFESTS = (
    Path(__file__).resolve().parents[1] / "docs" / "arm_decision_proof_v1" / "manifests"
)
SPEC = MANIFESTS / "third_scene_840920_task_a_simready_graph_asset_spec.v1.json"


def _spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _joint(spec: dict, joint_id: str) -> dict:
    for joint in spec["articulation_graph"]["joints"]:
        if joint["joint_id"] == joint_id:
            return joint
    raise AssertionError(f"joint not found: {joint_id}")


def _frame(spec: dict, joint_id: str) -> dict:
    for frame in spec["joint_frames"]:
        if frame["joint_id"] == joint_id:
            return frame
    raise AssertionError(f"joint frame not found: {joint_id}")


def _link(spec: dict, link_id: str) -> dict:
    for link in spec["links"]:
        if link["link_id"] == link_id:
            return link
    raise AssertionError(f"link not found: {link_id}")


def _parent_front_face_y(spec: dict, parent_link_id: str) -> float:
    """Smallest y any parent collision primitive occupies (the face the child faces)."""

    lowest = math.inf
    for geometry in _link(spec, parent_link_id).get("geometry", []):
        translate = geometry.get("translation_m") or [0.0, 0.0, 0.0]
        if geometry.get("kind") == "box":
            half = geometry["size_m"][1] / 2.0
        else:
            half = float(geometry.get("radius_m") or 0.0)
        lowest = min(lowest, translate[1] - half)
    assert lowest is not math.inf, "parent link has no collision geometry"
    return lowest


def test_spec_exists() -> None:
    assert SPEC.is_file()


def test_door_is_a_positive_only_target_joint() -> None:
    """The premise: positive rotation is this asset's declaration of 'open'."""

    door = _joint(_spec(), "door_hinge")
    assert door["role"] == "target"
    lower, upper = door["limits"]
    assert lower == 0.0 and upper > 0.0, (
        "this check assumes positive-only limits mean positive == opening"
    )


def test_commanded_direction_moves_the_door_away_from_the_cabinet() -> None:
    """The defect: +rotation drove the far edge toward the parent's front face."""

    spec = _spec()
    door = _joint(spec, "door_hinge")
    frame = _frame(spec, "door_hinge")

    axis_z = door["axis"][2]
    assert abs(axis_z) == pytest.approx(1.0), "door hinge is expected to be vertical"

    # Far edge of the disc, measured from the hinge, in the parent's frame.
    hinge_y = frame["parent_position_m"][1]
    radius = max(
        float(g.get("radius_m") or 0.0)
        for g in _link(spec, "door").get("geometry", [])
    )
    lever = 2.0 * radius  # hinge sits on the rim, so the far edge is a diameter away

    # Rotating by +theta about the vertical axis moves the far edge in y by
    # +lever*sin(theta) when axis_z is +1, and the other way when it is -1.
    upper = _joint(spec, "door_hinge")["limits"][1]
    delta_y = axis_z * lever * math.sin(min(upper, math.radians(10.0)))

    cabinet_front_y = _parent_front_face_y(spec, "body")
    assert hinge_y < cabinet_front_y, "door should start in front of the cabinet"

    assert delta_y < 0.0, (
        "commanded (positive) rotation moves the door's far edge from "
        f"y={hinge_y:.3f} toward the cabinet face at y={cabinet_front_y:.3f}. "
        "The door opens into the machine and wedges after a few degrees -- "
        "exactly the 6.01 degree stall observed on 2026-08-17. Flip the hinge "
        "axis sign so the commanded direction swings the door clear."
    )


def test_clearance_would_have_predicted_the_observed_stall() -> None:
    """Guards the guard: the pre-fix geometry really does jam near six degrees."""

    spec = _spec()
    frame = _frame(spec, "door_hinge")
    radius = max(
        float(g.get("radius_m") or 0.0)
        for g in _link(spec, "door").get("geometry", [])
    )
    clearance = _parent_front_face_y(spec, "body") - frame["parent_position_m"][1]
    jam_deg = math.degrees(math.asin(clearance / (2.0 * radius)))
    assert 4.0 <= jam_deg <= 9.0, (
        f"expected the wrong-way jam to land near the observed 6.01 deg, got {jam_deg:.2f}"
    )
