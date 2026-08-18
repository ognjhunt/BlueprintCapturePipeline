"""The scan authors the hinge; the sign is a theorem, not a keystroke.

Scene 840920 sealed a hand-typed axis of ``+Z`` where the measured geometry
demands ``-Z``; two paid runs read the jammed 6.01 degrees.  These tests build
a washer-shaped stage -- a cabinet box with a door shell protruding past its
front plate -- and require the derivation to verify a PROPOSED facing
direction (on the real scan the rear jumble out-shells the door, so facing
can never be auto-picked), find the plate, isolate the shell, and compute
the axis sign from the clearance rule.  Flipping the geometry must flip the
sign; a proposal naming a shell-less side must be refuted; no input can
reproduce the original bug, because the wrong sign is labelled a jam by
construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Usd, UsdGeom

from blueprint_pipeline.measured_articulation_derivation import (
    MeasuredArticulationError,
    derive_measured_articulation,
)


def _grid_face(mesh, *, axis, value, u_range, v_range, step=0.02):
    """A dense rectangle of vertices in a constant-<axis> plane."""

    points = list(mesh.GetPointsAttr().Get() or [])
    u = u_range[0]
    while u <= u_range[1] + 1e-9:
        v = v_range[0]
        while v <= v_range[1] + 1e-9:
            p = [0.0, 0.0, 0.0]
            p[axis] = value
            lateral = [i for i in range(3) if i != axis]
            p[lateral[0]] = u
            p[lateral[1]] = v
            points.append(tuple(p))
            v += step
        u += step
    mesh.GetPointsAttr().Set(points)


def _washer_stage(path: Path, *, door_forward: bool = True) -> Path:
    """Cabinet with a front plate and a protruding door shell.

    ``door_forward=True`` puts the door on the -y side (the 840920 layout);
    ``False`` mirrors the whole object to +y, which must flip the derived
    axis sign.
    """

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/scan")
    mesh.GetPointsAttr().Set([])
    sign = -1.0 if door_forward else 1.0

    # Cabinet: four vertical plates; the front one (at sign*0.24) is densest.
    for y in (sign * 0.24, -sign * 0.24):
        _grid_face(mesh, axis=1, value=y, u_range=(-0.3, 0.3), v_range=(0.0, 0.85))
    for x in (-0.3, 0.3):
        points = list(mesh.GetPointsAttr().Get() or [])
        z = 0.0
        while z <= 0.85:
            points.append((x, 0.0, z))
            z += 0.05
        mesh.GetPointsAttr().Set(points)
    # Door shell: a plate 6 cm proud of the front, narrower than the cabinet.
    _grid_face(
        mesh,
        axis=1,
        value=sign * 0.30,
        u_range=(-0.25, 0.25),
        v_range=(0.18, 0.68),
    )
    stage.GetRootLayer().Save()
    return path


def _derive(source, *, facing=-1, **kw):
    return derive_measured_articulation(
        source_usd_path=source,
        facing_outward_sign=facing,
        facing_proposed_by="test_scene_context",
        **kw,
    )


def test_front_plate_shell_and_axis_are_measured(tmp_path: Path) -> None:
    source = _washer_stage(tmp_path / "washer.usda")
    derived = _derive(source)

    assert derived["front_plate"]["plane_m"] == pytest.approx(-0.24, abs=0.011)
    assert derived["forward_shell"]["front_m"] == pytest.approx(-0.30, abs=0.011)

    joint = derived["target_joint"]
    assert joint["axis"] == [0.0, 0.0, -1.0]
    assert joint["pivot_asset_m"][0] == pytest.approx(-0.25, abs=0.02)
    assert joint["pivot_asset_m"][1] == pytest.approx(-0.30, abs=0.011)
    # The receipt shows the decision: the rejected sign is labelled a jam.
    verdicts = {c["axis_sign"]: c["verdict"] for c in joint["sign_candidates"]}
    assert verdicts[-1] == "opens_clear_of_parent"
    assert verdicts[1] == "jams_into_parent"
    assert derived["claim_boundary"]["axis_sign_is_derived_not_input"] is True
    assert derived["claim_boundary"]["physics_typed_by_hand"] is False
    assert derived["front_plate"]["facing_is_proposal_not_measurement"] is True
    assert derived["front_plate"]["facing_proposed_by"] == "test_scene_context"


def test_mirrored_geometry_flips_the_derived_sign(tmp_path: Path) -> None:
    """The sign follows the geometry, not a convention someone remembered."""

    mirrored = _washer_stage(tmp_path / "mirrored.usda", door_forward=False)
    derived = _derive(mirrored, facing=1)
    assert derived["target_joint"]["axis"] == [0.0, 0.0, 1.0]


def test_right_hinge_flips_the_sign_too(tmp_path: Path) -> None:
    source = _washer_stage(tmp_path / "washer.usda")
    left = _derive(source, hinge_side="left")
    right = _derive(source, hinge_side="right")
    assert left["target_joint"]["axis"][2] == -right["target_joint"]["axis"][2]
    assert right["target_joint"]["pivot_asset_m"][0] == pytest.approx(0.25, abs=0.02)


def test_derivation_is_replayable_byte_for_byte(tmp_path: Path) -> None:
    source = _washer_stage(tmp_path / "washer.usda")
    first = _derive(source)
    second = _derive(source)
    assert first == second
    assert first["derivation_digest"].startswith("sha256:")


def test_an_object_with_no_forward_shell_is_refused(tmp_path: Path) -> None:
    """A flat-fronted object has no measurable door; refusing beats guessing."""

    stage = Usd.Stage.CreateNew(str(tmp_path / "slab.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/scan")
    mesh.GetPointsAttr().Set([])
    _grid_face(mesh, axis=1, value=-0.24, u_range=(-0.3, 0.3), v_range=(0.0, 0.85))
    _grid_face(mesh, axis=1, value=0.24, u_range=(-0.3, 0.3), v_range=(0.0, 0.85))
    stage.GetRootLayer().Save()
    with pytest.raises(MeasuredArticulationError) as caught:
        _derive(tmp_path / "slab.usda")
    # A flat-fronted object cannot corroborate any facing proposal.
    assert "measured_facing_proposal_refuted" in caught.value.errors


def test_a_sparse_scan_is_refused(tmp_path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(tmp_path / "sparse.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/scan")
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 1, 1)])
    stage.GetRootLayer().Save()
    with pytest.raises(MeasuredArticulationError) as caught:
        _derive(tmp_path / "sparse.usda")
    assert "measured_source_too_sparse" in caught.value.errors


def test_a_proposal_naming_the_shell_less_side_is_refuted(tmp_path: Path) -> None:
    """The real-scan lesson: the wrong face must refuse, not quietly derive."""

    source = _washer_stage(tmp_path / "washer.usda")
    with pytest.raises(MeasuredArticulationError) as caught:
        _derive(source, facing=1)
    assert "measured_facing_proposal_refuted" in caught.value.errors


def test_a_proposal_without_provenance_is_refused(tmp_path: Path) -> None:
    source = _washer_stage(tmp_path / "washer.usda")
    with pytest.raises(MeasuredArticulationError) as caught:
        derive_measured_articulation(
            source_usd_path=source, facing_outward_sign=-1, facing_proposed_by="  "
        )
    assert "measured_facing_provenance_required" in caught.value.errors
