from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.sage_collision_identity import (
    SageCollisionIdentityError,
    inspect_sage_collision_identity,
)


def _box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> list[dict]:
    return [
        {"x": x, "y": y, "z": z}
        for z in (z0, z1)
        for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))
    ]


def _mesh(stage, path: str, minimum: tuple[float, float, float], maximum: tuple[float, float, float]):
    from pxr import UsdGeom, UsdPhysics

    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 4, 0, 3, 7]
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    from pxr import Usd, UsdGeom

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [
                {
                    "ins_id": "160",
                    "label": "canned beverage",
                    "bounding_box": _box(2.0, 2.0, 0.8, 2.1, 2.1, 1.0),
                },
                {
                    "ins_id": "22",
                    "label": "door",
                    "bounding_box": _box(0.0, 0.0, 0.0, 2.0, 0.2, 2.0),
                },
            ]
        ),
        encoding="utf-8",
    )
    collision = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(collision))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _mesh(stage, "/Root/Can", (2.0, 2.0, 0.8), (2.1, 2.1, 1.0))
    _mesh(stage, "/Root/DoorAssembly", (0.0, 0.0, 0.0), (2.0, 0.2, 2.0))
    _mesh(stage, "/Root/DoorLeft", (0.0, 0.0, 0.0), (0.9, 0.2, 2.0))
    _mesh(stage, "/Root/DoorRight", (1.1, 0.0, 0.0), (2.0, 0.2, 2.0))
    stage.GetRootLayer().Save()
    return labels, collision


def test_original_rigid_fixture_has_one_whole_object_match(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path)

    observed = inspect_sage_collision_identity(
        labels_path=labels,
        target_instance_id="160",
        sage_collision_usd_path=collision,
    )

    assert observed["whole_object_collision_identity_passed"] is True
    assert [row["prim_path"] for row in observed["whole_object_matches"]] == ["/Root/Can"]
    assert observed["candidate_subpart_count"] == 0
    assert observed["receipt_digest"].startswith("sha256:")


def test_articulated_fixture_reports_parts_without_promoting_them(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path)

    observed = inspect_sage_collision_identity(
        labels_path=labels,
        target_instance_id="22",
        sage_collision_usd_path=collision,
    )

    assert observed["whole_object_collision_identity_passed"] is True
    assert [row["prim_path"] for row in observed["candidate_subpart_meshes"]] == [
        "/Root/DoorLeft",
        "/Root/DoorRight",
    ]
    assert observed["claim_boundary"]["candidate_subpart_is_not_moving_link_proof"]
    assert observed["claim_boundary"]["joint_axis_or_limits_proven"] is False


def test_identity_inspection_fails_closed_for_unknown_instance(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path)

    with pytest.raises(SageCollisionIdentityError) as caught:
        inspect_sage_collision_identity(
            labels_path=labels,
            target_instance_id="missing",
            sage_collision_usd_path=collision,
        )

    assert caught.value.errors == ("interiorgs_target_instance_not_exactly_one",)
