from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.sage_collision_component_topology import (
    SageCollisionComponentTopologyError,
    inspect_sage_collision_component_topology,
)


def _label_box(
    instance_id: str,
    label: str,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> dict:
    return {
        "ins_id": instance_id,
        "label": label,
        "bounding_box": [
            {"x": x, "y": y, "z": z}
            for z in (minimum[2], maximum[2])
            for x, y in (
                (minimum[0], minimum[1]),
                (minimum[0], maximum[1]),
                (maximum[0], maximum[1]),
                (maximum[0], minimum[1]),
            )
        ],
    }


def _add_box(points: list, faces: list, minimum: tuple, maximum: tuple) -> None:
    start = len(points)
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    points.extend(
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
    faces.extend(
        [
            tuple(start + value for value in row)
            for row in (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            )
        ]
    )


def _add_open_receptacle(points: list, faces: list) -> None:
    _add_box(points, faces, (0.0, 0.0, 0.0), (1.0, 1.0, 0.1))
    _add_box(points, faces, (0.0, 0.0, 0.1), (0.1, 1.0, 1.0))
    _add_box(points, faces, (0.9, 0.0, 0.1), (1.0, 1.0, 1.0))
    _add_box(points, faces, (0.1, 0.0, 0.1), (0.9, 0.1, 1.0))
    _add_box(points, faces, (0.1, 0.9, 0.1), (0.9, 1.0, 1.0))
    # Join the separately authored wall boxes into one topological component
    # without adding a horizontal cap over the interior.
    faces.extend(
        [
            (8, 16, 17),
            (9, 24, 25),
            (10, 40, 41),
            (11, 32, 33),
        ]
    )


def _fixture(tmp_path: Path, *, capped: bool = False) -> tuple[Path, Path]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [
                _label_box("cloth", "towel", (2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
                _label_box("bin", "basket", (10.0, 0.0, 0.0), (11.0, 1.0, 1.0)),
            ]
        ),
        encoding="utf-8",
    )
    stage_path = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Combined")
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    _add_box(points, faces, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    _add_open_receptacle(points, faces)
    if capped:
        _add_box(points, faces, (0.1, 0.1, 0.9), (0.9, 0.9, 1.0))
        faces.append((8, len(points) - 1, len(points) - 2))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([value for face in faces for value in face])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    transform = UsdGeom.Xformable(mesh).AddTransformOp()
    transform.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(2.0, 0.0, 0.0)))
    # Reposition the receptacle component inside the same source mesh so that
    # its world identity is different from the first component.
    for index in range(8, len(points)):
        point = mesh.GetPointsAttr().Get()[index]
        point[0] += 8.0
        values = list(mesh.GetPointsAttr().Get())
        values[index] = point
        mesh.GetPointsAttr().Set(values)
    stage.GetRootLayer().Save()
    return labels, stage_path


def test_matches_transformed_components_and_proves_open_collision_cavity(
    tmp_path: Path,
) -> None:
    labels, collision = _fixture(tmp_path)

    result = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=["cloth", "bin"],
        opening_probe_instance_ids=["bin"],
        sage_collision_usd_path=collision,
    )

    assert result["all_component_collision_identities_passed"] is True
    by_id = {row["interiorgs_instance_id"]: row for row in result["targets"]}
    assert by_id["cloth"]["best_component"]["aabb_iou"] == pytest.approx(1.0)
    assert by_id["bin"]["best_component"]["aabb_iou"] == pytest.approx(1.0)
    opening = by_id["bin"]["opening_probe"]
    assert opening["open_collision_cavity_passed"] is True
    assert opening["center_first_hit_band"] == "floor"
    assert opening["cavity_depth_m"] == pytest.approx(0.9)
    assert result["coordinate_frame"][
        "authored_prim_local_to_world_transforms_applied"
    ] is True


def test_closed_cap_fails_open_cavity_probe(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path, capped=True)

    result = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=["bin"],
        opening_probe_instance_ids=["bin"],
        sage_collision_usd_path=collision,
        minimum_component_iou=0.7,
    )

    opening = result["targets"][0]["opening_probe"]
    assert opening["open_collision_cavity_passed"] is False
    assert opening["center_first_hit_band"] == "cap"


def test_rejects_unrequested_opening_probe_target(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path)

    with pytest.raises(SageCollisionComponentTopologyError) as caught:
        inspect_sage_collision_component_topology(
            labels_path=labels,
            target_instance_ids=["cloth"],
            opening_probe_instance_ids=["bin"],
            sage_collision_usd_path=collision,
        )

    assert caught.value.errors == ("opening_probe_target_not_in_requested_targets",)
