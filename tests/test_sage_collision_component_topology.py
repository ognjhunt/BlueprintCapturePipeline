from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import blueprint_pipeline.sage_collision_component_topology as topology_module
from blueprint_pipeline.sage_collision_component_topology import (
    SageCollisionComponentTopologyError,
    _opening_probe,
    inspect_sage_collision_component_topology,
    read_sage_collision_component_geometry,
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


def _add_floor_patch(
    points: list,
    faces: list,
    *,
    minimum_xy: tuple[float, float],
    maximum_xy: tuple[float, float],
) -> None:
    start = len(points)
    points.extend(
        [
            (minimum_xy[0], minimum_xy[1], 0.0),
            (maximum_xy[0], minimum_xy[1], 0.0),
            (maximum_xy[0], maximum_xy[1], 0.0),
            (minimum_xy[0], maximum_xy[1], 0.0),
        ]
    )
    faces.append(tuple(start + value for value in (0, 1, 2, 3)))


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
    assert result["coordinate_frame"]["authored_prim_local_to_world_transforms_applied"] is True


def test_labels_are_parsed_and_identified_from_one_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels, collision = _fixture(tmp_path)
    original_bytes = labels.read_bytes()
    original_loader = topology_module.load_interiorgs_labels
    swapped = False

    def swap_source_then_parse(snapshot_path: Path) -> list:
        nonlocal swapped
        if not swapped:
            swapped = True
            labels.write_text(
                json.dumps(
                    [
                        _label_box(
                            "bin",
                            "basket",
                            (100.0, 100.0, 100.0),
                            (101.0, 101.0, 101.0),
                        )
                    ]
                ),
                encoding="utf-8",
            )
        return original_loader(snapshot_path)

    monkeypatch.setattr(
        topology_module,
        "load_interiorgs_labels",
        swap_source_then_parse,
    )

    result = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=["bin"],
        opening_probe_instance_ids=["bin"],
        sage_collision_usd_path=collision,
    )

    assert swapped is True
    assert result["source_files"]["interiorgs_labels"]["size_bytes"] == len(original_bytes)
    assert result["source_files"]["interiorgs_labels"]["sha256"] == (
        "sha256:" + hashlib.sha256(original_bytes).hexdigest()
    )
    assert result["targets"][0]["label_world_aabb_min_m"] == pytest.approx([10.0, 0.0, 0.0])


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


def test_sparse_disconnected_floor_hits_cannot_claim_spanning_opening() -> None:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for x, y in ((0.5, 0.5), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        _add_floor_patch(
            points,
            faces,
            minimum_xy=(x - 0.04, y - 0.04),
            maximum_xy=(x + 0.04, y + 0.04),
        )

    opening = _opening_probe(
        vertices=points,
        faces=faces,
        bounds_min=[0.0, 0.0, 0.0],
        bounds_max=[1.0, 1.0, 1.0],
        grid_size=5,
        margin_fraction=0.0,
        floor_band_fraction=0.25,
        cap_band_fraction=0.75,
    )

    assert opening["floor_hit_count"] == 5
    assert opening["floor_hit_fraction"] == pytest.approx(0.2)
    assert opening["center_connected_floor_rectangle_cell_count"] == 1
    assert opening["center_connected_floor_rectangle_fraction"] == pytest.approx(0.04)
    assert opening["open_collision_cavity_passed"] is False
    assert opening["conservative_clear_opening"] is None


def test_asymmetric_all_floor_rectangle_preserves_measured_offset() -> None:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    _add_floor_patch(
        points,
        faces,
        minimum_xy=(0.45, 0.2),
        maximum_xy=(0.8, 0.8),
    )
    _add_box(points, faces, (0.0, 0.0, 0.0), (0.05, 1.0, 1.0))
    _add_box(points, faces, (0.95, 0.0, 0.0), (1.0, 1.0, 1.0))
    _add_box(points, faces, (0.05, 0.0, 0.0), (0.95, 0.05, 1.0))
    _add_box(points, faces, (0.05, 0.95, 0.0), (0.95, 1.0, 1.0))

    opening = _opening_probe(
        vertices=points,
        faces=faces,
        bounds_min=[0.0, 0.0, 0.0],
        bounds_max=[1.0, 1.0, 1.0],
        grid_size=5,
        margin_fraction=0.0,
        floor_band_fraction=0.25,
        cap_band_fraction=0.75,
    )

    clear = opening["conservative_clear_opening"]
    assert opening["open_collision_cavity_passed"] is True
    assert opening["center_connected_floor_rectangle_cell_count"] == 6
    assert clear["world_xy_min_m"] == pytest.approx([0.5, 0.25])
    assert clear["world_xy_max_m"] == pytest.approx([0.75, 0.75])
    assert clear["size_xy_m"] == pytest.approx([0.25, 0.5])
    assert clear["boundary_clearances_m"] == pytest.approx(
        {"x_min": 0.5, "x_max": 0.25, "y_min": 0.25, "y_max": 0.25}
    )
    assert opening["side_wall_probe"]["all_four_sides_passed"] is True
    assert opening["overhead_clearance_probe"]["clear_of_above_floor_projected_geometry"] is True


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


def test_opening_probe_grid_is_bounded() -> None:
    with pytest.raises(SageCollisionComponentTopologyError) as caught:
        _opening_probe(
            vertices=[],
            faces=[],
            bounds_min=[0.0, 0.0, 0.0],
            bounds_max=[1.0, 1.0, 1.0],
            grid_size=103,
            margin_fraction=0.1,
            floor_band_fraction=0.25,
            cap_band_fraction=0.75,
        )
    assert caught.value.errors == ("opening_probe_grid_size_must_be_odd_between_three_and_maximum",)


def test_replays_exact_component_in_bottom_center_local_frame(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path)
    topology = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=["bin"],
        opening_probe_instance_ids=["bin"],
        sage_collision_usd_path=collision,
    )
    best = topology["targets"][0]["best_component"]

    component = read_sage_collision_component_geometry(
        sage_collision_usd_path=collision,
        expected_source_sha256=topology["source_files"]["sage_collision_usd"]["sha256"],
        expected_source_size_bytes=topology["source_files"]["sage_collision_usd"]["size_bytes"],
        prim_path=best["prim_path"],
        component_index=best["component_index"],
        expected_geometry_digest=best["geometry_digest"],
    )

    assert component["source"]["geometry_digest"] == best["geometry_digest"]
    assert component["source"]["collision_api_applied"] is True
    assert component["world_aabb_size_m"] == pytest.approx([1.0, 1.0, 1.0])
    assert component["coordinate_frame"]["local_origin_world_m"] == pytest.approx([10.5, 0.5, 0.0])
    local_minimum = [
        min(point[axis] for point in component["vertices_local_m"]) for axis in range(3)
    ]
    local_maximum = [
        max(point[axis] for point in component["vertices_local_m"]) for axis in range(3)
    ]
    assert local_minimum == pytest.approx([-0.5, -0.5, 0.0])
    assert local_maximum == pytest.approx([0.5, 0.5, 1.0])
    assert component["vertex_count"] > 0
    assert component["face_count"] > 0
    assert component["receipt_digest"].startswith("sha256:")


def test_component_replay_rejects_source_drift_and_symlink(tmp_path: Path) -> None:
    labels, collision = _fixture(tmp_path)
    topology = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=["bin"],
        sage_collision_usd_path=collision,
    )
    best = topology["targets"][0]["best_component"]

    with pytest.raises(SageCollisionComponentTopologyError) as invalid_prim:
        read_sage_collision_component_geometry(
            sage_collision_usd_path=collision,
            expected_source_sha256=topology["source_files"]["sage_collision_usd"]["sha256"],
            expected_source_size_bytes=topology["source_files"]["sage_collision_usd"]["size_bytes"],
            prim_path=7,  # type: ignore[arg-type]
            component_index=best["component_index"],
            expected_geometry_digest=best["geometry_digest"],
        )
    assert invalid_prim.value.errors == ("sage_collision_component_prim_path_invalid",)

    with pytest.raises(SageCollisionComponentTopologyError) as drifted:
        read_sage_collision_component_geometry(
            sage_collision_usd_path=collision,
            expected_source_sha256=topology["source_files"]["sage_collision_usd"]["sha256"],
            expected_source_size_bytes=topology["source_files"]["sage_collision_usd"]["size_bytes"],
            prim_path=best["prim_path"],
            component_index=best["component_index"],
            expected_geometry_digest="sha256:" + "0" * 64,
        )
    assert drifted.value.errors == ("sage_collision_component_geometry_digest_mismatch",)

    with pytest.raises(SageCollisionComponentTopologyError) as source_drifted:
        read_sage_collision_component_geometry(
            sage_collision_usd_path=collision,
            expected_source_sha256="sha256:" + "0" * 64,
            expected_source_size_bytes=topology["source_files"]["sage_collision_usd"]["size_bytes"],
            prim_path=best["prim_path"],
            component_index=best["component_index"],
            expected_geometry_digest=best["geometry_digest"],
        )
    assert source_drifted.value.errors == ("sage_collision_source_identity_mismatch",)

    with pytest.raises(SageCollisionComponentTopologyError) as malformed:
        read_sage_collision_component_geometry(
            sage_collision_usd_path=collision,
            expected_source_sha256=topology["source_files"]["sage_collision_usd"]["sha256"],
            expected_source_size_bytes=topology["source_files"]["sage_collision_usd"]["size_bytes"],
            prim_path=best["prim_path"],
            component_index=best["component_index"],
            expected_geometry_digest="sha256:" + "z" * 64,
        )
    assert malformed.value.errors == ("sage_collision_expected_geometry_digest_invalid",)

    linked = tmp_path / "linked.usda"
    linked.symlink_to(collision)
    with pytest.raises(SageCollisionComponentTopologyError) as symlinked:
        read_sage_collision_component_geometry(
            sage_collision_usd_path=linked,
            expected_source_sha256=topology["source_files"]["sage_collision_usd"]["sha256"],
            expected_source_size_bytes=topology["source_files"]["sage_collision_usd"]["size_bytes"],
            prim_path=best["prim_path"],
            component_index=best["component_index"],
            expected_geometry_digest=best["geometry_digest"],
        )
    assert symlinked.value.errors == ("sage_collision_usd_missing",)
