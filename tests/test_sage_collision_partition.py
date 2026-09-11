from __future__ import annotations

import json

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sage_collision_identity import inspect_sage_collision_identity
from blueprint_pipeline.sage_collision_partition import (
    _record, materialize_sage_collision_partition, validate_partition,
)
from tests.test_sage_collision_identity import _box, _mesh


def combined_scene(tmp_path):
    labels = tmp_path / "labels.json"
    boxes = {"subject": ((.20, .20, .75), (.27, .26, .89)),
             "support": ((0., 0., 0.), (1.6, .8, .75)),
             "neighbor": ((.7, .3, .75), (.8, .4, .95))}
    labels.write_text(json.dumps([
        {"ins_id": name, "label": name, "bounding_box": _box(*lo, *hi)}
        for name, (lo, hi) in boxes.items()]))
    source = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    points, faces, counts = [], [], []
    for name, (lo, hi) in boxes.items():
        path = "/temporary_" + name
        _mesh(stage, path, lo, hi)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        faces.extend(int(i) + len(points) for i in mesh.GetFaceVertexIndicesAttr().Get())
        counts.extend(mesh.GetFaceVertexCountsAttr().Get())
        points.extend(mesh.GetPointsAttr().Get())
        stage.RemovePrim(path)
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Combined")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
    stage.GetRootLayer().Save()
    return source, labels


def test_grouped_source_keeps_every_face_and_separates_subject_from_table(tmp_path):
    source, labels = combined_scene(tmp_path)
    original = source.read_bytes()
    before = inspect_sage_collision_identity(labels_path=labels, target_instance_id="subject",
                                             sage_collision_usd_path=source)
    assert before["whole_object_collision_identity_passed"] is False
    result = materialize_sage_collision_partition(source_path=source, labels_path=labels,
        instance_ids=["subject", "support"], output_root=tmp_path / "partition")
    assert result["native_collision_cooking_qualified"] is False
    assert source.read_bytes() == original
    assert len(result["face_partitions"]) == 3
    assert sorted(i for row in result["face_partitions"] for i in row["source_face_indices"]) == list(range(18))
    target_paths = []
    for oid in ("subject", "support"):
        identity = inspect_sage_collision_identity(labels_path=labels, target_instance_id=oid,
            sage_collision_usd_path=result["output"]["path"])
        assert identity["whole_object_collision_identity_passed"] is True
        target_paths.append(identity["whole_object_matches"][0]["prim_path"])
    assert len(set(target_paths)) == 2
    assert materialize_sage_collision_partition(source_path=source, labels_path=labels,
        instance_ids=["subject", "support"], output_root=tmp_path / "partition") == result


def test_two_labels_cannot_claim_the_same_source_faces(tmp_path):
    source, labels = combined_scene(tmp_path)
    rows = json.loads(labels.read_text())
    rows[1]["bounding_box"] = rows[0]["bounding_box"]
    labels.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="selected_objects_share_faces"):
        materialize_sage_collision_partition(source_path=source, labels_path=labels,
            instance_ids=["subject", "support"], output_root=tmp_path / "partition")
    assert not (tmp_path / "partition").exists()


def test_resealed_changed_output_is_rejected_by_source_geometry_readback(tmp_path):
    source, labels = combined_scene(tmp_path)
    result = materialize_sage_collision_partition(source_path=source, labels_path=labels,
        instance_ids=["subject", "support"], output_root=tmp_path / "partition")
    output = tmp_path / "partition/partitioned_collision.usd"
    stage = Usd.Stage.Open(str(output))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(result["face_partitions"][0]["output_prim"]))
    points = mesh.GetPointsAttr().Get()
    points[0] = points[0] + (0.01, 0., 0.)
    mesh.GetPointsAttr().Set(points)
    stage.GetRootLayer().Save()
    result["output"] = _record(output)
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    with pytest.raises(ValueError, match="source_face_geometry_changed"):
        validate_partition(result, source_path=source, labels_path=labels, output_path=output)


def test_external_source_layer_is_not_composed(tmp_path):
    source, labels = combined_scene(tmp_path)
    source.write_text('#usda 1.0\n( subLayers = [@outside.usda@] )\n')
    with pytest.raises(ValueError, match="external_dependency_forbidden"):
        materialize_sage_collision_partition(source_path=source, labels_path=labels,
            instance_ids=["subject", "support"], output_root=tmp_path / "partition")
