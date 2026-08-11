from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.native_task_gpu_collision_qualification import (
    audit_native_task_gpu_collisions,
    author_native_task_gpu_qualified_collisions,
    NativeTaskGpuCollisionQualificationError,
    materialize_native_task_gpu_collision_static_qualification,
    validate_native_task_gpu_collision_static_qualification,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> None:
    path.write_text(
        '''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset"
{
    def Xform "dynamic" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Mesh "missing_approximation" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            point3f[] points = [(-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)]
        }
        def Mesh "oblong" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(-1, -0.004, -0.004), (1, 0.004, 0.004)]
        }
        def Mesh "regular" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
            point3f[] points = [(-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)]
        }
    }
    def Mesh "static" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        point3f[] points = [(-2, -2, 0), (2, 2, 0.1)]
    }
}
''',
        encoding="utf-8",
    )


def test_audit_reports_all_gpu_collision_blockers_in_one_pass(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _asset(source)

    audit = audit_native_task_gpu_collisions(source)

    assert audit["status"] == "blocked"
    assert audit["blockers"] == [
        "native_task_dynamic_convex_hull_gpu_oblong:/Asset/dynamic/oblong",
        (
            "native_task_dynamic_mesh_approximation_unsupported:"
            "/Asset/dynamic/missing_approximation"
        ),
    ]
    assert len(audit["dynamic_mesh_colliders"]) == 3


def test_authoring_preserves_source_and_emits_explicit_gpu_safe_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    destination = tmp_path / "qualified.usda"
    receipt = tmp_path / "receipt.json"
    _asset(source)
    source_digest = _digest(source)

    result = author_native_task_gpu_qualified_collisions(
        source_usd_path=source,
        destination_usd_path=destination,
        receipt_path=receipt,
    )

    assert _digest(source) == source_digest
    assert result["status"] == "authored_and_locally_qualified"
    assert result["after_audit"]["status"] == "qualified"
    assert result["changes"] == [
        {
            "prim_path": "/Asset/dynamic/missing_approximation",
            "old_approximation": "",
            "new_approximation": "convexHull",
        },
        {
            "prim_path": "/Asset/dynamic/oblong",
            "old_approximation": "convexHull",
            "new_approximation": "boundingCube",
        },
    ]
    stage = Usd.Stage.Open(str(destination))
    assert (
        UsdPhysics.MeshCollisionAPI(
            stage.GetPrimAtPath("/Asset/dynamic/oblong")
        ).GetApproximationAttr().Get()
        == "boundingCube"
    )
    assert receipt.is_file()


def test_audit_qualifies_exact_dynamic_primitive_collision_shapes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "primitive.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/Asset/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    cube = UsdGeom.Cube.Define(stage, "/Asset/body/collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(cube)
    stage.GetRootLayer().Save()

    audit = audit_native_task_gpu_collisions(path)

    assert audit["status"] == "qualified"
    assert audit["dynamic_mesh_colliders"] == []
    assert audit["dynamic_primitive_colliders"] == [
        {
            "prim_path": "/Asset/body/collider",
            "rigid_body_prim_path": "/Asset/body",
            "primitive_type": "Cube",
            "blockers": [],
        }
    ]
    assert audit["dynamic_collision_prim_count"] == 1

    receipt_path = tmp_path / "primitive_collision_qualification.json"
    receipt = materialize_native_task_gpu_collision_static_qualification(
        usd_path=path,
        destination=receipt_path,
    )
    assert receipt["status"] == "qualified"
    assert receipt["native_simulator_import_qualified"] is False
    assert validate_native_task_gpu_collision_static_qualification(receipt) == receipt
    assert receipt_path.is_file()

    receipt["native_simulator_import_qualified"] = True
    with pytest.raises(NativeTaskGpuCollisionQualificationError):
        validate_native_task_gpu_collision_static_qualification(receipt)


def test_checked_in_third_scene_collision_receipts_preserve_static_claim_ceiling() -> None:
    root = (
        Path(__file__).parents[1]
        / "docs/arm_decision_proof_v1/manifests"
    )
    rows = [
        validate_native_task_gpu_collision_static_qualification(
            json.loads((root / name).read_text(encoding="utf-8"))
        )
        for name in (
            "third_scene_840920_task_a_gpu_collision_static_qualification.v1.json",
            "third_scene_840920_task_b_gpu_collision_static_qualification.v1.json",
        )
    ]

    assert [row["source_usd_sha256"] for row in rows] == [
        "sha256:4e5caad7fb233f0560ae91d84b1911f9c27eeb1ad85201236f41c0e9c9f2c9ad",
        "sha256:821a28dc468cdc3f1b38d92bc7f0e158a63fc1432a54942b2144f5cb0416b7b4",
    ]
    assert [row["dynamic_collision_prim_count"] for row in rows] == [10, 2]
    assert all(row["status"] == "qualified" for row in rows)
    assert all(row["native_simulator_import_qualified"] is False for row in rows)
