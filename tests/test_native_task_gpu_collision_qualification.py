from __future__ import annotations

import hashlib
from pathlib import Path

from pxr import Usd, UsdPhysics

from blueprint_pipeline.native_task_gpu_collision_qualification import (
    audit_native_task_gpu_collisions,
    author_native_task_gpu_qualified_collisions,
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
