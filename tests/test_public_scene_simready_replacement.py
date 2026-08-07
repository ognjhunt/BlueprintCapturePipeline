from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_simready_replacement import (
    PublicSceneSimReadyReplacementError,
    materialize_simready_replacement,
)


def _write(path: Path, value: dict[str, object], *, digest_field: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if digest_field:
        value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_usd(path: Path, *, uneven: bool = False) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    target = UsdGeom.Mesh.Define(stage, "/Root/Target")
    target.CreatePointsAttr(
        [Gf.Vec3f(-0.03, -0.03, 0.5), Gf.Vec3f(0.03, -0.03, 0.5), Gf.Vec3f(0, 0.03, 0.67)]
    )
    target.CreateFaceVertexCountsAttr([3])
    target.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(target.GetPrim())

    support = UsdGeom.Mesh.Define(stage, "/Root/Support")
    high = 0.53 if uneven else 0.502
    support.CreatePointsAttr(
        [
            Gf.Vec3f(-0.2, -0.2, 0.502),
            Gf.Vec3f(0.2, -0.2, 0.502),
            Gf.Vec3f(0.2, 0.2, high),
            Gf.Vec3f(-0.2, 0.2, 0.502),
        ]
    )
    support.CreateFaceVertexCountsAttr([3, 3])
    support.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    UsdPhysics.CollisionAPI.Apply(support.GetPrim())
    stage.GetRootLayer().Save()


def _replacement_usd(path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/canned_beverage")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
    collider = UsdGeom.Cube.Define(stage, "/canned_beverage/colliders/body_collider")
    collider.CreateSizeAttr(0.06)
    collider.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.03))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    stage.GetRootLayer().Save()


def _fixture(tmp_path: Path, *, uneven: bool = False) -> dict[str, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    collision = evidence / "scene" / "collision.usda"
    collision.parent.mkdir(parents=True)
    _source_usd(collision, uneven=uneven)
    asset = repo / "assets" / "can.usda"
    asset.parent.mkdir(parents=True)
    _replacement_usd(asset)
    final_ply = evidence / "aura" / "artifacts" / "final.ply"
    final_ply.parent.mkdir(parents=True)
    final_ply.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")

    lower = [-0.03, -0.03, 0.5]
    upper = [0.03, 0.03, 0.67]
    corners = [
        [x, y, z]
        for z in (lower[2], upper[2])
        for x, y in (
            (lower[0], lower[1]),
            (upper[0], lower[1]),
            (upper[0], upper[1]),
            (lower[0], upper[1]),
        )
    ]
    sage = {
        "role": "sage3d_collision_companion",
        "scene_mapping": {"publisher_scene_id": "840313"},
        "target_binding": {
            "interiorgs_instance_id": "160",
            "semantic_label": "canned_beverage",
            "collision_prim_path": "/Root/Target",
            "support_collision_prim_path": "/Root/Support",
            "separately_removable": True,
            "obb_aabb_min_m": lower,
            "obb_aabb_max_m": upper,
        },
        "materialized_artifacts": [
            {
                "role": "static_collision_geometry",
                "external_relative_path": "scene/collision.usda",
                "sha256": _sha(collision),
                "size_bytes": collision.stat().st_size,
            }
        ],
    }
    _write(repo / "sage.json", sage, digest_field="manifest_digest")
    simready = {
        "status": "statically_validated",
        "source_scene_id": "840313",
        "source_instance_id": "160",
        "geometry": {"local_origin": "center_of_base_datum"},
        "usd": {
            "relative_path": "assets/can.usda",
            "sha256": _sha(asset),
            "size_bytes": asset.stat().st_size,
        },
    }
    _write(repo / "simready.json", simready, digest_field="receipt_digest")
    edit = {
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "160",
            "target_obb_corners_m": corners,
        }
    }
    _write(repo / "edit.json", edit, digest_field="receipt_digest")
    aura = {
        "status": "executed_candidate",
        "scene": {"publisher_scene_id": "840313", "target_instance_id": "ins160"},
        "execution": {
            "final_point_cloud": {
                "relative_path": "artifacts/final.ply",
                "sha256": _sha(final_ply),
                "size_bytes": final_ply.stat().st_size,
            }
        },
    }
    _write(repo / "aura.json", aura, digest_field="receipt_digest")
    request = {
        "schema_version": "adp009b_simready_replacement_request.v1",
        "sage_component_manifest_path": "sage.json",
        "simready_receipt_path": "simready.json",
        "edit_input_receipt_path": "edit.json",
        "aura_execution_receipt_path": "aura.json",
        "aura_execution_root_relative_path": "aura",
        "support_probe": {
            "footprint_margin_m": 0.01,
            "maximum_flatness_error_m": 0.002,
            "maximum_tilt_degrees": 5.0,
            "minimum_overlapping_area_m2": 0.003,
            "maximum_support_correction_m": 0.01,
        },
    }
    _write(repo / "request.json", request)
    return {
        "repo": repo,
        "evidence": evidence,
        "request": repo / "request.json",
        "collision": collision,
        "output": evidence / "replacement" / "collision_and_replacement.usda",
        "receipt": repo / "replacement_receipt.json",
    }


def test_materializer_measures_support_and_composes_replacement(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = materialize_simready_replacement(
        request_path=paths["request"],
        repo_root=paths["repo"],
        evidence_root=paths["evidence"],
        output_usda=paths["output"],
        output_receipt=paths["receipt"],
    )

    assert receipt["status"] == "composed_static_candidate"
    assert receipt["support_surface_measurement"]["height_span_m"] == pytest.approx(0.0)
    assert receipt["support_surface_measurement"]["maximum_tilt_degrees"] == pytest.approx(0.0)
    assert receipt["placement"]["support_aligned_base_placement_m"] == pytest.approx(
        [0.0, 0.0, 0.502]
    )
    assert receipt["placement"]["support_alignment_correction_m"] == pytest.approx(
        0.002, abs=1e-7
    )
    assert receipt["composition"]["source_target_collider_active"] is False
    assert receipt["composition"]["sage_source_bytes_modified"] is False
    assert receipt["composition"]["default_prim_path"] == "/World"
    assert receipt["composition"]["composed_target_collision_prim_path"] == (
        "/World/Environment/Target"
    )
    assert Path(receipt["composition"]["sage_collision_copy"]["relative_path"]).name == (
        paths["collision"].name
    )
    assert receipt["nvidia_agent_routing"]["geometry_smoothing_required"] is False


def test_materializer_rejects_changed_source_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["collision"].write_text("changed", encoding="utf-8")
    with pytest.raises(
        PublicSceneSimReadyReplacementError, match="sage_collision_usd_digest_mismatch"
    ):
        materialize_simready_replacement(
            request_path=paths["request"],
            repo_root=paths["repo"],
            evidence_root=paths["evidence"],
            output_usda=paths["output"],
            output_receipt=paths["receipt"],
        )


def test_materializer_rejects_scene_mismatch_and_caller_admission(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    request["status"] = "admitted"
    _write(paths["request"], request)
    with pytest.raises(
        PublicSceneSimReadyReplacementError, match="caller_asserted_admission_forbidden"
    ):
        materialize_simready_replacement(
            request_path=paths["request"],
            repo_root=paths["repo"],
            evidence_root=paths["evidence"],
            output_usda=paths["output"],
            output_receipt=paths["receipt"],
        )

    paths = _fixture(tmp_path / "mismatch")
    aura_path = paths["repo"] / "aura.json"
    aura = json.loads(aura_path.read_text(encoding="utf-8"))
    aura["scene"]["publisher_scene_id"] = "other"
    aura.pop("receipt_digest")
    _write(aura_path, aura, digest_field="receipt_digest")
    with pytest.raises(PublicSceneSimReadyReplacementError, match="replacement_scene_id_mismatch"):
        materialize_simready_replacement(
            request_path=paths["request"],
            repo_root=paths["repo"],
            evidence_root=paths["evidence"],
            output_usda=paths["output"],
            output_receipt=paths["receipt"],
        )


def test_materializer_rejects_nonflat_support(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, uneven=True)
    with pytest.raises(
        PublicSceneSimReadyReplacementError,
        match="support_surface_not_flat|horizontal_support_triangles_missing",
    ):
        materialize_simready_replacement(
            request_path=paths["request"],
            repo_root=paths["repo"],
            evidence_root=paths["evidence"],
            output_usda=paths["output"],
            output_receipt=paths["receipt"],
        )
