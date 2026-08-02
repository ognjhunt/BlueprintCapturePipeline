from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.external_scene_collision_candidate import (
    build_external_scene_collision_request,
    compile_external_scene_collision_candidate,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_glb_compiles_static_candidate_without_source_video(tmp_path) -> None:
    source = tmp_path / "scene.glb"
    source.write_bytes(trimesh.creation.box(extents=[4.0, 2.0, 6.0]).export(file_type="glb"))
    request = build_external_scene_collision_request(
        {
            "schema_version": "external_scene_collision_compilation_request.v1",
            "source_asset_digest": _digest(source),
            "source_format": "glb",
            "source_coordinate_frame": {"up_axis": "Y", "handedness": "right"},
            "metric_scale_status": "provider_declared_not_independently_validated",
            "source_video_available": False,
            "generated_fill_allowed": False,
            "collision_validated": False,
        }
    )
    output = tmp_path / "collision.usda"
    result = compile_external_scene_collision_candidate(
        source_path=source,
        request=request,
        output_path=output,
    )

    assert result["status"] == "candidate_compiled"
    assert result["source_video_required_for_candidate_compilation"] is False
    assert result["collision_validated"] is False
    assert result["metric_scale_status"] == ("provider_declared_not_independently_validated")
    assert result["bounds_stage_units"]["extents"] == [4.0, 6.0, 2.0]
    assert result["blockers"] == ["independent_metric_scale_pending"]
    assert result["qualification_gaps"] == [
        "independent_metric_scale_missing",
        "collider_contact_qualification_pending",
    ]

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Z"
    prim = stage.GetPrimAtPath("/World/BlueprintReconstruction/Collision/ExternalSceneMesh")
    assert prim.IsValid()
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI) is False


def test_validated_scale_removes_scale_blocker_but_not_claim_ceiling(tmp_path) -> None:
    source = tmp_path / "scene.glb"
    source.write_bytes(trimesh.creation.box().export(file_type="glb"))
    request = build_external_scene_collision_request(
        {
            "schema_version": "external_scene_collision_compilation_request.v1",
            "source_asset_digest": _digest(source),
            "source_format": "glb",
            "source_coordinate_frame": {"up_axis": "Z", "handedness": "right"},
            "metric_scale_status": "validated",
            "source_video_available": True,
            "generated_fill_allowed": False,
            "collision_validated": False,
        }
    )
    result = compile_external_scene_collision_candidate(
        source_path=source,
        request=request,
        output_path=tmp_path / "collision.usda",
    )

    assert result["blockers"] == []
    assert result["qualification_gaps"] == ["collider_contact_qualification_pending"]
    assert "metric_scale" not in result["unsupported_claims"]
    assert result["collision_validated"] is False
    assert result["claim_ceiling"] == "isaac_collision_candidate"
    request_schema = json.loads(
        (
            ROOT / "docs/schemas/external_scene_collision_compilation_request.v1.schema.json"
        ).read_text()
    )
    result_schema = json.loads(
        (ROOT / "docs/schemas/external_scene_collision_candidate.v1.schema.json").read_text()
    )
    jsonschema.validate(request, request_schema)
    jsonschema.validate(result, result_schema)


def test_cli_writes_digest_bound_request_result_and_usd(tmp_path) -> None:
    source = tmp_path / "scene.glb"
    source.write_bytes(trimesh.creation.box().export(file_type="glb"))
    request_path = tmp_path / "request-input.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "external_scene_collision_compilation_request.v1",
                "source_asset_digest": _digest(source),
                "source_format": "glb",
                "source_coordinate_frame": {"up_axis": "Y", "handedness": "right"},
                "metric_scale_status": "unverified",
                "source_video_available": False,
                "generated_fill_allowed": False,
                "collision_validated": False,
            }
        ),
        encoding="utf-8",
    )
    admitted_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    output = tmp_path / "collision.usda"

    assert (
        main(
            [
                "--source",
                str(source),
                "--request",
                str(request_path),
                "--output-usd",
                str(output),
                "--result-out",
                str(result_path),
                "--admitted-request-out",
                str(admitted_path),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert '"request_digest":"sha256:' in admitted_path.read_text(encoding="utf-8")
    assert '"collision_candidate_digest":"sha256:' in result_path.read_text(encoding="utf-8")
