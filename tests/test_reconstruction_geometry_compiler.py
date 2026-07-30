from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.nurec_openusd_packaging import package_nurec_openusd
from blueprint_pipeline.reconstruction_appearance_asset import (
    build_appearance_asset_manifest,
)
from blueprint_pipeline.reconstruction_geometry_compiler import (
    COMPILER_SCHEMA,
    QUALIFICATION_MEASUREMENT_SCHEMA,
    QUALIFICATION_REQUEST_SCHEMA,
    ReconstructionGeometryCompilerError,
    compile_collision_candidate,
    compile_metric_geometry,
    qualify_collision_candidate,
)
from blueprint_pipeline.reconstruction_geometry_contracts import (
    build_nurec_openusd_packaging_request,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import ToolRegistry, non_spend_tool_bindings


RECORDED_ARKITSCENES_SURFACE = (
    Path(__file__).parents[1]
    / "docs/evidence/arkitscenes_observed_surface_40958756_2ad2b7df.json"
)


def _digest_bytes(value: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _surface() -> dict:
    frame = {"frame": "blueprint_site", "units": "meters", "up_axis": "Z"}
    return {
        "schema_version": "observed_surface_mesh.v1",
        "coordinate_frame_declaration": frame,
        "vertices": [
            {
                "vertex_id": "v0",
                "position_m": [0.0, 0.0, 0.0],
                "confidence": 0.95,
                "region_id": "floor",
                "source_observation_ids": ["depth-0"],
                "generated": False,
            },
            {
                "vertex_id": "v1",
                "position_m": [1.0, 0.0, 0.0],
                "confidence": 0.95,
                "region_id": "floor",
                "source_observation_ids": ["depth-0"],
                "generated": False,
            },
            {
                "vertex_id": "v2",
                "position_m": [1.0, 1.0, 0.0],
                "confidence": 0.9,
                "region_id": "floor",
                "source_observation_ids": ["depth-1"],
                "generated": False,
            },
            {
                "vertex_id": "v3",
                "position_m": [0.0, 1.0, 0.0],
                "confidence": 0.9,
                "region_id": "floor",
                "source_observation_ids": ["depth-1"],
                "generated": False,
            },
            {
                "vertex_id": "v4",
                "position_m": [0.0, 0.0, 1.0],
                "confidence": 0.2,
                "region_id": "wall",
                "source_observation_ids": ["depth-2"],
                "generated": False,
            },
        ],
        "faces": [
            {
                "face_id": "f0",
                "vertex_ids": ["v0", "v1", "v2"],
                "region_id": "floor",
                "observed": True,
                "generated": False,
            },
            {
                "face_id": "f1",
                "vertex_ids": ["v0", "v2", "v3"],
                "region_id": "floor",
                "observed": True,
                "generated": False,
            },
            {
                "face_id": "f2",
                "vertex_ids": ["v0", "v1", "v4"],
                "region_id": "wall",
                "observed": True,
                "generated": False,
            },
        ],
    }


def _write_source(root: Path) -> tuple[Path, str]:
    path = root / "inputs" / "surface.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_surface(), sort_keys=True), encoding="utf-8")
    return path, _digest_bytes(path.read_bytes())


def _request(root: Path, **updates: object) -> dict:
    _, source_digest = _write_source(root)
    scale = {"status": "validated", "scale_error_fraction": 0.01}
    scale["metric_scale_validation_result_digest"] = canonical_digest(
        scale, digest_field="metric_scale_validation_result_digest"
    )
    value = {
        "schema_version": COMPILER_SCHEMA,
        "stable_run_identity": "geometry-run-1",
        "source_capture_identity": "capture-1",
        "source_capture_digest": "sha256:" + "1" * 64,
        "original_file_references": [
            {"artifact_id": "observed_surface", "digest": source_digest}
        ],
        "source_commit_sha": "a" * 40,
        "deterministic_configuration_digest": "sha256:" + "2" * 64,
        "train_heldout_split_digest": "sha256:" + "3" * 64,
        "camera_calibration_binding": {"digest": "sha256:" + "4" * 64},
        "coordinate_frame_declaration": {
            "frame": "blueprint_site",
            "units": "meters",
            "up_axis": "Z",
        },
        "authority_used": {"mode": "execute_non_spend"},
        "timestamp": "2026-07-30T22:00:00Z",
        "source_asset": {"relative_path": "inputs/surface.json", "digest": source_digest},
        "metric_scale_status": "validated",
        "metric_scale_validation": scale,
        "minimum_confidence": 0.8,
        "declared_region_ids": ["floor", "wall", "behind-cabinet"],
        "unsupported_region_ids": ["behind-cabinet"],
        "generated_fill_used": False,
        "appearance_asset_used_as_geometry_truth": False,
        "warnings": [],
        "blockers": [],
    }
    value.update(updates)
    value["source_artifact_digest"] = canonical_digest(
        value, digest_field="source_artifact_digest"
    )
    return value


def _qualification_request(
    root: Path, candidate: dict, *, measurement_updates: dict | None = None
) -> dict:
    evaluator = {
        "method_id": "blueprint.hermetic_independent_collider_evaluator",
        "method_version": "1.0.0",
        "candidate_method_independent": True,
    }
    measurements = {
        "scale_error_fraction": 0.01,
        "gravity_alignment_error_deg": 1.0,
        "floor_height_residual_m": 0.01,
        "wall_offset_residual_m": 0.02,
        "visual_to_collider_disagreement_m": 0.02,
        "clearance_error_m": 0.03,
        "mesh_coverage_fraction": 0.95,
        "minimum_obstacle_thickness_m": 0.04,
    }
    measurement = {
        "schema_version": QUALIFICATION_MEASUREMENT_SCHEMA,
        "collider_candidate_manifest_digest": candidate[
            "collider_candidate_manifest_digest"
        ],
        "collider_asset_digest": candidate["collider_asset_digest"],
        "evaluator": evaluator,
        "measurements": measurements,
        "evaluated_task_region_ids": ["floor"],
        "robot_footprint_navigability_checked": True,
        "candidate_self_graded": False,
        "thresholds_modified_after_measurement": False,
        "generated_geometry_promoted_to_collision_truth": False,
        "blockers": [],
    }
    if measurement_updates:
        measurement.update(measurement_updates)
    measurement_path = root / "inputs" / "collider_measurements.json"
    measurement_path.write_text(json.dumps(measurement, sort_keys=True), encoding="utf-8")
    measurement_digest = _digest_bytes(measurement_path.read_bytes())
    thresholds = {
        "scale_error_fraction": 0.03,
        "gravity_alignment_error_deg": 3.0,
        "floor_height_residual_m": 0.03,
        "wall_offset_residual_m": 0.05,
        "visual_to_collider_disagreement_m": 0.05,
        "clearance_error_m": 0.05,
        "mesh_coverage_fraction": 0.9,
        "minimum_obstacle_thickness_m": 0.03,
    }
    request = {
        "schema_version": QUALIFICATION_REQUEST_SCHEMA,
        "stable_run_identity": "collider-qualification-run-1",
        "source_capture_identity": candidate["source_capture_identity"],
        "source_capture_digest": candidate["source_capture_digest"],
        "original_file_references": [
            *candidate["original_file_references"],
            {"artifact_id": "collider_measurements", "digest": measurement_digest},
        ],
        "source_commit_sha": candidate["source_commit_sha"],
        "deterministic_configuration_digest": "sha256:" + "7" * 64,
        "train_heldout_split_digest": candidate["train_heldout_split_digest"],
        "camera_calibration_binding": candidate["camera_calibration_binding"],
        "coordinate_frame_declaration": candidate["coordinate_frame_declaration"],
        "authority_used": {"mode": "execute_non_spend"},
        "timestamp": "2026-07-30T22:30:00Z",
        "collider_candidate_manifest_digest": candidate[
            "collider_candidate_manifest_digest"
        ],
        "metric_scale_status": candidate["metric_scale_status"],
        "measurement_artifact": {
            "relative_path": "inputs/collider_measurements.json",
            "digest": measurement_digest,
        },
        "thresholds": thresholds,
        "qa_thresholds_digest": canonical_digest(
            thresholds, digest_field="qa_thresholds_digest"
        ),
        "independent_evaluator": evaluator,
        "task_region_ids": ["floor"],
        "warnings": [],
        "blockers": [],
    }
    request["collider_qualification_request_digest"] = canonical_digest(
        request, digest_field="collider_qualification_request_digest"
    )
    return request


def test_compiler_filters_low_confidence_observed_surface_without_fill(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manifest = compile_metric_geometry(
        source_artifact=request,
        output_root=tmp_path / "generated" / "metric",
        artifact_root=tmp_path,
    )

    assert manifest["metric_scale_status"] == "validated"
    assert manifest["observed_region_ids"] == ["floor"]
    assert manifest["unsupported_region_ids"] == ["behind-cabinet", "wall"]
    assert manifest["generated_fill_used"] is False
    assert manifest["topology"] == {
        "vertex_count": 4,
        "triangle_count": 2,
        "holes_closed": 0,
        "unseen_surfaces_created": 0,
    }
    confidence = manifest["confidence_filter"]
    assert confidence["rejected_vertex_count"] == 1
    assert confidence["rejected_face_count"] == 1
    assert confidence["rejected_faces"] == [
        {"face_id": "f2", "reason": "low_confidence_support"}
    ]
    emitted = tmp_path / manifest["geometry_asset_reference"]
    assert emitted.is_file()
    assert _digest_bytes(emitted.read_bytes()) == manifest["geometry_asset_digest"]


def test_recorded_arkitscenes_surface_cannot_bypass_coordinate_qualification(
    tmp_path: Path,
) -> None:
    receipt = json.loads(RECORDED_ARKITSCENES_SURFACE.read_text(encoding="utf-8"))
    request = _request(
        tmp_path,
        coordinate_frame_declaration=receipt["coordinate_frame_declaration"],
        metric_scale_status=receipt["surface_measurements"]["metric_scale_status"],
        metric_scale_validation=None,
    )

    with pytest.raises(
        ReconstructionGeometryCompilerError,
        match="metric_geometry_coordinate_frame_unqualified",
    ):
        compile_metric_geometry(
            source_artifact=request,
            output_root=tmp_path / "generated" / "metric",
            artifact_root=tmp_path,
        )


def test_metric_compilation_and_collider_baseline_replay_exactly(tmp_path: Path) -> None:
    request = _request(tmp_path)
    first = compile_metric_geometry(
        source_artifact=request,
        output_root=tmp_path / "generated" / "metric",
        artifact_root=tmp_path,
    )
    replay = compile_metric_geometry(
        source_artifact=request,
        output_root=tmp_path / "generated" / "metric",
        artifact_root=tmp_path,
    )
    assert replay == first

    candidate = compile_collision_candidate(
        source_artifact=first,
        output_root=tmp_path / "generated" / "collider",
        artifact_root=tmp_path,
    )
    candidate_replay = compile_collision_candidate(
        source_artifact=first,
        output_root=tmp_path / "generated" / "collider",
        artifact_root=tmp_path,
    )
    assert candidate_replay == candidate
    assert candidate["collision_validated"] is False
    assert candidate["unobserved_regions_filled"] is False
    assert candidate["candidate_operation"] == (
        "exact_observed_surface_copy_no_decimation_no_hole_fill"
    )
    assert candidate["observed_surface_asset_digest"] == first["geometry_asset_digest"]
    assert candidate["collider_asset_digest"] != first["geometry_asset_digest"]
    assert candidate["collider_asset_reference"].endswith(".usda")
    assert candidate["collider_source_prim_path"] == "/World/Collision"
    assert candidate["collision_api_configured"] is True
    assert candidate["component_statistics"] == {
        "count": 1,
        "disconnected_count": 0,
        "face_count": 2,
        "vertex_count": 4,
    }
    assert candidate["hole_statistics"]["boundary_edge_count"] == 4
    assert candidate["hole_statistics"]["count"] == 1


def test_independent_collider_qualification_accepts_only_frozen_measured_thresholds(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    metric = compile_metric_geometry(
        source_artifact=request,
        output_root=tmp_path / "generated" / "metric",
        artifact_root=tmp_path,
    )
    candidate = compile_collision_candidate(
        source_artifact=metric,
        output_root=tmp_path / "generated" / "collider",
        artifact_root=tmp_path,
    )
    qualification_request = _qualification_request(tmp_path, candidate)
    report = qualify_collision_candidate(
        source_artifact=candidate,
        output_root=tmp_path / "generated" / "qualification",
        artifact_root=tmp_path,
        qualification_request=qualification_request,
    )
    assert report["decision"] == "accepted_bounded_navigation"
    assert report["candidate_self_graded"] is False
    assert report["failed_threshold_ids"] == []
    assert set(report["unsupported_claims"]) >= {
        "grasping",
        "articulation",
        "contact_force",
        "deployment",
        "physical_success",
    }
    replay = qualify_collision_candidate(
        source_artifact=candidate,
        output_root=tmp_path / "generated" / "qualification",
        artifact_root=tmp_path,
        qualification_request=qualification_request,
    )
    assert replay == report


def test_collider_qualification_request_and_measurement_match_versioned_schemas(
    tmp_path: Path,
) -> None:
    metric = compile_metric_geometry(
        source_artifact=_request(tmp_path),
        output_root=tmp_path / "generated" / "metric",
        artifact_root=tmp_path,
    )
    candidate = compile_collision_candidate(
        source_artifact=metric,
        output_root=tmp_path / "generated" / "collider",
        artifact_root=tmp_path,
    )
    request = _qualification_request(tmp_path, candidate)
    schema_root = Path(__file__).parents[1] / "docs" / "schemas"
    request_schema = json.loads(
        (schema_root / "collider_qualification_request.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    measurement_schema = json.loads(
        (schema_root / "collider_qualification_measurements.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    measurement = json.loads(
        (tmp_path / request["measurement_artifact"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(request_schema).validate(request)
    Draft202012Validator(measurement_schema).validate(measurement)


def test_independent_collider_qualification_rejects_bad_clearance_and_self_grading(
    tmp_path: Path,
) -> None:
    metric = compile_metric_geometry(
        source_artifact=_request(tmp_path),
        output_root=tmp_path / "generated" / "metric",
        artifact_root=tmp_path,
    )
    candidate = compile_collision_candidate(
        source_artifact=metric,
        output_root=tmp_path / "generated" / "collider",
        artifact_root=tmp_path,
    )
    request = _qualification_request(
        tmp_path,
        candidate,
        measurement_updates={
            "measurements": {
                "scale_error_fraction": 0.01,
                "gravity_alignment_error_deg": 1.0,
                "floor_height_residual_m": 0.01,
                "wall_offset_residual_m": 0.02,
                "visual_to_collider_disagreement_m": 0.02,
                "clearance_error_m": 0.5,
                "mesh_coverage_fraction": 0.95,
                "minimum_obstacle_thickness_m": 0.04,
            }
        },
    )
    report = qualify_collision_candidate(
        source_artifact=candidate,
        output_root=tmp_path / "generated" / "qualification",
        artifact_root=tmp_path,
        qualification_request=request,
    )
    assert report["decision"] == "rejected"
    assert report["failed_threshold_ids"] == ["clearance_error_m"]

    request = _qualification_request(
        tmp_path, candidate, measurement_updates={"candidate_self_graded": True}
    )
    with pytest.raises(
        ReconstructionGeometryCompilerError, match="measurement_independence_invalid"
    ):
        qualify_collision_candidate(
            source_artifact=candidate,
            output_root=tmp_path / "generated" / "qualification",
            artifact_root=tmp_path,
            qualification_request=request,
        )


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda request: request["source_asset"].update(relative_path="../surface.json"), "source_asset_relative_path_unsafe"),
        (lambda request: request["source_asset"].update(digest="sha256:" + "f" * 64), "source_asset_digest_mismatch"),
        (lambda request: request.update(original_file_references=[]), "source_asset_provenance_binding_missing"),
        (lambda request: request.update(generated_fill_used=True), "generated_or_unseen_fill_forbidden"),
        (lambda request: request.update(appearance_asset_used_as_geometry_truth=True), "appearance_cannot_be_geometry_truth"),
    ],
)
def test_compiler_fails_closed_on_unsafe_or_promoted_sources(
    tmp_path: Path, mutation, expected: str
) -> None:
    request = _request(tmp_path)
    mutation(request)
    request["source_artifact_digest"] = canonical_digest(
        request, digest_field="source_artifact_digest"
    )
    with pytest.raises(ReconstructionGeometryCompilerError, match=expected):
        compile_metric_geometry(
            source_artifact=request,
            output_root=tmp_path / "generated" / "metric",
            artifact_root=tmp_path,
        )


def test_compiler_rejects_symlink_escape_and_unverified_validated_scale(tmp_path: Path) -> None:
    request = _request(tmp_path)
    source = tmp_path / "inputs" / "surface.json"
    outside = tmp_path / "outside.json"
    source.rename(outside)
    source.symlink_to(outside)
    request["source_asset"]["digest"] = _digest_bytes(outside.read_bytes())
    request["source_artifact_digest"] = canonical_digest(
        request, digest_field="source_artifact_digest"
    )
    with pytest.raises(ReconstructionGeometryCompilerError, match="source_asset_symlink_forbidden"):
        compile_metric_geometry(
            source_artifact=request,
            output_root=tmp_path / "generated" / "metric",
            artifact_root=tmp_path,
        )

    source.unlink()
    outside.rename(source)
    request = _request(tmp_path, metric_scale_validation={"status": "validated"})
    with pytest.raises(ReconstructionGeometryCompilerError, match="metric_scale_validation_invalid"):
        compile_metric_geometry(
            source_artifact=request,
            output_root=tmp_path / "generated" / "metric",
            artifact_root=tmp_path,
        )


def test_generated_source_face_is_rejected_instead_of_promoted(tmp_path: Path) -> None:
    path, digest = _write_source(tmp_path)
    surface = json.loads(path.read_text(encoding="utf-8"))
    surface["faces"][0]["generated"] = True
    path.write_text(json.dumps(surface, sort_keys=True), encoding="utf-8")
    digest = _digest_bytes(path.read_bytes())
    request = _request(tmp_path)
    # _request rewrites the clean fixture; replace it after request construction.
    path.write_text(json.dumps(surface, sort_keys=True), encoding="utf-8")
    request["source_asset"]["digest"] = digest
    request["original_file_references"][0]["digest"] = digest
    request["source_artifact_digest"] = canonical_digest(
        request, digest_field="source_artifact_digest"
    )
    with pytest.raises(ReconstructionGeometryCompilerError, match="observed_surface_face_invalid:f0"):
        compile_metric_geometry(
            source_artifact=request,
            output_root=tmp_path / "generated" / "metric",
            artifact_root=tmp_path,
        )


def test_registered_tools_use_repository_owned_geometry_runtimes(tmp_path: Path) -> None:
    request = _request(tmp_path)
    registry = ToolRegistry.default()
    metric_context = SupervisorContext(
        run_id="repository-owned-metric-runtime",
        customer_question="Compile metric geometry",
        supervisor_output_dir=str(tmp_path),
        metric_geometry_source=request,
    )
    authority = default_authority_envelope(
        run_id=metric_context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[request["source_artifact_digest"]],
    ).to_mapping()
    metric_bindings = {
        item.tool_id: item
        for item in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=metric_context,
            registry=registry,
            authority=authority,
        )
    }
    observation = metric_bindings["compile_metric_geometry"].invoke(
        {"source_artifact_digest": request["source_artifact_digest"]}
    )
    assert observation["status"] == "completed"
    manifest_path = (
        tmp_path
        / "generated"
        / "compile_metric_geometry"
        / "metric_geometry_manifest.v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    collider_context = SupervisorContext(
        run_id="repository-owned-collider-runtime",
        customer_question="Compile collider candidate",
        supervisor_output_dir=str(tmp_path),
        metric_geometry_manifest=manifest,
    )
    collider_authority = default_authority_envelope(
        run_id=collider_context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[manifest["metric_geometry_manifest_digest"]],
    ).to_mapping()
    collider_bindings = {
        item.tool_id: item
        for item in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=collider_context,
            registry=registry,
            authority=collider_authority,
        )
    }
    collider_observation = collider_bindings["compile_collision_candidate"].invoke(
        {"metric_geometry_manifest_digest": manifest["metric_geometry_manifest_digest"]}
    )
    assert collider_observation["status"] == "completed"
    candidate_path = (
        tmp_path
        / "generated"
        / "compile_collision_candidate"
        / "mesh_collider_candidate_manifest.v1.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["collision_validated"] is False
    assert candidate["unobserved_regions_filled"] is False

    qualification_request = _qualification_request(tmp_path, candidate)
    qualifier_context = SupervisorContext(
        run_id="repository-owned-qualifier-runtime",
        customer_question="Qualify collider candidate",
        supervisor_output_dir=str(tmp_path),
        collider_candidate_manifest=candidate,
        collider_qualification_request=qualification_request,
    )
    qualifier_authority = default_authority_envelope(
        run_id=qualifier_context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[candidate["collider_candidate_manifest_digest"]],
    ).to_mapping()
    qualifier_bindings = {
        item.tool_id: item
        for item in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=qualifier_context,
            registry=registry,
            authority=qualifier_authority,
        )
    }
    qualifier_observation = qualifier_bindings["qualify_collision_candidate"].invoke(
        {
            "collider_candidate_manifest_digest": candidate[
                "collider_candidate_manifest_digest"
            ]
        }
    )
    assert qualifier_observation["status"] == "completed"
    report_path = (
        tmp_path
        / "generated"
        / "qualify_collision_candidate"
        / "collider_qualification_report.v1.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["decision"] == "accepted_bounded_navigation"
    assert report["candidate_self_graded"] is False


def test_compiler_rejects_duplicate_json_keys_and_output_escape(tmp_path: Path) -> None:
    request = _request(tmp_path)
    source = tmp_path / "inputs" / "surface.json"
    source.write_text(
        '{"schema_version":"observed_surface_mesh.v1",'
        '"schema_version":"observed_surface_mesh.v1"}',
        encoding="utf-8",
    )
    digest = _digest_bytes(source.read_bytes())
    request["source_asset"]["digest"] = digest
    request["original_file_references"][0]["digest"] = digest
    request["source_artifact_digest"] = canonical_digest(
        request, digest_field="source_artifact_digest"
    )
    with pytest.raises(ReconstructionGeometryCompilerError, match="duplicate_json_key"):
        compile_metric_geometry(
            source_artifact=request,
            output_root=tmp_path / "generated" / "metric",
            artifact_root=tmp_path,
        )

    request = _request(tmp_path)
    escaped = tmp_path.parent / f"{tmp_path.name}-escaped-output"
    with pytest.raises(ReconstructionGeometryCompilerError, match="output_root_outside"):
        compile_metric_geometry(
            source_artifact=request,
            output_root=escaped,
            artifact_root=tmp_path,
        )
    assert not escaped.exists()


def test_measured_geometry_to_qualified_collider_to_openusd_package(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    metric = compile_metric_geometry(
        source_artifact=_request(tmp_path),
        output_root=tmp_path / "generated" / "metric",
        artifact_root=tmp_path,
    )
    candidate = compile_collision_candidate(
        source_artifact=metric,
        output_root=tmp_path / "generated" / "collider",
        artifact_root=tmp_path,
    )
    qualification = qualify_collision_candidate(
        source_artifact=candidate,
        output_root=tmp_path / "generated" / "qualification",
        artifact_root=tmp_path,
        qualification_request=_qualification_request(tmp_path, candidate),
    )

    appearance = tmp_path / "inputs" / "appearance.usda"
    stage = Usd.Stage.CreateNew(str(appearance))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.DefinePrim("/World/Appearance", "ParticleField3DGaussianSplat")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    appearance_digest = _digest_bytes(appearance.read_bytes())
    appearance_manifest = build_appearance_asset_manifest(
        {
            "stable_run_identity": "geometry-to-package-run-1",
            "source_capture_identity": candidate["source_capture_identity"],
            "source_capture_digest": candidate["source_capture_digest"],
            "original_file_references": candidate["original_file_references"],
            "producing_method": "fixture_particlefield_compiler",
            "implementation_version": "1.0.0",
            "container_image_digest": None,
            "source_commit_sha": candidate["source_commit_sha"],
            "deterministic_configuration_digest": "sha256:" + "9" * 64,
            "input_digests": [
                {"artifact_id": "appearance_candidate.ply", "digest": "sha256:" + "7" * 64},
                {"artifact_id": "training_result", "digest": "sha256:" + "6" * 64},
            ],
            "output_digests": [
                {"artifact_id": "inputs/appearance.usda", "digest": appearance_digest}
            ],
            "train_heldout_split_digest": candidate["train_heldout_split_digest"],
            "camera_calibration_binding": candidate["camera_calibration_binding"],
            "coordinate_frame_declaration": candidate["coordinate_frame_declaration"],
            "units": "meters",
            "metric_scale_status": "validated",
            "provider_runtime_identity": {"provider": "local"},
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "authority_used": {"mode": "execute_non_spend"},
            "warnings": [],
            "blockers": [],
            "proof_effect": "appearance_asset_candidate_only",
            "claim_ceiling": "appearance_reconstruction",
            "parent_artifact_or_event": {"digest": "sha256:" + "6" * 64},
            "timestamp": "2026-07-30T23:00:00Z",
            "status": "completed",
            "reconstruction_training_request_digest": "sha256:" + "5" * 64,
            "reconstruction_training_result_digest": "sha256:" + "6" * 64,
            "source_appearance_asset_reference": "appearance_candidate.ply",
            "source_appearance_asset_digest": "sha256:" + "7" * 64,
            "source_asset_format": "standard_3dgs_ply",
            "appearance_asset_reference": "inputs/appearance.usda",
            "appearance_asset_digest": appearance_digest,
            "appearance_asset_format": "particlefield_usd",
            "source_prim_path": "/World/Appearance",
            "splat_count": 1,
            "sh_degree": 0,
            "captured_observation": False,
            "raw_evidence": False,
            "metric_geometry_proven": False,
            "collision_geometry_proven": False,
            "heldout_evaluated": False,
        }
    )
    appearance_manifest_digest = appearance_manifest["appearance_asset_manifest_digest"]
    collider_path = tmp_path / candidate["collider_asset_reference"]
    assert _digest_bytes(collider_path.read_bytes()) == candidate["collider_asset_digest"]

    packaging = {
        "stable_run_identity": "geometry-to-package-run-1",
        "source_capture_identity": candidate["source_capture_identity"],
        "source_capture_digest": candidate["source_capture_digest"],
        "original_file_references": [
            {"artifact_id": "appearance", "digest": appearance_digest},
            {"artifact_id": "collider", "digest": candidate["collider_asset_digest"]},
        ],
        "producing_method": "blueprint.hermetic_packaging_request_compiler",
        "implementation_version": "1.0.0",
        "source_commit_sha": candidate["source_commit_sha"],
        "deterministic_configuration_digest": "sha256:" + "8" * 64,
        "input_digests": [
            {"artifact_id": "appearance", "digest": appearance_digest},
            {
                "artifact_id": "appearance_manifest",
                "digest": appearance_manifest_digest,
            },
            {"artifact_id": "collider", "digest": candidate["collider_asset_digest"]},
        ],
        "output_digests": [],
        "train_heldout_split_digest": candidate["train_heldout_split_digest"],
        "camera_calibration_binding": candidate["camera_calibration_binding"],
        "coordinate_frame_declaration": candidate["coordinate_frame_declaration"],
        "units": "meters",
        "provider_runtime_identity": {"provider": "local", "runtime": "python"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {"mode": "execute_non_spend"},
        "warnings": [],
        "blockers": [],
        "parent_artifact_or_event": {
            "digest": qualification["collider_qualification_digest"]
        },
        "timestamp": "2026-07-30T23:00:00Z",
        "appearance_asset": {
            "relative_path": "inputs/appearance.usda",
            "digest": appearance_digest,
            "source_prim_path": "/World/Appearance",
            "manifest_digest": appearance_manifest_digest,
        },
        "appearance_asset_manifest_digest": appearance_manifest_digest,
        "appearance_asset_manifest": appearance_manifest,
        "metric_geometry_manifest_digest": metric["metric_geometry_manifest_digest"],
        "collider_asset": {
            "relative_path": candidate["collider_asset_reference"],
            "digest": candidate["collider_asset_digest"],
            "source_prim_path": candidate["collider_source_prim_path"],
        },
        "collider_candidate_manifest_digest": candidate[
            "collider_candidate_manifest_digest"
        ],
        "collider_qualification_digest": qualification["collider_qualification_digest"],
        "collider_qualification_decision": qualification["decision"],
        "stage_meters_per_unit": 1.0,
        "up_axis": "Z",
        "shared_visual_physics_frame": True,
        "target_prim_paths": {
            "appearance": "/World/BlueprintReconstruction/Appearance",
            "collision": "/World/BlueprintReconstruction/Collision",
        },
        "output_format": "usdz",
        "output_name": "measured_geometry_fixture.usdz",
        "proof_effect": "packaging_request_only",
        "claim_ceiling": "none",
    }
    request = build_nurec_openusd_packaging_request(packaging)
    result = package_nurec_openusd(
        source_artifact=request,
        artifact_root=tmp_path,
        output_root=tmp_path / "generated" / "package",
    )
    replay = package_nurec_openusd(
        source_artifact=request,
        artifact_root=tmp_path,
        output_root=tmp_path / "generated" / "package",
    )
    assert replay == result
    assert result["collider_qualification_decision"] == "accepted_bounded_navigation"
    assert result["collision_api_configured"] is True
    package = tmp_path / "generated" / "package" / result["package_artifact_reference"]
    packaged_stage = Usd.Stage.Open(str(package))
    collision = packaged_stage.GetPrimAtPath("/World/BlueprintReconstruction/Collision")
    assert any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(collision))
