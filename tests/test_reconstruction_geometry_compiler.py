from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_geometry_compiler import (
    COMPILER_SCHEMA,
    ReconstructionGeometryCompilerError,
    compile_collision_candidate,
    compile_metric_geometry,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import ToolRegistry, non_spend_tool_bindings


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
    assert candidate["collider_asset_digest"] == first["geometry_asset_digest"]
    assert candidate["component_statistics"] == {
        "count": 1,
        "disconnected_count": 0,
        "face_count": 2,
        "vertex_count": 4,
    }
    assert candidate["hole_statistics"]["boundary_edge_count"] == 4
    assert candidate["hole_statistics"]["count"] == 1


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
