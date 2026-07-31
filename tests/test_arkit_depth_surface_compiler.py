from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from blueprint_pipeline.arkit_depth_surface_compiler import (
    REQUEST_SCHEMA,
    ArkitDepthSurfaceCompilerError,
    build_arkit_depth_surface_compilation_request,
    compile_arkit_depth_surface,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_geometry_compiler import compile_metric_geometry
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import (
    default_authority_envelope,
)
from blueprint_pipeline.task_evaluation_supervisor.tools import (
    ToolRegistry,
    non_spend_tool_bindings,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_digest(value: dict) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_inputs(
    root: Path,
    *,
    depth: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> tuple[Path, Path]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    depth_path = inputs / "depth.png"
    confidence_path = inputs / "confidence.png"
    depth_value = (
        np.full((3, 3), 1000, dtype=np.uint16)
        if depth is None
        else np.asarray(depth, dtype=np.uint16)
    )
    confidence_value = (
        np.full((3, 3), 2, dtype=np.uint8)
        if confidence is None
        else np.asarray(confidence, dtype=np.uint8)
    )
    Image.fromarray(depth_value).save(depth_path, format="PNG")
    Image.fromarray(confidence_value).save(confidence_path, format="PNG")
    return depth_path, confidence_path


def _request(
    root: Path,
    *,
    convention: str = "arkit_x_right_y_up_z_backward",
    split: str = "training",
    depth: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    **updates: object,
) -> dict:
    depth_path, confidence_path = _write_inputs(
        root, depth=depth, confidence=confidence
    )
    coordinate_frame = {
        "frame": "arkit_world",
        "units": "meters",
        "up_axis": "Z",
        "handedness": "right_handed",
    }
    value = {
        "schema_version": REQUEST_SCHEMA,
        "stable_run_identity": "arkit-depth-surface-run-1",
        "source_capture_identity": "capture-1",
        "source_capture_digest": "sha256:" + "1" * 64,
        "original_file_references": [
            {"artifact_id": "depth-frame-1", "digest": _digest(depth_path)},
            {"artifact_id": "confidence-frame-1", "digest": _digest(confidence_path)},
        ],
        "source_commit_sha": "a" * 40,
        "deterministic_configuration_digest": "sha256:" + "2" * 64,
        "train_heldout_split_digest": "sha256:" + "3" * 64,
        "camera_calibration_binding": {"digest": "sha256:" + "4" * 64},
        "coordinate_frame_declaration": coordinate_frame,
        "authority_used": {"mode": "execute_non_spend"},
        "timestamp": "2026-07-30T23:30:00Z",
        "capture_profile": "iphone_arkit_lidar",
        "camera_ray_convention": convention,
        "metric_scale_status": "sensor_metric_unvalidated",
        "pixel_stride": 1,
        "accepted_confidence_values": [2],
        "maximum_edge_length_m": 1.5,
        "maximum_depth_discontinuity_m": 0.2,
        "declared_region_ids": ["room-floor", "unseen-corner"],
        "unsupported_region_ids": ["unseen-corner"],
        "generated_fill_used": False,
        "candidate_may_read_hidden_heldout": False,
        "warnings": [],
        "frames": [
            {
                "frame_id": "frame-0001",
                "split": split,
                "region_id": "room-floor",
                "depth_asset": {
                    "relative_path": "inputs/depth.png",
                    "digest": _digest(depth_path),
                    "encoding": "uint16_png",
                    "scale_to_meters": 0.001,
                },
                "confidence_asset": {
                    "relative_path": "inputs/confidence.png",
                    "digest": _digest(confidence_path),
                    "encoding": "uint8_png",
                },
                "depth_intrinsics": {
                    "width": 3,
                    "height": 3,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 1.0,
                    "cy": 1.0,
                },
                "T_world_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    value.update(updates)
    value["arkit_depth_surface_compilation_request_digest"] = canonical_digest(
        value, digest_field="arkit_depth_surface_compilation_request_digest"
    )
    return value


def _compile(root: Path, request: dict) -> tuple[dict, dict]:
    result = compile_arkit_depth_surface(
        source_artifact=request,
        artifact_root=root,
        output_root=root / "generated" / "arkit-surface",
    )
    surface = json.loads(
        (root / result["surface_asset"]["relative_path"]).read_text(encoding="utf-8")
    )
    return result, surface


def _build_request_from_scaffold(
    root: Path, *, split: str = "training", up_axis: str | None = "Z"
) -> dict:
    depth_path, confidence_path = _write_inputs(root)
    capture_digest = "sha256:" + "1" * 64
    declaration = {
        "depth_encoding": "uint16_png",
        "scale_to_meters": 0.001,
        "camera_ray_convention": "arkit_x_right_y_up_z_backward",
        "depth_intrinsics": {
            "width": 3,
            "height": 3,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 1.0,
            "cy": 1.0,
        },
        "depth_registered_to_arkit_camera": True,
        "confidence_encoding": "uint8_png",
        "accepted_confidence_values": [2],
    }
    declaration["declaration_digest"] = canonical_digest(
        declaration, digest_field="declaration_digest"
    )
    binding = {
        "frame_id": "frame-0001",
        "depth_relative_path": "inputs/depth.png",
        "depth_digest": _digest(depth_path),
        "confidence_relative_path": "inputs/confidence.png",
        "confidence_digest": _digest(confidence_path),
    }
    scaffold = {
        "schema_version": "arkit_metric_scaffold.v1",
        "capture_digest": capture_digest,
        "coordinate_system": {
            "world_frame_definition": "arkit_world_origin_at_session_start",
            "units": "meters",
            "handedness": "right_handed",
            "gravity_aligned": True,
            "up_axis": up_axis,
        },
        "depth_confidence_pairs": [binding],
        "depth_surface_source_readiness": {
            "schema_version": "arkit_depth_surface_source_readiness.v1",
            "status": "ready_for_confidence_filtered_backprojection",
            "blockers": [],
            "source_declaration": declaration,
            "agent_may_override": False,
        },
        "source_artifact_digests": {
            "inputs/depth.png": _digest(depth_path),
            "inputs/confidence.png": _digest(confidence_path),
        },
    }
    scaffold_digest = _artifact_digest(scaffold)
    calibration = {
        "schema_version": "camera_calibration_manifest.v1",
        "capture_digest": capture_digest,
        "source_metric_scaffold_digest": scaffold_digest,
    }
    calibration["calibration_digest"] = canonical_digest(
        calibration, digest_field="calibration_digest"
    )
    observations = {
        "schema_version": "camera_observation_manifest.v1",
        "capture_digest": capture_digest,
        "split_digest": "sha256:" + "3" * 64,
        "calibration_digest": calibration["calibration_digest"],
        "candidate_splits_only": True,
        "hidden_heldout_pixels_included": False,
        "observations": [
            {
                "observation_id": "decoded-frame-0001",
                "capture_frame_id": "frame-0001",
                "split": split,
                "T_world_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "depth_confidence_binding": binding,
            }
        ],
    }
    observations["camera_observation_digest"] = canonical_digest(
        observations, digest_field="camera_observation_digest"
    )
    return build_arkit_depth_surface_compilation_request(
        stable_run_identity="arkit-depth-builder-run",
        source_capture_identity="capture-1",
        source_capture_digest=capture_digest,
        source_commit_sha="a" * 40,
        metric_scaffold=scaffold,
        metric_scaffold_digest=scaffold_digest,
        camera_observation_manifest=observations,
        camera_calibration_manifest=calibration,
        artifact_root=root,
        authority_used={"mode": "execute_non_spend"},
        timestamp="2026-07-30T23:30:00Z",
        pixel_stride=1,
        maximum_edge_length_m=1.5,
        maximum_depth_discontinuity_m=0.2,
    )


def test_arkit_depth_backprojection_is_metric_explicit_and_replayable(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result, surface = _compile(tmp_path, request)
    replay, replay_surface = _compile(tmp_path, request)

    assert replay == result
    assert replay_surface == surface
    assert result["accepted_high_confidence_pixel_count"] == 9
    assert result["emitted_vertex_count"] == 9
    assert result["emitted_triangle_count"] == 8
    assert result["hidden_heldout_observations_accessed"] is False
    assert result["generated_fill_used"] is False
    assert surface["unseen_or_rejected_depth_filled"] is False
    assert surface["unsupported_region_ids"] == ["unseen-corner"]
    center = next(
        row
        for row in surface["vertices"]
        if row["source_pixel"] == {"frame_id": "frame-0001", "u": 1, "v": 1}
    )
    assert center["position_m"] == pytest.approx([0.0, 0.0, -1.0])


def test_request_builder_joins_only_candidate_depth_observations(tmp_path: Path) -> None:
    request = _build_request_from_scaffold(tmp_path)
    result, surface = _compile(tmp_path, request)

    assert request["frames"][0]["split"] == "training"
    assert request["candidate_may_read_hidden_heldout"] is False
    assert request["coordinate_frame_declaration"]["up_axis"] == "Z"
    assert request["unsupported_region_ids"] == ["arkit-unobserved-regions"]
    assert result["hidden_heldout_observations_accessed"] is False
    assert surface["unsupported_region_ids"] == ["arkit-unobserved-regions"]

    other = tmp_path / "heldout"
    other.mkdir()
    with pytest.raises(ArkitDepthSurfaceCompilerError, match="hidden_or_invalid_split"):
        _build_request_from_scaffold(other, split="held_out")

    missing_axis = tmp_path / "missing-axis"
    missing_axis.mkdir()
    with pytest.raises(
        ArkitDepthSurfaceCompilerError,
        match="coordinate_frame_declaration_incomplete",
    ):
        _build_request_from_scaffold(missing_axis, up_axis=None)


def test_low_confidence_and_depth_discontinuity_remain_missing(tmp_path: Path) -> None:
    confidence = np.full((3, 3), 2, dtype=np.uint8)
    confidence[1, 1] = 1
    result, surface = _compile(
        tmp_path, _request(tmp_path, confidence=confidence)
    )
    assert result["accepted_high_confidence_pixel_count"] == 8
    assert result["rejected_or_missing_pixel_count"] == 1
    assert result["emitted_triangle_count"] == 2
    assert all(row["source_pixel"] != {"frame_id": "frame-0001", "u": 1, "v": 1} for row in surface["vertices"])

    other = tmp_path / "discontinuity"
    other.mkdir()
    depth = np.full((3, 3), 1000, dtype=np.uint16)
    depth[1, 1] = 5000
    result, _ = _compile(other, _request(other, depth=depth))
    assert result["accepted_high_confidence_pixel_count"] == 9
    assert result["discontinuity_rejected_triangle_count"] == 6
    assert result["emitted_triangle_count"] == 2


def test_camera_convention_is_declared_not_inferred(tmp_path: Path) -> None:
    _, arkit = _compile(tmp_path, _request(tmp_path))
    other = tmp_path / "opencv"
    other.mkdir()
    _, opencv = _compile(
        other,
        _request(other, convention="opencv_x_right_y_down_z_forward"),
    )
    arkit_center = next(row for row in arkit["vertices"] if row["source_pixel"]["u"] == 1 and row["source_pixel"]["v"] == 1)
    opencv_center = next(row for row in opencv["vertices"] if row["source_pixel"]["u"] == 1 and row["source_pixel"]["v"] == 1)
    assert arkit_center["position_m"][2] == pytest.approx(-1.0)
    assert opencv_center["position_m"][2] == pytest.approx(1.0)

    invalid = _request(tmp_path, convention="guess_from_frame_order")
    with pytest.raises(ArkitDepthSurfaceCompilerError, match="camera_ray_convention"):
        _compile(tmp_path, invalid)


def test_hidden_heldout_digest_mismatch_and_symlink_escape_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArkitDepthSurfaceCompilerError, match="hidden_or_invalid_split"):
        _compile(tmp_path, _request(tmp_path, split="hidden_heldout"))

    request = _request(tmp_path)
    request["frames"][0]["depth_asset"]["digest"] = "sha256:" + "f" * 64
    request["arkit_depth_surface_compilation_request_digest"] = canonical_digest(
        request, digest_field="arkit_depth_surface_compilation_request_digest"
    )
    with pytest.raises(ArkitDepthSurfaceCompilerError, match="digest_mismatch"):
        _compile(tmp_path, request)

    request = _request(tmp_path)
    depth_path = tmp_path / "inputs" / "depth.png"
    outside = tmp_path / "outside.png"
    depth_path.rename(outside)
    depth_path.symlink_to(outside)
    request["arkit_depth_surface_compilation_request_digest"] = canonical_digest(
        request, digest_field="arkit_depth_surface_compilation_request_digest"
    )
    with pytest.raises(ArkitDepthSurfaceCompilerError, match="symlink_forbidden"):
        _compile(tmp_path, request)


def test_observed_arkit_surface_feeds_metric_geometry_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    result, _ = _compile(tmp_path, _request(tmp_path))
    surface = result["surface_asset"]
    metric_request = {
        "schema_version": "metric_geometry_compilation_request.v1",
        "stable_run_identity": "arkit-metric-geometry-run-1",
        "source_capture_identity": result["source_capture_identity"],
        "source_capture_digest": result["source_capture_digest"],
        "original_file_references": [
            {"artifact_id": "arkit_observed_surface", "digest": surface["digest"]}
        ],
        "source_commit_sha": result["source_commit_sha"],
        "deterministic_configuration_digest": "sha256:" + "5" * 64,
        "train_heldout_split_digest": result["train_heldout_split_digest"],
        "camera_calibration_binding": result["camera_calibration_binding"],
        "coordinate_frame_declaration": result["coordinate_frame_declaration"],
        "authority_used": result["authority_used"],
        "timestamp": result["timestamp"],
        "source_asset": surface,
        "metric_scale_status": "sensor_metric_unvalidated",
        "minimum_confidence": 1.0,
        "declared_region_ids": ["room-floor", "unseen-corner"],
        "unsupported_region_ids": ["unseen-corner"],
        "generated_fill_used": False,
        "appearance_asset_used_as_geometry_truth": False,
        "warnings": [],
        "blockers": ["independent_metric_scale_validation_required"],
    }
    metric_request["source_artifact_digest"] = canonical_digest(
        metric_request, digest_field="source_artifact_digest"
    )
    manifest = compile_metric_geometry(
        source_artifact=metric_request,
        artifact_root=tmp_path,
        output_root=tmp_path / "generated" / "metric-geometry",
    )
    assert manifest["metric_scale_status"] == "sensor_metric_unvalidated"
    assert manifest["claim_ceiling"] == "metric_reference_geometry"
    assert manifest["blockers"] == ["independent_metric_scale_validation_required"]
    assert manifest["generated_fill_used"] is False


def test_request_and_result_validate_against_versioned_schemas(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result, _ = _compile(tmp_path, request)
    schema_root = Path(__file__).parents[1] / "docs" / "schemas"
    request_schema = json.loads(
        (schema_root / "arkit_depth_surface_compilation_request.v1.schema.json").read_text()
    )
    result_schema = json.loads(
        (schema_root / "arkit_depth_surface_compilation_result.v1.schema.json").read_text()
    )
    Draft202012Validator(request_schema).validate(request)
    Draft202012Validator(result_schema).validate(result)


def test_registered_tool_uses_digest_bound_repository_runtime(tmp_path: Path) -> None:
    request = _request(tmp_path)
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="registered-arkit-depth-surface",
        customer_question="Compile the observed ARKit depth surface.",
        supervisor_output_dir=str(tmp_path),
        arkit_depth_surface_compilation_request=request,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            request["arkit_depth_surface_compilation_request_digest"]
        ],
    ).to_mapping()
    binding = next(
        item
        for item in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
        if item.tool_id == "compile_arkit_observed_surface"
    )
    observation = binding.invoke(
        {
            "arkit_depth_surface_compilation_request_digest": request[
                "arkit_depth_surface_compilation_request_digest"
            ]
        }
    )

    assert observation["status"] == "completed"
    assert observation["proof_effect"] == "none"
    assert observation["typed_result"]["emitted_vertex_count"] == 9
    assert observation["typed_result"]["emitted_triangle_count"] == 8
    assert observation["typed_result"]["generated_fill_used"] is False
    assert observation["typed_result"]["raw_arkit_poses_modified"] is False
    assert {row["artifact_type"] for row in observation["produced_artifact_references"]} == {
        "arkit_depth_surface_compilation_result.v1",
        "observed_surface_mesh.v1",
    }

    refused = binding.invoke(
        {"arkit_depth_surface_compilation_request_digest": "sha256:" + "f" * 64}
    )
    assert refused["status"] == "refused"
    assert refused["typed_failure"]["retryable"] is False


def test_registered_tool_rejects_fabricated_compiler_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path)

    def malicious_compiler(**kwargs: object) -> dict:
        result = compile_arkit_depth_surface(**kwargs)
        result["surface_asset"]["digest"] = "sha256:" + "e" * 64
        result["arkit_depth_surface_compilation_result_digest"] = canonical_digest(
            result,
            digest_field="arkit_depth_surface_compilation_result_digest",
        )
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="malicious-arkit-depth-surface-output",
        customer_question="Compile the observed ARKit depth surface.",
        supervisor_output_dir=str(tmp_path),
        arkit_depth_surface_compilation_request=request,
        arkit_depth_surface_compiler=malicious_compiler,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            request["arkit_depth_surface_compilation_request_digest"]
        ],
    ).to_mapping()
    binding = next(
        item
        for item in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
        if item.tool_id == "compile_arkit_observed_surface"
    )

    observation = binding.invoke(
        {
            "arkit_depth_surface_compilation_request_digest": request[
                "arkit_depth_surface_compilation_request_digest"
            ]
        }
    )

    assert observation["status"] == "refused"
    assert observation["typed_failure"]["reason"] == "emitted_artifact_digest_mismatch"
    assert observation["produced_artifact_references"] == []
