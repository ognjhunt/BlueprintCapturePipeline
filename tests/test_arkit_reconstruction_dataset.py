from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.arkit_reconstruction_dataset import (
    ARKIT_RECONSTRUCTION_DATASET_REQUEST_SCHEMA_VERSION,
    ArkitReconstructionDatasetError,
    build_arkit_reconstruction_dataset_export_request,
    compile_arkit_reconstruction_dataset,
    export_bound_arkit_reconstruction_dataset,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_frame_dataset import compile_frozen_frame_dataset
from blueprint_pipeline.reconstruction_colmap_dataset import (
    ColmapTrainingDatasetError,
    bind_colmap_initialization_surface,
    export_colmap_training_dataset,
)
from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    SupervisorContext,
    ToolRegistry,
    build_capture_reconstruction_route,
    load_capture_build_ingress,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import (
    default_authority_envelope,
)
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


CAPTURE_DIGEST = "sha256:" + "a" * 64
IMPLEMENTATION_DIGEST = "sha256:" + "b" * 64
RUNTIME_DIGEST = "sha256:" + "c" * 64
SOURCE_COMMIT = "d" * 40


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifacts(root: Path) -> tuple[dict, dict, dict, dict, str]:
    frames: list[dict] = []
    for index in range(5):
        path = root / "frames" / f"decoded-{index:09d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"rgb-{index}".encode())
        frames.append(
            {
                "frame_id": f"decoded-{index:09d}",
                "decoded_frame_index": index,
                "t_video_sec": round(index * 0.1, 9),
                "source_pts_seconds": 10.0 + index * 0.1,
                "source_dts_seconds": None,
                "duration_seconds": 0.1,
                "key_frame": index == 0,
                "artifact_relative_path": path.relative_to(root).as_posix(),
                "digest": _file_digest(path),
                "image_metadata": {
                    "width": 64,
                    "height": 48,
                    "pixel_orientation": "encoded_source_no_autorotate",
                },
                "quality_signals": {"gradient_energy": 10.0},
            }
        )
    dataset = compile_frozen_frame_dataset(
        artifact_root=root,
        intake_id="intake-1",
        capture_digest=CAPTURE_DIGEST,
        capture_authority_profile="iphone_arkit_lidar",
        source_video_relative_path="walkthrough.mov",
        source_video_digest="sha256:" + "e" * 64,
        decoded_frame_count=5,
        selected_frames=frames,
        stream_metadata={"width": 64, "height": 48, "display_rotation_degrees": 90},
        runtime_identity="fixture-ffmpeg",
        runtime_digest=RUNTIME_DIGEST,
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        rights_and_retention={"external_processing": False},
        timestamp="2026-07-30T12:00:00Z",
    )
    base = next(root.glob("frozen_dataset_*"))
    split = json.loads((base / "frozen_split_manifest.json").read_text())
    candidate = json.loads((base / "candidate_dataset_manifest.json").read_text())
    pose = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    scaffold = {
        "schema_version": "arkit_metric_scaffold.v1",
        "capture_digest": CAPTURE_DIGEST,
        "coordinate_frame_session_id": "session-1",
        "coordinate_system": {
            "world_frame_definition": "arkit_world_origin_at_session_start",
            "units": "meters",
            "handedness": "right_handed",
            "gravity_aligned": True,
        },
        "intrinsics": {"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48},
        "camera_frames": [
            {
                "frame_id": f"capture-{index}",
                "encoded_frame_index": index,
                "t_video_sec": round(index * 0.1, 9),
                "t_capture_sec": round(index * 0.1, 9),
                "T_world_camera": pose,
            }
            for index in range(5)
        ],
        "depth_confidence_pairs": [
            {
                "frame_id": "capture-0",
                "depth_relative_path": "arkit/depth/0.bin",
                "depth_digest": "sha256:" + "f" * 64,
                "confidence_relative_path": "arkit/confidence/0.bin",
                "confidence_digest": "sha256:" + "1" * 64,
            }
        ],
        "source_artifact_digests": {},
    }
    scaffold_digest = "sha256:" + hashlib.sha256(
        (json.dumps(scaffold, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return dataset, split, candidate, scaffold, scaffold_digest


def test_arkit_export_is_candidate_only_idempotent_and_fail_closed(tmp_path: Path) -> None:
    dataset, split, candidate, scaffold, scaffold_digest = _source_artifacts(tmp_path / "source")
    kwargs = {
        "output_root": tmp_path / "export",
        "intake_id": "intake-1",
        "capture_digest": CAPTURE_DIGEST,
        "dataset_manifest": dataset,
        "split_manifest": split,
        "candidate_manifest": candidate,
        "metric_scaffold": scaffold,
        "metric_scaffold_digest": scaffold_digest,
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "source_commit_sha": SOURCE_COMMIT,
        "authority_used": {"external_processing": False},
        "timestamp": "2026-07-30T12:00:00Z",
    }

    first = compile_arkit_reconstruction_dataset(**kwargs)
    second = compile_arkit_reconstruction_dataset(**kwargs)

    assert first == second
    assert first["hidden_heldout_pixels_included"] is False
    assert first["raw_arkit_poses_modified"] is False
    assert first["metric_scale_validation_status"] == "not_executed"
    assert first["colmap_gsplat_export_status"] == (
        "candidate_only_raw_arkit_pose_request_ready"
    )
    root = next((tmp_path / "export").glob("arkit_export_*"))
    observations = json.loads(
        (root / "candidate_camera_observation_manifest.json").read_text()
    )
    assert observations["candidate_splits_only"] is True
    assert observations["hidden_heldout_pixels_included"] is False
    assert {row["split"] for row in observations["observations"]} <= {
        "training",
        "validation",
    }
    assert len(observations["observations"]) < len(split["assignments"])
    request = json.loads((root / "pose_refinement_request.json").read_text())
    assert request["candidate_may_change_input_poses"] is False
    assert request["maximum_pose_drift_threshold"] is None
    calibration = json.loads((root / "camera_calibration_manifest.json").read_text())
    colmap_request = json.loads(
        (root / "colmap_training_dataset_export_request.json").read_text()
    )
    assert first["colmap_training_dataset_export_request_digest"] == colmap_request[
        "colmap_training_dataset_export_request_digest"
    ]
    assert colmap_request["blockers"] == [
        "initialization_surface_not_bound",
        "pose_refinement_not_executed",
    ]
    colmap = export_colmap_training_dataset(
        source_artifact=colmap_request,
        artifact_root=next((tmp_path / "source").glob("frozen_dataset_*")),
        output_root=tmp_path / "colmap",
    )
    assert colmap["image_count"] == first["candidate_observation_count"]
    assert colmap["initialization_point_count"] == 0
    assert colmap["raw_input_poses_modified"] is False
    assert colmap["pose_refinement_executed"] is False
    assert colmap["hidden_heldout_pixels_included"] is False
    assert colmap["blockers"] == [
        "initialization_surface_not_bound",
        "pose_refinement_not_executed",
    ]
    surface_root = tmp_path / "surface-root"
    surface_path = surface_root / "generated/arkit_observed_surface.json"
    surface_path.parent.mkdir(parents=True)
    surface_value = {
        "schema_version": "observed_surface_mesh.v1",
        "source_capture_digest": CAPTURE_DIGEST,
        "train_heldout_split_digest": first["frozen_split_digest"],
        "generated_fill_used": False,
        "vertices": [
            {"vertex_id": "v0", "position_m": [0.0, 0.0, 0.0]},
            {"vertex_id": "v1", "position_m": [1.0, 0.0, 0.0]},
        ],
        "faces": [],
    }
    surface_path.write_text(json.dumps(surface_value), encoding="utf-8")
    surface_digest = _file_digest(surface_path)
    surface_result = {
        "schema_version": "arkit_depth_surface_compilation_result.v1",
        "status": "compiled_observed_surface_candidate",
        "source_capture_digest": CAPTURE_DIGEST,
        "train_heldout_split_digest": first["frozen_split_digest"],
        "camera_calibration_binding": {
            "calibration_digest": first["camera_calibration_digest"]
        },
        "coordinate_frame_declaration": colmap_request[
            "coordinate_frame_declaration"
        ],
        "surface_asset": {
            "relative_path": surface_path.relative_to(surface_root).as_posix(),
            "digest": surface_digest,
        },
        "hidden_heldout_observations_accessed": False,
        "generated_fill_used": False,
        "raw_arkit_poses_modified": False,
    }
    surface_result["arkit_depth_surface_compilation_result_digest"] = canonical_digest(
        surface_result,
        digest_field="arkit_depth_surface_compilation_result_digest",
    )
    initialized_request = bind_colmap_initialization_surface(
        source_artifact=colmap_request,
        surface_compilation_result=surface_result,
    )
    assert initialized_request["blockers"] == ["pose_refinement_not_executed"]
    initialized = export_colmap_training_dataset(
        source_artifact=initialized_request,
        artifact_root=next((tmp_path / "source").glob("frozen_dataset_*")),
        initialization_artifact_root=surface_root,
        output_root=tmp_path / "initialized-colmap",
    )
    assert initialized["initialization_surface_digest"] == surface_digest
    assert initialized["initialization_point_count"] == 2
    assert initialized["blockers"] == ["pose_refinement_not_executed"]
    generated_surface = json.loads(json.dumps(surface_result))
    generated_surface["generated_fill_used"] = True
    generated_surface["arkit_depth_surface_compilation_result_digest"] = canonical_digest(
        generated_surface,
        digest_field="arkit_depth_surface_compilation_result_digest",
    )
    with pytest.raises(
        ColmapTrainingDatasetError,
        match="colmap_surface_binding_truth_boundary_invalid",
    ):
        bind_colmap_initialization_surface(
            source_artifact=colmap_request,
            surface_compilation_result=generated_surface,
        )
    tampered = json.loads(json.dumps(colmap_request))
    tampered["camera_observation_manifest"]["observations"][0]["camera"][
        "T_world_camera"
    ][0][3] = 99.0
    tampered["camera_observation_manifest"]["camera_observation_digest"] = canonical_digest(
        tampered["camera_observation_manifest"],
        digest_field="camera_observation_digest",
    )
    tampered["colmap_training_dataset_export_request_digest"] = canonical_digest(
        tampered,
        digest_field="colmap_training_dataset_export_request_digest",
    )
    with pytest.raises(
        ColmapTrainingDatasetError,
        match="colmap_camera_projection_pose_mismatch",
    ):
        export_colmap_training_dataset(
            source_artifact=tampered,
            artifact_root=next((tmp_path / "source").glob("frozen_dataset_*")),
            output_root=tmp_path / "tampered-colmap",
        )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/arkit_reconstruction_dataset.v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    for artifact in (first, observations, request, calibration):
        validator.validate(artifact)


def test_arkit_export_rejects_pixel_calibration_and_digest_spoofing(tmp_path: Path) -> None:
    dataset, split, candidate, scaffold, scaffold_digest = _source_artifacts(tmp_path / "source")
    bad_candidate = json.loads(json.dumps(candidate))
    bad_candidate["frames"][0]["image_metadata"]["width"] = 65
    bad_candidate["candidate_dataset_digest"] = canonical_digest(
        bad_candidate, digest_field="candidate_dataset_digest"
    )
    base = {
        "output_root": tmp_path / "export",
        "intake_id": "intake-1",
        "capture_digest": CAPTURE_DIGEST,
        "dataset_manifest": dataset,
        "split_manifest": split,
        "metric_scaffold": scaffold,
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "source_commit_sha": SOURCE_COMMIT,
        "authority_used": {},
    }
    with pytest.raises(ArkitReconstructionDatasetError, match="pixel_binding_mismatch"):
        compile_arkit_reconstruction_dataset(
            **base,
            candidate_manifest=bad_candidate,
            metric_scaffold_digest=scaffold_digest,
        )
    with pytest.raises(ArkitReconstructionDatasetError, match="metric_scaffold_invalid"):
        compile_arkit_reconstruction_dataset(
            **base,
            candidate_manifest=candidate,
            metric_scaffold_digest="sha256:" + "9" * 64,
        )


def test_registered_arkit_export_is_digest_bound_and_candidate_only(tmp_path: Path) -> None:
    dataset, split, candidate, scaffold, scaffold_digest = _source_artifacts(
        tmp_path / "source"
    )
    request = build_arkit_reconstruction_dataset_export_request(
        {
            "schema_version": ARKIT_RECONSTRUCTION_DATASET_REQUEST_SCHEMA_VERSION,
            "intake_id": "intake-1",
            "source_capture_digest": CAPTURE_DIGEST,
            "dataset_manifest": dataset,
            "split_manifest": split,
            "candidate_manifest": candidate,
            "metric_scaffold": scaffold,
            "metric_scaffold_digest": scaffold_digest,
            "implementation_digest": IMPLEMENTATION_DIGEST,
            "source_commit_sha": SOURCE_COMMIT,
            "authority_used": {"external_processing": False},
            "timestamp": "2026-07-30T12:00:00Z",
        }
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/arkit_reconstruction_dataset.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(request)
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="arkit-export-tool",
        customer_question="Export candidate-only ARKit observations.",
        supervisor_output_dir=str(tmp_path / "run"),
        arkit_reconstruction_dataset_request=request,
        arkit_reconstruction_dataset_exporter=export_bound_arkit_reconstruction_dataset,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[request["arkit_export_request_digest"]],
    ).to_mapping()
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }

    observation = bindings["export_arkit_reconstruction_dataset"].invoke(
        {"arkit_export_request_digest": request["arkit_export_request_digest"]}
    )

    assert observation["status"] == "completed"
    assert observation["typed_result"]["hidden_heldout_pixels_included"] is False
    assert observation["typed_result"]["raw_arkit_poses_modified"] is False
    assert observation["typed_result"]["claim_ceiling"] == "calibrated_camera_trajectory"
    refused = bindings["export_arkit_reconstruction_dataset"].invoke(
        {"arkit_export_request_digest": "sha256:" + "9" * 64}
    )
    assert refused["status"] == "refused"
    assert "source_digest_mismatch" in refused["typed_failure"]["reason"]


def test_registered_arkit_scaffold_cannot_promote_sensor_scale(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_raw_capture_manifest.v1",
                "capture_id": "capture-arkit",
                "scene_id": "scene-arkit",
                "capture_authority_profile": "iphone_arkit_lidar",
                "capture_modality": "iphone_arkit_lidar",
            }
        ),
        encoding="utf-8",
    )
    capture_build = load_capture_build_ingress(capture_root)
    route = build_capture_reconstruction_route(capture_build)

    def compiler(*, request: dict, output_root: Path) -> dict:
        assert request["capture_authority_profile"] == "iphone_arkit_lidar"
        assert output_root.name == "arkit_metric_scaffold"
        return {
            "result_id": "arkit-scaffold-fixture",
            "intake_id": "intake-1",
            "capture_digest": CAPTURE_DIGEST,
            "method_id": "local_arkit_metric_scaffold",
            "method_version": "1",
            "method_profile_digest": "sha256:" + "2" * 64,
            "implementation_digest": IMPLEMENTATION_DIGEST,
            "provider_identity": "local",
            "runtime_identity": "fixture-runtime",
            "runtime_digest": RUNTIME_DIGEST,
            "outputs": ["calibrated_frames", "metric_reference_layer"],
            "source_frames": {},
            "camera_solution": {"status": "raw_contract_3_2_verified"},
            "coordinate_system": {"units": "meters"},
            "asset_references": {
                "metric_scaffold": {"uri": "local://scaffold", "digest": scaffold_digest},
                "arkit_reconstruction_dataset_export": {
                    "uri": "local://export",
                    "digest": "sha256:" + "3" * 64,
                },
                "arkit_raw_contract_validation": {
                    "uri": "local://raw-contract-validation",
                    "digest": "sha256:" + "4" * 64,
                },
            },
            "coverage_map": {},
            "observed_regions": [],
            "generated_regions": [],
            "uncertainty_map": {},
            "invalid_regions": [],
            "validation_metrics": {
                "decoded_pts_verified": True,
                "pose_refinement_executed": False,
                "independent_metric_scale_validation_passed": False,
                "arkit_raw_contract_validation_digest": "sha256:" + "5" * 64,
            },
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "rights_and_retention": {"external_processing": False},
            "claim_ceiling": {
                "calibrated_camera_poses": True,
                "sensor_declared_metric_scale": True,
                "metric_scale": False,
                "metric_reference_layer": False,
                "collision_geometry": False,
                "physical_task_success": False,
            },
        }

    scaffold_digest = "sha256:" + "1" * 64
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="arkit-scaffold-tool",
        customer_question="Compile the strict ARKit scaffold.",
        capture_build=capture_build,
        supervisor_output_dir=str(tmp_path / "run"),
        arkit_metric_scaffold_compiler=compiler,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[capture_build["capture_build_digest"]],
    ).to_mapping()
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }

    observation = bindings["compile_arkit_metric_scaffold"].invoke(
        {
            "capture_build_digest": capture_build["capture_build_digest"],
            "capture_reconstruction_route_digest": route[
                "capture_reconstruction_route_digest"
            ],
        }
    )

    assert observation["status"] == "completed"
    assert observation["typed_result"]["decoded_pts_verified"] is True
    assert observation["typed_result"]["metric_scale_independently_validated"] is False
    assert observation["typed_result"]["claim_ceiling"] == (
        "sensor_declared_metric_scaffold"
    )
    assert observation["proof_effect"] == "none"
