from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_heldout_request_compiler import (
    HeldoutRequestCompilationError,
    RENDER_MANIFEST_SCHEMA_VERSION,
    build_candidate_heldout_render_manifest,
    compile_heldout_appearance_evaluation_request,
    compile_heldout_appearance_supervisor_bindings,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    build_training_request,
    build_training_result,
)


D = ["sha256:" + f"{index:x}" * 64 for index in range(1, 10)]
SHA = "a" * 40
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64


def _training_request(split_digest: str) -> dict:
    return build_training_request(
        {
            "stable_run_identity": "training-1",
            "source_capture_identity": "capture-1",
            "source_capture_digest": D[0],
            "original_file_references": [{"artifact_id": "capture.mov", "digest": D[1]}],
            "producing_method": "fixture",
            "implementation_version": "1.0.0",
            "container_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "deterministic_configuration_digest": D[2],
            "input_digests": [{"artifact_id": "dataset", "digest": D[3]}],
            "output_digests": [],
            "train_heldout_split_digest": split_digest,
            "camera_calibration_binding": {"calibration_digest": D[5]},
            "coordinate_frame_declaration": {"frame": "capture_world", "units": "meters"},
            "units": "meters",
            "metric_scale_status": "sensor_metric_unvalidated",
            "provider_runtime_identity": {"provider": "vast", "runtime": "gpu-canary"},
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "authority_used": {"authority_id": "fixture"},
            "warnings": [],
            "blockers": [],
            "proof_effect": "none",
            "claim_ceiling": "request_only",
            "parent_artifact_or_event": {"dataset": D[3]},
            "timestamp": "2026-07-30T23:00:00Z",
            "method_profile_id": "nvidia_3dgrut_3dgut_mcmc_v1",
            "reconstruction_dataset_digest": D[3],
            "calibration_digest": D[5],
            "initialization_geometry_digest": D[6],
            "pose_result_digest": D[7],
            "worker_stack_manifest_digest": D[8],
            "evaluation_contract_digest": D[1],
            "camera_model": "PINHOLE",
            "densification_configuration": {"strategy": "mcmc"},
            "random_seed": 23,
            "iteration_budget": 30_000,
            "resource_request": {"gpu_count": 1, "minimum_vram_gb": 24},
            "timeout_seconds": 3600,
            "spend_cap_usd": 18.0,
            "output_contract": {"appearance_asset": "standard_3dgs_ply"},
            "candidate_dataset_contains_hidden_heldout_pixels": False,
            "candidate_can_change_split": False,
            "candidate_may_read_hidden_heldout": False,
            "trainer_may_grade_heldout": False,
        }
    )


def _training_result(request: dict) -> dict:
    return build_training_result(
        {
            **request,
            "producing_method": "nvidia-3dgrut-adapter",
            "output_digests": [
                {"artifact_id": "training.log", "digest": D[1]},
                {"artifact_id": "appearance_candidate.ply", "digest": D[2]},
            ],
            "proof_effect": "appearance_asset_candidate_only",
            "claim_ceiling": "appearance_reconstruction",
            "reconstruction_training_request_digest": request[
                "reconstruction_training_request_digest"
            ],
            "status": "succeeded",
            "failure_code": None,
            "checkpoint_references": [{"artifact_id": "checkpoint.pt", "digest": D[6]}],
            "training_metrics": {"iterations_completed": 30_000},
            "heldout_labels_included": False,
            "candidate_self_graded": False,
            "registered_observation_ids": ["train-1"],
            "rejected_observation_ids": [],
            "peak_resource_use": {"gpu_memory_gb": 12.0},
            "legal_next_actions": ["preserve_evidence_and_stop"],
        }
    )


def _dataset(split_digest: str) -> dict:
    value = {
        "schema_version": "reconstruction_dataset_manifest.v1",
        "stable_run_identity": "dataset-1",
        "source_capture_identity": "capture-1",
        "source_capture_digest": D[0],
        "train_heldout_split_digest": split_digest,
    }
    value["dataset_manifest_digest"] = canonical_digest(
        value, digest_field="dataset_manifest_digest"
    )
    return value


def _split() -> dict:
    value = {
        "schema_version": "frozen_reconstruction_split_manifest.v1",
        "frozen": True,
        "capture_digest": D[0],
        "deterministic_configuration_digest": D[5],
        "split_seed_digest": D[6],
        "assignments": [
            {
                "frame_id": "train-1",
                "decoded_frame_index": 1,
                "t_video_sec": 0.1,
                "frame_digest": D[7],
                "split": "training",
            },
            {
                "frame_id": "hidden-1",
                "decoded_frame_index": 2,
                "t_video_sec": 0.2,
                "frame_digest": D[8],
                "split": "held_out",
            },
        ],
        "candidate_can_change_assignments": False,
        "hidden_heldout_access": "independent_evaluator_only",
    }
    value["split_digest"] = canonical_digest(value, digest_field="split_digest")
    return value


def _hidden(split: dict) -> dict:
    value = {
        "schema_version": "hidden_heldout_evaluator_manifest.v1",
        "capture_digest": D[0],
        "split_digest": split["split_digest"],
        "access_scope": "independent_evaluator_only",
        "candidate_method_access_allowed": False,
        "frames": [
            {
                "frame_id": "hidden-1",
                "decoded_frame_index": 2,
                "t_video_sec": 0.2,
                "frame_digest": D[8],
                "evaluator_relative_path": "held_out/hidden-1.png",
            }
        ],
    }
    value["hidden_heldout_digest"] = canonical_digest(
        value, digest_field="hidden_heldout_digest"
    )
    return value


def _render(result: dict, split: dict) -> dict:
    return build_candidate_heldout_render_manifest(
        {
            "schema_version": RENDER_MANIFEST_SCHEMA_VERSION,
            "source_capture_digest": D[0],
            "frozen_split_digest": split["split_digest"],
            "reconstruction_training_result_digest": result[
                "reconstruction_training_result_digest"
            ],
            "appearance_asset_digest": D[2],
            "candidate_method_id": "nvidia_3dgrut_3dgut_mcmc_v1",
            "candidate_provider_identity": "vast-gpu-canary",
            "renderer_identity": "3dgrut-frozen-camera-renderer",
            "renderer_implementation_digest": D[7],
            "hidden_pixels_read_by_candidate": False,
            "heldout_labels_read_by_candidate": False,
            "candidate_selected_heldout": False,
            "candidate_self_grading": False,
            "frozen_camera_parameters_only": True,
            "renders": [
                {
                    "view_id": "hidden-1",
                    "real_view_digest": D[8],
                    "candidate_render_relative_path": "renders/hidden-1.png",
                    "candidate_render_digest": D[6],
                    "projection_form": "perspective_rgb",
                }
            ],
        }
    )


def _evaluator(split: dict) -> dict:
    value = {
        "stable_run_identity": "heldout-eval-1",
        "source_capture_digest": D[0],
        "frozen_split_digest": split["split_digest"],
        "candidate_provider_identity": "vast-gpu-canary",
        "evaluator_identity": "blueprint-independent-heldout-v1",
        "evaluator_provider_identity": "blueprint-local-evaluator",
        "evaluator_implementation_digest": D[5],
        "source_commit_sha": SHA,
        "thresholds_frozen_before_evaluation": True,
        "candidate_hidden_pixel_access_permitted": False,
        "candidate_self_grading_permitted": False,
        "thresholds": {
            "minimum_mean_psnr_db": 30.0,
            "minimum_mean_global_ssim": 0.95,
            "maximum_mean_absolute_error": 0.02,
        },
    }
    value["evaluation_contract_digest"] = canonical_digest(
        value, digest_field="evaluation_contract_digest"
    )
    return value


def _arguments() -> dict:
    split = _split()
    request = _training_request(split["split_digest"])
    result = _training_result(request)
    return {
        "training_request": request,
        "training_result": result,
        "reconstruction_dataset_manifest": _dataset(split["split_digest"]),
        "frozen_split_manifest": split,
        "hidden_heldout_manifest": _hidden(split),
        "candidate_render_manifest": _render(result, split),
        "evaluator_contract": _evaluator(split),
        "candidate_root": "/trusted/candidate",
        "evaluator_root": "/trusted/evaluator",
        "authority_used": {"local_evaluation_allowed": True},
        "timestamp": "2026-07-30T23:30:00Z",
    }


def test_compiler_binds_frozen_hidden_views_without_exposing_them_to_trainer() -> None:
    arguments = _arguments()
    request = compile_heldout_appearance_evaluation_request(**arguments)
    render_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/candidate_heldout_render_manifest.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(render_schema).validate(
        arguments["candidate_render_manifest"]
    )

    assert request["candidate_had_hidden_access"] is False
    assert request["candidate_self_grading"] is False
    assert request["pairs"][0]["view_id"] == "hidden-1"
    assert request["pairs"][0]["real_view_digest"] == D[8]
    assert request["pairs"][0]["real_view_relative_path"] == "held_out/hidden-1.png"
    bindings = compile_heldout_appearance_supervisor_bindings(**_arguments())
    assert set(bindings) == {
        "heldout_appearance_evaluation_request",
        "heldout_appearance_evaluator",
    }


def test_compiler_rejects_training_leakage_and_render_set_drift() -> None:
    arguments = _arguments()
    leaked = copy.deepcopy(arguments["training_result"])
    leaked["registered_observation_ids"].append("hidden-1")
    leaked["reconstruction_training_result_digest"] = canonical_digest(
        leaked, digest_field="reconstruction_training_result_digest"
    )
    arguments["training_result"] = leaked

    with pytest.raises(HeldoutRequestCompilationError) as error:
        compile_heldout_appearance_evaluation_request(**arguments)
    assert "heldout_training_observation_leakage" in error.value.codes


def test_render_manifest_refuses_hidden_pixels_and_self_grading() -> None:
    arguments = _arguments()
    render = dict(arguments["candidate_render_manifest"])
    render.pop("candidate_heldout_render_manifest_digest")
    render["hidden_pixels_read_by_candidate"] = True
    render["candidate_self_grading"] = True

    with pytest.raises(HeldoutRequestCompilationError) as error:
        build_candidate_heldout_render_manifest(render)
    assert {
        "heldout_render_manifest_forbidden:hidden_pixels_read_by_candidate",
        "heldout_render_manifest_forbidden:candidate_self_grading",
    } <= set(error.value.codes)


def test_compiler_rejects_threshold_contract_tampering() -> None:
    arguments = _arguments()
    evaluator = copy.deepcopy(arguments["evaluator_contract"])
    evaluator["thresholds"]["minimum_mean_psnr_db"] = 0.0
    arguments["evaluator_contract"] = evaluator

    with pytest.raises(
        HeldoutRequestCompilationError,
        match="heldout_evaluator_contract_invalid",
    ):
        compile_heldout_appearance_evaluation_request(**arguments)
