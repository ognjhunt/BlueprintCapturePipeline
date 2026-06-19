from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zipfile
from types import SimpleNamespace
from pathlib import Path

import pytest

from blueprint_pipeline import robot_eval_provider_input_setup as provider_input_setup
from blueprint_pipeline.evaluation_prep_stage import robot_eval_job_evaluation_prep_surface
from blueprint_pipeline.live_robot_eval_closure import build_live_robot_eval_closure_manifest
from blueprint_pipeline.post_training_data_package import build_post_training_data_package_export
from blueprint_pipeline.robot_eval_execution import build_scenario_eval_matrix
from blueprint_pipeline.robot_eval_job_orchestrator import (
    AgentsSdkRobotEvalJobAdapter,
    FakeRobotEvalJobAgentAdapter,
    _remote_cloud_execution_closure_manifest,
    _robot_team_grade_eval_closure_manifest,
    build_robot_eval_job,
    resolve_simulator_selection_policy,
    run_robot_eval_job_request_inbox,
)
from blueprint_pipeline.robot_eval_provider_input_setup import (
    prepare_robot_eval_provider_inputs,
)
from blueprint_pipeline.robot_eval_worker import _build_parser, run_robot_eval_worker


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_identity": {"site_id": "site-1"},
            "video_uri": "walkthrough.mov",
            "width": 1280,
            "height": 720,
            "frame_count": 3,
        },
    )
    _write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "status": "complete",
        },
    )
    (capture_root / "raw" / "walkthrough.mov").write_bytes(b"raw capture video\n")
    return capture_root


def _write_robot_eval_cards(
    capture_root: Path,
    *,
    scenario_variation_label: str = "derived",
    rights_blocked: bool = False,
) -> None:
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        robot_eval_dir / "site_card.json",
        {
            "schema_version": "real_site_robot_eval_site_card.v0.1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_id": "site-1",
            "site_type": "stockroom",
            "geometry": {
                "collider": {
                    "status": "review_input_present",
                    "collision_ready_claim_allowed": False,
                }
            },
            "provenance_rights_review_status": {
                "rights_privacy": {
                    "blocked": rights_blocked,
                    "rights_status": "blocked" if rights_blocked else "verified",
                }
            },
        },
    )
    _write_json(
        robot_eval_dir / "task_cards.json",
        {
            "schema_version": "real_site_robot_eval_task_cards.v0.1",
            "task_card_count": 1,
            "cards": [
                {
                    "task_card_id": "task_card_place_return_in_bin",
                    "task_id": "place_return_in_bin",
                    "task_statement": "Place the return item in the labeled bin",
                    "task_category": "pick_place",
                    "required_metrics": [
                        "success_rate",
                        "cycle_time",
                        "intervention_rate",
                        "unsafe_proximity",
                        "collision_risk",
                        "object_drop",
                        "wrong_object",
                        "timeout",
                        "recovery_success",
                        "world_model_uncertainty",
                        "sim_vs_real_calibration_score",
                        "placement_accuracy",
                    ],
                    "claim_boundary": "task_card_defines_eval_scope_not_robot_execution",
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "scenario_cards.json",
        {
            "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
            "scenario_card_count": 1,
            "cards": [
                {
                    "scenario_card_id": "scenario_card_place_return_in_bin_mobile",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "robot_profile_id": "mobile_manipulator_rgb_v1",
                    "target_object_ids": ["bin_0001"],
                    "target_objects": [
                        {
                            "object_id": "bin_0001",
                            "label": "returns bin",
                            "task_role": "target_container",
                            "center_xyz": [1.0, 0.0, 0.2],
                            "has_collision_hulls": True,
                            "has_support_surfaces": True,
                        }
                    ],
                    "start_zone": [-1.0, 0.0, 0.793],
                    "goal_zone": [1.0, 0.0, 0.793],
                    "start_zone_id": "start_zone_place_return_in_bin",
                    "goal_zone_id": "goal_zone_place_return_in_bin",
                    "spawn_candidates": [
                        {
                            "zone_id": "start_zone_place_return_in_bin",
                            "role": "robot_spawn",
                            "pose_xyz": [-1.0, 0.0, 0.793],
                            "validation_status": "validated_finite_site_pose",
                            "validated": True,
                            "label_source": "task_anchor_manifest",
                        }
                    ],
                    "target_candidates": [
                        {
                            "zone_id": "goal_zone_place_return_in_bin",
                            "role": "task_goal",
                            "pose_xyz": [1.0, 0.0, 0.793],
                            "validation_status": "validated_finite_site_pose",
                            "validated": True,
                            "label_source": "task_anchor_manifest",
                        }
                    ],
                    "semantic_spawn_target": {
                        "validated_spawn_target_pair": True,
                        "validated_spawn_candidate_count": 1,
                        "validated_target_candidate_count": 1,
                        "source": "task_anchor_manifest_site_zones",
                        "fallback_allowed_for_beta_release": False,
                    },
                    "normal_scenario": {
                        "statement": "Run the task under the capture-observed layout.",
                        "ground_truth_status": "derived_from_capture_package",
                    },
                    "variation": {
                        "statement": "Run under clutter variation.",
                        "ground_truth_status": f"{scenario_variation_label}_needs_review",
                    },
                    "edge_case": {
                        "statement": "Blocked path near the bin.",
                        "ground_truth_status": "agent_inferred_needs_review",
                    },
                    "observed_vs_inferred_labels": {
                        "layout": "capture_grounded",
                        "variation": scenario_variation_label,
                        "edge_case": "agent_inferred",
                    },
                    "required_missing_annotations": [
                        "needs_robot_pov",
                        "needs_action_logs",
                        "needs_actual_outcome",
                    ],
                    "claim_boundary": "scenario_card_is_review_scope_not_simulator_or_pilot_result",
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "scenario_family_library.json",
        {
            "schema_version": "real_site_robot_eval_scenario_family_library.v1",
            "family_count": 1,
            "variation_names_required": list(POLICY_REFERENCE_VARIATION_NAMES),
            "families": [
                {
                    "family_id": "family_scenario_place_return_in_bin_mobile",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "robot_profile_id": "mobile_manipulator_rgb_v1",
                    "status": "review_required",
                    "variation_count": len(POLICY_REFERENCE_VARIATION_NAMES),
                    "variations": [
                        {
                            "variation_id": variation_name,
                            "variation_name": variation_name,
                            "scenario_status": "review-only",
                        }
                        for variation_name in POLICY_REFERENCE_VARIATION_NAMES
                    ],
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "eval_card_count": 1,
            "cards": [
                {
                    "eval_card_id": "eval_card_place_return_in_bin_marble",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "prediction_source": "marble_review",
                    "engine_used": "hosted visual review",
                    "validation": {"actual_status": "needs_actual_outcome"},
                    "blocked_upgrades": [
                        "simulator_execution_completed",
                        "robot_policy_execution_proven",
                    ],
                    "proof_boundary": "prediction_only_no_actual_outcome_no_deployment_claim",
                }
            ],
        },
    )
    _write_json(
        robot_eval_dir / "proof_boundaries.json",
        {
            "schema_version": "real_site_robot_eval_proof_boundaries.v0.1",
            "simulator_execution_proven": False,
            "physics_contact_validation_proven": False,
            "robot_policy_execution_proven": False,
            "safety_validation_proven": False,
            "real_pilot_outcome_proven": False,
            "generated_scenarios_are_real_world_proof": False,
        },
    )


def _write_fixture_attempts(capture_root: Path, *, success: bool) -> None:
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "headless_fixture_attempts.json",
        {
            "schema_version": "site_eval_fixture_attempts.v1",
            "attempts": [
                {
                    "attempt_id": "fixture-attempt-1",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "policy_id": "policy-fixture-a",
                    "success": success,
                    "predicted_success": True,
                    "predicted_cycle_time_seconds": 12.5,
                    "predicted_intervention_count": 0,
                    "predicted_safety_event_count": 0,
                    "metrics": {
                        "cycle_time_seconds": 14.0,
                        "intervention_count": 0 if success else 1,
                        "contact_event_count": 0,
                        "safety_event_count": 0 if success else 1,
                    },
                    "failure_mode_ids": [] if success else ["failure_navigation_blocked"],
                    "breakage_categories": [] if success else ["blocked_path"],
                    "artifact_paths": {"trace": "fixtures/attempt-1.json"},
                    "owner_system": "BlueprintCapturePipeline.fixture",
                }
            ],
        },
    )


def test_post_training_export_includes_simulator_batch_trace_streams(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-batch-traces"
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "robot_eval_simulator_command_normalized_attempt_trace.v1",
            "attempts": [{"attempt_id": "attempt-run-a", "scenario_eval_run_id": "run-a"}],
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "robot_eval_simulator_command_failure_labels.v1",
            "labels": [],
        },
    )
    _write_json(
        job_dir / "visual_review_ledger.json",
        {
            "schema_version": "robot_eval_simulator_visual_review_ledger.v1",
            "status": "accepted",
            "records": [],
        },
    )
    _write_json(
        job_dir / "prediction_outcome_ledger.json",
        {
            "schema_version": "robot_eval_simulator_prediction_outcome_ledger.v1",
            "records": [],
        },
    )
    _write_json(
        job_dir / "calibration_report.json",
        {
            "schema_version": "robot_eval_simulator_calibration_report.v1",
            "records": [],
        },
    )
    _write_json(
        job_dir / "breakage_library.json",
        {
            "schema_version": "robot_eval_simulator_breakage_library.v1",
            "records": [],
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_trace_package_manifest.json",
        {
            "schema_version": "mujoco_g1_batch_trace_package.v1",
            "artifact_paths": {
                "attempt_trace_jsonl": "simulator_command_batch_attempt_trace.jsonl",
                "contact_stream_jsonl": "simulator_command_batch_contact_stream.jsonl",
                "planner_state_jsonl": "simulator_command_batch_planner_state.jsonl",
                "control_stream_jsonl": "simulator_command_batch_control_stream.jsonl",
                "visual_media_coverage": (
                    "simulator_command_batch_visual_media_coverage.json"
                ),
                "visual_review_ledger": (
                    "simulator_command_batch_visual_review_ledger.json"
                ),
            },
        },
    )
    for name in (
        "simulator_command_batch_attempt_trace.jsonl",
        "simulator_command_batch_contact_stream.jsonl",
        "simulator_command_batch_planner_state.jsonl",
        "simulator_command_batch_control_stream.jsonl",
        "simulator_command_batch_visual_media_coverage.json",
        "simulator_command_batch_visual_review_ledger.json",
        "simulator_command_digital_twin_fidelity_qa.json",
    ):
        path = job_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    package = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=job_dir,
    )

    assert package["status"] == "export_ready_review_required"
    assert package["included_artifacts"]["visual_review_ledger"] == "visual_review_ledger.json"
    assert package["included_artifacts"]["simulator_command_batch_visual_review_ledger"] == (
        "simulator_command_batch_visual_review_ledger.json"
    )
    assert package["export_policy"]["visual_review_ledger_included"] is True
    assert package["included_artifacts"]["simulator_command_batch_attempt_trace"] == (
        "simulator_command_batch_attempt_trace.jsonl"
    )
    assert package["included_artifacts"]["simulator_command_batch_contact_stream"] == (
        "simulator_command_batch_contact_stream.jsonl"
    )
    assert package["included_artifacts"]["simulator_command_batch_planner_state"] == (
        "simulator_command_batch_planner_state.jsonl"
    )
    assert package["included_artifacts"]["simulator_command_batch_control_stream"] == (
        "simulator_command_batch_control_stream.jsonl"
    )
    assert package["included_artifacts"]["simulator_command_batch_visual_media_coverage"] == (
        "simulator_command_batch_visual_media_coverage.json"
    )
    assert package["included_artifacts"]["simulator_command_digital_twin_fidelity_qa"] == (
        "simulator_command_digital_twin_fidelity_qa.json"
    )
    assert package["export_policy"]["simulator_command_batch_trace_streams_included"] is True


def test_scenario_eval_matrix_expands_requested_route_to_500_deterministic_runs(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_policy_reference_variation_instances(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = {
        "schema_version": "blueprint.robot_eval_execution_request.v1",
        "scenario_batch": {
            "target_scenario_eval_run_count": 500,
            "batching_reason": "robot_team_grade_mujoco_review",
        },
    }

    matrix = build_scenario_eval_matrix(
        capture_root=capture_root,
        job_dir=capture_root / "pipeline" / "robot_eval_jobs" / "job-500-matrix",
        job_request=request,
        generated_at="2026-06-14T00:00:00Z",
    )

    runs = matrix["runs"]
    assert matrix["status"] == "completed"
    assert matrix["base_scenario_eval_run_count"] == len(POLICY_REFERENCE_VARIATION_NAMES)
    assert matrix["target_scenario_eval_run_count"] == 500
    assert matrix["target_scenario_eval_run_count_source"] == (
        "execution_request.scenario_batch.target_scenario_eval_run_count"
    )
    assert matrix["scenario_eval_batch_expanded"] is True
    assert matrix["target_scenario_eval_run_count_satisfied"] is True
    assert matrix["scenario_eval_run_count"] == 500
    assert matrix["semantic_spawn_target_coverage_complete"] is True
    assert matrix["deterministic_fallback_spawn_target_run_count"] == 0
    assert matrix["fallback_spawn_target_run_ids"] == []
    assert len({run["scenario_eval_run_id"] for run in runs}) == 500
    assert set(matrix["variation_names_covered"]) == set(POLICY_REFERENCE_VARIATION_NAMES)
    assert matrix["episode_authoring_contract"][
        "spawn_target_variation_seed_handling"
    ] == "deterministic_frozen_matrix_rows"
    assert matrix["episode_authoring_contract"][
        "runtime_spawn_goal_variation_mutation_allowed"
    ] is False
    first_run = runs[0]
    repeated_run = runs[len(POLICY_REFERENCE_VARIATION_NAMES)]
    last_run = runs[-1]
    for run in (first_run, repeated_run, last_run):
        assert run["spawn_goal_variation_seed_frozen"] is True
        assert isinstance(run["deterministic_seed"], int)
        assert run["deterministic_seed"] == run["episode_seed"]
        assert len(run["spawn_pose"]) == 3
        assert len(run["target_pose"]) == 3
        assert run["validated_spawn_target_pair"] is True
        assert run["deterministic_spawn_target_fallback_used"] is False
        assert run["deterministic_scenario_parameters"][
            "semantic_spawn_target_validated"
        ] is True
        assert run["concrete_mutation"]["spawn_pose"] == run["spawn_pose"]
        assert run["concrete_mutation"]["target_pose"] == run["target_pose"]
        assert run["deterministic_scenario_parameters"][
            "runtime_spawn_goal_variation_mutation_allowed"
        ] is False
    assert repeated_run["batch_repeat_index"] == 1
    assert repeated_run["batch_source_scenario_eval_run_id"] == first_run[
        "scenario_eval_run_id"
    ]
    assert repeated_run["scenario_eval_run_id"] != first_run["scenario_eval_run_id"]
    assert last_run["batch_ordinal"] == 500

    repeat_matrix = build_scenario_eval_matrix(
        capture_root=capture_root,
        job_dir=capture_root / "pipeline" / "robot_eval_jobs" / "job-500-matrix-repeat",
        job_request=request,
        generated_at="2026-06-14T00:00:00Z",
    )
    assert repeat_matrix["deterministic_fingerprint"] == matrix["deterministic_fingerprint"]
    assert repeat_matrix["runs"][0] == first_run
    assert repeat_matrix["runs"][-1] == last_run


def test_minimal_webapp_request_gets_beta_enrichment_without_overwriting_source(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": "job-minimal-webapp",
        "buyer_request_id": "buyer-request-prod-001",
        "site_package": {
            "capture_root": str(capture_root),
            "site_submission_id": "site-submission-prod-001",
            "capture_job_id": "capture-job-prod-001",
            "capture_id": "capture-1",
        },
        "source_kind": "webapp_route_forwarding_proof",
    }
    request_path = tmp_path / "minimal-webapp-request.json"
    _write_json(request_path, request)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-minimal-webapp",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-minimal-webapp"
    source_request = _read_json(job_dir / "job_request_source.json")
    enriched_request = _read_json(job_dir / "job_request.json")
    enrichment = _read_json(job_dir / "job_request_enrichment_manifest.json")
    validation = _read_json(job_dir / "job_validation.json")
    policy_manifest = _read_json(job_dir / "policy_package_manifest.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert "customer" not in source_request
    assert "robot_profile" not in source_request
    assert "requested_tasks" not in source_request
    assert "policy_package" not in source_request
    assert enrichment["status"] == "enriched"
    assert enrichment["added_fields"] == [
        "customer",
        "robot_profile",
        "requested_tasks",
        "policy_package",
    ]
    assert enrichment["claim_boundary"].startswith("Fills missing beta orchestration inputs")
    assert enriched_request["customer"]["source"] == "pipeline_beta_request_enrichment"
    assert enriched_request["robot_profile"]["robot_profile_id"]
    assert enriched_request["requested_tasks"]
    assert enriched_request["policy_package"]["high_level_skill_trace"]["reference_only"] is True
    assert validation["status"] != "blocked"
    assert "missing_customer" not in validation["blockers"]
    assert "missing_robot_profile" not in validation["blockers"]
    assert "missing_requested_tasks" not in validation["blockers"]
    assert "missing_policy_evidence" not in validation["blockers"]
    assert policy_manifest["status"] == "review_required"
    assert policy_manifest["selected_modalities"] == ["high_level_skill_trace"]
    assert run_manifest["artifacts"]["job_request_source"] == "job_request_source.json"
    assert run_manifest["artifacts"]["job_request_enrichment_manifest"] == (
        "job_request_enrichment_manifest.json"
    )


def _write_minimal_glb_with_accessor_bounds(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "worldlabs_collider", "mesh": 0}],
        "meshes": [
            {
                "name": "scene_collider",
                "primitives": [{"attributes": {"POSITION": 0}}],
            }
        ],
        "accessors": [
            {
                "type": "VEC3",
                "componentType": 5126,
                "count": 8,
                "min": [-1.0, -2.0, 0.0],
                "max": [3.0, 4.0, 1.5],
            }
        ],
    }
    raw_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    raw_json += b" " * ((4 - (len(raw_json) % 4)) % 4)
    total_length = 12 + 8 + len(raw_json)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<II", len(raw_json), 0x4E4F534A)
        + raw_json
    )


def _write_cpu_preflight_ready_scene_asset(capture_root: Path) -> None:
    _write_minimal_glb_with_accessor_bounds(
        capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    )


POLICY_REFERENCE_VARIATION_NAMES = [
    "lighting_variation",
    "object_rotation",
    "cart_shifted",
    "blocked_path",
    "human_crossing",
    "forklift_nearby",
    "occlusion",
    "glare",
    "missing_label",
    "wrong_object_nearby",
    "narrow_approach_angle",
]


def _variation_detail_fields(variation_name: str) -> dict[str, object]:
    return {
        "concrete_mutation": {
            "fixture_variation": {
                "variation_name": variation_name,
                "parameter": "deterministic_fixture_mutation",
            }
        },
        "engine_mutations": {
            "fixture": {
                "operation_count": 1,
                "operations": [
                    {
                        "operation": "fixture.apply_variation",
                        "variation_name": variation_name,
                        "parameters": {"variation_name": variation_name},
                    }
                ],
            }
        },
    }


def _write_policy_reference_variation_instances(capture_root: Path) -> None:
    _write_json(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json",
        {
            "schema_version": "scenario_variation_instances.v1",
            "status": "completed",
            "required_variation_names": list(POLICY_REFERENCE_VARIATION_NAMES),
            "instance_count": len(POLICY_REFERENCE_VARIATION_NAMES),
            "instances": [
                {
                    "instance_id": _scenario_variation_instance_id(variation_name),
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": variation_name,
                    **_variation_detail_fields(variation_name),
                }
                for variation_name in POLICY_REFERENCE_VARIATION_NAMES
            ],
        },
    )


def _scenario_eval_run_id(variation_name: str, index: int) -> str:
    return (
        f"place_return_in_bin_scenario_place_return_in_bin_mobile_{variation_name}_run_{index:04d}"
    )


def _scenario_variation_instance_id(variation_name: str) -> str:
    return f"variation_place_return_in_bin_scenario_place_return_in_bin_mobile_{variation_name}"


def _policy_reference_attempts(
    *,
    policy_id: str,
    variation_names: list[str] | None = None,
) -> list[dict[str, object]]:
    names = variation_names or POLICY_REFERENCE_VARIATION_NAMES
    return [
        {
            "attempt_id": f"{policy_id}-{index:04d}",
            "scenario_eval_run_id": _scenario_eval_run_id(variation_name, index),
            "scenario_variation_instance_id": _scenario_variation_instance_id(variation_name),
            "variation_name": variation_name,
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "task_id": "place_return_in_bin",
            "policy_id": policy_id,
            "status": "completed",
            "success": True,
            "actions": [{"t": 0.0, "action": "navigate"}, {"t": 1.0, "action": "place"}],
            "metrics": {"cycle_time_seconds": 12.0 + index},
        }
        for index, variation_name in enumerate(names, start=1)
    ]


def _write_real_robot_pov_manifest(capture_root: Path) -> None:
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "real_robot_pov_manifest.json",
        {
            "schema_version": "real_robot_pov_manifest.v1",
            "owner_system": "robot-team-owner-system",
            "timestamp_alignment": "aligned_to_scenario_eval_run",
            "records": [
                {
                    "evidence_id": f"real-pov-{index:04d}",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "scenario_eval_run_id": _scenario_eval_run_id(variation_name, index),
                    "scenario_variation_instance_id": _scenario_variation_instance_id(
                        variation_name
                    ),
                    "variation_name": variation_name,
                    "robot_camera_video_uri": f"owner://pov/{variation_name}.mp4",
                    "action_log_uri": f"owner://actions/{variation_name}.jsonl",
                    "robot_state_log_uri": f"owner://state/{variation_name}.jsonl",
                    "owner_evidence_refs": {
                        "camera": f"owner://pov/{variation_name}.mp4",
                        "action_log": f"owner://actions/{variation_name}.jsonl",
                    },
                    "operator_attestation": {
                        "attested_by": "robot-team-ops",
                        "attestation": "Robot POV and action log are aligned to this eval run.",
                    },
                }
                for index, variation_name in enumerate(POLICY_REFERENCE_VARIATION_NAMES, start=1)
            ],
        },
    )


def _full_job_request(
    capture_root: Path,
    *,
    operation: str = "evaluate_only",
    rights_allowed: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "robot_eval_job_request.v1",
        "customer": {"id": "robot-team-a", "name": "Robot Team A"},
        "site_package": {
            "capture_root": str(capture_root),
            "site_id": "site-1",
            "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
        },
        "requested_tasks": [
            {
                "task_id": "place_return_in_bin",
                "scenario_ids": ["scenario_place_return_in_bin_mobile"],
            }
        ],
        "robot_profile": {
            "robot_profile_id": "mobile_manipulator_rgb_v1",
            "embodiment": "mobile_manipulator",
            "sensors": ["rgb", "depth"],
        },
        "policy_package": {
            "policy_api_endpoint": {
                "endpoint_url": "https://robot-team.example/policy",
                "observation_schema_ref": "schemas/obs-v1.json",
                "action_schema_ref": "schemas/action-v1.json",
            },
            "docker_container": {
                "image_ref": "registry.example/robot/policy:2026-06-04",
                "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            },
            "recorded_action_trace": {
                "trace_manifest_uri": "gs://robot-team/traces/trace-manifest.json",
                "checksum": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "timestamp_alignment": "aligned_to_capture_timestamps",
            },
            "high_level_skill_trace": {
                "skill_taxonomy_version": "skills-v1",
                "ordered_skill_sequence": ["navigate", "pick", "place"],
            },
            "teleop_demo": {
                "demo_artifact_uri": "gs://robot-team/demos/demo-1.json",
                "rights_privacy_attestation": "deidentified_operator_approved",
            },
            "sim_controller_plugin": {
                "simulator_framework": "fixture",
                "plugin_uri": "gs://robot-team/plugins/controller-fixture.json",
            },
        },
        "operation": operation,
        "simulator_preference": "fixture",
        "cosmos_training_preference": {"mode": "export_only"},
        "budget": {"budget_usd": 5.0, "timeout_seconds": 30},
        "rights_privacy_scope": {
            "status": "cleared_for_robot_eval" if rights_allowed else "blocked",
            "external_use_allowed": rights_allowed,
            "privacy_scope": "derived_deidentified_environment",
        },
        "owner_system": {"name": "robot-team-a", "request_id": "req-1"},
        "provenance": {
            "submitted_at": "2026-06-04T00:00:00+00:00",
            "timestamp_alignment": "trace_timestamps_aligned_to_capture",
        },
    }


def _webapp_execution_request() -> dict[str, object]:
    return {
        "schema_version": "blueprint.robot_eval_execution_request.v1",
        "webapp_role": "queue_and_forward_only",
        "scheduler_owner": "BlueprintCapturePipeline",
        "queueing": {
            "mode": "async_job",
            "customer_response": "job_id_and_status_only",
            "web_request_must_not_wait_for_simulator": True,
        },
        "preflight": {
            "cpu_preflight_required_before_gpu": True,
            "blocks_gpu_when_missing": True,
            "required_artifacts": [
                "scene_asset_inventory",
                "scene_asset_dependency_audit",
                "cpu_preflight_scorecard",
                "episode_spec_manifest",
                "gpu_handoff_packet",
            ],
        },
        "simulator_routing": {
            "requested_backend": "pipeline_selected",
            "allowed_backends": ["mujoco", "isaac_sim", "isaac_lab_arena", "fixture"],
            "default_first_pass_backend": "mujoco",
            "default_first_gpu_backend": "mujoco",
            "proxy_backends": ["mujoco", "fixture"],
            "escalation_backends": ["isaac_sim", "isaac_lab_arena"],
            "selection_policy": {
                "schema_version": "robot_eval_simulator_selection_policy.v1",
                "mode": "mujoco_first_unless_proof_requires_isaac",
                "first_pass_backend": "mujoco",
                "use_mujoco_when": [
                    "cheapest_first_real_simulator_pass",
                    "fast_cpu_or_low_cost_owner_runtime",
                    "compatible_mjcf_robot_asset_or_default_unitree_g1_smoke",
                    "early_policy_and_spawn_smoke_before_gpu_spend",
                ],
                "escalate_to_isaac_when": [
                    "rich_usd_or_openusd_scene_load_required",
                    "isaac_robot_asset_proof_required",
                    "rtx_sensor_or_camera_rendering_required",
                    "contact_or_physics_validation_requires_isaac_stack",
                ],
                "use_isaac_lab_arena_when": [
                    "isaac_lab_arena_batch_rollouts_required",
                    "large_scenario_matrix_or_sharded_eval_required",
                    "owner_arena_result_ingest_required",
                ],
            },
            "proof_boundaries": {
                "webapp_request_selects_policy_not_execution": True,
                "mujoco_proof_does_not_clear_isaac_sim_gate": True,
                "simulator_policy_does_not_prove_robot_readiness": True,
            },
            "isaac_gpu_constraint": "rtx_rt_core_required_no_a100_h100",
        },
        "gpu_allocation": {
            "mode": "on_demand_with_optional_warm_pool",
            "allocation_owner": "BlueprintCapturePipeline_or_owner_gpu_worker",
            "allocation_allowed_by_webapp": False,
            "gpu_spend_approved": False,
            "max_budget_usd": 0,
            "hard_timeout_seconds": 120,
            "idle_shutdown_required": True,
            "persistent_cache_recommended": True,
        },
        "artifact_contract": {
            "expected_outputs": [
                "job_run_manifest",
                "proof_boundary",
                "metrics",
                "trace",
                "simulator_pov",
                "stdout_log",
                "stderr_log",
            ],
            "simulator_execution_proven_by_webapp": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _webapp_execution_request_without_cpu_preflight() -> dict[str, object]:
    request = _webapp_execution_request()
    request["preflight"] = {
        "cpu_preflight_required_before_gpu": False,
        "blocks_gpu_when_missing": True,
        "required_artifacts": [],
    }
    return request


def test_simulator_selection_policy_defaults_to_mujoco_and_escalates_only_for_named_proof_classes(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    request = _full_job_request(capture_root)
    request["simulator_preference"] = "mujoco_first"
    request["execution_request"] = _webapp_execution_request()

    default_policy = resolve_simulator_selection_policy(request, selected_simulator="fixture")

    assert default_policy["mode"] == "mujoco_first_unless_proof_requires_isaac"
    assert default_policy["recommended_backend"] == "mujoco"
    assert default_policy["mujoco_first_applies"] is True
    assert default_policy["selected_backend"] == "fixture"
    assert default_policy["selected_backend_matches_recommendation"] is False
    assert "fixture_local_loop_does_not_satisfy_customer_eval_backend_policy" in default_policy[
        "non_blocking_warnings"
    ]
    assert default_policy["proof_boundary"]["mujoco_proof_does_not_clear_isaac_sim_gate"] is True

    isaac_request = {
        **request,
        "execution_request": {
            **_webapp_execution_request(),
            "simulator_routing": {
                **_webapp_execution_request()["simulator_routing"],
                "required_proof_classes": ["isaac_robot_asset_proof_required"],
            },
        },
    }
    isaac_policy = resolve_simulator_selection_policy(isaac_request, selected_simulator="mujoco")

    assert isaac_policy["recommended_backend"] == "isaac_sim"
    assert isaac_policy["escalation_required"] is True
    assert "isaac_robot_asset_proof_required" in isaac_policy["recommendation_reasons"]

    arena_request = {
        **request,
        "execution_request": {
            **_webapp_execution_request(),
            "simulator_routing": {
                **_webapp_execution_request()["simulator_routing"],
                "required_proof_classes": ["large_scenario_matrix_or_sharded_eval_required"],
            },
        },
    }
    arena_policy = resolve_simulator_selection_policy(
        arena_request,
        selected_simulator="isaac_lab_arena",
    )

    assert arena_policy["recommended_backend"] == "isaac_lab_arena"
    assert arena_policy["selected_backend_matches_recommendation"] is True


def test_robot_eval_job_fixture_path_runs_end_to_end_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-fixture-success",
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-fixture-success"
    required_outputs = {
        "job_request.json",
        "job_validation.json",
        "job_plan.json",
        "agent_orchestration_plan.json",
        "scheduler_decision.json",
        "worker_launch_plan.json",
        "worker_manifest.json",
        "gpu_provisioning_request.json",
        "gpu_provider_launch_request.json",
        "gpu_cost_control_ledger.json",
        "gpu_provisioning_result.json",
        "simulator_service_request.json",
        "simulator_service_result.json",
        "scenario_eval_matrix.json",
        "policy_package_manifest.json",
        "robot_pov_observation_manifest.json",
        "robot_pov_observations.jsonl",
        "robot_pov_frame_sequence_manifest.json",
        "robot_pov_render_storyboard.json",
        "policy_execution_manifest.json",
        "policy_execution_trace.json",
        "policy_execution_trace.jsonl",
        "training_request.json",
        "training_result.json",
        "evaluation_request.json",
        "evaluation_result.json",
        "robot_eval_report.json",
        "robot_eval_report.md",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
        "deployment_outcome_intake_manifest.json",
        "deployment_outcome_ledger.json",
        "sim_vs_real_calibration_report.json",
        "prediction_vs_actual_deployment_summary.json",
        "real_world_validation_followup_plan.json",
        "real_world_validation_followup_request_queue.json",
        "live_eval_closure_manifest.json",
        "post_training_data_package_export_manifest.json",
        "proof_boundary.json",
        "startup_architecture_audit.json",
        "job_run_manifest.json",
    }

    assert result["status"] == "fixture_evaluation_completed"
    assert required_outputs.issubset({path.name for path in job_dir.iterdir()})
    assert not (job_dir / "blocked_manifest.json").exists()

    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    startup_audit = _read_json(job_dir / "startup_architecture_audit.json")
    validation = _read_json(job_dir / "job_validation.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    scenario_eval_matrix = _read_json(job_dir / "scenario_eval_matrix.json")
    evaluation = _read_json(job_dir / "evaluation_result.json")
    robot_eval_report = _read_json(job_dir / "robot_eval_report.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    robot_pov = _read_json(job_dir / "robot_pov_observation_manifest.json")
    robot_pov_frames = _read_json(job_dir / "robot_pov_frame_sequence_manifest.json")
    robot_pov_storyboard = _read_json(job_dir / "robot_pov_render_storyboard.json")
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    data_package_export = _read_json(job_dir / "post_training_data_package_export_manifest.json")
    optional_exports = _read_json(job_dir / "optional_export_manifest.json")
    package_index = _read_json(job_dir / "package_index.json")
    checksums = _read_json(job_dir / "checksums.json")
    archive_manifest = _read_json(job_dir / "archive_manifest.json")

    assert validation["status"] == "passed"
    assert provisioning["status"] == "allocated"
    assert provisioning["provider"] == "fixture_local"
    assert simulator_result["status"] == "completed"
    assert simulator_result["framework"] == "fixture"
    assert simulator_result["simulator_execution_proven"] is False
    assert scenario_eval_matrix["status"] == "completed"
    assert scenario_eval_matrix["scenario_eval_run_count"] == 11
    assert set(scenario_eval_matrix["variation_names_covered"]) == {
        "lighting_variation",
        "object_rotation",
        "cart_shifted",
        "blocked_path",
        "human_crossing",
        "forklift_nearby",
        "occlusion",
        "glare",
        "missing_label",
        "wrong_object_nearby",
        "narrow_approach_angle",
    }
    assert evaluation["status"] == "completed"
    assert set(evaluation["standard_policy_scorecard"]) == {
        "success_rate",
        "cycle_time",
        "intervention_rate",
        "unsafe_proximity",
        "collision_risk",
        "object_drop",
        "wrong_object",
        "timeout",
        "recovery_success",
        "world_model_uncertainty",
        "sim_vs_real_calibration_score",
    }
    assert evaluation["standard_policy_scorecard"]["success_rate"] == 1.0
    assert evaluation["standard_policy_scorecard"]["cycle_time"]["mean_seconds"] == 14.0
    assert (
        evaluation["standard_policy_scorecard"]["cycle_time"]["sample_count"]
        == (scenario_eval_matrix["scenario_eval_run_count"])
    )
    assert evaluation["standard_policy_scorecard"]["intervention_rate"] == 0.0
    assert evaluation["standard_policy_scorecard"]["sim_vs_real_calibration_score"] is None
    assert robot_eval_report["schema_version"] == "robot_eval_job_report.v1"
    assert robot_eval_report["status"] == "generated"
    assert robot_eval_report["job_status"] == "fixture_evaluation_completed"
    assert (
        robot_eval_report["scenario_eval"]["scenario_eval_run_count"]
        == (scenario_eval_matrix["scenario_eval_run_count"])
    )
    assert robot_eval_report["evaluator_scores"]["success_rate"] == 1.0
    assert robot_eval_report["evaluator_scores"]["cycle_time"]["mean_seconds"] == 14.0
    assert robot_eval_report["policy_interface"]["executed_modalities"] == [
        "high_level_skill_trace"
    ]
    assert set(robot_eval_report["policy_interface"]["supported_modalities"]) == {
        "policy_api_endpoint",
        "docker_container",
        "recorded_action_trace",
        "high_level_skill_trace",
        "teleop_demo",
        "sim_controller_plugin",
    }
    assert robot_eval_report["live_eval_closure"]["status"] == (
        "local_artifacts_ready_live_external_blocked"
    )
    assert robot_eval_report["requirement_coverage"]["schema_version"] == (
        "live_robot_eval_requirement_coverage.v1"
    )
    assert robot_eval_report["proof_boundary"]["robot_readiness_proven"] is False
    assert "report_generated" in robot_eval_report["neutral_eval_harness_flow"]
    assert "# Robot Eval Report" in (job_dir / "robot_eval_report.md").read_text(encoding="utf-8")
    assert trace["attempt_count"] == scenario_eval_matrix["scenario_eval_run_count"]
    assert {attempt["scenario_eval_run_id"] for attempt in trace["attempts"]} == {
        run["scenario_eval_run_id"] for run in scenario_eval_matrix["runs"]
    }
    assert trace["attempts"][0]["success"] is True
    assert robot_pov["status"] == "completed"
    assert robot_pov["observation_count"] == scenario_eval_matrix["scenario_eval_run_count"]
    assert (
        robot_pov["local_render_frame_count"] >= 3 * scenario_eval_matrix["scenario_eval_run_count"]
    )
    assert robot_pov["robot_pov_evidence_proven"] is False
    assert robot_pov_frames["status"] == "completed"
    assert robot_pov_frames["sequence_count"] == scenario_eval_matrix["scenario_eval_run_count"]
    assert robot_pov_frames["total_frame_count"] >= (
        3 * scenario_eval_matrix["scenario_eval_run_count"]
    )
    assert all(
        (job_dir / path).is_file()
        for sequence in robot_pov_frames["sequences"]
        for path in sequence["frame_paths"]
    )
    assert robot_pov_storyboard["status"] == "completed"
    assert (
        robot_pov_storyboard["storyboard_count"] == scenario_eval_matrix["scenario_eval_run_count"]
    )
    assert robot_pov_storyboard["local_robot_pov_render_generated"] is True
    assert robot_pov_storyboard["robot_pov_evidence_proven"] is False
    assert policy_execution["status"] == "completed"
    assert (
        policy_execution["modality_results"]["high_level_skill_trace"]["attempt_count"]
        == (scenario_eval_matrix["scenario_eval_run_count"])
    )
    assert policy_execution["robot_policy_execution_proven"] is False
    assert deployment["status"] == "blocked_missing_real_world_outcomes"
    assert run_manifest["state"] == "completed"
    assert run_manifest["startup_architecture_audit_status"] == "passed"
    assert run_manifest["startup_architecture_audit_path"] == "startup_architecture_audit.json"
    assert run_manifest["startup_architecture_compliant"] is True
    assert run_manifest["artifacts"]["startup_architecture_audit"] == (
        "startup_architecture_audit.json"
    )
    assert startup_audit["architecture_compliant"] is True
    assert startup_audit["proof_boundary"]["simulator_execution_proven"] is False
    assert startup_audit["proof_boundary"]["robot_readiness_proven"] is False
    assert startup_audit["proof_boundary"]["public_claim_upgrade_allowed"] is False
    assert run_manifest["scenario_eval_matrix_status"] == "completed"
    assert (
        run_manifest["scenario_eval_run_count"] == scenario_eval_matrix["scenario_eval_run_count"]
    )
    assert run_manifest["scene_asset_preflight_status"] == "blocked"
    assert run_manifest["episode_spec_status"] == "compiled_review_required"
    assert run_manifest["cpu_simulator_preflight_status"] == (
        "ready_blocked_optional_dependencies_or_gates"
    )
    assert run_manifest["cpu_preflight_artifacts"]["episode_spec"] == (
        "../simulation_automation/episode_spec.v1.json"
    )
    assert run_manifest["public_claim_upgrade_allowed"] is False
    assert run_manifest["live_eval_closure_status"] == (
        "local_artifacts_ready_live_external_blocked"
    )
    assert run_manifest["live_end_to_end_verified"] is False
    assert (
        "live_simulator_execution:live_simulator_execution_not_proven"
        in run_manifest["live_eval_closure_blockers"]
    )
    assert proof_boundary["robot_readiness_proven"] is False
    assert proof_boundary["robot_policy_execution_proven"] is False
    assert proof_boundary["public_claim_upgrade_allowed"] is False
    assert proof_boundary["fixture_only_proof"] is True
    assert data_package_export["status"] == "export_ready_review_required"
    assert data_package_export["package_type"] == "post_training_data_package"
    assert data_package_export["included_artifacts"]["normalized_attempt_trace"] == (
        "normalized_attempt_trace.json"
    )
    assert data_package_export["included_artifacts"]["robot_pov_observation_manifest"] == (
        "robot_pov_observation_manifest.json"
    )
    assert data_package_export["included_artifacts"]["scenario_eval_matrix"] == (
        "scenario_eval_matrix.json"
    )
    assert data_package_export["export_policy"]["scenario_eval_matrix_included"] is True
    assert data_package_export["included_artifacts"]["policy_execution_trace"] == (
        "policy_execution_trace.json"
    )
    assert data_package_export["included_artifacts"]["deployment_outcome_intake_manifest"] == (
        "deployment_outcome_intake_manifest.json"
    )
    assert data_package_export["included_artifacts"]["live_eval_closure_manifest"] == (
        "live_eval_closure_manifest.json"
    )
    assert data_package_export["included_artifacts"]["robot_eval_report"] == (
        "robot_eval_report.json"
    )
    assert data_package_export["included_artifacts"]["robot_eval_report_markdown"] == (
        "robot_eval_report.md"
    )
    assert data_package_export["export_policy"]["deployment_outcome_intake_included"] is True
    assert data_package_export["export_policy"]["live_eval_closure_included"] is True
    assert data_package_export["export_policy"]["robot_eval_report_included"] is True
    assert data_package_export["claim_boundary"]["robot_readiness_proven"] is False
    assert live_closure["status"] == "local_artifacts_ready_live_external_blocked"
    assert live_closure["repo_local_artifacts_ready"] is True
    assert live_closure["live_external_ready"] is False
    assert live_closure["gates"]["scenario_library"]["passed"] is True
    assert live_closure["gates"]["robot_pov_generation"]["passed"] is True
    assert live_closure["gates"]["report_generation"]["passed"] is True
    requirement_coverage = live_closure["requirement_coverage"]
    assert requirement_coverage["schema_version"] == "live_robot_eval_requirement_coverage.v1"
    assert set(requirement_coverage["repo_local_requirement_ids"]) == {
        "site_capture",
        "task_definitions",
        "scenario_library",
        "scenario_variation_families",
        "robot_pov_generation",
        "scenario_eval_suite",
        "failure_labels",
        "evaluation_methodology",
        "robot_team_interface",
        "plugin_world_sim_engines",
        "neutral_eval_harness_report",
    }
    assert set(requirement_coverage["live_external_requirement_ids"]) == set()
    assert not (
        set(requirement_coverage["repo_local_requirement_ids"])
        - set(requirement_coverage["passed_requirement_ids"])
    )
    assert set(requirement_coverage["blocked_requirement_ids"]).isdisjoint(
        {
            "real_robot_pov_evidence",
            "real_world_validation_loop",
            "predicted_vs_actual_deployment_data",
        }
    )
    for export_name in ("rlds", "lerobot", "hdf5", "parquet", "video_bundle"):
        export_entry = optional_exports["formats"][export_name]
        assert export_entry["format_written"] is True
        assert export_entry["path"]
        assert (job_dir / export_entry["path"]).is_file()
    assert optional_exports["formats"]["rlds"]["status"] == "written_jsonl"
    assert optional_exports["formats"]["lerobot"]["status"] == "written_jsonl"
    assert optional_exports["formats"]["hdf5"]["status"] in {
        "written_native",
        "written_jsonl_fallback",
    }
    assert optional_exports["formats"]["parquet"]["status"] in {
        "written_native",
        "written_jsonl_fallback",
    }
    assert package_index["files"]["rlds_episodes"].startswith("exports/rlds/")
    assert package_index["files"]["lerobot_episodes"].startswith("exports/lerobot/")
    assert package_index["files"]["video_bundle_manifest"] == (
        "exports/video_bundle/clips_manifest.json"
    )
    assert checksums["files"]["rlds_episodes"]["exists"] is True
    assert "exports/rlds/episodes.jsonl" in archive_manifest["included_files"]


def test_robot_eval_job_blocks_unknown_requested_scenario_id(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request = _full_job_request(capture_root)
    request["requested_tasks"] = [
        {
            "task_id": "place_return_in_bin",
            "scenario_ids": ["scenario_not_in_library"],
        }
    ]
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-unknown-scenario",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    matrix = _read_json(job_dir / "scenario_eval_matrix.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    scenario_gate = live_closure["gates"]["scenario_eval_suite"]

    assert result["status"] == "blocked"
    assert matrix["status"] == "blocked_invalid_requested_scope"
    assert matrix["scenario_eval_run_count"] == 0
    assert matrix["unknown_requested_scenario_ids"] == ["scenario_not_in_library"]
    assert "scenario_eval_matrix_unknown_requested_scenarios" in matrix["blockers"]
    assert "scenario_eval_matrix_blocked" in blocked["blockers"]
    assert "scenario_eval_matrix_unknown_requested_scenarios" in blocked["missing_inputs"]
    assert "scenario_eval_matrix_not_completed" in scenario_gate["blockers"]
    assert "scenario_eval_matrix_unknown_requested_scenarios" in scenario_gate["blockers"]


def test_robot_eval_job_blocks_unknown_requested_task_without_defaulting_to_all_scenarios(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request = _full_job_request(capture_root)
    request["requested_tasks"] = [{"task_id": "unknown_task"}]
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-unknown-task",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    matrix = _read_json(job_dir / "scenario_eval_matrix.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    scenario_gate = live_closure["gates"]["scenario_eval_suite"]

    assert result["status"] == "blocked"
    assert matrix["status"] == "blocked_invalid_requested_scope"
    assert matrix["requested_scenario_count"] == 0
    assert matrix["scenario_eval_run_count"] == 0
    assert matrix["unknown_requested_task_ids"] == ["unknown_task"]
    assert "scenario_eval_matrix_unknown_requested_tasks" in matrix["blockers"]
    assert "scenario_eval_matrix_missing_requested_scenarios" in matrix["blockers"]
    assert "scenario_eval_matrix_unknown_requested_tasks" in scenario_gate["blockers"]


def test_robot_eval_job_live_closure_verifies_complete_external_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    webapp_ids = {
        "site_submission_id": "site-submission-live-001",
        "request_id": "request-live-001",
        "buyer_request_id": "buyer-request-live-001",
        "capture_job_id": "capture-job-live-001",
    }
    descriptor = _read_json(capture_root / "capture_descriptor.json")
    descriptor.update(webapp_ids)
    _write_json(capture_root / "capture_descriptor.json", descriptor)
    raw_manifest = _read_json(capture_root / "raw" / "manifest.json")
    raw_manifest["upstream_handoff"] = dict(webapp_ids)
    _write_json(capture_root / "raw" / "manifest.json", raw_manifest)
    _write_robot_eval_cards(capture_root)
    _write_real_robot_pov_manifest(capture_root)

    simulator_writer = tmp_path / "write_simulator_output.py"
    simulator_writer.write_text(
        "\n".join(
            [
                "import json, os",
                "with open(os.environ['BLUEPRINT_SCENARIO_EVAL_MATRIX'], encoding='utf-8') as f:",
                "    matrix = json.load(f)",
                "attempts = []",
                "for index, run in enumerate(matrix['runs'], start=1):",
                "    attempts.append({",
                "      'attempt_id': f'isaac-attempt-{index}',",
                "      'episode_id': f'isaac-episode-{index}',",
                "      'scenario_id': run['scenario_id'],",
                "      'scenario_run_id': f\"{run['scenario_eval_run_id']}__isaac_sim\",",
                "      'scenario_eval_run_id': run['scenario_eval_run_id'],",
                "      'scenario_variation_instance_id': run.get('scenario_variation_instance_id'),",
                "      'variation_name': run.get('variation_name'),",
                "      'task_id': run['task_id'],",
                "      'policy_id': 'policy-live-command',",
                "      'status': 'completed',",
                "      'success': True,",
                "      'metrics': {'cycle_time_seconds': 11.25, 'intervention_count': 0},",
                "      'actions': [{'t': 0.0, 'action': 'navigate'}, {'t': 1.0, 'action': 'place'}],",
                "      'contact_trace': [{'status': 'clear'}],",
                "      'safety_events': []",
                "    })",
                "payload = {",
                "  'attempts': attempts,",
                "  'isaac_sim_execution_proven': True,",
                "  'isaac_robot_asset_execution_proven': True,",
                "  'unitree_g1_asset_spawned': True,",
                "  'robot_asset': {",
                "    'name': 'Unitree G1',",
                "    'uri_or_path': 'Robots/Unitree/G1/g1.usd',",
                "    'source': 'isaac_sim_robot_assets',",
                "    'asset_class': 'humanoid',",
                "  },",
                "}",
                "with open(os.environ['BLUEPRINT_SIMULATOR_OUTPUT'], 'w', encoding='utf-8') as f:",
                "    json.dump(payload, f)",
            ]
        ),
        encoding="utf-8",
    )
    policy_writer = tmp_path / "write_policy_output.py"
    policy_writer.write_text(
        "\n".join(
            [
                "import json, os",
                "with open(os.environ['BLUEPRINT_POLICY_OBSERVATION_MANIFEST'], encoding='utf-8') as f:",
                "    observations = json.load(f)['observations']",
                "attempts = []",
                "for index, obs in enumerate(observations, start=1):",
                "    attempts.append({",
                "      'attempt_id': f'policy-attempt-{index}',",
                "      'observation_id': obs['observation_id'],",
                "      'scenario_id': obs['scenario_id'],",
                "      'scenario_eval_run_id': obs.get('scenario_eval_run_id'),",
                "      'scenario_variation_instance_id': obs.get('scenario_variation_instance_id'),",
                "      'variation_name': obs.get('variation_name'),",
                "      'task_id': obs['task_id'],",
                "      'policy_id': 'policy-live-command',",
                "      'status': 'completed',",
                "      'success': True,",
                "      'actions': [{'t': 0.0, 'action': 'pick'}, {'t': 1.0, 'action': 'place'}]",
                "    })",
                "payload = {'attempts': attempts}",
                "with open(os.environ['BLUEPRINT_POLICY_EXECUTION_OUTPUT'], 'w', encoding='utf-8') as f:",
                "    json.dump(payload, f)",
            ]
        ),
        encoding="utf-8",
    )
    evidence_root = tmp_path / "owner-evidence"
    methodology = evidence_root / "methodology.md"
    contact_validation = evidence_root / "contact_validation.json"
    safety_validation = evidence_root / "safety_validation.json"
    review_evidence = evidence_root / "review_acceptance.json"
    rights_clearance = evidence_root / "rights_clearance.json"
    for path, payload in (
        (methodology, "accepted safety/contact methodology\n"),
        (contact_validation, json.dumps({"status": "validated", "contact_clear": True})),
        (safety_validation, json.dumps({"status": "validated", "safety_validated": True})),
        (review_evidence, json.dumps({"status": "accepted", "reviewer": "owner-reviewer"})),
        (rights_clearance, json.dumps({"status": "accepted", "external_use_allowed": True})),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    request = _full_job_request(capture_root)
    request.update(webapp_ids)
    request["simulator_preference"] = "isaac_sim"
    request["actual_outcomes"] = {
        "records": [
            {
                "outcome_id": "pilot-live-1",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "scenario_eval_run_id": _scenario_eval_run_id("lighting_variation", 1),
                "scenario_variation_instance_id": _scenario_variation_instance_id(
                    "lighting_variation"
                ),
                "policy_id": "policy-live-command",
                "actual_success": True,
                "cycle_time_seconds": 11.5,
                "intervention_count": 0,
                "failure_mode_ids": [],
                "evidence_refs": {"pilot_log": "owner://pilot/live-1"},
            }
        ]
    }
    staged_closure_evidence_path = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "job-live-closure"
        / "live_eval_closure_evidence.json"
    )
    _write_json(
        staged_closure_evidence_path,
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "rights_privacy": {
                "accepted": True,
                "clearance_uri_or_path": str(rights_clearance),
                "operator_attestation": {
                    "attested_by": "rights-owner",
                    "attestation": "Owner approved deidentified external robot-eval use.",
                },
            },
            "review_acceptance": {
                "accepted": True,
                "reviewer": "owner-reviewer",
                "evidence_uri_or_path": str(review_evidence),
            },
            "delivery": {
                "storage_upload_performed": True,
                "signed_urls": ["https://signed-access.tryblueprint.io/package-live-001"],
                "entitlement_verified": True,
                "operator_attestation": {
                    "attested_by": "delivery-owner",
                    "attestation": "Signed delivery was uploaded and entitlement checked.",
                },
            },
            "safety_contact_physics": {
                "physics_contact_validated": True,
                "safety_validated": True,
                "robot_readiness_proven": True,
                "methodology_uri_or_path": str(methodology),
                "contact_validation_uri_or_path": str(contact_validation),
                "safety_validation_uri_or_path": str(safety_validation),
                "operator_attestation": {
                    "attested_by": "safety-owner",
                    "attestation": "Owner accepted contact, physics, and safety evidence.",
                },
            },
        },
    )
    request_path = tmp_path / "job-request-live.json"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-live-closure",
        provisioner="fixture_local",
        simulator="isaac_sim",
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim"],
        simulator_commands={"isaac_sim": f"{sys.executable} {simulator_writer}"},
        allow_policy_execution=True,
        policy_execution_commands={
            "policy_api_endpoint": f"{sys.executable} {policy_writer}",
        },
    )

    job_dir = Path(result["job_dir"])
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    data_package_export = _read_json(job_dir / "post_training_data_package_export_manifest.json")

    assert result["live_eval_closure_status"] == "live_end_to_end_verified"
    assert result["live_end_to_end_verified"] is True
    assert live_closure["status"] == "live_end_to_end_verified"
    assert str(staged_closure_evidence_path) in live_closure["evidence_sources"]
    assert live_closure["blockers"] == []
    assert all(gate["passed"] for gate in live_closure["gates"].values())
    assert proof_boundary["status"] == "live_end_to_end_verified"
    assert proof_boundary["simulator_execution_proven"] is True
    assert proof_boundary["robot_policy_execution_proven"] is True
    assert proof_boundary["robot_readiness_proven"] is True
    assert proof_boundary["public_claim_upgrade_allowed"] is True
    assert run_manifest["live_end_to_end_verified"] is True
    assert run_manifest["robot_readiness_proven"] is True
    assert run_manifest["public_claim_upgrade_allowed"] is True
    assert data_package_export["included_artifacts"]["live_eval_closure_manifest"] == (
        "live_eval_closure_manifest.json"
    )


def test_live_robot_eval_closure_blocks_rights_acceptance_without_owner_evidence(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-rights-accepted-only"
    request = _full_job_request(capture_root)
    request["rights_privacy_scope"] = {
        "status": "cleared_for_robot_eval",
        "external_use_allowed": True,
        "privacy_scope": "derived_deidentified_environment",
    }
    _write_json(
        job_dir / "live_eval_closure_evidence.json",
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "rights_privacy": {
                "accepted": True,
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    gate = manifest["gates"]["rights_privacy_scope"]
    assert gate["passed"] is False
    assert "rights_privacy_owner_evidence_missing" in gate["blockers"]
    assert gate["evidence"]["accepted"] is True
    assert gate["evidence"]["external_use_allowed"] is True
    assert gate["evidence"]["operator_attestation_present"] is False
    assert gate["evidence"]["evidence_proven"] is False


def test_live_robot_eval_closure_blocks_unverified_webapp_ids_only_in_job_request(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-request-only-webapp-ids"
    request = _full_job_request(capture_root)
    request.update(
        {
            "site_submission_id": "site-submission-live-001",
            "request_id": "request-live-001",
            "buyer_request_id": "buyer-request-live-001",
            "capture_job_id": "capture-job-live-001",
        }
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    gate = manifest["gates"]["webapp_upstream_truth"]
    assert gate["passed"] is False
    assert "webapp_upstream_ids_not_grounded_in_capture_or_webapp_source" in gate["blockers"]
    assert gate["evidence"]["ungrounded_fields"] == [
        "buyer_request_id",
        "capture_job_id",
        "request_id",
        "site_submission_id",
    ]
    assert gate["evidence"]["job_request_capture_root_matches"] is True
    assert gate["evidence"]["job_request_webapp_source_present"] is False


def test_live_robot_eval_closure_blocks_conflicting_webapp_upstream_sources(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-conflicting-webapp-ids"
    request = _full_job_request(capture_root)
    request.update(
        {
            "site_submission_id": "site-submission-live-001",
            "request_id": "request-live-001",
            "buyer_request_id": "buyer-request-live-001",
            "capture_job_id": "capture-job-live-001",
        }
    )
    descriptor = _read_json(capture_root / "capture_descriptor.json")
    descriptor.update(
        {
            "site_submission_id": "site-submission-other",
            "request_id": "request-live-001",
            "buyer_request_id": "buyer-request-live-001",
            "capture_job_id": "capture-job-live-001",
        }
    )
    _write_json(capture_root / "capture_descriptor.json", descriptor)

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    gate = manifest["gates"]["webapp_upstream_truth"]
    assert gate["passed"] is False
    assert "webapp_upstream_source_mismatch" in gate["blockers"]
    assert gate["evidence"]["mismatch_fields"] == ["site_submission_id"]
    assert gate["evidence"]["source_values"]["site_submission_id"] == {
        "job_request": "site-submission-live-001",
        "capture_descriptor": "site-submission-other",
    }


def test_live_robot_eval_closure_accepts_matching_webapp_route_forwarding_proof_lineage(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-prod-job-1"
    request = _full_job_request(capture_root)
    _write_json(
        capture_root
        / "pipeline"
        / "webapp_route_forwarding_proof"
        / "webapp_route_forwarding_proof.production.json",
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "forwarded_to_pipeline_intake",
            "capture_root": (
                "/var/lib/blueprint/pipeline-control-plane/captures/local-blueprint/"
                "scenes/scene-1/captures/capture-1"
            ),
            "pipeline_intake": {"status": "staged_for_control_plane"},
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "webapp-prod-job-1",
                "request_id": "request-prod-001",
                "buyer_request_id": "buyer-request-prod-001",
                "site_package": {
                    "site_submission_id": "site-submission-prod-001",
                    "capture_job_id": "capture-job-prod-001",
                    "capture_id": "capture-1",
                    "capture_root": (
                        "/var/lib/blueprint/pipeline-control-plane/captures/"
                        "local-blueprint/scenes/scene-1/captures/capture-1"
                    ),
                },
            },
        },
    )
    _write_json(
        job_dir / "runpod_live_execution_proof.zero-pods.json",
        {
            "schema_version": "runpod_live_execution_proof.v1",
            "status": "runpod_live_proof_collected",
            "api_call_performed": True,
            "runpod_side_effects_may_have_occurred": False,
            "active_pod_count_before": 0,
            "active_pod_count_after": 0,
            "secret_values_in_artifact": False,
            "raw_api_key_stored": False,
            "blockers": [],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    gate = manifest["gates"]["webapp_upstream_truth"]
    assert gate["passed"] is True
    assert gate["blockers"] == []
    assert gate["evidence"]["ids"] == {
        "site_submission_id": "site-submission-prod-001",
        "request_id": "request-prod-001",
        "buyer_request_id": "buyer-request-prod-001",
        "capture_job_id": "capture-job-prod-001",
    }
    assert gate["evidence"]["webapp_route_forwarding_proofs"][0]["grounding_verified"] is True
    assert gate["evidence"]["webapp_route_forwarding_proofs"][0]["lineage_matches"] is True
    assert gate["evidence"]["webapp_route_forwarding_proofs"][0]["job_id_matches"] is True
    beta_checks = {
        check["check_id"]: check for check in manifest["robot_team_beta_readiness"]["checks"]
    }
    assert beta_checks["production_or_staging_webapp_request_ids"]["passed"] is True
    assert beta_checks["real_capture_root_input"]["passed"] is True
    assert beta_checks["live_provider_worker_execution"]["passed"] is False
    assert (
        "remote_cloud_execution_closure_manifest_missing"
        in beta_checks["live_provider_worker_execution"]["blockers"]
    )
    assert beta_checks["fresh_live_robot_team_run_today"]["passed"] is False
    assert (
        "remote_cloud_execution_closure_manifest_missing"
        in beta_checks["fresh_live_robot_team_run_today"]["blockers"]
    )
    assert beta_checks["shutdown_and_cost_proof"]["evidence"][
        "runpod_zero_active_pods_proof"
    ]["zero_active_pods_now"] is True
    assert beta_checks["robot_pov_policy_evidence"]["passed"] is False
    assert (
        "real_robot_pov_evidence:real_robot_pov_evidence_not_proven"
        in beta_checks["robot_pov_policy_evidence"]["blockers"]
    )
    assert beta_checks["deployment_outcome_joins"]["passed"] is False
    assert (
        "deployment_outcome_intake_manifest_missing"
        in beta_checks["deployment_outcome_joins"]["blockers"]
    )


def test_live_robot_eval_closure_beta_accepts_schema_valid_deployment_join_artifacts(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-prod-join-artifacts"
    request = _full_job_request(capture_root)
    _write_json(
        job_dir / "deployment_outcome_intake_manifest.json",
        {
            "schema_version": "deployment_outcome_intake_manifest.v1",
            "status": "blocked_missing_real_world_outcomes",
            "record_count": 0,
            "blockers": [],
        },
    )
    _write_json(
        job_dir / "deployment_outcome_ledger.json",
        {
            "schema_version": "deployment_outcome_ledger.v1",
            "status": "blocked_missing_real_world_outcomes",
            "record_count": 0,
            "matched_prediction_record_count": 0,
            "exact_prediction_record_count": 0,
            "blockers": ["missing_real_world_outcome_records"],
        },
    )
    _write_json(
        job_dir / "prediction_vs_actual_deployment_summary.json",
        {
            "schema_version": "prediction_vs_actual_deployment_summary.v1",
            "status": "blocked_missing_real_world_outcomes",
            "matched_prediction_record_count": 0,
            "exact_prediction_record_count": 0,
        },
    )
    _write_json(
        job_dir / "sim_vs_real_calibration_report.json",
        {
            "schema_version": "sim_vs_real_calibration_report.v1",
            "status": "blocked_missing_real_world_outcomes",
            "matched_prediction_record_count": 0,
            "exact_prediction_record_count": 0,
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    beta_checks = {
        check["check_id"]: check
        for check in manifest["robot_team_beta_readiness"]["checks"]
    }
    deployment_check = beta_checks["deployment_outcome_joins"]
    assert deployment_check["passed"] is True
    assert deployment_check["blockers"] == []
    assert deployment_check["evidence"]["real_world_outcome_record_count"] == 0
    assert deployment_check["evidence"]["prediction_summary_status"] == (
        "blocked_missing_real_world_outcomes"
    )
    assert deployment_check["evidence"]["calibration_report_status"] == (
        "blocked_missing_real_world_outcomes"
    )


def test_live_robot_eval_closure_blocks_stale_remote_run_for_today_beta(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-prod-job-stale-live"
    request = _full_job_request(capture_root)
    _write_json(
        job_dir / "remote_cloud_execution_closure_manifest.json",
        {
            "schema_version": "remote_cloud_execution_closure_manifest.v1",
            "generated_at": "2000-01-01T00:00:00Z",
            "status": "remote_execution_completed_with_shutdown_proof",
            "remote_cloud_execution_proven": True,
            "clean_shutdown_proven": True,
            "live_provider_calls_performed": True,
            "blockers": [],
            "checks": {
                "artifact_output_uri_configured": True,
                "artifact_output_uri_provider_writable": True,
                "artifact_output_write_auth_contract_ready": True,
                "actual_gpu_time_record_present": True,
            },
            "outputs": {
                "artifact_output_uri": "gs://blueprint-8c1ca-scenes/jobs/webapp-prod-job-stale-live",
            },
            "cost_and_timeout_controls": {"actual_gpu_seconds": 120},
            "provider_input_setup": {
                "status": "ready_for_provider_launcher_inputs",
                "provider_inputs_uploaded": True,
                "blockers": [],
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    beta_checks = {
        check["check_id"]: check for check in manifest["robot_team_beta_readiness"]["checks"]
    }
    fresh_check = beta_checks["fresh_live_robot_team_run_today"]
    assert fresh_check["passed"] is False
    assert fresh_check["evidence"]["remote_closure_generated_date"] == "2000-01-01"
    assert "remote_closure_not_generated_today" in fresh_check["blockers"]
    assert beta_checks["live_provider_worker_execution"]["passed"] is True
    assert beta_checks["writable_artifact_output_uri"]["passed"] is True


def test_live_robot_eval_closure_blocks_writable_output_until_provider_inputs_uploaded(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-prod-job-gcs-blocked"
    request = _full_job_request(capture_root)
    _write_json(
        job_dir / "remote_cloud_execution_closure_manifest.json",
        {
            "schema_version": "remote_cloud_execution_closure_manifest.v1",
            "status": "blocked_before_remote_execution",
            "remote_cloud_execution_proven": False,
            "clean_shutdown_proven": False,
            "live_provider_calls_performed": False,
            "blockers": ["provider_input_setup:upload_failed:gs_billing_account_disabled"],
            "checks": {
                "artifact_output_uri_configured": True,
                "artifact_output_uri_provider_writable": True,
                "artifact_output_write_auth_contract_ready": True,
                "actual_gpu_time_record_present": False,
            },
            "outputs": {
                "artifact_output_uri": "gs://blueprint-8c1ca-scenes/jobs/webapp-prod-job-gcs-blocked",
                "artifact_output_uri_scheme": "gs",
            },
            "cost_and_timeout_controls": {"actual_gpu_seconds": None},
            "provider_input_setup": {
                "status": "prepared_with_external_blockers",
                "provider_inputs_uploaded": False,
                "blockers": [
                    "upload_failed:gs_billing_account_disabled",
                    "provider_inputs_upload_not_proven",
                ],
                "manifest_path": "provider_input_setup_manifest.json",
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    beta_checks = {
        check["check_id"]: check for check in manifest["robot_team_beta_readiness"]["checks"]
    }
    output_check = beta_checks["writable_artifact_output_uri"]
    assert output_check["passed"] is False
    assert "remote_artifact_output_upload_not_proven" in output_check["blockers"]
    assert (
        "provider_input_setup:upload_failed:gs_billing_account_disabled"
        in output_check["blockers"]
    )
    assert output_check["evidence"]["artifact_output_uri_provider_writable"] is True
    assert output_check["evidence"]["artifact_output_write_auth_contract_ready"] is True
    assert output_check["evidence"]["provider_input_setup"]["provider_inputs_uploaded"] is False


def test_live_robot_eval_closure_keeps_route_proof_without_request_id_blocked(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-prod-job-no-request-id"
    request = _full_job_request(capture_root)
    _write_json(
        capture_root
        / "pipeline"
        / "webapp_route_forwarding_proof"
        / "webapp_route_forwarding_proof.production.json",
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "forwarded_to_pipeline_intake",
            "capture_root": (
                "/var/lib/blueprint/pipeline-control-plane/captures/local-blueprint/"
                "scenes/scene-1/captures/capture-1"
            ),
            "pipeline_intake": {"status": "staged_for_control_plane"},
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "webapp-prod-job-no-request-id",
                "buyer_request_id": "buyer-request-prod-001",
                "site_package": {
                    "site_submission_id": "site-submission-prod-001",
                    "capture_job_id": "capture-job-prod-001",
                    "capture_id": "capture-1",
                },
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    gate = manifest["gates"]["webapp_upstream_truth"]
    assert gate["passed"] is False
    assert "missing_webapp_request_id" in gate["blockers"]
    assert gate["evidence"]["ids"]["buyer_request_id"] == "buyer-request-prod-001"
    assert gate["evidence"]["ids"]["site_submission_id"] == "site-submission-prod-001"
    assert gate["evidence"]["ids"]["capture_job_id"] == "capture-job-prod-001"
    assert gate["evidence"]["webapp_route_forwarding_proofs"][0]["grounding_verified"] is True
    assert (
        gate["evidence"]["webapp_route_forwarding_proofs"][0]["ids_present"]["request_id"]
        is False
    )
    beta_checks = {
        check["check_id"]: check for check in manifest["robot_team_beta_readiness"]["checks"]
    }
    assert beta_checks["production_or_staging_webapp_request_ids"]["passed"] is False
    assert "webapp_upstream_truth:missing_webapp_request_id" in (
        beta_checks["production_or_staging_webapp_request_ids"]["blockers"]
    )


def test_live_robot_eval_closure_uses_stored_webapp_doc_id_as_request_id(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-prod-doc-job"
    request = _full_job_request(capture_root)
    _write_json(
        capture_root
        / "pipeline"
        / "webapp_route_forwarding_proof"
        / "webapp_route_forwarding_proof.production.json",
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "forwarded_to_pipeline_intake",
            "capture_root": (
                "/var/lib/blueprint/pipeline-control-plane/captures/local-blueprint/"
                "scenes/scene-1/captures/capture-1"
            ),
            "pipeline_intake": {"status": "staged_for_control_plane"},
            "durable_store": {
                "firestore": {
                    "status": "stored",
                    "doc_id": "webapp-prod-doc-job",
                }
            },
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "webapp-prod-doc-job",
                "buyer_request_id": "buyer-request-prod-001",
                "site_package": {
                    "site_submission_id": "site-submission-prod-001",
                    "capture_job_id": "capture-job-prod-001",
                    "capture_id": "capture-1",
                },
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    gate = manifest["gates"]["webapp_upstream_truth"]
    assert gate["passed"] is True
    assert gate["evidence"]["ids"]["request_id"] == "webapp-prod-doc-job"
    assert gate["evidence"]["webapp_route_forwarding_proofs"][0]["id_source_fields"][
        "request_id"
    ] == "durable_store.firestore.doc_id"
    beta_checks = {
        check["check_id"]: check for check in manifest["robot_team_beta_readiness"]["checks"]
    }
    assert beta_checks["production_or_staging_webapp_request_ids"]["passed"] is True


def test_live_robot_eval_closure_does_not_ground_route_proof_for_another_job(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "current-webapp-job"
    request = _full_job_request(capture_root)
    _write_json(
        capture_root
        / "pipeline"
        / "webapp_route_forwarding_proof"
        / "webapp_route_forwarding_proof.stale.json",
        {
            "schema_version": "blueprint_webapp_route_forwarding_proof.v1",
            "status": "forwarded_to_pipeline_intake",
            "capture_root": (
                "/var/lib/blueprint/pipeline-control-plane/captures/local-blueprint/"
                "scenes/scene-1/captures/capture-1"
            ),
            "pipeline_intake": {"status": "staged_for_control_plane"},
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "previous-webapp-job",
                "request_id": "request-prod-001",
                "buyer_request_id": "buyer-request-prod-001",
                "site_package": {
                    "site_submission_id": "site-submission-prod-001",
                    "capture_job_id": "capture-job-prod-001",
                    "capture_id": "capture-1",
                },
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request=request,
    )

    gate = manifest["gates"]["webapp_upstream_truth"]
    assert gate["passed"] is False
    assert "missing_webapp_request_id" in gate["blockers"]
    assert "missing_webapp_buyer_request_id" in gate["blockers"]
    assert gate["evidence"]["webapp_route_forwarding_proofs"][0]["grounding_verified"] is False
    assert gate["evidence"]["webapp_route_forwarding_proofs"][0]["job_id_matches"] is False


def test_live_robot_eval_closure_revalidates_owner_gpu_proof_manifest_evidence(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-spoofed-owner-gpu-proof"
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "blockers": [],
            "missing_inputs": [],
            "evidence": {},
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_simulator_execution"]
    assert gate["passed"] is False
    assert "live_simulator_execution_not_proven" in gate["blockers"]
    assert "owner_gpu_proof_missing_required_manifest_fields" in gate["blockers"]
    assert "owner_gpu_proof_exit_code_not_zero" in gate["blockers"]
    assert "owner_gpu_proof_manifest_missing_required_evidence" in gate["blockers"]
    assert gate["evidence"]["owner_gpu_proof_audit"]["accepted"] is False
    assert gate["evidence"]["owner_gpu_proof_audit"]["missing_required_fields"] == [
        "owner_system_id",
        "simulator_backend",
        "simulator_version",
        "gpu_model",
        "proof_path",
    ]
    assert (
        "scene_loaded_in_owner_simulator"
        in gate["evidence"]["owner_gpu_proof_audit"]["missing_evidence_flags"]
    )
    assert (
        "sim_robot_pov_evidence_valid"
        in gate["evidence"]["owner_gpu_proof_audit"]["missing_evidence_flags"]
    )


def test_live_robot_eval_closure_blocks_mujoco_owner_proof_for_isaac_requirement(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-mujoco-owner-gpu-proof"
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        job_dir / "worker_manifest.json",
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "simulator": "isaac_sim",
            "allowed_simulators": ["isaac_sim"],
        },
    )
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_system_id": "runpod-a6000",
            "simulator_backend": "mujoco",
            "simulator_version": "3.9.0",
            "gpu_model": "NVIDIA RTX A6000",
            "proof_path": "pipeline/simulation_automation/gpu_owner_system_proof.json",
            "exit_code": 0,
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "isaac_sim_execution_proven": False,
            "isaac_robot_asset_execution_proven": False,
            "unitree_g1_asset_spawned": False,
            "blockers": [],
            "missing_inputs": [],
            "evidence": {
                "stdout_present": True,
                "stderr_present": True,
                "scene_load_trace_present": True,
                "scene_loaded_in_owner_simulator": True,
                "spawn_trace_present": True,
                "spawn_pose_loaded": True,
                "action_or_policy_trace_present": True,
                "action_or_policy_trace_valid": True,
                "default_smoke_policy_present": True,
                "default_smoke_policy_valid": True,
                "policy_execution_trace_present": True,
                "default_policy_execution_trace_valid": True,
                "sim_robot_pov_evidence_present": True,
                "sim_robot_pov_evidence_valid": True,
                "artifact_manifest_present": True,
                "artifact_manifest_valid": True,
                "robot_asset_trace_present": True,
                "robot_asset_matches_proof": True,
                "operator_attestation_present": True,
                "pass_fail_criteria_passed": True,
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_simulator_execution"]
    assert gate["passed"] is False
    assert "live_simulator_execution_not_proven" not in gate["blockers"]
    assert "isaac_sim_unitree_g1_execution_not_proven" in gate["blockers"]
    assert gate["evidence"]["owner_gpu_proof_audit"]["accepted"] is True
    assert gate["evidence"]["owner_isaac_unitree_g1_execution_proven"] is False


def test_live_robot_eval_closure_accepts_mujoco_owner_proof_for_mujoco_requirement(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-mujoco-owner-gpu-proof"
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        job_dir / "worker_manifest.json",
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "simulator": "mujoco",
            "allowed_simulators": ["mujoco"],
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "normalized_attempt_trace.v1",
            "status": "blocked",
            "attempt_count": 0,
        },
    )
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_system_id": "runpod-a6000",
            "simulator_backend": "mujoco",
            "simulator_version": "3.9.0",
            "gpu_model": "NVIDIA RTX A6000",
            "proof_path": "pipeline/simulation_automation/gpu_owner_system_proof.json",
            "exit_code": 0,
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "isaac_sim_execution_proven": False,
            "isaac_robot_asset_execution_proven": False,
            "mujoco_g1_asset_execution_proven": True,
            "mujoco_g1_asset_spawned": True,
            "unitree_g1_asset_spawned": False,
            "robot_asset": {
                "name": "Unitree G1",
                "source": "google_deepmind_mujoco_menagerie",
                "mujoco_g1_asset_execution_proven": True,
            },
            "blockers": [],
            "missing_inputs": [],
            "evidence": {
                "stdout_present": True,
                "stderr_present": True,
                "scene_load_trace_present": True,
                "scene_loaded_in_owner_simulator": True,
                "spawn_trace_present": True,
                "spawn_pose_loaded": True,
                "action_or_policy_trace_present": True,
                "action_or_policy_trace_valid": True,
                "default_smoke_policy_present": True,
                "default_smoke_policy_valid": True,
                "policy_execution_trace_present": True,
                "default_policy_execution_trace_valid": True,
                "sim_robot_pov_evidence_present": True,
                "sim_robot_pov_evidence_valid": True,
                "artifact_manifest_present": True,
                "artifact_manifest_valid": True,
                "robot_asset_trace_present": True,
                "robot_asset_matches_proof": True,
                "operator_attestation_present": True,
                "pass_fail_criteria_passed": True,
                "mujoco_g1_asset_spawned": True,
                "mujoco_g1_asset_valid": True,
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_simulator_execution"]
    assert gate["passed"] is True
    assert "isaac_sim_unitree_g1_execution_not_proven" not in gate["blockers"]
    assert "mujoco_g1_execution_not_proven" not in gate["blockers"]
    assert gate["evidence"]["expected_simulator"] == "mujoco"
    assert gate["evidence"]["owner_mujoco_unitree_g1_execution_proven"] is True
    assert gate["evidence"]["owner_gpu_proof_audit"]["accepted"] is True
    assert gate["evidence"]["normalized_attempt_trace_status"] == "blocked"


def test_live_robot_eval_closure_accepts_camel_case_owner_attestations(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-camel-evidence"
    refs_dir = tmp_path / "owner-evidence"
    methodology = refs_dir / "methodology.md"
    contact = refs_dir / "contact.json"
    safety = refs_dir / "safety.json"
    for path in (methodology, contact, safety):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("owner evidence\n", encoding="utf-8")
    _write_json(
        job_dir / "live_eval_closure_evidence.json",
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "reviewAcceptance": {
                "accepted": True,
                "operatorAttestation": {
                    "attestedBy": "owner-reviewer",
                    "acceptedClaimBoundary": "Owner accepted review evidence.",
                },
            },
            "delivery": {
                "storageUploadPerformed": True,
                "signedUrls": ["https://signed-access.tryblueprint.io/package-1"],
                "entitlementVerified": True,
                "operatorAttestation": {
                    "attestedBy": "delivery-owner",
                    "acceptedClaimBoundary": "Owner accepted signed delivery access.",
                },
            },
            "safetyContactPhysics": {
                "physicsContactValidated": True,
                "safetyValidated": True,
                "robotReadinessProven": True,
                "methodologyUriOrPath": str(methodology),
                "contactValidationUriOrPath": str(contact),
                "safetyValidationUriOrPath": str(safety),
                "operatorAttestation": {
                    "attestedBy": "safety-owner",
                    "acceptedClaimBoundary": (
                        "Owner accepted contact, physics, and safety evidence."
                    ),
                },
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    assert manifest["gates"]["review_acceptance"]["passed"] is True
    assert manifest["gates"]["review_acceptance"]["blockers"] == []
    assert manifest["gates"]["signed_delivery_access"]["passed"] is True
    assert manifest["gates"]["signed_delivery_access"]["blockers"] == []
    assert manifest["gates"]["safety_contact_physics_readiness"]["passed"] is True
    assert manifest["gates"]["safety_contact_physics_readiness"]["blockers"] == []


def test_live_robot_eval_closure_blocks_signed_delivery_without_operator_attestation(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-delivery-no-attestation"
    _write_json(
        job_dir / "live_eval_closure_evidence.json",
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "delivery": {
                "storage_upload_performed": True,
                "signed_urls": ["https://signed-access.tryblueprint.io/package-1"],
                "entitlement_verified": True,
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["signed_delivery_access"]
    assert gate["passed"] is False
    assert "signed_delivery_operator_attestation_missing" in gate["blockers"]
    assert gate["evidence"]["operator_attestation_present"] is False


def test_live_robot_eval_closure_blocks_mismatched_job_evidence(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-evidence-target"
    _write_json(
        job_dir / "live_eval_closure_evidence.json",
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "job_id": "other-job",
            "review_acceptance": {
                "accepted": True,
                "reviewer": "owner-reviewer",
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_evidence_integrity"]
    assert gate["passed"] is False
    assert "live_closure_evidence_job_id_mismatch" in gate["blockers"]
    assert gate["evidence"]["input_blockers"] == [
        {
            "blocker": "live_closure_evidence_job_id_mismatch",
            "source": str(job_dir / "live_eval_closure_evidence.json"),
            "expected_job_id": "job-evidence-target",
            "declared_job_id": "other-job",
        }
    ]
    assert "live_evidence_integrity:live_closure_evidence_job_id_mismatch" in manifest["blockers"]
    assert manifest["live_end_to_end_verified"] is False


def test_live_robot_eval_closure_blocks_missing_local_review_acceptance_ref(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-missing-review-ref"
    _write_json(
        job_dir / "live_eval_closure_evidence.json",
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "review_acceptance": {
                "accepted": True,
                "reviewer": "owner-reviewer",
                "evidence_uri_or_path": str(tmp_path / "missing-review-acceptance.json"),
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["review_acceptance"]
    assert gate["passed"] is False
    assert "review_acceptance_local_evidence_refs_missing" in gate["blockers"]
    assert gate["evidence"]["missing_local_ref_keys"] == ["evidence_uri_or_path"]


def test_live_robot_eval_closure_blocks_missing_local_safety_contact_refs(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-missing-safety-refs"
    refs_dir = tmp_path / "owner-evidence"
    _write_json(
        job_dir / "live_eval_closure_evidence.json",
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "safety_contact_physics": {
                "physics_contact_validated": True,
                "safety_validated": True,
                "robot_readiness_proven": True,
                "methodology_uri_or_path": str(refs_dir / "missing-methodology.md"),
                "contact_validation_uri_or_path": str(refs_dir / "missing-contact.json"),
                "safety_validation_uri_or_path": str(refs_dir / "missing-safety.json"),
                "operator_attestation": {
                    "attested_by": "safety-owner",
                    "attestation": "Owner accepted contact, physics, and safety evidence.",
                },
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["safety_contact_physics_readiness"]
    assert gate["passed"] is False
    assert "safety_contact_physics_local_evidence_refs_missing" in gate["blockers"]
    assert gate["evidence"]["missing_local_ref_keys"] == [
        "contact_validation_uri_or_path",
        "methodology_uri_or_path",
        "safety_validation_uri_or_path",
    ]


def test_live_robot_eval_closure_blocks_metadata_only_site_capture(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "site_identity": {"site_id": "site-1"},
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=capture_root / "pipeline" / "robot_eval_jobs" / "job-metadata-only",
    )

    gate = manifest["gates"]["site_capture"]
    assert gate["passed"] is False
    assert "missing_raw_capture_upload_completion" in gate["blockers"]
    assert "missing_raw_capture_evidence" in gate["blockers"]


def test_live_robot_eval_closure_blocks_incomplete_scenario_variation_library(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-incomplete-variation-library"
    _write_json(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json",
        {
            "schema_version": "scenario_variation_instances.v1",
            "status": "completed",
            "required_variation_names": list(POLICY_REFERENCE_VARIATION_NAMES),
            "variation_names_instantiated": ["lighting_variation"],
            "instance_count": 1,
            "instances": [
                {
                    "instance_id": "variation-place-return-lighting",
                    "variation_name": "lighting_variation",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["scenario_library"]
    assert gate["passed"] is False
    assert "scenario_variation_instances_missing_required_variations" in gate["blockers"]
    assert gate["evidence"]["missing_required_variation_names"] == [
        "object_rotation",
        "cart_shifted",
        "blocked_path",
        "human_crossing",
        "forklift_nearby",
        "occlusion",
        "glare",
        "missing_label",
        "wrong_object_nearby",
        "narrow_approach_angle",
    ]


def test_live_robot_eval_closure_blocks_scenario_variations_missing_per_scenario_coverage(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    scenario_cards = _read_json(robot_eval_dir / "scenario_cards.json")
    base_card = dict(scenario_cards["cards"][0])
    second_card = {
        **base_card,
        "scenario_card_id": "scenario_card_place_return_in_bin_secondary",
        "scenario_id": "scenario_place_return_in_bin_secondary",
    }
    scenario_cards["scenario_card_count"] = 2
    scenario_cards["cards"] = [base_card, second_card]
    _write_json(robot_eval_dir / "scenario_cards.json", scenario_cards)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-per-scenario-variation-library"
    _write_json(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json",
        {
            "schema_version": "scenario_variation_instances.v1",
            "status": "completed",
            "required_variation_names": ["lighting_variation", "glare"],
            "variation_names_instantiated": ["lighting_variation", "glare"],
            "instance_count": 2,
            "instances": [
                {
                    "instance_id": "variation-primary-lighting",
                    "variation_name": "lighting_variation",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                },
                {
                    "instance_id": "variation-secondary-glare",
                    "variation_name": "glare",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_secondary",
                },
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["scenario_library"]
    assert gate["passed"] is False
    assert (
        "scenario_variation_instances_missing_required_variations_per_scenario" in gate["blockers"]
    )
    assert gate["evidence"]["missing_required_variations_by_scenario"] == [
        {
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "missing_variation_names": ["glare"],
        },
        {
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_secondary",
            "missing_variation_names": ["lighting_variation"],
        },
    ]


def test_live_robot_eval_closure_blocks_name_only_scenario_variation_instances(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-name-only-variation-library"
    _write_json(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json",
        {
            "schema_version": "scenario_variation_instances.v1",
            "status": "completed",
            "required_variation_names": ["lighting_variation"],
            "variation_names_instantiated": ["lighting_variation"],
            "instance_count": 1,
            "instances": [
                {
                    "instance_id": "variation-primary-lighting",
                    "variation_name": "lighting_variation",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["scenario_library"]
    assert gate["passed"] is False
    assert "scenario_variation_instances_missing_concrete_mutation_details" in gate["blockers"]
    assert gate["evidence"]["variation_rows_missing_concrete_details"] == [
        {
            "row_index": 1,
            "instance_id": "variation-primary-lighting",
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "variation_name": "lighting_variation",
            "missing_fields": ["concrete_mutation", "engine_mutations"],
        }
    ]


def test_live_robot_eval_closure_blocks_card_counts_without_required_fields(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        robot_eval_dir / "task_cards.json",
        {
            "schema_version": "real_site_robot_eval_task_cards.v0.1",
            "task_card_count": 1,
            "cards": [{}],
        },
    )
    _write_json(
        robot_eval_dir / "scenario_cards.json",
        {
            "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
            "scenario_card_count": 1,
            "cards": [{}],
        },
    )
    _write_json(
        robot_eval_dir / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "eval_card_count": 1,
            "cards": [{}],
        },
    )
    _write_json(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json",
        {
            "schema_version": "scenario_variation_instances.v1",
            "status": "completed",
            "required_variation_names": list(POLICY_REFERENCE_VARIATION_NAMES),
            "variation_names_instantiated": list(POLICY_REFERENCE_VARIATION_NAMES),
            "instance_count": len(POLICY_REFERENCE_VARIATION_NAMES),
            "instances": [
                {
                    "instance_id": f"variation-{variation_name}",
                    "variation_name": variation_name,
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                }
                for variation_name in POLICY_REFERENCE_VARIATION_NAMES
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=capture_root / "pipeline" / "robot_eval_jobs" / "job-invalid-card-fields",
    )

    task_gate = manifest["gates"]["task_definitions"]
    scenario_gate = manifest["gates"]["scenario_library"]
    eval_gate = manifest["gates"]["scenario_eval_suite"]
    assert "task_definitions_cards_missing_required_fields" in task_gate["blockers"]
    assert "scenario_library_cards_missing_required_fields" in scenario_gate["blockers"]
    assert "scenario_eval_suite_cards_missing_required_fields" in eval_gate["blockers"]
    assert set(task_gate["evidence"]["cards_missing_required_fields"][0]["missing_fields"]) == {
        "task_id",
        "task_statement",
        "task_category",
        "required_metrics",
    }
    assert set(scenario_gate["evidence"]["cards_missing_required_fields"][0]["missing_fields"]) == {
        "scenario_id",
        "task_id",
        "robot_profile_id",
        "normal_scenario",
        "variation",
        "edge_case",
    }
    assert set(eval_gate["evidence"]["cards_missing_required_fields"][0]["missing_fields"]) == {
        "eval_card_id",
        "scenario_id",
        "task_id",
        "prediction_source",
        "validation",
        "proof_boundary",
    }


def test_live_robot_eval_closure_blocks_task_cards_missing_standard_metrics(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    task_path = capture_root / "pipeline" / "robot_eval_dataset" / "task_cards.json"
    task_payload = _read_json(task_path)
    task_payload["cards"][0]["required_metrics"] = ["cycle_time"]
    _write_json(task_path, task_payload)

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=capture_root / "pipeline" / "robot_eval_jobs" / "job-missing-task-metrics",
    )

    gate = manifest["gates"]["task_definitions"]
    assert gate["passed"] is False
    assert "task_definitions_missing_standard_required_metrics" in gate["blockers"]
    assert gate["evidence"]["cards_missing_standard_required_metrics"] == [
        {
            "index": 0,
            "task_id": "place_return_in_bin",
            "missing_metrics": [
                "collision_risk",
                "intervention_rate",
                "object_drop",
                "recovery_success",
                "sim_vs_real_calibration_score",
                "success_rate",
                "timeout",
                "unsafe_proximity",
                "world_model_uncertainty",
                "wrong_object",
            ],
        }
    ]


def test_live_robot_eval_closure_blocks_missing_robot_eval_report(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=capture_root / "pipeline" / "robot_eval_jobs" / "job-missing-report",
    )

    gate = manifest["gates"]["report_generation"]
    assert gate["passed"] is False
    assert "missing_robot_eval_report" in gate["blockers"]
    assert "missing_robot_eval_report_markdown" in gate["blockers"]


def test_live_robot_eval_closure_blocks_unlinked_robot_eval_report_stub(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-unlinked-report"
    _write_json(
        job_dir / "robot_eval_report.json",
        {
            "schema_version": "robot_eval_job_report.v1",
            "status": "generated",
            "neutral_eval_harness_flow": ["report_generated"],
            "scenario_eval": {"status": "completed", "scenario_eval_run_count": 1},
            "policy_interface": {
                "policy_execution_status": "completed",
                "selected_modalities": ["policy_api_endpoint"],
            },
            "evaluator_scores": {
                "success_rate": 1.0,
                "cycle_time": {"mean_seconds": 10.0, "sample_count": 1},
                "intervention_rate": 0.0,
                "unsafe_proximity": 0.0,
                "collision_risk": 0.0,
                "object_drop": 0,
                "wrong_object": 0,
                "timeout": 0,
                "recovery_success": {"rate": 1.0, "sample_count": 1},
                "world_model_uncertainty": {"mean": 0.1, "sample_count": 1},
                "sim_vs_real_calibration_score": None,
            },
            "real_world_validation": {
                "deployment_outcome_status": "completed",
                "real_world_outcome_records_present": True,
            },
            "predicted_vs_actual": {
                "sim_vs_real_calibration_status": "completed",
            },
            "live_eval_closure": {"status": "pending_live_eval_closure"},
            "requirement_coverage": {"schema_version": "live_robot_eval_requirement_coverage.v1"},
            "proof_boundary": {"robot_readiness_proven": False},
            "artifact_paths": {
                "scenario_eval_matrix": "missing-scenario-eval-matrix.json",
                "evaluation_result": "missing-evaluation-result.json",
                "policy_execution_manifest": "missing-policy-execution-manifest.json",
                "policy_execution_trace": "missing-policy-execution-trace.json",
                "deployment_outcome_ledger": "missing-deployment-outcome-ledger.json",
                "prediction_vs_actual_deployment_summary": "missing-prediction-summary.json",
                "proof_boundary": "missing-proof-boundary.json",
            },
        },
    )
    (job_dir / "robot_eval_report.md").parent.mkdir(parents=True, exist_ok=True)
    (job_dir / "robot_eval_report.md").write_text("# Robot Eval Report\n", encoding="utf-8")

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["report_generation"]
    assert gate["passed"] is False
    assert "robot_eval_report_referenced_artifacts_missing" in gate["blockers"]
    assert gate["evidence"]["artifact_audit"]["missing_artifact_file_keys"] == [
        "scenario_eval_matrix",
        "evaluation_result",
        "policy_execution_manifest",
        "policy_execution_trace",
        "deployment_outcome_ledger",
        "prediction_vs_actual_deployment_summary",
        "proof_boundary",
    ]


def test_live_robot_eval_closure_blocks_count_only_scenario_eval_matrix(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-count-only-matrix"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "required_variation_names": ["lighting_variation"],
            "variation_names_covered": ["lighting_variation"],
            "runs": [],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["scenario_eval_suite"]
    assert gate["passed"] is False
    assert "scenario_eval_matrix_missing_run_rows" in gate["blockers"]
    assert "scenario_eval_matrix_run_count_mismatch" in gate["blockers"]
    assert gate["evidence"]["scenario_eval_run_row_count"] == 0


def test_live_robot_eval_closure_requires_complete_simulator_plugin_registry(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-partial-plugins"
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "simulator_engine_plugin_registry.json",
        {
            "schema_version": "simulator_engine_plugin_registry.v1",
            "status": "ready_for_gated_managed_execution",
            "engine_targets": [
                "isaac_sim",
                "isaac_lab_arena",
                "mujoco",
                "pybullet",
                "newton",
            ],
            "plugin_count": 1,
            "plugins": {
                "mujoco": {
                    "plugin_id": "blueprint_mujoco_sim_engine_plugin",
                    "framework": "mujoco",
                    "adapter_contract_status": "ready",
                    "managed_execution_supported": True,
                }
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    plugin_gate = manifest["gates"]["simulator_engine_plugins"]
    assert plugin_gate["passed"] is False
    assert "simulator_engine_plugin_registry_missing_required_engines" in plugin_gate["blockers"]
    assert set(plugin_gate["evidence"]["missing_required_plugins"]) == {
        "isaac_sim",
        "isaac_lab_arena",
        "pybullet",
        "newton",
    }


def test_live_robot_eval_closure_rejects_unready_simulator_engine_plugins(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-unready-plugins"
    engines = [
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    ]
    plugins = {
        engine: {
            "plugin_id": f"blueprint_{engine}_sim_engine_plugin",
            "framework": engine,
            "adapter_contract_status": "ready",
            "managed_execution_supported": True,
        }
        for engine in engines
    }
    plugins["isaac_lab_arena"]["adapter_contract_status"] = "blocked_missing_owner_adapter"
    plugins["newton"]["managed_execution_supported"] = False
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "simulator_engine_plugin_registry.json",
        {
            "schema_version": "simulator_engine_plugin_registry.v1",
            "status": "ready_for_gated_managed_execution",
            "engine_targets": engines,
            "plugin_count": len(plugins),
            "plugins": plugins,
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    plugin_gate = manifest["gates"]["simulator_engine_plugins"]
    assert plugin_gate["passed"] is False
    assert "simulator_engine_plugins_not_ready" in plugin_gate["blockers"]
    assert plugin_gate["evidence"]["unready_plugins"] == [
        {
            "framework": "isaac_lab_arena",
            "adapter_contract_status": "blocked_missing_owner_adapter",
            "managed_execution_supported": True,
        },
        {
            "framework": "newton",
            "adapter_contract_status": "ready",
            "managed_execution_supported": False,
        },
    ]


def test_live_robot_eval_closure_requires_world_model_engine_plugins(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-missing-world-plugins"
    engines = [
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    ]
    plugins = {
        engine: {
            "plugin_id": f"blueprint_{engine}_sim_engine_plugin",
            "framework": engine,
            "adapter_contract_status": "ready",
            "managed_execution_supported": True,
        }
        for engine in engines
    }
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "simulator_engine_plugin_registry.json",
        {
            "schema_version": "simulator_engine_plugin_registry.v1",
            "status": "ready_for_gated_managed_execution",
            "engine_targets": engines,
            "world_model_engine_targets": [
                "worldlabs_world_model",
                "marble_simready",
                "cosmos_predict",
                "native_site_reference",
            ],
            "plugin_count": len(plugins),
            "world_model_plugin_count": 1,
            "plugins": plugins,
            "world_model_plugins": {
                "worldlabs_world_model": {
                    "plugin_id": "blueprint_worldlabs_world_model_engine_plugin",
                    "engine": "worldlabs_world_model",
                    "adapter_contract_status": "ready",
                    "managed_execution_supported": True,
                }
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    plugin_gate = manifest["gates"]["simulator_engine_plugins"]
    assert plugin_gate["passed"] is False
    assert (
        "simulator_engine_plugin_registry_missing_required_world_model_engines"
        in plugin_gate["blockers"]
    )
    assert set(plugin_gate["evidence"]["missing_required_world_model_plugins"]) == {
        "marble_simready",
        "cosmos_predict",
        "native_site_reference",
    }


def test_live_robot_eval_closure_blocks_ready_simulator_plugins_missing_local_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-plugin-inputs-missing"
    engines = [
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    ]
    plugins = {
        engine: {
            "plugin_id": f"blueprint_{engine}_sim_engine_plugin",
            "framework": engine,
            "adapter_contract_status": "ready",
            "managed_execution_supported": True,
            "inputs": {
                "simulation_automation_plan": "simulation_automation_plan.json",
                "asset_conversion_plan": "asset_conversion_plan.json",
                "scenario_variation_instances": "scenario_variation_instances.json",
                "episode_spec": "episode_spec.v1.json",
                "cpu_preflight_manifest": "cpu_simulator_preflight_manifest.json",
            },
        }
        for engine in engines
    }
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "simulator_engine_plugin_registry.json",
        {
            "schema_version": "simulator_engine_plugin_registry.v1",
            "status": "ready_for_gated_managed_execution",
            "engine_targets": engines,
            "plugin_count": len(plugins),
            "plugins": plugins,
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    plugin_gate = manifest["gates"]["simulator_engine_plugins"]
    assert plugin_gate["passed"] is False
    assert (
        "simulator_engine_plugin_registry_missing_local_input_artifacts" in plugin_gate["blockers"]
    )
    assert set(plugin_gate["evidence"]["missing_local_input_artifacts_by_plugin"]) == {
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    }
    assert plugin_gate["evidence"]["missing_local_input_artifacts_by_plugin"]["mujoco"] == [
        "asset_conversion_plan",
        "cpu_preflight_manifest",
        "episode_spec",
        "scenario_variation_instances",
        "simulation_automation_plan",
    ]


def test_live_robot_eval_closure_blocks_ready_world_model_plugins_missing_required_local_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-world-plugin-inputs-missing"
    engines = [
        "isaac_sim",
        "isaac_lab_arena",
        "mujoco",
        "pybullet",
        "newton",
    ]
    plugins = {
        engine: {
            "plugin_id": f"blueprint_{engine}_sim_engine_plugin",
            "framework": engine,
            "adapter_contract_status": "ready",
            "managed_execution_supported": True,
        }
        for engine in engines
    }
    world_engines = [
        "worldlabs_world_model",
        "marble_simready",
        "cosmos_predict",
        "native_site_reference",
    ]
    world_plugins = {
        engine: {
            "plugin_id": f"blueprint_{engine}_engine_plugin",
            "engine": engine,
            "adapter_contract_status": "ready",
            "managed_execution_supported": True,
            "source_status": "optional_missing",
            "inputs": {
                "simulation_automation_plan": "simulation_automation_plan.json",
                "scenario_variation_instances": "scenario_variation_instances.json",
                "site_card": "../robot_eval_dataset/site_card.json",
                "task_cards": "../robot_eval_dataset/task_cards.json",
                "scenario_cards": "../robot_eval_dataset/scenario_cards.json",
                "world_manifest": "../worldlabs_world_manifest.json",
            },
        }
        for engine in world_engines
    }
    _write_json(
        capture_root
        / "pipeline"
        / "simulation_automation"
        / "simulator_engine_plugin_registry.json",
        {
            "schema_version": "simulator_engine_plugin_registry.v1",
            "status": "ready_for_gated_managed_execution",
            "engine_targets": engines,
            "world_model_engine_targets": world_engines,
            "plugin_count": len(plugins),
            "world_model_plugin_count": len(world_plugins),
            "plugins": plugins,
            "world_model_plugins": world_plugins,
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    plugin_gate = manifest["gates"]["simulator_engine_plugins"]
    assert plugin_gate["passed"] is False
    assert (
        "world_model_engine_plugin_registry_missing_local_input_artifacts"
        in plugin_gate["blockers"]
    )
    missing_by_plugin = plugin_gate["evidence"][
        "missing_local_input_artifacts_by_world_model_plugin"
    ]
    assert set(missing_by_plugin) == set(world_engines)
    assert set(missing_by_plugin["worldlabs_world_model"]) == {
        "scenario_cards",
        "scenario_variation_instances",
        "simulation_automation_plan",
        "site_card",
        "task_cards",
    }
    assert "world_manifest" not in missing_by_plugin["worldlabs_world_model"]


def test_live_robot_eval_closure_blocks_incomplete_scenario_eval_matrix(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-incomplete-matrix"
    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "eval_card_count": 1,
            "cards": [{"eval_card_id": "eval-card-1", "task_id": "place_return_in_bin"}],
        },
    )
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "required_variation_names": ["lighting_variation", "glare"],
            "variation_names_covered": ["lighting_variation"],
            "missing_required_variation_names": ["glare"],
            "runs": [
                {
                    "scenario_eval_run_id": "place_return_in_bin_scenario_lighting_run_0001",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["scenario_eval_suite"]
    assert gate["passed"] is False
    assert "scenario_eval_matrix_missing_required_variations" in gate["blockers"]
    assert gate["evidence"]["missing_required_variation_names"] == ["glare"]


def test_live_robot_eval_closure_blocks_name_only_scenario_eval_matrix_runs(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-name-only-matrix"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "required_variation_names": ["lighting_variation"],
            "variation_names_covered": ["lighting_variation"],
            "missing_required_variation_names": [],
            "runs": [
                {
                    "scenario_eval_run_id": "primary-lighting-run-0001",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["scenario_eval_suite"]
    assert gate["passed"] is False
    assert "scenario_eval_matrix_runs_missing_concrete_variation_details" in gate["blockers"]
    assert gate["evidence"]["scenario_eval_runs_missing_concrete_details"] == [
        {
            "row_index": 1,
            "scenario_eval_run_id": "primary-lighting-run-0001",
            "scenario_variation_instance_id": "",
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "variation_name": "lighting_variation",
            "missing_fields": [
                "scenario_variation_instance_id",
                "concrete_mutation",
                "engine_mutations",
            ],
        }
    ]


def test_live_robot_eval_closure_blocks_scenario_eval_matrix_missing_per_scenario_coverage(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-per-scenario-matrix"
    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "eval_card_count": 2,
            "cards": [
                {
                    "eval_card_id": "eval-card-primary",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "task_id": "place_return_in_bin",
                    "prediction_source": "fixture",
                    "validation": {"actual_status": "needs_actual_outcome"},
                    "proof_boundary": "prediction_only_no_actual_outcome_no_deployment_claim",
                },
                {
                    "eval_card_id": "eval-card-secondary",
                    "scenario_id": "scenario_place_return_in_bin_secondary",
                    "task_id": "place_return_in_bin",
                    "prediction_source": "fixture",
                    "validation": {"actual_status": "needs_actual_outcome"},
                    "proof_boundary": "prediction_only_no_actual_outcome_no_deployment_claim",
                },
            ],
        },
    )
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 2,
            "required_variation_names": ["lighting_variation", "glare"],
            "variation_names_covered": ["glare", "lighting_variation"],
            "missing_required_variation_names": [],
            "runs": [
                {
                    "scenario_eval_run_id": "primary-lighting-run-0001",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                },
                {
                    "scenario_eval_run_id": "secondary-glare-run-0001",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_secondary",
                    "variation_name": "glare",
                },
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["scenario_eval_suite"]
    assert gate["passed"] is False
    assert "scenario_eval_matrix_missing_required_variations_per_scenario" in gate["blockers"]
    assert gate["evidence"]["missing_required_variations_by_scenario"] == [
        {
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "missing_variation_names": ["glare"],
        },
        {
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_secondary",
            "missing_variation_names": ["lighting_variation"],
        },
    ]


def test_live_robot_eval_closure_blocks_scenario_family_library_task_and_variation_gaps(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    task_path = robot_eval_dir / "task_cards.json"
    task_payload = _read_json(task_path)
    second_task = dict(task_payload["cards"][0])  # type: ignore[index]
    second_task.update(
        {
            "task_card_id": "task_card_inspect_label",
            "task_id": "inspect_label",
            "task_statement": "Inspect the shelf label before placing the item",
            "task_category": "inspection",
        }
    )
    task_payload["cards"].append(second_task)  # type: ignore[union-attr]
    task_payload["task_card_count"] = 2
    _write_json(task_path, task_payload)
    family_path = robot_eval_dir / "scenario_family_library.json"
    family_payload = _read_json(family_path)
    family_payload["families"][0]["variations"] = [  # type: ignore[index]
        variation
        for variation in family_payload["families"][0]["variations"]  # type: ignore[index]
        if variation["variation_id"] != "glare"
    ]
    family_payload["families"][0]["variation_count"] = len(  # type: ignore[index]
        family_payload["families"][0]["variations"]  # type: ignore[index]
    )
    _write_json(family_path, family_payload)
    _write_json(
        capture_root / "pipeline" / "simulation_automation" / "scenario_variation_instances.json",
        {
            "schema_version": "scenario_variation_instances.v1",
            "status": "completed",
            "required_variation_names": list(POLICY_REFERENCE_VARIATION_NAMES),
            "variation_names_instantiated": list(POLICY_REFERENCE_VARIATION_NAMES),
            "instance_count": len(POLICY_REFERENCE_VARIATION_NAMES),
            "instances": [
                {
                    "instance_id": f"variation-{variation_name}",
                    "variation_name": variation_name,
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                }
                for variation_name in POLICY_REFERENCE_VARIATION_NAMES
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=capture_root / "pipeline" / "robot_eval_jobs" / "job-scenario-family-gap",
    )

    gate = manifest["gates"]["scenario_library"]
    coverage = gate["evidence"]["scenario_family_task_coverage"]
    assert "scenario_family_library_missing_task_coverage" in gate["blockers"]
    assert "scenario_family_library_missing_required_variations" in gate["blockers"]
    assert coverage["missing_task_ids"] == ["inspect_label"]
    assert coverage["missing_required_variations_by_family"] == [
        {
            "family_id": "family_scenario_place_return_in_bin_mobile",
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "missing_variation_names": ["glare"],
        }
    ]


def test_live_robot_eval_closure_blocks_robot_pov_missing_required_run_ids(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-pov-coverage"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 2,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                },
                {
                    "scenario_eval_run_id": "scenario-run-2",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "glare",
                },
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_observation_manifest.json",
        {
            "schema_version": "robot_pov_observation_manifest.v1",
            "status": "completed",
            "observation_count": 2,
            "observations": [
                {
                    "observation_id": "obs-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                },
                {
                    "observation_id": "obs-2",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                },
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["robot_pov_generation"]
    assert gate["passed"] is False
    assert "robot_pov_missing_required_scenario_eval_run_ids" in gate["blockers"]
    assert gate["evidence"]["covered_scenario_eval_run_ids"] == ["scenario-run-1"]
    assert gate["evidence"]["missing_scenario_eval_run_ids"] == ["scenario-run-2"]


def test_live_robot_eval_closure_blocks_robot_pov_without_frame_sequence_artifacts(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-pov-missing-frames"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_observation_manifest.json",
        {
            "schema_version": "robot_pov_observation_manifest.v1",
            "status": "completed",
            "observation_count": 1,
            "local_render_sequence_count": 0,
            "local_render_frame_count": 0,
            "observations": [
                {
                    "observation_id": "obs-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "render_sequence_id": "sequence-missing",
                    "render_frame_paths": [],
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["robot_pov_generation"]
    assert gate["passed"] is False
    assert "missing_robot_pov_frame_sequence_manifest" in gate["blockers"]
    assert "missing_robot_pov_render_storyboard" in gate["blockers"]
    assert "robot_pov_local_render_frames_empty" in gate["blockers"]


def test_live_robot_eval_closure_blocks_robot_pov_observations_missing_required_fields(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-pov-missing-observation-fields"
    frame_path = job_dir / "robot_pov" / "frame-1.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"robot pov frame\n")
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_observation_manifest.json",
        {
            "schema_version": "robot_pov_observation_manifest.v1",
            "status": "completed",
            "observation_count": 1,
            "local_render_sequence_count": 1,
            "local_render_frame_count": 1,
            "observations": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_frame_sequence_manifest.json",
        {
            "schema_version": "robot_pov_frame_sequence_manifest.v1",
            "status": "completed",
            "sequence_count": 1,
            "total_frame_count": 1,
            "sequences": [
                {
                    "sequence_id": "sequence-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "frame_count": 1,
                    "frame_paths": ["robot_pov/frame-1.png"],
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_render_storyboard.json",
        {
            "schema_version": "robot_pov_render_storyboard.v1",
            "status": "completed",
            "storyboard_count": 1,
            "storyboards": [
                {
                    "storyboard_id": "storyboard-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "frames": [{"frame_path": "robot_pov/frame-1.png"}],
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["robot_pov_generation"]
    assert gate["passed"] is False
    assert "robot_pov_observations_missing_required_fields" in gate["blockers"]
    assert gate["evidence"]["observation_rows_missing_required_fields"] == [
        {
            "index": 0,
            "observation_id": None,
            "scenario_eval_run_id": "scenario-run-1",
            "missing_fields": [
                "camera",
                "generated_frame_path",
                "observation_id",
                "render_sequence_id",
                "render_storyboard_id",
                "scenario_id",
                "task_id",
            ],
        }
    ]


def test_live_robot_eval_closure_blocks_robot_pov_storyboard_missing_local_frame_files(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-pov-missing-storyboard-frame"
    frame_path = job_dir / "robot_pov" / "frame-1.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"robot pov frame\n")
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_observation_manifest.json",
        {
            "schema_version": "robot_pov_observation_manifest.v1",
            "status": "completed",
            "observation_count": 1,
            "local_render_sequence_count": 1,
            "local_render_frame_count": 1,
            "observations": [
                {
                    "observation_id": "obs-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "camera": {"name": "front", "frame": "base_link"},
                    "generated_frame_path": "robot_pov/frame-1.png",
                    "render_sequence_id": "sequence-1",
                    "render_storyboard_id": "storyboard-1",
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_frame_sequence_manifest.json",
        {
            "schema_version": "robot_pov_frame_sequence_manifest.v1",
            "status": "completed",
            "sequence_count": 1,
            "total_frame_count": 1,
            "sequences": [
                {
                    "sequence_id": "sequence-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "frame_count": 1,
                    "frame_paths": ["robot_pov/frame-1.png"],
                }
            ],
        },
    )
    _write_json(
        job_dir / "robot_pov_render_storyboard.json",
        {
            "schema_version": "robot_pov_render_storyboard.v1",
            "status": "completed",
            "storyboard_count": 1,
            "storyboards": [
                {
                    "storyboard_id": "storyboard-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "frames": [{"frame_path": "robot_pov/missing-frame.png"}],
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["robot_pov_generation"]
    assert gate["passed"] is False
    assert "robot_pov_storyboard_local_frame_files_missing" in gate["blockers"]
    assert gate["evidence"]["missing_storyboard_frame_paths"] == ["robot_pov/missing-frame.png"]


def test_live_robot_eval_closure_blocks_unlabeled_failed_scenario_eval_runs(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-failure-label-coverage"
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "robot_eval_simulator_command_normalized_attempt_trace.v1",
            "status": "completed",
            "attempt_count": 2,
            "attempts": [
                {
                    "attempt_id": "attempt-run-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "success": False,
                    "failure_mode_ids": ["failure_navigation_blocked"],
                },
                {
                    "attempt_id": "attempt-run-2",
                    "scenario_eval_run_id": "scenario-run-2",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "success": False,
                    "failure_mode_ids": ["failure_collision_risk"],
                },
            ],
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "robot_eval_simulator_command_failure_labels.v1",
            "status": "review_required",
            "label_count": 1,
            "labels": [
                {
                    "label_id": "label_attempt_run_1",
                    "attempt_id": "attempt-run-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "failure_mode_ids": ["failure_navigation_blocked"],
                    "status": "review_required",
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["failure_labels"]
    assert gate["passed"] is False
    assert "failure_labels_missing_failed_attempt_coverage" in gate["blockers"]
    assert gate["evidence"]["failed_attempt_count"] == 2
    assert gate["evidence"]["covered_failed_attempt_ids"] == ["attempt-run-1"]
    assert gate["evidence"]["missing_failed_attempt_ids"] == ["attempt-run-2"]
    assert gate["evidence"]["missing_failed_scenario_eval_run_ids"] == ["scenario-run-2"]


def test_live_robot_eval_closure_blocks_unlabeled_failed_policy_attempts(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-failure-labels"
    _write_json(
        job_dir / "policy_execution_trace.json",
        {
            "schema_version": "robot_policy_execution_trace.v1",
            "status": "completed",
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "policy-attempt-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "policy_id": "policy-api",
                    "status": "failed",
                    "success": False,
                    "actions": [{"type": "move_base", "target": "bin_approach"}],
                    "failure_mode_ids": ["failure_wrong_object"],
                }
            ],
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "robot_eval_failure_labels.v1",
            "status": "review_required",
            "label_count": 0,
            "labels": [],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["failure_labels"]
    assert gate["passed"] is False
    assert "failure_labels_missing_failed_attempt_coverage" in gate["blockers"]
    assert gate["evidence"]["failed_policy_attempt_count"] == 1
    assert gate["evidence"]["missing_failed_attempt_ids"] == ["policy-attempt-1"]
    assert gate["evidence"]["missing_failed_scenario_eval_run_ids"] == ["scenario-run-1"]


def test_live_robot_eval_closure_blocks_failure_labels_without_failure_modes(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-empty-failure-label"
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "robot_eval_simulator_command_normalized_attempt_trace.v1",
            "status": "completed",
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-run-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "success": False,
                    "failure_mode_ids": ["failure_navigation_blocked"],
                }
            ],
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "schema_version": "robot_eval_simulator_command_failure_labels.v1",
            "status": "review_required",
            "label_count": 1,
            "labels": [
                {
                    "label_id": "label_attempt_run_1",
                    "attempt_id": "attempt-run-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "failure_mode_ids": [],
                    "status": "review_required",
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["failure_labels"]
    assert gate["passed"] is False
    assert "failure_labels_missing_failure_mode_ids" in gate["blockers"]
    assert gate["evidence"]["labels_missing_failure_mode_ids"] == ["label_attempt_run_1"]


def test_live_robot_eval_closure_blocks_invalid_scorecard_metric_values(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-invalid-scorecard"
    _write_json(
        job_dir / "evaluation_result.json",
        {
            "schema_version": "robot_eval_evaluation_result.v1",
            "status": "completed",
            "standard_policy_scorecard": {
                "success_rate": "one",
                "cycle_time": {"mean_seconds": -1.0, "sample_count": 1},
                "intervention_rate": -0.25,
                "unsafe_proximity": {"event_count": -1},
                "collision_risk": {"event_count": 0},
                "object_drop": {"event_count": 0},
                "wrong_object": {"event_count": 0},
                "timeout": {"event_count": 0},
                "recovery_success": {
                    "success_rate": 2.0,
                    "success_count": 2,
                    "attempt_count": 1,
                },
                "world_model_uncertainty": {
                    "status": "scored",
                    "mean_score": 1.5,
                    "sample_count": 1,
                },
                "sim_vs_real_calibration_score": "n/a",
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["evaluation_methodology"]
    assert gate["passed"] is False
    assert "standard_policy_scorecard_invalid_metric_values" in gate["blockers"]
    assert set(gate["evidence"]["invalid_scorecard_fields"]) == {
        "success_rate",
        "cycle_time",
        "intervention_rate",
        "unsafe_proximity",
        "recovery_success",
        "world_model_uncertainty",
        "sim_vs_real_calibration_score",
    }


def test_live_robot_eval_closure_blocks_evaluation_scorecard_missing_run_coverage(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-incomplete-evaluation-coverage"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 2,
            "runs": [
                {"scenario_eval_run_id": "scenario-run-1"},
                {"scenario_eval_run_id": "scenario-run-2"},
            ],
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "robot_eval_job_normalized_attempt_trace.v1",
            "status": "completed",
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-run-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "success": True,
                    "metrics": {"cycle_time_seconds": 10.0},
                }
            ],
        },
    )
    _write_json(
        job_dir / "evaluation_result.json",
        {
            "schema_version": "robot_eval_evaluation_result.v1",
            "status": "completed",
            "normalized_attempt_trace_path": "normalized_attempt_trace.json",
            "standard_policy_scorecard": {
                "success_rate": 1.0,
                "cycle_time": {"mean_seconds": 10.0, "sample_count": 1},
                "intervention_rate": 0.0,
                "unsafe_proximity": {"event_count": 0},
                "collision_risk": {"event_count": 0},
                "object_drop": {"event_count": 0},
                "wrong_object": {"event_count": 0},
                "timeout": {"event_count": 0},
                "recovery_success": {
                    "success_rate": None,
                    "success_count": 0,
                    "attempt_count": 0,
                },
                "world_model_uncertainty": {
                    "status": "not_available",
                    "mean_score": None,
                    "sample_count": 0,
                },
                "sim_vs_real_calibration_score": None,
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["evaluation_methodology"]
    assert gate["passed"] is False
    assert "evaluation_scorecard_missing_required_scenario_eval_run_ids" in gate["blockers"]
    assert gate["evidence"]["required_scenario_eval_run_ids"] == [
        "scenario-run-1",
        "scenario-run-2",
    ]
    assert gate["evidence"]["scored_scenario_eval_run_ids"] == ["scenario-run-1"]
    assert gate["evidence"]["missing_scenario_eval_run_ids"] == ["scenario-run-2"]


def test_live_robot_eval_closure_blocks_invalid_selected_policy_modality(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-invalid-policy-interface"
    _write_json(
        job_dir / "policy_package_manifest.json",
        {
            "schema_version": "robot_eval_policy_package_manifest.v1",
            "status": "review_required",
            "selected_modalities": ["docker_container"],
            "modalities": {
                "policy_api_endpoint": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "docker_container": {
                    "status": "blocked",
                    "selected": True,
                    "reference": {"image_ref": "registry.example/robot/policy:latest"},
                    "missing_inputs": ["policy_package.docker_container.digest"],
                },
                "recorded_action_trace": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "high_level_skill_trace": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "teleop_demo": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "sim_controller_plugin": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["policy_interface"]
    assert gate["passed"] is False
    assert "policy_interface_selected_modalities_invalid" in gate["blockers"]
    assert gate["evidence"]["selected_modality_statuses"]["docker_container"] == "blocked"
    assert gate["evidence"]["selected_modality_missing_inputs"] == {
        "docker_container": ["policy_package.docker_container.digest"]
    }


def test_live_robot_eval_closure_blocks_missing_local_selected_policy_reference(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-missing-policy-ref"
    _write_json(
        job_dir / "policy_package_manifest.json",
        {
            "schema_version": "robot_eval_policy_package_manifest.v1",
            "status": "review_required",
            "selected_modalities": ["recorded_action_trace"],
            "modalities": {
                "policy_api_endpoint": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "docker_container": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "recorded_action_trace": {
                    "status": "reference_present_requires_owner_system_review",
                    "selected": True,
                    "reference": {
                        "trace_manifest_uri": "policy_refs/missing-recorded-action-trace.json",
                        "timestamp_alignment": "aligned_to_capture_timestamps",
                    },
                    "missing_inputs": [],
                },
                "high_level_skill_trace": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "teleop_demo": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
                "sim_controller_plugin": {
                    "status": "not_selected",
                    "selected": False,
                    "reference": {},
                    "missing_inputs": [],
                },
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["policy_interface"]
    assert gate["passed"] is False
    assert "policy_interface_selected_modalities_invalid" in gate["blockers"]
    assert gate["evidence"]["selected_modality_missing_local_ref_keys"] == {
        "recorded_action_trace": ["trace_manifest_uri"]
    }
    assert gate["evidence"]["selected_modality_missing_inputs"] == {
        "recorded_action_trace": [
            "policy_package.recorded_action_trace.trace_manifest_uri_local_file_missing"
        ]
    }


def test_live_robot_eval_closure_blocks_policy_execution_without_actions_or_skills(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-no-actions"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )
    _write_json(
        job_dir / "policy_execution_manifest.json",
        {
            "schema_version": "robot_policy_execution_manifest.v1",
            "status": "completed",
            "robot_policy_execution_proven": True,
            "attempt_count": 1,
            "selected_modalities": ["policy_api_endpoint"],
        },
    )
    _write_json(
        job_dir / "policy_execution_trace.json",
        {
            "schema_version": "robot_policy_execution_trace.v1",
            "status": "completed",
            "robot_policy_execution_proven": True,
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "policy_id": "policy-api",
                    "status": "completed",
                    "success": True,
                    "actions": [],
                    "skills": [],
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_policy_execution"]
    assert gate["passed"] is False
    assert "policy_execution_attempts_missing_action_or_skill_trace" in gate["blockers"]
    assert gate["evidence"]["attempts_missing_action_or_skill_trace"] == ["attempt-1"]


def test_live_robot_eval_closure_rejects_reference_replay_as_live_policy_proof(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-reference-spoof"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )
    _write_json(
        job_dir / "policy_execution_manifest.json",
        {
            "schema_version": "robot_policy_execution_manifest.v1",
            "status": "completed",
            "robot_policy_execution_proven": True,
            "attempt_count": 1,
            "selected_modalities": ["recorded_action_trace"],
            "modality_results": {
                "recorded_action_trace": {
                    "status": "completed_reference_replay",
                    "execution_performed": False,
                    "reference_replayed": True,
                    "attempt_count": 1,
                    "robot_policy_execution_proven": True,
                }
            },
        },
    )
    _write_json(
        job_dir / "policy_execution_trace.json",
        {
            "schema_version": "robot_policy_execution_trace.v1",
            "status": "completed",
            "robot_policy_execution_proven": True,
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "policy_id": "recorded-policy",
                    "status": "completed",
                    "success": True,
                    "actions": [{"type": "move_base", "target": "bin_approach"}],
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_policy_execution"]
    assert gate["passed"] is False
    assert "policy_execution_missing_proven_executed_modality" in gate["blockers"]
    assert "policy_execution_selected_modalities_reference_replay_only" in gate["blockers"]
    assert gate["evidence"]["policy_execution_result_audit"]["reference_only_modalities"] == [
        "recorded_action_trace"
    ]
    assert gate["evidence"]["policy_execution_result_audit"]["proven_executed_modalities"] == []


def test_live_robot_eval_closure_blocks_policy_execution_failed_status_even_with_proof_boolean(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-failed-status"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )
    _write_json(
        job_dir / "policy_execution_manifest.json",
        {
            "schema_version": "robot_policy_execution_manifest.v1",
            "status": "failed",
            "robot_policy_execution_proven": True,
            "attempt_count": 1,
            "selected_modalities": ["policy_api_endpoint"],
        },
    )
    _write_json(
        job_dir / "policy_execution_trace.json",
        {
            "schema_version": "robot_policy_execution_trace.v1",
            "status": "failed",
            "robot_policy_execution_proven": True,
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "policy_id": "policy-api",
                    "status": "completed",
                    "success": True,
                    "actions": [{"type": "move_base", "target": "bin_approach"}],
                    "skills": [],
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_policy_execution"]
    assert gate["passed"] is False
    assert "policy_execution_manifest_not_completed" in gate["blockers"]
    assert "policy_execution_trace_not_completed" in gate["blockers"]
    assert gate["evidence"]["policy_execution_manifest_status"] == "failed"
    assert gate["evidence"]["policy_execution_trace_status"] == "failed"


def test_live_robot_eval_closure_blocks_simulator_failed_status_even_with_proof_boolean(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-simulator-failed-status"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "variation_name": "lighting_variation",
                }
            ],
        },
    )
    _write_json(
        job_dir / "simulator_service_result.json",
        {
            "schema_version": "robot_eval_simulator_service_result.v1",
            "framework": "pybullet",
            "status": "failed",
            "simulators_run": True,
            "simulator_execution_proven": True,
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "robot_eval_simulator_command_normalized_attempt_trace.v1",
            "status": "failed",
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-run-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "success": True,
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["live_simulator_execution"]
    assert gate["passed"] is False
    assert "simulator_service_result_not_completed" in gate["blockers"]
    assert "normalized_attempt_trace_not_completed" in gate["blockers"]
    assert gate["evidence"]["simulator_status"] == "failed"
    assert gate["evidence"]["normalized_attempt_trace_status"] == "failed"


def test_robot_eval_job_runs_policy_command_and_pairs_real_world_outcomes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    _write_json(
        capture_root / "pipeline" / "robot_eval_inputs" / "actual_outcome_manifest.json",
        {
            "schema_version": "actual_outcome_manifest.v1",
            "records": [
                {
                    "outcome_id": "pilot-outcome-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "scenario_eval_run_id": _scenario_eval_run_id("lighting_variation", 1),
                    "scenario_variation_instance_id": _scenario_variation_instance_id(
                        "lighting_variation"
                    ),
                    "policy_id": "policy-command",
                    "actual_success": False,
                    "failure_mode_ids": ["failure_collision_risk"],
                    "cycle_time_seconds": 22.0,
                    "intervention_count": 1,
                    "tuning_hours": 3.5,
                    "tuning_iterations": 2,
                    "tuning_notes": ["slowed approach near bin"],
                    "site_modifications": [{"modification": "moved cart 0.5m from approach path"}],
                    "site_modifications_helped": True,
                    "evidence_refs": {"pilot_log": "file://pilot-log.json"},
                }
            ],
        },
    )
    policy_script = tmp_path / "policy_adapter.py"
    policy_script.write_text(
        "\n".join(
            [
                "import json, os",
                "out = os.environ['BLUEPRINT_POLICY_EXECUTION_OUTPUT']",
                "payload = {",
                "  'attempts': [{",
                "    'attempt_id': 'policy-command-attempt-1',",
                "    'task_id': 'place_return_in_bin',",
                "    'scenario_id': 'scenario_place_return_in_bin_mobile',",
                "    'policy_id': 'policy-command',",
                "    'status': 'completed',",
                "    'success': True,",
                "    'actions': [{'type': 'move_base', 'target': 'bin_approach'}],",
                "    'metrics': {'policy_latency_ms': 42}",
                "  }]",
                "}",
                "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-policy-and-real-world",
        provisioner="fixture_local",
        simulator="fixture",
        allow_policy_execution=True,
        policy_execution_commands={
            "policy_api_endpoint": f"{sys.executable} {policy_script}",
        },
    )

    job_dir = Path(result["job_dir"])
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")
    policy_trace = _read_json(job_dir / "policy_execution_trace.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    calibration = _read_json(job_dir / "sim_vs_real_calibration_report.json")
    deployment_summary = _read_json(job_dir / "prediction_vs_actual_deployment_summary.json")
    followup_plan = _read_json(job_dir / "real_world_validation_followup_plan.json")
    followup_queue = _read_json(job_dir / "real_world_validation_followup_request_queue.json")
    evaluation = _read_json(job_dir / "evaluation_result.json")
    package = _read_json(job_dir / "post_training_data_package_export_manifest.json")
    robot_eval_report = _read_json(job_dir / "robot_eval_report.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    beta_checks = {
        check["check_id"]: check
        for check in live_closure["robot_team_beta_readiness"]["checks"]
    }

    assert policy_execution["robot_policy_execution_proven"] is True
    assert live_closure["robot_team_beta_readiness"]["ready_for_beta"] is False
    assert "live_provider_worker_execution" in live_closure["robot_team_beta_readiness"][
        "blocked_check_ids"
    ]
    assert beta_checks["robot_pov_policy_evidence"]["passed"] is False
    assert "real_robot_pov_evidence:real_robot_pov_evidence_not_proven" in beta_checks[
        "robot_pov_policy_evidence"
    ]["blockers"]
    assert beta_checks["deployment_outcome_joins"]["passed"] is True
    assert beta_checks["deployment_outcome_joins"]["evidence"][
        "real_world_outcome_record_count"
    ] == 1
    assert (
        beta_checks["deployment_outcome_joins"]["evidence"][
            "predicted_vs_actual_gate_passed"
        ]
        is True
    )
    assert policy_execution["modality_results"]["policy_api_endpoint"]["status"] == "completed"
    assert policy_trace["attempt_count"] >= 1
    assert proof_boundary["robot_policy_execution_proven"] is True
    assert proof_boundary["real_world_outcome_proven"] is True

    assert deployment["status"] == "completed"
    assert deployment["real_world_outcome_records_present"] is True
    assert deployment["owner_evidence_record_count"] == 1
    assert deployment["missing_owner_evidence_record_ids"] == []
    assert deployment["real_world_outcome_proven"] is True
    assert deployment["records"][0]["owner_evidence_present"] is True
    assert calibration["status"] == "completed"
    assert calibration["sim_vs_real_calibration_score"] == 0.0
    assert calibration["exact_prediction_record_count"] == 1
    assert calibration["weak_prediction_match_record_count"] == 0
    assert evaluation["standard_policy_scorecard"]["sim_vs_real_calibration_score"] == 0.0
    assert (
        evaluation["sim_vs_real_calibration_report_path"] == "sim_vs_real_calibration_report.json"
    )
    assert calibration["missed_failure_count"] == 1
    assert calibration["site_modification_count"] == 1
    assert deployment_summary["how_much_real_world_tuning_was_needed"] == {
        "tuning_hours_total": 3.5,
        "tuning_iterations_total": 2,
        "records_with_tuning": 1,
    }
    assert (
        deployment_summary["whether_site_modifications_helped"][0]["site_modifications_helped"]
        is True
    )
    assert deployment_summary["real_world_validation_followup_plan_path"] == (
        "real_world_validation_followup_plan.json"
    )
    assert followup_plan["schema_version"] == "real_world_validation_followup_plan.v1"
    assert followup_plan["status"] == "review_required"
    assert followup_plan["source_artifacts"] == {
        "deployment_outcome_ledger": "deployment_outcome_ledger.json",
        "sim_vs_real_calibration_report": "sim_vs_real_calibration_report.json",
        "prediction_vs_actual_deployment_summary": ("prediction_vs_actual_deployment_summary.json"),
    }
    assert followup_plan["summary"] == {
        "action_count": 4,
        "scenario_rerun_count": 1,
        "scenario_library_update_count": 1,
        "robot_team_tuning_review_count": 1,
        "site_modification_review_count": 1,
        "unmatched_actual_review_count": 0,
    }
    action_types = {action["action_type"] for action in followup_plan["follow_up_actions"]}
    assert action_types == {
        "rerun_scenario_eval",
        "update_scenario_library_for_missed_failures",
        "robot_team_tuning_review",
        "site_modification_review",
    }
    rerun_action = next(
        action
        for action in followup_plan["follow_up_actions"]
        if action["action_type"] == "rerun_scenario_eval"
    )
    assert rerun_action["scenario_eval_run_id"] == _scenario_eval_run_id(
        "lighting_variation",
        1,
    )
    assert rerun_action["scenario_variation_instance_id"] == (
        _scenario_variation_instance_id("lighting_variation")
    )
    assert "actual_failed" in rerun_action["reasons"]
    scenario_update = next(
        action
        for action in followup_plan["follow_up_actions"]
        if action["action_type"] == "update_scenario_library_for_missed_failures"
    )
    assert scenario_update["missed_failures"] == ["failure_collision_risk"]
    assert scenario_update["variation_name"] == "lighting_variation"
    assert followup_queue["schema_version"] == "real_world_validation_followup_request_queue.v1"
    assert followup_queue["status"] == "ready_for_inbox_processing"
    assert followup_queue["queued_request_count"] == 1
    queued_request = followup_queue["queued_requests"][0]
    assert queued_request["schema_version"] == "robot_eval_job_request.v1"
    assert queued_request["job_id"] == "job-policy-and-real-world-followup-0001"
    assert queued_request["parent_job_id"] == "job-policy-and-real-world"
    assert queued_request["source_followup_action_id"] == rerun_action["action_id"]
    assert queued_request["requested_scenario_eval_runs"] == [
        {
            "scenario_eval_run_id": _scenario_eval_run_id("lighting_variation", 1),
            "scenario_variation_instance_id": _scenario_variation_instance_id("lighting_variation"),
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "variation_name": "lighting_variation",
            "source_followup_action_id": rerun_action["action_id"],
        }
    ]
    assert "actual_outcomes" not in queued_request
    queued_request_file = Path(followup_queue["queued_request_paths"][0])
    assert queued_request_file.is_file()
    assert _read_json(queued_request_file) == queued_request
    followup_result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=queued_request_file,
        job_id=queued_request["job_id"],
        provisioner="fixture_local",
        simulator="fixture",
    )
    followup_job_dir = Path(followup_result["job_dir"])
    followup_matrix = _read_json(followup_job_dir / "scenario_eval_matrix.json")
    followup_live_closure = _read_json(followup_job_dir / "live_eval_closure_manifest.json")
    assert followup_matrix["requested_scenario_eval_run_filter_count"] == 1
    assert followup_matrix["unmatched_requested_scenario_eval_run_filter_count"] == 0
    assert followup_matrix["scenario_eval_run_count"] == 1
    assert followup_matrix["runs"][0]["scenario_eval_run_id"] == _scenario_eval_run_id(
        "lighting_variation",
        1,
    )
    assert followup_matrix["runs"][0]["scenario_variation_instance_id"] == (
        _scenario_variation_instance_id("lighting_variation")
    )
    assert (
        followup_live_closure["gates"]["scenario_eval_suite"]["evidence"][
            "exact_followup_rerun_scope"
        ]
        is True
    )
    assert followup_live_closure["gates"]["scenario_eval_suite"]["passed"] is True
    assert robot_eval_report["real_world_validation"]["followup_plan_status"] == ("review_required")
    assert robot_eval_report["real_world_validation"]["followup_request_queue_status"] == (
        "ready_for_inbox_processing"
    )
    assert robot_eval_report["artifact_paths"]["real_world_validation_followup_plan"] == (
        "real_world_validation_followup_plan.json"
    )
    assert robot_eval_report["artifact_paths"]["real_world_validation_followup_request_queue"] == (
        "real_world_validation_followup_request_queue.json"
    )
    assert package["included_artifacts"]["sim_vs_real_calibration_report"] == (
        "sim_vs_real_calibration_report.json"
    )
    assert package["included_artifacts"]["deployment_outcome_ledger"] == (
        "deployment_outcome_ledger.json"
    )
    assert package["included_artifacts"]["real_world_validation_followup_plan"] == (
        "real_world_validation_followup_plan.json"
    )
    assert package["included_artifacts"]["real_world_validation_followup_request_queue"] == (
        "real_world_validation_followup_request_queue.json"
    )
    assert package["export_policy"]["policy_execution_trace_included"] is True
    assert package["export_policy"]["sim_vs_real_calibration_included"] is True
    assert package["export_policy"]["real_world_validation_followup_plan_included"] is True
    assert package["export_policy"]["real_world_validation_followup_queue_included"] is True
    assert run_manifest["robot_policy_execution_proven"] is True
    assert run_manifest["real_world_outcome_proven"] is True
    assert run_manifest["real_world_validation_followup_plan_status"] == "review_required"
    assert run_manifest["real_world_validation_followup_request_queue_status"] == (
        "ready_for_inbox_processing"
    )
    assert run_manifest["cpu_preflight_artifacts"]["real_world_validation_followup_plan"] == (
        "real_world_validation_followup_plan.json"
    )
    assert (
        run_manifest["cpu_preflight_artifacts"]["real_world_validation_followup_request_queue"]
        == "real_world_validation_followup_request_queue.json"
    )
    assert run_manifest["artifacts"]["real_world_validation_followup_plan"] == (
        "real_world_validation_followup_plan.json"
    )
    assert run_manifest["artifacts"]["real_world_validation_followup_request_queue"] == (
        "real_world_validation_followup_request_queue.json"
    )
    validation_gate = live_closure["gates"]["real_world_validation_loop"]
    assert validation_gate["evidence"]["followup_request_queue_status"] == (
        "ready_for_inbox_processing"
    )
    assert validation_gate["evidence"]["followup_request_queue_request_count"] == 1
    assert run_manifest["robot_readiness_proven"] is False


def test_robot_eval_job_runs_default_walk_to_target_policy_without_team_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = _full_job_request(capture_root)
    request.pop("policy_package")
    request["default_test_policy"] = {
        "policy_kind": "walk_to_target",
        "target": "receiving_dock_safe_spot",
    }
    request_path = tmp_path / "job-request-default-policy.json"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-default-walk-to-target",
        provisioner="fixture_local",
        simulator="fixture",
        allow_policy_execution=True,
    )

    job_dir = Path(result["job_dir"])
    policy_package = _read_json(job_dir / "policy_package_manifest.json")
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")
    policy_trace = _read_json(job_dir / "policy_execution_trace.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")

    assert policy_package["status"] == "review_required"
    assert policy_package["selected_modalities"] == ["high_level_skill_trace"]
    high_level_result = policy_execution["modality_results"]["high_level_skill_trace"]
    assert high_level_result["status"] == "completed"
    assert high_level_result["execution_performed"] is True
    assert high_level_result["reference_replayed"] is False
    assert high_level_result["default_test_policy"] is True
    assert high_level_result["default_test_policy_execution_proven"] is True
    assert high_level_result["robot_team_policy_execution_proven"] is False
    assert policy_execution["robot_policy_execution_proven"] is True
    assert policy_execution["default_test_policy_execution_proven"] is True
    assert policy_execution["robot_team_policy_execution_proven"] is False
    assert policy_execution["scenario_eval_run_coverage_complete"] is True
    assert policy_trace["robot_policy_execution_proven"] is True
    assert policy_trace["default_test_policy_execution_proven"] is True
    assert policy_trace["robot_team_policy_execution_proven"] is False
    assert all(
        attempt["policy_scope"] == "blueprint_default_test_policy"
        for attempt in policy_trace["attempts"]
    )
    assert all(
        attempt["target"] == "receiving_dock_safe_spot" for attempt in policy_trace["attempts"]
    )
    assert proof_boundary["robot_policy_execution_proven"] is True
    assert live_closure["gates"]["live_policy_execution"]["passed"] is True
    assert (
        live_closure["gates"]["live_policy_execution"]["evidence"][
            "default_test_policy_execution_proven"
        ]
        is True
    )
    assert (
        live_closure["gates"]["live_policy_execution"]["evidence"][
            "robot_team_policy_execution_proven"
        ]
        is False
    )
    assert live_closure["gates"]["real_robot_pov_evidence"]["passed"] is False


def test_robot_eval_job_ingests_inline_real_world_outcomes_from_job_request(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["actual_outcomes"] = {
        "schema_version": "actual_outcome_manifest.v1",
        "records": [
            {
                "outcome_id": "inline-pilot-outcome-1",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "policy_id": "fixture-policy",
                "actual_success": True,
                "failure_mode_ids": [],
                "cycle_time_seconds": 10.5,
                "intervention_count": 0,
                "real_world_tuning_needed": False,
                "site_modifications": [],
            }
        ],
    }
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-inline-real-world",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    calibration = _read_json(job_dir / "sim_vs_real_calibration_report.json")
    deployment_summary = _read_json(job_dir / "prediction_vs_actual_deployment_summary.json")
    intake = _read_json(job_dir / "deployment_outcome_intake_manifest.json")
    evaluation = _read_json(job_dir / "evaluation_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")

    assert deployment["status"] == "completed"
    assert deployment["real_world_outcome_records_present"] is True
    assert deployment["owner_evidence_record_count"] == 0
    assert deployment["missing_owner_evidence_record_ids"] == ["inline-pilot-outcome-1"]
    assert deployment["real_world_outcome_proven"] is False
    assert deployment["outcome_source"] == "job_request_inline_actual_outcomes"
    assert intake["status"] == "completed"
    assert intake["record_count"] == 1
    assert intake["real_world_outcome_records_present"] is True
    assert intake["real_world_outcome_proven"] is False
    assert calibration["status"] == "blocked_weak_prediction_matches"
    assert calibration["real_world_outcome_records_present"] is True
    assert calibration["real_world_outcome_proven"] is False
    assert calibration["sim_vs_real_calibration_score"] is None
    assert calibration["exact_prediction_record_count"] == 0
    assert calibration["weak_prediction_match_record_count"] == 1
    assert calibration["weak_prediction_match_record_ids"] == ["inline-pilot-outcome-1"]
    assert evaluation["standard_policy_scorecard"]["sim_vs_real_calibration_score"] is None
    assert deployment_summary["real_world_outcome_proven"] is False
    assert deployment_summary["weak_prediction_match_record_ids"] == ["inline-pilot-outcome-1"]
    assert deployment_summary["what_actually_happened"][0]["actual_success"] is True
    assert deployment_summary["what_actually_happened"][0]["exact_prediction_match"] is False
    predicted_gate = live_closure["gates"]["predicted_vs_actual_calibration"]
    assert predicted_gate["passed"] is False
    assert "predicted_vs_actual_weak_prediction_matches" in predicted_gate["blockers"]
    assert "predicted_vs_actual_no_exact_prediction_matches" in predicted_gate["blockers"]
    assert predicted_gate["evidence"]["weak_prediction_match_record_ids"] == [
        "inline-pilot-outcome-1"
    ]
    assert run_manifest["real_world_outcome_records_present"] is True
    assert run_manifest["owner_evidence_record_count"] == 0
    assert run_manifest["missing_owner_evidence_record_ids"] == ["inline-pilot-outcome-1"]
    assert run_manifest["real_world_outcome_proven"] is False


def test_robot_eval_job_marks_run_only_deployment_outcomes_as_missing_exact_join_keys(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["actual_outcomes"] = {
        "schema_version": "actual_outcome_manifest.v1",
        "records": [
            {
                "outcome_id": "pilot-run-only-1",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "scenario_eval_run_id": _scenario_eval_run_id("lighting_variation", 1),
                "policy_id": "fixture-policy",
                "actual_success": True,
                "failure_mode_ids": [],
                "cycle_time_seconds": 10.5,
                "intervention_count": 0,
                "evidence_refs": {"pilot_log": "owner://pilot/run-only-1"},
            }
        ],
    }
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-run-only-real-world",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    calibration = _read_json(job_dir / "sim_vs_real_calibration_report.json")
    summary = _read_json(job_dir / "prediction_vs_actual_deployment_summary.json")

    assert deployment["missing_exact_prediction_join_key_record_ids"] == ["pilot-run-only-1"]
    assert "deployment_outcomes_missing_exact_prediction_join_keys" in deployment["blockers"]
    assert deployment["records"][0]["prediction_match_level"] == "scenario_eval_run"
    assert deployment["records"][0]["exact_prediction_match"] is False
    assert calibration["status"] == "blocked_weak_prediction_matches"
    assert calibration["missing_exact_prediction_join_key_record_ids"] == ["pilot-run-only-1"]
    assert summary["missing_exact_prediction_join_key_record_ids"] == ["pilot-run-only-1"]


def test_robot_eval_job_blocks_deployment_outcomes_without_actual_result_signal(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["actual_outcomes"] = {
        "schema_version": "actual_outcome_manifest.v1",
        "records": [
            {
                "outcome_id": "pilot-missing-result-1",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "policy_id": "fixture-policy",
                "cycle_time_seconds": 10.5,
                "intervention_count": 0,
                "evidence_refs": {"pilot_log": "owner://pilot/missing-result-1"},
            }
        ],
    }
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-missing-actual-result",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    gate = live_closure["gates"]["real_world_validation_loop"]

    assert deployment["owner_evidence_record_count"] == 1
    assert deployment["missing_owner_evidence_record_ids"] == []
    assert deployment["missing_actual_result_signal_record_ids"] == ["pilot-missing-result-1"]
    assert "deployment_outcomes_missing_actual_result_signal" in deployment["blockers"]
    assert deployment["real_world_outcome_proven"] is False
    assert proof_boundary["real_world_outcome_proven"] is False
    assert run_manifest["real_world_outcome_proven"] is False
    assert gate["passed"] is False
    assert "deployment_outcomes_missing_actual_result_signal" in gate["blockers"]


def test_live_robot_eval_closure_recomputes_real_world_owner_evidence_from_rows(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-spoofed-real-world"
    _write_json(
        job_dir / "deployment_outcome_intake_manifest.json",
        {
            "schema_version": "deployment_outcome_intake_manifest.v1",
            "status": "completed",
            "record_count": 1,
            "real_world_outcome_records_present": True,
            "real_world_outcome_proven": False,
        },
    )
    _write_json(
        job_dir / "deployment_outcome_ledger.json",
        {
            "schema_version": "deployment_outcome_ledger.v1",
            "status": "completed",
            "record_count": 1,
            "real_world_outcome_records_present": True,
            "real_world_outcome_proven": True,
            "owner_evidence_record_count": 1,
            "missing_owner_evidence_record_ids": [],
            "missing_actual_result_signal_record_ids": [],
            "records": [
                {
                    "record_id": "spoofed-owner-evidence",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "actual_success": True,
                    "owner_evidence_present": True,
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["real_world_validation_loop"]
    assert gate["passed"] is False
    assert "deployment_outcomes_missing_owner_evidence" in gate["blockers"]
    assert gate["evidence"]["ledger_real_world_outcome_proven_claimed"] is True
    assert gate["evidence"]["record_level_real_world_outcome_proven"] is False
    assert gate["evidence"]["real_world_outcome_proven"] is False
    assert gate["evidence"]["missing_owner_evidence_record_ids"] == ["spoofed-owner-evidence"]


def test_live_robot_eval_closure_requires_real_world_followup_plan(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-missing-followup-plan"
    _write_json(
        job_dir / "deployment_outcome_intake_manifest.json",
        {
            "schema_version": "deployment_outcome_intake_manifest.v1",
            "status": "completed",
            "record_count": 1,
            "real_world_outcome_records_present": True,
            "real_world_outcome_proven": False,
        },
    )
    _write_json(
        job_dir / "deployment_outcome_ledger.json",
        {
            "schema_version": "deployment_outcome_ledger.v1",
            "status": "completed",
            "record_count": 1,
            "real_world_outcome_records_present": True,
            "real_world_outcome_proven": True,
            "owner_evidence_record_count": 1,
            "missing_owner_evidence_record_ids": [],
            "missing_actual_result_signal_record_ids": [],
            "records": [
                {
                    "record_id": "proven-without-followup",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "actual_success": True,
                    "owner_evidence_present": True,
                    "owner_evidence_refs": {"pilot_log": "owner://pilot/proven"},
                }
            ],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["real_world_validation_loop"]
    assert gate["passed"] is False
    assert "real_world_validation_followup_plan_missing" in gate["blockers"]
    assert gate["evidence"]["real_world_validation_followup_plan"]["exists"] is False


def test_robot_eval_job_blocks_predicted_vs_actual_with_unmatched_actual_run_id(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["actual_outcomes"] = {
        "schema_version": "actual_outcome_manifest.v1",
        "records": [
            {
                "outcome_id": "pilot-unmatched-run-1",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "scenario_eval_run_id": "not-in-scenario-eval-matrix",
                "policy_id": "fixture-policy",
                "actual_success": True,
                "failure_mode_ids": [],
                "cycle_time_seconds": 10.5,
                "intervention_count": 0,
                "evidence_refs": {"pilot_log": "owner://pilot/unmatched-run-1"},
            }
        ],
    }
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-unmatched-real-world-run",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    calibration = _read_json(job_dir / "sim_vs_real_calibration_report.json")
    deployment_summary = _read_json(job_dir / "prediction_vs_actual_deployment_summary.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    gate = live_closure["gates"]["predicted_vs_actual_calibration"]

    assert deployment.get("unmatched_actual_record_ids") == ["pilot-unmatched-run-1"]
    assert "deployment_outcomes_missing_matching_prediction" in deployment["blockers"]
    assert calibration.get("unmatched_actual_record_count") == 1
    assert calibration.get("unmatched_actual_record_ids") == ["pilot-unmatched-run-1"]
    assert deployment_summary.get("unmatched_actual_record_ids") == ["pilot-unmatched-run-1"]
    assert gate["passed"] is False
    assert "predicted_vs_actual_unmatched_actual_records" in gate["blockers"]
    assert gate["evidence"]["unmatched_actual_record_ids"] == ["pilot-unmatched-run-1"]


def test_live_robot_eval_closure_blocks_invalid_predicted_vs_actual_score_without_matches(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-invalid-calibration"
    _write_json(
        job_dir / "sim_vs_real_calibration_report.json",
        {
            "schema_version": "sim_vs_real_calibration_report.v1",
            "status": "completed",
            "sim_vs_real_calibration_score": 1.5,
            "matched_prediction_record_count": 0,
            "unmatched_actual_record_count": 0,
            "unmatched_actual_record_ids": [],
            "prediction_match_counts": {
                "scenario_eval_run_and_variation": 0,
                "scenario_eval_run": 0,
                "scenario_variation_instance": 0,
                "task_scenario_fallback": 0,
                "unmatched": 0,
            },
        },
    )
    _write_json(
        job_dir / "prediction_vs_actual_deployment_summary.json",
        {
            "schema_version": "prediction_vs_actual_deployment_summary.v1",
            "status": "completed",
            "sim_vs_real_calibration_score": 1.5,
            "matched_prediction_record_count": 0,
            "unmatched_actual_record_count": 0,
            "unmatched_actual_record_ids": [],
            "prediction_match_counts": {
                "scenario_eval_run_and_variation": 0,
                "scenario_eval_run": 0,
                "scenario_variation_instance": 0,
                "task_scenario_fallback": 0,
                "unmatched": 0,
            },
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["predicted_vs_actual_calibration"]
    assert gate["passed"] is False
    assert "sim_vs_real_calibration_score_invalid" in gate["blockers"]
    assert "predicted_vs_actual_no_matched_prediction_records" in gate["blockers"]
    assert gate["evidence"]["matched_prediction_record_count"] == 0


def test_live_robot_eval_closure_blocks_incomplete_predicted_vs_actual_summary(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-incomplete-pva-summary"
    _write_json(
        job_dir / "sim_vs_real_calibration_report.json",
        {
            "schema_version": "sim_vs_real_calibration_report.v1",
            "status": "completed",
            "sim_vs_real_calibration_score": 0.5,
            "matched_prediction_record_count": 1,
            "unmatched_actual_record_count": 0,
            "unmatched_actual_record_ids": [],
            "prediction_match_counts": {
                "scenario_eval_run_and_variation": 1,
                "scenario_eval_run": 0,
                "scenario_variation_instance": 0,
                "task_scenario_fallback": 0,
                "unmatched": 0,
            },
        },
    )
    _write_json(
        job_dir / "prediction_vs_actual_deployment_summary.json",
        {
            "schema_version": "prediction_vs_actual_deployment_summary.v1",
            "status": "completed",
            "sim_vs_real_calibration_score": 0.5,
            "matched_prediction_record_count": 1,
            "unmatched_actual_record_count": 0,
            "unmatched_actual_record_ids": [],
        },
    )

    manifest = build_live_robot_eval_closure_manifest(
        capture_root=capture_root,
        job_dir=job_dir,
    )

    gate = manifest["gates"]["predicted_vs_actual_calibration"]
    assert gate["passed"] is False
    assert "prediction_vs_actual_summary_missing_required_sections" in gate["blockers"]
    assert gate["evidence"]["missing_summary_sections"] == [
        "how_much_real_world_tuning_was_needed",
        "what_actually_happened",
        "what_eval_predicted",
        "whether_site_modifications_helped",
        "which_failures_were_missed",
        "which_scenarios_predicted_failure",
    ]


def test_robot_eval_job_ingests_deployment_outcome_inbox(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    inbox = capture_root / "pipeline" / "robot_eval_inputs" / "deployment_outcomes" / "inbox"
    inbox.mkdir(parents=True)
    _write_json(
        inbox / "pilot-outcome-001.json",
        {
            "schema_version": "deployment_outcome.v1",
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "policy_id": "fixture-policy",
            "actual_success": False,
            "failure_mode_ids": ["missed_blocked_path"],
            "cycle_time_seconds": 31.0,
            "intervention_count": 1,
            "real_world_tuning_needed": True,
            "tuning_iterations": 1,
            "tuning_hours": 0.75,
            "site_modifications": ["moved_cart_from_approach_lane"],
            "site_modifications_helped": True,
        },
    )
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-deployment-inbox",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    intake = _read_json(job_dir / "deployment_outcome_intake_manifest.json")
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    calibration = _read_json(job_dir / "sim_vs_real_calibration_report.json")
    evaluation = _read_json(job_dir / "evaluation_result.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")

    assert intake["status"] == "completed"
    assert intake["outcome_source"] == "deployment_outcome_inbox"
    assert intake["record_count"] == 1
    assert intake["real_world_outcome_records_present"] is True
    assert intake["real_world_outcome_proven"] is False
    assert intake["source_files"] == [str(inbox / "pilot-outcome-001.json")]
    assert deployment["status"] == "completed"
    assert deployment["outcome_source"] == "deployment_outcome_inbox"
    assert deployment["real_world_outcome_records_present"] is True
    assert deployment["owner_evidence_record_count"] == 0
    assert deployment["missing_owner_evidence_record_ids"] == ["deployment_outcome_0001"]
    assert deployment["real_world_outcome_proven"] is False
    assert deployment["records"][0]["missed_failures"] == ["missed_blocked_path"]
    assert deployment["records"][0]["owner_evidence_present"] is False
    assert calibration["status"] == "blocked_weak_prediction_matches"
    assert calibration["sim_vs_real_calibration_score"] is None
    assert calibration["exact_prediction_record_count"] == 0
    assert calibration["weak_prediction_match_record_ids"] == ["deployment_outcome_0001"]
    assert calibration["real_world_outcome_proven"] is False
    assert evaluation["standard_policy_scorecard"]["sim_vs_real_calibration_score"] is None
    assert proof_boundary["real_world_outcome_records_present"] is True
    assert proof_boundary["real_world_outcome_proven"] is False
    assert run_manifest["real_world_outcome_records_present"] is True
    assert run_manifest["real_world_outcome_proven"] is False
    validation_gate = live_closure["gates"]["real_world_validation_loop"]
    predicted_gate = live_closure["gates"]["predicted_vs_actual_calibration"]
    assert "deployment_outcomes_missing_owner_evidence" in validation_gate["blockers"]
    assert "predicted_vs_actual_weak_prediction_matches" in predicted_gate["blockers"]
    assert predicted_gate["evidence"]["weak_prediction_match_record_ids"] == [
        "deployment_outcome_0001"
    ]
    assert (
        "real_world_validation_loop:deployment_outcomes_missing_owner_evidence"
        in live_closure["blockers"]
    )


def test_robot_eval_job_ingests_job_specific_deployment_outcome_inbox(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    job_id = "job-specific-deployment-inbox"
    inbox = (
        capture_root / "pipeline" / "robot_eval_inputs" / job_id / "deployment_outcomes" / "inbox"
    )
    _write_json(
        inbox / "pilot-outcome-001.json",
        {
            "schema_version": "deployment_outcome.v1",
            "job_id": job_id,
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "policy_id": "fixture-policy",
            "actual_success": True,
            "failure_mode_ids": [],
            "cycle_time_seconds": 12.0,
            "intervention_count": 0,
            "evidence_refs": {"pilot_log": "owner://pilot/pilot-outcome-001"},
        },
    )
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id=job_id,
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    intake = _read_json(job_dir / "deployment_outcome_intake_manifest.json")
    deployment = _read_json(job_dir / "deployment_outcome_ledger.json")
    calibration = _read_json(job_dir / "sim_vs_real_calibration_report.json")

    assert intake["status"] == "completed"
    assert intake["outcome_source"] == "deployment_outcome_inbox"
    assert intake["source_files"] == [str(inbox / "pilot-outcome-001.json")]
    assert deployment["real_world_outcome_records_present"] is True
    assert deployment["owner_evidence_record_count"] == 1
    assert deployment["missing_owner_evidence_record_ids"] == []
    assert deployment["real_world_outcome_proven"] is True
    assert deployment["records"][0]["actual_success"] is True
    assert deployment["records"][0]["owner_evidence_present"] is True
    assert calibration["status"] == "blocked_weak_prediction_matches"
    assert calibration["sim_vs_real_calibration_score"] is None
    assert calibration["exact_prediction_record_count"] == 0
    assert calibration["weak_prediction_match_record_ids"] == ["deployment_outcome_0001"]


def test_robot_eval_job_normalizes_command_backed_simulator_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    simulator_script = tmp_path / "pybullet_runner.py"
    simulator_script.write_text(
        "\n".join(
            [
                "import json, os",
                "out = os.environ['BLUEPRINT_SIMULATOR_OUTPUT']",
                "matrix_path = os.environ['BLUEPRINT_SCENARIO_EVAL_MATRIX']",
                "matrix = json.load(open(matrix_path, encoding='utf-8'))",
                "required = [run['scenario_eval_run_id'] for run in matrix['runs']]",
                "payload = {",
                "  'required_scenario_eval_run_ids': required,",
                "  'attempts': [",
                "    {",
                "      'attempt_id': f'pybullet-attempt-{index + 1}',",
                "      'scenario_eval_run_id': run['scenario_eval_run_id'],",
                "      'task_id': run['task_id'],",
                "      'scenario_id': run['scenario_id'],",
                "      'scenario_variation_instance_id': run.get('scenario_variation_instance_id'),",
                "      'variation_name': run.get('variation_name'),",
                "      'policy_id': 'policy-command',",
                "      'status': 'completed',",
                "      'success': True,",
                "      'metrics': {'cycle_time_seconds': 11.0, 'intervention_count': 0},",
                "      'actions': [{'type': 'move_base', 'target': 'bin_approach'}]",
                "    }",
                "    for index, run in enumerate(matrix['runs'])",
                "  ]",
                "}",
                "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["simulator_preference"] = "pybullet"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-pybullet-command",
        provisioner="fixture_local",
        simulator="pybullet",
        allow_simulator_execution=True,
        allowed_simulators=["pybullet"],
        simulator_commands={"pybullet": f"{sys.executable} {simulator_script}"},
    )

    job_dir = Path(result["job_dir"])
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    provider_adapter = _read_json(job_dir / "simulator_provider_adapter_manifest.json")
    eval_result = _read_json(job_dir / "evaluation_result.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    matrix = _read_json(job_dir / "scenario_eval_matrix.json")
    prediction = _read_json(job_dir / "prediction_outcome_ledger.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    remote_closure = _read_json(job_dir / "remote_cloud_execution_closure_manifest.json")
    robot_team_closure = _read_json(job_dir / "robot_team_grade_eval_closure_manifest.json")
    package = _read_json(job_dir / "post_training_data_package_export_manifest.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    webapp_projection = _read_json(job_dir / "webapp_robot_eval_status_projection.json")

    assert result["status"] == "simulator_command_completed"
    assert simulator_result["status"] == "completed"
    assert simulator_result["simulator_execution_proven"] is True
    assert simulator_result["artifact_paths"]["simulator_provider_adapter_manifest"] == (
        "simulator_provider_adapter_manifest.json"
    )
    assert simulator_result["artifact_paths"]["normalized_attempt_trace"] == (
        "normalized_attempt_trace.json"
    )
    assert provider_adapter["schema_version"] == (
        "robot_eval_simulator_provider_adapter_manifest.v1"
    )
    assert provider_adapter["provider_profile"]["provider_family"] == "cpu_physics_engine"
    assert provider_adapter["gates"] == {
        "env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": True,
        "allow_simulator_execution_flag": True,
        "simulator_allowlisted": True,
        "command_configured": True,
        "blockers": [],
    }
    assert provider_adapter["command_ref"]["configured"] is True
    assert provider_adapter["command_ref"]["sha256"]
    assert provider_adapter["normalization"]["simulator_output_ingested"] is True
    assert eval_result["status"] == "completed"
    assert trace["attempt_count"] == matrix["scenario_eval_run_count"]
    assert trace["scenario_eval_run_coverage_complete"] is True
    assert trace["attempts"][0]["engine"] == "pybullet"
    assert prediction["records"][0]["predicted_success"] is True
    assert proof_boundary["simulator_execution_proven"] is True
    assert proof_boundary["robot_readiness_proven"] is False
    assert package["included_artifacts"]["simulator_command_artifacts_manifest"] == (
        "simulator_command_artifacts_manifest.json"
    )
    assert package["included_artifacts"]["simulator_provider_adapter_manifest"] == (
        "simulator_provider_adapter_manifest.json"
    )
    assert package["included_artifacts"]["webapp_robot_eval_status_projection"] == (
        "webapp_robot_eval_status_projection.json"
    )
    assert package["included_artifacts"]["remote_cloud_execution_closure_manifest"] == (
        "remote_cloud_execution_closure_manifest.json"
    )
    assert package["included_artifacts"]["robot_team_grade_eval_closure_manifest"] == (
        "robot_team_grade_eval_closure_manifest.json"
    )
    assert package["export_policy"]["simulator_provider_adapter_included"] is True
    assert run_manifest["simulator_execution_proven"] is True
    assert run_manifest["robot_readiness_proven"] is False
    assert run_manifest["webapp_robot_eval_status_projection_status"] == (
        "simulator_command_completed"
    )
    assert run_manifest["remote_cloud_execution_closure_status"] == (
        "not_required_for_local_execution"
    )
    assert run_manifest["remote_cloud_execution_proven"] is False
    assert run_manifest["remote_cloud_clean_shutdown_proven"] is False
    assert run_manifest["robot_team_grade_eval_closure_status"] == (
        "blocked_robot_team_grade_requirements"
    )
    assert run_manifest["robot_team_grade_evaluation_complete"] is False
    assert run_manifest["deployment_readiness_complete"] is False
    assert run_manifest["artifacts"]["webapp_robot_eval_status_projection"] == (
        "webapp_robot_eval_status_projection.json"
    )
    assert webapp_projection["schema_version"] == "webapp_robot_eval_status_projection.v1"
    assert webapp_projection["provider_complexity_hidden"] is True
    assert webapp_projection["provider_details_exposed"] is False
    assert webapp_projection["scenario_batch"]["scenario_eval_run_count"] == (
        matrix["scenario_eval_run_count"]
    )
    assert webapp_projection["scenario_batch"]["scenario_eval_run_coverage_complete"] is True
    assert webapp_projection["proof_boundary"]["simulator_execution_proven"] is True
    assert webapp_projection["proof_boundary"]["robot_readiness_proven"] is False
    assert webapp_projection["proof_boundary"]["public_claim_upgrade_allowed"] is False
    assert webapp_projection["artifact_paths"]["webapp_robot_eval_status_projection"] == (
        "webapp_robot_eval_status_projection.json"
    )
    assert webapp_projection["remote_cloud_execution"]["status"] == (
        "not_required_for_local_execution"
    )
    assert webapp_projection["remote_cloud_execution"]["remote_cloud_execution_proven"] is False
    assert webapp_projection["robot_team_grade_eval_closure"]["status"] == (
        "blocked_robot_team_grade_requirements"
    )
    assert (
        webapp_projection["robot_team_grade_eval_closure"][
            "robot_team_grade_evaluation_complete"
        ]
        is False
    )
    assert remote_closure["status"] == "not_required_for_local_execution"
    assert remote_closure["remote_cloud_execution_proven"] is False
    assert robot_team_closure["schema_version"] == "robot_team_grade_eval_closure.v1"
    assert robot_team_closure["robot_team_grade_evaluation_complete"] is False
    assert robot_team_closure["deployment_readiness_complete"] is False
    assert "task_success_metrics" in robot_team_closure["blocked_requirement_ids"]
    assert "digital_twin_fidelity_qa" in robot_team_closure["blocked_requirement_ids"]
    assert "full_trace_package" in robot_team_closure["blocked_requirement_ids"]
    assert "remote_cloud_execution_path" in robot_team_closure["blocked_requirement_ids"]
    full_trace_package = {
        requirement["requirement_id"]: requirement
        for requirement in robot_team_closure["requirements"]
    }["full_trace_package"]
    assert "missing_trace_artifact_metrics" in full_trace_package["blockers"]
    assert "missing_trace_artifact_visual_media_coverage" in full_trace_package[
        "blockers"
    ]
    assert "missing_trace_artifact_artifact_checksums" in full_trace_package[
        "blockers"
    ]
    assert "visual_media_coverage_manifest_missing" in full_trace_package["blockers"]
    task_metrics = {
        requirement["requirement_id"]: requirement
        for requirement in robot_team_closure["requirements"]
    }["task_success_metrics"]
    assert "missing_metric_clearance" in task_metrics["blockers"]
    assert "missing_metric_near_misses" in task_metrics["blockers"]
    assert "missing_metric_path_deviation" in task_metrics["blockers"]
    assert "missing_metric_policy_instability" in task_metrics["blockers"]
    assert "batch_metrics_artifact_missing" in task_metrics["blockers"]
    digital_twin = {
        requirement["requirement_id"]: requirement
        for requirement in robot_team_closure["requirements"]
    }["digital_twin_fidelity_qa"]
    assert "digital_twin_fidelity_qa_artifact_missing" in digital_twin["blockers"]
    assert webapp_projection["buyer_display_guardrails"]["provider_commands_exposed"] is False
    assert "provider_command" not in webapp_projection


def test_packaged_mujoco_g1_simulator_command_is_available() -> None:
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-m", "blueprint_pipeline.mujoco_g1_simulator_command", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )

    assert result.returncode == 0
    assert "--capture-root" in result.stdout
    assert "--g1-model-root" in result.stdout


def test_robot_eval_job_runs_packaged_mujoco_g1_simulator_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("mujoco")
    trimesh = pytest.importorskip("trimesh")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    scene_glb = capture_root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    scene_glb.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(0.4, 0.3, 0.2)).export(scene_glb)
    g1_root = tmp_path / "unitree_g1"
    g1_root.mkdir(parents=True)
    (g1_root / "g1.xml").write_text(
        "\n".join(
            [
                '<mujoco model="unitree_g1_test">',
                '  <option timestep="0.002"/>',
                '  <worldbody>',
                '    <body name="torso" pos="0 0 0.8">',
                '      <freejoint name="floating_base_joint"/>',
                '      <geom name="torso_geom" type="box" size="0.08 0.05 0.18"/>',
                "    </body>",
                "  </worldbody>",
                '  <keyframe><key name="stand" qpos="0 0 0.8 1 0 0 0"/></keyframe>',
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["simulator_preference"] = "mujoco"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-mujoco-g1-command",
        provisioner="fixture_local",
        simulator="mujoco",
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={
            "mujoco": (
                f"{sys.executable} -m blueprint_pipeline.mujoco_g1_simulator_command "
                f"--capture-root {capture_root} "
                f"--g1-model-root {g1_root} "
                "--steps 3 --skip-render-frames --no-fetch-g1-assets"
            )
        },
    )

    job_dir = Path(result["job_dir"])
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    matrix = _read_json(job_dir / "scenario_eval_matrix.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    eval_result = _read_json(job_dir / "evaluation_result.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")
    robot_team_closure = _read_json(job_dir / "robot_team_grade_eval_closure_manifest.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert result["status"] == "simulator_command_completed"
    assert simulator_result["status"] == "completed"
    assert simulator_result["framework"] == "mujoco"
    assert simulator_result["unitree_g1_asset_spawned"] is True
    assert simulator_result["simulator_execution_proven"] is True
    expected_run_ids = [
        row["scenario_eval_run_id"]
        for row in matrix["runs"]
        if row.get("scenario_eval_run_id")
    ]
    assert simulator_result["attempt_count"] == matrix["scenario_eval_run_count"]
    assert simulator_result["covered_scenario_eval_run_ids"] == sorted(expected_run_ids)
    assert simulator_result["missing_scenario_eval_run_ids"] == []
    assert trace["attempt_count"] == matrix["scenario_eval_run_count"]
    assert [
        attempt["scenario_eval_run_id"] for attempt in trace["attempts"]
    ] == expected_run_ids
    assert trace["attempts"][0]["policy_id"] == "blueprint_default_walk_to_target_smoke_policy"
    assert trace["attempts"][0]["artifact_paths"]["scene_trace"].endswith(
        "scene_load_trace.json"
    )
    assert eval_result["status"] == "completed"
    assert proof_boundary["simulator_execution_proven"] is True
    assert proof_boundary["robot_readiness_proven"] is False
    assert run_manifest["simulator_execution_proven"] is True
    assert run_manifest["sim_only_beta_core_complete"] is False
    assert robot_team_closure["sim_only_beta_core_complete"] is False
    assert "full_trace_package" in robot_team_closure["blocked_requirement_ids"]
    assert robot_team_closure["robot_team_grade_evaluation_complete"] is False
    assert run_manifest["robot_readiness_proven"] is False


def test_robot_eval_job_ingests_real_robot_pov_and_action_logs_separate_from_generated_support(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    _write_real_robot_pov_manifest(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-real-pov-evidence",
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = Path(result["job_dir"])
    robot_pov = _read_json(job_dir / "robot_pov_observation_manifest.json")
    frame_sequences = _read_json(job_dir / "robot_pov_frame_sequence_manifest.json")
    storyboard = _read_json(job_dir / "robot_pov_render_storyboard.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert robot_pov["robot_pov_generated"] is True
    assert robot_pov["generated_robot_pov_support_available"] is True
    assert robot_pov["real_robot_pov_evidence_record_count"] == len(
        POLICY_REFERENCE_VARIATION_NAMES
    )
    assert robot_pov["real_robot_pov_action_log_record_count"] == len(
        POLICY_REFERENCE_VARIATION_NAMES
    )
    assert robot_pov["missing_real_robot_pov_scenario_eval_run_ids"] == []
    assert robot_pov["robot_pov_evidence_proven"] is True
    assert robot_pov["real_robot_pov_manifest_path"] == (
        "../robot_eval_inputs/real_robot_pov_manifest.json"
    )
    assert all(
        observation["real_robot_pov_evidence"]["action_log_uri"]
        for observation in robot_pov["observations"]
    )
    assert all(
        observation["real_robot_pov_evidence"]["robot_camera_video_uri"]
        for observation in robot_pov["observations"]
    )
    assert frame_sequences["local_robot_pov_render_generated"] is True
    assert frame_sequences["robot_pov_evidence_proven"] is False
    assert storyboard["local_robot_pov_render_generated"] is True
    assert storyboard["robot_pov_evidence_proven"] is False
    assert run_manifest["robot_pov_evidence_proven"] is True
    assert run_manifest["robot_readiness_proven"] is False


def test_robot_eval_job_request_inbox_runs_webapp_job_request_automatically(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    inbox_dir = tmp_path / "webapp-robot-eval-job-requests"
    request = _full_job_request(capture_root)
    request["job_id"] = "webapp-job-1"
    request["source"] = {
        "system": "Blueprint-WebApp",
        "route": "/sites/sw-chi-01",
        "selection_state": {
            "site_slug": "sw-chi-01",
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
            "policy_id": "policy-fixture-a",
        },
    }
    _write_json(inbox_dir / "webapp-job-1.json", request)

    result = run_robot_eval_job_request_inbox(
        capture_root=capture_root,
        inbox_dir=inbox_dir,
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    queue_root = capture_root / "pipeline" / "robot_eval_job_requests"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "webapp-job-1"
    queue_manifest = _read_json(queue_root / "inbox_run_manifest.json")
    queued_request = _read_json(queue_root / "webapp-job-1" / "job_request.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert result["schema_version"] == "robot_eval_job_request_inbox_run.v1"
    assert result["status"] == "completed"
    assert result["processed_count"] == 1
    assert result["jobs"][0]["job_id"] == "webapp-job-1"
    assert result["jobs"][0]["status"] == "fixture_evaluation_completed"
    assert queued_request["schema_version"] == "robot_eval_job_request.v1"
    assert queued_request["source"]["system"] == "Blueprint-WebApp"
    assert queue_manifest["processed_count"] == 1
    assert queue_manifest["jobs"][0]["job_run_manifest_uri"].endswith(
        "/pipeline/robot_eval_jobs/webapp-job-1/job_run_manifest.json"
    )
    assert run_manifest["status"] == "fixture_evaluation_completed"
    assert run_manifest["public_claim_upgrade_allowed"] is False


def test_robot_eval_job_request_inbox_accepts_webapp_queue_envelope(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    inbox_dir = tmp_path / "webapp-robot-eval-job-requests"
    request = _full_job_request(capture_root)
    request["job_id"] = "webapp-envelope-job-1"
    request["buyer_request_id"] = "buyer-request-envelope-1"
    envelope = {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "queued_at_iso": "2026-06-07T00:00:00Z",
        "job_id": request["job_id"],
        "buyer_request_id": request["buyer_request_id"],
        "pipeline_command": "blueprint-run-robot-eval-job",
        "pipeline_consumer": "BlueprintCapturePipeline",
        "job_request": request,
    }
    _write_json(inbox_dir / "webapp-envelope-job-1.json", envelope)
    (inbox_dir / "._webapp-envelope-job-1.json").write_text("not json", encoding="utf-8")

    result = run_robot_eval_job_request_inbox(
        capture_root=capture_root,
        inbox_dir=inbox_dir,
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    queued_request = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_job_requests"
        / "webapp-envelope-job-1"
        / "job_request.json"
    )

    assert result["status"] == "completed"
    assert result["processed_count"] == 1
    assert result["jobs"][0]["job_id"] == "webapp-envelope-job-1"
    assert queued_request["schema_version"] == "robot_eval_job_request.v1"
    assert "queue_contract" not in queued_request


def test_robot_eval_job_request_inbox_processes_latest_webapp_identity_once(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    inbox_dir = tmp_path / "webapp-robot-eval-job-requests"
    identity = {
        "site_slug": "first-gpu-walkthrough-2",
        "site_submission_id": "site-submission-webapp-route-20260612",
        "capture_job_id": "capture-job-webapp-route-20260612",
        "capture_id": "downloads-walkthrough2-20260611",
        "source_kind": "webapp_route_forwarding_proof",
    }
    old_request = _full_job_request(capture_root)
    old_request["job_id"] = "old-walk-to-target-request"
    old_request["buyer_request_id"] = "buyer-request-webapp-route-20260611"
    old_request["source_kind"] = "webapp_route_forwarding_proof"
    old_request["requested_tasks"] = [{"task_id": "walk_to_target"}]
    old_request["source"] = {
        "system": "Blueprint-WebApp",
        "selection_state": {
            **{
                **identity,
                "site_submission_id": "site-submission-webapp-route-20260611",
                "capture_job_id": "capture-job-webapp-route-20260611",
            },
            "task_id": "walk_to_target",
            "scenario_id": "walk_to_target_pose",
        },
    }
    new_request = _full_job_request(capture_root)
    new_request["job_id"] = "new-dataset-aligned-request"
    new_request["buyer_request_id"] = "buyer-request-webapp-route-20260612"
    new_request["source_kind"] = "webapp_route_forwarding_proof"
    new_request["source"] = {
        "system": "Blueprint-WebApp",
        "selection_state": {
            **identity,
            "task_id": "place_return_in_bin",
            "scenario_id": "scenario_place_return_in_bin_mobile",
        },
    }
    old_path = inbox_dir / "old-walk-to-target-request.json"
    new_path = inbox_dir / "new-dataset-aligned-request.json"
    _write_json(old_path, old_request)
    _write_json(new_path, new_request)
    os.utime(old_path, (1, 1))
    os.utime(new_path, (2, 2))

    result = run_robot_eval_job_request_inbox(
        capture_root=capture_root,
        inbox_dir=inbox_dir,
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    queue_root = capture_root / "pipeline" / "robot_eval_job_requests"
    queue_manifest = _read_json(queue_root / "inbox_run_manifest.json")

    assert result["status"] == "completed"
    assert result["input_request_count"] == 2
    assert result["processed_count"] == 1
    assert result["superseded_request_count"] == 1
    assert result["jobs"][0]["job_id"] == "new-dataset-aligned-request"
    assert result["jobs"][0]["status"] == "fixture_evaluation_completed"
    assert result["superseded_requests"][0]["job_id"] == "old-walk-to-target-request"
    assert queue_manifest["superseded_request_count"] == 1
    assert not (capture_root / "pipeline" / "robot_eval_jobs" / "old-walk-to-target-request").exists()


def test_robot_eval_job_policy_manifest_includes_adapter_smoke_contracts(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-policy-smoke-contracts",
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )

    policy_manifest = _read_json(Path(result["job_dir"]) / "policy_package_manifest.json")
    observation_manifest = _read_json(
        Path(result["job_dir"]) / "robot_pov_observation_manifest.json"
    )
    assert policy_manifest["interface_contract"]["observation_schema_id"] == (
        "blueprint.robot_eval.observation.v1"
    )
    assert policy_manifest["interface_contract"]["action_schema_id"] == (
        "blueprint.robot_eval.action_trace.v1"
    )
    assert policy_manifest["interface_contract"]["reproducible_replay_required"] is True
    assert observation_manifest["policy_adapter_input_contract"]["observation_schema_id"] == (
        "blueprint.robot_eval.observation.v1"
    )
    assert observation_manifest["policy_adapter_input_contract"]["action_schema_id"] == (
        "blueprint.robot_eval.action_trace.v1"
    )
    assert observation_manifest["observations"][0]["observation_schema"]["schema_id"] == (
        "blueprint.robot_eval.observation.v1"
    )
    assert observation_manifest["observations"][0]["expected_action_schema"]["schema_id"] == (
        "blueprint.robot_eval.action_trace.v1"
    )
    for modality in (
        "policy_api_endpoint",
        "docker_container",
        "recorded_action_trace",
        "high_level_skill_trace",
        "teleop_demo",
        "sim_controller_plugin",
    ):
        contract = policy_manifest["modalities"][modality]["adapter_smoke_contract"]
        assert contract["schema_version"] == "policy_adapter_smoke_contract.v1"
        assert contract["modality"] == modality
        assert contract["observation_manifest_input"] == "robot_pov_observation_manifest.json"
        assert contract["observation_schema_id"] == "blueprint.robot_eval.observation.v1"
        assert contract["action_schema_id"] == "blueprint.robot_eval.action_trace.v1"
        assert "scenario_eval_run_id" in contract["required_attempt_fields"]
        assert "scenario_variation_instance_id" in contract["required_attempt_fields"]
        assert "scenario_eval_run_id" in contract["required_observation_fields"]
        assert "actions" in contract["required_action_output_fields"]
        assert contract["reproducible_replay_contract"][
            "runtime_spawn_goal_variation_mutation_allowed"
        ] is False
        interface_contract = policy_manifest["modalities"][modality]["interface_contract"]
        assert interface_contract["schema_version"] == "robot_team_policy_interface_contract.v1"
        assert interface_contract["observation_schema"]["schema_id"] == (
            "blueprint.robot_eval.observation.v1"
        )
        assert interface_contract["action_schema"]["schema_id"] == (
            "blueprint.robot_eval.action_trace.v1"
        )
        assert interface_contract["reproducible_replay"][
            "exact_scenario_eval_run_id_coverage_required"
        ] is True
        assert contract["proof_boundary"]["robot_readiness_proven"] is False
    assert (
        policy_manifest["modalities"]["policy_api_endpoint"]["adapter_smoke_contract"][
            "smoke_runner"
        ]
        == "http_policy_api_observation_probe"
    )
    assert (
        policy_manifest["modalities"]["docker_container"]["adapter_smoke_contract"]["smoke_runner"]
        == "docker_run_observation_manifest_probe"
    )
    docker_contract = policy_manifest["modalities"]["docker_container"]["interface_contract"]
    assert docker_contract["container_runtime"]["digest_required"] is True
    assert docker_contract["container_runtime"]["runtime_image_pinned_by_digest"] is True
    assert docker_contract["container_runtime"]["versioned_runtime_image_proven"] is True


def test_robot_eval_job_rights_privacy_block_prevents_execution(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, rights_blocked=True)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root, rights_allowed=False))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-rights-blocked",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-rights-blocked"
    blocked = _read_json(job_dir / "blocked_manifest.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert blocked["status"] == "blocked"
    assert "blocked_rights_privacy" in blocked["blockers"]
    assert blocked["missing_inputs"] == ["rights_privacy_clearance"]
    assert provisioning["status"] == "blocked"
    assert provisioning["execution_performed"] is False
    assert simulator_result["status"] == "blocked"
    assert simulator_result["execution_performed"] is False
    assert run_manifest["state"] == "blocked"
    assert run_manifest["public_claim_upgrade_allowed"] is False


def test_robot_eval_job_missing_policy_evidence_writes_exact_blockers(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = _full_job_request(capture_root)
    policy = dict(request["policy_package"])  # type: ignore[index]
    policy["teleop_demo"] = {
        "demo_artifact_uri": "gs://robot-team/demos/demo-1.json",
    }
    policy["docker_container"] = {"image_ref": "registry.example/robot/policy:latest"}
    request["policy_package"] = policy
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-policy-blocked",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-blocked"
    validation = _read_json(job_dir / "job_validation.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")
    policy_manifest = _read_json(job_dir / "policy_package_manifest.json")
    robot_team_closure = _read_json(job_dir / "robot_team_grade_eval_closure_manifest.json")

    assert validation["status"] == "blocked"
    assert validation["missing_evidence_statuses"] == [
        "needs_docker_container_ref",
        "needs_teleop_demo_ref",
    ]
    assert blocked["missing_inputs"] == [
        "policy_package.docker_container.digest",
        "policy_package.teleop_demo.rights_privacy_attestation",
    ]
    assert policy_manifest["modalities"]["docker_container"]["status"] == "blocked"
    assert policy_manifest["modalities"]["teleop_demo"]["status"] == "blocked"
    requirements = {item["requirement_id"]: item for item in robot_team_closure["requirements"]}
    policy_interface = requirements["robot_team_policy_interface"]
    assert policy_interface["passed"] is False
    assert "robot_team_policy_interface" in robot_team_closure["blocked_requirement_ids"]
    assert "policy_package_validation_blocked" in policy_interface["blockers"]
    assert "policy_interface_selected_modalities_invalid" in policy_interface["blockers"]
    assert (
        "policy_docker_container_runtime_image_not_versioned"
        in policy_interface["blockers"]
    )
    assert robot_team_closure["policy_interface_summary"]["invalid_selected_modalities"] == [
        "docker_container",
        "teleop_demo",
    ]


def test_robot_eval_job_accepts_one_complete_policy_modality(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = _full_job_request(capture_root)
    request["policy_package"] = {
        "policy_api_endpoint": {
            "endpoint_url": "https://robot-team.example/policy",
            "observation_schema_ref": "schemas/obs-v1.json",
            "action_schema_ref": "schemas/action-v1.json",
        }
    }
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-single-modality-policy",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-single-modality-policy"
    validation = _read_json(job_dir / "job_validation.json")
    policy_manifest = _read_json(job_dir / "policy_package_manifest.json")
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")

    assert "needs_robot_team_test_modality" not in validation["missing_evidence_statuses"]
    assert "needs_docker_container_ref" not in validation["missing_evidence_statuses"]
    assert policy_manifest["status"] == "review_required"
    assert policy_manifest["selected_modalities"] == ["policy_api_endpoint"]
    assert policy_manifest["modalities"]["docker_container"]["status"] == "not_selected"
    assert (
        policy_manifest["modalities"]["docker_container"]["owner_system_review_required"] is False
    )
    assert policy_execution["selected_modalities"] == ["policy_api_endpoint"]


def test_robot_eval_job_consumes_staged_policy_package(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    job_id = "job-staged-policy-package"
    staged_path = capture_root / "pipeline" / "robot_eval_inputs" / job_id / "policy_package.json"
    _write_json(
        staged_path,
        {
            "schema_version": "robot_team_policy_package.v1",
            "job_id": job_id,
            "policy_package": {
                "high_level_skill_trace": {
                    "skill_taxonomy_version": "skills-v1",
                    "ordered_skill_sequence": ["navigate", "pick", "place"],
                }
            },
        },
    )
    request = _full_job_request(capture_root)
    request.pop("policy_package")
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id=job_id,
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / job_id
    job_request = _read_json(job_dir / "job_request.json")
    validation = _read_json(job_dir / "job_validation.json")
    scenario_eval_matrix = _read_json(job_dir / "scenario_eval_matrix.json")
    policy_manifest = _read_json(job_dir / "policy_package_manifest.json")
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")

    assert job_request["external_input_sources"]["staged_policy_package"] == str(staged_path)
    assert validation["status"] != "blocked"
    assert policy_manifest["selected_modalities"] == ["high_level_skill_trace"]
    assert policy_execution["selected_modalities"] == ["high_level_skill_trace"]
    assert policy_execution["modality_results"]["high_level_skill_trace"]["status"] == (
        "completed_reference_replay"
    )
    assert policy_execution["attempt_count"] == scenario_eval_matrix["scenario_eval_run_count"]


def test_robot_eval_job_replays_all_reference_policy_modalities_with_matrix_coverage(
    tmp_path: Path,
) -> None:
    cases: dict[str, dict[str, object]] = {
        "policy_api_endpoint": {
            "endpoint_url": "https://robot-team.example/policy",
            "response_manifest_uri": "policy_refs/policy_api_endpoint.json",
        },
        "docker_container": {
            "image_ref": "registry.example/robot/policy:2026-06-04",
            "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "output_manifest_uri": "policy_refs/docker_container.json",
        },
        "recorded_action_trace": {
            "trace_manifest_uri": "policy_refs/recorded_action_trace.json",
            "timestamp_alignment": "aligned_to_capture_timestamps",
        },
        "high_level_skill_trace": {
            "skill_taxonomy_version": "skills-v1",
            "ordered_skill_sequence": ["navigate", "pick", "place"],
        },
        "teleop_demo": {
            "demo_artifact_uri": "policy_refs/teleop_demo.json",
            "rights_privacy_attestation": "deidentified_operator_approved",
        },
        "sim_controller_plugin": {
            "simulator_framework": "fixture",
            "plugin_uri": "policy_refs/sim_controller_plugin.json",
        },
    }
    for modality, payload in cases.items():
        capture_root = _build_capture_root(tmp_path / modality)
        _write_robot_eval_cards(capture_root)
        if modality != "high_level_skill_trace":
            reference_path = capture_root / "policy_refs" / f"{modality}.json"
            _write_json(
                reference_path,
                {
                    "schema_version": f"{modality}_policy_reference.v1",
                    "attempts": _policy_reference_attempts(policy_id=f"{modality}-policy"),
                },
            )
        request = _full_job_request(capture_root)
        request["policy_package"] = {modality: payload}
        request_path = tmp_path / modality / "job-request.json"
        _write_json(request_path, request)

        build_robot_eval_job(
            capture_root=capture_root,
            job_request=request_path,
            job_id=f"job-{modality}",
            provisioner="fixture_local",
            simulator="fixture",
        )

        job_dir = capture_root / "pipeline" / "robot_eval_jobs" / f"job-{modality}"
        scenario_eval_matrix = _read_json(job_dir / "scenario_eval_matrix.json")
        policy_execution = _read_json(job_dir / "policy_execution_manifest.json")
        policy_trace = _read_json(job_dir / "policy_execution_trace.json")
        live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
        modality_result = policy_execution["modality_results"][modality]  # type: ignore[index]

        assert policy_execution["selected_modalities"] == [modality]
        assert policy_execution["attempt_count"] == scenario_eval_matrix["scenario_eval_run_count"]
        assert policy_execution["scenario_eval_run_coverage_complete"] is True
        assert policy_execution["missing_scenario_eval_run_ids"] == []
        assert policy_trace["scenario_eval_run_coverage_complete"] is True
        assert modality_result["status"] == "completed_reference_replay"
        assert modality_result["reference_replayed"] is True
        assert modality_result["attempt_count"] == scenario_eval_matrix["scenario_eval_run_count"]
        assert modality_result["scenario_eval_run_coverage_complete"] is True
        assert modality_result["missing_scenario_eval_run_ids"] == []
        assert modality_result["robot_policy_execution_proven"] is False
        policy_gate = live_closure["gates"]["live_policy_execution"]
        assert (
            "policy_execution_missing_scenario_variation_run_coverage"
            not in policy_gate["blockers"]
        )
        assert (
            "policy_execution_missing_required_scenario_eval_run_ids" not in policy_gate["blockers"]
        )
        assert "live_policy_execution_not_proven" in policy_gate["blockers"]


def test_robot_eval_job_policy_reference_replay_reports_missing_matrix_coverage(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    reference_path = capture_root / "policy_refs" / "partial-recorded-action-trace.json"
    _write_json(
        reference_path,
        {
            "schema_version": "recorded_action_trace_policy_reference.v1",
            "attempts": _policy_reference_attempts(
                policy_id="partial-recorded-policy",
                variation_names=["lighting_variation"],
            ),
        },
    )
    request = _full_job_request(capture_root)
    request["policy_package"] = {
        "recorded_action_trace": {
            "trace_manifest_uri": "policy_refs/partial-recorded-action-trace.json",
            "timestamp_alignment": "aligned_to_capture_timestamps",
        }
    }
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, request)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-partial-reference-policy",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-partial-reference-policy"
    scenario_eval_matrix = _read_json(job_dir / "scenario_eval_matrix.json")
    policy_execution = _read_json(job_dir / "policy_execution_manifest.json")
    policy_trace = _read_json(job_dir / "policy_execution_trace.json")
    live_closure = _read_json(job_dir / "live_eval_closure_manifest.json")
    modality_result = policy_execution["modality_results"]["recorded_action_trace"]  # type: ignore[index]

    expected_missing = int(scenario_eval_matrix["scenario_eval_run_count"]) - 1
    assert policy_execution["attempt_count"] == 1
    assert policy_execution["scenario_eval_run_coverage_complete"] is False
    assert policy_execution["missing_scenario_eval_run_count"] == expected_missing
    assert policy_trace["missing_scenario_eval_run_count"] == expected_missing
    assert modality_result["missing_scenario_eval_run_count"] == expected_missing
    assert modality_result["scenario_eval_run_coverage_complete"] is False
    policy_gate = live_closure["gates"]["live_policy_execution"]
    assert policy_gate["evidence"]["missing_scenario_eval_run_count"] == expected_missing
    assert "policy_execution_missing_scenario_variation_run_coverage" in policy_gate["blockers"]
    assert "policy_execution_missing_required_scenario_eval_run_ids" in policy_gate["blockers"]


def test_robot_eval_job_real_provisioner_fails_closed_without_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "r2://blueprint-artifacts/jobs/job-real-provisioner-blocked",
    )
    monkeypatch.setenv(
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "r2://blueprint-artifacts/jobs/job-real-provisioner-blocked/worker_manifest.json",
    )

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-real-provisioner-blocked",
        provisioner="vast",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-real-provisioner-blocked"
    result = _read_json(job_dir / "gpu_provisioning_result.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")

    assert result["status"] == "blocked"
    assert result["provider"] == "vast"
    assert result["blockers"] == [
        "missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING",
        "missing_cli_allow_gpu_provisioning",
    ]
    assert "gpu_provisioning_blocked" in blocked["blockers"]


def test_remote_cloud_execution_closure_requires_provider_shutdown_evidence() -> None:
    worker_launch_plan = {
        "worker_image": {
            "configured_image_ref_present": True,
            "configured_image_ref_is_versioned": True,
            "configured_image_ref_fetchable_by_provider": True,
        },
        "input_bundle": {
            "capture_root_bundle_uri": "r2://blueprint-inputs/capture.zip",
            "capture_root_bundle_uri_scheme": "r2",
            "capture_root_bundle_uri_fetchable_by_provider": True,
        },
        "worker_manifest_input_contract": {
            "configured_worker_manifest_uri": "r2://blueprint-inputs/worker.json",
            "configured_worker_manifest_uri_present": True,
            "worker_manifest_uri_fetchable_by_provider": True,
        },
        "artifact_upload_contract": {
            "configured_artifact_output_uri": "r2://blueprint-artifacts/jobs/job-remote",
            "configured_artifact_output_uri_present": True,
            "artifact_output_uri_scheme": "r2",
            "artifact_output_uri_provider_writable": True,
            "artifact_output_write_auth_contract_ready": True,
            "artifact_output_write_auth": {
                "authorization_mode": "worker_storage_credentials",
                "write_auth_required_for_provider": True,
                "write_auth_contract_ready": True,
                "required_secret_env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
                "required_plaintext_env_vars": [
                    "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL",
                    "AWS_ENDPOINT_URL",
                ],
                "secret_values_in_artifact": False,
            },
            "upload_before_shutdown_required": True,
        },
        "launch_mode": {"hard_timeout_seconds": 120},
    }
    worker_manifest = {"status": "ready_for_worker_upload"}
    provider_launch_request = {
        "status": "request_manifest_ready",
        "provider_request_shape": {
            "inputs": {
                "manifest_uri": "r2://blueprint-inputs/worker.json",
                "capture_root_bundle_uri": "r2://blueprint-inputs/capture.zip",
            },
            "limits": {
                "hard_timeout_seconds": 120,
                "idle_shutdown_required": True,
            },
        },
    }
    base_ledger = {
        "status": "provider_runtime_observed",
        "live_provider_calls_performed": True,
        "budget": {"requested_budget_usd": 10},
        "worker_limits": {
            "hard_timeout_seconds": 120,
            "idle_shutdown_required": True,
        },
        "gpu_time": {"actual_gpu_time_record_present": True},
        "artifact_finalizer": {
            "upload_before_shutdown_required": True,
            "worker_artifacts_finalized_before_shutdown": True,
        },
    }
    missing_shutdown = _remote_cloud_execution_closure_manifest(
        job_id="job-remote",
        provisioner="runpod",
        simulator="mujoco",
        worker_launch_plan=worker_launch_plan,
        worker_manifest=worker_manifest,
        provider_launch_request=provider_launch_request,
        gpu_result={"live_provider_calls_performed": True},
        gpu_cost_ledger=base_ledger,
        sim_result={"simulator_execution_proven": True},
        generated_at="2026-06-15T00:00:00Z",
    )
    assert missing_shutdown["status"] == "remote_execution_completed_missing_shutdown_proof"
    assert missing_shutdown["remote_cloud_execution_proven"] is True
    assert missing_shutdown["clean_shutdown_proven"] is False
    assert "remote_artifact_upload_evidence_incomplete" in missing_shutdown["runtime_blockers"]
    assert "remote_provider_shutdown_not_proven" in missing_shutdown["runtime_blockers"]

    proven_ledger = {
        **base_ledger,
        "artifact_finalizer": {
            **base_ledger["artifact_finalizer"],  # type: ignore[index]
            "artifact_upload_evidence": {
                "status": "completed",
                "destination_uri": "r2://blueprint-artifacts/jobs/job-remote",
                "uploaded_file_count": 12,
                "object_keys": ["jobs/job-remote/job_run_manifest.json"],
            },
            "finalizer_refresh_upload_evidence": {
                "status": "completed",
                "destination_uri": "r2://blueprint-artifacts/jobs/job-remote",
                "uploaded_file_count": 4,
                "object_keys": ["jobs/job-remote/gpu_cost_control_ledger.json"],
            },
            "runtime_manifest_upload_evidence": {
                "status": "completed",
                "destination_uri": "r2://blueprint-artifacts/jobs/job-remote",
                "object_key": "jobs/job-remote/worker_runtime_manifest.json",
            },
            "provider_shutdown_proven": True,
            "provider_shutdown_evidence": {
                "provider_shutdown_proven": True,
                "zero_active_workers_after_run": True,
            },
        },
    }
    proven = _remote_cloud_execution_closure_manifest(
        job_id="job-remote",
        provisioner="runpod",
        simulator="mujoco",
        worker_launch_plan=worker_launch_plan,
        worker_manifest=worker_manifest,
        provider_launch_request=provider_launch_request,
        gpu_result={"live_provider_calls_performed": True},
        gpu_cost_ledger=proven_ledger,
        sim_result={"simulator_execution_proven": True},
        generated_at="2026-06-15T00:00:00Z",
    )
    assert proven["status"] == "remote_execution_completed_with_shutdown_proof"
    assert proven["remote_cloud_execution_proven"] is True
    assert proven["clean_shutdown_proven"] is True
    assert proven["runtime_blockers"] == []
    assert proven["checks"]["artifact_output_uri_provider_writable"] is True
    assert proven["checks"]["artifact_output_write_auth_contract_ready"] is True
    assert proven["checks"]["artifact_upload_evidence_complete"] is True
    assert proven["outputs"]["artifact_output_uri_scheme"] == "r2"
    assert proven["outputs"]["artifact_output_uri_provider_writable"] is True
    assert proven["outputs"]["artifact_output_write_auth"]["authorization_mode"] == (
        "worker_storage_credentials"
    )
    assert proven["outputs"]["artifact_upload_evidence"]["uploaded_file_count"] == 12


def test_remote_cloud_execution_closure_blocks_provider_input_setup_upload_not_proven() -> None:
    worker_launch_plan = {
        "worker_image": {
            "configured_image_ref_present": True,
            "configured_image_ref_is_versioned": True,
            "configured_image_ref_fetchable_by_provider": True,
        },
        "input_bundle": {
            "capture_root_bundle_uri": "gs://blueprint-inputs/capture.zip",
            "capture_root_bundle_uri_scheme": "gs",
            "capture_root_bundle_uri_fetchable_by_provider": True,
        },
        "worker_manifest_input_contract": {
            "configured_worker_manifest_uri": "gs://blueprint-inputs/worker.json",
            "configured_worker_manifest_uri_present": True,
            "worker_manifest_uri_fetchable_by_provider": True,
        },
        "artifact_upload_contract": {
            "configured_artifact_output_uri": "gs://blueprint-artifacts/jobs/job-remote",
            "configured_artifact_output_uri_present": True,
            "artifact_output_uri_scheme": "gs",
            "artifact_output_uri_provider_writable": True,
            "artifact_output_write_auth_contract_ready": True,
            "upload_before_shutdown_required": True,
        },
        "launch_mode": {"hard_timeout_seconds": 120},
    }
    worker_manifest = {"status": "ready_for_worker_upload"}
    provider_launch_request = {
        "status": "request_manifest_ready",
        "provider_input_setup": {
            "status": "prepared_with_external_blockers",
            "manifest_path": "provider_input_setup_manifest.json",
            "provider_inputs_uploaded": False,
            "blockers": ["upload_failed:gs_billing_account_disabled"],
            "artifact_output_uri": "gs://blueprint-artifacts/jobs/job-remote",
        },
        "provider_request_shape": {
            "inputs": {
                "manifest_uri": "gs://blueprint-inputs/worker.json",
                "capture_root_bundle_uri": "gs://blueprint-inputs/capture.zip",
            },
            "limits": {
                "hard_timeout_seconds": 120,
                "idle_shutdown_required": True,
            },
        },
    }
    gpu_cost_ledger = {
        "status": "blocked_before_allocation",
        "budget": {"requested_budget_usd": 10},
        "worker_limits": {
            "hard_timeout_seconds": 120,
            "idle_shutdown_required": True,
        },
        "gpu_time": {"actual_gpu_time_record_present": False},
        "artifact_finalizer": {"upload_before_shutdown_required": True},
    }

    closure = _remote_cloud_execution_closure_manifest(
        job_id="job-remote",
        provisioner="runpod",
        simulator="mujoco",
        worker_launch_plan=worker_launch_plan,
        worker_manifest=worker_manifest,
        provider_launch_request=provider_launch_request,
        gpu_result={},
        gpu_cost_ledger=gpu_cost_ledger,
        sim_result={"simulator_execution_proven": False},
        generated_at="2026-06-15T00:00:00Z",
    )

    assert closure["status"] == "blocked_before_remote_execution"
    assert closure["contract_ready_for_remote_runtime"] is False
    assert "provider_input_setup:upload_failed:gs_billing_account_disabled" in closure[
        "contract_blockers"
    ]
    assert "provider_input_setup:provider_inputs_upload_not_proven" in closure[
        "contract_blockers"
    ]
    assert closure["provider_input_setup"]["provider_inputs_uploaded"] is False


def test_remote_cloud_execution_closure_blocks_non_writable_artifact_output_uri() -> None:
    worker_launch_plan = {
        "worker_image": {
            "configured_image_ref_present": True,
            "configured_image_ref_is_versioned": True,
            "configured_image_ref_fetchable_by_provider": True,
        },
        "input_bundle": {
            "capture_root_bundle_uri": "r2://blueprint-inputs/capture.zip",
            "capture_root_bundle_uri_scheme": "r2",
            "capture_root_bundle_uri_fetchable_by_provider": True,
        },
        "worker_manifest_input_contract": {
            "configured_worker_manifest_uri": "r2://blueprint-inputs/worker.json",
            "configured_worker_manifest_uri_present": True,
            "worker_manifest_uri_fetchable_by_provider": True,
        },
        "artifact_upload_contract": {
            "configured_artifact_output_uri": "file:///tmp/job-remote",
            "configured_artifact_output_uri_present": True,
            "artifact_output_uri_scheme": "file",
            "artifact_output_uri_provider_writable": False,
            "upload_before_shutdown_required": True,
        },
        "launch_mode": {"hard_timeout_seconds": 120},
    }
    worker_manifest = {"status": "ready_for_worker_upload"}
    provider_launch_request = {
        "status": "request_manifest_ready",
        "provider_request_shape": {
            "inputs": {
                "manifest_uri": "r2://blueprint-inputs/worker.json",
                "capture_root_bundle_uri": "r2://blueprint-inputs/capture.zip",
            },
            "limits": {
                "hard_timeout_seconds": 120,
                "idle_shutdown_required": True,
            },
        },
    }
    gpu_cost_ledger = {
        "status": "provider_runtime_observed",
        "live_provider_calls_performed": True,
        "budget": {"requested_budget_usd": 10},
        "worker_limits": {
            "hard_timeout_seconds": 120,
            "idle_shutdown_required": True,
        },
        "gpu_time": {"actual_gpu_time_record_present": True},
        "artifact_finalizer": {
            "upload_before_shutdown_required": True,
            "worker_artifacts_finalized_before_shutdown": True,
            "provider_shutdown_proven": True,
        },
    }

    closure = _remote_cloud_execution_closure_manifest(
        job_id="job-remote",
        provisioner="runpod",
        simulator="mujoco",
        worker_launch_plan=worker_launch_plan,
        worker_manifest=worker_manifest,
        provider_launch_request=provider_launch_request,
        gpu_result={"live_provider_calls_performed": True},
        gpu_cost_ledger=gpu_cost_ledger,
        sim_result={"simulator_execution_proven": True},
        generated_at="2026-06-15T00:00:00Z",
    )

    assert closure["status"] == "blocked_before_remote_execution"
    assert closure["contract_ready_for_remote_runtime"] is False
    assert "remote_artifact_output_uri_not_provider_writable" in closure["contract_blockers"]
    assert closure["checks"]["artifact_output_uri_provider_writable"] is False
    assert closure["outputs"]["artifact_output_uri_scheme"] == "file"
    assert closure["outputs"]["artifact_output_uri_provider_writable"] is False


def test_robot_team_grade_closure_accepts_explicitly_blocked_scenario_runs(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "robot_eval_jobs" / "job-explicit-block"
    scenario_eval_matrix = {
        "status": "completed",
        "scenario_eval_run_count": 2,
        "runs": [
            {"scenario_eval_run_id": "run-covered", "task_id": "task-1", "scenario_id": "s-1"},
            {"scenario_eval_run_id": "run-blocked", "task_id": "task-1", "scenario_id": "s-2"},
        ],
    }
    _write_json(job_dir / "scenario_eval_matrix.json", scenario_eval_matrix)
    _write_json(
        job_dir / "simulator_command_batch_closure_manifest.json",
        {
            "required_scenario_eval_run_count": 2,
            "covered_scenario_eval_run_count": 1,
            "missing_scenario_eval_run_count": 1,
            "scenario_eval_run_coverage_complete": False,
            "required_scenario_eval_run_ids": ["run-covered", "run-blocked"],
            "covered_scenario_eval_run_ids": ["run-covered"],
            "missing_scenario_eval_run_ids": ["run-blocked"],
            "explicitly_blocked_scenario_eval_runs": [
                {
                    "scenario_eval_run_id": "run-blocked",
                    "stage": "provider_runtime_preflight",
                    "reason": "worker image digest missing before launch",
                    "blockers": ["worker_image_digest_missing"],
                }
            ],
        },
    )

    closure = _robot_team_grade_eval_closure_manifest(
        job_dir=job_dir,
        job_id="job-explicit-block",
        scene_id="scene-1",
        capture_id="capture-1",
        status="blocked",
        blockers=[],
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result={},
        copied_artifacts={},
        robot_pov_manifest={},
        policy_manifest={"interface_contract": {"reproducible_replay_required": True}},
        policy_execution_manifest={},
        evaluation_result={},
        proof_boundary={"public_claim_upgrade_allowed": False, "robot_readiness_proven": False},
        live_closure={},
        remote_cloud_closure={},
        webapp_status_projection={},
        data_package_export={},
        generated_at="2026-06-15T00:00:00Z",
    )

    requirements = {item["requirement_id"]: item for item in closure["requirements"]}
    assert requirements["batch_scenario_execution"]["passed"] is True
    assert "batch_scenario_execution" not in closure["blocked_requirement_ids"]
    assert closure["scenario_execution_summary"]["selected_scenario_runs_closed"] is True
    assert closure["scenario_execution_summary"]["explicitly_blocked_scenario_eval_run_ids"] == [
        "run-blocked"
    ]


def test_robot_team_grade_closure_marks_sim_only_beta_core_complete_with_trace_media_metrics(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "robot_eval_jobs" / "job-sim-beta-complete"
    scenario_eval_matrix = {
        "status": "completed",
        "scenario_eval_run_count": 1,
        "runs": [{"scenario_eval_run_id": "run-1", "task_id": "task-1", "scenario_id": "s-1"}],
    }
    task_success_summary = {
        "task_success_rate": 1.0,
        "successful_attempt_count": 1,
        "failed_attempt_count": 0,
        "goal_reached_attempt_count": 1,
        "fall_attempt_count": 0,
        "min_clearance_m": 0.15,
        "scene_contact_attempt_count": 0,
        "near_miss_event_count": 0,
        "max_path_deviation_m": 0.0,
        "stuck_attempt_count": 0,
        "policy_instability_attempt_count": 0,
    }
    _write_json(job_dir / "scenario_eval_matrix.json", scenario_eval_matrix)
    _write_json(
        job_dir / "simulator_command_batch_closure_manifest.json",
        {
            "required_scenario_eval_run_count": 1,
            "covered_scenario_eval_run_count": 1,
            "missing_scenario_eval_run_count": 0,
            "scenario_eval_run_coverage_complete": True,
            "required_scenario_eval_run_ids": ["run-1"],
            "covered_scenario_eval_run_ids": ["run-1"],
            "missing_scenario_eval_run_ids": [],
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "status": "completed",
            "attempt_count": 1,
            "required_scenario_eval_run_ids": ["run-1"],
            "covered_scenario_eval_run_ids": ["run-1"],
            "missing_scenario_eval_run_ids": [],
            "scenario_eval_run_coverage_complete": True,
            "task_success_summary": task_success_summary,
        },
    )
    _write_json(job_dir / "failure_labels.json", {"status": "no_failure_labels", "labels": []})
    _write_json(
        job_dir / "simulator_command_batch_metrics.json",
        {
            "attempt_metric_row_count": 1,
            "missing_metric_row_count": 0,
            "metric_coverage_complete": True,
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_visual_media_coverage.json",
        {
            "all_required_runs_have_visual_recording": True,
            "all_required_runs_have_robot_pov_video": True,
            "all_required_runs_have_third_person_video": True,
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_trace_package_manifest.json",
        {
            "contact_stream_record_count": 1,
            "planner_state_coverage_complete": True,
            "control_stream_coverage_complete": True,
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_artifact_checksums.json",
        {
            "artifacts": {
                "attempt_trace_jsonl": {"present": True},
                "contact_stream_jsonl": {"present": True},
                "planner_state_jsonl": {"present": True},
                "control_stream_jsonl": {"present": True},
                "metrics": {"present": True},
                "failure_labels": {"present": True},
                "visual_media_coverage": {"present": True},
            }
        },
    )
    for name in (
        "simulator_command_batch_attempt_trace.jsonl",
        "simulator_command_batch_contact_stream.jsonl",
        "simulator_command_batch_planner_state.jsonl",
        "simulator_command_batch_control_stream.jsonl",
        "robot_pov_observations.jsonl",
    ):
        (job_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (job_dir / name).write_text("{}\n", encoding="utf-8")
    for name in (
        "robot_pov_observation_manifest.json",
        "robot_pov_frame_sequence_manifest.json",
        "live_eval_closure_manifest.json",
        "proof_boundary.json",
        "post_training_data_package_export_manifest.json",
        "webapp_robot_eval_status_projection.json",
    ):
        _write_json(job_dir / name, {"status": "present"})
    evaluation_result = {
        "standard_policy_scorecard": {
            "cycle_time": {"sample_count": 1},
            "collision_risk": {"event_count": 0},
            "unsafe_proximity": {"event_count": 0},
        }
    }

    closure = _robot_team_grade_eval_closure_manifest(
        job_dir=job_dir,
        job_id="job-sim-beta-complete",
        scene_id="scene-1",
        capture_id="capture-1",
        status="simulator_command_completed",
        blockers=[],
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result={
            "status": "completed",
            "required_scenario_eval_run_ids": ["run-1"],
            "covered_scenario_eval_run_ids": ["run-1"],
            "missing_scenario_eval_run_ids": [],
        },
        copied_artifacts={},
        robot_pov_manifest={},
        policy_manifest={"status": "blocked"},
        policy_execution_manifest={},
        evaluation_result=evaluation_result,
        proof_boundary={"public_claim_upgrade_allowed": False, "robot_readiness_proven": False},
        live_closure={},
        remote_cloud_closure={},
        webapp_status_projection={
            "provider_complexity_hidden": True,
            "provider_details_exposed": False,
        },
        data_package_export={"status": "export_ready_review_required"},
        generated_at="2026-06-15T00:00:00Z",
    )

    assert closure["sim_only_beta_core_complete"] is True
    assert closure["robot_team_grade_evaluation_complete"] is False
    assert "digital_twin_fidelity_qa" in closure["blocked_requirement_ids"]
    assert "robot_team_policy_interface" in closure["blocked_requirement_ids"]
    sim_only_blockers = [
        item["requirement_id"]
        for item in closure["requirements"]
        if item["sim_only_beta_required"] and not item["passed"]
    ]
    assert sim_only_blockers == []


def test_robot_team_grade_closure_blocks_weak_explicit_scenario_block_records(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "robot_eval_jobs" / "job-weak-block"
    scenario_eval_matrix = {
        "status": "completed",
        "scenario_eval_run_count": 2,
        "runs": [
            {"scenario_eval_run_id": "run-covered", "task_id": "task-1", "scenario_id": "s-1"},
            {"scenario_eval_run_id": "run-blocked", "task_id": "task-1", "scenario_id": "s-2"},
        ],
    }
    _write_json(job_dir / "scenario_eval_matrix.json", scenario_eval_matrix)
    _write_json(
        job_dir / "simulator_command_batch_closure_manifest.json",
        {
            "required_scenario_eval_run_count": 2,
            "covered_scenario_eval_run_count": 1,
            "missing_scenario_eval_run_count": 1,
            "scenario_eval_run_coverage_complete": False,
            "required_scenario_eval_run_ids": ["run-covered", "run-blocked"],
            "covered_scenario_eval_run_ids": ["run-covered"],
            "missing_scenario_eval_run_ids": ["run-blocked"],
            "explicitly_blocked_scenario_eval_runs": [
                {"scenario_eval_run_id": "run-blocked"}
            ],
        },
    )

    closure = _robot_team_grade_eval_closure_manifest(
        job_dir=job_dir,
        job_id="job-weak-block",
        scene_id="scene-1",
        capture_id="capture-1",
        status="blocked",
        blockers=[],
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result={},
        copied_artifacts={},
        robot_pov_manifest={},
        policy_manifest={"interface_contract": {"reproducible_replay_required": True}},
        policy_execution_manifest={},
        evaluation_result={},
        proof_boundary={"public_claim_upgrade_allowed": False, "robot_readiness_proven": False},
        live_closure={},
        remote_cloud_closure={},
        webapp_status_projection={},
        data_package_export={},
        generated_at="2026-06-15T00:00:00Z",
    )

    requirements = {item["requirement_id"]: item for item in closure["requirements"]}
    assert requirements["batch_scenario_execution"]["passed"] is False
    assert "batch_scenario_execution" in closure["blocked_requirement_ids"]
    assert "scenario_eval_run_missing_without_explicit_blockers" in requirements[
        "batch_scenario_execution"
    ]["blockers"]
    assert "scenario_eval_run_blocker_records_missing_required_fields" in requirements[
        "batch_scenario_execution"
    ]["blockers"]
    assert closure["scenario_execution_summary"]["selected_scenario_runs_closed"] is False
    assert closure["scenario_execution_summary"]["invalid_explicit_blocker_record_run_ids"] == [
        "run-blocked"
    ]


def test_robot_team_grade_closure_blocks_incomplete_trace_stream_coverage(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "robot_eval_jobs" / "job-trace-streams"
    scenario_eval_matrix = {
        "status": "completed",
        "scenario_eval_run_count": 1,
        "runs": [{"scenario_eval_run_id": "run-1", "task_id": "task-1", "scenario_id": "s-1"}],
    }
    for name in (
        "simulator_command_batch_attempt_trace.jsonl",
        "simulator_command_batch_contact_stream.jsonl",
        "simulator_command_batch_planner_state.jsonl",
        "simulator_command_batch_control_stream.jsonl",
    ):
        (job_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (job_dir / name).write_text("{}\n", encoding="utf-8")
    _write_json(job_dir / "scenario_eval_matrix.json", scenario_eval_matrix)
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "required_scenario_eval_run_count": 1,
            "covered_scenario_eval_run_count": 1,
            "missing_scenario_eval_run_count": 0,
            "scenario_eval_run_coverage_complete": True,
            "task_success_rate": 1.0,
            "successful_task_attempt_count": 1,
            "failed_task_attempt_count": 0,
            "task_success_summary": {
                "goal_reached_attempt_count": 1,
                "fall_attempt_count": 0,
                "min_clearance_m": 0.42,
                "scene_contact_attempt_count": 0,
                "near_miss_event_count": 0,
                "max_path_deviation_m": 0.1,
                "stuck_attempt_count": 0,
                "policy_instability_attempt_count": 0,
            },
        },
    )
    _write_json(job_dir / "failure_labels.json", {"labels": [], "failed_attempt_count": 0})
    _write_json(
        job_dir / "simulator_command_batch_metrics.json",
        {
            "metric_coverage_complete": True,
            "attempt_metric_row_count": 1,
            "missing_metric_row_count": 0,
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_visual_media_coverage.json",
        {
            "all_required_runs_have_visual_recording": True,
            "all_required_runs_have_robot_pov_video": True,
            "all_required_runs_have_third_person_video": True,
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_artifact_checksums.json",
        {
            "artifacts": {
                "attempt_trace_jsonl": {"present": True},
                "contact_stream_jsonl": {"present": True},
                "planner_state_jsonl": {"present": True},
                "control_stream_jsonl": {"present": True},
                "metrics": {"present": True},
                "failure_labels": {"present": True},
                "visual_media_coverage": {"present": True},
            }
        },
    )
    _write_json(
        job_dir / "simulator_command_batch_trace_package_manifest.json",
        {
            "status": "completed",
            "contact_stream_record_count": 1,
            "planner_state_coverage_complete": False,
            "control_stream_coverage_complete": False,
        },
    )
    _write_json(job_dir / "robot_pov_observation_manifest.json", {"status": "completed"})
    _write_json(job_dir / "robot_pov_frame_sequence_manifest.json", {"status": "completed"})

    closure = _robot_team_grade_eval_closure_manifest(
        job_dir=job_dir,
        job_id="job-trace-streams",
        scene_id="scene-1",
        capture_id="capture-1",
        status="completed",
        blockers=[],
        scenario_eval_matrix=scenario_eval_matrix,
        simulator_result={},
        copied_artifacts={},
        robot_pov_manifest={},
        policy_manifest={
            "status": "review_required",
            "selected_modalities": ["policy_api_endpoint"],
            "interface_contract": {
                "observation_schema_id": "blueprint.robot_eval.observation.v1",
                "action_schema_id": "blueprint.robot_eval.action_trace.v1",
                "reproducible_replay_required": True,
            },
            "modalities": {
                "policy_api_endpoint": {
                    "status": "reference_present_requires_owner_system_review",
                    "selected": True,
                    "missing_inputs": [],
                }
            },
        },
        policy_execution_manifest={},
        evaluation_result={
            "standard_policy_scorecard": {
                "cycle_time": {"sample_count": 1},
                "collision_risk": {"event_count": 0},
                "unsafe_proximity": {"event_count": 0},
            }
        },
        proof_boundary={"public_claim_upgrade_allowed": False, "robot_readiness_proven": False},
        live_closure={},
        remote_cloud_closure={},
        webapp_status_projection={},
        data_package_export={},
        generated_at="2026-06-15T00:00:00Z",
    )

    requirements = {item["requirement_id"]: item for item in closure["requirements"]}
    full_trace = requirements["full_trace_package"]
    assert full_trace["passed"] is False
    assert "missing_trace_artifact_third_person_video_manifest" not in full_trace["blockers"]
    assert "planner_state_coverage_not_complete" in full_trace["blockers"]
    assert "control_stream_coverage_not_complete" in full_trace["blockers"]
    assert closure["closure_audit_summary"]["full_trace_package_complete"] is False


def test_webapp_execution_request_writes_scheduler_decision_and_blocks_gpu_without_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request()
    request["budget"] = {"budget_usd": 0, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "r2://blueprint-artifacts/jobs/job-webapp-execution-request",
    )
    monkeypatch.setenv(
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "r2://blueprint-artifacts/jobs/job-webapp-execution-request/worker_manifest.json",
    )

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-webapp-execution-request",
        provisioner="runpod",
        simulator="isaac_sim",
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim"],
        simulator_commands={"isaac_sim": f"{sys.executable} -c \"print('isaac ok')\""},
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-webapp-execution-request"
    scheduler = _read_json(job_dir / "scheduler_decision.json")
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_manifest = _read_json(job_dir / "worker_manifest.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    cost_ledger = _read_json(job_dir / "gpu_cost_control_ledger.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")

    assert scheduler["schema_version"] == "robot_eval_execution_scheduler_decision.v1"
    assert scheduler["webapp_role"] == "queue_and_forward_only"
    assert scheduler["scheduler_owner"] == "BlueprintCapturePipeline"
    assert scheduler["selection"]["simulator"] == "isaac_sim"  # type: ignore[index]
    assert scheduler["selection"]["recommended_simulator"] == "mujoco"  # type: ignore[index]
    assert scheduler["selection"]["simulator_selection_policy_mode"] == (  # type: ignore[index]
        "mujoco_first_unless_proof_requires_isaac"
    )
    assert scheduler["selection"]["selected_simulator_matches_request_policy"] is False  # type: ignore[index]
    simulator_policy = scheduler["simulator_selection_policy"]
    assert simulator_policy["recommended_backend"] == "mujoco"  # type: ignore[index]
    assert simulator_policy["selected_backend"] == "isaac_sim"  # type: ignore[index]
    assert simulator_policy["mujoco_first_applies"] is True  # type: ignore[index]
    assert simulator_policy["proof_boundary"]["mujoco_proof_does_not_clear_isaac_sim_gate"] is True  # type: ignore[index]
    assert "selected_simulator_differs_from_request_policy_recommendation" in simulator_policy[
        "non_blocking_warnings"
    ]
    assert scheduler["selection"]["worker_profile"]["worker_image_family"] == (  # type: ignore[index]
        "isaac-eval-worker"
    )
    assert scheduler["gpu_allocation"]["recommended_action"] == "do_not_allocate_gpu"  # type: ignore[index]
    assert scheduler["cpu_preflight_gate"]["required_before_gpu"] is True  # type: ignore[index]
    assert "scheduler_cpu_preflight_not_ready_for_gpu" in scheduler["blockers"]
    assert worker_plan["schema_version"] == "robot_eval_worker_launch_plan.v1"
    assert worker_plan["status"] == "blocked_by_scheduler"
    assert worker_plan["worker_image"]["image_family"] == "isaac-eval-worker"  # type: ignore[index]
    assert worker_plan["worker_image"]["dockerfile_path"] == (  # type: ignore[index]
        "deploy/docker/robot_eval_worker/isaac/Dockerfile"
    )
    assert worker_plan["worker_image"]["entrypoint"] == "blueprint-run-robot-eval-worker"  # type: ignore[index]
    assert worker_plan["worker_image"]["prebuilt_image_required"] is True  # type: ignore[index]
    assert worker_plan["worker_image"]["runtime_dependency_install_disallowed"] is True  # type: ignore[index]
    assert worker_plan["gpu_selection"]["preferred_gpu_class"] == (  # type: ignore[index]
        "rtx_rt_core_24gb_or_larger"
    )
    assert worker_plan["gpu_selection"]["disallowed_gpu_classes"] == ["a100", "h100"]  # type: ignore[index]
    assert "isaac_kit_cache" in worker_plan["cache_plan"]["targets"]  # type: ignore[index]
    assert worker_plan["launch_mode"]["mode"] == "on_demand_with_optional_warm_pool"  # type: ignore[index]
    assert worker_plan["launch_mode"]["idle_shutdown_required"] is True  # type: ignore[index]
    assert worker_plan["launch_mode"]["idle_timeout_seconds"] == 60  # type: ignore[index]
    assert worker_plan["launch_mode"]["external_watchdog_ttl_required"] is True  # type: ignore[index]
    assert worker_plan["launch_mode"]["external_watchdog_ttl_seconds"] == 180  # type: ignore[index]
    assert worker_plan["worker_entrypoint_contract"]["expected_command_shape"] == (  # type: ignore[index]
        "blueprint-run-robot-eval-worker --manifest ${BLUEPRINT_EVAL_MANIFEST_URI}"
    )
    assert worker_plan["worker_entrypoint_contract"]["package_console_script"] == (  # type: ignore[index]
        "blueprint-run-robot-eval-worker"
    )
    assert worker_plan["worker_entrypoint_contract"]["delegates_to_console_script"] == (  # type: ignore[index]
        "blueprint-run-robot-eval-job"
    )
    assert worker_plan["secret_policy"]["provider_credential_env_vars"] == ["RUNPOD_API_KEY"]  # type: ignore[index]
    assert worker_plan["secret_policy"]["store_provider_credentials_in_artifacts"] is False  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["upload_before_shutdown_required"] is True  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["manifest_input_uri_schemes"] == [  # type: ignore[index]
        "file",
        "http",
        "https",
        "gs",
        "s3",
        "r2",
    ]
    assert worker_plan["artifact_upload_contract"]["artifact_output_uri_schemes"] == [  # type: ignore[index]
        "file",
        "gs",
        "s3",
        "r2",
    ]
    assert worker_plan["artifact_upload_contract"]["artifact_output_uri_scheme"] == "r2"  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["artifact_output_uri_provider_writable"] is True  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["artifact_output_write_auth_contract_ready"] is True  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["artifact_output_write_auth"][  # type: ignore[index]
        "required_secret_env_vars"
    ] == ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    assert worker_plan["artifact_upload_contract"]["artifact_output_write_auth"][  # type: ignore[index]
        "secret_values_in_artifact"
    ] is False
    assert worker_plan["artifact_upload_contract"]["s3_compatible_storage_supported"] is True  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["r2_requires_endpoint_env"] is True  # type: ignore[index]
    assert worker_plan["runtime_preflight_contract"]["required_before_scene_load"] is True  # type: ignore[index]
    assert worker_plan["runtime_preflight_contract"][  # type: ignore[index]
        "worker_blocks_scene_load_on_failed_preflight"
    ] is True
    assert worker_plan["runtime_preflight_contract"]["result_artifact"] == (  # type: ignore[index]
        "worker_runtime_preflight.json"
    )
    assert "vulkan_device" in worker_plan["runtime_preflight_contract"]["required_checks"]  # type: ignore[index]
    assert worker_plan["runtime_preflight_contract"]["runtime_preflight_is_not_simulator_proof"] is True  # type: ignore[index]
    assert worker_plan["worker_manifest_input_contract"]["worker_manifest_uri_env_var"] == (  # type: ignore[index]
        "BLUEPRINT_EVAL_MANIFEST_URI"
    )
    assert worker_plan["worker_manifest_input_contract"]["configured_worker_manifest_uri"] == (  # type: ignore[index]
        "r2://blueprint-artifacts/jobs/job-webapp-execution-request/worker_manifest.json"
    )
    assert worker_plan["worker_manifest_input_contract"][  # type: ignore[index]
        "worker_manifest_uri_fetchable_by_provider"
    ] is True
    assert worker_manifest["schema_version"] == "robot_eval_worker_manifest.v1"
    assert worker_manifest["capture_root"] == str(capture_root)
    assert worker_manifest["provisioner"] == "runpod"
    assert worker_manifest["simulator"] == "isaac_sim"
    assert worker_manifest["worker_manifest_uri"] == (
        "r2://blueprint-artifacts/jobs/job-webapp-execution-request/worker_manifest.json"
    )
    assert worker_manifest["worker_manifest_uri_required"] is True
    assert worker_manifest["worker_manifest_uri_fetchable_by_provider"] is True
    assert worker_manifest["runtime_preflight_contract"]["result_artifact"] == (  # type: ignore[index]
        "worker_runtime_preflight.json"
    )
    assert worker_manifest["artifact_output_uri"] == (
        "r2://blueprint-artifacts/jobs/job-webapp-execution-request"
    )
    assert worker_manifest["artifact_output_uri_required"] is True
    assert worker_manifest["artifact_output_uri_scheme"] == "r2"
    assert worker_manifest["artifact_output_uri_provider_writable"] is True
    assert worker_manifest["artifact_output_write_auth_contract_ready"] is True
    assert worker_manifest["artifact_output_write_auth"]["authorization_mode"] == (
        "worker_storage_credentials"
    )
    assert worker_manifest["job_request"]["schema_version"] == "robot_eval_job_request.v1"  # type: ignore[index]
    assert provider_launch["schema_version"] == "robot_eval_gpu_provider_launch_request.v1"
    assert provider_launch["provider"] == "runpod"
    assert provider_launch["status"] == "blocked_by_scheduler"
    assert provider_launch["reason"] == "scheduler_decision_blocked"
    assert provider_launch["operation"] == "enqueue_runpod_serverless_or_on_demand_worker"
    assert provider_launch["live_provider_calls_performed"] is False
    assert provider_launch["worker_launch_plan_path"] == "worker_launch_plan.json"
    assert provider_launch["worker_launch_plan_status"] == "blocked_by_scheduler"
    assert provider_launch["worker_manifest_path"] == "worker_manifest.json"
    assert provider_launch["worker_manifest_status"] == "ready_for_worker_upload"
    assert provider_launch["provider_request_shape"]["image"]["image_family"] == (  # type: ignore[index]
        "isaac-eval-worker"
    )
    assert provider_launch["provider_request_shape"]["image"]["dockerfile_path"] == (  # type: ignore[index]
        "deploy/docker/robot_eval_worker/isaac/Dockerfile"
    )
    provider_command = provider_launch["provider_request_shape"]["command"]  # type: ignore[index]
    assert provider_command.startswith("blueprint-run-robot-eval-worker ")
    assert "--allow-gpu-provisioning" in provider_command
    assert "--allow-simulator-execution" in provider_command
    assert "--allowed-simulator isaac_sim" in provider_command
    assert "--simulator-command" in provider_command
    assert "isaac_sim=" in provider_command
    assert provider_launch["provider_request_shape"]["inputs"]["worker_manifest_path"] == (  # type: ignore[index]
        "worker_manifest.json"
    )
    assert provider_launch["provider_request_shape"]["inputs"]["worker_manifest_schema"] == (  # type: ignore[index]
        "robot_eval_worker_manifest.v1"
    )
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "worker_manifest_local_path_ready"
    ] is True
    assert provider_launch["provider_request_shape"]["inputs"]["manifest_uri"] == (  # type: ignore[index]
        "r2://blueprint-artifacts/jobs/job-webapp-execution-request/worker_manifest.json"
    )
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "manifest_uri_configured"
    ] is True
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "manifest_uri_fetchable_by_provider"
    ] is True
    assert provider_launch["provider_request_shape"]["runtime_preflight"][  # type: ignore[index]
        "required_before_scene_load"
    ] is True
    assert provider_launch["provider_request_shape"]["runtime_preflight"][  # type: ignore[index]
        "worker_blocks_scene_load_on_failed_preflight"
    ] is True
    assert provider_launch["provider_request_shape"]["runtime_preflight"][  # type: ignore[index]
        "result_artifact"
    ] == "worker_runtime_preflight.json"
    assert "isaac_headless_launch" in provider_launch["provider_request_shape"]["runtime_preflight"]["required_checks"]  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["inputs"]["artifact_output_uri"] == (  # type: ignore[index]
        "r2://blueprint-artifacts/jobs/job-webapp-execution-request"
    )
    assert provider_launch["provider_request_shape"]["inputs"]["artifact_output_uri_scheme"] == "r2"  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_uri_provider_writable"
    ] is True
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_write_auth_contract_ready"
    ] is True
    assert provider_launch["provider_request_shape"]["inputs"]["artifact_output_write_auth"][  # type: ignore[index]
        "required_secret_env_vars"
    ] == ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "provider_writable_artifact_output_uri_schemes"
    ] == ["gs", "r2", "s3"]
    assert provider_launch["provider_request_shape"]["gpu"]["preferred_gpu_class"] == (  # type: ignore[index]
        "rtx_rt_core_24gb_or_larger"
    )
    assert provider_launch["provider_request_shape"]["gpu"]["disallowed_gpu_classes"] == [  # type: ignore[index]
        "a100",
        "h100",
    ]
    assert provider_launch["provider_request_shape"]["limits"]["max_active_workers"] == 1  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["limits"]["idle_shutdown_required"] is True  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["limits"]["idle_timeout_seconds"] == 60  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["limits"]["external_watchdog_ttl_required"] is True  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["limits"]["external_watchdog_ttl_seconds"] == 180  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["limits"]["external_watchdog_owner"] == (  # type: ignore[index]
        "provider_launcher_or_owner_control_plane"
    )
    assert "BLUEPRINT_EVAL_MANIFEST_URI" in provider_launch["provider_request_shape"]["environment"]["plaintext_env_var_names"]  # type: ignore[index]
    assert "RUNPOD_API_KEY" in provider_launch["provider_request_shape"]["environment"]["secret_env_var_names"]  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["environment"]["secret_values_in_artifact"] is False  # type: ignore[index]
    assert provider_launch["gate_requirements"]["provider_credential_env_vars"] == [  # type: ignore[index]
        "RUNPOD_API_KEY"
    ]
    assert provider_launch["gate_requirements"]["env_BLUEPRINT_ALLOW_GPU_PROVISIONING_present"] is False  # type: ignore[index]
    assert provider_launch["gate_requirements"]["cli_allow_gpu_provisioning_present"] is True  # type: ignore[index]
    assert "scheduler_cpu_preflight_not_ready_for_gpu" in provider_launch["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING" in provider_launch["blockers"]
    assert cost_ledger["schema_version"] == "robot_eval_gpu_cost_control_ledger.v1"
    assert cost_ledger["provider"] == "runpod"
    assert cost_ledger["status"] == "blocked_before_allocation"
    assert cost_ledger["live_provider_calls_performed"] is False
    assert cost_ledger["budget"]["requested_budget_usd"] == 0  # type: ignore[index]
    assert cost_ledger["budget"]["gpu_spend_approved_by_webapp"] is False  # type: ignore[index]
    assert cost_ledger["worker_limits"]["max_active_workers"] == 1  # type: ignore[index]
    assert cost_ledger["worker_limits"]["hard_timeout_seconds"] == 120  # type: ignore[index]
    assert cost_ledger["worker_limits"]["max_billable_gpu_seconds"] == 120  # type: ignore[index]
    assert cost_ledger["worker_limits"]["idle_shutdown_required"] is True  # type: ignore[index]
    assert cost_ledger["worker_limits"]["external_watchdog_ttl_required"] is True  # type: ignore[index]
    assert cost_ledger["worker_limits"]["external_watchdog_ttl_seconds"] == 180  # type: ignore[index]
    assert cost_ledger["gpu_time"]["estimated_gpu_seconds"] == 0  # type: ignore[index]
    assert cost_ledger["gpu_time"]["actual_gpu_seconds"] is None  # type: ignore[index]
    assert cost_ledger["gpu_time"]["actual_gpu_time_record_required"] is True  # type: ignore[index]
    assert cost_ledger["artifact_finalizer"]["upload_before_shutdown_required"] is True  # type: ignore[index]
    assert cost_ledger["artifact_finalizer"]["shutdown_after_artifacts_required"] is True  # type: ignore[index]
    assert "scheduler_cpu_preflight_not_ready_for_gpu" in cost_ledger["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING" in cost_ledger["blockers"]
    assert provisioning["status"] == "blocked"
    assert provisioning["reason"] == "scheduler_decision_blocked"
    assert "scheduler_cpu_preflight_not_ready_for_gpu" in provisioning["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_GPU_PROVISIONING" in provisioning["blockers"]
    assert provisioning["worker_launch_plan_path"] == "worker_launch_plan.json"
    assert provisioning["worker_launch_plan_status"] == "blocked_by_scheduler"
    assert provisioning["gpu_provider_launch_request_path"] == "gpu_provider_launch_request.json"
    assert provisioning["gpu_provider_launch_request_status"] == "blocked_by_scheduler"
    assert provisioning["gpu_cost_control_ledger_path"] == "gpu_cost_control_ledger.json"
    assert provisioning["gpu_cost_control_ledger_status"] == "blocked_before_allocation"
    assert run_manifest["scheduler_decision_status"] == "blocked"
    assert run_manifest["worker_launch_plan_status"] == "blocked_by_scheduler"
    assert run_manifest["worker_launch_plan_path"] == "worker_launch_plan.json"
    assert run_manifest["gpu_provider_launch_request_status"] == "blocked_by_scheduler"
    assert run_manifest["gpu_provider_launch_request_path"] == "gpu_provider_launch_request.json"
    assert run_manifest["gpu_cost_control_ledger_status"] == "blocked_before_allocation"
    assert run_manifest["gpu_cost_control_ledger_path"] == "gpu_cost_control_ledger.json"
    assert run_manifest["startup_architecture_audit_status"] == "blocked"
    assert run_manifest["startup_architecture_audit_path"] == "startup_architecture_audit.json"
    assert run_manifest["startup_architecture_compliant"] is False
    assert run_manifest["cpu_preflight_artifacts"]["scheduler_decision"] == (
        "scheduler_decision.json"
    )
    assert run_manifest["cpu_preflight_artifacts"]["worker_launch_plan"] == (
        "worker_launch_plan.json"
    )
    assert run_manifest["cpu_preflight_artifacts"]["gpu_provider_launch_request"] == (
        "gpu_provider_launch_request.json"
    )
    assert run_manifest["cpu_preflight_artifacts"]["gpu_cost_control_ledger"] == (
        "gpu_cost_control_ledger.json"
    )
    assert blocked["evidence"]["worker_launch_plan_status"] == "blocked_by_scheduler"
    assert blocked["evidence"]["gpu_provider_launch_request_status"] == "blocked_by_scheduler"
    assert blocked["evidence"]["gpu_cost_control_ledger_status"] == "blocked_before_allocation"
    assert "gpu_provisioning_blocked" in blocked["blockers"]


def test_live_provider_launch_blocks_without_versioned_worker_image_ref(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_cpu_preflight_ready_scene_asset(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request()
    request["budget"] = {"budget_usd": 10, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-missing-image-ref",
    )
    monkeypatch.setenv(
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-missing-image-ref/worker_manifest.json",
    )
    monkeypatch.delenv("BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF", raising=False)
    monkeypatch.delenv("BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF", raising=False)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-live-provider-missing-image-ref",
        provisioner="runpod",
        simulator="isaac_sim",
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim"],
        simulator_commands={"isaac_sim": f"{sys.executable} -c \"print('isaac ok')\""},
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / (
        "job-live-provider-missing-image-ref"
    )
    scheduler = _read_json(job_dir / "scheduler_decision.json")
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    cost_ledger = _read_json(job_dir / "gpu_cost_control_ledger.json")
    remote_closure = _read_json(job_dir / "remote_cloud_execution_closure_manifest.json")

    assert scheduler["status"] == "awaiting_explicit_gpu_and_simulator_gates"
    assert scheduler["blockers"] == []
    assert worker_plan["status"] == "blocked_missing_prebuilt_worker_image_ref"
    assert "missing_prebuilt_worker_image_ref" in worker_plan["blockers"]
    assert remote_closure["status"] == "blocked_before_remote_execution"
    assert "remote_worker_image_not_pinned_or_fetchable" in remote_closure["contract_blockers"]
    assert worker_plan["worker_image"]["published_image_ref_required"] is True  # type: ignore[index]
    assert worker_plan["worker_image"]["image_ref_env_var"] == (  # type: ignore[index]
        "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF"
    )
    assert worker_plan["worker_image"]["configured_image_ref_present"] is False  # type: ignore[index]
    assert provider_launch["status"] == "blocked_by_worker_plan"
    assert provider_launch["reason"] == "worker_launch_plan_blocked"
    assert provider_launch["provider_request_shape"]["image"][  # type: ignore[index]
        "owner_published_image_ref_required"
    ] is True
    assert provider_launch["provider_request_shape"]["image"][  # type: ignore[index]
        "configured_image_ref_present"
    ] is False
    assert "missing_prebuilt_worker_image_ref" in provider_launch["blockers"]
    assert provisioning["status"] == "blocked"
    assert provisioning["reason"] == "worker_launch_plan_blocked"
    assert "missing_prebuilt_worker_image_ref" in provisioning["blockers"]
    assert cost_ledger["status"] == "blocked_before_allocation"
    assert "missing_prebuilt_worker_image_ref" in cost_ledger["blockers"]
    assert cost_ledger["gpu_time"]["estimated_gpu_seconds"] == 0  # type: ignore[index]


def test_live_provider_launch_blocks_without_worker_manifest_uri(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_cpu_preflight_ready_scene_asset(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request()
    request["budget"] = {"budget_usd": 10, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-missing-manifest-uri",
    )
    monkeypatch.setenv(
        "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "registry.example/blueprint/isaac-eval-worker:2026-06-12",
    )
    monkeypatch.delenv("BLUEPRINT_EVAL_MANIFEST_URI", raising=False)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-live-provider-missing-manifest-uri",
        provisioner="runpod",
        simulator="isaac_sim",
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim"],
        simulator_commands={"isaac_sim": f"{sys.executable} -c \"print('isaac ok')\""},
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / (
        "job-live-provider-missing-manifest-uri"
    )
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_manifest = _read_json(job_dir / "worker_manifest.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    cost_ledger = _read_json(job_dir / "gpu_cost_control_ledger.json")

    assert worker_plan["status"] == "blocked_missing_worker_manifest_uri"
    assert "missing_worker_manifest_uri" in worker_plan["blockers"]
    assert worker_plan["worker_manifest_input_contract"][  # type: ignore[index]
        "worker_manifest_uri_required_for_provider"
    ] is True
    assert worker_plan["worker_manifest_input_contract"][  # type: ignore[index]
        "configured_worker_manifest_uri_present"
    ] is False
    assert worker_manifest["status"] == "blocked"
    assert worker_manifest["worker_manifest_uri"] is None
    assert worker_manifest["worker_manifest_uri_required"] is True
    assert "missing_worker_manifest_uri" in worker_manifest["blockers"]
    assert provider_launch["status"] == "blocked_by_worker_plan"
    assert provider_launch["reason"] == "worker_launch_plan_blocked"
    assert provider_launch["provider_request_shape"]["inputs"]["manifest_uri"] is None  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "manifest_uri_configured"
    ] is False
    assert "missing_worker_manifest_uri" in provider_launch["blockers"]
    assert provisioning["status"] == "blocked"
    assert "missing_worker_manifest_uri" in provisioning["blockers"]
    assert cost_ledger["status"] == "blocked_before_allocation"
    assert "missing_worker_manifest_uri" in cost_ledger["blockers"]


def test_live_provider_launch_blocks_https_artifact_output_uri(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_cpu_preflight_ready_scene_asset(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request()
    request["budget"] = {"budget_usd": 10, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "https://example.com/blueprint-artifacts/jobs/job-live-provider-https-output",
    )
    monkeypatch.setenv(
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-https-output/worker_manifest.json",
    )
    monkeypatch.setenv(
        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-https-output/capture-root.zip",
    )
    monkeypatch.setenv(
        "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "registry.example/blueprint/isaac-eval-worker:2026-06-12",
    )

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-live-provider-https-output",
        provisioner="runpod",
        simulator="isaac_sim",
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim"],
        simulator_commands={"isaac_sim": f"{sys.executable} -c \"print('isaac ok')\""},
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / (
        "job-live-provider-https-output"
    )
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_manifest = _read_json(job_dir / "worker_manifest.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    remote_closure = _read_json(job_dir / "remote_cloud_execution_closure_manifest.json")

    assert worker_plan["status"] == "blocked_invalid_worker_artifact_output_uri"
    assert "worker_artifact_output_uri_not_provider_writable" in worker_plan["blockers"]
    assert worker_plan["artifact_upload_contract"]["artifact_output_uri_scheme"] == "https"  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["artifact_output_uri_provider_writable"] is False  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["artifact_output_write_auth_contract_ready"] is False  # type: ignore[index]
    assert worker_plan["artifact_upload_contract"]["artifact_output_write_auth"][  # type: ignore[index]
        "authorization_mode"
    ] == "unsupported_uri_scheme"
    assert worker_plan["artifact_upload_contract"][  # type: ignore[index]
        "remote_provider_writable_artifact_output_uri_schemes"
    ] == ["gs", "r2", "s3"]
    assert worker_manifest["status"] == "blocked"
    assert worker_manifest["artifact_output_uri_scheme"] == "https"
    assert worker_manifest["artifact_output_uri_provider_writable"] is False
    assert worker_manifest["artifact_output_write_auth_contract_ready"] is False
    assert "worker_artifact_output_uri_not_provider_writable" in worker_manifest["blockers"]
    assert provider_launch["status"] == "blocked_by_worker_plan"
    assert provider_launch["provider_request_shape"]["inputs"]["artifact_output_uri_scheme"] == "https"  # type: ignore[index]
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_uri_provider_writable"
    ] is False
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "artifact_output_write_auth_contract_ready"
    ] is False
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "provider_writable_artifact_output_uri_schemes"
    ] == ["gs", "r2", "s3"]
    assert "worker_artifact_output_uri_not_provider_writable" in provider_launch["blockers"]
    assert remote_closure["status"] == "blocked_before_remote_execution"
    assert "remote_artifact_output_uri_not_provider_writable" in remote_closure[
        "contract_blockers"
    ]
    assert remote_closure["checks"]["artifact_output_uri_provider_writable"] is False
    assert remote_closure["checks"]["artifact_output_write_auth_contract_ready"] is False
    assert remote_closure["outputs"]["artifact_output_uri_scheme"] == "https"


def test_live_provider_launch_blocks_without_capture_root_bundle_uri(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_cpu_preflight_ready_scene_asset(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request()
    request["budget"] = {"budget_usd": 10, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-missing-capture-bundle",
    )
    monkeypatch.setenv(
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-missing-capture-bundle/worker_manifest.json",
    )
    monkeypatch.setenv(
        "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "registry.example/blueprint/isaac-eval-worker:2026-06-12",
    )
    monkeypatch.delenv("BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI", raising=False)

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-live-provider-missing-capture-bundle",
        provisioner="runpod",
        simulator="isaac_sim",
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim"],
        simulator_commands={"isaac_sim": f"{sys.executable} -c \"print('isaac ok')\""},
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / (
        "job-live-provider-missing-capture-bundle"
    )
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_manifest = _read_json(job_dir / "worker_manifest.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")

    assert worker_plan["status"] == "blocked_missing_capture_root_bundle_uri"
    assert "missing_capture_root_bundle_uri" in worker_plan["blockers"]
    assert worker_plan["input_bundle"][  # type: ignore[index]
        "capture_root_bundle_uri_required_for_provider"
    ] is True
    assert worker_plan["input_bundle"][  # type: ignore[index]
        "capture_root_bundle_uri_fetchable_by_provider"
    ] is False
    assert worker_manifest["capture_root_bundle_uri"] is None
    assert provider_launch["status"] == "blocked_by_worker_plan"
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "capture_root_bundle_uri_configured"
    ] is False
    assert "missing_capture_root_bundle_uri" in provider_launch["blockers"]
    assert "missing_capture_root_bundle_uri" in provisioning["blockers"]


def test_live_provider_launch_blocks_when_cpu_preflight_gate_is_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_cpu_preflight_ready_scene_asset(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request_without_cpu_preflight()
    request["budget"] = {"budget_usd": 10, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-disabled-cpu-preflight",
    )
    monkeypatch.setenv(
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-disabled-cpu-preflight/worker_manifest.json",
    )
    monkeypatch.setenv(
        "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "registry.example/blueprint/isaac-eval-worker:2026-06-12",
    )

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-live-provider-disabled-cpu-preflight",
        provisioner="runpod",
        simulator="isaac_sim",
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
        allowed_simulators=["isaac_sim"],
        simulator_commands={"isaac_sim": f"{sys.executable} -c \"print('isaac ok')\""},
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / (
        "job-live-provider-disabled-cpu-preflight"
    )
    scheduler = _read_json(job_dir / "scheduler_decision.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    cost_ledger = _read_json(job_dir / "gpu_cost_control_ledger.json")

    assert scheduler["status"] == "blocked"
    assert scheduler["cpu_preflight_gate"]["required_before_gpu"] is True  # type: ignore[index]
    assert "execution_request_cpu_preflight_gate_disabled_for_gpu" in scheduler["blockers"]
    assert provider_launch["status"] == "blocked_by_scheduler"
    assert provider_launch["reason"] == "scheduler_decision_blocked"
    assert "execution_request_cpu_preflight_gate_disabled_for_gpu" in provider_launch["blockers"]
    assert provisioning["status"] == "blocked"
    assert provisioning["reason"] == "scheduler_decision_blocked"
    assert cost_ledger["status"] == "blocked_before_allocation"
    assert cost_ledger["gpu_time"]["estimated_gpu_seconds"] == 0  # type: ignore[index]


def test_live_provider_launch_accepts_versioned_worker_image_ref_after_cpu_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_cpu_preflight_ready_scene_asset(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request()
    request["budget"] = {"budget_usd": 10, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv(
        "BLUEPRINT_ARTIFACT_OUTPUT_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref",
    )
    monkeypatch.setenv(
        "BLUEPRINT_EVAL_MANIFEST_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref/worker_manifest.json",
    )
    monkeypatch.setenv(
        "BLUEPRINT_CAPTURE_ROOT_BUNDLE_URI",
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref/capture-root.zip",
    )
    monkeypatch.setenv(
        "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "registry.example/blueprint/isaac-eval-worker:2026-06-12",
    )

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-live-provider-versioned-image-ref",
        provisioner="runpod",
        simulator="isaac_sim",
        allow_gpu_provisioning=True,
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / (
        "job-live-provider-versioned-image-ref"
    )
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_manifest = _read_json(job_dir / "worker_manifest.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    provisioning = _read_json(job_dir / "gpu_provisioning_result.json")
    cost_ledger = _read_json(job_dir / "gpu_cost_control_ledger.json")
    remote_closure = _read_json(job_dir / "remote_cloud_execution_closure_manifest.json")

    assert worker_plan["status"] == "awaiting_explicit_provider_gate"
    assert worker_plan["blockers"] == []
    assert worker_plan["input_bundle"]["capture_root_bundle_uri"] == (  # type: ignore[index]
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref/capture-root.zip"
    )
    assert worker_plan["input_bundle"][  # type: ignore[index]
        "capture_root_bundle_uri_fetchable_by_provider"
    ] is True
    assert worker_plan["worker_image"]["configured_image_ref"] == (  # type: ignore[index]
        "registry.example/blueprint/isaac-eval-worker:2026-06-12"
    )
    assert worker_plan["worker_image"]["configured_image_ref_is_versioned"] is True  # type: ignore[index]
    assert worker_plan["worker_image"]["configured_image_ref_fetchable_by_provider"] is True  # type: ignore[index]
    assert worker_plan["runtime_preflight_contract"]["required_before_scene_load"] is True  # type: ignore[index]
    assert worker_plan["runtime_preflight_contract"]["vulkan_required"] is True  # type: ignore[index]
    assert worker_manifest["status"] == "ready_for_worker_upload"
    assert worker_manifest["worker_manifest_uri"] == (
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref/worker_manifest.json"
    )
    assert worker_manifest["capture_root_bundle_uri"] == (
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref/capture-root.zip"
    )
    assert worker_manifest["worker_manifest_uri_required"] is True
    assert worker_manifest["worker_manifest_uri_fetchable_by_provider"] is True
    assert "test_frame_render" in worker_manifest["runtime_preflight_contract"]["required_checks"]  # type: ignore[index]
    assert worker_manifest["artifact_output_uri"] == (
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref"
    )
    assert provider_launch["status"] == "request_manifest_ready"
    assert provider_launch["reason"] == "provider_launch_request_ready_for_explicit_launcher"
    provider_command = provider_launch["provider_request_shape"]["command"]  # type: ignore[index]
    assert provider_command.startswith("blueprint-run-robot-eval-worker ")
    assert "--allow-gpu-provisioning" in provider_command
    assert "--allow-simulator-execution" not in provider_command
    assert "--allowed-simulator" not in provider_command
    assert "--simulator-command" not in provider_command
    assert provider_launch["provider_request_shape"]["image"]["configured_image_ref"] == (  # type: ignore[index]
        "registry.example/blueprint/isaac-eval-worker:2026-06-12"
    )
    assert provider_launch["provider_request_shape"]["image"][  # type: ignore[index]
        "configured_image_ref_fetchable_by_provider"
    ] is True
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "worker_manifest_local_path_ready"
    ] is True
    assert provider_launch["provider_request_shape"]["inputs"]["manifest_uri"] == (  # type: ignore[index]
        "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref/worker_manifest.json"
    )
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "manifest_uri_fetchable_by_provider"
    ] is True
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "capture_root_bundle_uri"
    ] == "r2://blueprint-artifacts/jobs/job-live-provider-versioned-image-ref/capture-root.zip"
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "capture_root_bundle_uri_fetchable_by_provider"
    ] is True
    assert provider_launch["provider_request_shape"]["runtime_preflight"][  # type: ignore[index]
        "vulkan_required"
    ] is True
    assert "rtx_renderer_available" in provider_launch["provider_request_shape"]["runtime_preflight"]["required_checks"]  # type: ignore[index]
    assert provider_launch["live_provider_calls_performed"] is False
    assert provisioning["status"] == "request_manifest_ready"
    assert provisioning["live_provider_calls_performed"] is False
    assert cost_ledger["status"] == "ready_for_explicit_provider_launcher"
    assert cost_ledger["gpu_time"]["estimated_gpu_seconds"] == 120  # type: ignore[index]
    assert cost_ledger["gpu_time"]["actual_gpu_seconds"] is None  # type: ignore[index]
    assert remote_closure["status"] == "ready_for_explicit_provider_runtime"
    assert remote_closure["contract_ready_for_remote_runtime"] is True
    assert remote_closure["remote_cloud_execution_proven"] is False
    assert remote_closure["clean_shutdown_proven"] is False
    assert remote_closure["checks"]["versioned_worker_image_ref_pinned"] is True
    assert remote_closure["checks"]["worker_manifest_uri_fetchable"] is True
    assert remote_closure["checks"]["capture_root_bundle_uri_fetchable"] is True
    assert remote_closure["checks"]["artifact_output_uri_configured"] is True
    assert remote_closure["runtime_blockers"] == ["remote_provider_runtime_not_executed"]


def test_provider_input_setup_prepares_capture_bundle_and_worker_manifest(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_cpu_preflight_ready_scene_asset(capture_root)
    request = _full_job_request(capture_root)
    request["execution_request"] = _webapp_execution_request()
    request["budget"] = {"budget_usd": 10, "timeout_seconds": 120}
    request_path = tmp_path / "webapp-job-request.json"
    _write_json(request_path, request)

    result = prepare_robot_eval_provider_inputs(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-provider-input-setup",
        artifact_root_uri="r2://blueprint-artifacts/jobs/job-provider-input-setup",
        simulator="mujoco",
        provisioner="runpod",
        image_ref="registry.example/blueprint/mujoco-eval-worker:2026-06-12",
        output_dir=tmp_path / "provider_inputs",
        upload=False,
        allow_gpu_provisioning=True,
        allow_simulator_execution=False,
        timeout_seconds=600,
        budget_usd=10,
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-provider-input-setup"
    worker_plan = _read_json(job_dir / "worker_launch_plan.json")
    worker_manifest = _read_json(job_dir / "worker_manifest.json")
    provider_launch = _read_json(job_dir / "gpu_provider_launch_request.json")
    remote_closure = _read_json(job_dir / "remote_cloud_execution_closure_manifest.json")

    assert result["status"] == "prepared_with_external_blockers"
    assert result["blockers"] == ["provider_inputs_upload_not_proven"]
    assert result["bundle"]["raw_media_excluded_by_default"] is True  # type: ignore[index]
    assert Path(result["bundle"]["path"]).is_file()  # type: ignore[index]
    assert result["capture_root_bundle_uri"] == (
        "r2://blueprint-artifacts/jobs/job-provider-input-setup/capture-root.zip"
    )
    assert result["worker_manifest_uri"] == (
        "r2://blueprint-artifacts/jobs/job-provider-input-setup/worker_manifest.json"
    )
    assert result["artifact_output_uri"] == (
        "r2://blueprint-artifacts/jobs/job-provider-input-setup/artifacts"
    )
    assert worker_plan["input_bundle"]["capture_root_bundle_uri"] == result[  # type: ignore[index]
        "capture_root_bundle_uri"
    ]
    assert worker_manifest["capture_root_bundle_uri"] == result["capture_root_bundle_uri"]
    assert worker_manifest["runtime_preflight_command"] == (
        "python -m blueprint_pipeline.mujoco_worker_runtime_preflight --smoke-steps 2"
    )
    assert provider_launch["provider_request_shape"]["inputs"][  # type: ignore[index]
        "capture_root_bundle_uri_fetchable_by_provider"
    ] is True
    assert provider_launch["status"] == "blocked_provider_input_setup"
    assert "provider_input_setup_blocked" in provider_launch["blockers"]
    assert "provider_inputs_upload_not_proven" in provider_launch["blockers"]
    assert remote_closure["status"] == "blocked_before_remote_execution"
    assert remote_closure["contract_ready_for_remote_runtime"] is False
    assert remote_closure["provider_input_setup"]["provider_inputs_uploaded"] is False
    assert "provider_input_setup:provider_inputs_upload_not_proven" in remote_closure[
        "contract_blockers"
    ]
    assert (tmp_path / "provider_inputs" / "provider_input_env.sh").is_file()
    assert (tmp_path / "provider_inputs" / "provider_input_setup_manifest.json").is_file()
    publish_script = tmp_path / "provider_inputs" / "provider_publish_resolution.sh"
    assert publish_script.is_file()
    publish_text = publish_script.read_text(encoding="utf-8")
    assert "docker push" in publish_text
    assert "aws s3 cp" in publish_text
    assert "RUNPOD_API_KEY" not in publish_text


def test_provider_input_upload_failure_is_recorded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "capture-root.zip"
    source.write_text("zip-bytes", encoding="utf-8")

    def _fail_upload(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("billing disabled")

    monkeypatch.setattr(provider_input_setup, "_upload_file_to_gs", _fail_upload)

    result = provider_input_setup.upload_file(
        source,
        "gs://blueprint-artifacts/jobs/job-1/capture-root.zip",
    )

    assert result["status"] == "blocked"
    assert result["storage_scheme"] == "gs"
    assert result["blockers"] == ["upload_failed:RuntimeError"]
    assert "billing disabled" in str(result["error"])


def test_provider_input_upload_failure_classifies_disabled_gcs_billing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "capture-root.zip"
    source.write_text("zip-bytes", encoding="utf-8")

    def _fail_upload(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("The billing account for the owning project is disabled in state absent")

    monkeypatch.setattr(provider_input_setup, "_upload_file_to_gs", _fail_upload)

    result = provider_input_setup.upload_file(
        source,
        "gs://blueprint-artifacts/jobs/job-1/capture-root.zip",
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["upload_failed:gs_billing_account_disabled"]


def test_robot_eval_worker_runs_local_manifest_and_copies_artifacts(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request = _full_job_request(capture_root)
    request["job_id"] = "worker-fixture-job"
    manifest_path = tmp_path / "worker_manifest.json"
    artifact_output_dir = tmp_path / "worker_artifacts"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-fixture-job",
            "provisioner": "fixture_local",
            "simulator": "fixture",
            "job_request": request,
            "artifact_output_uri": str(artifact_output_dir),
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    worker_manifest = _read_json(tmp_path / "worker" / "worker_runtime_manifest.json")
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "worker-fixture-job"

    assert runtime["status"] == "completed"
    assert worker_manifest["status"] == "completed"
    assert worker_manifest["job_status"] == "fixture_evaluation_completed"
    assert worker_manifest["artifact_upload"]["status"] == "completed"  # type: ignore[index]
    assert worker_manifest["artifact_upload"]["worker_runtime_manifest_included"] is True  # type: ignore[index]
    assert worker_manifest["live_provider_calls_performed"] is False
    assert worker_manifest["simulator_execution_proven"] is False
    assert (job_dir / "job_run_manifest.json").is_file()
    assert (job_dir / "startup_architecture_audit.json").is_file()
    assert (job_dir / "worker_runtime_preflight.json").is_file()
    assert (job_dir / "worker_runtime_manifest.json").is_file()
    assert (artifact_output_dir / "job_run_manifest.json").is_file()
    assert (artifact_output_dir / "startup_architecture_audit.json").is_file()
    assert (artifact_output_dir / "worker_runtime_preflight.json").is_file()
    assert (artifact_output_dir / "worker_runtime_manifest.json").is_file()
    assert (artifact_output_dir / "worker_manifest.json").is_file()
    assert (artifact_output_dir / "worker_launch_plan.json").is_file()
    assert (artifact_output_dir / "gpu_provider_launch_request.json").is_file()
    assert (artifact_output_dir / "gpu_cost_control_ledger.json").is_file()


def test_robot_eval_worker_runtime_manifest_propagates_command_simulator_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    simulator_script = tmp_path / "write_simulator_output.py"
    simulator_script.write_text(
        "\n".join(
            [
                "import json, os",
                "out = os.environ['BLUEPRINT_SIMULATOR_OUTPUT']",
                "matrix_path = os.environ['BLUEPRINT_SCENARIO_EVAL_MATRIX']",
                "matrix = json.load(open(matrix_path, encoding='utf-8'))",
                "required = [run['scenario_eval_run_id'] for run in matrix['runs']]",
                "payload = {",
                "  'required_scenario_eval_run_ids': required,",
                "  'attempts': [",
                "    {",
                "      'attempt_id': f'worker-command-attempt-{index + 1}',",
                "      'scenario_eval_run_id': run['scenario_eval_run_id'],",
                "      'task_id': run['task_id'],",
                "      'scenario_id': run['scenario_id'],",
                "      'scenario_variation_instance_id': run.get('scenario_variation_instance_id'),",
                "      'variation_name': run.get('variation_name'),",
                "      'policy_id': 'policy-command',",
                "      'status': 'completed',",
                "      'success': True,",
                "      'metrics': {'cycle_time_seconds': 9.0, 'intervention_count': 0},",
                "      'actions': [{'type': 'move_base', 'target': 'bin_approach'}]",
                "    }",
                "    for index, run in enumerate(matrix['runs'])",
                "  ]",
                "}",
                "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    preflight_script = tmp_path / "runtime_preflight.py"
    preflight_script.write_text(
        "\n".join(
            [
                "import json, os",
                "detail = os.environ.get('BLUEPRINT_RUNTIME_PREFLIGHT_DETAIL_OUTPUT')",
                "if detail:",
                "    open(detail, 'w', encoding='utf-8').write(json.dumps({'status': 'passed', 'blockers': []}))",
            ]
        ),
        encoding="utf-8",
    )
    request = _full_job_request(capture_root)
    request["job_id"] = "worker-command-sim-job"
    request["simulator_preference"] = "pybullet"
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-command-sim-job",
            "provisioner": "fixture_local",
            "simulator": "pybullet",
            "allowed_simulators": ["pybullet"],
            "simulator_commands": {"pybullet": f"{sys.executable} {simulator_script}"},
            "runtime_preflight_command": f"{sys.executable} {preflight_script}",
            "job_request": request,
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
        allow_simulator_execution=True,
    )

    worker_manifest = _read_json(tmp_path / "worker" / "worker_runtime_manifest.json")

    assert runtime["status"] == "completed"
    assert worker_manifest["job_status"] == "simulator_command_completed"
    assert worker_manifest["simulator_execution_proven"] is True
    assert worker_manifest["robot_readiness_proven"] is False
    assert worker_manifest["public_claim_upgrade_allowed"] is False


def test_robot_eval_worker_runs_sim_only_command_when_full_job_scope_blocks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    matrix_rel = (
        "pipeline/robot_eval_jobs/worker-sim-only-matrix/scenario_eval_matrix.json"
    )
    matrix_path = capture_root / matrix_rel
    _write_json(
        matrix_path,
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "blockers": [],
            "scenario_eval_run_count": 1,
            "runs": [
                {
                    "scenario_eval_run_id": "run-sim-only-1",
                    "task_id": "place_return_in_bin",
                    "scenario_id": "scenario_place_return_in_bin_mobile",
                    "scenario_variation_instance_id": "variation-sim-only-1",
                }
            ],
        },
    )
    simulator_script = tmp_path / "write_sim_only_output.py"
    simulator_script.write_text(
        "\n".join(
            [
                "import json, os",
                "assert os.environ['BLUEPRINT_SCENARIO_EVAL_MATRIX'].endswith('scenario_eval_matrix.json')",
                "out = os.environ['BLUEPRINT_SIMULATOR_OUTPUT']",
                "payload = {",
                "  'status': 'completed',",
                "  'simulator_execution_proven': True,",
                "  'mujoco_g1_asset_execution_proven': True,",
                "  'mujoco_g1_asset_spawned': True,",
                "  'attempts': [{'attempt_id': 'run-sim-only-1', 'status': 'completed', 'success': True}],",
                "}",
                "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    preflight_script = tmp_path / "runtime_preflight.py"
    preflight_script.write_text(
        "\n".join(
            [
                "import json, os",
                "detail = os.environ.get('BLUEPRINT_RUNTIME_PREFLIGHT_DETAIL_OUTPUT')",
                "if detail:",
                "    open(detail, 'w', encoding='utf-8').write(json.dumps({'status': 'passed', 'blockers': []}))",
            ]
        ),
        encoding="utf-8",
    )
    request = _full_job_request(capture_root)
    request["job_id"] = "worker-sim-only-blocked-full-job"
    request["requested_tasks"] = [
        {
            "task_id": "unknown_task_for_full_job_block",
            "scenario_ids": ["unknown_scenario_for_full_job_block"],
        }
    ]
    request["simulator_preference"] = "pybullet"
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-sim-only-blocked-full-job",
            "provisioner": "runpod",
            "simulator": "pybullet",
            "allowed_simulators": ["pybullet"],
            "simulator_commands": {"pybullet": f"{sys.executable} {simulator_script}"},
            "scenario_eval_matrix_path": matrix_rel,
            "runtime_preflight_contract": {
                "required_before_scene_load": True,
                "worker_blocks_scene_load_on_failed_preflight": True,
                "run_before": "scene_load_and_policy_execution",
                "result_artifact": "worker_runtime_preflight.json",
                "required_checks": ["pybullet_runtime_smoke"],
                "runtime_preflight_is_not_simulator_proof": True,
            },
            "runtime_preflight_command": f"{sys.executable} {preflight_script}",
            "artifact_output_uri_required": False,
            "job_request": request,
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
        allow_gpu_provisioning=True,
        allow_simulator_execution=True,
    )

    worker_manifest = _read_json(tmp_path / "worker" / "worker_runtime_manifest.json")
    assert runtime["status"] == "completed"
    assert worker_manifest["status"] == "completed"
    assert worker_manifest["job_status"] == "simulator_command_completed"
    assert worker_manifest["full_job_status"] == "blocked"
    assert "scenario_eval_matrix_blocked" in worker_manifest["full_job_blockers"]
    assert worker_manifest["job_blockers"] == []
    assert worker_manifest["scenario_eval_matrix_status"] == "completed"
    assert worker_manifest["simulator_service_status"] == "completed"
    assert worker_manifest["evaluation_status"] == "completed"
    assert worker_manifest["simulator_execution_proven"] is True
    assert worker_manifest["robot_readiness_proven"] is False
    assert worker_manifest["public_claim_upgrade_allowed"] is False
    assert worker_manifest["provider_runtime_simulator_command_result"]["status"] == "completed"


def test_robot_eval_worker_downloads_capture_root_bundle_before_running(
    tmp_path: Path,
) -> None:
    source_capture_root = _build_capture_root(tmp_path / "source")
    _write_robot_eval_cards(source_capture_root)
    _write_fixture_attempts(source_capture_root, success=True)
    request = _full_job_request(source_capture_root)
    request["job_id"] = "worker-capture-bundle-job"
    bundle_path = tmp_path / "capture-root.zip"
    parts = source_capture_root.parts
    archive_root = Path(*parts[: parts.index("scenes") - 1])
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for source in sorted(source_capture_root.rglob("*")):
            if source.is_file():
                archive.write(source, source.relative_to(archive_root))
    manifest_path = tmp_path / "worker_manifest.json"
    artifact_output_dir = tmp_path / "worker_artifacts"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root_bundle_uri": str(bundle_path),
            "job_id": "worker-capture-bundle-job",
            "provisioner": "fixture_local",
            "simulator": "fixture",
            "job_request": request,
            "artifact_output_uri": str(artifact_output_dir),
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    extracted_capture_root = (
        tmp_path
        / "worker"
        / "capture_root_bundle"
        / "local-blueprint"
        / "scenes"
        / "scene-1"
        / "captures"
        / "capture-1"
    )
    job_dir = (
        extracted_capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "worker-capture-bundle-job"
    )
    bundle_manifest = _read_json(job_dir / "capture_root_bundle_manifest.json")

    assert runtime["status"] == "completed"
    assert runtime["capture_root"] == str(extracted_capture_root)
    assert runtime["capture_root_bundle_uri"] == str(bundle_path)
    assert runtime["capture_root_bundle"]["status"] == "extracted"  # type: ignore[index]
    assert bundle_manifest["capture_root"] == str(extracted_capture_root)
    assert (job_dir / "job_run_manifest.json").is_file()
    assert (artifact_output_dir / "capture_root_bundle_manifest.json").is_file()
    assert (artifact_output_dir / "worker_runtime_manifest.json").is_file()


def test_robot_eval_worker_blocks_without_capture_root(tmp_path: Path) -> None:
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "job_id": "missing-capture-root",
            "job_request": {"schema_version": "robot_eval_job_request.v1"},
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert "missing_capture_root" in runtime["blockers"]
    assert (tmp_path / "worker" / "worker_runtime_manifest.json").is_file()


def test_robot_eval_worker_blocks_live_provider_without_artifact_output_uri(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-missing-artifact-output",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "worker-missing-artifact-output",
            },
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert runtime["provisioner"] == "runpod"
    assert runtime["artifact_output_uri_required"] is True
    assert "missing_artifact_output_uri" in runtime["blockers"]
    assert runtime["artifact_upload"]["status"] == "blocked"  # type: ignore[index]
    assert not (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "worker-missing-artifact-output"
    ).exists()


def test_robot_eval_worker_blocks_live_provider_non_writable_artifact_output_uri(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    artifact_output_uri = (tmp_path / "worker_artifacts").as_uri()
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-non-writable-artifact-output",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "artifact_output_uri": artifact_output_uri,
            "artifact_output_uri_required": True,
            "artifact_output_uri_scheme": "file",
            "artifact_output_uri_provider_writable": False,
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "worker-non-writable-artifact-output",
            },
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert runtime["provisioner"] == "runpod"
    assert runtime["artifact_output_uri_required"] is True
    assert runtime["artifact_output_uri"] == artifact_output_uri
    assert runtime["artifact_output_uri_scheme"] == "file"
    assert runtime["artifact_output_uri_provider_writable"] is False
    assert runtime["blockers"] == ["artifact_output_uri_not_provider_writable"]
    assert runtime["artifact_upload"]["status"] == "blocked"  # type: ignore[index]
    assert runtime["artifact_upload"]["reason"] == (  # type: ignore[index]
        "artifact_output_uri_not_provider_writable"
    )
    assert not (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "worker-non-writable-artifact-output"
    ).exists()


def test_robot_eval_worker_blocks_live_provider_missing_artifact_write_auth_contract(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-missing-artifact-write-auth",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "artifact_output_uri": "r2://blueprint-artifacts/jobs/worker-missing-auth",
            "artifact_output_uri_required": True,
            "artifact_output_uri_scheme": "r2",
            "artifact_output_uri_provider_writable": True,
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "worker-missing-artifact-write-auth",
            },
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert runtime["artifact_output_uri_required"] is True
    assert runtime["artifact_output_uri_scheme"] == "r2"
    assert runtime["artifact_output_uri_provider_writable"] is True
    assert runtime["artifact_output_write_auth_contract_ready"] is False
    assert runtime["blockers"] == ["artifact_output_write_auth_contract_missing"]
    assert runtime["artifact_upload"]["status"] == "blocked"  # type: ignore[index]
    assert runtime["artifact_upload"]["reason"] == (  # type: ignore[index]
        "artifact_output_write_auth_contract_missing"
    )
    assert not (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "worker-missing-artifact-write-auth"
    ).exists()


def test_robot_eval_worker_blocks_live_provider_without_worker_manifest_schema(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_job_request.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-invalid-schema",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "artifact_output_uri": str(tmp_path / "worker_artifacts"),
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert runtime["expected_worker_manifest_schema"] == "robot_eval_worker_manifest.v1"
    assert runtime["actual_worker_manifest_schema"] == "robot_eval_job_request.v1"
    assert runtime["blockers"] == ["invalid_or_missing_worker_manifest_schema"]
    assert not (
        capture_root / "pipeline" / "robot_eval_jobs" / "worker-invalid-schema"
    ).exists()


def test_robot_eval_worker_parser_defaults_artifact_output_uri_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_EVAL_MANIFEST_URI", "gs://blueprint/manifests/worker.json")
    monkeypatch.setenv("BLUEPRINT_ARTIFACT_OUTPUT_URI", "gs://blueprint/artifacts/job")

    args = _build_parser().parse_args([])

    assert args.manifest == "gs://blueprint/manifests/worker.json"
    assert args.artifact_output_uri == "gs://blueprint/artifacts/job"


def test_robot_eval_worker_blocks_live_provider_without_embedded_job_request(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-missing-job-request",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "artifact_output_uri": str(tmp_path / "worker_artifacts"),
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert runtime["expected_worker_manifest_schema"] == "robot_eval_worker_manifest.v1"
    assert runtime["blockers"] == ["missing_worker_manifest_job_request"]
    assert not (
        capture_root / "pipeline" / "robot_eval_jobs" / "worker-missing-job-request"
    ).exists()


def test_robot_eval_worker_blocks_live_provider_without_runtime_preflight_contract(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-missing-runtime-preflight",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "artifact_output_uri": str(tmp_path / "worker_artifacts"),
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "worker-missing-runtime-preflight",
            },
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert runtime["blockers"] == ["invalid_worker_runtime_preflight_contract"]
    assert "runtime_preflight_required_checks_missing" in runtime[
        "runtime_preflight_contract_blockers"
    ]
    assert "runtime_preflight_missing_check:vulkan_device" in runtime[
        "runtime_preflight_contract_blockers"
    ]
    assert runtime["simulator_execution_proven"] is False
    assert not (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "worker-missing-runtime-preflight"
    ).exists()


def test_robot_eval_worker_blocks_simulator_execution_without_runtime_preflight_command(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    artifact_output_dir = tmp_path / "worker_artifacts"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-missing-runtime-preflight-command",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "artifact_output_uri": str(artifact_output_dir),
            "runtime_preflight_contract": {
                "required_before_scene_load": True,
                "worker_blocks_scene_load_on_failed_preflight": True,
                "run_before": "scene_load_and_policy_execution",
                "result_artifact": "worker_runtime_preflight.json",
                "runtime_preflight_is_not_simulator_proof": True,
                "vulkan_required": True,
                "test_frame_render_required": True,
                "required_checks": [
                    "nvidia_smi_gpu_inventory",
                    "driver_version",
                    "vulkan_device",
                    "rtx_renderer_available",
                    "isaac_headless_launch",
                    "blank_scene_load",
                    "test_frame_render",
                ],
            },
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "worker-missing-runtime-preflight-command",
            },
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
        capture_root=capture_root,
        allow_simulator_execution=True,
    )

    preflight = _read_json(tmp_path / "worker" / "worker_runtime_preflight.json")
    assert runtime["status"] == "blocked"
    assert runtime["blockers"] == ["worker_runtime_preflight_blocked"]
    assert runtime["runtime_preflight_status"] == "blocked"
    assert runtime["runtime_preflight_blockers"] == ["missing_runtime_preflight_command"]
    assert preflight["status"] == "blocked"
    assert preflight["blockers"] == ["missing_runtime_preflight_command"]
    assert (artifact_output_dir / "worker_runtime_preflight.json").is_file()
    assert (artifact_output_dir / "worker_runtime_manifest.json").is_file()
    assert not (
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "worker-missing-runtime-preflight-command"
    ).exists()


def test_robot_eval_worker_signed_put_runs_on_missing_runtime_preflight_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    signed_url_signature_param = "x-goog-" + "signature="
    signed_capture_bundle_url = (
        "https://storage.example/capture-root.zip?"
        f"{signed_url_signature_param}capture-secret-signature&x-goog-date=20260612"
    )
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "capture_root_bundle_uri": signed_capture_bundle_url,
            "job_id": "worker-signed-put-runtime-preflight-blocked",
            "provisioner": "runpod",
            "simulator": "mujoco",
            "artifact_output_uri_required": False,
            "runtime_preflight_contract": {
                "required_before_scene_load": True,
                "worker_blocks_scene_load_on_failed_preflight": True,
                "run_before": "scene_load_and_policy_execution",
                "result_artifact": "worker_runtime_preflight.json",
                "runtime_preflight_is_not_simulator_proof": True,
                "nvidia_smi_required": False,
                "egl_required_when_rendering": True,
                "blank_scene_or_model_load_required": True,
                "test_frame_render_required": False,
                "required_checks": [
                    "python_import_mujoco",
                    "headless_context_selection",
                    "egl_context_when_rendering",
                    "blank_model_or_scene_load",
                    "short_rollout_smoke",
                ],
            },
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "worker-signed-put-runtime-preflight-blocked",
            },
        },
    )
    monkeypatch.setenv(
            "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL",
            (
                "https://storage.example/signed-runtime-manifest?"
                f"{signed_url_signature_param}runtime-put-secret-signature&x-goog-date=20260612"
            ),
        )
    uploads: list[dict[str, object]] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b""

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        uploads.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "body": json.loads(request.data.decode("utf-8")),
                "headers": dict(request.header_items()),
            }
        )
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
        capture_root=capture_root,
        allow_simulator_execution=True,
    )

    assert runtime["status"] == "blocked"
    assert runtime["blockers"] == ["worker_runtime_preflight_blocked"]
    assert runtime["runtime_preflight_blockers"] == ["missing_runtime_preflight_command"]
    assert len(uploads) >= 2
    assert uploads[0]["method"] == "PUT"
    assert uploads[0]["url"] == (
        "https://storage.example/signed-runtime-manifest?"
        f"{signed_url_signature_param}runtime-put-secret-signature&x-goog-date=20260612"
    )
    assert "capture-secret-signature" not in json.dumps(uploads[-1]["body"])
    assert "runtime-put-secret-signature" not in json.dumps(uploads[-1]["body"])
    assert uploads[-1]["body"]["signed_put_runtime_manifest_upload"]["status"] == "completed"  # type: ignore[index]
    persisted = _read_json(tmp_path / "worker" / "worker_runtime_manifest.json")
    persisted_text = json.dumps(persisted)
    assert "capture-secret-signature" not in persisted_text
    assert "runtime-put-secret-signature" not in persisted_text
    assert persisted["capture_root_bundle_uri"] == (
        "https://storage.example/capture-root.zip?"
        f"{signed_url_signature_param}<redacted:signed-url-signature>&x-goog-date=20260612"
    )
    assert persisted["signed_put_runtime_manifest_upload"]["status"] == "completed"
    assert persisted["signed_put_runtime_manifest_upload"]["signed_url_stored"] is False


def test_robot_eval_worker_copies_runtime_preflight_logs_to_artifact_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request = _full_job_request(capture_root)
    request["job_id"] = "worker-runtime-preflight-logs"
    manifest_path = tmp_path / "worker_manifest.json"
    artifact_output_dir = tmp_path / "worker_artifacts"
    monkeypatch.setenv("RUNPOD_API_KEY", "runtime-preflight-secret-value")
    monkeypatch.setenv("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME", "true")
    monkeypatch.setenv(
        "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF",
        "registry.example/blueprint/isaac-eval-worker:2026-06-12",
    )
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "worker-runtime-preflight-logs",
            "provisioner": "runpod",
            "simulator": "isaac_sim",
            "robot": "unitree_g1",
            "secret_env_var_names": ["RUNPOD_API_KEY"],
            "artifact_output_uri": str(artifact_output_dir),
            "runtime_preflight_contract": {
                "required_before_scene_load": True,
                "worker_blocks_scene_load_on_failed_preflight": True,
                "run_before": "scene_load_and_policy_execution",
                "result_artifact": "worker_runtime_preflight.json",
                "runtime_preflight_is_not_simulator_proof": True,
                "vulkan_required": True,
                "test_frame_render_required": True,
                "required_checks": [
                    "nvidia_smi_gpu_inventory",
                    "driver_version",
                    "vulkan_device",
                    "rtx_renderer_available",
                    "isaac_headless_launch",
                    "blank_scene_load",
                    "test_frame_render",
                ],
            },
            "runtime_preflight_command": (
                f"{sys.executable} -c \"import os, sys; "
                "print(os.environ['BLUEPRINT_SIMULATOR_FRAMEWORK']); "
                "print(os.environ['RUNPOD_API_KEY']); "
                "print(os.environ['RUNPOD_API_KEY'], file=sys.stderr)\""
            ),
            "job_request": request,
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
        allow_simulator_execution=True,
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / (
        "worker-runtime-preflight-logs"
    )
    preflight = _read_json(job_dir / "worker_runtime_preflight.json")
    assert runtime["status"] == "blocked"
    assert runtime["robot"] == "unitree_g1"
    assert runtime["runtime_preflight_status"] == "passed"
    assert preflight["status"] == "passed"
    assert preflight["capture_root"] == str(capture_root)
    assert preflight["secret_values_in_artifact"] is False
    assert preflight["stdout_stderr_secret_redaction_enabled"] is True
    assert "RUNPOD_API_KEY" in preflight["redacted_secret_env_var_names"]
    assert preflight["redacted_secret_value_count"] >= 1
    assert "raw_command" not in preflight
    assert preflight["command"]["raw_command_stored"] is False
    assert preflight["simulator_execution_proven"] is False
    stdout = (job_dir / "worker_runtime_preflight.stdout.log").read_text(
        encoding="utf-8"
    )
    stderr = (job_dir / "worker_runtime_preflight.stderr.log").read_text(
        encoding="utf-8"
    )
    assert "isaac_sim" in stdout
    assert "<redacted:RUNPOD_API_KEY>" in stdout
    assert "<redacted:RUNPOD_API_KEY>" in stderr
    assert "runtime-preflight-secret-value" not in stdout
    assert "runtime-preflight-secret-value" not in stderr
    assert (job_dir / "worker_runtime_preflight.stderr.log").is_file()
    assert (artifact_output_dir / "worker_runtime_preflight.json").is_file()
    assert (artifact_output_dir / "worker_runtime_preflight.stdout.log").is_file()
    assert (artifact_output_dir / "worker_runtime_preflight.stderr.log").is_file()
    assert (artifact_output_dir / "startup_architecture_audit.json").is_file()
    assert (artifact_output_dir / "job_run_manifest.json").is_file()
    assert (artifact_output_dir / "worker_runtime_manifest.json").is_file()
    refreshed_audit = _read_json(job_dir / "startup_architecture_audit.json")
    refreshed_run_manifest = _read_json(job_dir / "job_run_manifest.json")
    cost_ledger = _read_json(job_dir / "gpu_cost_control_ledger.json")
    finalizer_proof = _read_json(job_dir / "provider_runtime_finalizer_proof.json")
    remote_closure = _read_json(job_dir / "remote_cloud_execution_closure_manifest.json")
    assert runtime["startup_architecture_blockers"] == refreshed_audit["blockers"]
    assert runtime["provider_runtime_accounting"]["status"] == "recorded"
    assert cost_ledger["status"] == "provider_runtime_observed"
    assert cost_ledger["live_provider_calls_performed"] is True
    assert cost_ledger["gpu_time"]["actual_gpu_time_record_present"] is True  # type: ignore[index]
    assert cost_ledger["gpu_time"]["actual_gpu_time_source"] == (  # type: ignore[index]
        "worker_runtime_wall_clock_seconds"
    )
    assert cost_ledger["artifact_finalizer"][  # type: ignore[index]
        "worker_artifacts_finalized_before_shutdown"
    ] is True
    assert cost_ledger["artifact_finalizer"][  # type: ignore[index]
        "provider_shutdown_proven"
    ] is False
    assert finalizer_proof["worker_artifacts_finalized_before_shutdown"] is True
    assert finalizer_proof["provider_shutdown_proven"] is False
    assert "provider_shutdown_evidence_missing" in finalizer_proof["blockers"]
    assert remote_closure["clean_shutdown_proven"] is False
    assert "remote_provider_shutdown_not_proven" in remote_closure["runtime_blockers"]
    assert refreshed_run_manifest["remote_cloud_clean_shutdown_proven"] is False
    assert refreshed_run_manifest["artifacts"]["provider_runtime_finalizer_proof"] == (
        "provider_runtime_finalizer_proof.json"
    )
    assert (artifact_output_dir / "gpu_cost_control_ledger.json").is_file()
    assert (artifact_output_dir / "provider_runtime_finalizer_proof.json").is_file()
    assert (artifact_output_dir / "remote_cloud_execution_closure_manifest.json").is_file()
    assert "cost:gpu_time_recorded_or_blocked" not in refreshed_audit["blockers"]
    redaction_check = next(
        check
        for check in refreshed_audit["checks"]
        if check["id"] == "worker_runtime:preflight_command_redacted"
    )
    assert redaction_check["status"] == "passed"
    assert "worker_runtime:preflight_command_redacted" not in refreshed_audit["blockers"]
    assert refreshed_run_manifest["startup_architecture_audit_status"] == refreshed_audit[
        "status"
    ]
    assert (
        refreshed_run_manifest["artifacts"]["worker_runtime_manifest"]
        == "worker_runtime_manifest.json"
    )
    assert (
        refreshed_run_manifest["artifacts"]["startup_architecture_audit"]
        == "startup_architecture_audit.json"
    )


def test_robot_eval_worker_can_require_artifact_output_for_fixture_jobs(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    manifest_path = tmp_path / "worker_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "robot_eval_worker_manifest.v1",
            "capture_root": str(capture_root),
            "job_id": "fixture-missing-artifact-output",
            "provisioner": "fixture_local",
            "simulator": "fixture",
            "artifact_output_uri_required": True,
            "job_request": {
                "schema_version": "robot_eval_job_request.v1",
                "job_id": "fixture-missing-artifact-output",
            },
        },
    )

    runtime = run_robot_eval_worker(
        manifest_uri=str(manifest_path),
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "blocked"
    assert runtime["provisioner"] == "fixture_local"
    assert runtime["artifact_output_uri_required"] is True
    assert runtime["blockers"] == ["missing_artifact_output_uri"]


def test_robot_eval_worker_uses_s3_manifest_and_r2_artifact_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request = _full_job_request(capture_root)
    request["job_id"] = "worker-object-storage-job"
    worker_manifest = {
        "schema_version": "robot_eval_worker_manifest.v1",
        "capture_root": str(capture_root),
        "job_id": "worker-object-storage-job",
        "provisioner": "fixture_local",
        "simulator": "fixture",
        "job_request": request,
        "artifact_output_uri": "r2://blueprint-artifacts/jobs/worker-object-storage-job",
    }
    uploaded_keys: list[str] = []
    client_kwargs: list[dict[str, object]] = []

    class FakeS3Client:
        def download_file(self, bucket: str, key: str, target: str) -> None:
            assert bucket == "blueprint-manifests"
            assert key == "worker/job.json"
            Path(target).write_text(json.dumps(worker_manifest), encoding="utf-8")

        def upload_file(self, source: str, bucket: str, key: str) -> None:
            assert Path(source).is_file()
            assert bucket == "blueprint-artifacts"
            uploaded_keys.append(key)

    fake_client = FakeS3Client()

    def fake_client_factory(service_name: str, **kwargs: object) -> FakeS3Client:
        assert service_name == "s3"
        client_kwargs.append(dict(kwargs))
        return fake_client

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_client_factory))
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://example-r2.invalid")

    runtime = run_robot_eval_worker(
        manifest_uri="s3://blueprint-manifests/worker/job.json",
        work_dir=tmp_path / "worker",
    )

    assert runtime["status"] == "completed"
    assert runtime["artifact_upload"]["status"] == "completed"  # type: ignore[index]
    assert runtime["artifact_upload"]["worker_runtime_manifest_included"] is True  # type: ignore[index]
    assert runtime["artifact_upload"]["storage_scheme"] == "r2"  # type: ignore[index]
    assert runtime["artifact_upload"]["s3_compatible_endpoint_configured"] is True  # type: ignore[index]
    assert "jobs/worker-object-storage-job/job_run_manifest.json" in uploaded_keys
    assert "jobs/worker-object-storage-job/startup_architecture_audit.json" in uploaded_keys
    assert "jobs/worker-object-storage-job/worker_runtime_preflight.json" in uploaded_keys
    assert "jobs/worker-object-storage-job/worker_runtime_manifest.json" in uploaded_keys
    assert "jobs/worker-object-storage-job/worker_manifest.json" in uploaded_keys
    assert "jobs/worker-object-storage-job/worker_launch_plan.json" in uploaded_keys
    assert "jobs/worker-object-storage-job/gpu_provider_launch_request.json" in uploaded_keys
    assert "jobs/worker-object-storage-job/gpu_cost_control_ledger.json" in uploaded_keys
    assert any(call.get("endpoint_url") == "https://example-r2.invalid" for call in client_kwargs)


def test_robot_eval_job_fixture_failure_records_failed_attempt_without_overclaiming(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=False)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-fixture-failure",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-fixture-failure"
    evaluation = _read_json(job_dir / "evaluation_result.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    labels = _read_json(job_dir / "failure_labels.json")
    proof_boundary = _read_json(job_dir / "proof_boundary.json")

    assert evaluation["status"] == "completed_with_failures"
    assert trace["attempts"][0]["success"] is False
    assert labels["labels"][0]["failure_mode_ids"] == ["failure_navigation_blocked"]
    assert proof_boundary["robot_readiness_proven"] is False


def test_robot_eval_job_generated_scenarios_stay_review_required_until_accepted(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root, scenario_variation_label="agent_inferred")
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-review-required",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-review-required"
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")

    assert simulator_result["status"] == "blocked"
    assert trace["status"] == "blocked"
    assert "generated_or_inferred_scenarios_require_review" in blocked["blockers"]


def test_command_simulator_blocks_without_env_gate(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-missing-env",
        provisioner="fixture_local",
        simulator="mujoco",
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": f"{sys.executable} -c \"print('sim ok')\""},
    )

    result = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-env"
        / "simulator_service_result.json"
    )
    provider_adapter = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-env"
        / "simulator_provider_adapter_manifest.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"]
    assert result["artifact_paths"]["simulator_provider_adapter_manifest"] == (
        "simulator_provider_adapter_manifest.json"
    )
    assert provider_adapter["status"] == "blocked"
    assert provider_adapter["provider_profile"]["provider_family"] == "cpu_physics_engine"
    assert provider_adapter["gates"] == {
        "env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION": False,
        "allow_simulator_execution_flag": True,
        "simulator_allowlisted": True,
        "command_configured": True,
        "blockers": ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"],
    }
    assert provider_adapter["command_ref"]["configured"] is True


def test_isaac_lab_arena_simulator_surfaces_packet_and_blocks_without_env_gate(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-arena-missing-env",
        provisioner="fixture_local",
        simulator="isaac_lab_arena",
        allow_simulator_execution=True,
        allowed_simulators=["isaac_lab_arena"],
        simulator_commands={"isaac_lab_arena": f"{sys.executable} -c \"print('arena sim ok')\""},
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-arena-missing-env"
    request = _read_json(job_dir / "simulator_service_request.json")
    result = _read_json(job_dir / "simulator_service_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    arena_packet = _read_json(
        capture_root / "pipeline" / "simulation_automation" / "arena_environment_packet.json"
    )

    assert request["framework"] == "isaac_lab_arena"
    assert request["arena_environment_packet_path"] == (
        "../simulation_automation/arena_environment_packet.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"]
    assert run_manifest["cpu_preflight_artifacts"]["arena_environment_packet"] == (
        "../simulation_automation/arena_environment_packet.json"
    )
    assert arena_packet["backend"] == "isaac_lab_arena"
    assert arena_packet["simulator_execution_proven"] is False
    assert run_manifest["simulator_execution_proven"] is False
    assert run_manifest["robot_readiness_proven"] is False


def test_command_simulator_blocks_without_cli_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-missing-cli",
        provisioner="fixture_local",
        simulator="mujoco",
        allowed_simulators=["mujoco"],
        simulator_commands={"mujoco": f"{sys.executable} -c \"print('sim ok')\""},
    )

    result = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-cli"
        / "simulator_service_result.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_cli_allow_simulator_execution"]


def test_command_simulator_blocks_without_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-missing-command",
        provisioner="fixture_local",
        simulator="mujoco",
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
    )

    result = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-command-missing-command"
        / "simulator_service_result.json"
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["missing_simulator_command_mujoco"]


def test_command_simulator_records_stdout_stderr_and_exit_code_when_allowed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_SIMULATOR_EXECUTION", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-command-allowed",
        provisioner="fixture_local",
        simulator="mujoco",
        allow_simulator_execution=True,
        allowed_simulators=["mujoco"],
        simulator_commands={
            "mujoco": (
                f'{sys.executable} -c "import sys; '
                "print('sim stdout'); print('sim stderr', file=sys.stderr)\""
            )
        },
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-command-allowed"
    result = _read_json(job_dir / "simulator_service_result.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert "sim stdout" in result["stdout"]
    assert "sim stderr" in result["stderr"]
    assert run_manifest["public_claim_upgrade_allowed"] is False


def test_training_request_is_export_only_and_training_result_blocks_without_gates(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root, operation="train_then_evaluate"))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-training-blocked",
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-training-blocked"
    request = _read_json(job_dir / "training_request.json")
    result = _read_json(job_dir / "training_result.json")
    blocked = _read_json(job_dir / "blocked_manifest.json")

    assert request["status"] == "export_manifest_only"
    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "missing_env_BLUEPRINT_ALLOW_COSMOS_TRAINING",
        "missing_cli_allow_training",
        "missing_training_command",
    ]
    assert "training_blocked" in blocked["blockers"]


def test_fake_and_agents_sdk_agent_adapters_write_advisory_plans(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-fake-agent",
        agent_adapter=FakeRobotEvalJobAgentAdapter(),
        provisioner="fixture_local",
        simulator="fixture",
    )
    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-agents-sdk-blocked",
        agent_adapter=AgentsSdkRobotEvalJobAdapter(
            agents_sdk_available=False,
            openai_api_key="",
            live_env_allowed=False,
            allow_live_operator=False,
        ),
        provisioner="fixture_local",
        simulator="fixture",
    )

    fake_plan = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-fake-agent"
        / "agent_orchestration_plan.json"
    )
    agents_plan = _read_json(
        capture_root
        / "pipeline"
        / "robot_eval_jobs"
        / "job-agents-sdk-blocked"
        / "agent_orchestration_plan.json"
    )

    assert fake_plan["status"] == "completed"
    assert fake_plan["decisions"][0]["next_command"] == "validate_job_request"
    assert agents_plan["status"] == "blocked"
    assert agents_plan["blockers"] == [
        "missing_openai_agents_sdk",
        "missing_openai_api_key",
        "missing_cli_allow_live_agent_operator",
        "missing_env_BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS",
    ]
    assert agents_plan["agent_authority"] == "live_operator_when_gated"
    assert agents_plan["proof_booleans_mutable_by_agent"] is False


def test_agents_sdk_robot_eval_live_operator_logs_decisions_without_proof_upgrade(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))

    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-agents-sdk-live",
        agent_adapter=AgentsSdkRobotEvalJobAdapter(
            agents_sdk_available=True,
            openai_api_key="sk-test",
            live_env_allowed=True,
            allow_live_operator=True,
            executor=lambda _prompt, _context: {
                "final_output": "Validation passed; run fixture evaluator, then summarize.",
                "commands_chosen": ["run_fixture_evaluation"],
                "tool_call_summaries": [
                    {"tool_name": "read_manifest", "summary": "checked job_validation.json"}
                ],
                "decisions": [
                    {
                        "decision": "run_fixture_evaluation",
                        "summary": "Deterministic validation already passed.",
                    }
                ],
            },
        ),
        provisioner="fixture_local",
        simulator="fixture",
    )

    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-agents-sdk-live"
    agents_plan = _read_json(job_dir / "agent_orchestration_plan.json")
    run_manifest = _read_json(job_dir / "job_run_manifest.json")

    assert agents_plan["status"] == "operator_completed"
    assert agents_plan["execution_performed"] is True
    assert agents_plan["operator_mode"] == "live_operator"
    assert agents_plan["operator_ledger"]["commands_chosen"] == [
        "choose_next_deterministic_robot_eval_command",
        "run_fixture_evaluation",
    ]
    assert agents_plan["operator_ledger"]["tool_call_summaries"][0]["tool_name"] == (
        "read_manifest"
    )
    assert agents_plan["proof_effect"]["direct_proof_booleans_set_true"] == []
    assert run_manifest["agent_operator_mode"] == "live_operator"
    assert run_manifest["robot_readiness_proven"] is False


def test_evaluation_prep_surfaces_robot_eval_job_artifacts_without_overclaiming(
    tmp_path: Path,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    _write_fixture_attempts(capture_root, success=True)
    request_path = tmp_path / "job-request.json"
    _write_json(request_path, _full_job_request(capture_root))
    build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-surfaced",
        provisioner="fixture_local",
        simulator="fixture",
    )
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-surfaced"
    _write_json(
        job_dir / "gpu_provider_launcher_result.json",
        {
            "schema_version": "robot_eval_provider_launcher_result.v1",
            "status": "dry_run",
            "live_provider_calls_performed": False,
            "simulator_execution_proven": False,
        },
    )
    (job_dir / "gpu_provider_launcher.stdout.log").write_text(
        "provider launcher dry run\n",
        encoding="utf-8",
    )
    (job_dir / "gpu_provider_launcher.stderr.log").write_text("", encoding="utf-8")
    _write_json(
        job_dir / "runpod_provider_adapter_result.json",
        {
            "schema_version": "runpod_provider_adapter_result.v1",
            "status": "dry_run",
            "live_runpod_api_call_performed": False,
            "simulator_execution_proven": False,
        },
    )
    _write_json(
        job_dir / "worker_runtime_manifest.json",
        {
            "schema_version": "robot_eval_worker_runtime_manifest.v1",
            "status": "blocked",
            "simulator_execution_proven": False,
        },
    )
    _write_json(
        job_dir / "worker_runtime_preflight.json",
        {
            "schema_version": "robot_eval_worker_runtime_preflight.v1",
            "status": "blocked",
            "simulator_execution_proven": False,
        },
    )
    (job_dir / "worker_runtime_preflight.stdout.log").write_text(
        "runtime preflight not executed\n",
        encoding="utf-8",
    )
    (job_dir / "worker_runtime_preflight.stderr.log").write_text("", encoding="utf-8")

    eval_dir = capture_root / "pipeline" / "evaluation_prep"
    eval_dir.mkdir(parents=True, exist_ok=True)
    surface = robot_eval_job_evaluation_prep_surface(
        capture_root=capture_root,
        eval_dir=eval_dir,
    )

    assert surface["schema_version"] == "robot_eval_job_evaluation_prep_surface.v1"
    assert surface["status"] == "fixture_evaluation_completed"
    assert surface["public_claim_upgrade_allowed"] is False
    assert surface["simulator_execution_proven"] is False
    assert surface["robot_readiness_proven"] is False
    assert surface["artifacts"]["robot_eval_job_job-surfaced_run_manifest"] == (
        "../robot_eval_jobs/job-surfaced/job_run_manifest.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_scheduler_decision"] == (
        "../robot_eval_jobs/job-surfaced/scheduler_decision.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_worker_launch_plan"] == (
        "../robot_eval_jobs/job-surfaced/worker_launch_plan.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_worker_manifest"] == (
        "../robot_eval_jobs/job-surfaced/worker_manifest.json"
    )
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_gpu_provider_launch_request"
    ] == ("../robot_eval_jobs/job-surfaced/gpu_provider_launch_request.json")
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_gpu_provider_launcher_result"
    ] == ("../robot_eval_jobs/job-surfaced/gpu_provider_launcher_result.json")
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_gpu_provider_launcher_stdout_log"
    ] == ("../robot_eval_jobs/job-surfaced/gpu_provider_launcher.stdout.log")
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_gpu_provider_launcher_stderr_log"
    ] == ("../robot_eval_jobs/job-surfaced/gpu_provider_launcher.stderr.log")
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_runpod_provider_adapter_result"
    ] == ("../robot_eval_jobs/job-surfaced/runpod_provider_adapter_result.json")
    assert surface["artifacts"]["robot_eval_job_job-surfaced_gpu_cost_control_ledger"] == (
        "../robot_eval_jobs/job-surfaced/gpu_cost_control_ledger.json"
    )
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_remote_cloud_execution_closure_manifest"
    ] == ("../robot_eval_jobs/job-surfaced/remote_cloud_execution_closure_manifest.json")
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_robot_team_grade_eval_closure_manifest"
    ] == ("../robot_eval_jobs/job-surfaced/robot_team_grade_eval_closure_manifest.json")
    assert surface["artifacts"]["robot_eval_job_job-surfaced_gpu_provisioning_result"] == (
        "../robot_eval_jobs/job-surfaced/gpu_provisioning_result.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_startup_architecture_audit"] == (
        "../robot_eval_jobs/job-surfaced/startup_architecture_audit.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_worker_runtime_manifest"] == (
        "../robot_eval_jobs/job-surfaced/worker_runtime_manifest.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_worker_runtime_preflight"] == (
        "../robot_eval_jobs/job-surfaced/worker_runtime_preflight.json"
    )
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_worker_runtime_preflight_stdout_log"
    ] == ("../robot_eval_jobs/job-surfaced/worker_runtime_preflight.stdout.log")
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_worker_runtime_preflight_stderr_log"
    ] == ("../robot_eval_jobs/job-surfaced/worker_runtime_preflight.stderr.log")
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_robot_pov_frame_sequence_manifest"
    ] == ("../robot_eval_jobs/job-surfaced/robot_pov_frame_sequence_manifest.json")
    assert surface["artifacts"]["robot_eval_job_job-surfaced_scenario_eval_matrix"] == (
        "../robot_eval_jobs/job-surfaced/scenario_eval_matrix.json"
    )
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_deployment_outcome_intake_manifest"
    ] == ("../robot_eval_jobs/job-surfaced/deployment_outcome_intake_manifest.json")
    assert surface["artifacts"]["robot_eval_job_job-surfaced_live_eval_closure_manifest"] == (
        "../robot_eval_jobs/job-surfaced/live_eval_closure_manifest.json"
    )
    assert surface["artifacts"][
        "robot_eval_job_job-surfaced_real_world_validation_followup_plan"
    ] == ("../robot_eval_jobs/job-surfaced/real_world_validation_followup_plan.json")
    assert (
        surface["artifacts"][
            "robot_eval_job_job-surfaced_real_world_validation_followup_request_queue"
        ]
        == "../robot_eval_jobs/job-surfaced/real_world_validation_followup_request_queue.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_robot_eval_report"] == (
        "../robot_eval_jobs/job-surfaced/robot_eval_report.json"
    )
    assert surface["artifacts"]["robot_eval_job_job-surfaced_robot_eval_report_markdown"] == (
        "../robot_eval_jobs/job-surfaced/robot_eval_report.md"
    )
    assert (
        surface["artifacts"][
            "robot_eval_job_job-surfaced_post_training_data_package_export_manifest"
        ]
        == "../robot_eval_jobs/job-surfaced/post_training_data_package_export_manifest.json"
    )
    assert (
        surface["artifacts"]["robot_eval_job_job-surfaced_webapp_robot_eval_status_projection"]
        == "../robot_eval_jobs/job-surfaced/webapp_robot_eval_status_projection.json"
    )
    assert surface["artifact_uris"]["robot_eval_job_job-surfaced_run_manifest_uri"].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/job_run_manifest.json"
    )
    assert surface["artifact_uris"][
        "robot_eval_job_job-surfaced_gpu_provider_launch_request_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/gpu_provider_launch_request.json")
    assert surface["artifact_uris"][
        "robot_eval_job_job-surfaced_gpu_provider_launcher_result_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/gpu_provider_launcher_result.json")
    assert surface["artifact_uris"][
        "robot_eval_job_job-surfaced_runpod_provider_adapter_result_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/runpod_provider_adapter_result.json")
    assert surface["artifact_uris"][
        "robot_eval_job_job-surfaced_gpu_cost_control_ledger_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/gpu_cost_control_ledger.json")
    assert surface["artifact_uris"][
        "robot_eval_job_startup_architecture_audit_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/startup_architecture_audit.json")
    assert surface["artifact_uris"][
        "robot_eval_job_runpod_provider_adapter_result_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/runpod_provider_adapter_result.json")
    assert surface["artifact_uris"][
        "robot_eval_job_worker_runtime_preflight_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/worker_runtime_preflight.json")
    webapp_startup_aliases = {
        "robot_eval_scheduler_decision_uri": "scheduler_decision.json",
        "robot_eval_worker_launch_plan_uri": "worker_launch_plan.json",
        "robot_eval_worker_manifest_uri": "worker_manifest.json",
        "robot_eval_gpu_provider_launch_request_uri": "gpu_provider_launch_request.json",
        "robot_eval_gpu_provider_launcher_result_uri": "gpu_provider_launcher_result.json",
        "robot_eval_runpod_provider_adapter_result_uri": "runpod_provider_adapter_result.json",
        "robot_eval_gpu_cost_control_ledger_uri": "gpu_cost_control_ledger.json",
        "robot_eval_startup_architecture_audit_uri": "startup_architecture_audit.json",
        "robot_eval_worker_runtime_manifest_uri": "worker_runtime_manifest.json",
        "robot_eval_worker_runtime_preflight_uri": "worker_runtime_preflight.json",
    }
    for alias_key, filename in webapp_startup_aliases.items():
        assert surface["artifact_uris"][alias_key].endswith(
            f"/pipeline/robot_eval_jobs/job-surfaced/{filename}"
        )
    assert surface["artifact_uris"]["robot_eval_job_evaluation_result_uri"].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/evaluation_result.json"
    )
    assert surface["artifact_uris"]["robot_eval_job_scenario_eval_matrix_uri"].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/scenario_eval_matrix.json"
    )
    assert surface["artifact_uris"][
        "robot_eval_job_post_training_data_package_export_manifest_uri"
    ].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/post_training_data_package_export_manifest.json"
    )
    assert surface["artifact_uris"][
        "robot_eval_job_webapp_robot_eval_status_projection_uri"
    ].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/webapp_robot_eval_status_projection.json"
    )
    assert surface["artifact_uris"][
        "robot_eval_job_remote_cloud_execution_closure_manifest_uri"
    ].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/remote_cloud_execution_closure_manifest.json"
    )
    assert surface["artifact_uris"][
        "robot_eval_job_robot_team_grade_eval_closure_manifest_uri"
    ].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/robot_team_grade_eval_closure_manifest.json"
    )
    assert surface["artifact_uris"]["robot_eval_job_live_eval_closure_manifest_uri"].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/live_eval_closure_manifest.json"
    )
    assert surface["artifact_uris"][
        "robot_eval_job_real_world_validation_followup_plan_uri"
    ].endswith("/pipeline/robot_eval_jobs/job-surfaced/real_world_validation_followup_plan.json")
    assert surface["artifact_uris"][
        "robot_eval_job_real_world_validation_followup_request_queue_uri"
    ].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/real_world_validation_followup_request_queue.json"
    )
    assert surface["artifact_uris"]["robot_eval_job_robot_eval_report_uri"].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/robot_eval_report.json"
    )
    assert surface["artifact_uris"]["robot_eval_job_robot_eval_report_markdown_uri"].endswith(
        "/pipeline/robot_eval_jobs/job-surfaced/robot_eval_report.md"
    )


def _write_arena_rollout_results(results_dir: Path, *, count: int = 500) -> None:
    video_dir = results_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "episode.mp4").write_bytes(b"fake arena video bytes")
    (results_dir / "stdout.txt").write_text("arena rollout completed\n", encoding="utf-8")
    (results_dir / "stderr.txt").write_text("", encoding="utf-8")
    episodes = []
    for index in range(count):
        success = (index + 1) % 10 != 0
        episodes.append(
            {
                "episode_id": f"episode-{index + 1:04d}",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "scenario_run_id": f"scenario_place_return_in_bin_mobile__arena_run_{index + 1:04d}",
                "task_id": "place_return_in_bin",
                "shard_id": f"arena_shard_{(index // 125) + 1:04d}",
                "status": "completed" if success else "failed",
                "success": success,
                "failure_reason": None if success else "threshold_miss_timeout",
                "metrics": {
                    "cycle_time_seconds": 12.0 + (index % 7),
                    "placement_accuracy": 1.0 if success else 0.2,
                },
                "start_time_seconds": float(index),
                "end_time_seconds": float(index + 1),
                "video_path": "videos/episode.mp4",
                "stdout_path": "stdout.txt",
                "stderr_path": "stderr.txt",
            }
        )
    _write_json(
        results_dir / "rollout_manifest.json",
        {
            "schema_version": "isaac_lab_arena_rollout_manifest.fixture.v1",
            "episodes": episodes,
        },
    )
    _write_json(
        results_dir / "review_resolutions.json",
        {
            "schema_version": "arena_review_resolutions.fixture.v1",
            "resolutions": [
                {
                    "label_id": "label_arena_attempt_0010",
                    "decision": "accepted",
                    "reviewer": "fixture-reviewer",
                    "evidence_uri": "review://accepted/0010",
                }
            ],
        },
    )


def test_isaac_lab_arena_results_feed_eval_package_and_delivery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS", "true")
    capture_root = _build_capture_root(tmp_path)
    _write_robot_eval_cards(capture_root)
    results_dir = tmp_path / "arena-results"
    _write_arena_rollout_results(results_dir)
    request_path = tmp_path / "job-request.json"
    request = _full_job_request(capture_root)
    request["simulator_preference"] = "isaac_lab_arena"
    _write_json(request_path, request)

    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request_path,
        job_id="job-arena-ingest",
        provisioner="fixture_local",
        simulator="isaac_lab_arena",
        arena_results_dir=results_dir,
        arena_scenario_count=500,
        arena_shard_size=125,
        arena_num_envs=32,
        arena_retry_budget=3,
        arena_operator_mode="fake",
    )

    job_dir = Path(result["job_dir"])
    run_manifest = _read_json(job_dir / "job_run_manifest.json")
    simulator_result = _read_json(job_dir / "simulator_service_result.json")
    eval_result = _read_json(job_dir / "evaluation_result.json")
    schedule = _read_json(job_dir / "arena_eval_schedule.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    labels = _read_json(job_dir / "failure_labels.json")
    clips = _read_json(job_dir / "clips_manifest.json")
    package = _read_json(job_dir / "post_training_data_package_export_manifest.json")
    archive = _read_json(job_dir / "archive_manifest.json")
    delivery = _read_json(job_dir / "delivery_manifest.json")
    operators = _read_json(job_dir / "live_operator_ledger.json")

    assert result["status"] == "completed_with_failures"
    assert simulator_result["status"] == "completed_from_supplied_arena_results"
    assert simulator_result["simulator_execution_proven"] is False
    assert eval_result["status"] == "completed_with_failures"
    assert run_manifest["arena_result_ingest_status"] == "completed"
    assert run_manifest["simulator_execution_proven"] is False
    assert run_manifest["robot_readiness_proven"] is False

    assert schedule["scenario_count"] == 500
    assert schedule["shard_count"] == 4
    assert schedule["num_envs"] == 32
    assert trace["attempt_count"] == 500
    assert labels["label_count"] == 50
    assert clips["clip_count"] == 500

    assert (job_dir / "arena_eval_metrics.json").is_file()
    assert (job_dir / "dataset_card.json").is_file()
    assert (job_dir / "license_manifest.json").is_file()
    assert (job_dir / "checksums.json").is_file()
    assert package["status"] == "export_ready_review_required"
    assert package["archive_manifest_path"] == "archive_manifest.json"
    assert archive["archive"]["exists"] is True
    assert delivery["status"] == "local_delivery_bundle_ready"
    assert operators["status"] == "completed"
    assert operators["agents_sdk_operator_performed"] is True
    assert operators["codex_sdk_operator_performed"] is True
    assert operators["public_claim_upgrade_allowed"] is False
