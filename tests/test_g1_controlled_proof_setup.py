from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.g1_controlled_proof_setup import (
    DEFAULT_POLICY_ID,
    DEFAULT_ROBOT_MAKE_MODEL,
    DEFAULT_ROBOT_PROFILE_ID,
    G1_CONTROLLED_PROOF_SETUP_SCHEMA_VERSION,
    OFFICIAL_UNITREE_G1_POLICY_CANDIDATE_SCHEMA_VERSION,
    UNITREE_MUJOCO_MAIN_REF,
    UNITREE_RL_GYM_MAIN_REF,
    UNITREE_RL_LAB_MAIN_REF,
    build_g1_controlled_proof_setup,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_job_request(capture_root: Path, job_id: str) -> Path:
    path = capture_root / "pipeline" / "robot_eval_jobs" / job_id / "job_request.json"
    _write_json(
        path,
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "buyer_request_id": "buyer-123",
            "requested_tasks": [
                {
                    "task_id": "walk_to_target",
                    "scenario_ids": ["site-a_walk_to_target_pose"],
                }
            ],
            "site_package": {
                "site_slug": "site-a",
                "site_submission_id": "site-submission-123",
                "buyer_request_id": "buyer-123",
                "capture_job_id": "capture-job-123",
                "capture_id": "capture-123",
            },
            "source": {
                "selection_state": {
                    "task_id": "walk_to_target",
                    "scenario_id": "site-a_walk_to_target_pose",
                }
            },
        },
    )
    _write_json(
        capture_root / "pipeline" / "robot_eval_jobs" / job_id / "gpu_provider_launch_request.json",
        {
            "schema_version": "robot_eval_gpu_provider_launch_request.v1",
            "status": "blocked_provider_input_setup",
            "provider": "runpod",
        },
    )
    return path


def test_g1_setup_writes_unitree_default_artifacts_without_secrets(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    job_id = "robot-eval-test"
    job_request_path = _seed_job_request(capture_root, job_id)

    manifest = build_g1_controlled_proof_setup(capture_root=capture_root, job_id=job_id)

    assert manifest["schema_version"] == G1_CONTROLLED_PROOF_SETUP_SCHEMA_VERSION
    assert manifest["status"] == "setup_ready_external_operator_inputs_required"
    assert manifest["default_robot"]["make_model"] == DEFAULT_ROBOT_MAKE_MODEL  # type: ignore[index]
    assert manifest["default_robot"]["robot_profile_id"] == DEFAULT_ROBOT_PROFILE_ID  # type: ignore[index]
    assert manifest["job_context"]["job_request_path"] == str(job_request_path)  # type: ignore[index]
    artifacts = manifest["artifacts"]  # type: ignore[assignment]
    setup_manifest_path = Path(artifacts["setup_manifest"])  # type: ignore[index]
    assert setup_manifest_path.is_file()
    policy = _read_json(Path(artifacts["robot_team_policy_package"]))  # type: ignore[index]
    assert policy["policy_id"] == DEFAULT_POLICY_ID
    assert "https://github.com/unitreerobotics/unitree_rl_gym" in json.dumps(policy)
    assert "https://github.com/unitreerobotics/unitree_rl_lab" in json.dumps(policy)
    assert "https://github.com/unitreerobotics/unitree_mujoco" in json.dumps(policy)
    assert UNITREE_RL_GYM_MAIN_REF in json.dumps(policy)
    assert UNITREE_RL_LAB_MAIN_REF in json.dumps(policy)
    assert UNITREE_MUJOCO_MAIN_REF in json.dumps(policy)
    candidate = _read_json(Path(artifacts["official_g1_policy_candidate"]))  # type: ignore[index]
    assert candidate["schema_version"] == OFFICIAL_UNITREE_G1_POLICY_CANDIDATE_SCHEMA_VERSION
    assert candidate["status"] == "candidate_selected_execution_required"
    assert candidate["robot_profile_id"] == DEFAULT_ROBOT_PROFILE_ID
    assert candidate["proof_boundary"]["robot_team_policy_performance_proven"] is False  # type: ignore[index]
    selected = candidate["candidate_package"]["selected_default"]  # type: ignore[index]
    assert selected["source_ref"] == UNITREE_RL_GYM_MAIN_REF
    assert selected["physical_deploy_config"] == "deploy/deploy_real/configs/g1.yaml"
    assert "deploy_real.py" in selected["physical_deploy_command_template"]
    assert "https://github.com/unitreerobotics/unitree_rl_lab" in json.dumps(candidate)
    assert "https://github.com/unitreerobotics/unitree_rl_gym" in json.dumps(candidate)
    assert "https://github.com/unitreerobotics/unitree_mujoco" in json.dumps(candidate)
    assert UNITREE_RL_GYM_MAIN_REF in json.dumps(candidate)
    assert "deploy/deploy_real/README.md" in json.dumps(candidate)
    assert "<" not in json.dumps(candidate)
    pov = _read_json(Path(artifacts["real_robot_pov_manifest"]))  # type: ignore[index]
    assert pov["schema_version"] == "real_robot_pov_manifest.v1"
    assert pov["records"][0]["robot_camera_video_uri"] == "<physical-g1-pov-camera-video-uri>"  # type: ignore[index]
    runpod_plan = _read_json(Path(artifacts["runpod_low_cost_launch_plan"]))  # type: ignore[index]
    assert runpod_plan["preferred_gpu_type_id"] == "NVIDIA RTX A4000"
    assert runpod_plan["preferred_serverless_gpu_tier"]["gpu_type_ids"] == [  # type: ignore[index]
        "NVIDIA RTX A4000",
        "NVIDIA RTX A4500",
        "NVIDIA RTX 4000 Ada Generation",
    ]
    assert runpod_plan["preferred_serverless_gpu_tier"]["reference_cost_per_second_usd"] == 0.00016  # type: ignore[index]
    assert runpod_plan["expected_max_pod_compute_cost_usd_reference"] < 0.02
    assert runpod_plan["expected_max_serverless_compute_cost_usd_reference"] < 0.03
    assert runpod_plan["max_budget_usd"] == 2.0
    assemble_script = Path(artifacts["assemble_script"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "python -m blueprint_pipeline.g1_controlled_run_evidence" in assemble_script
    assert Path(artifacts["evidence_input_template"]).parent.is_dir()  # type: ignore[index]
    field_kit_path = Path(artifacts["field_run_capture_kit"])  # type: ignore[index]
    assert field_kit_path.is_file()
    field_kit = _read_json(field_kit_path)
    assert field_kit["default_robot"]["make_model"] == DEFAULT_ROBOT_MAKE_MODEL  # type: ignore[index]
    capture_script = Path(field_kit["artifacts"]["capture_script"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "BLUEPRINT_ALLOW_G1_PHYSICAL_RUN=true" in capture_script
    assert "BLUEPRINT_G1_CAMERA_SOURCE" in capture_script
    assert "BLUEPRINT_G1_POLICY_COMMAND" in capture_script
    stage_script = Path(artifacts["stage_script"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "blueprint-intake-live-pipeline-inputs" in stage_script
    assert "BLUEPRINT_ALLOW_STAGING_G1_CONTROLLED_RUN_INPUTS=true" in stage_script
    runpod_script = Path(artifacts["runpod_script"]).read_text(encoding="utf-8")  # type: ignore[index]
    assert "RUNPOD_API_KEY" in runpod_script
    assert "blueprint-run-runpod-provider-adapter" in runpod_script
    assert "rpa_" not in json.dumps(manifest)
    assert "rpa_" not in assemble_script
    assert "rpa_" not in stage_script
    assert "rpa_" not in runpod_script
    assert manifest["proof_boundary"]["physical_robot_readiness_proven"] is False  # type: ignore[index]


def test_g1_setup_cli_writes_manifest(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"

    exit_code = main(["--capture-root", str(capture_root)])

    assert exit_code == 0
    assert (
        capture_root
        / "pipeline"
        / "g1_controlled_proof_setup"
        / "g1_controlled_proof_setup_manifest.json"
    ).is_file()
