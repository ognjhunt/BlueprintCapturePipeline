from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline import wam_fixture_evaluator as wam_fixture_module
from blueprint_pipeline.wam_fixture_evaluator import main, run_wam_eval_job


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_job(tmp_path: Path) -> tuple[Path, Path]:
    capture_root = tmp_path / "capture-root"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-wam-fixture"
    job_dir.mkdir(parents=True)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "status": "completed",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "scenario_variation_instance_id": "variation-clearance",
                    "task_id": "tote_transfer",
                    "scenario_id": "blocked_path",
                    "variation_name": "blocked_path",
                    "split": "train",
                },
                {
                    "scenario_eval_run_id": "run_heldout_grasp_glare",
                    "scenario_variation_instance_id": "variation-grasp-glare",
                    "task_id": "tote_transfer",
                    "scenario_id": "glare_grasp",
                    "variation_name": "glare_grasp",
                    "split": "heldout",
                },
            ],
        },
    )
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-wam-fixture",
            "evaluation_substrate": "fixture_wam",
            "policy_candidates": [
                {
                    "policy_id": "baseline_policy",
                    "capabilities": ["clearance_aware_navigation"],
                },
                {
                    "policy_id": "site_finetune_policy",
                    "capabilities": [
                        "clearance_aware_navigation",
                        "visual_recheck",
                        "grasp_alignment_correction",
                    ],
                },
            ],
        },
    )
    _write_json(
        job_dir / "policy_package_manifest.json",
        {
            "schema_version": "robot_eval_policy_package_manifest.v1",
            "status": "configured",
        },
    )
    return capture_root, job_dir


def test_fixture_wam_eval_job_writes_rollouts_labels_scorecard_and_boundaries(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    _write_json(
        job_dir / "deployment_outcome_ledger.json",
        {
            "records": [
                {
                    "scenarioEvalRunId": "run_heldout_grasp_glare",
                    "policyId": "site_finetune_policy",
                    "hardwareId": "unitree-g1",
                    "success": True,
                    "operatorAttestation": "operator-reviewed-rollout",
                },
                {
                    "id": "missing-owner-evidence",
                    "scenario_eval_run_id": "run_train_clearance",
                    "policy_id": "baseline_policy",
                    "hardware_id": "unitree-g1",
                },
            ]
        },
    )

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    expected = {
        "evaluation_substrate_registry.json",
        "wam_evaluation_request.json",
        "wam_rollout_manifest.json",
        "wam_rollout_results.json",
        "vision_success_labels.json",
        "normalized_attempt_trace.json",
        "failure_labels.json",
        "policy_ranking_scorecard.json",
        "wam_eval_claim_boundary.json",
        "real_world_validation_followup_request.json",
        "srcc_validation_plan.json",
        "customer_handoff_report.json",
        "customer_handoff_report.md",
    }
    assert expected.issubset({path.name for path in job_dir.iterdir()})

    rollouts = _read_json(job_dir / "wam_rollout_results.json")
    labels = _read_json(job_dir / "vision_success_labels.json")
    trace = _read_json(job_dir / "normalized_attempt_trace.json")
    scorecard = _read_json(job_dir / "policy_ranking_scorecard.json")
    claim_boundary = _read_json(job_dir / "wam_eval_claim_boundary.json")
    srcc_plan = _read_json(job_dir / "srcc_validation_plan.json")
    anchor_manifest = _read_json(job_dir / "wam_real_world_validation_anchor_manifest.json")

    assert rollouts["rollout_count"] == 4
    assert labels["label_count"] == 4
    assert trace["attempt_count"] == 4
    assert scorecard["top_policy_id"] == "site_finetune_policy"
    assert scorecard["policy_count"] == 2
    assert claim_boundary["generated_rollouts_are_model_derived_support_artifacts"] is True
    assert claim_boundary["live_provider_calls_performed"] is False
    assert claim_boundary["customer_specific_srcc_claimed"] is False
    assert claim_boundary["passing_wam_heldout_eval_is_not_deployment_approval"] is True
    assert srcc_plan["status"] == "requires_real_world_rollout_anchors"
    assert srcc_plan["customer_specific_srcc_claimed"] is False
    assert anchor_manifest["deployment_outcome_ledger_path"] == "deployment_outcome_ledger.json"
    assert anchor_manifest["usable_anchor_count"] == 1
    assert anchor_manifest["missing_or_incomplete_anchor_count"] == 1
    assert anchor_manifest["anchors"][0]["actual_success"] is True
    assert anchor_manifest["missing_anchor_requirements"][0]["record_id"] == "missing-owner-evidence"


def test_fixture_wam_cli_and_live_provider_blocked_manifest(tmp_path: Path) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--job-dir",
                str(job_dir),
                "--evaluation-substrate",
                "fixture_wam",
            ]
        )
        == 0
    )

    blocked = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert blocked["status"] == "blocked"
    assert "cosmos3_wam_provider_adapter_not_configured_for_local_run" in blocked["blockers"]
    blocked_request = _read_json(job_dir / "wam_evaluation_request.json")
    blocked_boundary = _read_json(job_dir / "wam_eval_claim_boundary.json")
    assert blocked_request["status"] == "blocked"
    assert blocked_request["evaluation_substrate"] == "cosmos3_wam"
    assert blocked_boundary["live_provider_calls_performed"] is False

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--job-dir",
                str(job_dir),
                "--evaluation-substrate",
                "mujoco",
            ]
        )
        == 1
    )


def test_fixture_wam_blocks_missing_scenario_matrix(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture-root"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-wam-missing"
    job_dir.mkdir(parents=True)

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "scenario_eval_matrix_missing_or_empty" in result["blockers"]
    request = _read_json(job_dir / "wam_evaluation_request.json")
    assert request["status"] == "blocked"


def test_fixture_wam_uses_wam_request_policy_fields_and_redacts_references(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    _write_json(
        job_dir / "job_request.json",
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "job-wam-fixture",
            "wamEvaluation": {
                "policies": [
                    {
                        "name": "universal_policy",
                        "checkpoint": "ckpt-001",
                        "capabilities": ["all"],
                        "api_key": "must-not-persist",
                    }
                ]
            },
            "policyPackage": {"actionInterface": "joint_position_targets"},
            "robotProfile": {"robotModel": "unitree-g1"},
            "taskRequest": {"taskFamily": "warehouse_tote_transfer"},
            "classicalSimCrossChecks": ["isaac_sim", "mujoco", "unknown"],
        },
    )

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    policy_binding = _read_json(job_dir / "wam_policy_interface_binding.json")
    scorecard = _read_json(job_dir / "policy_ranking_scorecard.json")
    validation = _read_json(job_dir / "wam_customer_validation_envelope.json")
    cross_check = _read_json(job_dir / "wam_classical_sim_cross_check_plan.json")
    assert policy_binding["policies"][0]["policy_id"] == "policy_candidate_01"
    assert policy_binding["policies"][0]["checkpoint_id"] == "ckpt-001"
    assert policy_binding["policies"][0]["reference"]["api_key"] == "<redacted>"
    assert policy_binding["action_interface"] == "joint_position_targets"
    assert scorecard["top_policy_id"] == "policy_candidate_01"
    assert validation["hardware_id"] == "unitree-g1"
    assert validation["task_family"] == "warehouse_tote_transfer"
    assert cross_check["recommended_cross_checks"] == [
        "classical_sim_isaac",
        "classical_sim_mujoco",
    ]


def test_live_wam_provider_command_normalizes_rollouts_and_upload_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    script = tmp_path / "provider_fixture.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "payload = {",
                "    'wam_rollout_results': {",
                "        'rollouts': [",
                "            None,",
                "            {",
                "                'policyId': 'provider-policy',",
                "                'success': True,",
                "                'uncertainty_score': '0.21',",
                "                'failure_mode_ids': 'fixture_failure',",
                "                'ood_flags': 'glare',",
                "                'metrics': {'latency_ms': 12},",
                "                'claim_boundary': {'provider_reported': True},",
                "            },",
                "        ]",
                "    },",
                "    'artifact_upload_evidence': {",
                "        'upload_complete': True,",
                "        'api_key': 'secret-value',",
                "    },",
                "}",
                "with open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8') as fh:",
                "    json.dump(payload, fh)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", "true")
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_API_KEY", "test-auth")
    monkeypatch.setenv(
        "BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND",
        f"{sys.executable} {script}",
    )

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        allow_live_provider=True,
        artifact_output_uri="gs://bucket/wam-output",
        budget_usd=12.5,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    assert result["claim_boundary"]["live_provider_calls_performed"] is True
    provider_execution = _read_json(job_dir / "wam_provider_execution_manifest.json")
    provider_upload = _read_json(job_dir / "wam_provider_artifact_upload_proof.json")
    rollouts = _read_json(job_dir / "wam_rollout_results.json")
    assert provider_execution["status"] == "completed"
    assert provider_execution["provider_command_used"] is True
    assert provider_execution["attempt_count"] == 1
    assert provider_execution["detail"]["normalized_rollout_count"] == 1
    assert provider_upload["status"] == "upload_proven"
    assert provider_upload["evidence"]["api_key"] == "<redacted>"
    assert rollouts["rollouts"][0]["scenario_eval_run_id"] == "scenario_eval_run_0002"
    assert rollouts["rollouts"][0]["policy_id"] == "provider-policy"
    assert rollouts["rollouts"][0]["failure_mode_ids"] == ["fixture_failure"]
    assert rollouts["rollouts"][0]["ood_flags"] == ["glare"]


def test_live_wam_provider_requires_explicit_and_env_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    monkeypatch.delenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", raising=False)
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_API_KEY", "test-auth")
    command = f"{sys.executable} -c 'print(1)'"

    missing_env = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        allow_live_provider=True,
        provider_command=command,
        generated_at="2026-06-20T00:00:00+00:00",
    )
    assert missing_env["status"] == "blocked"
    assert "BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER_not_enabled" in missing_env["blockers"]

    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", "true")
    missing_explicit = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        provider_command=command,
        generated_at="2026-06-20T00:00:01+00:00",
    )
    assert missing_explicit["status"] == "blocked"
    assert "allow_live_wam_provider_not_enabled" in missing_explicit["blockers"]


def test_live_wam_provider_command_blocks_when_output_has_no_rollouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    script = tmp_path / "provider_empty.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "with open(os.environ['BLUEPRINT_WAM_PROVIDER_OUTPUT'], 'w', encoding='utf-8') as fh:",
                "    json.dump({'rollouts': []}, fh)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER", "true")
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_API_KEY", "test-auth")

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="cosmos3_wam",
        allow_live_provider=True,
        provider_command=f"{sys.executable} {script}",
        max_retries=1,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "wam_provider_output_missing_rollouts" in result["blockers"]
    provider_execution = _read_json(job_dir / "wam_provider_execution_manifest.json")
    assert provider_execution["status"] == "blocked"
    assert provider_execution["attempt_count"] == 2


def test_fixture_wam_helper_edges_cover_forced_failure_and_generated_run_ids(
    tmp_path: Path,
) -> None:
    assert wam_fixture_module._string_list("") == []
    assert wam_fixture_module._number(True, default=None) is None

    runs = wam_fixture_module._matrix_runs(
        {
            "runs": [
                "not-a-mapping",
                {
                    "task_id": "tote_transfer",
                    "scenario_id": "human_crossing",
                    "variation_name": "human_crossing",
                },
            ]
        }
    )
    assert runs[0]["scenario_eval_run_id"] == "scenario_eval_run_0002"

    rollout = wam_fixture_module._rollout_for_run(
        job_dir=tmp_path,
        substrate="fixture_wam",
        policy={
            "policy_id": "forced_failure_policy",
            "capabilities": ["all"],
            "fixture_success_profile": {
                "fail_scenario_eval_run_ids": ["scenario_eval_run_0002"],
            },
        },
        run=runs[0],
        index=1,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert rollout["predicted_success"] is False
    assert "fixture_forced_failure" in rollout["ood_flags"]
    assert "fixture_policy_failure" in rollout["failure_mode_ids"]
    assert "dynamic_agent_safety_failure" in rollout["failure_mode_ids"]


def test_fixture_wam_blocks_when_policy_candidate_resolution_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _fixture_job(tmp_path)
    monkeypatch.setattr(wam_fixture_module, "_policy_candidates", lambda **_: [])

    result = run_wam_eval_job(
        capture_root=capture_root,
        job_dir=job_dir,
        evaluation_substrate="fixture_wam",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "policy_candidates_missing" in result["blockers"]
