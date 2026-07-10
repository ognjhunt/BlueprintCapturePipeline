import hashlib
import json
import os
import sys
from pathlib import Path

from blueprint_pipeline.policy_autoresearch import run_policy_autoresearch
from blueprint_pipeline import policy_autoresearch as autoresearch_module
from blueprint_pipeline.policy_autoresearch_mujoco_evaluator import (
    run_mujoco_policy_evaluator,
)
from blueprint_pipeline.policy_autoresearch_owner_gpu_evaluator import (
    run_owner_gpu_policy_evaluator,
)

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _job_dir(tmp_path: Path) -> tuple[Path, Path]:
    capture_root = tmp_path / "capture-root"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job-policy-autoresearch"
    job_dir.mkdir(parents=True)
    return capture_root, job_dir


def _seed_recipe(path: Path, extra: dict | None = None) -> Path:
    payload = {
        "schema_version": "policy_autoresearch_recipe.v1",
        "policy_id": "site_policy_seed",
        "policy_kind": "code_as_policy_navigation_heuristic",
        "mutable_parameters": {
            "planner": "direct",
            "clearance_margin_m": 0.05,
            "dynamic_obstacle_yield": False,
            "perception_vote_count": 1,
            "retry_budget": 0,
            "max_speed_mps": 0.9,
            "grasp_alignment_correction": False,
        },
    }
    if extra:
        payload.update(extra)
    recipe_path = path / "seed_policy_recipe.json"
    _write_json(recipe_path, payload)
    return recipe_path


def _group_artifacts(path: Path, group_values: dict[str, str]) -> dict[str, dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    for field, group_id in group_values.items():
        key = hashlib.sha256(f"{field}:{group_id}".encode()).hexdigest()[:12]
        source = path / "group-artifacts" / f"{field}-{key}.source.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps(
                {
                    "schema_version": "test_group_source.v1",
                    "axis": field,
                    "group_id": group_id,
                    "source_payload": f"frozen-source-for:{field}:{group_id}",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        artifact = path / "group-artifacts" / f"{field}-{key}.manifest.json"
        _write_json(
            artifact,
            {
                "schema_version": "policy_autoresearch_group_artifact.v1",
                "axis": field,
                "group_id": group_id,
                "source_content_sha256": source_sha256,
                "source_artifact": {
                    "path": str(source),
                    "sha256": source_sha256,
                },
            },
        )
        artifacts[field] = {
            "path": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    return artifacts


def test_policy_autoresearch_promotes_perfect_heldout_candidate_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    train_clearance_group_artifacts = _group_artifacts(
        tmp_path,
        {
            "site_id": "site-train-clearance",
            "scene_id": "scene-train-clearance",
            "task_id": "navigate_to_target_train",
            "object_id": "target-clearance",
            "source_trajectory_id": "trajectory-train-clearance",
            "policy_lineage_id": "lineage-train",
        },
    )
    train_dynamic_group_artifacts = _group_artifacts(
        tmp_path,
        {
            "site_id": "site-train-dynamic",
            "scene_id": "scene-train-dynamic",
            "task_id": "navigate_to_target_train",
            "object_id": "target-dynamic",
            "source_trajectory_id": "trajectory-train-dynamic",
            "policy_lineage_id": "lineage-train",
        },
    )
    train_visual_group_artifacts = _group_artifacts(
        tmp_path,
        {
            "site_id": "site-train-visual",
            "scene_id": "scene-train-visual",
            "task_id": "navigate_to_target_train",
            "object_id": "target-visual",
            "source_trajectory_id": "trajectory-train-visual",
            "policy_lineage_id": "lineage-train",
        },
    )
    dev_group_artifacts = _group_artifacts(
        tmp_path,
        {
            "site_id": "site-dev",
            "scene_id": "scene-dev",
            "task_id": "navigate_to_target_dev",
            "object_id": "target-dev",
            "source_trajectory_id": "trajectory-dev",
            "policy_lineage_id": "lineage-dev",
        },
    )
    locked_group_artifacts = _group_artifacts(
        tmp_path,
        {
            "site_id": "site-locked-test",
            "scene_id": "scene-locked-test",
            "task_id": "navigate_to_target_locked_test",
            "object_id": "target-locked-test",
            "source_trajectory_id": "trajectory-locked-test",
            "policy_lineage_id": "lineage-locked-test",
        },
    )
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "scenario_variation_instance_id": "variation_clearance",
                    "task_id": "navigate_to_target_train",
                    "scenario_id": "blocked_path",
                    "scene_id": "scene-train-clearance",
                    "site_id": "site-train-clearance",
                    "object_id": "target-clearance",
                    "source_trajectory_id": "trajectory-train-clearance",
                    "policy_lineage_id": "lineage-train",
                    "group_artifacts": train_clearance_group_artifacts,
                    "variation_name": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
                {
                    "scenario_eval_run_id": "run_train_dynamic",
                    "scenario_variation_instance_id": "variation_dynamic",
                    "task_id": "navigate_to_target_train",
                    "scenario_id": "human_crossing",
                    "scene_id": "scene-train-dynamic",
                    "site_id": "site-train-dynamic",
                    "object_id": "target-dynamic",
                    "source_trajectory_id": "trajectory-train-dynamic",
                    "policy_lineage_id": "lineage-train",
                    "group_artifacts": train_dynamic_group_artifacts,
                    "variation_name": "human_crossing",
                    "split": "train",
                    "required_policy_capabilities": ["dynamic_obstacle_yield"],
                },
                {
                    "scenario_eval_run_id": "run_train_visual",
                    "scenario_variation_instance_id": "variation_visual",
                    "task_id": "navigate_to_target_train",
                    "scenario_id": "occluded_target",
                    "scene_id": "scene-train-visual",
                    "site_id": "site-train-visual",
                    "object_id": "target-visual",
                    "source_trajectory_id": "trajectory-train-visual",
                    "policy_lineage_id": "lineage-train",
                    "group_artifacts": train_visual_group_artifacts,
                    "variation_name": "occlusion_glare",
                    "split": "train",
                    "required_policy_capabilities": ["visual_recheck"],
                },
                {
                    "scenario_eval_run_id": "run_heldout_combined",
                    "scenario_variation_instance_id": "variation_heldout",
                    "task_id": "navigate_to_target_dev",
                    "scenario_id": "site_eval_combined_holdout",
                    "scene_id": "scene-dev",
                    "site_id": "site-dev",
                    "object_id": "target-dev",
                    "source_trajectory_id": "trajectory-dev",
                    "policy_lineage_id": "lineage-dev",
                    "group_artifacts": dev_group_artifacts,
                    "variation_name": "blocked_path_human_crossing_occlusion",
                    "split": "heldout",
                    "required_policy_capabilities": [
                        "clearance_aware_navigation",
                        "dynamic_obstacle_yield",
                        "visual_recheck",
                    ],
                },
                {
                    "scenario_eval_run_id": "run_locked_test_combined",
                    "scenario_variation_instance_id": "variation_locked_test",
                    "task_id": "navigate_to_target_locked_test",
                    "scenario_id": "site_eval_combined_locked_test",
                    "scene_id": "scene-locked-test",
                    "site_id": "site-locked-test",
                    "object_id": "target-locked-test",
                    "source_trajectory_id": "trajectory-locked-test",
                    "policy_lineage_id": "lineage-locked-test",
                    "group_artifacts": locked_group_artifacts,
                    "variation_name": "blocked_path_human_crossing_occlusion",
                    "split": "locked_test",
                    "required_policy_capabilities": [
                        "clearance_aware_navigation",
                        "dynamic_obstacle_yield",
                        "visual_recheck",
                    ],
                },
            ],
        },
    )

    reviewed_examples_path = tmp_path / "reviewed_examples.json"
    _write_json(
        reviewed_examples_path,
        {
            "schema_version": "policy_autoresearch_reviewed_examples.v1",
            "examples": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "label": "failure",
                    "reviewer_notes": "Seed policy clips the blocked route.",
                },
                {
                    "scenario_eval_run_id": "run_heldout_combined",
                    "label": "success",
                    "reviewer_notes": "Candidate must clear blocked route, dynamic obstacle, and glare.",
                },
            ],
        },
    )
    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        reviewed_examples_path=reviewed_examples_path,
        max_iterations=6,
        agent_count=3,
        target_success_rate=1.0,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["status"] == "promoted"
    assert result["report"]["target_success_reached"] is True
    assert result["report"]["baseline_heldout_success_rate"] == 0.0
    assert result["report"]["best_heldout_success_rate"] == 1.0
    assert result["heldout_eval_result"]["safety_contact_gate_passed"] is True
    assert result["heldout_eval_result"]["task_success_summary"]["task_success_rate"] == 1.0
    assert result["report"]["split_design_validation"]["status"] == "validated"
    assert result["report"]["locked_test_eval_count"] == 1
    assert result["locked_test_eval_result"]["phase"] == "locked_test"
    assert result["locked_test_eval_result"]["task_success_rate"] == 1.0
    assert all(
        "locked_test_success_rate" not in record for record in result["report"]["iteration_records"]
    )

    expected_artifacts = {
        "policy_autoresearch_report": "policy_autoresearch_report.json",
        "agent_idea_tree": "agent_idea_tree.json",
        "policy_candidate_package": "policy_candidate_package.json",
        "heldout_eval_result": "heldout_eval_result.json",
        "locked_test_eval_result": "locked_test_eval_result.json",
        "followup_real_world_validation_request": "followup_real_world_validation_request.json",
        "budget_ledger": "budget_ledger.json",
    }
    assert result["report"]["artifact_paths"] == expected_artifacts
    for relative_path in expected_artifacts.values():
        assert (job_dir / "policy_autoresearch" / relative_path).is_file()

    package = json.loads(
        (job_dir / "policy_autoresearch" / "policy_candidate_package.json").read_text(
            encoding="utf-8"
        )
    )
    assert package["status"] == "promoted_sim_only_policy_candidate"
    assert package["frozen_verifier_sha256"] == result["verifier_manifest"]["verifier_sha256"]
    assert result["verifier_manifest"]["reviewed_examples_path"] == str(
        reviewed_examples_path.resolve()
    )
    assert (
        result["verifier_manifest"]["reviewed_examples_payload"]["examples"][0]["label"]
        == "failure"
    )
    assert package["rank_fidelity_result_proven"] is False
    assert package["public_claim_upgrade_allowed"] is False
    assert result["budget_ledger"]["usage"]["candidate_evaluations"] >= 1
    assert any(
        event["event"] == "parallel_branch_batch_started"
        for event in result["budget_ledger"]["events"]
    )
    assert any(
        record["parallel_branch"] is True for record in result["report"]["iteration_records"]
    )
    assert set(package["recipe"]["derived_policy_capabilities"]) == {
        "clearance_aware_navigation",
        "dynamic_obstacle_yield",
        "visual_recheck",
    }

    rerun = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        reviewed_examples_path=reviewed_examples_path,
        output_dir=job_dir / "different-output-dir",
        max_iterations=6,
        agent_count=3,
        target_success_rate=1.0,
        generated_at="2026-06-18T00:01:00+00:00",
    )
    assert rerun["report"]["status"] == "completed_no_promotion"
    assert rerun["report"]["locked_test_eval_count"] == 0
    assert "locked_test_already_consumed_for_this_verifier" in rerun["report"]["blockers"]

    copied_job_dir = job_dir.parent / "job-policy-autoresearch-copy"
    copied_job_dir.mkdir(parents=True)
    (copied_job_dir / "scenario_eval_matrix.json").write_bytes(
        (job_dir / "scenario_eval_matrix.json").read_bytes()
    )
    copied_job_rerun = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=copied_job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        reviewed_examples_path=reviewed_examples_path,
        max_iterations=6,
        agent_count=3,
        target_success_rate=1.0,
        generated_at="2026-06-18T00:02:00+00:00",
    )
    assert (
        copied_job_rerun["verifier_manifest"]["verifier_sha256"]
        == result["verifier_manifest"]["verifier_sha256"]
    )
    assert copied_job_rerun["report"]["locked_test_eval_count"] == 0
    assert (
        "locked_test_already_consumed_for_this_verifier" in copied_job_rerun["report"]["blockers"]
    )

    changed_threshold_rerun = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        reviewed_examples_path=reviewed_examples_path,
        output_dir=job_dir / "changed-threshold-output",
        max_iterations=6,
        agent_count=3,
        target_success_rate=0.5,
        generated_at="2026-06-18T00:03:00+00:00",
    )
    assert (
        changed_threshold_rerun["verifier_manifest"]["verifier_sha256"]
        != result["verifier_manifest"]["verifier_sha256"]
    )
    assert (
        changed_threshold_rerun["verifier_manifest"]["locked_test_identity_sha256"]
        == result["verifier_manifest"]["locked_test_identity_sha256"]
    )
    assert changed_threshold_rerun["report"]["locked_test_eval_count"] == 0
    assert (
        "locked_test_already_consumed_for_this_verifier"
        in (changed_threshold_rerun["report"]["blockers"])
    )


def test_policy_autoresearch_blocks_policy_recipe_that_tries_to_move_goalposts(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train",
                    "task_id": "navigate_to_target",
                    "scenario_id": "base",
                    "split": "train",
                },
                {
                    "scenario_eval_run_id": "run_heldout",
                    "task_id": "navigate_to_target",
                    "scenario_id": "base_holdout",
                    "split": "heldout",
                },
            ],
        },
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(
            tmp_path,
            {
                "reward_function": "return 1.0",
                "verifier_override": {"target_success_rate": 0.0},
            },
        ),
        max_iterations=2,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["status"] == "blocked"
    assert result["report"]["target_success_reached"] is False
    assert "forbidden_policy_recipe_keys" in result["report"]["blockers"]
    assert result["policy_candidate_package"]["status"] == "blocked"


def test_policy_autoresearch_does_not_promote_train_only_improvement(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path",
                    "variation_name": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
                {
                    "scenario_eval_run_id": "run_heldout_dynamic",
                    "task_id": "navigate_to_target",
                    "scenario_id": "human_crossing",
                    "variation_name": "human_crossing",
                    "split": "heldout",
                    "required_policy_capabilities": ["dynamic_obstacle_yield"],
                },
            ],
        },
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        max_iterations=1,
        agent_count=1,
        target_success_rate=1.0,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["status"] == "completed_no_promotion"
    assert result["report"]["best_train_success_rate"] == 1.0
    assert result["report"]["best_heldout_success_rate"] == 0.0
    assert result["policy_candidate_package"]["status"] == "not_promoted"
    assert "dev_target_success_not_reached" in result["report"]["blockers"]


def test_policy_autoresearch_stops_before_branch_when_candidate_budget_exhausted(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path",
                    "variation_name": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
                {
                    "scenario_eval_run_id": "run_heldout_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path_holdout",
                    "variation_name": "blocked_path_holdout",
                    "split": "heldout",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
            ],
        },
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        max_iterations=3,
        agent_count=2,
        max_candidate_evaluations=0,
        target_success_rate=1.0,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["status"] == "completed_no_promotion"
    assert result["budget_ledger"]["status"] == "budget_exhausted"
    assert result["budget_ledger"]["usage"]["candidate_evaluations"] == 0
    assert "candidate_evaluation_budget_exhausted" in result["report"]["blockers"]
    assert not result["report"]["iteration_records"]
    assert (job_dir / "policy_autoresearch" / "budget_ledger.json").is_file()


def test_policy_autoresearch_can_use_external_evaluator_command(tmp_path: Path) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
                {
                    "scenario_eval_run_id": "run_heldout_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path_holdout",
                    "split": "heldout",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
            ],
        },
    )
    evaluator = tmp_path / "fake_policy_evaluator.py"
    evaluator.write_text(
        """
import json
import os

matrix = json.loads(open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_MATRIX"]).read())
recipe = json.loads(open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_RECIPE"]).read())
output = os.environ["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"]
params = recipe.get("mutable_parameters", {})
success = params.get("planner") == "clearance_aware"
attempts = []
for run in matrix["runs"]:
    attempts.append({
        "scenario_eval_run_id": run["scenario_eval_run_id"],
        "scenario_variation_instance_id": run.get("scenario_variation_instance_id"),
        "task_id": run.get("task_id"),
        "scenario_id": run.get("scenario_id"),
        "variation_name": run.get("variation_name"),
        "policy_id": recipe.get("candidate_id") or recipe.get("policy_id"),
        "success": success,
        "task_success": success,
        "metrics": {
            "simulator_execution_performed": True,
            "safety_event_count": 0,
            "contact_event_count": 0 if success else 1,
        },
        "claim_boundary": {"simulator_execution_performed": True},
        "failure_mode_ids": [] if success else ["failure_clearance_near_miss"],
    })
json.dump({
    "phase": os.environ["BLUEPRINT_POLICY_AUTORESEARCH_PHASE"],
    "frozen_verifier_sha256": os.environ["BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256"],
    "attempts": attempts,
}, open(output, "w"))
""".strip(),
        encoding="utf-8",
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        max_iterations=1,
        agent_count=1,
        target_success_rate=1.0,
        evaluator_command=f"{sys.executable} {evaluator}",
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["status"] == "completed_no_promotion"
    assert result["heldout_eval_result"]["evaluator_command_used"] is True
    assert result["heldout_eval_result"]["task_success_summary"]["task_success_rate"] == 1.0


def test_policy_autoresearch_can_call_wam_evaluator_command(tmp_path: Path) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "task_id": "tote_transfer",
                    "scenario_id": "blocked_path",
                    "variation_name": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
                {
                    "scenario_eval_run_id": "run_heldout_clearance",
                    "task_id": "tote_transfer",
                    "scenario_id": "heldout_blocked_path",
                    "variation_name": "heldout_blocked_path",
                    "split": "heldout",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
            ],
        },
    )
    evaluator = tmp_path / "fake_wam_evaluator.py"
    evaluator.write_text(
        """
import json
import os

matrix = json.loads(open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_MATRIX"]).read())
recipe = json.loads(open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_RECIPE"]).read())
output = os.environ["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"]
substrate = os.environ["BLUEPRINT_POLICY_AUTORESEARCH_EVALUATION_SUBSTRATE"]
assert substrate == "fixture_wam"
assert matrix["evaluation_substrate"] == "fixture_wam"
params = recipe.get("mutable_parameters", {})
success = params.get("planner") == "clearance_aware"
attempts = []
for run in matrix["runs"]:
    attempts.append({
        "scenario_eval_run_id": run["scenario_eval_run_id"],
        "scenario_variation_instance_id": run.get("scenario_variation_instance_id"),
        "task_id": run.get("task_id"),
        "scenario_id": run.get("scenario_id"),
        "variation_name": run.get("variation_name"),
        "policy_id": recipe.get("candidate_id") or recipe.get("policy_id"),
        "evaluation_substrate": substrate,
        "success": success,
        "task_success": success,
        "metrics": {
            "simulator_execution_performed": False,
            "safety_event_count": 0,
            "contact_event_count": 0 if success else 1,
            "world_model_uncertainty": 0.14,
        },
        "failure_mode_ids": [] if success else ["failure_clearance_near_miss"],
        "claim_boundary": {
            "evaluation_substrate": substrate,
            "generated_wam_rollout": True,
            "simulator_execution_performed": False,
            "rank_fidelity_result_proven": False
        }
    })
json.dump({
    "phase": os.environ["BLUEPRINT_POLICY_AUTORESEARCH_PHASE"],
    "frozen_verifier_sha256": os.environ["BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256"],
    "evaluation_substrate": substrate,
    "attempts": attempts,
}, open(output, "w"))
""".strip(),
        encoding="utf-8",
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        max_iterations=1,
        agent_count=1,
        target_success_rate=1.0,
        evaluation_substrates=("fixture_wam",),
        evaluator_command=f"{sys.executable} {evaluator}",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["report"]["status"] == "completed_no_promotion"
    assert result["report"]["requested_evaluation_substrates"] == ["fixture_wam"]
    assert result["report"]["wam_evaluation_substrate_requested"] is True
    assert result["report"]["simulator_execution_proven"] is False
    assert result["heldout_eval_result"]["evaluation_substrate"] == "fixture_wam"
    assert result["heldout_eval_result"]["status"] == "not_promoted"
    assert result["baseline_heldout_eval_result"]["status"] == ("failed_evaluator_contract")
    assert (
        "external_evaluator_simulator_execution_not_proven:0"
        in result["heldout_eval_result"]["contract_blockers"]
    )
    assert result["heldout_eval_result"]["attempts"] == []
    assert result["policy_candidate_package"]["status"] == "not_promoted"
    assert (
        result["policy_candidate_package"]["sim_only_policy_improvement_support_artifact"] is False
    )
    assert result["policy_candidate_package"]["wam_policy_improvement_support_artifact"] is True
    assert result["policy_candidate_package"]["customer_specific_srcc_claimed"] is False


def test_policy_autoresearch_blocks_unsupported_evaluation_substrate(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                }
            ],
        },
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        evaluation_substrates=("hardwired_private_model",),
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["report"]["status"] == "blocked"
    assert result["report"]["blockers"] == [
        "unsupported_evaluation_substrate:Unsupported evaluation substrate: hardwired_private_model"
    ]
    assert result["policy_candidate_package"]["status"] == "blocked"


def test_policy_autoresearch_rejects_external_evaluator_engine_mismatch(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
                {
                    "scenario_eval_run_id": "run_heldout_clearance",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path_holdout",
                    "split": "heldout",
                    "required_policy_capabilities": ["clearance_aware_navigation"],
                },
            ],
        },
    )
    evaluator = tmp_path / "wrong_engine_evaluator.py"
    evaluator.write_text(
        """
import json
import os

matrix = json.loads(open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_MATRIX"]).read())
recipe = json.loads(open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_RECIPE"]).read())
attempts = []
for run in matrix["runs"]:
    attempts.append({
        "scenario_eval_run_id": run["scenario_eval_run_id"],
        "policy_id": recipe.get("policy_id"),
        "simulator_engine": "mujoco",
        "success": True,
        "task_success": True,
        "metrics": {
            "simulator_execution_performed": True,
            "safety_event_count": 0,
            "contact_event_count": 0,
        },
        "claim_boundary": {"simulator_execution_performed": True},
    })
json.dump(
    {"simulator_engine": "mujoco", "attempts": attempts},
    open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"], "w"),
)
""".strip(),
        encoding="utf-8",
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        max_iterations=1,
        agent_count=1,
        target_success_rate=1.0,
        simulator_engines=("isaac_sim",),
        evaluator_commands_by_engine={"isaac_sim": f"{sys.executable} {evaluator}"},
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["status"] == "completed_no_promotion"
    assert result["baseline_heldout_eval_result"]["status"] == ("failed_evaluator_engine_mismatch")
    assert result["baseline_heldout_eval_result"]["failure_mode_ids"] == [
        "external_evaluator_engine_mismatch"
    ]
    assert result["report"]["proven_simulator_engines"] == []
    assert result["policy_candidate_package"]["simulator_execution_proven"] is False


def test_policy_autoresearch_does_not_count_isaac_without_specific_proof(
    tmp_path: Path,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_isaac_heldout",
                    "task_id": "walk_to_target",
                    "scenario_id": "owner_isaac_route",
                    "split": "heldout",
                }
            ],
        },
    )
    evaluator = tmp_path / "isaac_no_specific_proof_evaluator.py"
    evaluator.write_text(
        """
import json
import os

matrix = json.loads(open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_MATRIX"]).read())
attempts = []
for run in matrix["runs"]:
    attempts.append({
        "scenario_eval_run_id": run["scenario_eval_run_id"],
        "simulator_engine": "isaac_sim",
        "success": True,
        "task_success": True,
        "metrics": {
            "simulator_execution_performed": True,
            "safety_event_count": 0,
            "contact_event_count": 0,
        },
        "claim_boundary": {"simulator_execution_performed": True},
    })
json.dump(
    {"simulator_engine": "isaac_sim", "attempts": attempts},
    open(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT"], "w"),
)
""".strip(),
        encoding="utf-8",
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        max_iterations=1,
        agent_count=1,
        target_success_rate=1.0,
        simulator_engines=("isaac_sim",),
        evaluator_commands_by_engine={"isaac_sim": f"{sys.executable} {evaluator}"},
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["proven_simulator_engines"] == []
    assert result["policy_candidate_package"]["simulator_execution_proven"] is False
    assert result["heldout_eval_result"]["attempts"] == []
    assert "autoresearch_dev_split_missing" in result["report"]["blockers"]
    assert result["locked_test_eval_result"]["status"] == ("blocked_locked_test_split_contract")


def _owner_capture_root(tmp_path: Path) -> Path:
    capture_root = (
        tmp_path
        / "storage"
        / "local-blueprint"
        / "scenes"
        / "site-owner"
        / "captures"
        / "capture-owner"
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "site-owner", "capture_id": "capture-owner"},
    )
    return capture_root


def _owner_policy_command_script(path: Path, *, write_attempt_trace: bool = True) -> None:
    attempt_trace_block = (
        """
write(os.environ["BLUEPRINT_POLICY_AUTORESEARCH_OWNER_ATTEMPT_TRACE"], {
    "attempts": [{
        "attempt_id": "owner-policy-attempt-1",
        "scenario_eval_run_id": "run_owner_heldout",
        "task_id": "walk_to_target",
        "scenario_id": "owner_isaac_route",
        "variation_name": "heldout_owner_route",
        "success": True,
        "task_success": True,
        "metrics": {
            "safety_event_count": 0,
            "contact_event_count": 0
        },
        "failure_mode_ids": []
    }]
})
"""
        if write_attempt_trace
        else ""
    )
    path.write_text(
        f"""
import json
import os
from pathlib import Path

def write(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

robot_asset = {{
    "name": os.environ["BLUEPRINT_ROBOT_ASSET_NAME"],
    "uri_or_path": os.environ["BLUEPRINT_ROBOT_ASSET_URI_OR_PATH"],
    "source": os.environ["BLUEPRINT_ROBOT_ASSET_SOURCE"],
    "asset_class": os.environ["BLUEPRINT_ROBOT_ASSET_CLASS"],
}}
write(os.environ["BLUEPRINT_SCENE_LOAD_TRACE"], {{
    "status": "loaded",
    "scene_loaded": True,
    "robot_asset": robot_asset,
}})
write(os.environ["BLUEPRINT_SPAWN_TRACE"], {{
    "status": "validated",
    "spawn_pose_loaded": True,
    "robot_asset": robot_asset,
}})
write(os.environ["BLUEPRINT_POLICY_EXECUTION_TRACE"], {{
    "status": "completed",
    "default_policy_executed": True,
    "policy_execution_completed": True,
    "actions": [{{"name": "walk_to_target", "status": "completed"}}],
}})
write(os.environ["BLUEPRINT_SIM_ROBOT_POV_EVIDENCE"], {{
    "status": "complete",
    "sim_robot_pov_captured": True,
    "frames": [{{"camera": "front_rgbd", "path": "owner-frame-0001.png"}}],
}})
write(os.environ["BLUEPRINT_ARTIFACT_MANIFEST"], {{
    "status": "complete",
    "artifact_manifest_complete": True,
    "artifacts": [
        {{"kind": "policy_trace", "path": os.environ["BLUEPRINT_POLICY_EXECUTION_TRACE"]}},
        {{"kind": "sim_robot_pov", "path": os.environ["BLUEPRINT_SIM_ROBOT_POV_EVIDENCE"]}}
    ],
}})
{attempt_trace_block}
""".strip(),
        encoding="utf-8",
    )


def test_owner_gpu_policy_evaluator_requires_owner_proof_and_scores_attempt_trace(
    tmp_path: Path,
) -> None:
    capture_root = _owner_capture_root(tmp_path)
    recipe_path = _seed_recipe(tmp_path, {"policy_id": "owner_isaac_policy"})
    matrix_path = tmp_path / "owner_split_matrix.json"
    _write_json(
        matrix_path,
        {
            "schema_version": "policy_autoresearch_split_matrix.v1",
            "phase": "heldout",
            "simulator_engine": "isaac_sim",
            "runs": [
                {
                    "scenario_eval_run_id": "run_owner_heldout",
                    "task_id": "walk_to_target",
                    "scenario_id": "owner_isaac_route",
                    "variation_name": "heldout_owner_route",
                }
            ],
        },
    )
    command_script = tmp_path / "owner_policy_command.py"
    _owner_policy_command_script(command_script)
    output_path = tmp_path / "owner_policy_evaluator_output.json"

    payload = run_owner_gpu_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        capture_root=capture_root,
        owner_command=f"{sys.executable} {command_script}",
        simulator_engine="isaac_sim",
        simulator_version="6.0.0",
        gpu_model="L40S 48GB",
        operator_id="operator-1",
        operator_attestation="I ran this command on the owner GPU VM.",
        timeout_seconds=30,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert output_path.is_file()
    assert payload["status"] == "completed"
    assert payload["owner_gpu_simulator_execution_proven"] is True
    assert payload["isaac_sim_execution_proven"] is True
    assert payload["policy_attempt_trace_present"] is True
    assert payload["attempts"][0]["task_success"] is True
    assert payload["attempts"][0]["metrics"]["simulator_execution_performed"] is True
    assert payload["attempts"][0]["claim_boundary"]["rank_fidelity_result_proven"] is False


def test_owner_gpu_policy_evaluator_does_not_score_success_without_attempt_trace(
    tmp_path: Path,
) -> None:
    capture_root = _owner_capture_root(tmp_path)
    recipe_path = _seed_recipe(tmp_path, {"policy_id": "owner_isaac_policy"})
    matrix_path = tmp_path / "owner_split_matrix.json"
    _write_json(
        matrix_path,
        {
            "schema_version": "policy_autoresearch_split_matrix.v1",
            "phase": "heldout",
            "simulator_engine": "isaac_sim",
            "runs": [
                {
                    "scenario_eval_run_id": "run_owner_heldout",
                    "task_id": "walk_to_target",
                    "scenario_id": "owner_isaac_route",
                    "variation_name": "heldout_owner_route",
                }
            ],
        },
    )
    command_script = tmp_path / "owner_policy_command_without_attempt_trace.py"
    _owner_policy_command_script(command_script, write_attempt_trace=False)

    payload = run_owner_gpu_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=tmp_path / "owner_policy_evaluator_output.json",
        capture_root=capture_root,
        owner_command=f"{sys.executable} {command_script}",
        simulator_engine="isaac_sim",
        simulator_version="6.0.0",
        gpu_model="L40S 48GB",
        operator_id="operator-1",
        operator_attestation="I ran this command on the owner GPU VM.",
        timeout_seconds=30,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert payload["status"] == "blocked_no_policy_attempt_trace"
    assert payload["owner_gpu_simulator_execution_proven"] is True
    assert payload["policy_attempt_trace_present"] is False
    assert payload["attempts"][0]["task_success"] is False
    assert payload["attempts"][0]["metrics"]["simulator_execution_performed"] is True
    assert payload["attempts"][0]["failure_mode_ids"] == ["owner_gpu_policy_attempt_trace_missing"]


def test_policy_autoresearch_local_replay_evaluator_uses_existing_attempt_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root, job_dir = _job_dir(tmp_path)
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "robot_eval_scenario_eval_matrix.v1",
            "runs": [
                {
                    "scenario_eval_run_id": "run_train_blocked_path",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path",
                    "variation_name": "blocked_path",
                    "split": "train",
                    "required_policy_capabilities": [
                        "clearance_aware_navigation",
                        "retry_recovery",
                    ],
                },
                {
                    "scenario_eval_run_id": "run_heldout_blocked_path",
                    "task_id": "navigate_to_target",
                    "scenario_id": "blocked_path_holdout",
                    "variation_name": "blocked_path_holdout",
                    "split": "heldout",
                    "required_policy_capabilities": [
                        "clearance_aware_navigation",
                        "retry_recovery",
                    ],
                },
            ],
        },
    )
    trace_path = job_dir / "simulator_command_batch_attempt_trace.jsonl"
    trace_rows = [
        {
            "attempt_id": "attempt-train",
            "scenario_eval_run_id": "run_train_blocked_path",
            "task_id": "navigate_to_target",
            "scenario_id": "blocked_path",
            "variation_name": "blocked_path",
            "status": "failed_task_criteria",
            "success": False,
            "task_success": False,
            "failure_mode_ids": [
                "failure_target_not_reached",
                "failure_endpoint_not_clean",
                "failure_timeout",
                "failure_clearance_near_miss",
            ],
            "metrics": {
                "clearance_threshold_violation": True,
                "timeout_count": 1,
                "safety_event_count": 0,
                "contact_event_count": 1,
            },
        },
        {
            "attempt_id": "attempt-heldout",
            "scenario_eval_run_id": "run_heldout_blocked_path",
            "task_id": "navigate_to_target",
            "scenario_id": "blocked_path_holdout",
            "variation_name": "blocked_path_holdout",
            "status": "failed_task_criteria",
            "success": False,
            "task_success": False,
            "failure_mode_ids": [
                "failure_target_not_reached",
                "failure_endpoint_not_clean",
                "failure_timeout",
                "failure_clearance_near_miss",
            ],
            "metrics": {
                "clearance_threshold_violation": True,
                "timeout_count": 1,
                "safety_event_count": 0,
                "contact_event_count": 1,
            },
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in trace_rows) + "\n",
        encoding="utf-8",
    )
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv(
        "PYTHONPATH",
        src_path + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )

    result = run_policy_autoresearch(
        capture_root=capture_root,
        job_dir=job_dir,
        policy_recipe_path=_seed_recipe(tmp_path),
        max_iterations=1,
        agent_count=2,
        target_success_rate=1.0,
        evaluator_command=f"{sys.executable} -m blueprint_pipeline.policy_autoresearch_local_evaluator",
        evaluator_attempt_trace_path=trace_path,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert result["report"]["status"] == "completed_no_promotion"
    assert result["report"]["baseline_heldout_success_rate"] == 0.0
    assert result["report"]["best_heldout_success_rate"] == 0.0
    assert result["heldout_eval_result"]["status"] == "not_promoted"
    assert result["baseline_heldout_eval_result"]["status"] == ("failed_evaluator_contract")
    assert (
        "external_evaluator_simulator_execution_not_proven:0"
        in result["heldout_eval_result"]["contract_blockers"]
    )
    assert result["heldout_eval_result"]["attempts"] == []


def test_mujoco_policy_evaluator_executes_candidate_route_matrix(tmp_path: Path) -> None:
    recipe_path = _seed_recipe(
        tmp_path,
        {
            "policy_id": "route_candidate",
            "mutable_parameters": {
                "planner": "clearance_aware",
                "clearance_margin_m": 0.2,
                "retry_budget": 1,
                "max_speed_mps": 0.7,
            },
        },
    )
    matrix_path = tmp_path / "split_matrix.json"
    _write_json(
        matrix_path,
        {
            "schema_version": "policy_autoresearch_split_matrix.v1",
            "phase": "heldout",
            "runs": [
                {
                    "scenario_eval_run_id": "run-heldout-route",
                    "task_id": "walk_to_target",
                    "scenario_id": "warehouse_route",
                    "variation_name": "blocked_path",
                    "start_xyz": [8.0, -0.2, 0.793],
                    "target_xyz": [-8.0, -5.7, 0.793],
                    "route_waypoints": [
                        [8.0, -0.2, 0.793],
                        [-8.0, -5.7, 0.793],
                    ],
                }
            ],
        },
    )
    captured: dict[str, Path] = {}

    def fake_simulator_runner(**kwargs):
        candidate_matrix_path = Path(kwargs["scenario_eval_matrix_path"])
        captured["candidate_matrix_path"] = candidate_matrix_path
        candidate_matrix = json.loads(candidate_matrix_path.read_text(encoding="utf-8"))
        route = candidate_matrix["runs"][0]["route_waypoints"]
        assert candidate_matrix["source_scenario_eval_matrix_mutated"] is False
        assert candidate_matrix["runs"][0]["policy_id"] == "route_candidate"
        assert len(route) > 2
        assert route != [[8.0, -0.2, 0.793], [-8.0, -5.7, 0.793]]
        output_path = Path(kwargs["simulator_output_path"])
        payload = {
            "status": "completed",
            "simulator_execution_proven": True,
            "attempts": [
                {
                    "attempt_id": "mujoco-route-attempt",
                    "scenario_eval_run_id": "run-heldout-route",
                    "task_id": "walk_to_target",
                    "scenario_id": "warehouse_route",
                    "variation_name": "blocked_path",
                    "task_success": True,
                    "success": True,
                    "failure_mode_ids": [],
                    "metrics": {
                        "robot_scene_contact_event_count": 0,
                        "near_miss_event_count": 0,
                        "collision_response_event_count": 0,
                        "fall_count": 0,
                        "unsafe_proximity_event_count": 0,
                    },
                    "claim_boundary": "fake_mujoco_runner",
                }
            ],
        }
        _write_json(output_path, payload)
        return payload

    output_path = tmp_path / "evaluator_output.json"
    payload = run_mujoco_policy_evaluator(
        recipe_path=recipe_path,
        matrix_path=matrix_path,
        output_path=output_path,
        capture_root=tmp_path,
        g1_model_root=tmp_path / "g1",
        simulator_runner=fake_simulator_runner,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert output_path.is_file()
    assert captured["candidate_matrix_path"].is_file()
    assert payload["status"] == "completed"
    assert payload["simulator_execution_proven"] is True
    assert payload["candidate_route_strategies"] == ["policy_perimeter_clearance_route"]
    assert payload["attempts"][0]["task_success"] is True
    assert payload["attempts"][0]["metrics"]["simulator_execution_performed"] is True
    assert payload["attempts"][0]["metrics"]["contact_event_count"] == 0
    assert payload["attempts"][0]["claim_boundary"]["rank_fidelity_result_proven"] is False


def test_locked_test_access_claim_is_atomic_and_one_time(tmp_path: Path) -> None:
    first, path = autoresearch_module._claim_locked_test_once(
        claim_root=tmp_path,
        verifier_sha256="a" * 64,
        candidate_id="candidate-a",
        generated_at="now",
    )
    second, second_path = autoresearch_module._claim_locked_test_once(
        claim_root=tmp_path,
        verifier_sha256="a" * 64,
        candidate_id="candidate-b",
        generated_at="later",
    )

    assert first is True
    assert second is False
    assert second_path == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidate_id"] == "candidate-a"
    assert payload["locked_test_access_count"] == 1


def test_split_design_blocks_each_group_axis_and_embedded_outcomes() -> None:
    train = {
        "scenario_eval_run_id": "train",
        "site_id": "site-train",
        "scene_id": "scene-train",
        "task_id": "task-train",
        "object_id": "object-train",
        "source_trajectory_id": "shared-trajectory",
        "policy_lineage_id": "lineage-train",
    }
    dev = {
        "scenario_eval_run_id": "dev",
        "site_id": "site-dev",
        "scene_id": "scene-dev",
        "task_id": "task-dev",
        "object_id": "object-dev",
        "source_trajectory_id": "shared-trajectory",
        "policy_lineage_id": "lineage-dev",
    }
    locked_test = {
        "scenario_eval_run_id": "test",
        "site_id": "site-test",
        "scene_id": "scene-test",
        "task_id": "task-test",
        "object_id": "object-test",
        "source_trajectory_id": "trajectory-test",
        "policy_lineage_id": "lineage-test",
        "metrics": {"private_locked_outcome": 1.0},
    }

    validation = autoresearch_module._split_design_validation(
        train_runs=[train],
        dev_runs=[dev],
        locked_test_runs=[locked_test],
    )

    assert validation["status"] == "blocked"
    assert (
        "autoresearch_group_axis_leakage:source_trajectory_id:train:dev" in validation["blockers"]
    )
    assert any(
        blocker.startswith("autoresearch_locked_test_outcome_fields_forbidden:0")
        for blocker in validation["blockers"]
    )


def test_verifier_digest_changes_for_full_split_affecting_run_edits(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    run = {
        "scenario_eval_run_id": "train",
        "site_id": "site",
        "scene_id": "scene",
        "task_id": "task",
        "object_id": "object-a",
        "source_trajectory_id": "trajectory",
        "policy_lineage_id": "lineage",
        "split": "train",
    }
    _write_json(matrix_path, {"runs": [run]})
    first = autoresearch_module._build_verifier_manifest(
        matrix_path=matrix_path,
        reviewed_examples_path=None,
        reviewed_examples_payload=None,
        runs=[run],
        train_runs=[run],
        dev_runs=[],
        locked_test_runs=[],
        split_design_validation={"status": "blocked"},
        split_source="test",
        target_success_rate=1.0,
        generated_at="now",
    )
    changed = {**run, "object_id": "object-b"}
    _write_json(matrix_path, {"runs": [changed]})
    second = autoresearch_module._build_verifier_manifest(
        matrix_path=matrix_path,
        reviewed_examples_path=None,
        reviewed_examples_payload=None,
        runs=[changed],
        train_runs=[changed],
        dev_runs=[],
        locked_test_runs=[],
        split_design_validation={"status": "blocked"},
        split_source="test",
        target_success_rate=1.0,
        generated_at="now",
    )

    assert first["verifier_sha256"] != second["verifier_sha256"]
    assert first["scenario_eval_matrix_sha256"] != second["scenario_eval_matrix_sha256"]


def test_locked_test_identity_is_order_invariant_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    locked_a = {
        "scenario_eval_run_id": "locked-a",
        "scenario_variation_instance_id": "variation-a",
        "task_id": "task-a",
        "scenario_id": "scenario-a",
        "split": "locked_test",
    }
    locked_b = {
        "scenario_eval_run_id": "locked-b",
        "scenario_variation_instance_id": "variation-b",
        "task_id": "task-b",
        "scenario_id": "scenario-b",
        "split": "locked_test",
    }
    _write_json(matrix_path, {"runs": [locked_a, locked_b]})

    def build(rows):
        return autoresearch_module._build_verifier_manifest(
            matrix_path=matrix_path,
            reviewed_examples_path=None,
            reviewed_examples_payload=None,
            runs=rows,
            train_runs=[],
            dev_runs=[],
            locked_test_runs=rows,
            split_design_validation={"status": "validated", "blockers": []},
            split_source="registered",
            target_success_rate=1.0,
            generated_at="now",
        )

    forward = build([locked_a, locked_b])
    reversed_order = build([locked_b, locked_a])

    assert forward["locked_test_identity_sha256"] == reversed_order["locked_test_identity_sha256"]
    assert [
        row["scenario_eval_run_id"]
        for row in forward["locked_test_identity_payload"]["locked_test_runs"]
    ] == ["locked-a", "locked-b"]

    duplicate = build([locked_a, locked_a])
    assert duplicate["split_design_validation"]["status"] == "blocked"
    assert (
        "autoresearch_locked_test_canonical_identity_duplicate"
        in duplicate["split_design_validation"]["blockers"]
    )
