from __future__ import annotations

from pathlib import Path

import pytest

import blueprint_pipeline.benchmark_protocol as benchmark_protocol
from blueprint_pipeline.common import write_json


BOOTSTRAP_REPLICATES = benchmark_protocol.BOOTSTRAP_REPLICATES
EXTERNAL_REFERENCE_SCHEMA_VERSION = benchmark_protocol.EXTERNAL_REFERENCE_SCHEMA_VERSION
RESULTS_SCHEMA_VERSION = benchmark_protocol.RESULTS_SCHEMA_VERSION
SPEC_SCHEMA_VERSION = benchmark_protocol.SPEC_SCHEMA_VERSION
build_benchmark_report = benchmark_protocol.build_benchmark_report
canonical_sha256 = benchmark_protocol.canonical_sha256
compile_benchmark_protocol = benchmark_protocol.compile_benchmark_protocol
execute_benchmark_protocol_request = benchmark_protocol.execute_benchmark_protocol_request
validate_benchmark_spec = benchmark_protocol.validate_benchmark_spec
write_benchmark_report = benchmark_protocol.write_benchmark_report


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64
DIGEST_F = "f" * 64


@pytest.fixture(autouse=True)
def _short_bootstrap(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(benchmark_protocol, "_BOOTSTRAP_EXECUTION_REPLICATES", 200)


def _policy(policy_id: str, digest: str, *, public: bool) -> dict:
    row = {
        "policy_id": policy_id,
        "policy_family": f"family-{policy_id}",
        "checkpoint_id": f"checkpoint-{policy_id}",
        "checkpoint_sha256": digest,
        "adapter_code_sha256": DIGEST_C,
        "embodiment_id": "widowx",
        "public": public,
        "runner": {
            "schema_version": "reproducible_policy_runner.v1",
            "command": f"python run.py --policy {policy_id}",
            "runner_manifest_sha256": DIGEST_D,
            "source_revision": f"refs/tags/{policy_id}-v1",
        },
    }
    if public:
        row.update(
            {
                "source_uri": f"https://example.org/{policy_id}",
                "license": "Apache-2.0",
            }
        )
    return row


def valid_spec() -> dict:
    scenarios = []
    for index, split in enumerate(("train", "dev", "public_test", "hidden_test")):
        label = "seen" if index % 2 == 0 else "unseen"
        scenarios.append(
            {
                "scenario_id": f"scenario-{split}",
                "task_id": "drawer-close",
                "split": split,
                "seed": 100 + index,
                "initial_condition_sha256": DIGEST_E,
                "generalization": {
                    "task": label,
                    "scene": label,
                    "object": label,
                    "camera": label,
                    "lighting": label,
                    "embodiment": label,
                },
            }
        )
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "benchmark_id": "blueprint-drawer",
        "benchmark_version": "2026.1",
        "protocol_version": "1",
        "title": "Blueprint drawer benchmark",
        "description": "Frozen exact-site policy comparison.",
        "frozen": True,
        "preregistration_sha256": DIGEST_A,
        "tasks": [
            {
                "task_id": "drawer-close",
                "instruction": "Close the drawer.",
                "reset_protocol": "Reset drawer to registered open pose.",
                "success_definition": "Drawer joint is within the registered closed tolerance.",
                "timeout_seconds": 30,
                "partial_progress_predicates": [
                    {"predicate_id": "drawer-motion", "weight": 0.4},
                    {"predicate_id": "drawer-closed", "weight": 0.6},
                ],
            }
        ],
        "action_space": {
            "schema_ref": "blueprint://schemas/robot_eval_action_trace.v1",
            "dimension": 7,
            "coordinate_frame": "robot_base",
            "timestamp_semantics": "monotonic_chunk_start_and_per_sample_offsets",
            "normalization_manifest_sha256": DIGEST_B,
        },
        "environment": {
            "site_id": "captured-site-drawer",
            "site_package_sha256": DIGEST_C,
            "representation_type": "captured_3dgs_site_memory",
            "observation_calibration_sha256": DIGEST_D,
            "physics_authority": "mujoco",
            "physics_asset_sha256": DIGEST_E,
            "same_site_capture": True,
        },
        "evaluator_runtime": {
            "evaluator_id": "blueprint-wam-runtime",
            "evaluator_version": "2026.1",
            "runner_manifest_sha256": DIGEST_A,
            "source_revision": "refs/tags/blueprint-wam-2026.1",
            "success_evaluator_sha256": DIGEST_B,
            "robot_adapter_sha256": DIGEST_C,
            "observation_adapter_sha256": DIGEST_D,
            "action_adapter_sha256": DIGEST_E,
            "deterministic_seeding": True,
        },
        "scenarios": scenarios,
        "rollout_protocol": {
            "fixed_rollouts_per_scenario_policy": 2,
            "cherry_picking_prohibited": True,
            "result_replacement_prohibited": True,
            "infrastructure_retries_scored_as_new_attempts": True,
        },
        "required_episode_evidence": {
            "video": True,
            "action_trace": True,
            "evaluator_output": True,
            "content_digests": True,
        },
        "scoring": {
            "metrics": [
                "full_task_success",
                "partial_progress",
                "efficiency",
                "safety_interventions",
                "evaluator_abstention",
            ],
            "confidence_intervals_required": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "public_baselines": [_policy("baseline", DIGEST_A, public=True)],
        "candidate_policies": [_policy("candidate", DIGEST_B, public=False)],
    }


def _results(plan: dict) -> dict:
    rows = []
    for attempt in plan["attempts"]:
        success = attempt["policy_id"] == "candidate"
        rows.append(
            {
                **{
                    key: attempt[key]
                    for key in (
                        "attempt_id",
                        "policy_id",
                        "checkpoint_sha256",
                        "scenario_id",
                        "split",
                        "seed",
                        "rollout_index",
                        "initial_condition_sha256",
                        "environment_sha256",
                        "evaluator_runtime_sha256",
                    )
                },
                "status": "completed",
                "selected_for_reporting": True,
                "replacement_attempt": False,
                "full_task_success": success,
                "partial_progress": 1.0 if success else 0.5,
                "efficiency": {
                    "normalized_score": 0.9 if success else 0.4,
                    "duration_seconds": 12.0,
                    "path_length_m": 1.5,
                },
                "safety": {
                    "intervention_count": 0 if success else 1,
                    "collision_count": 0,
                    "unsafe_event_count": 0,
                },
                "evaluator_abstained": False,
                "evidence": {
                    "video": {"uri": f"gs://bucket/{attempt['attempt_id']}.mp4", "sha256": DIGEST_C},
                    "action_trace": {"uri": f"gs://bucket/{attempt['attempt_id']}.jsonl", "sha256": DIGEST_D},
                    "evaluator_output": {"uri": f"gs://bucket/{attempt['attempt_id']}.json", "sha256": DIGEST_E},
                },
            }
        )
    return {"schema_version": RESULTS_SCHEMA_VERSION, "attempts": rows}


def test_valid_spec_compiles_hidden_split_without_public_identifier_leak(tmp_path: Path):
    spec = valid_spec()
    assert validate_benchmark_spec(spec)["status"] == "validated"

    compiled = compile_benchmark_protocol(spec, output_dir=tmp_path, generated_at="2026-07-21T00:00:00Z")

    card_text = (tmp_path / "benchmark_card.json").read_text()
    assert "scenario-hidden_test" not in card_text
    assert compiled["benchmark_card"]["split_summary"]["counts"]["hidden_test"] == 1
    assert compiled["benchmark_card"]["split_summary"]["hidden_test_identifiers_redacted"] is True
    assert compiled["execution_plan"]["attempt_count"] == 8
    assert len({row["attempt_id"] for row in compiled["execution_plan"]["attempts"]}) == 8
    assert compiled["evaluation_run_task_scenario_pack"]["adapter_id"] == "benchmark_task_scenario_pack"
    assert compiled["webapp_projection"]["hidden_scenario_identifiers_included"] is False
    assert (
        compiled["webapp_projection"]["environment_summary"]["representation_type"]
        == "captured_3dgs_site_memory"
    )
    assert (tmp_path / "benchmark_split_manifest.private.json").stat().st_mode & 0o077 == 0


def test_report_has_all_metric_confidence_intervals_breakdowns_and_external_fidelity(tmp_path: Path):
    spec = valid_spec()
    compiled = compile_benchmark_protocol(spec, output_dir=tmp_path)
    results = _results(compiled["execution_plan"])
    reference = {
        "schema_version": EXTERNAL_REFERENCE_SCHEMA_VERSION,
        "reference_id": "partner-real-robot-2026",
        "reference_type": "real_robot",
        "site_alignment": "same_site",
        "independently_accepted": True,
        "source_uri": "https://example.org/reference.json",
        "source_artifact_sha256": DIGEST_F,
        "task_mapping_sha256": DIGEST_E,
        "policy_results": [
            {"policy_id": "baseline", "checkpoint_sha256": DIGEST_A, "score": 0.2},
            {"policy_id": "candidate", "checkpoint_sha256": DIGEST_B, "score": 0.8},
            # Three exact matches are required; an unmatched external row cannot satisfy it.
            {"policy_id": "external-only", "checkpoint_sha256": DIGEST_C, "score": 0.5},
        ],
    }
    report = build_benchmark_report(
        spec=spec,
        plan=compiled["execution_plan"],
        results=results,
        external_reference=reference,
        seed=7,
    )

    assert report["status"] == "complete"
    assert report["anti_cherry_picking_verified"] is True
    assert report["evidence_summary"] == {
        "attempt_count": 8,
        "video_count": 8,
        "action_trace_count": 8,
        "evaluator_output_count": 8,
        "all_attempts_digest_bound": True,
    }
    assert len(report["evidence_index_sha256"]) == 64
    for policy in report["policy_aggregates"]:
        for metric in policy["metrics"].values():
            assert len(metric["confidence_interval_95"]) == 2
            assert metric["bootstrap_replicates"] == 200
    assert set(report["breakdowns"]["generalization"]) == {
        "task", "scene", "object", "camera", "lighting", "embodiment"
    }
    assert report["external_rank_fidelity"]["status"] == "blocked"
    assert "external_rank_fidelity_requires_three_exact_checkpoint_matches" in report["external_rank_fidelity"]["blockers"]
    written = write_benchmark_report(
        spec=spec,
        plan=compiled["execution_plan"],
        results=results,
        output_dir=tmp_path,
        external_reference=reference,
        seed=7,
    )
    assert written["webapp_projection"]["evidence_summary"]["video_count"] == 8
    evidence_index_path = tmp_path / "benchmark_evidence_index.private.json"
    assert evidence_index_path.is_file()
    assert evidence_index_path.stat().st_mode & 0o077 == 0


def test_external_rank_fidelity_measures_three_exact_checkpoints(tmp_path: Path):
    spec = valid_spec()
    spec["candidate_policies"].append(_policy("candidate-two", DIGEST_F, public=False))
    compiled = compile_benchmark_protocol(spec, output_dir=tmp_path)
    results = _results(compiled["execution_plan"])
    for row in results["attempts"]:
        if row["policy_id"] == "candidate-two":
            row["full_task_success"] = True
            row["partial_progress"] = 0.8
            row["efficiency"]["normalized_score"] = 0.7
    reference = {
        "schema_version": EXTERNAL_REFERENCE_SCHEMA_VERSION,
        "reference_id": "partner-real-robot-2026",
        "reference_type": "real_robot",
        "site_alignment": "different_site",
        "independently_accepted": True,
        "source_uri": "https://example.org/reference.json",
        "source_artifact_sha256": DIGEST_F,
        "task_mapping_sha256": DIGEST_E,
        "policy_results": [
            {"policy_id": "baseline", "checkpoint_sha256": DIGEST_A, "score": 0.1},
            {"policy_id": "candidate-two", "checkpoint_sha256": DIGEST_F, "score": 0.7},
            {"policy_id": "candidate", "checkpoint_sha256": DIGEST_B, "score": 0.9},
        ],
    }
    report = build_benchmark_report(
        spec=spec,
        plan=compiled["execution_plan"],
        results=results,
        external_reference=reference,
        seed=11,
    )
    external = report["external_rank_fidelity"]
    assert external["status"] == "measured"
    assert external["measurement_scope"] == "cross_site_real_robot_rank_concordance"
    assert external["metrics"]["spearman"]["estimate"] is not None
    assert external["metrics"]["mmrv"]["confidence_interval_95"][0] is not None
    assert external["claim_boundary"]["different_site_comparison_is_not_site_specific_validation"] is True
    assert external["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert external["claim_boundary"]["cross_site_rank_concordance_proven"] is True

    reference["site_alignment"] = "same_site"
    same_site = build_benchmark_report(
        spec=spec,
        plan=compiled["execution_plan"],
        results=results,
        external_reference=reference,
        seed=11,
    )["external_rank_fidelity"]
    assert same_site["measurement_scope"] == "same_site_real_robot_rank_fidelity"
    assert same_site["claim_boundary"]["rank_fidelity_result_proven"] is True
    assert same_site["claim_boundary"]["public_claim_upgrade_allowed"] is False

    reference["independently_accepted"] = False
    unaccepted = build_benchmark_report(
        spec=spec,
        plan=compiled["execution_plan"],
        results=results,
        external_reference=reference,
        seed=11,
    )["external_rank_fidelity"]
    assert unaccepted["status"] == "blocked"
    assert "external_reference_not_independently_accepted" in unaccepted["blockers"]


def test_report_blocks_missing_attempt_and_missing_evidence(tmp_path: Path):
    spec = valid_spec()
    compiled = compile_benchmark_protocol(spec, output_dir=tmp_path)
    results = _results(compiled["execution_plan"])
    results["attempts"].pop()
    results["attempts"][0]["evidence"].pop("video")
    report = build_benchmark_report(
        spec=spec,
        plan=compiled["execution_plan"],
        results=results,
        seed=13,
    )
    assert report["status"] == "blocked"
    assert "result_attempt_coverage_not_exact" in report["blockers"]
    assert "result_evidence_missing_or_invalid:0:video" in report["blockers"]
    assert report["anti_cherry_picking_verified"] is False


def test_report_blocks_result_environment_or_evaluator_binding_mismatch(tmp_path: Path):
    spec = valid_spec()
    compiled = compile_benchmark_protocol(spec, output_dir=tmp_path)
    results = _results(compiled["execution_plan"])
    results["attempts"][0]["environment_sha256"] = DIGEST_F
    results["attempts"][1].pop("evaluator_runtime_sha256")

    report = build_benchmark_report(
        spec=spec,
        plan=compiled["execution_plan"],
        results=results,
        seed=17,
    )

    assert report["status"] == "blocked"
    assert "result_attempt_binding_mismatch:0:environment_sha256" in report["blockers"]
    assert (
        "result_attempt_binding_mismatch:1:evaluator_runtime_sha256"
        in report["blockers"]
    )


def test_invalid_spec_rejects_unfrozen_hidden_and_missing_seen_unseen_axis():
    spec = valid_spec()
    spec["frozen"] = False
    spec["scenarios"] = [row for row in spec["scenarios"] if row["split"] != "hidden_test"]
    for row in spec["scenarios"]:
        row["generalization"]["camera"] = "seen"
    spec["environment"]["physics_asset_sha256"] = ""
    spec["evaluator_runtime"]["deterministic_seeding"] = False
    blockers = validate_benchmark_spec(spec)["blockers"]
    assert "benchmark_must_be_frozen" in blockers
    assert "required_split_missing:hidden_test" in blockers
    assert "seen_unseen_coverage_missing:camera" in blockers
    assert "environment_physics_asset_digest_required" in blockers
    assert "evaluator_runtime_deterministic_seeding_required" in blockers


def test_benchmark_grade_job_request_compiles_private_plan_and_redacted_projection(
    tmp_path: Path,
):
    spec = valid_spec()
    spec_path = tmp_path / "private" / "benchmark_spec.json"
    write_json(spec_path, spec)
    status = execute_benchmark_protocol_request(
        {
            "benchmark_protocol_request": {
                "schema_version": "blueprint_benchmark_protocol_request.v1",
                "mode": "benchmark_grade",
                "benchmark_spec_uri": "private/benchmark_spec.json",
                "benchmark_spec_sha256": canonical_sha256(spec),
                "frozen_hidden_splits_required": True,
                "fixed_rollouts_required": True,
                "confidence_intervals_required": True,
                "exact_checkpoint_digests_required": True,
                "private_split_material_allowed_in_webapp": False,
                "scheduler_owner": "BlueprintCapturePipeline",
            }
        },
        output_dir=tmp_path / "job" / "benchmark_protocol",
        allowed_root=tmp_path,
    )
    assert status["status"] == "planned"
    assert status["claim_boundary"]["private_split_material_exported"] is False
    assert (tmp_path / "job/benchmark_protocol/benchmark_split_manifest.private.json").is_file()
    projection_text = (
        tmp_path / "job/benchmark_protocol/webapp_benchmark_projection.json"
    ).read_text()
    assert "scenario-hidden_test" not in projection_text


def test_benchmark_grade_job_request_blocks_digest_mismatch(tmp_path: Path):
    spec_path = tmp_path / "benchmark_spec.json"
    write_json(spec_path, valid_spec())
    status = execute_benchmark_protocol_request(
        {
            "benchmark_protocol_request": {
                "schema_version": "blueprint_benchmark_protocol_request.v1",
                "mode": "benchmark_grade",
                "benchmark_spec_uri": "benchmark_spec.json",
                "benchmark_spec_sha256": DIGEST_F,
                "frozen_hidden_splits_required": True,
                "fixed_rollouts_required": True,
                "confidence_intervals_required": True,
                "exact_checkpoint_digests_required": True,
                "private_split_material_allowed_in_webapp": False,
                "scheduler_owner": "BlueprintCapturePipeline",
            }
        },
        output_dir=tmp_path / "job" / "benchmark_protocol",
        allowed_root=tmp_path,
    )
    assert status["status"] == "blocked"
    assert status["blockers"] == ["benchmark_spec_sha256_mismatch"]
