from __future__ import annotations

import builtins
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import robot_eval_execution as ree


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (root / "pipeline" / "robot_eval_dataset").mkdir(parents=True, exist_ok=True)
    return root


def test_robot_eval_execution_private_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    assert ree._number(True, default=3.5) == 3.5
    assert ree._string_list("single") == ["single"]
    assert ree._read_optional_mapping(tmp_path / "missing.json") == {}
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert ree._read_optional_mapping(invalid_json) == {}

    monkeypatch.setattr(ree.os.path, "relpath", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")))
    assert ree._relative_to(tmp_path, tmp_path / "x") == str(tmp_path / "x")

    assert ree._source_trace_base_dir({"artifact_paths": {"batch_trace_package_manifest": "missing.json"}}) is None
    copied, records = ree._copy_command_batch_trace_artifacts(
        job_dir=job_dir,
        trace_package={
            "artifact_paths": {
                "attempt_trace_jsonl": "",
                "contact_stream_jsonl": "gs://remote/contact.jsonl",
                "planner_state_jsonl": "missing-planner.jsonl",
            }
        },
        source_base_dir=tmp_path,
    )
    assert copied == {}
    assert records["attempt_trace_jsonl"]["status"] == "missing_source_ref"
    assert records["contact_stream_jsonl"]["status"] == "remote_source_not_copied"
    assert records["planner_state_jsonl"]["status"] == "missing_source_file"

    assert ree.default_test_policy_package_from_request({"default_test_policy": {"enabled": False}}) == {}
    assert ree.default_test_policy_package_from_request(
        {"default_test_policy": {"policy_kind": "unsupported"}}
    ) == {}
    manipulation = ree.default_test_policy_package_from_request(
        {
            "default_test_policy": {
                "policy_kind": "mobile_manipulation_pick_carry_place",
                "objectId": "tote-1",
            }
        }
    )
    assert manipulation["high_level_skill_trace"]["object_id"] == "tote-1"
    missing_target_manipulation = ree.default_test_policy_package_from_request(
        {
            "default_test_policy": {
                "policy_kind": "mobile_manipulation_pick_carry_place",
            }
        }
    )
    assert missing_target_manipulation["high_level_skill_trace"]["object_id"] == ""
    assert missing_target_manipulation["high_level_skill_trace"]["blockers"] == [
        "default_manipulation_policy_object_id_missing"
    ]

    inline_pov, _, inline_source = ree._load_real_robot_pov_payload(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={"realRobotPov": {"records": [{"evidence_id": "pov-1"}]}},
    )
    assert inline_pov["records"][0]["evidence_id"] == "pov-1"
    assert inline_source == "job_request_inline_real_robot_pov"
    pov_ref = job_dir / "pov.json"
    _write_json(pov_ref, {"records": [{"evidence_id": "pov-ref"}]})
    ref_pov, _, ref_source = ree._load_real_robot_pov_payload(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={"realRobotPovManifestUri": "pov.json"},
    )
    assert ref_pov["records"][0]["evidence_id"] == "pov-ref"
    assert ref_source == "job_request_real_robot_pov_manifest_ref"

    assert ree._local_reference_path(f"file://{pov_ref}", capture_root=capture_root, job_dir=job_dir) == pov_ref
    assert ree._local_reference_path("https://example.com/pov.json", capture_root=capture_root, job_dir=job_dir) is None
    assert ree._local_reference_path(str(pov_ref), capture_root=capture_root, job_dir=job_dir) == pov_ref
    job_relative = job_dir / "relative.json"
    job_relative.write_text("{}", encoding="utf-8")
    assert ree._local_reference_path("relative.json", capture_root=capture_root, job_dir=job_dir) == job_relative

    scenario_cards = {"cards": [{"task_id": "task-a", "scenario_id": "scenario-a"}]}
    assert ree._requested_scenarios({"requestedTasks": ["bad"]}, scenario_cards) == []
    assert ree._requested_scenarios({}, scenario_cards) == [
        {"task_id": "task-a", "scenario_id": "scenario-a"}
    ]
    assert ree._requested_scenario_eval_run_filters({"requestedScenarioEvalRuns": {"scenarioEvalRunId": "run-1"}}) == [
        {
            "scenario_eval_run_id": "run-1",
            "scenario_variation_instance_id": "",
            "variation_name": "",
            "task_id": "",
            "scenario_id": "",
            "source_followup_action_id": "",
        }
    ]
    assert ree._requested_scenario_eval_run_filters({"requestedScenarioEvalRuns": "bad"}) == []
    assert ree._requested_scenario_eval_run_filters({"requestedScenarioEvalRuns": ["bad"]}) == []
    assert ree._scenario_card_rows({"cards": "bad"}) == []
    assert ree._scenario_variation_rows_by_scenario(
        {"instances": ["bad", {"task_id": "task-a"}, {"task_id": "task-a", "scenario_id": "scenario-a"}]}
    ) == {("task-a", "scenario-a"): [{"task_id": "task-a", "scenario_id": "scenario-a"}]}

    assert ree._pose_triplet({"position": {"x": 1, "y": 2}}) == [1.0, 2.0, 0.793]
    assert ree._pose_triplet({"x": "bad", "y": 2}) is None
    assert ree._pose_triplet([1]) is None
    assert ree._pose_triplet(["bad", 2]) is None
    assert ree._first_valid_candidate(
        [{"pose": ["bad", 1]}, {"pose": [1, 2], "validated": False}, {"pose": [1, 2], "validation_status": "blocked"}]
    ) is None
    assert ree._scenario_card_spawn_target_context(None)["blockers"] == ["scenario_card_missing"]
    assert ree._mutation_pose({"spawn_pose": [1, 2]}, "spawn_pose") == ([1.0, 2.0, 0.793], "spawn_pose")
    assert ree._stable_scenario_seed({"deterministic_seed": "42"}, ordinal=1, repeat_index=0) == 42
    assert ree._attempt_video_index({"attempts": ["bad", {"scenario_id": "s", "artifact_paths": {"video": "v.mp4"}}]}) == {"s": "v.mp4"}
    assert ree._records_from_payload([{"status": "completed"}, "bad"]) == [{"status": "completed"}]
    assert ree._redact({"api_token": "secret", "nested": [{"password": "pw"}]}) == {
        "api_token": "<redacted>",
        "nested": [{"password": "<redacted>"}],
    }
    assert ree._docker_command({}) == ""
    assert ree._attestation_present("operator attests")
    assert not ree._attestation_present([])


def test_robot_eval_execution_matrix_policy_and_command_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job"
    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        robot_eval_dir / "scenario_cards.json",
        {
            "cards": [
                {
                    "task_id": "task-a",
                    "scenario_id": "scenario-a",
                    "spawn_candidates": [{"zone_id": "spawn", "pose": [0, 0, 0.8]}],
                    "target_candidates": [{"zone_id": "target", "pose": [1, 1, 0.8]}],
                }
            ]
        },
    )
    matrix = ree.build_scenario_eval_matrix(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "requested_tasks": [
                {"task_id": "unknown-task", "scenario_ids": ["unknown-scenario"]},
                {"task_id": "task-b", "scenario_ids": ["scenario-a"]},
                {"task_id": "task-a", "scenario_ids": ["scenario-a"]},
            ],
            "requested_scenario_eval_runs": {"scenario_eval_run_id": "missing-run"},
        },
        generated_at="2026-06-01T00:00:00Z",
    )
    assert "scenario_eval_matrix_unknown_requested_tasks" in matrix["blockers"]
    assert "scenario_eval_matrix_requested_task_scenario_mismatch" in matrix["blockers"]
    assert "scenario_eval_matrix_unknown_requested_eval_runs" in matrix["blockers"]
    assert matrix["missing_variation_scenarios"] == ["scenario-a"]

    attempts = ree._normalize_policy_attempts(
        payload={"status": "completed"},
        modality="recorded_action_trace",
        observations=[],
        generated_at="2026-06-01T00:00:00Z",
    )
    assert attempts[0]["observation_id"] == "observation_1"
    assert ree._normalize_policy_attempts(
        payload=[{"status": "failed", "success": False}],
        modality="teleop_demo",
        observations=[{"observation_id": "obs-1"}],
        generated_at="2026-06-01T00:00:00Z",
    )[0]["success"] is False
    expanded = ree._normalize_policy_attempts(
        payload={},
        modality="high_level_skill_trace",
        observations=[{"observation_id": "obs-1"}, {"observation_id": "obs-2"}],
        generated_at="2026-06-01T00:00:00Z",
    )
    assert [attempt["observation_id"] for attempt in expanded] == ["obs-1", "obs-2"]

    status, payload, detail = ree._run_command(
        command_text="/definitely/missing/blueprint-command",
        output_path=tmp_path / "policy.json",
        observation_manifest_path=tmp_path / "observations.json",
        modality="recorded_action_trace",
        timeout_seconds=1,
    )
    assert status == "blocked"
    assert payload is None
    assert detail["blockers"] == ["missing_policy_command_dependency"]

    def raise_timeout(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise subprocess.TimeoutExpired("policy", 1, output="slow")

    monkeypatch.setattr(ree.subprocess, "run", raise_timeout)
    timeout_status, _, timeout_detail = ree._run_command(
        command_text="python",
        output_path=tmp_path / "timeout.json",
        observation_manifest_path=tmp_path / "observations.json",
        modality="recorded_action_trace",
        timeout_seconds=1,
    )
    assert timeout_status == "failed"
    assert timeout_detail["blockers"] == ["policy_command_timeout"]

    class Response:
        status = 202

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"attempts":[{"status":"completed"}]}'

    monkeypatch.setattr(
        ree,
        "fetch_bounded_https",
        lambda *args, **kwargs: type(
            "BoundedResponse",
            (),
            {
                "body": b'{"attempts":[{"status":"completed"}]}',
                "status": 202,
            },
        )(),
    )
    api_status, api_payload, api_detail = ree._call_policy_api(
        endpoint="https://policy.example",
        observation_manifest={"observations": []},
        timeout_seconds=1,
    )
    assert api_status == "completed"
    assert api_payload["attempts"][0]["status"] == "completed"
    assert api_detail["http_status"] == 202
    monkeypatch.setattr(
        ree,
        "fetch_bounded_https",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("down")),
    )
    failed_api = ree._call_policy_api(
        endpoint="https://policy.example",
        observation_manifest={"observations": []},
        timeout_seconds=1,
    )
    assert failed_api[0] == "failed"

    observation_manifest = {"observations": [{"observation_id": "obs-1", "scenario_eval_run_id": "run-1"}]}
    blocked = ree.build_policy_execution_bundle(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={"use_default_test_policy": True},
        observation_manifest=observation_manifest,
        allow_policy_execution=False,
        generated_at="2026-06-01T00:00:00Z",
    )
    assert blocked["manifest"]["modality_results"]["high_level_skill_trace"]["status"] == "blocked_policy_execution_gate"
    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    completed = ree.build_policy_execution_bundle(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={"use_default_test_policy": True},
        observation_manifest=observation_manifest,
        allow_policy_execution=True,
        generated_at="2026-06-01T00:00:00Z",
    )
    assert completed["manifest"]["modality_results"]["high_level_skill_trace"]["status"] == "completed"
    missing_manipulation_target = ree.build_policy_execution_bundle(
        capture_root=capture_root,
        job_dir=tmp_path / "job-missing-manipulation-target",
        job_request={
            "default_test_policy": {
                "policy_kind": "mobile_manipulation_pick_carry_place",
            }
        },
        observation_manifest=observation_manifest,
        allow_policy_execution=True,
        generated_at="2026-06-01T00:00:00Z",
    )
    missing_target_result = missing_manipulation_target["manifest"]["modality_results"][
        "high_level_skill_trace"
    ]
    assert missing_target_result["status"] == "blocked_missing_policy_execution_trace"
    assert missing_target_result["default_test_policy_execution_proven"] is False
    assert missing_target_result["detail"]["blockers"] == [
        "default_manipulation_policy_object_id_missing"
    ]


def test_robot_eval_execution_simulator_and_actual_outcome_edges(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "batch_manifest.json").write_text("{}", encoding="utf-8")
    (source_dir / "attempts.jsonl").write_text("{}\n", encoding="utf-8")
    (job_dir / "scenario_eval_matrix.json").parent.mkdir(parents=True, exist_ok=True)
    (job_dir / "scenario_eval_matrix.json").write_text("{", encoding="utf-8")

    artifacts = ree.build_simulator_command_artifacts(
        job_dir=job_dir,
        simulator="mujoco",
        simulator_output={
            "missing_scenario_eval_run_ids": ["run-missing"],
            "artifact_paths": {
                "batch_trace_package_manifest": str(source_dir / "batch_manifest.json"),
                "digital_twin_fidelity_qa": "gs://remote/qa.json",
            },
            "batch_trace_package": {
                "artifact_paths": {
                    "attempt_trace_jsonl": "attempts.jsonl",
                    "contact_stream_jsonl": "gs://remote/contact.jsonl",
                    "planner_state_jsonl": "missing-planner.jsonl",
                }
            },
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_eval_run_id": "run-1",
                    "scenario_id": "scenario-a",
                    "task_id": "task-a",
                    "status": "failed",
                    "success": False,
                    "task_outcome": {
                        "goal_reached": False,
                        "endpoint_clean": False,
                        "timeout": True,
                        "fall_detected": True,
                        "stuck_detected": True,
                        "policy_instability_detected": True,
                        "clearance_threshold_violation": True,
                        "near_miss_event_count": 2,
                        "final_target_error_m": 1.2,
                        "max_path_deviation_m": 0.5,
                        "min_clearance_m": 0.1,
                        "clearance_threshold_m": 0.2,
                        "robot_scene_contact_event_count": 1,
                    },
                    "failure_reason": "timeout",
                    "artifact_paths": {
                        "scene_trace": "scene.json",
                        "spawn_trace": "spawn.json",
                        "policy_trace": "policy.json",
                        "sim_robot_pov_evidence": "pov.json",
                        "frames": ["f1.png", "f2.png", "f3.png", "f4.png"],
                    },
                }
            ],
        },
        generated_at="2026-06-01T00:00:00Z",
    )
    assert artifacts["normalized_attempt_trace"]["status"] == "completed"
    assert artifacts["normalized_attempt_trace"]["missing_scenario_eval_run_ids"] == ["run-missing"]
    assert artifacts["failure_labels"]["labels"][0]["primary_failure_mode"] == "timeout"
    assert artifacts["visual_review_ledger"]["records"][0]["media_evidence_present"] is True
    assert artifacts["simulator_command_batch_trace_package_manifest"]["job_artifact_copy_status"] == "partial_or_missing"
    assert artifacts["simulator_command_batch_trace_package_manifest"]["job_artifact_copy_records"]["attempt_trace_jsonl"]["status"] == "copied"
    assert artifacts["manifest"]["simulator_command_digital_twin_fidelity_qa_copy_record"]["status"] == "remote_source_not_copied"
    trace = artifacts["normalized_attempt_trace"]
    assert trace["task_success_label_provenance_counts"] == {
        "simulator_trace_or_physics": 1
    }
    assert trace["success_rate_provenance_disclosed"] is True
    assert trace["success_rate_buyer_display_allowed"] is True
    assert trace["attempts"][0]["task_success_label_provenance"][
        "provenance_type"
    ] == "simulator_trace_or_physics"

    generated_video_label = ree.build_simulator_command_artifacts(
        job_dir=tmp_path / "job-generated-video-label",
        simulator="wam",
        simulator_output={
            "attempts": [
                {
                    "attempt_id": "wam-attempt-1",
                    "status": "completed",
                    "success": True,
                    "task_success": True,
                    "success_label_source": "openai_generated_video_frame_judge",
                    "wam_success_label_from_generated_video": True,
                    "artifact_paths": {"video_path": "generated-rollout.mp4"},
                }
            ],
        },
        generated_at="2026-06-01T00:00:00Z",
    )
    generated_trace = generated_video_label["normalized_attempt_trace"]
    assert generated_trace["task_success_label_provenance_counts"] == {
        "generated_video_vlm_judge": 1
    }
    assert generated_trace["generated_video_vlm_judged_attempt_count"] == 1
    assert generated_trace["success_rate_buyer_display_allowed"] is True
    assert "model-derived generated rollout video" in generated_trace[
        "attempts"
    ][0]["task_success_label_provenance"]["buyer_disclosure"]

    blocked_coverage = ree.build_simulator_command_artifacts(
        job_dir=tmp_path / "job-blocked-coverage",
        simulator="mujoco",
        simulator_output={
            "required_scenario_eval_run_ids": ["run-missing"],
            "attempts": [{"attempt_id": "attempt-1", "scenario_eval_run_id": "run-1"}],
        },
        generated_at="2026-06-01T00:00:00Z",
    )
    assert blocked_coverage["normalized_attempt_trace"]["status"] == (
        "blocked_incomplete_scenario_eval_run_coverage"
    )

    blocked_execution = ree.build_simulator_command_artifacts(
        job_dir=tmp_path / "job-blocked-execution",
        simulator="isaac_sim",
        simulator_output={
            "simulator_execution_proven": False,
            "required_scenario_eval_run_ids": ["run-1"],
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_eval_run_id": "run-1",
                    "status": "blocked",
                    "success": False,
                    "failure_reason": "isaac_runtime_or_authorized_gpu_unavailable",
                }
            ],
        },
        generated_at="2026-06-01T00:00:00Z",
    )
    assert blocked_execution["normalized_attempt_trace"]["status"] == (
        "blocked_simulator_execution_not_proven"
    )
    assert blocked_execution["normalized_attempt_trace"]["simulator_execution_proven"] is False
    assert blocked_execution["manifest"]["simulator_execution_proven"] is False

    missing_qa = ree.build_simulator_command_artifacts(
        job_dir=tmp_path / "job-missing-qa",
        simulator="mujoco",
        simulator_output={"artifact_paths": {"digital_twin_fidelity_qa": "missing-qa.json"}, "attempts": []},
        generated_at="2026-06-01T00:00:00Z",
    )
    assert missing_qa["manifest"]["simulator_command_digital_twin_fidelity_qa_copy_record"]["status"] == "missing_source_file"

    capture_root = _capture_root(tmp_path / "capture")
    outcome_ref = tmp_path / "outcomes.json"
    _write_json(outcome_ref, {"records": [{"actual_status": "passed"}]})
    loaded, source = ree._load_actual_outcome_payload(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={"actualOutcomeManifestUri": str(outcome_ref)},
    )
    assert ree._records_from_payload(loaded)[0]["actual_status"] == "passed"
    assert source == "job_request_outcome_manifest_ref"

    inbox = job_dir / "actual_outcomes" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / ".hidden.json").write_text("{}", encoding="utf-8")
    (inbox / "bad.json").write_text("{", encoding="utf-8")
    _write_json(inbox / "empty.json", {"records": []})
    _write_json(inbox / "ready.json", {"records": [{"actual_status": "failed"}]})
    inbox_payload = ree._load_actual_outcome_inbox(capture_root=capture_root, job_dir=job_dir)
    assert inbox_payload["status"] == "ready"
    assert len(inbox_payload["blockers"]) == 2

    predictions = ree._prediction_index(
        {"records": ["bad", {"task_id": "task-a", "scenario_id": "scenario-a", "predicted_status": "failed"}]},
        {"attempts": ["bad", {"task_id": "task-b", "scenario_id": "scenario-b", "scenario_eval_run_id": "run-b"}]},
    )
    prediction, level = ree._prediction_for_actual(
        predictions,
        task_id="task-a",
        scenario_id="scenario-a",
        scenario_eval_run_id="",
        scenario_variation_instance_id="",
    )
    assert level == "task_scenario_fallback"
    assert ree._predicted_success(prediction) is False
    assert ree._predicted_success({"failure_mode_ids": ["collision"]}) is False
    assert ree._actual_success({"actual_status": "completed"}) is True
    assert ree._actual_success({"status": "collision"}) is False
    assert ree._actual_signal_present({"failure_mode_ids": ["collision"]})
    assert ree._predicted_success({"predicted_status": "passed"}) is True

    followup = ree._build_real_world_validation_followup_plan(
        rows=[
            {
                "record_id": "row-1",
                "matched_prediction": False,
                "actual_success": False,
                "actual_result_signal_present": False,
                "missed_failures": ["collision"],
                "real_world_tuning_needed": True,
                "tuning_hours": 1.5,
                "tuning_iterations": 2,
                "tuning_notes": ["tighten policy"],
                "site_modifications": {"not": "a-list"},
            }
        ],
        generated_at="2026-06-01T00:00:00Z",
        outcome_source="fixture",
        calibration_status="review_required",
    )
    assert followup["summary"]["scenario_rerun_count"] == 1
    assert followup["summary"]["robot_team_tuning_review_count"] == 1
    assert followup["summary"]["site_modification_review_count"] == 0


def test_robot_eval_execution_remaining_policy_command_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    job_dir = tmp_path / "job-remaining"
    observation_manifest = {"observations": [{"observation_id": "obs-1", "scenario_eval_run_id": "run-1"}]}

    original_import = builtins.__import__

    def import_without_pil(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL":
            raise ImportError("no pillow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pil)
    assert ree._write_observation_png(tmp_path / "obs.png", ["line"]) is False
    monkeypatch.setattr(builtins, "__import__", original_import)

    def completed_invalid_json(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="{bad", stderr="")

    monkeypatch.setattr(ree.subprocess, "run", completed_invalid_json)
    invalid_status, invalid_payload, invalid_detail = ree._run_command(
        command_text="python -c pass",
        output_path=tmp_path / "invalid-policy-output.json",
        observation_manifest_path=tmp_path / "observations.json",
        modality="recorded_action_trace",
        timeout_seconds=1,
    )
    assert invalid_status == "failed"
    assert invalid_payload is None
    assert invalid_detail["blockers"] == ["policy_command_exit:0"]

    command_blocked = ree.build_policy_execution_bundle(
        capture_root=capture_root,
        job_dir=job_dir,
        job_request={
            "policy_package": {
                "recorded_action_trace": {
                    "execution_command": "python -c pass",
                    "token": "secret",
                }
            }
        },
        observation_manifest=observation_manifest,
        allow_policy_execution=False,
        generated_at="2026-06-01T00:00:00Z",
    )
    recorded = command_blocked["manifest"]["modality_results"]["recorded_action_trace"]
    assert recorded["status"] == "blocked_policy_execution_gate"
    assert recorded["launch_reviewable_without_execution"] is True
    assert "recorded_action_trace" in command_blocked["manifest"]["reviewable_policy_adapter_modes"]
    assert (
        command_blocked["manifest"]["policy_adapter_pack_contract"][
            "same_observation_action_contract_for_all_modes"
        ]
        is True
    )
    assert command_blocked["manifest"]["policy_adapter_pack_contract"][
        "execution_claim_requires_gated_policy_execution"
    ] is True
    assert recorded["reference"]["token"] == "<redacted>"

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"attempts": [{"scenario_eval_run_id": "run-1", "status": "completed"}]}).encode(
                "utf-8"
            )

    monkeypatch.setenv("BLUEPRINT_ALLOW_POLICY_EXECUTION", "true")
    monkeypatch.setattr(
        ree,
        "fetch_bounded_https",
        lambda *args, **kwargs: type(
            "BoundedResponse",
            (),
            {
                "body": json.dumps(
                    {"attempts": [{"scenario_eval_run_id": "run-1", "status": "completed"}]}
                ).encode("utf-8"),
                "status": 200,
            },
        )(),
    )
    api_bundle = ree.build_policy_execution_bundle(
        capture_root=capture_root,
        job_dir=tmp_path / "job-api",
        job_request={"policy_package": {"policy_api_endpoint": {"endpoint_url": "https://policy.example"}}},
        observation_manifest=observation_manifest,
        allow_policy_execution=True,
        generated_at="2026-06-01T00:00:00Z",
    )
    assert api_bundle["manifest"]["modality_results"]["policy_api_endpoint"]["status"] == "completed"

    manipulation_payload = ree._default_test_policy_execution_payload(
        payload={
            "policy_kind": "mobile_manipulation_pick_carry_place",
            "object_id": "tote-9",
        },
        observations=[
            {
                "observation_id": "obs-1",
                "scenario_id": "scenario-a",
                "scenario_eval_run_id": "run-1",
                "scenario_variation_instance_id": "variation-1",
                "variation_name": "clutter",
                "task_id": "task-a",
            }
        ],
    )
    assert manipulation_payload["status"] == "completed"
    assert manipulation_payload["attempts"][0]["policy_kind"] == "mobile_manipulation_pick_carry_place"

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "batch_manifest.json").write_text("{}", encoding="utf-8")
    (source_dir / "qa.json").write_text("{}", encoding="utf-8")
    copied_qa = ree.build_simulator_command_artifacts(
        job_dir=tmp_path / "job-copied-qa",
        simulator="mujoco",
        simulator_output={
            "artifact_paths": {
                "batch_trace_package_manifest": str(source_dir / "batch_manifest.json"),
                "digital_twin_fidelity_qa": "qa.json",
            },
            "attempts": [{"attempt_id": "attempt-1", "scenario_eval_run_id": "run-1"}],
        },
        generated_at="2026-06-01T00:00:00Z",
    )
    assert copied_qa["manifest"]["simulator_command_digital_twin_fidelity_qa_copy_record"]["status"] == "copied"

    _write_json(
        capture_root / "pipeline" / "robot_eval_dataset" / "scenario_cards.json",
        {"cards": [{"task_id": "task-a", "scenario_id": "scenario-a"}]},
    )
    duplicate_filter_matrix = ree.build_scenario_eval_matrix(
        capture_root=capture_root,
        job_dir=tmp_path / "job-duplicate-filter",
        job_request={
            "requested_tasks": [{"task_id": "task-a", "scenario_ids": ["scenario-a"]}],
            "requested_scenario_eval_runs": [
                {"scenario_id": "scenario-a"},
                {"scenario_id": "scenario-a"},
            ],
        },
        generated_at="2026-06-01T00:00:00Z",
    )
    assert duplicate_filter_matrix["scenario_eval_run_count"] == 1
