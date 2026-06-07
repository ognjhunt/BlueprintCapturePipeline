from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.arena_package_audit import build_arena_package_proof_boundary_audit
from blueprint_pipeline.arena_result_ingest import build_arena_result_ingest


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    robot_eval = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(robot_eval / "site_card.json", {"site_id": "site-1"})
    _write_json(
        robot_eval / "task_cards.json",
        {"cards": [{"task_id": "task-1", "task_statement": "Move a tote"}]},
    )
    _write_json(
        robot_eval / "scenario_cards.json",
        {
            "cards": [
                {
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "robot_profile_id": "robot-1",
                }
            ]
        },
    )
    _write_json(robot_eval / "eval_cards.json", {"cards": [{"scenario_id": "scenario-1"}]})
    _write_json(robot_eval / "proof_boundaries.json", {"robot_readiness_proven": False})
    return capture_root


def _arena_results(tmp_path: Path) -> Path:
    results_dir = tmp_path / "arena-results"
    (results_dir / "videos").mkdir(parents=True, exist_ok=True)
    (results_dir / "videos" / "episode.mp4").write_bytes(b"fake video")
    _write_json(
        results_dir / "rollout_manifest.json",
        {
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "scenario_id": "scenario-1",
                    "scenario_run_id": "scenario-1__arena_run_0001",
                    "status": "failed",
                    "success": False,
                    "failure_reason": "occlusion_threshold_miss",
                    "metrics": {"cycle_time_seconds": 42.0},
                    "video_path": "videos/episode.mp4",
                }
            ]
        },
    )
    return results_dir


def test_arena_result_ingest_writes_package_and_blocks_live_gates(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    results_dir = _arena_results(tmp_path)
    output_dir = tmp_path / "arena-package"

    result = build_arena_result_ingest(
        capture_root=capture_root,
        arena_results_dir=results_dir,
        output_dir=output_dir,
        job_request={
            "policy_package": {
                "policy_api_endpoint": {"endpoint_url": "https://robot.example/policy"}
            }
        },
        scenario_count=500,
        shard_size=100,
        allow_rollout_vision_labeling=True,
        vision_labeling_command="fake-rollout-labeler",
        allow_delivery_upload=True,
        delivery_command="fake-delivery-upload",
        operator_mode="agents-sdk",
        allow_live_agents_sdk=True,
        allow_live_codex_sdk=True,
    )

    schedule = _read_json(output_dir / "arena_eval_schedule.json")
    trace = _read_json(output_dir / "normalized_attempt_trace.json")
    vision = _read_json(output_dir / "rollout_vision_labels.json")
    signed_access = _read_json(output_dir / "signed_access_manifest.json")
    operators = _read_json(output_dir / "live_operator_ledger.json")
    package = _read_json(output_dir / "post_training_data_package_export_manifest.json")

    assert result["status"] == "completed"
    assert schedule["scenario_count"] == 500
    assert schedule["shard_count"] == 5
    assert trace["attempt_count"] == 1
    assert vision["status"] == "blocked_review_required"
    assert "missing_env_BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING" in vision["blockers"]
    assert signed_access["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD" in signed_access["blockers"]
    assert operators["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS" in operators["blockers"]
    assert "missing_openai_api_key" in operators["blockers"]
    assert package["status"] == "export_ready_review_required"
    assert (output_dir / "dataset_card.json").is_file()
    assert (output_dir / "archives" / "post_training_data_package.tar.gz").is_file()

    audit = build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=output_dir,
        expected_scenario_count=500,
    )
    assert audit["status"] == "passed"
    assert audit["summary"]["attempt_count"] == 1
    assert audit["summary"]["clip_count"] == 1
    assert audit["proof_boundary_violations"] == []
    assert (output_dir / "arena_package_proof_boundary_audit.json").is_file()


def test_arena_package_audit_blocks_illegal_proof_upgrade(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    results_dir = _arena_results(tmp_path)
    output_dir = tmp_path / "arena-package"
    build_arena_result_ingest(
        capture_root=capture_root,
        arena_results_dir=results_dir,
        output_dir=output_dir,
        scenario_count=500,
        shard_size=100,
    )
    evaluation_result = _read_json(output_dir / "arena_result_ingest_run_manifest.json")
    evaluation_result["robot_readiness_proven"] = True
    _write_json(output_dir / "arena_result_ingest_run_manifest.json", evaluation_result)

    audit = build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=output_dir,
        expected_scenario_count=500,
    )

    assert audit["status"] == "blocked"
    assert "proof_boundary_violation:arena_result_ingest_run_manifest.json:robot_readiness_proven" in audit[
        "blockers"
    ]
