from __future__ import annotations

import json
import sys
from pathlib import Path

from blueprint_pipeline.arena_package_audit import build_arena_package_proof_boundary_audit
from blueprint_pipeline.arena_package_delivery_local import build_local_delivery_command_manifest
from blueprint_pipeline.arena_result_ingest import build_arena_result_ingest
from blueprint_pipeline.rollout_vision_label_openai import build_openai_rollout_vision_labels


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


def test_arena_result_ingest_consumes_review_required_vision_command_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    results_dir = _arena_results(tmp_path)
    output_dir = tmp_path / "arena-package"
    writer = tmp_path / "write_vision_labels.py"
    writer.write_text(
        "\n".join(
            [
                "import json",
                "payload = {",
                "  'schema_version': 'arena_rollout_vision_command_labels.v1',",
                "  'provider': 'fake-command',",
                "  'model': 'fake-vision-model',",
                "  'visual_evidence_used': True,",
                "  'labels': [{",
                "    'vision_label_id': 'vision-command-1',",
                "    'source_failure_label_id': 'failure_episode-1',",
                "    'attempt_id': 'episode-1',",
                "    'status': 'accepted',",
                "    'object_state': 'object partially occluded',",
                "    'contact': 'review_required',",
                "    'occlusion': 'present',",
                "    'threshold_miss': True,",
                "    'failure_evidence': ['occlusion_threshold_miss'],",
                "    'label_source': 'fake-command',",
                "    'visual_evidence_used': True,",
                "  }],",
                "}",
                "with open('rollout_vision_labels.command.json', 'w', encoding='utf-8') as f:",
                "    json.dump(payload, f)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", "true")

    build_arena_result_ingest(
        capture_root=capture_root,
        arena_results_dir=results_dir,
        output_dir=output_dir,
        scenario_count=500,
        shard_size=100,
        allow_rollout_vision_labeling=True,
        vision_labeling_command=f"{sys.executable} {writer}",
    )

    vision = _read_json(output_dir / "rollout_vision_labels.json")

    assert vision["status"] == "completed_review_required"
    assert vision["vision_model_labeling_performed"] is True
    assert vision["command_labels"]["status"] == "completed"
    assert vision["command_labels"]["provider"] == "fake-command"
    assert vision["labels"][0]["vision_label_id"] == "vision-command-1"
    assert vision["labels"][0]["status"] == "review_required"
    assert vision["labels"][0]["label_source"] == "fake-command"
    assert vision["claim_boundary"]["robot_readiness_proven"] is False
    assert vision["claim_boundary"]["vision_model_labeling_performed"] is True


def test_openai_rollout_vision_labeler_fails_closed_without_gate_key_or_keyframes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_json(
        tmp_path / "failure_labels.json",
        {
            "labels": [
                {
                    "label_id": "failure-1",
                    "attempt_id": "attempt-1",
                    "threshold_miss": True,
                    "failure_categories": ["threshold_miss"],
                }
            ]
        },
    )
    _write_json(
        tmp_path / "clips_manifest.json",
        {
            "clips": [
                {
                    "clip_id": "clip-attempt-1",
                    "attempt_id": "attempt-1",
                    "clip_path": "clips/missing.mp4",
                }
            ]
        },
    )

    result = build_openai_rollout_vision_labels(output_dir=tmp_path)

    assert result["status"] == "blocked_review_required"
    assert "missing_env_BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING" in result["blockers"]
    assert "missing_openai_api_key" in result["blockers"]
    assert "missing_visual_evidence_keyframes" in result["blockers"]
    assert result["label_count"] == 0
    assert result["public_claim_upgrade_allowed"] is False
    assert (tmp_path / "rollout_vision_labels.command.json").is_file()


def test_arena_result_ingest_consumes_local_delivery_command_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    results_dir = _arena_results(tmp_path)
    output_dir = tmp_path / "arena-package"
    delivery_root = tmp_path / "local-delivery-root"
    writer = tmp_path / "write_delivery_manifest.py"
    writer.write_text(
        "\n".join(
            [
                "import json, pathlib, shutil",
                f"delivery_root = pathlib.Path({str(delivery_root)!r})",
                "bundle = pathlib.Path('delivery_bundle')",
                "target = delivery_root / pathlib.Path.cwd().name",
                "target.mkdir(parents=True, exist_ok=True)",
                "paths = []",
                "for source in sorted(bundle.rglob('*')):",
                "    if not source.is_file():",
                "        continue",
                "    rel = source.relative_to(bundle)",
                "    dest = target / rel",
                "    dest.parent.mkdir(parents=True, exist_ok=True)",
                "    shutil.copy2(source, dest)",
                "    paths.append({'relative_path': str(rel), 'delivered_path': str(dest), 'size_bytes': dest.stat().st_size})",
                "payload = {",
                "  'schema_version': 'arena_delivery_command_manifest.v1',",
                "  'status': 'local_delivery_ready_review_required',",
                "  'provider': 'local_filesystem',",
                "  'delivery_root': str(delivery_root),",
                "  'signed_urls': [],",
                "  'local_access_paths': paths,",
                "  'storage_upload_performed': False,",
                "}",
                "with open('delivery_upload.command.json', 'w', encoding='utf-8') as f:",
                "    json.dump(payload, f)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "true")
    monkeypatch.setenv("BLUEPRINT_LOCAL_DELIVERY_ROOT", str(delivery_root))

    build_arena_result_ingest(
        capture_root=capture_root,
        arena_results_dir=results_dir,
        output_dir=output_dir,
        scenario_count=500,
        shard_size=100,
        allow_delivery_upload=True,
        delivery_command=f"{sys.executable} {writer}",
    )

    signed_access = _read_json(output_dir / "signed_access_manifest.json")
    command_manifest = _read_json(output_dir / "delivery_upload.command.json")

    assert signed_access["status"] == "local_delivery_ready_review_required"
    assert "signed_urls_not_provided_by_local_delivery_command" in signed_access["blockers"]
    assert signed_access["signed_urls"] == []
    assert signed_access["local_access_paths"]
    assert signed_access["storage_upload_performed"] is False
    assert command_manifest["status"] == "local_delivery_ready_review_required"
    assert command_manifest["storage_upload_performed"] is False
    assert (delivery_root / output_dir.name / "customer_handoff_report.md").is_file()


def test_local_delivery_command_fails_closed_without_gate_or_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", raising=False)
    monkeypatch.delenv("BLUEPRINT_LOCAL_DELIVERY_ROOT", raising=False)

    result = build_local_delivery_command_manifest(output_dir=tmp_path)

    assert result["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD" in result["blockers"]
    assert "missing_env_BLUEPRINT_LOCAL_DELIVERY_ROOT" in result["blockers"]
    assert "missing_delivery_bundle" in result["blockers"]
    assert result["storage_upload_performed"] is False
    assert (tmp_path / "delivery_upload.command.json").is_file()
