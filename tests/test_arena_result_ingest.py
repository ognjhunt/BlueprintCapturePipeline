from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import blueprint_pipeline.arena_result_ingest as arena
from blueprint_pipeline.arena_package_audit import build_arena_package_proof_boundary_audit
from blueprint_pipeline.arena_package_delivery_local import build_local_delivery_command_manifest
from blueprint_pipeline.arena_result_ingest import build_arena_result_ingest
from blueprint_pipeline.rollout_vision_label_openai import build_openai_rollout_vision_labels


pytestmark = [pytest.mark.slow, pytest.mark.integration]


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
    _write_json(robot_eval / "proof_boundaries.json", {"rank_fidelity_result_proven": False})
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
                    "clip_curation": {
                        "evidence_source": "synthetic_fixture_declared",
                        "frame_count": 32,
                        "camera_motion_m": 0.01,
                        "action_motion_score": 0.5,
                        "visible_skeleton_fraction": 0.9,
                        "sharpness_score": 40.0,
                        "semantic_dedup_key": "scenario-1|move-tote|episode-1",
                    },
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
    buyer_report = _read_json(output_dir / "task_eval_run_report.json")
    assert buyer_report["schema_version"] == "task_eval_run_buyer_report.v1"
    # local file ingest carries no layer contracts, so the ledger must
    # truthfully report no_claim and never a bare success boolean
    assert buyer_report["evidence_level"] == "no_claim"
    assert "success" not in buyer_report
    assert buyer_report["success_claim_ledger"]["highest_truthful_claim"] == "no_claim"
    condition = buyer_report["scorecard"]["conditions"][0]
    assert condition["trials"] == 1
    assert condition["successes"] == 0
    # no_claim withholds the numeric success_rate (only the factual counts stand);
    # publishing a rate + "completed" here would over-claim a run with no grounding.
    assert condition["success_rate"] is None
    assert buyer_report["scorecard"]["status"] == "rates_withheld_insufficient_evidence"
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
    evaluation_result["rank_fidelity_result_proven"] = True
    _write_json(output_dir / "arena_result_ingest_run_manifest.json", evaluation_result)

    audit = build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=output_dir,
        expected_scenario_count=500,
    )

    assert audit["status"] == "blocked"
    assert "proof_boundary_violation:arena_result_ingest_run_manifest.json:rank_fidelity_result_proven" in audit[
        "blockers"
    ]


def test_arena_package_audit_allows_live_closure_backed_proof_upgrade(tmp_path: Path) -> None:
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
    closure_boundary = {
        "live_end_to_end_verified": True,
        "simulator_execution_proven": True,
        "robot_policy_execution_proven": True,
        "rank_fidelity_result_proven": True,
        "physics_contact_validated": True,
        "non_ranking_operational_claim_validated": True,
        "public_claim_upgrade_allowed": True,
    }
    _write_json(
        output_dir / "live_eval_closure_manifest.json",
        {
            "schema_version": "live_robot_eval_closure_manifest.v1",
            "status": "live_end_to_end_verified",
            "live_end_to_end_verified": True,
            "blockers": [],
            "proof_boundary": closure_boundary,
        },
    )
    for name, payload in {
        "job_request.json": {"schema_version": "robot_eval_job_request.v1"},
        "simulator_service_result.json": {"simulator_execution_proven": True},
        "evaluation_result.json": {"rank_fidelity_result_proven": True},
        "proof_boundary.json": dict(closure_boundary),
        "job_run_manifest.json": dict(closure_boundary),
    }.items():
        _write_json(output_dir / name, payload)

    audit = build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=output_dir,
        expected_scenario_count=500,
        require_job_artifacts=True,
    )

    assert audit["status"] == "passed"
    assert audit["proof_boundary_violations"] == []


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
    assert vision["claim_boundary"]["rank_fidelity_result_proven"] is False
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


def test_arena_result_ingest_small_helper_edges(tmp_path: Path) -> None:
    assert arena._string_list(None) == []
    assert arena._string_list("one") == ["one"]
    assert arena._string_list(7) == ["7"]
    assert arena._boolish("passed") is True
    assert arena._read_optional_json(tmp_path / "missing.json") is None
    assert arena._artifact_ref(tmp_path, tmp_path / "missing.bin")["exists"] is False
    assert arena._cards({"scenarios": [{"scenario_id": "scenario-1"}, "skip"]}) == [
        {"scenario_id": "scenario-1"}
    ]
    assert arena._cards({}) == []
    assert arena._load_scenario_cards(tmp_path / "pipeline")[0]["scenario_id"] == (
        "arena_placeholder_scenario"
    )
    assert arena._redact({"api_key": "secret", "items": [{"token": "hidden"}]}) == {
        "api_key": "<redacted>",
        "items": [{"token": "<redacted>"}],
    }

    modality_payloads = {
        "policy_api_endpoint": {"endpoint_url": ""},
        "docker_container": {"image_ref": "", "digest": "latest"},
        "recorded_action_trace": {"trace_manifest_uri": ""},
        "high_level_skill_trace": {"ordered_skill_sequence": []},
        "teleop_demo": {"demo_artifact_uri": ""},
        "sim_controller_plugin": {"plugin_uri": ""},
    }
    missing_by_modality = {
        modality: arena._policy_adapter_status(modality, payload)[1]
        for modality, payload in modality_payloads.items()
    }
    assert missing_by_modality["policy_api_endpoint"] == ["endpoint_url"]
    assert missing_by_modality["docker_container"] == ["image_ref", "digest"]
    assert missing_by_modality["recorded_action_trace"] == ["trace_manifest_uri"]
    assert missing_by_modality["high_level_skill_trace"] == ["ordered_skill_sequence"]
    assert missing_by_modality["teleop_demo"] == ["demo_artifact_uri"]
    assert missing_by_modality["sim_controller_plugin"] == ["plugin_uri"]

    assert arena._candidate_json_files(tmp_path / "not-a-dir") == []
    results_dir = tmp_path / "results"
    _write_json(results_dir / "review_resolutions.json", {"ignored": True})
    (results_dir / "null.json").write_text("null", encoding="utf-8")
    _write_json(results_dir / "single.json", {"episode_id": "episode-single"})
    _write_json(results_dir / "list.json", [{"episode_id": "episode-list"}])
    records, blockers = arena._extract_episode_records(results_dir)
    assert blockers == []
    assert {record["episode_id"] for record in records} == {"episode-single", "episode-list"}
    assert arena._extract_episode_records(tmp_path / "missing-results")[1] == [
        "arena_results_dir_missing"
    ]

    normalized = arena._normalize_attempts(
        records=[
            {"episode_id": "passed", "status": "passed", "metrics": {}},
            {"episode_id": "failed", "status": "failed", "metrics": {}},
        ],
        results_dir=results_dir,
        generated_at="now",
    )
    by_episode = {attempt["episode_id"]: attempt for attempt in normalized["attempts"]}
    assert by_episode["passed"]["success"] is True
    assert by_episode["failed"]["failure_reason"] == "threshold_miss_or_failed_status"

    failure_labels = arena._build_failure_labels(
        {"attempts": ["skip", {"attempt_id": "ok", "success": True}]},
        generated_at="now",
    )
    assert failure_labels["label_count"] == 0
    assert "metric_out_of_bounds" in arena._failure_categories(
        "threshold miss",
        {"score_delta": -1.0},
    )
    assert arena._copy_clip_source(tmp_path / "missing.mp4", tmp_path / "clip.mp4") is False
    clips = arena._build_clips_manifest(
        attempt_trace={"attempts": ["skip"]},
        output_dir=tmp_path / "clips-out",
        generated_at="now",
    )
    assert clips["clip_count"] == 0


def test_arena_result_ingest_command_output_and_delivery_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_command = arena._run_optional_command(
        "/definitely/missing/blueprint-arena-command",
        timeout_seconds=1,
        cwd=tmp_path,
    )
    assert missing_command["reason"] == "missing_command_dependency"

    def raise_timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(
            cmd=["blueprint-arena-command"],
            timeout=1,
            output="stdout",
            stderr="stderr",
        )

    monkeypatch.setattr(arena.subprocess, "run", raise_timeout)
    timed_out = arena._run_optional_command(
        "blueprint-arena-command",
        timeout_seconds=1,
        cwd=tmp_path,
    )
    assert timed_out["reason"] == "timeout"

    assert arena._load_command_vision_labels(tmp_path / "missing-vision")["status"] == "missing"
    vision_not_object = tmp_path / "vision-not-object"
    vision_not_object.mkdir()
    (vision_not_object / "rollout_vision_labels.command.json").write_text("[]", encoding="utf-8")
    assert arena._load_command_vision_labels(vision_not_object)["blockers"] == [
        "vision_command_output_not_object"
    ]
    vision_missing_labels = tmp_path / "vision-missing-labels"
    _write_json(vision_missing_labels / "rollout_vision_labels.command.json", {"provider": "x"})
    assert arena._load_command_vision_labels(vision_missing_labels)["blockers"] == [
        "vision_command_output_missing_labels"
    ]

    monkeypatch.setenv("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", "true")
    vision_cli_missing = arena._build_vision_labels(
        failure_labels={"labels": [{"label_id": "failure-1", "attempt_id": "attempt-1"}]},
        output_dir=tmp_path / "vision-cli-missing",
        allow_vision_labeling=False,
        vision_labeling_command="labeler",
        timeout_seconds=1,
        generated_at="now",
    )
    assert "missing_cli_allow_rollout_vision_labeling" in vision_cli_missing["blockers"]
    vision_command_missing = arena._build_vision_labels(
        failure_labels={"labels": [{"label_id": "failure-1", "attempt_id": "attempt-1"}]},
        output_dir=tmp_path / "vision-command-missing",
        allow_vision_labeling=True,
        vision_labeling_command=None,
        timeout_seconds=1,
        generated_at="now",
    )
    assert "missing_vision_labeling_command" in vision_command_missing["blockers"]
    monkeypatch.setattr(
        arena,
        "_run_optional_command",
        lambda *_args, **_kwargs: {"status": "failed", "reason": "exit_code:1"},
    )
    vision_command_failed = arena._build_vision_labels(
        failure_labels={"labels": [{"label_id": "failure-1", "attempt_id": "attempt-1"}]},
        output_dir=tmp_path / "vision-command-failed",
        allow_vision_labeling=True,
        vision_labeling_command="labeler",
        timeout_seconds=1,
        generated_at="now",
    )
    assert "vision_labeling_command_failed" in vision_command_failed["blockers"]
    no_labels = arena._build_vision_labels(
        failure_labels={"labels": []},
        output_dir=tmp_path / "vision-no-labels",
        allow_vision_labeling=False,
        vision_labeling_command=None,
        timeout_seconds=1,
        generated_at="now",
    )
    assert no_labels["status"] == "no_failure_labels"

    assert arena._load_delivery_command_output(tmp_path / "missing-delivery")["status"] == "missing"
    delivery_not_object = tmp_path / "delivery-not-object"
    delivery_not_object.mkdir()
    (delivery_not_object / "delivery_upload.command.json").write_text("[]", encoding="utf-8")
    assert arena._load_delivery_command_output(delivery_not_object)["blockers"] == [
        "delivery_command_output_not_object"
    ]

    monkeypatch.setenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "true")
    delivery_cli_missing = arena._build_delivery_artifacts(
        output_dir=tmp_path / "delivery-cli-missing",
        allow_delivery_upload=False,
        delivery_command="uploader",
        timeout_seconds=1,
        generated_at="now",
    )
    assert "missing_cli_allow_delivery_upload" in (
        delivery_cli_missing["storage_upload_performed"] is False
        and _read_json(tmp_path / "delivery-cli-missing" / "signed_access_manifest.json")["blockers"]
    )
    arena._build_delivery_artifacts(
        output_dir=tmp_path / "delivery-command-missing",
        allow_delivery_upload=True,
        delivery_command=None,
        timeout_seconds=1,
        generated_at="now",
    )
    assert "missing_delivery_upload_command" in _read_json(
        tmp_path / "delivery-command-missing" / "signed_access_manifest.json"
    )["blockers"]
    monkeypatch.setattr(
        arena,
        "_run_optional_command",
        lambda *_args, **_kwargs: {"status": "failed", "reason": "exit_code:1"},
    )
    arena._build_delivery_artifacts(
        output_dir=tmp_path / "delivery-command-failed",
        allow_delivery_upload=True,
        delivery_command="uploader",
        timeout_seconds=1,
        generated_at="now",
    )
    assert "delivery_upload_failed" in _read_json(
        tmp_path / "delivery-command-failed" / "signed_access_manifest.json"
    )["blockers"]

    def completed_command(_command: str, _timeout: int, cwd: Path) -> dict[str, object]:
        _write_json(
            cwd / "delivery_upload.command.json",
            {
                "status": "completed",
                "signed_urls": ["https://example.test/package.zip"],
                "storage_upload_performed": True,
                "entitlement_verified": True,
                "buyer_access_check": {
                    "buyer_access_checked": True,
                    "buyer_accessible": True,
                    "status": "signed_url_minted",
                },
                "operator_attestation": "delivery owner accepted signed buyer access",
            },
        )
        return {"status": "completed", "reason": None}

    monkeypatch.setattr(arena, "_run_optional_command", completed_command)
    arena._build_delivery_artifacts(
        output_dir=tmp_path / "delivery-signed",
        allow_delivery_upload=True,
        delivery_command="uploader",
        timeout_seconds=1,
        generated_at="now",
    )
    assert _read_json(tmp_path / "delivery-signed" / "signed_access_manifest.json")[
        "status"
    ] == "signed_access_ready"
    signed_manifest = _read_json(tmp_path / "delivery-signed" / "signed_access_manifest.json")
    assert signed_manifest["entitlement_verified"] is True
    assert signed_manifest["buyer_access_check"]["buyer_access_checked"] is True
    assert signed_manifest["operator_attestation"] == (
        "delivery owner accepted signed buyer access"
    )

    def completed_command_without_access(_command: str, _timeout: int, cwd: Path) -> dict[str, object]:
        _write_json(cwd / "delivery_upload.command.json", {"status": "completed"})
        return {"status": "completed", "reason": None}

    monkeypatch.setattr(arena, "_run_optional_command", completed_command_without_access)
    arena._build_delivery_artifacts(
        output_dir=tmp_path / "delivery-no-access",
        allow_delivery_upload=True,
        delivery_command="uploader",
        timeout_seconds=1,
        generated_at="now",
    )
    assert _read_json(tmp_path / "delivery-no-access" / "signed_access_manifest.json")[
        "status"
    ] == "delivery_command_completed_review_required"


def test_arena_result_ingest_review_rerun_operator_and_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "review_resolutions.json",
        {"resolutions": [{"label_id": "label_attempt-1", "decision": "accepted", "reviewer": "qa"}]},
    )
    review = arena._build_review_resolution(
        results_dir=results_dir,
        failure_labels={
            "labels": [{"label_id": "label_attempt-1", "attempt_id": "attempt-1"}]
        },
        vision_labels={"label_count": 0},
        output_dir=tmp_path / "review-out",
        generated_at="now",
    )
    assert review["status"] == "accepted_labels_ready"
    accepted = _read_json(tmp_path / "review-out" / "accepted_failure_labels.json")
    assert accepted["labels"][0]["reviewer"] == "qa"

    rerun = arena._build_rerun_plan(
        attempt_trace={
            "attempts": [
                "skip-me",
                {
                    "attempt_id": "attempt-1",
                    "scenario_run_id": "run-1",
                    "scenario_id": "scenario-1",
                    "status": "timeout",
                    "success": False,
                    "video_path": "",
                },
            ]
        },
        review_ledger={"entries": [{"label_id": "label_attempt-1", "decision": "pending"}]},
        output_dir=tmp_path / "rerun-out",
        generated_at="now",
        retry_budget=1,
        cost_budget_usd=0.0,
    )
    assert rerun["status"] == "blocked_cost_budget_exhausted"
    assert rerun["queue"][0]["eligible"] is False
    assert set(rerun["queue"][0]["rerun_reasons"]) >= {"timeout", "missing_artifact"}

    fake_blocked = arena._build_live_operator_ledger(
        output_dir=tmp_path / "fake-blocked",
        rerun_plan=rerun,
        allow_live_agents_sdk=False,
        allow_live_codex_sdk=False,
        operator_mode="fake",
        timeout_seconds=1,
        generated_at="now",
    )
    assert fake_blocked["blockers"] == ["missing_env_BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS"]
    monkeypatch.setenv("BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS", "true")
    fake_completed = arena._build_live_operator_ledger(
        output_dir=tmp_path / "fake-completed",
        rerun_plan=rerun,
        allow_live_agents_sdk=False,
        allow_live_codex_sdk=False,
        operator_mode="fake",
        timeout_seconds=1,
        generated_at="now",
    )
    assert fake_completed["agents_sdk_operator_performed"] is True
    assert fake_completed["codex_sdk_operator_performed"] is True

    original_live_agents_blockers = arena._live_agents_blockers
    original_live_codex_blockers = arena._live_codex_blockers
    original_run_agents_sdk_operator = arena._run_agents_sdk_operator
    original_run_codex_sdk_operator = arena._run_codex_sdk_operator
    monkeypatch.setattr(arena, "_live_agents_blockers", lambda _allow: [])
    monkeypatch.setattr(arena, "_live_codex_blockers", lambda _allow: [])
    monkeypatch.setattr(
        arena,
        "_run_agents_sdk_operator",
        lambda *_args, **_kwargs: {"operator": "agents", "decision": "ok"},
    )
    monkeypatch.setattr(
        arena,
        "_run_codex_sdk_operator",
        lambda *_args, **_kwargs: {"operator": "codex", "decision": "ok"},
    )
    live_completed = arena._build_live_operator_ledger(
        output_dir=tmp_path / "live-completed",
        rerun_plan=rerun,
        allow_live_agents_sdk=True,
        allow_live_codex_sdk=True,
        operator_mode="agents-sdk",
        timeout_seconds=1,
        generated_at="now",
    )
    assert live_completed["status"] == "completed"

    monkeypatch.delenv(arena.LIVE_AGENTS_SDK_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(arena, "_module_available", lambda _candidates: None)
    assert "missing_cli_allow_live_agents_sdk" in original_live_agents_blockers(False)
    monkeypatch.delenv(arena.LIVE_CODEX_SDK_ENV, raising=False)
    monkeypatch.setattr(arena, "codex_cli_path", lambda: None)
    codex_blockers = original_live_codex_blockers(False)
    assert "missing_cli_allow_live_codex_sdk" in codex_blockers
    assert "missing_codex_cli" in codex_blockers

    class FakeAgent:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class FakeRunner:
        @staticmethod
        async def run(_agent: object, _prompt: str) -> object:
            return types.SimpleNamespace(final_output="agent output")

    monkeypatch.setitem(
        sys.modules,
        "agents",
        types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner),
    )
    agents_decision = original_run_agents_sdk_operator(tmp_path, timeout_seconds=1)
    assert agents_decision["decision"] == "live_agent_completed"

    class FakeSandbox:
        workspace_write = object()

    class FakeThread:
        def run(self, _prompt: str) -> object:
            return types.SimpleNamespace(final_response="codex output")

    class FakeCodex:
        def __enter__(self) -> "FakeCodex":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def thread_start(self, **_kwargs: object) -> FakeThread:
            return FakeThread()

    monkeypatch.setitem(
        sys.modules,
        "openai_codex",
        types.SimpleNamespace(Codex=FakeCodex, Sandbox=FakeSandbox),
    )
    codex_decision = original_run_codex_sdk_operator(tmp_path, timeout_seconds=1)
    assert codex_decision["decision"] == "live_codex_completed"
    monkeypatch.delitem(sys.modules, "openai_codex", raising=False)
    monkeypatch.setattr(
        arena,
        "run_codex_cli_operator",
        lambda _config: {"final_output": "cli output"},
    )
    codex_cli_decision = original_run_codex_sdk_operator(tmp_path, timeout_seconds=1)
    assert codex_cli_decision["decision"] == "live_codex_cli_completed"

    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "completed",
            "manifest_path": str(tmp_path / "arena_result_ingest_run_manifest.json"),
        }

    job_request = tmp_path / "job-request.json"
    _write_json(job_request, {"job_id": "job-1"})
    monkeypatch.setattr(arena, "build_arena_result_ingest", fake_build)
    exit_code = arena.main(
        [
            "--capture-root",
            str(tmp_path / "capture"),
            "--job-dir",
            str(tmp_path / "job"),
            "--arena-results-dir",
            str(tmp_path / "results"),
            "--output-dir",
            str(tmp_path / "out"),
            "--job-request",
            str(job_request),
            "--scenario-count",
            "7",
            "--shard-size",
            "3",
            "--num-envs",
            "2",
            "--timeout-seconds",
            "4",
            "--retry-budget",
            "5",
            "--cost-budget-usd",
            "6.5",
            "--allow-rollout-vision-labeling",
            "--vision-labeling-command",
            "labeler",
            "--allow-delivery-upload",
            "--delivery-command",
            "uploader",
            "--operator-mode",
            "fake",
            "--allow-live-agents-sdk",
            "--allow-live-codex-sdk",
        ]
    )
    assert exit_code == 0
    assert captured["scenario_count"] == 7
    assert captured["cost_budget_usd"] == 6.5
    assert "[arena-result-ingest] status=completed" in capsys.readouterr().out


def test_arena_ingest_small_helpers_cover_missing_and_fallback_edges(tmp_path: Path) -> None:
    assert arena._string_list(None) == []
    assert arena._string_list("one") == ["one"]
    assert arena._string_list(7) == ["7"]
    assert arena._boolish("passed") is True
    assert arena._read_optional_json(tmp_path / "missing.json") is None

    missing_ref = arena._artifact_ref(tmp_path, tmp_path / "missing.bin")
    assert missing_ref["exists"] is False
    assert missing_ref["sha256"] is None
    assert arena._cards({"scenarios": [{"scenario_id": "s1"}, "skip"]}) == [{"scenario_id": "s1"}]
    assert arena._load_scenario_cards(tmp_path)[0]["scenario_id"] == "arena_placeholder_scenario"

    redacted = arena._redact({"api_key": "secret", "items": [{"password": "hidden", "safe": "ok"}]})
    assert redacted == {"api_key": "<redacted>", "items": [{"password": "<redacted>", "safe": "ok"}]}
    assert arena._candidate_json_files(tmp_path / "does-not-exist") == []

    ignored = tmp_path / "review_resolutions.json"
    included = tmp_path / "episode.json"
    _write_json(ignored, {"ignored": True})
    _write_json(included, {"episode_id": "episode-1", "success": True})
    assert arena._candidate_json_files(tmp_path) == [included]


def test_arena_ingest_policy_adapter_status_validates_each_modality() -> None:
    assert arena._policy_adapter_status("policy_api_endpoint", {}) == (
        "blocked_missing_reference",
        ["policy_package.policy_api_endpoint"],
    )
    assert arena._policy_adapter_status("policy_api_endpoint", {"endpoint_url": ""}) == (
        "blocked_missing_fields",
        ["endpoint_url"],
    )
    assert arena._policy_adapter_status("docker_container", {"image_ref": "image", "digest": "bad"}) == (
        "blocked_missing_fields",
        ["digest"],
    )
    assert arena._policy_adapter_status("docker_container", {"digest": "sha256:abc"}) == (
        "blocked_missing_fields",
        ["image_ref"],
    )
    assert arena._policy_adapter_status("recorded_action_trace", {})[1] == [
        "policy_package.recorded_action_trace"
    ]
    assert arena._policy_adapter_status("recorded_action_trace", {"trace_manifest_uri": ""})[1] == [
        "trace_manifest_uri"
    ]
    assert arena._policy_adapter_status("high_level_skill_trace", {"ordered_skill_sequence": []})[1] == [
        "ordered_skill_sequence"
    ]
    assert arena._policy_adapter_status("teleop_demo", {"demo_artifact_uri": ""})[1] == [
        "demo_artifact_uri"
    ]
    assert arena._policy_adapter_status("sim_controller_plugin", {"plugin_uri": ""})[1] == [
        "plugin_uri"
    ]


def test_arena_ingest_extracts_episode_record_shapes_and_normalizes_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert arena._extract_episode_records(tmp_path / "missing") == ([], ["arena_results_dir_missing"])

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    missing_candidate = results_dir / "missing.json"
    monkeypatch.setattr(arena, "_candidate_json_files", lambda _results_dir: [missing_candidate])
    assert arena._extract_episode_records(results_dir) == ([], ["missing_episode_records"])

    monkeypatch.undo()
    _write_json(results_dir / "single.json", {"episode_id": "direct", "status": "completed"})
    _write_json(results_dir / "list.json", [{"episode_id": "listed", "status": "failed"}])
    records, blockers = arena._extract_episode_records(results_dir)

    assert blockers == []
    assert {record["episode_id"] for record in records} == {"direct", "listed"}

    trace = arena._normalize_attempts(
        records=[
            {"episode_id": "ok", "status": "completed"},
            {"episode_id": "bad", "status": "failed"},
        ],
        results_dir=results_dir,
        generated_at="now",
    )
    assert trace["attempts"][0]["success"] is True
    assert trace["attempts"][1]["failure_reason"] == "threshold_miss_or_failed_status"


def test_arena_ingest_labels_clips_commands_and_command_output_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failures = arena._build_failure_labels(
        {
            "attempts": [
                "skip",
                {"attempt_id": "success", "success": True},
                {
                    "attempt_id": "failed",
                    "success": False,
                    "failure_reason": "collision threshold",
                    "metrics": {"score": -1},
                },
            ]
        },
        "now",
    )
    assert failures["label_count"] == 1
    assert "metric_out_of_bounds" in failures["labels"][0]["failure_categories"]

    assert arena._copy_clip_source(tmp_path / "missing.mp4", tmp_path / "clips" / "missing.mp4") is False
    clips = arena._build_clips_manifest(
        attempt_trace={"attempts": ["skip", {"attempt_id": "a1", "video_path": ""}]},
        output_dir=tmp_path,
        generated_at="now",
    )
    assert clips["clip_count"] == 1
    assert clips["clips"][0]["status"] == "blocked_missing_video"

    def _missing_run(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(arena.subprocess, "run", _missing_run)
    missing = arena._run_optional_command("blueprint-definitely-missing-command", 1, tmp_path)
    assert missing["status"] == "blocked"
    assert missing["reason"] == "missing_command_dependency"

    def _timeout_run(*_args, **_kwargs):
        raise arena.subprocess.TimeoutExpired(cmd=["slow"], timeout=1, output="out", stderr="err")

    monkeypatch.setattr(arena.subprocess, "run", _timeout_run)
    timed_out = arena._run_optional_command("slow", 1, tmp_path)
    assert timed_out["status"] == "failed"
    assert timed_out["reason"] == "timeout"

    assert arena._load_command_vision_labels(tmp_path)["status"] == "missing"
    (tmp_path / "rollout_vision_labels.command.json").write_text("[]", encoding="utf-8")
    assert arena._load_command_vision_labels(tmp_path)["blockers"] == ["vision_command_output_not_object"]
    _write_json(tmp_path / "rollout_vision_labels.command.json", {"labels": "not-list"})
    assert arena._load_command_vision_labels(tmp_path)["blockers"] == [
        "vision_command_output_missing_labels"
    ]

    assert arena._load_delivery_command_output(tmp_path / "delivery-missing")["status"] == "missing"
    delivery_dir = tmp_path / "delivery-output"
    delivery_dir.mkdir()
    (delivery_dir / "delivery_upload.command.json").write_text("[]", encoding="utf-8")
    assert arena._load_delivery_command_output(delivery_dir)["blockers"] == [
        "delivery_command_output_not_object"
    ]


def test_arena_ingest_vision_review_and_rerun_edge_states(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vision = arena._build_vision_labels(
        failure_labels={"labels": ["skip"]},
        output_dir=tmp_path,
        allow_vision_labeling=False,
        vision_labeling_command=None,
        timeout_seconds=1,
        generated_at="now",
    )
    assert vision["status"] == "no_failure_labels"

    monkeypatch.setenv("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", "true")
    missing_cli_gate = arena._build_vision_labels(
        failure_labels={"labels": [{"label_id": "label_a", "attempt_id": "a"}]},
        output_dir=tmp_path,
        allow_vision_labeling=False,
        vision_labeling_command=f"{sys.executable} -c pass",
        timeout_seconds=1,
        generated_at="now",
    )
    assert "missing_cli_allow_rollout_vision_labeling" in missing_cli_gate["blockers"]

    missing_command = arena._build_vision_labels(
        failure_labels={"labels": [{"label_id": "label_b", "attempt_id": "b"}]},
        output_dir=tmp_path,
        allow_vision_labeling=True,
        vision_labeling_command=None,
        timeout_seconds=1,
        generated_at="now",
    )
    assert "missing_vision_labeling_command" in missing_command["blockers"]

    failed_command = arena._build_vision_labels(
        failure_labels={"labels": [{"label_id": "label_c", "attempt_id": "c"}]},
        output_dir=tmp_path,
        allow_vision_labeling=True,
        vision_labeling_command=f"{sys.executable} -c 'import sys; sys.exit(3)'",
        timeout_seconds=5,
        generated_at="now",
    )
    assert "vision_labeling_command_failed" in failed_command["blockers"]

    results_dir = tmp_path / "review-results"
    results_dir.mkdir()
    _write_json(
        results_dir / "review_resolutions.json",
        {"resolutions": [{"label_id": "label_c", "decision": "accepted", "reviewer": "owner"}]},
    )
    review = arena._build_review_resolution(
        results_dir=results_dir,
        failure_labels={"labels": [{"label_id": "label_c", "attempt_id": "c"}]},
        vision_labels={"label_count": 1},
        output_dir=tmp_path,
        generated_at="now",
    )
    assert review["status"] == "accepted_labels_ready"
    accepted = _read_json(tmp_path / "accepted_failure_labels.json")
    assert accepted["labels"][0]["review_status"] == "accepted"

    rerun = arena._build_rerun_plan(
        attempt_trace={
            "attempts": [
                "skip",
                {
                    "attempt_id": "c",
                    "scenario_id": "scenario-c",
                    "scenario_run_id": "run-c",
                    "status": "timeout",
                    "success": False,
                },
            ]
        },
        review_ledger={"entries": [{"label_id": "label_c", "decision": "pending"}]},
        output_dir=tmp_path,
        generated_at="now",
        retry_budget=1,
        cost_budget_usd=0,
    )
    assert rerun["status"] == "blocked_cost_budget_exhausted"
    assert rerun["queue"][0]["eligible"] is False
    assert set(rerun["queue"][0]["rerun_reasons"]) == {
        "failed",
        "missing_artifact",
        "review_required",
        "timeout",
    }


def test_arena_ingest_delivery_upload_edge_states(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "package"
    output_dir.mkdir()
    for name in (
        "customer_handoff_report.md",
        "customer_handoff_report.json",
        "post_training_data_package_export_manifest.json",
        "package_index.json",
        "archive_manifest.json",
    ):
        (output_dir / name).write_text(name, encoding="utf-8")

    monkeypatch.setenv("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "true")
    missing_cli = arena._build_delivery_artifacts(
        output_dir=output_dir,
        allow_delivery_upload=False,
        delivery_command=f"{sys.executable} -c pass",
        timeout_seconds=5,
        generated_at="now",
    )
    assert _read_json(output_dir / "signed_access_manifest.json")["blockers"] == [
        "missing_cli_allow_delivery_upload"
    ]
    assert missing_cli["storage_upload_performed"] is False

    missing_command = arena._build_delivery_artifacts(
        output_dir=output_dir,
        allow_delivery_upload=True,
        delivery_command=None,
        timeout_seconds=5,
        generated_at="now",
    )
    assert _read_json(output_dir / "signed_access_manifest.json")["blockers"] == [
        "missing_delivery_upload_command"
    ]
    assert missing_command["storage_upload_performed"] is False

    failed_upload = arena._build_delivery_artifacts(
        output_dir=output_dir,
        allow_delivery_upload=True,
        delivery_command=f"{sys.executable} -c 'import sys; sys.exit(4)'",
        timeout_seconds=5,
        generated_at="now",
    )
    assert "delivery_upload_failed" in _read_json(output_dir / "signed_access_manifest.json")["blockers"]
    assert failed_upload["storage_upload_performed"] is False

    writer = tmp_path / "write_signed.py"
    writer.write_text(
        "\n".join(
            [
                "import json",
                "json.dump({'signed_urls': ['https://example.invalid/package'], 'storage_upload_performed': True}, open('delivery_upload.command.json', 'w'))",
            ]
        ),
        encoding="utf-8",
    )
    signed = arena._build_delivery_artifacts(
        output_dir=output_dir,
        allow_delivery_upload=True,
        delivery_command=f"{sys.executable} {writer}",
        timeout_seconds=5,
        generated_at="now",
    )
    assert _read_json(output_dir / "signed_access_manifest.json")["status"] == "signed_access_ready"
    assert signed["storage_upload_performed"] is True

    writer.write_text(
        "import json; json.dump({'status': 'completed'}, open('delivery_upload.command.json', 'w'))",
        encoding="utf-8",
    )
    arena._build_delivery_artifacts(
        output_dir=output_dir,
        allow_delivery_upload=True,
        delivery_command=f"{sys.executable} {writer}",
        timeout_seconds=5,
        generated_at="now",
    )
    manifest = _read_json(output_dir / "signed_access_manifest.json")
    assert manifest["status"] == "delivery_command_completed_review_required"
    assert manifest["blockers"] == ["delivery_command_output_missing_signed_or_local_access"]


def test_arena_ingest_live_operator_gates_and_fake_sdk_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS", raising=False)
    blocked_fake = arena._build_live_operator_ledger(
        output_dir=tmp_path,
        rerun_plan={},
        allow_live_agents_sdk=False,
        allow_live_codex_sdk=False,
        operator_mode="fake",
        timeout_seconds=1,
        generated_at="now",
    )
    assert blocked_fake["blockers"] == ["missing_env_BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS"]

    monkeypatch.setenv("BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS", "true")
    fake = arena._build_live_operator_ledger(
        output_dir=tmp_path,
        rerun_plan={"status": "reruns_queued", "eligible_count": 1},
        allow_live_agents_sdk=False,
        allow_live_codex_sdk=False,
        operator_mode="fake",
        timeout_seconds=1,
        generated_at="now",
    )
    assert fake["status"] == "completed"
    assert fake["agents_sdk_operator_performed"] is True
    assert fake["codex_sdk_operator_performed"] is True

    monkeypatch.setattr(arena, "_live_agents_blockers", lambda allowed: [])
    monkeypatch.setattr(arena, "_live_codex_blockers", lambda allowed: [])
    monkeypatch.setattr(arena, "_run_agents_sdk_operator", lambda output_dir, timeout_seconds: {"operator": "agents"})
    monkeypatch.setattr(arena, "_run_codex_sdk_operator", lambda output_dir, timeout_seconds: {"operator": "codex"})
    live = arena._build_live_operator_ledger(
        output_dir=tmp_path,
        rerun_plan={},
        allow_live_agents_sdk=True,
        allow_live_codex_sdk=True,
        operator_mode="agents-sdk",
        timeout_seconds=1,
        generated_at="now",
    )
    assert live["status"] == "completed"
    assert [decision["operator"] for decision in live["decisions"]] == ["agents", "codex"]

    monkeypatch.undo()
    monkeypatch.setattr(arena, "_module_available", lambda candidates: None)
    monkeypatch.setattr(arena, "codex_cli_path", lambda: None)
    monkeypatch.delenv(arena.LIVE_AGENTS_SDK_ENV, raising=False)
    monkeypatch.delenv(arena.LIVE_CODEX_SDK_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "missing_cli_allow_live_agents_sdk" in arena._live_agents_blockers(False)
    assert "missing_openai_agents_sdk" in arena._live_agents_blockers(True)
    codex_blockers = arena._live_codex_blockers(False)
    assert "missing_cli_allow_live_codex_sdk" in codex_blockers
    assert "missing_codex_cli" in codex_blockers


def test_arena_ingest_live_agents_and_codex_operator_success_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunner:
        @staticmethod
        async def run(_agent, _prompt):
            return types.SimpleNamespace(final_output="agent output")

    monkeypatch.setitem(sys.modules, "agents", types.SimpleNamespace(Agent=FakeAgent, Runner=FakeRunner))
    agents_result = arena._run_agents_sdk_operator(tmp_path, 1)
    assert agents_result["decision"] == "live_agent_completed"
    assert agents_result["tool_call_summary"] == {"final_output": "agent output"}

    class FakeThread:
        def run(self, _prompt):
            return types.SimpleNamespace(final_response="codex response")

    class FakeCodex:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def thread_start(self, **_kwargs):
            return FakeThread()

    monkeypatch.setitem(
        sys.modules,
        "openai_codex",
        types.SimpleNamespace(Codex=FakeCodex, Sandbox=types.SimpleNamespace(workspace_write="workspace-write")),
    )
    codex_result = arena._run_codex_sdk_operator(tmp_path, 1)
    assert codex_result["decision"] == "live_codex_completed"
    assert codex_result["tool_call_summary"] == {"final_response": "codex response"}

    monkeypatch.setitem(sys.modules, "openai_codex", None)
    monkeypatch.setattr(arena, "run_codex_cli_operator", lambda config: {"final_output": "cli output"})
    cli_result = arena._run_codex_sdk_operator(tmp_path, 1)
    assert cli_result["decision"] == "live_codex_cli_completed"
    assert cli_result["tool_call_summary"]["final_output"] == "cli output"


def test_arena_ingest_main_returns_success_and_failure(tmp_path: Path, capsys) -> None:
    capture_root = _capture_root(tmp_path)
    results_dir = _arena_results(tmp_path)
    output_dir = tmp_path / "main-package"
    job_request = tmp_path / "job_request.json"
    _write_json(job_request, {"policy_package": {"policy_api_endpoint": {"endpoint_url": "https://policy"}}})

    assert arena.main(
        [
            "--capture-root",
            str(capture_root),
            "--arena-results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--job-request",
            str(job_request),
            "--scenario-count",
            "1",
            "--shard-size",
            "1",
            "--num-envs",
            "1",
            "--timeout-seconds",
            "1",
            "--retry-budget",
            "1",
            "--cost-budget-usd",
            "1",
            "--operator-mode",
            "none",
        ]
    ) == 0
    assert "[arena-result-ingest] status=completed" in capsys.readouterr().out

    assert arena.main(
        [
            "--capture-root",
            str(capture_root),
            "--arena-results-dir",
            str(tmp_path / "missing-results"),
            "--output-dir",
            str(tmp_path / "blocked-package"),
        ]
    ) == 1
