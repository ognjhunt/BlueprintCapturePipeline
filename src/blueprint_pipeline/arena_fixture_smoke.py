"""Local 500-scenario Arena package smoke.

This command creates a synthetic Arena-style fixture under an output directory,
runs the real ``blueprint-ingest-arena-results`` CLI path against it, audits the
package artifacts, and writes a compact proof-boundary summary. It is local-only:
it does not run Isaac Lab-Arena, upload storage, call vision models, or prove
robot readiness.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .arena_package_audit import build_arena_package_proof_boundary_audit
from .arena_result_ingest import CLAIM_BOUNDARY, main as arena_result_ingest_main
from .common import ensure_dir, read_json_any, utc_now_iso, write_json


ARENA_FIXTURE_SMOKE_SCHEMA_VERSION = "arena_fixture_smoke.v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(path, dict(payload))


def _read_mapping(path: Path) -> Dict[str, Any]:
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_capture_fixture(capture_root: Path) -> None:
    _write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "arena-smoke-site",
            "capture_id": "arena-smoke-capture",
            "fixture_only": True,
            "proof_boundary": "synthetic_local_smoke_not_webapp_or_site_operator_truth",
        },
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "arena-smoke-site",
            "capture_id": "arena-smoke-capture",
            "fixture_only": True,
        },
    )
    robot_eval = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        robot_eval / "site_card.json",
        {
            "schema_version": "real_site_robot_eval_site_card.v0.1",
            "site_id": "arena-smoke-site",
            "site_type": "synthetic_fixture",
        },
    )
    _write_json(
        robot_eval / "task_cards.json",
        {
            "schema_version": "real_site_robot_eval_task_cards.v0.1",
            "cards": [
                {
                    "task_id": "move-tote",
                    "task_statement": "Move the tote to the marked staging area.",
                }
            ],
        },
    )
    _write_json(
        robot_eval / "scenario_cards.json",
        {
            "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
            "cards": [
                {
                    "scenario_id": "arena-smoke-clear-path",
                    "task_id": "move-tote",
                    "robot_profile_id": "mobile-manipulator-rgbd",
                },
                {
                    "scenario_id": "arena-smoke-occlusion",
                    "task_id": "move-tote",
                    "robot_profile_id": "mobile-manipulator-rgbd",
                },
                {
                    "scenario_id": "arena-smoke-missing-artifact",
                    "task_id": "move-tote",
                    "robot_profile_id": "mobile-manipulator-rgbd",
                },
            ],
        },
    )
    _write_json(
        robot_eval / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "cards": [{"scenario_id": "arena-smoke-clear-path"}],
        },
    )
    _write_json(
        robot_eval / "proof_boundaries.json",
        {
            "simulator_execution_proven": False,
            "robot_policy_execution_proven": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    )


def _write_results_fixture(results_dir: Path) -> None:
    videos = results_dir / "videos"
    logs = results_dir / "logs"
    ensure_dir(videos)
    ensure_dir(logs)
    (videos / "episode-success.mp4").write_bytes(b"synthetic-success-video")
    (videos / "episode-failure.mp4").write_bytes(b"synthetic-failure-video")
    (logs / "episode-success.jsonl").write_text('{"event":"success"}\n', encoding="utf-8")
    (logs / "episode-failure.jsonl").write_text('{"event":"occlusion"}\n', encoding="utf-8")
    _write_json(
        results_dir / "rollout_manifest.json",
        {
            "schema_version": "arena_fixture_rollout_manifest.v1",
            "fixture_only": True,
            "episodes": [
                {
                    "episode_id": "episode-success",
                    "scenario_id": "arena-smoke-clear-path",
                    "scenario_run_id": "arena-smoke-clear-path__arena_run_0001",
                    "task_id": "move-tote",
                    "status": "success",
                    "success": True,
                    "metrics": {
                        "cycle_time_seconds": 18.5,
                        "intervention_count": 0,
                        "safety_event_count": 0,
                    },
                    "video_path": "videos/episode-success.mp4",
                    "log_path": "logs/episode-success.jsonl",
                    "start_time_seconds": 0.0,
                    "end_time_seconds": 18.5,
                },
                {
                    "episode_id": "episode-failure",
                    "scenario_id": "arena-smoke-occlusion",
                    "scenario_run_id": "arena-smoke-occlusion__arena_run_0002",
                    "task_id": "move-tote",
                    "status": "failed",
                    "success": False,
                    "failure_reason": "occlusion_threshold_miss",
                    "metrics": {
                        "cycle_time_seconds": 42.0,
                        "intervention_count": 1,
                        "safety_event_count": 0,
                    },
                    "video_path": "videos/episode-failure.mp4",
                    "log_path": "logs/episode-failure.jsonl",
                    "start_time_seconds": 0.0,
                    "end_time_seconds": 42.0,
                },
                {
                    "episode_id": "episode-missing-artifact",
                    "scenario_id": "arena-smoke-missing-artifact",
                    "scenario_run_id": "arena-smoke-missing-artifact__arena_run_0003",
                    "task_id": "move-tote",
                    "status": "timeout",
                    "success": False,
                    "failure_reason": "timeout_missing_artifact",
                    "metrics": {
                        "cycle_time_seconds": 120.0,
                        "intervention_count": 1,
                        "safety_event_count": 0,
                    },
                    "start_time_seconds": 0.0,
                    "end_time_seconds": 120.0,
                },
            ],
        },
    )


def _write_job_request(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "arena-fixture-smoke-job",
            "policy_package": {
                "policy_api_endpoint": {"endpoint_url": "https://robot.example/policy"},
                "docker_container": {
                    "image_ref": "registry.example/robot/policy:smoke",
                    "digest": "sha256:" + "a" * 64,
                },
                "recorded_action_trace": {
                    "trace_manifest_uri": "fixture://traces/action-trace-manifest.json"
                },
                "high_level_skill_trace": {
                    "ordered_skill_sequence": ["navigate", "pick", "place"]
                },
                "teleop_demo": {
                    "demo_artifact_uri": "fixture://demos/teleop-demo.json",
                },
                "sim_controller_plugin": {
                    "plugin_uri": "fixture://plugins/controller.json",
                },
            },
            "rights_privacy_scope": {
                "status": "fixture_local_only",
                "external_use_allowed": False,
            },
        },
    )


def _write_vision_command(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import json",
                "payload = {",
                "  'schema_version': 'arena_rollout_vision_command_labels.v1',",
                "  'provider': 'local_fixture_vision',",
                "  'model': 'deterministic-fixture',",
                "  'visual_evidence_used': True,",
                "  'labels': [{",
                "    'vision_label_id': 'fixture-vision-occlusion',",
                "    'source_failure_label_id': 'label_arena_attempt_0002',",
                "    'attempt_id': 'arena_attempt_0002',",
                "    'status': 'accepted',",
                "    'object_state': 'tote occluded by fixture obstacle',",
                "    'contact': 'review_required',",
                "    'occlusion': 'present',",
                "    'threshold_miss': True,",
                "    'failure_evidence': ['occlusion_threshold_miss'],",
                "    'label_source': 'local_fixture_vision',",
                "    'visual_evidence_used': True",
                "  }]",
                "}",
                "with open('rollout_vision_labels.command.json', 'w', encoding='utf-8') as f:",
                "    json.dump(payload, f)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _set_env(name: str, value: str, original: Dict[str, str | None]) -> None:
    if name not in original:
        original[name] = os.environ.get(name)
    os.environ[name] = value


def _restore_env(original: Mapping[str, str | None]) -> None:
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def build_arena_fixture_smoke(
    *,
    output_dir: str | Path,
    scenario_count: int = 500,
    shard_size: int = 100,
    retry_budget: int = 2,
) -> Dict[str, Any]:
    root = Path(output_dir).resolve()
    capture_root = root / "local-blueprint" / "scenes" / "arena-smoke-site" / "captures" / "arena-smoke-capture"
    results_dir = root / "arena-results"
    package_dir = root / "arena-package"
    delivery_root = root / "local-deliveries"
    job_request_path = root / "job_request.json"
    vision_script = root / "write_fixture_vision_labels.py"
    ensure_dir(root)
    _write_capture_fixture(capture_root)
    _write_results_fixture(results_dir)
    _write_job_request(job_request_path)
    _write_vision_command(vision_script)

    original_env: Dict[str, str | None] = {}
    try:
        src_root = Path(__file__).resolve().parents[1]
        existing_pythonpath = os.environ.get("PYTHONPATH")
        pythonpath = (
            f"{src_root}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(src_root)
        )
        _set_env("PYTHONPATH", pythonpath, original_env)
        _set_env("BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING", "true", original_env)
        _set_env("BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD", "true", original_env)
        _set_env("BLUEPRINT_LOCAL_DELIVERY_ROOT", str(delivery_root), original_env)
        _set_env("BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS", "true", original_env)
        ingest_exit_code = arena_result_ingest_main(
            [
                "--capture-root",
                str(capture_root),
                "--arena-results-dir",
                str(results_dir),
                "--output-dir",
                str(package_dir),
                "--job-request",
                str(job_request_path),
                "--scenario-count",
                str(scenario_count),
                "--shard-size",
                str(shard_size),
                "--retry-budget",
                str(retry_budget),
                "--allow-rollout-vision-labeling",
                "--vision-labeling-command",
                f"{shlex.quote(sys.executable)} {shlex.quote(str(vision_script))}",
                "--allow-delivery-upload",
                "--delivery-command",
                (
                    f"{shlex.quote(sys.executable)} -m "
                    "blueprint_pipeline.arena_package_delivery_local --output-dir ."
                ),
                "--operator-mode",
                "fake",
            ]
        )
    finally:
        _restore_env(original_env)

    audit = build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=package_dir,
        expected_scenario_count=scenario_count,
    )
    run_manifest = _read_mapping(package_dir / "arena_result_ingest_run_manifest.json")
    schedule = _read_mapping(package_dir / "arena_eval_schedule.json")
    trace = _read_mapping(package_dir / "normalized_attempt_trace.json")
    clips = _read_mapping(package_dir / "clips_manifest.json")
    rerun = _read_mapping(package_dir / "arena_rerun_plan.json")
    operators = _read_mapping(package_dir / "live_operator_ledger.json")
    signed_access = _read_mapping(package_dir / "signed_access_manifest.json")
    policy = _read_mapping(package_dir / "policy_adapter_manifest.json")
    result = {
        "schema_version": ARENA_FIXTURE_SMOKE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if ingest_exit_code == 0 and audit.get("status") == "passed" else "blocked",
        "output_dir": str(root),
        "capture_root": str(capture_root),
        "arena_results_dir": str(results_dir),
        "package_dir": str(package_dir),
        "ingest_exit_code": ingest_exit_code,
        "audit_status": audit.get("status"),
        "audit_manifest": str(package_dir / "arena_package_proof_boundary_audit.json"),
        "run_status": run_manifest.get("status"),
        "scenario_count": schedule.get("scenario_count"),
        "shard_count": schedule.get("shard_count"),
        "attempt_count": trace.get("attempt_count"),
        "clip_count": clips.get("clip_count"),
        "rerun_status": rerun.get("status"),
        "rerun_eligible_count": rerun.get("eligible_count"),
        "operator_status": operators.get("status"),
        "signed_access_status": signed_access.get("status"),
        "policy_adapter_status": policy.get("status"),
        "proof_boundary": {
            **dict(CLAIM_BOUNDARY),
            "fixture_smoke_only": True,
            "webapp_upstream_truth_proven": False,
            "owner_system_arena_execution_proven": False,
        },
        "blockers": list(audit.get("blockers") or []),
    }
    write_json(root / "arena_fixture_smoke_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local fixture smoke for the 500-scenario Arena package pipeline"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-count", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--retry-budget", type=int, default=2)
    args = parser.parse_args(argv)
    result = build_arena_fixture_smoke(
        output_dir=args.output_dir,
        scenario_count=args.scenario_count,
        shard_size=args.shard_size,
        retry_budget=args.retry_budget,
    )
    print(f"[arena-fixture-smoke] manifest={Path(args.output_dir).resolve() / 'arena_fixture_smoke_manifest.json'}")
    print(f"[arena-fixture-smoke] status={result['status']}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
