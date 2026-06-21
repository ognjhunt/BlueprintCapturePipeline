from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import oscar_cosmos_wam_evaluator as evaluator


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _input_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "mujoco_job"
    _write_json(
        job_dir / "scenario_eval_matrix.json",
        {
            "schema_version": "matrix",
            "runs": [
                {
                    "scenario_eval_run_id": "run_1",
                    "task_id": "approach_target",
                    "spawn_id": "doorway",
                    "task_prompt": "Approach the target.",
                }
            ],
        },
    )
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "schema_version": "attempts",
            "attempts": [
                {
                    "attempt_id": "attempt_1",
                    "scenario_eval_run_id": "run_1",
                    "task_id": "approach_target",
                    "spawn_id": "doorway",
                    "success": True,
                }
            ],
        },
    )
    _write_jsonl(
        job_dir / "normalized_policy_action_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "normalized_action": {"action_type": "waypoint"},
                "policy_id": "endpoint_policy",
            }
        ],
    )
    _write_jsonl(
        job_dir / "g1_mujoco_locomotion_trace.jsonl",
        [
            {
                "scenario_eval_run_id": "run_1",
                "root_position": [0.0, 0.0, 0.79],
                "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        ],
    )
    _write_json(
        job_dir / "review_video_selection_manifest.json",
        {
            "schema_version": "review_videos",
            "selected_review_videos": [
                {"episode_id": "episode_1", "camera": "third_person", "path": "/tmp/review.mp4"}
            ],
        },
    )
    return job_dir


def test_oscar_cosmos_wam_evaluator_writes_blocked_dry_run_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)
    monkeypatch.delenv("HF_TOKEN_FILE", raising=False)
    monkeypatch.delenv("NGC_API_KEY_FILE", raising=False)
    input_job = _input_job(tmp_path)
    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_job",
        generated_at="now",
    )
    assert summary["status"] == "completed"
    assert summary["learned_wam_model_ran"] is False
    assert "blocked_missing_wam_runtime" in summary["blockers"]
    assert "blocked_missing_wam_model_checkpoint" in summary["blockers"]
    required = [
        "wam_model_runtime_discovery.json",
        "wam_rollout_input_manifest.json",
        "wam_action_conditioning_manifest.json",
        "wam_generated_rollout_manifest.json",
        "wam_generated_rollout_results.json",
        "wam_consistency_checks.json",
        "wam_success_labels.json",
        "wam_policy_scorecard.json",
        "wam_evaluator_trace_binding.json",
        "wam_evaluator_truth_boundary.json",
        "policy_model_truth_boundary.json",
        "policy_model_endpoint_readiness_manifest.json",
        "policy_model_endpoint_creation_plan.json",
        "policy_cloud_gpu_setup_manifest.json",
        "local_model_source_tree_discovery.json",
    ]
    for filename in required:
        assert (tmp_path / "wam_job" / filename).is_file()
    consistency = json.loads(
        (tmp_path / "wam_job" / "wam_consistency_checks.json").read_text(encoding="utf-8")
    )
    assert consistency["forward_inverse_consistency_proven"] is False
    assert consistency["action_conditioned_video_rollout_generated"] is False
    truth = json.loads(
        (tmp_path / "wam_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["learned_wam_model_ran"] is False
    policy_truth = json.loads(
        (tmp_path / "wam_job" / "policy_model_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy_truth["schema_version"] == "policy_model_truth_boundary.v1"
    assert policy_truth["replaceable_model_adapter_boundary"] is True
    readiness = json.loads(
        (tmp_path / "wam_job" / "policy_model_endpoint_readiness_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["http_endpoint_wrapper_available"] is True
    assert readiness["real_model_ready_candidate_count"] == 0
    assert readiness["claim_boundary"]["endpoint_creation_is_not_model_execution_proof"] is True
    oscar = next(row for row in readiness["candidates"] if row["candidate_id"] == "oscar_wam")
    assert oscar["endpoint_wrapper_can_be_created"] is False
    assert "set_BLUEPRINT_OSCAR_WAM_COMMAND_to_runnable_adapter_command" in oscar[
        "what_is_needed_to_make_true"
    ]
    creation_plan = json.loads(
        (tmp_path / "wam_job" / "policy_model_endpoint_creation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert creation_plan["http_wrapper_binary_available"] is True
    assert creation_plan["can_create_real_model_endpoint_now"] is False
    assert creation_plan["claim_boundary"]["http_endpoint_creation_is_not_model_execution_proof"] is True
    assert "runnable adapter command" in " ".join(
        creation_plan["minimum_user_supplied_inputs"]
    )
    source_discovery = json.loads(
        (tmp_path / "wam_job" / "local_model_source_tree_discovery.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        source_discovery["claim_boundary"]["source_tree_present_is_not_model_runtime_proof"]
        is True
    )


def test_oscar_cosmos_wam_evaluator_reports_file_auth_without_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    hf_file = tmp_path / "hf-token"
    ngc_file = tmp_path / "ngc-token"
    hf_file.write_text("hf-secret-value\n", encoding="utf-8")
    ngc_file.write_text("ngc-secret-value\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_FILE", str(hf_file))
    monkeypatch.setenv("NGC_API_KEY_FILE", str(ngc_file))

    discovery = evaluator.discover_wam_model_runtimes(generated_at="now")

    assert discovery["model_access_secret_status"]["huggingface"]["auth_ready"] is True
    assert discovery["model_access_secret_status"]["ngc"]["auth_ready"] is True
    serialized = json.dumps(discovery, sort_keys=True)
    assert "hf-secret-value" not in serialized
    assert "ngc-secret-value" not in serialized


def test_oscar_cosmos_wam_evaluator_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert evaluator._repo_root().name == "BlueprintCapturePipeline"
    assert evaluator._timestamp().endswith("Z")
    assert evaluator._string_list("one") == ["one"]
    assert evaluator._load_json(tmp_path / "missing.json") == {}
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('\n{"ok": 1}\n[]\n', encoding="utf-8")
    assert evaluator._read_jsonl(rows_path) == [{"ok": 1}]
    assert evaluator._read_jsonl(tmp_path / "missing.jsonl") == []
    assert evaluator._command_available("'unterminated") is False
    assert evaluator._command_available("   ") is False
    assert evaluator._relative_or_absolute(tmp_path / "a", tmp_path) == "a"
    assert evaluator._relative_or_absolute(Path("/tmp/outside-blueprint-test"), tmp_path).startswith("/")

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x")
    assert evaluator._checkpoint_like_files(checkpoint)["checkpoint_files_found"][0]["relative_path"] == checkpoint.name
    oscar_checkpoint = tmp_path / "__0_0.distcp"
    oscar_checkpoint.write_bytes(b"x" * (51 * 1024 * 1024))
    oscar_scan = evaluator._checkpoint_like_files(oscar_checkpoint)
    assert oscar_scan["checkpoint_files_found"][0]["relative_path"] == oscar_checkpoint.name
    assert oscar_scan["checkpoint_files_found"][0]["large_enough_for_wam_or_vla_weights"] is True
    not_checkpoint = tmp_path / "not-checkpoint.txt"
    not_checkpoint.write_text("not weights", encoding="utf-8")
    assert evaluator._checkpoint_like_files(not_checkpoint)["checkpoint_files_found"] == []
    assert evaluator._checkpoint_like_files(tmp_path / "missing-root")["files_scanned"] == 0

    class OSErrorFile:
        suffix = ".pt"
        name = "bad.pt"

        def is_file(self) -> bool:
            return True

        def stat(self) -> object:
            raise OSError("no stat")

    assert evaluator._checkpoint_like_files(OSErrorFile())["checkpoint_files_found"][0]["size_bytes"] is None

    scan_root = tmp_path / "scan-root"
    scan_root.mkdir()
    for index in range(13):
        (scan_root / f"model-{index}.pt").write_bytes(b"x")
    scan = evaluator._checkpoint_like_files(scan_root)
    assert scan["truncated"] is True
    assert len(scan["checkpoint_files_found"]) == 12
    many_files = tmp_path / "many-files"
    many_files.mkdir()
    for index in range(3):
        (many_files / f"file-{index}.txt").write_text("x", encoding="utf-8")
    assert evaluator._checkpoint_like_files(many_files, max_files_scanned=2)["truncated"] is True

    bad_stat = tmp_path / "bad-stat-root"
    bad_stat.mkdir()
    bad_file = bad_stat / "bad.pt"
    bad_file.write_bytes(b"x")
    original_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object):
        if self == bad_file:
            raise OSError("bad stat")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    assert evaluator._checkpoint_like_files(bad_stat)["checkpoint_files_found"][0]["size_bytes"] is None


def test_oscar_cosmos_wam_evaluator_host_probe_and_plan_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        def __init__(self, returncode: int, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(evaluator.platform, "system", lambda: "Linux")
    monkeypatch.setattr(evaluator.shutil, "which", lambda _name: None)
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: Completed(0, '{"cuda": true}'))
    assert evaluator._local_host_probe()["torch_cuda_available"] is True
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: Completed(1, ""))
    assert evaluator._local_host_probe()["torch_probe_error_type"] == "torch_probe_subprocess_failed"
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe")))
    assert evaluator._local_host_probe()["torch_probe_error_type"] == "RuntimeError"

    monkeypatch.setattr(
        evaluator,
        "_source_roots_for_candidate",
        lambda _candidate: [
            {"label": "missing", "path": tmp_path / "missing-source", "configured_by_env": False}
        ],
    )
    assert "blocked_missing_local_model_source_tree" in evaluator._local_source_tree_probe("oscar_wam")["blockers"]
    readiness = evaluator.build_policy_model_endpoint_readiness_manifest(
        generated_at="now",
        candidates=("oscar_wam",),
        explicit_candidate_id="oscar_wam",
        explicit_command="definitely-not-a-real-command",
        explicit_checkpoint=tmp_path / "missing-checkpoint",
    )
    row = readiness["candidates"][0]
    assert "make_configured_model_command_executable_or_on_path" in row["what_is_needed_to_make_true"]
    assert "download_or_mount_configured_model_checkpoint_path" in row["what_is_needed_to_make_true"]
    plan = evaluator.build_policy_model_endpoint_creation_plan(
        generated_at="now",
        readiness_manifest={"candidates": ["bad", row]},
    )
    assert len(plan["candidate_creation_plans"]) == 1

    ready_checkpoint = tmp_path / "ready-checkpoint"
    ready_checkpoint.mkdir()
    ready_command = tmp_path / "ready-command.py"
    ready_command.write_text("print('ok')\n", encoding="utf-8")
    discovery = evaluator.discover_wam_model_runtimes(
        candidates=("oscar_wam", "cosmos_wam"),
        generated_at="now",
        explicit_candidate_id="oscar_wam",
        explicit_command=f"{sys.executable} {ready_command}",
        explicit_checkpoint=ready_checkpoint,
    )
    assert discovery["selected_candidate"] == "oscar_wam"
    assert discovery["selected_candidate_blockers"] == []
    assert discovery["blockers"] == []
    assert "blocked_missing_wam_runtime" in discovery["all_candidate_blockers"]
    assert "local_host_probe" in discovery

    monkeypatch.setattr(evaluator.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(evaluator.shutil, "which", lambda _name: None)
    host_blocked = evaluator.discover_wam_model_runtimes(
        candidates=("oscar_wam",),
        generated_at="now",
        explicit_candidate_id="oscar_wam",
        explicit_command=f"{sys.executable} -m blueprint_pipeline.oscar_wam_command_adapter",
        explicit_checkpoint=ready_checkpoint,
    )
    host_blocked_row = host_blocked["candidates"][0]
    assert host_blocked_row["configured_command_checkpoint_ready"] is True
    assert host_blocked_row["provider_or_linux_cuda_runtime_required"] is True
    assert "blocked_oscar_linux_cuda_runtime_required" in host_blocked_row[
        "official_adapter_host_preflight_blockers"
    ]

    status_dir = tmp_path / "status-videos"
    _write_json(status_dir / "video_generation_status.json", {"videos": ["bad", {"path": "fallback.mp4"}]})
    assert evaluator._review_videos(status_dir) == [{"path": "fallback.mp4"}]


def test_oscar_cosmos_wam_command_and_rollout_edge_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_manifest = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    _write_json(input_manifest, {"schema_version": "input"})
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("cmd", 1)),
    )
    payload, detail = evaluator._run_local_wam_command(
        command="python missing.py",
        input_manifest_path=input_manifest,
        output_path=output_path,
        candidate_id="oscar_wam",
        checkpoint_path="checkpoint.pt",
        timeout_seconds=1,
    )
    assert payload == {}
    assert detail["blockers"][0].startswith("wam_model_command_failed:")

    class Completed:
        returncode = 0
        stdout = "not-json"
        stderr = ""

    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: Completed())
    payload, detail = evaluator._run_local_wam_command(
        command="python missing.py",
        input_manifest_path=input_manifest,
        output_path=output_path,
        candidate_id="oscar_wam",
        checkpoint_path=None,
        timeout_seconds=1,
    )
    assert payload == {}
    assert detail["blockers"] == ["wam_model_stdout_json_invalid"]

    class NonMappingCompleted:
        returncode = 0
        stdout = "[]"
        stderr = ""

    monkeypatch.setattr(evaluator.subprocess, "run", lambda *_args, **_kwargs: NonMappingCompleted())
    payload, detail = evaluator._run_local_wam_command(
        command="python missing.py",
        input_manifest_path=input_manifest,
        output_path=output_path,
        candidate_id="oscar_wam",
        checkpoint_path=None,
        timeout_seconds=1,
    )
    assert payload == {}
    assert detail["status"] == "completed"

    assert evaluator._wam_rollout_blocked_reason(["blocked_local_wam_model_run_not_enabled"]) == "blocked_local_wam_model_run_not_enabled"
    assert evaluator._wam_rollout_blocked_reason(["wam_model_command_failed:TimeoutExpired"]) == "blocked_wam_model_command_failed"
    assert evaluator._wam_rollout_blocked_reason(["blocked_missing_wam_model_checkpoint"]) == "blocked_missing_wam_model_checkpoint"
    assert evaluator._wam_rollout_blocked_reason(["blocked_missing_wam_runtime", "blocked_missing_wam_model_checkpoint"]) == "blocked_missing_wam_runtime_and_checkpoint"
    assert evaluator._wam_rollout_blocked_reason(["blocked_custom"]) == "blocked_custom"
    assert evaluator._wam_rollout_blocked_reason([]) == "blocked_missing_wam_model_runtime_or_checkpoint"
    assert evaluator._rollout_video_path({}, base_dir=tmp_path) is None
    assert evaluator._rollout_video_path({"generated_video_path": "video.mp4"}, base_dir=tmp_path) == tmp_path / "video.mp4"


def test_oscar_cosmos_wam_evaluator_reports_source_tree_without_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source_root = tmp_path / "source-only-oscar"
    source_root.mkdir()
    (source_root / "README.md").write_text("source checkout only", encoding="utf-8")
    (source_root / "camera.pt").write_bytes(b"not model weights")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", str(source_root))
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", raising=False)

    discovery = evaluator.discover_wam_model_runtimes(
        generated_at="now",
        candidates=("oscar_wam",),
    )
    oscar = discovery["candidates"][0]
    source = oscar["local_source_discovery"]

    assert source["source_tree_present"] is True
    assert source["present_source_tree_count"] >= 1
    assert "blocked_source_tree_present_without_runnable_adapter_command" in source["blockers"]
    assert "blocked_source_tree_present_without_configured_checkpoint" in source["blockers"]
    assert "blocked_missing_wam_runtime" in oscar["blockers"]
    assert "blocked_missing_wam_model_checkpoint" in oscar["blockers"]


def test_oscar_cosmos_wam_evaluator_reports_checkpointed_missing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_COMMAND", raising=False)
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "oscar-checkpoint"
    checkpoint.mkdir()

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_checkpointed_job",
        model_candidates=("oscar_wam",),
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["blockers"] == ["blocked_missing_wam_runtime"]
    assert summary["wam_generated_rollout_status"] == "blocked_missing_wam_runtime"
    generated = json.loads(
        (tmp_path / "wam_checkpointed_job" / "wam_generated_rollout_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["blocked_reason"] == "blocked_missing_wam_runtime"
    consistency = json.loads(
        (tmp_path / "wam_checkpointed_job" / "wam_consistency_checks.json").read_text(
            encoding="utf-8"
        )
    )
    assert consistency["generated_rollout_termination_reason"] == "blocked_missing_wam_runtime"


def test_oscar_cosmos_wam_evaluator_runs_configured_command_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_command.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
video = output.parent / "rollout_1.mp4"
video.write_bytes(b"mp4-placeholder")
payload = {
    "rollouts": [
        {
            "rollout_id": "rollout_1",
            "policy_id": "unit_test_wam",
            "scenario_eval_run_id": "run_1",
            "generated_video_path": str(video),
            "model_rollout_confidence": 0.42,
        }
    ]
}
output.write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({"status": "completed"}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_COMMAND", f"{sys.executable} {command_path}")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_model_job",
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is True
    assert summary["wam_generated_rollout_status"] == "completed"
    creation_plan = json.loads(
        (tmp_path / "wam_model_job" / "policy_model_endpoint_creation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert creation_plan["can_create_real_model_endpoint_now"] is True
    generated = json.loads(
        (tmp_path / "wam_model_job" / "wam_generated_rollout_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["rollout_count"] == 1
    truth = json.loads(
        (tmp_path / "wam_model_job" / "wam_evaluator_truth_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth["action_conditioned_video_rollout_generated"] is True


def test_oscar_cosmos_wam_evaluator_uses_env_command_and_blocks_missing_rollout_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_missing_video.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
payload = {"rollouts": ["bad", {"rollout_id": "rollout_missing", "generated_video_path": "missing.mp4"}]}
output.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_COMMAND", f"{sys.executable} {command_path}")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", str(checkpoint))

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_missing_video_job",
        model_candidates=("oscar_wam",),
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert "blocked_generated_rollout_video_missing" in summary["blockers"]


def test_policy_endpoint_readiness_reports_missing_file_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = tmp_path / "wam-command"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_text("weights", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    monkeypatch.setattr(evaluator, "normalize_model_access_env", lambda: None)
    monkeypatch.setattr(
        evaluator,
        "model_access_secret_status",
        lambda: {"huggingface": {"auth_ready": False}},
    )

    manifest = evaluator.build_policy_model_endpoint_readiness_manifest(
        generated_at="now",
        candidates=("oscar_wam",),
        explicit_candidate_id="oscar_wam",
        explicit_command=str(command),
        explicit_checkpoint=checkpoint,
    )

    row = manifest["candidates"][0]
    assert row["status"] == "blocked"
    assert row["model_access_auth_ready"] == {"huggingface": False}
    assert "configure_file_based_huggingface_auth" in row["what_is_needed_to_make_true"]
    assert "configure_file_based_huggingface_auth" in manifest["blockers"]


def test_oscar_cosmos_wam_evaluator_uses_default_job_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_root=tmp_path / "jobs",
        model_candidates=("oscar_wam",),
        generated_at="now",
    )

    assert Path(summary["job_dir"]).parent == tmp_path / "jobs"


def test_oscar_cosmos_wam_evaluator_propagates_adapter_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "wam_model_command.py"
    command_path.write_text(
        """
import json
import os
from pathlib import Path

output = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
output.write_text(json.dumps({
    "status": "blocked",
    "blockers": ["blocked_oscar_requires_cuda_gpu_runtime"],
}), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_model_job",
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["wam_generated_rollout_status"] == "blocked_oscar_requires_cuda_gpu_runtime"
    assert "blocked_oscar_requires_cuda_gpu_runtime" in summary["blockers"]


def test_configured_wam_command_failure_is_not_reported_as_missing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    input_job = _input_job(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "model"
    checkpoint.mkdir(parents=True)
    command_path = tmp_path / "failing_wam_model_command.py"
    command_path.write_text(
        "import sys\nsys.stderr.write('cuda runtime unavailable\\n')\nsys.exit(7)\n",
        encoding="utf-8",
    )

    summary = evaluator.run_oscar_cosmos_wam_evaluator(
        input_job_dir=input_job,
        job_dir=tmp_path / "wam_failed_runtime_job",
        wam_model_command=f"{sys.executable} {command_path}",
        wam_model_checkpoint=checkpoint,
        allow_wam_model_run=True,
        generated_at="now",
    )

    assert summary["learned_wam_model_ran"] is False
    assert summary["wam_generated_rollout_status"] == "blocked_wam_model_command_failed"
    assert "wam_model_command_nonzero_exit" in summary["blockers"]
    assert "blocked_missing_wam_runtime" not in summary["blockers"]
    assert "blocked_missing_wam_model_checkpoint" not in summary["blockers"]
    generated = json.loads(
        (tmp_path / "wam_failed_runtime_job" / "wam_generated_rollout_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["blocked_reason"] == "blocked_wam_model_command_failed"


def test_oscar_cosmos_wam_evaluator_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    input_job = _input_job(tmp_path)
    exit_code = evaluator.main(
        ["--input-job-dir", str(input_job), "--job-dir", str(tmp_path / "cli_wam_job")]
    )
    assert exit_code == 0
    assert '"learned_wam_model_ran": false' in capsys.readouterr().out
