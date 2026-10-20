"""Pinned offline runtime and cohort commands, without weights or GPU calls."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline import flora_rewarder_runtime as runtime
from tests.test_episode_interpretation import _episode_root, _request
from tests.test_flora_episode_interpretation import _profile


@pytest.mark.parametrize("av_version", ["15.1.0", "16.1.0+local"])
def test_diagnostic_rejects_unpinned_pyav_before_preprocessing(tmp_path, monkeypatch, av_version):
    versions = {"vllm": "0.24.0", "transformers": "5.8.1", "qwen-vl-utils": "0.0.14",
                "decord": "0.6.0", "av": av_version, "numpy": "2.2.6", "Pillow": "12.2.0",
                "huggingface_hub": "1.21.0", "torch": "2.11.0+cu130"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(importlib, "import_module", lambda _: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    report_path = tmp_path / "diagnostic.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", str(tmp_path), str(tmp_path),
                                     str(tmp_path / "must-not-read-jobs.json"), str(report_path), "2", "cpu"])
    exec(runtime._DIAGNOSTIC, {})
    report = json.loads(report_path.read_text())
    assert report["blockers"] == ["dependency_version_mismatch:av"]
    assert report["media"] == []


def test_diagnostic_checkpoints_and_logs_each_completed_episode(tmp_path, monkeypatch, capsys):
    versions = {"vllm": "0.24.0", "transformers": "5.8.1", "qwen-vl-utils": "0.0.14",
                "decord": "0.6.0", "av": "16.1.0", "numpy": "2.2.6", "Pillow": "12.2.0",
                "huggingface_hub": "1.21.0", "torch": "2.11.0+cu130"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(importlib, "import_module", lambda _: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    report_path = tmp_path / "diagnostic.json"
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps([{"id": str(index), "videos": ["fake.mp4"], "task": "fixture",
        "rubric": [], "robot": "fixture", "cameras": ["fixture"]} for index in range(2)]))
    monkeypatch.setattr(sys, "argv", ["diagnostic", str(tmp_path), str(tmp_path),
                                     str(jobs_path), str(report_path), "2", "cpu"])
    def video_inputs(_):
        partial = json.loads(report_path.read_text())
        assert partial["completed"] is False
        assert len(partial["media"]) == len(checkpoints)
        checkpoints.append(partial)
    checkpoints = []
    core = SimpleNamespace(
        Collage=lambda _: SimpleNamespace(num_frames=1, fps=1, duration_s=1,
            frames=lambda _: None, release=lambda: None),
        dense_windows=lambda *a: [], terminal_frame_indices=lambda *a: [0],
        dense_regex=lambda: "fixture", terminal_messages=lambda *a: [],
        rubric_prompt=lambda *a: "fixture", rubric_header=lambda *a: "fixture",
        rubric_messages=lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "batch_utils", SimpleNamespace(core=core,
        engine=SimpleNamespace(load_processor=lambda _: None, render=lambda *a, **k: None),
        rubric_reward=SimpleNamespace(video_inputs=video_inputs)))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **k: None))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(StructuredOutputsParams=lambda **k: None))
    exec(runtime._DIAGNOSTIC, {})
    report = json.loads(report_path.read_text())
    assert report["completed"] is True and report["blockers"] == []
    assert len(checkpoints) == len(report["media"]) == 2
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["event"] for row in progress] == ["flora_preflight_episode_complete"] * 2
    assert [row["completed_episode_count"] for row in progress] == [1, 2]
    assert all(row["requested_episode_count"] == 2 for row in progress)


@pytest.mark.parametrize("cpu_only,profile_timeout,expected", [
    (True, 3600, 1800), (False, 3600, 300), (True, 73, 73), (False, 73, 73)])
def test_preflight_timeout_is_cpu_specific_and_respects_profile(tmp_path, monkeypatch, cpu_only, profile_timeout, expected):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    profile["runtime_config"].update(python_executable=__file__, timeout_seconds=profile_timeout)
    monkeypatch.setattr(runtime, "SOURCE_SHA256", {})
    monkeypatch.setattr(runtime, "MODEL_FILES", {})
    observed = []
    def execute(argv, **kwargs):
        observed.append(kwargs["timeout"])
        Path(argv[-3]).write_text(json.dumps({"blockers": [], "media": [], "completed": True}))
        return {"returncode": 0, "timed_out": False}
    monkeypatch.setattr(runtime, "_execute", execute)
    report = runtime._preflight_flora_runtime(profile, output_root=tmp_path / "preflight", cpu_import_only=cpu_only)
    assert observed == [expected]
    assert report["execute_ready"] is (not cpu_only)


@pytest.mark.parametrize("timed_out", [True, False])
def test_partial_cpu_diagnostic_is_retained_but_never_ready(tmp_path, monkeypatch, timed_out):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    profile["runtime_config"]["python_executable"] = __file__
    monkeypatch.setattr(runtime, "SOURCE_SHA256", {})
    monkeypatch.setattr(runtime, "MODEL_FILES", {})
    partial = {"blockers": [], "media": [{"id": str(index)} for index in range(7)],
               "completed": False, "requested_episode_count": 20}
    def execute(argv, **kwargs):
        Path(argv[-3]).write_text(json.dumps(partial))
        return {"returncode": -9 if timed_out else 0, "timed_out": timed_out}
    monkeypatch.setattr(runtime, "_execute", execute)
    report = runtime.preflight_flora_cpu_imports(profile, output_root=tmp_path / "preflight")
    assert report["diagnostics"] == partial
    assert report["execute_ready"] is False
    assert report["cpu_imports_and_preprocessing_ready"] is False
    assert "flora_runtime_diagnostic_incomplete" in report["blockers"]
    if timed_out:
        assert "flora_runtime_diagnostic_timed_out" in report["blockers"]


@pytest.mark.parametrize("parent_exit", [0, 1])
def test_subprocess_cleanup_stops_orphan_worker_after_parent_exit(tmp_path, parent_exit):
    log = tmp_path / "worker.log"
    script = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(p.pid,flush=True); "
        f"sys.exit({parent_exit})"
    )
    result = runtime._execute([sys.executable, "-c", script], log_path=log, timeout=5)
    assert result["returncode"] == parent_exit
    assert result["process_group_cleanup_attempted"] is True
    child = int(log.read_text().strip())
    # An orphan may briefly remain a zombie until init reaps it; neither an
    # absent process nor a zombie can retain a CUDA context or keep computing.
    for _ in range(50):
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(child)], capture_output=True, text=True).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.01)
    else:
        os.kill(child, 9)
        pytest.fail("orphan model worker survived process-group cleanup")


def test_missing_runtime_bytes_block_before_any_subprocess(tmp_path, monkeypatch):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    monkeypatch.setattr(runtime, "_execute", lambda *a, **k: pytest.fail("preflight must not execute"))
    report = runtime.preflight_flora_runtime(profile, output_root=tmp_path / "preflight")
    assert report["execute_ready"] is False
    assert any("source_missing" in blocker for blocker in report["blockers"])
    assert any("checkpoint_missing" in blocker for blocker in report["blockers"])
    assert report["learned_model_executed"] is False
    assert report["resource_allocation_performed"] is False


def test_pinned_source_tampering_is_detected_before_import(tmp_path, monkeypatch):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    source = tmp_path / "source"
    for name in runtime.SOURCE_SHA256:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise RuntimeError('must never execute')")
    profile["runtime_config"]["source_root"] = str(source)
    monkeypatch.setattr(runtime, "_execute", lambda *a, **k: pytest.fail("must not import changed code"))
    result = runtime.preflight_flora_runtime(profile, output_root=tmp_path / "preflight")
    assert "flora_source_digest_mismatch:batch_utils/core.py" in result["blockers"]


def test_cohort_uses_three_upstream_batch_commands_and_offline_pinned_model(tmp_path, monkeypatch):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    root = tmp_path / "runtime"
    root.mkdir()
    jobs = [{"id": f"blinded-{i}", "task": "Move cup", "videos": ["camera.mp4"],
             "robot": "arm", "cameras": ["top"], "rubric": profile["task"]["rubric"]} for i in range(20)]
    monkeypatch.setattr(runtime, "preflight_flora_runtime", lambda *a, **k: {
        "execute_ready": True, "verified_source_root": str(tmp_path / "verified")})
    calls = []
    def execute(argv, *, log_path, timeout):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.write_text("".join(json.dumps({"id": job["id"]}) + "\n" for job in jobs))
        return {"returncode": 0, "timed_out": False, "wall_time_seconds": 0.01}
    monkeypatch.setattr(runtime, "_execute", execute)
    result = runtime.FloraSubprocessRunner().run_batch(profile=profile, jobs=jobs, output_root=root)
    assert len(calls) == 3
    assert set(result["records"]) == set(runtime.MODES)
    for argv in calls:
        assert "--jobs" in argv
        assert "--input" not in argv
        assert "--overwrite" not in argv
        assert argv[argv.index("--model") + 1] == "/opt/blueprint/flora/model"
        assert argv[argv.index("--revision") + 1] == runtime.FLORA_REVISION
        assert "-I" in argv and "-B" in argv
    rubric = calls[-1]
    assert rubric[rubric.index("--num-samples") + 1] == "9"
    assert rubric[rubric.index("--max-frames") + 1] == "166"
    assert rubric[rubric.index("--fps") + 1] == "2"
    env = runtime._offline_environment()
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert "PYTHONPATH" not in env


def test_runtime_stops_after_failed_mode_without_reloading_for_remaining_modes(tmp_path, monkeypatch):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    root = tmp_path / "runtime"
    root.mkdir()
    monkeypatch.setattr(runtime, "preflight_flora_runtime", lambda *a, **k: {
        "execute_ready": True, "verified_source_root": str(tmp_path / "verified")})
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        return {"returncode": 1, "timed_out": False, "wall_time_seconds": 0.01}
    monkeypatch.setattr(runtime, "_execute", execute)
    result = runtime.FloraSubprocessRunner().run_batch(profile=profile, jobs=[], output_root=root)
    assert len(calls) == 1
    assert result["errors"] == ["flora_mode_execution_failed:terminal"]


def test_cpu_preflight_uses_same_diagnostic_without_claiming_gpu_or_weight_readiness(tmp_path, monkeypatch):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text("{}")
    profile["runtime_config"].update(model_path=str(model), python_executable=__file__)
    monkeypatch.setattr(runtime, "SOURCE_SHA256", {})
    monkeypatch.setattr(runtime, "MODEL_FILES", {
        "config.json": {"sha256": runtime.file_digest(config), "size_bytes": 2},
        "model-00001-of-00001.safetensors": {"sha256": "sha256:" + "a" * 64, "size_bytes": 1000}})
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        Path(argv[-3]).write_text(json.dumps({"blockers": [], "versions": {}, "media": [], "completed": True}))
        return {"returncode": 0, "timed_out": False}
    monkeypatch.setattr(runtime, "_execute", execute)
    result = runtime.preflight_flora_cpu_imports(profile, output_root=tmp_path / "cpu-preflight")
    assert result["execute_ready"] is False
    assert result["cpu_imports_and_preprocessing_ready"] is True
    assert result["cuda_checks_required"] is False
    assert result["model_weight_checks_required"] is False
    assert calls[0][-1] == "cpu"
    assert calls[0][4] == runtime._DIAGNOSTIC
    gpu = runtime.preflight_flora_runtime(profile, output_root=tmp_path / "gpu-preflight")
    assert gpu["execute_ready"] is False
    assert "flora_checkpoint_missing:model-00001-of-00001.safetensors" in gpu["blockers"]
    assert len(calls) == 1
