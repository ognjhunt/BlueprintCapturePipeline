"""Pinned offline runtime and cohort commands, without weights or GPU calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from blueprint_pipeline import flora_rewarder_runtime as runtime
from tests.test_episode_interpretation import _episode_root, _request
from tests.test_flora_episode_interpretation import _profile


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
        Path(argv[-3]).write_text(json.dumps({"blockers": [], "versions": {}, "media": []}))
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
