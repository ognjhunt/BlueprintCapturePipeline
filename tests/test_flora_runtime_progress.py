"""Artifact-based monitoring without model, provider, or grade claims."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import flora_provider_runtime as worker
from blueprint_pipeline.flora_runtime_progress import (
    FloraArtifactProgress, progress_cursor, progress_marker,
)


def _observer(tmp_path: Path):
    root = tmp_path / "batch"
    root.mkdir()
    lines = []
    observer = FloraArtifactProgress(batch_root=root, episode_ids={"episode-a", "episode-b"},
                                     instruction="Move the cup", emit=lines.append)
    return root, observer, lines


def _row(identifier: str) -> str:
    return json.dumps({"id": identifier, "task": "Move the cup"}) + "\n"


def test_preflight_and_modes_emit_only_actual_record_advancement(tmp_path):
    root, observer, lines = _observer(tmp_path)
    observer.poll()
    assert lines == []
    preflight = root / "preflight/diagnostic.json"
    preflight.parent.mkdir()
    preflight.write_text(json.dumps({"requested_episode_count": 2, "completed": False,
                                    "media": [{"id": "episode-a"}]}))
    observer.poll()
    assert lines == [progress_marker("preflight", 1, 2)]
    observer.poll()
    assert len(lines) == 1
    preflight.write_text(json.dumps({"requested_episode_count": 2, "completed": True,
        "media": [{"id": "episode-a"}, {"id": "episode-b"}]}))
    terminal = root / "terminal.jsonl"
    terminal.write_text(_row("episode-a") + '{"id":"episode-b"')
    observer.poll()
    assert lines[-2:] == [progress_marker("preflight", 2, 2), progress_marker("terminal", 1, 2)]
    terminal.write_text(_row("episode-a") + _row("episode-b"))
    observer.poll()
    (root / "dense.jsonl").write_text(_row("episode-a"))
    observer.poll()
    assert lines[-2:] == [progress_marker("terminal", 2, 2), progress_marker("dense", 1, 2)]
    assert all(line.endswith(":benchmark_complete=false") for line in lines)
    # Observation never writes a report, receipt, grade, or completion marker.
    assert not list(root.glob("*report*")) and not list(root.glob("*receipt*"))
    assert not any("COMPLETED_OR_BLOCKED" in line for line in lines)


@pytest.mark.parametrize("contents", [
    _row("episode-a") * 2,
    _row("foreign-episode"),
    json.dumps({"id": "episode-a", "task": "different task"}) + "\n",
    "Accepted publickey for root\n",
    "{bad-json}\n",
])
def test_duplicate_foreign_or_malformed_records_cannot_make_progress(tmp_path, contents):
    root, observer, lines = _observer(tmp_path)
    (root / "terminal.jsonl").write_text(contents)
    observer.poll()
    assert lines == []


def test_partial_bytes_metadata_changes_and_regression_never_refresh_progress(tmp_path):
    root, observer, lines = _observer(tmp_path)
    path = root / "terminal.jsonl"
    path.write_text(_row("episode-a"))
    observer.poll()
    expected = [progress_marker("terminal", 1, 2)]
    for tail in ('{"id":', '{"id":"episode-b"', '{"id":"episode-b","task":'):
        path.write_text(_row("episode-a") + tail)
        observer.poll()
        assert lines == expected
    os.utime(path, None)
    observer.poll()
    path.write_text("")
    observer.poll()
    assert lines == expected


def test_invalid_preflight_cohort_and_symlinks_do_not_count(tmp_path):
    root, observer, lines = _observer(tmp_path)
    preflight = root / "preflight/diagnostic.json"
    preflight.parent.mkdir()
    for report in (
        {"requested_episode_count": 3, "media": [{"id": "episode-a"}]},
        {"requested_episode_count": 2, "media": [{"id": "episode-a"}] * 2},
        {"requested_episode_count": 2, "media": [{"id": "foreign"}]},
    ):
        preflight.write_text(json.dumps(report))
        observer.poll()
        assert lines == []
    outside = tmp_path / "outside.jsonl"
    outside.write_text(_row("episode-a"))
    (root / "terminal.jsonl").symlink_to(outside)
    observer.poll()
    assert lines == []


def test_progress_cursor_uses_monotonic_stage_count_without_a_completion_claim():
    first = progress_marker("terminal", 1, 20)
    later = progress_marker("terminal", 2, 20)
    assert progress_cursor(first + "\n" + later + "\n" + first) == (2, 2)
    assert progress_cursor(progress_marker("dense", 0, 20)) == (3, 0)
    for text in ("sshd session opened", first.replace("false", "true"),
                 first.replace("records=1", "records=21"), first.replace("terminal", "completed")):
        assert progress_cursor(text) is None


def test_provider_supervisor_observes_child_artifacts_and_keeps_raw_stdout_private(tmp_path, monkeypatch):
    root, observer, lines = _observer(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    waits = []

    class Process:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, *, timeout):
            waits.append(timeout)
            if len(waits) <= 2:
                (root / "terminal.jsonl").write_text(_row("episode-a") +
                    (_row("episode-b") if len(waits) == 2 else ""))
                raise subprocess.TimeoutExpired("fixture-child", timeout)
            return 2

    def popen(command, **kwargs):
        assert command == ["fixture-child"] and kwargs["cwd"] == tmp_path
        assert kwargs["env"] == {"EXACT": "environment"}
        kwargs["stdout"].write("Accepted publickey for root\nraw-model-text\n")
        return Process()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    code = worker._run_benchmark_with_progress(["fixture-child"], benchmark=tmp_path,
        output=output, environment={"EXACT": "environment"}, progress=observer)
    assert code == 2 and waits == [5, 5, 5]
    assert lines == [progress_marker("terminal", 1, 2), progress_marker("terminal", 2, 2)]
    assert (output / "benchmark.log").read_text() == "Accepted publickey for root\nraw-model-text\n"
    assert not (output / "benchmark/benchmark_report.json").exists()


@pytest.mark.slow
def test_provider_supervisor_preserves_real_subprocess_exit_and_log(tmp_path):
    root, observer, lines = _observer(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    script = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); print('raw child log'); sys.exit(2)"
    code = worker._run_benchmark_with_progress(
        [sys.executable, "-I", "-B", "-c", script, str(root / "terminal.jsonl"), _row("episode-a")],
        benchmark=tmp_path, output=output, environment=dict(os.environ), progress=observer)
    assert code == 2 and lines == [progress_marker("terminal", 1, 2)]
    assert (output / "benchmark.log").read_text() == "raw child log\n"


def test_monitor_failure_cannot_change_child_result(tmp_path, monkeypatch):
    class Process:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, *, timeout):
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: Process())

    def unavailable():
        raise OSError("private diagnostic detail")

    assert worker._run_benchmark_with_progress(["fixture-child"], benchmark=tmp_path,
        output=tmp_path, environment={}, progress=SimpleNamespace(poll=unavailable)) == 0
    assert (tmp_path / "benchmark.log").read_text() == "FLORA_PROGRESS_OBSERVATION_UNAVAILABLE:OSError\n"


def test_provider_supervisor_preserves_direct_child_cancellation_cleanup(tmp_path, monkeypatch):
    events = []

    class Process:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("exit")

        def wait(self, *, timeout=None):
            if timeout is not None:
                raise KeyboardInterrupt
            events.append("wait_after_kill")
            return -9

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: Process())
    with pytest.raises(KeyboardInterrupt):
        worker._run_benchmark_with_progress(["fixture-child"], benchmark=tmp_path,
            output=tmp_path, environment={}, progress=SimpleNamespace(poll=lambda: None))
    assert events == ["kill", "wait_after_kill", "exit"]


def test_provider_startup_and_partial_outputs_never_claim_benchmark_completion(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import flora_dependency_bootstrap as bootstrap
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    root, output = tmp_path / "bundle", tmp_path / "output"
    runtime, benchmark = root / "provider_runtime", root / "benchmark"
    runtime.mkdir(parents=True)
    benchmark.mkdir()
    (runtime / "evaluator_provider_runtime_manifest.json").write_text(json.dumps({"source_commit": "a" * 40}))
    (runtime / "evaluator_input_manifest.json").write_text(json.dumps({
        "model_id": "nvidia/Flora-Rewarder", "model_revision": "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367",
        "profile_relative_path": "profile.json", "manifest_relative_path": "manifest.json",
        "rights_relative_path": "rights"}))
    digests = ["sha256:" + character * 64 for character in ("a", "b")]
    (benchmark / "manifest.json").write_text(json.dumps({"episodes": [{"input_bundle_digest": value} for value in digests]}))
    (benchmark / "profile.json").write_text(json.dumps({"task": {"instruction": "Move the cup"},
        "runtime_config": {"model_path": "/opt/blueprint/flora/model", "source_root": "/opt/blueprint/flora/source"}}))
    monkeypatch.setenv("BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR", str(root))
    monkeypatch.setenv("BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR", str(output))
    monkeypatch.setattr(worker.sys, "path", list(worker.sys.path))
    monkeypatch.setattr(worker, "_verify", lambda *a: None)
    monkeypatch.setattr(worker, "_offline_benchmark_input_preflight", lambda *a: {"eligible_episode_count": 2})
    monkeypatch.setattr(bootstrap, "ensure_flora_runtime_dependencies", lambda **kw: {"status": "ready"})
    versions = {"vllm": "0.24.0", "transformers": "5.8.1", "qwen-vl-utils": "0.0.14",
                "decord": "0.6.0", "av": "16.1.0", "numpy": "2.2.6", "Pillow": "12.2.0",
                "huggingface_hub": "1.21.0"}
    monkeypatch.setattr(worker.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(worker.importlib, "import_module", lambda _: None)
    model = tmp_path / "fixture-model"
    model.mkdir()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kw: str(model)))
    monkeypatch.setattr(worker, "Path", lambda value: tmp_path / Path(value).name
        if str(value).startswith("/opt/blueprint/flora/") else Path(value))

    def partial_child(command, **kwargs):
        progress = kwargs["progress"]
        expected = canonical_digest({"inputs": digests}).split(":")[1][:24]
        assert progress.batch_root == output / "benchmark/raw" / ("batch-" + expected)
        assert progress.episode_ids == {"episode-" + value[7:39] for value in digests}
        progress.batch_root.mkdir(parents=True)
        (progress.batch_root / "terminal.jsonl").write_text(json.dumps({
            "id": "episode-" + "a" * 32, "task": "Move the cup"}) + "\n")
        progress.poll()
        assert command[1:3] == ["-m", "blueprint_pipeline.flora_episode_benchmark"]
        return 2

    monkeypatch.setattr(worker, "_run_benchmark_with_progress", partial_child)
    assert worker.main() == 2
    result = json.loads((output / "evaluator_runtime_result.json").read_text())
    assert result["status"] == "blocked" and result["benchmark_exit_code"] == 2
    assert result["blockers"] == ["flora_benchmark_failed_or_report_missing"]
    assert result.get("runtime_execution_verified") is not True
    assert result.get("benchmark_grades_complete") is not True
    lines = capsys.readouterr().out.splitlines()
    assert lines == [progress_marker("startup", 1, 1), progress_marker("terminal", 1, 2)]
    assert all(line.endswith(":benchmark_complete=false") for line in lines)
