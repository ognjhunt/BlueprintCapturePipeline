"""Partial native evidence survives failures without fabricating a score."""
from copy import deepcopy
import json
import time

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_beta_runtime as runtime
from blueprint_pipeline.beta_policy_screening_g1 import _episode_outcome
from tests.test_beta_policy_screening_g1 import session_result, SHA, DP


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _stage_query(output, cell, query):
    path = output / "cells" / cell["cell_id"] / "episode/query_checkpoints/query_0000.v1.json"
    runtime._write(path, query)
    return path


def test_worker_failure_retains_actual_query_bytes_and_load_proof(tmp_path):
    plan, candidate, source_result = session_result()
    source = source_result["episodes"][0]
    cell = candidate["scenario_cells"][0]
    query_path = _stage_query(tmp_path, cell, source["query_checkpoints"][0])
    worker = _seal({"schema_version": "native_g1_development_worker_result.v1", "candidate_id": DP,
        "request_digest": SHA, "scene_plan_digest": SHA, "status": "blocked", "phase_reached": "episode",
        "blocker": {"message": "g1_beta_checkpoint_load_unverified"}}, "result_digest")
    cell_root = query_path.parents[2]
    runtime._write(cell_root / runtime.WORKER_RESULT, worker)
    result = runtime.collect_cell(root=tmp_path, cell_root=cell_root, cell=cell, scene_plan_digest=SHA,
        candidate_id=DP, request_digest=SHA, returncode=None, reason=None)
    assert result["status"] == "blocked"
    assert result["reason"] == "g1_beta_checkpoint_load_unverified"
    assert result["query_checkpoints"] == source["query_checkpoints"]
    assert len(result["artifact_records"]) == 2
    outcome, _reason, queried = _episode_outcome(result, candidate_id=DP,
        contract_digest=candidate["execution_spec"]["task_success_contract_digest"])
    assert outcome == "void" and queried


def test_timeout_without_worker_final_keeps_response_and_explicit_reason(tmp_path):
    _plan, candidate, source_result = session_result()
    cell = candidate["scenario_cells"][0]
    path = _stage_query(tmp_path, cell, source_result["episodes"][0]["query_checkpoints"][0])
    result = runtime.collect_cell(root=tmp_path, cell_root=path.parents[2], cell=cell,
        scene_plan_digest=SHA, candidate_id=DP, request_digest=SHA, returncode=None,
        reason="g1_beta_native_worker_deadline")
    assert result["reason"] == "g1_beta_native_worker_deadline"
    assert result["status"] == "blocked" and "worker_result" not in result
    assert len(result["query_checkpoints"]) == 1


def test_query_file_closure_and_scene_binding_are_not_self_asserted(tmp_path):
    _plan, candidate, source_result = session_result()
    cell = candidate["scenario_cells"][0]
    query = deepcopy(source_result["episodes"][0]["query_checkpoints"][0])
    query["scene_plan_digest"] = "sha256:" + "f" * 64
    _seal(query, "checkpoint_digest")
    path = _stage_query(tmp_path, cell, query)
    with pytest.raises(ValueError, match="checkpoint_binding_invalid"):
        runtime.collect_cell(root=tmp_path, cell_root=path.parents[2], cell=cell, scene_plan_digest=SHA,
            candidate_id=DP, request_digest=SHA, returncode=None, reason=None)
    path.write_text(json.dumps(source_result["episodes"][0]["query_checkpoints"][0]))
    (path.parent / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="checkpoint_closure_invalid"):
        runtime.collect_cell(root=tmp_path, cell_root=path.parents[2], cell=cell, scene_plan_digest=SHA,
            candidate_id=DP, request_digest=SHA, returncode=None, reason=None)


def test_unstarted_cells_are_individual_void_rows_and_no_query_claim():
    plan, candidate, _source_result = session_result()
    authority = {"candidate_ids": [DP], "cells": candidate["scenario_cells"],
        "authority_digest": SHA, "runtime_inputs_digest": SHA,
        "task_success_contract_digest": candidate["execution_spec"]["task_success_contract_digest"],
        "beta_screening_admission": {"plan_digest": plan["plan_digest"], "configuration_digest": candidate["configuration_digest"],
            "admission_digest": candidate["admission_digest"], "runtime_identity_digest": candidate["execution_spec"]["runtime_digest"],
            "source": plan["source"], "scenario_plan_digest": plan["scenario_plan_digest"]}}
    rows = [runtime._empty_cell(cell) for cell in authority["cells"]]
    result = runtime._result(authority, rows, stage="policy_dependencies", blockers=["g1_beta_policy_dependencies_failed"])
    assert len(result["episodes"]) == 4 and result["candidate_policy_queried"] is False
    assert all(row["status"] == "not_attempted" and row["reason"] for row in result["episodes"])
    assert "session_closeout" not in result
    assert result["physical_outcome_claimed"] is False


def test_missing_image_git_retains_each_cell_reason_without_install_or_model_fetch(tmp_path, monkeypatch):
    from tests.test_native_g1_beta_session import inputs
    from blueprint_pipeline.native_g1_beta_session import build_session_authority
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    authority = build_session_authority(plan=plan, candidate=candidate, spec=spec, attempt_id="screen-" + "a" * 32)
    monkeypatch.setattr(runtime, "_inputs", lambda root: (authority, spec, tmp_path / "native"))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    monkeypatch.setenv(runtime.DEADLINE_ENV, str(time.time() + 1800))
    monkeypatch.setattr(runtime, "_build_runtime", lambda *args: pytest.fail("unexpected bootstrap"))
    monkeypatch.setattr("blueprint_pipeline.native_g1_beta_download.stage_selected_assets", lambda **kwargs: pytest.fail("unexpected model fetch"))
    output = tmp_path / "output"
    output.mkdir()
    result = runtime.run_session(tmp_path, output)
    assert result["candidate_policy_queried"] is False
    assert len(result["episodes"]) == 4
    for row in result["episodes"]:
        assert row["reason"] == "g1_beta_image_git_unavailable"
        assert _episode_outcome(row, candidate_id=DP, contract_digest=spec["task_success_contract_digest"])[1] == row["reason"]


@pytest.mark.parametrize(("message", "expected"), [
    ("g1_beta_dependency_environment_invalid", "g1_beta_dependency_environment_invalid"),
    ("g1_beta_dependency_wheel_file_invalid:triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
     "g1_beta_dependency_wheel_file_invalid__triton_3_6_0_cp312_cp312_manylinux_2_27_x86_64_manylinux_2_28_x86_64_whl"),
    ("local_policy_download_bytes_changed", "g1_beta_policy_dependencies_local_policy_download_bytes_changed"),
    ("fetch failed for https://files.example/a.whl", "g1_beta_policy_dependencies_failed__valueerror"),
    ("g1_beta_x:/var/lib/blueprint/site", "g1_beta_policy_dependencies_failed__valueerror"),
])
def test_a_bootstrap_failure_keeps_a_bounded_specific_reason(message, expected):
    import re

    reason = runtime.failure_reason("policy_dependencies", ValueError(message))
    assert reason == expected
    assert re.fullmatch(r"g1_[a-z0-9_]{1,120}", reason)


def test_a_bootstrap_failure_reason_never_exceeds_the_projection_bound():
    import re

    reason = runtime.failure_reason("policy_dependencies", ValueError("g1_beta_dependency_wheel_file_invalid:" + "a" * 200 + ".whl"))
    assert re.fullmatch(r"g1_[a-z0-9_]{1,120}", reason)


def test_a_closed_cells_frames_go_once_its_videos_are_encoded(tmp_path):
    media = tmp_path / "episode" / "media"
    encoded, unencoded, empty = (media / name for name in ("dp", "no_video", "empty_video"))
    for directory in (encoded, unencoded, empty):
        (directory / "frames").mkdir(parents=True)
        (directory / "frames" / "head_0000.png").write_bytes(b"p" * 100)
        (directory / "multicamera_frame_manifest.json").write_text("{}")
    (encoded / "head.mp4").write_bytes(b"m" * 10)
    (empty / "head.mp4").write_bytes(b"")

    assert runtime.drop_encoded_frames(media) == 100
    assert not (encoded / "frames").exists() and (encoded / "head.mp4").is_file()
    assert (encoded / "multicamera_frame_manifest.json").is_file()
    assert (unencoded / "frames/head_0000.png").is_file() and (empty / "frames/head_0000.png").is_file()
    assert runtime.drop_encoded_frames(tmp_path / "missing") == 0


def test_symlinked_frames_are_never_followed(tmp_path):
    import os

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.png").write_bytes(b"k")
    directory = tmp_path / "media" / "dp"
    directory.mkdir(parents=True)
    (directory / "head.mp4").write_bytes(b"m")
    os.symlink(outside, directory / "frames")

    assert runtime.drop_encoded_frames(tmp_path / "media") == 0
    assert (outside / "keep.png").is_file()


def test_a_worker_that_keeps_growing_prints_progress_and_a_stalled_one_goes_quiet(tmp_path, monkeypatch, capsys):
    """2026-11-09: DP and pi0.5 each ended about 30 minutes after their provider log's last line, with no output:
    the worker writes only to its own log, and the controller watches only the provider log."""
    import itertools
    import re
    import subprocess

    from blueprint_pipeline import native_g1_beta_runtime as runtime

    class Worker:
        def __init__(self, argv, stdout, **_):
            self.log, self.calls, self.returncode, self.pid = stdout, 0, None, 1

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls <= 3:  # episodes running: the worker's log grows
                self.log.write(b"x" * 10)
                self.log.flush()
                raise subprocess.TimeoutExpired("worker", timeout)
            if self.calls <= 6:  # stalled: nothing grows, so nothing is printed
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = 0
            return 0

    clock = itertools.count(1_000_000, 61)
    monkeypatch.setattr(runtime.time, "time", lambda: next(clock))
    monkeypatch.setattr(runtime, "remaining_seconds", lambda _deadline: 5)
    monkeypatch.setattr(runtime, "_retained_bytes", lambda _output: 0)
    monkeypatch.setattr(runtime.subprocess, "Popen", Worker)
    cell_root = tmp_path / "cells" / "cell-000"
    cell_root.parent.mkdir()

    assert runtime._run_worker(tmp_path / "request.json", cell_root, deadline=10**12, output=tmp_path) == (0, None)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("BLUEPRINT_G1_")]
    assert lines[0] == "BLUEPRINT_G1_CELL_STARTED:cell-000"
    assert lines[-1] == "BLUEPRINT_G1_CELL_EXITED:cell-000:0:exited"
    working = [int(re.search(r":log_bytes=(\d+):", line).group(1)) for line in lines if "RUNTIME_WORKING" in line]
    # Each line reports growth; the stalled stretch adds none, so a hung worker still trips the controller.
    assert len(working) >= 2 and working == sorted(set(working))
    assert all("_PROGRESS:" not in line for line in lines)


def test_the_policy_runtime_build_prints_while_it_grows_and_nothing_while_it_stalls(tmp_path, monkeypatch, capsys):
    """The build fetches wheels and installs about 8 GB, bounded only by the session deadline: a slow fetch or disk
    could pass the controller's 30-minute window with nothing in the provider log."""
    import re
    import time

    from blueprint_pipeline import native_g1_beta_runtime as runtime

    monkeypatch.setattr(runtime, "PROGRESS_REPORT_SECONDS", 0.05)
    build = tmp_path / "policy-runtime-build"
    with runtime._GrowthReporter(build, "policy_runtime_build"):
        time.sleep(0.15)  # nothing there yet: no line
        build.mkdir()
        for index in range(3):
            (build / f"wheel-{index}").write_bytes(b"x" * 1000)
            time.sleep(0.15)
        time.sleep(0.3)  # stalled: no new lines
    sizes = [int(re.search(r":bytes=(\d+)$", line).group(1)) for line in capsys.readouterr().out.splitlines()
             if line.startswith("BLUEPRINT_G1_RUNTIME_WORKING:stage=policy_runtime_build:")]
    assert len(sizes) >= 2 and sizes == sorted(set(sizes)) and sizes[-1] == 3000


def test_a_compiling_policy_checks_its_toolchain_before_the_first_episode(tmp_path, monkeypatch, capsys):
    """2026-11-09, the library job's pi0.5 run: its first inference failed in torch.compile, ``Failed to find C
    compiler``, about 40 minutes in. The same compile on a tiny function now runs before any episode."""
    import json
    import subprocess
    import types

    from blueprint_pipeline import native_g1_beta_runtime as runtime

    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "config.json").write_text(json.dumps({"type": "pi05", "compile_model": True}))
    assert runtime.policy_compiles(policy)
    (policy / "config.json").write_text(json.dumps({"type": "diffusion", "compile_model": False}))
    assert not runtime.policy_compiles(policy)

    calls, envs = [], []

    def run(argv, **kwargs):
        calls.append(argv)
        envs.append(kwargs.get("env"))
        return outcome

    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/isaac-sim/kit/python/lib")
    monkeypatch.setenv("PYTHONHOME", "/isaac-sim/kit/python")
    outcome = types.SimpleNamespace(returncode=0, stdout="", stderr="")
    runtime.check_policy_compile(tmp_path / "python", deadline=runtime.time.time() + 3600)
    assert "torch.compile" in calls[0][-1] and "cuda" in calls[0][-1]
    # Started like the policy server: isolated from Isaac's Python variables, so it tests the venv's own torch.
    from blueprint_pipeline.native_g1_policy_server_supervisor import ISOLATED_POLICY_ENV_DROP
    assert runtime.ISOLATED_POLICY_ENV_DROP == ISOLATED_POLICY_ENV_DROP
    assert calls[0][1:3] == ["-I", "-c"]
    assert envs[0] is not None and not ISOLATED_POLICY_ENV_DROP & set(envs[0])
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["BLUEPRINT_G1_STAGE:policy_compile_check", "BLUEPRINT_G1_STAGE:policy_compile_ready"]

    outcome = types.SimpleNamespace(returncode=1, stdout="", stderr="Traceback\ntorch._inductor.exc.InductorError: "
                                    "RuntimeError: Failed to find C compiler.")
    with pytest.raises(ValueError, match=runtime.COMPILE_TOOLCHAIN_REASON):
        runtime.check_policy_compile(tmp_path / "python", deadline=runtime.time.time() + 3600)
    out = capsys.readouterr().out
    assert "BLUEPRINT_G1_COMPILE_CHECK:" in out and "Failed to find C compiler" in out
    assert runtime.failure_reason("policy_compile_check", ValueError(runtime.COMPILE_TOOLCHAIN_REASON)) \
        == runtime.COMPILE_TOOLCHAIN_REASON

    def timed_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

    monkeypatch.setattr(runtime.subprocess, "run", timed_out)
    with pytest.raises(ValueError, match=runtime.COMPILE_TOOLCHAIN_REASON):
        runtime.check_policy_compile(tmp_path / "python", deadline=runtime.time.time() + 3600)
