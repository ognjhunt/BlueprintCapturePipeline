"""The test job matrix: every variation ends where its row says, in parallel as in series, with no network.

The two full runs are marked slow (deselected by default); the guard, scrub and comparison tests are fast.
"""
# Covers: src/blueprint_pipeline/test_job_matrix.py
# Covers: tests/job_matrix_steps.py
# Covers: tests/fixtures/job_matrix_v1.json
from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from blueprint_pipeline import test_job_matrix as matrix

MATRIX = Path(__file__).resolve().parent / "fixtures" / "job_matrix_v1.json"


@pytest.mark.slow
def test_every_variation_ends_where_its_row_says_and_no_step_passes_unearned(tmp_path):
    pytest.importorskip("pxr")
    report = matrix.run_matrix(MATRIX, jobs=3, work_root=tmp_path)
    assert report["passed"], [(r["label"], r["differences"]) for r in report["variations"] if r["differences"]]
    for result in report["variations"]:
        statuses = [result["steps"][name]["status"] for name in result["steps"]]
        cut = statuses.index("stopped") if "stopped" in statuses else len(statuses)
        assert all(s in {"passed", "not_applicable", "not_run_here"} for s in statuses[:cut]), result["label"]
        assert all(s == "not_reached" for s in statuses[cut + 1:]), result["label"]
        for name, step in result["steps"].items():
            assert step["fed_from"] or step["status"] != "passed", (result["label"], name)
            assert step.get("reason") or step["status"] not in {"not_applicable", "not_run_here"}, (result["label"], name)
    assert report["replay_set"]["network_attempts"] == [] and all(r["network_attempts"] == [] for r in report["variations"])


@pytest.mark.slow
def test_series_and_parallel_give_the_same_report_apart_from_times(tmp_path):
    pytest.importorskip("pxr")
    rows = ["unknown-kind", "planar-push-not-run"]
    one = matrix.run_matrix(MATRIX, jobs=1, only=rows, work_root=tmp_path / "one")
    two = matrix.run_matrix(MATRIX, jobs=2, only=rows, work_root=tmp_path / "two")
    assert one["passed"] and two["passed"]
    assert [matrix.stable(r) for r in one["variations"]] == [matrix.stable(r) for r in two["variations"]]


def _reach_out() -> list:
    import socket
    import subprocess

    refused = []
    for call in (lambda: socket.create_connection(("example.test", 443), timeout=1),
                 lambda: socket.getaddrinfo("example.test", 443), lambda: socket.gethostbyname("example.test"),
                 lambda: subprocess.run(["curl", "https://example.test"], check=False)):
        try:
            call()
        except OSError as exc:
            refused.append(str(exc).split(":")[0])
    for argv in (["git", "fetch"], ["git", "-C", ".", "ls-remote", "origin"], ["git", "-c", "core.fsmonitor=x", "status"],
                 ["git", "-C", ".", "rev-parse", "--git-dir"], ["/tmp/git", "-C", ".", "rev-parse", "HEAD"]):
        try:
            subprocess.run(argv, capture_output=True, check=False)
        except OSError as exc:
            refused.append(str(exc).split(":")[0])
    git = subprocess.run(["git", "-C", str(matrix.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                         check=False).args[3]
    return [refused, [row["call"] for row in matrix._ATTEMPTS], git, os.environ.get("HTTPS_PROXY"),
            os.environ.get("MATRIX_PROBE_TOKEN")]


def test_a_variation_process_refuses_the_network_and_any_program_but_two_local_git_reads(monkeypatch):
    monkeypatch.setenv("MATRIX_PROBE_TOKEN", "x")
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=matrix._child_setup, initargs=(str(matrix.REPO_ROOT),)) as pool:
        refused, calls, git, proxy, token = pool.submit(_reach_out).result()
    assert refused == ["test_job_matrix_network_refused"] * 3 + ["test_job_matrix_subprocess_refused"] * 6
    assert calls == ["create_connection", "getaddrinfo", "gethostbyname"] + ["subprocess"] * 6
    assert git == "rev-parse"  # a local read of the two the rows use runs
    assert proxy == "http://127.0.0.1:9" and token is None
    assert os.environ["MATRIX_PROBE_TOKEN"] == "x"  # the caller's own environment is untouched


def test_a_difference_in_any_compared_field_fails_the_row():
    result = {"last_stage": "company_zip", "first_stop": {"kind": "completed", "code": "zip_published"},
              "physics_authority": "qualified_replacement_asset", "robot_free_probe": None,
              "network_attempts": [{"call": "connect"}]}
    assert matrix.compare(result, {"last_stage": "company_zip", "physics_authority": "static_candidate_no_simulator_run",
                                   "robot_free_probe": "not_queued"}) == [
        "physics_authority: expected 'static_candidate_no_simulator_run', got 'qualified_replacement_asset'",
        "robot_free_probe: expected 'not_queued', got None", "network_attempts: 1"]


def test_a_row_without_its_expected_end_is_refused(tmp_path):
    value = json.loads(MATRIX.read_text())
    del value["variations"][0]["expected"]["last_stage"]
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="incomplete"):
        matrix.load_matrix(path)
