"""The launch reconciler tick is the retention producer that files the owner
terminal receipts the scene-progression reconciler joins (R8).

Every tick it bridges every owner-bound policy-canary launch run and indexes
every canary root that carries a sealed ``dispatch_receipt.json`` -- read-only
over evidence, idempotent, never launching or retrying. The launch run root and
the canary root here come from the real producers (the launch-dispatcher run
shape and the REAL dispatcher resume path).
"""
from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.task_evaluation_launch_reconciler import main as reconcile_main
from tests.test_task_evaluation_policy_canary_dispatcher import materialize_canary_root
from tests.test_task_evaluation_scene_terminal_reconciler import _env
from tests.test_task_evaluation_scene_terminal_result_index import (
    RUN_ID, _files, _owner_launch_run, _reconcile, _terminal_dir,
)


def _tick(tmp_path, env, *, canary_root: Path, configured: bool = True):
    args = ["--queue-root", str(tmp_path / "queue"), "--state-root", str(tmp_path / "state"),
            "--guard-report", str(tmp_path / "guard-missing.json"), "--report-out", str(tmp_path / "report.json")]
    if configured:
        args += ["--policy-canary-dispatch-root", str(canary_root.parent),
                 "--terminal-result-root", env["config"]["terminal_result_root"],
                 "--scene-intent-root", env["config"]["intent_root"]]
    code = reconcile_main(args)
    return code, json.loads((tmp_path / "report.json").read_text())


def test_tick_files_the_owner_terminal_set_from_the_real_run_roots(tmp_path, monkeypatch):
    env = _env(tmp_path)
    _owner_launch_run(tmp_path / "state", env)
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    code, report = _tick(tmp_path, env, canary_root=run["root"])
    statuses = sorted(row["status"] for row in report["terminal_index"])
    assert statuses == ["launch_bridge_indexed", "policy_canary_terminal_indexed"], report["terminal_index"]
    assert all(row["provider_mutation_performed"] is False for row in report["terminal_index"])
    assert code == 0 and report["status"] == "passed", report
    assert "policy_canary_dispatch_receipt.json" in _files(_terminal_dir(env))
    assert _reconcile(env)["status"] == "blocked"
    # The next tick is byte-identical and still reports the set as indexed.
    _, again = _tick(tmp_path, env, canary_root=run["root"])
    assert sorted(row["status"] for row in again["terminal_index"]) == statuses


def test_tick_without_terminal_roots_says_so_and_files_nothing(tmp_path, monkeypatch):
    env = _env(tmp_path)
    _owner_launch_run(tmp_path / "state", env)
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    _, report = _tick(tmp_path, env, canary_root=run["root"], configured=False)
    assert [row["status"] for row in report["terminal_index"]] == ["terminal_index_not_configured"]
    assert not Path(env["config"]["terminal_result_root"]).exists()


def test_tick_reports_an_index_failure_as_a_blocked_alarm(tmp_path, monkeypatch):
    env = _env(tmp_path)
    _owner_launch_run(tmp_path / "state", env)
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    projection_path = run["root"] / "artifacts/result_delivery/policy_canary_result_projection.json"
    projection_path.write_text(projection_path.read_text() + "\n")  # drifted from the sealed receipt
    code, report = _tick(tmp_path, env, canary_root=run["root"])
    blocked = [row for row in report["terminal_index"] if row["status"] == "terminal_index_blocked"]
    assert blocked and blocked[0]["blockers"] == ["terminal_result_index_dispatch_receipt_binding_invalid"]
    assert blocked[0]["canary_run_root"] == str(run["root"])
    assert code == 2 and report["status"] == "blocked"
    # The bridge still filed (independent), the canary set did not.
    assert _files(_terminal_dir(env)) == {"launch_request.json", "launch_profile.json"}
