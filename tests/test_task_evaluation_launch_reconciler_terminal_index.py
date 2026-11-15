"""The launch reconciler tick is the retention producer that files the owner
terminal receipts the scene-progression reconciler joins (R8).

Every tick it offers every launch run root to the launch bridge and indexes
every canary root under the dispatch root -- read-only over evidence,
idempotent, never launching or retrying. The website job's own launch is a
scene-configuration launch: the bridge passes it over silently, files nothing
for it and never blocks the owner-bound launch beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_reconciler import main as reconcile_main
from tests.test_task_evaluation_scene_terminal_reconciler import COMMIT, _env, _write
from tests.test_task_evaluation_scene_terminal_result_index import (
    RUN_ID, _files, _owner_launch_run, _terminal_dir,
)

SCENE_RUN_ID = "scene-839873-scene-configuration-run"


def _scene_configuration_launch_run(state_root: Path, *, launch_id: str = "launch-scene-1") -> Path:
    """The launch dispatcher's retained run root for a website scene-configuration launch.

    Its profile carries no owner policy binding: there is no policy run to file.
    """
    root = state_root / launch_id
    profile = {"schema_version": "task_evaluation_launch_profile.v1",
               "profile_id": "scene-configuration-profile", "source_commit": COMMIT,
               "claim_ceiling": "development_only", "profile_digest": ""}
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    request = {"schema_version": "task_evaluation_launch_request.v1", "run_id": SCENE_RUN_ID,
               "launch_id": launch_id, "source_commit": COMMIT,
               "request_digest": "sha256:" + "4" * 64, "launch_profile_digest": profile["profile_digest"]}
    _write(root / "launch_profile.json", profile)
    _write(root / "launch_request.json", request)
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1", "status": "queued_for_no_spend_preparation",
        "launch_id": launch_id, "run_id": SCENE_RUN_ID, "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"], "execute_requested": False,
        "provider_mutation_attempted": False, "allocator_invoked": False, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(root / "launch_receipt.json", receipt)
    identity = {key: receipt[key] for key in ("launch_id", "run_id", "request_digest", "receipt_digest")}
    attempt = {"schema_version": "task_evaluation_launch_webapp_sync_result.v1", "status": "succeeded",
               "provider_mutation_performed": False, "attempt_number": 1,
               "attempted_at": "2026-09-06T00:00:00+00:00", **identity,
               "response": {"schema_version": "task_evaluation_launch_web_sync_receipt.v1",
                            "status": receipt["status"], "already_exists": False, **identity},
               "sync_result_digest": ""}
    attempt["sync_result_digest"] = canonical_digest(attempt, digest_field="sync_result_digest")
    _write(root / "webapp_sync_succeeded.json", attempt)
    return root


def _tick(tmp_path, env, *, configured: bool = True):
    args = ["--queue-root", str(tmp_path / "queue"), "--state-root", str(tmp_path / "state"),
            "--guard-report", str(tmp_path / "guard-missing.json"), "--report-out", str(tmp_path / "report.json")]
    if configured:
        args += ["--policy-canary-dispatch-root", str(tmp_path / "canaries"),
                 "--terminal-result-root", env["config"]["terminal_result_root"],
                 "--scene-intent-root", env["config"]["intent_root"]]
    code = reconcile_main(args)
    return code, json.loads((tmp_path / "report.json").read_text())


def test_tick_files_the_owner_bridge_and_passes_over_the_scene_configuration_launch(tmp_path):
    env = _env(tmp_path)
    _owner_launch_run(tmp_path / "state", env)
    _scene_configuration_launch_run(tmp_path / "state")
    code, report = _tick(tmp_path, env)
    statuses = sorted(row["status"] for row in report["terminal_index"])
    assert statuses == ["launch_bridge_indexed"], report["terminal_index"]
    assert report["terminal_index"][0]["run_id"] == RUN_ID
    assert all(row["provider_mutation_performed"] is False for row in report["terminal_index"])
    assert code == 0 and report["status"] == "passed", report
    assert _files(_terminal_dir(env)) == {"launch_request.json", "launch_profile.json"}
    # The next tick is byte-identical and still reports the set as indexed.
    _, again = _tick(tmp_path, env)
    assert sorted(row["status"] for row in again["terminal_index"]) == statuses


def test_tick_without_terminal_roots_says_so_and_files_nothing(tmp_path):
    env = _env(tmp_path)
    _owner_launch_run(tmp_path / "state", env)
    _scene_configuration_launch_run(tmp_path / "state")
    _, report = _tick(tmp_path, env, configured=False)
    assert [row["status"] for row in report["terminal_index"]] == ["terminal_index_not_configured"]
    assert not Path(env["config"]["terminal_result_root"]).exists()


def test_tick_reports_an_index_failure_as_a_blocked_alarm(tmp_path):
    env = _env(tmp_path)
    _owner_launch_run(tmp_path / "state", env)
    scene_run = _scene_configuration_launch_run(tmp_path / "state")
    profile_path = scene_run / "launch_profile.json"
    profile = json.loads(profile_path.read_text())
    profile["profile_id"] = "drifted-after-sealing"
    profile_path.write_text(json.dumps(profile))
    code, report = _tick(tmp_path, env)
    blocked = [row for row in report["terminal_index"] if row["status"] == "terminal_index_blocked"]
    assert blocked and blocked[0]["blockers"] == ["terminal_result_index_launch_profile_invalid"]
    assert blocked[0]["launch_run_root"] == str(scene_run)
    assert code == 2 and report["status"] == "blocked"
    # The owner's bridge still filed (independent of the refused run).
    assert _files(_terminal_dir(env)) == {"launch_request.json", "launch_profile.json"}
