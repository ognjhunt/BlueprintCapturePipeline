"""The offline consent-withdrawal drill (tracker G-02, exit criterion E5): $0, temporary roots and fakes only.

Beta journey step: site withdrawal. The WebApp's shared withdrawal vector becomes a
tombstone, the intake's automatic deletion plan is recorded with no operator, a paid
admission for the site is refused, and an authorized apply on the synthetic site seals
a receipt naming every retention-policy data class.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_drill.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_deletion.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal.py
from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_automatic_plan as bounds
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_drill as drill

ROOT = Path(__file__).resolve().parents[1]
VECTOR = ROOT / "tests/fixtures/website-withdrawal-vector.json"
# The WebApp's strict acknowledgement schema (Blueprint-WebApp server/utils/websiteCaptureWithdrawal.ts,
# ``websiteWithdrawalReceipt``): an unknown key leaves the WebApp's withdrawal pending.
WEBAPP_RECEIPT_KEYS = {
    "schema_version", "request_id", "scene_id", "capture_id", "withdrawal_id", "command_digest", "tombstone_digest",
    "digest", "state", "pipeline_acknowledged", "serve_allowed", "future_processing_allowed",
    "local_cleanup_verified", "local_cleanup_receipt_digest", "deletion_confirmed", "provider_acknowledgement",
    "cloud_storage_cleanup_verified", "delivered_copy_cleanup_verified", "retained_audit_records", "legal_hold",
    "cloud_object_absence_verified", "cloud_object_cleanup_receipt_digest", "cloud_object_current_absence_status",
    "cloud_object_absence_observed_at_iso", "cleanup_scope", "external_cleanup_required"}


@pytest.fixture(autouse=True)
def _host_lock_in_tmp(tmp_path, monkeypatch):
    # Never the host's lock path: every automatic plan here locks inside the test's own directory.
    monkeypatch.setenv(bounds.LOCK_ENV, str(tmp_path / "host-lock/plan.lock"))


def test_withdrawal_drill_runs_offline_from_the_webapp_vector_to_a_sealed_receipt(tmp_path, capsys):
    result = drill.run_drill(tmp_path, vector_path=VECTOR)
    # (a) the paid screening launch's current-source check refuses the withdrawn site.
    assert result["paid_admission_refusal"] == "team_scene_capture_withdrawn"
    # The acknowledgement still parses under the WebApp's strict schema.
    assert set(result["acknowledgement"]) <= WEBAPP_RECEIPT_KEYS
    receipt = result["receipt"]
    # (b) every data class the retention policy names is in the receipt, removed or remaining with a reason.
    policy = json.loads((ROOT / "docs/beta_data_retention_policy_2026-07-09.json").read_text())
    classes = {row["class"]: row for row in receipt["policy_copy_classes"]}
    assert set(classes) == set(policy["data_classes"]) == set(deletion.POLICY_COPY_CLASSES)
    for row in classes.values():
        assert row["removed"] or row["remaining"]
        assert all(item["reason"] for item in row["remaining"])
    holders = {row["holder"] for row in receipt["remaining"]}
    assert {"gpu_and_remote_cpu_providers", "model_and_reconstruction_providers",
            "object_store_provider_physical_erasure", "webapp_and_delivered_copies",
            "pipeline_root_not_covered", "retained_audit_and_lineage_metadata"} <= holders
    assert "capture_source_bucket" not in holders  # the drill named the (fake) source bucket
    uncovered = next(row for row in receipt["remaining"] if row["holder"] == "pipeline_root_not_covered")
    assert uncovered["path"] == str(tmp_path / "inputs/sam31-preparations")
    assert "reproducible SAM preparation artifacts" in uncovered["reason"]
    # The counts the record prints: every covered copy went from some to none.
    counts = {row["copy_class"]: (row["before"], row["after"]) for row in result["record"] if row["before"] is not None}
    assert counts == {"scene_files": (5, 0), "run_directories": (3, 0), "run_pointers": (1, 0),
                      "b2_object_versions": (5, 0), "capture_source_bucket_objects": (1, 0)}
    assert receipt["covered_copies_deleted"] is True and receipt["provider_copies_deleted"] is False
    # (c) the other site's capture, run, B2 object and bucket object are untouched (checked inside the drill too).
    assert (tmp_path / "capture-partition/scenes/site-drill-other/captures/walkthrough-drill-other/raw/video.mp4"
            ).read_bytes() == b"synthetic-video drill-other"
    assert (tmp_path / "control-plane/task-evaluation-launch-runs/launch-drill-other-construction-0003").is_dir()
    with capsys.disabled():
        sys.stdout.write("\n" + drill.format_record(result["record"]) + "\n")


def test_the_deletion_plan_is_recorded_after_an_acknowledgement_with_no_operator_call(tmp_path):
    world = drill.build_world(tmp_path, "req1")
    vector = json.loads(VECTOR.read_text())
    before = drill._snapshot([tmp_path])
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"], command=vector["command"])
    started: list[list[str]] = []

    def launch(arguments):
        # The intake's own process runs exactly this command; here it runs in-process on the drill's roots.
        started.append(arguments)
        deletion.main([*arguments[3:], "--record-automatic-plan", *drill._cli_root_arguments(world["resolution"])])

    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "started"
    assert started == [[sys.executable, "-m", bounds.CHILD_MODULE, "--capture-root", str(world["capture"].absolute())]]
    assert bounds.read_state(world["site"] / "website_withdrawal")["state"] == "recorded"
    path = world["site"] / "website_withdrawal" / withdrawal.AUTOMATIC_PLAN_FILE
    record = json.loads(path.read_text())
    assert record["schema_version"] == deletion.AUTOMATIC_PLAN_SCHEMA
    assert record["started_by"] == "pipeline_after_withdrawal_acknowledgement" and record["applied"] is False
    assert record["plan_digest"] == record["plan"]["digest"]
    assert record["summary"]["runs"] == 4 and record["summary"]["scene_files"] == 5
    # No object store was listed: the intake's own run makes no provider call, so the objects stay unresolved.
    assert record["summary"]["object_store_listed"] is False
    assert record["summary"]["unresolved_reasons"] == {"website_withdrawal_deletion_object_store_unconfigured": 4}
    assert {row["holder"] for row in record["remaining"]} >= {"capture_source_bucket", "webapp_and_delivered_copies"}
    assert {row["class"] for row in record["policy_copy_classes"]} == set(deletion.POLICY_COPY_CLASSES)
    # Plan only: every file that existed before is still there, unchanged.
    after = drill._snapshot([tmp_path])
    assert {key: value for key, value in after.items() if key in before} == before
    assert all("/website_withdrawal/" in key or "/host-lock/" in key for key in set(after) - set(before))
    # Recorded once: the next acknowledgement starts nothing, and a record never changes.
    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "recorded"
    assert len(started) == 1 and json.loads(path.read_text()) == record


def test_the_automatic_plan_command_refuses_apply_and_needs_a_tombstone(tmp_path):
    world = drill.build_world(tmp_path, "req1")
    roots = drill._cli_root_arguments(world["resolution"])
    with pytest.raises(ValueError, match="website_withdrawal_tombstone_required"):
        withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=lambda _: None)
    vector = json.loads(VECTOR.read_text())
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"], command=vector["command"])
    with pytest.raises(deletion.WithdrawalDeletionError, match="automatic_plan_never_applies"):
        deletion.main(["--capture-root", str(world["capture"]), "--record-automatic-plan", "--apply",
                       "--authorize-complete-deletion", *roots])
    assert not (world["site"] / "website_withdrawal" / withdrawal.AUTOMATIC_PLAN_FILE).exists()


def test_drill_entry_point_prints_the_record(tmp_path, capsys):
    assert drill.main(["--work-root", str(tmp_path / "drill"), "--vector", str(VECTOR)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("copy_class | before | after | remaining_reason")
    assert "paid admission after tombstone: refused (team_scene_capture_withdrawn)" in out


def test_a_run_root_the_planner_cannot_list_is_unresolved_not_skipped(tmp_path, monkeypatch):
    world = drill.build_world(tmp_path, "req1")
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"],
                                      command=json.loads(VECTOR.read_text())["command"])
    unreadable = world["resolution"]["run_roots"][0]
    original = Path.iterdir

    def iterdir(path):
        if path == unreadable:
            raise PermissionError("synthetic")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    plan = deletion.plan_withdrawal_deletion(capture_root=world["capture"], **world["resolution"])
    assert {"path": str(unreadable), "reason": "website_withdrawal_deletion_root_unreadable"} in plan["unresolved"]


def _acknowledged(tmp_path, request_id="req1"):
    world = drill.build_world(tmp_path / request_id, request_id)
    command = json.loads(VECTOR.read_text())["command"]
    if request_id != "req1":
        command = {**command, "request_id": request_id, "scene_id": f"site-{request_id}",
                   "capture_id": f"walkthrough-{request_id}"}
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"], command=command)
    world["journal"] = world["site"] / "website_withdrawal"
    return world


def _plan_path(world):
    return world["journal"] / withdrawal.AUTOMATIC_PLAN_FILE


def _state(world):
    return bounds.read_state(world["journal"])


def _in_process(world):
    """A launcher that runs the child's plan-only command in this process, on the world's roots."""
    def launch(arguments):
        deletion.main([*arguments[3:], "--record-automatic-plan", *drill._cli_root_arguments(world["resolution"])])
    return launch


def _alive_launcher(started):
    """A launcher whose 'child' is this live process: the state stays launched while it runs."""
    def launch(arguments):
        started.append(arguments)
        return os.getpid()
    return launch


def test_one_automatic_plan_at_a_time_on_the_host_and_a_second_request_leaves_it_owed(tmp_path):
    first, second = _acknowledged(tmp_path, "req1"), _acknowledged(tmp_path, "req2")
    started = []
    # A burst: the first acknowledgement launches; the decision is taken under the host lock with the
    # child's pid as holder, so every later request, for this site or another, starts nothing.
    assert withdrawal.request_automatic_deletion_plan(capture_root=first["capture"],
                                                      launch=_alive_launcher(started)) == "started"
    assert _state(first)["state"] == "launched" and _state(first)["last_launch"]["pid"] == os.getpid()
    assert withdrawal.request_automatic_deletion_plan(capture_root=first["capture"],
                                                      launch=_alive_launcher(started)) == "running"
    assert withdrawal.request_automatic_deletion_plan(capture_root=second["capture"],
                                                      launch=_alive_launcher(started)) == "owed"
    assert len(started) == 1
    assert _state(second)["state"] == "owed" and _state(second)["owed_behind"] == str(first["journal"])
    # A child that finds the lock held scans nothing and says what it is behind.
    lock = bounds.host_lock()
    with bounds.hold_host_lock(lock, first["journal"]) as held:
        assert held
        assert deletion.record_automatic_plan(capture_root=second["capture"], **second["resolution"]) == {"state": "owed"}
    assert not _plan_path(second).exists() and _state(second)["failed_attempts"] == 0


def test_a_spawn_that_raises_is_recorded_as_a_failure(tmp_path):
    world = _acknowledged(tmp_path)

    def refuse(_arguments):
        raise BlockingIOError(11, "Resource temporarily unavailable")

    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=refuse) == "failed"
    state = _state(world)
    assert state["state"] == "failed" and state["last_failure"]["reason"] == "automatic_plan_spawn_failed_EAGAIN"
    assert state["last_launch"]["attempt"] == 1 and state["next_attempt_not_before_epoch"] > time.time() + 3500
    assert not bounds.host_lock().with_name(bounds.host_lock().name + ".holder.json").exists()


def test_a_child_that_dies_before_the_lock_is_recorded_on_the_next_request(tmp_path):
    import subprocess

    world = _acknowledged(tmp_path)
    # The real child command, with an argument its parser refuses: it dies before the lock and records nothing.
    environment = bounds.child_environment()
    environment[bounds.LOCK_ENV] = str(bounds.host_lock())
    child = []

    def launch(arguments):
        process = subprocess.Popen([*arguments, "--no-such-argument"], env=environment, cwd="/",
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child.append(process)
        return process.pid

    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "started"
    assert _state(world)["state"] == "launched"
    assert child[0].wait(timeout=60) != 0
    assert _state(world)["state"] == "launched" and not _plan_path(world).exists()
    # The next acknowledgement finds the launched child gone with no plan: a failure, then the backoff.
    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "backoff"
    state = _state(world)
    assert state["state"] == "failed"
    assert state["last_failure"]["reason"] == "automatic_plan_child_died_before_recording"
    assert len(child) == 1


def test_failures_back_off_and_stop_after_the_cap_leaving_a_record_for_an_operator(tmp_path, monkeypatch, caplog):
    world = _acknowledged(tmp_path)
    clock = [time.time()]
    monkeypatch.setattr(bounds.time, "time", lambda: clock[0])
    original = deletion.plan_withdrawal_deletion

    def fail(**_kwargs):
        raise bounds.PlanTimedOut()

    monkeypatch.setattr(deletion, "plan_withdrawal_deletion", fail)
    launches = []

    def launch(arguments):
        launches.append(arguments)
        with pytest.raises(bounds.PlanTimedOut):
            _in_process(world)(arguments)

    for attempt, wait in enumerate(bounds.BACKOFF_SECONDS, start=1):
        assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "started"
        state = _state(world)
        assert state["state"] == "failed" and state["failed_attempts"] == attempt
        assert state["last_failure"]["reason"] == "automatic_plan_timed_out"
        # Within the backoff nothing starts (the hourly resend just finds it waiting).
        clock[0] += wait - 1
        assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "backoff"
        clock[0] += 2
    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "started"
    state = _state(world)
    assert state["state"] == "stopped_needs_operator" and state["failed_attempts"] == bounds.MAX_FAILED_ATTEMPTS
    assert "run the plan by hand" in state["needs_operator"]
    # Stopped: no automatic attempt ever again, and every acknowledgement logs it.
    clock[0] += 30 * 24 * 3600
    with caplog.at_level("WARNING"):
        assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "stopped"
    assert "website_withdrawal_automatic_plan_stopped" in caplog.text
    assert len(launches) == bounds.MAX_FAILED_ATTEMPTS
    # An operator's plan by hand still lands, and supersedes the failure.
    monkeypatch.setattr(deletion, "plan_withdrawal_deletion", original)
    with contextlib.redirect_stdout(io.StringIO()):
        _in_process(world)(["python", "-m", bounds.CHILD_MODULE, "--capture-root", str(world["capture"])])
    state = _state(world)
    assert state["state"] == "recorded" and state["last_failure"] is None
    assert state["superseded_failure"]["reason"] == "automatic_plan_timed_out"


def test_a_dead_or_overdue_lock_holder_is_stale_and_a_stuck_one_stays_the_holder(tmp_path, monkeypatch):
    import subprocess

    dead_site, overdue_site, stuck_site, next_site = (
        _acknowledged(tmp_path, name) for name in ("req1", "req2", "req3", "req4"))
    lock = bounds.host_lock()
    lock.parent.mkdir(parents=True)
    holder = lock.with_name(lock.name + ".holder.json")
    finished = subprocess.Popen(["true"])
    finished.wait()
    holder.write_text(json.dumps({"pid": finished.pid, "process_start": "1", "started_at_epoch": time.time(),
                                  "journal": str(dead_site["journal"])}))
    started = []
    assert withdrawal.request_automatic_deletion_plan(capture_root=next_site["capture"],
                                                      launch=_alive_launcher(started)) == "started"
    assert _state(dead_site)["last_failure"]["reason"] == "automatic_plan_process_died"
    holder.unlink()
    bounds.write_state(next_site["journal"], "recorded")
    overdue = time.time() - bounds.WALL_CLOCK_SECONDS - bounds.STALE_GRACE_SECONDS - 1
    hung = subprocess.Popen(["sleep", "60"])
    try:
        holder.write_text(json.dumps({"pid": hung.pid, "process_start": bounds._process(hung.pid)[1],
                                      "started_at_epoch": overdue, "journal": str(overdue_site["journal"])}))
        with open(lock, "a+") as handle:  # the hung holder's flock
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            assert bounds.reap_stale(lock) == "reaped"
        assert hung.wait(timeout=10) == -9
        assert _state(overdue_site)["last_failure"]["reason"] == "automatic_plan_timed_out_killed"
        assert not holder.exists()
    finally:
        hung.kill()
    # A process that survives SIGKILL (uninterruptible IO) is not cleared: it stays the holder, recorded stuck.
    survivor = subprocess.Popen(["sleep", "60"])
    try:
        record = {"pid": survivor.pid, "process_start": bounds._process(survivor.pid)[1],
                  "started_at_epoch": overdue, "journal": str(stuck_site["journal"])}
        holder.write_text(json.dumps(record))
        monkeypatch.setattr(bounds.os, "kill", lambda *_: None)  # the kill is delivered but the process stays
        monkeypatch.setattr(bounds.time, "sleep", lambda _: None)
        assert bounds.reap_stale(lock) == "fresh"
        assert json.loads(holder.read_text()) == record
        assert _state(stuck_site)["state"] == "stuck"
        assert _state(stuck_site)["stuck"]["reason"] == "automatic_plan_survived_sigkill"
    finally:
        survivor.kill()


@pytest.mark.parametrize("error, reason", [
    (lambda: bounds.PlanTimedOut(), "automatic_plan_timed_out"),
    (lambda: RuntimeError("boom"), "automatic_plan_failed_RuntimeError"),
    (lambda: deletion.WithdrawalDeletionError("website_withdrawal_deletion_record_unreadable"),
     "website_withdrawal_deletion_record_unreadable")])
def test_a_failed_or_timed_out_plan_is_recorded_beside_the_tombstone(tmp_path, monkeypatch, error, reason):
    world = _acknowledged(tmp_path)

    def fail(**_kwargs):
        raise error()

    monkeypatch.setattr(deletion, "plan_withdrawal_deletion", fail)
    with pytest.raises(type(error())):
        deletion.record_automatic_plan(capture_root=world["capture"], **world["resolution"])
    state = _state(world)
    assert state["state"] == "failed" and state["last_failure"]["reason"] == reason
    assert state["failed_attempts"] == 1 and state["last_failure"]["at_iso"]
    assert not _plan_path(world).exists()
    assert not bounds.host_lock().with_name(bounds.host_lock().name + ".holder.json").exists()


def test_a_dangling_symlink_at_the_plan_path_is_refused_without_a_scan(tmp_path, monkeypatch):
    world = _acknowledged(tmp_path)
    _plan_path(world).symlink_to(tmp_path / "nowhere.json")
    scans = []
    monkeypatch.setattr(deletion, "plan_withdrawal_deletion", lambda **kwargs: scans.append(kwargs))
    started = []
    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=started.append) == "refused"
    assert started == []
    assert _state(world)["last_failure"]["reason"] == "website_withdrawal_deletion_automatic_plan_path_unsafe"
    with pytest.raises(deletion.WithdrawalDeletionError, match="automatic_plan_path_unsafe"):
        deletion.record_automatic_plan(capture_root=world["capture"], **world["resolution"])
    assert scans == [] and _plan_path(world).is_symlink() and not (tmp_path / "nowhere.json").exists()
    # Counted like any failure, so it stops too instead of refusing forever at no record.
    for _ in range(bounds.MAX_FAILED_ATTEMPTS):
        withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=started.append)
    assert _state(world)["state"] == "stopped_needs_operator" and started == []


def test_the_child_environment_is_the_explicit_list_and_no_secret():
    environment = bounds.child_environment({
        "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT": "/q", "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE": "/p",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE": "/etc/secret",
        "BLUEPRINT_PIPELINE_SYNC_TOKEN_PATH": "/etc/token", "PIPELINE_SYNC_WEBAPP_URL": "https://x",
        "BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL": "https://hook", "AWS_SECRET_ACCESS_KEY": "s"})
    assert set(environment) == {"PATH", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", bounds.LOCK_ENV,
                                "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT", "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE"}


_RECORD_READS = """
import contextlib, io, json, os, sys
class Reads(dict):
    names = set()
    def get(self, key, default=None):
        Reads.names.add(key)
        return dict.get(self, key, default)
    def __getitem__(self, key):
        Reads.names.add(key)
        return dict.__getitem__(self, key)
    def __contains__(self, key):
        Reads.names.add(key)
        return dict.__contains__(self, key)
os.environ = Reads(os.environ)
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
with contextlib.redirect_stdout(io.StringIO()):
    deletion.main(sys.argv[1:])
print(json.dumps(sorted(Reads.names)))
"""


def test_every_variable_the_plan_reads_for_a_root_is_passed_to_the_child(tmp_path):
    import subprocess

    world = _acknowledged(tmp_path)
    environment = bounds.child_environment()
    completed = subprocess.run(
        [sys.executable, "-c", _RECORD_READS, "--capture-root", str(world["capture"]), "--record-automatic-plan",
         *drill._cli_root_arguments(world["resolution"])], env=environment, cwd="/", capture_output=True,
        text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    read = {name for name in json.loads(completed.stdout.splitlines()[-1]) if name.startswith("BLUEPRINT_")}
    assert read - {bounds.LOCK_ENV} <= set(bounds.PLAN_ROOT_VARIABLES), sorted(read - set(bounds.PLAN_ROOT_VARIABLES))


def test_the_bounded_child_applies_its_limits_first_and_records_its_plan(tmp_path):
    import resource
    import subprocess

    world = _acknowledged(tmp_path)
    command = [sys.executable, "-m", bounds.CHILD_MODULE, "--capture-root", str(world["capture"]),
               *drill._cli_root_arguments(world["resolution"]), "--uncovered-root", str(tmp_path / "absent")]
    completed = subprocess.run(command, env=bounds.child_environment(), cwd="/", capture_output=True, text=True,
                               timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    record = json.loads(_plan_path(world).read_text())
    limits = record["child_limits"]
    assert limits["order"][:3] == ["oom_score_adj_1000", "nice_19", "io_class_idle"]  # applied first
    assert limits["wall_clock_seconds"] == bounds.WALL_CLOCK_SECONDS
    assert limits["address_space_bytes"] is True and limits["cpu_seconds"] is True and limits["nice_19"] is True
    assert _state(world)["state"] == "recorded"
    # The child refuses --apply before importing anything heavy.
    refused = subprocess.run([*command, "--apply"], env=bounds.child_environment(), cwd="/", capture_output=True,
                             text=True, timeout=60, check=False)
    assert refused.returncode != 0 and "never_applies" in refused.stderr
    # This test process itself is not bounded.
    assert resource.getrlimit(resource.RLIMIT_AS)[0] != bounds.ADDRESS_SPACE_BYTES


def test_a_request_that_fails_before_deciding_is_recorded(tmp_path, monkeypatch):
    world = _acknowledged(tmp_path)

    def broken(*_args, **_kwargs):
        raise PermissionError(13, "lock directory not writable")

    monkeypatch.setattr(bounds, "_open_lock", broken)
    with pytest.raises(PermissionError):
        withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=lambda _: None)
    assert _state(world)["last_failure"]["reason"] == "automatic_plan_request_failed_PermissionError"
