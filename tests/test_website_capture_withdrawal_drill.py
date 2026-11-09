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

import json
import sys
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
        deletion.main([*arguments[3:], *drill._cli_root_arguments(world["resolution"])])

    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=launch) == "started"
    assert started == [[sys.executable, "-m", "blueprint_pipeline.website_capture_withdrawal_deletion",
                        "--capture-root", str(world["capture"].absolute()), "--record-automatic-plan"]]
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
    return world


def _plan_path(world):
    return world["site"] / "website_withdrawal" / withdrawal.AUTOMATIC_PLAN_FILE


def _failure(world):
    return json.loads((world["site"] / "website_withdrawal" / bounds.FAILURE_FILE).read_text())


def test_one_automatic_plan_at_a_time_on_the_host_and_a_second_request_leaves_it_owed(tmp_path):
    first, second = _acknowledged(tmp_path, "req1"), _acknowledged(tmp_path, "req2")
    lock = bounds.host_lock()
    started = []
    with bounds.hold_host_lock(lock, first["site"] / "website_withdrawal") as held:
        assert held
        # Another site's request while the first scan runs starts nothing, scans nothing and records nothing.
        assert withdrawal.request_automatic_deletion_plan(capture_root=second["capture"], launch=started.append) == "owed"
        assert deletion.record_automatic_plan(capture_root=second["capture"], **second["resolution"]) == {"state": "owed"}
        assert started == [] and not _plan_path(second).exists()
        assert not (second["site"] / "website_withdrawal" / bounds.FAILURE_FILE).exists()
    # Once it is free, the next acknowledgement picks the owed plan up.
    assert withdrawal.request_automatic_deletion_plan(capture_root=second["capture"], launch=started.append) == "started"
    assert deletion.record_automatic_plan(capture_root=second["capture"], **second["resolution"])["applied"] is False


def test_a_dead_or_overdue_lock_holder_is_stale_and_its_site_gets_a_failure_record(tmp_path):
    import subprocess
    import time

    dead_site, overdue_site, next_site = (_acknowledged(tmp_path, name) for name in ("req1", "req2", "req3"))
    lock = bounds.host_lock()
    lock.parent.mkdir(parents=True)
    finished = subprocess.Popen(["true"])
    finished.wait()
    holder = lock.with_name(lock.name + ".holder.json")
    holder.write_text(json.dumps({"pid": finished.pid, "process_start": "1", "started_at_epoch": time.time(),
                                  "journal": str(dead_site["site"] / "website_withdrawal")}))
    started = []
    # Dead: nothing holds the flock, so the request records the death and starts the plan.
    assert withdrawal.request_automatic_deletion_plan(capture_root=next_site["capture"], launch=started.append) == "started"
    assert _failure(dead_site)["reason"] == "automatic_plan_process_died" and not holder.exists()
    # Overdue: a live holder past the wall clock plus grace is killed, only if its start time still matches.
    hung = subprocess.Popen(["sleep", "60"])
    try:
        record = {"pid": hung.pid, "process_start": bounds._process_start(hung.pid),
                  "started_at_epoch": time.time() - bounds.WALL_CLOCK_SECONDS - bounds.STALE_GRACE_SECONDS - 1,
                  "journal": str(overdue_site["site"] / "website_withdrawal")}
        holder.write_text(json.dumps(record))
        with open(lock, "a+") as handle:  # the hung holder's flock
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            assert withdrawal.request_automatic_deletion_plan(capture_root=next_site["capture"],
                                                              launch=started.append) == "started"
        assert hung.wait(timeout=10) == -9
        assert _failure(overdue_site)["reason"] == "automatic_plan_timed_out_killed"
        # A live holder within its limit stays the holder: the request leaves the plan owed.
        fresh = subprocess.Popen(["sleep", "60"])
        holder.write_text(json.dumps({**record, "pid": fresh.pid, "process_start": bounds._process_start(fresh.pid),
                                      "started_at_epoch": time.time()}))
        with open(lock, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            assert withdrawal.request_automatic_deletion_plan(capture_root=next_site["capture"],
                                                              launch=started.append) == "owed"
        assert fresh.poll() is None
        fresh.kill()
        fresh.wait()
    finally:
        hung.kill()
    assert len(started) == 2


@pytest.mark.parametrize("error, reason", [
    (lambda: bounds.PlanTimedOut(), "automatic_plan_timed_out"),
    (lambda: RuntimeError("boom"), "automatic_plan_failed_RuntimeError"),
    (lambda: deletion.WithdrawalDeletionError("website_withdrawal_deletion_record_unreadable"),
     "website_withdrawal_deletion_record_unreadable")])
def test_a_failed_or_timed_out_plan_is_recorded_beside_the_tombstone_and_retried(tmp_path, monkeypatch, error, reason):
    world = _acknowledged(tmp_path)
    original = deletion.plan_withdrawal_deletion

    def fail(**_kwargs):
        raise error()

    monkeypatch.setattr(deletion, "plan_withdrawal_deletion", fail)
    with pytest.raises(type(error())):
        deletion.record_automatic_plan(capture_root=world["capture"], **world["resolution"])
    failure = _failure(world)
    assert failure["reason"] == reason and failure["attempts"] == 1 and failure["at_iso"]
    assert not _plan_path(world).exists()
    assert not bounds.host_lock().with_name(bounds.host_lock().name + ".holder.json").exists()
    # The next acknowledgement retries, and a later success lands.
    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=lambda _: None) == "started"
    monkeypatch.setattr(deletion, "plan_withdrawal_deletion", original)
    assert deletion.record_automatic_plan(capture_root=world["capture"], **world["resolution"])["applied"] is False


def test_a_dangling_symlink_at_the_plan_path_is_refused_once_per_request_without_a_scan(tmp_path, monkeypatch):
    world = _acknowledged(tmp_path)
    _plan_path(world).symlink_to(tmp_path / "nowhere.json")
    scans = []
    monkeypatch.setattr(deletion, "plan_withdrawal_deletion", lambda **kwargs: scans.append(kwargs))
    started = []
    assert withdrawal.request_automatic_deletion_plan(capture_root=world["capture"], launch=started.append) == "refused"
    assert started == [] and _failure(world)["reason"] == "website_withdrawal_deletion_automatic_plan_path_unsafe"
    with pytest.raises(deletion.WithdrawalDeletionError, match="automatic_plan_path_unsafe"):
        deletion.record_automatic_plan(capture_root=world["capture"], **world["resolution"])
    assert scans == [] and _plan_path(world).is_symlink() and not (tmp_path / "nowhere.json").exists()
    assert _failure(world)["attempts"] == 2


def test_the_child_environment_carries_roots_and_paths_but_no_secret():
    environment = bounds.child_environment({
        "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT": "/q", "BLUEPRINT_TEAM_EVALUATION_REGISTRY_PATH": "/r",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE": "/etc/secret",
        "BLUEPRINT_PIPELINE_SYNC_TOKEN_PATH": "/etc/token", "PIPELINE_SYNC_WEBAPP_URL": "https://x",
        "BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL": "https://hook", "AWS_SECRET_ACCESS_KEY": "s"})
    assert environment["BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT"] == "/q"
    assert environment["BLUEPRINT_TEAM_EVALUATION_REGISTRY_PATH"] == "/r"
    assert set(environment) == {"PATH", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", bounds.CHILD_ENV,
                                bounds.LOCK_ENV, "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT",
                                "BLUEPRINT_TEAM_EVALUATION_REGISTRY_PATH"}


def test_the_bounded_child_records_its_plan_and_its_limits(tmp_path):
    import resource
    import subprocess

    world = _acknowledged(tmp_path)
    command = [sys.executable, "-m", "blueprint_pipeline.website_capture_withdrawal_deletion", "--capture-root",
               str(world["capture"]), "--record-automatic-plan", *drill._cli_root_arguments(world["resolution"]),
               "--uncovered-root", str(tmp_path / "absent")]
    environment = bounds.child_environment()
    environment[bounds.LOCK_ENV] = str(bounds.host_lock())
    completed = subprocess.run(command, env=environment, cwd="/", capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    record = json.loads(_plan_path(world).read_text())
    limits = record["child_limits"]
    assert limits["wall_clock_seconds"] == bounds.WALL_CLOCK_SECONDS
    assert limits["address_space_bytes"] is True and limits["cpu_seconds"] is True and limits["nice_19"] is True
    assert set(limits) >= {"io_class_idle", "oom_score_adj_1000"}  # best effort; recorded either way
    # This test process itself is not bounded.
    assert resource.getrlimit(resource.RLIMIT_AS)[0] != bounds.ADDRESS_SPACE_BYTES
