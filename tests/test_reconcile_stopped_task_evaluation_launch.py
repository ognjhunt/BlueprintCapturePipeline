import importlib.util
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.test_task_evaluation_launch_reconciler_stale_profile import LAUNCH_ID, _profile, _zero_guard

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("stopped_launch", SCRIPTS / "reconcile_stopped_task_evaluation_launch.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def run(tmp_path, monkeypatch):
    now = datetime(2026, 9, 15, 7, 20, tzinfo=timezone.utc)
    root = tmp_path / "state" / LAUNCH_ID
    root.mkdir(parents=True)
    profile = _profile(reconciliation=True)
    request = {"launch_id": LAUNCH_ID, "request_digest": "sha256:" + "a" * 64,
               "launch_profile_digest": profile["profile_digest"]}
    started = {"launch_id": LAUNCH_ID, "request_digest": request["request_digest"],
               "process_id": 999999999, "started_at": (now - timedelta(minutes=10)).isoformat(),
               "hard_ttl_seconds": 27000}
    for name, value in (("launch_profile.json", profile), ("launch_started.json", started)):
        (root / name).write_text(json.dumps(value))
    claim = tmp_path / "queue" / "processing" / (LAUNCH_ID + "-claim.json")
    claim.parent.mkdir(parents=True)
    claim.write_text(json.dumps(request))
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps(_zero_guard(observed_at=now)))
    unit = {"LoadState": "loaded", "ActiveState": "failed", "MainPID": "0", "ControlGroup": "",
            "ExecMainExitTimestamp": "Tue 2026-09-15 07:19:00 UTC"}
    held = []

    @contextmanager
    def locks(_):
        held.append(True)
        try:
            yield
        finally:
            held.pop()

    def observe():
        assert held
        return unit

    monkeypatch.setattr(recovery, "_holding_paid_launch_locks", locks)
    monkeypatch.setattr(recovery, "_observe_dispatcher", observe)
    monkeypatch.setattr(recovery, "_process_alive", lambda _: False)
    args = dict(queue_root=claim.parents[1], state_root=root.parent, guard_report_path=guard,
                launch_id=LAUNCH_ID, now=now)
    return args, root, claim, guard, unit


def test_closes_dead_claim_before_ttl_without_launch_or_refund(run):
    args, root, claim, _, _ = run
    original = claim.read_bytes()
    result = recovery.reconcile_stopped_launch(**args)
    assert result["status"] == "provider_zero_confirmed"
    assert result["historical_spend_settled"] is False
    assert result["allocator_invoked"] is False
    assert result["lease_age_seconds"] < result["hard_ttl_seconds"]
    assert not claim.exists()
    blocked = args["queue_root"] / "blocked" / claim.name
    assert blocked.read_bytes() == original
    # A crash after receipt creation but before the queue move is recoverable.
    blocked.replace(claim)
    again = recovery.reconcile_stopped_launch(**{**args, "now": args["now"] + timedelta(seconds=1)})
    assert again == result


@pytest.mark.parametrize("field,value", [("ActiveState", "active"), ("MainPID", "123"),
    ("ControlGroup", "/still-has-children"), ("LoadState", "not-found")])
def test_refuses_unproven_drain(run, field, value):
    args, root, claim, _, unit = run
    unit[field] = value
    with pytest.raises(recovery.TaskEvaluationLaunchError, match="not_drained"):
        recovery.reconcile_stopped_launch(**args)
    assert claim.exists() and not (root / "orphan_recovery_receipt.json").exists()


@pytest.mark.parametrize("live_count,age", [(1, 0), (0, 120), (0, 600)])
def test_refuses_live_stale_or_pre_stop_inventory(run, live_count, age):
    args, root, claim, guard, _ = run
    guard.write_text(json.dumps(_zero_guard(observed_at=args["now"] - timedelta(seconds=age), live_count=live_count)))
    with pytest.raises(recovery.TaskEvaluationLaunchError, match="provider_zero_required"):
        recovery.reconcile_stopped_launch(**args)
    assert claim.exists() and not (root / "orphan_recovery_receipt.json").exists()


def test_refuses_live_original_dispatcher_pid(run, monkeypatch):
    args, _, claim, _, _ = run
    monkeypatch.setattr(recovery, "_process_alive", lambda _: True)
    with pytest.raises(recovery.TaskEvaluationLaunchError, match="writer_or_binding"):
        recovery.reconcile_stopped_launch(**args)
    assert claim.exists()


def test_retains_guard_snapshot_and_discards_only_temporary_secret_copies(run):
    args, root, _, guard, _ = run
    source_guard = json.loads(guard.read_text())
    secrets = root / "allocator/scene-configuration-job/runtime-secrets"
    secrets.mkdir(parents=True)
    (secrets / "OPENAI_CONTENT_AGENTS_API_KEY_FILE").write_text("test-only-copy")
    result = recovery.reconcile_stopped_launch(**args)
    guard.write_text("{}")
    retained = Path(result["guard_report_path"])
    assert json.loads(retained.read_text()) == source_guard
    assert "sha256:" + hashlib.sha256(retained.read_bytes()).hexdigest() == result["guard_report_sha256"]
    assert not secrets.exists()
    assert result["temporary_runtime_secrets_removed"] is True


def test_refuses_symlinked_secret_directory(run, tmp_path):
    args, root, claim, _, _ = run
    outside = tmp_path / "canonical-secrets"
    outside.mkdir()
    key = outside / "key"
    key.write_text("preserve")
    secrets = root / "allocator/scene-configuration-job/runtime-secrets"
    secrets.parent.mkdir(parents=True)
    secrets.symlink_to(outside, target_is_directory=True)
    with pytest.raises(recovery.TaskEvaluationLaunchError, match="secret_root_unsafe"):
        recovery.reconcile_stopped_launch(**args)
    assert key.read_text() == "preserve" and claim.exists()
