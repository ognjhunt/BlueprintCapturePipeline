"""A launch lease that never reached a run waits one dispatcher pass, not 4 h.

2026-10-31: capture-447ae4f4's launch was refused in the dispatcher before
``create_launch_output``. The request stayed in ``processing/`` for independent
reconciliation, scene progression counted it as an active writer, and the
reconciler gave it the 4 h unknown-run fallback before proving provider zero.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher_module
from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest, process_launch_queue
from blueprint_pipeline.task_evaluation_launch_dispatcher_ownership import DISPATCHER_LOCK_NAME
from blueprint_pipeline.task_evaluation_launch_reconciler import (
    PRE_RUN_STALE_SECONDS,
    reconcile_launches,
)
from blueprint_pipeline.task_evaluation_owner_dispatch_scope import SCOPE_ENV
from tests.test_task_evaluation_launch_reconciler_stale_profile import (
    LAUNCH_ID,
    PROFILE_ID,
    _profile,
    _write,
    _zero_guard,
)


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.delenv(SCOPE_ENV, raising=False)
    profile = _profile(reconciliation=True)
    _write(tmp_path / "profiles" / f"{PROFILE_ID}.json", profile)
    return {
        "queue": tmp_path / "queue",
        "state": tmp_path / "state",
        "profiles": tmp_path / "profiles",
        "guard": tmp_path / "guard.json",
    }


def _stage(roots: dict[str, Path], launch_id: str, *, age_seconds: float) -> Path:
    """A pending request staged ``age_seconds`` ago; claiming keeps that mtime."""
    profile = json.loads((roots["profiles"] / f"{PROFILE_ID}.json").read_text())
    request = {
        "launch_id": launch_id,
        "run_id": launch_id,
        "launch_profile_id": PROFILE_ID,
        "launch_profile_digest": profile["profile_digest"],
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    pending = roots["queue"] / "pending" / f"{launch_id}-pre-run.json"
    _write(pending, request)
    staged = datetime.now(timezone.utc).timestamp() - age_seconds
    os.utime(pending, (staged, staged))
    return roots["queue"] / "processing" / pending.name


def _reconcile(roots: dict[str, Path], *, live_count: int = 0) -> list[dict]:
    observed_at = datetime.now(timezone.utc)
    _write(roots["guard"], _zero_guard(observed_at=observed_at, live_count=live_count))
    report = reconcile_launches(
        queue_root=roots["queue"],
        state_root=roots["state"],
        profile_dir=roots["profiles"],
        guard_report_path=roots["guard"],
        now=observed_at,
        publish_progress=False,
    )
    return report["launches"]


def _refused_before_run_directory(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch, *, age_seconds: float
) -> Path:
    """Run the real queue; its dispatch fails before create_launch_output."""
    processing = _stage(roots, LAUNCH_ID, age_seconds=age_seconds)

    def refuse(**_kwargs):
        raise RuntimeError("publisher admission refused")

    monkeypatch.setattr(dispatcher_module, "dispatch_launch_request", refuse)
    queue_report = process_launch_queue(
        queue_root=roots["queue"], profile_dir=roots["profiles"], state_root=roots["state"]
    )
    assert queue_report["receipts"][0]["retain_processing_for_reconciliation"] is True
    assert processing.is_file()
    assert not (roots["state"] / LAUNCH_ID).exists()
    return processing


def test_unowned_pre_run_lease_is_contained_after_one_dispatcher_pass(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    processing = _refused_before_run_directory(roots, monkeypatch, age_seconds=20 * 60)

    rows = _reconcile(roots)

    assert rows[0]["status"] == "provider_zero_confirmed"
    assert not processing.exists()
    assert (roots["queue"] / "blocked" / processing.name).is_file()
    recovery = json.loads(
        (roots["state"] / LAUNCH_ID / "orphan_recovery_receipt.json").read_text()
    )
    assert recovery["provider_zero_confirmed"] is True
    assert recovery["allocator_invoked"] is False
    assert recovery["automatic_retry_performed"] is False
    assert recovery["hard_ttl_seconds"] == PRE_RUN_STALE_SECONDS


def test_fresh_unowned_pre_run_lease_still_waits_out_a_dispatcher_pass(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    processing = _refused_before_run_directory(roots, monkeypatch, age_seconds=60)

    rows = _reconcile(roots)

    assert rows[0]["status"] == "processing_within_ttl"
    assert processing.is_file()


@pytest.mark.parametrize("age_seconds", [20 * 60, 3 * 3600])
def test_live_dispatcher_keeps_its_claimed_batch_while_it_waits_for_serial_dispatch(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch, age_seconds: int
) -> None:
    """Codex on #3203: a whole batch is claimed first and dispatched one by one."""
    launch_ids = [f"{LAUNCH_ID}-a", f"{LAUNCH_ID}-b"]
    for launch_id in launch_ids:
        _stage(roots, launch_id, age_seconds=age_seconds)
    seen: dict[str, list] = {}

    def dispatch(**kwargs):
        request = json.loads(Path(kwargs["request_path"]).read_text(encoding="utf-8"))
        if not seen:
            # The first request is in dispatch; the second waits in processing
            # with no run directory and the mtime it was staged with.
            seen["rows"] = _reconcile(roots)
            seen["processing"] = sorted(
                path.name for path in (roots["queue"] / "processing").glob("*.json")
            )
        return {"status": "dry_run_completed", "launch_id": request["launch_id"]}

    monkeypatch.setattr(dispatcher_module, "dispatch_launch_request", dispatch)
    report = process_launch_queue(
        queue_root=roots["queue"],
        profile_dir=roots["profiles"],
        state_root=roots["state"],
        max_messages=10,
    )

    assert report["processed_count"] == 2
    assert [row["status"] for row in seen["rows"]] == ["processing_within_ttl"] * 2
    assert seen["processing"] == [f"{launch_id}-pre-run.json" for launch_id in launch_ids]
    assert not list((roots["queue"] / "blocked").glob("*.json"))
    assert not list(roots["state"].glob("*/orphan_recovery_receipt.json"))


def test_failed_provider_zero_pass_does_not_restore_the_full_fallback(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex on #3203: the unproven pass writes run_root/reconciliations itself."""
    processing = _refused_before_run_directory(roots, monkeypatch, age_seconds=20 * 60)
    run_root = roots["state"] / LAUNCH_ID

    first = _reconcile(roots, live_count=1)

    assert first[0]["status"] == "recovery_pending"
    assert processing.is_file()
    assert list((run_root / "reconciliations").glob("*.json"))
    assert not (run_root / "orphan_recovery_receipt.json").exists()

    second = _reconcile(roots)

    assert second[0]["status"] == "provider_zero_confirmed"
    assert (roots["queue"] / "blocked" / processing.name).is_file()
    recovery = json.loads((run_root / "orphan_recovery_receipt.json").read_text())
    assert recovery["allocator_invoked"] is False


@pytest.mark.parametrize(
    "evidence", ["launch_started", "launch_request", "empty_run_directory"]
)
def test_a_lease_the_dispatcher_got_into_keeps_its_existing_wait(
    roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch, evidence: str
) -> None:
    processing = _refused_before_run_directory(roots, monkeypatch, age_seconds=20 * 60)
    run_root = roots["state"] / LAUNCH_ID
    run_root.mkdir(parents=True)
    if evidence == "launch_started":
        started_at = datetime.now(timezone.utc) - timedelta(minutes=20)
        _write(
            run_root / "launch_started.json",
            {"started_at": started_at.isoformat(), "hard_ttl_seconds": 7200},
        )
    elif evidence == "launch_request":
        _write(run_root / "launch_request.json", json.loads(processing.read_text()))

    rows = _reconcile(roots)

    assert rows[0]["status"] == "processing_within_ttl"
    assert processing.is_file()
    assert not (run_root / "orphan_recovery_receipt.json").exists()


def test_no_shortcut_before_a_lock_taking_dispatcher_has_run(roots: dict[str, Path]) -> None:
    """After a deploy, a dispatcher started on older code takes no lock."""
    processing = _stage(roots, LAUNCH_ID, age_seconds=20 * 60)
    processing.parent.mkdir(parents=True)
    (roots["queue"] / "pending" / processing.name).replace(processing)

    rows = _reconcile(roots)

    assert rows[0]["status"] == "processing_within_ttl"
    assert processing.is_file()
    assert not (roots["queue"] / DISPATCHER_LOCK_NAME).exists()
