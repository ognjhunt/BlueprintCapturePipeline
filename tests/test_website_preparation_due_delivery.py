"""AUT-03: settled history never consumes a due callback's dispatch budget."""
from __future__ import annotations

import json
import importlib
import os
import subprocess
import sys
import time

import pytest

from blueprint_pipeline import website_preparation_delivery_index as index
from blueprint_pipeline import website_preparation_status as status
from blueprint_pipeline import website_task_context as context
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_website_preparation_terminal_status import website_capture, _claim, _finish, _acceptance  # noqa: F401, F811


@pytest.mark.slow
@pytest.mark.parametrize("corrupt_oldest", [False, True])
def test_new_callback_amid_ten_thousand_settled_ledgers_is_attempted_within_two_ticks(
    website_capture, monkeypatch, corrupt_oldest,  # noqa: F811
):
    source = website_capture
    ledger = _claim(source)
    _finish(source, ledger, status="failed_retryable", last_error="retained dependency delay")
    status.retain_preparation_wakeup(source.root)
    for number in range(10000):
        write_json(source.storage_root / f"history/scenes/site-{number:05d}/captures/walkthrough-{number:05d}/pipeline_job_ledger.json",
                   {"schema_version": "pipeline_job_ledger.v1", "revision": 1, "attempt_count": 1, "status": "completed"})
    now = time.time()
    if corrupt_oldest:
        broken = _synthetic_pending(source.storage_root, 0, now=now)
        (broken / status.DELIVERY_FILE).write_bytes(b'{"state":')
    calls = []
    def receive(**arguments):
        calls.append(arguments)
        return _acceptance(source, status.STATUS_V2)
    monkeypatch.setattr(context, "website_webapp_request", receive)
    if corrupt_oldest:
        first = status.reconcile_preparation_wakeups(source.storage_root, limit=1, now_epoch=now)
        assert first["attempted"] == 0 and not calls
    result = status.reconcile_preparation_wakeups(source.storage_root, limit=1, now_epoch=now + 1)
    assert result["attempted"] == result["delivered"] == 1
    assert len(calls) == 1 and calls[0]["payload"] == source.selected
    metrics = json.loads((source.storage_root / index.METRICS_FILE).read_text())
    assert metrics["repair_scanned"] == 1
    assert metrics["successful_delivery_lag_seconds"]
    if corrupt_oldest:
        assert (broken / status.DELIVERY_FILE).read_bytes() == b'{"state":'


@pytest.mark.parametrize("damage", ["missing_hint", "corrupt_hint", "corrupt_cursor"])
def test_index_is_rebuilt_from_canonical_delivery_after_restart(website_capture, monkeypatch, damage):  # noqa: F811
    source = website_capture
    _finish(source, _claim(source), status="failed_retryable", last_error="retained failure")
    status.retain_preparation_wakeup(source.root)
    hint = next((source.storage_root / index.DIRECTORY).glob("*.json"))
    if damage == "missing_hint":
        hint.unlink()
    elif damage == "corrupt_hint":
        hint.write_text("{broken")
    else:
        (source.storage_root / ".website_preparation_delivery_cursor.json").write_text("{broken")
    monkeypatch.setattr(context, "website_webapp_request", lambda **_: _acceptance(source, status.STATUS_V1))
    assert status.reconcile_preparation_wakeups(source.storage_root, limit=1)["delivered"] == 1
    assert json.loads((source.root / status.DELIVERY_FILE).read_text())["state"] == "delivered"
    assert status.reconcile_preparation_wakeups(source.storage_root, limit=1)["attempted"] == 0


def _synthetic_pending(storage, number, *, now):
    request = f"req-{number:03d}"
    root = storage / f"bucket/scenes/site-{request}/captures/walkthrough-{request}"
    selected = {"request_id": request, "scene_id": f"site-{request}", "capture_id": f"walkthrough-{request}",
        "completion_marker_generation": "1", "producer_delivery_key": "sha256:" + "a" * 64,
        "source_payload_sha256": "b" * 64, "task_context_digest": "sha256:" + "c" * 64}
    ledger = {"schema_version": "pipeline_job_ledger.v1", "status": "failed_retryable", "revision": 1,
              "attempt_count": 1, "source_payload_sha256": selected["source_payload_sha256"]}
    delivery = {"schema_version": "website_preparation_status_delivery.v1", "state": "pending",
        "selectors": selected, "revision": 1, "delivery_attempt_count": 0,
        "pending_since_epoch": now - 120, "next_attempt_at_epoch": 0.0}
    write_json(root / "pipeline_job_ledger.json", ledger)
    (root / ".pipeline_job_ledger.json.lock").touch()
    write_json(root / status.DELIVERY_FILE, delivery)
    index.update(root, delivery)
    return root


def _synthetic_status(*, capture_root, selectors, schema_version=status.STATUS_V1):
    ledger = json.loads((capture_root / "pipeline_job_ledger.json").read_text())
    value = {"schema_version": schema_version, **selectors, "revision": ledger["revision"],
             "attempt_count": ledger["attempt_count"], "state": "failed_retryable"}
    value["status_digest"] = cross_runtime_canonical_digest(value)
    return value


def _synthetic_ack(root):
    delivery = json.loads((root / status.DELIVERY_FILE).read_text())
    observed = _synthetic_status(capture_root=root, selectors=delivery["selectors"])
    return {"schema_version": "website_preparation_status_acceptance.v1", **delivery["selectors"],
            "accepted": True, "state": observed["state"], "status_digest": observed["status_digest"],
            "native_execution_complete": False,
            "correlation_id": "bp-prep-" + observed["status_digest"].removeprefix("sha256:")[:16]}


def test_hundred_due_callbacks_progress_fairly_despite_failures_and_new_revision(tmp_path, monkeypatch):
    now = 100000.0
    roots = [_synthetic_pending(tmp_path, number, now=now) for number in range(100)]
    by_capture = {root.name: root for root in roots}
    calls = []
    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: None)
    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)
    def receive(*, capture_id, **_):
        root = by_capture[capture_id]
        calls.append(capture_id)
        ack = _synthetic_ack(root)
        if root == roots[0] and calls.count(capture_id) == 1:
            # A new revision races the old acceptance. It stays pending.
            ledger = json.loads((root / "pipeline_job_ledger.json").read_text())
            write_json(root / "pipeline_job_ledger.json", {**ledger, "revision": 2})
            delivery = json.loads((root / status.DELIVERY_FILE).read_text())
            delivery.update(revision=2, pending_since_epoch=now, next_attempt_at_epoch=0.0)
            write_json(root / status.DELIVERY_FILE, delivery)
            index.update(root, delivery)
        elif int(capture_id.rsplit("-", 1)[1]) % 5 == 0:
            raise TimeoutError("private response must not enter delivery")
        return ack
    monkeypatch.setattr(context, "website_webapp_request", receive)
    first = status.reconcile_preparation_wakeups(tmp_path, limit=50, now_epoch=now)
    second = status.reconcile_preparation_wakeups(tmp_path, limit=50, now_epoch=now)
    assert first["attempted"] == second["attempted"] == 50
    assert set(calls) == set(by_capture)
    assert len(calls) == 100
    raced = json.loads((roots[0] / status.DELIVERY_FILE).read_text())
    assert raced["state"] == "pending" and raced["revision"] == 2
    failed = json.loads((roots[5] / status.DELIVERY_FILE).read_text())
    assert failed["state"] == "pending" and failed["next_attempt_at_epoch"] > now
    assert "private response" not in json.dumps(failed)
    metrics = json.loads((tmp_path / index.METRICS_FILE).read_text())
    assert metrics["oldest_due_age_seconds"] >= 120
    # Newer revision is eligible immediately; failed sites keep their backoff.
    assert status.reconcile_preparation_wakeups(tmp_path, limit=50, now_epoch=now)["attempted"] == 1


def test_index_cannot_select_an_arbitrary_capture_path(tmp_path):
    directory = tmp_path / index.DIRECTORY
    directory.mkdir()
    write_json(directory / ("a" * 64 + ".json"), {"schema_version": index.SCHEMA_VERSION,
               "capture_relative_path": "../../outside/scenes/site-x/captures/walkthrough-x"})
    roots, metrics = index.due(tmp_path, now=10, limit=1)
    assert roots == [] and metrics["invalid_hint_count"] == 1


def test_source_lookup_outage_backs_off_without_monopolizing_healthy_callback(tmp_path, monkeypatch):
    now = 100000.0
    broken, healthy = [_synthetic_pending(tmp_path, number, now=now) for number in range(2)]
    calls = []
    def retain(root):
        if root == broken:
            raise TimeoutError("private current-source lookup failure")
    monkeypatch.setattr(status, "retain_preparation_wakeup", retain)
    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)
    def receive(*, capture_id, **_):
        calls.append(capture_id)
        assert capture_id == healthy.name
        return _synthetic_ack(healthy)
    monkeypatch.setattr(context, "website_webapp_request", receive)
    for tick in range(3):
        status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now + tick)
    assert calls == [healthy.name]
    failed = json.loads((broken / status.DELIVERY_FILE).read_text())
    assert failed["state"] == "pending" and failed["next_attempt_at_epoch"] > now + 2
    assert failed["delivery_attempt_count"] == 0 and failed["delivery_failure_count"] == 1
    assert failed["last_error_type"] == "TimeoutError"
    assert "private current-source" not in json.dumps(failed)


def test_already_due_retry_is_not_starved_by_five_new_pending_revisions(tmp_path, monkeypatch):
    now = 100000.0
    retry = _synthetic_pending(tmp_path, 0, now=now)
    delivery = json.loads((retry / status.DELIVERY_FILE).read_text())
    delivery.update(delivery_attempt_count=1, next_attempt_at_epoch=now - 1)
    write_json(retry / status.DELIVERY_FILE, delivery)
    index.update(retry, delivery)
    roots = {retry.name: retry}
    calls = []
    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: None)
    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)
    def receive(*, capture_id, **_):
        calls.append(capture_id)
        return _synthetic_ack(roots[capture_id])
    monkeypatch.setattr(context, "website_webapp_request", receive)
    for tick in range(5):
        fresh = _synthetic_pending(tmp_path, tick + 1, now=now + tick)
        fresh_delivery = json.loads((fresh / status.DELIVERY_FILE).read_text())
        fresh_delivery["pending_since_epoch"] = now + tick
        write_json(fresh / status.DELIVERY_FILE, fresh_delivery)
        index.update(fresh, fresh_delivery)
        roots[fresh.name] = fresh
        status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now + tick)
    assert calls[0] == retry.name
    assert len(set(calls)) == 5
    assert json.loads((retry / status.DELIVERY_FILE).read_text())["state"] == "delivered"


@pytest.mark.parametrize("corrupt_bytes", [b'{"state":', b'[]', b'{}', b'{"state":"pending"}'])
def test_unreadable_oldest_delivery_cannot_starve_healthy_peer_after_restart(
    tmp_path, monkeypatch, corrupt_bytes,
):
    now = 100000.0
    broken, healthy = [_synthetic_pending(tmp_path, number, now=now) for number in range(2)]
    corrupt_path = broken / status.DELIVERY_FILE
    corrupt_path.write_bytes(corrupt_bytes)
    ledger_bytes = (broken / "pipeline_job_ledger.json").read_bytes()
    calls = []
    # The canonical repair path also refuses this record; it cannot silently
    # replace unreadable delivery history with a new pending attempt.
    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: status._read(root / status.DELIVERY_FILE))
    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)
    def receive(*, capture_id, **_):
        calls.append(capture_id)
        assert capture_id == healthy.name
        return _synthetic_ack(healthy)
    monkeypatch.setattr(context, "website_webapp_request", receive)
    ticks = []
    for tick in range(3):
        # No process-local rotation/backoff state may be required for progress.
        importlib.reload(index)
        ticks.append(status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now + tick))
    assert calls == [healthy.name], ticks
    assert ticks[1]["delivered"] == 1
    assert corrupt_path.read_bytes() == corrupt_bytes
    assert (broken / "pipeline_job_ledger.json").read_bytes() == ledger_bytes
    assert json.loads((healthy / status.DELIVERY_FILE).read_text())["state"] == "delivered"


def test_unavailable_hint_backoff_survives_same_revision_repair_and_is_bounded(tmp_path):
    now = 100000.0
    root = _synthetic_pending(tmp_path, 0, now=now)
    canonical = (root / status.DELIVERY_FILE).read_bytes()
    delivery = json.loads(canonical)
    for failure in range(1, 9):
        selected = index.due(tmp_path, now=now, limit=1)[0][0]
        index.defer_unavailable(selected, now=now)
        # A late duplicate failure and historical repair cannot shorten or
        # repeatedly extend the deferral for the already-selected hint.
        index.defer_unavailable(selected, now=now + 1)
        index.update(root, delivery)
        importlib.reload(index)
        assert index.due(tmp_path, now=now + 1, limit=1)[0] == []
        hint = json.loads(next((tmp_path / index.DIRECTORY).glob("*.json")).read_text())
        interval = hint["unavailable_next_attempt_at_epoch"] - now
        assert interval == min(1800, 60 * 2 ** min(failure - 1, 5))
        now += interval
    assert (root / status.DELIVERY_FILE).read_bytes() == canonical
    assert len(list((tmp_path / index.DIRECTORY).glob("*.json"))) == 1


@pytest.mark.parametrize("damage", [
    "missing_anchor", "far_deadline", "huge_deadline", "wrong_interval", "future_anchor",
    "invalid_count", "huge_count", "nonfinite_anchor", "negative_deadline", "typed_deadline",
])
def test_canonical_repair_discards_invalid_hint_backoff_without_mutating_delivery(tmp_path, damage):
    now = time.time()
    root = _synthetic_pending(tmp_path, 0, now=now)
    canonical = (root / status.DELIVERY_FILE).read_bytes()
    index.defer_unavailable(index.due(tmp_path, now=now, limit=1)[0][0], now=now)
    hint_path = next((tmp_path / index.DIRECTORY).glob("*.json"))
    hint = json.loads(hint_path.read_text())
    if damage == "missing_anchor":
        hint.pop("unavailable_last_attempt_at_epoch")
    elif damage == "far_deadline":
        hint["unavailable_next_attempt_at_epoch"] = now + 10**12
    elif damage == "huge_deadline":
        hint["unavailable_next_attempt_at_epoch"] = 10**500
    elif damage == "wrong_interval":
        hint["unavailable_next_attempt_at_epoch"] = now + 1800
    elif damage == "future_anchor":
        hint.update(unavailable_last_attempt_at_epoch=now + 10**12,
                    unavailable_next_attempt_at_epoch=now + 10**12 + 60)
    elif damage == "invalid_count":
        hint["unavailable_failure_count"] = True
    elif damage == "huge_count":
        hint["unavailable_failure_count"] = 10**500
    elif damage == "nonfinite_anchor":
        hint["unavailable_last_attempt_at_epoch"] = float("nan")
    elif damage == "negative_deadline":
        hint["unavailable_next_attempt_at_epoch"] = -1
    else:
        hint["unavailable_next_attempt_at_epoch"] = str(now + 60)
    write_json(hint_path, hint)
    index.update(root, json.loads(canonical))
    rebuilt = json.loads(hint_path.read_text())
    assert not any(key.startswith("unavailable_") for key in rebuilt)
    selected, _ = index.due(tmp_path, now=time.time(), limit=1)
    assert len(selected) == 1 and selected[0].capture_root == root
    assert (root / status.DELIVERY_FILE).read_bytes() == canonical


@pytest.mark.parametrize("field", [
    "next_attempt_at_epoch", "pending_since_epoch", "unavailable_next_attempt_at_epoch",
    "unavailable_last_attempt_at_epoch", "unavailable_failure_count",
])
def test_oversized_hint_numbers_cannot_abort_healthy_callback_dispatch(tmp_path, monkeypatch, field):
    now = 100000.0
    roots = [_synthetic_pending(tmp_path, number, now=now) for number in range(2)]
    corrupted = next(path for path in (tmp_path / index.DIRECTORY).glob("*.json")
                     if json.loads(path.read_text())["capture_relative_path"].endswith(roots[0].name))
    hint = json.loads(corrupted.read_text())
    write_json(corrupted, {**hint, field: 10**500})
    by_capture = {root.name: root for root in roots}
    calls = []
    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: None)
    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)

    def receive(*, capture_id, **_):
        calls.append(capture_id)
        return _synthetic_ack(by_capture[capture_id])

    monkeypatch.setattr(context, "website_webapp_request", receive)
    for tick in range(2):
        result = status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now + tick)
        assert result["attempted"] == result["delivered"] == 1
    assert set(calls) == set(by_capture) and len(calls) == 2
    metrics = json.loads((tmp_path / index.METRICS_FILE).read_text())
    assert not metrics.get("index_unavailable", False)


def test_missing_canonical_delivery_cannot_monopolize_a_healthy_peer(tmp_path, monkeypatch):
    now = 100000.0
    missing, healthy = [_synthetic_pending(tmp_path, number, now=now) for number in range(2)]
    path = missing / status.DELIVERY_FILE
    path.unlink()
    ledger_bytes = (missing / "pipeline_job_ledger.json").read_bytes()
    calls = []
    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: None)
    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)

    def receive(*, capture_id, **_):
        calls.append(capture_id)
        return _synthetic_ack(healthy)

    monkeypatch.setattr(context, "website_webapp_request", receive)
    first = status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now)
    importlib.reload(index)
    second = status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now + 1)
    assert first["unavailable"] >= 1 and first["attempted"] == 0
    assert second["delivered"] == 1 and calls == [healthy.name]
    assert not path.exists()
    assert (missing / "pipeline_job_ledger.json").read_bytes() == ledger_bytes


def test_persistent_corruption_does_not_starve_new_healthy_peers_when_backoff_expires(tmp_path, monkeypatch):
    baseline = 100000.0
    broken = _synthetic_pending(tmp_path, 0, now=baseline)
    corrupt_path = broken / status.DELIVERY_FILE
    corrupt_path.write_bytes(b'{"state":')
    roots, calls = {}, []
    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: status._read(root / status.DELIVERY_FILE))
    monkeypatch.setattr(status, "read_preparation_status", _synthetic_status)
    def receive(*, capture_id, **_):
        calls.append(capture_id)
        return _synthetic_ack(roots[capture_id])
    monkeypatch.setattr(context, "website_webapp_request", receive)
    for cycle in range(4):
        # Every cycle starts after even the maximum deferral could expire.
        now = baseline + cycle * 3600
        healthy = _synthetic_pending(tmp_path, cycle + 1, now=now)
        roots[healthy.name] = healthy
        for tick in range(2):
            importlib.reload(index)
            result = status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now + tick)
            assert result["attempted"] <= 1
        assert calls == list(roots)
        assert corrupt_path.read_bytes() == b'{"state":'


def test_late_unavailable_hint_cannot_delay_new_revision_and_new_revision_resets_deferral(tmp_path):
    now = 100000.0
    root = _synthetic_pending(tmp_path, 0, now=now)
    selected = index.due(tmp_path, now=now, limit=1)[0][0]
    path = root / status.DELIVERY_FILE
    delivery = json.loads(path.read_text())
    delivery.update(revision=2, pending_since_epoch=now)
    write_json(path, delivery)
    index.update(root, delivery)
    index.defer_unavailable(selected, now=now)
    pending = index.due(tmp_path, now=now, limit=1)[0]
    assert len(pending) == 1 and pending[0].hint["revision"] == 2
    index.defer_unavailable(pending[0], now=now)
    assert index.due(tmp_path, now=now, limit=1)[0] == []
    delivery.update(revision=3)
    write_json(path, delivery)
    index.update(root, delivery)
    pending = index.due(tmp_path, now=now, limit=1)[0]
    assert len(pending) == 1 and pending[0].hint["revision"] == 3
    assert "unavailable_next_attempt_at_epoch" not in pending[0].hint


@pytest.mark.slow
def test_unreadable_delivery_deferral_is_visible_to_a_fresh_dispatcher_process(tmp_path, monkeypatch):
    now = 100000.0
    broken, healthy = [_synthetic_pending(tmp_path, number, now=now) for number in range(2)]
    corrupt_path = broken / status.DELIVERY_FILE
    corrupt_path.write_bytes(b'{"state":')
    monkeypatch.setattr(status, "retain_preparation_wakeup", lambda root: None)
    assert status.reconcile_preparation_wakeups(tmp_path, limit=1, now_epoch=now)["attempted"] == 0
    result = subprocess.run([sys.executable, "-c",
        "import json,sys; from pathlib import Path; "
        "from blueprint_pipeline.website_preparation_delivery_index import due; "
        "rows,_=due(Path(sys.argv[1]),now=float(sys.argv[2]),limit=1); "
        "print(json.dumps([str(row.capture_root) for row in rows]))",
        str(tmp_path), str(now + 1)], check=True, capture_output=True, text=True, timeout=20,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONDONTWRITEBYTECODE": "1"})
    assert json.loads(result.stdout) == [str(healthy)]
    assert corrupt_path.read_bytes() == b'{"state":'


@pytest.mark.parametrize("damage", ["schema", "selectors", "counter", "timestamp"])
def test_canonical_repair_preserves_malformed_delivery_without_fabricating_history(
    website_capture, damage,  # noqa: F811
):
    source = website_capture
    _finish(source, _claim(source), status="failed_retryable", last_error="retained failure")
    status.retain_preparation_wakeup(source.root)
    path = source.root / status.DELIVERY_FILE
    delivery = json.loads(path.read_text())
    if damage == "schema":
        delivery["schema_version"] = "unknown"
    elif damage == "selectors":
        delivery["selectors"] = {}
    elif damage == "counter":
        delivery["delivery_failure_count"] = "unknown"
    else:
        delivery["next_attempt_at_epoch"] = float("nan")
    write_json(path, delivery)
    before = path.read_bytes()
    with pytest.raises(status.PreparationStatusUnavailable):
        status.retain_preparation_wakeup(source.root)
    assert path.read_bytes() == before
