"""AUT-03: settled history never consumes a due callback's dispatch budget."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import website_preparation_delivery_index as index
from blueprint_pipeline import website_preparation_status as status
from blueprint_pipeline import website_task_context as context
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from tests.test_website_preparation_terminal_status import website_capture, _claim, _finish, _acceptance  # noqa: F401, F811


@pytest.mark.slow
def test_new_callback_amid_ten_thousand_settled_ledgers_is_attempted_first_tick(website_capture, monkeypatch):  # noqa: F811
    source = website_capture
    ledger = _claim(source)
    _finish(source, ledger, status="failed_retryable", last_error="retained dependency delay")
    status.retain_preparation_wakeup(source.root)
    for number in range(10000):
        write_json(source.storage_root / f"history/scenes/site-{number:05d}/captures/walkthrough-{number:05d}/pipeline_job_ledger.json",
                   {"schema_version": "pipeline_job_ledger.v1", "revision": 1, "attempt_count": 1, "status": "completed"})
    calls = []
    def receive(**arguments):
        calls.append(arguments)
        return _acceptance(source, status.STATUS_V2)
    monkeypatch.setattr(context, "website_webapp_request", receive)
    result = status.reconcile_preparation_wakeups(source.storage_root, limit=1)
    assert result["attempted"] == result["delivered"] == 1
    assert len(calls) == 1 and calls[0]["payload"] == source.selected
    metrics = json.loads((source.storage_root / index.METRICS_FILE).read_text())
    assert metrics["repair_scanned"] == 1
    assert metrics["successful_delivery_lag_seconds"]


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
