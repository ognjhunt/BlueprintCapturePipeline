"""An owner-approved retirement of a parked website preparation delivery (2026-11-06).

Seven owner test captures from 10-28/29 stayed parked for over a week: their
status read can never succeed again. Retiring one is general: any delivery
parked for a day can be retired, with the owner's approval on its record, and
it is then never due, retried or counted. A newer producer revision still
starts a new delivery. Nothing else in the capture changes.
"""
from __future__ import annotations

import json
import time

import pytest

from blueprint_pipeline import website_preparation_delivery_index as index
from blueprint_pipeline import website_preparation_delivery_retirement as retirement
from blueprint_pipeline import website_preparation_status as status
from blueprint_pipeline.common import write_json
from tests.test_website_preparation_due_delivery import _synthetic_pending
from tests.test_website_preparation_delivery_parking import _hint, _record, _set
from tests.test_website_preparation_terminal_status import website_capture, _claim, _finish  # noqa: F401, F811

DAY = 24 * 3600.0
NOW = 5_000_000.0
EVIDENCE = "Owner, 2026-11-06 session: \"yes retire the 7\""


def _parked(storage, number=0, *, parked_for=8 * DAY):
    root = _synthetic_pending(storage, number, now=NOW)
    _set(root, pending_since_epoch=NOW - 9 * DAY, parked_reason=status.PARKED_REASON,
         parked_at_epoch=NOW - parked_for, status_unavailable_since_epoch=NOW - 9 * DAY,
         last_unavailable_reason="task_context_changed", next_attempt_at_epoch=NOW + 3600)
    return root


def _relative(storage, root):
    return root.relative_to(storage).as_posix()


def _retire(storage, root, **kwargs):
    return retirement.retire_parked_delivery(storage_root=storage, capture_relative_path=_relative(storage, root),
                                             approval_evidence=EVIDENCE, now=NOW, **kwargs)


def test_a_plan_names_what_it_would_retire_and_writes_nothing(tmp_path):
    root = _parked(tmp_path)
    before = (root / status.DELIVERY_FILE).read_bytes()
    plan = _retire(tmp_path, root)
    assert plan["outcome"] == "planned" and plan["apply"] is False
    assert plan["delivery"]["parked_reason"] == status.PARKED_REASON
    assert plan["would_set"] == {"state": "retired", "retired_at_epoch": NOW, "retired_reason": status.RETIRED_REASON,
                                 "retirement_approval_evidence": EVIDENCE}
    assert (root / status.DELIVERY_FILE).read_bytes() == before and _hint(tmp_path, root) is not None


def test_apply_retires_a_parked_delivery_keeps_its_fields_and_drops_it_from_every_count(tmp_path):
    root = _parked(tmp_path)
    live = _synthetic_pending(tmp_path, 1, now=NOW)
    before, ledger = _record(root), (root / "pipeline_job_ledger.json").read_bytes()
    assert index.due(tmp_path, now=NOW, limit=5)[1]["parked_count"] == 1
    done = _retire(tmp_path, root, apply=True, door_caller="operator-1",
                   door_request_id="20261106T170000Z-retire-parked-delivery-0123abcd")
    assert done["outcome"] == "retired" and done["record_sha256"] != done["prior_record_sha256"]
    record = _record(root)
    assert {key: record[key] for key in before if key != "state"} == {key: before[key] for key in before
                                                                      if key != "state"}
    assert (record["state"], record["retired_at_epoch"], record["retired_reason"]) == (
        "retired", NOW, "owner_retired_parked_delivery")
    assert record["retirement_approval_evidence"] == EVIDENCE
    assert record["retired_by_door_caller"] == "operator-1"
    assert record["retirement_door_request_id"] == "20261106T170000Z-retire-parked-delivery-0123abcd"
    assert (root / "pipeline_job_ledger.json").read_bytes() == ledger
    assert _hint(tmp_path, root) is None
    rows, metrics = index.due(tmp_path, now=NOW, limit=5)
    assert [row.capture_root for row in rows] == [live]
    assert (metrics["parked_count"], metrics["live_count"]) == (0, 1)


def test_apply_is_idempotent_and_clears_a_hint_an_interrupted_apply_left(tmp_path):
    root = _parked(tmp_path)
    _retire(tmp_path, root, apply=True)
    record = (root / status.DELIVERY_FILE).read_bytes()
    # An apply interrupted between the record and the index left the hint behind.
    stale = json.loads(json.dumps(_record(root)))
    stale["state"] = "pending"
    index.update(root, stale)
    assert _hint(tmp_path, root) is not None
    again = _retire(tmp_path, root, apply=True)
    assert again["outcome"] == "already_retired"
    assert (root / status.DELIVERY_FILE).read_bytes() == record and _hint(tmp_path, root) is None
    assert _retire(tmp_path, root)["outcome"] == "already_retired"


def test_the_controller_never_retries_a_retired_delivery(tmp_path, monkeypatch):
    root = _parked(tmp_path)
    _retire(tmp_path, root, apply=True)
    record = (root / status.DELIVERY_FILE).read_bytes()
    monkeypatch.setattr(status, "read_preparation_status", lambda **_: pytest.fail("a retired delivery is read"))
    counts = status.reconcile_preparation_wakeups(tmp_path, limit=5, now_epoch=NOW + 30 * DAY)
    assert counts == {"attempted": 0, "delivered": 0, "unavailable": counts["unavailable"]}
    assert (root / status.DELIVERY_FILE).read_bytes() == record and _hint(tmp_path, root) is None


@pytest.mark.parametrize(("change", "code"), [
    ({"parked_reason": None, "parked_at_epoch": None}, "parked_delivery_not_parked"),
    ({"parked_at_epoch": NOW - 3600}, "parked_delivery_parked_under_a_day"),
    ({"parked_reason": "something_else"}, "parked_delivery_not_parked"),
    ({"state": "delivered"}, "parked_delivery_not_pending"),
    ({"state": "retired"}, "parked_delivery_record_invalid"),  # a retired state without its fields
])
def test_only_a_delivery_parked_for_a_day_can_be_retired(tmp_path, change, code):
    root = _parked(tmp_path)
    record = {key: value for key, value in {**_record(root), **change}.items() if value is not None}
    write_json(root / status.DELIVERY_FILE, record)
    before = (root / status.DELIVERY_FILE).read_bytes()
    with pytest.raises(retirement.DeliveryRetirementRefused) as refused:
        _retire(tmp_path, root, apply=True)
    assert refused.value.code == code
    assert (root / status.DELIVERY_FILE).read_bytes() == before


def test_a_live_delivery_is_refused_even_in_a_plan(tmp_path):
    root = _synthetic_pending(tmp_path, 0, now=NOW)
    with pytest.raises(retirement.DeliveryRetirementRefused) as refused:
        _retire(tmp_path, root)
    assert refused.value.code == "parked_delivery_not_parked"


@pytest.mark.parametrize("relative", [
    "/abs/scenes/site-a/captures/walkthrough-a", "bucket/scenes/site-a/captures/../captures/walkthrough-a",
    "bucket/scenes/site-a/captures/other-a", "bucket/scenes/scene-a/captures/walkthrough-a",
    "bucket/scenes/site-a/captures/walkthrough-a/raw", "bucket/scenes/site-a", "", 7,
])
def test_only_a_capture_path_the_index_keys_is_accepted(tmp_path, relative):
    with pytest.raises(retirement.DeliveryRetirementRefused) as refused:
        retirement.retire_parked_delivery(storage_root=tmp_path, capture_relative_path=relative,
                                          approval_evidence=EVIDENCE, now=NOW)
    assert refused.value.code == "parked_delivery_capture_invalid"


def test_a_missing_capture_or_ledger_is_refused_and_nothing_is_created(tmp_path):
    relative = "bucket/scenes/site-req-009/captures/walkthrough-req-009"
    with pytest.raises(retirement.DeliveryRetirementRefused) as refused:
        retirement.retire_parked_delivery(storage_root=tmp_path, capture_relative_path=relative,
                                          approval_evidence=EVIDENCE, apply=True, now=NOW)
    assert refused.value.code == "parked_delivery_capture_absent" and not (tmp_path / "bucket").exists()


@pytest.mark.parametrize("evidence", ["", " padded ", "line\nbreak", "x" * 501, None])
def test_the_approval_evidence_is_required_and_bounded(tmp_path, evidence):
    root = _parked(tmp_path)
    with pytest.raises(retirement.DeliveryRetirementRefused) as refused:
        retirement.retire_parked_delivery(storage_root=tmp_path, capture_relative_path=_relative(tmp_path, root),
                                          approval_evidence=evidence, apply=True, now=NOW)
    assert refused.value.code == "parked_delivery_approval_evidence_invalid"


def test_a_newer_producer_revision_starts_a_new_delivery(website_capture):  # noqa: F811
    source = website_capture
    _finish(source, _claim(source), status="failed_retryable", last_error="retained failure")
    status.retain_preparation_wakeup(source.root)
    now = time.time()
    _set(source.root, parked_reason=status.PARKED_REASON, parked_at_epoch=now - 2 * DAY,
         status_unavailable_since_epoch=now - 3 * DAY)
    relative = source.root.relative_to(source.storage_root).as_posix()
    retirement.retire_parked_delivery(storage_root=source.storage_root, capture_relative_path=relative,
                                      approval_evidence=EVIDENCE, apply=True, now=now)
    retired = _record(source.root)
    # Retaining the same revision's wake-up again keeps it retired, and its hint gone.
    status.retain_preparation_wakeup(source.root)
    assert _record(source.root) == retired and _hint(source.storage_root, source.root) is None
    # A newer attempt commits a newer revision: a fresh pending delivery, live and due.
    _finish(source, _claim(source), status="failed_retryable", last_error="next attempt")
    status.retain_preparation_wakeup(source.root)
    record = _record(source.root)
    assert record["revision"] > retired["revision"] and record["state"] == "pending"
    assert not any(key.startswith("retire") for key in record)
    rows, metrics = index.due(source.storage_root, now=now + 1, limit=5)
    assert [row.capture_root for row in rows] == [source.root]
    assert (metrics["parked_count"], metrics["live_count"]) == (0, 1)


def test_the_command_prints_a_typed_result_and_exits_by_outcome(tmp_path, monkeypatch, capsys):
    root = _parked(tmp_path)
    monkeypatch.setenv(index.STORAGE_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(retirement.time, "time", lambda: NOW)
    argv = ["--capture", _relative(tmp_path, root), "--approval-evidence", EVIDENCE]
    assert retirement.main(argv) == 0 and json.loads(capsys.readouterr().out)["outcome"] == "planned"
    assert retirement.main([*argv, "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "retired"
    assert retirement.main(["--capture", "bad", "--approval-evidence", EVIDENCE]) == 2
    assert json.loads(capsys.readouterr().out) == {"schema_version": retirement.SCHEMA, "outcome": "refused",
                                                   "code": "parked_delivery_capture_invalid"}


def _orphan(storage, number=0, *, parked_for=8 * DAY):
    """A parked hint whose capture workspace was retired: the hint is all that is left."""
    import shutil

    root = _parked(storage, number, parked_for=parked_for)
    hint = _hint(storage, root)
    assert hint is not None
    shutil.rmtree(root)
    return root


def test_an_orphaned_parked_hint_is_planned_then_archived_never_deleted(tmp_path):
    # 2026-11-06: four of the seven parked deliveries pointed at captures whose workspaces were gone.
    root = _orphan(tmp_path)
    assert index.due(tmp_path, now=NOW, limit=5)[1]["parked_count"] == 1
    plan = _retire(tmp_path, root)
    assert plan["outcome"] == "planned" and plan["capture_absent"] is True
    assert _hint(tmp_path, root) is not None
    done = _retire(tmp_path, root, apply=True, door_caller="operator-1",
                   door_request_id="20261106T170000Z-retire-parked-delivery-0123abcd")
    assert done["outcome"] == "retired" and done["capture_absent"] is True
    assert _hint(tmp_path, root) is None and not root.exists()
    assert index.due(tmp_path, now=NOW, limit=5)[1]["parked_count"] == 0
    key = __import__("hashlib").sha256(_relative(tmp_path, root).encode()).hexdigest()
    archived = json.loads((tmp_path / index.DIRECTORY / retirement.ORPHAN_DIRECTORY / f"{key}.json").read_text())
    assert archived["hint"]["capture_relative_path"] == _relative(tmp_path, root)
    assert (archived["retired_reason"], archived["retirement_approval_evidence"], archived["retired_by_door_caller"]) == (
        retirement.ORPHAN_REASON, EVIDENCE, "operator-1")
    # Idempotent: a second apply finds it archived.
    assert _retire(tmp_path, root, apply=True)["outcome"] == "already_retired"


def test_an_orphaned_hint_must_still_be_parked_for_a_day_and_its_capture_still_absent(tmp_path):
    root = _orphan(tmp_path, parked_for=DAY / 2)
    with pytest.raises(retirement.DeliveryRetirementRefused, match="parked_delivery_parked_under_a_day"):
        _retire(tmp_path, root, apply=True)
    assert _hint(tmp_path, root) is not None
    other = _synthetic_pending(tmp_path, 3, now=NOW)  # a live hint whose capture is then removed
    import shutil
    shutil.rmtree(other)
    with pytest.raises(retirement.DeliveryRetirementRefused, match="parked_delivery_not_parked"):
        _retire(tmp_path, other, apply=True)
    missing = tmp_path / _relative(tmp_path, root).replace("walkthrough-", "walkthrough-x")
    with pytest.raises(retirement.DeliveryRetirementRefused, match="parked_delivery_capture_absent"):
        retirement.retire_parked_delivery(storage_root=tmp_path, capture_relative_path=_relative(tmp_path, missing),
                                          approval_evidence=EVIDENCE, now=NOW, apply=True)
