"""Owner-ledger wiring using synthetic grants/HTTP and temporary existing stores."""

import copy
import json
import os
import shutil
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from blueprint_pipeline import parallel_findall_execution as execution
from blueprint_pipeline import parallel_findall_owner as owner
from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission_grant,
)
from blueprint_pipeline.safe_outbound_http import SafeHttpResponse
from tools.daily_research.firestore import FirestoreLedger
from tools.daily_research.runner import Ledger, Refusal

DAY = "2026-10-03"
NEXT_DAY = "2026-10-04"
OPERATION = "synthetic-owner-operation"
RUN_ID = "findall_synthetic_owner"
FAKE_GRANT = object()  # A test double; no grant issuer is called in these tests.


@pytest.fixture(autouse=True)
def isolated_credentials_and_transport(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    def forbid(*args, **kwargs):
        raise AssertionError("network forbidden in owner-ledger tests")

    def validate(grant, **binding):
        if grant is None:
            require_paid_resource_admission_grant(grant, **binding)
        assert grant is FAKE_GRANT
        assert binding["resource_class"] == "parallel_findall"
        assert binding["require_allocation_binding"] is True

    monkeypatch.setattr(socket.socket, "connect", forbid)
    monkeypatch.setattr(execution.safe_outbound_http, "open_request", forbid)
    monkeypatch.setattr(owner, "require_paid_resource_admission_grant", validate)
    monkeypatch.setattr(execution, "require_paid_resource_admission_grant", validate)


@pytest.fixture
def spec():
    return json.loads((Path(__file__).resolve().parents[1] /
                       "docs/examples/parallel_findall_spec.json").read_text())


def row(day=DAY):
    return {"date": day, "run_key": "blueprint-researcher:" + day, "state": "running",
            "metadata": {"preserved": True}, "unrelated": {"values": [1, 2]}}


@pytest.fixture
def ledger(tmp_path):
    value = Ledger(tmp_path)
    value.put(row())
    yield value
    value.db.close()


def slot(ledger, day=DAY):
    return next(iter(ledger.get(day)[owner.SUBMISSIONS_FIELD].values()))


def payload():
    return {"findall_id": RUN_ID, "generator": "base",
            "status": {"status": "queued", "is_active": True},
            "future_field": {"basis": [{"reasoning": "unknown remains unknown",
                                       "citations": [{"url": "https://example.test/evidence"}]}],
                             "weight": 1.25, "provider_match": "conditional"}}


def submit(ledger, spec, **overrides):
    return owner.create_with_owner_ledger(
        execution.AdmittedFindAllClient("synthetic-key-never-sent"), spec,
        **{"ledger": ledger, "day": DAY, "operation_id": OPERATION,
           "maximum_cost_usd": "1.00", "paid_resource_admission_grant": FAKE_GRANT,
           "current_authority": lambda row, prepared: True, **overrides},
    )


def transport(monkeypatch, ledger, *, result=None, failure=None, inside=None):
    calls = []

    def respond(request, **options):
        entry = slot(ledger)
        assert entry["state"] == "submission_unresolved"
        assert entry["prepared"]["body_json"] == json.loads(request.data)
        calls.append(request)
        if inside:
            inside()
        if failure:
            raise failure
        return SafeHttpResponse(200, json.dumps(result or payload()).encode(),
                                request.full_url, request.full_url)

    monkeypatch.setattr(execution.safe_outbound_http, "open_request", respond)
    return calls


def test_existing_sqlite_commit_and_raw_receipt_are_preserved(monkeypatch, ledger, spec):
    before = ledger.get(DAY)
    calls = transport(monkeypatch, ledger)
    assert submit(ledger, spec) == payload()
    after = ledger.get(DAY)
    assert {k: v for k, v in after.items() if k != owner.SUBMISSIONS_FIELD} == before
    entry = slot(ledger)
    assert entry["state"] == "receipt_retained"
    assert entry["findall_id"] == RUN_ID
    assert json.loads(ledger.read_bytes(entry["receipt_file"])) == payload()
    assert len(calls) == 1


@pytest.mark.parametrize("changed_body", [False, True])
def test_restart_and_changed_body_never_replay(monkeypatch, ledger, spec, changed_body):
    calls = transport(monkeypatch, ledger)
    submit(ledger, spec)
    restarted = Ledger(ledger.root)
    try:
        if changed_body:
            spec["objective"] += " changed"
        with pytest.raises(execution.FindAllError, match="already_claimed"):
            submit(restarted, spec)
        assert len(calls) == 1
    finally:
        restarted.db.close()


def test_operation_cannot_restart_on_another_daily_record(monkeypatch, ledger, spec):
    calls = transport(monkeypatch, ledger)
    submit(ledger, spec)
    ledger.put(row(NEXT_DAY))
    with pytest.raises(execution.FindAllError, match="already_claimed"):
        submit(ledger, spec, day=NEXT_DAY)
    assert owner.SUBMISSIONS_FIELD not in ledger.get(NEXT_DAY)
    assert len(calls) == 1


@pytest.mark.parametrize("decision", [False, None, 1])
def test_current_authority_must_return_literal_true(ledger, spec, decision):
    with pytest.raises(execution.FindAllError, match="not_authorized"):
        submit(ledger, spec, current_authority=lambda r, p: decision)
    assert owner.SUBMISSIONS_FIELD not in ledger.get(DAY)


def test_current_authority_runs_under_existing_lock_and_cannot_rewrite_inputs(monkeypatch, ledger, spec):
    other = Ledger(ledger.root)
    calls = transport(monkeypatch, ledger)

    def check(current, prepared):
        with pytest.raises(Refusal, match="runner_overlap"), other.lock():
            pass
        current["unrelated"] = "changed only in copy"
        prepared["body_json"]["objective"] = "changed only in copy"
        return True

    try:
        submit(ledger, spec, current_authority=check)
        assert ledger.get(DAY)["unrelated"] == {"values": [1, 2]}
        assert json.loads(calls[0].data)["objective"] == spec["objective"]
    finally:
        other.db.close()


def test_missing_grant_refuses_before_owner_state_is_written(ledger, spec):
    with pytest.raises(PaidResourceAdmissionBlocked):
        submit(ledger, spec, paid_resource_admission_grant=None)
    assert owner.SUBMISSIONS_FIELD not in ledger.get(DAY)


def test_requires_existing_owner_record_without_creating_one(ledger, spec):
    with pytest.raises(execution.FindAllError, match="claim_failed"):
        submit(ledger, spec, day=NEXT_DAY)
    assert ledger.get(NEXT_DAY) is None


def test_uncertain_post_keeps_claim_and_prevents_replay(monkeypatch, ledger, spec):
    calls = transport(monkeypatch, ledger, failure=TimeoutError("synthetic private error"))
    with pytest.raises(execution.FindAllSubmissionUnresolved) as caught:
        submit(ledger, spec)
    assert caught.value.findall_id is None
    assert slot(ledger)["state"] == "submission_unresolved"
    with pytest.raises(execution.FindAllError, match="already_claimed"):
        submit(ledger, spec)
    assert len(calls) == 1


def test_committed_claim_with_failed_sidecar_never_posts_or_replays(monkeypatch, ledger, spec):
    original = ledger.put

    def fail_after_commit(value):
        original(value)
        raise OSError("synthetic private disk error")

    monkeypatch.setattr(ledger, "put", fail_after_commit)
    with pytest.raises(execution.FindAllError, match="claim_failed"):
        submit(ledger, spec)
    monkeypatch.setattr(ledger, "put", original)
    with pytest.raises(execution.FindAllError, match="already_claimed"):
        submit(ledger, spec)


@pytest.mark.parametrize("failure", ["write", "readback"])
def test_known_id_is_committed_before_artifact_failure(monkeypatch, ledger, spec, failure):
    calls = transport(monkeypatch, ledger)
    if failure == "write":
        def fail(*args):
            raise OSError("synthetic private artifact error")
        monkeypatch.setattr(ledger, "write_bytes", fail)
    else:
        monkeypatch.setattr(ledger, "read_bytes", lambda name: b"wrong receipt")
    with pytest.raises(execution.FindAllSubmissionUnresolved) as caught:
        submit(ledger, spec)
    assert caught.value.findall_id == RUN_ID
    assert slot(ledger)["findall_id"] == RUN_ID
    assert slot(ledger)["state"] == "provider_id_recorded"
    with pytest.raises(execution.FindAllError, match="already_claimed"):
        submit(ledger, spec)
    assert len(calls) == 1


def test_invalid_provider_status_still_retains_raw_receipt_and_id(monkeypatch, ledger, spec):
    value = payload()
    value["status"] = {"status": 0}
    transport(monkeypatch, ledger, result=value)
    with pytest.raises(execution.FindAllSubmissionUnresolved) as caught:
        submit(ledger, spec)
    assert caught.value.findall_id == RUN_ID
    assert json.loads(ledger.read_bytes(slot(ledger)["receipt_file"])) == value


def test_owner_lock_is_held_through_post_and_receipt(monkeypatch, ledger, spec):
    other = Ledger(ledger.root)

    def overlap():
        with pytest.raises(execution.FindAllError, match="owner_ledger_failed"):
            submit(other, spec, operation_id="different-synthetic-operation")

    try:
        calls = transport(monkeypatch, ledger, inside=overlap)
        submit(ledger, spec)
        assert len(calls) == 1
    finally:
        other.db.close()


def test_release_failure_after_success_preserves_known_id(monkeypatch, ledger, spec):
    lock = ledger.lock

    @contextmanager
    def fail_release():
        with lock():
            yield
        raise OSError("synthetic private release error")

    monkeypatch.setattr(ledger, "lock", fail_release)
    transport(monkeypatch, ledger)
    with pytest.raises(execution.FindAllSubmissionUnresolved) as caught:
        submit(ledger, spec)
    assert caught.value.findall_id == RUN_ID
    assert slot(ledger)["state"] == "receipt_retained"


def test_post_response_read_and_release_failure_preserve_known_id(monkeypatch, ledger, spec):
    lock, get = ledger.lock, ledger.get

    @contextmanager
    def fail_release():
        try:
            with lock():
                yield
        finally:
            raise OSError("synthetic private release error")

    def fail_read(day):
        raise OSError("synthetic private post-response read error")

    monkeypatch.setattr(ledger, "lock", fail_release)
    calls = transport(monkeypatch, ledger,
                      inside=lambda: monkeypatch.setattr(ledger, "get", fail_read))
    with pytest.raises(execution.FindAllSubmissionUnresolved) as caught:
        submit(ledger, spec)
    assert caught.value.findall_id == RUN_ID
    assert str(caught.value) == "findall_submission_unresolved:" + RUN_ID
    monkeypatch.setattr(ledger, "get", get)
    assert slot(ledger)["state"] == "submission_unresolved"
    monkeypatch.setattr(ledger, "lock", lock)
    with pytest.raises(execution.FindAllError, match="already_claimed"):
        submit(ledger, spec)
    assert len(calls) == 1


def test_malformed_existing_journal_fails_before_dispatch(ledger, spec):
    value = ledger.get(DAY)
    value[owner.SUBMISSIONS_FIELD] = {"wrong-key": {"operation_id": OPERATION}}
    ledger.put(value)
    with pytest.raises(execution.FindAllError, match="claim_failed"):
        submit(ledger, spec)


class SyntheticBridge:
    """Real FirestoreLedger wrapper, with an in-memory lease/store transport."""

    def __init__(self):
        self.locked = False
        self.record = row()
        self.files = {}

    def call(self, op, **fields):
        import base64
        if op == "acquire":
            assert not self.locked
            self.locked = True
        elif op == "release":
            self.locked = False
        elif op == "rows":
            return [copy.deepcopy(self.record)]
        elif op == "get":
            return copy.deepcopy(self.record) if fields["day"] == DAY else None
        elif op == "put":
            assert self.locked
            self.record = copy.deepcopy(fields["row"])
        elif op == "file_put":
            assert self.locked
            name = fields["name"]
            assert name.startswith(DAY + "-tool-findall-") and name.endswith(".json")
            raw = base64.b64decode(fields["bytes"])
            assert name not in self.files or self.files[name] == raw
            self.files[name] = raw
        elif op == "file_get":
            return base64.b64encode(self.files[fields["name"]]).decode()
        else:
            raise AssertionError("unexpected bridge operation")


def test_existing_firestore_ledger_interface_needs_no_new_store_or_access(monkeypatch, spec):
    bridge = SyntheticBridge()
    ledger = FirestoreLedger(bridge)
    calls = transport(monkeypatch, ledger)
    assert submit(ledger, spec) == payload()
    assert bridge.locked is False
    assert slot(ledger)["state"] == "receipt_retained"
    assert len(calls) == 1


@pytest.mark.slow
def test_raw_receipt_obeys_actual_firestore_artifact_contract(monkeypatch, ledger, spec):
    """The existing JS store accepts this name and rejects an altered receipt."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the existing Firestore store contract")
    transport(monkeypatch, ledger)
    submit(ledger, spec)
    entry = slot(ledger)
    root = Path(__file__).resolve().parents[1]
    script = r"""
      import assert from 'node:assert/strict';
      import {readFileSync} from 'node:fs';
      globalThis.fetch = () => {throw new Error('network forbidden');};
      const {Store, ROOT} = await import(process.argv[1]);
      const {MemoryFirestore} = await import(process.argv[2]);
      const {name, raw} = JSON.parse(readFileSync(0, 'utf8'));
      const db = new MemoryFirestore(); db.values.set(ROOT, {enabled:true});
      const store = new Store(db, () => 1000, 'synthetic-owner');
      await store.acquire();
      const encoded = Buffer.from(raw).toString('base64');
      assert.equal(await store.filePut(name, encoded), true);
      assert.equal(Buffer.from(await store.fileGet(name), 'base64').toString(), raw);
      assert.equal(await store.filePut(name, encoded), true);
      await assert.rejects(store.filePut(name, Buffer.from('changed receipt').toString('base64')),
        /artifact_identity_conflict/);
      await store.release();
      await assert.rejects(store.filePut(name, encoded), /firestore_lease_lost/);
    """
    # Do not inherit any provider, Firebase, OAuth, or Node preload binding.
    result = subprocess.run(
        [node, "--input-type=module", "-e", script,
         (root / "tools/daily_research/firestore_bridge.mjs").as_uri(),
         (root / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()],
        input=json.dumps({"name": entry["receipt_file"],
                          "raw": ledger.read_bytes(entry["receipt_file"]).decode()}),
        env={"PATH": os.defpath}, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
