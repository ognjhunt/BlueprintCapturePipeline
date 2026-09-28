"""Validator verdicts are proven once per operation and once per content change, never trusted blindly."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

import pytest

from blueprint_pipeline import task_evaluation_release_identity as identity
from blueprint_pipeline import validation_verdict_store as store
from blueprint_pipeline.task_evaluation_sam31_prefix_evidence import file_records, reuse_verdict
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import read
from blueprint_pipeline.validation_file_digests import MINIMUM_BYTES, digest_scope_stats, file_digest_scope

COMMIT = "c" * 40


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


@pytest.fixture
def persisted_root(tmp_path, monkeypatch, _single_threaded_verdict_persistence):
    root = tmp_path / "verdicts"
    monkeypatch.setenv(store.ROOT_ENV, str(root))
    monkeypatch.setattr(identity, "running_release_commit", lambda *args, **kwargs: COMMIT)
    return root


def _inputs(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    big = artifacts / "splat.ply"
    big.write_bytes(os.urandom(1024) * (MINIMUM_BYTES // 1024 + 1))
    small = artifacts / "receipt.json"
    small.write_text(json.dumps({"kind": "receipt", "value": 1}))
    aside = artifacts / "aside.json"
    aside.write_text(json.dumps({"consulted": "by the validator, not listed as a record"}))
    documents = {"splat": _record(big), "nested": [{"receipt": _record(small)}], "note": "no path here"}
    return big, small, aside, documents


def _validator(aside, calls):
    def compute():
        calls.append(1)
        read(aside)  # a file the validator consults without a record in its inputs
        return {"pin": "x", "files": [1, 2], "pair": (3, 4)}
    return compute


def test_file_records_walks_nested_documents(tmp_path):
    big, small, aside, documents = _inputs(tmp_path)
    assert [row["path"] for row in file_records(documents)] == [str(big), str(small)]
    assert file_records({"path": "x", "sha256": "y"}) == []  # incomplete records are not files


def test_verdict_is_computed_once_per_operation_and_copied(tmp_path, persisted_root):
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        first = reuse_verdict("t", ("k", 1), documents, _validator(aside, calls))
        first["pin"] = "mutated"
        second = reuse_verdict("t", ("k", 1), documents, _validator(aside, calls))
        stats = digest_scope_stats()
    assert calls == [1]
    assert second == {"pin": "x", "files": [1, 2], "pair": [3, 4]}  # normalized, unaffected by the caller's mutation
    assert stats["verdicts_computed"] == 1 and stats["verdicts_reused"] == 0


def test_concurrent_thread_refuses_verdict_persistence(tmp_path, monkeypatch):
    root = tmp_path / "verdicts"
    monkeypatch.setenv(store.ROOT_ENV, str(root))
    monkeypatch.setattr(store.threading, "active_count", lambda: 2)
    _big, _small, aside, documents = _inputs(tmp_path)
    with file_digest_scope():
        assert reuse_verdict("t", ("k",), documents, _validator(aside, []))["pin"] == "x"
    assert not list(root.rglob("*.json"))


def test_verdict_is_reused_across_operations_when_nothing_changed(tmp_path, persisted_root):
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        assert reuse_verdict("t", ("k",), documents, _validator(aside, calls))["pin"] == "x"
    entries = list(persisted_root.rglob("*.json"))
    assert len(entries) == 1
    entry = json.loads(entries[0].read_text())
    assert entry["source_commit"] == COMMIT and entry["entry_digest"] and entry["code"]
    assert {row["path"] for row in entry["files"]} == {str(big), str(small), str(aside)}
    assert oct(entries[0].stat().st_mode & 0o777) == "0o640"
    calls.clear()
    with file_digest_scope():
        assert reuse_verdict("t", ("k",), documents, _validator(aside, calls)) == {"pin": "x", "files": [1, 2], "pair": [3, 4]}
        assert digest_scope_stats()["verdicts_reused"] == 1
    assert calls == []
    # An identical rewrite moves the inode identity; the bytes are re-hashed and still reused.
    big.write_bytes(big.read_bytes())
    calls.clear()
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    assert calls == []


def test_changed_consulted_bytes_recompute(tmp_path, persisted_root):
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    # A consulted file that is not a listed record still invalidates the stored verdict.
    aside.write_text(json.dumps({"consulted": "changed"}))
    calls.clear()
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    assert calls == [1]
    # A large artifact with changed bytes recomputes; a listed record that changed refuses outright.
    big.write_bytes(os.urandom(1024) * (MINIMUM_BYTES // 1024 + 1))
    calls.clear()
    with file_digest_scope():
        with pytest.raises(ValueError, match="input_bytes_mismatch"):
            reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    assert calls == []
    documents["splat"] = _record(big)  # the caller's own inputs now name the new bytes: fresh key, fresh proof
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    assert calls == [1]


def test_other_key_or_changed_validator_code_never_reuses(tmp_path, persisted_root, monkeypatch):
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    with file_digest_scope():
        reuse_verdict("t", ("other",), documents, _validator(aside, calls))
    assert calls == [1, 1]
    entry = json.loads(next(persisted_root.glob("t-*.json")).read_text())
    modules = {row["module"] for row in entry["code"]}
    assert {"blueprint_pipeline.validation_verdict_store", "blueprint_pipeline.task_evaluation_sam31_prefix_evidence",
            "blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs"} <= modules
    assert entry["source_commit"] == COMMIT  # provenance only; not part of the reuse decision
    # A validator module whose source changed on disk invalidates every verdict that loaded it.
    original = store._module_digest
    evidence_file = str(Path(__import__("blueprint_pipeline.task_evaluation_sam31_prefix_evidence").task_evaluation_sam31_prefix_evidence.__file__))
    monkeypatch.setattr(store, "_module_digest",
                        lambda path: "sha256:changed" if str(path) == evidence_file else original(path))
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    assert calls == [1, 1, 1]
    monkeypatch.setattr(store, "code_identity", lambda: None)
    monkeypatch.setattr(store, "executed_code_identity", lambda run, always=(): (run(), None))
    before = sorted(persisted_root.glob("t-*.json"))
    with file_digest_scope():
        reuse_verdict("t", ("fresh",), documents, _validator(aside, calls))
        reuse_verdict("t", ("fresh",), documents, _validator(aside, calls))
    with file_digest_scope():
        reuse_verdict("t", ("fresh",), documents, _validator(aside, calls))
    assert calls == [1, 1, 1, 1, 1]  # no code identity: once per operation, never persisted
    assert sorted(persisted_root.glob("t-*.json")) == before


def test_tampered_or_foreign_entries_are_ignored(tmp_path, persisted_root):
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    entry = next(persisted_root.rglob("*.json"))
    value = json.loads(entry.read_text())
    value["verdict"]["pin"] = "forged"
    entry.write_text(json.dumps(value))
    with file_digest_scope():
        assert reuse_verdict("t", ("k",), documents, _validator(aside, calls))["pin"] == "x"
    assert calls == [1, 1]
    entry.write_text("not json")
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    assert calls == [1, 1, 1]


def test_unpersistable_or_unwritable_verdicts_still_work(tmp_path, persisted_root, monkeypatch):
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        value = reuse_verdict("t", ("k",), documents, lambda: (calls.append(1) or {"path": Path("/x")}))
    assert value == {"path": Path("/x")} and not list(persisted_root.rglob("*.json"))
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the root should be")
    monkeypatch.setenv(store.ROOT_ENV, str(blocker / "verdicts"))
    with file_digest_scope():
        assert reuse_verdict("t", ("k",), documents, _validator(aside, calls))["pin"] == "x"
    assert store.store(name="t", key=("k",), files=[], verdict={}) is None
    assert store.lookup(name="t", key=("k",)) is None


def test_source_inputs_are_derived_once_per_operation(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_inputs as inputs
    from tests.test_task_evaluation_scene_configuration_submission import production_fixture
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    task["expected_production_commit"] = "b" * 40
    kwargs = dict(installation_path=fixture["installation_receipt"], preparation_path=fixture["source_preparation"],
                  publisher_path=fixture["publisher_intake"], task=task, commit="b" * 40)
    derivations = []
    original = inputs._source_inputs
    monkeypatch.setattr(inputs, "_source_inputs", lambda **kw: (derivations.append(1), original(**kw))[1])
    with file_digest_scope():
        first = inputs.source_inputs(**kwargs)
        second = inputs.source_inputs(**kwargs)
    assert derivations == [1] and first == second
    inputs.source_inputs(**kwargs)
    assert derivations == [1, 1]  # outside a scope every call derives


def test_worker_advancement_runs_as_one_digest_scoped_operation(tmp_path):
    from blueprint_pipeline import task_evaluation_sam31_preparation_queue as queue
    seen = []

    def advancer(context):
        seen.append((digest_scope_stats() is not None, context["expected_source_commit"]))
        raise ValueError("stop_here")

    with pytest.raises(ValueError, match="stop_here"):
        queue.advance_sam31_for_preparation(queue_root=tmp_path, approved_roots=(tmp_path,), advancer=advancer,
            envelope_context={"request": {"preparation_id": "prep-1", "expected_production_commit": "a" * 40},
                              "request_digest": "sha256:" + "a" * 64})
    assert seen == [(True, "a" * 40)]
    assert digest_scope_stats() is None


def test_verdict_code_is_the_code_that_ran(tmp_path, persisted_root):
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    entry = json.loads(next(persisted_root.rglob("*.json")).read_text())
    modules = {row["module"] for row in entry["code"]}
    assert "blueprint_pipeline.task_evaluation_sam31_prefix_evidence" in modules
    assert "blueprint_pipeline.validation_file_digests" in modules
    assert "blueprint_pipeline.validation_verdict_store" in modules
    assert "blueprint_pipeline.control_plane_storage_gc" not in modules
    assert "blueprint_pipeline.task_evaluation_scene_configuration_artifixer_driver" not in modules
    assert len(modules) < 40


def test_deploy_outside_the_executed_code_keeps_verdicts(tmp_path, persisted_root, monkeypatch):
    """2026-09-13: eight deploys that never touched a validator each cost an eighteen-minute
    re-validation because every loaded module was part of the verdict key."""
    big, small, aside, documents = _inputs(tmp_path)
    calls = []
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    original = store._module_digest

    def changed_outside(path):
        if path.name == "control_plane_storage_gc.py":
            return "sha256:" + "0" * 64
        return original(path)

    monkeypatch.setattr(store, "_module_digest", changed_outside)
    calls.clear()
    with file_digest_scope():
        assert reuse_verdict("t", ("k",), documents, _validator(aside, calls))["pin"] == "x"
    assert calls == []  # a deploy outside the executed code keeps the verdict

    def changed_inside(path):
        if path.name == "validation_file_digests.py":
            return "sha256:" + "1" * 64
        return original(path)

    monkeypatch.setattr(store, "_module_digest", changed_inside)
    calls.clear()
    with file_digest_scope():
        reuse_verdict("t", ("k",), documents, _validator(aside, calls))
    assert calls == [1]  # a change inside the executed code recomputes


def test_a_thread_started_during_a_validation_is_not_traced_after_it(monkeypatch):
    """A worker started inside a validation inherits its tracer through threading.settrace and kept
    it for life: it re-derived code dependencies for unrelated work after the verdict was returned,
    and raised into that work once a caller's test double replaced find_spec (2026-09-28 shard 1)."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        during, code = store.executed_code_identity(lambda: pool.submit(sys.gettrace).result())
        assert during is not None and code is None  # while it ran, the worker was traced and refused persistence
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
        assert pool.submit(store.verdict_root).result() == store.verdict_root()
        assert pool.submit(sys.gettrace).result() is not during


def _tool_hook(frame, event, arg):  # stands in for a tool's hook, such as coverage's or a debugger's
    return None


@pytest.mark.parametrize("hook", [None, _tool_hook], ids=["no hook", "a tool's hook"])
def test_overlapping_validations_in_two_threads_restore_the_thread_hook(hook):
    """Two dispatcher workers validating at once ended out of order: the first restored the hook it
    saw, then the second restored the first one's finished tracer as the process-wide hook, which
    every thread started afterwards inherited."""
    before = threading.gettrace()
    threading.settrace(hook)
    first_running, second_running, first_ended = threading.Event(), threading.Event(), threading.Event()
    waited = {}

    def first():
        waited["second started"] = store.executed_code_identity(
            lambda: (first_running.set(), second_running.wait(5))[1])[0]
        first_ended.set()

    def second():
        first_running.wait(5)
        waited["first ended"] = store.executed_code_identity(
            lambda: (second_running.set(), first_ended.wait(5))[1])[0]

    workers = [threading.Thread(target=first), threading.Thread(target=second)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
    leaked = threading.gettrace()
    threading.settrace(before)  # keep a failure here from reaching later tests
    assert waited == {"second started": True, "first ended": True}  # they overlapped; the first ended first
    assert leaked is hook


def test_nested_validations_restore_each_hook_and_hand_a_worker_outward():
    """An inner validation restores the enclosing one's tracer, for its thread and for new threads, and
    a worker it started is handed to that tracer, which keeps refusing persistence. The outer one
    restores a debugger's tracer and a tool's thread hook."""
    seen = {}

    def debugger(frame, event, arg):
        return None

    def outer():
        seen["outer"] = sys.gettrace()
        with ThreadPoolExecutor(max_workers=1) as pool:
            seen["inner code"] = store.executed_code_identity(lambda: pool.submit(sys.gettrace).result())[1]
            seen["after inner"] = [sys.gettrace(), threading.gettrace(), pool.submit(sys.gettrace).result()]
        return "outer"

    local_before, threads_before = sys.gettrace(), threading.gettrace()
    sys.settrace(debugger)
    threading.settrace(_tool_hook)
    try:
        result = store.executed_code_identity(outer)
        after = [sys.gettrace(), threading.gettrace()]
    finally:
        sys.settrace(local_before)
        threading.settrace(threads_before)
    assert seen["after inner"] == [seen["outer"]] * 3 and seen["inner code"] is None
    assert result == ("outer", None) and after == [debugger, _tool_hook]


@dataclass
class _ValueEqualHook:
    """A thread hook compared by value, hence unhashable, as a tool's callable dataclass may be."""
    threads: list = field(default_factory=list)

    def __call__(self, frame, event, arg):
        self.threads.append(threading.get_ident())


def test_an_unhashable_thread_hook_is_restored_and_raises_into_no_thread():
    """Resolving the saved hook hashed it: every validation raised TypeError from its finally, left its
    dependency context set and its finished tracer installed, and that tracer raised into new threads."""
    hook, before, context = _ValueEqualHook(), threading.gettrace(), contextvars.copy_context()
    threading.settrace(hook)
    started_after = []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = context.run(store.executed_code_identity, lambda: pool.submit(lambda: "run").result())[0]
            restored = threading.gettrace()
            outlived = pool.submit(lambda: "worker").result()  # started inside the run, handed to the hook
        thread = threading.Thread(target=lambda: started_after.append("thread"))
        thread.start()
        thread.join()
    finally:
        threading.settrace(before)
    assert (result, outlived, started_after) == ("run", "worker", ["thread"])
    assert restored is hook and context.get(store._DEPENDENCIES, ()) == ()


def test_a_validation_ending_during_another_ones_restore_leaves_no_finished_tracer(monkeypatch):
    """Forced ordering: the validation owning the hook resolved what to restore, the older one it had
    replaced then ended, and the restore installed that finished tracer as the process-wide hook."""
    before, resolve = threading.gettrace(), store._live_hook
    started, older_running, resolving, older_ended = (threading.Barrier(2), threading.Event(),
                                                      threading.Event(), threading.Event())
    owner, waited = {}, {}

    def paused(hook):
        resolved = resolve(hook)
        if threading.get_ident() == owner.get("thread") and not resolving.is_set():
            resolving.set()
            older_ended.wait(0.5)  # unserialized, the older validation ended right here
        return resolved

    monkeypatch.setattr(store, "_live_hook", paused)

    def older():
        started.wait(5)
        waited["older saw the resolution"] = store.executed_code_identity(
            lambda: (older_running.set(), resolving.wait(5))[1])[0]
        older_ended.set()

    def newer():
        owner["thread"] = threading.get_ident()
        started.wait(5)
        older_running.wait(5)
        store.executed_code_identity(lambda: None)

    workers = [threading.Thread(target=older), threading.Thread(target=newer)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
    leaked = threading.gettrace()
    threading.settrace(before)
    assert waited == {"older saw the resolution": True} and older_ended.is_set()
    assert leaked is before
