# Covers: src/blueprint_pipeline/control_plane_lane_scratch_retention.py
"""ADP-009D/day-28: lane observations grant no cleanup authority."""
from __future__ import annotations

import json
import os
import errno
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_lane_scratch as producer
from blueprint_pipeline import control_plane_lane_scratch_retention as retention
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.fixture
def roots(tmp_path, monkeypatch):
    root, pins = tmp_path.resolve() / "lanes", tmp_path.resolve() / "pins"
    root.mkdir()
    pins.mkdir()
    monkeypatch.setattr(retention, "_ALLOWED_ROOTS", frozenset({str(root)}))
    return root, pins


def lease(**changes):
    value = producer._creation_lease("lane-1", "folder-1", owner="owner-1", reason="fixture",
                                   class_intent="scratch", cleanup="delete", ttl_seconds=100,
                                   run_ref="run-1", now=lambda: 10)
    value.update(changes)
    value["lease_digest"] = canonical_digest(value, digest_field="lease_digest")
    return value


def folder(roots, value=None, raw=None):
    value = value or lease()
    path = roots[0] / value["lane"] / value["name"]
    path.mkdir(parents=True, exist_ok=True)
    (path / producer.LEASE_FILE).write_bytes(raw if raw is not None else json.dumps(value).encode())
    (path / "payload").write_bytes(b"tiny")
    return path


def observe(roots, **kwargs):
    return retention.observe_lane_scratch_retention((str(roots[0]),), pins_root=str(roots[1]),
        observed_at_epoch=50, enabled_requested=True, **kwargs)


def safe(result):
    assert result["status"] in {"report_only", "not_configured"}
    assert result["mutations"] == result["removed_bytes"] == 0
    assert result["candidate_bytes"] is None
    for key in ("execution_authorized", "apply_supported", "general_reference_inventory_complete",
                "queues_checked", "processes_checked", "consumer_fence_checked",
                "owner_approval_checked", "evidence_policy_checked", "restore_checked"):
        assert result[key] is False
    assert all(row["kept"] is True for row in result["rows"])
    assert all(row["bytes"] is None for row in result["retained_by_reason"].values())


@pytest.mark.parametrize("cleanup", ["delete", "offload", "owner_review"])
@pytest.mark.parametrize("state", ["live", "expired", "released"])
@pytest.mark.parametrize("reference", ["run_ref", "scene_ref"])
def test_valid_lease_states_and_intents_are_kept(roots, cleanup, state, reference):
    value = lease(cleanup=cleanup, expires_at_epoch=20 if state == "expired" else 110,
                  released_at_epoch=15 if state == "released" else None)
    if reference == "scene_ref":
        del value["run_ref"]
        value["scene_ref"] = "scene-1"
        value["lease_digest"] = canonical_digest(value, digest_field="lease_digest")
    path = folder(roots, value)
    result = observe(roots)
    safe(result)
    assert result["complete"] is True
    assert result["registered_count"] == result["observed_registered_count"] == 1
    row, = result["rows"]
    assert row["lease_status"] == state
    assert row["reference"] == {reference: value[reference]}
    assert row["logical_bytes"] == sum(p.stat().st_size for p in path.iterdir())
    assert row["pin_match"] == "no_pin_match_in_observed_ledger"
    assert "owner_approval_missing" in row["keep_reasons"]
    assert result["logical_bytes"] == row["logical_bytes"]


def test_valid_cache_and_renewal_producer_contract(roots):
    value = lease(class_intent="cache", size_budget_bytes=16, renewed_at_epoch=30,
                  expires_at_epoch=130)
    folder(roots, value)
    result = observe(roots)
    safe(result)
    assert result["complete"] and result["rows"][0]["class_intent"] == "cache"
    assert "live_lease" in result["rows"][0]["keep_reasons"]


@pytest.mark.parametrize("edit", [
    {"created_at_epoch": True}, {"expires_at_epoch": False}, {"renewed_at_epoch": True},
    {"released_at_epoch": True}, {"created_at_epoch": 10**400}, {"expires_at_epoch": float("inf")},
    {"schema_version": "foreign"}, {"lane": "foreign"}, {"owner": "bad/id"},
    {"size_budget_bytes": True}, {"class_intent": "cache", "size_budget_bytes": None},
    {"extra": "PRIVATE_MARKER"}, {"scene_ref": "scene-1"}, {"lease_digest": "sha256:" + "0" * 64},
])
def test_malformed_sealed_lease_is_unknown(roots, edit):
    value = lease()
    value.update(edit)
    if edit.keys() != {"lease_digest"}:
        try:
            value["lease_digest"] = canonical_digest(value, digest_field="lease_digest")
        except ValueError:
            pass
    path = folder(roots)
    (path / producer.LEASE_FILE).write_bytes(json.dumps(value).encode())
    result = observe(roots)
    safe(result)
    assert not result["complete"] and "lane_lease_invalid" in result["blockers"]
    assert result["registered_count"] is result["logical_bytes"] is result["allocated_bytes"] is None
    assert "PRIVATE_MARKER" not in json.dumps(result)


@pytest.mark.parametrize("raw", [b'{"owner":"one","owner":"two"}', b"[]", b"{", b"x" * 8193])
def test_bad_raw_lease_is_bounded_unknown(roots, raw):
    folder(roots, raw=raw)
    result = observe(roots)
    safe(result)
    assert not result["complete"]


def test_unregistered_folder_is_counted_without_descent(roots, monkeypatch):
    path = roots[0] / "lane-1" / "unregistered"
    path.mkdir(parents=True)
    (path / "not-observed").symlink_to("/private/foreign")
    result = observe(roots)
    safe(result)
    assert result["complete"]
    assert result["unregistered_count"] == 1 and result["registered_count"] == 0
    assert result["rows"] == [] and result["logical_bytes"] == 0


def test_missing_root_is_local_unknown_and_never_created(roots):
    roots[0].rmdir()
    result = observe(roots)
    safe(result)
    assert not result["complete"] and "lane_root_unavailable" in result["blockers"]
    assert not roots[0].exists()


def test_no_configured_roots_does_not_observe_host(roots, monkeypatch):
    monkeypatch.setattr(retention, "observe_storage_pins", lambda *a, **k: pytest.fail("no host scan"))
    result = retention.observe_lane_scratch_retention((), pins_root=str(roots[1]),
        observed_at_epoch=50, enabled_requested=False)
    safe(result)
    assert result["status"] == "not_configured" and not result["complete"]


def test_enabled_observation_uses_no_mutating_seam(roots, monkeypatch):
    folder(roots)
    opened = os.open
    def read_only(name, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        return opened(name, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", read_only)
    for name in ("unlink", "mkdir", "rename", "replace", "rmdir", "write"):
        monkeypatch.setattr(os, name, lambda *a, **k: pytest.fail("mutation"))
    monkeypatch.setattr(producer, "list_lane_scratch", lambda *a, **k: pytest.fail("mutable reader"))
    result = observe(roots)
    safe(result)
    assert result["complete"] and not (roots[0] / ".lane-scratch.lock").exists()


@pytest.mark.parametrize("changes", [
    {"lane_roots": ["/mnt/blueprint-work/lanes"]},
    {"lane_roots": ("/foreign",)}, {"lane_roots": ("/mnt/blueprint-work/lanes/",)},
    {"lane_roots": ("/mnt/blueprint-work/lanes/../lanes",)},
    {"lane_roots": ("/mnt/blueprint-work/lanes",) * 3},
    {"enabled_requested": 1}, {"observed_at_epoch": True}, {"observed_at_epoch": 10**400},
    {"time_budget_seconds": 0}, {"time_budget_seconds": 11}, {"time_budget_seconds": True},
    {"pins_root": "/pins/../foreign"},
    {"pins_root": "/pins/<foreign>"}, {"pins_root": "/pins/*"},
])
def test_parameters_fail_with_fixed_typed_error(roots, changes):
    arguments = dict(lane_roots=(str(roots[0]),), pins_root=str(roots[1]),
                     enabled_requested=False, observed_at_epoch=50)
    arguments.update(changes)
    with pytest.raises(retention.LaneScratchRetentionError, match="^lane_parameters_invalid$"):
        retention.observe_lane_scratch_retention(**arguments)


def test_renewal_during_pin_observation_is_not_a_complete_old_lease(roots, monkeypatch):
    path = folder(roots)
    real = retention.observe_storage_pins
    def changed(*args, **kwargs):
        result = real(*args, **kwargs)
        (path / producer.LEASE_FILE).write_text(json.dumps(lease(renewed_at_epoch=30, expires_at_epoch=130)))
        return result
    monkeypatch.setattr(retention, "observe_storage_pins", changed)
    result = observe(roots)
    safe(result)
    assert not result["complete"]
    assert result["rows"][0]["logical_bytes"] is None
    assert "changed_lease" in result["rows"][0]["keep_reasons"]


@pytest.mark.parametrize("kind", ["root", "lane", "folder", "lease", "payload"])
def test_named_replacement_during_pin_observation_refuses_stability(roots, monkeypatch, kind):
    path = folder(roots)
    target = {"root": roots[0], "lane": path.parent, "folder": path,
              "lease": path / producer.LEASE_FILE, "payload": path / "payload"}[kind]
    real = retention.observe_storage_pins
    def swapped(*args, **kwargs):
        result = real(*args, **kwargs)
        target.rename(target.with_name(target.name + "-old"))
        if kind in {"lease", "payload"}:
            target.write_bytes(b"replacement")
        else:
            target.mkdir()
        return result
    monkeypatch.setattr(retention, "observe_storage_pins", swapped)
    result = observe(roots)
    safe(result)
    assert not result["complete"] and result["logical_bytes"] is None
    assert result["rows"][0]["logical_bytes"] is None


def test_unique_inode_accounting_within_and_across_folders(roots):
    first = folder(roots)
    second = folder(roots, lease(name="folder-2"))
    (second / "payload").unlink()
    os.link(first / "payload", first / "another")
    os.link(first / "payload", second / "payload")
    os.link(first / "payload", roots[0].parent / "external")
    result = observe(roots)
    safe(result)
    assert result["complete"]
    unique = {(p.stat().st_dev, p.stat().st_ino): p.stat()
              for path in (first, second) for p in path.iterdir()}
    assert result["logical_bytes"] == sum(v.st_size for v in unique.values())
    assert result["allocated_bytes"] == sum(v.st_blocks * 512 for v in unique.values())
    assert sum(r["logical_bytes"] for r in result["rows"]) > result["logical_bytes"]
    assert all("hardlinked_payload" in r["keep_reasons"] for r in result["rows"])


def test_sparse_empty_and_nested_regular_payload_never_read(roots, monkeypatch):
    path = folder(roots)
    (path / "nested").mkdir()
    (path / "nested" / "empty").write_bytes(b"")
    with (path / "nested" / "sparse").open("wb") as stream:
        stream.seek(4095)
        stream.write(b"x")
    real = os.read
    def lease_only(fd, *args):
        value = os.fstat(fd)
        assert value.st_ino == (path / producer.LEASE_FILE).stat().st_ino
        return real(fd, *args)
    monkeypatch.setattr(os, "read", lease_only)
    result = observe(roots)
    assert result["complete"]
    assert result["logical_bytes"] == sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    assert result["allocated_bytes"] == sum(p.stat().st_blocks * 512 for p in path.rglob("*") if p.is_file())


@pytest.mark.parametrize("kind", ["payload_link", "payload_fifo", "lease_link", "lease_fifo", "folder_link",
                                 "lane_link", "root_link", "ancestor_link", "staging", "root_unknown", "lock_link"])
def test_unsafe_entries_are_incomplete_without_following_or_blocking(roots, kind, monkeypatch):
    path = folder(roots)
    if kind.startswith("payload") or kind.startswith("lease"):
        target = path / (producer.LEASE_FILE if kind.startswith("lease") else "payload")
        target.unlink()
        os.mkfifo(target) if kind.endswith("fifo") else target.symlink_to("/private/foreign")
    elif kind in {"root_link", "lane_link", "folder_link"}:
        target = {"root_link": roots[0], "lane_link": path.parent, "folder_link": path}[kind]
        moved = target.with_name(target.name + "-moved")
        target.rename(moved)
        target.symlink_to(moved)
    elif kind == "ancestor_link":
        alias = roots[0].parent / "alias"
        alias.symlink_to(roots[0].parent, target_is_directory=True)
        root = str(alias / roots[0].name)
        monkeypatch.setattr(retention, "_ALLOWED_ROOTS", frozenset({root}))
        roots = (alias / roots[0].name, roots[1])
    elif kind == "staging":
        (path.parent / ".folder-2.01234567.tmp").mkdir()
    elif kind == "root_unknown":
        (roots[0] / "foreign").write_bytes(b"x")
    else:
        (roots[0] / ".lane-scratch.lock").symlink_to("/private/foreign")
    result = observe(roots)
    safe(result)
    assert not result["complete"] and result["logical_bytes"] is None


def test_regular_existing_lock_is_observed_without_creation_or_acquisition(roots):
    folder(roots)
    lock = roots[0] / ".lane-scratch.lock"
    lock.write_bytes(b"")
    before = lock.stat()
    result = observe(roots)
    assert result["complete"] and lock.stat() == before


@pytest.mark.parametrize("code", ["pin_deadline_exceeded", "pin_clock_invalid", "pin_inventory_busy", "pin_root_missing"])
def test_empty_partial_pin_observation_keeps_references_unknown(roots, monkeypatch, code):
    folder(roots)
    monkeypatch.setattr(retention, "observe_storage_pins", lambda *a, **k: SimpleNamespace(
        complete=False, protected_paths=(), blockers=(code,)))
    result = observe(roots)
    safe(result)
    assert not result["complete"] and not result["pin_observation_complete"]
    assert result["rows"][0]["pin_match"] == "references_unknown"
    assert result["retained_by_reason"] == {"references_unknown": {"count": 1, "bytes": None}}


@pytest.mark.parametrize("relation,matched", [("equal", True), ("ancestor", True), ("descendant", True), ("sibling", False)])
@pytest.mark.parametrize("complete", [True, False])
def test_pin_matching_is_component_exact_and_never_clears_other_gates(roots, monkeypatch, relation, matched, complete):
    path = folder(roots, lease(expires_at_epoch=20))
    protected = {"equal": str(path), "ancestor": str(path.parent), "descendant": str(path / "payload"),
                 "sibling": str(path) + "-foreign"}[relation]
    monkeypatch.setattr(retention, "observe_storage_pins", lambda *a, **k: SimpleNamespace(
        complete=complete, protected_paths=(protected,)))
    result = observe(roots)
    safe(result)
    row, = result["rows"]
    assert (row["pin_match"] == "referenced") is matched
    assert all(gate in row["keep_reasons"] for gate in retention._GATES)
    assert result["complete"] is complete
    assert ("references_unknown" in row["keep_reasons"]) is (not complete)


def test_expired_unreleased_pin_dependency_is_protective(roots):
    from blueprint_pipeline import control_plane_storage_pins as pins
    path = folder(roots)
    pins.write_storage_pin(kind="preparation", owner_id="dep", paths=[str(path)], depends_on=[],
        pins_root=roots[1], ttl_seconds=20, now=lambda: 10)
    pins.release_storage_pin(kind="preparation", owner_id="dep", pins_root=roots[1], now=lambda: 15)
    pins.write_storage_pin(kind="activation", owner_id="parent", paths=[], depends_on=[{"kind": "preparation", "owner_id": "dep"}],
        pins_root=roots[1], ttl_seconds=20, now=lambda: 10)
    result = observe(roots)
    assert result["complete"] and result["rows"][0]["pin_match"] == "referenced"


@pytest.mark.parametrize("cap,value", [("MAX_LANES", 0), ("MAX_FOLDERS", 0), ("MAX_ENTRIES", 0),
                                     ("MAX_DEPTH", 0), ("MAX_TOTAL_LEASE_BYTES", 1), ("MAX_OUTPUT_BYTES", 1)])
def test_injected_resource_caps_are_unknown_never_truncated_complete(roots, monkeypatch, cap, value):
    path = folder(roots)
    (path / "nested").mkdir()
    monkeypatch.setattr(retention, cap, value)
    result = observe(roots)
    safe(result)
    assert not result["complete"] and result["logical_bytes"] is None and result["blockers"]


@pytest.mark.parametrize("clock", [lambda: float("nan"), lambda: True, lambda: 10**400])
def test_invalid_clock_is_bounded_empty_unknown(roots, clock):
    folder(roots)
    result = observe(roots, monotonic=clock)
    safe(result)
    assert not result["complete"] and result["rows"] == []
    assert "lane_clock_invalid" in result["blockers"]


def test_deadline_after_first_row_discards_only_as_unknown_and_closes_handles(roots, monkeypatch):
    folder(roots)
    real = retention._Observation.folder
    expired = False
    def accepted(self, *args, **kwargs):
        nonlocal expired
        real(self, *args, **kwargs)
        assert len(self.rows) == 1
        expired = True
    monkeypatch.setattr(retention._Observation, "folder", accepted)
    opened, closed = [], []
    open_real, close_real = os.open, os.close
    def opening(*args, **kwargs):
        fd = open_real(*args, **kwargs)
        opened.append(fd)
        return fd
    def closing(fd):
        close_real(fd)
        closed.append(fd)
    monkeypatch.setattr(os, "open", opening)
    monkeypatch.setattr(os, "close", closing)
    result = observe(roots, monotonic=lambda: 20 if expired else 0)
    safe(result)
    assert result["rows"] == [] and "lane_deadline_exceeded" in result["blockers"]
    assert sorted(opened) == sorted(closed)


def test_definite_one_shot_close_failure_is_retained_for_cleanup(roots, monkeypatch):
    folder(roots)
    real = os.close
    failed = False
    target = None
    def closing(fd):
        nonlocal failed, target
        if not failed:
            failed, target = True, fd
            raise OSError(errno.EIO, "synthetic")
        return real(fd)
    monkeypatch.setattr(os, "close", closing)
    result = observe(roots)
    safe(result)
    assert not result["complete"] and "lane_descriptor_close_failed" in result["blockers"]
    with pytest.raises(OSError):
        os.fstat(target)


@pytest.mark.parametrize("kind", ["cross_device", "permission", "vanished"])
def test_payload_syscall_failures_are_local_and_sizes_unknown(roots, monkeypatch, kind):
    folder(roots)
    real = os.stat
    def checking(name, *args, **kwargs):
        value = real(name, *args, **kwargs)
        if name == "payload" and kwargs.get("dir_fd") is not None:
            if kind == "permission":
                raise PermissionError("PRIVATE_MARKER")
            if kind == "vanished":
                raise FileNotFoundError("PRIVATE_MARKER")
            return SimpleNamespace(st_dev=value.st_dev + 1)
        return value
    monkeypatch.setattr(os, "stat", checking)
    result = observe(roots)
    safe(result)
    assert not result["complete"] and result["rows"][0]["logical_bytes"] is None
    assert "PRIVATE_MARKER" not in json.dumps(result)


def test_injected_caps_accept_the_inclusive_boundary(roots, monkeypatch):
    path = folder(roots)
    raw_size = (path / producer.LEASE_FILE).stat().st_size
    for cap, value in (("MAX_LANES", 1), ("MAX_FOLDERS", 1), ("MAX_ENTRIES", 4),
                       ("MAX_DEPTH", 0), ("MAX_LEASE_BYTES", raw_size), ("MAX_TOTAL_LEASE_BYTES", raw_size * 2 + 1)):
        monkeypatch.setattr(retention, cap, value)
    result = observe(roots)
    assert result["complete"]
    assert result["registered_count"] == 1


def test_reference_comparison_cap_never_becomes_negative_clearance(roots, monkeypatch):
    folder(roots)
    folder(roots, lease(name="folder-2"))
    monkeypatch.setattr(retention, "MAX_COMPARISONS", 1)
    monkeypatch.setattr(retention, "observe_storage_pins", lambda *a, **k: SimpleNamespace(
        complete=True, protected_paths=("/unrelated/one", "/unrelated/two")))
    result = observe(roots)
    safe(result)
    assert not result["complete"] and "lane_reference_comparisons_limit" in result["blockers"]
    assert all(r["pin_match"] == "references_unknown" for r in result["rows"])


def test_missing_and_busy_real_pin_ledgers_are_unknown(roots):
    import fcntl
    folder(roots)
    roots[1].rmdir()
    result = observe(roots)
    assert not result["complete"] and result["rows"][0]["pin_match"] == "references_unknown"
    roots[1].mkdir()
    fd = os.open(roots[1], os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = observe(roots)
        assert not result["complete"] and result["rows"][0]["pin_match"] == "references_unknown"
    finally:
        os.close(fd)


def test_unregistered_payload_is_never_enumerated(roots, monkeypatch):
    path = roots[0] / "lane-1" / "unregistered"
    path.mkdir(parents=True)
    real = os.scandir
    def checking(fd):
        assert os.fstat(fd).st_ino != path.stat().st_ino
        return real(fd)
    monkeypatch.setattr(os, "scandir", checking)
    result = observe(roots)
    assert result["complete"] and result["unregistered_count"] == 1


def test_iterator_bound_stops_before_collecting_an_extra_entry(roots, monkeypatch):
    folder(roots)
    monkeypatch.setattr(retention, "MAX_ENTRIES", 0)
    real = os.scandir
    yielded = 0
    class Counting:
        def __init__(self, fd):
            self.iterator = real(fd)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.iterator.close()
        def __iter__(self):
            nonlocal yielded
            for value in self.iterator:
                yielded += 1
                yield value
    monkeypatch.setattr(os, "scandir", Counting)
    result = observe(roots)
    assert not result["complete"] and yielded == 1


@pytest.mark.parametrize("mode", ["invalid", "backwards", "deadline"])
def test_finalization_guard_precedes_serialization(roots, monkeypatch, mode):
    folder(roots)
    finalizing = False
    real = retention._Observation.result
    def result(self, *args, **kwargs):
        nonlocal finalizing
        finalizing = True
        return real(self, *args, **kwargs)
    monkeypatch.setattr(retention._Observation, "result", result)
    def clock():
        return {"invalid": float("nan"), "backwards": -1, "deadline": 20}[mode] if finalizing else 0
    result = observe(roots, monotonic=clock)
    safe(result)
    assert not result["complete"] and result["rows"] == []
    expected = "lane_deadline_exceeded" if mode == "deadline" else "lane_clock_invalid"
    assert expected in result["blockers"]


def test_bad_later_lease_retains_valid_positive_rows_with_available_budget(roots):
    folder(roots)
    folder(roots, lease(name="folder-2"), raw=b"bad")
    result = observe(roots)
    safe(result)
    assert not result["complete"] and len(result["rows"]) == 1
    assert result["rows"][0]["logical_bytes"] is not None
    assert result["observed_registered_count"] == 1 and result["registered_count"] is None


@pytest.mark.slow
def test_cold_import_preserves_legacy_reader_contract(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", """
import inspect
from blueprint_pipeline import control_plane_lane_scratch as producer
from blueprint_pipeline import control_plane_storage_pins as legacy
before = (producer.list_lane_scratch, legacy.load_storage_pins, inspect.signature(legacy.load_storage_pins))
from blueprint_pipeline import control_plane_lane_scratch_retention
assert before == (producer.list_lane_scratch, legacy.load_storage_pins, inspect.signature(legacy.load_storage_pins))
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_pin_clock_is_guarded_by_the_shared_lane_deadline(roots, monkeypatch):
    folder(roots)
    expired = False
    def pins(*args, monotonic, **kwargs):
        nonlocal expired
        expired = True
        with pytest.raises(retention._Blocked) as blocked:
            monotonic()
        assert blocked.value.code == "lane_deadline_exceeded"
        return SimpleNamespace(complete=False, protected_paths=())
    monkeypatch.setattr(retention, "observe_storage_pins", pins)
    result = observe(roots, monotonic=lambda: 10 if expired else 0)
    safe(result)
    assert not result["complete"] and result["rows"] == []
    assert "lane_deadline_exceeded" in result["blockers"]


def test_cross_device_root_lock_metadata_is_incomplete(roots, monkeypatch):
    folder(roots)
    (roots[0] / ".lane-scratch.lock").write_bytes(b"")
    real = os.stat
    def mounted(name, *args, **kwargs):
        value = real(name, *args, **kwargs)
        if name == ".lane-scratch.lock" and kwargs.get("dir_fd") is not None:
            return SimpleNamespace(st_mode=value.st_mode, st_dev=value.st_dev + 1,
                                   **{key: getattr(value, key) for key in dir(value)
                                      if key.startswith("st_") and key not in {"st_mode", "st_dev"}})
        return value
    monkeypatch.setattr(os, "stat", mounted)
    result = observe(roots)
    safe(result)
    assert not result["complete"] and result["logical_bytes"] is None
    assert "lane_metadata_unsafe" in result["blockers"]


@pytest.mark.parametrize("state", ["published", "renewed", "released"])
def test_real_producer_publication_renewal_and_release_bytes_are_compatible(roots, state):
    path = producer.create_lane_scratch("lane-1", "folder-1", owner="owner-1", reason="fixture",
        class_intent="scratch", cleanup="owner_review", ttl_seconds=100,
        scene_ref="scene-1", root=roots[0], now=lambda: 10)
    before = json.loads((path / producer.LEASE_FILE).read_bytes())
    common = dict(root=roots[0], lane="lane-1", name="folder-1", owner="owner-1",
                  expected_digest=before["lease_digest"], now=lambda: 20)
    if state == "renewed":
        producer.renew_lane_scratch(**common, ttl_seconds=100)
    elif state == "released":
        producer.release_lane_scratch(**common)
    expected = json.loads((path / producer.LEASE_FILE).read_bytes())
    result = observe(roots)
    safe(result)
    assert result["complete"]
    assert result["rows"][0]["lease_digest"] == expected["lease_digest"]
    assert result["rows"][0]["lease_status"] == ("released" if state == "released" else "live")
