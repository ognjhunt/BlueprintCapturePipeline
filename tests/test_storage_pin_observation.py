# Covers: src/blueprint_pipeline/control_plane_storage_pin_observation.py
"""ADP-009D/day-28: strict observations cannot turn missing pins into clearance."""
from __future__ import annotations

import dataclasses
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from blueprint_pipeline import control_plane_storage_pin_observation as observer
from blueprint_pipeline import control_plane_storage_pins as legacy


def pin(*, kind="preparation", owner="owner-1", paths=None, dependencies=None,
        created=10, expires=100, released=None):
    return {"schema_version": legacy.SCHEMA_VERSION, "kind": kind, "owner_id": owner,
            "paths": ["/payload/candidate"] if paths is None else paths,
            "depends_on": [] if dependencies is None else dependencies,
            "created_at_epoch": created, "expires_at_epoch": expires,
            "released_at_epoch": released}


def write(root, value, *, filename=None, raw=None):
    directory = root / value["kind"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (filename or value["owner_id"] + ".json")
    payload = raw if raw is not None else json.dumps(value).encode()
    path.write_bytes(payload)
    return path


@pytest.fixture
def root(tmp_path):
    root = tmp_path.resolve() / "pins"
    root.mkdir()
    return root


def observe(root, **kwargs):
    return observer.observe_storage_pins(str(root), observed_at_epoch=50, **kwargs)


def incomplete(result, code=None):
    assert result.complete is False
    assert result.blockers
    assert all(len(code) < 100 and "/" not in code for code in result.blockers)
    assert result.general_reference_inventory_complete is False
    assert result.consumer_fence_checked is False
    assert result.execution_authorized is False
    assert result.mutations == 0
    if code:
        assert code in result.blockers


def test_missing_root_is_unknown_and_is_not_created(root, monkeypatch):
    missing = root / "missing"
    def forbidden(*args, **kwargs):
        pytest.fail("observer must not mutate or use the permissive loader")
    monkeypatch.setattr(legacy, "storage_pin_guard", forbidden)
    monkeypatch.setattr(legacy, "load_storage_pins", forbidden)
    monkeypatch.setattr(legacy, "release_storage_pin", forbidden)
    monkeypatch.setattr(legacy, "write_storage_pin", forbidden)
    incomplete(observe(missing), "pin_root_missing")
    assert not missing.exists()


def test_existing_empty_root_has_a_complete_pin_only_observation(root):
    result = observe(root)
    assert result.complete is True
    assert result.scope == "storage_pins_only"
    assert result.rows == result.protected_paths == result.protected_identities == ()
    assert result.root_identity == (root.stat().st_dev, root.stat().st_ino)
    assert result.blockers == ()
    assert result.general_reference_inventory_complete is False


def test_borrowed_root_lock_is_retained_through_observation(root):
    held = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    probe = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = observe(root, _held_root_fd=held)
        assert result.complete and result.root_identity == (root.stat().st_dev, root.stat().st_ino)
        os.fstat(held)  # Observation did not close the caller's lock descriptor.
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
    finally:
        os.close(probe)
        os.close(held)


def test_valid_row_is_frozen_and_keeps_raw_identity(root):
    path = write(root, pin(paths=["/payload/b", "/payload/a", "/payload/a"]))
    raw = path.read_bytes()
    result = observe(root)
    assert result.complete
    row, = result.rows
    assert row.paths == ("/payload/a", "/payload/b")
    assert row.raw_sha256 == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert row.raw_size_bytes == len(raw)
    assert row.row_path == str(path)
    assert row.status == "live"
    assert result.protected_paths == row.paths
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.owner_id = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.complete = False


@pytest.mark.parametrize("edit", [
    lambda row: row.update(schema_version="unknown"),
    lambda row: row.update(owner_id="different-owner"),
    lambda row: row.update(created_at_epoch=True),
    lambda row: row.update(expires_at_epoch=False),
    lambda row: row.update(expires_at_epoch=10),
    lambda row: row.update(created_at_epoch=-1),
    lambda row: row.update(released_at_epoch="bad"),
    lambda row: row.update(released_at_epoch=9),
    lambda row: row.update(released_at_epoch=51),
    lambda row: row.update(paths="/payload"),
    lambda row: row.update(paths=["relative"]),
    lambda row: row.update(paths=["/payload/../foreign"]),
    lambda row: row.update(paths=["/payload/<redacted>"]),
    lambda row: row.update(depends_on=[{"kind": "unknown", "owner_id": "other"}]),
    lambda row: row.update(depends_on=[{"kind": "activation", "owner_id": "bad/id"}]),
    lambda row: row.update(depends_on=[{"kind": "activation", "owner_id": "other", "extra": True}]),
    lambda row: row.update(extra="unexpected"),
])
def test_strict_row_structure_never_becomes_empty_clearance(root, edit):
    row = pin()
    edit(row)
    write(root, row, filename="owner-1.json")
    incomplete(observe(root), "pin_row_invalid")


@pytest.mark.parametrize("raw", [b'not-json', b'\xff', b'{"x":NaN}', b'{"x":Infinity}',
                                 b'{"x":1e400}', b'{"x":' + b'9' * 400 + b'}',
                                 b'{"x":"\\ud800"}', b'{"x":1,"x":2}'])
def test_json_invalid_is_a_fixed_unknown_reason(root, raw):
    write(root, pin(), raw=raw)
    incomplete(observe(root), "pin_row_invalid")


@pytest.mark.parametrize("entry", ["unknown-kind", "foreign-file", ".pin-stage"])
def test_unknown_root_or_kind_entries_make_inventory_incomplete(root, entry):
    if entry == "unknown-kind":
        (root / entry).mkdir()
    elif entry == "foreign-file":
        (root / entry).write_text("unknown")
    else:
        (root / "preparation").mkdir()
        (root / "preparation" / entry).write_text("unpublished")
    incomplete(observe(root), "pin_entry_unknown")


def test_missing_dependency_is_unknown_even_on_a_released_row(root):
    write(root, pin(released=40, dependencies=[{"kind": "compilation", "owner_id": "absent"}]))
    incomplete(observe(root), "pin_dependency_unavailable")


def test_permissive_existing_loader_still_skips_invalid_and_expired_paths(root):
    path = write(root, pin(expires=20))
    (path.parent / "invalid.json").write_text("invalid")
    assert len(legacy.load_storage_pins(root, now=lambda: 50)) == 1
    assert legacy.live_pinned_paths(root, now=lambda: 50) == set()
    incomplete(observe(root))


@pytest.mark.parametrize("root_value", [None, "relative", "/bad/../root", "/bad//root", "/bad/<redacted>"])
def test_bad_api_root_is_typed_without_echoing_values(root_value):
    with pytest.raises(observer.StoragePinObservationError) as error:
        observer.observe_storage_pins(root_value, observed_at_epoch=50)
    assert str(error.value) == "pin_parameters_invalid"


@pytest.mark.parametrize("field,value", [("observed_at_epoch", True), ("observed_at_epoch", float('nan')),
                                        ("observed_at_epoch", 10**400), ("time_budget_seconds", 0),
                                        ("time_budget_seconds", True), ("time_budget_seconds", 6)])
def test_bad_numeric_parameters_are_typed(root, field, value):
    kwargs = {"observed_at_epoch": 50, field: value}
    with pytest.raises(observer.StoragePinObservationError, match="^pin_parameters_invalid$"):
        observer.observe_storage_pins(str(root), **kwargs)


@pytest.mark.parametrize("level", ["ancestor", "root", "kind", "row"])
def test_symlinks_are_never_followed(root, level):
    path = write(root, pin())
    if level == "row":
        other = root / "foreign.json"
        path.rename(other)
        path.symlink_to(other)
    elif level == "kind":
        other = root / "foreign-kind"
        path.parent.rename(other)
        (root / "preparation").symlink_to(other, target_is_directory=True)
    elif level == "root":
        other = root.with_name("foreign-root")
        root.rename(other)
        root.symlink_to(other, target_is_directory=True)
    else:
        linked = root.parent / "linked-ancestor"
        linked.symlink_to(root.parent, target_is_directory=True)
        root = linked / root.name
    incomplete(observe(root))


def test_nonblocking_guard_reports_busy_without_waiting(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        incomplete(observe(root), "pin_inventory_busy")
    finally:
        os.close(fd)


@pytest.mark.parametrize("fault", ["open", "fstat", "read", "scandir", "stat", "flock"])
def test_syscall_faults_are_fixed_and_all_owned_fds_close(root, monkeypatch, fault):
    write(root, pin())
    opened = set()
    real_open, real_close = observer.os.open, observer.os.close

    def tracked_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.add(fd)
        return fd

    def tracked_close(fd):
        opened.remove(fd)
        real_close(fd)

    monkeypatch.setattr(observer.os, "open", tracked_open)
    monkeypatch.setattr(observer.os, "close", tracked_close)
    def fail(*args, **kwargs):
        raise PermissionError("PRIVATE_TEST_MARKER")
    if fault == "flock":
        monkeypatch.setattr(observer.fcntl, "flock", fail)
    elif fault == "open":
        def fail_row(name, *args, **kwargs):
            if name.endswith('.json'):
                return fail()
            return tracked_open(name, *args, **kwargs)
        monkeypatch.setattr(observer.os, "open", fail_row)
    else:
        monkeypatch.setattr(observer.os, fault, fail)
    result = observe(root)
    incomplete(result)
    assert not opened
    assert "PRIVATE_TEST_MARKER" not in str(result)


@pytest.mark.parametrize("race", ["truncate", "replace", "remove", "fifo", "kind", "root"])
def test_deterministic_changes_at_read_edge_cannot_be_complete(root, monkeypatch, race):
    path = write(root, pin())
    real_read = observer.os.read
    fired = False
    def racing_read(fd, count):
        nonlocal fired
        raw = real_read(fd, count)
        if not fired:
            fired = True
            if race == "truncate":
                path.write_bytes(b'{}')
            elif race == "replace":
                path.unlink()
                path.write_bytes(raw)
            elif race == "remove":
                path.unlink()
            elif race == "fifo":
                path.unlink()
                os.mkfifo(path)
            elif race == "kind":
                path.parent.rename(root / "old-kind")
                (root / "preparation").mkdir()
            else:
                root.rename(root.with_name("old-pins"))
                root.mkdir()
        return raw
    monkeypatch.setattr(observer.os, "read", racing_read)
    incomplete(observe(root))
    assert fired


def test_row_modified_after_read_is_detected_by_final_identity(root, monkeypatch):
    path = write(root, pin())
    real_validate = observer._Scan.validate
    def changed_after_validation(scan, *args, **kwargs):
        row = real_validate(scan, *args, **kwargs)
        changed = pin(paths=["/payload/different"])
        path.write_text(json.dumps(changed))
        return row
    monkeypatch.setattr(observer._Scan, "validate", changed_after_validation)
    incomplete(observe(root), "pin_row_changed")


def test_fifo_present_before_open_never_reads_or_blocks(root, monkeypatch):
    directory = root / "preparation"
    directory.mkdir()
    os.mkfifo(directory / "owner-1.json")
    def forbidden(*args, **kwargs):
        pytest.fail("must prove regular before attempting a read")
    monkeypatch.setattr(observer.os, "read", forbidden)
    incomplete(observe(root), "pin_row_unavailable")


def test_budget_rejects_declared_next_row_before_read_or_parse(root, monkeypatch):
    first = write(root, pin(owner="a"))
    write(root, pin(owner="b"))
    monkeypatch.setattr(observer, "MAX_TOTAL_BYTES", first.stat().st_size)
    parsed = []
    real_json = observer._json
    def tracked(raw):
        parsed.append(raw)
        return real_json(raw)
    monkeypatch.setattr(observer, "_json", tracked)
    incomplete(observe(root), "pin_bytes_limit")
    assert len(parsed) == 1


def test_entry_budget_stops_iteration_before_sort_or_full_collection(root, monkeypatch):
    for name in ("a", "b", "c", "d"):
        (root / name).mkdir()
    monkeypatch.setattr(observer, "MAX_ENTRIES", 2)
    counted = []
    real_scandir = observer.os.scandir
    class Entries:
        def __init__(self, fd):
            self.iterator = real_scandir(fd)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.iterator.close()
        def __iter__(self):
            for item in self.iterator:
                counted.append(item.name)
                yield item
    monkeypatch.setattr(observer.os, "scandir", Entries)
    incomplete(observe(root), "pin_entries_limit")
    assert len(counted) == 3  # One bounded overflow sentinel, never the whole directory.


@pytest.mark.parametrize("limit,value,code", [
    ("MAX_ROW_BYTES", 2, "pin_row_bytes_limit"), ("MAX_TOTAL_BYTES", 2, "pin_bytes_limit"),
    ("MAX_ROWS", 0, "pin_rows_limit"), ("MAX_VALUES", 0, "pin_values_limit"),
    ("MAX_OUTPUT_BYTES", 8, "pin_output_limit"),
])
def test_tiny_injected_caps_are_incomplete(root, monkeypatch, limit, value, code):
    write(root, pin())
    monkeypatch.setattr(observer, limit, value)
    incomplete(observe(root), code)


def test_exact_row_total_entry_value_and_output_caps_are_inclusive(root, monkeypatch):
    path = write(root, pin())
    expected = observe(root)
    monkeypatch.setattr(observer, "MAX_ROW_BYTES", path.stat().st_size)
    monkeypatch.setattr(observer, "MAX_TOTAL_BYTES", path.stat().st_size)
    monkeypatch.setattr(observer, "MAX_ROWS", 1)
    monkeypatch.setattr(observer, "MAX_ENTRIES", 2)
    monkeypatch.setattr(observer, "MAX_VALUES", 1)
    monkeypatch.setattr(observer, "MAX_OUTPUT_BYTES", len(json.dumps(dataclasses.asdict(expected), ensure_ascii=False).encode()))
    assert observe(root) == expected


@pytest.mark.parametrize("clock", [lambda: float('nan'), lambda: True,
                                   lambda: 10**400,
                                   lambda: (_ for _ in ()).throw(RuntimeError("PRIVATE_TEST_MARKER"))])
def test_bad_clock_is_fixed_unknown_not_a_traceback(root, clock):
    incomplete(observe(root, monotonic=clock), "pin_clock_invalid")


def test_deadline_and_post_open_deadline_close_owned_descriptors(root, monkeypatch):
    write(root, pin())
    opened = set()
    real_open, real_close = observer.os.open, observer.os.close
    def tracked_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.add(fd)
        return fd
    def tracked_close(fd):
        opened.remove(fd)
        real_close(fd)
    monkeypatch.setattr(observer.os, "open", tracked_open)
    monkeypatch.setattr(observer.os, "close", tracked_close)
    values = iter([0.0, 6.0])
    incomplete(observe(root, monotonic=lambda: next(values, 6.0)), "pin_deadline_exceeded")
    assert not opened


def test_backwards_clock_is_unknown_and_cannot_extend_deadline(root):
    write(root, pin())
    values = iter([1.0, 0.0])
    incomplete(observe(root, monotonic=lambda: next(values, 0.0)), "pin_clock_invalid")


def test_derived_row_provenance_path_obeys_the_same_lexical_cap(root, monkeypatch):
    write(root, pin())
    monkeypatch.setattr(observer, "MAX_PATH_BYTES", len(str(root).encode()))
    incomplete(observe(root), "pin_row_invalid")


@pytest.mark.parametrize("has_row", [False, True])
def test_one_shot_definite_close_failure_retains_ownership_for_cleanup(root, monkeypatch, has_row):
    if has_row:
        write(root, pin())
    opened = set()
    real_open, real_close = observer.os.open, observer.os.close
    failed = False
    def tracked_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.add(fd)
        return fd
    def fault_close(fd):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError(errno.EACCES, "PRIVATE_TEST_MARKER")
        opened.remove(fd)
        real_close(fd)
    monkeypatch.setattr(observer.os, "open", tracked_open)
    monkeypatch.setattr(observer.os, "close", fault_close)
    incomplete(observe(root), "pin_descriptor_close_failed")
    assert not opened


def test_close_error_after_descriptor_was_closed_never_recloses_foreign_inode(root, monkeypatch):
    write(root, pin())
    foreign = root.parent / "foreign"
    foreign.write_bytes(b'foreign')
    real_open, real_close = observer.os.open, observer.os.close
    replacement = None
    fired = False
    def raced_close(fd):
        nonlocal fired, replacement
        if not fired:
            fired = True
            real_close(fd)
            replacement = real_open(foreign, os.O_RDONLY)
            assert replacement == fd
            raise OSError(errno.EIO, "ambiguous completion")
        assert fd != replacement, "must preserve foreign reused descriptor"
        real_close(fd)
    monkeypatch.setattr(observer.os, "close", raced_close)
    try:
        incomplete(observe(root), "pin_descriptor_close_failed")
        assert os.read(replacement, 7) == b'foreign'
    finally:
        if replacement is not None:
            real_close(replacement)


@pytest.mark.parametrize("failure", [ValueError, TypeError, UnicodeError, OverflowError])
def test_final_result_canonicalization_fault_is_fixed_incomplete(root, monkeypatch, failure):
    write(root, pin())
    real_asdict = observer.asdict
    def faulty(value):
        if isinstance(value, observer.StoragePinObservation):
            raise failure("PRIVATE_TEST_MARKER")
        return real_asdict(value)
    monkeypatch.setattr(observer, "asdict", faulty)
    result = observe(root)
    incomplete(result, "pin_result_invalid")
    assert "PRIVATE_TEST_MARKER" not in str(result)


def test_expired_unreleased_pins_and_released_dependencies_stay_protective(root):
    write(root, pin(kind="activation", owner="active", expires=20,
                    dependencies=[{"kind": "preparation", "owner_id": "released"}]))
    write(root, pin(owner="released", paths=["/payload/dependency"], released=40))
    write(root, pin(kind="compilation", owner="unneeded", paths=["/payload/not-protected"], released=40))
    result = observe(root)
    assert result.complete
    assert result.protected_identities == (observer.PinIdentity("activation", "active"),
                                           observer.PinIdentity("preparation", "released"))
    assert result.protected_paths == ("/payload/candidate", "/payload/dependency")
    assert next(row for row in result.rows if row.owner_id == "active").status == "expired_unreleased"


def test_cycles_are_finite_and_duplicate_dependency_values_normalize(root):
    identity = {"kind": "compilation", "owner_id": "b"}
    write(root, pin(owner="a", dependencies=[identity, identity]))
    write(root, pin(kind="compilation", owner="b",
                    dependencies=[{"kind": "preparation", "owner_id": "a"}]))
    result = observe(root)
    assert result.complete
    assert len(result.protected_identities) == 2
    assert next(row for row in result.rows if row.owner_id == "a").depends_on == (observer.PinIdentity(**identity),)


def test_duplicates_count_toward_value_work_cap_before_normalization(root, monkeypatch):
    write(root, pin(paths=["/payload/a"] * 3))
    monkeypatch.setattr(observer, "MAX_VALUES", 2)
    incomplete(observe(root), "pin_values_limit")


def test_valid_partial_rows_survive_later_bad_row_but_never_clear_references(root):
    write(root, pin(owner="a"))
    write(root, pin(owner="z"), raw=b'invalid')
    result = observe(root)
    incomplete(result, "pin_row_invalid")
    assert [row.owner_id for row in result.rows] == ["a"]
    assert result.protected_paths == ("/payload/candidate",)


def test_empty_paths_are_producer_compatible_and_still_protect_identity(root):
    write(root, pin(paths=[]))
    result = observe(root)
    assert result.complete
    assert result.protected_paths == ()
    assert result.protected_identities == (observer.PinIdentity("preparation", "owner-1"),)
    assert result.general_reference_inventory_complete is False


def test_expiry_boundary_is_unreleased_and_future_release_is_unknown(root):
    path = write(root, pin(expires=50))
    assert observe(root).rows[0].status == "expired_unreleased"
    path.write_text(json.dumps(pin(released=50.01)))
    incomplete(observe(root), "pin_row_invalid")


def test_stable_observations_sort_identically_without_mutation_or_payload_access(root, monkeypatch):
    write(root, pin(owner="z"))
    write(root, pin(kind="activation", owner="a"))
    expected = observe(root)
    assert [(row.kind, row.owner_id) for row in expected.rows] == [("activation", "a"), ("preparation", "z")]
    original_open, original_stat = observer.os.open, observer.os.stat
    def safe_open(name, flags, *args, **kwargs):
        assert not flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND | os.O_RDWR | os.O_WRONLY)
        assert "/payload" not in str(name)
        return original_open(name, flags, *args, **kwargs)
    def safe_stat(name, *args, **kwargs):
        assert "/payload" not in str(name)
        return original_stat(name, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("read-only pin observer must never mutate")
    monkeypatch.setattr(observer.os, "open", safe_open)
    monkeypatch.setattr(observer.os, "stat", safe_stat)
    for name in ("mkdir", "unlink", "replace", "rename", "write"):
        monkeypatch.setattr(observer.os, name, forbidden)
    assert observe(root) == expected


def test_distinct_fixed_blockers_are_bounded(root, monkeypatch):
    write(root, pin(owner="a"), raw=b'invalid')
    (root / "preparation" / ".pin-stage").write_text("stage")
    monkeypatch.setattr(observer, "MAX_BLOCKERS", 1)
    result = observe(root)
    incomplete(result)
    assert len(result.blockers) <= 2
    assert "pin_blockers_truncated" in result.blockers


@pytest.mark.slow
def test_cold_import_remains_leaf_only_and_does_not_import_paid_runtime():
    source = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    code = (
        "import sys; import blueprint_pipeline.control_plane_storage_pin_observation; "
        "expected={'blueprint_pipeline','blueprint_pipeline.control_plane_storage_pins',"
        "'blueprint_pipeline.control_plane_storage_pin_observation'}; "
        "assert {name for name in sys.modules if name.startswith('blueprint_pipeline')} == expected"
    )
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True,
                            text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_oversized_lexical_root_refuses_before_component_allocation(monkeypatch):
    class NoSlice(str):
        def __getitem__(self, key):
            pytest.fail("an over-budget path must not be sliced/split into components")
    monkeypatch.setattr(observer, "MAX_PATH_BYTES", 4)
    with pytest.raises(observer.StoragePinObservationError, match="^pin_parameters_invalid$"):
        observer.observe_storage_pins(NoSlice("/oversized"), observed_at_epoch=50)


@pytest.mark.parametrize("failure,code", [("deadline", "pin_deadline_exceeded"),
                                          ("invalid", "pin_clock_invalid"),
                                          ("backwards", "pin_clock_invalid")])
@pytest.mark.parametrize("edge", ["second_row", "finalize"])
def test_accepted_first_row_then_clock_failure_uses_early_empty_fallback(root, monkeypatch, failure, code, edge):
    import builtins
    write(root, pin(owner="a"))
    write(root, pin(owner="z"))
    state = {"failed": False, "accepted": False}
    opened = set()
    real_open, real_close = observer.os.open, observer.os.close
    real_read_row, real_result = observer._Scan.read_row, observer._Scan.result
    real_asdict = observer.asdict

    def clock():
        if not state["failed"]:
            return 1.0
        return {"deadline": 7.0, "invalid": float('nan'), "backwards": 0.0}[failure]

    def read_row(scan, kind_fd, kind, name):
        if name == "z.json":
            assert len(scan.rows) == 1 and scan.rows[0].owner_id == "a"
            state["accepted"] = True
            if edge == "second_row":
                state["failed"] = True
        return real_read_row(scan, kind_fd, kind, name)

    def result(scan):
        if edge == "finalize":
            assert len(scan.rows) == 2
            state["failed"] = True
        return real_result(scan)

    def guarded_sorted(values, *args, **kwargs):
        if state["failed"] and values and isinstance(values, list) and isinstance(values[0], observer.ObservedStoragePin):
            pytest.fail("expired/invalid finalization must stop before sorting accepted evidence")
        return builtins.sorted(values, *args, **kwargs)

    def guarded_asdict(value):
        if state["failed"]:
            pytest.fail("expired/invalid finalization must not convert accepted evidence")
        return real_asdict(value)

    def tracked_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.add(fd)
        return fd

    def tracked_close(fd):
        opened.remove(fd)
        real_close(fd)

    monkeypatch.setattr(observer._Scan, "read_row", read_row)
    monkeypatch.setattr(observer._Scan, "result", result)
    monkeypatch.setattr(observer, "sorted", guarded_sorted, raising=False)
    monkeypatch.setattr(observer, "asdict", guarded_asdict)
    monkeypatch.setattr(observer.os, "open", tracked_open)
    monkeypatch.setattr(observer.os, "close", tracked_close)
    report = observe(root, monotonic=clock)
    incomplete(report, code)
    assert state["accepted"] is True
    assert report.rows == report.protected_identities == report.protected_paths == ()
    assert not opened
