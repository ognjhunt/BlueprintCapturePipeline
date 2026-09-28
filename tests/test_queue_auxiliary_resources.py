# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_queue_auxiliary_observation.py src/blueprint_pipeline/control_plane_queue_observation.py
"""Invocation-wide bounds apply to nested receipt locations too."""
import json
import os
import inspect

import pytest

from blueprint_pipeline import control_plane_queue_auxiliary_observation as auxiliary
from blueprint_pipeline import control_plane_queue_observation as primary
from tests.test_queue_auxiliary_layouts import CHILD, HEX, OTHER, STEM, observe, root_for, write


def test_unknown_group_is_retained_as_unknown_location_without_traversal(tmp_path):
    root = root_for(tmp_path, "sam")
    write(root, "progress/foreign/never-read.json")
    result = observe(("sam", root))
    assert not result.complete and result.rows == ()
    assert any(item.relative_path == "progress/foreign" and item.status == "unknown_layout"
               for item in result.roots[0].role_directories)


@pytest.mark.parametrize("field,value", [("family", "other"), ("family", []),
                                         ("root_path", "relative"), ("root_path", "/a/../b")])
def test_invalid_parameters_refuse_before_open(tmp_path, monkeypatch, field, value):
    values = {"family": "sam", "root_path": str(tmp_path)}
    values[field] = value
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("invalid API touched filesystem"))
    with pytest.raises(auxiliary.AuxiliaryQueueObservationError, match="^auxiliary_parameters_invalid$"):
        observe_contract = auxiliary.AuxiliaryQueueContract(**values)
        auxiliary.observe_preparation_sam_auxiliaries([observe_contract], observed_at_epoch=1)


@pytest.mark.parametrize("parameter,value", [("observed_at_epoch", True), ("observed_at_epoch", -1),
                                             ("observed_at_epoch", float("inf")),
                                             ("time_budget_seconds", 0), ("time_budget_seconds", 5.01),
                                             ("time_budget_seconds", True), ("monotonic", None)])
def test_invalid_clock_options_are_fixed_refusals(tmp_path, parameter, value):
    kwargs = {"observed_at_epoch": 1, parameter: value}
    with pytest.raises(auxiliary.AuxiliaryQueueObservationError, match="^auxiliary_parameters_invalid$"):
        auxiliary.observe_preparation_sam_auxiliaries(
            [auxiliary.AuxiliaryQueueContract("sam", str(tmp_path))], **kwargs)


def test_overlapping_roots_refuse(tmp_path):
    with pytest.raises(auxiliary.AuxiliaryQueueObservationError):
        observe(("sam", tmp_path), ("preparation", tmp_path / "nested"))


@pytest.mark.parametrize("limit,cap,blocker", [("MAX_ROWS", 1, "queue_rows_limit"),
                                             ("MAX_TOTAL_BYTES", 18, "queue_bytes_limit"),
                                             ("MAX_VALUES", 4, "queue_values_limit")])
def test_row_budgets_span_two_roots_and_groups(tmp_path, monkeypatch, limit, cap, blocker):
    roots = [root_for(tmp_path, "sam", name) for name in ("a", "b")]
    for root in roots:
        write(root, f"progress/{CHILD}/000001.json", '{"x":1}')
    # Two7-byte rows make aggregate14; use an exact smaller aggregate below.
    if limit == "MAX_TOTAL_BYTES":
        cap = 13
    monkeypatch.setattr(primary, limit, cap)
    result = observe(*[("sam", root) for root in roots])
    assert not result.complete and blocker in result.blockers


def test_directory_budget_spans_discovered_groups(tmp_path, monkeypatch):
    root = root_for(tmp_path, "preparation")
    write(root, f"source-progress/{STEM}/000001-{HEX}.json")
    write(root, f"source-progress/other-{OTHER}/000001-{HEX}.json")
    monkeypatch.setattr(auxiliary, "MAX_DIRECTORIES", 5)
    result = observe(("preparation", root))
    assert not result.complete and "queue_directories_limit" in result.blockers


def test_fd_budget_refuses_before_next_open_and_closes_owned_fds(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    live = set()
    opened, closed = os.open, os.close
    def track_open(*args, **kwargs):
        assert len(live) < 3
        fd = opened(*args, **kwargs)
        live.add(fd)
        return fd
    def track_close(fd):
        closed(fd)
        live.discard(fd)
    monkeypatch.setattr(auxiliary, "MAX_FDS", 3)
    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", track_close)
    result = observe(("sam", root))
    assert "queue_fds_limit" in result.blockers and not live


@pytest.mark.parametrize("cap,complete", [(7, True), (6, False)])
def test_auxiliary_row_policy_exact_boundary(tmp_path, monkeypatch, cap, complete):
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json", '{"x":1}')
    monkeypatch.setattr(auxiliary, "MAX_AUXILIARY_ROW_BYTES", cap)
    result = observe(("sam", root))
    assert result.complete is complete
    if not complete:
        assert "queue_row_bytes_limit" in result.blockers


@pytest.mark.parametrize("constant,cap,blocker", [("MAX_DEPTH", 1, "queue_depth_limit"),
                                               ("MAX_ENTRIES", 3, "queue_entries_limit"),
                                               ("MAX_OUTPUT_BYTES", 1, "queue_output_limit")])
def test_other_bounds_refuse_conservatively(tmp_path, monkeypatch, constant, cap, blocker):
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json", '{"x":{"y":1}}')
    monkeypatch.setattr(primary, constant, cap)
    result = observe(("sam", root))
    assert not result.complete and blocker in result.blockers


def test_deadline_mid_preflight_stops_before_parser(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json", '{"x":"' + "a" * 2500 + '"}')
    original = primary._Scan.preflight
    active = False
    checkpoints = 0
    def clock():
        nonlocal checkpoints
        if active:
            checkpoints += 1
        return 10 if checkpoints >= 2 else 0
    def preflight(self, text):
        nonlocal active
        active = True
        return original(self, text)
    monkeypatch.setattr(primary._Scan, "preflight", preflight)
    monkeypatch.setattr(json, "loads", lambda *a, **k: pytest.fail("parsed expired preflight"))
    result = observe(("sam", root), monotonic=clock)
    assert result.rows == () and "queue_deadline_exceeded" in result.blockers


@pytest.mark.parametrize("clock", [lambda: float("nan"), lambda: True])
def test_invalid_clock_empty_incomplete(tmp_path, clock):
    result = observe(("sam", tmp_path), monotonic=clock)
    assert not result.complete and result.rows == () and "queue_clock_invalid" in result.blockers


@pytest.mark.parametrize("name", ["unsafe[child]", "a" * 256])
def test_unsafe_or_overlong_group_is_not_opened(tmp_path, monkeypatch, name):
    root = root_for(tmp_path, "sam")
    if len(name) <= 255:
        (root / "progress" / name).mkdir()
    original = primary._Scan.names
    opened = primary._Scan.open
    def names(self, fd, pass_index):
        values = original(self, fd, pass_index)
        if os.fstat(fd).st_ino == (root / "progress").stat().st_ino:
            return (name,)
        return values
    def open_checked(self, value, *args, **kwargs):
        assert value != name
        return opened(self, value, *args, **kwargs)
    monkeypatch.setattr(primary._Scan, "names", names)
    monkeypatch.setattr(primary._Scan, "open", open_checked)
    result = observe(("sam", root))
    assert not result.complete and result.rows == ()
    assert "auxiliary_location_invalid" in result.blockers


def test_fixed_blockers_cap_and_no_raw_errors(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    write(root, "results/unknown.json", "broken")
    write(root, "started/staged.tmp")
    monkeypatch.setattr(primary, "MAX_BLOCKERS", 1)
    result = observe(("sam", root))
    assert not result.complete and len(result.blockers) == 2
    assert "queue_blockers_truncated" in result.blockers


def test_same_inode_root_alias_is_unknown_not_deduplicated(tmp_path, monkeypatch):
    roots = [root_for(tmp_path, "sam", name) for name in ("a", "b")]
    original = primary._Scan.walk
    def walk(self, path):
        return original(self, str(roots[0]) if path == str(roots[1]) else path)
    monkeypatch.setattr(primary._Scan, "walk", walk)
    result = observe(*[("sam", root) for root in roots])
    assert not result.complete and "queue_root_alias" in result.blockers
    assert len(result.roots) == 2


def test_output_exact_boundary_includes_auxiliary_metadata(tmp_path, monkeypatch):
    from dataclasses import asdict
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json", '{"x":"é\\n"}')
    result = observe(("sam", root))
    size = len(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False).encode())
    monkeypatch.setattr(primary, "MAX_OUTPUT_BYTES", size)
    assert observe(("sam", root)) == result
    monkeypatch.setattr(primary, "MAX_OUTPUT_BYTES", size - 1)
    refused = observe(("sam", root))
    assert not refused.complete and refused.roots == refused.rows == ()


@pytest.mark.parametrize("failure", ["expired", "invalid"])
def test_clock_failure_before_root_finalization_never_constructs_typed_evidence(tmp_path, monkeypatch, failure):
    root = root_for(tmp_path, "sam")
    for group in ("foreign-a", "foreign-b"):
        (root / "progress" / group).mkdir()
    expired = False
    constructed_after_failure = []
    live = set()
    opened, closed = os.open, os.close
    constructor = auxiliary.ObservedAuxiliaryRoot
    def clock():
        nonlocal expired
        caller = inspect.currentframe().f_back.f_back
        if caller.f_code.co_name == "observe_aux_root" and caller.f_locals.get("group") == "foreign-b":
            expired = True
        return (6 if failure == "expired" else float("nan")) if expired else 0
    def construct(*args, **kwargs):
        if expired:
            constructed_after_failure.append(args)
        return constructor(*args, **kwargs)
    def track_open(*args, **kwargs):
        fd = opened(*args, **kwargs)
        live.add(fd)
        return fd
    def track_close(fd):
        closed(fd)
        live.discard(fd)
    monkeypatch.setattr(auxiliary, "ObservedAuxiliaryRoot", construct)
    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", track_close)
    result = observe(("sam", root), monotonic=clock)
    assert expired and not result.complete and result.roots == result.rows == ()
    assert not constructed_after_failure
    assert not live
    assert ("queue_deadline_exceeded" if failure == "expired" else "queue_clock_invalid") in result.blockers
