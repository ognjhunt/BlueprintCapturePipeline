"""Covers: bounded primary queue observation contracts, raw receipts and syntax.

ADP-009D/day-28: this helper observes selected states without cleanup authority.
"""
import hashlib
import importlib
import json
from dataclasses import asdict

import pytest


def _module():
    return importlib.import_module("blueprint_pipeline.control_plane_queue_observation")


def _observe(root, states=("pending",), **kwargs):
    module = _module()
    return module.observe_queue_states(
        [module.QueueRootContract(str(root), states)], observed_at_epoch=10,
        monotonic=lambda: 0, **kwargs,
    )


def _row(root, state="pending", name="item.json", raw=b'{"path":"/work/a"}\n'):
    directory = root / state
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(raw)
    return path


def test_missing_root_is_unknown_without_creation(tmp_path):
    root = tmp_path / "missing"
    result = _observe(root)
    assert not result.complete and "queue_root_missing" in result.blockers
    assert result.rows == () and not root.exists()


def test_stable_missing_selected_state_is_empty_only_in_existing_scope(tmp_path):
    result = _observe(tmp_path, ("pending", "processing"))
    assert result.complete and result.rows == ()
    assert result.roots[0].missing_states == ("pending", "processing")
    assert result.scope == "selected_primary_queue_states_only"
    assert not result.general_reference_inventory_complete
    assert not result.consumer_fence_checked and not result.producer_seals_verified
    assert result.mutations == 0 and not result.execution_authorized


PRIMARY = {
    "launches": ("pending", "processing", "completed", "blocked"),
    "preparations": ("pending", "processing", "awaiting_source_preparation", "awaiting_capacity", "materialized", "completed", "blocked"),
    "compilations": ("pending", "processing", "completed", "blocked"),
    "activations": ("pending", "processing", "prepared", "blocked"),
    "canaries": ("pending", "processing", "completed", "blocked", "stranded"),
    "constructions": ("pending", "processing", "completed", "blocked"),
    "sam": ("pending", "processing", "waiting_external", "completed", "failed"),
    "releases": ("pending", "processing", "completed", "blocked"),
}


@pytest.mark.parametrize("family,state", [(family, state) for family, states in PRIMARY.items() for state in states])
def test_every_reviewed_primary_state_is_observed_without_basename_fallback(tmp_path, family, state):
    root = tmp_path / ("configured-" + family)
    path = _row(root, state)
    result = _observe(root, PRIMARY[family])
    assert result.complete and len(result.rows) == 1
    assert result.rows[0].state == state and result.rows[0].row_path == str(path)
    assert result.roots[0].selected_states == tuple(sorted(PRIMARY[family]))


def test_raw_identity_and_duplicate_names_across_states_are_preserved(tmp_path):
    raw = b'{ "b":2, "a":1 }\n'
    _row(tmp_path, "pending", raw=raw)
    _row(tmp_path, "processing", raw=raw)
    result = _observe(tmp_path, ("processing", "pending"))
    assert result.complete and len(result.rows) == 2
    assert [row.state for row in result.rows] == ["pending", "processing"]
    assert all(row.raw_text == raw.decode() and row.raw_size_bytes == len(raw) for row in result.rows)
    assert all(row.raw_sha256 == "sha256:" + hashlib.sha256(raw).hexdigest() for row in result.rows)
    assert result.rows[0].raw_sha256 != "sha256:" + hashlib.sha256(b'{"a":1,"b":2}').hexdigest()


@pytest.mark.parametrize("raw", [b"\xff", b"[]", b"null", b"{} trailing", b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":"\\ud800"}', b'{"x":'])
def test_malformed_row_never_yields_complete_empty_scope(tmp_path, raw):
    _row(tmp_path, raw=raw)
    result = _observe(tmp_path)
    assert not result.complete and result.rows == ()
    assert "queue_row_invalid" in result.blockers


@pytest.mark.parametrize("states", [(), ("pending", "pending"), ("..",), ("pending/x",), ("Pending",), ("",), (True,), ("x" * 65,)])
def test_invalid_state_contract_refuses_before_filesystem(tmp_path, states, monkeypatch):
    module = _module()
    monkeypatch.setattr(module.os, "open", lambda *a, **kw: pytest.fail("filesystem called"))
    with pytest.raises(module.QueueObservationError, match="^queue_parameters_invalid$"):
        _observe(tmp_path, states)


@pytest.mark.parametrize("root", ["relative", "/x/../y", "/x/./y", "/x//y", "//x", "/x/", "/x/<secret>", "/x/\n", "/" + "/".join(["x"] * 65), "/" + "x" * 4096, "/x/?", "/x/[y]", "/x/*", "/x/\ud800"], ids=lambda value: str(value)[:20])
def test_invalid_root_contract_is_bounded_and_sanitized(root):
    module = _module()
    with pytest.raises(module.QueueObservationError, match="^queue_parameters_invalid$"):
        module.observe_queue_states([module.QueueRootContract(root, ("pending",))], observed_at_epoch=1)


def test_oversized_root_refuses_before_slice_or_component_allocation(monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "MAX_PATH_BYTES", 2)

    class SliceSpy(str):
        def __getitem__(self, key):
            if isinstance(key, slice):
                pytest.fail("oversized input sliced before refusal")
            return super().__getitem__(key)

    with pytest.raises(module.QueueObservationError, match="^queue_parameters_invalid$"):
        module.observe_queue_states([module.QueueRootContract(SliceSpy("/x/y"), ("pending",))], observed_at_epoch=1)


def test_duplicate_and_overlapping_root_contracts_refuse(tmp_path):
    module = _module()
    for paths in [(tmp_path, tmp_path), (tmp_path, tmp_path / "nested")]:
        with pytest.raises(module.QueueObservationError):
            module.observe_queue_states([module.QueueRootContract(str(p), ("pending",)) for p in paths], observed_at_epoch=1)


@pytest.mark.parametrize("field,value", [("observed_at_epoch", True), ("observed_at_epoch", -1), ("observed_at_epoch", float("nan")), ("observed_at_epoch", 10**1000), ("time_budget_seconds", 0), ("time_budget_seconds", 6), ("time_budget_seconds", True), ("monotonic", None)])
def test_invalid_api_values_refuse(tmp_path, field, value):
    module = _module()
    kwargs = {"observed_at_epoch": 1, field: value}
    with pytest.raises(module.QueueObservationError):
        module.observe_queue_states([module.QueueRootContract(str(tmp_path), ("pending",))], **kwargs)


def test_unselected_auxiliary_entries_are_excluded_not_invented_empty_inventory(tmp_path):
    _row(tmp_path)
    _row(tmp_path, "results", raw=b'{"path":"/work/retained"}')
    (tmp_path / "source-resume-completed").mkdir()
    result = _observe(tmp_path)
    assert result.complete and len(result.rows) == 1
    assert result.roots[0].unobserved_root_entries == ("results", "source-resume-completed")
    assert not result.general_reference_inventory_complete
    assert json.loads(json.dumps(asdict(result)))["producer_seals_verified"] is False


def test_selected_unknown_entry_keeps_accepted_partial_row(tmp_path):
    _row(tmp_path, name="a.json")
    (tmp_path / "pending" / "b.tmp").write_text("unfinished")
    result = _observe(tmp_path)
    assert not result.complete and len(result.rows) == 1
    assert "queue_entry_unknown" in result.blockers
