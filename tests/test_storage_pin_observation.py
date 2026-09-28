# Covers: src/blueprint_pipeline/control_plane_storage_pin_observation.py
"""ADP-009D/day-28: strict observations cannot turn missing pins into clearance."""
from __future__ import annotations

import dataclasses
import hashlib
import json

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
