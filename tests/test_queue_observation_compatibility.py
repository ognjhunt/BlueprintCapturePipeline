"""Covers: queue observation partial evidence, legacy callers and cold purity."""
import importlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest


def _module():
    return importlib.import_module("blueprint_pipeline.control_plane_queue_observation")


def _row(root, state="pending", name="a.json", raw=b'{"path":"/work/a"}'):
    directory = root / state
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(raw)
    return path


def _observe(module, root, states=("pending",)):
    return module.observe_queue_states([module.QueueRootContract(str(root), states)],
                                       observed_at_epoch=10, monotonic=lambda: 0)


def test_malformed_later_row_preserves_accepted_partial_evidence(tmp_path):
    module = _module()
    _row(tmp_path)
    _row(tmp_path, name="b.json", raw=b'{"broken":')
    result = _observe(module, tmp_path)
    assert not result.complete and len(result.rows) == 1 and result.blockers
    assert result.rows[0].raw_text == '{"path":"/work/a"}'


def test_hardlinked_rows_and_deterministic_order_preserve_separate_provenance(tmp_path):
    module = _module()
    first = _row(tmp_path, name="b.json")
    os.link(first, first.with_name("a.json"))
    one, two = _observe(module, tmp_path), _observe(module, tmp_path)
    assert one == two and one.complete and len(one.rows) == 2
    assert [Path(row.row_path).name for row in one.rows] == ["a.json", "b.json"]
    assert one.rows[0].row_identity == one.rows[1].row_identity


def test_legacy_permissive_and_strict_queue_readers_are_unchanged(tmp_path):
    from blueprint_pipeline.control_plane_storage_references import QueueReferenceUnreadable, queue_reference_text
    _row(tmp_path, raw=b'{"pending":1}')
    _row(tmp_path, "awaiting_capacity", raw=b'{"waiting":2}')
    _row(tmp_path, "pending", "invalid.json", raw=b'\xff')
    _row(tmp_path, "pending", "target.dat", raw=b'{}')
    (tmp_path / "pending" / "linked.json").symlink_to(tmp_path / "pending" / "target.dat")
    assert queue_reference_text([tmp_path]) == '{"pending":1}'
    with pytest.raises(QueueReferenceUnreadable):
        queue_reference_text([tmp_path], strict=True)
    assert queue_reference_text([tmp_path / "missing"], strict=True) == ""
    assert queue_reference_text([tmp_path], states=("awaiting_capacity",), strict=True) == '{"waiting":2}'
    result = _observe(_module(), tmp_path, ("pending", "awaiting_capacity"))
    assert not result.complete and len(result.rows) == 2


def test_output_size_limit_is_proved_without_building_uncapped_encoded_strings(tmp_path, monkeypatch):
    module = _module()
    _row(tmp_path, raw=b'{"x":"\\\\\\\\\\\\\\\\"}')
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 3)
    monkeypatch.setattr(module.json.JSONEncoder, "iterencode", lambda *a, **kw: pytest.fail("encoder allocated before output proof"))
    result = _observe(module, tmp_path)
    assert not result.complete and result.rows == () and "queue_output_limit" in result.blockers


def test_output_string_size_walk_checks_clock_in_loop(tmp_path, monkeypatch):
    module = _module()
    _row(tmp_path, raw=json.dumps({"x": "a" * 40}).encode())
    monkeypatch.setattr(module, "PREFLIGHT_CHECK_CHARS", 4)
    checkpoints = 0

    def clock():
        nonlocal checkpoints
        frame = inspect.currentframe().f_back.f_back
        if frame.f_code.co_name == "string_size":
            checkpoints += 1
        return 6 if checkpoints >= 3 else 0

    result = module.observe_queue_states([module.QueueRootContract(str(tmp_path), ("pending",))],
                                         observed_at_epoch=10, monotonic=clock)
    assert not result.complete and result.rows == () and "queue_deadline_exceeded" in result.blockers


@pytest.mark.parametrize("raw", [b'{"x":"plain"}', b'{"x":"\\n\\t\\r\\b\\f\\u0001\\\"\\\\"}', '{"x":"é中😀"}'.encode()])
def test_exact_output_size_matches_standard_json_representation(tmp_path, monkeypatch, raw):
    module = _module()
    _row(tmp_path, raw=raw)
    result = _observe(module, tmp_path)
    size = len(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False).encode())
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", size)
    assert _observe(module, tmp_path) == result
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", size - 1)
    assert not _observe(module, tmp_path).complete


def test_root_inode_alias_refuses_instead_of_deduplicating_configured_scope(tmp_path, monkeypatch):
    module = _module()
    first, second = tmp_path / "a", tmp_path / "b"
    _row(first)
    second.mkdir()
    real = module._Scan.walk

    def alias(self, path):
        return real(self, str(first) if path == str(second) else path)

    monkeypatch.setattr(module._Scan, "walk", alias)
    result = module.observe_queue_states([module.QueueRootContract(str(p), ("pending",)) for p in (first, second)],
                                        observed_at_epoch=10, monotonic=lambda: 0)
    assert not result.complete and "queue_root_alias" in result.blockers
    assert len(result.roots) == 2


def test_more_than_root_and_state_caps_refuses_before_scan(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "MAX_ROOTS", 1)
    with pytest.raises(module.QueueObservationError):
        module.observe_queue_states([module.QueueRootContract(str(tmp_path / x), ("pending",)) for x in ("a", "b")], observed_at_epoch=1)
    monkeypatch.setattr(module, "MAX_STATES", 1)
    with pytest.raises(module.QueueObservationError):
        _observe(module, tmp_path, ("pending", "processing"))


def test_reference_payload_paths_are_never_statted(tmp_path, monkeypatch):
    module = _module()
    _row(tmp_path, raw=b'{"path":"/not-present/private-payload"}')
    real = module.os.stat
    calls = []

    def statted(path, *args, **kwargs):
        calls.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "stat", statted)
    assert _observe(module, tmp_path).complete
    assert "/not-present/private-payload" not in calls


def test_fixed_blockers_are_capped_with_one_overflow_marker(tmp_path, monkeypatch):
    module = _module()
    _row(tmp_path, raw=b"broken")
    (tmp_path / "pending" / "other.tmp").write_bytes(b"staged")
    monkeypatch.setattr(module, "MAX_BLOCKERS", 1)
    result = _observe(module, tmp_path)
    assert not result.complete and len(result.blockers) == 2
    assert "queue_blockers_truncated" in result.blockers


@pytest.mark.slow
def test_first_observation_in_fresh_process_has_no_runtime_import_or_mutation(tmp_path):
    _row(tmp_path)
    script = r'''
import importlib.abc, os, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("blueprint_pipeline.") and fullname != "blueprint_pipeline.control_plane_queue_observation":
            raise AssertionError("unexpected runtime dependency " + fullname)
sys.meta_path.insert(0, Reject())
def refuse(*args, **kwargs):
    raise AssertionError("mutation attempted")
for name in ("mkdir", "unlink", "remove", "replace", "rename", "write", "link", "symlink"):
    setattr(os, name, refuse)
from blueprint_pipeline.control_plane_queue_observation import QueueRootContract, observe_queue_states
value = observe_queue_states([QueueRootContract(sys.argv[1], ("pending",))], observed_at_epoch=10, monotonic=lambda:0)
assert value.complete and len(value.rows) == 1 and value.mutations == 0
assert value.general_reference_inventory_complete is False and value.consumer_fence_checked is False
print("pure selected observation")
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True,
                            timeout=10, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src:."})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pure selected observation"
