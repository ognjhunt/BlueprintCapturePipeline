# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_queue_auxiliary_observation.py src/blueprint_pipeline/control_plane_queue_observation.py
"""The unused auxiliary seam preserves primary defaults and imports no runtime."""
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_queue_auxiliary_observation as auxiliary
from blueprint_pipeline import control_plane_queue_observation as primary
from tests.test_queue_auxiliary_layouts import CHILD, observe, root_for, write


def test_auxiliary_byte_policy_does_not_reduce_primary_default(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json", '{"x":1}')
    primary_root = tmp_path / "primary"
    write(primary_root, "pending/row.json", '{"x":1}')
    monkeypatch.setattr(auxiliary, "MAX_AUXILIARY_ROW_BYTES", 6)
    monkeypatch.setattr(primary, "MAX_ROW_BYTES", 7)
    refused = observe(("sam", root))
    assert not refused.complete and "queue_row_bytes_limit" in refused.blockers
    accepted = primary.observe_queue_states(
        [primary.QueueRootContract(str(primary_root), ("pending",))], observed_at_epoch=1)
    assert accepted.complete and len(accepted.rows) == 1
    assert primary._Scan((), 1, lambda: 0, 5)._row_bytes_limit == 7


def test_repeated_observation_is_deterministic_and_frozen(tmp_path):
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json")
    first = observe(("sam", root))
    assert first == observe(("sam", root))
    with pytest.raises(FrozenInstanceError):
        first.complete = False
    with pytest.raises(FrozenInstanceError):
        first.rows[0].layout_role = "authority"


def test_json_payload_paths_never_become_filesystem_targets(tmp_path, monkeypatch):
    root = root_for(tmp_path, "sam")
    target = "/private/foreign-payload-never-read"
    write(root, f"started/{CHILD}.json", '{"path":"' + target + '"}')
    statted, opened = os.stat, os.open
    def stat_no_payload(path, *args, **kwargs):
        assert str(path) != target
        return statted(path, *args, **kwargs)
    def open_no_payload(path, *args, **kwargs):
        assert str(path) != target
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(os, "stat", stat_no_payload)
    monkeypatch.setattr(os, "open", open_no_payload)
    assert observe(("sam", root)).complete


@pytest.mark.parametrize("constant,cap,complete", [("MAX_ENTRIES", 5, True), ("MAX_ENTRIES", 4, False)])
def test_entry_budget_inclusive_for_stable_empty_selected_roles(tmp_path, monkeypatch, constant, cap, complete):
    root = root_for(tmp_path, "sam")
    monkeypatch.setattr(primary, constant, cap)
    assert observe(("sam", root)).complete is complete


@pytest.mark.parametrize("cap,complete", [(6, True), (5, False)])
def test_directory_budget_inclusive_for_root_and_five_roles(tmp_path, monkeypatch, cap, complete):
    root = root_for(tmp_path, "sam")
    monkeypatch.setattr(auxiliary, "MAX_DIRECTORIES", cap)
    assert observe(("sam", root)).complete is complete


@pytest.mark.slow
def test_first_auxiliary_call_in_cold_interpreter_has_no_runtime_or_mutation(tmp_path):
    root = root_for(tmp_path, "sam")
    write(root, f"started/{CHILD}.json")
    script = r'''
import importlib.abc, os, sys
allowed = {
    "blueprint_pipeline.control_plane_queue_observation",
    "blueprint_pipeline.control_plane_queue_auxiliary_observation",
}
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("blueprint_pipeline.") and fullname not in allowed:
            raise AssertionError("runtime import attempted " + fullname)
sys.meta_path.insert(0, Reject())
def refuse(*args, **kwargs):
    raise AssertionError("mutation attempted")
for name in ("mkdir", "unlink", "remove", "replace", "rename", "write", "link", "symlink"):
    setattr(os, name, refuse)
original = os.open
def read_only(name, flags, *args, **kwargs):
    assert not flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND)
    return original(name, flags, *args, **kwargs)
os.open = read_only
from blueprint_pipeline.control_plane_queue_auxiliary_observation import (
    AuxiliaryQueueContract, observe_preparation_sam_auxiliaries,
)
result = observe_preparation_sam_auxiliaries(
    [AuxiliaryQueueContract("sam", sys.argv[1])], observed_at_epoch=1,
)
assert result.complete and len(result.rows) == 1
assert not result.consumer_bindings_verified and not result.general_reference_inventory_complete
'''
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    completed = subprocess.run([sys.executable, "-c", script, str(root)], env=environment,
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
