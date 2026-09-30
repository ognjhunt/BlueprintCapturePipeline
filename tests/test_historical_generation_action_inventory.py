# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_generation.py
"""ADP-009D/day28: exact historical payload before a separate owner action."""
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def historical_tree():
    with TemporaryDirectory(prefix='historical-generation-') as temp:
        parent = Path(temp).resolve()
        target = parent / 'old-diagnostics'
        target.mkdir()
        (target / 'nested').mkdir()
        (target / 'one.bin').write_bytes(b'owned original bytes')
        (target / 'nested' / 'two.log').write_bytes(b'original log\n')
        yield parent, target


def inventory(parent, target, **kwargs):
    from blueprint_pipeline.control_plane_lane_historical_generation import inventory_historical_generation
    return inventory_historical_generation(target, allowed_roots=(parent,), **kwargs)


def test_exact_historical_generation_binds_original_bytes_without_action(historical_tree):
    parent, target = historical_tree
    before = {str(p): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    observed = inventory(parent, target)
    assert observed['target_path'] == str(target) and observed['parent_path'] == str(parent)
    assert observed['root_identity']['ino'] == parent.stat().st_ino
    assert observed['target_identity']['ino'] == target.stat().st_ino
    assert observed['execution_authorized'] is False
    assert observed['logical_payload_bytes'] == sum(map(len, before.values()))
    assert observed['member_count'] == 4  # Root, directory and two regular files.
    files = [row for row in observed['members'] if row['kind'] == 'file']
    assert len(files) == 2
    assert all(row['sha256'] == 'sha256:' + hashlib.sha256((target / row['path']).read_bytes()).hexdigest()
               for row in files)
    assert before == {str(p): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    assert not (target / '.lane-scratch.v1.json').exists()
    (target / 'one.bin').write_bytes(b'different generation')
    assert inventory(parent, target)['generation_digest'] != observed['generation_digest']


@pytest.mark.parametrize('alias', ['symlink', 'hardlink'])
def test_aliases_cannot_become_historical_action_members(historical_tree, alias):
    parent, target = historical_tree
    if alias == 'symlink':
        (target / 'alias').symlink_to(target / 'one.bin')
    else:
        os.link(target / 'one.bin', target / 'alias')
    with pytest.raises(ValueError, match='historical_generation_member_unsupported'):
        inventory(parent, target)


def test_only_direct_configured_children_are_action_candidates(historical_tree):
    parent, target = historical_tree
    with pytest.raises(ValueError, match='historical_generation_scope_invalid'):
        inventory(parent, target / 'nested')


@pytest.mark.parametrize('limit', ['members', 'bytes'])
def test_inventory_refuses_before_exceeding_finite_original_bounds(historical_tree, limit):
    parent, target = historical_tree
    options = {'max_members': 3} if limit == 'members' else {'max_payload_bytes': 16}
    with pytest.raises(ValueError, match='historical_generation_limit'):
        inventory(parent, target, **options)


def test_changed_payload_during_hash_is_not_a_reviewable_generation(historical_tree, monkeypatch):
    parent, target = historical_tree
    from blueprint_pipeline import control_plane_lane_historical_generation as module
    prior = module.os.read
    inode = (target / 'one.bin').stat().st_ino
    changed = False
    def rewriting(fd, amount):
        nonlocal changed
        result = prior(fd, amount)
        if not changed and os.fstat(fd).st_ino == inode:
            changed = True
            (target / 'one.bin').write_bytes(b'new bytes while hashing')
        return result
    monkeypatch.setattr(module.os, 'read', rewriting)
    with pytest.raises(ValueError, match='historical_generation_changed'):
        inventory(parent, target)
    assert changed


def test_unrelated_ancestor_namespace_change_does_not_change_selected_generation(historical_tree, monkeypatch):
    parent, target = historical_tree
    from blueprint_pipeline import control_plane_lane_historical_generation as module
    original = module.os.read
    changed = False
    sibling = parent.parent / (parent.name + '-unselected')
    def read(fd, amount):
        nonlocal changed
        value = original(fd, amount)
        if not changed and os.fstat(fd).st_ino == (target / 'one.bin').stat().st_ino:
            changed = True
            sibling.mkdir()
        return value
    monkeypatch.setattr(module.os, 'read', read)
    try:
        observed = inventory(parent, target)
        assert changed and observed['member_count'] == 4
    finally:
        sibling.rmdir()


def test_original_selected_parent_change_during_hash_still_refuses(historical_tree, monkeypatch):
    parent, target = historical_tree
    from blueprint_pipeline import control_plane_lane_historical_generation as module
    original = module.os.read
    changed = False
    def read(fd, amount):
        nonlocal changed
        value = original(fd, amount)
        if not changed and os.fstat(fd).st_ino == (target / 'one.bin').stat().st_ino:
            changed = True
            (parent / 'changed-selected-parent').mkdir()
        return value
    monkeypatch.setattr(module.os, 'read', read)
    with pytest.raises(ValueError, match='historical_generation_changed'):
        inventory(parent, target)
    assert changed


def test_unselected_ancestor_sibling_during_descriptor_acquisition(historical_tree, monkeypatch):
    parent, target = historical_tree
    from blueprint_pipeline import control_plane_lane_historical_generation as module
    original = module.os.open
    changed = False
    sibling = parent.parent / (parent.name + '-acquisition-sibling')
    def opening(name, flags, *args, **kwargs):
        nonlocal changed
        if not changed and name == parent.parent.name:
            changed = True
            sibling.mkdir()
        return original(name, flags, *args, **kwargs)
    monkeypatch.setattr(module.os, 'open', opening)
    try:
        assert inventory(parent, target)['member_count'] == 4
        assert changed
    finally:
        sibling.rmdir()


def test_elapsed_deadline_refuses_generation(historical_tree):
    parent, target = historical_tree
    calls = iter([0.0, 2.0])
    with pytest.raises(ValueError, match='historical_generation_deadline'):
        inventory(parent, target, max_seconds=1.0, monotonic=lambda: next(calls))
